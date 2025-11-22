import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import copy
import contextlib

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss_multi import lossAV, lossA, lossV
from model.loconet_encoder import locoencoder

import torch.distributed as dist
from utils.distributed import all_gather, all_reduce
from copy import deepcopy
import matplotlib.pyplot as plt
import shutil
import cv2
import imageio
from PIL import Image
import math
from typing import List, Mapping, Optional, Tuple, Union
import numpy as np


class Loconet(nn.Module):

    def __init__(self, cfg, n_channel, layer, consistency = -1, consistency_method = "kl", talknce_lambda = 0.0):
        super(Loconet, self).__init__()
        self.cfg = cfg
        self.model = locoencoder(cfg)
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

        self.n_channel = n_channel
        self.landmark_bottleneck = nn.Conv2d(in_channels=164, out_channels=n_channel, kernel_size=(1, 1))

        # insert before layer
        self.layer = layer

        if layer == 0:
            self.bottle_neck = nn.Conv2d(in_channels=(1 + n_channel), out_channels=1, kernel_size=(1, 1))
        elif layer == 1:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1))
        elif layer == 2:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1))
        elif layer == 3:
            self.bottle_neck = nn.Conv2d(in_channels=(128 + n_channel), out_channels=128, kernel_size=(1, 1))
        elif layer == 4:
            self.bottle_neck = nn.Conv2d(in_channels=(256 + n_channel), out_channels=256, kernel_size=(1, 1))

        self.consistency_lambda = consistency
        self.consistency_method = consistency_method
        self.talknce_lambda = talknce_lambda

    def create_landmark_tensor(self, landmark, dtype, device):
        """
            landmark has shape (b, s, t, 82, 2)
            return tensor has shape (b, s, t, 164, W, H)
        """
        landmarkTensor = None
        
        if self.layer == 0:
            W, H = 112, 112
        elif self.layer == 1:
            W, H = 28, 28
        elif self.layer == 2:
            W, H = 28, 28
        elif self.layer == 3:
            W, H = 14, 14
        elif self.layer == 4:
            W, H = 7, 7

        b, s, t, _, _ = landmark.shape
        landmarkTensor = torch.zeros((b, s, t, 82, 2, W, H), dtype=dtype, device=device)
        landmark_idx = ((landmark > 0.0) | torch.isclose(landmark, torch.tensor(0.0))) & ((landmark < 1.0) | landmark.isclose(landmark, torch.tensor(1.0)))

        landmark_masked = torch.where(landmark_idx, landmark, torch.tensor(float('nan')))

        coordinate = torch.where(torch.isnan(landmark_masked), torch.tensor(float('nan')), torch.min(torch.floor(landmark_masked * W), torch.tensor(W - 1)))

        # Convert coordinates to long, handling NaN to avoid indexing issues
        coord_0 = coordinate[..., 0].long()
        coord_1 = coordinate[..., 1].long()
        
        # Create a mask for valid coordinates (non-NaN)
        valid_mask = ~torch.isnan(coordinate[..., 0]) & ~torch.isnan(coordinate[..., 1])

        # Get valid indices
        b_id, s_id, t_id, lip_id = torch.nonzero(valid_mask, as_tuple=True)

        if b_id.numel() > 0:  # Ensure there are valid indices
            landmarkTensor[b_id, s_id, t_id, lip_id, :, coord_0[b_id, s_id, t_id, lip_id], coord_1[b_id, s_id, t_id, lip_id]] = landmark[b_id, s_id, t_id, lip_id, :]

        landmarkTensor = landmarkTensor.reshape(b * s * t, -1, W, H)

        assert (landmarkTensor.shape[1] == 164)
        return landmarkTensor

     # hook the gradient of the activation
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_forward_activation(self, x, landmarkFeature):
        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = x.transpose(0, 1).transpose(1, 2)
        batchsize = x.shape[0]
        x = self.model.visualFrontend.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3],
                              x.shape[4]) # b * s * t, c, w, h
        
        # inject before self.layer
        # landmarkFeature has shape (b * s * t, n_channel, H, W)
        # x has shape (b * s * t, c, W, H)

        layers = [self.model.visualFrontend.resnet.layer1, self.model.visualFrontend.resnet.layer2, self.model.visualFrontend.resnet.layer3, self.model.visualFrontend.resnet.layer4]
        for i in range(3):
            if i == self.layer - 1:
                x = torch.cat((x, landmarkFeature), dim = 1)
                x = self.bottle_neck(x)
                x = layers[i](x)
            else:
                x = layers[i](x)
        return x
    
    def get_activations(self, visualFeature, landmark):
        b, s, t = visualFeature.shape[:3]

        # landmark recomposition
        # initial shape: b, s, t, 82, 2
        # transform to (b * s * t, 164, H, W) to run 1x1 kernel
        print(landmark.shape)
        landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
        landmarkFeature = self.landmark_bottleneck(landmarkFeature)
        
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        visualEmbed = self.get_forward_activation(visualFeature, landmarkFeature)

        return visualEmbed

    def forward_visual_frontend(self, x, landmarkFeature):
        B, T, W, H = x.shape
        if self.layer == 0:
            x = x.view(B * T, 1, W, H)
            x = torch.cat((x, landmarkFeature), dim = 1)
            x = self.bottle_neck(x)
            x = x.view(B, T, W, H)
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = x.transpose(0, 1).transpose(1, 2)
        batchsize = x.shape[0]
        x = self.model.visualFrontend.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3],
                              x.shape[4]) # b * s * t, c, w, h
        
        # inject before self.layer
        # landmarkFeature has shape (b * s * t, n_channel, H, W)
        # x has shape (b * s * t, c, W, H)

        layers = [self.model.visualFrontend.resnet.layer1, self.model.visualFrontend.resnet.layer2, self.model.visualFrontend.resnet.layer3, self.model.visualFrontend.resnet.layer4]
        for i in range(4):
            if i == self.layer - 1:
                x = torch.cat((x, landmarkFeature), dim = 1)
                x = self.bottle_neck(x)
                x = layers[i](x)
            else:
                x = layers[i](x)
            if i == 2:
                # hook the gradient
                if x.requires_grad:
                    x.register_hook(self.activations_hook)

        x = self.model.visualFrontend.resnet.avgpool(x)
        x = x.reshape(batchsize, -1, 512)
        x = x.transpose(1, 2)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.model.visualTCN(x)
        x = self.model.visualConv1D(x)
        x = x.transpose(1, 2)
        return x
    
    def info_nce_loss1(self, features_vis, features_aud, labels):
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() #[T, T]
        labels = labels.to('cuda')

        features_vis = F.normalize(features_vis, dim=1) #[T,128]
        features_aud = F.normalize(features_aud, dim=1) #[T,128]

        similarity_matrix = torch.matmul(features_vis, features_aud.T) # [T,T]

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda')

        # select and combine multiple positives
        positives = similarity_matrix[mask].view(labels.shape[0], -1)
        # print(positives)

        # select only the negatives the negatives
        negatives = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

        logits = logits / 0.07
        return logits, labels

    def forward(self, audioFeature, visualFeature, landmark, labels, masks):
        b, s, t = visualFeature.shape[:3]

        # landmark recomposition
        # initial shape: b, s, t, 82, 2
        # transform to (b * s * t, 164, H, W) to run 1x1 kernel
        landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
        landmarkFeature = self.landmark_bottleneck(landmarkFeature)
        
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])

        visualFeatureOriginal = visualFeature.clone()
        audioFeatureOriginal = audioFeature.clone()

        audioEmbed = self.model.forward_audio_frontend(audioFeature)    # B, C, T, 4
        visualEmbed = self.forward_visual_frontend(visualFeature, landmarkFeature)
        audioEmbed = audioEmbed.repeat(s, 1, 1)

        # add talknce loss
        nce_loss = 0
        if self.cfg.use_talknce:
            new_labels = labels[0].reshape((-1))
            tri_vis = visualEmbed[0].reshape(-1,128)
            tri_aud = audioEmbed[0].reshape(-1,128) #[T*128]

            active_index = np.where(new_labels.cpu()==1) # get active segments
            if len(active_index[0]) > 0:
                tri_vis2 = torch.stack([tri_vis[i,:] for i in active_index[0]], dim=0)
                tri_aud2 = torch.stack([tri_aud[j,:] for j in active_index[0]], dim=0)
                nce_label = torch.ones_like(torch.Tensor(active_index[0])).to('cuda')

                logits, labels_nce = self.info_nce_loss1(tri_vis2, tri_aud2, nce_label)
                # print(labels_nce.shape)
                # print(logits.shape)
                nce_loss = self.criterion(logits, labels_nce) #input, target
            else:
                nce_loss=0
        else:
            assert nce_loss == 0

        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)

        labels = labels.reshape((-1))
        masks = masks.reshape((-1))
        nlossAV, predScoresLandmark, _, prec = self.lossAV.forward(outsAV, labels, masks)
        nlossA = self.lossA.forward(outsA, labels, masks)
        nlossV = self.lossV.forward(outsV, labels, masks)

        consistency_loss = 0
        if self.consistency_lambda > 0.0:
            audioEmbed = self.model.forward_audio_frontend(audioFeatureOriginal)    # B, C, T, 4
            visualEmbed = self.forward_visual_frontend(visualFeatureOriginal, torch.zeros_like(landmarkFeature))
            audioEmbed = audioEmbed.repeat(s, 1, 1)
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAVNonLandmark = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)

            if self.consistency_method == "mse":
                consistency_loss = self.consistency_lambda * self.mse_loss(outsAVNonLandmark, outsAV)
            else:
                labels = labels.reshape((-1))
                masks = masks.reshape((-1))
                _, predScoresNonLandmark, _, _ = self.lossAV.forward(outsAVNonLandmark, labels, masks)
                consistency_loss = self.consistency_lambda * self.kl_loss(predScoresNonLandmark.log(), predScoresLandmark.detach())
                assert consistency_loss >= 0.0
                
            
            # print(consistency_loss)

        nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV + self.talknce_lambda*nce_loss + consistency_loss

        num_frames = masks.sum()
        return nloss, prec, num_frames

    def forward_evaluation(self, audioFeature, visualFeature, landmark, labels, masks, useLandmark = True):
        if labels == None:
            b, s, t = visualFeature.shape[:3]

            # landmark recomposition
            # initial shape: b, s, t, 82, 2
            # transform to (b * s * t, 164, H, W) to run 1x1 kernel
            landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
            landmarkFeature = self.landmark_bottleneck(landmarkFeature)
            if not useLandmark:
                landmarkFeature = torch.zeros_like(landmarkFeature)
            
            visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])

            audioEmbed = self.model.forward_audio_frontend(audioFeature)    # B, C, T, 4
            visualEmbed = self.forward_visual_frontend(visualFeature, landmarkFeature)
            audioEmbed = audioEmbed.repeat(s, 1, 1)

            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
            outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
            predScore = self.lossAV.forward(outsAV, labels, masks)

            return predScore
        b, s, t = visualFeature.shape[:3]

        # landmark recomposition
        # initial shape: b, s, t, 82, 2
        # transform to (b * s * t, 164, H, W) to run 1x1 kernel
        landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
        landmarkFeature = self.landmark_bottleneck(landmarkFeature)
        if not useLandmark:
            landmarkFeature = torch.zeros_like(landmarkFeature)
        
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])

        audioEmbed = self.model.forward_audio_frontend(audioFeature)    # B, C, T, 4
        visualEmbed = self.forward_visual_frontend(visualFeature, landmarkFeature)
        audioEmbed = audioEmbed.repeat(s, 1, 1)

        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        labels = labels.reshape((-1))
        masks = masks.reshape((-1))
        outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
        labels = labels.view(b, s, t)[:, 0, :].view(b * t).cuda()
        masks = masks.view(b, s, t)[:, 0, :].view(b * t)
        _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)

        return predScore

class loconet(nn.Module):
    

    def __init__(self, cfg, n_channel = 1, layer = 1, rank=None, device=None, consistency_method = "kl", consistency_lambda = -1, talknce_lambda = 0.0):
        super(loconet, self).__init__()
        self.cfg = cfg
        self.rank = rank
        if rank != None:
            self.rank = rank
            self.device = device
            self.n_channel = n_channel
            self.layer = layer
            self.model = Loconet(cfg, n_channel, layer, consistency_lambda, consistency_method, talknce_lambda).to(device)
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[rank],
                                                             output_device=rank,
                                                             find_unused_parameters=False)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.BASE_LR)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                             step_size=1,
                                                             gamma=self.cfg.SOLVER.SCHEDULER.GAMMA)
        else:
            self.n_channel = n_channel
            self.layer = layer
            self.model = Loconet(cfg, n_channel, layer).cuda()
            self.evalDataType = cfg.evalDataType
            self.method = consistency_method
            self.consistency_lambda = consistency_lambda

        print(
            time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
            (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.model.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        loader.sampler.set_epoch(epoch)
        device = self.device

        pbar = enumerate(loader, start=1)
        if self.rank == 0:
            pbar = tqdm.tqdm(pbar, total=loader.__len__())

        for num, (audioFeature, visualFeature, landmarks, labels, masks) in pbar:
            audioFeature = audioFeature.to(device)
            visualFeature = visualFeature.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            landmarks = landmarks.to(device) # b, s, t, 82, 2
            nloss, prec, num_frames = self.model(
                audioFeature,
                visualFeature,
                landmarks,
                labels,
                masks,
            )

            self.optim.zero_grad()
            nloss.backward()
            self.optim.step()

            [nloss, prec, num_frames] = all_reduce([nloss, prec, num_frames], average=False)
            top1 += prec.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            index += int(num_frames.detach().cpu().item())
            if self.rank == 0:
                pbar.set_postfix(
                    dict(epoch=epoch,
                         lr=lr,
                         loss=loss / (num * self.cfg.NUM_GPUS),
                         acc=(top1 / index)))
        acc= top1 / index
        dist.barrier()
        return loss / num, lr, acc

    def evaluate_network(self, epoch, loader, useLandmark = True):
        self.eval()
        predScores = []
        evalCsvSave = os.path.join(self.cfg.WORKSPACE, "{}_res_{}_{}_{}_{}_{}_shifted_{}.csv".format(epoch, self.n_channel, self.layer, self.method, self.consistency_lambda, useLandmark, self.cfg.shift_factor))
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, landmarks, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                landmarks = landmarks.cuda()
                predScore = self.model.forward_evaluation(audioFeature, visualFeature, landmarks, labels, masks, useLandmark)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig,
                                                                                      evalCsvSave)
        mAP = float(
            str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        return mAP
    
    def evaluate(self, epoch, loader, useLandmark = True):
        self.eval()
        predScores = []
        evalCsvSave = os.path.join(self.cfg.WORKSPACE, "{}_res_{}_{}_{}_{}_{}_{}_talknce:{}_shifted_{}".format(epoch, "reverse" if self.cfg.evalDataType == "test_reverse" else "val", self.n_channel, self.layer, self.method, self.consistency_lambda, useLandmark, self.cfg.use_talknce, self.cfg.shift_factor))
        if self.cfg.use_full_landmark:
            evalCsvSave += "_full_landmark"
        if self.cfg.only_landmark:
            evalCsvSave += "_only_landmark"
        evalCsvSave += ".csv"
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, landmarks, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                landmarks = landmarks.cuda()
                predScore = self.model.forward_evaluation(audioFeature, visualFeature, landmarks, labels, masks, useLandmark)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        # evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' if score >= 0.5 else 'NOT_SPEAKING' for score in predScores])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        print(evalCsvSave)  

    def evaluate_grad_cam(self, epoch, loader):
        self.eval()
        predScores = []
        savePath = os.path.join(self.cfg.WORKSPACE, "grad_cam_our")
        if os.path.exists(savePath):
            shutil.rmtree(savePath)
        os.makedirs(savePath)
        cnt = 0
        for id, (audioReverseFeature, visualFeature, landmarks, labels, masks) in tqdm.tqdm(enumerate(loader)):
            if id < 1:
                continue
            b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
            
            visualFeature = visualFeature.cuda()
            audioReverseFeature = audioReverseFeature.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            landmarks = landmarks.cuda()

            visualFeatureOriginal = deepcopy(visualFeature)
            print(visualFeatureOriginal.shape)
            landmarksOriginal = deepcopy(landmarks)

            pred_score = self.model.forward_evaluation(audioReverseFeature, visualFeature, landmarks, None, None)

            loss = torch.sum(pred_score[:, 1])
            loss.backward()

            gradients = self.model.get_activations_gradient()

            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[2, 3])

            visualReverseCopy = deepcopy(visualFeatureOriginal)
            activations = self.model.get_activations(visualReverseCopy, landmarksOriginal).detach()
            # weight the channels by corresponding gradients
            print(activations.shape, pooled_gradients.shape)
            for i in range(activations.shape[1]):
                for j in range(activations.shape[0]):
                    activations[j, i, :, :] *= pooled_gradients[j, i]

            grad_cam = torch.sum(activations, dim = 1)
            grad_cam = torch.nn.functional.relu(grad_cam)
            grad_cam = grad_cam.reshape(s, t, *grad_cam.shape[1:])[0]

            for index in range(t):
                heatmap = grad_cam[index].detach().cpu().numpy()

                # Normalize the heatmap
                heatmap /= numpy.max(heatmap)

                # Add a channel dimension to make it (H, W, 1)
                heatmap = numpy.expand_dims(heatmap, axis=-1)
                # heatmap = heatmap.swapaxes(0,1).swapaxes(1, 2)
                heatmap = cv2.resize(heatmap, (112, 112))
                heatmap = numpy.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                superimposed_img = heatmap * 0.4 + visualFeatureOriginal[0, 0, index:(index+1)].transpose(0,1).transpose(1, 2).cpu().numpy()
                cv2.imwrite(os.path.join(savePath, f"detecton_frame_{index}_prediction_{pred_score[index, 1] >= 0}.png"), superimposed_img)
            break
        

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location='cpu')
        if self.rank != None:
            info = self.load_state_dict(loadedState)
        else:
            new_state = {}
            
            for k, v in loadedState.items():
                new_state[k.replace("model.module.", "model.")] = v
            # print(new_state.keys())
            # print()
            info = self.load_state_dict(new_state, strict=False)
        print(info)