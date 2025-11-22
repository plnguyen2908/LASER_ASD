import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
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

import numpy as np
from PIL import Image


class Loconet(nn.Module):

    def __init__(self, cfg):
        super(Loconet, self).__init__()
        self.cfg = cfg
        self.model = locoencoder(cfg)
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
        self.criterion = nn.CrossEntropyLoss()

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

        # select only the negatives the negatives
        negatives = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

        logits = logits / 0.07
        return logits, labels

    def forward(self, audioFeature, visualFeature, labels, masks):
        b, s, t = visualFeature.shape[:3]
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])

        audioEmbed = self.model.forward_audio_frontend(audioFeature)    # B, C, T, 4
        visualEmbed = self.model.forward_visual_frontend(visualFeature)
        audioEmbed = audioEmbed.repeat(s, 1, 1)

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
                nce_loss = self.criterion(logits, labels_nce) #input, target
            else:
                nce_loss=0

        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)

        labels = labels.reshape((-1))
        masks = masks.reshape((-1))
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, masks)
        nlossA = self.lossA.forward(outsA, labels, masks)
        nlossV = self.lossV.forward(outsV, labels, masks)

        nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV + 0.3 * nce_loss

        num_frames = masks.sum()
        return nloss, prec, num_frames


class loconet(nn.Module):

    def __init__(self, cfg, rank=None, device=None):
        super(loconet, self).__init__()
        self.cfg = cfg
        self.rank = rank
        if rank != None:
            self.rank = rank
            self.device = device

            self.model = Loconet(cfg).to(device)
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
            self.model = locoencoder(cfg).cuda()
            self.lossAV = lossAV().cuda()
            self.lossA = lossA().cuda()
            self.lossV = lossV().cuda()

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

        for num, (audioFeature, visualFeature, labels, masks) in pbar:

            audioFeature = audioFeature.to(device)
            visualFeature = visualFeature.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            nloss, prec, num_frames = self.model(
                audioFeature,
                visualFeature,
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
        acc = top1 / index
        dist.barrier()
        return loss / num, lr, acc

    def evaluate_network(self, epoch, loader):
        self.eval()
        predScores = []
        evalCsvSave = os.path.join(self.cfg.WORKSPACE, "{}_res_shifted_{}.csv".format(epoch, self.cfg.shift_factor))
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
                visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
                labels = labels.view(b * s, *labels.shape[2:])
                masks = masks.view(b * s, *masks.shape[2:])
                audioEmbed = self.model.forward_audio_frontend(audioFeature)
                visualEmbed = self.model.forward_visual_frontend(visualFeature)
                audioEmbed = audioEmbed.repeat(s, 1, 1)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
                labels = labels.reshape((-1))
                masks = masks.reshape((-1))
                outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                labels = labels.view(b, s, t)[:, 0, :].view(b * t).cuda()
                masks = masks.view(b, s, t)[:, 0, :].view(b * t)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)
                predScore = predScore[:, 1].detach().cpu().numpy()
                assert predScore.shape[0] == t
                x = len(predScores)
                predScores.extend(predScore)
                assert(len(predScores) == x + t)
        evalLines = open(evalOrig).read().splitlines()[1:]
        print(len(predScores), len(evalLines))
        assert (len(predScores) == len(evalLines))
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

    def evaluate(self, epoch, loader):
        self.eval()
        predScores = []
        evalCsvSave = os.path.join(self.cfg.WORKSPACE, "{}_res_shifted_{}.csv".format(epoch, self.cfg.shift_factor))
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
                visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
                labels = labels.view(b * s, *labels.shape[2:])
                masks = masks.view(b * s, *masks.shape[2:])
                audioEmbed = self.model.forward_audio_frontend(audioFeature)
                visualEmbed = self.model.forward_visual_frontend(visualFeature)
                audioEmbed = audioEmbed.repeat(s, 1, 1)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
                labels = labels.reshape((-1))
                masks = masks.reshape((-1))
                outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                labels = labels.view(b, s, t)[:, 0, :].view(b * t).cuda()
                masks = masks.view(b, s, t)[:, 0, :].view(b * t)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' if score >= 0.5 else 'NOT_SPEAKING' for score in predScores])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)

    def evaluate_grad_cam(self, epoch, loader):
        self.eval()
        predScores = []
        savePath = os.path.join(self.cfg.WORKSPACE, "grad_cam")
        if os.path.exists(savePath):
            shutil.rmtree(savePath)
        os.makedirs(savePath)
        cnt = 0
        for id, (audioFeature, audioReverseFeature, visualFeature, labels, masks) in tqdm.tqdm(enumerate(loader)):
            if id < 1:
                continue

            visualFeatureOriginal = deepcopy(visualFeature)
            
            audioReverseFeature = audioReverseFeature.cuda()
            visualFeature = visualFeature.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
            visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
            labels = labels.view(b * s, *labels.shape[2:])
            masks = masks.view(b * s, *masks.shape[2:])
            audioEmbed = self.model.forward_audio_frontend(audioReverseFeature)
            visualEmbed = self.model.forward_visual_frontend(visualFeature)
            audioEmbed = audioEmbed.repeat(s, 1, 1)
            audioEmbed, visualEmbed = self.model.forward_cross_attention(
                audioEmbed, visualEmbed)
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
            outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
            predScore= self.lossAV.forward(outsAV, None, None)
            predScoreNum = predScore[:, 1].detach().cpu().numpy()
            wrong_indices = numpy.argwhere(predScoreNum >= 0)

            print(wrong_indices)
            
            if wrong_indices.size != 0:
                
                
                # get the gradient of the output with respect to the parameters of the model
                loss = torch.sum(predScore[:, 1])
                loss.backward()

                # pull the gradients out of the model
                gradients = self.model.visualFrontend.get_activations_gradient()

                # pool the gradients across the channels
                pooled_gradients = torch.mean(gradients, dim=[2, 3])

                visualReverseCopy = deepcopy(visualFeatureOriginal)

                # get the activations of the last convolutional layer
                visualReverseCopy = visualReverseCopy.view(b * s, *visualReverseCopy.shape[2:]).cuda()
                B, T, W, H = visualReverseCopy.shape
                visualReverseCopy = visualReverseCopy.view(B * T, 1, 1, W, H)
                visualReverseCopy = (visualReverseCopy / 255 - 0.4161) / 0.1688
                activations = self.model.visualFrontend.get_activations(visualReverseCopy).detach()
                # weight the channels by corresponding gradients
                for i in range(activations.shape[1]):
                    for j in range(activations.shape[0]):
                        activations[j, i, :, :] *= pooled_gradients[j, i]

                grad_cam = torch.mean(activations, dim = 1)
                grad_cam = torch.nn.functional.relu(grad_cam)
                grad_cam = grad_cam.reshape(s, t, *grad_cam.shape[1:])[0]

                for index in tqdm.tqdm(wrong_indices):
                    heatmap = grad_cam[index].detach().cpu().numpy()
                    heatmap /= numpy.max(heatmap)
                    heatmap = heatmap.swapaxes(0,1).swapaxes(1, 2)
                    heatmap = cv2.resize(heatmap, (W, H))
                    heatmap = numpy.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = heatmap * 0.4 + visualFeatureOriginal[0, 0, index].transpose(0,1).transpose(1, 2).cpu().numpy()
                    cv2.imwrite(os.path.join(savePath, f"wrong_detecton_frame_{index}.png"), superimposed_img)

                visual = deepcopy(visualFeatureOriginal)
                audioFeature = audioFeature.cuda()
                visual = visual.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                b, s, t = visual.shape[0], visual.shape[1], visual.shape[2]
                visual = visual.view(b * s, *visual.shape[2:])
                audio_embed = self.model.forward_audio_frontend(audioFeature)
                visual_embed = self.model.forward_visual_frontend(visual)
                audio_embed = audio_embed.repeat(s, 1, 1)
                audio_embed, visual_embed = self.model.forward_cross_attention(
                    audio_embed, visual_embed)
                outs_AV = self.model.forward_audio_visual_backend(audio_embed, visual_embed, b, s)
                outs_AV = outs_AV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                pred_score= self.lossAV.forward(outs_AV, None, None)

                # get the gradient of the output with respect to the parameters of the model
                loss1 = torch.sum(pred_score[:, 1])
                loss1.backward()

                # pull the gradients out of the model
                gradients = self.model.visualFrontend.get_activations_gradient()

                # pool the gradients across the channels
                pooled_gradients = torch.mean(gradients, dim=[2, 3])

                visualCopy = deepcopy(visualFeatureOriginal)

                # get the activations of the last convolutional layer
                visualCopy = visualCopy.view(b * s, *visualCopy.shape[2:]).cuda()
                B, T, W, H = visualCopy.shape
                visualCopy = visualCopy.view(B * T, 1, 1, W, H)
                visualCopy = (visualCopy / 255 - 0.4161) / 0.1688
                activations = self.model.visualFrontend.get_activations(visualCopy).detach()
                # weight the channels by corresponding gradients
                for i in range(activations.shape[1]):
                    for j in range(activations.shape[0]):
                        activations[j, i, :, :] *= pooled_gradients[j, i]

                grad_cam = torch.mean(activations, dim = 1)
                grad_cam = torch.nn.functional.relu(grad_cam)
                grad_cam = grad_cam.reshape(s, t, *grad_cam.shape[1:])[0]

                for index in tqdm.tqdm(wrong_indices):
                    heatmap = grad_cam[index].detach().cpu().numpy()
                    heatmap /= numpy.max(heatmap)
                    heatmap = heatmap.swapaxes(0,1).swapaxes(1, 2)
                    heatmap = cv2.resize(heatmap, (W, H))
                    heatmap = numpy.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = heatmap * 0.4 + visualFeatureOriginal[0, 0, index].transpose(0,1).transpose(1, 2).cpu().numpy()
                    cv2.imwrite(os.path.join(savePath, f"original_detecton_frame_{index}_label_{labels[0, index].item()}.png"), superimposed_img)
                
                for index in wrong_indices:
                    images = [Image.open(x) for x in [os.path.join(savePath, f"original_detecton_frame_{index}_label_{labels[0, index].item()}.png"), os.path.join(savePath, f"wrong_detecton_frame_{index}.png")]]
                    widths, heights = zip(*(i.size for i in images))
                    total_width = sum(widths)
                    max_height = max(heights)
                    new_im = Image.new('RGB', (total_width, max_height))
                    x_offset = 0
                    for im in images:
                        new_im.paste(im, (x_offset,0))
                        x_offset += im.size[0]
                    predicted = 1 if pred_score[index, 1] >= 0 else 0

                    img = numpy.asarray(new_im)
                    plt.imshow(img)
                    plt.title(f"Normal vs reverse audio grad-cam frame: {index}. GT: {labels[0, index].item()}, Pred: {predicted}")
                    plt.savefig(os.path.join(savePath, f"combined_grad_cam_original_{index}_label_{labels[0, index].item()}_predict_{predicted}.png"))
                    plt.close()

                fp_in = os.path.join(savePath, "combined_*.png")
                fp_out = os.path.join(savePath, "combine.gif")

                # use exit stack to automatically close opened images
                with contextlib.ExitStack() as stack:

                    # lazily load images
                    imgs = (stack.enter_context(Image.open(f))
                            for f in sorted(glob.glob(fp_in)))

                    # extract  first image from iterator
                    img = next(imgs)

                    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
                    img.save(fp=fp_out, format='GIF', append_images=imgs,
                            save_all=True, duration=800, loop=0)
            
            predScores.extend(predScoreNum)
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
                new_state[k.replace("model.module.", "")] = v
            info = self.load_state_dict(new_state, strict=False)
        print(info)
