import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

class talkNet(nn.Module):
    def __init__(self, n_channel, layer, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()        
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.n_channel = n_channel
        self.landmark_bottleneck = nn.Conv2d(in_channels=164, out_channels=n_channel, kernel_size=(1, 1)).cuda()

        # insert before layer
        self.layer = layer

        if layer == 0:
            self.bottle_neck = nn.Conv2d(in_channels=(1 + n_channel), out_channels=1, kernel_size=(1, 1)).cuda()
        elif layer == 1:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1)).cuda()
        elif layer == 2:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1)).cuda()
        elif layer == 3:
            self.bottle_neck = nn.Conv2d(in_channels=(128 + n_channel), out_channels=128, kernel_size=(1, 1)).cuda()
        elif layer == 4:
            self.bottle_neck = nn.Conv2d(in_channels=(256 + n_channel), out_channels=256, kernel_size=(1, 1)).cuda()
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

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

        # change back to 0, 1 if not work
        if b_id.numel() > 0:  # Ensure there are valid indices
            landmarkTensor[b_id, s_id, t_id, lip_id, :, coord_0[b_id, s_id, t_id, lip_id], coord_1[b_id, s_id, t_id, lip_id]] = landmark[b_id, s_id, t_id, lip_id, :]

        landmarkTensor = landmarkTensor.reshape(b * s * t, -1, W, H)

        assert (landmarkTensor.shape[1] == 164)
        return landmarkTensor
    
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
            # if i == 2:
            #     # hook the gradient
            #     if x.requires_grad:
            #         x.register_hook(self.activations_hook)

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

    def train_network(self, loader, epoch, weight, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, landmark, labels) in enumerate(loader, start=1):
            self.zero_grad()
            landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
            landmarkFeature = self.landmark_bottleneck(landmarkFeature.cuda())
            
            visualFeatureOriginal = visualFeature.clone()
            audioFeatureOriginal = audioFeature.clone()
            
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
            visualEmbed = self.forward_visual_frontend(visualFeature[0].cuda(), landmarkFeature.cuda())


            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, predScoreLandmark, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)

            consistency_loss = 0
            audioEmbed = self.model.forward_audio_frontend(audioFeatureOriginal[0].cuda()) # feedForward
            visualEmbed = self.forward_visual_frontend(visualFeatureOriginal[0].cuda(), torch.zeros_like(landmarkFeature).cuda())


            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAVNonLandmark= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            _, predScoreNonLandmark, _, _ = self.lossAV.forward(outsAVNonLandmark, labels)
            consistency_loss = self.kl_loss(predScoreNonLandmark.log(), predScoreLandmark.detach())

            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV + consistency_loss * weight
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, useLandmark=True, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, landmark, labels in tqdm.tqdm(loader):
            with torch.no_grad():   
                landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
                landmarkFeature = self.landmark_bottleneck(landmarkFeature.cuda())
                if not useLandmark:
                    landmarkFeature = torch.zeros_like(landmarkFeature)      

                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
                visualEmbed = self.forward_visual_frontend(visualFeature[0].cuda(), landmarkFeature.cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()     
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP
    
    def evaluate(self, loader, evalCsvSave, evalOrig, useLandmark=True,  **kwargs):
        self.eval()
        predScores = []
        
        for audioFeature, visualFeature, landmark, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                landmarkFeature = self.create_landmark_tensor(landmark, visualFeature.dtype, visualFeature.device)
                landmarkFeature = self.landmark_bottleneck(landmarkFeature.cuda())
                if not useLandmark:
                    landmarkFeature = torch.zeros_like(landmarkFeature)    

                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
                visualEmbed = self.forward_visual_frontend(visualFeature[0].cuda(), landmarkFeature.cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
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
        print(evalOrig)
        print(evalCsvSave)
        return None

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
