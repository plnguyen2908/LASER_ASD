import time, os, torch, argparse, warnings, glob, pandas, json

from utils.tools import *
from dlhammer.dlhammer import bootstrap
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb

# from xxlib.utils.distributed import all_gather, all_reduce
from torch import nn
from dataLoader_multiperson_landmark import train_loader, val_loader

from landmark_loconet import loconet
import argparse

class MyCollator(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, data):
        audiofeatures = [item[0] for item in data]
        visualfeatures = [item[1] for item in data]
        landmarks = [item[2] for item in data]
        labels = [item[3] for item in data]
        masks = [item[4] for item in data]
        cut_limit = self.cfg.MODEL.CLIP_LENGTH
        # pad audio
        lengths = torch.tensor([t.shape[1] for t in audiofeatures])
        max_len = max(lengths)
        padded_audio = torch.stack([
            torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2]))], 1)
            for i in audiofeatures
        ], 0)

        if max_len > cut_limit * 4:
            padded_audio = padded_audio[:, :, :cut_limit * 4, ...]

        # pad video
        lengths = torch.tensor([t.shape[1] for t in visualfeatures])
        max_len = max(lengths)
        padded_video = torch.stack([
            torch.cat(
                [i, i.new_zeros((i.shape[0], max_len - i.shape[1], i.shape[2], i.shape[3]))], 1)
            for i in visualfeatures
        ], 0)
        padded_landmark = torch.stack([
            torch.cat(
                [i, i.new_full((i.shape[0], max_len - i.shape[1], i.shape[2], i.shape[3]), -1)], 1
            ) for i in landmarks
        ], 0)
        padded_labels = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in labels], 0)
        padded_masks = torch.stack(
            [torch.cat([i, i.new_zeros((i.shape[0], max_len - i.shape[1]))], 1) for i in masks], 0)

        if max_len > cut_limit:
            padded_video = padded_video[:, :, :cut_limit, ...]
            padded_labels = padded_labels[:, :, :cut_limit, ...]
            padded_masks = padded_masks[:, :, :cut_limit, ...]
            padded_landmark = padded_landmark[:,:,:cut_limit,...]

        return padded_audio, padded_video, padded_landmark, padded_labels, padded_masks


class DataPrep():

    def __init__(self, cfg, world_size, rank):
        self.cfg = cfg
        self.world_size = world_size
        self.rank = rank

    def train_dataloader(self):
        if self.cfg.only_landmark:
            print("only_landmark")

        loader = train_loader(self.cfg, trialFileName = self.cfg.trainTrialAVA, \
                          audioPath      = os.path.join(self.cfg.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(self.cfg.visualPathAVA, 'train'), \
                          num_speakers=self.cfg.MODEL.NUM_SPEAKERS,
                          )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            loader, num_replicas=self.world_size, rank=self.rank)
        collator = MyCollator(self.cfg)
        trainLoader = torch.utils.data.DataLoader(loader,
                                                  batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                                  pin_memory=False,
                                                  num_workers=self.cfg.NUM_WORKERS,
                                                  collate_fn=collator,
                                                  sampler=train_sampler)
        return trainLoader

    def val_dataloader(self):
        loader = val_loader(self.cfg, trialFileName = self.cfg.evalTrialAVA, \
                            audioPath     = os.path.join(self.cfg
                                .audioPathAVA , self.cfg
                                .evalDataType), \
                            visualPath    = os.path.join(self.cfg
                                .visualPathAVA, self.cfg
                                .evalDataType), \
                            num_speakers = self.cfg.MODEL.NUM_SPEAKERS
                                )
        valLoader = torch.utils.data.DataLoader(loader,
                                                batch_size=self.cfg.VAL.BATCH_SIZE,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=16)

        return valLoader


def prepare_context_files(cfg):
    path = os.path.join(cfg.DATA.dataPathAVA, "csv")
    for phase in ["train", "val"]:
        if not cfg.only_landmark:
            csv_f = f"{phase}_loader.csv"
            csv_orig = f"{phase}_orig.csv"
        else:
            csv_f = f"{phase}_loader_only_landmark.csv"
            csv_orig = f"{phase}_orig_only_landmark.csv"
        entity_f = os.path.join(path, phase + "_entity.json")
        ts_f = os.path.join(path, phase + "_ts.json")
        # if os.path.exists(entity_f) and os.path.exists(ts_f):
        #     continue
        orig_df = pandas.read_csv(os.path.join(path, csv_orig))
        print(orig_df.shape[0])
        entity_data = {}
        ts_to_entity = {}

        for index, row in orig_df.iterrows():

            entity_id = row['entity_id']
            video_id = row['video_id']
            if row['label'] == "SPEAKING_AUDIBLE":
                label = 1
            else:
                label = 0
            ts = float(row['frame_timestamp'])
            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
            if ts not in entity_data[video_id][entity_id].keys():
                entity_data[video_id][entity_id][ts] = []

            entity_data[video_id][entity_id][ts] = label

            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}
            if ts not in ts_to_entity[video_id].keys():
                ts_to_entity[video_id][ts] = []
            ts_to_entity[video_id][ts].append(entity_id)

        with open(entity_f, 'w') as f:
            json.dump(entity_data, f)

        with open(ts_f, 'w') as f:
            json.dump(ts_to_entity, f)


def main(gpu, world_size):
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    cfg = bootstrap(print_cfg=False)
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    make_deterministic(seed=int(cfg.SEED))
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))

    warnings.filterwarnings("ignore")

    cfg = init_args(cfg)

    data = DataPrep(cfg, world_size, rank)
    
    if cfg.downloadAVA == True:
        preprocess_AVA(cfg)
        quit()
        
    prepare_context_files(cfg)

    cfg.modelSavePath += '_' + str(cfg.n_channel) + '_' + str(cfg.layer)
    if cfg.use_consistency:
        assert cfg.consistency_method in ["kl", "mse"]
        cfg.modelSavePath += f'_consistency_{cfg.consistency_method}_lambda_{cfg.consistency_lambda}'
    if not cfg.use_talknce:
        cfg.modelSavePath += f'_talknce:{cfg.use_talknce}'
    if cfg.use_full_landmark:
        cfg.modelSavePath += f"_full_landmark"
    if cfg.only_landmark:
        cfg.modelSavePath += f"_only_landmark"
    os.makedirs(cfg.modelSavePath, exist_ok=True)
    print(cfg.modelSavePath)
    modelfiles = glob.glob('%s/model_0*.model' % (cfg.modelSavePath))
    modelfiles.sort()
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:10]) + 1
        if not cfg.use_consistency:
            s = loconet(cfg, cfg.n_channel, cfg.layer, rank, device, talknce_lambda=cfg.talknce_lambda)
            s.loadParameters(modelfiles[-1])
        else:
            s = loconet(cfg, cfg.n_channel, cfg.layer, rank, device, cfg.consistency_method, cfg.consistency_lambda, cfg.talknce_lambda)
            s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        if not cfg.use_consistency:
            s = loconet(cfg, cfg.n_channel, cfg.layer, rank, device, talknce_lambda=cfg.talknce_lambda)
        else:
            s = loconet(cfg, cfg.n_channel, cfg.layer, rank, device, cfg.consistency_method, cfg.consistency_lambda, cfg.talknce_lambda)

    if rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Landmark_LoCoNet_consistency",
            # name=f"inject_talknce_nchannel_{cfg.n_channel}_layer_{cfg.layer}_method_{cfg.consistency_method}_lambda_{cfg.consistency_lambda}_talknce:{cfg.use_talknce}",
            name = f"inject_talknce_nchannel_{cfg.n_channel}_layer_{cfg.layer}_full_landmark_{cfg.use_full_landmark}_only_landmark_{cfg.only_landmark}",
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.SOLVER.BASE_LR,
                "architecture": "landmark_talknce_loconet",
                "dataset": "AVA Active Speaker",
                "epochs": 25,
            }
        )

    print("--training--")
    while (1):

        loss, lr, acc = s.train_network(epoch=epoch, loader=data.train_dataloader())

        s.saveParameters(cfg.modelSavePath + "/model_%04d_acc%.02f.model" % (epoch, acc))

        if epoch >= cfg.TRAIN.MAX_EPOCH:
            break

        if rank == 0:
            wandb.log({"acc": acc, "loss": loss, "lr": lr})
        epoch += 1
    
    # if rank == 0:
    #     mAP = s.evaluate_network(epoch=epoch, loader=data.val_dataloader())
    #     wandb.summary['test_accuracy (mAP)'] = mAP
    wandb.finish()


if __name__ == '__main__':

    cfg = bootstrap()
    world_size = cfg.NUM_GPUS    #
    os.environ['MASTER_ADDR'] = '127.0.0.1'    #
    os.environ['MASTER_PORT'] = str(random.randint(4000, 8888))    #
    mp.spawn(main, nprocs=cfg.NUM_GPUS, args=(world_size,))
