import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import math
from einops import rearrange
import time
import random
import string
import h5py
import SimpleITK as sitk
from tqdm import tqdm
import webdataset as wds
import logging
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from engine import train_one_epoch, evaluate
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import utils
from models import Clipper
from models import BrainNetwork, EmotionNetwork, Inception_Extension, RidgeRegression
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from collections import OrderedDict

class GlobalBehaviorDataset(Dataset):
    def __init__(self, subjects, datadir, batch_size):
        self.subjects = subjects
        self.samples = []
        
        for subj_id in subjects:
            # 0:正常人 1:抑郁症
            if subj_id[0] == 'H':
                label = 0
            elif subj_id[0] == 'M':
                label = 1
            img = sitk.ReadImage(os.path.join(datadir, f"{subj_id}.nii.gz"))
            fmri_data = torch.from_numpy(sitk.GetArrayFromImage(img)/10000.0).to("cpu").to(data_type)
            mean_voxel = fmri_data.mean(dim=0, keepdim=True)
            fmri_data = (fmri_data - mean_voxel) / (mean_voxel + 1e-8) * 100
            mu = fmri_data.mean(dim=0, keepdim=True)
            sigma = fmri_data.std (dim=0, unbiased=False, keepdim=True) + 1e-8
            fmri_data = (fmri_data - mu) / sigma
            
            for i in range(0, 230, batch_size):
                fmri_window = fmri_data[i:i+batch_size]
                image_window = image_data[i*10:(i+batch_size)*10]
                if fmri_window.shape[0] == batch_size:
                    self.samples.append((
                        fmri_window.numpy(),
                        image_window.numpy(),
                        i//batch_size,
                        label
                    ))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fmri, images, win_idx, label = self.samples[idx]
        score_idx = win_idx % behav_scores.shape[0]
        return torch.from_numpy(fmri), torch.from_numpy(images), behav_scores[score_idx].float(), label

class MDDModule(nn.Module):
    def __init__(self, volumn_dim=13148,hidden_dim=2048, n_blocks=4,
                 clip_seq_dim=256, clip_emb_dim=1664, patch_size=450,
                 clip_scale=1.0, drop_rate=0.1, batch_size=5, device=torch.device):
        super(MDDModule, self).__init__()
        self.volumn_dim = volumn_dim
        self.clip_seq_dim = clip_seq_dim
        self.clip_emb_dim = clip_emb_dim
        self.patch_size = patch_size
        self.clip_scale = clip_scale

        self.backbone = BrainNetwork(h=hidden_dim, in_dim=volumn_dim, seq_len=1, n_blocks=n_blocks,
                        clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                        clip_scale=clip_scale, num_tokens=512, patch_size=patch_size, 
                        drop_rate=drop_rate)
        self.fmri = Inception_Extension(h=256, in_dim=volumn_dim, out_dim=volumn_dim, expand=batch_size*2, seq_len=1)
        utils.count_params(self.backbone)
        utils.count_params(self.fmri)
        self.emotion = EmotionNetwork(h=256, in_dim=clip_emb_dim*clip_seq_dim, out_dim=2)
        utils.count_params(self.emotion)

    def forward(self, x):
        return x

def test(args):
    print("Begin Testing!")
    data_type = torch.float
    device = torch.device(args.device)
    print(device)
    log_file_path = os.path.join(args.outdir, 'test_logstruth.txt')
    subj_list = [
        filename.split('.')[0]
        for filename in os.listdir(args.datadir)
        if filename.endswith('.nii.gz')
    ]
    subj_list = sorted(subj_list, key=lambda x: (x[0], int(x[1:])))
    behav_scores = np.load(args.behavdir)  # (47,2)
    behav_scores = torch.from_numpy(behav_scores[1:, :]).to("cpu").to(data_type)  # 去掉前十秒第一个clip (46,2)
    image_data = torch.load(args.imgdir).to("cpu").to(data_type) / 255.0
    print("Data ok!")
    test_subjs = subj_list
    test_set = GlobalBehaviorDataset(test_subjs, args.datadir, args.batch_size)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size_all,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    clip_img = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14", version="laion2b_s39b_b160k",
            output_tokens=True, only_tokens=True)
    clip_img.to(device)

    model = MDDModule(
        volumn_dim=args.volumn_dim,
        hidden_dim=args.hidden_dim, 
        n_blocks=args.n_blocks,
        clip_seq_dim=args.clip_seq_dim, 
        clip_emb_dim=args.clip_emb_dim,
        patch_size=args.patch_size,
        clip_scale=args.clip_scale, 
        drop_rate=args.drop_rate, 
        batch_size=args.batch_size,
        device=device
    ).to(device)
    utils.count_params(model)
    checkpoint = torch.load(f'/home/students/ljx_4090_2/MDD/src/MDDresult/reconweight/ckpttruth.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint

    # get all reconstructions
    model.eval().requires_grad_(False)
    clip_score = 0.0
    emo_score = 0.0
    ano_score = 0.0
    final = []
    clipl = []
    emol = []

    with torch.no_grad():
        for test_i, (samples, targets, emo, label) in enumerate(test_loader):

            samples = samples.to(device)
            samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
            samples = samples.unsqueeze(1)
            targets = targets.to(device)
            targets = targets.view(-1, *targets.shape[2:])
            emo = emo.to(device)
            samples = model.fmri(samples)
            clip_target = clip_img(targets).to(data_type)
            assert not torch.any(torch.isnan(clip_target))

            backbone, clips = model.backbone(samples)
            if args.clip_scale > 0:
                clip_voxels_norm = nn.functional.normalize(clips.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                clip_voxels_norm = clip_voxels_norm.to(data_type)
                clip_target_norm = clip_target_norm.to(data_type)
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)
                loss_clip *= args.clip_scale
                clipl.append(loss_clip)
                clip_score += loss_clip.item()
                               
            if args.emo_scale > 0:
                clips = clips.flatten(1).unsqueeze(1)
                finals = model.emotion(clips)
                final.append(finals)
                mse = torch.mean((finals - emo) ** 2)
                mse *= args.emo_scale
                emol.append(mse)
                emo_score += mse
                
            if (test_i+1)*args.batch_size_all % 46 == 0:
                final = torch.cat(final, dim=0)
                clipl = torch.vstack(clipl)
                emol = torch.vstack(emol)
                new_column = torch.full((46, 1), (test_i+1)*args.batch_size_all // 46).to(device)
                final_new = torch.cat([new_column, final, clipl, emol], dim=1)
                df = pd.DataFrame(final_new.cpu().numpy())
                file_path = '/home/students/ljx_4090_2/MDD/src/emoresulttruth.csv'
                if not os.path.exists(file_path):
                    df.to_csv(file_path, index=False, header=False, mode='w')
                else:
                    df.to_csv(file_path, index=False, header=False, mode='a')
                final = []
                clipl = []
                emol = []
                ano_score = clip_score + emo_score
                print(f"subj {(test_i+1)*args.batch_size_all // 46}/200 | label {label.item()} | ano_score {ano_score / 46}")
                logs = {"clip/score": clip_score / 46,
                        "emo/score": emo_score / 46,
                        "ano/score": ano_score / 46,
                        "label": label.item()
                        }
                log_str = f"subj {(test_i+1)*args.batch_size_all // 46}/200 | "
                log_str += " | ".join([f"{k}: {v:.6f}" if isinstance(v, (float, np.float32, np.float64))
                                       else f"{k}: {v}" for k, v in logs.items()])
                
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = f"[{timestamp}] {log_str}\n"

                # 写入文件
                with open(log_file_path, 'a') as f:
                    f.write(log_entry)

                clip_score = 0.0
                emo_score = 0.0
                ano_score = 0.0
    
    print("\n===Finished!===\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/students/ljx_4090_2/MDD")
    parser.add_argument("--datadir", type=str, default="/home/students/ljx_4090_2/MDD/fsaverage5")
    parser.add_argument("--imgdir", type=str, default="/home/students/ljx_4090_2/MDD/image_tensor.pt")
    parser.add_argument("--behavdir", type=str, default="/home/students/ljx_4090_2/MDD/behav.npy")
    parser.add_argument("--outdir", type=str, default="/home/students/ljx_4090_2/MDD/src/MDDresult")
    parser.add_argument("--use_emo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--batch_size_all", type=int, default=1)
    parser.add_argument("--mixup_pct",type=float,default=0.)
    parser.add_argument("--blur_scale",type=float,default=0.5)
    parser.add_argument("--clip_scale",type=float,default=1.)
    parser.add_argument("--emo_scale",type=float,default=100.)
    parser.add_argument("--prior_scale",type=float,default=30)
    parser.add_argument("--num_epochs",type=int,default=150)
    parser.add_argument("--n_blocks",type=int,default=4)
    parser.add_argument("--hidden_dim",type=int,default=1024)
    parser.add_argument('--clip_seq_dim', type=int, default=256)
    parser.add_argument('--clip_emb_dim', type=int, default=1664)
    parser.add_argument("--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'])
    parser.add_argument("--ckpt_saving",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--ckpt_interval",type=int,default=50)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max_lr",type=float,default=3e-4)
    parser.add_argument("--patch_size", type=int, default=450)
    parser.add_argument("--train_val_split_ratio",type=float,default=0.9)
    parser.add_argument("--volumn_dim",type=int,default=13148)
    parser.add_argument("--time_steps",type=int,default=230)
    parser.add_argument("--drop_rate",type=float,default=0.1)
    parser.add_argument('--device', type=str, default='cuda:4', help='Override device if not distributed')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    # Initialize dataset and model
    # prepare data
    data_type = torch.float
    behav_scores = np.load(args.behavdir)
    behav_scores = torch.from_numpy(behav_scores[1:, :]).to("cpu").to(data_type)  # 去掉前十秒第一个clip (46,2)
    image_data = torch.load(args.imgdir).to("cpu").to(data_type)

    os.makedirs(args.outdir, exist_ok=True)
    test(args)