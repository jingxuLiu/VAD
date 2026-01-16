import os
import sys
import json
import argparse
import numpy as np
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
from models import BrainNetwork, EmotionNetwork, Inception_Extension
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder


class GlobalBehaviorDataset(Dataset):
    def __init__(self, subjects, datadir, batch_size):
        self.subjects = subjects
        self.samples = []
        
        for subj_id in subjects:
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
                        i//batch_size
                    ))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fmri, images, win_idx = self.samples[idx]
        score_idx = win_idx % behav_scores.shape[0]
        return torch.from_numpy(fmri), torch.from_numpy(images), behav_scores[score_idx].float()

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

def train(args):
    print("Begin Training!")
    data_type = torch.float
    device = torch.device(args.device)
    print(device)

    log_file_path = os.path.join(args.outdir, 'training_logstruth.txt')
    subj_list = [
        filename.split('.')[0]
        for filename in os.listdir(args.datadir)
        if filename.endswith('.nii.gz')
    ]
    subj_list = sorted(subj_list, key=lambda x: (x[0], int(x[1:])))
    behav_scores = np.load(args.behavdir)
    behav_scores = torch.from_numpy(behav_scores[1:, :]).to("cpu").to(data_type)  # 去掉前十秒第一个clip (46,2)
    image_data = torch.load(args.imgdir).to("cpu").to(data_type) / 255.0
    print("Data ok!")

    train_subjs = subj_list[:70]
    train_set = GlobalBehaviorDataset(train_subjs, args.datadir, args.batch_size)
    print("Dataset ok")
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size_all,
        shuffle=True,
        drop_last=False,
        pin_memory=True
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
    
    def save_ckpt(tag):
        ckpt_path = args.outdir+f'/reconweight/{tag}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
        }, ckpt_path)
        print(f"\n---saved {args.outdir}/{tag} ckpt!---\n")

    def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=args.outdir): 
        print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
        checkpoint = torch.load(outdir+'/reconweight'+f'/{tag}.pth', map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=strict)
        if load_epoch:
            epoch = checkpoint['epoch']
            print("Epoch",epoch)
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_lr:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        del checkpoint
        return epoch

    print("\nDone with model preparations!")
    num_params = utils.count_params(model)

    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)
    total_steps = args.num_epochs * len(train_loader)
    if args.lr_scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps,
        last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, total_steps=total_steps,
            pct_start=2/args.num_epochs, final_div_factor=1000,
            last_epoch=-1
        )
    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_train_loss = 1e9
    best_test_loss = 1e9
    # loss functions
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    progress_bar = tqdm(range(epoch, args.num_epochs), ncols=1200, disable=(args.local_rank != 0))
    
    for epoch in progress_bar:
        model.train()
        
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        
        loss_clip_total = 0.
        loss_emo_total = 0.
        train_i = 0
        print("start")
        for samples, targets, emo in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss = 0.0
            train_i += 1
            samples = samples.to(device)   
            samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
            samples = samples.unsqueeze(1) 
            targets = targets.to(device)
            targets = targets.view(-1, *targets.shape[2:])
            emo = emo.to(device)
            # forward
            samples = model.fmri(samples)
            if epoch < int(args.mixup_pct * args.num_epochs):
                samples, perm, betas, select = utils.mixco(samples)
            clip_target = clip_img(targets).to(data_type)
            assert not torch.any(torch.isnan(clip_target))

            backbone, clips = model.backbone(samples)
            if args.clip_scale > 0:
                clip_voxels_norm = nn.functional.normalize(clips.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                if epoch < int(args.mixup_pct * args.num_epochs):
                    clip_voxels_norm = clip_voxels_norm.to(data_type)
                    clip_target_norm = clip_target_norm.to(data_type)
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                    clip_voxels_norm = clip_voxels_norm.to(data_type)
                    clip_target_norm = clip_target_norm.to(data_type)
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)
                        
                loss_clip_total += loss_clip.item()
                loss_clip *= args.clip_scale
                loss += loss_clip
            
            if args.use_emo:
                clips = clips.flatten(1).unsqueeze(1)
                finals = model.emotion(clips)
                loss_emo = mse(emo, finals)
                loss_emo_total += loss_emo.item()
                loss_emo *= args.emo_scale
                loss += loss_emo

            if args.clip_scale > 0:
                # forward and backward top 1 accuracy
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                                labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                                labels, k=1).item()

            # backward
            utils.check_loss(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            if args.lr_scheduler_type is not None:
                scheduler.step()
        
        print(f'Epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]:.6f}, loss: {loss.item():.4f}, loss_mean: {np.mean(losses[-(train_i+1):]):.4f}')
                
        model.eval()
        with torch.no_grad():
            logs = {"train/loss": np.mean(losses[-(train_i + 1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "train/loss_clip_total": loss_clip_total / (train_i + 1),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "train/loss_emo_total": loss_emo_total / (train_i + 1)
                    }

            log_str = f"Epoch {epoch} | "
            log_str += " | ".join([f"{k}: {v:.6f}" if isinstance(v, (float, np.float32, np.float64))
                                    else f"{k}: {v}" for k, v in logs.items()])
                    
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {log_str}\n"
            with open(log_file_path, 'a') as f:
                f.write(log_entry)
                    
            progress_bar.set_postfix(**logs)
        
        if np.mean(losses[-(train_i + 1):]) <= best_train_loss:
            best_train_loss = np.mean(losses[-(train_i + 1):])
            save_ckpt(f'ckpttruth')
        print("finished training Epoch %d" % epoch)
    
    print("\n===Finished!===\n")
    checkpoint = torch.load(args.outdir+f'/reconweight/ckpttruth.pth', map_location='cpu')
    best_epoch = checkpoint['epoch']
    print("The train_ckpt saved is epoch %d" % best_epoch)
    del checkpoint

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
    parser.add_argument("--mixup_pct",type=float,default=0.33)
    parser.add_argument("--blur_scale",type=float,default=0.5)
    parser.add_argument("--clip_scale",type=float,default=1.)
    parser.add_argument("--emo_scale",type=float,default=1.)
    parser.add_argument("--prior_scale",type=float,default=30)
    parser.add_argument("--num_epochs",type=int,default=100)
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
    parser.add_argument('--device', type=str, default='cuda:6', help='Override device if not distributed')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help='Local rank passed by torch.distributed.launch or torchrun')
    parser.add_argument('--local-rank', dest='local_rank', type=int,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()
    utils.seed_everything(args.seed)

    data_type = torch.float
    print(args.local_rank)
    subj_list = [
        filename.split('.')[0]
        for filename in os.listdir(args.datadir)
        if filename.endswith('.nii.gz')
    ]
    subj_list = sorted(subj_list, key=lambda x: (x[0], int(x[1:])))
    behav_scores = np.load(args.behavdir)
    behav_scores = torch.from_numpy(behav_scores[1:, :]).to("cpu").to(data_type)  # 去掉前十秒第一个clip (46,2)
    image_data = torch.load(args.imgdir).to("cpu").to(data_type)
    print("Data ok!")
    train_subjs = subj_list[:70]
    train_set = GlobalBehaviorDataset(train_subjs, args.datadir, args.batch_size)
    print("Dataset ok")
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size_all,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    os.makedirs(args.outdir, exist_ok=True)
    train(args)
