import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import ModelEma
from model import utils
import numpy as np
from model.models import Clipper


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    #image_loss = contrastive_loss(similarity.t())
    #return (caption_loss + image_loss)/2
    return caption_loss

def ContrastiveLoss(fmri,image):

    logit_scale=torch.tensor([8]).to(fmri.device)

    fmri = fmri / fmri.norm(p=2, dim=-1, keepdim=True)
    image = image / image.norm(p=2, dim=-1, keepdim=True)

    logit_scale = logit_scale.exp()
    # logits_per_fmri = torch.matmul(fmri.t(), image) * logit_scale  # (768, 197376)
    logits_per_fmri = torch.matmul(fmri, image.t()) * logit_scale  # (64, 64)
    #logits_per_fmri = torch.matmul(fmri, image.t())
    logits_per_image = logits_per_fmri.t()
    
    return clip_loss(logits_per_fmri)



def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, num_epochs: int, mixup_pct: float):

    model.train()
    mse = nn.MSELoss()
    clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=True)
    metric_logger = utils.MetricLogger(delimiter="  ")  # 自定义指标记录器
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10  # 每5个batch打印一次日志
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    for samples, targets, emo in metric_logger.log_every(data_loader, print_freq, header):
        # optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # samples = samples.to(device, non_blocking=True)  # (1, 5, 20484)    
            samples = samples.to(device)
            samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
            samples = samples.unsqueeze(1).half()
            samples = model.fmri(samples)  # (50, 20484) 
            # samples = samples.repeat_interleave(10, dim=0)  # (50, 20484)  
            if epoch < int(mixup_pct * num_epochs):
                samples, perm, betas, select = utils.mixco(samples)
            # targets = targets.to(device, non_blocking=True)  # (b, 50,3,720,1280)
            targets = targets.to(device)
            targets = targets.view(-1, *targets.shape[2:])  # (b*50,3,720,1280)
            targets = clip_extractor.embed_image(targets).float()   # (b*50, 257, 768) 
            # targets = targets.view(-1, 50, 257, 768).reshape(-1, 50, 257 * 768)  # (b, 50, 197376)
            backbone, clips = model.backbone(samples)  # (50, 257, 768)
            clip_voxels_norm = nn.functional.normalize(clips.flatten(1), dim=-1)  # (50, 197376)
            clip_target_norm = nn.functional.normalize(targets.flatten(1), dim=-1)  # (50, 197376)
            
            if epoch < int(mixup_pct * num_epochs):
                clip_voxels_norm = clip_voxels_norm.to(torch.float16)
                clip_target_norm = clip_target_norm.to(torch.float16)
                loss_clip = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
            else:
                epoch_temp = soft_loss_temps[epoch - int(mixup_pct * num_epochs)]
                clip_voxels_norm = clip_voxels_norm.to(torch.float16)
                clip_target_norm = clip_target_norm.to(torch.float16)
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)

            loss = loss_clip
            # loss = ContrastiveLoss(targets, clips)
            # loss = mse(emo, finals)
            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            lr_scheduler.step()
        
            torch.cuda.synchronize(device)
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_prior(model: torch.nn.Module, diffusion_prior: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, model_ema: Optional[ModelEma] = None):

    diffusion_prior.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)     
        targets = targets.to(device, non_blocking=True)
        outputs = model(samples)
        loss, _ = diffusion_prior(text_embed=outputs.unsqueeze(1), image_embed=targets.unsqueeze(1))
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    
        torch.cuda.synchronize(device)
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=True)
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for samples, targets, emo in metric_logger.log_every(data_loader, 5, header):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                samples = samples.to(device, non_blocking=True)  # (1, 5, 20484)    
                samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
                samples = samples.unsqueeze(1).half()
                samples = model.fmri(samples)  # (50, 20484)
                # samples = samples.repeat_interleave(10, dim=0)  # (50, 20484)
                targets = targets.to(device, non_blocking=True)  # (b, 50,3,720,1280)
                targets = targets.view(-1, *targets.shape[2:])  # (b*50,3,720,1280)
                targets = clip_extractor.embed_image(targets).float()   # (b*50, 257, 768)
                backbone, clips = model.backbone(samples)  # (50, 257, 768)

                clip_voxels_norm = nn.functional.normalize(clips.flatten(1), dim=-1)  # (50, 197376)
                clip_target_norm = nn.functional.normalize(targets.flatten(1), dim=-1)  # (50, 197376)
                clip_voxels_norm = clip_voxels_norm.to(torch.float16)
                clip_target_norm = clip_target_norm.to(torch.float16)
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)

                loss = loss_clip
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                torch.cuda.synchronize(device)
                metric_logger.update(loss=loss_value)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_prior(data_loader, model, diffusion_prior, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    diffusion_prior.eval()
    loss = 0
    with torch.no_grad():
        for test_input, target in metric_logger.log_every(data_loader, 10, header):
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            #with torch.cuda.amp.autocast():
            output = model(test_input)
            loss_prior, _ = diffusion_prior(text_embed=output.unsqueeze(1), image_embed=target.unsqueeze(1))
            loss += loss_prior

    return loss
