# VAD: Visual–Affective Decoding Framework

This is the official code for the paper "Visual-affective decoding enhances neurofunctional objective and generalizable signatures in unseen adolescent depression diagnosis" ([https://www.nature.com/articles/s41746-026-03413-8]()). [**npj Digital Medicine 2026**]


---

## Overview

VAD is a dual-pathway normative encoding framework that learns healthy brain representations from naturalistic movie-watching fMRI, then identifies adolescent major depressive disorder (MDD) as systematic deviations from this norm. The model is trained exclusively on healthy controls (HC) and evaluates both HC and MDD subjects against the learned HC-derived norm.

At test time, the framework produces three scores per subject from the two pathways:
- **Visual decoding score (clip_score)**: SoftCLIP contrastive loss between predicted and ground-truth CLIP embeddings — quantifies visual representation fidelity.
- **Affective decoding score (emo_score)**: MSE between predicted and ground-truth valence/arousal ratings — quantifies emotion prediction error.
- **Anomaly score (ano_score)**: sum of the two scores, reflecting overall deviation from the HC normative model. Each subject is classified as MDD if their anomaly score exceeds a fixed threshold (Youden index determined from internal cross-validation).

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch ≥ 2.0, SimpleITK, einops, accelerate, timm, diffusers, open-clip-torch.

---

## Repository Structure

```
VAD_code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train_vad.py                       # Training script (HC-only, normative encoding)
├── infer_vad.py                       # Inference + anomaly detection + MDD classification
├── generative_models/                 # FrozenOpenCLIPImageEmbedder and Stable Diffusion modules
├── save_models/
│   └── ckpt.pth                       # Pretrained model checkpoint
├── model/
│   ├── models.py                      # Model architectures (BrainNetwork, EmotionNetwork, Inception_Extension)
│   ├── utils.py                       # Loss functions (SoftCLIP, MixCo, cosine similarity, etc.)
│   ├── engine.py                      # Training/evaluation engine utilities
│   └── video_decoder.py               # Video decoder module
└── data/
    ├── video.py                       # Video frame extraction from movie file
    ├── preprocess_fmri.py             # Standalone fMRI preprocessing documentation
    ├── create_voxel_mask.py           # Significant-voxel mask creation from activated_voxels_idx.txt
    ├── activated_voxels_idx.txt       # Indices of significantly activated voxels
    ├── dataset.py                     # Dataset and data loading utilities
    └── download.py                    # Data download helper (HuggingFace Hub)
```

---

## Data Preparation

### Required User-Provided Data

| Item | Description |
|------|-------------|
| `{data_dir}/{subj_id}.nii.gz` | Preprocessed fMRI for each subject in fsaverage5 surface space: `(N_TR, 20484)` shape, where N_TR is the total number of TRs covering the movie duration. The code reads the actual shape at runtime. |
| `movie.mp4` | The naturalistic movie stimulus viewed by subjects. Frame extraction is handled by `data/video.py`. |
| `behav.npy` | Per-segment valence/arousal ratings. Sampled every 5 seconds; shape `(N_TR/5 + 1, 2)` for columns `[valence, arousal]`. The first segment serves as baseline and is excluded from loss computation. |

### Subject Naming Convention

- `H###.nii.gz` — Healthy Controls (HC)
- `M###.nii.gz` — MDD patients

The code automatically infers group labels from the filename prefix.

---

## fMRI Preprocessing Pipeline

The following preprocessing steps are applied **inside the Dataset class** in `train_vad.py` and `infer_vad.py`. A standalone documented reference is provided in `data/preprocess_fmri.py`.

### Step 1: Percent Signal Change Normalization

```python
fmri_data = raw_fmri / 10000.0                                    # Scale raw intensities
mean_voxel = fmri_data.mean(dim=0, keepdim=True)
fmri_data = (fmri_data - mean_voxel) / (mean_voxel + 1e-8) * 100  # % signal change
```

### Step 2: Z-Score Normalization (per voxel)

```python
mu = fmri_data.mean(dim=0, keepdim=True)
sigma = fmri_data.std(dim=0, unbiased=False, keepdim=True) + 1e-8
fmri_data = (fmri_data - mu) / sigma
```

Each voxel is independently standardized to zero mean and unit variance across time.

### Step 3: Temporal Alignment — TRs to Video Segments

The movie is divided into segments of 10 seconds each (the user's behavioral annotation rate of 5 seconds per label means 2 labels per segment in training; the code framework generalizes to arbitrary segment durations). Each segment spans `TRs_per_segment` consecutive TRs (typically 5 TRs at TR = 2 s, i.e., 10 s per segment).

```python
batch_size = TRs_per_segment  # e.g., 5
for i in range(0, N_TR, batch_size):
    fmri_window = fmri_data[i : i + batch_size]
```

The first segment (baseline, frames 0–9) is excluded from behavioral alignment. Behavioral ratings are indexed as `score_idx = window_idx % n_behavior_segments`.

### Step 4: Image Alignment

Video frames are temporally aligned to fMRI windows with a fixed sampling ratio:

```python
frames_per_tr = 2  # e.g., 10 frames per 5-TR segment
image_window = image_data[i * frames_per_tr : (i + batch_size) * frames_per_tr]
```

### Step 5: Voxel Selection

After loading the raw fMRI data in fsaverage5 space, only voxels showing significant stimulus-evoked activation are retained for model input, reducing the dimensionality from the full fsaverage5 vertex count to a parsimonious subset:

```python
# voxel_mask: bool array of shape (20484,) — True at significant voxel indices
fmri_data = fmri_data[:, voxel_mask]
```

This step is applied once per subject at data loading. The mask file is generated by `data/create_voxel_mask.py` based on the provided `activated_voxels_idx.txt`.

---

## Anomaly Score Calculation

At test time, each subject is evaluated against the HC-trained model. Two pathway-level deviation scores are computed, averaged across all segments.

### Visual Pathway Score (clip_score)

```python
clip_voxels_norm = normalize(predicted_CLIP_embedding)
clip_target_norm = normalize(true_CLIP_embedding)
loss_clip = SoftCLIP_loss(clip_voxels_norm, clip_target_norm, temp=0.006)
```

SoftCLIP contrastive loss between the fMRI-predicted CLIP token sequence and the ground-truth video frame CLIP embedding. **Higher clip_score → greater deviation from HC visual representation norm.**

### Affective Pathway Score (emo_score)

```python
predicted_va = model.emotion(predicted_CLIP_features)
loss_emo = MSE(predicted_va, true_valence_arousal)
```

MSE between predicted and ground-truth valence/arousal. **Higher emo_score → greater deviation from HC affective processing norm.**

### Combined Anomaly Score and Classification

```python
ano_score = clip_score * clip_scale + emo_score * emo_scale
```

Since the model is trained only on HC data, higher anomaly scores indicate greater deviation from healthy brain function. Each subject is classified as MDD if their anomaly score exceeds a fixed threshold (the Youden index determined from internal cross-validation, passed via `--ano_threshold`):

```python
predicted_MDD = (ano_score >= ano_threshold).astype(int)  # 1 = MDD, 0 = HC
```

Per-subject scores are accumulated across segments and averaged:

```python
clip_score += loss_clip.item()     # Sum across segments
emo_score += loss_emo               # Sum across segments
# After all segments for a subject:
ano_score = clip_score + emo_score  # Total per subject
avg_clip = clip_score / n_segments
avg_emo  = emo_score / n_segments
avg_ano  = ano_score / n_segments
```

---

## Training

Train the VAD model on HC subjects only (normative encoding):

```bash
python train_vad.py \
    --datadir ./data/fmri \
    --imgdir ./data/image_tensor.pt \
    --behavdir ./data/behav.npy \
    --outdir ./results \
    --device cuda:0 \
    --batch_size 5 \
    --batch_size_all 1 \
    --num_epochs 100 \
    --clip_scale 1.0 \
    --emo_scale 1.0 \
    --max_lr 3e-4 \
    --seed 42
```

**Key Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--volumn_dim` | N_significant | Number of input fMRI voxels (after significant-voxel selection) |
| `--hidden_dim` | 1024 | Hidden dimension of transformer backbone |
| `--n_blocks` | 4 | Number of transformer blocks |
| `--clip_seq_dim` | 256 | CLIP token sequence length |
| `--clip_emb_dim` | 1664 | CLIP embedding dimension (ViT-bigG-14) |
| `--patch_size` | 450 | Patch size for voxel embedding |
| `--batch_size` | 5 | TRs per fMRI input window |
| `--mixup_pct` | 0.33 | Fraction of epochs using MixCo augmentation |
| `--clip_scale` | 1.0 | Weight for visual pathway loss |
| `--emo_scale` | 1.0 | Weight for affective pathway loss |
| `--lr_scheduler_type` | cycle | Learning rate schedule (OneCycleLR or linear) |

The training loop uses MixCo (Mixup Contrastive) augmentation during the initial epochs, then switches to SoftCLIP contrastive loss, with an AdamW optimizer and OneCycleLR scheduler.

---

## Inference

Run inference on all subjects; the model outputs per-subject clip scores, emotion scores, anomaly scores, and binary MDD classification:

```bash
python infer_vad.py \
    --datadir ./data/fmri \
    --imgdir ./data/image_tensor.pt \
    --behavdir ./data/behav.npy \
    --outdir ./results \
    --ckpt_path ./save_models \
    --ano_threshold 13.135 \
    --device cuda:0 \
    --batch_size 5 \
    --batch_size_all 1 \
    --clip_scale 1.0 \
    --emo_scale 100.0 \
    --seed 42
```

The script loads the trained checkpoint from `--ckpt_path/ckpt.pth` and produces:
- Console log (`test_logstruth.txt`): per-subject clip/emo/anomaly scores and label
- CSV output (`emoresulttruth.csv`): predicted valence, predicted arousal, clip_loss, emo_loss per segment
- **Final classification output** (`classification_results.csv`): per-subject anomaly score, binary MDD prediction, and overall accuracy/sensitivity/specificity

---

## Citation

If you use this code, please cite the corresponding paper.
