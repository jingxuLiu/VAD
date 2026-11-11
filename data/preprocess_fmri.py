"""
=============================================================================
fMRI Preprocessing, Temporal Alignment, and Anomaly Score Calculation
=============================================================================
This script documents the exact preprocessing pipeline used in the VAD framework.
It is provided for clarity and reproducibility — the actual preprocessing is
embedded in train_vad.py and infer_vad.py. This script can be run standalone
to verify the pipeline on sample data.

Steps:
1. Load raw fMRI (NIfTI, fsaverage5 surface) — (N_TR, 20484 voxels)
2. Voxel selection: 20484 → 13148 significant voxels
3. Percent signal change normalization
4. Z-score normalization (per voxel)
5. Temporal alignment: N_TR → M segments (batch_size TRs per segment)
6. Anomaly score calculation (clip_score + emo_score) + softmax classification
=============================================================================
"""

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path


def select_significant_voxels(fmri_data, mask_path=None):
    """
    Step 2: Select 13,148 significantly activated voxels from the full 20,484 fsaverage5 vertices.

    The mask (model_13148_mask.npy) is a bool array of shape (20484,) — True at the
    13,148 indices listed in activated_voxels_idx.txt where stimulus-evoked BOLD
    activation exceeded a statistical threshold.

    Parameters
    ----------
    fmri_data : np.ndarray, shape (N_TR, 20484)
        Raw fMRI data in fsaverage5 space
    mask_path : str, optional
        Path to model_13148_mask.npy

    Returns
    -------
    fmri_selected : np.ndarray, shape (N_TR, 13148)
    """
    if mask_path is None:
        mask_path = "model_13148_mask.npy"
    mask = np.load(mask_path)  # bool, shape (20484,)
    return fmri_data[:, mask]


def load_and_preprocess_fmri(nifti_path, mask_path=None):
    """
    Steps 1-4: Load raw fMRI, select voxels, and apply preprocessing.

    Parameters
    ----------
    nifti_path : str
        Path to subject's .nii.gz fMRI file
    mask_path : str, optional
        Path to model_13148_mask.npy

    Returns
    -------
    fmri_normalized : np.ndarray, shape (N_TR, 13148)
        Preprocessed fMRI data, z-scored per voxel
    """
    # Step 1: Load raw fMRI — shape (N_TR, 20484), determined at runtime
    img = sitk.ReadImage(nifti_path)
    fmri_data = sitk.GetArrayFromImage(img).astype(np.float32) / 10000.0

    # Step 2: Voxel selection (20484 → 13148)
    if mask_path is not None or Path("model_13148_mask.npy").exists():
        fmri_data = select_significant_voxels(fmri_data, mask_path)

    # Step 3: Percent signal change
    #   fmri = (fmri - mean_voxel) / (mean_voxel + 1e-8) * 100
    mean_voxel = fmri_data.mean(axis=0, keepdims=True)
    fmri_data = (fmri_data - mean_voxel) / (mean_voxel + 1e-8) * 100

    # Step 4: Z-score per voxel (standardize to zero mean, unit variance)
    mu = fmri_data.mean(axis=0, keepdims=True)
    sigma = fmri_data.std(axis=0, ddof=0, keepdims=True) + 1e-8
    fmri_normalized = (fmri_data - mu) / sigma

    return fmri_normalized


def temporal_align(fmri_normalized, n_tr, batch_size=5):
    """
    Step 5: Temporal alignment — group TRs into video segments.

    Each segment spans batch_size TRs. At TR = 2 s, batch_size = 5 means
    10-second segments matching the behavioral annotation rate.

    Parameters
    ----------
    fmri_normalized : np.ndarray, shape (N_TR, n_voxels)
    n_tr : int
        Total number of TRs to use (truncates to n_tr)
    batch_size : int
        TRs per segment, default 5

    Returns
    -------
    segments : np.ndarray, shape (n_segments, batch_size, n_voxels)
    n_segments : int
    """
    n_segments = n_tr // batch_size
    fmri_truncated = fmri_normalized[:n_segments * batch_size]
    segments = fmri_truncated.reshape(n_segments, batch_size, -1)
    return segments, n_segments


def calculate_anomaly_and_classify(clip_scores, emo_scores):
    """
    Step 6: Anomaly score calculation and softmax classification.

    anomaly_score[i] = clip_score[i] + emo_score[i]

    Since the model is trained only on HC data, higher anomaly scores indicate
    greater deviation from healthy brain function. A softmax function applied
    across the test cohort produces per-subject MDD probability estimates.

    Parameters
    ----------
    clip_scores : np.ndarray, shape (N_subjects,)
        Per-subject visual pathway scores
    emo_scores : np.ndarray, shape (N_subjects,)
        Per-subject affective pathway scores

    Returns
    -------
    anomaly_scores : np.ndarray, shape (N_subjects,)
    mdd_probabilities : np.ndarray, shape (N_subjects,)
        Softmax-normalized MDD probabilities
    predictions : np.ndarray, shape (N_subjects,)
        Binary predictions (1 = MDD) at threshold = 0.5
    """
    anomaly_scores = clip_scores + emo_scores

    # Softmax classification
    anomaly_tensor = torch.tensor(anomaly_scores, dtype=torch.float32)
    mdd_probabilities = F.softmax(anomaly_tensor, dim=0).numpy()
    predictions = (mdd_probabilities > 0.5).astype(int)

    return anomaly_scores, mdd_probabilities, predictions


# ======== Standalone verification ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAD fMRI Preprocessing — Verify the pipeline on a single subject")
    parser.add_argument("--nifti", type=str, required=True,
                        help="Path to subject .nii.gz file")
    parser.add_argument("--mask", type=str, default="model_13148_mask.npy",
                        help="Path to model_13148_mask.npy")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="TRs per video segment")

    args = parser.parse_args()

    print("=" * 60)
    print("VAD fMRI Preprocessing Pipeline — Verification Run")
    print("=" * 60)

    # Steps 1-4: Load, select voxels, preprocess
    print(f"\nLoading: {args.nifti}")
    fmri = load_and_preprocess_fmri(args.nifti, args.mask)
    n_tr_total = fmri.shape[0]
    n_voxels = fmri.shape[1]
    print(f"  Shape after preprocessing: ({n_tr_total}, {n_voxels})")
    print(f"  Global mean: {fmri.mean():.6f}, std: {fmri.std():.6f}")

    # Step 5: Temporal alignment
    segments, n_segments = temporal_align(fmri, n_tr_total, args.batch_size)
    print(f"  Temporal alignment: {n_tr_total} TR → {n_segments} segments "
          f"({args.batch_size} TR/segment)")

    print("\nPipeline verified. Output is ({n_segments}, {batch_size}, {n_voxels}) "
          "segmented fMRI data, ready for the BrainNetwork backbone.")
    print("See the README for full anomaly score calculation and softmax classification details.")
