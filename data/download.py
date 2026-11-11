import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id, branch, exclude_dirs, exclude_files = "pscotti/mindeyev2", "main", ["train_logs", "evals"], ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]

include_specific_files = ["evals/all_images.pt", "evals/all_captions.pt", "evals/all_git_generated_captions.pt"]

def download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files):
    files = list_repo_files(repo_id, repo_type="dataset", revision=branch)
    for file_path in files:
        if (not any(ex_dir in file_path for ex_dir in exclude_dirs) or file_path in include_specific_files) and not any(ex_file in file_path for ex_file in exclude_files):
            hf_hub_download(repo_id, filename=file_path, repo_type="dataset", revision=branch, local_dir=os.getcwd())

download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files)