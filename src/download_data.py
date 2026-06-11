import os
from pathlib import Path

def download_dataset():
    # Determine the project root directory
    # (src/my_code/download_data.py -> src/my_code -> src -> project_root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    res_dir = project_root / "res" / "dataset"
    
    print(f"Target directory for dataset: {res_dir.resolve()}")
    
    # We will use huggingface_hub to download the repository files
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Installing it now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    print("Downloading dataset 'AIML-TUDA/dlam-ts-project-data-2026' from Hugging Face...")
    
    # Download files to the 'res' folder
    snapshot_download(
        repo_id="AIML-TUDA/dlam-ts-project-data-2026",
        repo_type="dataset",
        local_dir=res_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=[".git*", "README.md"]  # Ignore git metadata and Hugging Face README
    )
    
    print("Download completed successfully! The files are saved in the 'res' folder.")

if __name__ == "__main__":
    download_dataset()
