#!/usr/bin/env python3
"""
Download audio files from Hugging Face afrispeech-dialog dataset.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Configuration
REPO_ID = "intronhealth/afrispeech-dialog"
REPO_TYPE = "dataset"
DATA_FOLDER = "data"
OUTPUT_DIR = "Audio"
HF_TOKEN = os.getenv("HF_TOKEN", None)
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]

def get_audio_files_from_repo(repo_id, repo_type, data_folder, token=None):
    """List all audio files in the repository."""
    print(f"Listing files in {repo_id}/{data_folder}...")
    
    list_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": data_folder
    }
    if token:
        list_kwargs["token"] = token
    
    try:
        files = list_repo_files(**list_kwargs)
        # Filter for .wav files
        audio_files = [f for f in files if f.endswith('.wav')]
        print(f"Found {len(audio_files)} audio files")
        return audio_files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def download_audio_files(repo_id, repo_type, data_folder, file_names, output_dir, token=None):
    """Download audio files directly from Hugging Face Hub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(file_names)} files to {output_dir}...")
    
    downloaded = 0
    failed = 0
    
    for file_name in tqdm(file_names, desc="Downloading"):
        try:
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": file_name,
                "local_dir": output_dir,
                "local_dir_use_symlinks": False
            }
            if token:
                download_kwargs["token"] = token
            
            hf_hub_download(**download_kwargs)
            downloaded += 1
        except Exception as e:
            print(f"\nError downloading {file_name}: {e}")
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed: {failed}")

def main():
    token = HF_TOKEN
    
    if not token:
        print("\nNote: No token provided. If you get authentication errors, you can:")
        print("  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token")
        print("  2. Pass token as argument: python download_afrispeech_dialog_audio.py your_token")
        print()
    
    # Get list of audio files from the repository
    audio_files = get_audio_files_from_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        data_folder=DATA_FOLDER,
        token=token
    )
    
    if not audio_files:
        print("No audio files found or error occurred. Exiting.")
        return
    
    # Download the files
    download_audio_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        data_folder=DATA_FOLDER,
        file_names=audio_files,
        output_dir=OUTPUT_DIR,
        token=token
    )

if __name__ == "__main__":
    main()


