#!/usr/bin/env python3
"""
Download audio files from Hugging Face dataset repository.
"""
import csv
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# Configuration
REPO_ID = "intronhealth/afrispeech-parliament"
REPO_TYPE = "dataset"
DATA_FOLDER = "data"
METADATA_CSV = "Entailment/metadata_afrispeech-parliament.csv"
OUTPUT_DIR = "Audio"

# Token can be passed as environment variable HF_TOKEN or as command line argument
HF_TOKEN = os.getenv("HF_TOKEN", None)
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]

def get_file_list_from_metadata(csv_path):
    """Extract list of audio file names from metadata CSV."""
    file_names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get('file_name', '').strip()
            if file_name:
                # Remove 'data/' prefix if present
                if file_name.startswith('data/'):
                    file_name = file_name[5:]
                file_names.append(file_name)
    return file_names

def download_audio_files(repo_id, repo_type, data_folder, file_names, output_dir, token=None):
    """Download audio files from Hugging Face repository."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    print(f"Downloading {len(file_names)} audio files to {output_dir}...")
    
    for file_name in tqdm(file_names, desc="Downloading"):
        try:
            # Download file from Hugging Face
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": f"{data_folder}/{file_name}",
                "local_dir": output_dir,
                "local_dir_use_symlinks": False
            }
            if token:
                download_kwargs["token"] = token
            
            local_path = hf_hub_download(**download_kwargs)
            downloaded += 1
        except Exception as e:
            print(f"\nError downloading {file_name}: {e}")
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed: {failed}")

def main():
    token = HF_TOKEN
    
    # Get list of files from metadata
    print(f"Reading metadata from {METADATA_CSV}...")
    file_names = get_file_list_from_metadata(METADATA_CSV)
    print(f"Found {len(file_names)} files to download")
    
    if not token:
        print("\nNote: No token provided. If you get authentication errors, you can:")
        print("  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token")
        print("  2. Run: huggingface_hub login")
        print("  3. Pass token as argument: python download_audio.py your_token")
        print()
    
    # Download files
    download_audio_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        data_folder=DATA_FOLDER,
        file_names=file_names,
        output_dir=OUTPUT_DIR,
        token=token
    )

if __name__ == "__main__":
    main()

