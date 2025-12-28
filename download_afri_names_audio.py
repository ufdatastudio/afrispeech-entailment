#!/usr/bin/env python3
"""
Download audio files from Hugging Face afri-names dataset.
"""
import csv
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Configuration
REPO_ID = "intronhealth/afri-names"
REPO_TYPE = "dataset"
METADATA_CSV = "Entailment/metadata_afri-names.csv"
OUTPUT_DIR = "Audio"
HF_TOKEN = os.getenv("HF_TOKEN", None)
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]

def get_file_paths_from_metadata(csv_path):
    """Extract list of audio file paths from metadata CSV."""
    file_paths = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get('file_name', '').strip()
            if file_path:
                # Keep the full path including 'data/' prefix as files are in data/ folder
                file_paths.append(file_path)
    return file_paths

def download_audio_files(repo_id, repo_type, metadata_csv, output_dir, token=None):
    """Download audio files directly from Hugging Face Hub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = get_file_paths_from_metadata(metadata_csv)
    print(f"Found {len(file_paths)} files in metadata")
    print(f"Downloading to {output_dir}...")
    
    downloaded = 0
    failed = 0
    
    for file_path in tqdm(file_paths, desc="Downloading"):
        try:
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": file_path,
                "local_dir": output_dir,
                "local_dir_use_symlinks": False
            }
            if token:
                download_kwargs["token"] = token
            
            hf_hub_download(**download_kwargs)
            downloaded += 1
        except Exception as e:
            print(f"\nError downloading {file_path}: {e}")
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed: {failed}")

def main():
    token = HF_TOKEN
    
    if not token:
        print("\nNote: No token provided. If you get authentication errors, you can:")
        print("  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token")
        print("  2. Pass token as argument: python download_afri_names_audio.py your_token")
        print()
    
    download_audio_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        metadata_csv=METADATA_CSV,
        output_dir=OUTPUT_DIR,
        token=token
    )

if __name__ == "__main__":
    main()

