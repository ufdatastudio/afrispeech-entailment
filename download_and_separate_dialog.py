#!/usr/bin/env python3
"""
Download afrispeech-dialog dataset and separate medical from general domain data.
"""
import csv
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import pandas as pd

# Configuration
REPO_ID = "intronhealth/afrispeech-dialog"
REPO_TYPE = "dataset"
DATA_FOLDER = "data"
OUTPUT_DIR = "Audio"
METADATA_DIR = "Entailment"
METADATA_MEDICAL_CSV = f"{METADATA_DIR}/metadata_afrispeech-dialog_medical.csv"
METADATA_GENERAL_CSV = f"{METADATA_DIR}/metadata_afrispeech-dialog_general.csv"
HF_TOKEN = os.getenv("HF_TOKEN", None)
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]

def download_metadata(repo_id, repo_type, token=None):
    """Download metadata.csv file."""
    print("Downloading metadata.csv...")
    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "filename": "metadata.csv"
    }
    if token:
        download_kwargs["token"] = token
    
    try:
        metadata_path = hf_hub_download(**download_kwargs)
        print(f"Metadata downloaded to: {metadata_path}")
        return metadata_path
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return None

def separate_metadata_by_domain(metadata_path):
    """Read metadata and separate by domain."""
    print(f"Reading metadata from {metadata_path}...")
    
    df = pd.read_csv(metadata_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'domain' not in df.columns:
        print("Error: 'domain' column not found in metadata")
        return None, None
    
    # Separate by domain
    medical_df = df[df['domain'].str.lower() == 'medical'].copy()
    general_df = df[df['domain'].str.lower() == 'general'].copy()
    
    print(f"\nDomain distribution:")
    print(f"  Medical: {len(medical_df)} samples")
    print(f"  General: {len(general_df)} samples")
    print(f"  Total: {len(df)} samples")
    
    return medical_df, general_df

def save_separated_metadata(medical_df, general_df):
    """Save separated metadata to CSV files."""
    # Ensure output directory exists
    Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Keep only file_name and transcript columns
    columns_to_keep = ['file_name', 'transcript']
    
    if medical_df is not None and len(medical_df) > 0:
        medical_output = Path(METADATA_MEDICAL_CSV)
        medical_df[columns_to_keep].to_csv(medical_output, index=False, encoding='utf-8')
        print(f"\nSaved medical metadata to: {METADATA_MEDICAL_CSV} ({len(medical_df)} rows)")
    
    if general_df is not None and len(general_df) > 0:
        general_output = Path(METADATA_GENERAL_CSV)
        general_df[columns_to_keep].to_csv(general_output, index=False, encoding='utf-8')
        print(f"Saved general metadata to: {METADATA_GENERAL_CSV} ({len(general_df)} rows)")

def get_audio_files_from_metadata(df):
    """Extract audio file names from metadata dataframe."""
    file_names = []
    for file_name in df['file_name']:
        # Remove 'data/' prefix if present
        if file_name.startswith('data/'):
            file_name = file_name[5:]
        file_names.append(file_name)
    return file_names

def download_audio_files(repo_id, repo_type, data_folder, file_names, output_dir, token=None):
    """Download audio files from Hugging Face Hub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {len(file_names)} audio files to {output_dir}...")
    
    downloaded = 0
    failed = 0
    
    for file_name in tqdm(file_names, desc="Downloading"):
        try:
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": f"{data_folder}/{file_name}",
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
        print("  2. Pass token as argument: python download_and_separate_dialog.py your_token")
        print()
        return
    
    # Step 1: Download metadata
    metadata_path = download_metadata(REPO_ID, REPO_TYPE, token)
    if not metadata_path:
        print("Failed to download metadata. Exiting.")
        return
    
    # Step 2: Separate metadata by domain
    medical_df, general_df = separate_metadata_by_domain(metadata_path)
    if medical_df is None or general_df is None:
        print("Failed to separate metadata. Exiting.")
        return
    
    # Step 3: Save separated metadata files
    save_separated_metadata(medical_df, general_df)
    
    # Step 4: Download all audio files
    all_file_names = get_audio_files_from_metadata(pd.concat([medical_df, general_df]))
    download_audio_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        data_folder=DATA_FOLDER,
        file_names=all_file_names,
        output_dir=OUTPUT_DIR,
        token=token
    )
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Medical samples: {len(medical_df)}")
    print(f"  General samples: {len(general_df)}")
    print(f"  Total samples: {len(medical_df) + len(general_df)}")
    print(f"  Metadata files created:")
    print(f"    - {METADATA_MEDICAL_CSV}")
    print(f"    - {METADATA_GENERAL_CSV}")
    print(f"  Audio files downloaded to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
