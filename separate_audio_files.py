#!/usr/bin/env python3
"""
Separate audio files in Audio/data into medical and general folders based on metadata.
"""
import csv
import os
import shutil
from pathlib import Path
import pandas as pd

# Configuration
AUDIO_DATA_DIR = "Audio/data"
OUTPUT_MEDICAL_DIR = "Audio/medical"
OUTPUT_GENERAL_DIR = "Audio/general"

# Try to find metadata files
METADATA_FILES = [
    "Entailment/metadata_afrispeech-dialog_medical.csv",
    "Entailment/metadata_afrispeech-dialog_general.csv",
    "Entailment/metadata_medical.csv",
    "Entailment/metadata_afrispeech-general.csv"
]

def find_metadata_files():
    """Find available metadata files."""
    found_files = []
    for metadata_file in METADATA_FILES:
        if Path(metadata_file).exists():
            found_files.append(metadata_file)
    return found_files

def get_file_mapping_from_metadata():
    """Create a mapping of file_name -> domain from metadata files."""
    file_to_domain = {}
    
    # Check for dialog-specific metadata first
    medical_meta = Path("Entailment/metadata_afrispeech-dialog_medical.csv")
    general_meta = Path("Entailment/metadata_afrispeech-dialog_general.csv")
    
    if medical_meta.exists() and general_meta.exists():
        print(f"Using dialog-specific metadata files...")
        # Read medical metadata
        with open(medical_meta, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get('file_name', '').strip()
                # Remove 'data/' prefix if present
                if file_name.startswith('data/'):
                    file_name = file_name[5:]
                # Get just the filename (last part after /)
                file_name = file_name.split('/')[-1]
                file_to_domain[file_name] = 'medical'
        
        # Read general metadata
        with open(general_meta, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get('file_name', '').strip()
                # Remove 'data/' prefix if present
                if file_name.startswith('data/'):
                    file_name = file_name[5:]
                # Get just the filename (last part after /)
                file_name = file_name.split('/')[-1]
                file_to_domain[file_name] = 'general'
        
        return file_to_domain
    
    # Fallback: check if we need to download metadata first
    print("Dialog metadata files not found. Checking if we need to download them...")
    return None

def separate_audio_files():
    """Separate audio files into medical and general folders."""
    audio_data_path = Path(AUDIO_DATA_DIR)
    
    if not audio_data_path.exists():
        print(f"Error: {AUDIO_DATA_DIR} does not exist")
        return
    
    # Get file to domain mapping
    file_to_domain = get_file_mapping_from_metadata()
    
    if file_to_domain is None:
        print("Error: Could not find metadata files to determine domain")
        print("Please run download_and_separate_dialog.py first")
        return
    
    # Create output directories
    medical_dir = Path(OUTPUT_MEDICAL_DIR)
    general_dir = Path(OUTPUT_GENERAL_DIR)
    medical_dir.mkdir(parents=True, exist_ok=True)
    general_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(audio_data_path.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {AUDIO_DATA_DIR}")
    
    medical_count = 0
    general_count = 0
    not_found = []
    
    for audio_file in audio_files:
        file_name = audio_file.name
        
        # Check if we have domain info for this file
        domain = file_to_domain.get(file_name)
        
        if domain == 'medical':
            dest = medical_dir / file_name
            shutil.copy2(audio_file, dest)
            medical_count += 1
        elif domain == 'general':
            dest = general_dir / file_name
            shutil.copy2(audio_file, dest)
            general_count += 1
        else:
            not_found.append(file_name)
    
    print(f"\nSeparation complete!")
    print(f"  Medical files: {medical_count} copied to {OUTPUT_MEDICAL_DIR}")
    print(f"  General files: {general_count} copied to {OUTPUT_GENERAL_DIR}")
    
    if not_found:
        print(f"\n  Warning: {len(not_found)} files not found in metadata:")
        for f in not_found[:10]:  # Show first 10
            print(f"    - {f}")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")

def main():
    print("Separating audio files by domain...")
    separate_audio_files()

if __name__ == "__main__":
    main()


