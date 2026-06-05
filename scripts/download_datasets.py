#!/usr/bin/env python3
"""
Script to download datasets for DSRQS paper.
Datasets will be available after paper acceptance.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile

DATASET_URLS = {
    "orphanet_fq274": "https://data.example.com/orphanet_fq274.zip",  # Placeholder URL
    "disgenet_rd411": "https://data.example.com/disgenet_rd411.zip",
    "omim_hop3": "https://data.example.com/omim_hop3.zip"
}

def download_dataset(dataset_name, data_dir, force=False):
    """Download a single dataset."""
    dataset_dir = data_dir / dataset_name
    zip_path = data_dir / f"{dataset_name}.zip"
    
    if dataset_dir.exists() and not force:
        print(f"✓ Dataset {dataset_name} already exists at {dataset_dir}")
        return True
    
    print(f"Downloading {dataset_name}...")
    print(f"Note: This is a placeholder script.")
    print(f"Actual download URLs will be available upon paper acceptance.")
    print(f"For now, using existing data in {data_dir}")
    
    # Check if we already have the data
    full_json = dataset_dir / f"{dataset_name}_full.json"
    if full_json.exists():
        print(f"✓ Found existing data for {dataset_name}")
        return True
    
    print(f"⚠ Dataset {dataset_name} full data not found.")
    print(f"Please contact authors for access, or place data in {dataset_dir}/")
    return False

def download_all_datasets(data_dir):
    """Download all datasets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("DSRQS Dataset Downloader")
    print("=" * 80)
    print()
    
    results = {}
    for dataset_name in DATASET_URLS.keys():
        results[dataset_name] = download_dataset(dataset_name, data_dir)
    
    print()
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    for dataset_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset_name}")
    print("=" * 80)
    print()
    
    return all(results.values())

def main():
    parser = argparse.ArgumentParser(
        description="Download DSRQS datasets (available upon paper acceptance)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory to save datasets (default: data)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "orphanet_fq274", "disgenet_rd411", "omim_hop3"],
        default="all",
        help="Dataset to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if args.dataset == "all":
        success = download_all_datasets(data_dir)
    else:
        success = download_dataset(args.dataset, data_dir, args.force)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
