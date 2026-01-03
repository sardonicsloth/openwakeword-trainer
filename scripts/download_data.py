#!/usr/bin/env python3
"""
Download required datasets for openWakeWord training.

This script downloads:
1. AudioSet - Real-world sounds for negative samples
2. Free Music Archive (FMA) - Music for negative samples
3. MIT Room Impulse Responses - For audio augmentation
4. ACAV100M pre-computed features - Negative sample embeddings
5. Validation features - For false positive validation

Usage:
    python download_data.py --output-dir /data --datasets all
    python download_data.py --output-dir /data --datasets audioset fma
"""

import argparse
import os
import subprocess
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_file(url: str, dest: Path, desc: str = None):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc or dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def download_audioset(output_dir: Path, num_parts: int = 1):
    """Download AudioSet balanced train segments."""
    print("\n=== Downloading AudioSet ===")

    audioset_dir = output_dir / "audioset"
    audioset_16k_dir = output_dir / "audioset_16k"
    audioset_dir.mkdir(parents=True, exist_ok=True)
    audioset_16k_dir.mkdir(parents=True, exist_ok=True)

    # Download balanced train parts (bal_train00.tar through bal_train09.tar)
    base_url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/"

    for i in range(num_parts):
        fname = f"bal_train{i:02d}.tar"
        tar_path = audioset_dir / fname

        print(f"\nDownloading {fname}...")
        download_file(base_url + fname, tar_path)

        # Extract
        print(f"Extracting {fname}...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(audioset_dir)

    # Convert to 16kHz
    print("\nConverting to 16kHz...")
    flac_files = list(audioset_dir.glob("**/*.flac"))
    for flac_file in tqdm(flac_files, desc="Converting"):
        wav_file = audioset_16k_dir / (flac_file.stem + ".wav")
        if not wav_file.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", str(flac_file),
                "-ar", "16000", "-ac", "1", str(wav_file)
            ], capture_output=True)


def download_fma(output_dir: Path):
    """Download Free Music Archive small subset."""
    print("\n=== Downloading Free Music Archive ===")

    fma_dir = output_dir / "fma"
    fma_dir.mkdir(parents=True, exist_ok=True)

    # FMA small subset (~8GB compressed, ~30GB extracted)
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    zip_path = fma_dir / "fma_small.zip"

    print("Downloading FMA small (this may take a while)...")
    download_file(url, zip_path)

    print("Extracting FMA...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(fma_dir)])

    # Convert to 16kHz WAV
    print("Converting to 16kHz...")
    mp3_files = list(fma_dir.glob("**/*.mp3"))
    for mp3_file in tqdm(mp3_files[:1000], desc="Converting"):  # Limit for speed
        wav_file = fma_dir / (mp3_file.stem + ".wav")
        if not wav_file.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", str(mp3_file),
                "-ar", "16000", "-ac", "1", "-t", "30", str(wav_file)
            ], capture_output=True)


def download_rirs(output_dir: Path):
    """Download MIT Room Impulse Responses."""
    print("\n=== Downloading MIT RIRs ===")

    rir_dir = output_dir / "mit_rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)

    # MIT Acoustical Reverberation Scene Statistics Survey
    url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
    zip_path = rir_dir / "mit_rirs.zip"

    print("Downloading MIT RIRs...")
    download_file(url, zip_path)

    print("Extracting...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(rir_dir)])


def download_features(output_dir: Path):
    """Download pre-computed ACAV100M features."""
    print("\n=== Downloading Pre-computed Features ===")

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # ACAV100M features (2000 hours, ~2GB)
    acav_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    acav_path = features_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

    print("Downloading ACAV100M features (~2GB)...")
    download_file(acav_url, acav_path)

    # Validation features (~100MB)
    val_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    val_path = features_dir / "validation_set_features.npy"

    print("Downloading validation features...")
    download_file(val_url, val_path)


def main():
    parser = argparse.ArgumentParser(description="Download openWakeWord training data")
    parser.add_argument("--output-dir", type=str, default="/data",
                        help="Output directory for downloaded data")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=["all", "audioset", "fma", "rirs", "features"],
                        help="Datasets to download")
    parser.add_argument("--audioset-parts", type=int, default=2,
                        help="Number of AudioSet parts to download (0-9)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets = args.datasets

    if "all" in datasets:
        datasets = ["audioset", "fma", "rirs", "features"]

    if "features" in datasets:
        download_features(output_dir)

    if "rirs" in datasets:
        download_rirs(output_dir)

    if "audioset" in datasets:
        download_audioset(output_dir, args.audioset_parts)

    if "fma" in datasets:
        download_fma(output_dir)

    print("\n=== Download Complete ===")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
