#!/usr/bin/env python3
"""
Generate synthetic speech samples for wake word training using Piper TTS.

This script uses piper-sample-generator to create diverse synthetic
speech samples of a given wake word phrase.

Usage:
    python generate_samples.py --wake-word "assistant" --count 5000 --output-dir /data/samples
"""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic wake word samples")
    parser.add_argument("--wake-word", type=str, required=True, help="Wake word phrase to synthesize")
    parser.add_argument("--count", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="/data/output/samples", help="Output directory")
    parser.add_argument("--model", type=str,
                        default="/app/piper-sample-generator/models/en_US-libritts_r-medium.pt",
                        help="Piper TTS model path")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--max-speakers", type=int, default=50, help="Maximum number of speaker variations")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} samples of '{args.wake_word}'...")
    print(f"Output directory: {output_dir}")

    # Run piper-sample-generator
    cmd = [
        "python", "-m", "piper_sample_generator",
        "--model", args.model,
        "--text", args.wake_word,
        "--output-dir", str(output_dir),
        "--count", str(args.count),
        "--batch-size", str(args.batch_size),
        "--max-speakers", str(args.max_speakers)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Count generated files
    wav_files = list(output_dir.glob("*.wav"))
    print(f"\nGenerated {len(wav_files)} WAV files in {output_dir}")


if __name__ == "__main__":
    main()
