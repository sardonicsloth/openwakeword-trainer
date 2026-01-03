#!/usr/bin/env python3
"""
Wrapper script for openWakeWord training.

This script provides a simplified interface to the openWakeWord train.py,
handling config generation and the full training pipeline.

Usage:
    # Full training pipeline
    python train.py --wake-word "hey assistant" --output-dir /data/output

    # With custom config
    python train.py --config /data/my_config.yml

    # Individual steps
    python train.py --config /data/my_config.yml --step generate
    python train.py --config /data/my_config.yml --step augment
    python train.py --config /data/my_config.yml --step train
    python train.py --config /data/my_config.yml --step convert
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
from shutil import copy2


def create_config(wake_word: str, output_dir: Path, **kwargs) -> Path:
    """Create a training config from template."""

    # Load template
    template_path = Path("/app/configs/template.yml")
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update with wake word settings
    model_name = wake_word.lower().replace(" ", "_")
    config["model_name"] = model_name
    config["target_phrase"] = [wake_word]
    config["output_dir"] = str(output_dir)

    # Apply any overrides
    for key, value in kwargs.items():
        if value is not None and key in config:
            config[key] = value

    # Save config
    config_path = output_dir / f"{model_name}.yml"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created config: {config_path}")
    return config_path


def run_training_step(config_path: Path, step: str):
    """Run a specific training step."""

    train_script = "/app/openWakeWord/openwakeword/train.py"

    step_flags = {
        "generate": "--generate_clips",
        "augment": "--augment_clips",
        "train": "--train_model",
        "convert": "--convert_to_tflite",
    }

    if step not in step_flags:
        raise ValueError(f"Unknown step: {step}. Must be one of: {list(step_flags.keys())}")

    cmd = [
        sys.executable,
        train_script,
        "--training_config", str(config_path),
        step_flags[step]
    ]

    print(f"\n{'='*60}")
    print(f"Running step: {step}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + "\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Step {step} failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train custom openWakeWord model")

    # Config options
    parser.add_argument("--config", type=str, help="Path to existing config YAML")
    parser.add_argument("--wake-word", type=str, help="Wake word phrase to train")
    parser.add_argument("--output-dir", type=str, default="/data/output",
                        help="Output directory")

    # Training parameters
    parser.add_argument("--n-samples", type=int, help="Number of positive samples")
    parser.add_argument("--steps", type=int, help="Training steps")
    parser.add_argument("--layer-size", type=int, help="Model layer size")

    # Step control
    parser.add_argument("--step", type=str, choices=["generate", "augment", "train", "convert", "all"],
                        default="all", help="Training step to run")

    args = parser.parse_args()

    # Determine config path
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
    elif args.wake_word:
        config_path = create_config(
            args.wake_word,
            Path(args.output_dir),
            n_samples=args.n_samples,
            steps=args.steps,
            layer_size=args.layer_size,
        )
    else:
        print("Error: Must provide either --config or --wake-word")
        sys.exit(1)

    # Run training steps
    if args.step == "all":
        steps = ["generate", "augment", "train", "convert"]
    else:
        steps = [args.step]

    for step in steps:
        run_training_step(config_path, step)

    # Print output location
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    model_name = config["model_name"]

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model files:")
    print(f"  ONNX:   {output_dir}/{model_name}.onnx")
    print(f"  TFLite: {output_dir}/{model_name}.tflite")
    print("\nDeploy the .tflite file to your openWakeWord instance.")


if __name__ == "__main__":
    main()
