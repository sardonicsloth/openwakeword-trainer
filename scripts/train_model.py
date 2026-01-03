#!/usr/bin/env python3
"""
Train a custom wake word model using openWakeWord.

This script:
1. Uses pre-generated positive samples from piper-sample-generator
2. Generates negative samples (silence and noise)
3. Extracts features using openWakeWord's feature extractor
4. Trains a DNN classifier
5. Exports to ONNX format

Usage:
    python train_model.py --wake-word "assistant" --samples-dir /data/samples --output-dir /data/output
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import scipy.io.wavfile

# Add openWakeWord to path
sys.path.insert(0, "/app/openWakeWord")
import openwakeword
from openwakeword.utils import AudioFeatures


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom wake word model")
    parser.add_argument("--wake-word", type=str, required=True, help="Wake word name")
    parser.add_argument("--samples-dir", type=str, default="/data/output/samples",
                        help="Directory containing positive WAV samples")
    parser.add_argument("--output-dir", type=str, default="/data/output/model",
                        help="Output directory for trained model")
    parser.add_argument("--train-positive", type=int, default=4000,
                        help="Number of positive training samples")
    parser.add_argument("--val-positive", type=int, default=1000,
                        help="Number of positive validation samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--layer-dim", type=int, default=32, help="Hidden layer dimension")
    return parser.parse_args()


def load_wav_as_features(wav_path, feature_extractor):
    """Load a WAV file and extract openWakeWord features."""
    sr, audio = scipy.io.wavfile.read(wav_path)
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        audio = (audio * 32767).astype(np.int16)

    # Pad to at least 1.5 seconds for feature extraction
    min_samples = int(1.5 * 16000)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

    # Get embedding features
    features = feature_extractor.embed_clips(audio.reshape(1, -1))
    return features[0] if len(features.shape) > 1 else features


class WakeWordModel(nn.Module):
    """Simple DNN classifier for wake word detection."""

    def __init__(self, input_shape, layer_dim=32):
        super().__init__()
        input_size = input_shape[0] * input_shape[1] if len(input_shape) > 1 else input_shape[0]
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def main():
    args = parse_args()

    print(f"Starting wake word training for '{args.wake_word}'...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    print("Initializing feature extractor...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = AudioFeatures(device=device)

    # Load positive samples
    samples_dir = Path(args.samples_dir)
    print(f"Loading positive samples from {samples_dir}...")
    positive_files = sorted(samples_dir.glob("*.wav"))[:args.train_positive + args.val_positive]
    print(f"Found {len(positive_files)} positive samples")

    if len(positive_files) < args.train_positive + args.val_positive:
        print(f"Warning: Only {len(positive_files)} samples available, need {args.train_positive + args.val_positive}")

    positive_features = []
    for wav_file in tqdm(positive_files, desc="Extracting positive features"):
        try:
            features = load_wav_as_features(str(wav_file), feature_extractor)
            positive_features.append(features)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

    positive_features = np.array(positive_features)
    print(f"Extracted features shape: {positive_features.shape}")

    # Generate negative samples (silence and random noise)
    print("Generating negative samples...")
    n_negative = len(positive_features)
    negative_features = []

    for i in tqdm(range(n_negative), desc="Generating negative features"):
        if i % 2 == 0:
            # Silence with small noise
            audio = np.random.randint(-100, 100, size=int(1.5 * 16000), dtype=np.int16)
        else:
            # Random noise
            audio = np.random.randint(-10000, 10000, size=int(1.5 * 16000), dtype=np.int16)

        features = feature_extractor.embed_clips(audio.reshape(1, -1))
        negative_features.append(features[0] if len(features.shape) > 1 else features)

    negative_features = np.array(negative_features)
    print(f"Negative features shape: {negative_features.shape}")

    # Split into train/val
    X_train_pos = positive_features[:args.train_positive]
    X_val_pos = positive_features[args.train_positive:]
    X_train_neg = negative_features[:args.train_positive]
    X_val_neg = negative_features[args.train_positive:]

    # Create training data
    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.hstack([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])

    X_val = np.vstack([X_val_pos, X_val_neg])
    y_val = np.hstack([np.ones(len(X_val_pos)), np.zeros(len(X_val_neg))])

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Create model
    input_shape = X_train.shape[1:]
    model = WakeWordModel(input_shape, layer_dim=args.layer_dim)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    print(f"Model created, using device: {torch_device}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # Training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = correct / total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(torch_device)
    onnx_path = output_dir / f"{args.wake_word}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")

    print("\nTraining complete!")
    print(f"Model files saved to {output_dir}")
    print(f"\nNext step: Convert to TFLite using scripts/convert_tflite.py")


if __name__ == "__main__":
    main()
