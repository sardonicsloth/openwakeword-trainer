# openWakeWord Trainer

A Docker-based pipeline for training custom wake word models compatible with [openWakeWord](https://github.com/dscripka/openWakeWord).

Based on the [official training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).

## Features

- Synthetic speech generation using [Piper TTS](https://github.com/rhasspy/piper)
- Proper negative samples from AudioSet and Free Music Archive
- Room impulse response augmentation for realistic conditions
- Pre-computed ACAV100M features (2000 hours of audio)
- GPU-accelerated training with automatic model export
- TFLite conversion for deployment on edge devices

## Requirements

- Docker with NVIDIA GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 11.8+ support
- ~50GB disk space for training data
- ~8GB GPU memory recommended

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t openwakeword-trainer .
```

### 2. Download Training Data

Download the required datasets (AudioSet, FMA, ACAV100M features):

```bash
# Create data directory
mkdir -p data

# Download all required data (~10GB+ download)
docker run --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/download_data.py --output-dir /data --datasets all
```

For faster iteration, you can download just the essential features:

```bash
# Minimal: just pre-computed features (~2GB)
docker run --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/download_data.py --output-dir /data --datasets features rirs
```

### 3. Train Your Wake Word

```bash
docker run --gpus all --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/train.py \
        --wake-word "hey assistant" \
        --output-dir /data/output
```

This runs the full pipeline:
1. **Generate** - Create synthetic positive samples with Piper TTS
2. **Augment** - Apply room impulse responses and background noise
3. **Train** - Train the wake word classifier with hard negative mining
4. **Convert** - Export to ONNX and TFLite formats

### 4. Run Individual Steps

```bash
# Generate synthetic clips only
docker run --gpus all --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/train.py --config /data/output/hey_assistant.yml --step generate

# Augment clips
docker run --gpus all --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/train.py --config /data/output/hey_assistant.yml --step augment

# Train model
docker run --gpus all --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/train.py --config /data/output/hey_assistant.yml --step train

# Convert to TFLite
docker run --gpus all --rm -v $(pwd)/data:/data openwakeword-trainer \
    python /app/scripts/train.py --config /data/output/hey_assistant.yml --step convert
```

## Deployment

Copy the generated `.tflite` file to your openWakeWord instance:

```bash
# For Wyoming satellite
scp data/output/hey_assistant.tflite user@satellite:/opt/docker/wyoming/models/

# Configure openwakeword container
docker run -d \
    -p 10400:10400 \
    -v /opt/docker/wyoming/models:/custom \
    rhasspy/wyoming-openwakeword \
        --preload-model hey_assistant \
        --custom-model-dir /custom
```

## Configuration

Copy and customize `configs/template.yml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_phrase` | required | Wake word phrase(s) |
| `n_samples` | 10000 | Positive training samples |
| `n_samples_val` | 2000 | Validation samples |
| `steps` | 50000 | Training iterations |
| `layer_size` | 32 | Hidden layer dimension |
| `max_negative_weight` | 1500 | Hard negative mining weight |
| `target_false_positives_per_hour` | 0.2 | Target FP rate |

## Training Data

The training pipeline uses:

1. **Positive samples**: Synthetic TTS from Piper with multiple voices
2. **Negative samples**:
   - AudioSet: Real-world sounds (speech, music, noise)
   - FMA: Music samples
   - ACAV100M: 2000 hours of pre-computed embeddings
3. **Augmentation**:
   - MIT Room Impulse Responses
   - Background noise mixing
   - Audio transformations

## Model Architecture

The default model is a simple DNN:
- Input: openWakeWord audio embeddings (96-dim mel spectrogram → 96-dim embedding)
- 2x hidden layers with 32 units each
- Sigmoid output for binary classification
- ~120KB TFLite model size

## Troubleshooting

### Model triggers too often (false positives)
- Increase `target_false_positives_per_hour` threshold
- Add more negative samples via `custom_negative_phrases`
- Increase training `steps`

### Model doesn't trigger (false negatives)
- Generate more positive samples with `n_samples`
- Decrease `target_false_positives_per_hour`
- Try different wake word phrase

### Out of memory during training
- Reduce `batch_n_per_class` values
- Use a GPU with more VRAM

## License

MIT License
