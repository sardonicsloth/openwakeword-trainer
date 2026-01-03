# openWakeWord Trainer

A Docker-based pipeline for training custom wake word models compatible with [openWakeWord](https://github.com/dscripka/openWakeWord).

## Features

- Synthetic speech sample generation using [Piper TTS](https://github.com/rhasspy/piper)
- GPU-accelerated training with PyTorch
- ONNX export with TFLite conversion for deployment
- Compatible with Wyoming protocol voice satellites

## Requirements

- Docker with NVIDIA GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 11.8+ support
- At least 8GB GPU memory recommended

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t openwakeword-trainer .
```

### 2. Generate Training Samples

Generate synthetic speech samples for your wake word:

```bash
docker run --gpus all --rm \
  -v $(pwd)/data:/data \
  openwakeword-trainer \
  python /app/scripts/generate_samples.py \
    --wake-word "assistant" \
    --count 5000 \
    --output-dir /data/output/samples
```

### 3. Train the Model

Train the wake word classifier:

```bash
docker run --gpus all --rm \
  -v $(pwd)/data:/data \
  openwakeword-trainer \
  python /app/scripts/train_model.py \
    --wake-word "assistant" \
    --samples-dir /data/output/samples \
    --output-dir /data/output/model \
    --epochs 100
```

### 4. Convert to TFLite

The TFLite conversion requires specific package versions. Run in a separate container:

```bash
docker run --rm \
  -v $(pwd)/data/output/model:/model \
  python:3.10 bash -c "
    pip install numpy==1.23.5 onnx onnx-tf tensorflow==2.11.0 tensorflow_probability==0.19.0 &&
    python -c \"
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load('/model/assistant.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('/model/saved_model')

converter = tf.lite.TFLiteConverter.from_saved_model('/model/saved_model')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('/model/assistant.tflite', 'wb') as f:
    f.write(tflite_model)
print('Saved assistant.tflite')
\"
  "
```

## Deployment

Copy the generated `.tflite` file to your openWakeWord custom models directory:

```bash
# For Wyoming satellite
scp data/output/model/assistant.tflite user@satellite:/opt/docker/wyoming/models/

# Restart openwakeword container with custom model
docker run -d \
  -p 10400:10400 \
  -v /opt/docker/wyoming/models:/custom \
  rhasspy/wyoming-openwakeword \
    --preload-model assistant \
    --custom-model-dir /custom
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wake-word` | required | Wake word phrase |
| `--samples-dir` | /data/output/samples | Directory with WAV samples |
| `--output-dir` | /data/output/model | Output directory |
| `--train-positive` | 4000 | Training samples |
| `--val-positive` | 1000 | Validation samples |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--layer-dim` | 32 | Hidden layer size |

## Model Architecture

The wake word model is a simple DNN classifier:
- Input: openWakeWord audio embeddings (from melspectrogram + embedding model)
- 2x hidden layers with LayerNorm and ReLU (32 units each)
- Sigmoid output for binary classification

## License

MIT License
