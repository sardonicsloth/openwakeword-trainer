FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Clone repositories
RUN git clone https://github.com/dscripka/openWakeWord.git /app/openWakeWord
RUN git clone https://github.com/rhasspy/piper-sample-generator.git /app/piper-sample-generator

# Download Piper TTS model
RUN mkdir -p /app/piper-sample-generator/models && \
    wget -O /app/piper-sample-generator/models/en_US-libritts_r-medium.pt \
    'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'

# Install piper-phonemize from wheel
RUN pip install https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/piper_phonemize-1.1.0-cp310-cp310-manylinux_2_28_x86_64.whl

# Install PyTorch with CUDA
RUN pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install training dependencies (matching notebook versions)
RUN pip install --no-cache-dir \
    'numpy<2' \
    pyyaml \
    webrtcvad \
    mutagen==1.47.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.2.0 \
    speechbrain==0.5.14 \
    audiomentations==0.33.0 \
    torch-audiomentations==0.11.0 \
    acoustics==0.2.6 \
    pronouncing==0.2.0 \
    datasets==2.14.6 \
    deep-phonemizer==0.0.19 \
    librosa \
    soundfile \
    scipy \
    scikit-learn \
    tqdm \
    requests \
    matplotlib \
    onnx \
    onnxruntime-gpu

# Install TensorFlow for TFLite conversion
RUN pip install tensorflow==2.11.0 tensorflow_probability==0.19.0 onnx-tf==1.10.0

# Install openWakeWord in editable mode (with training support)
RUN pip install -e /app/openWakeWord

# Install piper-sample-generator
RUN pip install -e /app/piper-sample-generator

# Download openWakeWord embedding models
RUN mkdir -p /app/openWakeWord/openwakeword/resources/models && \
    wget -O /app/openWakeWord/openwakeword/resources/models/embedding_model.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx && \
    wget -O /app/openWakeWord/openwakeword/resources/models/embedding_model.tflite \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite && \
    wget -O /app/openWakeWord/openwakeword/resources/models/melspectrogram.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx && \
    wget -O /app/openWakeWord/openwakeword/resources/models/melspectrogram.tflite \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite

# Copy training scripts
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

WORKDIR /data
