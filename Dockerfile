FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
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

# Install PyTorch 1.x with CUDA first
RUN pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install piper-phonemize from wheel
RUN pip install https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/piper_phonemize-1.1.0-cp310-cp310-manylinux_2_28_x86_64.whl

# Install Python dependencies
RUN pip install --no-cache-dir \
    'numpy<2' \
    webrtcvad \
    'audiomentations==0.33.0' \
    torchinfo \
    torchmetrics \
    onnx \
    onnx-tf \
    'tensorflow<2.12' \
    pronouncing \
    datasets \
    librosa \
    soundfile \
    scipy \
    scikit-learn \
    torch_audiomentations \
    tqdm \
    requests \
    mutagen \
    matplotlib \
    onnxruntime-gpu

# Install openWakeWord in editable mode
RUN pip install -e /app/openWakeWord

# Install piper-sample-generator
RUN pip install -e /app/piper-sample-generator

# CRITICAL: Reinstall correct PyTorch versions AFTER all other installations
RUN pip install --force-reinstall torch==1.13.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Now install speechbrain with the correct torchaudio in place
RUN pip install speechbrain==0.5.16

# Download openWakeWord models during build
RUN python -c "from openwakeword.utils import download_models; download_models()"

WORKDIR /data
