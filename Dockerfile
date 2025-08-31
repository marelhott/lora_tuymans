# Optimalizovaný multi-stage Dockerfile pro RunPod deployment
# Používá pip místo conda pro menší velikost image
# Aktualizováno pro RTX 5090 podporu (vyžaduje PyTorch 2.8+)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as base

# Argumenty pro multi-platform build
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Nastavení pracovního adresáře
WORKDIR /app

# Instalace pouze nezbytných systémových závislostí
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install code-server and FTP Server
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN apt-get update && apt-get install -y \
    vsftpd \
    && apt-get clean

# Install FileBrowser
RUN wget -qO- https://github.com/filebrowser/filebrowser/releases/latest/download/linux-amd64-filebrowser.tar.gz | tar -xz -C /usr/local/bin/

# Upgrade pip a instalace wheel
RUN pip install --upgrade pip setuptools wheel

# Kopírování requirements a instalace Python závislostí
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Vytvoření adresářů pro cache a persistentní disk
RUN mkdir -p .streamlit /data/models /data/loras /root/.cache/huggingface

# Kopírování pouze aplikačních souborů (bez modelů)
COPY app.py requirements.txt ./
COPY .streamlit/ .streamlit/

# Environment variables pro RunPod optimalizaci + RTX 5090 optimalizace
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    FORCE_CPU=false \
    MAX_MEMORY_GB=24 \
    ENABLE_ATTENTION_SLICING=true \
    ENABLE_CPU_OFFLOAD=auto \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface \
    LORA_MODELS_PATH=/data/loras \
    FULL_MODELS_PATH=/data/models

# Volume pro persistentní disk (bude mountován z RunPod storage)
VOLUME ["/data"]

# Expose ports for Streamlit, FTP, Code Server, and FileBrowser
EXPOSE 8501 21 20 10000-10100 8080 8083

# Copy startup script
COPY start_services.sh /app/start_services.sh
RUN chmod +x /app/start_services.sh

# Run both applications
CMD ["/app/start_services.sh"]