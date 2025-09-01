# Multi-stage build pro optimalizaci velikosti - AMD64 pro RunPod
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as builder

# Nastavení environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Aktualizace a instalace systémových závislostí
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Vytvoření Python virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Kopírování requirements a instalace Python závislostí
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage - optimalizovaný image s debug nástroji
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Nastavení environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalace runtime závislostí a debug nástrojů
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    curl \
    wget \
    netstat-nat \
    net-tools \
    netcat \
    lsof \
    procps \
    htop \
    nano \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Kopírování Python environment z builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Vytvoření non-root uživatele pro security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app /app/static /data/loras /data/models /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /data /home/appuser/.cache

# Nastavení pracovního adresáře
WORKDIR /app

# Kopírování aplikačních souborů
COPY app.py .
COPY app_no_websocket.py .
COPY .streamlit/ .streamlit/
COPY start_services.sh .
COPY start_no_websocket.sh .
COPY start_runpod.sh .
COPY start_websocket_free.sh .
COPY nginx-runpod.conf .
COPY monitor.sh .

# Nastavení oprávnění
RUN chown -R appuser:appuser /app && \
    chmod +x start_services.sh && \
    chmod +x start_no_websocket.sh && \
    chmod +x start_runpod.sh && \
    chmod +x start_websocket_free.sh && \
    chmod +x monitor.sh

# Expose port pro Streamlit
EXPOSE 8505

# Robustní healthcheck s fallback metodami
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8505/_stcore/health 2>/dev/null || \
        curl -f http://localhost:8505/ 2>/dev/null || \
        (ps aux | grep -q "streamlit.*8505" && netstat -tuln | grep -q ":8505") || \
        exit 1

# Přepnutí na non-root uživatele
USER appuser

# Environment variables pro aplikaci
ENV FORCE_CPU=false
ENV MAX_MEMORY_GB=16
ENV ENABLE_ATTENTION_SLICING=true
ENV ENABLE_CPU_OFFLOAD=auto
ENV LORA_MODELS_PATH=/data/loras
ENV FULL_MODELS_PATH=/data/models
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0

# ENTRYPOINT pro kompletně WebSocket-free režim
ENTRYPOINT ["/app/start_websocket_free.sh"]
