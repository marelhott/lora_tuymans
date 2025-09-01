#!/bin/bash

# LoRA Tuymans - RunPod Optimized Startup Script
# Optimalizováno pro RunPod HTTP proxy s diagnostikou

set -euo pipefail

# Logging funkce
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1"; }
log_success() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"; }
log_warning() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1" >&2; }

log_info "🚀 Spouštím LoRA Tuymans pro RunPod..."

# RunPod environment detection
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    log_success "RunPod prostředí detekováno: Pod ID = $RUNPOD_POD_ID"
    RUNPOD_MODE=true
else
    log_warning "RunPod prostředí nedetekováno, spouštím v lokálním režimu"
    RUNPOD_MODE=false
fi

# Systémové informace
log_info "📊 Systémové informace:"
echo "  - Hostname: $(hostname)"
echo "  - User: $(whoami)"
echo "  - Working directory: $(pwd)"
echo "  - RunPod Mode: $RUNPOD_MODE"
echo "  - Available memory: $(free -h | grep Mem | awk '{print $7}' 2>/dev/null || echo 'N/A')"
echo "  - Available disk: $(df -h /app 2>/dev/null | tail -1 | awk '{print $4}' || echo 'N/A')"

# GPU kontrola
if command -v nvidia-smi &> /dev/null; then
    log_success "🎮 GPU detekována:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "GPU info nedostupná"
else
    log_warning "🖥️  GPU není dostupná, používám CPU režim"
    export FORCE_CPU=true
fi

# Port diagnostika
log_info "🔍 Port diagnostika:"
if netstat -tuln 2>/dev/null | grep -q ":8505"; then
    log_warning "Port 8505 je již používán:"
    netstat -tuln | grep ":8505" || true
else
    log_success "Port 8505 je volný"
fi

# RunPod proxy URL generování
if [[ "$RUNPOD_MODE" == "true" ]]; then
    PROXY_URL="https://${RUNPOD_POD_ID}-8505.proxy.runpod.net"
    log_info "🌐 RunPod Proxy URL: $PROXY_URL"
fi

# Environment variables pro Streamlit
export STREAMLIT_SERVER_PORT=8505
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

log_info "⚙️  Streamlit konfigurace:"
echo "  - Port: $STREAMLIT_SERVER_PORT"
echo "  - Address: $STREAMLIT_SERVER_ADDRESS"
echo "  - Headless: $STREAMLIT_SERVER_HEADLESS"

# Pre-startup checks
log_info "🔧 Pre-startup kontroly..."

# Kontrola app.py
if [[ ! -f "/app/app.py" ]]; then
    log_error "app.py nenalezen v /app/"
    exit 1
fi

# Kontrola .streamlit konfigurace
if [[ ! -d "/app/.streamlit" ]]; then
    log_warning ".streamlit adresář nenalezen, vytvářím..."
    mkdir -p /app/.streamlit
fi

# Spuštění Streamlit s RunPod optimalizací
log_info "🎨 Spouštím Streamlit aplikaci..."
log_info "📡 Server address: 0.0.0.0:8505"

if [[ "$RUNPOD_MODE" == "true" ]]; then
    log_info "🌐 RunPod proxy bude dostupný na: $PROXY_URL"
    log_info "⏱️  Proxy aktivace: ~30-60 sekund"
fi

# Spuštění s explicitními parametry pro RunPod
exec streamlit run /app/app.py \
    --server.port=8505 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=true \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --server.allowRunOnSave=false \
    --server.runOnSave=false \
    --server.enableStaticServing=false \
    --server.baseUrlPath="" \
    --global.disableFolderWatchdog=true \
    --runner.fastReruns=false \
    --client.toolbarMode=minimal \
    --logger.level=info \
    --logger.enableRich=false