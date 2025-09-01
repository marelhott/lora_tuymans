#!/bin/bash

# LoRA Tuymans - Kompletně WebSocket-Free Start Script
# Optimalizováno pro RunPod proxy bez jakýchkoliv WebSocket závislostí

set -euo pipefail

# Logging funkce
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1"; }
log_success() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"; }
log_warning() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1" >&2; }

log_info "🚀 Spouštím LoRA Tuymans v KOMPLETNĚ WebSocket-Free režimu..."

# Systémové informace
log_info "📊 Systémové informace:"
echo "  - Hostname: $(hostname)"
echo "  - User: $(whoami)"
echo "  - Working directory: $(pwd)"
echo "  - WebSocket Mode: COMPLETELY DISABLED"
echo "  - Communication: HTTP POLLING ONLY"
echo "  - RunPod Proxy: OPTIMIZED"

# GPU kontrola
if command -v nvidia-smi &> /dev/null; then
    log_success "🎮 GPU detekována:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "GPU info nedostupná"
else
    log_warning "🖥️  GPU není dostupná, používám CPU režim"
    export FORCE_CPU=true
fi

# Environment variables pro WebSocket-free režim
export STREAMLIT_SERVER_PORT=8505
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DISABLE_WATCHDOG_WARNING=true

log_info "⚙️  WebSocket-Free konfigurace:"
echo "  - Port: $STREAMLIT_SERVER_PORT"
echo "  - Address: $STREAMLIT_SERVER_ADDRESS"
echo "  - Headless: $STREAMLIT_SERVER_HEADLESS"
echo "  - WebSocket: COMPLETELY DISABLED"
echo "  - CORS: DISABLED"
echo "  - XSRF Protection: DISABLED"
echo "  - WebSocket Compression: DISABLED"
echo "  - Usage Stats: DISABLED"

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

# Spuštění Streamlit v kompletně WebSocket-free režimu
log_info "🎨 Spouštím Streamlit v WebSocket-Free režimu..."
log_info "📡 Server: 0.0.0.0:8505 (HTTP POLLING ONLY)"
log_info "🌐 RunPod proxy: https://[POD_ID]-8505.proxy.runpod.net"
log_info "⚠️  WebSocket chyby v konzoli jsou NORMÁLNÍ - WebSocket je vypnutý!"

# Spuštění s maximálním vypnutím WebSocket funkcí
exec streamlit run /app/app.py \
    --server.port=8505 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --server.allowRunOnSave=false \
    --server.runOnSave=false \
    --server.enableStaticServing=false \
    --server.baseUrlPath="" \
    --global.disableWatchdogWarning=true \
    --runner.fastReruns=false \
    --client.toolbarMode=minimal \
    --browser.gatherUsageStats=false \
    --logger.level=info \
    --logger.enableRich=false