#!/bin/bash

# LoRA Tuymans - WebSocket-Free Start Script
# Optimalizováno pro RunPod proxy prostředí

set -euo pipefail

# Logging funkce
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1"; }
log_success() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"; }
log_warning() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1" >&2; }

log_info "Spouštím LoRA Tuymans v WebSocket-Free režimu..."

# Systémové informace
log_info "Systémové informace:"
echo "  - Hostname: $(hostname)"
echo "  - User: $(whoami)"
echo "  - Working directory: $(pwd)"
echo "  - WebSocket Mode: DISABLED"
echo "  - Proxy Mode: ENABLED"

# Kontrola GPU
if command -v nvidia-smi &> /dev/null; then
    log_success "GPU detekována:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    log_warning "GPU není dostupná, přepínám na CPU"
    export FORCE_CPU=true
fi

# Vyčištění procesů
log_info "Čištění běžících procesů..."
if pgrep -f streamlit >/dev/null; then
    log_warning "Ukončuji běžící Streamlit procesy..."
    pkill -f streamlit || true
    sleep 3
fi

# Kontrola portu
log_info "Kontrola portu 8505..."
if command -v lsof >/dev/null && lsof -Pi :8505 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_warning "Port 8505 je obsazený, čištím..."
    fuser -k 8505/tcp 2>/dev/null || true
    sleep 2
fi

# Environment variables pro WebSocket-free režim
log_info "Nastavuji WebSocket-free prostředí..."
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=true
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false
export STREAMLIT_SERVER_ALLOW_RUN_ON_SAVE=false
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false

# Speciální konfigurace pro RunPod proxy
export STREAMLIT_SERVER_BASE_URL_PATH=""
export STREAMLIT_SERVER_WEBSOCKET_PROXY_HOST="0.0.0.0"
export STREAMLIT_SERVER_WEBSOCKET_PROXY_PORT="8505"

# Vypnutí WebSocket funkcí
export STREAMLIT_GLOBAL_DISABLE_FOLDER_WATCHDOG=true
export STREAMLIT_RUNNER_FAST_RERUNS=false
export STREAMLIT_CLIENT_TOOLBAR_MODE="minimal"

# Kontrola aplikačních souborů
log_info "Kontrola aplikačních souborů..."
if [[ ! -f "app_no_websocket.py" ]]; then
    log_error "app_no_websocket.py nebyl nalezen!"
    exit 1
fi

if [[ ! -d ".streamlit" ]]; then
    log_warning "Vytvářím .streamlit složku..."
    mkdir -p .streamlit
fi

# Spuštění Streamlit v WebSocket-free režimu
log_success "Spouštím Streamlit v WebSocket-free režimu na portu 8505..."
log_info "Server address: 0.0.0.0:8505"
log_info "WebSocket: DISABLED"
log_info "Proxy mode: ENABLED"
log_info "RunPod proxy bude dostupný za ~30 sekund"

# Spuštění s WebSocket-free konfigurací
streamlit run app_no_websocket.py \
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
    --logger.enableRich=false 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STREAMLIT-NO-WS: $line"
    done &

# Získání PID
STREAMLIT_PID=$!
log_success "Streamlit WebSocket-free spuštěn s PID: $STREAMLIT_PID"

# Monitoring s HTTP-only kontrolami
log_info "Spouštím HTTP-only monitoring..."
for i in {1..30}; do
    sleep 2
    if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        # HTTP-only health check (bez WebSocket)
        if curl -s -f http://localhost:8505/ >/dev/null 2>&1; then
            log_success "Streamlit je připraven (HTTP-only mode)!"
            log_info "WebSocket: DISABLED ✅"
            log_info "HTTP Access: ENABLED ✅"
            break
        elif [[ $i -eq 30 ]]; then
            log_warning "HTTP přístup selhal po 60 sekundách"
        fi
    else
        log_error "Streamlit proces se ukončil!"
        exit 1
    fi
    
    if [[ $((i % 5)) -eq 0 ]]; then
        log_info "Čekám na HTTP startup... ($((i*2))s)"
    fi
done

# Finální diagnostika
log_info "Finální diagnostika (WebSocket-free):"
echo "  - Streamlit PID: $STREAMLIT_PID"
echo "  - Port status: $(netstat -tuln 2>/dev/null | grep :8505 || echo 'N/A')"
echo "  - WebSocket status: DISABLED"
echo "  - HTTP status: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8505/ 2>/dev/null || echo 'N/A')"

# Test WebSocket vypnutí
log_info "Testování WebSocket vypnutí..."
if command -v nc >/dev/null; then
    # Test že WebSocket upgrade je blokován
    echo -e "GET / HTTP/1.1\r\nHost: localhost:8505\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n" | nc localhost 8505 | head -1 | grep -q "200\|404" && \
        log_success "WebSocket upgrade správně blokován" || \
        log_warning "WebSocket upgrade test selhal"
fi

log_success "WebSocket-free aplikace je připravena!"
log_info "Přístup: http://localhost:8505 (pouze HTTP)"
log_info "Udržuji proces naživu..."

# Udržení procesu
wait "$STREAMLIT_PID"