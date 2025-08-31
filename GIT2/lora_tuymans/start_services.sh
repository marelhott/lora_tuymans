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
echo "  - Available memory: $(free -h | grep Mem | awk '{print $7}')"
echo "  - Available disk: $(df -h /app | tail -1 | awk '{print $4}')"

# Kontrola GPU dostupnosti
if command -v nvidia-smi &> /dev/null; then
    log_success "GPU detekována:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    log_warning "GPU není dostupná, přepínám na CPU"
    export FORCE_CPU=true
    export CUDA_VISIBLE_DEVICES=""
fi

# Kontrola CUDA s detailním výstupem
log_info "Kontrola PyTorch CUDA podpory..."
if python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null; then
    log_success "PyTorch CUDA je funkční"
else
    log_warning "PyTorch CUDA není dostupná nebo má problémy"
fi

# Vytvoření potřebných adresářů
mkdir -p /data/loras /data/models /home/appuser/.cache/huggingface

# Nastavení Streamlit konfigurace pro RunPod
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=true
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

# Vyčištění případných běžících procesů s lepší diagnostikou
log_info "Čištění běžících procesů..."
if pgrep -f streamlit >/dev/null; then
    log_warning "Nalezeny běžící Streamlit procesy, ukončuji je..."
    pkill -f streamlit || true
    sleep 3
    # Ověření ukončení
    if pgrep -f streamlit >/dev/null; then
        log_error "Streamlit procesy se nepodařilo ukončit, force kill..."
        pkill -9 -f streamlit || true
        sleep 2
    fi
else
    log_info "Žádné běžící Streamlit procesy nenalezeny"
fi

# Ověření dostupnosti portu s lepší diagnostikou
log_info "Kontrola portu 8505..."
if command -v lsof >/dev/null && lsof -Pi :8505 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_warning "Port 8505 je obsazený, ukončuji procesy..."
    if command -v fuser >/dev/null; then
        fuser -k 8505/tcp 2>/dev/null || true
    else
        # Fallback using netstat and kill
        local pid=$(netstat -tulpn 2>/dev/null | grep :8505 | awk '{print $7}' | cut -d'/' -f1 | head -1)
        if [[ -n "$pid" && "$pid" != "-" ]]; then
            log_warning "Ukončuji proces $pid na portu 8505"
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    sleep 3
else
    log_success "Port 8505 je volný"
fi

# Příprava prostředí pro Streamlit
log_info "Příprava prostředí pro Streamlit..."
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=true
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Kontrola aplikačních souborů
log_info "Kontrola aplikačních souborů..."
if [[ ! -f "app.py" ]]; then
    log_error "app.py nebyl nalezen v $(pwd)"
    exit 1
fi

if [[ ! -d ".streamlit" ]]; then
    log_warning ".streamlit složka neexistuje, vytvářím..."
    mkdir -p .streamlit
fi

# Spuštění Streamlit aplikace s monitoring
log_success "Spouštím Streamlit UI na portu 8505..."
log_info "Server address: 0.0.0.0:8505"
log_info "RunPod proxy bude dostupný za ~30 sekund"
log_info "Logs budou dostupné v real-time..."

# Spuštění WebSocket-free aplikace
streamlit run app.py \
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

# Získání PID Streamlit procesu
STREAMLIT_PID=$!
log_success "Streamlit spuštěn s PID: $STREAMLIT_PID"

# Monitoring loop
log_info "Spouštím monitoring loop..."
for i in {1..30}; do
    sleep 2
    if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        if curl -s http://localhost:8505/_stcore/health >/dev/null 2>&1; then
            log_success "Streamlit je připraven a odpovídá na portu 8505!"
            log_info "Health check: ✅ PASSED"
            break
        elif [[ $i -eq 30 ]]; then
            log_warning "Streamlit běží, ale health check selhal po 60 sekundách"
        fi
    else
        log_error "Streamlit proces se ukončil neočekávaně!"
        exit 1
    fi
    
    if [[ $((i % 5)) -eq 0 ]]; then
        log_info "Čekám na Streamlit startup... ($((i*2))s)"
    fi
done

# Finální diagnostika
log_info "Finální diagnostika:"
echo "  - Streamlit PID: $STREAMLIT_PID"
echo "  - Port status: $(netstat -tuln 2>/dev/null | grep :8505 || echo 'N/A')"
echo "  - Process status: $(ps aux | grep streamlit | grep -v grep || echo 'N/A')"

# Udržení procesu naživu
log_success "Aplikace je připravena! Udržuji proces naživu..."
wait "$STREAMLIT_PID"

# RunPod Multi-Service Startup Script
# Spouští Streamlit aplikaci a File Manager s vylepšeným error handlingem

echo "🚀 Starting RunPod Multi-Service Environment..."

# Vytvoření potřebných složek s detailním logováním
echo "📁 Creating directory structure..."
mkdir -p /data/loras /data/models /workspace/models/full /workspace/models/lora
chmod 755 /data /data/loras /data/models /workspace /workspace/models /workspace/models/full /workspace/models/lora

# Ověření vytvoření složek
for dir in "/data/loras" "/data/models" "/workspace/models/full" "/workspace/models/lora"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory created: $dir"
    else
        echo "❌ Failed to create: $dir"
    fi
done

# Nastavení environment variables
export PYTHONPATH="/app:$PYTHONPATH"
export HF_HOME="/data/.cache/huggingface"
export TORCH_HOME="/data/.cache/torch"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50000  # 50GB in MB
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50000  # 50GB in MB

# Funkce pro graceful shutdown
cleanup() {
    echo "🛑 Shutting down services..."
    kill $STREAMLIT_PID 2>/dev/null
    wait
    echo "✅ All services stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

# File Manager removed from deployment

# Start FTP Server for Mac Finder access
echo "🚀 Starting FTP Server on port 21..."

# Create FTP user and set permissions
adduser --disabled-password --gecos "" ftpuser
echo "ftpuser:password123" | chpasswd
chown -R ftpuser:ftpuser /data
chmod 755 /data

# Configure vsftpd
cat > /etc/vsftpd.conf << EOF
listen=YES
local_enable=YES
write_enable=YES
local_umask=022
dirmessage_enable=YES
use_localtime=YES
xferlog_enable=YES
connect_from_port_20=YES
chroot_local_user=YES
allow_writeable_chroot=YES
secure_chroot_dir=/var/run/vsftpd/empty
pam_service_name=vsftpd
rsa_cert_file=/etc/ssl/certs/ssl-cert-snakeoil.pem
rsa_private_key_file=/etc/ssl/private/ssl-cert-snakeoil.key
ssl_enable=NO
pasv_enable=YES
pasv_min_port=10000
pasv_max_port=10100
EOF

# Start FTP server
service vsftpd start > /tmp/ftp.log 2>&1 &
FTP_PID=$!
echo "FTP Server PID: $FTP_PID"

# Start Code Server
echo "🚀 Starting Code Server on port 8080..."
code-server --bind-addr 0.0.0.0:8080 --auth none /data > /tmp/codeserver.log 2>&1 &
CODE_PID=$!
echo "Code Server PID: $CODE_PID"

# Start FileBrowser
echo "🚀 Starting FileBrowser on port 8083..."
mkdir -p /data/filebrowser
filebrowser --port 8083 --address 0.0.0.0 --root /data --database /data/filebrowser/filebrowser.db --noauth > /tmp/filebrowser.log 2>&1 &
FILEBROWSER_PID=$!
echo "FileBrowser PID: $FILEBROWSER_PID"

# Spuštění Streamlit App s error handlingem
echo "🎨 Starting Streamlit App on port 8501..."
python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "🎨 Streamlit PID: $STREAMLIT_PID"

# Čekání na spuštění služeb s kontrolou
echo "⏳ Waiting for services to start..."
sleep 10

# Kontrola, zda služby běží
if kill -0 $FILEMANAGER_PID 2>/dev/null; then
    echo "✅ File Manager is running (PID: $FILEMANAGER_PID)"
else
    echo "❌ File Manager failed to start"
    cat /tmp/filemanager.log
fi

if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "✅ Streamlit is running (PID: $STREAMLIT_PID)"
else
    echo "❌ Streamlit failed to start"
    cat /tmp/streamlit.log
fi

echo "✅ Services started!"
echo "🎨 Streamlit App: http://localhost:8501"
echo "📁 File Manager: http://localhost:8502"
echo "📋 Logs: /tmp/streamlit.log, /tmp/filemanager.log"

# Monitoring loop s lepším error handlingem
while true; do
    # Kontrola Streamlit
    if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
        echo "❌ Streamlit crashed, restarting..."
        echo "📋 Last Streamlit log:"
        tail -20 /tmp/streamlit.log
        python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true > /tmp/streamlit.log 2>&1 &
        STREAMLIT_PID=$!
        echo "🔄 Streamlit restarted (PID: $STREAMLIT_PID)"
    fi
    
    # Kontrola File Manager
    # File Manager monitoring removed
    
    # Kontrola FTP Server
     if ! pgrep vsftpd > /dev/null; then
         echo "❌ FTP Server crashed, restarting..."
         echo "📋 Last FTP log:"
         tail -20 /tmp/ftp.log
         service vsftpd start > /tmp/ftp.log 2>&1 &
         FTP_PID=$!
         echo "🔄 FTP Server restarted (PID: $FTP_PID)"
     fi
     
     # Kontrola Code Server
     if ! kill -0 $CODE_PID 2>/dev/null; then
         echo "❌ Code Server crashed, restarting..."
         echo "📋 Last Code Server log:"
         tail -20 /tmp/codeserver.log
         code-server --bind-addr 0.0.0.0:8080 --auth none /data > /tmp/codeserver.log 2>&1 &
         CODE_PID=$!
         echo "🔄 Code Server restarted (PID: $CODE_PID)"
     fi
     
     # Kontrola FileBrowser
     if ! kill -0 $FILEBROWSER_PID 2>/dev/null; then
         echo "❌ FileBrowser crashed, restarting..."
         echo "📋 Last FileBrowser log:"
         tail -20 /tmp/filebrowser.log
         filebrowser --port 8083 --address 0.0.0.0 --root /data --database /data/filebrowser/filebrowser.db --noauth > /tmp/filebrowser.log 2>&1 &
         FILEBROWSER_PID=$!
         echo "🔄 FileBrowser restarted (PID: $FILEBROWSER_PID)"
     fi
    
    sleep 30
done