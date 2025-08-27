#!/bin/bash

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