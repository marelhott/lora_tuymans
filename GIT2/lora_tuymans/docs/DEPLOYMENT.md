# 🚀 Deployment Guide

## Přehled

Tento guide pokrývá deployment aplikace LoRA Tuymans Style Transfer na různých platformách a prostředích.

## 📋 Obsah

- [🏠 Lokální deployment](#-lokální-deployment)
- [🐳 Docker deployment](#-docker-deployment)
- [☁️ Cloud platformy](#️-cloud-platformy)
- [🖥️ RunPod deployment](#️-runpod-deployment)
- [🔧 Production setup](#-production-setup)
- [📊 Monitoring](#-monitoring)
- [🔒 Security](#-security)

---

## 🏠 Lokální Deployment

### Požadavky

**Minimální:**
- Python 3.10+
- 8GB RAM
- 10GB volného místa
- Internet připojení

**Doporučené:**
- Python 3.11
- NVIDIA GPU (RTX 3080+)
- 16GB+ RAM
- 50GB+ volného místa
- Rychlé SSD

### Instalace

#### 1. Příprava prostředí

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev git

# CentOS/RHEL
sudo yum install python311 python311-devel git

# macOS (Homebrew)
brew install python@3.11 git

# Windows
# Stáhněte Python 3.11 z python.org
# Nainstalujte Git z git-scm.com
```

#### 2. Klonování a setup

```bash
# Klonování repozitáře
git clone https://github.com/your-username/lora_tuymans.git
cd lora_tuymans

# Vytvoření virtual environment
python3.11 -m venv venv

# Aktivace (Linux/Mac)
source venv/bin/activate

# Aktivace (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Instalace dependencies
pip install -r requirements.txt
```

#### 3. Konfigurace

```bash
# Vytvoření složek
mkdir -p data/loras data/models

# Kopírování konfigurace
cp .streamlit/config.toml.example .streamlit/config.toml

# Editace konfigurace (volitelné)
nano .streamlit/config.toml
```

#### 4. Spuštění

```bash
# Development mode
streamlit run app.py

# Production mode
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Environment Variables

```bash
# .env soubor
FORCE_CPU=false
MAX_MEMORY_GB=16
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=auto
LORA_MODELS_PATH=./data/loras
FULL_MODELS_PATH=./data/models
```

---

## 🐳 Docker Deployment

### Rychlé spuštění

```bash
# Pull a spuštění
docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/data \
  mulenmara1505/lora_tuymans:latest
```

### Lokální build

```bash
# Build image
docker build -t lora_tuymans:local .

# Spuštění
docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/data \
  -e MAX_MEMORY_GB=16 \
  lora_tuymans:local
```

### Docker Compose

**`docker-compose.yml`**
```yaml
version: '3.8'

services:
  lora_tuymans:
    image: mulenmara1505/lora_tuymans:latest
    container_name: lora_tuymans
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./data/loras:/data/loras
      - ./data/models:/data/models
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - MAX_MEMORY_GB=16
      - ENABLE_ATTENTION_SLICING=true
      - ENABLE_CPU_OFFLOAD=auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  huggingface_cache:
```

```bash
# Spuštění
docker-compose up -d

# Logs
docker-compose logs -f

# Zastavení
docker-compose down
```

### Multi-stage Production Build

**`Dockerfile.prod`**
```dockerfile
# Build stage
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app /data/loras /data/models /root/.cache/huggingface && \
    chown -R appuser:appuser /app /data

WORKDIR /app

# Copy application
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser .streamlit/ .streamlit/
COPY --chown=appuser:appuser start_services.sh .

RUN chmod +x start_services.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

USER appuser

EXPOSE 8501

ENTRYPOINT ["/app/start_services.sh"]
```

---

## ☁️ Cloud Platformy

### AWS EC2

#### Instance doporučení

| Instance Type | vCPU | RAM | GPU | VRAM | Cena/hod | Použití |
|---------------|------|-----|-----|------|----------|----------|
| g4dn.xlarge | 4 | 16GB | T4 | 16GB | $0.526 | Development |
| g4dn.2xlarge | 8 | 32GB | T4 | 16GB | $0.752 | Production |
| g5.xlarge | 4 | 16GB | A10G | 24GB | $1.006 | High-end |
| g5.2xlarge | 8 | 32GB | A10G | 24GB | $1.212 | Enterprise |

#### Setup

```bash
# 1. Spuštění instance
# - AMI: Deep Learning AMI (Ubuntu 20.04)
# - Instance type: g4dn.xlarge nebo vyšší
# - Security group: Port 8501 open

# 2. Připojení
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Instalace Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker ubuntu

# 4. NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# 5. Deployment
docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 8501:8501 \
  --restart unless-stopped \
  mulenmara1505/lora_tuymans:latest
```

### Google Cloud Platform

#### Compute Engine

```bash
# 1. Vytvoření instance
gcloud compute instances create lora-tuymans-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# 2. SSH připojení
gcloud compute ssh lora-tuymans-vm --zone=us-central1-a

# 3. Docker deployment
sudo docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 8501:8501 \
  --restart unless-stopped \
  mulenmara1505/lora_tuymans:latest
```

### Azure

#### Container Instances

```yaml
# azure-container.yaml
apiVersion: 2019-12-01
location: eastus
name: lora-tuymans
properties:
  containers:
  - name: lora-tuymans
    properties:
      image: mulenmara1505/lora_tuymans:latest
      ports:
      - port: 8501
        protocol: TCP
      resources:
        requests:
          cpu: 4
          memoryInGB: 16
          gpu:
            count: 1
            sku: K80
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8501
  restartPolicy: Always
tags: {}
type: Microsoft.ContainerInstance/containerGroups
```

```bash
# Deployment
az container create --resource-group myResourceGroup --file azure-container.yaml
```

---

## 🖥️ RunPod Deployment

### Template Setup

#### 1. Custom Template

```json
{
  "name": "LoRA Tuymans Style Transfer",
  "imageName": "mulenmara1505/lora_tuymans:latest",
  "dockerArgs": "",
  "ports": "8501/http",
  "volumeInPath": "/workspace",
  "volumeMountPath": "/data",
  "env": [
    {
      "key": "MAX_MEMORY_GB",
      "value": "16"
    },
    {
      "key": "ENABLE_ATTENTION_SLICING",
      "value": "true"
    }
  ]
}
```

#### 2. Doporučené GPU

| GPU | VRAM | Cena/hod | Performance |
|-----|------|----------|-------------|
| RTX 3080 | 10GB | $0.34 | Základní |
| RTX 3090 | 24GB | $0.50 | Doporučené |
| RTX 4090 | 24GB | $0.79 | Optimální |
| A100 | 40GB | $1.89 | Enterprise |

#### 3. Startup Script

```bash
#!/bin/bash
# runpod-startup.sh

# Vytvoření složek
mkdir -p /workspace/loras /workspace/models

# Symbolické linky
ln -sf /workspace/loras /data/loras
ln -sf /workspace/models /data/models

# Spuštění aplikace
cd /app
./start_services.sh
```

### Persistent Storage

```bash
# Nahrání modelů
# 1. Připojte se k RunPod instance
# 2. Nahrajte modely do /workspace/
wget -O /workspace/loras/tuymans_style.safetensors "https://your-model-url"

# 3. Ověření
ls -la /workspace/loras/
ls -la /workspace/models/
```

---

## 🔧 Production Setup

### Reverse Proxy (Nginx)

**`/etc/nginx/sites-available/lora_tuymans`**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # File upload limit
    client_max_body_size 10G;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    
    # Health check endpoint
    location /_stcore/health {
        proxy_pass http://localhost:8501;
        access_log off;
    }
}
```

### Systemd Service

**`/etc/systemd/system/lora_tuymans.service`**
```ini
[Unit]
Description=LoRA Tuymans Style Transfer
After=network.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 127.0.0.1:8501:8501 \
  --restart unless-stopped \
  -v /opt/lora_tuymans/data:/data \
  -e MAX_MEMORY_GB=16 \
  mulenmara1505/lora_tuymans:latest

ExecStop=/usr/bin/docker stop lora_tuymans
ExecStopPost=/usr/bin/docker rm lora_tuymans

[Install]
WantedBy=multi-user.target
```

```bash
# Aktivace služby
sudo systemctl enable lora_tuymans
sudo systemctl start lora_tuymans
sudo systemctl status lora_tuymans
```

### Load Balancing

**`docker-compose.prod.yml`**
```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - app1
      - app2
    restart: unless-stopped

  app1:
    image: mulenmara1505/lora_tuymans:latest
    environment:
      - MAX_MEMORY_GB=8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    volumes:
      - ./data:/data
    restart: unless-stopped

  app2:
    image: mulenmara1505/lora_tuymans:latest
    environment:
      - MAX_MEMORY_GB=8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    volumes:
      - ./data:/data
    restart: unless-stopped
```

---

## 📊 Monitoring

### Prometheus + Grafana

**`monitoring/docker-compose.yml`**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  node_exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'

  nvidia_gpu_exporter:
    image: mindprince/nvidia_gpu_prometheus_exporter
    ports:
      - "9445:9445"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

volumes:
  prometheus_data:
  grafana_data:
```

### Health Checks

```bash
#!/bin/bash
# health_check.sh

# Kontrola aplikace
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application: OK"
else
    echo "❌ Application: FAILED"
    exit 1
fi

# Kontrola GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU: OK"
else
    echo "❌ GPU: FAILED"
fi

# Kontrola paměti
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -lt 90 ]; then
    echo "✅ Memory: ${MEM_USAGE}%"
else
    echo "⚠️ Memory: ${MEM_USAGE}% (HIGH)"
fi

# Kontrola disku
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 85 ]; then
    echo "✅ Disk: ${DISK_USAGE}%"
else
    echo "⚠️ Disk: ${DISK_USAGE}% (HIGH)"
fi
```

---

## 🔒 Security

### Firewall (UFW)

```bash
# Základní konfigurace
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH
sudo ufw allow ssh

# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Streamlit (pouze z localhost)
sudo ufw allow from 127.0.0.1 to any port 8501

# Aktivace
sudo ufw enable
```

### SSL/TLS (Let's Encrypt)

```bash
# Instalace Certbot
sudo apt install certbot python3-certbot-nginx

# Získání certifikátu
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Přidat: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Environment Security

```bash
# .env.production
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_GATHER_USAGE_STATS=false

# Omezení přístupu
STREAMLIT_SERVER_ADDRESS=127.0.0.1  # Pouze localhost
```

### Docker Security

```bash
# Spuštění s omezenými oprávněními
docker run -d \
  --name lora_tuymans \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/tmp \
  --cap-drop ALL \
  --cap-add CHOWN \
  --cap-add SETUID \
  --cap-add SETGID \
  --security-opt no-new-privileges \
  --gpus all \
  -p 127.0.0.1:8501:8501 \
  mulenmara1505/lora_tuymans:latest
```

---

## 🚨 Troubleshooting

### Časté problémy

#### Port již používán
```bash
# Najít proces
sudo netstat -tulpn | grep 8501
sudo lsof -i :8501

# Ukončit proces
sudo kill -9 PID
```

#### GPU nedostupná
```bash
# Kontrola NVIDIA drivers
nvidia-smi

# Kontrola Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Restart NVIDIA services
sudo systemctl restart nvidia-docker
```

#### Out of Memory
```bash
# Kontrola využití
free -h
nvidia-smi

# Vyčištění cache
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
docker system prune -f
```

### Logs

```bash
# Application logs
docker logs lora_tuymans
docker logs -f lora_tuymans  # Follow

# System logs
sudo journalctl -u lora_tuymans
sudo journalctl -f  # Follow

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 📈 Performance Tuning

### GPU Optimalizace

```bash
# NVIDIA persistence mode
sudo nvidia-smi -pm 1

# Max performance mode
sudo nvidia-smi -ac 877,1215  # Memory,Graphics clock

# Power limit
sudo nvidia-smi -pl 300  # 300W
```

### System Optimalizace

```bash
# Kernel parameters
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Docker Optimalizace

```json
// /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
```

---

**📧 Support:** [your-email@example.com](mailto:your-email@example.com)  
**📖 Dokumentace:** [README.md](../README.md)  
**🐛 Issues:** [GitHub Issues](https://github.com/your-username/lora_tuymans/issues)