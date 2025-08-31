# 🐛 Troubleshooting Guide

## Přehled

Komplexní guide pro řešení problémů s aplikací LoRA Tuymans Style Transfer.

## 📋 Obsah

- [🚨 Kritické chyby](#-kritické-chyby)
- [⚡ Výkonnostní problémy](#-výkonnostní-problémy)
- [🖥️ GPU problémy](#️-gpu-problémy)
- [🐳 Docker problémy](#-docker-problémy)
- [🌐 Síťové problémy](#-síťové-problémy)
- [📁 Problémy s modely](#-problémy-s-modely)
- [🔧 Diagnostické nástroje](#-diagnostické-nástroje)
- [📞 Získání podpory](#-získání-podpory)

---

## 🚨 Kritické chyby

### "CUDA out of memory"

**Příznaky:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Příčiny:**
- Nedostatek VRAM
- Příliš velký batch size
- Memory leak
- Jiné procesy používají GPU

**Řešení:**

#### 1. Okamžité řešení
```bash
# Restart aplikace
docker restart lora_tuymans
# nebo
Ctrl+C a znovu spustit
```

#### 2. Optimalizace paměti
```bash
# Environment variables
export ENABLE_ATTENTION_SLICING=true
export ENABLE_CPU_OFFLOAD=true
export MAX_MEMORY_GB=6
```

#### 3. Snížení parametrů
- Batch Count: 3 → 1
- Upscaling: Vypnout
- Steps: 30 → 20

#### 4. Vyčištění GPU paměti
```python
# V Python konzoli
import torch
torch.cuda.empty_cache()
```

#### 5. Kontrola jiných procesů
```bash
# Zobrazit GPU procesy
nvidia-smi

# Ukončit jiné procesy
sudo kill -9 PID
```

### "ModuleNotFoundError"

**Příznaky:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'diffusers'
```

**Řešení:**

#### 1. Kontrola virtual environment
```bash
# Aktivace venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Kontrola aktivace
which python
which pip
```

#### 2. Reinstalace dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstalace
pip install --force-reinstall -r requirements.txt

# Nebo jednotlivě
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install streamlit
```

#### 3. Kontrola Python verze
```bash
# Minimální verze: 3.10
python --version

# Pokud je starší
sudo apt install python3.11
python3.11 -m venv venv
```

### "Permission denied"

**Příznaky:**
```
PermissionError: [Errno 13] Permission denied: '/data/loras'
FileNotFoundError: [Errno 2] No such file or directory
```

**Řešení:**

#### 1. Oprávnění složek
```bash
# Vytvoření složek
mkdir -p data/loras data/models

# Nastavení oprávnění
chmod 755 data/
chmod 755 data/loras/
chmod 755 data/models/

# Vlastnictví
sudo chown -R $USER:$USER data/
```

#### 2. Docker oprávnění
```bash
# Přidání uživatele do docker skupiny
sudo usermod -aG docker $USER

# Restart session
newgrp docker
```

#### 3. SELinux/AppArmor
```bash
# Kontrola SELinux
getenforce

# Dočasné vypnutí
sudo setenforce 0

# Trvalé vypnutí
sudo nano /etc/selinux/config
# SELINUX=disabled
```

---

## ⚡ Výkonnostní problémy

### Pomalé generování

**Diagnostika:**
```bash
# Monitoring GPU
watch -n 1 nvidia-smi

# Monitoring CPU
htop

# Monitoring paměti
free -h
```

**Optimalizace:**

#### 1. GPU optimalizace
```bash
# Persistence mode
sudo nvidia-smi -pm 1

# Performance mode
sudo nvidia-smi -ac 877,1215

# Power limit (podle GPU)
sudo nvidia-smi -pl 300
```

#### 2. System optimalizace
```bash
# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Cache pressure
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
```

#### 3. Aplikační optimalizace
```bash
# Environment variables
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=false  # Pokud máte dostatek VRAM
MAX_MEMORY_GB=16
```

### Vysoké využití RAM

**Monitoring:**
```bash
# Continuous monitoring
watch -n 1 'free -h && ps aux --sort=-%mem | head -10'

# Memory map
sudo pmap -x $(pgrep python)
```

**Řešení:**

#### 1. Memory cleanup
```bash
# Clear cache
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches

# Docker cleanup
docker system prune -f
docker volume prune -f
```

#### 2. Aplikační optimalizace
```python
# V kódu - garbage collection
import gc
gc.collect()

# CUDA cache cleanup
import torch
torch.cuda.empty_cache()
```

---

## 🖥️ GPU problémy

### GPU není detekována

**Diagnostika:**
```bash
# Základní kontrola
nvidia-smi
lspci | grep -i nvidia

# CUDA kontrola
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

**Řešení:**

#### 1. Driver problémy
```bash
# Kontrola verze
nvidia-smi

# Reinstalace (Ubuntu)
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install nvidia-driver-535
sudo reboot
```

#### 2. CUDA toolkit
```bash
# Instalace CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run

# Environment
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 3. Docker GPU support
```bash
# NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### GPU přehřívání

**Monitoring:**
```bash
# Teplota
watch -n 1 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits'

# Fan speed
nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits
```

**Řešení:**

#### 1. Fan curve
```bash
# Manual fan control
sudo nvidia-smi -pm 1
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=80"
```

#### 2. Power limiting
```bash
# Snížení power limitu
sudo nvidia-smi -pl 250  # Z 300W na 250W
```

#### 3. Undervolting
```bash
# Pomocí MSI Afterburner nebo nvidia-settings
# Snížení core voltage o 50-100mV
```

---

## 🐳 Docker problémy

### Container se nespustí

**Diagnostika:**
```bash
# Logs
docker logs lora_tuymans

# Inspect
docker inspect lora_tuymans

# Events
docker events --filter container=lora_tuymans
```

**Časté chyby:**

#### 1. Port již používán
```bash
# Najít proces
sudo netstat -tulpn | grep 8501
sudo lsof -i :8501

# Ukončit
sudo kill -9 PID

# Nebo použít jiný port
docker run -p 8502:8501 ...
```

#### 2. Volume problémy
```bash
# Kontrola cest
ls -la $(pwd)/data

# Oprávnění
sudo chown -R 1000:1000 data/

# SELinux context
sudo chcon -Rt svirt_sandbox_file_t data/
```

#### 3. GPU access
```bash
# Test GPU v containeru
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Pokud nefunguje
sudo systemctl restart nvidia-docker
sudo systemctl restart docker
```

### Image build problémy

**Časté chyby:**

#### 1. Network timeout
```bash
# Proxy nastavení
docker build --build-arg HTTP_PROXY=http://proxy:8080 .

# DNS
echo '{"dns": ["8.8.8.8", "8.8.4.4"]}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

#### 2. Disk space
```bash
# Cleanup
docker system prune -a
docker volume prune

# Kontrola místa
df -h
docker system df
```

#### 3. Memory limit
```bash
# Zvýšení build memory
docker build --memory=8g .

# Nebo v daemon.json
{
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Name": "memlock",
      "Soft": -1
    }
  }
}
```

---

## 🌐 Síťové problémy

### Aplikace není dostupná

**Diagnostika:**
```bash
# Port listening
sudo netstat -tulpn | grep 8501

# Firewall
sudo ufw status
sudo iptables -L

# Connectivity
curl -I http://localhost:8501
telnet localhost 8501
```

**Řešení:**

#### 1. Firewall konfigurace
```bash
# UFW
sudo ufw allow 8501/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

#### 2. Streamlit konfigurace
```toml
# .streamlit/config.toml
[server]
address = "0.0.0.0"  # Ne 127.0.0.1
port = 8501
enableCORS = true
```

#### 3. Cloud provider security groups
```bash
# AWS Security Group
# Inbound: TCP 8501 from 0.0.0.0/0

# GCP Firewall
gcloud compute firewall-rules create allow-streamlit \
  --allow tcp:8501 \
  --source-ranges 0.0.0.0/0
```

### Pomalé stahování modelů

**Diagnostika:**
```bash
# Rychlost připojení
speedtest-cli

# DNS
nslookup huggingface.co

# Traceroute
traceroute huggingface.co
```

**Optimalizace:**

#### 1. Mirror/CDN
```bash
# HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Nebo lokální cache
export HF_HOME=/fast/cache/huggingface
```

#### 2. Parallel downloads
```python
# V kódu
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="model_name",
    max_workers=8,
    resume_download=True
)
```

---

## 📁 Problémy s modely

### Model se nenačte

**Diagnostika:**
```bash
# Kontrola souboru
file model.safetensors
ls -la model.safetensors

# Integrita
sha256sum model.safetensors

# Obsah
python -c "from safetensors import safe_open; print(list(safe_open('model.safetensors', framework='pt').keys())[:10])"
```

**Časté problémy:**

#### 1. Poškozený soubor
```bash
# Re-download
wget -c -O model.safetensors "MODEL_URL"

# Nebo z backup
cp backup/model.safetensors data/loras/
```

#### 2. Nekompatibilní formát
```python
# Konverze z .ckpt
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_ckpt("model.ckpt")
pipe.save_pretrained("converted_model", safe_serialization=True)
```

#### 3. Špatná cesta
```bash
# Kontrola struktury
tree data/

# Správné umístění
# LoRA: data/loras/model.safetensors
# Full: data/models/model.safetensors
```

### Špatné výsledky

**Diagnostika:**
- Zkontrolujte typ modelu (LoRA vs Full)
- Ověřte kompatibilitu s base modelem
- Testujte s různými parametry

**Řešení:**

#### 1. Model compatibility
```python
# Kontrola base modelu
print(f"Base model: {BASE_MODEL}")
print(f"LoRA model: {model_path}")

# Kompatibilní kombinace:
# SDXL base + SDXL LoRA
# SD 1.5 base + SD 1.5 LoRA
```

#### 2. Parameter tuning
```python
# Pro LoRA modely
strength = 0.6-0.8
cfg_scale = 7-9

# Pro Full modely
strength = 0.7-0.9
cfg_scale = 6-8
```

---

## 🔧 Diagnostické nástroje

### System Information Script

```bash
#!/bin/bash
# system_info.sh

echo "=== SYSTEM INFORMATION ==="
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"

echo "\n=== GPU INFORMATION ==="nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "\n=== DOCKER INFORMATION ==="
docker --version
docker info | grep -E "Server Version|Storage Driver|Logging Driver"

echo "\n=== PYTHON ENVIRONMENT ==="
python --version
pip list | grep -E "torch|diffusers|streamlit"

echo "\n=== APPLICATION STATUS ==="
curl -s http://localhost:8501/_stcore/health && echo "✅ App: OK" || echo "❌ App: FAILED"

echo "\n=== DISK USAGE ==="
du -sh data/loras/ data/models/ 2>/dev/null || echo "Model directories not found"
```

### Performance Monitor

```bash
#!/bin/bash
# monitor.sh

while true; do
    clear
    echo "=== $(date) ==="
    
    echo "\n🖥️  CPU:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    
    echo "\n💾 RAM:"
    free -h | grep Mem | awk '{printf "Used: %s / %s (%.1f%%)\n", $3, $2, $3/$2*100}'
    
    echo "\n🎮 GPU:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU: %s%% | VRAM: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4}'
    
    echo "\n🐳 Docker:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -2
    
    sleep 5
done
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

ERRORS=0

# Application health
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application: OK"
else
    echo "❌ Application: FAILED"
    ((ERRORS++))
fi

# GPU health
if nvidia-smi > /dev/null 2>&1; then
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if [ $TEMP -lt 85 ]; then
        echo "✅ GPU: OK (${TEMP}°C)"
    else
        echo "⚠️  GPU: HOT (${TEMP}°C)"
        ((ERRORS++))
    fi
else
    echo "❌ GPU: FAILED"
    ((ERRORS++))
fi

# Memory health
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -lt 90 ]; then
    echo "✅ Memory: OK (${MEM_USAGE}%)"
else
    echo "⚠️  Memory: HIGH (${MEM_USAGE}%)"
    ((ERRORS++))
fi

# Disk health
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 85 ]; then
    echo "✅ Disk: OK (${DISK_USAGE}%)"
else
    echo "⚠️  Disk: HIGH (${DISK_USAGE}%)"
    ((ERRORS++))
fi

# Model directories
if [ -d "data/loras" ] && [ -d "data/models" ]; then
    LORA_COUNT=$(find data/loras -name "*.safetensors" | wc -l)
    MODEL_COUNT=$(find data/models -name "*.safetensors" | wc -l)
    echo "✅ Models: ${LORA_COUNT} LoRA, ${MODEL_COUNT} Full"
else
    echo "❌ Model directories missing"
    ((ERRORS++))
fi

echo "\n=== SUMMARY ==="
if [ $ERRORS -eq 0 ]; then
    echo "🎉 All systems operational!"
    exit 0
else
    echo "⚠️  Found $ERRORS issues"
    exit 1
fi
```

---

## 📞 Získání podpory

### Před kontaktováním podpory

1. **Spusťte diagnostiku:**
   ```bash
   ./system_info.sh > system_report.txt
   ./health_check.sh >> system_report.txt
   ```

2. **Shromážděte logy:**
   ```bash
   # Application logs
   docker logs lora_tuymans > app_logs.txt 2>&1
   
   # System logs
   journalctl -u docker > system_logs.txt
   
   # GPU logs
   nvidia-smi -q > gpu_info.txt
   ```

3. **Připravte informace:**
   - Verze aplikace
   - Operační systém
   - Hardware specifikace
   - Kroky k reprodukci problému
   - Chybové hlášky

### Kontaktní informace

**🐛 GitHub Issues:**
- URL: https://github.com/your-username/lora_tuymans/issues
- Template: Bug report, Feature request
- Přiložte: system_report.txt, logy

**💬 GitHub Discussions:**
- URL: https://github.com/your-username/lora_tuymans/discussions
- Kategorie: Q&A, General, Ideas

**📧 Email podpora:**
- Email: your-email@example.com
- Předmět: "[LoRA Tuymans] Popis problému"
- Přiložte: diagnostické soubory

**📱 Community:**
- Discord: [Server link]
- Reddit: r/StableDiffusion
- Stack Overflow: tag "lora-tuymans"

### Template pro hlášení chyb

```markdown
## 🐛 Bug Report

**Popis problému:**
Stručný popis toho, co se děje.

**Kroky k reprodukci:**
1. Krok 1
2. Krok 2
3. Krok 3

**Očekávané chování:**
Co by se mělo stát.

**Skutečné chování:**
Co se skutečně stalo.

**Prostředí:**
- OS: [Ubuntu 22.04]
- GPU: [RTX 4090]
- Docker: [Yes/No]
- Verze: [v2.0.0]

**Logy:**
```
Vložte relevantní logy zde
```

**Screenshots:**
Přiložte screenshots pokud je to relevantní.

**Dodatečné informace:**
Jakékoliv další informace o problému.
```

---

**🔧 Tento troubleshooting guide pokrývá většinu běžných problémů. Pokud váš problém není uveden, neváhejte kontaktovat podporu!**