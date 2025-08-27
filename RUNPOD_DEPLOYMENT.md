# 🚀 RunPod Deployment Guide

Komplexní průvodce pro nasazení AI Style Transfer aplikace na RunPod GPU servery.

## 📋 Přehled

Tato aplikace je optimalizována pro RunPod deployment s:
- ✅ Automatickým stažením base modelů
- ✅ GPU optimalizací a memory management
- ✅ Podporou externího storage pro LoRA modely
- ✅ Progress tracking a real-time monitoring
- ✅ Streamlit web interface na portu 8501

## 🛠️ Příprava Docker Image

### 1. Build a Push na Docker Hub

```bash
# Udělte execute práva build scriptu
chmod +x build_and_push.sh

# Build a push (nahraďte 'your-username' vaším Docker Hub username)
./build_and_push.sh your-username ai-style-transfer latest
```

### 2. Manuální Build (alternativa)

```bash
# Build pro linux/amd64 platformu
docker build --platform linux/amd64 -t your-username/ai-style-transfer:latest .

# Push na Docker Hub
docker login
docker push your-username/ai-style-transfer:latest
```

## 🎯 RunPod Template Setup

### 1. Vytvoření Template

1. **Přihlaste se do RunPod Console**
2. **Jděte na Templates → Create Template**
3. **Vyplňte základní informace:**
   - **Name:** `AI Style Transfer - GPU Optimized`
   - **Image:** `your-username/ai-style-transfer:latest`
   - **Description:** `Pokročilá aplikace pro přenos malířského stylu`

### 2. Konfigurace Template

#### Container Settings
- **Container Disk:** `50 GB`
- **Volume Size:** `20 GB`
- **Volume Mount Path:** `/workspace/lora_models`
- **Expose Ports:** `8501/http`

#### Environment Variables
```bash
FORCE_CPU=false
MAX_MEMORY_GB=24
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=auto
# BASE_MODEL - nepoužíváme base modely, pouze uživatelské
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
HF_HOME=/root/.cache/huggingface
TRANSFORMERS_CACHE=/root/.cache/huggingface
LORA_MODELS_PATH=/workspace/lora_models
```

### 3. Import Template (rychlá možnost)

```bash
# Použijte připravený template soubor
cp runpod_template.json runpod_template_configured.json

# Upravte imageName v souboru na vaši Docker image
sed -i 's/your-username/ACTUAL_USERNAME/g' runpod_template_configured.json
```

## 🚀 Deployment

### 1. Spuštění Podu

1. **Vyberte Template:** `AI Style Transfer - GPU Optimized`
2. **Vyberte GPU:** Doporučeno RTX 3080+ (min. 8GB VRAM)
3. **Nastavte Volume:** 20 GB pro LoRA modely
4. **Deploy Pod**

### 2. Doporučené GPU Konfigurace

| GPU Model | VRAM | MAX_MEMORY_GB | CPU_OFFLOAD | Poznámky |
|-----------|------|---------------|-------------|----------|
| RTX 4090 | 24GB | 24 | false | Optimální výkon |
| RTX 3090 | 24GB | 24 | auto | Velmi dobrý výkon |
| RTX 3080 | 10GB | 10 | true | Dobrý výkon |
| RTX 3070 | 8GB | 8 | true | Základní výkon |
| A100 | 40GB | 40 | false | Nejlepší výkon |

### 3. Přístup k Aplikaci

1. **Počkejte na inicializaci** (2-5 minut pro stažení base modelu)
2. **Otevřete port 8501** v RunPod Console
3. **Aplikace je dostupná** na `https://your-pod-id-8501.proxy.runpod.net`

## 📁 Správa LoRA Modelů

### 1. Nahrání LoRA Modelů

```bash
# Připojte se k podu přes SSH nebo Terminal
# Nahrajte LoRA modely do volume
cp your-lora-model.safetensors /workspace/lora_models/
```

### 2. Podporované Formáty

- ✅ **LoRA modely:** `.safetensors` (10-500 MB)
- ✅ **Full modely:** `.safetensors` (až 7 GB)
- ✅ **Automatická detekce** typu modelu

### 3. Organizace Modelů

```
/workspace/lora_models/
├── anime-style.safetensors
├── oil-painting.safetensors
├── watercolor.safetensors
└── sketch-style.safetensors
```

## 🔧 Optimalizace Výkonu

### 1. Memory Management

```bash
# Pro GPU s méně než 12 GB VRAM
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=true
MAX_MEMORY_GB=8

# Pro GPU s 16+ GB VRAM
ENABLE_ATTENTION_SLICING=false
ENABLE_CPU_OFFLOAD=false
MAX_MEMORY_GB=16
```

### 2. CUDA Optimalizace

```bash
# Optimální nastavení pro většinu GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# Pro GPU s více VRAM
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.8
```

## 🐛 Troubleshooting

### Časté Problémy

#### 1. Out of Memory (OOM)
```bash
# Snižte memory limit
MAX_MEMORY_GB=6
ENABLE_CPU_OFFLOAD=true
ENABLE_ATTENTION_SLICING=true
```

#### 2. Pomalé Načítání
```bash
# Zkontrolujte, zda je base model stažen
docker logs container_id | grep "Base model downloaded"

# Restartujte pod pokud se zasekl
```

#### 3. Port Nedostupný
```bash
# Zkontrolujte health check
curl -f http://localhost:8501/_stcore/health

# Zkontrolujte logy
docker logs container_id
```

### Monitoring

```bash
# GPU utilization
nvidia-smi

# Memory usage
free -h

# Disk space
df -h
```

## 📊 Výkon a Benchmarky

### Typické Časy Generování

| GPU | Rozlišení | Kroky | Čas |
|-----|-----------|-------|-----|
| RTX 4090 | 512x512 | 20 | 3-5s |
| RTX 3090 | 512x512 | 20 | 5-8s |
| RTX 3080 | 512x512 | 20 | 8-12s |
| RTX 3070 | 512x512 | 20 | 12-18s |

### Doporučené Nastavení

- **Rychlost:** `num_inference_steps=15`, `guidance_scale=7.0`
- **Kvalita:** `num_inference_steps=25`, `guidance_scale=8.5`
- **Vysoká kvalita:** `num_inference_steps=35`, `guidance_scale=10.0`

## 🔐 Bezpečnost

### 1. Environment Variables
- ❌ Nikdy necommitujte API klíče
- ✅ Používejte RunPod Secrets pro citlivé údaje
- ✅ Omezte přístup k volume s modely

### 2. Network Security
- ✅ Používejte HTTPS proxy RunPod
- ✅ Omezte přístup k SSH pokud není potřeba
- ✅ Pravidelně aktualizujte Docker image

## 📞 Podpora

### Užitečné Odkazy
- [RunPod Documentation](https://docs.runpod.io/)
- [Docker Hub Repository](https://hub.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Kontakt
Pro technickou podporu vytvořte issue v GitHub repository.

---

**🎉 Gratulujeme! Vaše AI Style Transfer aplikace je nyní připravena pro produkční použití na RunPod.**