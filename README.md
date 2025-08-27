# AI Style Transfer - Přenos malířského stylu

Pokročilá Streamlit aplikace pro přenos malířského stylu pomocí LoRA modelů a plných safetensors modelů s podporou až 7 GB souborů.

## 🚀 Funkce

- **Podpora velkých modelů**: Nahrávání souborů až do 10 GB
- **Dva typy modelů**: LoRA modely (.safetensors) a plné safetensors modely
- **Automatická detekce**: Rozpoznání typu modelu na základě obsahu
- **GPU optimalizace**: Automatická detekce a využití dostupného hardware
- **Progress tracking**: Sledování průběhu nahrávání a zpracování
- **Memory management**: Pokročilá správa paměti pro velké modely
- **Docker podpora**: Připraveno pro deployment na RunPod

## 📋 Požadavky

### Lokální spuštění
- Python 3.8+
- CUDA kompatibilní GPU (doporučeno)
- Minimálně 8 GB RAM (16+ GB doporučeno)
- 50+ GB volného místa na disku

### RunPod deployment
- GPU instance s minimálně 16 GB VRAM
- Docker support

## 🛠️ Instalace

### Lokální instalace

```bash
# Klonování repozitáře
git clone <repository-url>
cd ai-style-transfer

# Instalace závislostí
pip install -r requirements.txt

# Spuštění aplikace
streamlit run app.py
```

### Docker instalace

```bash
# Build Docker image
docker build -t ai-style-transfer .

# Spuštění s Docker Compose
docker-compose up
```

## 🌐 RunPod Deployment

### Příprava

1. **Nahrajte Docker image na Docker Hub**:
```bash
docker build -t your-username/ai-style-transfer:latest .
docker push your-username/ai-style-transfer:latest
```

2. **Nastavte RunPod API klíč**:
```bash
export RUNPOD_API_KEY="your-api-key"
```

3. **Spusťte deployment skript**:
```bash
python runpod_deploy.py
```

### Manuální deployment na RunPod

1. Vytvořte nový Pod na RunPod
2. Použijte Docker image: `your-username/ai-style-transfer:latest`
3. Nastavte porty: `8501/http`
4. Doporučené GPU: RTX A5000 nebo lepší
5. Minimální VRAM: 16 GB
6. Volume: 100 GB pro modely

### Environment Variables

```bash
FORCE_CPU=false                    # Vynutit CPU místo GPU
MAX_MEMORY_GB=24                   # Maximální paměť v GB
ENABLE_ATTENTION_SLICING=true      # Povolить attention slicing
ENABLE_CPU_OFFLOAD=auto            # CPU offload (true/false/auto)
BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0  # Základní model
```

## 📖 Použití

### Podporované formáty modelů

1. **LoRA modely** (10-500 MB):
   - Malé soubory .safetensors
   - Vyžadují základní SDXL model
   - Rychlejší načítání

2. **Plné safetensors modely** (1-7 GB):
   - Kompletní modely v jednom souboru
   - Samostatné, nevyžadují základní model
   - Pomalejší načítání, ale často lepší kvalita

### Kroky použití

1. **Nahrajte vstupní obrázek** (PNG, JPG, JPEG)
2. **Nahrajte model** (.safetensors soubor)
3. **Nastavte parametry**:
   - **Strength** (0.1-1.0): Síla aplikace stylu
   - **Guidance Scale** (1-20): Řízení generování
   - **Inference Steps** (10-50): Počet kroků generování
4. **Klikněte na "🎨 Aplikovat styl"**
5. **Stáhněte výsledek**

## ⚙️ Konfigurace

### Streamlit konfigurace (.streamlit/config.toml)

```toml
[server]
maxUploadSize = 10240  # 10 GB limit
gatherUsageStats = false

[theme]
base = "dark"
```

### Optimalizace paměti

Aplikace automaticky detekuje dostupnou paměť a aplikuje optimalizace:

- **Attention Slicing**: Snižuje VRAM požadavky
- **CPU Offload**: Přesouvá části modelu na CPU
- **Memory Cleanup**: Automatické čištění paměti
- **Chunked Loading**: Postupné načítání velkých souborů

## 🐛 Řešení problémů

### Časté problémy

1. **"CUDA out of memory"**:
   - Snižte rozlišení vstupního obrázku
   - Nastavte `ENABLE_CPU_OFFLOAD=true`
   - Snižte `num_inference_steps`

2. **"PEFT backend is required"**:
   - Aplikace automaticky nainstaluje PEFT
   - Restartujte aplikaci po instalaci

3. **Pomalé nahrávání velkých souborů**:
   - Zkontrolujte internetové připojení
   - Progress bar zobrazuje průběh

### Logy a debugging

```bash
# Spuštění s debug módem
streamlit run app.py --logger.level=debug

# Sledování GPU paměti
nvidia-smi -l 1
```

## 📊 Výkon

### Doporučené konfigurace

| GPU | VRAM | Doporučené nastavení |
|-----|------|----------------------|
| RTX 3060 | 12 GB | CPU Offload: true, Attention Slicing: true |
| RTX 3080 | 10 GB | CPU Offload: auto, Attention Slicing: true |
| RTX 4090 | 24 GB | CPU Offload: false, Attention Slicing: false |
| A100 | 40 GB | Všechny optimalizace vypnuté |

### Časy generování (přibližné)

- **LoRA model**: 30-60 sekund
- **Plný safetensors**: 60-120 sekund
- **CPU pouze**: 5-15 minut