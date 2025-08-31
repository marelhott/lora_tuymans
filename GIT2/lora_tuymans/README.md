# 🎨 LoRA Tuymans Style Transfer

**Profesionální AI aplikace pro stylový přenos obrazů pomocí LoRA fine-tuned Stable Diffusion XL v charakteristickém stylu belgického malíře Luca Tuymanse.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF6B6B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

## 📋 Obsah

- [🚀 Rychlý start](#-rychlý-start)
- [✨ Funkce](#-funkce)
- [🛠️ Instalace](#️-instalace)
- [🐳 Docker deployment](#-docker-deployment)
- [📖 Použití](#-použití)
- [⚙️ Konfigurace](#️-konfigurace)
- [🔧 API dokumentace](#-api-dokumentace)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Přispívání](#-přispívání)

## 🚀 Rychlý start

### Lokální spuštění
```bash
# Klonování repozitáře
git clone https://github.com/your-username/lora_tuymans.git
cd lora_tuymans

# Instalace dependencies
pip install -r requirements.txt

# Spuštění aplikace
streamlit run app.py
```

### Docker spuštění
```bash
# Pull a spuštění
docker run -d \
  --gpus all \
  -p 8501:8501 \
  -v ./data/loras:/data/loras \
  -v ./data/models:/data/models \
  mulenmara1505/lora_tuymans:latest
```

## ✨ Funkce

### 🎨 **AI Style Transfer**
- **LoRA modely**: Podpora custom LoRA modelů
- **Full modely**: Podpora .safetensors modelů
- **Batch processing**: Generování více variant najednou
- **Real-time progress**: Live sledování postupu generování

### 🖥️ **Moderní UI**
- **Lobe UI design**: Profesionální a čistý interface
- **Responsive layout**: Optimalizováno pro všechny obrazovky
- **Lightbox galerie**: Kliknutelné zvětšení obrázků
- **Drag & drop**: Jednoduché nahrávání souborů

### ⚡ **Výkon & Optimalizace**
- **Model caching**: Rychlejší načítání modelů
- **Memory management**: Automatické čištění CUDA cache
- **GPU acceleration**: Plná podpora NVIDIA GPU
- **CPU fallback**: Automatický přepínač při CUDA chybách

### 🔒 **Bezpečnost**
- **XSRF protection**: Ochrana proti CSRF útokům
- **Input validation**: Validace nahrávaných souborů
- **Non-root Docker**: Bezpečné spouštění v kontejneru
- **Environment isolation**: Izolované prostředí

## 🛠️ Instalace

### Systémové požadavky

**Minimální:**
- Python 3.10+
- 8GB RAM
- 10GB volného místa

**Doporučené:**
- Python 3.11
- NVIDIA GPU s 12GB+ VRAM
- 16GB+ RAM
- 50GB+ volného místa

### Krok za krokem

1. **Příprava prostředí**
```bash
# Vytvoření virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# nebo
venv\Scripts\activate     # Windows
```

2. **Instalace dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Konfigurace**
```bash
# Vytvoření složek pro modely
mkdir -p data/loras data/models

# Kopírování konfigurace (volitelné)
cp .streamlit/config.toml.example .streamlit/config.toml
```

4. **Spuštění**
```bash
streamlit run app.py
```

## 🐳 Docker Deployment

### Lokální build
```bash
# Build image
docker build -t lora_tuymans .

# Spuštění
docker run -d \
  --name lora_tuymans \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/data \
  lora_tuymans
```

### RunPod deployment
```bash
# Použití pre-built image
docker run -d \
  --gpus all \
  -p 8501:8501 \
  -e MAX_MEMORY_GB=16 \
  -e ENABLE_ATTENTION_SLICING=true \
  -v /workspace/loras:/data/loras \
  -v /workspace/models:/data/models \
  mulenmara1505/lora_tuymans:latest
```

### Environment variables
```bash
# Výkon
FORCE_CPU=false                    # Vynutit CPU režim
MAX_MEMORY_GB=16                    # Limit paměti
ENABLE_ATTENTION_SLICING=true       # Memory optimization
ENABLE_CPU_OFFLOAD=auto             # CPU offloading

# Cesty
LORA_MODELS_PATH=/data/loras        # Cesta k LoRA modelům
FULL_MODELS_PATH=/data/models       # Cesta k full modelům
HF_HOME=/root/.cache/huggingface    # HuggingFace cache

# Model
BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
```

## 📖 Použití

### Základní workflow

1. **📁 Nahrání obrázku**
   - Klikněte na "Browse files" v pravém sidebaru
   - Podporované formáty: PNG, JPG, JPEG
   - Maximální velikost: 10GB

2. **🎨 Výběr modelu**
   - **LoRA modely**: Umístěte .safetensors soubory do `data/loras/`
   - **Full modely**: Umístěte .safetensors soubory do `data/models/`
   - Klikněte na kroužek vedle názvu modelu pro výběr

3. **⚙️ Nastavení parametrů**
   - **Strength** (0.1-1.0): Síla aplikace stylu
   - **CFG Scale** (1-20): Guidance scale pro generování
   - **Steps** (10-50): Počet inference kroků
   - **Clip Skip** (1-2): CLIP layer skip
   - **Batch Count** (1-3): Počet generovaných variant

4. **🚀 Generování**
   - Klikněte "Generovat"
   - Sledujte progress bar
   - Výsledky se zobrazí automaticky

5. **🖼️ Prohlížení výsledků**
   - Klikněte na obrázek pro lightbox
   - Dvojklik pro zavření lightboxu
   - Automatické uložení do session

### Pokročilé funkce

**🎲 Random Seed**
```
☑️ Random Seed
Seed: 42
```

**🔄 Variance Seed**
```
☑️ Variance Seed
Variance Seed: 123
Variance Strength: 0.1
```

**⬆️ Upscaling**
```
☑️ Upscaling
Factor: 2x nebo 4x
```

## ⚙️ Konfigurace

### Streamlit konfigurace

**`.streamlit/config.toml`**
```toml
[server]
maxUploadSize = 10240           # 10GB limit
gatherUsageStats = false
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = true               # CORS podpora
enableXsrfProtection = true     # CSRF ochrana
enableWebsocketCompression = true

[theme]
base = "light"
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"
```

### Docker konfigurace

**Dockerfile optimalizace:**
- Multi-stage build pro menší image
- Non-root user pro bezpečnost
- Health checks pro monitoring
- Optimalizované layer caching

## 🔧 API Dokumentace

### Hlavní funkce

#### `apply_style()`
```python
def apply_style(
    input_image: PIL.Image,
    model_path: str,
    model_type: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    progress_callback: callable = None,
    clip_skip: int = 2,
    seed: Optional[int] = None,
    upscale_factor: int = 1,
    num_images: int = 1,
    sampler: str = "DPMSolverMultistepScheduler",
    variance_seed: Optional[int] = None,
    variance_strength: float = 0.0
) -> List[PIL.Image]
```

**Parametry:**
- `input_image`: Vstupní obrázek (PIL.Image)
- `model_path`: Cesta k modelu (.safetensors)
- `model_type`: Typ modelu ("lora" nebo "full_model")
- `strength`: Síla aplikace stylu (0.1-1.0)
- `guidance_scale`: CFG scale (1.0-20.0)
- `num_inference_steps`: Počet kroků (10-50)
- `clip_skip`: CLIP skip (1-2)
- `seed`: Random seed (volitelné)
- `upscale_factor`: Upscaling faktor (1, 2, 4)
- `num_images`: Počet generovaných obrázků (1-3)

**Návratová hodnota:**
- `List[PIL.Image]`: Seznam vygenerovaných obrázků

#### `load_base_model()` (Cached)
```python
@st.cache_resource
def load_base_model() -> StableDiffusionXLImg2ImgPipeline
```
Načte a cachuje base model pro LoRA.

#### `load_full_model()` (Cached)
```python
@st.cache_resource
def load_full_model(model_path: str) -> StableDiffusionXLImg2ImgPipeline
```
Načte a cachuje full model z cesty.

### Utility funkce

#### `get_optimal_device()`
```python
def get_optimal_device() -> Tuple[str, str]
```
Detekuje optimální device (CUDA/CPU) s fallback logikou.

#### `detect_model_type()`
```python
def detect_model_type(model_path: str) -> str
```
Detekuje typ modelu na základě cesty a obsahu.

#### `get_lora_models_list()`
```python
def get_lora_models_list() -> List[Dict[str, str]]
```
Vrací seznam dostupných LoRA modelů.

#### `get_full_models_list()`
```python
def get_full_models_list() -> List[Dict[str, str]]
```
Vrací seznam dostupných full modelů.

## 🐛 Troubleshooting

### Časté problémy

**❌ CUDA Out of Memory**
```bash
# Řešení 1: Snížit batch size
Batch Count: 1

# Řešení 2: Povolit CPU offload
ENABLE_CPU_OFFLOAD=true

# Řešení 3: Attention slicing
ENABLE_ATTENTION_SLICING=true

# Řešení 4: Vynutit CPU
FORCE_CPU=true
```

**❌ Model se nenačte**
```bash
# Zkontrolovat cestu
ls -la data/loras/
ls -la data/models/

# Zkontrolovat formát
file model.safetensors

# Zkontrolovat oprávnění
chmod 644 model.safetensors
```

**❌ Streamlit se nespustí**
```bash
# Zkontrolovat port
netstat -tulpn | grep 8501

# Zkontrolovat dependencies
pip check

# Reinstalace
pip install --force-reinstall streamlit
```

**❌ Docker problémy**
```bash
# Zkontrolovat GPU
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Zkontrolovat volumes
docker inspect container_name

# Logs
docker logs container_name
```

### Performance tuning

**🚀 Optimalizace rychlosti:**
```python
# V .env nebo environment
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=false  # Pokud máte dostatek VRAM
MAX_MEMORY_GB=16          # Podle vaší GPU
```

**💾 Optimalizace paměti:**
```python
# Pro nižší VRAM
ENABLE_ATTENTION_SLICING=true
ENABLE_CPU_OFFLOAD=true
MAX_MEMORY_GB=8
```

## 🤝 Přispívání

### Development setup

```bash
# Fork a clone
git clone https://github.com/your-username/lora_tuymans.git
cd lora_tuymans

# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Spuštění testů
pytest tests/
```

### Coding standards

- **Python**: PEP 8, type hints
- **Commits**: Conventional commits
- **Testing**: pytest, coverage >80%
- **Documentation**: Docstrings, README updates

### Pull Request proces

1. **Fork** repozitáře
2. **Vytvořte** feature branch
3. **Implementujte** změny s testy
4. **Aktualizujte** dokumentaci
5. **Otevřete** Pull Request

## 📄 Licence

MIT License - viz [LICENSE](LICENSE) soubor.

## 🙏 Poděkování

- **Luc Tuymans** - inspirace stylem
- **Stability AI** - Stable Diffusion XL
- **HuggingFace** - Diffusers library
- **Streamlit** - Web framework

---

**📧 Kontakt:** [your-email@example.com](mailto:your-email@example.com)  
**🐛 Issues:** [GitHub Issues](https://github.com/your-username/lora_tuymans/issues)  
**💬 Diskuze:** [GitHub Discussions](https://github.com/your-username/lora_tuymans/discussions)
- ✅ **Multiple variants** (1-3 obrázky)
- ✅ **Upscaling support** (2x, 4x)
- ✅ **Seed control** pro reprodukovatelnost
- ✅ **Advanced parameters** (CFG, steps, clip skip)
- ✅ **Lightbox gallery** s dvojklikem
- ✅ **Responsive design** pro všechny zařízení

## 🐳 Docker Image

**Repository**: `mulenmara1505/lora_tuymans_cursor:latest`

**Features**:
- Multi-stage build pro minimalizaci velikosti
- CUDA 12.1.1 + cuDNN 8 support
- Non-root user pro security
- Health checks pro RunPod
- Optimalizované pro GPU inference

## 📁 Struktura projektu

```
lora_tuymans/
├── app.py                 # Hlavní Streamlit aplikace
├── Dockerfile            # Optimalizovaný pro RunPod
├── requirements.txt      # Pinned verze dependencies
├── start_services.sh     # RunPod start skript
├── build_and_push.sh     # Build a push skript
└── .streamlit/           # Streamlit konfigurace
```

## 🚀 Lokální vývoj

```bash
# Clone repository
git clone <repository-url>
cd lora_tuymans

# Instalace dependencies
pip install -r requirements.txt

# Spuštění aplikace
streamlit run app.py
```

## 🔒 Security Features

- Non-root user v Docker containeru
- Minimal base image (Ubuntu 22.04)
- Health checks pro monitoring
- Environment variable validation
- Secure file permissions

## 📊 Performance

- **GPU acceleration** s CUDA 12.1.1
- **Memory efficient** attention slicing
- **CPU offloading** pro velké modely
- **Optimized inference** pro RTX 4090/5090

## 🤝 Contributing

Tato verze je optimalizovaná pro RunPod deployment s vylepšeným UI. Backend logika zůstává identická s původní verzí.

## 📄 License

Původní model a LoRA weights podléhají licenčním podmínkám Stable Diffusion XL a Luc Tuymans uměleckého stylu.