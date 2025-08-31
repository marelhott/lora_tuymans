# 🔧 API Dokumentace

## Přehled

Tato dokumentace popisuje všechny funkce a API endpointy aplikace LoRA Tuymans Style Transfer.

## Hlavní funkce

### `apply_style()`

**Hlavní funkce pro aplikaci stylu na vstupní obrázek.**

```python
def apply_style(
    input_image: PIL.Image.Image,
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
) -> List[PIL.Image.Image]
```

#### Parametry

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `input_image` | `PIL.Image.Image` | - | Vstupní obrázek pro stylový přenos |
| `model_path` | `str` | - | Absolutní cesta k modelu (.safetensors) |
| `model_type` | `str` | - | Typ modelu: "lora" nebo "full_model" |
| `strength` | `float` | 0.8 | Síla aplikace stylu (0.1-1.0) |
| `guidance_scale` | `float` | 7.5 | CFG scale pro generování (1.0-20.0) |
| `num_inference_steps` | `int` | 20 | Počet inference kroků (10-50) |
| `progress_callback` | `callable` | None | Callback funkce pro progress tracking |
| `clip_skip` | `int` | 2 | CLIP layer skip (1-2) |
| `seed` | `Optional[int]` | None | Random seed pro reprodukovatelnost |
| `upscale_factor` | `int` | 1 | Upscaling faktor (1, 2, 4) |
| `num_images` | `int` | 1 | Počet generovaných variant (1-3) |
| `sampler` | `str` | "DPMSolverMultistepScheduler" | Typ sampleru |
| `variance_seed` | `Optional[int]` | None | Seed pro variance generování |
| `variance_strength` | `float` | 0.0 | Síla variance (0.0-1.0) |

#### Návratová hodnota

- **Typ:** `List[PIL.Image.Image]`
- **Popis:** Seznam vygenerovaných obrázků

#### Výjimky

- `FileNotFoundError`: Model soubor neexistuje
- `RuntimeError`: CUDA chyba nebo nedostatek paměti
- `ValueError`: Neplatné parametry

#### Příklad použití

```python
from PIL import Image

# Načtení vstupního obrázku
input_img = Image.open("input.jpg")

# Aplikace stylu
results = apply_style(
    input_image=input_img,
    model_path="/data/loras/tuymans_style.safetensors",
    model_type="lora",
    strength=0.8,
    guidance_scale=7.5,
    num_inference_steps=20,
    num_images=2
)

# Uložení výsledků
for i, img in enumerate(results):
    img.save(f"output_{i}.jpg")
```

---

## Model Management

### `load_base_model()`

**Načte a cachuje base model pro LoRA.**

```python
@st.cache_resource
def load_base_model() -> StableDiffusionXLImg2ImgPipeline
```

#### Návratová hodnota

- **Typ:** `StableDiffusionXLImg2ImgPipeline`
- **Popis:** Načtený a konfigurovaný pipeline

#### Chování

- Model je cachován pomocí `@st.cache_resource`
- Automatická detekce device (CUDA/CPU)
- Fallback na CPU při CUDA chybách
- Memory efficient optimalizace

### `load_full_model()`

**Načte a cachuje full model z cesty.**

```python
@st.cache_resource
def load_full_model(model_path: str) -> StableDiffusionXLImg2ImgPipeline
```

#### Parametry

| Parametr | Typ | Popis |
|----------|-----|-------|
| `model_path` | `str` | Absolutní cesta k .safetensors souboru |

#### Návratová hodnota

- **Typ:** `StableDiffusionXLImg2ImgPipeline`
- **Popis:** Načtený pipeline specifický pro model

---

## Utility funkce

### `get_optimal_device()`

**Detekuje optimální device pro inference.**

```python
def get_optimal_device() -> Tuple[str, str]
```

#### Návratová hodnota

- **Typ:** `Tuple[str, str]`
- **Popis:** (device, reason)
  - `device`: "cuda" nebo "cpu"
  - `reason`: Důvod výběru device

#### Logika detekce

1. Kontrola `FORCE_CPU` environment variable
2. Dostupnost CUDA
3. Dostupnost GPU s dostatečnou VRAM
4. Fallback na CPU

### `detect_model_type()`

**Detekuje typ modelu na základě cesty a obsahu.**

```python
def detect_model_type(model_path: str) -> str
```

#### Parametry

| Parametr | Typ | Popis |
|----------|-----|-------|
| `model_path` | `str` | Cesta k modelu |

#### Návratová hodnota

- **Typ:** `str`
- **Hodnoty:** "lora" nebo "full_model"

#### Detekční logika

1. Kontrola cesty (obsahuje "lora")
2. Velikost souboru (LoRA < 1GB, Full > 1GB)
3. Analýza obsahu safetensors

### `get_lora_models_list()`

**Vrací seznam dostupných LoRA modelů.**

```python
def get_lora_models_list() -> List[Dict[str, str]]
```

#### Návratová hodnota

- **Typ:** `List[Dict[str, str]]`
- **Struktura:**
  ```python
  [
      {
          "name": "model_name.safetensors",
          "path": "/absolute/path/to/model.safetensors",
          "size": "150MB"
      }
  ]
  ```

### `get_full_models_list()`

**Vrací seznam dostupných full modelů.**

```python
def get_full_models_list() -> List[Dict[str, str]]
```

#### Návratová hodnota

- **Typ:** `List[Dict[str, str]]`
- **Struktura:** Stejná jako `get_lora_models_list()`

---

## System Information

### `get_system_info()`

**Vrací informace o systému.**

```python
def get_system_info() -> Dict[str, Any]
```

#### Návratová hodnota

```python
{
    "platform": "Linux-5.4.0-x86_64",
    "python_version": "3.11.5",
    "torch_version": "2.2.1",
    "cuda_available": True,
    "cuda_version": "12.1",
    "gpu_name": "NVIDIA RTX 4090",
    "gpu_memory": "24GB",
    "cpu_count": 16,
    "memory_total": "32GB",
    "memory_available": "28GB"
}
```

---

## Progress Tracking

### Progress Callback

**Funkce pro sledování postupu generování.**

```python
def progress_callback(progress: float, text: str = "") -> None
```

#### Parametry

| Parametr | Typ | Popis |
|----------|-----|-------|
| `progress` | `float` | Progress hodnota (0.0-1.0) |
| `text` | `str` | Volitelný popisný text |

#### Fáze generování

| Progress | Fáze | Popis |
|----------|------|-------|
| 0.0-0.2 | Inicializace | Příprava prostředí |
| 0.2-0.5 | Model Loading | Načítání modelu |
| 0.5-0.6 | Preprocessing | Příprava dat |
| 0.6-0.85 | Generation | Generování obrázků |
| 0.85-0.95 | Postprocessing | Upscaling, cleanup |
| 0.95-1.0 | Finalization | Dokončení |

---

## Error Handling

### Typy chyb

#### `CUDAOutOfMemoryError`
```python
# Automatické řešení
1. Snížení batch size
2. Povolení CPU offload
3. Attention slicing
4. Fallback na CPU
```

#### `ModelLoadError`
```python
# Možné příčiny
1. Neexistující soubor
2. Poškozený model
3. Nekompatibilní formát
4. Nedostatečná oprávnění
```

#### `ValidationError`
```python
# Validace parametrů
1. Strength: 0.1 <= value <= 1.0
2. Guidance scale: 1.0 <= value <= 20.0
3. Steps: 10 <= value <= 50
4. Batch count: 1 <= value <= 3
```

---

## Environment Variables

### Výkon

| Variable | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `FORCE_CPU` | `bool` | `false` | Vynutit CPU režim |
| `MAX_MEMORY_GB` | `float` | `8` | Limit paměti v GB |
| `ENABLE_ATTENTION_SLICING` | `bool` | `true` | Memory optimization |
| `ENABLE_CPU_OFFLOAD` | `str` | `auto` | CPU offloading |

### Cesty

| Variable | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `LORA_MODELS_PATH` | `str` | `/data/loras` | Cesta k LoRA modelům |
| `FULL_MODELS_PATH` | `str` | `/data/models` | Cesta k full modelům |
| `HF_HOME` | `str` | `/root/.cache/huggingface` | HuggingFace cache |

### Model

| Variable | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `BASE_MODEL` | `str` | `stabilityai/stable-diffusion-xl-base-1.0` | Base model pro LoRA |

---

## Session State

### Klíče session state

```python
# Model selection
st.session_state.selected_lora: Optional[str]
st.session_state.selected_model: Optional[str]
st.session_state.current_model_path: Optional[str]

# Generated images
st.session_state.generated_images: List[PIL.Image.Image]
st.session_state.gallery_cache_key: int

# Progress tracking
st.session_state.current_step: int  # 0=idle, 1=loading, 2=generating
st.session_state.progress_value: float  # 0-100
st.session_state.is_processing: bool

# UI state
st.session_state.batch_count: int  # 1-3
```

---

## Schedulers

### Dostupné schedulers

```python
scheduler_map = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DDIMScheduler": DDIMScheduler,
    "LMSDiscreteScheduler": LMSDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler
}
```

### Doporučené nastavení

| Scheduler | Rychlost | Kvalita | Použití |
|-----------|----------|---------|----------|
| DPMSolverMultistepScheduler | ⭐⭐⭐ | ⭐⭐⭐ | Výchozí, vyvážené |
| EulerDiscreteScheduler | ⭐⭐⭐⭐ | ⭐⭐ | Rychlé prototypování |
| DDIMScheduler | ⭐⭐ | ⭐⭐⭐⭐ | Vysoká kvalita |

---

## Performance Benchmarks

### Typické časy generování

| GPU | VRAM | Steps | Time (1 image) | Time (3 images) |
|-----|------|-------|----------------|------------------|
| RTX 4090 | 24GB | 20 | 8-12s | 20-30s |
| RTX 4080 | 16GB | 20 | 12-18s | 30-45s |
| RTX 3080 | 10GB | 20 | 18-25s | 45-60s |
| CPU | 32GB | 20 | 180-300s | 450-750s |

### Memory usage

| Model Type | VRAM Usage | RAM Usage |
|------------|------------|----------|
| LoRA | 6-8GB | 4-6GB |
| Full Model | 8-12GB | 6-8GB |
| CPU Mode | 0GB | 12-16GB |

---

## Changelog

### v2.0.0 (Current)
- ✅ Model caching implementace
- ✅ Security improvements (XSRF protection)
- ✅ Updated dependencies
- ✅ Improved error handling
- ✅ Docker optimalizace

### v1.0.0
- ✅ Základní LoRA a Full model podpora
- ✅ Streamlit UI
- ✅ Progress tracking
- ✅ Lightbox galerie
- ✅ Docker deployment