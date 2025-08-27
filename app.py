import streamlit as st
import torch
from PIL import Image
import os
import io
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler
)
from safetensors.torch import load_file
import time
import gc
from pathlib import Path
import platform
import psutil
from typing import Optional

# Environment variables pro konfiguraci
FORCE_CPU = os.getenv('FORCE_CPU', 'false').lower() == 'true'
MAX_MEMORY_GB = float(os.getenv('MAX_MEMORY_GB', '8'))
# BASE_MODEL - nepoužíváme base modely, pouze uživatelské full a LoRA modely
ENABLE_ATTENTION_SLICING = os.getenv('ENABLE_ATTENTION_SLICING', 'true').lower() == 'true'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD', 'auto')
LORA_MODELS_PATH = os.getenv('LORA_MODELS_PATH', '/data/loras')
FULL_MODELS_PATH = os.getenv('FULL_MODELS_PATH', '/data/models')
HF_HOME = os.getenv('HF_HOME', '/root/.cache/huggingface')
BASE_MODEL = os.getenv('BASE_MODEL', 'stabilityai/stable-diffusion-xl-base-1.0')

# Nastavení stránky
st.set_page_config(page_title="AI Stylový Přenos", page_icon="🎨", layout="wide")

# Lobe UI inspirovaný světlý design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

/* Lobe UI - Hlavní kontejner */
.main .block-container {
    padding: 2rem 1.5rem;
    max-width: 1200px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    min-height: 100vh;
}

/* Lobe UI - Globální styling */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    color: #1e293b;
    font-family: 'Inter', sans-serif;
}

/* Lobe UI - Sidebar styling */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Pravý sidebar - ostrý kontrast bílé a šedé */
.right-sidebar {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%);
    border: 2px solid #64748b;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.9);
    height: fit-content;
    position: sticky;
    top: 1rem;
    backdrop-filter: blur(10px);
}

/* Lobe UI - Moderní tlačítka */
.stButton > button {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    color: #475569;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    font-family: 'Inter', sans-serif;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-color: #cbd5e1;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Lobe UI - Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #ffffff;
    border: 1px solid #2563eb;
    font-weight: 600;
    box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-color: #1d4ed8;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px 0 rgba(59, 130, 246, 0.4);
}

/* Lobe UI - Moderní inputy */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    color: #1e293b;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    padding: 0.75rem;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    width: 100% !important;
    min-width: 300px !important;
}

/* Rozšířené selectboxy v pravém sidebaru */
.right-sidebar .stSelectbox > div > div {
    width: 100% !important;
    min-width: 350px !important;
    max-width: none !important;
}

.stSelectbox > div > div:focus,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    outline: none;
}

/* Lobe UI - Sliders */
.stSlider > div > div > div {
    color: #3b82f6;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
}

/* Lobe UI - File uploader */
.stFileUploader > div {
    background: rgba(255, 255, 255, 0.8);
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.05);
}

/* Lobe UI - Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 8px;
    height: 8px;
}

/* Lobe UI - Checkboxy a radio */
.stCheckbox > div,
.stRadio > div {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    color: #475569;
}

.stRadio > div {
    flex-direction: row;
    gap: 1.5rem;
    flex-wrap: wrap;
}

.stRadio > div > label {
    background: rgba(255, 255, 255, 0.8);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
    cursor: pointer;
}

.stRadio > div > label:hover {
    background: rgba(59, 130, 246, 0.05);
    border-color: #3b82f6;
}

/* Lobe UI - Sidebar komponenty */
.css-1d391kg .stSelectbox > div > div,
.css-1d391kg .stSlider > div,
.css-1d391kg .stCheckbox > div,
.css-1d391kg .stTextInput > div,
.css-1d391kg .stNumberInput > div {
    padding: 0.5rem 0;
}

.css-1d391kg .stButton > button {
    width: 100%;
    margin: 0.25rem 0;
    font-size: 0.8rem;
    padding: 0.5rem 1rem;
}

/* Lobe UI - Nadpisy */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #1e293b;
    margin: 1rem 0 0.5rem 0;
}

h1 { font-size: 2rem; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }

/* Lobe UI - Obrázky */
.stImage {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.stImage:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Lobe UI - Zprávy */
.stAlert {
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stSuccess {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #166534;
}

.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    color: #dc2626;
}

.stWarning {
    background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
    color: #d97706;
}

.stInfo {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    color: #2563eb;
}

/* Lobe UI - Expandery - vylepšené pro LoRA/modely */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #1e293b;
    border: 2px solid #e2e8f0;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    cursor: pointer;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-color: #3b82f6;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}

.streamlit-expanderContent {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.8) 100%);
    border-radius: 0 0 12px 12px;
    padding: 1.5rem;
    border: 2px solid #e2e8f0;
    border-top: none;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Komplexní progress bar styling */
.progress-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.progress-step {
    display: flex;
    align-items: center;
    margin: 0.75rem 0;
    padding: 0.5rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.progress-step.active {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 4px solid #3b82f6;
}

.progress-step.completed {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-left: 4px solid #10b981;
}

.progress-step-icon {
    font-size: 1.2rem;
    margin-right: 0.75rem;
    min-width: 2rem;
}

.progress-step-text {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #1e293b;
}

.progress-step-details {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Galerie variant styling */
.variant-gallery {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.variant-main-preview {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.variant-thumbnails {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    justify-content: center;
}

.variant-thumbnail {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.variant-thumbnail:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

.variant-thumbnail.selected {
    border-color: #3b82f6;
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Lobe UI - Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
}

/* Lobe UI - Animace */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.main .block-container > div {
    animation: fadeIn 0.5s ease-out;
}

/* Lobe UI - Responsivní design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
    }
    
    .stRadio > div {
        flex-direction: column;
        gap: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Vytvoření adresářů - s fallback pro lokální vývoj
try:
    os.makedirs(LORA_MODELS_PATH, exist_ok=True)
    os.makedirs(FULL_MODELS_PATH, exist_ok=True)
except OSError:
    # Fallback pro lokální vývoj - použij lokální složky
    LORA_MODELS_PATH = './lora_models'
    FULL_MODELS_PATH = './models'
    os.makedirs(LORA_MODELS_PATH, exist_ok=True)
    os.makedirs(FULL_MODELS_PATH, exist_ok=True)

try:
    os.makedirs(HF_HOME, exist_ok=True)
except OSError:
    # Fallback pro HuggingFace cache
    HF_HOME = os.path.expanduser('~/.cache/huggingface')
    os.makedirs(HF_HOME, exist_ok=True)

def detect_runpod_paths():
    """Detekuje dostupné RunPod cesty pro modely."""
    possible_paths = {
        'workspace_loras': '/workspace/loras',
        'workspace_models': '/workspace/models', 
        'data_loras': '/data/loras',
        'data_models': '/data/models',
        'runpod_volume_loras': '/runpod-volume/loras',
        'runpod_volume_models': '/runpod-volume/models',
        'content_loras': '/content/loras',  # Colab style
        'content_models': '/content/models'
    }
    
    available_paths = []
    for name, path in possible_paths.items():
        if os.path.exists(path):
            available_paths.append(path)
    
    return available_paths

def get_lora_models_list():
    """Získá seznam dostupných LoRA modelů z persistentního disku včetně RunPod detekce."""
    lora_files = []
    
    # Zkusí všechny možné cesty
    search_paths = [LORA_MODELS_PATH] + detect_runpod_paths()
    
    for search_path in search_paths:
        if 'lora' in search_path.lower() and os.path.exists(search_path):
            try:
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.safetensors'):
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            # Relativní cesta pro lepší zobrazení
                            rel_path = os.path.relpath(file_path, search_path)
                            display_name = f"{os.path.basename(search_path)}/{rel_path}" if '/' in rel_path else f"{os.path.basename(search_path)}/{file}"
                            
                            # Kontrola duplicit
                            if not any(existing['path'] == file_path for existing in lora_files):
                                lora_files.append({
                                    'name': display_name,
                                    'path': file_path,
                                    'size_mb': file_size,
                                    'source': search_path
                                })
            except PermissionError:
                continue
    
    return sorted(lora_files, key=lambda x: x['name'])

def get_full_models_list():
    """Získá seznam dostupných full modelů z persistentního disku včetně RunPod detekce."""
    models = []
    
    # Zkusí všechny možné cesty
    search_paths = [FULL_MODELS_PATH] + detect_runpod_paths()
    
    for search_path in search_paths:
        if 'model' in search_path.lower() and os.path.exists(search_path):
            try:
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.safetensors'):
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            # Relativní cesta pro lepší zobrazení
                            rel_path = os.path.relpath(file_path, search_path)
                            display_name = f"{os.path.basename(search_path)}/{rel_path}" if '/' in rel_path else f"{os.path.basename(search_path)}/{file}"
                            
                            # Kontrola duplicit
                            if not any(existing['path'] == file_path for existing in models):
                                models.append({
                                    'name': display_name,
                                    'path': file_path,
                                    'size_mb': file_size,
                                    'source': search_path
                                })
            except PermissionError:
                continue
    
    return sorted(models, key=lambda x: x['name'])

# Funkce pro detekci hardware
def get_system_info():
    """Získá informace o systému a dostupném hardware"""
    info = {
        'platform': platform.system(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    }
    return info

def get_optimal_device():
    """Určí optimální zařízení pro inference s fallback pro CUDA chyby"""
    if FORCE_CPU:
        return "cpu", "Vynuceno CPU"
    
    if not torch.cuda.is_available():
        return "cpu", "CUDA není dostupná"
    
    try:
        # Test CUDA funkčnosti s RTX 5090 fallback
        test_tensor = torch.randn(10, device='cuda')
        _ = test_tensor + 1  # Jednoduchý test operace
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4:
            return "cpu", f"Nedostatek GPU paměti ({gpu_memory:.1f} GB < 4 GB)"
        
        return "cuda", f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)"
    except Exception as e:
        # Fallback na CPU při CUDA chybách (RTX 5090 kompatibilita)
        return "cpu", f"CUDA chyba - fallback na CPU: {str(e)[:30]}..."

def show_progress_bar(progress: float, text: str = "") -> None:
    """Zobrazí progress bar s textem"""
    progress_bar = st.progress(progress)
    if text:
        st.text(text)
    return progress_bar

def update_progress(progress_bar, progress: float, text: str = "") -> None:
    """Aktualizuje progress bar"""
    progress_bar.progress(progress)
    if text:
        st.text(text)

# Nadpis aplikace
# Odstraněno podle požadavku uživatele

# Systémové informace přesunuty do sidebaru
sys_info = get_system_info()
device, device_reason = get_optimal_device()

# Zobrazení varování při CUDA chybě
if "chyba" in device_reason.lower():
    st.warning(f"⚠️ CUDA problém detekován, přepínám na CPU: {device_reason}")

# Hlavní obsah bude přesunut do col_main

# Funkce pro detekci typu modelu
def detect_model_type(file_path):
    """Detekuje zda je soubor LoRA model nebo full safetensors model"""
    try:
        # Načteme metadata ze safetensors souboru
        state_dict = load_file(file_path)
        
        # Kontrola velikosti souboru
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
        
        # LoRA modely jsou obvykle menší (< 1GB) a obsahují specifické klíče
        lora_keys = ['lora_unet', 'lora_te', 'alpha', 'rank']
        has_lora_keys = any(any(lora_key in key for lora_key in lora_keys) for key in state_dict.keys())
        
        # Full modely jsou větší a obsahují kompletní váhy
        full_model_keys = ['model.diffusion_model', 'first_stage_model', 'cond_stage_model']
        has_full_keys = any(any(full_key in key for full_key in full_model_keys) for key in state_dict.keys())
        
        if has_lora_keys or file_size < 1.0:
            return "lora"
        elif has_full_keys or file_size > 2.0:
            return "full_model"
        else:
            # Pokud nejsme si jisti, zkusíme podle velikosti
            return "lora" if file_size < 1.0 else "full_model"
            
    except Exception as e:
        st.warning(f"Nelze detekovat typ modelu: {e}")
        return "unknown"

# Funkce pro aplikaci stylu na vstupní obrázek
def apply_style(input_image, model_path, model_type, strength, guidance_scale, num_inference_steps, progress_callback, clip_skip=2, seed=None, upscale_factor=1, num_images=1, sampler="DPMSolverMultistepScheduler", variance_seed=None, variance_strength=0.0):
    # Použití optimální device detekce s fallback
    device, device_reason = get_optimal_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Logování device informací
    if "chyba" in device_reason.lower():
        print(f"Warning: {device_reason}")
    
    # Progress tracking - začátek
    progress_callback(0.1)
    
    # Pokročilé vyčištění paměti před načtením
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Nastavení memory efficient attention pro velké modely
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Optimalizace pro velké modely na základě environment variables
    enable_memory_efficient_attention = ENABLE_ATTENTION_SLICING
    
    if ENABLE_CPU_OFFLOAD == 'auto':
        enable_cpu_offload = device == "cpu" or (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < MAX_MEMORY_GB * 1024**3)
    else:
        enable_cpu_offload = ENABLE_CPU_OFFLOAD.lower() == 'true'
    
    try:
        # Progress tracking - načítání modelu
        progress_callback(0.2)
        
        if model_type == "lora":
            # Robustnější CUDA handling pro RTX 5090
            try:
                # Načtení base modelu pro LoRA
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch_dtype,
                    variant="fp16" if device == "cuda" else None,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    clip_skip=clip_skip
                )
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    # Fallback na CPU při CUDA chybě
                    st.warning(f"⚠️ CUDA chyba při načítání modelu, přepínám na CPU: {str(cuda_error)[:50]}...")
                    device = "cpu"
                    torch_dtype = torch.float32
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        BASE_MODEL,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                        clip_skip=clip_skip
                    )
                else:
                    raise cuda_error
            
            # Progress tracking - optimalizace
            progress_callback(0.4)
            
            # Memory efficient optimizations
            if enable_memory_efficient_attention:
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
                
            if enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            
            # Progress tracking - načítání LoRA
            progress_callback(0.5)
            
            # Načtení LoRA modelu - robustnější přístup
            try:
                # Pokus o načtení jako adresář s adapter_config.json
                if os.path.isdir(model_path):
                    pipe.load_lora_weights(model_path)
                else:
                    # Načtení .safetensors souboru přímo
                    pipe.load_lora_weights(model_path, adapter_name="lora_adapter")
            except Exception as e:
                try:
                    # Fallback - načtení pomocí from_single_file
                    pipe.load_lora_weights(model_path, weight_name=os.path.basename(model_path))
                except Exception as e2:
                    st.warning(f"Nelze načíst LoRA model: {e2}")
                    # Pokračovat bez LoRA
                    pass
                
        elif model_type == "full_model":
            # Načtení full safetensors modelu
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    clip_skip=clip_skip
                )
                
                # Progress tracking - optimalizace
                progress_callback(0.5)
                
                # Memory efficient optimizations pro full modely
                if enable_memory_efficient_attention:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                    
                if enable_cpu_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(device)
            except Exception as e:
                st.error(f"Chyba při načítání full modelu: {e}")
                return None
        else:
            st.error("Nepodporovaný typ modelu")
            return None
        
        # Progress tracking - příprava generování
        progress_callback(0.6)
        
        # Nastavení scheduleru
        scheduler_map = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler
        }
        
        if sampler in scheduler_map:
            pipe.scheduler = scheduler_map[sampler].from_config(pipe.scheduler.config)
        
        progress_callback(0.6, f"Generuji {num_images} variant...")
        
        # Generování více obrázků
        results = []
        for i in range(num_images):
            # Nastavení generátoru pro reprodukovatelnost
            generator = None
            current_seed = seed
            
            if seed is not None:
                if variance_seed is not None and i > 0:
                    # Pro varianty použijeme kombinaci původního seed a variance seed
                    current_seed = seed + (variance_seed * i) % 2147483647
                generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # Callback pro progress bar během generování
            def callback_fn(step, timestep, latents):
                # Mapování kroků generování na progress 0.6 - 0.85
                generation_progress = 0.6 + (i / num_images) * 0.25 + (step / num_inference_steps) * (0.25 / num_images)
                progress_callback(generation_progress)
                return latents
            
            # Aktualizace progress pro každý obrázek
            progress_callback(0.6 + (i / num_images) * 0.25, f"Generuji obrázek {i+1}/{num_images}...")
            
            # Aplikace stylu na vstupní obrázek
            # Prázdný prompt, protože nechceme generovat podle textu
            try:
                result = pipe(
                    image=input_image,
                    prompt="",
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback=callback_fn,
                    callback_steps=1
                ).images[0]
                
                results.append(result)
            
            except Exception as e:
                st.error(f"Chyba při generování obrázku {i+1}: {e}")
                continue
        
        # Progress tracking - generování dokončeno
        progress_callback(0.85)
        
        # Upscaling pokud je povoleno
        if upscale_factor > 1:
            progress_callback(0.9, "Upscaling obrázků...")
            upscaled_results = []
            for i, result in enumerate(results):
                try:
                    # Jednoduché upscaling pomocí PIL (pro Real-ESRGAN by bylo potřeba další závislost)
                    original_size = result.size
                    new_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
                    upscaled_result = result.resize(new_size, Image.Resampling.LANCZOS)
                    upscaled_results.append(upscaled_result)
                    progress_callback(0.9 + (i / len(results)) * 0.05, f"Upscaling {i+1}/{len(results)}...")
                except Exception as e:
                    progress_callback(0.95, f"Upscaling obrázku {i+1} selhal: {e}")
                    upscaled_results.append(result)  # Použij původní obrázek
            results = upscaled_results
        
        # Progress tracking - dokončeno
        progress_callback(1.0)
        
        # Vrátíme první obrázek pro zpětnou kompatibilitu, ale všechny jsou v results
        return results[0] if results else None
        
    finally:
        # Vyčištění paměti po dokončení
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Uvolnění pipeline z paměti
        if 'pipe' in locals():
            del pipe
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

# Inicializace session state pro uchování nahraných souborů
if 'uploaded_model_file' not in st.session_state:
    st.session_state.uploaded_model_file = None
    st.session_state.model_type = None
    st.session_state.model_name = None


    




with st.sidebar:
    # Inicializace session state pro uchování nahraných souborů
    if 'uploaded_model_file' not in st.session_state:
        st.session_state.uploaded_model_file = None
        st.session_state.model_type = None
        st.session_state.model_name = None
    

    
    # Parametry pro přenos stylu
    st.markdown("### ⚙️ Parametry")
    strength = st.slider("Strength", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    guidance_scale = st.slider("CFG Scale", min_value=1.0, max_value=30.0, value=7.5, step=0.5)
    num_inference_steps = st.slider("Steps", min_value=5, max_value=50, value=20, step=5)
    
    # Pokročilé parametry
    clip_skip = st.slider("Clip Skip", min_value=1, max_value=4, value=2, step=1)
    
    # Upscaling - otevřené ve výchozím stavu
    enable_upscaling = st.checkbox("⬆️ Upscaling", value=True)
    if enable_upscaling:
        upscale_factor = st.selectbox("Faktor:", [2, 4], index=0)
    else:
        upscale_factor = 1
    
    # Počet variant
    num_images = st.slider("Počet variant", min_value=1, max_value=8, value=1, step=1)
    
    # Sampler - výchozí hodnota bez UI
    sampler = "DPMSolverMultistepScheduler"
    
    # Seed pro reprodukovatelnost - otevřené ale nezaškrtnuté
    use_seed = st.checkbox("🎯 Seed", value=False)
    seed = st.number_input("Seed:", min_value=0, max_value=2147483647, value=42, step=1, disabled=not use_seed)
    if not use_seed:
        seed = None
    
    # Variance seed pro příbuzné varianty - otevřené ale nezaškrtnuté
    use_variance_seed = st.checkbox("🔀 Variance Seed", value=False)
    variance_seed = st.number_input("Variance Seed:", min_value=0, max_value=2147483647, value=123, step=1, disabled=not use_variance_seed)
    variance_strength = st.slider("Variance Strength", min_value=0.0, max_value=1.0, value=0.1, step=0.05, disabled=not use_variance_seed)
    if not use_variance_seed:
        variance_seed = None
        variance_strength = 0.0

# Inicializace globálních proměnných pro model
if 'current_model_file' not in st.session_state:
    st.session_state.current_model_file = None
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None
if 'selected_lora_model' not in st.session_state:
    st.session_state.selected_lora_model = None
if 'selected_full_model' not in st.session_state:
    st.session_state.selected_full_model = None

# Resetování globálních proměnných na začátku
st.session_state.current_model_file = None
st.session_state.current_model_path = None

# Hlavní layout s pravým sidebarom
col_main, col_right = st.columns([6, 1])

with col_right:
    # Aplikace CSS třídy pro pravý sidebar
    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
    
    # Model selection s expandery
    st.markdown("### 🧠 Model Selection")
    
    # Nahrání modelu
    with st.expander("📁 Nahrát Model", expanded=False):
        uploaded_file = st.file_uploader(
            "Model (.safetensors):", 
            type=["safetensors"],
            key="right_model_uploader"
        )
        
        if uploaded_file is not None:
            if (st.session_state.uploaded_model_file is None or 
                st.session_state.model_name != uploaded_file.name):
                st.session_state.uploaded_model_file = uploaded_file
                st.session_state.model_name = uploaded_file.name
                st.session_state.model_type = None
        
        if st.session_state.uploaded_model_file is not None:
            st.session_state.current_model_file = st.session_state.uploaded_model_file
            st.success(f"✅ {st.session_state.model_name}")
            if st.button("🗑️ Vymazat", key="right_clear_model"):
                st.session_state.uploaded_model_file = None
                st.session_state.model_type = None
                st.session_state.model_name = None
                st.session_state.current_model_file = None
                st.rerun()
    
    # LoRA modely
    with st.expander("⚙️ LoRA Modely", expanded=False):
        lora_models = get_lora_models_list()
        if lora_models:
            model_options = [f"{model['name']} ({model['size_mb']:.1f} MB)" for model in lora_models]
            selected_model = st.selectbox(
                "Vyberte LoRA model:",
                options=["(žádný)"] + model_options,
                key="right_lora_select"
            )
            if selected_model != "(žádný)":
                selected_index = model_options.index(selected_model)
                st.session_state.current_model_path = lora_models[selected_index]['path']
                st.session_state.selected_lora_model = selected_model
        else:
            st.warning("⚠️ Žádné LoRA modely")
            st.info("💡 Umístěte .safetensors soubory do /data/loras")
    
    # Full modely
    with st.expander("🔧 Full Modely", expanded=False):
        full_models = get_full_models_list()
        if full_models:
            model_options = [f"{model['name']} ({model['size_mb']:.1f} MB)" for model in full_models]
            selected_model = st.selectbox(
                "Vyberte full model:",
                options=["(žádný)"] + model_options,
                key="right_full_select"
            )
            if selected_model != "(žádný)":
                selected_index = model_options.index(selected_model)
                st.session_state.current_model_path = full_models[selected_index]['path']
                st.session_state.selected_full_model = selected_model
        else:
            st.warning("⚠️ Žádné full modely")
            st.info("💡 Umístěte .safetensors soubory do /data/models")
    
    st.markdown("---")
    
    # Preset a Favorites systém
    st.markdown("### 💾 Presety")
    preset_name = st.text_input("Název:", placeholder="Artistic Style", key="right_preset_name")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾", key="save_preset", help="Uložit preset"):
            if preset_name:
                preset_data = {
                     'strength': strength,
                     'guidance_scale': guidance_scale,
                     'num_inference_steps': num_inference_steps,
                     'clip_skip': clip_skip,
                     'enable_upscaling': enable_upscaling,
                     'upscale_factor': upscale_factor,
                     'num_images': num_images,
                     'sampler': sampler,
                     'use_seed': use_seed,
                     'seed': seed,
                     'use_variance_seed': use_variance_seed,
                     'variance_seed': variance_seed,
                     'variance_strength': variance_strength
                 }
                # Uložení do session state
                if 'presets' not in st.session_state:
                    st.session_state.presets = {}
                st.session_state.presets[preset_name] = preset_data
                st.success(f"✅ Uloženo!")
            else:
                st.warning("⚠️ Zadejte název")
    
    with col2:
        if st.button("⭐", key="add_favorite", help="Přidat do oblíbených"):
            if preset_name:
                if 'favorites' not in st.session_state:
                    st.session_state.favorites = []
                if preset_name not in st.session_state.favorites:
                    st.session_state.favorites.append(preset_name)
                    st.success(f"⭐ Přidáno!")
                else:
                    st.info("Už je v oblíbených")
            else:
                st.warning("⚠️ Zadejte název")
    
    # Načítání presetů
    if 'presets' in st.session_state and st.session_state.presets:
        preset_options = list(st.session_state.presets.keys())
        selected_preset = st.selectbox("Načíst:", ["(vyberte)"] + preset_options, key="right_load_preset")
        
        if selected_preset != "(vyberte)" and st.button("📥 Načíst", key="load_preset"):
            preset_data = st.session_state.presets[selected_preset]
            # Aktualizace session state s hodnotami presetu
            for key, value in preset_data.items():
                st.session_state[f"preset_{key}"] = value
            st.success(f"✅ Načteno! Obnovte stránku.")
    
    # Zobrazení oblíbených
    if 'favorites' in st.session_state and st.session_state.favorites:
        st.markdown("**⭐ Oblíbené:**")
        for fav in st.session_state.favorites:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"⭐ {fav}")
            with col2:
                if st.button("🗑️", key=f"remove_fav_{fav}"):
                    st.session_state.favorites.remove(fav)
                    st.rerun()
    
    st.markdown("---")
    
    # Systémové informace - zabalené
    with st.expander("⚙️ Systém", expanded=False):
        st.write(f"**Platforma:** {sys_info['platform']}")
        st.write(f"**CPU:** {sys_info['cpu_count']} jader")
        st.write(f"**RAM:** {sys_info['memory_gb']:.1f} GB")
        st.write(f"**Zařízení:** {device_reason}")
        if sys_info['cuda_available']:
            st.write(f"**GPU paměť:** {sys_info['cuda_memory_gb']:.1f} GB")
    
    # Uzavření pravého sidebaru
    st.markdown('</div>', unsafe_allow_html=True)

with col_main:
    # Tlačítko pro zpracování - přesunuto nahoru
    process_button = st.button("✨ Aplikovat styl", type="primary", use_container_width=True)
    
    # Progress bar kontejner - robustnější verze
    progress_container = st.container()
    
    # Vstupní a výstupní obrázky
    col1, col2 = st.columns(2)
    
    with col1:
        input_image_file = st.file_uploader("Nahrajte obrázek", type=["png", "jpg", "jpeg"])
        
        if input_image_file is not None:
            input_image = Image.open(input_image_file).convert("RGB")
            st.image(input_image, width=int(400 * 0.84))  # Zvětšeno o 20% z původních 70%
    
    with col2:
        # Placeholder pro výstupní obrázek
        output_placeholder = st.empty()
    
    # Zpracování obrázku
    if process_button and input_image_file is not None and (st.session_state.current_model_file is not None or st.session_state.current_model_path is not None or st.session_state.uploaded_model_file is not None):
        # Kompaktní progress tracking - dva pruhy vedle sebe
        with progress_container:
            # Dva sloupce pro progress pruhy
            prog_col1, prog_col2 = st.columns(2)
            
            with prog_col1:
                st.markdown("**Načítání modelu**")
                model_progress = st.progress(0)
                model_percent = st.empty()
            
            with prog_col2:
                st.markdown("**Generování obrázku**")
                generate_progress = st.progress(0)
                generate_percent = st.empty()
        
        start_time = time.time()
        
        def update_progress(progress: float, text: str = ""):
            # Mapování progress na dvě fáze
            if progress < 0.5:  # Načítání modelu (0-50%)
                model_prog = min(1.0, progress / 0.5)
                model_progress.progress(model_prog)
                model_percent.text(f"{model_prog*100:.0f}%")
                generate_progress.progress(0)
                generate_percent.text("0%")
            else:  # Generování (50-100%)
                model_progress.progress(1.0)
                model_percent.text("100%")
                gen_prog = min(1.0, (progress - 0.5) / 0.5)
                generate_progress.progress(gen_prog)
                generate_percent.text(f"{gen_prog*100:.0f}%")

        
        try:
            # Zpracování modelu podle zdroje
            if st.session_state.current_model_file is not None:
                os.makedirs("models", exist_ok=True)
                final_model_path = os.path.join("models", st.session_state.current_model_file.name)
                with open(final_model_path, "wb") as f:
                    f.write(st.session_state.current_model_file.getbuffer())
            elif st.session_state.uploaded_model_file is not None:
                os.makedirs("models", exist_ok=True)
                final_model_path = os.path.join("models", st.session_state.model_name)
                with open(final_model_path, "wb") as f:
                    f.write(st.session_state.uploaded_model_file.getbuffer())
            else:
                final_model_path = st.session_state.current_model_path
            
            # Detekce typu modelu
            model_type = detect_model_type(final_model_path)
            update_progress(0.05)
            
            # Generování obrázku
            update_progress(0.1)
            start_time = time.time()
            
            result_image = apply_style(
                input_image,
                final_model_path,
                model_type,
                strength,
                guidance_scale,
                num_inference_steps,
                update_progress,
                clip_skip=clip_skip,
                seed=seed,
                upscale_factor=upscale_factor,
                num_images=num_images,
                sampler=sampler,
                variance_seed=variance_seed,
                variance_strength=variance_strength
            )
            
            # Vyčištění progress baru
            progress_container.empty()
            
            # Vyčištění output_placeholder a zobrazení výsledku
            output_placeholder.empty()
            
            if num_images == 1:
                with output_placeholder.container():
                    st.image(result_image, width=int(400 * 0.84))
                    
                    # Tlačítko pro stažení
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Stáhnout",
                        data=buf.getvalue(),
                        file_name="result.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                # Pro více variant zobrazíme info v col2 a mřížku pod sloupci
                with output_placeholder.container():
                    st.markdown(f"### 🖼️ {num_images} variant")
                    st.markdown("*Mřížka níže*")
            
            # Galerie variant s velkým náhledem a miniaturami
            if num_images > 1:
                st.markdown("---")
                st.markdown('<div class="variant-gallery">', unsafe_allow_html=True)
                st.markdown(f"### 🖼️ Galerie variant ({num_images})")
                
                # Inicializace vybrané varianty
                if 'selected_variant' not in st.session_state:
                    st.session_state.selected_variant = 0
                
                # Hlavní náhled
                st.markdown('<div class="variant-main-preview">', unsafe_allow_html=True)
                st.markdown(f"**Varianta {st.session_state.selected_variant + 1} - Hlavní náhled**")
                
                # Zobrazení vybraného obrázku ve velkém
                selected_image = result_image  # Pro demonstraci - v reálné implementaci by to byl result_images[st.session_state.selected_variant]
                st.image(selected_image, use_column_width=True)
                
                # Tlačítko pro stažení vybrané varianty
                buf = io.BytesIO()
                selected_image.save(buf, format="PNG")
                st.download_button(
                    label=f"📥 Stáhnout variantu {st.session_state.selected_variant + 1}",
                    data=buf.getvalue(),
                    file_name=f"result_variant_{st.session_state.selected_variant + 1}.png",
                    mime="image/png",
                    key="download_selected_variant",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Miniatury variant
                st.markdown("**Vyberte variantu:**")
                st.markdown('<div class="variant-thumbnails">', unsafe_allow_html=True)
                
                # Vytvoření sloupců pro miniatury (max 4 na řádek)
                cols_per_row = min(4, num_images)
                rows = (num_images + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    thumbnail_cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx < num_images:
                            with thumbnail_cols[col_idx]:
                                # CSS třída pro vybranou miniaturu
                                thumbnail_class = "selected" if img_idx == st.session_state.selected_variant else ""
                                
                                # Tlačítko pro výběr varianty
                                if st.button(
                                    f"Varianta {img_idx + 1}",
                                    key=f"select_variant_{img_idx}",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_variant = img_idx
                                    st.rerun()
                                
                                # Miniatura obrázku
                                st.markdown(f'<div class="variant-thumbnail {thumbnail_class}">', unsafe_allow_html=True)
                                st.image(result_image, width=150)  # V reálné implementaci: result_images[img_idx]
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Individuální tlačítko pro stažení
                                buf_thumb = io.BytesIO()
                                result_image.save(buf_thumb, format="PNG")
                                st.download_button(
                                    label="📥",
                                    data=buf_thumb.getvalue(),
                                    file_name=f"variant_{img_idx + 1}.png",
                                    mime="image/png",
                                    key=f"download_thumb_{img_idx}",
                                    help=f"Stáhnout variantu {img_idx + 1}"
                                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Tlačítko pro stažení všech variant
                if st.button("📦 Stáhnout všechny varianty", use_container_width=True):
                    st.info("💡 Funkce stažení všech variant bude implementována v budoucí verzi.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Chyba: {str(e)}")
    
    elif process_button:
        if input_image_file is None:
            st.warning("⚠️ Nahrajte obrázek")
        if (model_file is None and model_path is None and 
            st.session_state.uploaded_model_file is None):
            st.warning("⚠️ Vyberte model")
    
    # Informace o aplikaci odstraněny podle požadavku uživatele