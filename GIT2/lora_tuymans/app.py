#!/usr/bin/env python3
"""
LoRA Tuymans Style Transfer
Kompletní aplikace pro stylový přenos pomocí LoRA modelů
Vygenerováno na základě dokumentace z 30.8.2025
"""

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
from typing import Optional, List
import numpy as np
import random
import base64

# Environment variables pro konfiguraci
FORCE_CPU = os.getenv('FORCE_CPU', 'false').lower() == 'true'
MAX_MEMORY_GB = float(os.getenv('MAX_MEMORY_GB', '8'))
ENABLE_ATTENTION_SLICING = os.getenv('ENABLE_ATTENTION_SLICING', 'true').lower() == 'true'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD', 'auto')
LORA_MODELS_PATH = os.getenv('LORA_MODELS_PATH', '/data/loras')
FULL_MODELS_PATH = os.getenv('FULL_MODELS_PATH', '/data/models')
HF_HOME = os.getenv('HF_HOME', '/home/appuser/.cache/huggingface')
BASE_MODEL = os.getenv('BASE_MODEL', 'stabilityai/stable-diffusion-xl-base-1.0')

# Nastavení stránky
st.set_page_config(
    page_title="LoRA Tuymans Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state inicializace - VŽDY vyčistit při restartu
# Detekce nové session (restart aplikace)
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    # Vyčištění všech dat při novém startu
    st.session_state.current_model_path = None
    st.session_state.current_model_type = None
    st.session_state.pipeline = None
    st.session_state.cached_pipeline = None
    st.session_state.current_lora_path = None
    st.session_state.all_generated_images = []  # ČISTÝ START
    st.session_state.initialized = True
    
# Fallback pro starší session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_model_path = None
    st.session_state.current_model_type = None
    st.session_state.pipeline = None
    st.session_state.cached_pipeline = None
    st.session_state.current_lora_path = None
    st.session_state.all_generated_images = []

# Lobe UI inspirovaný design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

/* Lobe UI - Hlavní kontejner bez jakýchkoliv omezení */
.main .block-container {
    padding: 1rem;
    max-width: none !important;
    width: 100% !important;
    height: auto !important;
    max-height: none !important;
    min-height: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
}

/* Lobe UI - Globální styling bez omezení */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    color: #1e293b;
    font-family: 'Inter', sans-serif;
    height: auto !important;
    max-height: none !important;
    min-height: auto !important;
    overflow: visible !important;
    overflow-y: visible !important;
}

/* Odstranění všech omezení výšky */
.main, .main > div, .block-container, .element-container {
    height: auto !important;
    max-height: none !important;
    min-height: auto !important;
    overflow: visible !important;
    overflow-y: visible !important;
}

/* Zajištění neomezeného scrollování */
body, html {
    height: auto !important;
    max-height: none !important;
    overflow-y: auto !important;
    scroll-behavior: smooth;
}

/* Oprava Streamlit scrollování */
.main, .main > div, .stApp {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: auto !important;
    max-height: none !important;
}

/* Odstranění fixed positioning */
.stApp > header {
    position: relative !important;
}

/* Zajištění správného scrollování pro celou aplikaci */
#root, [data-testid="stAppViewContainer"] {
    height: auto !important;
    max-height: none !important;
    overflow-y: auto !important;
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

/* Pouze upload ikona - žádný text */
.stFileUploader {
    width: 60px;
    height: 60px;
}

.stFileUploader > div {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 0;
    text-align: center;
    transition: all 0.3s ease;
    width: 60px !important;
    height: 60px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.stFileUploader > div:hover {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
    transform: scale(1.05);
}

/* Skrytí všech textových elementů */
.stFileUploader > div > div,
.stFileUploader > div > div > div,
.stFileUploader > div > div > div > div,
.stFileUploader > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div,
.stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div > div {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Zobrazení pouze ikony */
.stFileUploader > div::before {
    content: "📁";
    font-size: 24px;
    display: block;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 999;
}

/* Dvoufázový progress bar */
.dual-progress {
    margin: 1rem 0;
}

.progress-phase {
    margin-bottom: 0.5rem;
}

.progress-label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #475569;
    margin-bottom: 0.25rem;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 8px;
    height: 8px;
}

/* Galerie obrázků - absolutně neomezený layout */
.image-gallery {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 30px;
    margin: 1rem 0;
    padding-bottom: 5rem;
    width: 100% !important;
    max-width: none !important;
    height: auto !important;
    max-height: none !important;
    min-height: auto !important;
    overflow: visible !important;
    overflow-y: visible !important;
    position: relative;
    z-index: 1;
}

.image-container {
    position: relative;
    display: inline-block;
    margin: 0 auto;
}

.gallery-image {
    max-width: 1000px;
    max-height: 800px;
    width: auto;
    height: auto;
    object-fit: contain;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: all 0.3s ease;
    display: block;
    margin: 0 auto;
}

.gallery-image:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.15);
}

.download-icon {
    position: absolute;
    bottom: 10px;
    right: 10px;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
    z-index: 10;
}

.download-icon:hover {
    background: rgba(0, 0, 0, 0.9);
    transform: scale(1.1);
}

/* Modal pro zvětšení obrázku */
.image-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.9);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    cursor: pointer;
}

.modal-image {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 8px;
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
</style>

<script>
function openImageFullscreen(imageSrc) {
    // Otevření nového okna
    const win = window.open("", "_blank");
    win.document.write(`
        <html>
        <head>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background: black;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    overflow: hidden;
                }
                img {
                    max-width: 100vw;
                    max-height: 100vh;
                    object-fit: contain;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <img src="${imageSrc}" ondblclick="window.close()">
        </body>
        </html>
    `);
}

function handleImageClick(imageSrc) {
    openImageFullscreen(imageSrc);
}

function downloadImage(imageSrc, filename) {
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = filename || 'generated_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
</script>
""", unsafe_allow_html=True)

# Vytvoření adresářů
try:
    os.makedirs(LORA_MODELS_PATH, exist_ok=True)
    os.makedirs(FULL_MODELS_PATH, exist_ok=True)
except OSError:
    LORA_MODELS_PATH = './lora_models'
    FULL_MODELS_PATH = './models'
    os.makedirs(LORA_MODELS_PATH, exist_ok=True)
    os.makedirs(FULL_MODELS_PATH, exist_ok=True)

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
    """Určí optimální zařízení pro inference"""
    if FORCE_CPU:
        return "cpu", "Vynuceno CPU"
    
    if not torch.cuda.is_available():
        return "cpu", "CUDA není dostupná"
    
    try:
        test_tensor = torch.randn(10, device='cuda')
        _ = test_tensor + 1
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4:
            return "cpu", f"Nedostatek GPU paměti ({gpu_memory:.1f} GB < 4 GB)"
        
        return "cuda", f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)"
    except Exception as e:
        return "cpu", f"CUDA chyba - fallback na CPU: {str(e)[:30]}..."

def get_lora_models_list():
    """Získá seznam dostupných LoRA modelů"""
    lora_files = []
    
    if os.path.exists(LORA_MODELS_PATH):
        for root, dirs, files in os.walk(LORA_MODELS_PATH):
            for file in files:
                if file.endswith('.safetensors'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    rel_path = os.path.relpath(file_path, LORA_MODELS_PATH)
                    
                    lora_files.append({
                        'name': rel_path,
                        'path': file_path,
                        'size_mb': file_size
                    })
    
    return sorted(lora_files, key=lambda x: x['name'])

def get_full_models_list():
    """Získá seznam dostupných full modelů"""
    models = []
    
    if os.path.exists(FULL_MODELS_PATH):
        for root, dirs, files in os.walk(FULL_MODELS_PATH):
            for file in files:
                if file.endswith('.safetensors'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    rel_path = os.path.relpath(file_path, FULL_MODELS_PATH)
                    
                    models.append({
                        'name': rel_path,
                        'path': file_path,
                        'size_mb': file_size
                    })
    
    return sorted(models, key=lambda x: x['name'])

def detect_model_type(file_path):
    """Detekuje zda je soubor LoRA model nebo full safetensors model"""
    try:
        state_dict = load_file(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
        
        lora_keys = ['lora_unet', 'lora_te', 'alpha', 'rank']
        has_lora_keys = any(any(lora_key in key for lora_key in lora_keys) for key in state_dict.keys())
        
        full_model_keys = ['model.diffusion_model', 'first_stage_model', 'cond_stage_model']
        has_full_keys = any(any(full_key in key for full_key in full_model_keys) for key in state_dict.keys())
        
        if has_lora_keys or file_size < 1.0:
            return "lora"
        elif has_full_keys or file_size > 2.0:
            return "full_model"
        else:
            return "lora" if file_size < 1.0 else "full_model"
            
    except Exception as e:
        st.warning(f"Nelze detekovat typ modelu: {e}")
        return "unknown"

def load_base_model():
    """Načtení základního SDXL modelu - BEZ CACHE pro správné LoRA switching"""
    try:
        device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
        
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None
        )
        
        pipeline = pipeline.to(device)
        
        if ENABLE_ATTENTION_SLICING and device == "cuda":
            pipeline.enable_attention_slicing()
        
        if ENABLE_CPU_OFFLOAD == "true" or (ENABLE_CPU_OFFLOAD == "auto" and device == "cuda"):
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    except Exception as e:
        st.error(f"Chyba při načítání základního modelu: {e}")
        return None

def load_lora_model(pipeline, lora_path: str, current_lora_path: str = None):
    """Načtení LoRA modelu na fresh pipeline"""
    try:
        # Validace lora_path
        if lora_path is None:
            st.error("LoRA path je None - nebyl vybrán žádný model")
            return None
        
        if not os.path.exists(lora_path):
            st.error(f"LoRA soubor neexistuje: {lora_path}")
            return None
        
        # Vyčištění GPU paměti před načtením
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Načtení LoRA modelu na fresh pipeline
        st.write(f"🔧 Načítám LoRA: {os.path.basename(lora_path)}")
        pipeline.load_lora_weights(lora_path)
        
        # Ověření načtení
        st.success(f"✅ LoRA model načten: {os.path.basename(lora_path)}")
        return pipeline
    except Exception as e:
        st.error(f"Chyba při načítání LoRA modelu: {e}")
        return None

def load_full_model(model_path: str):
    """Načtení plného safetensors modelu"""
    try:
        # Validace model_path
        if model_path is None:
            st.error("Model path je None - nebyl vybrán žádný model")
            return None
        
        if not os.path.exists(model_path):
            st.error(f"Model soubor neexistuje: {model_path}")
            return None
        
        device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
        
        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        pipeline = pipeline.to(device)
        
        if ENABLE_ATTENTION_SLICING and device == "cuda":
            pipeline.enable_attention_slicing()
        
        if ENABLE_CPU_OFFLOAD == "true" or (ENABLE_CPU_OFFLOAD == "auto" and device == "cuda"):
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    except Exception as e:
        st.error(f"Chyba při načítání plného modelu: {e}")
        return None

def apply_style(
    input_image: Image.Image,
    model_path: str,
    model_type: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    progress_callback=None,
    clip_skip: int = 2,
    upscale_factor: int = 1,
    num_images: int = 1
) -> List[Image.Image]:
    """Aplikace stylu na vstupní obrázek podle API dokumentace"""
    
    device, device_reason = get_optimal_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Fáze 1: Načítání modelu
        if progress_callback:
            progress_callback(0.1, "Načítání modelu...", "model")
        
        # Načtení modelu podle typu - VŽDY FRESH pro správné LoRA switching
        if model_type == "lora":
            # VŽDY načíst fresh base model pro LoRA
            pipeline = load_base_model()
            if pipeline is None:
                return []
            if progress_callback:
                progress_callback(0.5, "Načítání LoRA...", "model")
            
            # Načtení LoRA modelu na fresh base
            pipeline = load_lora_model(pipeline, model_path, None)
            
            # Debug info
            st.info(f"🔄 Načten LoRA model: {os.path.basename(model_path)}")
        else:
            # Full model handling - také fresh
            pipeline = load_full_model(model_path)
            st.info(f"🔄 Načten full model: {os.path.basename(model_path)}")
            
        # Vyčištění cache pro zajištění fresh loading
        st.session_state.cached_pipeline = None
        
        # Vyčištění všech torch cache pro memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if pipeline is None:
            return []
        
        if progress_callback:
            progress_callback(1.0, "Model načten", "model")
        
        # Nastavení scheduleru pro konzistentní výsledky
        # Použití EulerDiscreteScheduler pro lepší determinismus
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        # Debug scheduler info
        st.write(f"🔧 Scheduler: {type(pipeline.scheduler).__name__}")
        st.write(f"📊 Scheduler config: steps={pipeline.scheduler.num_train_timesteps}")
        
        # Force deterministic behavior
        if hasattr(pipeline.scheduler, 'set_timesteps'):
            pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Zachování aspect ratio vstupního obrázku
        original_width, original_height = input_image.size
        aspect_ratio = original_width / original_height
        
        # Nastavení rozměrů s dodržením aspect ratio (max 1200px)
        if aspect_ratio > 1:  # landscape
            new_width = min(1200, original_width)
            new_height = int(new_width / aspect_ratio)
        else:  # portrait nebo square
            new_height = min(1200, original_height)
            new_width = int(new_height * aspect_ratio)
        
        # Zajištění že rozměry jsou dělitelné 8 (požadavek SDXL)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        input_image = input_image.resize((new_width, new_height))
        
        results = []
        
        # Fáze 2: Generování obrázků
        for i in range(num_images):
            if progress_callback:
                progress_callback(0.0, f"Generování obrázku {i+1} z {num_images}", "generation")
            
            # Generování s random seed pro každý obrázek
            import hashlib
            unique_string = f"{time.time()}_{model_path}_{i}_{random.random()}"
            current_seed = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
            
            # Reset všech random generátorů pro konzistenci
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)
            
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # Debug info
            st.write(f"🎲 Random seed: {current_seed}")
            st.write(f"🎨 Model: {os.path.basename(model_path)}")
            st.write(f"💪 Strength: {strength}, Steps: {num_inference_steps}")
            
            # Generování s progress updates
            if progress_callback:
                progress_callback(0.2, f"Zpracování obrázku {i+1} z {num_images} (20%)", "generation")
            
            with torch.no_grad():
                result = pipeline(
                    prompt="",
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    num_images_per_prompt=1
                )
            
            if progress_callback:
                progress_callback(0.8, f"Dokončování obrázku {i+1} z {num_images} (80%)", "generation")
            
            generated_image = result.images[0]
            
            # Upscaling pokud je povoleno
            if upscale_factor > 1:
                if progress_callback:
                    progress_callback(0.9, f"Upscaling obrázku {i+1} z {num_images}", "generation")
                original_size = generated_image.size
                new_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
                generated_image = generated_image.resize(new_size, Image.Resampling.LANCZOS)
            
            results.append(generated_image)
            
            if progress_callback:
                progress_callback(1.0, f"Hotovo: obrázek {i+1} z {num_images}", "generation")
        
        if progress_callback:
            progress_callback(1.0, "Dokončeno!", "generation")
        
        return results
        
    except Exception as e:
        st.error(f"Chyba při generování: {e}")
        return []
    
    finally:
        # Vyčištění paměti
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# Inicializace session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Systémové informace
sys_info = get_system_info()
device, device_reason = get_optimal_device()

# Zobrazení varování při CUDA chybě
if "chyba" in device_reason.lower():
    st.warning(f"⚠️ CUDA problém detekován, přepínám na CPU: {device_reason}")

# Layout aplikace s užšími sidebary
col_left, col_main, col_right = st.columns([0.8, 2.4, 0.8])

# Levý sidebar - parametry
with col_left:
    st.header("Parametry")
    
    # Parametry pro přenos stylu
    strength = st.slider("Strength", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    guidance_scale = st.slider("CFG Scale", min_value=1.0, max_value=30.0, value=7.5, step=0.5)
    num_inference_steps = st.slider("Steps", min_value=5, max_value=50, value=20, step=5)
    
    # Pokročilé parametry
    clip_skip = st.slider("Clip Skip", min_value=1, max_value=4, value=2, step=1)
    
    # Upscaling
    st.subheader("Upscaling")
    enable_upscaling = st.checkbox("Povolit upscaling")
    upscale_factor = st.selectbox("Faktor", [1, 2, 4], index=0) if enable_upscaling else 1
    
    # Batch generování
    st.subheader("Batch")
    num_images = st.slider("Počet variant", min_value=1, max_value=3, value=1)
    
    # Sampler odstraněn
    
    # Seed sekce odstraněna - používá se automatický random seed

# Pravý sidebar - modely a systém
with col_right:
    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
    
    # Kompaktní upload
    input_image_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    
    # Malá miniatura
    if input_image_file is not None:
        input_image = Image.open(input_image_file).convert("RGB")
        st.image(input_image, width=80)
    
    st.divider()
    
    st.header("Modely")
    
    # LoRA modely
    st.subheader("LoRA Modely")
    lora_models = get_lora_models_list()
    
    if lora_models:
        model_options = [f"{model['name']} ({model['size_mb']:.1f} MB)" for model in lora_models]
        selected_lora = st.selectbox(
            "Vyberte LoRA model:",
            options=["(žádný)"] + model_options,
            key="lora_select"
        )
        if selected_lora != "(žádný)":
            selected_index = model_options.index(selected_lora)
            new_model_path = lora_models[selected_index]['path']
            # Detekce změny modelu
            if st.session_state.current_model_path != new_model_path:
                st.session_state.current_model_path = new_model_path
                st.session_state.current_model_type = "lora"
                # Vyčištění cache při změně modelu
                st.session_state.cached_pipeline = None
                # Vyčištění GPU paměti
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                # Reset full model selectbox
                if 'full_select' in st.session_state:
                    st.session_state.full_select = "(žádný)"
        else:
            # Reset při výběru "(žádný)"
            if st.session_state.current_model_path is not None and st.session_state.current_model_type == "lora":
                st.session_state.current_model_path = None
                st.session_state.current_model_type = None
                st.session_state.cached_pipeline = None
                st.session_state.current_lora_path = None
                # Vyčištění paměti při resetu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    else:
        st.warning("Žádné LoRA modely")
        st.info("Umístěte .safetensors soubory do /data/loras")
    
    st.divider()
    
    # Full modely
    st.subheader("Full Modely")
    full_models = get_full_models_list()
    
    if full_models:
        model_options = [f"{model['name']} ({model['size_mb']:.1f} MB)" for model in full_models]
        selected_full = st.selectbox(
            "Vyberte full model:",
            options=["(žádný)"] + model_options,
            key="full_select"
        )
        if selected_full != "(žádný)":
            selected_index = model_options.index(selected_full)
            new_model_path = full_models[selected_index]['path']
            # Detekce změny modelu
            if st.session_state.current_model_path != new_model_path:
                st.session_state.current_model_path = new_model_path
                st.session_state.current_model_type = "full_model"
                # Vyčištění cache při změně modelu
                st.session_state.cached_pipeline = None
                st.session_state.current_lora_path = None
                # Vyčištění GPU paměti
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                # Reset LoRA selectbox
                if 'lora_select' in st.session_state:
                    st.session_state.lora_select = "(žádný)"
        else:
            # Reset při výběru "(žádný)"
            if st.session_state.current_model_path is not None and st.session_state.current_model_type == "full_model":
                st.session_state.current_model_path = None
                st.session_state.current_model_type = None
                st.session_state.cached_pipeline = None
                st.session_state.current_lora_path = None
                # Vyčištění paměti při resetu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    else:
        st.warning("Žádné full modely")
        st.info("Umístěte .safetensors soubory do /data/models")
    

    
    st.markdown('</div>', unsafe_allow_html=True)

# Hlavní oblast
with col_main:
    # Tlačítka pro zpracování a vymazání
    col1, col2 = st.columns([3, 1])
    with col1:
        process_button = st.button("Aplikovat styl", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️", help="Vymazat všechny obrázky", use_container_width=True):
            st.session_state.all_generated_images = []
            # Vyčištění paměti po smazání obrázků
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            st.rerun()
    
    # Progress bar kontejner - jednoduchý
    progress_container = st.empty()
    
    # Zobrazení galerie obrázků
    if st.session_state.all_generated_images:
        st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
        
        for i, img in enumerate(st.session_state.all_generated_images):
            # Konverze obrázku na base64 pro JavaScript
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            img_data_url = f"data:image/png;base64,{img_base64}"
            
            # Zobrazení obrázku s download ikonou
            st.markdown(
                f'''<div style="text-align: center; margin: 15px 0;">
                    <div class="image-container">
                        <img src="{img_data_url}" 
                             class="gallery-image" 
                             onclick="handleImageClick('{img_data_url}')" 
                             alt="Generated image {i+1}"
                             style="max-width: 1000px; max-height: 800px; width: auto; height: auto; display: block; margin: 0 auto; cursor: pointer;" />
                        <button class="download-icon" 
                                onclick="event.stopPropagation(); downloadImage('{img_data_url}', 'generated_image_{i+1}.png')" 
                                title="Stáhnout obrázek">
                            ⬇️
                        </button>
                    </div>
                </div>''',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Kontejner pro výstup
    output_placeholder = st.empty()
    
    # Zpracování obrázku
    if process_button and input_image_file is not None and hasattr(st.session_state, 'current_model_path'):
         # Jednoduchý progress bar
         progress_bar = progress_container.progress(0)
         status_text = st.empty()
         
         def update_progress(progress: float, text: str = "", phase: str = "generation"):
              status_text.text(text if text else f"Zpracování... {int(progress*100)}%")
              progress_bar.progress(progress)
         
         start_time = time.time()
         
         try:
             # Generování obrázku
             result_images = apply_style(
                  input_image,
                  st.session_state.current_model_path,
                  st.session_state.current_model_type,
                  strength,
                  guidance_scale,
                  num_inference_steps,
                  update_progress,
                  clip_skip=clip_skip,
                  upscale_factor=upscale_factor,
                  num_images=num_images
              )
             
             # Vyčištění progress baru
             progress_container.empty()
             status_text.empty()
             
             # Zobrazení výsledků - postupné přidávání obrázků
             if result_images:
                 # Inicializace session state pro obrázky pokud neexistuje
                 if 'all_generated_images' not in st.session_state:
                     st.session_state.all_generated_images = []
                 
                 # Přidání nových obrázků k existujícím
                 st.session_state.all_generated_images.extend(result_images)
                 
                 # Zobrazení všech obrázků - persistentní způsob
                 st.rerun()
                 
                 end_time = time.time()
                 st.success(f"Generování dokončeno za {end_time - start_time:.1f} sekund")
             
         except Exception as e:
             progress_container.empty()
             st.error(f"Chyba: {str(e)}")
    
    elif process_button:
         if input_image_file is None:
             st.warning("Nahrajte obrázek")
         if not hasattr(st.session_state, 'current_model_path'):
             st.warning("Vyberte model")

# Footer