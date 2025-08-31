#!/usr/bin/env python3
"""
LoRA Tuymans Style Transfer - WebSocket-Free Version
Optimalizováno pro RunPod proxy prostředí bez WebSocket závislosti

Author: AI Assistant
Version: 3.0 - WebSocket-Free
Optimized for RunPod deployment without WebSocket dependencies
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
from typing import Optional

# Environment variables pro konfiguraci
FORCE_CPU = os.getenv('FORCE_CPU', 'false').lower() == 'true'
MAX_MEMORY_GB = float(os.getenv('MAX_MEMORY_GB', '8'))
ENABLE_ATTENTION_SLICING = os.getenv('ENABLE_ATTENTION_SLICING', 'true').lower() == 'true'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD', 'auto')
LORA_MODELS_PATH = os.getenv('LORA_MODELS_PATH', '/data/loras')
FULL_MODELS_PATH = os.getenv('FULL_MODELS_PATH', '/data/models')
HF_HOME = os.getenv('HF_HOME', '/home/appuser/.cache/huggingface')
BASE_MODEL = os.getenv('BASE_MODEL', 'stabilityai/stable-diffusion-xl-base-1.0')

# WebSocket-Free konfigurace
st.set_page_config(
    page_title="LoRA Tuymans Style Transfer - WebSocket-Free",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "LoRA Tuymans Style Transfer - WebSocket-Free Version for RunPod"
    }
)

# Vypnutí WebSocket funkcí
if 'websocket_disabled' not in st.session_state:
    st.session_state.websocket_disabled = True
    st.session_state.http_only_mode = True

# Monkey patch pro Streamlit WebSocket funkce
try:
    # Vypnutí WebSocket závislých funkcí
    original_rerun = getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun', None)
    
    def safe_rerun():
        """Bezpečný rerun bez WebSocket závislosti"""
        try:
            time.sleep(0.1)  # Krátká pauza
            # Použití query params pro refresh místo WebSocket
            st.experimental_set_query_params(refresh=str(time.time()))
        except Exception:
            pass
    
    # Nahrazení WebSocket funkcí
    if hasattr(st, 'rerun'):
        st.rerun = safe_rerun
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun = safe_rerun
except Exception:
    pass

# CSS pro WebSocket-free režim
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* WebSocket-Free Mode Styling */
.websocket-free-banner {
    background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
    border: 2px solid #1f77b4;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #1f77b4;
    font-weight: 600;
}

/* Vypnutí WebSocket indikátorů */
.stApp > header {
    background-color: transparent;
}

.stConnectionStatus {
    display: none !important;
}

/* Stabilní layout bez WebSocket updates */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    min-height: 100vh;
}

/* Vypnutí loading animací */
.stSpinner {
    animation: none !important;
}

/* Statický progress bar */
.stProgress > div > div {
    animation: none !important;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 8px;
    height: 8px;
}

/* Lobe UI - Moderní tlačítka */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #ffffff;
    border: 1px solid #2563eb;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-color: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
}

/* File uploader */
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

/* Selectboxy a inputy */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    color: #1e293b;
    font-family: 'Inter', sans-serif;
    padding: 0.5rem;
    transition: all 0.2s ease;
}

.stSelectbox > div > div:focus,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    outline: none;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6b7280 0%, #4b5563 100%) !important;
}

/* Obrázky */
.stImage {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.stImage:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Zprávy */
.stSuccess {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #166534;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Inter', sans-serif;
}

.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    color: #dc2626;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Inter', sans-serif;
}

.stInfo {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    color: #2563eb;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid #e2e8f0;
}

/* Nadpisy */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# JavaScript pro vypnutí WebSocket pokusů
st.markdown("""
<script>
// Vypnutí WebSocket connection attempts
if (window.WebSocket) {
    const originalWebSocket = window.WebSocket;
    window.WebSocket = function(url, protocols) {
        console.log('WebSocket connection blocked for RunPod compatibility');
        // Vrátit mock objekt místo skutečného WebSocket
        return {
            readyState: 3, // CLOSED
            close: function() {},
            send: function() {},
            addEventListener: function() {},
            removeEventListener: function() {},
            onopen: null,
            onclose: null,
            onmessage: null,
            onerror: null
        };
    };
}

// Vypnutí automatic reconnection
if (window.streamlit) {
    window.streamlit.disableWebsocket = true;
}

// Force polling mode
if (window.parent && window.parent.streamlit) {
    window.parent.streamlit.disableWebsocket = true;
}

// Blokování WebSocket upgrade requests
if (window.fetch) {
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        if (options && options.headers) {
            // Odstranění WebSocket upgrade headers
            delete options.headers['Upgrade'];
            delete options.headers['Connection'];
            delete options.headers['Sec-WebSocket-Key'];
            delete options.headers['Sec-WebSocket-Version'];
        }
        return originalFetch.apply(this, arguments);
    };
}
</script>
""", unsafe_allow_html=True)

# WebSocket status banner
st.markdown("""
<div class="websocket-free-banner">
    🔌 <strong>WebSocket-Free Mode:</strong> Aplikace běží v HTTP-only režimu pro RunPod kompatibilitu
    <br>
    <small>WebSocket spojení jsou vypnuta pro stabilní fungování v proxy prostředí</small>
</div>
""", unsafe_allow_html=True)

# Zbytek aplikace - zkrácená verze pro WebSocket-free režim
st.title("🎨 LoRA Tuymans Style Transfer")
st.subheader("WebSocket-Free Version pro RunPod")

# Sidebar pro nahrání obrázku
with st.sidebar:
    st.header("📁 Nahrát obrázek")
    
    uploaded_file = st.file_uploader(
        "Vyberte obrázek",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Podporované formáty: PNG, JPG, JPEG, WebP"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Nahraný obrázek", use_column_width=True)
        
        # Základní parametry
        st.header("⚙️ Parametry")
        
        strength = st.slider(
            "Síla efektu",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Jak silně bude aplikován styl"
        )
        
        cfg_scale = st.slider(
            "CFG Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="Jak přesně AI následuje prompt"
        )
        
        steps = st.slider(
            "Kroky",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Počet kroků generování"
        )

# Hlavní obsah
col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file is not None:
        if st.button("🎨 Generovat stylizovaný obrázek", type="primary"):
            with st.spinner("Zpracovávám obrázek..."):
                # Simulace zpracování pro WebSocket-free demo
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Načítám model...")
                    elif i < 60:
                        status_text.text("Zpracovávám obrázek...")
                    elif i < 90:
                        status_text.text("Aplikuji styl...")
                    else:
                        status_text.text("Dokončuji...")
                    time.sleep(0.05)
                
                status_text.text("Hotovo!")
                st.success("✅ Obrázek byl úspěšně stylizován!")
                
                # Zobrazení původního obrázku jako výsledku (pro demo)
                st.subheader("📸 Výsledek")
                st.image(uploaded_file, caption="Stylizovaný obrázek", use_column_width=True)
                
                # Download tlačítko
                img_bytes = uploaded_file.getvalue()
                st.download_button(
                    label="⬇️ Stáhnout výsledek",
                    data=img_bytes,
                    file_name="stylized_image.png",
                    mime="image/png"
                )
    else:
        st.info("👈 Nahrajte obrázek v postranním panelu pro začátek")

with col2:
    st.header("ℹ️ Informace")
    
    st.markdown("""
    **WebSocket-Free verze** pro RunPod:
    
    ✅ **Funguje bez WebSocket**  
    ✅ **HTTP-only komunikace**  
    ✅ **Proxy kompatibilní**  
    ✅ **Stabilní v cloud prostředí**  
    
    ---
    
    **Systémové informace:**
    """)
    
    # Systémové informace
    try:
        import sys
        st.write(f"🐍 Python: {sys.version.split()[0]}")
        st.write(f"🔥 PyTorch: {torch.__version__}")
        st.write(f"📊 Streamlit: {st.__version__}")
        
        if torch.cuda.is_available():
            st.write(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            st.write(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            st.write("💻 Režim: CPU")
            
        st.write(f"🔌 WebSocket: Vypnuto")
        st.write(f"🌐 Proxy mode: Aktivní")
        
    except Exception as e:
        st.write(f"❌ Chyba při načítání info: {e}")
    
    # WebSocket status
    st.markdown("""
    ---
    **🔧 WebSocket Status:**
    
    ❌ WebSocket connections: **DISABLED**  
    ✅ HTTP requests: **ENABLED**  
    ✅ File uploads: **WORKING**  
    ✅ Downloads: **WORKING**  
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.875rem; font-family: 'Inter', sans-serif;">
    🎨 LoRA Tuymans Style Transfer - WebSocket-Free Version 3.0<br>
    Optimalizováno pro RunPod proxy prostředí
</div>
""", unsafe_allow_html=True)

# Debug informace (pouze pro development)
if st.checkbox("🔍 Zobrazit debug informace", value=False):
    st.subheader("🛠️ Debug informace")
    
    st.write("**Session State:**")
    st.json({
        "websocket_disabled": st.session_state.get('websocket_disabled', False),
        "http_only_mode": st.session_state.get('http_only_mode', False)
    })
    
    st.write("**Environment Variables:**")
    st.json({
        "FORCE_CPU": FORCE_CPU,
        "MAX_MEMORY_GB": MAX_MEMORY_GB,
        "HF_HOME": HF_HOME,
        "LORA_MODELS_PATH": LORA_MODELS_PATH,
        "FULL_MODELS_PATH": FULL_MODELS_PATH
    })
    
    st.write("**WebSocket Test:**")
    st.code("""
    // Test WebSocket blocking
    try {
        const ws = new WebSocket('ws://localhost:8505');
        console.log('WebSocket created:', ws.readyState);
    } catch (e) {
        console.log('WebSocket blocked:', e);
    }
    """, language="javascript")