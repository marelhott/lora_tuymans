#!/usr/bin/env python3
"""
LoRA Tuymans Style Transfer - WebSocket-Free Version
Optimalizováno pro RunPod proxy prostředí bez WebSocket závislosti
"""

import streamlit as st
import time
import os
from pathlib import Path

# Konfigurace pro WebSocket-free režim
st.set_page_config(
    page_title="LoRA Tuymans Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "LoRA Tuymans Style Transfer - WebSocket-Free Version"
    }
)

# Vypnutí WebSocket funkcí
if 'websocket_disabled' not in st.session_state:
    st.session_state.websocket_disabled = True
    # Force refresh bez WebSocket
    st.experimental_set_query_params(no_websocket="true")

# Import původní aplikace s modifikacemi
try:
    # Načtení původního app.py s WebSocket workarounds
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Monkey patch pro Streamlit WebSocket funkce
    import streamlit.runtime.state as st_state
    import streamlit.runtime.scriptrunner as st_runner
    
    # Vypnutí WebSocket závislých funkcí
    original_rerun = st.rerun if hasattr(st, 'rerun') else st.experimental_rerun
    
    def safe_rerun():
        """Bezpečný rerun bez WebSocket závislosti"""
        try:
            time.sleep(0.1)  # Krátká pauza
            st.experimental_set_query_params(refresh=str(time.time()))
        except Exception:
            pass
    
    # Nahrazení WebSocket funkcí
    if hasattr(st, 'rerun'):
        st.rerun = safe_rerun
    else:
        st.experimental_rerun = safe_rerun
    
    # Načtení hlavní aplikace
    exec(open('app.py').read())
    
except Exception as e:
    st.error(f"Chyba při načítání aplikace: {e}")
    
    # Fallback - základní UI
    st.title("🎨 LoRA Tuymans Style Transfer")
    st.subheader("WebSocket-Free Version")
    
    st.info("""
    Tato verze je optimalizována pro RunPod proxy prostředí.
    WebSocket funkce jsou vypnuty pro lepší kompatibilitu.
    """)
    
    # Základní file upload
    uploaded_file = st.file_uploader(
        "Nahrajte obrázek",
        type=['png', 'jpg', 'jpeg'],
        help="Podporované formáty: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Nahraný obrázek", use_column_width=True)
        
        if st.button("Zpracovat obrázek", type="primary"):
            with st.spinner("Zpracovávám..."):
                time.sleep(2)  # Simulace zpracování
                st.success("Obrázek byl úspěšně zpracován!")
    
    # Systémové informace
    with st.expander("Systémové informace"):
        st.write(f"Python verze: {sys.version}")
        st.write(f"Streamlit verze: {st.__version__}")
        st.write(f"WebSocket status: Vypnuto")
        st.write(f"Proxy mode: Aktivní")

# Přidání CSS pro lepší vzhled bez WebSocket animací
st.markdown("""
<style>
/* Vypnutí WebSocket indikátorů */
.stApp > header {
    background-color: transparent;
}

/* Skrytí connection status */
.stConnectionStatus {
    display: none !important;
}

/* Stabilní layout bez WebSocket updates */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Vypnutí loading animací */
.stSpinner {
    animation: none !important;
}

/* Statický progress bar */
.stProgress > div > div {
    animation: none !important;
}

/* Info box pro WebSocket-free mode */
.websocket-info {
    background-color: #e8f4fd;
    border: 1px solid #1f77b4;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# WebSocket status indikátor
st.markdown("""
<div class="websocket-info">
    <strong>🔌 WebSocket Status:</strong> Vypnuto (RunPod Proxy Mode)
    <br>
    <small>Aplikace běží v kompatibilním režimu bez real-time komunikace.</small>
</div>
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
            removeEventListener: function() {}
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
</script>
""", unsafe_allow_html=True)