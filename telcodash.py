import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Telco Digital AI Dashboard", layout="wide")

# CSS untuk menyembunyikan Streamlit UI dan membuat iframe fullscreen
st.markdown("""
<style>
/* Sembunyikan semua UI Streamlit */
[data-testid="stHeader"], footer, #MainMenu, .block-container, [data-testid="stSidebar"] {
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Pastikan iframe memenuhi layar */
iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    z-index: 999999 !important;
}

/* Pastikan body memenuhi layar */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

SE_DIR = Path(__file__).parent / "se"
file_html = "Live_DC_ASPAC.html"

if (SE_DIR / file_html).exists():
    html = (SE_DIR / file_html).read_text(encoding="utf-8")
    # Gunakan height besar untuk menghindari scroll ganda
    components.html(html, height=10000, scrolling=False)
else:
    st.error(f"File {file_html} tidak ditemukan.")
