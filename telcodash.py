import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Set tampilan halaman
st.set_page_config(page_title="Telco Digital AI Dashboard", layout="wide")

# Path ke folder 'se' di dalam repository
SE_DIR = Path(__file__).parent / "se"

# Nama file HTML yang akan dibuka langsung
file_html = "Live_DC_ASPAC.html"

# Cek apakah file ada
if (SE_DIR / file_html).exists():
    html = (SE_DIR / file_html).read_text(encoding="utf-8")
    components.html(html, height=950, scrolling=True)
else:
    st.error(f"File {file_html} tidak ditemukan di folder 'se'.")
