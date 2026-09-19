import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Telco Digital AI Dashboard", layout="wide")

SE_DIR = Path(__file__).parent / "se"

pilihan = st.sidebar.selectbox("Dashboard", [
    "Live_DC_ASPAC.html",
    "18092026-Live_FOSubsea_ASPAC.html",
])

html = (SE_DIR / pilihan).read_text(encoding="utf-8")
components.html(html, height=950, scrolling=True)
