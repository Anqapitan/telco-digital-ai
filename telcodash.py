import streamlit as st

# URL GitHub Pages kamu (Ganti dengan URL yang benar setelah aktifkan GitHub Pages)
GITHUB_PAGES_URL = "https://anqapitan.github.io/telco-digital-ai/se/Live_DC_ASPAC.html"

st.markdown(f"""
    <meta http-equiv="refresh" content="0; url={GITHUB_PAGES_URL}" />
    <style>
        /* Sembunyikan SEMUA elemen Streamlit */
        [data-testid="stHeader"], footer, #MainMenu, .block-container, [data-testid="stSidebar"], header, nav {
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            height: 100% !important;
            overflow: hidden !important;
        }
    </style>
    <div style="text-align:center; padding: 20px; color: #333; font-family: Arial, sans-serif;">
        Redirecting to Dashboard...<br>
        <small>Jika tidak redirect otomatis, <a href="{GITHUB_PAGES_URL}" style="color: #2f6fed; font-weight: bold;">klik di sini</a></small>
    </div>
    """, unsafe_allow_html=True)
