import streamlit as st

# URL HTML GitHub Pages kamu (harus sudah aktif di GitHub Pages /se/)
# Contoh: https://anqapitan.github.io/telco-digital-ai/se/Live_DC_ASPAC.html
# Jika belum aktif, aktifkan GitHub Pages di Settings -> Pages -> Source: /se
GITHUB_PAGES_URL = "https://anqapitan.github.io/telco-digital-ai/se/Live_DC_ASPAC.html"

st.markdown(
    f"""
    <meta http-equiv="refresh" content="0; url={GITHUB_PAGES_URL}" />
    <style>
        /* Sembunyikan SEMUA elemen Streamlit agar terlihat seperti redirect murni */
        [data-testid="stHeader"], footer, #MainMenu, .block-container, [data-testid="stSidebar"], header {
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
    <div style="text-align:center; padding: 20px; color: #333;">
        Redirecting to Dashboard...<br>
        <small>Jika tidak redirect, <a href="{GITHUB_PAGES_URL}" style="color: #2f6fed;">klik di sini</a></small>
    </div>
    """,
    unsafe_allow_html=True
)
