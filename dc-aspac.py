import streamlit as st

# URL yang benar sesuai permintaan kamu
TARGET_URL = "https://anqapitan.github.io/telco-digital-ai/Live_DC_ASPAC.html#v=ID"

st.markdown(f"""
    <meta http-equiv="refresh" content="0; url={TARGET_URL}" />
    <div style="text-align: center; margin-top: 20px; font-family: Arial, sans-serif;">
        Redirecting to Dashboard...<br>
        <a href="{TARGET_URL}" target="_blank" style="color: #2f6fed; font-weight: bold;">Klik jika tidak redirect</a>
    </div>
    """, unsafe_allow_html=True)
