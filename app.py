import requests
import streamlit as st
import base64
from datetime import datetime
from typing import List, Dict, Any
import json

# Optional: install dulu jika belum
# pip install duckduckgo-search

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# ============================================================
# LOGO SVG INLINE
# ============================================================

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" width="300" height="300">
  <rect width="300" height="300" fill="none"/>
  <polygon points="42,90 78,70 78,230 42,210" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
  <polygon points="112,50 170,26 186,50 186,176" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
  <polygon points="112,96 186,222 148,272 112,252" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
  <polygon points="222,68 258,88 258,230 222,210" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
</svg>
"""
LOGO_BASE64 = base64.b64encode(LOGO_SVG.encode("utf-8")).decode("utf-8")
LOGO_DATA_URI = f"data:image/svg+xml;base64,{LOGO_BASE64}"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ID Telco Digital AI Assistant",
    page_icon=LOGO_DATA_URI,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS (termasuk PowerShell style history)
# ============================================================

st.markdown(f"""
<style>
.main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: 5%;
    padding-right: 5%;
    max-width: 1400px;
}}
.logo-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0.8rem;
}}
.logo-header img {{
    height: 58px;
    width: auto;
    filter: drop-shadow(0 3px 8px rgba(0,0,0,0.25));
}}
.app-title {{
    font-size: 2.3rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}}
.app-caption {{
    font-size: 0.95rem;
    opacity: 0.75;
    margin-bottom: 1.2rem;
}}
textarea {{
    min-height: 150px !important;
    resize: vertical !important;
    white-space: pre-wrap !important;
}}
.stButton > button {{
    width: 100%;
    min-height: 48px;
    font-weight: 600;
    border-radius: 8px;
}}
.answer-container {{
    margin-top: 1.5rem;
    padding: 1.25rem;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}}
.history-box {{
    background: #012456 !important;
    border: 2px solid #00bfff !important;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    max-height: 480px;
    overflow-y: auto;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #ffff00 !important;
    box-shadow: 0 0 12px rgba(0, 191, 255, 0.35);
}}
.history-box strong {{ color: #00ff9f !important; }}
.history-box code {{
    background: #003366 !important;
    color: #7dffb0 !important;
    padding: 2px 7px;
    border-radius: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DAFTAR MODEL + KETERANGAN
# ============================================================

MODELS = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

# Keterangan singkat setiap model
MODEL_INFO = {
    "Qwen/Qwen2.5-VL-72B-Instruct": "Multimodal kuat (Gambar+Dokumen) • Terbaik untuk analisis visual & Mermaid",
    "Qwen/Qwen2.5-72B-Instruct": "Text-only • Sangat kuat reasoning & analisis panjang",
    "meta-llama/Llama-3.1-8B-Instruct": "Text-only • Ringan, cepat, bagus untuk percakapan umum",
    "google/gemma-3-4b-it": "Multimodal (Gambar) • Paling ringan & hemat free tier",
    "google/gemma-3-12b-it": "Multimodal (Gambar) • Seimbang antara kualitas & kecepatan",
    "google/gemma-3-27b-it": "Multimodal (Gambar) • Kualitas tertinggi di keluarga Gemma-3",
}

# Capability (untuk fitur upload)
MODEL_CAPABILITIES = {
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "type": "multimodal",
        "max_files": 5,
        "max_size_mb": 10,
        "accept": ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]
    },
    "Qwen/Qwen2.5-72B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "meta-llama/Llama-3.1-8B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "google/gemma-3-4b-it": {
        "type": "vision",
        "max_files": 3,
        "max_size_mb": 5,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
    "google/gemma-3-12b-it": {
        "type": "vision",
        "max_files": 4,
        "max_size_mb": 8,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
    "google/gemma-3-27b-it": {
        "type": "vision",
        "max_files": 5,
        "max_size_mb": 10,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
}

# ============================================================
# WEB SEARCH TOOL (GRATIS - DuckDuckGo)
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Web search gratis menggunakan DuckDuckGo (tanpa API key)"""
    if not DDG_AVAILABLE:
        return "Library duckduckgo-search belum terinstall. Jalankan: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "Tidak ditemukan hasil pencarian."

        output = f"Hasil pencarian untuk: **{query}**\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "Tanpa Judul")
            href = r.get("href", "")
            body = r.get("body", "")[:300]
            output += f"{i}. **{title}**\n{body}...\nSumber: {href}\n\n"
        return output
    except Exception as e:
        return f"Gagal melakukan pencarian: {str(e)}"

# ============================================================
# SESSION STATE
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""
if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = MODELS[0]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "enable_web_search" not in st.session_state:
    st.session_state["enable_web_search"] = True

# ============================================================
# DEEP LINKING
# ============================================================

q = st.query_params
if "prompt" in q and q["prompt"]:
    st.session_state["prompt_history"] = q["prompt"]
if "model" in q and q["model"] in MODELS:
    st.session_state["model_selected"] = q["model"]

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="logo-header">
    <img src="{LOGO_DATA_URI}" alt="Logo">
    <div>
        <div class="app-title">ID Telco Digital AI Assistant</div>
        <div class="app-caption" style="margin-bottom:0">
            Gen-AI Literature Analytics by nap@iicf.or.id
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-caption">
AI assistant untuk analisis Telco, ICT, Digital Transformation, Fiber Optic, 5G, 
Satellite, Data Center, Regulation, Project & Risk Management.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR / PENGATURAN
# ============================================================

with st.expander("⚙️ Pengaturan Web Search & Tool", expanded=False):
    st.session_state["enable_web_search"] = st.checkbox(
        "Aktifkan Web Search (DuckDuckGo - Gratis)",
        value=st.session_state["enable_web_search"]
    )
    st.caption("Menggunakan DuckDuckGo. Tidak memerlukan API key.")

# ============================================================
# PILIH MODEL
# ============================================================

current_model = st.session_state.get("model_selected", MODELS[0])
try:
    model_index = MODELS.index(current_model)
except:
    model_index = 0

model = st.selectbox(
    "Pilih Model AI",
    options=MODELS,
    index=model_index,
    format_func=lambda x: f"{x}  →  {MODEL_INFO.get(x, '')}",
    help="Model multimodal akan menampilkan opsi upload file."
)
st.session_state["model_selected"] = model
cap = MODEL_CAPABILITIES.get(model, {"type": "text"})

if cap["type"] == "text":
    st.caption(f"**{model}** | Tipe: Text-only")
elif cap["type"] == "vision":
    st.caption(f"**{model}** | Tipe: Vision (Gambar) | Max {cap['max_files']} file")
else:
    st.caption(f"**{model}** | Tipe: Multimodal | Max {cap['max_files']} file")

# ============================================================
# UPLOAD FILE
# ============================================================

uploaded_files = []
if cap["type"] in ["vision", "multimodal"]:
    uploaded_files = st.file_uploader(
        "📎 Upload file pendukung (opsional)",
        type=cap["accept"],
        accept_multiple_files=True,
        help=f"Maksimal {cap['max_files']} file"
    )
    if uploaded_files and len(uploaded_files) > cap["max_files"]:
        uploaded_files = uploaded_files[:cap["max_files"]]
        st.warning(f"Hanya {cap['max_files']} file pertama yang digunakan.")

# ============================================================
# INPUT PROMPT
# ============================================================

prompt = st.text_area(
    "Masukkan Pertanyaan Anda",
    value=st.session_state["prompt_history"],
    height=160,
    placeholder="Contoh:\nApa tren terbaru Data Center di Asia Pacific 2026?\nBuatkan diagram Mermaid arsitektur 5G Core.\nJelaskan regulasi AI di Indonesia.",
    key="p"
)

# ============================================================
# TOMBOL TANYA AI
# ============================================================

if st.button("🚀 Tanya AI", type="primary", use_container_width=True):

    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu.")
        st.stop()

    # ---------- System Prompt ----------
    system_prompt = """
Anda adalah Telco Digital AI, asisten profesional di bidang Telecommunications, ICT, 
Digital Transformation, Data Center, Fiber Optic, Submarine Cable, 5G, Satellite, 
Regulation, Project & Risk Management.

ATURAN:
1. Selalu jawab dalam Bahasa Indonesia yang profesional dan jelas.
2. Jika ada hasil web search, gunakan informasi tersebut dan sebutkan sumbernya.
3. Jika diminta membuat diagram, hasilkan kode Mermaid yang valid di dalam blok ```mermaid.
4. Bedakan fakta, analisis, dan rekomendasi.
"""

    # ---------- Web Search (jika diaktifkan) ----------
    search_context = ""
    if st.session_state["enable_web_search"]:
        with st.spinner("🔍 Sedang mencari informasi terbaru di web..."):
            search_context = web_search(prompt.strip(), max_results=5)

    # ---------- Siapkan messages ----------
    user_content = []
    
    # Tambahkan hasil search jika ada
    final_prompt = prompt.strip()
    if search_context:
        final_prompt = f"""Berikut hasil pencarian web terkait pertanyaan Anda:

{search_context}

---
Pertanyaan pengguna:
{prompt.strip()}

Jawab berdasarkan informasi di atas + pengetahuan Anda. Sebutkan sumber jika relevan."""

    user_content.append({"type": "text", "text": final_prompt})

    # Handle file upload (multimodal)
    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            mime = f.type or "application/octet-stream"
            b64 = base64.b64encode(file_bytes).decode()

            if any(f.name.lower().endswith(ext) for ext in [".png",".jpg",".jpeg",".webp",".gif"]):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            else:
                try:
                    text_content = file_bytes.decode("utf-8", errors="ignore")[:6000]
                    user_content.append({
                        "type": "text",
                        "text": f"\n\n[Isi file {f.name}]\n{text_content}"
                    })
                except:
                    pass

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content if len(user_content) > 1 else final_prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.7
    }

    # ---------- Panggil API ----------
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except:
        st.error("❌ HF_TOKEN belum diatur di Streamlit Secrets.")
        st.stop()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    try:
        with st.spinner("🤖 AI sedang memproses..."):
            response = requests.post(API_URL, json=payload, headers=headers, timeout=180)

        response.raise_for_status()
        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        # Simpan history
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["chat_history"].append({
            "role": "user", "content": prompt.strip(), "model": model, "time": now
        })
        st.session_state["chat_history"].append({
            "role": "assistant", "content": answer, "model": model, "time": now
        })

        st.markdown("### ✅ Jawaban AI")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        if search_context:
            with st.expander("🔍 Lihat hasil Web Search yang digunakan"):
                st.markdown(search_context)

        st.session_state["prompt_history"] = prompt
        st.query_params["prompt"] = prompt
        st.query_params["model"] = model

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")

# ============================================================
# RIWAYAT PERCAKAPAN
# ============================================================

if st.session_state["chat_history"]:
    st.markdown("---")
    st.subheader("📜 Riwayat Percakapan (Session ini)")

    history_html = ""
    history_md = "# Riwayat Chat - ID Telco Digital AI\n\n"
    history_plain = ""
    history_wa = ""

    for item in st.session_state["chat_history"]:
        role_label = "👤 ANDA" if item["role"] == "user" else "🤖 AI"
        role_color = "#00ff9f" if item["role"] == "user" else "#ffcc00"

        history_html += f"""
        <div style="margin-bottom:1.1rem;">
            <strong style="color:{role_color}">{role_label}</strong>
            <span style="color:#7ec8ff;font-size:0.82rem;">({item['time']})</span>
            <code>{item['model']}</code><br><br>
            <div style="color:#ffff00;white-space:pre-wrap;">{item['content']}</div>
        </div>
        <hr style="border:none;border-top:1px dashed #00bfff;margin:1.1rem 0;">
        """

        history_md += f"**{role_label}** ({item['time']}) — `{item['model']}`\n\n{item['content']}\n\n---\n\n"
        role_plain = "Anda" if item["role"] == "user" else "AI"
        history_plain += f"[{item['time']}] {role_plain} ({item['model']}):\n{item['content']}\n\n"
        history_wa += f"*{role_plain}* ({item['time']})\n{item['content']}\n\n"

    st.markdown(f'<div class="history-box">{history_html}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("⬇️ Markdown", history_md, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md", "text/markdown", use_container_width=True)
    with col2:
        st.download_button("⬇️ Plain Text", history_plain, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col3:
        st.download_button("⬇️ WhatsApp", history_wa, f"chat_wa_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col4:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
