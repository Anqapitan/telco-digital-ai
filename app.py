import requests
import streamlit as st
import base64
from datetime import datetime
from io import BytesIO
import json

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
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="ID Telco Digital AI Assistant",
    page_icon=LOGO_DATA_URI,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
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
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }}
    div[data-baseweb="select"] {{ width: 100%; }}
    .stButton > button {{
        width: 100%;
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }}
    .answer-container {{
        margin-top: 1.5rem;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        overflow-wrap: anywhere;
        word-wrap: break-word;
    }}
/* ========== RIWAYAT PERCAKAPAN (PowerShell Style) ========== */
    .history-box {{
        background: #012456 !important;          /* Biru tua khas PowerShell */
        border: 2px solid #00bfff !important;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        max-height: 480px;
        overflow-y: auto;
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #ffff00 !important;               /* Kuning terang */
        box-shadow: 0 0 12px rgba(0, 191, 255, 0.35);
    }}

    .history-box strong {{
        color: #00ff9f !important;               /* Hijau neon untuk role */
    }}

    .history-box code {{
        background: #003366 !important;
        color: #7dffb0 !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.82rem;
    }}

    .history-box hr {{
        border: none;
        border-top: 1px dashed #00bfff;
        margin: 1.1rem 0;
    }}

    /* Scrollbar PowerShell style */
    .history-box::-webkit-scrollbar {{
        width: 10px;
    }}
    .history-box::-webkit-scrollbar-track {{
        background: #001a33;
    }}
    .history-box::-webkit-scrollbar-thumb {{
        background: #00bfff;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# API & MODEL CONFIG
# ============================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"

# Model list (Gemma-2 dihapus, diganti Qwen2.5-VL)
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",          # Text-only
    "meta-llama/Llama-3.1-8B-Instruct",    # Text-only
    "google/gemma-3-4b-it",                # Multimodal (image)
    "Qwen/Qwen2.5-VL-72B-Instruct",        # Multimodal (image + document)
]

# Definisi kemampuan model
MODEL_CAPABILITIES = {
    "Qwen/Qwen2.5-72B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "meta-llama/Llama-3.1-8B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "google/gemma-3-4b-it": {
        "type": "vision",
        "max_files": 3,
        "max_size_mb": 5,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "type": "multimodal",
        "max_files": 5,
        "max_size_mb": 10,
        "accept": ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]
    },
}

# ============================================================
# SESSION STATE
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""

if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = MODELS[0]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # list of dict {role, content, model, time}

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

st.markdown(
    f"""
    <div class="logo-header">
        <img src="{LOGO_DATA_URI}" alt="Logo">
        <div>
            <div class="app-title">ID Telco Digital AI Assistant</div>
            <div class="app-caption" style="margin-bottom:0">
                Gen-AI Literature Analytics by nap@iicf.or.id
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-caption">
    AI assistant untuk analisis Telco, ICT, Digital Transformation,
    AI regulation, Fiber Optic, 5G, Satellite, Data Center, Regulation,
    Project & Risk Management.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PILIH MODEL
# ============================================================

current_model = st.session_state.get("model_selected", MODELS[0])
try:
    model_index = MODELS.index(current_model)
except ValueError:
    model_index = 0

model = st.selectbox(
    "Pilih Model AI",
    MODELS,
    index=model_index,
    help="Model multimodal akan menampilkan opsi upload file."
)
st.session_state["model_selected"] = model

cap = MODEL_CAPABILITIES.get(model, {"type": "text"})

# Info model
if cap["type"] == "text":
    st.caption(f"Model aktif: **{model}** | Tipe: Text-only | Bahasa: Indonesia")
elif cap["type"] == "vision":
    st.caption(f"Model aktif: **{model}** | Tipe: Vision (Gambar) | Max {cap['max_files']} file, {cap['max_size_mb']} MB/file")
else:
    st.caption(f"Model aktif: **{model}** | Tipe: Multimodal (Gambar + Dokumen) | Max {cap['max_files']} file, {cap['max_size_mb']} MB/file")

# ============================================================
# UPLOAD FILE (hanya muncul jika model mendukung)
# ============================================================

uploaded_files = []
if cap["type"] in ["vision", "multimodal"]:
    accept_types = cap["accept"]
    help_text = (
        f"Anda bisa upload maksimal **{cap['max_files']} file**, "
        f"ukuran masing-masing ≤ **{cap['max_size_mb']} MB**.\n\n"
        f"Format yang didukung: {', '.join(accept_types)}"
    )
    uploaded_files = st.file_uploader(
        "📎 Upload file pendukung (opsional)",
        type=accept_types,
        accept_multiple_files=True,
        help=help_text
    )
    if uploaded_files and len(uploaded_files) > cap["max_files"]:
        st.warning(f"Maksimal {cap['max_files']} file. Hanya {cap['max_files']} file pertama yang akan digunakan.")
        uploaded_files = uploaded_files[:cap["max_files"]]

# ============================================================
# INPUT PROMPT
# ============================================================

prompt = st.text_area(
    "Masukkan Pertanyaan Anda",
    value=st.session_state["prompt_history"],
    height=160,
    placeholder=(
        "Contoh:\n"
        "Apa itu 5G?\n"
        "Bagaimana cara kerja Fiber Optic?\n"
        "Apa risiko pembangunan Data Center?\n"
        "Bagaimana model bisnis Submarine Cable?\n"
        "Jelaskan dalam konteks Indonesia."
    ),
    key="p"
)

# ============================================================
# TOMBOL TANYA AI
# ============================================================

if st.button("🚀 Tanya AI", type="primary", use_container_width=True):

    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu.")
        st.stop()

    system_prompt = """
Anda adalah Telco Digital AI, seorang AI assistant profesional
untuk bidang Telecommunications, ICT, Digital Transformation,
Business Analysis, Project Management, Risk Management,
Data Center, Fiber Optic, Submarine Cable, Satellite,
5G, IoT, Cloud, Cybersecurity, Regulation dan Digital Infrastructure.

ATURAN BAHASA:
1. Selalu jawab dalam Bahasa Indonesia.
2. Gunakan Bahasa Indonesia yang profesional, jelas dan natural.
3. Istilah teknis internasional boleh tetap menggunakan istilah bahasa Inggris apabila istilah tersebut merupakan terminology standar industri.
4. Jika pengguna bertanya menggunakan bahasa selain Bahasa Indonesia, tetap berikan jawaban utama dalam Bahasa Indonesia.
5. Jangan mengubah pertanyaan pengguna ke bahasa lain sebelum menjawab.
6. Gunakan struktur yang mudah dibaca seperti heading, bullet point, numbered list atau tabel apabila memang membantu.
7. Jangan mengatakan bahwa Anda tidak dapat menggunakan Bahasa Indonesia hanya karena model dasar Anda menggunakan bahasa lain.

ATURAN KUALITAS:
- Bedakan antara fakta, asumsi, analisis dan rekomendasi.
- Jika informasi bersifat tidak pasti, jelaskan tingkat ketidakpastiannya.
- Jangan mengarang sumber, data, regulasi atau angka.
- Untuk persoalan teknis, jelaskan konsep secara sistematis.
- Untuk persoalan bisnis/proyek, pertimbangkan Business, Technology, Project, Risk, Governance dan O&M apabila relevan.
"""

    # Siapkan messages
    messages = [
        {"role": "system", "content": system_prompt.strip()},
    ]

    # Handle multimodal content
    user_content = [{"type": "text", "text": prompt.strip()}]

    if uploaded_files and cap["type"] in ["vision", "multimodal"]:
        for f in uploaded_files:
            file_bytes = f.read()
            mime = f.type or "application/octet-stream"
            b64 = base64.b64encode(file_bytes).decode("utf-8")

            if f.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            else:
                # Untuk PDF/TXT/MD — kirim sebagai text jika memungkinkan
                try:
                    text_content = file_bytes.decode("utf-8", errors="ignore")
                    user_content.append({
                        "type": "text",
                        "text": f"\n\n[Isi file: {f.name}]\n{text_content[:8000]}"
                    })
                except Exception:
                    st.warning(f"File {f.name} tidak dapat dibaca sebagai teks.")

    messages.append({"role": "user", "content": user_content if len(user_content) > 1 else prompt.strip()})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.7
    }

    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error("❌ HF_TOKEN belum ditemukan di Streamlit Secrets.")
        st.stop()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        with st.spinner("🤖 AI sedang memproses pertanyaan..."):
            response = requests.post(API_URL, json=payload, headers=headers, timeout=180)

        response.raise_for_status()
        result_data = response.json()

        if "choices" not in result_data or not result_data["choices"]:
            st.error("❌ AI tidak memberikan jawaban.")
            st.stop()

        answer = result_data["choices"][0].get("message", {}).get("content", "")

        if not answer:
            st.error("❌ Content jawaban AI kosong.")
            st.stop()

        # Simpan ke history
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["chat_history"].append({
            "role": "user",
            "content": prompt.strip(),
            "model": model,
            "time": now
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "model": model,
            "time": now
        })

        st.markdown("### ✅ Jawaban AI")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["prompt_history"] = prompt
        st.query_params["prompt"] = prompt
        st.query_params["model"] = model

        st.caption(f"Model: `{model}` | Response language: Bahasa Indonesia")

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout. Model membutuhkan waktu lebih lama.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP/API Error: {e}")
        try:
            st.code(str(response.json()), language="json")
        except Exception:
            st.code(response.text)
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

# ============================================================
# CHAT HISTORY + DOWNLOAD / COPY
# ============================================================

if st.session_state["chat_history"]:
    st.markdown("---")
    st.subheader("📜 Riwayat Percakapan (Session ini)")

    # --- Buat versi HTML (untuk tampilan PowerShell) ---
    history_html = ""
    # --- Buat versi Markdown (untuk download) ---
    history_md = "# Riwayat Chat - ID Telco Digital AI\n\n"
    # --- Buat versi Plain Text ---
    history_plain = ""
    # --- Buat versi WhatsApp ---
    history_wa = ""

    for item in st.session_state["chat_history"]:
        role_label = "👤 ANDA" if item["role"] == "user" else "🤖 AI"
        role_color = "#00ff9f" if item["role"] == "user" else "#ffcc00"
        
        # HTML untuk tampilan
        history_html += f"""
        <div style="margin-bottom: 1.1rem;">
            <strong style="color:{role_color}">{role_label}</strong> 
            <span style="color:#7ec8ff; font-size:0.82rem;">({item['time']})</span> 
            <code>{item['model']}</code>
            <br><br>
            <div style="color:#ffff00; white-space: pre-wrap;">{item['content']}</div>
        </div>
        <hr style="border: none; border-top: 1px dashed #00bfff; margin: 1.1rem 0;">
        """

        # Markdown untuk download
        history_md += f"**{role_label}** ({item['time']}) — `{item['model']}`\n\n{item['content']}\n\n---\n\n"

        # Plain Text
        role_plain = "Anda" if item["role"] == "user" else "AI"
        history_plain += f"[{item['time']}] {role_plain} ({item['model']}):\n{item['content']}\n\n"

        # WhatsApp format
        history_wa += f"*{role_plain}* ({item['time']})\n{item['content']}\n\n"

    # Tampilkan dengan tema PowerShell
    st.markdown(f'<div class="history-box">{history_html}</div>', unsafe_allow_html=True)

    # Tombol aksi
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "⬇️ Download Markdown",
            data=history_md,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        st.download_button(
            "⬇️ Download Plain Text",
            data=history_plain,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col3:
        st.download_button(
            "⬇️ Download WhatsApp",
            data=history_wa,
            file_name=f"chat_wa_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col4:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
