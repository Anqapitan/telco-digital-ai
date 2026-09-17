import requests
import streamlit as st
import base64
from datetime import datetime
from io import BytesIO

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
        margin-top: 1rem;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        overflow-wrap: anywhere;
        word-wrap: break-word;
    }}
    .chat-bubble-user {{
        background: #e8f4fd;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #1f8bd6;
    }}
    .chat-bubble-ai {{
        background: #f0fdf4;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #34b56a;
    }}
    @media only screen and (max-width: 768px) {{
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .app-title {{ font-size: 1.7rem; }}
        .logo-header img {{ height: 46px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# API & MODEL
# ============================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"

MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",   # Multimodal (Vision)
    "google/gemma-3-4b-it",           # Multimodal (Vision)
]

# Model yang mendukung vision/multimodal
MULTIMODAL_MODELS = {
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "google/gemma-3-4b-it",
}

# ============================================================
# SESSION STATE
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""

if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = MODELS[0]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # list of dict: role, content, model, time, has_image

# ============================================================
# DEEP LINKING
# ============================================================

q = st.query_params
if "prompt" in q and q["prompt"]:
    st.session_state["prompt_history"] = q["prompt"]
if "model" in q and q["model"] in MODELS:
    st.session_state["model_selected"] = q["model"]

# ============================================================
# HEADER + LOGO
# ============================================================

st.markdown(
    f"""
    <div class="logo-header">
        <img src="{LOGO_DATA_URI}" alt="Logo Narational">
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
    help="Model multimodal (Vision) akan menampilkan opsi upload gambar."
)
st.session_state["model_selected"] = model

is_multimodal = model in MULTIMODAL_MODELS

st.caption(
    f"Model aktif: **{model}** | "
    f"{'🖼️ Multimodal (Vision)' if is_multimodal else '📝 Text-only'} | "
    "Bahasa respons default: **Bahasa Indonesia**"
)

# ============================================================
# INPUT PROMPT + UPLOAD GAMBAR (otomatis jika multimodal)
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

uploaded_file = None
if is_multimodal:
    uploaded_file = st.file_uploader(
        "📷 Upload Gambar (opsional) — hanya untuk model Vision/Multimodal",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        help="Model Vision akan menganalisis gambar yang diunggah bersama pertanyaan Anda."
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview gambar yang akan dianalisis", use_container_width=True)

# ============================================================
# FUNGSI BANTUAN: Chat History
# ============================================================

def format_history_markdown(history):
    lines = ["# Riwayat Chat - ID Telco Digital AI\n"]
    for i, msg in enumerate(history, 1):
        role = "👤 User" if msg["role"] == "user" else "🤖 AI"
        time_str = msg.get("time", "")
        model_str = f" ({msg.get('model', '')})" if msg["role"] == "assistant" else ""
        img_note = " *[ada gambar]*" if msg.get("has_image") else ""
        lines.append(f"### {i}. {role}{model_str} — {time_str}{img_note}\n")
        lines.append(msg["content"] + "\n")
    return "\n".join(lines)

def format_history_whatsapp(history):
    lines = []
    for msg in history:
        time_str = msg.get("time", "")
        if msg["role"] == "user":
            prefix = f"[{time_str}] Anda:"
        else:
            prefix = f"[{time_str}] AI ({msg.get('model', '')}):"
        content = msg["content"]
        if msg.get("has_image"):
            content = "[Gambar dilampirkan]\n" + content
        lines.append(f"{prefix}\n{content}\n")
    return "\n".join(lines)

# ============================================================
# TAMPILKAN CHAT HISTORY
# ============================================================

if st.session_state["chat_history"]:
    st.markdown("### 💬 Riwayat Percakapan (Session ini)")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user"><b>Anda</b> <small>({msg.get("time","")})</small><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-ai"><b>AI</b> <small>({msg.get("model","")} • {msg.get("time","")})</small><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    col1, col2, col3 = st.columns(3)
    with col1:
        md_content = format_history_markdown(st.session_state["chat_history"])
        st.download_button(
            "📥 Download Markdown",
            data=md_content,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        wa_content = format_history_whatsapp(st.session_state["chat_history"])
        st.download_button(
            "📱 Download format WhatsApp",
            data=wa_content,
            file_name=f"chat_whatsapp_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col3:
        if st.button("📋 Copy ke Clipboard (WhatsApp)", use_container_width=True):
            st.code(wa_content, language=None)
            st.success("Teks di atas siap di-copy (Ctrl+A → Ctrl+C)")

    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

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

    # Siapkan content user
    user_content = prompt.strip()
    has_image = False

    if is_multimodal and uploaded_file is not None:
        # Encode gambar ke base64
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime = uploaded_file.type or "image/jpeg"
        data_url = f"data:{mime};base64,{img_b64}"

        user_content = [
            {"type": "text", "text": prompt.strip()},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
        has_image = True

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 8192,
        "temperature": 0.7
    }

    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error("❌ HF_TOKEN belum ditemukan.")
        st.info("Tambahkan HF_TOKEN pada Streamlit Secrets.")
        st.stop()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        with st.spinner("🤖 AI sedang memproses pertanyaan..." + (" (dengan gambar)" if has_image else "")):
            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=180
            )

        response.raise_for_status()
        result_data = response.json()

        if "choices" not in result_data or not result_data["choices"]:
            st.error("❌ AI tidak memberikan jawaban.")
            st.stop()

        answer = result_data["choices"][0].get("message", {}).get("content", "")

        if not answer:
            st.error("❌ Content jawaban AI kosong.")
            st.stop()

        # Simpan ke chat history
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state["chat_history"].append({
            "role": "user",
            "content": prompt.strip(),
            "time": now,
            "has_image": has_image
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "model": model,
            "time": now,
            "has_image": False
        })

        st.session_state["prompt_history"] = prompt
        st.session_state["model_selected"] = model
        st.query_params["prompt"] = prompt
        st.query_params["model"] = model

        st.markdown("### ✅ Jawaban AI")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"Model: `{model}` | Response language: `Bahasa Indonesia`")
        st.rerun()   # refresh agar history muncul di atas

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout. Model membutuhkan waktu lebih lama untuk merespons.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP/API Error: {e}")
        try:
            st.code(str(response.json()), language="json")
        except Exception:
            st.code(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network/API error: {e}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Pastikan HF_TOKEN valid dan model yang dipilih tersedia pada Hugging Face Inference Providers.")
