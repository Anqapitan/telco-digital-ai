import requests
import streamlit as st
import base64

# ============================================================
# LOGO SVG INLINE (berdasarkan logo yang Anda lampirkan)
# ============================================================

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" width="300" height="300">
  <!-- Background transparan -->
  <rect width="300" height="300" fill="none"/>
  
  <!-- Bentuk putih kiri -->
  <polygon points="42,90 78,70 78,230 42,210" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
  
  <!-- Bentuk merah tengah atas -->
  <polygon points="112,50 170,26 186,50 186,176" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
  
  <!-- Bentuk putih tengah bawah -->
  <polygon points="112,96 186,222 148,272 112,252" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
  
  <!-- Bentuk merah kanan -->
  <polygon points="222,68 258,88 258,230 222,210" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
</svg>
"""

# Convert SVG ke base64 untuk favicon & logo
LOGO_BASE64 = base64.b64encode(LOGO_SVG.encode("utf-8")).decode("utf-8")
LOGO_DATA_URI = f"data:image/svg+xml;base64,{LOGO_BASE64}"

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="ID Telco Digital AI Assistant",
    page_icon=LOGO_DATA_URI,          # ← Favicon dari logo yang ditanamkan
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS RESPONSIVE + LOGO STYLE
# ============================================================

st.markdown(
    f"""
    <style>
    /* --------------------------------------------------------
       GLOBAL
       -------------------------------------------------------- */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 5%;
        padding-right: 5%;
        max-width: 1400px;
    }}

    /* --------------------------------------------------------
       LOGO HEADER
       -------------------------------------------------------- */
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

    /* --------------------------------------------------------
       TITLE
       -------------------------------------------------------- */
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

    /* --------------------------------------------------------
       PROMPT TEXTAREA
       -------------------------------------------------------- */
    textarea {{
        min-height: 150px !important;
        resize: vertical !important;
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }}

    /* --------------------------------------------------------
       SELECTBOX & BUTTON
       -------------------------------------------------------- */
    div[data-baseweb="select"] {{
        width: 100%;
    }}
    .stButton > button {{
        width: 100%;
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }}

    /* --------------------------------------------------------
       ANSWER AREA
       -------------------------------------------------------- */
    .answer-container {{
        margin-top: 2rem;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        overflow-wrap: anywhere;
        word-wrap: break-word;
    }}

    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */
    @media only screen and (max-width: 768px) {{
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .app-title {{
            font-size: 1.7rem;
        }}
        .app-caption {{
            font-size: 0.82rem;
            line-height: 1.4;
        }}
        .logo-header img {{
            height: 46px;
        }}
        textarea {{
            min-height: 150px !important;
            font-size: 0.95rem !important;
        }}
        .stButton > button {{
            min-height: 50px;
        }}
    }}

    @media only screen and (max-width: 480px) {{
        .app-title {{
            font-size: 1.45rem;
        }}
        .logo-header img {{
            height: 40px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"

# ============================================================
# DAFTAR MODEL
# ============================================================

MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "google/gemma-3-4b-it",
]

# ============================================================
# SESSION STATE
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""

if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = MODELS[0]

# ============================================================
# DEEP LINKING
# ============================================================

q = st.query_params

if "prompt" in q:
    deep_link_prompt = q["prompt"]
    if deep_link_prompt:
        st.session_state["prompt_history"] = deep_link_prompt

if "model" in q:
    deep_link_model = q["model"]
    if deep_link_model in MODELS:
        st.session_state["model_selected"] = deep_link_model

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
# PILIH MODEL AI
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
    help="Model yang dipilih akan selalu diarahkan untuk memberikan jawaban dalam Bahasa Indonesia."
)

st.session_state["model_selected"] = model

# ============================================================
# INFO MODEL
# ============================================================

st.caption(
    f"Model aktif: **{model}** | Bahasa respons default: **Bahasa Indonesia**"
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

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()}
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
        with st.spinner("🤖 AI sedang memproses pertanyaan..."):
            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=120
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

        st.markdown("### ✅ Jawaban AI")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["prompt_history"] = prompt
        st.session_state["model_selected"] = model

        st.query_params["prompt"] = prompt
        st.query_params["model"] = model

        st.caption(f"Model: `{model}` | Response language: `Bahasa Indonesia`")

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
