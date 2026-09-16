import requests
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Telco Digital AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS RESPONSIVE
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
       -------------------------------------------------------- */

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 5%;
        padding-right: 5%;
        max-width: 1400px;
    }

    /* --------------------------------------------------------
       TITLE
       -------------------------------------------------------- */

    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }

    .app-caption {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    /* --------------------------------------------------------
       PROMPT TEXTAREA
       -------------------------------------------------------- */

    textarea {
        min-height: 150px !important;
        resize: vertical !important;

        /*
        Memastikan teks membungkus otomatis sesuai
        lebar layar/browser.
        */
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    /* --------------------------------------------------------
       SELECTBOX
       -------------------------------------------------------- */

    div[data-baseweb="select"] {
        width: 100%;
    }

    /* --------------------------------------------------------
       BUTTON
       -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }

    /* --------------------------------------------------------
       ANSWER AREA
       -------------------------------------------------------- */

    .answer-container {
        margin-top: 2rem;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        overflow-wrap: anywhere;
        word-wrap: break-word;
    }

    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media only screen and (max-width: 768px) {

        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-title {
            font-size: 1.75rem;
        }

        .app-caption {
            font-size: 0.82rem;
            line-height: 1.4;
        }

        textarea {
            min-height: 150px !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }

        .stButton > button {
            min-height: 50px;
            font-size: 1rem;
        }

        .answer-container {
            padding: 1rem;
            font-size: 0.95rem;
        }
    }

    /* --------------------------------------------------------
       VERY SMALL MOBILE
       -------------------------------------------------------- */

    @media only screen and (max-width: 480px) {

        .main .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        .app-title {
            font-size: 1.5rem;
        }

        textarea {
            min-height: 145px !important;
        }
    }

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
#
# Contoh:
#
# ?prompt=Apa%20itu%205G
#
# ?prompt=Apa%20itu%205G&model=Qwen/Qwen2.5-72B-Instruct
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
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🤖 Telco Digital AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-caption">
    AI assistant untuk analisis Telco, ICT, Digital Transformation,
    Fiber Optic, 5G, Satellite, Data Center, Regulation,
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

current_model = st.session_state.get(
    "model_selected",
    MODELS[0]
)

try:
    model_index = MODELS.index(current_model)
except ValueError:
    model_index = 0


model = st.selectbox(
    "Pilih Model AI",
    MODELS,
    index=model_index,
    help=(
        "Model yang dipilih akan selalu diarahkan untuk "
        "memberikan jawaban dalam Bahasa Indonesia."
    )
)


# Update session state apabila user mengganti model
st.session_state["model_selected"] = model


# ============================================================
# INFO MODEL
# ============================================================

st.caption(
    f"Model aktif: **{model}** | "
    "Bahasa respons default: **Bahasa Indonesia**"
)


# ============================================================
# TOMBOL TANYA AI
# ============================================================

if st.button(
    "🚀 Tanya AI",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDASI PROMPT
    # --------------------------------------------------------

    if not prompt.strip():

        st.warning(
            "⚠️ Mohon isi pertanyaan terlebih dahulu."
        )

        st.stop()


    # --------------------------------------------------------
    # SYSTEM PROMPT
    #
    # Instruksi ini sengaja dibuat eksplisit agar model
    # apa pun yang dipilih diarahkan menggunakan Bahasa Indonesia.
    # --------------------------------------------------------

    system_prompt = """
Anda adalah Telco Digital AI, seorang AI assistant profesional
untuk bidang Telecommunications, ICT, Digital Transformation,
Business Analysis, Project Management, Risk Management,
Data Center, Fiber Optic, Submarine Cable, Satellite,
5G, IoT, Cloud, Cybersecurity, Regulation dan Digital Infrastructure.

ATURAN BAHASA:

1. Selalu jawab dalam Bahasa Indonesia.
2. Gunakan Bahasa Indonesia yang profesional, jelas dan natural.
3. Istilah teknis internasional boleh tetap menggunakan istilah
   bahasa Inggris apabila istilah tersebut merupakan terminology
   standar industri.
4. Jika pengguna bertanya menggunakan bahasa selain Bahasa Indonesia,
   tetap berikan jawaban utama dalam Bahasa Indonesia.
5. Jangan mengubah pertanyaan pengguna ke bahasa lain sebelum menjawab.
6. Gunakan struktur yang mudah dibaca seperti heading, bullet point,
   numbered list atau tabel apabila memang membantu.
7. Jangan mengatakan bahwa Anda tidak dapat menggunakan Bahasa Indonesia
   hanya karena model dasar Anda menggunakan bahasa lain.

ATURAN KUALITAS:

- Bedakan antara fakta, asumsi, analisis dan rekomendasi.
- Jika informasi bersifat tidak pasti, jelaskan tingkat ketidakpastiannya.
- Jangan mengarang sumber, data, regulasi atau angka.
- Untuk persoalan teknis, jelaskan konsep secara sistematis.
- Untuk persoalan bisnis/proyek, pertimbangkan Business,
  Technology, Project, Risk, Governance dan O&M apabila relevan.
"""


    # ========================================================
    # PAYLOAD API
    # ========================================================

    payload = {

        "model": model,

        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": prompt.strip()
            }
        ],

        "max_tokens": 8192,

        "temperature": 0.7
    }


    # ========================================================
    # HEADER
    # ========================================================

    try:

        hf_token = st.secrets["HF_TOKEN"]

    except Exception:

        st.error(
            "❌ HF_TOKEN belum ditemukan."
        )

        st.info(
            "Tambahkan HF_TOKEN pada Streamlit Secrets."
        )

        st.stop()


    headers = {

        "Authorization": f"Bearer {hf_token}",

        "Content-Type": "application/json",

        "Accept": "application/json"
    }


    # ========================================================
    # REQUEST KE HUGGING FACE
    # ========================================================

    try:

        with st.spinner(
            "🤖 AI sedang memproses pertanyaan..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=120
            )


        # ----------------------------------------------------
        # ERROR HTTP
        # ----------------------------------------------------

        response.raise_for_status()


        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        result_data = response.json()


        if "choices" not in result_data:

            st.error(
                "❌ Response API tidak memiliki field 'choices'."
            )

            st.json(result_data)

            st.stop()


        if not result_data["choices"]:

            st.error(
                "❌ AI tidak memberikan jawaban."
            )

            st.stop()


        answer = (
            result_data["choices"][0]
            .get("message", {})
            .get("content", "")
        )


        if not answer:

            st.error(
                "❌ Content jawaban AI kosong."
            )

            st.stop()


        # ====================================================
        # TAMPILKAN JAWABAN
        # ====================================================

        st.markdown(
            "### ✅ Jawaban AI"
        )

        st.markdown(
            '<div class="answer-container">',
            unsafe_allow_html=True
        )

        st.markdown(answer)

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # UPDATE SESSION STATE
        # ====================================================

        st.session_state["prompt_history"] = prompt

        st.session_state["model_selected"] = model


        # ====================================================
        # UPDATE URL / DEEP LINK
        # ====================================================

        st.query_params["prompt"] = prompt

        st.query_params["model"] = model


        # ====================================================
        # INFORMASI MODEL
        # ====================================================

        st.caption(
            f"Model: `{model}` | "
            "Response language: `Bahasa Indonesia`"
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ Request timeout. "
            "Model membutuhkan waktu lebih lama untuk merespons."
        )


    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ HTTP/API Error: {e}"
        )

        # Tampilkan detail response apabila tersedia
        try:

            error_detail = response.json()

            st.code(
                str(error_detail),
                language="json"
            )

        except Exception:

            st.code(
                response.text
            )


    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ Network/API error: {e}"
        )


    except Exception as e:

        st.error(
            f"❌ Terjadi kesalahan: {e}"
        )

        st.info(
            "Pastikan HF_TOKEN valid dan model yang dipilih "
            "tersedia pada Hugging Face Inference Providers."
        )
