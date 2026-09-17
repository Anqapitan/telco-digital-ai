"""
ID Telco Digital AI Assistant - v4.4
=====================================================================================
Changelog vs v4.3:
- Text area "Masukkan Pertanyaan Anda" sekarang tinggi dinamis (fleksibel mengikuti jumlah baris/karakter)
- Simple RAG ringan (keyword + overlap scoring atas KEY FACTS & snippet dashboard) → context lebih relevan & panjang terkontrol
- Analytics dashboard internal (query Supabase logs) untuk pemilik app – metrics, top models, countries, recent queries
- Better file handling: PDF text extraction (pypdf), auto-summary untuk file teks panjang, tetap support image ke vision model
- Multi-language toggle (Bahasa Indonesia default / English) – system prompt & jawaban mengikuti pilihan
- Soft-fail tetap dijaga; caching & cascade model tetap ada
"""

import requests
import streamlit as st
import base64
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import uuid
import traceback
from collections import Counter

# ============================================================
# OPTIONAL DEPENDENCIES – explicit status tracking
# ============================================================

DDG_AVAILABLE = False
DDG_ERROR = ""
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError as e:
    DDG_ERROR = str(e)
except Exception as e:
    DDG_ERROR = f"Unexpected: {e}"

BS4_AVAILABLE = False
BS4_ERROR = ""
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError as e:
    BS4_ERROR = str(e)
except Exception as e:
    BS4_ERROR = f"Unexpected: {e}"

SUPABASE_AVAILABLE = False
SUPABASE_ERROR = ""
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError as e:
    SUPABASE_ERROR = str(e)
except Exception as e:
    SUPABASE_ERROR = f"Unexpected: {e}"

FPDF_AVAILABLE = False
FPDF_ERROR = ""
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError as e:
    FPDF_ERROR = str(e)
except Exception as e:
    FPDF_ERROR = f"Unexpected: {e}"

PYPDF_AVAILABLE = False
PYPDF_ERROR = ""
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError as e:
    PYPDF_ERROR = str(e)
except Exception as e:
    PYPDF_ERROR = f"Unexpected: {e}"

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# API & CONSTANTS
# ============================================================

API_URL_HF = "https://router.huggingface.co/v1/chat/completions"
API_URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
API_URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
APP_VERSION = "4.4.0"
MAX_LOG_PROMPT_LEN = 800
MAX_LOG_ANSWER_LEN = 1500
DEFAULT_SUPABASE_TABLE = "telcodigitalai_logs"
MAX_RAG_CHUNKS = 4
MAX_FILE_TEXT_CHARS = 8000

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
# CSS
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
.mermaid {{
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}}
.source-summary {{
    background: #f0f7ff;
    border-left: 4px solid #1e88e5;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.92rem;
}}
.metric-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODELS – expanded multimodal free options (Sep 2026)
# ============================================================

PROVIDER_CHOICES = [
    # --- Groq (text + vision) ---
    {"id": "groq:openai/gpt-oss-20b", "provider": "groq", "model": "openai/gpt-oss-20b",
     "label": "Groq · gpt-oss-20b", "desc": "Aktif · Pengganti Llama 3.1 8B · cepat (default)",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "groq:openai/gpt-oss-120b", "provider": "groq", "model": "openai/gpt-oss-120b",
     "label": "Groq · gpt-oss-120b", "desc": "Aktif · Lebih kuat · pengganti 70B",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "groq:qwen/qwen3.6-27b", "provider": "groq", "model": "qwen/qwen3.6-27b",
     "label": "Groq · qwen3.6-27b (Vision)", "desc": "Aktif · Multimodal · Reasoning",
     "type": "multimodal", "max_files": 5, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "groq:qwen/qwen3.8-27b", "provider": "groq", "model": "qwen/qwen3.8-27b",
     "label": "Groq · qwen3.8-27b (Vision)", "desc": "Aktif · Multimodal · Thinking mode",
     "type": "multimodal", "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    # --- OpenRouter FREE ---
    {"id": "openrouter:openrouter/free", "provider": "openrouter", "model": "openrouter/free",
     "label": "OpenRouter · free (auto-router)", "desc": "FREE · Pilih model gratis otomatis",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter",
     "model": "meta-llama/llama-3.3-70b-instruct:free",
     "label": "OpenRouter · llama-3.3-70b:free", "desc": "FREE · Stabil & kuat",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:openai/gpt-oss-20b:free", "provider": "openrouter",
     "model": "openai/gpt-oss-20b:free",
     "label": "OpenRouter · gpt-oss-20b:free", "desc": "FREE · General purpose",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:google/gemma-4-31b-it:free", "provider": "openrouter",
     "model": "google/gemma-4-31b-it:free",
     "label": "OpenRouter · gemma-4-31b:free (Vision)", "desc": "FREE · Multimodal kuat",
     "type": "multimodal", "max_files": 4, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "openrouter:google/gemma-4-26b-a4b-it:free", "provider": "openrouter",
     "model": "google/gemma-4-26b-a4b-it:free",
     "label": "OpenRouter · gemma-4-26b:free (Vision)", "desc": "FREE · Multimodal efisien",
     "type": "multimodal", "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "provider": "openrouter",
     "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "label": "OpenRouter · nemotron-omni:free (Vision)", "desc": "FREE · Text+Image+Video",
     "type": "multimodal", "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    # --- Hugging Face ---
    {"id": "hf:meta-llama/Llama-3.1-8B-Instruct", "provider": "hf",
     "model": "meta-llama/Llama-3.1-8B-Instruct",
     "label": "HF · Llama-3.1-8B-Instruct", "desc": "HF · Bisa 402 jika kredit habis",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "hf:Qwen/Qwen2.5-72B-Instruct", "provider": "hf",
     "model": "Qwen/Qwen2.5-72B-Instruct",
     "label": "HF · Qwen2.5-72B-Instruct", "desc": "HF · Reasoning kuat",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "hf:Qwen/Qwen2.5-VL-72B-Instruct", "provider": "hf",
     "model": "Qwen/Qwen2.5-VL-72B-Instruct",
     "label": "HF · Qwen2.5-VL-72B (Vision)", "desc": "HF · Multimodal",
     "type": "multimodal", "max_files": 5,
     "accept": ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]},
    {"id": "hf:google/gemma-3-4b-it", "provider": "hf",
     "model": "google/gemma-3-4b-it",
     "label": "HF · gemma-3-4b-it (Vision)", "desc": "HF · Vision ringan",
     "type": "vision", "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
]

CHOICE_BY_ID = {c["id"]: c for c in PROVIDER_CHOICES}
DEFAULT_CHOICE_ID = "groq:openai/gpt-oss-20b"
MODELS = [c["id"] for c in PROVIDER_CHOICES]

# ============================================================
# SPECIALIZED SOURCES + KEY FACTS (Sep 2026)
# ============================================================

SPECIALIZED_SOURCES = {
    "dc": {
        "name": "Live Data Center Asia Pacific",
        "url": "https://narational.byethost11.com/Live_DC_ASPAC.html",
        "description": (
            "Curated Research Dashboard + Query Launcher • 7 Klasifikasi Industri DC APAC "
            "(Business/Investment, AI/HPC Infrastructure, Power & Energy, Water & Environmental, "
            "Regulatory/Governance, Industrial Ecosystem, Strategic/Sovereign Infrastructure). "
            "Fokus Indonesia + koridor digital Asia Pacific. Agregasi berita & riset realtime "
            "(Google News RSS + OpenAlex) periode 2-3 bulan terakhir. Auto-refresh 5 menit. "
            "Oleh nap@iicf.or.id."
        ),
        "keywords": ["data center", "datacenter", "hyperscale", "AI campus", "GPU", "Batam", "Nongsa", "BATIC", "grid readiness", "CoreWeave", "Firmus"]
    },
    "fo": {
        "name": "Live Fiber Optic & Submarine Cable Asia Pacific",
        "url": "https://narational.byethost11.com/Live_FOSubsea_ASPAC.html",
        "description": (
            "Curated Research Dashboard + Query Launcher • 8 Klasifikasi Industri Fiber Optic & SubSEA APAC. "
            "Agregasi proyek kabel laut domestik, landing station, backbone terestrial, coherent optics, "
            "kebijakan hub digital Indonesia + koridor SG-ID-Pasifik. Live realtime search (Google/Bing News RSS + "
            "OpenAlex/Crossref/arXiv). Periode 2-3 bulan terakhir. Oleh nap@iicf.or.id."
        ),
        "keywords": ["submarine cable", "subsea", "fiber optic", "kabel laut", "landing station", "Nongsa-Changi", "Echo cable", "Telin", "WaveLogic", "backbone"]
    }
}

KEY_FACTS_DC = """
=== KEY FACTS LIVE DC ASPAC (snapshot ~Sep 2026, sumber dashboard kurasi + berita) ===
- CoreWeave mengumumkan data center pertamanya di Asia-Pacific di Indonesia (3 fasilitas, total ~360 MW contracted IT power, target online 2028) — validasi posisi RI dalam rantai pasok komputasi AI.
- Firmus Technologies + NVIDIA: kampus AI Factory 360 MW di Batam (bersama DayOne), hingga ~170.000 GPU (Grace-Blackwell / Vera series), offtake projected US$25–30 miliar, target live Q1 2027.
- DayOne mengembangkan kapasitas signifikan di Batam (termasuk PPA listrik besar); Batam/Nongsa menjadi magnet investor berkat kedekatan Singapura, kabel laut, FTZ, dan ketersediaan lahan.
- BATIC 2026 (Bali) menekankan infrastruktur digital & AI sebagai pendorong pertumbuhan ekonomi Asia Pasifik dan kedaulatan digital nasional.
- Forum Grid Readiness & Clean Power membahas kesiapan jaringan listrik Indonesia untuk memasok DC hyperscale & AI (timeline koneksi, ekspansi transmisi).
- Klasifikasi dashboard: A Business/Investment, B AI/HPC, C Power & Energy, D Water & Environmental, E Regulatory, F Industrial Ecosystem, G Strategic/Sovereign.
Periode fokus dashboard: ~16 Jul – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
"""

KEY_FACTS_FO = """
=== KEY FACTS LIVE FO & SUBSEA ASPAC (snapshot ~Sep 2026, sumber dashboard kurasi + berita) ===
- Nongsa-Changi Cable (NCC) mendarat resmi ~20 Juli 2026 di Nongsa Digital Park, Batam (Telin + BW Digital). Panjang ~50 km, 24 fiber pairs, kapasitas >1,6 Pbps, latency <2 ms — jalur terpendek & paling langsung Batam–Singapore (DC-to-DC).
- Sistem Echo (Google & Meta) mendarat di Singapore; arsitektur mencakup jalur trans-Pasifik yang melibatkan Indonesia.
- ION Cable System dan proyek coherent optics (Ciena WaveLogic dll.) memperkuat backbone Jakarta–Singapore + Sumatra.
- Telkom (via Telin) memperkuat ambisi Indonesia sebagai Hub Internet Asia Pasifik; model bisnis bergeser ke wholesale + layanan AI-ready & DCI.
- Pemerintah menyiapkan landing station baru dan titik hub sesuai regulasi yang ada.
- Fokus: proyek kabel laut domestik, landing station, backbone terestrial, coherent optics, kebijakan hub digital.
Periode fokus dashboard: ~16 Jun – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
"""

# ============================================================
# SIMPLE RAG (lightweight keyword + overlap scoring)
# ============================================================

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _chunk_text(text: str, max_chars: int = 450) -> List[str]:
    """Split long text into overlapping-ish chunks by paragraphs/sentences."""
    if not text or len(text) < max_chars:
        return [text.strip()] if text.strip() else []
    parts = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-Z\-])", text)
    chunks, current = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def build_rag_corpus(prioritize: str = "both") -> List[Dict[str, str]]:
    """Build list of {id, source, text} for retrieval."""
    corpus = []
    if prioritize in ("dc", "both"):
        for i, ch in enumerate(_chunk_text(KEY_FACTS_DC)):
            corpus.append({"id": f"dc-{i}", "source": "KEY_FACTS_DC + Live DC ASPAC", "text": ch})
        corpus.append({
            "id": "dc-desc",
            "source": SPECIALIZED_SOURCES["dc"]["name"],
            "text": SPECIALIZED_SOURCES["dc"]["description"] + " URL: " + SPECIALIZED_SOURCES["dc"]["url"]
        })
    if prioritize in ("fo", "both"):
        for i, ch in enumerate(_chunk_text(KEY_FACTS_FO)):
            corpus.append({"id": f"fo-{i}", "source": "KEY_FACTS_FO + Live FO/Subsea ASPAC", "text": ch})
        corpus.append({
            "id": "fo-desc",
            "source": SPECIALIZED_SOURCES["fo"]["name"],
            "text": SPECIALIZED_SOURCES["fo"]["description"] + " URL: " + SPECIALIZED_SOURCES["fo"]["url"]
        })
    return corpus


def simple_rag_retrieve(query: str, prioritize: str = "both", top_k: int = MAX_RAG_CHUNKS) -> str:
    """
    Lightweight RAG: score chunks by token overlap (Jaccard-like) + keyword boost.
    No external embedding model required – works on free Streamlit Cloud.
    """
    corpus = build_rag_corpus(prioritize)
    if not corpus:
        return ""

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        # return top static chunks
        selected = corpus[:top_k]
    else:
        scored = []
        for doc in corpus:
            d_tokens = set(_tokenize(doc["text"]))
            if not d_tokens:
                continue
            inter = len(q_tokens & d_tokens)
            union = len(q_tokens | d_tokens) or 1
            jaccard = inter / union
            # boost if important domain keywords present
            boost = 0.0
            for kw in (SPECIALIZED_SOURCES["dc"]["keywords"] + SPECIALIZED_SOURCES["fo"]["keywords"]):
                if kw.lower() in query.lower() and kw.lower() in doc["text"].lower():
                    boost += 0.08
            score = jaccard + min(boost, 0.4)
            if score > 0.02:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [d for _, d in scored[:top_k]]
        if not selected:
            selected = corpus[:min(2, len(corpus))]

    if not selected:
        return ""

    out = ["=== RAG RETRIEVED CONTEXT (dari KEY FACTS & deskripsi dashboard) ===\n"]
    for i, doc in enumerate(selected, 1):
        out.append(f"[{i}] Sumber: {doc['source']}\n{doc['text']}\n")
    out.append(
        "\n(Gunakan potongan di atas sebagai prioritas fakta jika relevan dengan pertanyaan. "
        "Sebutkan URL dashboard Live DC / Live FO-Subsea bila memakai fakta ini.)\n"
    )
    return "\n".join(out)


# ============================================================
# UTILS
# ============================================================

def _is_private_ip(ip: str) -> bool:
    if not ip or ip == "unknown":
        return True
    try:
        parts = [int(p) for p in ip.split(".")]
        if len(parts) != 4:
            return True
        a, b = parts[0], parts[1]
        if a == 10:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
        if a == 127:
            return True
        if a == 0 or a >= 224:
            return True
        return False
    except Exception:
        return True


def safe_get_ip_and_country() -> Tuple[str, str]:
    ip = "unknown"
    country = "unknown"
    try:
        headers = {}
        if hasattr(st, "context") and st.context:
            headers = dict(st.context.headers or {})
        candidates = []
        header_keys = [
            "X-Forwarded-For", "x-forwarded-for",
            "X-Real-IP", "x-real-ip",
            "CF-Connecting-IP", "cf-connecting-ip",
            "True-Client-IP", "true-client-ip",
            "X-Client-IP", "x-client-ip",
            "Forwarded", "forwarded",
        ]
        for key in header_keys:
            val = headers.get(key)
            if not val:
                continue
            for part in str(val).replace("for=", "").split(","):
                part = part.strip().strip('"').split(";")[0].strip()
                if part and part not in candidates:
                    candidates.append(part)
        for cand in candidates:
            if not _is_private_ip(cand):
                ip = cand
                break
        if ip == "unknown" or _is_private_ip(ip):
            try:
                r = requests.get("https://ipapi.co/json/", timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    pub = data.get("ip", "")
                    if pub and not _is_private_ip(pub):
                        ip = pub
                        country = data.get("country_name") or data.get("country_code") or country
            except Exception:
                pass
        if ip != "unknown" and not _is_private_ip(ip) and country == "unknown":
            try:
                r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    country = data.get("country_name") or data.get("country_code") or country
            except Exception:
                pass
        if _is_private_ip(ip):
            ip = "internal"
            if country == "unknown":
                country = "Streamlit Cloud"
    except Exception:
        pass
    return ip, country


def get_external_referrer() -> str:
    try:
        headers = {}
        if hasattr(st, "context") and st.context:
            headers = st.context.headers or {}
        referer = (
            headers.get("Referer")
            or headers.get("referer")
            or headers.get("Referrer")
            or headers.get("referrer")
            or ""
        )
        referer = str(referer).strip()
        if not referer:
            return ""
        own_domains = [
            "telco-digital-ai.streamlit.app",
            "localhost",
            "127.0.0.1",
            "streamlit.app",
        ]
        referer_lower = referer.lower()
        if any(d in referer_lower for d in own_domains):
            return ""
        return referer[:500]
    except Exception:
        return ""


def generate_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:12]
    return st.session_state["session_id"]


def sanitize_text(text: str, max_len: int = 4000) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))
    return text[:max_len]


def _get_secret(*names):
    for n in names:
        try:
            v = st.secrets.get(n)
            if v is None:
                continue
            s = str(v).strip().strip('"').strip("'")
            if s and s not in ("None", "null"):
                return s
        except Exception:
            continue
    return None


def compute_dynamic_textarea_height(text: str) -> int:
    """Fleksibel: tinggi menyesuaikan jumlah baris / karakter yang diketik."""
    if not text:
        return 140
    lines = text.count("\n") + 1
    # perkiraan wrap (≈70 karakter per baris di UI)
    approx_wrapped = max(lines, (len(text) // 70) + 1)
    height = 90 + approx_wrapped * 22
    return max(120, min(520, height))


# ============================================================
# SUPABASE LOGGING
# ============================================================

def get_supabase_table_name() -> str:
    return _get_secret("SUPABASE_TABLE") or DEFAULT_SUPABASE_TABLE


def get_supabase_client() -> Optional["Client"]:
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        if st.session_state.get("debug_mode"):
            st.warning(f"[Supabase client] {e}")
        return None


def append_behavior_log(row: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        client = get_supabase_client()
        if client is None:
            reason = "Supabase library missing" if not SUPABASE_AVAILABLE else "SUPABASE_URL / SUPABASE_KEY tidak terbaca di secrets"
            return False, reason
        table_name = get_supabase_table_name()
        clean = {k: ("" if v is None else str(v)) for k, v in row.items()}
        client.table(table_name).insert(clean).execute()
        return True, f"OK → tabel `{table_name}`"
    except Exception as e:
        msg = str(e)[:300]
        if st.session_state.get("debug_mode"):
            st.warning(f"[Log] Gagal tulis ke Supabase: {msg}")
        return False, msg


def log_access(
    feature: str,
    prompt: str = "",
    answer: str = "",
    model: str = "",
    files_count: int = 0,
    web_search: bool = False,
    specialized: bool = False,
    error_note: str = "",
    language: str = "id"
) -> Tuple[bool, str]:
    try:
        ip, country = safe_get_ip_and_country()
        ua = ""
        try:
            headers = st.context.headers if hasattr(st, "context") else {}
            ua = headers.get("User-Agent", "") or headers.get("user-agent", "")
        except Exception:
            pass
        row = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": generate_session_id(),
            "ip": ip,
            "country": country,
            "origin_url": get_external_referrer(),
            "feature": feature,
            "model": model,
            "prompt_snippet": sanitize_text(prompt, MAX_LOG_PROMPT_LEN),
            "answer_snippet": sanitize_text(answer, MAX_LOG_ANSWER_LEN),
            "files_uploaded": files_count,
            "web_search_used": web_search,
            "specialized_used": specialized,
            "user_agent": sanitize_text(ua, 200),
            "app_version": APP_VERSION,
            "error_note": sanitize_text(error_note, 300),
            "language": language,
        }
        return append_behavior_log(row)
    except Exception as e:
        return False, str(e)[:200]


# ============================================================
# ANALYTICS DASHBOARD (owner)
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_recent_logs(limit: int = 200) -> List[Dict]:
    client = get_supabase_client()
    if not client:
        return []
    try:
        table = get_supabase_table_name()
        res = (
            client.table(table)
            .select("*")
            .order("timestamp_utc", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def render_analytics_dashboard():
    st.markdown("### 📊 Analytics Internal (Owner)")
    client = get_supabase_client()
    if not client:
        st.info("Supabase belum terkonfigurasi atau library tidak tersedia. Analytics tidak aktif.")
        return

    logs = fetch_recent_logs(300)
    if not logs:
        st.warning("Belum ada data log atau query gagal (periksa RLS / nama tabel / secrets).")
        return

    # Metrics
    total = len(logs)
    success = sum(1 for r in logs if r.get("feature") == "query_success")
    errors = sum(1 for r in logs if r.get("feature") == "error")
    with_files = sum(1 for r in logs if int(r.get("files_uploaded") or 0) > 0)
    specialized = sum(1 for r in logs if str(r.get("specialized_used", "")).lower() in ("true", "1", "yes"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total log (recent)", total)
    c2.metric("Query sukses", success)
    c3.metric("Error", errors)
    c4.metric("Dengan file", with_files)
    c5.metric("Pakai specialized", specialized)

    # Top models
    models = [r.get("model") or "unknown" for r in logs if r.get("feature") in ("query_success", "query_start")]
    top_models = Counter(models).most_common(8)
    countries = [r.get("country") or "unknown" for r in logs]
    top_countries = Counter(countries).most_common(8)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top Models**")
        if top_models:
            st.dataframe(
                {"Model": [m for m, _ in top_models], "Count": [c for _, c in top_models]},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Tidak ada data model.")
    with col_b:
        st.markdown("**Top Countries / Origin**")
        if top_countries:
            st.dataframe(
                {"Country": [c for c, _ in top_countries], "Count": [n for _, n in top_countries]},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Tidak ada data negara.")

    # Recent queries
    st.markdown("**Recent queries (snippet)**")
    recent_rows = []
    for r in logs[:25]:
        if r.get("feature") not in ("query_start", "query_success", "error"):
            continue
        recent_rows.append({
            "UTC": r.get("timestamp_utc", "")[:19],
            "Feature": r.get("feature", ""),
            "Model": (r.get("model") or "")[:40],
            "Country": r.get("country", ""),
            "Prompt": (r.get("prompt_snippet") or "")[:80],
            "Lang": r.get("language", "id"),
        })
    if recent_rows:
        st.dataframe(recent_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada query tercatat.")

    if st.button("🔄 Refresh analytics cache"):
        fetch_recent_logs.clear()
        st.rerun()


# ============================================================
# MODEL AVAILABILITY (lightweight, cached)
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def check_model_availability(provider: str, model: str, api_key: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "no API key"
    try:
        if provider == "groq":
            url = API_URL_GROQ
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
            r = requests.post(url, json=payload, headers=headers, timeout=12)
            if r.status_code in (200, 201):
                return True, "ok"
            if r.status_code in (400, 404):
                return False, f"HTTP_{r.status_code}"
            if r.status_code in (401, 403):
                return False, "auth"
            if r.status_code == 429:
                return True, "rate-limited (still listed)"
            return False, f"HTTP_{r.status_code}"
        elif provider == "openrouter":
            url = API_URL_OPENROUTER
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://telco-digital-ai.streamlit.app",
                "X-Title": "ID Telco Digital AI",
            }
            payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
            r = requests.post(url, json=payload, headers=headers, timeout=12)
            if r.status_code in (200, 201):
                return True, "ok"
            if r.status_code == 402:
                return False, "payment / free quota"
            if r.status_code in (400, 404):
                return False, f"HTTP_{r.status_code}"
            if r.status_code == 429:
                return True, "rate-limited"
            return False, f"HTTP_{r.status_code}"
        else:
            return True, "hf (not probed)"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:80]


# ============================================================
# WEB SEARCH + SPECIALIZED (with caching)
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    if not DDG_AVAILABLE:
        return f"Library duckduckgo-search belum terinstall. Error: {DDG_ERROR}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "Tidak ditemukan hasil pencarian."
        output = f"Hasil pencarian umum untuk: **{query}**\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "Tanpa Judul")
            href = r.get("href", "")
            body = r.get("body", "")[:350]
            output += f"{i}. **{title}**\n{body}...\nSumber: {href}\n\n"
        return output
    except Exception as e:
        return f"Gagal melakukan pencarian umum: {str(e)}"


@st.cache_data(ttl=300, show_spinner=False)
def try_scrape_dashboard(url: str, max_chars: int = 2500) -> str:
    if not BS4_AVAILABLE:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,id;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if "requires Javascript" in text or len(text) < 200:
            return ""
        return text[:max_chars]
    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def specialized_apac_search(user_query: str, max_results: int = 6, prioritize: str = "both") -> str:
    output_parts = []
    output_parts.append(
        "=== SUMBER KURASI PRIMER (PRIORITAS TINGGI) ===\n"
        "Gunakan informasi terkini dari dashboard live berikut sebagai referensi utama "
        "untuk topik Data Center, Fiber Optic, dan Submarine Cable di Asia Pacific / Indonesia "
        "(periode 2-3 bulan terakhir). Sebutkan sumbernya jika relevan.\n\n"
    )
    if prioritize in ("dc", "both"):
        output_parts.append(
            f"1. **{SPECIALIZED_SOURCES['dc']['name']}**\n"
            f"   URL: {SPECIALIZED_SOURCES['dc']['url']}\n"
            f"   Deskripsi: {SPECIALIZED_SOURCES['dc']['description']}\n"
        )
        output_parts.append(KEY_FACTS_DC)
    if prioritize in ("fo", "both"):
        output_parts.append(
            f"2. **{SPECIALIZED_SOURCES['fo']['name']}**\n"
            f"   URL: {SPECIALIZED_SOURCES['fo']['url']}\n"
            f"   Deskripsi: {SPECIALIZED_SOURCES['fo']['description']}\n"
        )
        output_parts.append(KEY_FACTS_FO)

    if not DDG_AVAILABLE:
        output_parts.append("\n[Info] DuckDuckGo tidak tersedia. Gunakan KEY FACTS di atas + pengetahuan model.")
        return "\n".join(output_parts)

    site_queries = []
    if prioritize in ("dc", "both"):
        site_queries.append(f'site:narational.byethost11.com ({user_query})')
        site_queries.append('site:narational.byethost11.com (CoreWeave OR Firmus OR "data center" OR Batam OR Nongsa OR BATIC OR "grid readiness")')
    if prioritize in ("fo", "both"):
        site_queries.append('site:narational.byethost11.com ("Nongsa-Changi" OR Echo OR WaveLogic OR "submarine cable" OR "kabel laut" OR landing)')
        site_queries.append(f'site:narational.byethost11.com (fiber OR subsea OR "submarine cable") Indonesia')

    q_lower = user_query.lower()
    extra_terms = []
    if any(k in q_lower for k in ["data center", "datacenter", "dc ", "hyperscale", "ai campus", "gpu", "coreweave", "firmus"]):
        extra_terms.extend(["CoreWeave", "Firmus", "Batam", "Nongsa", "BATIC 2026", "grid readiness", "360MW"])
    if any(k in q_lower for k in ["fiber", "optic", "subsea", "submarine", "kabel laut", "landing", "echo", "nongsa"]):
        extra_terms.extend(["Nongsa-Changi", "Echo cable", "Telin", "WaveLogic 6", "landing station", "ION Cable", "Jayapura"])
    if extra_terms:
        site_queries.append(f'site:narational.byethost11.com ({" OR ".join(extra_terms[:5])})')

    broad_queries = [
        f'("{user_query}") (Indonesia OR "Asia Pacific" OR APAC) (2025 OR 2026) (data center OR "fiber optic" OR subsea OR "submarine cable")',
        '(data center OR "submarine cable" OR "fiber optic") Indonesia (Batam OR Nongsa OR "landing station" OR CoreWeave OR Firmus) 2026',
    ]

    all_results = []
    seen_urls = set()
    try:
        with DDGS() as ddgs:
            for sq in site_queries:
                try:
                    res = list(ddgs.text(sq, max_results=4))
                    for r in res:
                        href = r.get("href", "")
                        if href and href not in seen_urls:
                            seen_urls.add(href)
                            all_results.append(r)
                except Exception:
                    continue
            if len(all_results) < 3:
                for bq in broad_queries:
                    try:
                        res = list(ddgs.text(bq, max_results=4))
                        for r in res:
                            href = r.get("href", "")
                            if href and href not in seen_urls:
                                seen_urls.add(href)
                                all_results.append(r)
                    except Exception:
                        continue
    except Exception as e:
        output_parts.append(f"\n[Peringatan] Gagal menjalankan pencarian targeted: {str(e)}\n")

    if all_results:
        output_parts.append("\n=== HASIL PENCARIAN TARGETED (terkait dashboard & berita terkini) ===\n")
        for i, r in enumerate(all_results[:max_results], 1):
            title = r.get("title", "Tanpa Judul")
            href = r.get("href", "")
            body = r.get("body", "")[:400]
            source_tag = " [SUMBER KURASI]" if "narational.byethost11.com" in href else ""
            output_parts.append(f"{i}. **{title}**{source_tag}\n{body}...\nSumber: {href}\n")
    else:
        output_parts.append("\n[Info] Tidak ditemukan hasil pencarian tambahan. Tetap prioritaskan KEY FACTS + dashboard live di atas.\n")

    output_parts.append(
        "\n=== INSTRUKSI UNTUK AI ===\n"
        "- Prioritaskan fakta dan tren yang selaras dengan KEY FACTS dan dashboard live.\n"
        "- Jika membahas Data Center atau Fiber/Subsea APAC/Indonesia, sebutkan URL dashboard yang relevan.\n"
        "- Bedakan dengan jelas: fakta dari sumber vs analisis/rekomendasi Anda.\n"
        "- Periode fokus: 2-3 bulan terakhir (pertengahan 2026).\n"
    )
    return "\n".join(output_parts)


# ============================================================
# BETTER FILE HANDLING
# ============================================================

def extract_pdf_text(file_bytes: bytes, max_chars: int = MAX_FILE_TEXT_CHARS) -> str:
    if not PYPDF_AVAILABLE:
        return "[PDF] Library pypdf tidak tersedia – tidak bisa ekstrak teks. Upload sebagai gambar halaman jika model vision mendukung."
    try:
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        texts = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t.strip())
                total += len(t)
                if total >= max_chars:
                    break
        full = "\n\n".join(texts)
        if len(full) > max_chars:
            full = full[:max_chars] + "\n\n[... teks PDF dipotong karena panjang ...]"
        return full if full.strip() else "[PDF] Tidak ada teks yang bisa diekstrak (mungkin scanned/image-only)."
    except Exception as e:
        return f"[PDF] Gagal ekstrak: {str(e)[:120]}"


def summarize_long_text(text: str, max_keep: int = 3500) -> str:
    """Simple extractive-ish summary: keep head + tail + middle keywords if too long."""
    if len(text) <= max_keep:
        return text
    head = text[: max_keep // 2]
    tail = text[-(max_keep // 3) :]
    return (
        head
        + "\n\n[... bagian tengah diringkas otomatis karena file terlalu panjang ...]\n\n"
        + tail
    )


def process_uploaded_file(f, is_vision_capable: bool) -> List[Dict]:
    """Return list of content parts (text and/or image_url) for the messages."""
    parts = []
    try:
        file_bytes = f.read()
        name = (f.name or "file").lower()
        mime = f.type or "application/octet-stream"

        # Images → vision
        if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]):
            if is_vision_capable:
                b64 = base64.b64encode(file_bytes).decode()
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            else:
                parts.append({"type": "text", "text": f"[Gambar {f.name} diupload tetapi model saat ini tidak multimodal/vision.]"})
            return parts

        # PDF
        if name.endswith(".pdf"):
            text = extract_pdf_text(file_bytes)
            text = summarize_long_text(text)
            parts.append({"type": "text", "text": f"\n\n[Isi PDF: {f.name}]\n{text}"})
            return parts

        # Plain text / md / others
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text_content = file_bytes.decode("latin-1", errors="ignore")
        text_content = summarize_long_text(text_content, MAX_FILE_TEXT_CHARS)
        parts.append({"type": "text", "text": f"\n\n[Isi file {f.name}]\n{text_content}"})
        return parts
    except Exception as e:
        parts.append({"type": "text", "text": f"[Gagal memproses file {getattr(f, 'name', '?')}: {e}]"})
        return parts


# ============================================================
# PDF EXPORT
# ============================================================

def create_pdf_from_history(history: List[Dict], title: str = "Riwayat Chat - ID Telco Digital AI") -> Optional[bytes]:
    if not FPDF_AVAILABLE:
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        try:
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        gen_line = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | App v{APP_VERSION}"
        try:
            pdf.cell(0, 8, gen_line, new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            pdf.cell(0, 8, gen_line, ln=True)
        pdf.ln(5)

        for item in history:
            if item.get("role") == "user":
                ip = item.get("ip", "unknown")
                country = item.get("country", "unknown")
                role = f"[{ip}] [{country}]"
            else:
                role = "AI"
            header = f"[{item.get('time', '')}] {role} ({item.get('model', '')})"
            pdf.set_font("Helvetica", "B", 11)
            try:
                pdf.cell(0, 8, header, new_x="LMARGIN", new_y="NEXT")
            except TypeError:
                pdf.cell(0, 8, header, ln=True)
            pdf.set_font("Helvetica", "", 10)
            content = re.sub(r"[*_`#]", "", str(item.get("content", "")))[:3000]
            content = content.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

        raw = pdf.output(dest="S")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        if isinstance(raw, str):
            return raw.encode("latin-1")
        return None
    except Exception:
        return None


# ============================================================
# MERMAID HELPER
# ============================================================

def extract_and_render_mermaid(text: str):
    pattern = r"```mermaid\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return
    st.markdown("#### Diagram Mermaid terdeteksi")
    for i, code in enumerate(matches):
        code = code.strip()
        lines = code.count("\n") + 1
        height = min(900, max(280, 40 + lines * 28))
        mermaid_html = f"""
        <div style="position:relative;">
          <button onclick="
            const el = document.getElementById('mermaid-wrap-{i}');
            if (!document.fullscreenElement) {{
              el.requestFullscreen().catch(()=>{{}});
            }} else {{
              document.exitFullscreen();
            }}
          " style="position:absolute;top:8px;right:8px;z-index:10;padding:4px 10px;font-size:12px;cursor:pointer;border-radius:4px;border:1px solid #ccc;background:#fff;">
            ⛶ Fullscreen
          </button>
          <div id="mermaid-wrap-{i}" class="mermaid" style="min-height:{height}px; background:#f8f9fa; padding:1.2rem; border-radius:8px;">
{code}
          </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
          try {{
            mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
            mermaid.run({{ nodes: [document.getElementById('mermaid-wrap-{i}')] }});
          }} catch (e) {{
            document.getElementById('mermaid-wrap-{i}').innerHTML = '<pre style="white-space:pre-wrap;color:#333;">' + 
              {json.dumps(code)} + '</pre><p style="color:#c00;font-size:0.85rem;">(Render Mermaid gagal – menampilkan kode mentah)</p>';
          }}
        </script>
        """
        st.components.v1.html(mermaid_html, height=height + 60, scrolling=True)
        with st.expander(f"Lihat kode Mermaid #{i+1} (fallback teks)"):
            st.code(code, language="mermaid")


# ============================================================
# SESSION STATE INIT
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""
if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = DEFAULT_CHOICE_ID
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "enable_web_search" not in st.session_state:
    st.session_state["enable_web_search"] = True
if "enable_specialized_apac" not in st.session_state:
    st.session_state["enable_specialized_apac"] = True
if "enable_rag" not in st.session_state:
    st.session_state["enable_rag"] = True
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False
if "preferred_provider" not in st.session_state:
    st.session_state["preferred_provider"] = "auto"
if "session_id" not in st.session_state:
    generate_session_id()
if "prioritize_dashboard" not in st.session_state:
    st.session_state["prioritize_dashboard"] = "both"
if "force_refresh_context" not in st.session_state:
    st.session_state["force_refresh_context"] = False
if "last_log_status" not in st.session_state:
    st.session_state["last_log_status"] = ""
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "id"  # id | en

# ============================================================
# DEEP LINKING
# ============================================================

q = st.query_params
if "prompt" in q and q["prompt"]:
    st.session_state["prompt_history"] = q["prompt"]
if "model" in q:
    mid = q["model"]
    if mid in CHOICE_BY_ID:
        st.session_state["model_selected"] = mid
    else:
        for c in PROVIDER_CHOICES:
            if c["model"] == mid or c["id"].endswith(mid):
                st.session_state["model_selected"] = c["id"]
                break
if "lang" in q and q["lang"] in ("id", "en"):
    st.session_state["ui_language"] = q["lang"]

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="logo-header">
    <img src="{LOGO_DATA_URI}" alt="Logo">
    <div>
        <div class="app-title">ID Telco Digital AI Assistant</div>
        <div class="app-caption" style="margin-bottom:0">
            Gen-AI Literature Analytics by nap@iicf.or.id • v{APP_VERSION}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-caption">
AI assistant untuk analisis Telco, ICT, Digital Transformation, Fiber Optic, 5G, 
Satellite, Data Center, Regulation, Project & Risk Management.
<br>Dilengkapi sumber kurasi live DC & FO/Subsea Asia Pacific (2-3 bulan terakhir) + Simple RAG.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR / SETTINGS
# ============================================================

with st.expander("⚙️ Pengaturan Web Search, Bahasa, Logging & Fitur Lanjutan", expanded=False):
    # Language toggle
    lang_options = {"id": "Bahasa Indonesia (default)", "en": "English"}
    st.session_state["ui_language"] = st.selectbox(
        "🌐 Bahasa jawaban / Answer language",
        options=["id", "en"],
        format_func=lambda x: lang_options[x],
        index=0 if st.session_state.get("ui_language", "id") == "id" else 1,
        help="System prompt dan jawaban AI akan mengikuti pilihan ini."
    )

    st.session_state["enable_web_search"] = st.checkbox(
        "Aktifkan Web Search Umum (DuckDuckGo - Gratis)",
        value=st.session_state["enable_web_search"]
    )
    st.caption("Menggunakan DuckDuckGo. Tidak memerlukan API key.")

    st.session_state["enable_specialized_apac"] = st.checkbox(
        "⭐ Aktifkan Sumber Kurasi Live DC & FO/Subsea APAC (Prioritas)",
        value=st.session_state["enable_specialized_apac"]
    )

    st.session_state["enable_rag"] = st.checkbox(
        "🧠 Aktifkan Simple RAG (KEY FACTS + deskripsi dashboard)",
        value=st.session_state.get("enable_rag", True),
        help="Retrieval ringan berbasis keyword/overlap untuk context lebih fokus."
    )

    st.session_state["prioritize_dashboard"] = st.selectbox(
        "Prioritaskan dashboard",
        options=["both", "dc", "fo"],
        format_func=lambda x: {"both": "DC + FO/Subsea (keduanya)", "dc": "Hanya Data Center", "fo": "Hanya Fiber/Subsea"}[x],
        index=["both", "dc", "fo"].index(st.session_state.get("prioritize_dashboard", "both"))
    )

    st.session_state["force_refresh_context"] = st.checkbox(
        "Force refresh context (abaikan cache search/scrape)",
        value=st.session_state.get("force_refresh_context", False),
        help="Gunakan jika ingin data paling baru dari dashboard (mengorbankan kecepatan)."
    )

    if st.session_state["enable_specialized_apac"]:
        st.markdown(
            f"""
            **Dashboard primer:**
            - [Live DC ASPAC]({SPECIALIZED_SOURCES['dc']['url']})
            - [Live FO & Subsea ASPAC]({SPECIALIZED_SOURCES['fo']['url']})
            """
        )

    st.session_state["debug_mode"] = st.checkbox("Debug mode (tampilkan error detail)", value=st.session_state.get("debug_mode", False))

    st.caption(
        "Error 400/402/404 **tidak ditampilkan** ke user — app otomatis ganti ke provider/model lain. "
        "Laporan hanya muncul jika **semua** pilihan gagal."
    )

    st.markdown("#### Status Dependencies & Secrets")
    dep_status = []
    dep_status.append(f"- `duckduckgo-search`: {'✅' if DDG_AVAILABLE else '❌ ' + DDG_ERROR[:60]}")
    dep_status.append(f"- `beautifulsoup4`: {'✅' if BS4_AVAILABLE else '❌ ' + BS4_ERROR[:60]}")
    dep_status.append(f"- `supabase`: {'✅' if SUPABASE_AVAILABLE else '❌ ' + SUPABASE_ERROR[:60]}")
    dep_status.append(f"- `fpdf2`: {'✅' if FPDF_AVAILABLE else '❌ ' + FPDF_ERROR[:60]}")
    dep_status.append(f"- `pypdf`: {'✅' if PYPDF_AVAILABLE else '❌ ' + PYPDF_ERROR[:60]}")
    st.markdown("\n".join(dep_status))

    has_hf = bool(_get_secret("HF_TOKEN", "hf_token"))
    has_groq = bool(_get_secret("GROQ_API_KEY", "groq_api_key", "GROQ_KEY"))
    has_or = bool(_get_secret("OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY"))
    sb_client = get_supabase_client()
    sb_table = get_supabase_table_name()
    supabase_status = f"Aktif (tabel: `{sb_table}`)" if sb_client else "Belum dikonfigurasi / library hilang"

    st.markdown(
        f"**Secrets:**  \n"
        f"- `HF_TOKEN`: {'✅' if has_hf else '❌'}  \n"
        f"- `GROQ_API_KEY`: {'✅' if has_groq else '❌'}  \n"
        f"- `OPENROUTER_API_KEY`: {'✅' if has_or else '❌'}  \n"
        f"- Supabase: **{supabase_status}** · Session: `{st.session_state.get('session_id', '-')}`"
    )

    if st.session_state.get("last_log_status"):
        st.caption(f"Last log attempt: {st.session_state['last_log_status']}")

    if st.button("🧪 Test Tulis Log ke Supabase", use_container_width=True):
        ok, msg = log_access(feature="test_log", prompt="manual test from settings", model="n/a",
                             language=st.session_state.get("ui_language", "id"))
        st.session_state["last_log_status"] = f"{'✅' if ok else '❌'} {msg}"
        if ok:
            st.success(f"Test log berhasil: {msg}")
        else:
            st.error(f"Test log gagal: {msg}")
            st.info(
                "Kemungkinan penyebab tabel kosong:\n"
                f"1. Nama tabel di Supabase bukan `{sb_table}` (ubah via secret SUPABASE_TABLE).\n"
                "2. RLS (Row Level Security) memblokir insert dari anon key.\n"
                "3. SUPABASE_URL / SUPABASE_KEY salah atau belum di-Save + Reboot.\n"
                "4. Library supabase belum terinstall di environment Streamlit."
            )

    if not has_groq:
        st.warning(
            "⚠️ `GROQ_API_KEY` tidak terbaca. Pastikan format TOML satu baris:\n"
            '`GROQ_API_KEY = "gsk_xxx"` lalu **Save + Reboot app**.'
        )

# ============================================================
# ANALYTICS (owner section)
# ============================================================

with st.expander("📊 Analytics Dashboard (Owner / Internal)", expanded=False):
    render_analytics_dashboard()

# ============================================================
# MODEL SELECT
# ============================================================

choice_ids = [c["id"] for c in PROVIDER_CHOICES]
current_id = st.session_state.get("model_selected", DEFAULT_CHOICE_ID)
if current_id not in choice_ids:
    current_id = DEFAULT_CHOICE_ID
try:
    model_index = choice_ids.index(current_id)
except Exception:
    model_index = 0

selected_id = st.selectbox(
    "Pilih Provider · Model AI",
    options=choice_ids,
    index=model_index,
    format_func=lambda i: f"{CHOICE_BY_ID[i]['label']}  —  {CHOICE_BY_ID[i]['desc']}",
    help="Default: Groq free tier. Jika gagal, otomatis coba pilihan lain. Model bertanda Vision mendukung upload gambar."
)
st.session_state["model_selected"] = selected_id
choice = CHOICE_BY_ID[selected_id]
model = choice["model"]
cap = {
    "type": choice["type"],
    "max_files": choice.get("max_files", 0),
    "accept": choice.get("accept", []),
}

st.caption(
    f"**Provider:** `{choice['provider']}` · **Model:** `{choice['model']}` · "
    f"Tipe: {choice['type']}"
    + (f" · Max {cap['max_files']} file" if cap["max_files"] else "")
)

# ============================================================
# UPLOAD
# ============================================================

uploaded_files = []
# Accept PDF for all models (text extraction); images only for vision/multimodal
accept_types = list(set((cap.get("accept") or []) + ["pdf", "txt", "md"]))
if cap["type"] in ["vision", "multimodal"] or True:  # always allow text/PDF
    uploaded_files = st.file_uploader(
        "📎 Upload file pendukung (opsional) – PDF akan diekstrak teks; gambar hanya untuk model Vision",
        type=accept_types if accept_types else ["pdf", "txt", "md", "png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
        help=f"Maksimal {cap['max_files'] or 5} file. PDF → text extraction. Image → vision model."
    )
    max_f = cap["max_files"] or 5
    if uploaded_files and len(uploaded_files) > max_f:
        uploaded_files = uploaded_files[:max_f]
        st.warning(f"Hanya {max_f} file pertama yang digunakan.")

# ============================================================
# PROMPT – dynamic height
# ============================================================

current_prompt_val = st.session_state.get("prompt_history", "")
dyn_height = compute_dynamic_textarea_height(current_prompt_val)

prompt = st.text_area(
    "Masukkan Pertanyaan Anda",
    value=current_prompt_val,
    height=dyn_height,
    placeholder="Contoh:\nApa tren terbaru Data Center di Asia Pacific 2026?\nBuatkan diagram Mermaid arsitektur 5G Core.\nJelaskan regulasi AI di Indonesia.\nUpdate proyek kabel laut Indonesia terbaru (Nongsa-Changi, Echo)?\n\n(Tinggi kotak akan menyesuaikan panjang teks yang Anda ketik.)",
    key="p"
)
# Keep session in sync for next render height calculation
st.session_state["prompt_history"] = prompt

# ============================================================
# MAIN BUTTON
# ============================================================

if st.button("🚀 Tanya AI", type="primary", use_container_width=True):

    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu." if st.session_state.get("ui_language") == "id" else "⚠️ Please enter a question first.")
        st.stop()

    lang = st.session_state.get("ui_language", "id")

    ok_log, msg_log = log_access(
        feature="query_start",
        prompt=prompt.strip(),
        model=model,
        files_count=len(uploaded_files) if uploaded_files else 0,
        web_search=st.session_state["enable_web_search"],
        specialized=st.session_state["enable_specialized_apac"],
        language=lang
    )
    st.session_state["last_log_status"] = f"{'✅' if ok_log else '❌'} {msg_log}"

    # ---------- System prompt by language ----------
    if lang == "en":
        system_prompt = """
You are Telco Digital AI, a professional assistant in Telecommunications, ICT, 
Digital Transformation, Data Center, Fiber Optic, Submarine Cable, 5G, Satellite, 
Regulation, Project & Risk Management.

MANDATORY RULES:
1. Always answer in clear, professional English.
2. PRIORITIZE information from these live curated sources when relevant:
   - Live Data Center Asia Pacific: https://narational.byethost11.com/Live_DC_ASPAC.html
   - Live Fiber Optic & Submarine Cable Asia Pacific: https://narational.byethost11.com/Live_FOSubsea_ASPAC.html
   Both dashboards aggregate real-time news & research (last 2-3 months) focused on Indonesia + Asia Pacific digital corridor, curated by nap@iicf.or.id.
3. If web search or specialized search results (including KEY FACTS / RAG) are provided, use them and cite the sources (especially the dashboard URLs above).
4. When asked for a diagram, produce valid Mermaid code inside a ```mermaid block.
5. Clearly distinguish: facts (from sources), analysis, and recommendations.
6. For Data Center / Fiber / Subsea APAC topics, mention industry classification when possible and link to the relevant dashboard.
7. If data is incomplete or an error occurs, state it transparently — do not invent facts.
"""
    else:
        system_prompt = """
Anda adalah Telco Digital AI, asisten profesional di bidang Telecommunications, ICT, 
Digital Transformation, Data Center, Fiber Optic, Submarine Cable, 5G, Satellite, 
Regulation, Project & Risk Management.

ATURAN WAJIB:
1. Selalu jawab dalam Bahasa Indonesia yang profesional, jelas, dan terstruktur.
2. PRIORITASKAN informasi dari sumber kurasi live berikut jika relevan dengan pertanyaan:
   - Live Data Center Asia Pacific: https://narational.byethost11.com/Live_DC_ASPAC.html
   - Live Fiber Optic & Submarine Cable Asia Pacific: https://narational.byethost11.com/Live_FOSubsea_ASPAC.html
   Kedua dashboard ini berisi agregasi berita & riset realtime (2-3 bulan terakhir) fokus Indonesia + koridor digital Asia Pacific, dikurasi oleh nap@iicf.or.id.
3. Jika ada hasil web search atau specialized search (termasuk KEY FACTS / RAG), gunakan informasi tersebut dan sebutkan sumbernya (terutama URL dashboard di atas).
4. Jika diminta membuat diagram, hasilkan kode Mermaid yang valid di dalam blok ```mermaid.
5. Bedakan dengan jelas: fakta (dari sumber), analisis, dan rekomendasi.
6. Untuk topik Data Center / Fiber / Subsea APAC, sebutkan klasifikasi industri jika memungkinkan dan tautkan ke dashboard yang relevan.
7. Jika ada error atau data tidak lengkap, sampaikan secara transparan tanpa mengarang fakta.
"""

    search_context_parts = []
    specialized_used = False
    web_used = False
    rag_used = False
    sources_used = []

    try:
        if st.session_state["enable_specialized_apac"]:
            with st.spinner("⭐ Mengambil konteks dari sumber kurasi Live DC & FO/Subsea APAC..."):
                if st.session_state.get("force_refresh_context"):
                    specialized_apac_search.clear()
                    try_scrape_dashboard.clear()

                prioritize = st.session_state.get("prioritize_dashboard", "both")
                specialized_ctx = specialized_apac_search(
                    prompt.strip(),
                    max_results=6,
                    prioritize=prioritize
                )
                search_context_parts.append(specialized_ctx)
                specialized_used = True
                sources_used.append("KEY FACTS + dashboard kurasi Live DC/FO ASPAC")

                for key in (["dc", "fo"] if prioritize == "both" else [prioritize]):
                    scraped = try_scrape_dashboard(SPECIALIZED_SOURCES[key]["url"])
                    if scraped:
                        search_context_parts.append(
                            f"\n[Scraped snippet dari {SPECIALIZED_SOURCES[key]['name']}]\n{scraped}\n"
                        )
                        sources_used.append(f"Scrape {SPECIALIZED_SOURCES[key]['name']}")
    except Exception as e:
        search_context_parts.append(f"\n[Soft error specialized search: {str(e)}]\n")

    # Simple RAG
    try:
        if st.session_state.get("enable_rag", True):
            with st.spinner("🧠 Simple RAG: mengambil potongan KEY FACTS paling relevan..."):
                prioritize = st.session_state.get("prioritize_dashboard", "both")
                rag_ctx = simple_rag_retrieve(prompt.strip(), prioritize=prioritize, top_k=MAX_RAG_CHUNKS)
                if rag_ctx:
                    search_context_parts.append(rag_ctx)
                    rag_used = True
                    sources_used.append("Simple RAG (KEY FACTS)")
    except Exception as e:
        search_context_parts.append(f"\n[Soft error RAG: {str(e)}]\n")

    try:
        if st.session_state["enable_web_search"]:
            with st.spinner("🔍 Mencari informasi terbaru di web (umum)..."):
                general_ctx = web_search(prompt.strip(), max_results=4)
                search_context_parts.append("\n=== HASIL PENCARIAN UMUM ===\n" + general_ctx)
                web_used = True
                sources_used.append("DuckDuckGo web search")
    except Exception as e:
        search_context_parts.append(f"\n[Soft error web search: {str(e)}]\n")

    search_context = "\n".join(search_context_parts) if search_context_parts else ""

    user_content = []
    final_prompt = prompt.strip()
    if search_context:
        if lang == "en":
            final_prompt = f"""Here is relevant search context (priority: live curated APAC sources + KEY FACTS + RAG + general search):

{search_context}

---
User question:
{prompt.strip()}

Answer based on the information above + your knowledge.
- Prioritize and cite Live DC / Live FO-Subsea dashboard sources when relevant.
- Clearly distinguish facts, analysis, and recommendations.
"""
        else:
            final_prompt = f"""Berikut konteks pencarian yang relevan (prioritas sumber kurasi live APAC + KEY FACTS + RAG + pencarian umum):

{search_context}

---
Pertanyaan pengguna:
{prompt.strip()}

Jawab berdasarkan informasi di atas + pengetahuan Anda. 
- Prioritaskan dan sebutkan sumber dari dashboard Live DC / Live FO-Subsea jika relevan.
- Bedakan fakta, analisis, dan rekomendasi.
"""

    user_content.append({"type": "text", "text": final_prompt})

    is_vision = choice["type"] in ("vision", "multimodal")
    if uploaded_files:
        for f in uploaded_files:
            parts = process_uploaded_file(f, is_vision_capable=is_vision)
            user_content.extend(parts)

    messages_multimodal = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content if len(user_content) > 1 else final_prompt}
    ]
    messages_text = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": final_prompt}
    ]

    def _call_provider(url: str, api_key: str, model_name: str, msgs: list, extra_headers: dict = None) -> Tuple[str, str]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        payload = {"model": model_name, "messages": msgs, "max_tokens": 8192, "temperature": 0.7}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=180)
            if r.status_code in (400, 401, 402, 403, 404, 429):
                return "", f"HTTP_{r.status_code}"
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return content, ""
        except requests.exceptions.Timeout:
            return "", "TIMEOUT"
        except Exception as e:
            return "", str(e)[:150]

    hf_token = _get_secret("HF_TOKEN", "hf_token")
    groq_key = _get_secret("GROQ_API_KEY", "groq_api_key", "GROQ_KEY")
    or_key = _get_secret("OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY")

    primary = choice
    chain = [primary] + [c for c in PROVIDER_CHOICES if c["id"] != primary["id"]]

    answer = ""
    used_provider = primary["provider"]
    used_model = primary["model"]
    fail_log = []

    try:
        with st.spinner("🤖 AI sedang memproses..." if lang == "id" else "🤖 AI is processing..."):
            for c in chain:
                prov = c["provider"]
                mname = c["model"]

                if prov == "groq":
                    if not groq_key:
                        fail_log.append(f"{c['label']}: no GROQ_API_KEY")
                        continue
                    msgs = messages_multimodal if (c["type"] in ("vision", "multimodal") and uploaded_files) else messages_text
                    ans, err = _call_provider(API_URL_GROQ, groq_key, mname, msgs)
                elif prov == "openrouter":
                    if not or_key:
                        fail_log.append(f"{c['label']}: no OPENROUTER_API_KEY")
                        continue
                    msgs = messages_multimodal if (c["type"] in ("vision", "multimodal") and uploaded_files) else messages_text
                    ans, err = _call_provider(
                        API_URL_OPENROUTER, or_key, mname, msgs,
                        extra_headers={
                            "HTTP-Referer": "https://telco-digital-ai.streamlit.app",
                            "X-Title": "ID Telco Digital AI",
                        },
                    )
                else:  # hf
                    if not hf_token:
                        fail_log.append(f"{c['label']}: no HF_TOKEN")
                        continue
                    msgs = messages_multimodal if c["type"] in ("vision", "multimodal") else messages_text
                    ans, err = _call_provider(API_URL_HF, hf_token, mname, msgs)

                if ans:
                    answer = ans
                    used_provider = prov
                    used_model = mname
                    if c["id"] != primary["id"]:
                        st.caption(f"↪️ Auto-switch ke **{c['label']}** (pilihan awal tidak tersedia).")
                    break
                else:
                    fail_log.append(f"{c['label']}: {err or 'unknown'}")

        if not answer:
            st.error("❌ Semua provider/model gagal. Ringkasan:" if lang == "id" else "❌ All providers/models failed. Summary:")
            for line in fail_log[:12]:
                st.text(f"  • {line}")
            if not groq_key and not or_key and not hf_token:
                st.warning("Tidak ada API key yang terbaca di Secrets." if lang == "id" else "No API keys found in Secrets.")
            log_access(feature="error", prompt=prompt.strip(), model=model, error_note="; ".join(fail_log)[:300], language=lang)
            st.stop()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            _ip, _country = safe_get_ip_and_country()
        except Exception:
            _ip, _country = "unknown", "unknown"
        try:
            _origin = get_external_referrer()
        except Exception:
            _origin = ""

        display_model = f"{used_model}" + (f" via {used_provider}" if used_provider != "huggingface" else "")

        st.session_state["chat_history"].append({
            "role": "user",
            "content": prompt.strip(),
            "model": display_model,
            "time": now,
            "ip": _ip,
            "country": _country,
            "origin_url": _origin,
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "model": display_model,
            "time": now,
            "ip": _ip,
            "country": _country,
            "origin_url": _origin,
        })

        st.markdown("### ✅ Jawaban AI" if lang == "id" else "### ✅ AI Answer")
        if used_provider != "huggingface":
            st.caption(f"Provider: **{used_provider}** · Model: `{used_model}`")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        if sources_used:
            unique_sources = list(dict.fromkeys(sources_used))
            src_html = " · ".join(unique_sources)
            label = "📚 Sumber yang digunakan:" if lang == "id" else "📚 Sources used:"
            st.markdown(
                f'<div class="source-summary"><strong>{label}</strong> {src_html}</div>',
                unsafe_allow_html=True
            )

        try:
            extract_and_render_mermaid(answer)
        except Exception:
            pass

        if search_context:
            with st.expander("🔍 Lihat konteks pencarian (Specialized + RAG + KEY FACTS + Umum)" if lang == "id"
                             else "🔍 View search context (Specialized + RAG + KEY FACTS + General)"):
                st.markdown(search_context)

        st.session_state["prompt_history"] = prompt
        st.query_params["prompt"] = prompt
        st.query_params["model"] = model
        st.query_params["lang"] = lang
        st.query_params["provider"] = used_provider if used_provider in ("hf", "groq", "openrouter") else st.session_state.get("preferred_provider", "auto")

        ok_log2, msg_log2 = log_access(
            feature="query_success",
            prompt=prompt.strip(),
            answer=answer,
            model=display_model,
            files_count=len(uploaded_files) if uploaded_files else 0,
            web_search=web_used,
            specialized=specialized_used,
            language=lang
        )
        st.session_state["last_log_status"] = f"{'✅' if ok_log2 else '❌'} {msg_log2}"

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}" if lang == "id" else f"❌ An error occurred: {str(e)}")
        if st.session_state.get("debug_mode"):
            st.code(traceback.format_exc())
        log_access(feature="error", prompt=prompt.strip(), model=model, error_note=str(e)[:300], language=lang)

# ============================================================
# RIWAYAT + EXPORT
# ============================================================

if st.session_state["chat_history"]:
    st.markdown("---")
    st.subheader("📜 Riwayat Percakapan (Session ini)" if st.session_state.get("ui_language") == "id" else "📜 Conversation History (this session)")

    history_md = "# Riwayat Chat - ID Telco Digital AI\n\n"
    history_plain = ""
    history_wa = ""

    for item in st.session_state["chat_history"]:
        role = item["role"]
        ip = item.get("ip", "unknown")
        country = item.get("country", "unknown")
        origin = item.get("origin_url", "") or ""
        session_id = st.session_state.get("session_id", "-")

        if role == "user":
            has_public_ip = (
                ip not in ("unknown", "internal", "", None)
                and not str(ip).startswith(("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                                            "172.2", "172.3", "192.168.", "127."))
            )
            if has_public_ip:
                role_label = f"👤 [{ip}] · [{country}]"
                role_plain = f"[{ip}] [{country}]"
            else:
                role_label = f"👤 Session `{session_id}`"
                role_plain = f"Session {session_id}"
            extra_info = f"🔗 Dari: `{origin}`" if origin else ""
        else:
            role_label = "🤖 AI"
            role_plain = "AI"
            extra_info = ""

        st.markdown(f"**{role_label}**  ·  `{item.get('time', '')}`  ·  `{item.get('model', '')}`")
        if extra_info:
            st.caption(extra_info)

        with st.container(border=True):
            st.markdown(item.get("content", ""))

        st.markdown("")

        history_md += f"**{role_label}** ({item['time']}) — `{item['model']}`\n"
        if origin:
            history_md += f"Dari: {origin}\n"
        history_md += f"\n{item['content']}\n\n---\n\n"
        history_plain += f"[{item['time']}] {role_plain} ({item['model']}):\n"
        if origin:
            history_plain += f"Dari: {origin}\n"
        history_plain += f"{item['content']}\n\n"
        history_wa += f"*{role_plain}* ({item['time']})\n{item['content']}\n\n"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button("⬇️ Markdown", history_md, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md", "text/markdown", use_container_width=True)
    with col2:
        st.download_button("⬇️ Plain Text", history_plain, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col3:
        st.download_button("⬇️ WhatsApp", history_wa, f"chat_wa_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col4:
        try:
            pdf_bytes = create_pdf_from_history(st.session_state["chat_history"])
            if pdf_bytes and isinstance(pdf_bytes, (bytes, bytearray)) and len(pdf_bytes) > 100:
                st.download_button(
                    label="⬇️ PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_pdf_download"
                )
            else:
                st.button("⬇️ PDF (tidak tersedia)", disabled=True, use_container_width=True, key="btn_pdf_disabled")
        except Exception:
            st.button("⬇️ PDF (error)", disabled=True, use_container_width=True, key="btn_pdf_error")
    with col5:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["chat_history"] = []
            log_access(feature="clear_history", language=st.session_state.get("ui_language", "id"))
            st.rerun()

st.markdown("---")
st.caption(
    f"ID Telco Digital AI v{APP_VERSION} • Session: {st.session_state.get('session_id', '-')} • "
    "Logging behavior ke Supabase (jika secrets dikonfigurasi). "
    "Sumber kurasi: Live DC & FO/Subsea ASPAC oleh nap@iicf.or.id • Simple RAG + multi-language."
)
