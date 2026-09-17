"""
ID Telco Digital AI Assistant - v5.1.0 (Production-Ready)
==========================================================
Refactor menyeluruh dari v5.0.0-rc1. Memperbaiki:
- 3 bug blocking (pandas import, accept_multiple, context guard flow)
- Regresi dari v4.3/v4.4 (KEY FACTS hardcoded, multi-bahasa, PDF extraction, model availability)
- Keamanan (admin password dari secrets, Mermaid strict, HTML escape)
- Arsitektur context guard (state machine yang benar, tidak butuh klik 2x)

Ringkasan fitur:
1. Context Guard dengan RAG filter → konfirmasi user → eksekusi sekali klik
2. KEY FACTS hardcoded + auto-fetch RSS (fallback bertingkat)
3. Multi-bahasa (ID/EN/MS/Auto)
4. Ekstraksi PDF (pypdf) + auto-summary untuk dokumen panjang
5. Feedback RLHF dengan korelasi query
6. Admin analytics internal (PIN dari secrets)
7. Rate limiting per session
8. Deteksi ketersediaan model (cached 10 menit)
9. Simple RAG (TF-IDF fallback keyword-overlap)
10. Cascade 13 model dengan timeout per model
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────
import base64
import hashlib
import html
import io
import json
import re
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY (wajib)
# ─────────────────────────────────────────────────────────────
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCIES — status tracking eksplisit
# ─────────────────────────────────────────────────────────────
DDG_AVAILABLE, DDG_ERROR = False, ""
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    DDG_ERROR = str(e)[:120]

BS4_AVAILABLE, BS4_ERROR = False, ""
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    BS4_ERROR = str(e)[:120]

SUPABASE_AVAILABLE, SUPABASE_ERROR = False, ""
try:
    from supabase import Client, create_client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    SUPABASE_ERROR = str(e)[:120]

FPDF_AVAILABLE, FPDF_ERROR = False, ""
try:
    from fpdf import FPDF  # type: ignore
    FPDF_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    FPDF_ERROR = str(e)[:120]

PYPDF_AVAILABLE, PYPDF_ERROR = False, ""
try:
    from pypdf import PdfReader  # type: ignore
    PYPDF_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    PYPDF_ERROR = str(e)[:120]

PANDAS_AVAILABLE, PANDAS_ERROR = False, ""
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    PANDAS_ERROR = str(e)[:120]

SKLEARN_AVAILABLE, SKLEARN_ERROR = False, ""
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    SKLEARN_ERROR = str(e)[:120]


# ═════════════════════════════════════════════════════════════
# KONSTANTA
# ═════════════════════════════════════════════════════════════
APP_VERSION = "5.1.0"

API_URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
API_URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
API_URL_HF = "https://router.huggingface.co/v1/chat/completions"

DEFAULT_SUPABASE_TABLE = "telcodigitalai_logs"
MAX_LOG_PROMPT_LEN = 800
MAX_LOG_ANSWER_LEN = 1500

# Context guard
MAX_CONTEXT_CHARS = 14000       # ≈ 4000 token
RAG_TOP_K = 8
RAG_CHUNK_CHARS = 900
SUMMARY_TARGET_CHARS = 6000

# Rate limit per session
RATE_LIMIT_WINDOW_SEC = 600     # 10 menit
RATE_LIMIT_MAX_QUERIES = 25

# LLM cascade
LLM_TIMEOUT_PER_MODEL = 60      # detik
LLM_MAX_ATTEMPTS = 6            # batasi chain agar tidak 13×60s

# ─────────────────────────────────────────────────────────────
# LOGO SVG INLINE
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ID Telco Digital AI Assistant",
    page_icon=LOGO_DATA_URI,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.main .block-container {
    padding-top: 1.5rem; padding-bottom: 3rem;
    padding-left: 5%; padding-right: 5%; max-width: 1400px;
}
.logo-header { display:flex; align-items:center; gap:16px; margin-bottom:0.8rem; }
.logo-header img { height:58px; width:auto; filter:drop-shadow(0 3px 8px rgba(0,0,0,0.25)); }
.app-title { font-size:2.3rem; font-weight:700; line-height:1.2; margin-bottom:0.15rem; }
.app-caption { font-size:0.95rem; opacity:0.75; margin-bottom:1.2rem; }
.stButton > button { width:100%; min-height:44px; font-weight:600; border-radius:8px; }
.answer-container {
    margin-top:1.5rem; padding:1.25rem; border-radius:10px;
    border:1px solid rgba(128,128,128,0.25);
}
.source-summary {
    background:#f0f7ff; border-left:4px solid #1e88e5;
    padding:0.75rem 1rem; margin:1rem 0; border-radius:0 6px 6px 0; font-size:0.92rem;
}
.context-warning {
    background:#fff8e1; border-left:4px solid #ffb300;
    padding:0.9rem 1.1rem; margin:1rem 0; border-radius:0 6px 6px 0; font-size:0.92rem;
}
.history-box {
    background:#012456 !important; border:2px solid #00bfff !important;
    border-radius:8px; padding:1rem 1.2rem; margin-top:0.6rem;
    font-family:'Consolas','Courier New',monospace !important;
    font-size:0.9rem; color:#ffff00 !important;
}
.mermaid { background:#f8f9fa; padding:1rem; border-radius:8px; margin:1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═════════════════════════════════════════════════════════════
PROVIDER_CHOICES: List[Dict[str, Any]] = [
    # ─── Groq ────────────────────────────────────────────────
    {"id": "groq:openai/gpt-oss-20b", "provider": "groq",
     "model": "openai/gpt-oss-20b", "label": "Groq · gpt-oss-20b",
     "desc": "Aktif · cepat (default)",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "groq:openai/gpt-oss-120b", "provider": "groq",
     "model": "openai/gpt-oss-120b", "label": "Groq · gpt-oss-120b",
     "desc": "Aktif · lebih kuat",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "groq:qwen/qwen3.6-27b", "provider": "groq",
     "model": "qwen/qwen3.6-27b", "label": "Groq · qwen3.6-27b (Vision)",
     "desc": "Aktif · Multimodal · Reasoning",
     "type": "multimodal", "max_files": 5,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "groq:qwen/qwen3.8-27b", "provider": "groq",
     "model": "qwen/qwen3.8-27b", "label": "Groq · qwen3.8-27b (Vision)",
     "desc": "Aktif · Multimodal · Thinking",
     "type": "multimodal", "max_files": 3,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    # ─── OpenRouter FREE ─────────────────────────────────────
    {"id": "openrouter:meta-llama/llama-3.3-70b-instruct:free",
     "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free",
     "label": "OpenRouter · llama-3.3-70b:free", "desc": "FREE · Stabil & kuat",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:openai/gpt-oss-20b:free", "provider": "openrouter",
     "model": "openai/gpt-oss-20b:free",
     "label": "OpenRouter · gpt-oss-20b:free", "desc": "FREE · General purpose",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:google/gemma-4-31b-it:free", "provider": "openrouter",
     "model": "google/gemma-4-31b-it:free",
     "label": "OpenRouter · gemma-4-31b:free (Vision)", "desc": "FREE · Multimodal kuat",
     "type": "multimodal", "max_files": 4,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "provider": "openrouter",
     "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "label": "OpenRouter · nemotron-omni:free (Vision)",
     "desc": "FREE · Text+Image+Video",
     "type": "multimodal", "max_files": 3,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    # ─── Hugging Face ────────────────────────────────────────
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
     "type": "vision", "max_files": 3,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
]
CHOICE_BY_ID: Dict[str, Dict[str, Any]] = {c["id"]: c for c in PROVIDER_CHOICES}
DEFAULT_CHOICE_ID = "groq:openai/gpt-oss-20b"

# ─────────────────────────────────────────────────────────────
# MULTI-LANGUAGE
# ─────────────────────────────────────────────────────────────
LANG_OPTIONS = {
    "id": "🇮🇩 Bahasa Indonesia (default)",
    "en": "🇬🇧 English",
    "ms": "🇲🇾 Bahasa Melayu",
    "auto": "🌐 Auto (ikuti bahasa pertanyaan)",
}
LANG_INSTRUCTIONS = {
    "id": "Selalu jawab dalam Bahasa Indonesia yang profesional, jelas, dan terstruktur.",
    "en": "Always answer in professional, clear, and well-structured English.",
    "ms": "Sentiasa jawab dalam Bahasa Melayu yang profesional, jelas, dan berstruktur.",
    "auto": ("Jawab dalam bahasa yang sama dengan pertanyaan pengguna. "
             "Jika ambigu, gunakan Bahasa Indonesia."),
}


# ═════════════════════════════════════════════════════════════
# SPECIALIZED SOURCES + KEY FACTS FALLBACK
# ═════════════════════════════════════════════════════════════
SPECIALIZED_SOURCES = {
    "dc": {
        "name": "Live Data Center Asia Pacific",
        "url": "https://narational.byethost11.com/Live_DC_ASPAC.html",
        "description": (
            "Curated Research Dashboard + Query Launcher • 7 Klasifikasi Industri DC APAC. "
            "Fokus Indonesia + koridor digital Asia Pacific. Agregasi berita & riset realtime "
            "periode 2-3 bulan terakhir. Oleh nap@iicf.or.id."
        ),
        "keywords": ["data center", "datacenter", "hyperscale", "AI campus", "GPU",
                     "Batam", "Nongsa", "BATIC", "grid readiness", "CoreWeave", "Firmus"],
    },
    "fo": {
        "name": "Live Fiber Optic & Submarine Cable Asia Pacific",
        "url": "https://narational.byethost11.com/Live_FOSubsea_ASPAC.html",
        "description": (
            "Curated Research Dashboard • 8 Klasifikasi Industri Fiber Optic & SubSEA APAC. "
            "Proyek kabel laut domestik, landing station, backbone terestrial, coherent optics. "
            "Oleh nap@iicf.or.id."
        ),
        "keywords": ["submarine cable", "subsea", "fiber optic", "kabel laut",
                     "landing station", "Nongsa-Changi", "Echo cable", "Telin",
                     "WaveLogic", "backbone"],
    },
}

KEY_FACTS_DC = """=== KEY FACTS LIVE DC ASPAC (snapshot ~Sep 2026) ===
- CoreWeave mengumumkan data center pertama di Asia-Pacific di Indonesia (target online 2028).
- Firmus Technologies + NVIDIA: kampus AI Factory 360 MW di Batam (bersama DayOne), hingga ~170.000 GPU, target live Q1 2027.
- DayOne mengembangkan kapasitas signifikan di Batam; Batam/Nongsa menjadi magnet investor.
- BATIC 2026 (Bali) menekankan infrastruktur digital & AI sebagai pendorong pertumbuhan.
- Forum Grid Readiness membahas kesiapan jaringan listrik Indonesia untuk DC hyperscale & AI.
- Klasifikasi dashboard: A Business, B AI/HPC, C Power, D Water, E Regulatory, F Industrial, G Sovereign.
- Periode fokus: ~16 Jul – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
[CATATAN: KEY FACTS ini snapshot; verifikasi angka spesifik langsung ke dashboard.]
"""

KEY_FACTS_FO = """=== KEY FACTS LIVE FO & SUBSEA ASPAC (snapshot ~Sep 2026) ===
- Nongsa-Changi Cable (NCC) mendarat ~20 Juli 2026 di Nongsa Digital Park, Batam (Telin + BW Digital). Panjang ~50 km, 24 fiber pairs, kapasitas >1,6 Pbps.
- Sistem Echo (Google & Meta) mendarat di Singapore; arsitektur melibatkan Indonesia.
- ION Cable System dan coherent optics (Ciena WaveLogic) memperkuat backbone Jakarta–Singapore + Sumatra.
- Telkom via Telin memperkuat ambisi Indonesia sebagai Hub Internet Asia Pasifik.
- Pemerintah menyiapkan landing station baru sesuai regulasi.
- Periode fokus: ~16 Jun – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
[CATATAN: KEY FACTS ini snapshot; verifikasi angka spesifik langsung ke dashboard.]
"""


# ═════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════
def _is_private_ip(ip: str) -> bool:
    """Cek apakah IP privat, loopback, link-local, atau CGNAT."""
    if not ip or ip == "unknown":
        return True
    try:
        parts = [int(p) for p in ip.split(".")]
        if len(parts) != 4:
            return True
        a, b = parts[0], parts[1]
        if a == 10 or a == 127 or a == 0 or a >= 224:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
        if a == 169 and b == 254:  # link-local
            return True
        if a == 100 and 64 <= b <= 127:  # CGNAT
            return True
        return False
    except Exception:  # noqa: BLE001
        return True


def safe_get_ip_and_country() -> Tuple[str, str]:
    """Best-effort IP + country. Soft-fail → ('unknown', 'unknown')."""
    ip, country = "unknown", "unknown"
    try:
        headers: Dict[str, str] = {}
        if hasattr(st, "context") and st.context:
            try:
                headers = dict(st.context.headers or {})
            except Exception:  # noqa: BLE001
                headers = {}
        candidates: List[str] = []
        for key in ("X-Forwarded-For", "x-forwarded-for",
                    "X-Real-IP", "x-real-ip",
                    "CF-Connecting-IP", "cf-connecting-ip",
                    "True-Client-IP", "true-client-ip"):
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
                        country = (data.get("country_name")
                                   or data.get("country_code")
                                   or country)
            except Exception:  # noqa: BLE001
                pass
        if ip != "unknown" and not _is_private_ip(ip) and country == "unknown":
            try:
                r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    country = (data.get("country_name")
                               or data.get("country_code")
                               or country)
            except Exception:  # noqa: BLE001
                pass
        if _is_private_ip(ip):
            ip = "internal"
            if country == "unknown":
                country = "Streamlit Cloud"
    except Exception:  # noqa: BLE001
        pass
    return ip, country


def get_external_referrer() -> str:
    """Referer eksternal (bukan domain app sendiri). Empty string jika tidak ada."""
    try:
        headers: Dict[str, str] = {}
        if hasattr(st, "context") and st.context:
            try:
                headers = st.context.headers or {}
            except Exception:  # noqa: BLE001
                headers = {}
        referer = str(headers.get("Referer")
                      or headers.get("referer")
                      or headers.get("Referrer")
                      or headers.get("referrer")
                      or "").strip()
        if not referer:
            return ""
        own_domains = ("telco-digital-ai.streamlit.app", "localhost",
                       "127.0.0.1", "streamlit.app")
        low = referer.lower()
        if any(d in low for d in own_domains):
            return ""
        return referer[:500]
    except Exception:  # noqa: BLE001
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


def _get_secret(*names: str) -> Optional[str]:
    for n in names:
        try:
            v = st.secrets.get(n)
            if v is None:
                continue
            s = str(v).strip().strip('"').strip("'")
            if s and s not in ("None", "null"):
                return s
        except Exception:  # noqa: BLE001
            continue
    return None


def get_api_keys() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    hf = _get_secret("HF_TOKEN", "hf_token")
    groq = _get_secret("GROQ_API_KEY", "groq_api_key", "GROQ_KEY")
    ork = _get_secret("OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY")
    return hf, groq, ork


def estimate_tokens(text: str) -> int:
    """Estimasi kasar: ~3.5 karakter/token untuk campuran ID/EN."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


# ═════════════════════════════════════════════════════════════
# RATE LIMITING
# ═════════════════════════════════════════════════════════════
def _rate_limit_check() -> Tuple[bool, str]:
    """Return (allowed, message). Sliding window per session."""
    now = time.time()
    timestamps: deque = st.session_state.get("_rate_ts", deque())
    # Buang timestamp di luar window
    while timestamps and (now - timestamps[0]) > RATE_LIMIT_WINDOW_SEC:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT_MAX_QUERIES:
        wait = int(RATE_LIMIT_WINDOW_SEC - (now - timestamps[0])) + 1
        st.session_state["_rate_ts"] = timestamps
        return False, (f"Batas {RATE_LIMIT_MAX_QUERIES} query / "
                       f"{RATE_LIMIT_WINDOW_SEC // 60} menit tercapai. "
                       f"Coba lagi dalam ~{wait} detik.")
    timestamps.append(now)
    st.session_state["_rate_ts"] = timestamps
    return True, ""


# ═════════════════════════════════════════════════════════════
# SUPABASE LOGGING
# ═════════════════════════════════════════════════════════════
def get_supabase_table_name() -> str:
    return _get_secret("SUPABASE_TABLE") or DEFAULT_SUPABASE_TABLE


def get_supabase_client(admin: bool = False) -> Optional["Client"]:
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = _get_secret("SUPABASE_URL")
        if admin:
            key = (_get_secret("SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
                   or _get_secret("SUPABASE_KEY", "SUPABASE_ANON_KEY"))
        else:
            key = _get_secret("SUPABASE_KEY", "SUPABASE_ANON_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:  # noqa: BLE001
        if st.session_state.get("debug_mode"):
            st.warning(f"[Supabase client] {e}")
        return None


def append_behavior_log(row: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        client = get_supabase_client()
        if client is None:
            reason = ("Supabase library missing" if not SUPABASE_AVAILABLE
                      else "SUPABASE_URL / SUPABASE_KEY tidak terbaca")
            return False, reason
        table = get_supabase_table_name()
        clean = {k: ("" if v is None else str(v)) for k, v in row.items()}
        client.table(table).insert(clean).execute()
        return True, f"OK → tabel `{table}`"
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:300]
        if st.session_state.get("debug_mode"):
            st.warning(f"[Log] Gagal tulis: {msg}")
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
    feedback: str = "",
) -> Tuple[bool, str]:
    try:
        ip, country = safe_get_ip_and_country()
        ua = ""
        try:
            headers = st.context.headers if hasattr(st, "context") else {}
            ua = headers.get("User-Agent", "") or headers.get("user-agent", "")
        except Exception:  # noqa: BLE001
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
            "feedback": feedback,
        }
        return append_behavior_log(row)
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:200]


# ═════════════════════════════════════════════════════════════
# MODEL AVAILABILITY (cached)
# ═════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def check_model_availability(provider: str, model: str, api_key: str) -> Tuple[bool, str]:
    """Probe ringan. Return (ok, note)."""
    if not api_key:
        return False, "no API key"
    try:
        if provider == "groq":
            url = API_URL_GROQ
            headers = {"Authorization": f"Bearer {api_key}",
                       "Content-Type": "application/json"}
        elif provider == "openrouter":
            url = API_URL_OPENROUTER
            headers = {"Authorization": f"Bearer {api_key}",
                       "Content-Type": "application/json",
                       "HTTP-Referer": "https://telco-digital-ai.streamlit.app",
                       "X-Title": "ID Telco Digital AI"}
        else:
            return True, "hf (not probed)"
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "ping"}],
                   "max_tokens": 1}
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        if r.status_code in (200, 201):
            return True, "ok"
        if r.status_code == 429:
            return True, "rate-limited (masih terdaftar)"
        if r.status_code in (401, 403):
            return False, "auth"
        if r.status_code == 402:
            return False, "payment / free quota"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:80]


# ═════════════════════════════════════════════════════════════
# SEARCH & SCRAPE (dengan caching)
# ═════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def ddg_fetch(query: str, max_results: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    if not DDG_AVAILABLE:
        return [], f"duckduckgo-search tidak tersedia ({DDG_ERROR})"
    try:
        with DDGS() as ddgs:
            res = list(ddgs.text(query, max_results=max_results))
        return res, ""
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:150]
        if "429" in msg or "Ratelimit" in msg.lower():
            return [], "DuckDuckGo rate-limit. Coba beberapa menit lagi."
        return [], msg


@st.cache_data(ttl=300, show_spinner=False)
def scrape_dashboard(url: str, max_chars: int = 2500) -> Tuple[str, str, str]:
    """Return (text, status, detail). Status: ok|blocked|js_required|empty|error|library_missing."""
    if not BS4_AVAILABLE:
        return "", "library_missing", "beautifulsoup4 tidak terinstall"
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,id;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        low = (r.text or "")[:4000].lower()
        if (r.status_code in (403, 503)
                or "cloudflare" in low
                or "just a moment" in low
                or "checking your browser" in low
                or "access denied" in low):
            return "", "blocked", f"HTTP {r.status_code} — proteksi anti-bot"
        if r.status_code != 200:
            return "", "error", f"HTTP {r.status_code}"
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if "requires javascript" in text.lower()[:800] or "enable javascript" in low:
            return "", "js_required", "halaman butuh JavaScript"
        if len(text) < 200:
            return "", "empty", f"konten hanya {len(text)} karakter"
        return text[:max_chars], "ok", ""
    except requests.exceptions.Timeout:
        return "", "error", "timeout"
    except Exception as e:  # noqa: BLE001
        return "", "error", str(e)[:120]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rss_headlines(category: str) -> List[str]:
    """Google News RSS — ambil 5 judul terbaru. Soft-fail → []."""
    rss_map = {
        "dc": ("https://news.google.com/rss/search?"
               "q=Data+Center+Indonesia+Batam+OR+CoreWeave+OR+Firmus&hl=id&gl=ID&ceid=ID:id"),
        "fo": ("https://news.google.com/rss/search?"
               "q=Submarine+Cable+Indonesia+OR+Nongsa+OR+Echo+Cable&hl=id&gl=ID&ceid=ID:id"),
    }
    url = rss_map.get(category)
    if not url:
        return []
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item")[:5]:
            t = item.findtext("title")
            if t:
                titles.append(t.strip())
        return titles
    except Exception:  # noqa: BLE001
        return []


# ═════════════════════════════════════════════════════════════
# SIMPLE RAG (TF-IDF → fallback keyword overlap)
# ═════════════════════════════════════════════════════════════
_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "ini", "itu",
    "apa", "bagaimana", "mengapa", "siapa", "kapan", "dimana", "jelaskan",
    "berikan", "adalah", "atau", "juga", "akan", "bisa", "dapat", "the", "and",
    "or", "for", "with", "what", "how", "please", "you", "are", "can", "to",
    "of", "in", "on", "data",
}


def _split_paragraphs(text: str) -> List[str]:
    """Split dengan toleransi separator ganda."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _score_paragraphs_tfidf(query: str, paras: List[str]) -> List[float]:
    if not SKLEARN_AVAILABLE or len(paras) < 2:
        return [0.0] * len(paras)
    try:
        vec = TfidfVectorizer(stop_words=None, min_df=1)
        matrix = vec.fit_transform(paras + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        return [float(s) for s in scores]
    except Exception:  # noqa: BLE001
        return [0.0] * len(paras)


def _score_paragraphs_keyword(query: str, paras: List[str]) -> List[float]:
    qtok = {w for w in re.findall(r"[a-z0-9\-]{3,}", query.lower())
            if w not in _STOPWORDS}
    if not qtok:
        return [0.0] * len(paras)
    scores = []
    for p in paras:
        ptok = set(re.findall(r"[a-z0-9\-]{3,}", p.lower()))
        scores.append(len(qtok & ptok) / max(1, len(qtok)))
    return scores


def simple_rag_filter(query: str, context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Filter konteks berdasarkan relevansi. Mempertahankan urutan asli paragraf.
    Jika tidak ada paragraf relevan → potong mentah.
    """
    if not context:
        return ""
    if len(context) <= max_chars:
        return context

    paras = _split_paragraphs(context)
    if not paras:
        return context[:max_chars] + "\n\n[...dipotong karena batas memori...]"

    scores = _score_paragraphs_tfidf(query, paras)
    if all(s == 0.0 for s in scores):
        scores = _score_paragraphs_keyword(query, paras)

    # Seleksi top-K berdasarkan skor, tapi pertahankan urutan asli
    indexed = list(enumerate(zip(paras, scores)))
    indexed.sort(key=lambda x: x[1][1], reverse=True)

    selected_idx: List[int] = []
    total_len = 0
    for idx, (p, sc) in indexed:
        if total_len + len(p) + 2 > max_chars:
            break
        if sc <= 0.0 and selected_idx:
            # Sudah ada yang relevan; jangan tambahkan yang tidak relevan
            continue
        selected_idx.append(idx)
        total_len += len(p) + 2
        if len(selected_idx) >= RAG_TOP_K * 2:
            break

    if not selected_idx:
        return context[:max_chars] + "\n\n[...dipotong karena batas memori...]"

    selected_idx.sort()
    result = "\n\n".join(paras[i] for i in selected_idx)
    if len(result) < len(context):
        result += f"\n\n[...{len(context) - len(result)} karakter tidak relevan disaring oleh RAG...]"
    return result[:max_chars]


# ═════════════════════════════════════════════════════════════
# FILE HANDLING
# ═════════════════════════════════════════════════════════════
def _extract_pdf_text(data: bytes, max_chars: int = 12000) -> str:
    if not PYPDF_AVAILABLE:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        parts, total = [], 0
        for page in reader.pages[:30]:
            try:
                t = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                t = ""
            parts.append(t)
            total += len(t)
            if total > max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception:  # noqa: BLE001
        return ""


def prepare_file_parts(
    files: List[Tuple[str, str, bytes]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (parts_untuk_llm, catatan_untuk_ui)."""
    parts: List[Dict[str, Any]] = []
    notes: List[str] = []
    for name, mime, data in files:
        try:
            lname = name.lower()
            if lname.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                b64 = base64.b64encode(data).decode()
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime or 'image/png'};base64,{b64}"},
                })
                notes.append(f"🖼️ {name} dilampirkan untuk model Vision.")
            elif lname.endswith(".pdf"):
                txt = _extract_pdf_text(data)
                if txt and len(txt.strip()) > 100:
                    if len(txt) > 8000:
                        txt = txt[:8000] + "\n[...teks PDF dipotong...]"
                        notes.append(f"📄 {name}: teks PDF diekstrak & dipotong.")
                    else:
                        notes.append(f"📄 {name}: teks PDF diekstrak.")
                    parts.append({"type": "text",
                                  "text": f"\n\n[Isi PDF {name}]\n{txt}"})
                else:
                    notes.append(
                        f"📄 {name}: PDF tampaknya scan tanpa lapisan teks. "
                        f"Konversi ke PNG/JPG lalu upload agar dibaca model Vision."
                    )
            else:
                try:
                    txt = data.decode("utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    txt = ""
                if len(txt) > 8000:
                    txt = txt[:8000] + "\n[...dipotong...]"
                    notes.append(f"📃 {name}: file teks panjang dipotong.")
                else:
                    notes.append(f"📃 {name}: isi file dilampirkan.")
                if txt:
                    parts.append({"type": "text",
                                  "text": f"\n\n[Isi file {name}]\n{txt}"})
        except Exception as e:  # noqa: BLE001
            notes.append(f"⚠️ Gagal memproses {name}: {str(e)[:120]}")
    return parts, notes


# ═════════════════════════════════════════════════════════════
# LLM CASCADE
# ═════════════════════════════════════════════════════════════
def _call_provider(
    url: str,
    api_key: str,
    model_name: str,
    msgs: List[Dict[str, Any]],
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = LLM_TIMEOUT_PER_MODEL,
) -> Tuple[str, str]:
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    payload = {"model": model_name, "messages": msgs,
               "max_tokens": 8192, "temperature": 0.7}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code in (400, 401, 402, 403, 404, 429):
            return "", f"HTTP {r.status_code}"
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content or "", ""
    except requests.exceptions.Timeout:
        return "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return "", str(e)[:150]


def call_chain(
    messages_text: List[Dict[str, Any]],
    messages_multimodal: Optional[List[Dict[str, Any]]] = None,
    force_text_only: bool = False,
) -> Tuple[str, str, str, List[str], bool]:
    """
    Cascade provider. Return (answer, provider, model, fail_log, switched).
    Membatasi jumlah percobaan agar responsif.
    """
    hf_key, groq_key, or_key = get_api_keys()
    primary_id = st.session_state.get("model_selected", DEFAULT_CHOICE_ID)
    primary = CHOICE_BY_ID.get(primary_id) or CHOICE_BY_ID[DEFAULT_CHOICE_ID]
    chain = [primary] + [c for c in PROVIDER_CHOICES if c["id"] != primary["id"]]
    if force_text_only:
        chain = [c for c in chain if c.get("type") == "text"]

    fail_log: List[str] = []
    attempts = 0
    for c in chain:
        if attempts >= LLM_MAX_ATTEMPTS:
            break
        prov, mname = c["provider"], c["model"]
        mm_ok = c.get("type") in ("vision", "multimodal")
        msgs = (messages_multimodal
                if (mm_ok and messages_multimodal is not None)
                else messages_text)
        try:
            if prov == "groq":
                if not groq_key:
                    fail_log.append(f"{c['label']}: no GROQ_API_KEY")
                    continue
                attempts += 1
                ans, err = _call_provider(API_URL_GROQ, groq_key, mname, msgs)
            elif prov == "openrouter":
                if not or_key:
                    fail_log.append(f"{c['label']}: no OPENROUTER_API_KEY")
                    continue
                attempts += 1
                ans, err = _call_provider(
                    API_URL_OPENROUTER, or_key, mname, msgs,
                    extra_headers={
                        "HTTP-Referer": "https://telco-digital-ai.streamlit.app",
                        "X-Title": "ID Telco Digital AI",
                    },
                )
            else:  # hf
                if not hf_key:
                    fail_log.append(f"{c['label']}: no HF_TOKEN")
                    continue
                attempts += 1
                ans, err = _call_provider(API_URL_HF, hf_key, mname, msgs)
            if ans:
                switched = (c["id"] != primary_id)
                return ans, prov, mname, fail_log, switched
            fail_log.append(f"{c['label']}: {err or 'unknown'}")
        except Exception as e:  # noqa: BLE001
            fail_log.append(f"{c['label']}: {str(e)[:80]}")
    return "", "", "", fail_log, False


def llm_summarize(text: str, instruction: str) -> str:
    """Ringkas teks via cascade. Soft-fail → ''."""
    msgs = [
        {"role": "system",
         "content": ("Anda adalah peringkas dokumen teknis telekomunikasi. "
                     "Pertahankan angka, nama proyek, organisasi, tanggal, dan URL penting.")},
        {"role": "user", "content": instruction + "\n\n" + text[:24000]},
    ]
    ans, _, _, _, _ = call_chain(msgs, None, force_text_only=True)
    return ans or ""


# ═════════════════════════════════════════════════════════════
# BUILD SEARCH CONTEXT
# ═════════════════════════════════════════════════════════════
def build_search_context(
    prompt: str,
    prioritize: str,
    use_web: bool,
    use_spec: bool,
) -> Tuple[str, List[str], List[Tuple[str, str, str]], bool, bool]:
    """
    Return: (context, sources_used, notes[(name,status,detail)], web_used, spec_used).
    """
    parts: List[str] = []
    sources_used: List[str] = []
    notes: List[Tuple[str, str, str]] = []
    web_used = False
    spec_used = False

    if use_spec:
        spec_used = True
        parts.append(
            "=== SUMBER KURASI PRIMER (PRIORITAS TINGGI) ===\n"
            "Gunakan informasi terkini dari dashboard live berikut sebagai referensi utama "
            "untuk topik Data Center, Fiber Optic, dan Submarine Cable di Asia Pacific / Indonesia "
            "(periode 2-3 bulan terakhir).\n"
        )
        if prioritize in ("dc", "both"):
            src = SPECIALIZED_SOURCES["dc"]
            parts.append(f"1. **{src['name']}**\n   URL: {src['url']}\n   {src['description']}\n")
            parts.append(KEY_FACTS_DC)
            headlines = fetch_rss_headlines("dc")
            if headlines:
                parts.append("=== JUDUL BERITA TERKINI (RSS, DC) ===\n"
                             + "\n".join(f"- {h}" for h in headlines))
        if prioritize in ("fo", "both"):
            src = SPECIALIZED_SOURCES["fo"]
            parts.append(f"2. **{src['name']}**\n   URL: {src['url']}\n   {src['description']}\n")
            parts.append(KEY_FACTS_FO)
            headlines = fetch_rss_headlines("fo")
            if headlines:
                parts.append("=== JUDUL BERITA TERKINI (RSS, FO) ===\n"
                             + "\n".join(f"- {h}" for h in headlines))
        sources_used.append("KEY FACTS + dashboard kurasi Live DC/FO ASPAC")

        # Scrape dashboard (dengan status)
        for key in (["dc", "fo"] if prioritize == "both" else [prioritize]):
            src = SPECIALIZED_SOURCES[key]
            text, status, detail = scrape_dashboard(src["url"])
            if status == "ok":
                parts.append(f"[Scraped dari {src['name']}]\n{text}")
                sources_used.append(f"Scrape {src['name']}")
            else:
                notes.append((src["name"], status, detail))

        # DDG targeted
        site_q: List[str] = []
        if prioritize in ("dc", "both"):
            site_q.append(f'site:narational.byethost11.com ({prompt})')
            site_q.append('site:narational.byethost11.com (CoreWeave OR Firmus '
                          'OR "data center" OR Batam OR Nongsa)')
        if prioritize in ("fo", "both"):
            site_q.append('site:narational.byethost11.com ("Nongsa-Changi" OR Echo '
                          'OR "submarine cable" OR "kabel laut")')
        all_res: List[Dict[str, Any]] = []
        seen: set = set()
        for sq in site_q:
            res, err = ddg_fetch(sq, 3)
            if err:
                notes.append(("DuckDuckGo", "error", err))
            for r in res:
                href = r.get("href", "")
                if href and href not in seen:
                    seen.add(href)
                    all_res.append(r)
        if all_res:
            lines = ["=== HASIL PENCARIAN TARGETED ==="]
            for i, r in enumerate(all_res[:6], 1):
                tag = " [SUMBER KURASI]" if "narational" in r.get("href", "") else ""
                lines.append(f"{i}. {r.get('title', '')}{tag}\n"
                             f"{(r.get('body', '') or '')[:300]}...\n"
                             f"Sumber: {r.get('href', '')}")
            parts.append("\n".join(lines))
            sources_used.append("DuckDuckGo site-search (kurasi)")

    if use_web:
        res, err = ddg_fetch(prompt, 4)
        if err:
            notes.append(("DuckDuckGo (umum)", "error", err))
        if res:
            lines = [f"=== HASIL PENCARIAN UMUM: {prompt} ==="]
            for i, r in enumerate(res, 1):
                lines.append(f"{i}. {r.get('title', '')}\n"
                             f"{(r.get('body', '') or '')[:300]}...\n"
                             f"Sumber: {r.get('href', '')}")
            parts.append("\n".join(lines))
            sources_used.append("DuckDuckGo web search")
            web_used = True

    return "\n\n".join(parts), sources_used, notes, web_used, spec_used


# ═════════════════════════════════════════════════════════════
# PDF EXPORT
# ═════════════════════════════════════════════════════════════
def create_pdf_from_history(
    history: List[Dict[str, Any]],
    title: str = "Riwayat Chat - ID Telco Digital AI",
) -> Optional[bytes]:
    if not FPDF_AVAILABLE:
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                       f"| App v{APP_VERSION}", ln=True)
        pdf.ln(5)
        for item in history:
            role = "User" if item.get("role") == "user" else "AI"
            header = f"[{item.get('time', '')}] {role} ({item.get('model', '')})"
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, header, ln=True)
            pdf.set_font("Helvetica", "", 10)
            content = re.sub(r"[*_`#]", "", str(item.get("content", "")))[:3000]
            content = content.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)
        raw = pdf.output(dest="S")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        if isinstance(raw, str):
            return raw.encode("latin-1")
        return None
    except Exception:  # noqa: BLE001
        return None


# ═════════════════════════════════════════════════════════════
# MERMAID
# ═════════════════════════════════════════════════════════════
def extract_and_render_mermaid(text: str) -> None:
    pattern = r"```mermaid\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return
    st.markdown("#### Diagram Mermaid terdeteksi")
    for i, code in enumerate(matches):
        code = code.strip()
        if len(code) > 8000:
            st.warning(f"Diagram #{i+1} terlalu besar — tampilkan sebagai kode mentah.")
            st.code(code, language="mermaid")
            continue
        lines = code.count("\n") + 1
        height = min(900, max(280, 40 + lines * 28))
        safe_code = json.dumps(code)
        mermaid_html = f"""
        <div style="position:relative;">
          <button onclick="
            const el = document.getElementById('mermaid-wrap-{i}');
            if (!document.fullscreenElement) {{ el.requestFullscreen().catch(()=>{{}}); }}
            else {{ document.exitFullscreen(); }}
          " style="position:absolute;top:8px;right:8px;z-index:10;padding:4px 10px;
                   font-size:12px;cursor:pointer;border-radius:4px;border:1px solid #ccc;
                   background:#fff;">⛶ Fullscreen</button>
          <div id="mermaid-wrap-{i}" class="mermaid"
               style="min-height:{height}px;background:#f8f9fa;padding:1.2rem;
                      border-radius:8px;overflow:auto;"></div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
        (function() {{
          var code = {safe_code};
          try {{
            mermaid.initialize({{ startOnLoad: false, theme: 'default',
                                 securityLevel: 'strict' }});
            mermaid.render('mermaid-svg-{i}', code).then(function(res) {{
              document.getElementById('mermaid-wrap-{i}').innerHTML = res.svg;
            }}).catch(function(err) {{
              document.getElementById('mermaid-wrap-{i}').innerHTML =
                '<pre style="white-space:pre-wrap;color:#333;">' +
                code.replace(/[<>&]/g, function(c) {{
                  return {{'<':'&lt;','>':'&gt;','&':'&amp;'}}[c];
                }}) + '</pre><p style="color:#c00;font-size:0.85rem;">' +
                '(Render Mermaid gagal — menampilkan kode mentah)</p>';
            }});
          }} catch (e) {{
            document.getElementById('mermaid-wrap-{i}').innerHTML =
              '<pre style="white-space:pre-wrap;color:#333;">' + code + '</pre>';
          }}
        }})();
        </script>
        """
        st.components.v1.html(mermaid_html, height=height + 60, scrolling=True)
        with st.expander(f"Lihat kode Mermaid #{i+1}"):
            st.code(code, language="mermaid")


# ═════════════════════════════════════════════════════════════
# RENDER NOTES
# ═════════════════════════════════════════════════════════════
SCRAPE_EXPLAIN = {
    "blocked": ("🛡️ {name} dilindungi anti-bot (HTTP {detail}). "
                "Fallback aktif: DDG site-search + KEY FACTS — jawaban tetap akurat."),
    "js_required": ("🧩 {name} butuh JavaScript. Fallback: DDG site-search + KEY FACTS."),
    "empty": ("📭 {name} mengembalikan konten kosong dari server cloud. "
              "Fallback: DDG + KEY FACTS."),
    "error": ("⚠️ {name} gagal diakses ({detail}). Fallback: DDG + KEY FACTS."),
    "library_missing": ("📦 beautifulsoup4 tidak terinstall — scraping dilewati."),
}


def render_context_notes(notes: List[Tuple[str, str, str]]) -> None:
    for name, status, detail in notes:
        if status == "ok":
            continue
        tpl = SCRAPE_EXPLAIN.get(status)
        if tpl:
            st.markdown(
                f'<div class="context-warning">{tpl.format(name=name, detail=detail)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"ℹ️ {name}: {status} — {detail}")


# ═════════════════════════════════════════════════════════════
# ADMIN ANALYTICS
# ═════════════════════════════════════════════════════════════
def _admin_password_ok(pwd: str) -> bool:
    """Bandingkan hash SHA-256. Password diambil dari secrets."""
    expected_hash = _get_secret("ADMIN_PASSWORD_HASH")
    if not expected_hash:
        return False
    if len(expected_hash) != 64:
        return False
    try:
        return hashlib.sha256(pwd.encode()).hexdigest() == expected_hash.lower()
    except Exception:  # noqa: BLE001
        return False


def render_admin_dashboard() -> None:
    st.header("🔐 Admin Analytics Dashboard")

    if not _get_secret("ADMIN_PASSWORD_HASH"):
        st.error("ADMIN_PASSWORD_HASH belum diset di Streamlit Secrets. "
                 "Panel owner dinonaktifkan.")
        st.caption("Tambahkan ke secrets: `ADMIN_PASSWORD_HASH = \"<sha256hex>\"`")
        return

    if not st.session_state.get("is_admin"):
        pwd = st.text_input("Password Admin", type="password", key="admin_pwd")
        if st.button("Login"):
            if _admin_password_ok(pwd):
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Password salah.")
        return

    st.success("Login berhasil.")
    if st.button("Logout"):
        st.session_state["is_admin"] = False
        st.rerun()

    if not PANDAS_AVAILABLE:
        st.warning(f"pandas tidak tersedia: {PANDAS_ERROR}")
        return

    client = get_supabase_client(admin=True)
    if client is None:
        st.warning("Supabase (admin) tidak terkonfigurasi. "
                   "Set SUPABASE_SERVICE_KEY untuk SELECT analytics.")
        return

    try:
        resp = (client.table(get_supabase_table_name())
                .select("*").order("timestamp_utc", desc=True)
                .limit(1000).execute())
        rows = resp.data or []
        if not rows:
            st.info("Tabel log masih kosong.")
            return

        df = pd.DataFrame(rows)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

        q_success = df[df.get("feature") == "querysuccess"] if "feature" in df else df
        fb = df[df.get("feature") == "feedback"] if "feature" in df else pd.DataFrame()
        fb_up = int((fb.get("feedback", pd.Series(dtype=str)) == "positive").sum())
        fb_down = int((fb.get("feedback", pd.Series(dtype=str)) == "negative").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total baris", f"{len(df):,}")
        c2.metric("Query sukses", f"{len(q_success):,}")
        c3.metric("Sesi unik", f"{df['session_id'].nunique():,}" if "session_id" in df else "-")
        c4.metric("Feedback 👍 / 👎", f"{fb_up} / {fb_down}")
        ratio = (fb_up / max(1, fb_up + fb_down)) * 100
        c5.metric("Rating positif", f"{ratio:.0f}%")

        col1, col2 = st.columns(2)
        with col1:
            if "timestamp_utc" in q_success.columns and not q_success.empty:
                st.markdown("**Tren harian (query sukses)**")
                daily = q_success.set_index("timestamp_utc").resample("D").size()
                st.line_chart(daily)
            if "country" in df.columns:
                st.markdown("**Top negara**")
                st.bar_chart(df["country"].value_counts().head(8))
        with col2:
            if "model" in q_success.columns:
                st.markdown("**Top model**")
                st.bar_chart(q_success["model"].value_counts().head(8))
            if "feature" in df.columns:
                st.markdown("**Distribusi fitur**")
                st.bar_chart(df["feature"].value_counts().head(10))

        st.markdown("**50 log terbaru**")
        show_cols = [c for c in ["timestamp_utc", "feature", "country", "model",
                                 "prompt_snippet", "error_note"]
                     if c in df.columns]
        st.dataframe(df[show_cols].head(50), use_container_width=True)

    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal query analytics: {str(e)[:250]}")


# ═════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═════════════════════════════════════════════════════════════
_DEFAULTS: Dict[str, Any] = {
    "prompt_history": "",
    "model_selected": DEFAULT_CHOICE_ID,
    "chat_history": [],
    "enable_web_search": True,
    "enable_specialized_apac": True,
    "enable_rag": True,
    "debug_mode": False,
    "prioritize_dashboard": "both",
    "force_refresh_context": False,
    "answer_lang": "id",
    "last_log_status": "",
    "pending_query": None,      # {prompt, raw_context, files, sources, web_used, spec_used, notes}
    "is_admin": False,
    "_rate_ts": deque(),
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

generate_session_id()


# ═════════════════════════════════════════════════════════════
# DEEP LINKING
# ═════════════════════════════════════════════════════════════
_q = st.query_params
if _q.get("prompt"):
    st.session_state["prompt_history"] = _q["prompt"]
if _q.get("model"):
    mid = _q["model"]
    if mid in CHOICE_BY_ID:
        st.session_state["model_selected"] = mid
    else:
        for c in PROVIDER_CHOICES:
            if c["model"] == mid or c["id"].endswith(mid):
                st.session_state["model_selected"] = c["id"]
                break

# Admin routing
if _q.get("admin") == "1":
    render_admin_dashboard()
    st.stop()


# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════
st.markdown(
    f"""
<div class="logo-header">
  <img src="{LOGO_DATA_URI}" alt="Logo">
  <div>
    <div class="app-title">ID Telco Digital AI Assistant</div>
    <div class="app-caption" style="margin-bottom:0">
      Gen-AI Literature Analytics by nap@iicf.or.id • v{APP_VERSION}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="app-caption">
AI assistant untuk analisis Telco, ICT, Digital Transformation, Fiber Optic, 5G,
Satellite, Data Center, Regulation, Project &amp; Risk Management.
Dilengkapi sumber kurasi live DC &amp; FO/Subsea Asia Pacific, Simple RAG, dan Context Guard.
</div>
""",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Pengaturan")

    # Bahasa
    lang_keys = list(LANG_OPTIONS.keys())
    lang_labels = list(LANG_OPTIONS.values())
    cur_lang = st.session_state.get("answer_lang", "id")
    idx = lang_keys.index(cur_lang) if cur_lang in lang_keys else 0
    chosen_label = st.radio("🌐 Bahasa jawaban AI", lang_labels, index=idx)
    st.session_state["answer_lang"] = lang_keys[lang_labels.index(chosen_label)]

    st.session_state["enable_web_search"] = st.checkbox(
        "Aktifkan Web Search Umum (DuckDuckGo)",
        value=st.session_state["enable_web_search"])
    st.session_state["enable_specialized_apac"] = st.checkbox(
        "⭐ Aktifkan Sumber Kurasi Live DC & FO/Subsea APAC",
        value=st.session_state["enable_specialized_apac"])

    if st.session_state["enable_specialized_apac"]:
        st.session_state["prioritize_dashboard"] = st.selectbox(
            "Prioritaskan dashboard",
            options=["both", "dc", "fo"],
            format_func=lambda x: {"both": "DC + FO/Subsea",
                                   "dc": "Hanya Data Center",
                                   "fo": "Hanya Fiber/Subsea"}[x],
            index=["both", "dc", "fo"].index(
                st.session_state.get("prioritize_dashboard", "both")),
        )
        st.session_state["enable_rag"] = st.checkbox(
            "🧠 Simple RAG (hemat token)",
            value=st.session_state["enable_rag"],
            help=f"TF-IDF jika sklearn tersedia, else keyword-overlap. "
                 f"Top {RAG_TOP_K} paragraf paling relevan.")

    st.session_state["force_refresh_context"] = st.checkbox(
        "Force refresh (abaikan cache)",
        value=st.session_state["force_refresh_context"])

    st.session_state["debug_mode"] = st.checkbox(
        "Debug mode",
        value=st.session_state["debug_mode"])

    st.markdown("---")
    st.markdown("#### Status Dependencies")
    st.markdown(f"- duckduckgo-search: {'✅' if DDG_AVAILABLE else '❌ ' + DDG_ERROR}")
    st.markdown(f"- beautifulsoup4: {'✅' if BS4_AVAILABLE else '❌ ' + BS4_ERROR}")
    st.markdown(f"- supabase: {'✅' if SUPABASE_AVAILABLE else '❌ ' + SUPABASE_ERROR}")
    st.markdown(f"- fpdf2: {'✅' if FPDF_AVAILABLE else '❌ ' + FPDF_ERROR}")
    st.markdown(f"- pypdf: {'✅' if PYPDF_AVAILABLE else '⚠️ ' + PYPDF_ERROR}")
    st.markdown(f"- pandas: {'✅' if PANDAS_AVAILABLE else '⚠️ ' + PANDAS_ERROR}")
    st.markdown(f"- scikit-learn: {'✅ TF-IDF' if SKLEARN_AVAILABLE else '⚠️ keyword-overlap'}")

    st.markdown("---")
    has_hf, has_groq, has_or = get_api_keys()
    st.markdown(
        f"**Secrets:**  \n"
        f"- HF_TOKEN: {'✅' if has_hf else '❌'}  \n"
        f"- GROQ_API_KEY: {'✅' if has_groq else '❌'}  \n"
        f"- OPENROUTER_API_KEY: {'✅' if has_or else '❌'}  \n"
        f"- ADMIN_PASSWORD_HASH: {'✅' if _get_secret('ADMIN_PASSWORD_HASH') else '❌'}  \n"
        f"- Supabase: **{'Aktif' if get_supabase_client() else 'Belum dikonfigurasi'}**  \n"
        f"- Session: `{st.session_state.get('session_id', '-')}`"
    )
    if st.session_state.get("last_log_status"):
        st.caption(f"Last log: {st.session_state['last_log_status']}")

    if st.button("🧪 Test Tulis Log", use_container_width=True):
        ok, msg = log_access(feature="test_log", prompt="manual test")
        st.session_state["last_log_status"] = f"{'✅' if ok else '❌'} {msg}"
        st.info(msg)

    if st.button("🔎 Cek ketersediaan model terpilih", use_container_width=True):
        c = CHOICE_BY_ID.get(st.session_state["model_selected"],
                             CHOICE_BY_ID[DEFAULT_CHOICE_ID])
        key_map = {"groq": has_groq, "openrouter": has_or, "hf": has_hf}
        api_key = {"groq": _get_secret("GROQ_API_KEY", "groq_api_key"),
                   "openrouter": _get_secret("OPENROUTER_API_KEY"),
                   "hf": _get_secret("HF_TOKEN")}[c["provider"]]
        avail, note = check_model_availability(c["provider"], c["model"], api_key or "")
        if avail:
            st.success(f"✅ {c['label']} — {note}")
        else:
            st.warning(f"⚠️ {c['label']} — {note} (cascade akan fallback)")

    if not has_groq:
        st.warning("GROQ_API_KEY tidak terbaca di secrets.")


# ═════════════════════════════════════════════════════════════
# MODEL SELECT
# ═════════════════════════════════════════════════════════════
choice_ids = [c["id"] for c in PROVIDER_CHOICES]
cur_id = st.session_state.get("model_selected", DEFAULT_CHOICE_ID)
if cur_id not in choice_ids:
    cur_id = DEFAULT_CHOICE_ID
try:
    model_idx = choice_ids.index(cur_id)
except Exception:  # noqa: BLE001
    model_idx = 0

selected_id = st.selectbox(
    "Pilih Provider · Model AI",
    options=choice_ids,
    index=model_idx,
    format_func=lambda i: f"{CHOICE_BY_ID[i]['label']} — {CHOICE_BY_ID[i]['desc']}",
    help="Cascade otomatis jika model gagal.",
)
st.session_state["model_selected"] = selected_id
choice = CHOICE_BY_ID[selected_id]
model = choice["model"]
cap = {"type": choice["type"],
       "max_files": choice.get("max_files", 0),
       "accept": choice.get("accept", [])}

st.caption(
    f"Provider: `{choice['provider']}` · Model: `{choice['model']}` · "
    f"Tipe: {cap['type']}"
    + (f" · Max {cap['max_files']} file" if cap["max_files"] else "")
)


# ═════════════════════════════════════════════════════════════
# UPLOAD
# ═════════════════════════════════════════════════════════════
uploaded_files: List[Any] = []
if cap["type"] in ("vision", "multimodal") or cap["accept"]:
    accept_types = cap["accept"] or ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]
    uploaded_files = st.file_uploader(
        "📎 Upload file pendukung (opsional)",
        type=accept_types,
        accept_multiple_files=True,
        help=f"Maksimal {max(cap.get('max_files', 0), 3)} file",
    )
    max_f = max(cap.get("max_files", 0), 3)
    if uploaded_files and len(uploaded_files) > max_f:
        uploaded_files = uploaded_files[:max_f]
        st.warning(f"Hanya {max_f} file pertama yang digunakan.")


# ═════════════════════════════════════════════════════════════
# PROMPT
# ═════════════════════════════════════════════════════════════
prompt = st.text_area(
    "Masukkan Pertanyaan Anda",
    value=st.session_state.get("prompt_history", ""),
    height=150,
    placeholder=(
        "Contoh:\n"
        "Apa tren terbaru Data Center di Asia Pacific 2026?\n"
        "Buatkan diagram Mermaid arsitektur 5G Core.\n"
        "Update proyek kabel laut Indonesia (Nongsa-Changi, Echo)?"
    ),
    key="prompt_input",
)


# ═════════════════════════════════════════════════════════════
# EXECUTE QUERY — dipanggil sekali, dari tombol utama atau review UI
# ═════════════════════════════════════════════════════════════
def _execute_query(
    prompt_text: str,
    files_payload: List[Tuple[str, str, bytes]],
    search_context: str,
    sources_used: List[str],
    web_used: bool,
    spec_used: bool,
) -> None:
    """Jalankan query ke LLM dan tampilkan hasilnya."""
    lang_key = st.session_state.get("answer_lang", "id")
    lang_rule = LANG_INSTRUCTIONS.get(lang_key, LANG_INSTRUCTIONS["id"])

    system_prompt = f"""Anda adalah Telco Digital AI, asisten profesional di bidang
Telecommunications, ICT, Digital Transformation, Data Center, Fiber Optic,
Submarine Cable, 5G, Satellite, Regulation, Project & Risk Management.

ATURAN WAJIB:
1. {lang_rule}
2. PRIORITASKAN informasi dari sumber kurasi live berikut jika relevan:
   - Live Data Center Asia Pacific: {SPECIALIZED_SOURCES['dc']['url']}
   - Live Fiber Optic & Submarine Cable Asia Pacific: {SPECIALIZED_SOURCES['fo']['url']}
3. Jika ada konteks pencarian, gunakan dan sebutkan sumbernya.
4. Jika diminta diagram, hasilkan kode Mermaid valid dalam blok ```mermaid.
5. Bedakan dengan jelas: fakta (dari sumber), analisis, dan rekomendasi.
6. Jika data tidak lengkap, sampaikan transparan tanpa mengarang fakta."""

    file_parts, file_notes = prepare_file_parts(files_payload)
    for fn in file_notes:
        st.caption(fn)

    final_prompt = prompt_text.strip()
    if search_context:
        final_prompt = (
            "Berikut konteks pencarian yang relevan (prioritas sumber kurasi live APAC):\n\n"
            f"{search_context}\n\n"
            f"Pertanyaan pengguna:\n{prompt_text.strip()}\n\n"
            "Jawab berdasarkan informasi di atas + pengetahuan Anda. "
            "Prioritaskan dan sebutkan sumber dari dashboard Live DC / Live FO-Subsea "
            "jika relevan. Bedakan fakta, analisis, dan rekomendasi."
        )

    messages_text = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    messages_multimodal = None
    if file_parts:
        messages_multimodal = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": [{"type": "text", "text": final_prompt}] + file_parts},
        ]

    with st.spinner("🤖 AI sedang memproses..."):
        answer, used_provider, used_model, fail_log, switched = call_chain(
            messages_text, messages_multimodal)

    if not answer:
        st.error("❌ Semua provider/model gagal. Ringkasan:")
        for line in fail_log[:12]:
            st.text(f"  • {line}")
        log_access(feature="error", prompt=prompt_text.strip(),
                   model=model, error_note="; ".join(fail_log)[:300])
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ip, country = safe_get_ip_and_country()
    except Exception:  # noqa: BLE001
        ip, country = "unknown", "unknown"
    try:
        origin = get_external_referrer()
    except Exception:  # noqa: BLE001
        origin = ""

    display_model = f"{used_model} via {used_provider}"

    st.session_state["chat_history"].append({
        "role": "user", "content": prompt_text.strip(),
        "model": display_model, "time": now,
        "ip": ip, "country": country, "origin_url": origin,
    })
    st.session_state["chat_history"].append({
        "role": "assistant", "content": answer,
        "model": display_model, "time": now,
        "ip": ip, "country": country, "origin_url": origin,
        "user_prompt": prompt_text.strip(),
        "feedback": None,
        "sources": list(sources_used),
    })

    st.markdown("### ✅ Jawaban AI")
    if switched:
        st.caption(f"↪️ Auto-switch ke **{used_model}** via {used_provider}.")
    else:
        st.caption(f"Provider: **{used_provider}** · Model: `{used_model}`")

    # Render markdown normal — Streamlit menangani formatting dengan benar
    with st.container(border=True):
        st.markdown(answer)

    if sources_used:
        uniq = list(dict.fromkeys(sources_used))
        st.markdown(
            f'<div class="source-summary"><strong>📚 Sumber yang digunakan:</strong> '
            f'{html.escape(" · ".join(uniq))}</div>',
            unsafe_allow_html=True,
        )

    try:
        extract_and_render_mermaid(answer)
    except Exception:  # noqa: BLE001
        pass

    if search_context and st.session_state.get("debug_mode"):
        with st.expander("🔍 Lihat konteks pencarian (debug)"):
            st.markdown(search_context)
            st.caption(f"📏 {len(search_context):,} karakter "
                       f"(±{estimate_tokens(search_context):,} token)")

    st.session_state["prompt_history"] = prompt_text
    try:
        st.query_params["prompt"] = prompt_text[:500]
        st.query_params["model"] = model
    except Exception:  # noqa: BLE001
        pass

    ok_log, msg_log = log_access(
        feature="query_success",
        prompt=prompt_text.strip(), answer=answer,
        model=display_model, files_count=len(files_payload),
        web_search=web_used, specialized=spec_used,
    )
    st.session_state["last_log_status"] = f"{'✅' if ok_log else '❌'} {msg_log}"


# ═════════════════════════════════════════════════════════════
# TOMBOL UTAMA — build pending query
# ═════════════════════════════════════════════════════════════
if st.button("🚀 Tanya AI", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu.")
        st.stop()

    allowed, msg = _rate_limit_check()
    if not allowed:
        st.error(f"🚦 {msg}")
        st.stop()

    log_access(feature="query_start", prompt=prompt.strip(), model=model,
               files_count=len(uploaded_files or []),
               web_search=st.session_state["enable_web_search"],
               specialized=st.session_state["enable_specialized_apac"])

    # Baca file bytes
    files_payload: List[Tuple[str, str, bytes]] = []
    for f in (uploaded_files or []):
        try:
            files_payload.append((f.name, f.type or "application/octet-stream",
                                  f.getvalue()))
        except Exception:  # noqa: BLE001
            try:
                files_payload.append((f.name, f.type or "application/octet-stream",
                                      f.read()))
            except Exception:  # noqa: BLE001
                pass

    with st.spinner("⭐ Menyusun konteks (kurasi APAC + Simple RAG + web)..."):
        if st.session_state.get("force_refresh_context"):
            try:
                ddg_fetch.clear()
                scrape_dashboard.clear()
                fetch_rss_headlines.clear()
            except Exception:  # noqa: BLE001
                pass

        raw_context, sources_used, notes, web_used, spec_used = build_search_context(
            prompt=prompt.strip(),
            prioritize=st.session_state.get("prioritize_dashboard", "both"),
            use_web=st.session_state["enable_web_search"],
            use_spec=st.session_state["enable_specialized_apac"],
        )

    st.session_state["pending_query"] = {
        "prompt": prompt.strip(),
        "raw_context": raw_context,
        "files": files_payload,
        "sources_used": sources_used,
        "notes": notes,
        "web_used": web_used,
        "spec_used": spec_used,
    }


# ═════════════════════════════════════════════════════════════
# HANDLER PENDING QUERY — satu kali, tanpa klik dua kali
# ═════════════════════════════════════════════════════════════
pq = st.session_state.get("pending_query")
if pq:
    raw_ctx = pq["raw_context"]
    if len(raw_ctx) > MAX_CONTEXT_CHARS:
        # ─────── CONTEXT GUARD ───────
        rag_filtered = (
            simple_rag_filter(pq["prompt"], raw_ctx, MAX_CONTEXT_CHARS)
            if st.session_state.get("enable_rag", True)
            else raw_ctx[:MAX_CONTEXT_CHARS]
        )
        st.markdown(
            f'<div class="context-warning">'
            f'⚠️ <strong>Konteks sangat panjang</strong> '
            f'({len(raw_ctx):,} karakter ≈ {estimate_tokens(raw_ctx):,} token) — '
            f'melebihi batas aman {MAX_CONTEXT_CHARS:,} karakter '
            f'(≈{estimate_tokens("x"*MAX_CONTEXT_CHARS):,} token). '
            f'RAG menyaring menjadi {len(rag_filtered):,} karakter '
            f'(≈{estimate_tokens(rag_filtered):,} token). '
            f'Silakan konfirmasi cara melanjutkan.'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("🔍 Pratinjau konteks terfilter (RAG)", expanded=True):
            st.text_area("Konteks terfilter", rag_filtered, height=220,
                         disabled=True, label_visibility="collapsed")
        with st.expander("🔍 Lihat konteks penuh sebelum dipotong"):
            st.text_area("Konteks penuh", raw_ctx, height=220,
                         disabled=True, label_visibility="collapsed")

        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Pakai Konteks Terfilter (disarankan)",
                     type="primary", use_container_width=True):
            _execute_query(pq["prompt"], pq["files"], rag_filtered,
                           pq["sources_used"] + ["Context-Guard: RAG filter"],
                           pq["web_used"], pq["spec_used"])
            st.session_state["pending_query"] = None

        if c2.button("🚀 Paksa Konteks Penuh (risiko token limit)",
                     use_container_width=True):
            _execute_query(pq["prompt"], pq["files"], raw_ctx,
                           pq["sources_used"] + ["Context-Guard: forced full"],
                           pq["web_used"], pq["spec_used"])
            st.session_state["pending_query"] = None

        if c3.button("❌ Batal", use_container_width=True):
            st.session_state["pending_query"] = None
            st.rerun()
    else:
        # Konteks aman — eksekusi langsung
        _execute_query(pq["prompt"], pq["files"], raw_ctx,
                       pq["sources_used"], pq["web_used"], pq["spec_used"])
        st.session_state["pending_query"] = None
        render_context_notes(pq["notes"])


# ═════════════════════════════════════════════════════════════
# RIWAYAT + FEEDBACK + EXPORT
# ═════════════════════════════════════════════════════════════
if st.session_state["chat_history"]:
    st.markdown("---")
    st.subheader("📜 Riwayat Percakapan (Session ini)")

    history_md_parts: List[str] = ["# Riwayat Chat - ID Telco Digital AI\n"]
    history_txt_parts: List[str] = []
    history_wa_parts: List[str] = []

    for idx, item in enumerate(st.session_state["chat_history"]):
        role = item["role"]
        ip = item.get("ip", "unknown")
        country = item.get("country", "unknown")
        origin = item.get("origin_url", "") or ""
        session_id = st.session_state.get("session_id", "-")

        if role == "user":
            has_public = (ip not in ("unknown", "internal", "", None)
                          and not str(ip).startswith(("10.", "172.", "192.168.",
                                                      "127.", "169.254.", "100.6")))
            if has_public:
                role_label = f"👤 [{ip}] · [{country}]"
                role_plain = f"[{ip}] [{country}]"
            else:
                role_label = f"👤 Session `{session_id}`"
                role_plain = f"Session {session_id}"
        else:
            role_label = "🤖 AI"
            role_plain = "AI"

        st.markdown(f"**{role_label}** · `{item.get('time','')}` · "
                    f"`{item.get('model','')}`")
        if role == "user" and origin:
            st.caption(f"🔗 Dari: {origin}")

        with st.container(border=True):
            st.markdown(item.get("content", ""))

        # Feedback
        if role == "assistant":
            fb = item.get("feedback")
            if fb:
                st.caption(f"✅ Feedback tercatat: {'👍' if fb == 1 else '👎'}")
            else:
                fc1, fc2, _ = st.columns([1, 1, 6])
                if fc1.button("👍 Bermanfaat", key=f"fb_up_{idx}",
                              use_container_width=True):
                    item["feedback"] = 1
                    log_access(feature="feedback", feedback="positive",
                               prompt=item.get("user_prompt", "")[:MAX_LOG_PROMPT_LEN],
                               answer=item.get("content", "")[:600],
                               model=item.get("model", ""))
                    st.rerun()
                if fc2.button("👎 Kurang akurat", key=f"fb_dn_{idx}",
                              use_container_width=True):
                    item["feedback"] = -1
                    log_access(feature="feedback", feedback="negative",
                               prompt=item.get("user_prompt", "")[:MAX_LOG_PROMPT_LEN],
                               answer=item.get("content", "")[:600],
                               model=item.get("model", ""))
                    st.rerun()

        # Akumulasi export
        history_md_parts.append(f"**{role_label}** ({item['time']}) — `{item['model']}`\n")
        if origin:
            history_md_parts.append(f"Dari: {origin}\n")
        history_md_parts.append(f"\n{item['content']}\n\n---\n")
        history_txt_parts.append(f"[{item['time']}] {role_plain} "
                                 f"({item['model']}):\n{item['content']}\n")
        history_wa_parts.append(f"*{role_plain}* ({item['time']})\n"
                                f"{item['content']}\n")

    history_md = "\n".join(history_md_parts)
    history_txt = "\n".join(history_txt_parts)
    history_wa = "\n".join(history_wa_parts)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button("⬇️ Markdown", history_md, f"chat_{ts}.md",
                           "text/markdown", use_container_width=True)
    with col2:
        st.download_button("⬇️ Plain Text", history_txt, f"chat_{ts}.txt",
                           "text/plain", use_container_width=True)
    with col3:
        st.download_button("⬇️ WhatsApp", history_wa, f"chat_wa_{ts}.txt",
                           "text/plain", use_container_width=True)
    with col4:
        pdf_bytes = create_pdf_from_history(st.session_state["chat_history"])
        if pdf_bytes and len(pdf_bytes) > 100:
            st.download_button("⬇️ PDF", pdf_bytes, f"chat_{ts}.pdf",
                               "application/pdf", use_container_width=True,
                               key="btn_pdf_dl")
        else:
            st.button("⬇️ PDF (tidak tersedia)", disabled=True,
                      use_container_width=True, key="btn_pdf_dis")
    with col5:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["chat_history"] = []
            log_access(feature="clear_history")
            st.rerun()


# ═════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    f"ID Telco Digital AI v{APP_VERSION} · "
    f"Session: `{st.session_state.get('session_id', '-')}` · "
    f"Context Guard ±{MAX_CONTEXT_CHARS:,} char · "
    f"RAG {'TF-IDF' if SKLEARN_AVAILABLE else 'keyword-overlap'} · "
    "Sumber kurasi: Live DC & FO/Subsea ASPAC oleh nap@iicf.or.id."
)
