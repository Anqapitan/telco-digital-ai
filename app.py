"""
ID Telco Digital AI Assistant - v5.3.1 (GitHub Pages Migration)
================================================================
Changelog v5.3.1 (dari v5.3.0):
- Migrasi dashboard HTML dari ByetHost → GitHub Pages
- Konstanta GITHUB_PAGES_BASE + SITE_DOMAIN (single source of truth)
- Scraping dashboard kini mengembalikan status 'ok' (tidak lagi 'blocked')
- Pesan UI diperbarui (tidak lagi menyebut ByetHost-specific)
- KEY FACTS hardcoded menyertakan URL dashboard resmi
- BUGFIX: tambah `import random` (digunakan di ddg_fetch)
- BUGFIX: tambah import RatelimitException & TimeoutException dari ddgs
- BUGFIX: DDGS(timeout=15) fallback jika versi ddgs tidak mendukung
- BUGFIX: admin fallback password disamakan dengan caption ('admin')

Struktur repo GitHub (WAJIB):
    repo-anda/
    ├── app_telcodigitalai_v5.3.1.py    ← file ini
    ├── requirements.txt
    ├── .streamlit/
    │   └── config.toml
    ├── Live_DC_ASPAC.html              ← dashboard DC (di root)
    ├── Live_FOSubsea_ASPAC.html        ← dashboard FO (di root)
    └── README.md (opsional)

URL publik dashboard setelah GitHub Pages aktif:
    https://anqapitan.github.io/telco-digital-ai/Live_DC_ASPAC.html
    https://anqapitan.github.io/telco-digital-ai/Live_FOSubsea_ASPAC.html

SETUP SUPABASE (jalankan di SQL Editor):
-----------------------------------------
CREATE TABLE IF NOT EXISTS telcodigitalai_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ DEFAULT now(),
    session_id TEXT, ip TEXT, country TEXT, origin_url TEXT,
    feature TEXT, model TEXT, prompt_snippet TEXT, answer_snippet TEXT,
    files_uploaded INT, web_search_used BOOLEAN, specialized_used BOOLEAN,
    user_agent TEXT, app_version TEXT, error_note TEXT,
    feedback TEXT, response_time_ms INT, answer_id TEXT
);
ALTER TABLE telcodigitalai_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_anon_insert_only"
ON telcodigitalai_logs FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "allow_service_role_all"
ON telcodigitalai_logs FOR ALL TO service_role USING (true);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON telcodigitalai_logs (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_logs_feature ON telcodigitalai_logs (feature);

STREAMLIT SECRETS yang dibutuhkan:
- HF_TOKEN, GROQ_API_KEY, OPENROUTER_API_KEY
- SUPABASE_URL, SUPABASE_KEY (anon, untuk insert)
- SUPABASE_SERVICE_KEY (untuk analytics SELECT — JANGAN dipakai untuk insert)
- ADMIN_PASSWORD atau ADMIN_PASSWORD_HASH (opsional; fallback prototype: "admin")
- SUPABASE_TABLE (opsional)
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
import random          # ★ BUGFIX v5.3.1: dibutuhkan oleh ddg_fetch()
import re
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY WAJIB
# ─────────────────────────────────────────────────────────────
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCIES
# ─────────────────────────────────────────────────────────────
DDG_AVAILABLE, DDG_ERROR = False, ""
RatelimitException: type = Exception  # type: ignore
TimeoutException: type = Exception    # type: ignore
try:
    from ddgs import DDGS  # type: ignore
    try:
        from ddgs.exceptions import RatelimitException, TimeoutException  # type: ignore
    except ImportError:
        # Fallback: gunakan nama-nama yang mungkin tersedia
        try:
            from ddgs.exceptions import (  # type: ignore
                DuckDuckGoSearchException as RatelimitException,
            )
        except ImportError:
            pass
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

PIL_AVAILABLE, PIL_ERROR = False, ""
try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    PIL_ERROR = str(e)[:120]

# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────
APP_VERSION = "5.3.1"

# ═══ GitHub Pages Base URL (dashboard HTML) ═══
# Ganti 2 baris ini jika pindah ke custom domain di masa depan.
GITHUB_PAGES_BASE = "https://anqapitan.github.io/telco-digital-ai"
SITE_DOMAIN = "anqapitan.github.io"  # untuk query DDG site:

API_URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
API_URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
API_URL_HF = "https://router.huggingface.co/v1/chat/completions"

DEFAULT_SUPABASE_TABLE = "telcodigitalai_logs"
MAX_LOG_PROMPT_LEN = 800
MAX_LOG_ANSWER_LEN = 1500

MAX_CONTEXT_CHARS = 14000
RAG_TOP_K = 8
RAG_CHUNK_CHARS = 900
SUMMARY_TARGET_CHARS = 6000

RATE_LIMIT_WINDOW_SEC = 600
RATE_LIMIT_MAX_QUERIES = 25
PROBE_COOLDOWN_SEC = 60

LLM_TIMEOUT_PER_MODEL = 60
LLM_MAX_ATTEMPTS = 6

MAX_UPLOAD_MB = 8.0
MAX_TOTAL_UPLOAD_MB = 20.0
MAX_IMAGE_DIMENSION = 1600

ADMIN_MAX_ATTEMPTS = 5
ADMIN_LOCKOUT_SEC = 300
ADMIN_FALLBACK_PASSWORD = "admin"  # ganti via secrets ADMIN_PASSWORD

PROTECTED_MARKER = "<!-- PROTECTED_KEY_FACTS -->"

# ─────────────────────────────────────────────────────────────
# MODEL PROFILES
# ─────────────────────────────────────────────────────────────
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-20b":              {"max_output": 4096, "context": 8192},
    "openai/gpt-oss-120b":             {"max_output": 4096, "context": 8192},
    "qwen/qwen3.6-27b":                {"max_output": 2048, "context": 8192},
    "qwen/qwen3.8-27b":                {"max_output": 2048, "context": 8192},
    "meta-llama/llama-3.3-70b-instruct:free": {"max_output": 2048, "context": 8192},
    "openai/gpt-oss-20b:free":         {"max_output": 2048, "context": 8192},
    "google/gemma-4-31b-it:free":      {"max_output": 2048, "context": 8192},
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": {"max_output": 2048, "context": 8192},
    "meta-llama/Llama-3.1-8B-Instruct": {"max_output": 2048, "context": 8192},
    "Qwen/Qwen2.5-72B-Instruct":       {"max_output": 2048, "context": 8192},
    "Qwen/Qwen2.5-VL-72B-Instruct":    {"max_output": 2048, "context": 8192},
    "google/gemma-3-4b-it":            {"max_output": 2048, "context": 8192},
}
DEFAULT_PROFILE = {"max_output": 2048, "context": 8192}

# ─────────────────────────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────────────────────────
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" width="300" height="300">
<rect width="300" height="300" fill="none"/>
<polygon points="42,90 78,70 78,230 42,210" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
<polygon points="112,50 170,26 186,50 186,176" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
<polygon points="112,96 186,222 148,272 112,252" fill="#fbfcfe" stroke="#0b0b0b" stroke-width="8"/>
<polygon points="222,68 258,88 258,230 222,210" fill="#f2453d" stroke="#0b0b0b" stroke-width="8"/>
</svg>"""
LOGO_DATA_URI = f"data:image/svg+xml;base64,{base64.b64encode(LOGO_SVG.encode()).decode()}"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ID Telco Digital AI Assistant",
    page_icon=LOGO_DATA_URI,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME DETECTION
# ─────────────────────────────────────────────────────────────
def _detect_theme() -> str:
    """Return 'light' atau 'dark'."""
    try:
        if hasattr(st, "context") and hasattr(st.context, "theme"):
            t = st.context.theme
            if hasattr(t, "type"):
                return t.type
            if isinstance(t, dict):
                return t.get("base", "light")
    except Exception:  # noqa: BLE001
        pass
    try:
        b = st.get_option("theme.base")
        if b in ("light", "dark"):
            return b
    except Exception:  # noqa: BLE001
        pass
    return "light"

IS_DARK = (_detect_theme() == "dark")

# ─────────────────────────────────────────────────────────────
# ADAPTIVE PALETTE
# ─────────────────────────────────────────────────────────────
if IS_DARK:
    C = {
        "bg": "#0e1117", "surface": "#1c1f26",
        "surface_alt": "rgba(255,255,255,0.04)",
        "border": "rgba(255,255,255,0.15)",
        "text": "#fafafa", "text_muted": "#b0b3b8",
        "accent": "#4a9eff", "accent_bg": "rgba(74,158,255,0.12)",
        "warn": "#ffb74d", "warn_bg": "rgba(255,183,77,0.14)",
        "warn_border": "#ffb74d",
        "error": "#ef5350", "error_bg": "rgba(239,83,80,0.12)",
        "code_bg": "rgba(255,255,255,0.05)",
        "mermaid_theme": "dark",
    }
else:
    C = {
        "bg": "#ffffff", "surface": "#ffffff",
        "surface_alt": "rgba(0,0,0,0.02)",
        "border": "rgba(0,0,0,0.12)",
        "text": "#1a1a1a", "text_muted": "#5a5a5a",
        "accent": "#1e88e5", "accent_bg": "#f0f7ff",
        "warn": "#b26a00", "warn_bg": "#fff8e1",
        "warn_border": "#ffb300",
        "error": "#c62828", "error_bg": "#ffebee",
        "code_bg": "#f5f5f5",
        "mermaid_theme": "default",
    }

# ─────────────────────────────────────────────────────────────
# CSS — ADAPTIVE
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.main .block-container {{
    padding-top: 1.5rem; padding-bottom: 3rem;
    padding-left: 5%; padding-right: 5%; max-width: 1400px;
}}
.logo-header {{ display:flex; align-items:center; gap:16px; margin-bottom:0.8rem; }}
.logo-header img {{ height:58px; width:auto; filter:drop-shadow(0 3px 8px rgba(0,0,0,0.25)); }}
.app-title {{ font-size:2.3rem; font-weight:700; line-height:1.2; margin-bottom:0.15rem; color:{C['text']}; }}
.app-caption {{ font-size:0.95rem; opacity:0.75; margin-bottom:1.2rem; color:{C['text_muted']}; }}
.stButton > button {{ width:100%; min-height:44px; font-weight:600; border-radius:8px; }}
.answer-container {{ margin-top:1.5rem; padding:1.25rem; border-radius:10px;
    border:1px solid {C['border']}; background:{C['surface_alt']}; color:{C['text']}; }}
.source-summary {{ background:{C['accent_bg']}; border-left:4px solid {C['accent']};
    color:{C['text']}; padding:0.75rem 1rem; margin:1rem 0;
    border-radius:0 6px 6px 0; font-size:0.92rem; line-height:1.5; }}
.source-summary strong {{ color:{C['accent']}; }}
.context-warning {{ background:{C['warn_bg']}; border-left:4px solid {C['warn_border']};
    color:{C['text']}; padding:0.9rem 1.1rem; margin:1rem 0;
    border-radius:0 6px 6px 0; font-size:0.92rem; line-height:1.55; }}
.privacy-note {{ background:{C['surface_alt']}; border:1px solid {C['border']};
    border-radius:6px; padding:0.5rem 0.75rem; margin:0.5rem 0;
    font-size:0.8rem; color:{C['text_muted']}; }}
section[data-testid="stSidebar"] .stRadio > label {{
    font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;
    color:{C['text_muted']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
PROVIDER_CHOICES: List[Dict[str, Any]] = [
    {"id": "groq:openai/gpt-oss-20b", "provider": "groq",
     "model": "openai/gpt-oss-20b", "label": "Groq · gpt-oss-20b",
     "desc": "Aktif · cepat (default)", "type": "text",
     "max_files": 0, "accept": []},
    {"id": "groq:openai/gpt-oss-120b", "provider": "groq",
     "model": "openai/gpt-oss-120b", "label": "Groq · gpt-oss-120b",
     "desc": "Aktif · lebih kuat", "type": "text",
     "max_files": 0, "accept": []},
    {"id": "groq:qwen/qwen3.6-27b", "provider": "groq",
     "model": "qwen/qwen3.6-27b", "label": "Groq · qwen3.6-27b (Vision)",
     "desc": "Multimodal · Reasoning", "type": "multimodal",
     "max_files": 5, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "groq:qwen/qwen3.8-27b", "provider": "groq",
     "model": "qwen/qwen3.8-27b", "label": "Groq · qwen3.8-27b (Vision)",
     "desc": "Multimodal · Thinking", "type": "multimodal",
     "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "openrouter:meta-llama/llama-3.3-70b-instruct:free",
     "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free",
     "label": "OpenRouter · llama-3.3-70b:free", "desc": "FREE · Stabil",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:openai/gpt-oss-20b:free", "provider": "openrouter",
     "model": "openai/gpt-oss-20b:free", "label": "OpenRouter · gpt-oss-20b:free",
     "desc": "FREE · General", "type": "text", "max_files": 0, "accept": []},
    {"id": "openrouter:google/gemma-4-31b-it:free", "provider": "openrouter",
     "model": "google/gemma-4-31b-it:free",
     "label": "OpenRouter · gemma-4-31b:free (Vision)",
     "desc": "FREE · Multimodal", "type": "multimodal",
     "max_files": 4, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "provider": "openrouter",
     "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "label": "OpenRouter · nemotron-omni:free (Vision)",
     "desc": "FREE · Text+Image", "type": "multimodal",
     "max_files": 3, "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
    {"id": "hf:meta-llama/Llama-3.1-8B-Instruct", "provider": "hf",
     "model": "meta-llama/Llama-3.1-8B-Instruct",
     "label": "HF · Llama-3.1-8B-Instruct", "desc": "Bisa 402",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "hf:Qwen/Qwen2.5-72B-Instruct", "provider": "hf",
     "model": "Qwen/Qwen2.5-72B-Instruct",
     "label": "HF · Qwen2.5-72B-Instruct", "desc": "Reasoning kuat",
     "type": "text", "max_files": 0, "accept": []},
    {"id": "hf:Qwen/Qwen2.5-VL-72B-Instruct", "provider": "hf",
     "model": "Qwen/Qwen2.5-VL-72B-Instruct",
     "label": "HF · Qwen2.5-VL-72B (Vision)", "desc": "Multimodal",
     "type": "multimodal", "max_files": 5,
     "accept": ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]},
    {"id": "hf:google/gemma-3-4b-it", "provider": "hf",
     "model": "google/gemma-3-4b-it",
     "label": "HF · gemma-3-4b-it (Vision)", "desc": "Vision ringan",
     "type": "vision", "max_files": 3,
     "accept": ["png", "jpg", "jpeg", "webp", "gif"]},
]
CHOICE_BY_ID = {c["id"]: c for c in PROVIDER_CHOICES}
DEFAULT_CHOICE_ID = "groq:openai/gpt-oss-20b"

# ─────────────────────────────────────────────────────────────
# MULTI-LANGUAGE
# ─────────────────────────────────────────────────────────────
LANG_OPTIONS = {
    "id": "🇮🇩 Bahasa Indonesia", "en": "🇬🇧 English",
    "ms": "🇲🇾 Bahasa Melayu", "auto": "🌐 Auto",
}
LANG_INSTRUCTIONS = {
    "id": "Selalu jawab dalam Bahasa Indonesia yang profesional, jelas, dan terstruktur.",
    "en": "Always answer in professional, clear, and well-structured English.",
    "ms": "Sentiasa jawab dalam Bahasa Melayu yang profesional, jelas, dan berstruktur.",
    "auto": ("Jawab dalam bahasa yang sama dengan pertanyaan pengguna. "
             "Jika ambigu, gunakan Bahasa Indonesia."),
}

RESEARCH_MODE_INSTRUCTION = """
Gunakan format riset formal berikut:
## 1. Ringkasan Eksekutif
## 2. Fakta Utama (dengan sumber)
## 3. Analisis
## 4. Risiko / Ketidakpastian
## 5. Sumber
"""

# ─────────────────────────────────────────────────────────────
# SOURCES — GitHub Pages (v5.3.1)
# ─────────────────────────────────────────────────────────────
SPECIALIZED_SOURCES = {
    "dc": {
        "name": "Live Data Center Asia Pacific",
        "url": f"{GITHUB_PAGES_BASE}/Live_DC_ASPAC.html",
        "description": ("Curated Research Dashboard • 7 klasifikasi industri DC APAC. "
                        "Fokus Indonesia + APAC. Oleh nap@iicf.or.id."),
        "keywords": ["data center", "datacenter", "hyperscale", "AI campus", "GPU",
                     "Batam", "Nongsa", "BATIC", "CoreWeave", "Firmus"],
    },
    "fo": {
        "name": "Live Fiber Optic & Submarine Cable Asia Pacific",
        "url": f"{GITHUB_PAGES_BASE}/Live_FOSubsea_ASPAC.html",
        "description": ("Curated Research Dashboard • 8 klasifikasi industri FO & SubSEA APAC. "
                        "Oleh nap@iicf.or.id."),
        "keywords": ["submarine cable", "subsea", "fiber optic", "kabel laut",
                     "landing station", "Nongsa-Changi", "Echo", "Telin"],
    },
}

KEY_FACTS_DC = f"""=== KEY FACTS LIVE DC ASPAC (snapshot ~Sep 2026) ===
- CoreWeave: DC pertama di Asia-Pacific di Indonesia (target 2028).
- Firmus Technologies + NVIDIA: AI Factory 360 MW di Batam, ~170.000 GPU, target Q1 2027.
- DayOne mengembangkan kapasitas signifikan di Batam.
- BATIC 2026 (Bali) menekankan infrastruktur digital & AI.
- Periode fokus: ~16 Jul – 16 Sep 2026.
Dashboard resmi: {GITHUB_PAGES_BASE}/Live_DC_ASPAC.html
[CATATAN: verifikasi angka spesifik ke dashboard.]"""

KEY_FACTS_FO = f"""=== KEY FACTS LIVE FO & SUBSEA ASPAC (snapshot ~Sep 2026) ===
- Nongsa-Changi Cable (NCC) mendarat ~20 Juli 2026 di Nongsa Digital Park
  (Telin + BW Digital). 24 fiber pairs, >1.6 Pbps.
- Sistem Echo (Google & Meta) mendarat di Singapore; melibatkan Indonesia.
- ION Cable System & Ciena WaveLogic memperkuat backbone Jakarta–Singapore.
- Telkom via Telin memperkuat ambisi Indonesia sebagai Hub Internet APAC.
- Periode fokus: ~16 Jun – 16 Sep 2026.
Dashboard resmi: {GITHUB_PAGES_BASE}/Live_FOSubsea_ASPAC.html
[CATATAN: verifikasi angka spesifik ke dashboard.]"""

# ─────────────────────────────────────────────────────────────
# SYNONYMS untuk RAG
# ─────────────────────────────────────────────────────────────
SYNONYMS = {
    "submarine cable": ["kabel laut", "subsea", "kabel bawah laut", "submarine"],
    "data center": ["dc", "datacenter", "pusat data", "hyperscale"],
    "ai factory": ["gpu", "ai campus", "hyperscale", "ai datacenter"],
    "fiber optic": ["fiber", "serat optik", "optical fiber"],
    "landing station": ["pendaratan kabel", "cable landing"],
}

# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────
def _is_private_ip(ip: str) -> bool:
    if not ip or ip == "unknown":
        return True
    try:
        p = [int(x) for x in ip.split(".")]
        if len(p) != 4:
            return True
        a, b = p[0], p[1]
        if a in (10, 127, 0) or a >= 224:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
        if a == 169 and b == 254:
            return True
        if a == 100 and 64 <= b <= 127:
            return True
        return False
    except Exception:  # noqa: BLE001
        return True

def safe_get_ip_and_country() -> Tuple[str, str]:
    ip, country = "unknown", "unknown"
    try:
        headers: Dict[str, str] = {}
        if hasattr(st, "context") and st.context:
            try:
                headers = dict(st.context.headers or {})
            except Exception:  # noqa: BLE001
                headers = {}
        cands: List[str] = []
        for k in ("X-Forwarded-For", "x-forwarded-for", "X-Real-IP", "x-real-ip",
                  "CF-Connecting-IP", "cf-connecting-ip",
                  "True-Client-IP", "true-client-ip"):
            v = headers.get(k)
            if not v:
                continue
            for part in str(v).replace("for=", "").split(","):
                p = part.strip().strip('"').split(";")[0].strip()
                if p and p not in cands:
                    cands.append(p)
        for c in cands:
            if not _is_private_ip(c):
                ip = c
                break
        if ip == "unknown" or _is_private_ip(ip):
            try:
                r = requests.get("https://ipapi.co/json/", timeout=4)
                if r.status_code == 200:
                    d = r.json()
                    pub = d.get("ip", "")
                    if pub and not _is_private_ip(pub):
                        ip = pub
                        country = d.get("country_name") or d.get("country_code") or country
            except Exception:  # noqa: BLE001
                pass
        if ip != "unknown" and not _is_private_ip(ip) and country == "unknown":
            try:
                r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=4)
                if r.status_code == 200:
                    d = r.json()
                    country = d.get("country_name") or d.get("country_code") or country
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
    try:
        headers: Dict[str, str] = {}
        if hasattr(st, "context") and st.context:
            try:
                headers = st.context.headers or {}
            except Exception:  # noqa: BLE001
                headers = {}
        ref = str(headers.get("Referer") or headers.get("referer") or "").strip()
        if not ref:
            return ""
        own = ("telco-digital-ai.streamlit.app", "localhost", "127.0.0.1", "streamlit.app")
        if any(d in ref.lower() for d in own):
            return ""
        return ref[:500]
    except Exception:  # noqa: BLE001
        return ""

def generate_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:12]
    return st.session_state["session_id"]

def sanitize_text(text: str, max_len: int = 4000) -> str:
    if not text:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))[:max_len]

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
    return (_get_secret("HF_TOKEN", "hf_token"),
            _get_secret("GROQ_API_KEY", "groq_api_key", "GROQ_KEY"),
            _get_secret("OPENROUTER_API_KEY", "openrouter_api_key", "OPENROUTER_KEY"))

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5)) if text else 0

def hash_ip(ip: str) -> str:
    if ip in ("unknown", "internal", ""):
        return ip
    return "h:" + hashlib.sha256(f"tdai-{ip}".encode()).hexdigest()[:16]

# ─────────────────────────────────────────────────────────────
# RATE LIMIT
# ─────────────────────────────────────────────────────────────
def _rate_limit_check() -> Tuple[bool, str]:
    now = time.time()
    ts: deque = st.session_state.get("_rate_ts", deque())
    while ts and (now - ts[0]) > RATE_LIMIT_WINDOW_SEC:
        ts.popleft()
    if len(ts) >= RATE_LIMIT_MAX_QUERIES:
        wait = int(RATE_LIMIT_WINDOW_SEC - (now - ts[0])) + 1
        st.session_state["_rate_ts"] = ts
        return False, (f"Batas {RATE_LIMIT_MAX_QUERIES} query / "
                       f"{RATE_LIMIT_WINDOW_SEC // 60} menit tercapai. "
                       f"Coba lagi ~{wait} detik.")
    ts.append(now)
    st.session_state["_rate_ts"] = ts
    return True, ""

def _probe_cooldown_ok() -> bool:
    now = time.time()
    last = st.session_state.get("_probe_ts", 0)
    if now - last < PROBE_COOLDOWN_SEC:
        return False
    st.session_state["_probe_ts"] = now
    return True

# ─────────────────────────────────────────────────────────────
# SUPABASE LOGGING
# ─────────────────────────────────────────────────────────────
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
    """
    Insert via raw requests dengan 'Prefer: return=minimal' untuk menghindari
    PostgREST menambahkan RETURNING (yang butuh SELECT policy untuk anon).
    """
    try:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_KEY", "SUPABASE_ANON_KEY")
        if not url or not key:
            return False, "SUPABASE_URL / SUPABASE_KEY tidak terbaca di secrets"

        table = get_supabase_table_name()
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (bool, int, float)):
                clean[k] = v
            else:
                clean[k] = str(v)

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        r = requests.post(endpoint, json=clean, headers=headers, timeout=10)

        if r.status_code in (200, 201, 204):
            return True, f"OK → tabel `{table}`"

        body = (r.text or "")[:300]
        if r.status_code == 409:
            return False, f"Konflik (duplikat?) — HTTP {r.status_code}"
        if r.status_code == 401:
            return False, "Auth gagal — cek SUPABASE_KEY"
        if r.status_code == 404:
            return False, f"Tabel `{table}` tidak ditemukan"
        return False, f"HTTP {r.status_code}: {body}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:300]

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
    response_time_ms: Optional[int] = None,
    answer_id: str = "",
) -> Tuple[bool, str]:
    try:
        ip, country = safe_get_ip_and_country()
        ua = ""
        try:
            h = st.context.headers if hasattr(st, "context") else {}
            ua = h.get("User-Agent", "") or h.get("user-agent", "")
        except Exception:  # noqa: BLE001
            pass
        ip_hashed = hash_ip(ip) if st.session_state.get("privacy_mode", True) else ip
        row: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": generate_session_id(),
            "ip": ip_hashed,
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
        if response_time_ms is not None:
            row["response_time_ms"] = int(response_time_ms)
        if answer_id:
            row["answer_id"] = answer_id
        return append_behavior_log(row)
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:200]

# ─────────────────────────────────────────────────────────────
# MODEL AVAILABILITY
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def check_model_availability(provider: str, model: str, api_key: str) -> Tuple[bool, str]:
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
            return True, "rate-limited"
        if r.status_code in (401, 403):
            return False, "auth"
        if r.status_code == 402:
            return False, "payment"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:80]

# ─────────────────────────────────────────────────────────────
# SEARCH & SCRAPE
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def ddg_fetch(query: str, max_results: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    """DDG search dengan exponential backoff + jitter."""
    if not DDG_AVAILABLE:
        return [], f"ddgs tidak tersedia ({DDG_ERROR})"

    max_retries = 3
    initial_delay = 2.0

    for attempt in range(max_retries):
        try:
            try:
                ddgs_inst = DDGS(timeout=15)
            except TypeError:
                ddgs_inst = DDGS()
            with ddgs_inst as d:
                res = list(d.text(query, max_results=max_results))
            return res, ""
        except RatelimitException:
            if attempt == max_retries - 1:
                return [], "DuckDuckGo rate-limit. Coba beberapa menit lagi."
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
        except TimeoutException:
            if attempt == max_retries - 1:
                return [], "Timeout saat menghubungi DuckDuckGo."
            time.sleep(initial_delay)
        except Exception as e:  # noqa: BLE001
            return [], str(e)[:150]
    return [], "Gagal setelah beberapa percobaan."

@st.cache_data(ttl=300, show_spinner=False)
def scrape_dashboard(url: str, max_chars: int = 2500) -> Tuple[str, str, str]:
    """Return (text, status, detail). Status: ok|blocked|js_required|empty|error|library_missing."""
    if not BS4_AVAILABLE:
        return "", "library_missing", "beautifulsoup4 tidak terinstall"
    try:
        h = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,id;q=0.9",
        }
        r = requests.get(url, headers=h, timeout=10, allow_redirects=True)
        low = (r.text or "")[:4000].lower()
        if (r.status_code in (403, 503) or "cloudflare" in low
                or "just a moment" in low or "checking your browser" in low
                or "access denied" in low):
            return "", "blocked", f"HTTP {r.status_code} — anti-bot"
        if r.status_code != 200:
            return "", "error", f"HTTP {r.status_code}"
        soup = BeautifulSoup(r.text, "lxml")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        if "requires javascript" in text.lower()[:800]:
            return "", "js_required", "butuh JS"
        if len(text) < 200:
            return "", "empty", f"{len(text)} char"
        return text[:max_chars], "ok", ""
    except requests.exceptions.Timeout:
        return "", "error", "timeout"
    except Exception as e:  # noqa: BLE001
        return "", "error", str(e)[:120]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rss_headlines(cat: str) -> List[str]:
    rss = {
        "dc": ("https://news.google.com/rss/search?"
               "q=Data+Center+Indonesia+Batam+OR+CoreWeave+OR+Firmus&hl=id&gl=ID&ceid=ID:id"),
        "fo": ("https://news.google.com/rss/search?"
               "q=Submarine+Cable+Indonesia+OR+Nongsa+OR+Echo+Cable&hl=id&gl=ID&ceid=ID:id"),
    }
    url = rss.get(cat)
    if not url:
        return []
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        return [i.text.strip() for i in root.findall(".//item")[:5] if i.text][:5]
    except Exception:  # noqa: BLE001
        return []

# ─────────────────────────────────────────────────────────────
# SIMPLE RAG
# ─────────────────────────────────────────────────────────────
_STOP = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "ini", "itu",
    "apa", "bagaimana", "mengapa", "siapa", "kapan", "dimana", "jelaskan",
    "berikan", "adalah", "atau", "juga", "akan", "bisa", "dapat", "the", "and",
    "or", "for", "with", "what", "how", "please", "you", "are", "can", "to",
    "of", "in", "on",
}

def _expand_query_synonyms(query: str) -> str:
    low = query.lower()
    extras: List[str] = []
    for key, syns in SYNONYMS.items():
        if key in low:
            extras.extend(syns)
        else:
            for s in syns:
                if s in low:
                    extras.append(key)
                    extras.extend([x for x in syns if x != s])
                    break
    if extras:
        return query + " " + " ".join(set(extras))
    return query

def _split_paras(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def _score_tfidf(q: str, paras: List[str]) -> List[float]:
    if not SKLEARN_AVAILABLE or len(paras) < 2:
        return [0.0] * len(paras)
    try:
        v = TfidfVectorizer(stop_words=None, min_df=1)
        m = v.fit_transform(paras + [q])
        return [float(s) for s in cosine_similarity(m[-1], m[:-1]).flatten()]
    except Exception:  # noqa: BLE001
        return [0.0] * len(paras)

def _score_kw(q: str, paras: List[str]) -> List[float]:
    qt = {w for w in re.findall(r"[a-z0-9\-]{3,}", q.lower()) if w not in _STOP}
    if not qt:
        return [0.0] * len(paras)
    return [len(qt & set(re.findall(r"[a-z0-9\-]{3,}", p.lower()))) / max(1, len(qt))
            for p in paras]

def simple_rag_filter(query: str, ctx: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not ctx:
        return ""
    if len(ctx) <= max_chars:
        return ctx

    paras = _split_paras(ctx)
    if not paras:
        return ctx[:max_chars] + "\n\n[...dipotong...]"

    protected: List[Tuple[int, str]] = []
    dynamic: List[Tuple[int, str]] = []
    for i, p in enumerate(paras):
        if PROTECTED_MARKER in p or p.startswith("=== KEY FACTS"):
            protected.append((i, p))
        else:
            dynamic.append((i, p))

    expanded_q = _expand_query_synonyms(query)
    dynamic_texts = [p for _, p in dynamic]
    if dynamic_texts:
        scores = _score_tfidf(expanded_q, dynamic_texts)
        if all(s == 0.0 for s in scores):
            scores = _score_kw(expanded_q, dynamic_texts)
    else:
        scores = []

    protected_text = "\n\n".join(p for _, p in protected)
    budget = max(500, max_chars - len(protected_text) - 100)

    scored = sorted(zip(dynamic, scores), key=lambda x: x[1], reverse=True)
    selected_dynamic: List[Tuple[int, str]] = []
    total = 0
    for (idx, p), sc in scored:
        if total + len(p) + 2 > budget:
            break
        if sc <= 0.0 and selected_dynamic:
            continue
        selected_dynamic.append((idx, p))
        total += len(p) + 2
        if len(selected_dynamic) >= RAG_TOP_K * 2:
            break

    all_selected = sorted(protected + selected_dynamic, key=lambda x: x[0])
    result = "\n\n".join(p for _, p in all_selected)
    if len(result) < len(ctx):
        result += f"\n\n[...{len(ctx) - len(result)} karakter tidak relevan disaring RAG...]"
    return result[:max_chars]

# ─────────────────────────────────────────────────────────────
# FILE HANDLING
# ─────────────────────────────────────────────────────────────
def _extract_pdf(data: bytes, max_chars: int = 12000) -> str:
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

def _compress_image(data: bytes, mime: str,
                    max_dim: int = MAX_IMAGE_DIMENSION,
                    quality: int = 80) -> Tuple[bytes, str]:
    if not PIL_AVAILABLE:
        return data, mime
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[-1])
            else:
                bg.paste(img)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        new_data = buf.getvalue()
        if len(new_data) < len(data):
            return new_data, "image/jpeg"
        return data, mime
    except Exception:  # noqa: BLE001
        return data, mime

def validate_upload_size(files: List[Any]
                          ) -> Tuple[List[Tuple[str, str, bytes]], List[str]]:
    valid: List[Tuple[str, str, bytes]] = []
    warnings: List[str] = []
    total_mb = 0.0
    for f in files:
        try:
            data = f.getvalue()
        except Exception:  # noqa: BLE001
            try:
                data = f.read()
            except Exception:  # noqa: BLE001
                continue
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            warnings.append(f"⚠️ {f.name} terlalu besar ({size_mb:.1f} MB > {MAX_UPLOAD_MB} MB). Dilewati.")
            continue
        if total_mb + size_mb > MAX_TOTAL_UPLOAD_MB:
            warnings.append(f"⚠️ Total upload melebihi {MAX_TOTAL_UPLOAD_MB} MB. {f.name} dilewati.")
            continue
        total_mb += size_mb
        valid.append((f.name, f.type or "application/octet-stream", data))
    return valid, warnings

def prepare_file_parts(files: List[Tuple[str, str, bytes]]
                       ) -> Tuple[List[Dict[str, Any]], List[str]]:
    parts: List[Dict[str, Any]] = []
    notes: List[str] = []
    for name, mime, data in files:
        try:
            lname = name.lower()
            if lname.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                comp_data, comp_mime = _compress_image(data, mime)
                b64 = base64.b64encode(comp_data).decode()
                saved = len(data) - len(comp_data)
                note = f"🖼️ {name} dilampirkan"
                if saved > 0:
                    note += f" (kompresi −{saved // 1024} KB)"
                notes.append(note + ".")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{comp_mime};base64,{b64}"},
                })
            elif lname.endswith(".pdf"):
                txt = _extract_pdf(data)
                if txt and len(txt.strip()) > 100:
                    if len(txt) > 8000:
                        txt = txt[:8000] + "\n[...dipotong...]"
                        notes.append(f"📄 {name}: PDF diekstrak & dipotong.")
                    else:
                        notes.append(f"📄 {name}: PDF diekstrak.")
                    parts.append({"type": "text", "text": f"\n\n[Isi PDF {name}]\n{txt}"})
                else:
                    notes.append(f"📄 {name}: PDF scan tanpa teks — konversi ke PNG/JPG.")
            else:
                try:
                    txt = data.decode("utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    txt = ""
                if len(txt) > 8000:
                    txt = txt[:8000] + "\n[...dipotong...]"
                    notes.append(f"📃 {name}: dipotong.")
                else:
                    notes.append(f"📃 {name}: dilampirkan.")
                if txt:
                    parts.append({"type": "text", "text": f"\n\n[Isi file {name}]\n{txt}"})
        except Exception as e:  # noqa: BLE001
            notes.append(f"⚠️ Gagal {name}: {str(e)[:120]}")
    return parts, notes

# ─────────────────────────────────────────────────────────────
# LLM CASCADE
# ─────────────────────────────────────────────────────────────
def _choose_max_tokens(msgs: List[Dict[str, Any]], model: str) -> int:
    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)
    try:
        text = json.dumps(msgs, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = str(msgs)
    est_in = estimate_tokens(text)

    context_window = profile.get("context", 8192)
    max_output = profile.get("max_output", 2048)
    reserved = 512

    available = context_window - est_in - reserved
    if available < 256:
        return 256
    return min(available, max_output)

def _call_provider(
    url: str,
    key: str,
    model: str,
    msgs: List[Dict[str, Any]],
    extra: Optional[Dict[str, str]] = None,
    timeout: int = LLM_TIMEOUT_PER_MODEL,
) -> Tuple[str, str]:
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": _choose_max_tokens(msgs, model),
        "temperature": 0.7,
    }
    try:
        r = requests.post(url, json=payload, headers=h, timeout=timeout)
        if r.status_code in (400, 401, 402, 403, 404, 429):
            return "", f"HTTP {r.status_code}"
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or "", ""
    except requests.exceptions.Timeout:
        return "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return "", str(e)[:150]

def call_chain(
    text_msgs: List[Dict[str, Any]],
    mm_msgs: Optional[List[Dict[str, Any]]] = None,
    force_text: bool = False,
) -> Tuple[str, str, str, List[str], bool]:
    hf, groq, ork = get_api_keys()
    pid = st.session_state.get("model_selected", DEFAULT_CHOICE_ID)
    primary = CHOICE_BY_ID.get(pid) or CHOICE_BY_ID[DEFAULT_CHOICE_ID]
    chain = [primary] + [c for c in PROVIDER_CHOICES if c["id"] != primary["id"]]
    if force_text:
        chain = [c for c in chain if c.get("type") == "text"]

    fails: List[str] = []
    attempts = 0
    for c in chain:
        if attempts >= LLM_MAX_ATTEMPTS:
            break
        p, m = c["provider"], c["model"]
        mm_ok = c.get("type") in ("vision", "multimodal")
        msgs = mm_msgs if (mm_ok and mm_msgs is not None) else text_msgs
        try:
            if p == "groq":
                if not groq:
                    fails.append(f"{c['label']}: no GROQ_API_KEY")
                    continue
                attempts += 1
                ans, err = _call_provider(API_URL_GROQ, groq, m, msgs)
            elif p == "openrouter":
                if not ork:
                    fails.append(f"{c['label']}: no OPENROUTER_API_KEY")
                    continue
                attempts += 1
                ans, err = _call_provider(
                    API_URL_OPENROUTER, ork, m, msgs,
                    extra={"HTTP-Referer": "https://telco-digital-ai.streamlit.app",
                           "X-Title": "ID Telco Digital AI"})
            else:
                if not hf:
                    fails.append(f"{c['label']}: no HF_TOKEN")
                    continue
                attempts += 1
                ans, err = _call_provider(API_URL_HF, hf, m, msgs)
            if ans:
                return ans, p, m, fails, (c["id"] != pid)
            fails.append(f"{c['label']}: {err or 'unknown'}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"{c['label']}: {str(e)[:80]}")
    return "", "", "", fails, False

def llm_summarize(text: str, instr: str) -> str:
    msgs = [
        {"role": "system", "content": ("Peringkas dokumen teknis telekomunikasi. "
                                       "Pertahankan angka, nama proyek, tanggal, URL.")},
        {"role": "user", "content": instr + "\n\n" + text[:24000]},
    ]
    ans, _, _, _, _ = call_chain(msgs, None, force_text=True)
    return ans or ""

# ─────────────────────────────────────────────────────────────
# BUILD CONTEXT (v5.3.1 — GitHub Pages)
# ─────────────────────────────────────────────────────────────
def build_search_context(
    prompt: str, prioritize: str, use_web: bool, use_spec: bool,
) -> Tuple[str, List[str], List[Tuple[str, str, str]], bool, bool]:
    parts: List[str] = []
    sources: List[str] = []
    notes: List[Tuple[str, str, str]] = []
    web_used, spec_used = False, False

    if use_spec:
        spec_used = True
        parts.append("=== SUMBER KURASI PRIMER (PRIORITAS TINGGI) ===\n"
                     "Gunakan dashboard live berikut sebagai referensi utama untuk "
                     "topik DC, FO, Submarine APAC.")

        if prioritize in ("dc", "both"):
            s = SPECIALIZED_SOURCES["dc"]
            parts.append(f"1. **{s['name']}**\n   URL: {s['url']}\n   {s['description']}\n")
            parts.append(f"{PROTECTED_MARKER}\n{KEY_FACTS_DC}")
            h = fetch_rss_headlines("dc")
            if h:
                parts.append("=== JUDUL BERITA DC ===\n" + "\n".join(f"- {x}" for x in h))

        if prioritize in ("fo", "both"):
            s = SPECIALIZED_SOURCES["fo"]
            parts.append(f"2. **{s['name']}**\n   URL: {s['url']}\n   {s['description']}\n")
            parts.append(f"{PROTECTED_MARKER}\n{KEY_FACTS_FO}")
            h = fetch_rss_headlines("fo")
            if h:
                parts.append("=== JUDUL BERITA FO ===\n" + "\n".join(f"- {x}" for x in h))

        sources.append("KEY FACTS + dashboard kurasi Live DC/FO ASPAC")

        for k in (["dc", "fo"] if prioritize == "both" else [prioritize]):
            s = SPECIALIZED_SOURCES[k]
            txt, status, detail = scrape_dashboard(s["url"])
            if status == "ok":
                parts.append(f"[Scraped dari {s['name']}]\n{txt}")
                sources.append(f"Scrape {s['name']}")
            else:
                notes.append((s["name"], status, detail))

        # DDG site-search — v5.3.1 pakai SITE_DOMAIN (GitHub Pages)
        site_q: List[str] = []
        if prioritize in ("dc", "both"):
            site_q.append(f'site:{SITE_DOMAIN} ({prompt})')
            site_q.append(f'site:{SITE_DOMAIN} (CoreWeave OR Firmus '
                          f'OR "data center" OR Batam)')
        if prioritize in ("fo", "both"):
            site_q.append(f'site:{SITE_DOMAIN} ("Nongsa-Changi" OR Echo '
                          f'OR "submarine cable")')

        all_res: List[Dict[str, Any]] = []
        seen: set = set()
        for q in site_q:
            res, err = ddg_fetch(q, 3)
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
                # v5.3.1: cek SITE_DOMAIN (bukan "narational" lagi)
                tag = " [SUMBER KURASI]" if SITE_DOMAIN in r.get("href", "") else ""
                lines.append(f"{i}. {r.get('title', '')}{tag}\n"
                             f"{(r.get('body', '') or '')[:300]}...\n"
                             f"Sumber: {r.get('href', '')}")
            parts.append("\n".join(lines))
            sources.append("DuckDuckGo site-search (kurasi)")

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
            sources.append("DuckDuckGo web search")
            web_used = True

    return "\n\n".join(parts), sources, notes, web_used, spec_used

# ─────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────
def create_pdf_from_history(history: List[Dict[str, Any]],
                            title: str = "Riwayat Chat") -> Optional[bytes]:
    if not FPDF_AVAILABLE:
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"v{APP_VERSION}", ln=True)
        pdf.ln(5)
        for item in history:
            role = "User" if item.get("role") == "user" else "AI"
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, f"[{item.get('time', '')}] {role} ({item.get('model', '')})", ln=True)
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

# ─────────────────────────────────────────────────────────────
# MERMAID
# ─────────────────────────────────────────────────────────────
def _sanitize_mermaid_code(code: str) -> str:
    def _quote(match: re.Match) -> str:
        nid = match.group(1)
        label = match.group(2).strip().replace('"', '\\"')
        return f'{nid}["{label}"]'

    code = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)\[([^"\]]*[()&,][^"\]]*)\]',
                  _quote, code)
    code = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)\(([^")]*[&,][^")]*)\)',
                  lambda m: f'{m.group(1)}("{m.group(2).strip()}")', code)
    return code

def extract_and_render_mermaid(text: str) -> None:
    pattern = r"```mermaid\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return
    st.markdown("#### 📊 Diagram Mermaid terdeteksi")

    mermaid_theme = C["mermaid_theme"]
    bg = C["surface_alt"] if IS_DARK else "#ffffff"
    border = C["border"]
    text_color = C["text"]
    error_bg = C["error_bg"]
    error_fg = C["error"]

    for i, code in enumerate(matches):
        code = code.strip()
        if not code:
            continue
        if len(code) > 12000:
            st.warning(f"Diagram #{i+1} terlalu besar ({len(code):,} karakter). "
                       f"Menampilkan sebagai kode.")
            st.code(code, language="mermaid")
            continue

        sanitized = _sanitize_mermaid_code(code)
        safe_code = json.dumps(sanitized)

        lines = sanitized.count("\n") + 1
        height = min(900, max(320, 80 + lines * 22))

        mermaid_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    html, body {{ margin:0; padding:0; background:transparent; color:{text_color};
                  font-family:-apple-system,system-ui,sans-serif; }}
    #wrap {{ position:relative; background:{bg}; border:1px solid {border};
             border-radius:8px; padding:1rem; min-height:{height}px;
             overflow:auto; color:{text_color}; }}
    #target svg {{ max-width:100%; height:auto; }}
    .fs-btn {{ position:absolute; top:8px; right:8px; z-index:10;
               padding:4px 10px; font-size:12px; cursor:pointer;
               border-radius:4px; border:1px solid {border};
               background:{bg}; color:{text_color}; font-family:inherit; }}
    .fs-btn:hover {{ opacity:0.8; }}
    .code-fallback {{ background:{error_bg}; border:1px solid {border};
                      border-left:4px solid {error_fg}; border-radius:6px;
                      padding:0.8rem; color:{text_color};
                      font-family:'Consolas','Courier New',monospace;
                      font-size:0.85rem; white-space:pre-wrap;
                      word-break:break-word; margin:0; }}
    .err-msg {{ color:{error_fg}; font-size:0.85rem; font-weight:600;
                margin-top:0.6rem; }}
    .loading {{ color:{text_color}; opacity:0.6; font-size:0.9rem; }}
</style>
</head>
<body>
<div id="wrap">
    <button class="fs-btn" onclick="toggleFs()">⛶ Fullscreen</button>
    <div id="target"><div class="loading">⏳ Memuat diagram...</div></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
(function() {{
    const code = {safe_code};
    const targetId = 'target';

    function escapeHtml(s) {{
        return String(s).replace(/[&<>"']/g, c => ({{
            '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
        }})[c]);
    }}

    function showError(msg) {{
        document.getElementById(targetId).innerHTML =
            '<pre class="code-fallback">' + escapeHtml(code) + '</pre>' +
            '<p class="err-msg">⚠️ Render gagal: ' + escapeHtml(msg || 'unknown') + '</p>';
    }}

    function waitAndRender() {{
        if (typeof mermaid === 'undefined') return setTimeout(waitAndRender, 80);
        try {{
            mermaid.initialize({{
                startOnLoad: false,
                theme: '{mermaid_theme}',
                securityLevel: 'strict',
                fontFamily: 'inherit',
            }});
        }} catch (e) {{ return showError('init: ' + (e.message || e)); }}

        const rid = 'svg-' + Date.now() + '-' + Math.floor(Math.random()*1000);
        try {{
            const result = mermaid.render(rid, code);
            if (result && typeof result.then === 'function') {{
                result.then(r => {{
                    document.getElementById(targetId).innerHTML = r.svg || r;
                }}).catch(e => showError(e && e.message ? e.message : String(e)));
            }} else if (typeof result === 'string') {{
                document.getElementById(targetId).innerHTML = result;
            }} else {{
                showError('Unexpected render result');
            }}
        }} catch (e) {{
            showError(e.message || String(e));
        }}
    }}

    window.toggleFs = function() {{
        const el = document.getElementById('wrap');
        if (!document.fullscreenElement) {{
            el.requestFullscreen().catch(() => {{}});
        }} else {{
            document.exitFullscreen();
        }}
    }};

    waitAndRender();
}})();
</script>
</body>
</html>"""
        st.components.v1.html(mermaid_html, height=height + 80, scrolling=True)

        with st.expander(f"Lihat kode Mermaid #{i+1}"):
            st.code(sanitized, language="mermaid")
            if sanitized != code:
                st.caption("ℹ️ Kode di-auto-quote agar valid Mermaid.")
                with st.expander("Kode asli dari AI"):
                    st.code(code, language="mermaid")

# ─────────────────────────────────────────────────────────────
# RENDER NOTES (v5.3.1 — generic messages)
# ─────────────────────────────────────────────────────────────
SCRAPE_MSG = {
    "blocked": ("🛡️ <strong>{name}</strong> menolak akses otomatis (HTTP {detail}). "
                "Fallback aktif: DDG site-search + KEY FACTS."),
    "js_required": "🧩 <strong>{name}</strong> butuh JavaScript. Fallback: DDG + KEY FACTS.",
    "empty": "📭 <strong>{name}</strong> mengembalikan konten kosong. Fallback: DDG + KEY FACTS.",
    "error": "⚠️ <strong>{name}</strong> gagal diakses ({detail}). Fallback: DDG + KEY FACTS.",
    "library_missing": "📦 beautifulsoup4 tidak terinstall — scraping dilewati.",
}

def render_context_notes(notes: List[Tuple[str, str, str]]) -> None:
    for name, status, detail in notes:
        if status == "ok":
            continue
        msg = SCRAPE_MSG.get(status)
        if msg:
            safe = msg.format(name=html.escape(name),
                              detail=html.escape(detail or ""))
            st.markdown(f'<div class="context-warning">{safe}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption(f"ℹ️ {html.escape(name)}: {html.escape(status)} — "
                       f"{html.escape(detail or '')}")

# ─────────────────────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────────────────────
def _admin_ok(pwd: str) -> bool:
    """
    Auth admin 3 mode (urut prioritas):
    1. ADMIN_PASSWORD (plaintext) di secrets
    2. ADMIN_PASSWORD_HASH (sha256) di secrets
    3. Fallback prototype: ADMIN_FALLBACK_PASSWORD
    """
    plain = _get_secret("ADMIN_PASSWORD", "ADMIN_PIN", "ADMIN_PWD")
    if plain:
        return pwd == plain

    exp_hash = _get_secret("ADMIN_PASSWORD_HASH")
    if exp_hash and len(exp_hash) == 64:
        try:
            return hashlib.sha256(pwd.encode()).hexdigest() == exp_hash.lower()
        except Exception:  # noqa: BLE001
            return False

    return pwd == ADMIN_FALLBACK_PASSWORD

def _admin_attempt_allowed() -> bool:
    now = time.time()
    attempts: List[float] = st.session_state.get("_admin_attempts", [])
    attempts = [t for t in attempts if now - t < ADMIN_LOCKOUT_SEC]
    if len(attempts) >= ADMIN_MAX_ATTEMPTS:
        st.session_state["_admin_attempts"] = attempts
        return False
    attempts.append(now)
    st.session_state["_admin_attempts"] = attempts
    return True

def render_admin_analytics() -> None:
    if not st.session_state.get("is_admin"):
        st.caption(f"🔓 **Prototype mode** — login: `{ADMIN_FALLBACK_PASSWORD}` "
                   f"(ganti nanti via secrets `ADMIN_PASSWORD`)")
        pwd = st.text_input("Password Admin", type="password", key="admin_pwd")
        if st.button("🔓 Login", use_container_width=False):
            if not _admin_attempt_allowed():
                st.error(f"Terlalu banyak percobaan. Coba lagi "
                         f"{ADMIN_LOCKOUT_SEC // 60} menit lagi.")
            elif _admin_ok(pwd):
                st.session_state["is_admin"] = True
                st.session_state["_admin_attempts"] = []
                st.rerun()
            else:
                st.error("Password salah.")
        return

    st.success("Login berhasil.")
    if st.button("🚪 Logout"):
        st.session_state["is_admin"] = False
        st.rerun()

    if not PANDAS_AVAILABLE:
        st.warning(f"pandas tidak tersedia: {PANDAS_ERROR}")
        return

    client = get_supabase_client(admin=True)
    if client is None:
        st.warning("Supabase admin tidak terkonfigurasi "
                   "(set SUPABASE_SERVICE_KEY).")
        return

    limit_rows = st.selectbox("Jumlah baris log", [100, 300, 500, 1000, 2000], index=2)

    try:
        resp = (client.table(get_supabase_table_name())
                .select("*").order("timestamp_utc", desc=True)
                .limit(limit_rows).execute())
        rows = resp.data or []
        if not rows:
            st.info("Tabel log masih kosong.")
            return

        df = pd.DataFrame(rows)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

        if "feature" in df.columns:
            qs = df[df["feature"].isin(["query_success", "querysuccess"])]
            fb = df[df["feature"] == "feedback"]
        else:
            qs, fb = df, pd.DataFrame()

        fb_up = int((fb.get("feedback", pd.Series(dtype=str)) == "positive").sum())
        fb_down = int((fb.get("feedback", pd.Series(dtype=str)) == "negative").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total baris", f"{len(df):,}")
        c2.metric("Query sukses", f"{len(qs):,}")
        c3.metric("Sesi unik",
                  f"{df['session_id'].nunique():,}" if "session_id" in df else "-")
        c4.metric("👍 / 👎", f"{fb_up} / {fb_down}")
        c5.metric("Rating positif", f"{(fb_up / max(1, fb_up + fb_down) * 100):.0f}%")

        if "response_time_ms" in qs.columns and not qs.empty:
            try:
                rt = pd.to_numeric(qs["response_time_ms"], errors="coerce").dropna()
                if len(rt) > 0:
                    st.markdown(f"**Response time (ms):** "
                                f"median {int(rt.median())}, "
                                f"p95 {int(rt.quantile(0.95))}, "
                                f"max {int(rt.max())}")
            except Exception:  # noqa: BLE001
                pass

        col1, col2 = st.columns(2)
        with col1:
            if "timestamp_utc" in qs.columns and not qs.empty:
                st.markdown("**Tren harian (query sukses)**")
                st.line_chart(qs.set_index("timestamp_utc").resample("D").size())
            if "country" in df.columns:
                st.markdown("**Top negara**")
                st.bar_chart(df["country"].value_counts().head(8))
        with col2:
            if "model" in qs.columns:
                st.markdown("**Top model**")
                st.bar_chart(qs["model"].value_counts().head(8))
            if "feature" in df.columns:
                st.markdown("**Distribusi fitur**")
                st.bar_chart(df["feature"].value_counts().head(10))

        st.markdown("**50 log terbaru**")
        cols = [c for c in ["timestamp_utc", "feature", "country", "model",
                            "prompt_snippet", "response_time_ms", "error_note"]
                if c in df.columns]
        st.dataframe(df[cols].head(50), use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal query: {str(e)[:250]}")

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "prompt_history": "",
    "model_selected": DEFAULT_CHOICE_ID,
    "chat_history": [],
    "enable_web_search": True,
    "enable_specialized_apac": True,
    "enable_rag": True,
    "enable_research_mode": False,
    "privacy_mode": True,
    "debug_mode": False,
    "prioritize_dashboard": "both",
    "force_refresh_context": False,
    "answer_lang": "id",
    "last_log_status": "",
    "pending_query": None,
    "is_admin": False,
    "_rate_ts": deque(),
    "_admin_attempts": [],
    "_probe_ts": 0,
    "nav_section": "💬 Chat",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
generate_session_id()

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
if _q.get("admin") == "1":
    st.session_state["nav_section"] = "📊 Logs & Analytics"

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="logo-header">
  <img src="{LOGO_DATA_URI}" alt="Logo">
  <div>
    <div class="app-title">ID Telco Digital AI Assistant</div>
    <div class="app-caption" style="margin-bottom:0">
      Gen-AI Literature Analytics by nap@iicf.or.id • v{APP_VERSION} · Theme: {_detect_theme()}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EXECUTE QUERY
# ─────────────────────────────────────────────────────────────
def _execute_query(prompt_text: str,
                   files_payload: List[Tuple[str, str, bytes]],
                   search_context: str,
                   sources_used: List[str],
                   web_used: bool, spec_used: bool) -> None:
    lang_key = st.session_state.get("answer_lang", "id")
    lang_rule = LANG_INSTRUCTIONS.get(lang_key, LANG_INSTRUCTIONS["id"])
    research_block = (RESEARCH_MODE_INSTRUCTION
                      if st.session_state.get("enable_research_mode") else "")

    sys_prompt = f"""Anda adalah Telco Digital AI, asisten profesional di bidang
Telecommunications, ICT, Data Center, Fiber Optic, Submarine Cable, 5G, Satellite, Regulation.

ATURAN WAJIB:
1. {lang_rule}
2. PRIORITASKAN sumber kurasi live:
   - Live Data Center APAC: {SPECIALIZED_SOURCES['dc']['url']}
   - Live Fiber & Submarine APAC: {SPECIALIZED_SOURCES['fo']['url']}
3. Sebutkan sumber jika memakai konteks.
4. Jika diminta diagram, hasilkan kode Mermaid valid dalam blok ```mermaid.
5. Bedakan fakta, analisis, dan rekomendasi.
{research_block}"""

    file_parts, file_notes = prepare_file_parts(files_payload)
    for fn in file_notes:
        st.caption(fn)

    final_prompt = prompt_text.strip()
    if search_context:
        final_prompt = (f"Konteks pencarian:\n\n{search_context}\n\n"
                        f"Pertanyaan:\n{prompt_text.strip()}\n\n"
                        f"Jawab berdasarkan konteks + pengetahuan. "
                        f"Sebutkan sumber dashboard bila relevan.")

    msg_text = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": final_prompt}]
    msg_mm = None
    if file_parts:
        msg_mm = [{"role": "system", "content": sys_prompt},
                  {"role": "user",
                   "content": [{"type": "text", "text": final_prompt}] + file_parts}]

    start_time = time.time()
    with st.spinner("🤖 AI sedang memproses..."):
        answer, prov, mdl, fails, switched = call_chain(msg_text, msg_mm)
    response_time_ms = int((time.time() - start_time) * 1000)

    if not answer:
        st.error("❌ Semua provider/model gagal. Ringkasan:")
        for line in fails[:12]:
            st.text(f"  • {line}")
        log_access(feature="error", prompt=prompt_text, model="",
                   error_note="; ".join(fails)[:300],
                   response_time_ms=response_time_ms)
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

    display_model = f"{mdl} via {prov}"
    answer_id = str(uuid.uuid4())[:10]

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
        "answer_id": answer_id,
        "response_time_ms": response_time_ms,
    })

    st.markdown("### ✅ Jawaban AI")
    if switched:
        st.caption(f"↪️ Auto-switch ke **{mdl}** via {prov} · {response_time_ms} ms")
    else:
        st.caption(f"Provider: **{prov}** · Model: `{mdl}` · {response_time_ms} ms")

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
        with st.expander("🔍 Konteks pencarian (debug)"):
            st.markdown(search_context)
            st.caption(f"📏 {len(search_context):,} karakter "
                       f"(±{estimate_tokens(search_context):,} token)")

    st.session_state["prompt_history"] = prompt_text
    try:
        st.query_params["prompt"] = prompt_text[:500]
        st.query_params["model"] = mdl
    except Exception:  # noqa: BLE001
        pass

    ok, msg = log_access(
        feature="query_success", prompt=prompt_text, answer=answer,
        model=display_model, files_count=len(files_payload),
        web_search=web_used, specialized=spec_used,
        response_time_ms=response_time_ms, answer_id=answer_id,
    )
    st.session_state["last_log_status"] = f"{'✅' if ok else '❌'} {msg}"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigasi")
    nav = st.radio(
        "Pilih bagian",
        ["💬 Chat", "🔍 Search & RAG", "📊 Logs & Analytics", "ℹ️ Diagnostics"],
        key="nav_section", label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model & Bahasa")

    choice_ids = [c["id"] for c in PROVIDER_CHOICES]
    cur_id = st.session_state.get("model_selected", DEFAULT_CHOICE_ID)
    if cur_id not in choice_ids:
        cur_id = DEFAULT_CHOICE_ID
    try:
        idx = choice_ids.index(cur_id)
    except Exception:  # noqa: BLE001
        idx = 0
    selected_id = st.selectbox(
        "Model AI", options=choice_ids, index=idx,
        format_func=lambda i: CHOICE_BY_ID[i]["label"])
    st.session_state["model_selected"] = selected_id
    choice = CHOICE_BY_ID[selected_id]
    model = choice["model"]
    cap = {"type": choice["type"], "max_files": choice.get("max_files", 0),
           "accept": choice.get("accept", [])}

    lang_keys = list(LANG_OPTIONS.keys())
    lang_labels = list(LANG_OPTIONS.values())
    cur_lang = st.session_state.get("answer_lang", "id")
    li = lang_keys.index(cur_lang) if cur_lang in lang_keys else 0
    st.session_state["answer_lang"] = lang_keys[lang_labels.index(
        st.selectbox("Bahasa jawaban", lang_labels, index=li))]

    st.session_state["enable_research_mode"] = st.checkbox(
        "📋 Mode riset formal (Ringkasan/Fakta/Analisis/Risiko/Sumber)",
        value=st.session_state["enable_research_mode"])

    st.caption(f"`{choice['provider']}` · `{choice['model']}`")

    st.markdown("---")
    st.markdown("### 📎 Upload")
    uploaded_files = []
    if cap["type"] in ("vision", "multimodal") or cap["accept"]:
        accept = cap["accept"] or ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]
        uploaded_files = st.file_uploader(
            "File pendukung", type=accept, accept_multiple_files=True,
            label_visibility="collapsed")
        mf = max(cap.get("max_files", 0), 3)
        if uploaded_files and len(uploaded_files) > mf:
            uploaded_files = uploaded_files[:mf]
            st.caption(f"⚠️ Hanya {mf} file pertama dipakai.")
        if uploaded_files:
            st.caption(f"📏 Batas: {MAX_UPLOAD_MB} MB/file · "
                       f"{MAX_TOTAL_UPLOAD_MB} MB total"
                       + ("" if PIL_AVAILABLE else " · Pillow tidak ada, tanpa kompresi"))

    st.markdown("---")
    st.markdown("### 🔒 Privasi")
    st.session_state["privacy_mode"] = st.checkbox(
        "Hash IP di log (recommended)",
        value=st.session_state["privacy_mode"])
    st.markdown(
        '<div class="privacy-note">'
        'Pertanyaan dapat dicatat terbatas untuk analisis riset. '
        'Jangan masukkan data rahasia, kredensial, atau informasi pribadi sensitif.'
        '</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.get('session_id', '-')}`")
    if st.session_state.get("last_log_status"):
        st.caption(f"Last log: {st.session_state['last_log_status']}")

# ─────────────────────────────────────────────────────────────
# SECTION: CHAT
# ─────────────────────────────────────────────────────────────
if nav == "💬 Chat":
    prompt = st.text_area(
        "Masukkan Pertanyaan Anda",
        value=st.session_state.get("prompt_history", ""),
        height=140,
        placeholder=("Contoh:\n"
                     "Apa tren terbaru Data Center di Asia Pacific 2026?\n"
                     "Buatkan diagram Mermaid arsitektur 5G Core.\n"
                     "Update proyek kabel laut Indonesia (Nongsa-Changi, Echo)?"),
        key="prompt_input")

    if st.button("🚀 Tanya AI", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Isi pertanyaan terlebih dahulu.")
            st.stop()
        allowed, msg = _rate_limit_check()
        if not allowed:
            st.error(f"🚦 {msg}")
            st.stop()

        log_access(feature="query_start", prompt=prompt, model=model,
                   files_count=len(uploaded_files or []),
                   web_search=st.session_state["enable_web_search"],
                   specialized=st.session_state["enable_specialized_apac"])

        files_payload, upload_warnings = validate_upload_size(uploaded_files or [])
        for w in upload_warnings:
            st.warning(w)

        with st.spinner("⭐ Menyusun konteks..."):
            if st.session_state.get("force_refresh_context"):
                try:
                    ddg_fetch.clear()
                    scrape_dashboard.clear()
                    fetch_rss_headlines.clear()
                except Exception:  # noqa: BLE001
                    pass
            raw, sources, notes, w_used, s_used = build_search_context(
                prompt=prompt.strip(),
                prioritize=st.session_state.get("prioritize_dashboard", "both"),
                use_web=st.session_state["enable_web_search"],
                use_spec=st.session_state["enable_specialized_apac"])
        st.session_state["pending_query"] = {
            "prompt": prompt.strip(), "raw": raw, "files": files_payload,
            "sources": sources, "notes": notes,
            "w_used": w_used, "s_used": s_used}

    pq = st.session_state.get("pending_query")
    if pq:
        raw = pq["raw"]
        if len(raw) > MAX_CONTEXT_CHARS:
            filtered = (simple_rag_filter(pq["prompt"], raw, MAX_CONTEXT_CHARS)
                        if st.session_state.get("enable_rag", True)
                        else raw[:MAX_CONTEXT_CHARS])
            st.markdown(
                f'<div class="context-warning">'
                f'⚠️ <strong>Konteks sangat panjang</strong> '
                f'({len(raw):,} char ≈ {estimate_tokens(raw):,} token) — melebihi '
                f'batas {MAX_CONTEXT_CHARS:,} char. RAG menyaring jadi '
                f'{len(filtered):,} char (≈{estimate_tokens(filtered):,} token). '
                f'Konteks KEY FACTS dipertahankan. Konfirmasi cara lanjut:'
                f'</div>', unsafe_allow_html=True)
            with st.expander("🔍 Pratinjau konteks terfilter (RAG)", expanded=True):
                st.text_area("Konteks terfilter", filtered, height=200,
                             disabled=True, label_visibility="collapsed")
            with st.expander("🔍 Konteks penuh (sebelum filter)"):
                st.text_area("Konteks penuh", raw, height=200,
                             disabled=True, label_visibility="collapsed")
            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Pakai Konteks Terfilter", type="primary",
                         use_container_width=True):
                _execute_query(pq["prompt"], pq["files"], filtered,
                               pq["sources"] + ["Context-Guard: RAG filter"],
                               pq["w_used"], pq["s_used"])
                st.session_state["pending_query"] = None
                render_context_notes(pq["notes"])
            if c2.button("🚀 Paksa Penuh (risiko token limit)",
                         use_container_width=True):
                _execute_query(pq["prompt"], pq["files"], raw,
                               pq["sources"] + ["Context-Guard: forced full"],
                               pq["w_used"], pq["s_used"])
                st.session_state["pending_query"] = None
                render_context_notes(pq["notes"])
            if c3.button("❌ Batal", use_container_width=True):
                st.session_state["pending_query"] = None
                st.rerun()
        else:
            _execute_query(pq["prompt"], pq["files"], raw, pq["sources"],
                           pq["w_used"], pq["s_used"])
            st.session_state["pending_query"] = None
            render_context_notes(pq["notes"])

    if st.session_state["chat_history"]:
        st.markdown("---")
        st.subheader("📜 Riwayat Percakapan")
        md_parts = ["# Riwayat Chat\n"]
        txt_parts: List[str] = []
        wa_parts: List[str] = []
        for i, item in enumerate(st.session_state["chat_history"]):
            role = item["role"]
            ip = item.get("ip", "unknown")
            country = item.get("country", "unknown")
            origin = item.get("origin_url", "") or ""
            sid = st.session_state.get("session_id", "-")

            if role == "user":
                has_pub = (ip not in ("unknown", "internal", "", None)
                           and not str(ip).startswith(("10.", "172.", "192.168.",
                                                       "127.", "169.254.", "100.6", "h:")))
                label = f"👤 [{ip}] · [{country}]" if has_pub else f"👤 Session `{sid}`"
                plain = f"[{ip}] [{country}]" if has_pub else f"Session {sid}"
            else:
                label, plain = "🤖 AI", "AI"

            st.markdown(f"**{label}** · `{item.get('time','')}` · "
                        f"`{item.get('model','')}`")
            if role == "user" and origin:
                st.caption(f"🔗 Dari: {origin}")

            with st.container(border=True):
                st.markdown(item.get("content", ""))

            if role == "assistant":
                aid = item.get("answer_id", "")
                fb = item.get("feedback")
                if fb:
                    st.caption(f"✅ Feedback: {'👍' if fb == 1 else '👎'}"
                               + (f" · id `{aid}`" if aid else ""))
                else:
                    fc1, fc2, _ = st.columns([1, 1, 6])
                    if fc1.button("👍 Bermanfaat", key=f"u_{i}",
                                  use_container_width=True):
                        item["feedback"] = 1
                        log_access(feature="feedback", feedback="positive",
                                   prompt=item.get("user_prompt", "")[:MAX_LOG_PROMPT_LEN],
                                   answer=item.get("content", "")[:600],
                                   model=item.get("model", ""),
                                   answer_id=aid)
                        st.rerun()
                    if fc2.button("👎 Kurang akurat", key=f"d_{i}",
                                  use_container_width=True):
                        item["feedback"] = -1
                        log_access(feature="feedback", feedback="negative",
                                   prompt=item.get("user_prompt", "")[:MAX_LOG_PROMPT_LEN],
                                   answer=item.get("content", "")[:600],
                                   model=item.get("model", ""),
                                   answer_id=aid)
                        st.rerun()

            md_parts.append(f"**{label}** ({item['time']}) — `{item['model']}`\n"
                            + (f"Dari: {origin}\n" if origin else "")
                            + f"\n{item['content']}\n\n---\n")
            txt_parts.append(f"[{item['time']}] {plain} ({item['model']}):\n"
                             f"{item['content']}\n")
            wa_parts.append(f"*{plain}* ({item['time']})\n{item['content']}\n")

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("⬇️ Markdown", "\n".join(md_parts),
                               f"chat_{ts}.md", "text/markdown",
                               use_container_width=True)
        with c2:
            st.download_button("⬇️ TXT", "\n".join(txt_parts),
                               f"chat_{ts}.txt", "text/plain",
                               use_container_width=True)
        with c3:
            st.download_button("⬇️ WA", "\n".join(wa_parts),
                               f"chat_wa_{ts}.txt", "text/plain",
                               use_container_width=True)
        with c4:
            pdf = create_pdf_from_history(st.session_state["chat_history"])
            if pdf and len(pdf) > 100:
                st.download_button("⬇️ PDF", pdf, f"chat_{ts}.pdf",
                                   "application/pdf", use_container_width=True,
                                   key="pdf_dl")
            else:
                st.button("⬇️ PDF (n/a)", disabled=True, use_container_width=True)
        with c5:
            if st.button("🗑️ Hapus", use_container_width=True):
                st.session_state["chat_history"] = []
                log_access(feature="clear_history")
                st.rerun()

# ─────────────────────────────────────────────────────────────
# SECTION: SEARCH & RAG
# ─────────────────────────────────────────────────────────────
elif nav == "🔍 Search & RAG":
    st.header("🔍 Search & RAG Configuration")
    st.caption("Atur perilaku DuckDuckGo, RAG, dan sumber kurasi.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌐 DuckDuckGo Web Search")
        st.session_state["enable_web_search"] = st.checkbox(
            "Aktifkan web search umum", value=st.session_state["enable_web_search"])
        if not DDG_AVAILABLE:
            st.error(f"❌ ddgs tidak tersedia: {DDG_ERROR}")
        else:
            st.success("✅ ddgs tersedia")
            test_q = st.text_input("Test query", value="data center Indonesia 2026")
            if st.button("🔎 Test Search", use_container_width=False):
                with st.spinner("Mencari..."):
                    res, err = ddg_fetch(test_q, 3)
                if err:
                    st.error(f"Error: {err}")
                elif not res:
                    st.warning("Tidak ada hasil.")
                else:
                    st.success(f"{len(res)} hasil ditemukan")
                    for r in res:
                        st.markdown(f"**{r.get('title', '')}**")
                        st.caption((r.get("body", "") or "")[:200])
                        st.caption(f"🔗 {r.get('href', '')}")

    with col2:
        st.markdown("### 🧠 Simple RAG Filter")
        st.session_state["enable_rag"] = st.checkbox(
            "Aktifkan RAG filter", value=st.session_state["enable_rag"])
        st.caption(f"Filter konteks > {MAX_CONTEXT_CHARS:,} char ke top-{RAG_TOP_K} "
                   f"paragraf relevan. KEY FACTS selalu dipertahankan.")
        if SKLEARN_AVAILABLE:
            st.success("✅ TF-IDF (scikit-learn) tersedia")
        else:
            st.warning("⚠️ sklearn tidak tersedia — fallback: keyword overlap")
        st.caption("📖 Synonym expansion aktif untuk submarine cable, data center, "
                   "AI factory, fiber optic.")

        st.markdown("### 📚 Sumber Kurasi")
        st.session_state["enable_specialized_apac"] = st.checkbox(
            "Aktifkan dashboard kurasi DC & FO",
            value=st.session_state["enable_specialized_apac"])
        if st.session_state["enable_specialized_apac"]:
            st.session_state["prioritize_dashboard"] = st.selectbox(
                "Prioritaskan",
                options=["both", "dc", "fo"],
                format_func=lambda x: {"both": "DC + FO",
                                       "dc": "Hanya DC",
                                       "fo": "Hanya FO"}[x],
                index=["both", "dc", "fo"].index(
                    st.session_state.get("prioritize_dashboard", "both")))
            st.session_state["force_refresh_context"] = st.checkbox(
                "Force refresh context (abaikan cache)",
                value=st.session_state["force_refresh_context"])

        st.markdown("### 🧪 Test RAG Filter")
        sample_ctx = st.text_area(
            "Sample konteks (dummy)",
            "Data center Batam berkembang pesat dengan investasi CoreWeave.\n\n"
            "Kabel laut Nongsa-Changi baru saja mendarat.\n\n"
            "Cuaca di Jakarta hari ini cerah.\n\n"
            "Firmus membangun AI Factory 360 MW di Batam.",
            height=120)
        sample_q = st.text_input("Query untuk test",
                                 value="kapan Nongsa-Changi mendarat?")
        if st.button("🧠 Test RAG", use_container_width=False):
            filtered = simple_rag_filter(sample_q, sample_ctx, 300)
            st.markdown("**Hasil filter:**")
            st.markdown(f'<div class="context-warning">{html.escape(filtered)}</div>',
                        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SECTION: LOGS & ANALYTICS
# ─────────────────────────────────────────────────────────────
elif nav == "📊 Logs & Analytics":
    st.header("📊 Logs & Analytics")
    tab1, tab2 = st.tabs(["🔧 Status Supabase", "📈 Admin Analytics"])

    with tab1:
        st.markdown("### 🔧 Status Supabase Logging")
        if not SUPABASE_AVAILABLE:
            st.error(f"❌ Library supabase tidak terinstall: {SUPABASE_ERROR}")
        else:
            sb = get_supabase_client()
            if sb is None:
                st.error("❌ Supabase belum dikonfigurasi "
                         "(SUPABASE_URL / SUPABASE_KEY)")
            else:
                st.success(f"✅ Terhubung ke tabel `{get_supabase_table_name()}`")
                st.caption("Insert menggunakan anon key. SELECT analytics memakai "
                           "SUPABASE_SERVICE_KEY (set untuk mengaktifkan).")
                if st.button("🧪 Test Tulis Log"):
                    ok, msg = log_access(feature="test_log",
                                         prompt="manual test from UI")
                    st.session_state["last_log_status"] = f"{'✅' if ok else '❌'} {msg}"
                    st.info(msg)

        st.markdown("### 🔑 Status Secrets")
        hf, groq, ork = get_api_keys()
        st.markdown(f"""
| Secret | Status |
|---|---|
| HF_TOKEN | {'✅' if hf else '❌'} |
| GROQ_API_KEY | {'✅' if groq else '❌'} |
| OPENROUTER_API_KEY | {'✅' if ork else '❌'} |
| SUPABASE_URL | {'✅' if _get_secret('SUPABASE_URL') else '❌'} |
| SUPABASE_KEY (anon) | {'✅' if _get_secret('SUPABASE_KEY', 'SUPABASE_ANON_KEY') else '❌'} |
| SUPABASE_SERVICE_KEY | {'✅' if _get_secret('SUPABASE_SERVICE_KEY') else '—'} |
| ADMIN_PASSWORD | {'✅' if _get_secret('ADMIN_PASSWORD') else f'⚠️ (fallback: {ADMIN_FALLBACK_PASSWORD})'} |
""")

        st.markdown("### 📋 Log Terbaru (via session ini)")
        if st.session_state["chat_history"]:
            recent = [h for h in st.session_state["chat_history"]
                      if h["role"] == "user"][-10:]
            for r in reversed(recent):
                st.caption(f"🕒 {r.get('time', '')} · {r.get('model', '')} · "
                           f"{r.get('ip', '?')} ({r.get('country', '?')})")
                st.markdown(f"> {r.get('content', '')[:200]}...")
        else:
            st.info("Belum ada aktivitas di session ini.")

    with tab2:
        render_admin_analytics()

# ─────────────────────────────────────────────────────────────
# SECTION: DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
elif nav == "ℹ️ Diagnostics":
    st.header("ℹ️ System Diagnostics")

    st.markdown("### 🌐 Sumber Kurasi (GitHub Pages)")
    st.markdown(f"- **Base URL:** `{GITHUB_PAGES_BASE}`")
    st.markdown(f"- **Site domain (DDG):** `{SITE_DOMAIN}`")
    st.markdown(f"- [Live DC ASPAC]({SPECIALIZED_SOURCES['dc']['url']})")
    st.markdown(f"- [Live FO & Subsea ASPAC]({SPECIALIZED_SOURCES['fo']['url']})")

    st.markdown("### 📦 Optional Dependencies")
    deps = [
        ("ddgs", DDG_AVAILABLE, DDG_ERROR),
        ("beautifulsoup4", BS4_AVAILABLE, BS4_ERROR),
        ("supabase", SUPABASE_AVAILABLE, SUPABASE_ERROR),
        ("fpdf2", FPDF_AVAILABLE, FPDF_ERROR),
        ("pypdf", PYPDF_AVAILABLE, PYPDF_ERROR),
        ("pandas", PANDAS_AVAILABLE, PANDAS_ERROR),
        ("scikit-learn", SKLEARN_AVAILABLE, SKLEARN_ERROR),
        ("Pillow (image compression)", PIL_AVAILABLE, PIL_ERROR),
    ]
    for name, avail, err in deps:
        st.markdown(f"- **{name}**: "
                    f"{'✅ tersedia' if avail else f'❌ {err}'}")

    st.markdown("### 🔬 Model Availability")
    st.caption("⚠️ Probe memakai sebagian kecil kuota API. Dibatasi 1×/60 detik.")
    c = CHOICE_BY_ID.get(st.session_state["model_selected"],
                         CHOICE_BY_ID[DEFAULT_CHOICE_ID])
    if st.button("🔎 Cek model terpilih", use_container_width=False):
        if not _probe_cooldown_ok():
            st.warning(f"Tunggu {PROBE_COOLDOWN_SEC} detik sebelum probe lagi.")
        else:
            hf, groq, ork = get_api_keys()
            api_key = {"groq": groq, "openrouter": ork, "hf": hf}[c["provider"]]
            avail, note = check_model_availability(c["provider"], c["model"],
                                                    api_key or "")
            (st.success if avail else st.warning)(f"{c['label']} — {note}")

    if st.button("🔎 Cek 6 provider pertama", use_container_width=False):
        if not _probe_cooldown_ok():
            st.warning(f"Tunggu {PROBE_COOLDOWN_SEC} detik sebelum probe lagi.")
        else:
            hf, groq, ork = get_api_keys()
            for ch in PROVIDER_CHOICES[:6]:
                api_key = {"groq": groq, "openrouter": ork, "hf": hf}[ch["provider"]]
                if not api_key:
                    st.caption(f"⏭️ {ch['label']}: skip (no key)")
                    continue
                with st.spinner(f"Cek {ch['label']}..."):
                    avail, note = check_model_availability(
                        ch["provider"], ch["model"], api_key)
                icon = "✅" if avail else "⚠️"
                st.caption(f"{icon} {ch['label']}: {note}")

    st.markdown("### 🧪 Tools")
    if st.button("🧹 Clear semua cache", use_container_width=False):
        try:
            ddg_fetch.clear()
            scrape_dashboard.clear()
            fetch_rss_headlines.clear()
            check_model_availability.clear()
            st.success("Cache dibersihkan.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Gagal: {e}")

    st.markdown("### 🔐 Generate Admin Password Hash")
    pwd_in = st.text_input("Password yang ingin di-hash", type="password")
    if pwd_in:
        st.code(hashlib.sha256(pwd_in.encode()).hexdigest(), language="text")
        st.caption('Copy ke Streamlit Secrets: `ADMIN_PASSWORD_HASH = "..."`')

    st.markdown("### 📊 Model Profiles (max_tokens dinamis)")
    for m, prof in list(MODEL_PROFILES.items())[:6]:
        st.caption(f"`{m}` → output max {prof['max_output']}, "
                   f"context {prof['context']}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"ID Telco Digital AI v{APP_VERSION} · "
    f"Session `{st.session_state.get('session_id', '-')}` · "
    f"Theme: {_detect_theme()} · "
    f"Context Guard ±{MAX_CONTEXT_CHARS:,} char · "
    f"RAG {'TF-IDF' if SKLEARN_AVAILABLE else 'keyword'} · "
    f"Sources: GitHub Pages Live DC & FO/Subsea ASPAC"
)
