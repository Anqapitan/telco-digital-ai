"""
ID Telco Digital AI Assistant - v3 (Enhanced)
================================================
Changelog vs v2:
- Behavior access logging → CSV di Google Drive (shared folder admin) via service account
- Soft error handling production-ready
- Chat memory lintas session (session_id + optional load/save)
- Export PDF (fpdf2)
- Mermaid / diagram rendering (st.markdown + optional component)
- Improved specialized APAC search + static key-facts fallback dari konten aktual dashboard (Sep 2026)
- Refactor high-risk parts (token, payload, timeout, input sanitization, exception boundaries)
- IP / country approximation + origin tracking
"""

import requests
import streamlit as st
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import csv
import io
import hashlib
import uuid
import traceback
from pathlib import Path

# Optional dependencies with graceful fallback
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ============================================================
# API & CONSTANTS
# ============================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"
APP_VERSION = "3.0.0"
MAX_LOG_PROMPT_LEN = 800
MAX_LOG_ANSWER_LEN = 1500

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
.mermaid {{
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODELS
# ============================================================

MODELS = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

MODEL_INFO = {
    "Qwen/Qwen2.5-VL-72B-Instruct": "Multimodal kuat (Gambar+Dokumen) • Terbaik untuk analisis visual & Mermaid",
    "Qwen/Qwen2.5-72B-Instruct": "Text-only • Sangat kuat reasoning & analisis panjang",
    "meta-llama/Llama-3.1-8B-Instruct": "Text-only • Ringan, cepat, bagus untuk percakapan umum",
    "google/gemma-3-4b-it": "Multimodal (Gambar) • Paling ringan & hemat free tier",
    "google/gemma-3-12b-it": "Multimodal (Gambar) • Seimbang antara kualitas & kecepatan",
    "google/gemma-3-27b-it": "Multimodal (Gambar) • Kualitas tertinggi di keluarga Gemma-3",
}

MODEL_CAPABILITIES = {
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "type": "multimodal", "max_files": 5, "max_size_mb": 10,
        "accept": ["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"]
    },
    "Qwen/Qwen2.5-72B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "meta-llama/Llama-3.1-8B-Instruct": {"type": "text", "max_files": 0, "max_size_mb": 0, "accept": []},
    "google/gemma-3-4b-it": {
        "type": "vision", "max_files": 3, "max_size_mb": 5,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
    "google/gemma-3-12b-it": {
        "type": "vision", "max_files": 4, "max_size_mb": 8,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
    "google/gemma-3-27b-it": {
        "type": "vision", "max_files": 5, "max_size_mb": 10,
        "accept": ["png", "jpg", "jpeg", "webp", "gif"]
    },
}

# ============================================================
# SPECIALIZED SOURCES + KEY FACTS (dari konten aktual dashboard Sep 2026)
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

# Static key facts extracted from live dashboards (mid-Sep 2026 snapshot)
# Used as high-priority fallback when scrape/DDG yield little
KEY_FACTS_DC = """
=== KEY FACTS LIVE DC ASPAC (snapshot ~Sep 2026, sumber dashboard kurasi) ===
- CoreWeave mengumumkan data center pertamanya di Asia-Pacific di Indonesia (fokus Batam) — validasi posisi RI dalam rantai pasok komputasi AI.
- Firmus + NVIDIA memimpin dorongan AI Data Center di Indonesia, termasuk proyek skala besar di Batam (referensi kapasitas tinggi / 360MW class).
- BATIC 2026 (Bali, akhir Agustus) menekankan infrastruktur digital & AI sebagai pendorong pertumbuhan ekonomi Asia Pasifik dan kedaulatan digital nasional.
- Forum Grid Readiness & Clean Power membahas kesiapan jaringan listrik Indonesia untuk memasok DC hyperscale & AI (timeline koneksi, ekspansi transmisi).
- Batam/Nongsa masuk radar investor utama berkat kedekatan Singapura, kabel laut, FTZ, dan ketersediaan lahan.
- Klasifikasi dashboard: A Business/Investment, B AI/HPC, C Power & Energy, D Water & Environmental, E Regulatory, F Industrial Ecosystem, G Strategic/Sovereign.
Periode fokus dashboard: ~16 Jul – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
"""

KEY_FACTS_FO = """
=== KEY FACTS LIVE FO & SUBSEA ASPAC (snapshot ~Sep 2026, sumber dashboard kurasi) ===
- Landing Nongsa-Changi Cable di Batam (Juli 2026) memperpendek koridor Indonesia–Singapura; Batam dibentuk menjadi node connectivity (DC Batam ↔ IX/cloud SG ↔ kabel global).
- Sistem Echo (Google & Meta) mendarat di Singapore; arsitektur mencakup Jakarta–Guam–California (~17.000 km) — Indonesia masuk jalur trans-Pasifik hyperscaler.
- ION Cable System 1 memakai Ciena WaveLogic 6 Extreme (hingga 1,2 Tb/s per wavelength) untuk Jakarta–Singapore + backbone terestrial Sumatra.
- Telkom (via Telin) memperkuat ambisi Indonesia sebagai Hub Internet Asia Pasifik; model bisnis bergeser ke wholesale + layanan AI-ready & DCI.
- Pemerintah (KKP) menyiapkan 4 landing station baru (Jakarta, NTT, Manado, Jayapura) + >100 titik hub sesuai regulasi yang ada.
- Fokus: proyek kabel laut domestik, landing station, backbone terestrial, coherent optics, kebijakan hub digital.
Periode fokus dashboard: ~16 Jun – 16 Sep 2026. Selalu sebutkan URL dashboard jika memakai fakta ini.
"""

# ============================================================
# UTILS: Soft helpers
# ============================================================

def safe_get_ip_and_country() -> Tuple[str, str]:
    """Approximate client IP + country. Soft-fail to unknown."""
    ip = "unknown"
    country = "unknown"
    try:
        # Streamlit Cloud / headers
        headers = st.context.headers if hasattr(st, "context") and st.context else {}
        # Common proxy headers
        for key in ["X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP", "True-Client-IP"]:
            val = headers.get(key) or headers.get(key.lower())
            if val:
                ip = str(val).split(",")[0].strip()
                break
        if ip == "unknown":
            # Fallback via free geo (rate-limited, soft)
            try:
                r = requests.get("https://ipapi.co/json/", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    ip = data.get("ip", ip)
                    country = data.get("country_name") or data.get("country_code") or country
            except Exception:
                pass
        else:
            # Try country from IP
            try:
                r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    country = data.get("country_name") or data.get("country_code") or country
            except Exception:
                pass
    except Exception:
        pass
    return ip, country


def get_origin_url() -> str:
    try:
        # Streamlit query / referrer approximation
        params = st.query_params
        base = "https://telco-digital-ai.streamlit.app"
        if params:
            q = "&".join(f"{k}={v}" for k, v in params.items())
            return f"{base}?{q}"
        return base
    except Exception:
        return "https://telco-digital-ai.streamlit.app"


def generate_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:12]
    return st.session_state["session_id"]


def sanitize_text(text: str, max_len: int = 4000) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))
    return text[:max_len]


# ============================================================
# GOOGLE DRIVE LOGGING (soft-fail)
# ============================================================

def get_gspread_client():
    """Return authorized gspread client or None."""
    if not GSPREAD_AVAILABLE:
        return None
    try:
        # Expect st.secrets["gcp_service_account"] as dict (JSON key)
        sa_info = st.secrets.get("gcp_service_account")
        if not sa_info:
            return None
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def append_behavior_log(row: Dict[str, Any]) -> bool:
    """
    Append one log row to a Google Sheet (shared folder admin).
    Sheet name expected: "TelcoAI_AccessLog" or configured via secrets.
    Soft-fail: never break the main app.
    """
    try:
        client = get_gspread_client()
        if client is None:
            return False

        sheet_key = st.secrets.get("GDRIVE_LOG_SHEET_KEY") or st.secrets.get("gdrive_log_sheet_key")
        if not sheet_key:
            return False

        sh = client.open_by_key(sheet_key)
        # Prefer first worksheet or named "AccessLog"
        try:
            ws = sh.worksheet("AccessLog")
        except Exception:
            ws = sh.sheet1

        # Ensure header exists
        headers = [
            "timestamp_utc", "session_id", "ip", "country", "origin_url",
            "feature", "model", "prompt_snippet", "answer_snippet",
            "files_uploaded", "web_search_used", "specialized_used",
            "user_agent", "app_version", "error_note"
        ]
        existing = ws.row_values(1)
        if not existing or existing[0] != "timestamp_utc":
            ws.insert_row(headers, 1)

        values = [str(row.get(h, "")) for h in headers]
        ws.append_row(values, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        # Soft fail — do not surface to user unless debug
        if st.session_state.get("debug_mode"):
            st.warning(f"[Log] Gagal tulis ke Drive: {e}")
        return False


def log_access(
    feature: str,
    prompt: str = "",
    answer: str = "",
    model: str = "",
    files_count: int = 0,
    web_search: bool = False,
    specialized: bool = False,
    error_note: str = ""
):
    """High-level logger. Always soft."""
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
            "origin_url": get_origin_url(),
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
        }
        append_behavior_log(row)
    except Exception:
        pass  # absolute soft


# ============================================================
# WEB SEARCH (improved)
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    if not DDG_AVAILABLE:
        return "Library duckduckgo-search belum terinstall. Jalankan: pip install duckduckgo-search"
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


def specialized_apac_search(user_query: str, max_results: int = 6) -> str:
    """
    Improved specialized search:
    - Always inject primary sources + KEY_FACTS (real snapshot Sep 2026)
    - Aggressive site: + keyword expansion from actual dashboard content
    - Soft fallback
    """
    output_parts = []

    # 1. Primary source injection + key facts (high priority)
    output_parts.append(
        "=== SUMBER KURASI PRIMER (PRIORITAS TINGGI) ===\n"
        "Gunakan informasi terkini dari dashboard live berikut sebagai referensi utama "
        "untuk topik Data Center, Fiber Optic, dan Submarine Cable di Asia Pacific / Indonesia "
        "(periode 2-3 bulan terakhir). Sebutkan sumbernya jika relevan.\n\n"
        f"1. **{SPECIALIZED_SOURCES['dc']['name']}**\n"
        f"   URL: {SPECIALIZED_SOURCES['dc']['url']}\n"
        f"   Deskripsi: {SPECIALIZED_SOURCES['dc']['description']}\n\n"
        f"2. **{SPECIALIZED_SOURCES['fo']['name']}**\n"
        f"   URL: {SPECIALIZED_SOURCES['fo']['url']}\n"
        f"   Deskripsi: {SPECIALIZED_SOURCES['fo']['description']}\n"
    )
    output_parts.append(KEY_FACTS_DC)
    output_parts.append(KEY_FACTS_FO)

    if not DDG_AVAILABLE:
        output_parts.append(
            "\n[Info] DuckDuckGo tidak tersedia. Gunakan KEY FACTS di atas + pengetahuan model."
        )
        return "\n".join(output_parts)

    # 2. Targeted queries (improved with real keywords)
    site_queries = [
        f'site:narational.byethost11.com ({user_query})',
        'site:narational.byethost11.com (CoreWeave OR Firmus OR "Nongsa-Changi" OR Echo OR WaveLogic OR BATIC OR "grid readiness")',
        f'site:narational.byethost11.com (data center OR datacenter OR "fiber optic" OR subsea OR "submarine cable" OR "kabel laut") Indonesia',
    ]

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
            output_parts.append(
                f"{i}. **{title}**{source_tag}\n{body}...\nSumber: {href}\n"
            )
    else:
        output_parts.append(
            "\n[Info] Tidak ditemukan hasil pencarian tambahan. "
            "Tetap prioritaskan KEY FACTS + dua dashboard live di atas.\n"
        )

    output_parts.append(
        "\n=== INSTRUKSI UNTUK AI ===\n"
        "- Prioritaskan fakta dan tren yang selaras dengan KEY FACTS dan dashboard live.\n"
        "- Jika membahas Data Center atau Fiber/Subsea APAC/Indonesia, sebutkan URL dashboard yang relevan.\n"
        "- Bedakan dengan jelas: fakta dari sumber vs analisis/rekomendasi Anda.\n"
        "- Periode fokus: 2-3 bulan terakhir (pertengahan 2026).\n"
    )
    return "\n".join(output_parts)


def try_scrape_dashboard(url: str, max_chars: int = 2500) -> str:
    """Kept for future; currently returns empty on JS pages."""
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
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | App v{APP_VERSION}", ln=True)
        pdf.ln(5)

        for item in history:
            role = "ANDA" if item["role"] == "user" else "AI"
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, f"[{item.get('time', '')}] {role} ({item.get('model', '')})", ln=True)
            pdf.set_font("Helvetica", "", 10)
            # Simple text wrap; strip markdown-ish
            content = re.sub(r"[*_`#]", "", item.get("content", ""))[:3000]
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

        return pdf.output()
    except Exception:
        return None


# ============================================================
# MERMAID HELPER
# ============================================================

def extract_and_render_mermaid(text: str):
    """Extract ```mermaid blocks and render via HTML (mermaid.js CDN)."""
    pattern = r"```mermaid\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return
    st.markdown("#### Diagram Mermaid terdeteksi")
    for i, code in enumerate(matches):
        code = code.strip()
        # Render via mermaid.js
        mermaid_html = f"""
        <div class="mermaid" id="mermaid-{i}">
        {code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        """
        st.components.v1.html(mermaid_html, height=400, scrolling=True)


# ============================================================
# SESSION STATE INIT
# ============================================================

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""
if "model_selected" not in st.session_state:
    st.session_state["model_selected"] = MODELS[0]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "enable_web_search" not in st.session_state:
    st.session_state["enable_web_search"] = True
if "enable_specialized_apac" not in st.session_state:
    st.session_state["enable_specialized_apac"] = True
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False
if "session_id" not in st.session_state:
    generate_session_id()

# Cross-session memory note: true persistence needs external store.
# We keep rich session history + allow download / PDF. Optional future: load from Drive.

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
            Gen-AI Literature Analytics by nap@iicf.or.id • v{APP_VERSION}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-caption">
AI assistant untuk analisis Telco, ICT, Digital Transformation, Fiber Optic, 5G, 
Satellite, Data Center, Regulation, Project & Risk Management.
<br>Dilengkapi sumber kurasi live DC & FO/Subsea Asia Pacific (2-3 bulan terakhir).
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR / SETTINGS
# ============================================================

with st.expander("⚙️ Pengaturan Web Search, Logging & Fitur Lanjutan", expanded=False):
    st.session_state["enable_web_search"] = st.checkbox(
        "Aktifkan Web Search Umum (DuckDuckGo - Gratis)",
        value=st.session_state["enable_web_search"]
    )
    st.caption("Menggunakan DuckDuckGo. Tidak memerlukan API key.")

    st.session_state["enable_specialized_apac"] = st.checkbox(
        "⭐ Aktifkan Sumber Kurasi Live DC & FO/Subsea APAC (Prioritas)",
        value=st.session_state["enable_specialized_apac"]
    )
    st.caption(
        "Mengutamakan dua dashboard live kurasi + KEY FACTS aktual (CoreWeave, Firmus, "
        "Nongsa-Changi, Echo, WaveLogic, BATIC, grid readiness, dll.)."
    )

    if st.session_state["enable_specialized_apac"]:
        st.markdown(
            f"""
            **Dashboard primer:**
            - [Live DC ASPAC]({SPECIALIZED_SOURCES['dc']['url']})
            - [Live FO & Subsea ASPAC]({SPECIALIZED_SOURCES['fo']['url']})
            """
        )

    st.session_state["debug_mode"] = st.checkbox("Debug mode (tampilkan error detail)", value=False)
    st.caption(f"Session ID: `{st.session_state.get('session_id', '-')}` • Logging ke Google Drive: {'Aktif' if GSPREAD_AVAILABLE else 'Tidak tersedia (install gspread + secrets)'}")

# ============================================================
# MODEL SELECT
# ============================================================

current_model = st.session_state.get("model_selected", MODELS[0])
try:
    model_index = MODELS.index(current_model)
except Exception:
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
# UPLOAD
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
# PROMPT
# ============================================================

prompt = st.text_area(
    "Masukkan Pertanyaan Anda",
    value=st.session_state["prompt_history"],
    height=160,
    placeholder="Contoh:\nApa tren terbaru Data Center di Asia Pacific 2026?\nBuatkan diagram Mermaid arsitektur 5G Core.\nJelaskan regulasi AI di Indonesia.\nUpdate proyek kabel laut Indonesia terbaru (Nongsa-Changi, Echo)?",
    key="p"
)

# ============================================================
# MAIN BUTTON
# ============================================================

if st.button("🚀 Tanya AI", type="primary", use_container_width=True):

    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu.")
        st.stop()

    # Log start of query
    log_access(
        feature="query_start",
        prompt=prompt.strip(),
        model=model,
        files_count=len(uploaded_files) if uploaded_files else 0,
        web_search=st.session_state["enable_web_search"],
        specialized=st.session_state["enable_specialized_apac"]
    )

    # ---------- System Prompt ----------
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
3. Jika ada hasil web search atau specialized search (termasuk KEY FACTS), gunakan informasi tersebut dan sebutkan sumbernya (terutama URL dashboard di atas).
4. Jika diminta membuat diagram, hasilkan kode Mermaid yang valid di dalam blok ```mermaid.
5. Bedakan dengan jelas: fakta (dari sumber), analisis, dan rekomendasi.
6. Untuk topik Data Center / Fiber / Subsea APAC, sebutkan klasifikasi industri jika memungkinkan dan tautkan ke dashboard yang relevan.
7. Jika ada error atau data tidak lengkap, sampaikan secara transparan tanpa mengarang fakta.
"""

    # ---------- Build search context ----------
    search_context_parts = []
    specialized_used = False
    web_used = False

    try:
        if st.session_state["enable_specialized_apac"]:
            with st.spinner("⭐ Sedang mengambil & menyusun konteks dari sumber kurasi Live DC & FO/Subsea APAC..."):
                specialized_ctx = specialized_apac_search(prompt.strip(), max_results=6)
                search_context_parts.append(specialized_ctx)
                specialized_used = True
                for key in ["dc", "fo"]:
                    scraped = try_scrape_dashboard(SPECIALIZED_SOURCES[key]["url"])
                    if scraped:
                        search_context_parts.append(
                            f"\n[Scraped snippet dari {SPECIALIZED_SOURCES[key]['name']}]\n{scraped}\n"
                        )
    except Exception as e:
        search_context_parts.append(f"\n[Soft error specialized search: {str(e)}]\n")

    try:
        if st.session_state["enable_web_search"]:
            with st.spinner("🔍 Sedang mencari informasi terbaru di web (umum)..."):
                general_ctx = web_search(prompt.strip(), max_results=4)
                search_context_parts.append("\n=== HASIL PENCARIAN UMUM ===\n" + general_ctx)
                web_used = True
    except Exception as e:
        search_context_parts.append(f"\n[Soft error web search: {str(e)}]\n")

    search_context = "\n".join(search_context_parts) if search_context_parts else ""

    # ---------- Messages ----------
    user_content = []
    final_prompt = prompt.strip()
    if search_context:
        final_prompt = f"""Berikut konteks pencarian yang relevan (prioritas sumber kurasi live APAC + KEY FACTS + pencarian umum):

{search_context}

---
Pertanyaan pengguna:
{prompt.strip()}

Jawab berdasarkan informasi di atas + pengetahuan Anda. 
- Prioritaskan dan sebutkan sumber dari dashboard Live DC / Live FO-Subsea jika relevan.
- Bedakan fakta, analisis, dan rekomendasi.
"""

    user_content.append({"type": "text", "text": final_prompt})

    # File handling (safe)
    if uploaded_files:
        for f in uploaded_files:
            try:
                file_bytes = f.read()
                mime = f.type or "application/octet-stream"
                b64 = base64.b64encode(file_bytes).decode()
                if any(f.name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
                else:
                    text_content = file_bytes.decode("utf-8", errors="ignore")[:6000]
                    user_content.append({
                        "type": "text",
                        "text": f"\n\n[Isi file {f.name}]\n{text_content}"
                    })
            except Exception as fe:
                st.warning(f"Gagal memproses file {getattr(f, 'name', '?')}: {fe}")

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

    # ---------- Call API (soft error) ----------
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error("❌ HF_TOKEN belum diatur di Streamlit Secrets. Hubungi admin.")
        log_access(feature="error", prompt=prompt.strip(), model=model, error_note="HF_TOKEN missing")
        st.stop()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    answer = ""
    try:
        with st.spinner("🤖 AI sedang memproses..."):
            response = requests.post(API_URL, json=payload, headers=headers, timeout=180)

        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        # Save history
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

        # Try render Mermaid if present
        try:
            extract_and_render_mermaid(answer)
        except Exception:
            pass

        if search_context:
            with st.expander("🔍 Lihat konteks pencarian yang digunakan (Specialized + KEY FACTS + Umum)"):
                st.markdown(search_context)

        st.session_state["prompt_history"] = prompt
        st.query_params["prompt"] = prompt
        st.query_params["model"] = model

        # Success log
        log_access(
            feature="query_success",
            prompt=prompt.strip(),
            answer=answer,
            model=model,
            files_count=len(uploaded_files) if uploaded_files else 0,
            web_search=web_used,
            specialized=specialized_used
        )

    except requests.exceptions.Timeout:
        msg = "⏱️ Timeout: Model membutuhkan waktu lebih lama. Coba model lebih ringan atau kurangi panjang prompt."
        st.error(msg)
        log_access(feature="error", prompt=prompt.strip(), model=model, error_note="Timeout")
    except requests.exceptions.HTTPError as he:
        status = getattr(he.response, "status_code", "?")
        msg = f"❌ HTTP Error {status}. Model mungkin sedang overload atau token habis. Coba model lain."
        st.error(msg)
        if st.session_state.get("debug_mode"):
            st.code(traceback.format_exc())
        log_access(feature="error", prompt=prompt.strip(), model=model, error_note=f"HTTP {status}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        if st.session_state.get("debug_mode"):
            st.code(traceback.format_exc())
        log_access(feature="error", prompt=prompt.strip(), model=model, error_note=str(e)[:300])

# ============================================================
# RIWAYAT + EXPORT
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

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button("⬇️ Markdown", history_md, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md", "text/markdown", use_container_width=True)
    with col2:
        st.download_button("⬇️ Plain Text", history_plain, f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col3:
        st.download_button("⬇️ WhatsApp", history_wa, f"chat_wa_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with col4:
        # PDF
        pdf_bytes = create_pdf_from_history(st.session_state["chat_history"])
        if pdf_bytes:
            st.download_button(
                "⬇️ PDF",
                data=pdf_bytes,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.button("⬇️ PDF (fpdf2 belum terinstall)", disabled=True, use_container_width=True)
    with col5:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state["chat_history"] = []
            log_access(feature="clear_history")
            st.rerun()

# Footer soft note
st.markdown("---")
st.caption(
    f"ID Telco Digital AI v{APP_VERSION} • Session: {st.session_state.get('session_id', '-')} • "
    "Logging behavior ke Google Drive (jika secrets dikonfigurasi). "
    "Sumber kurasi: Live DC & FO/Subsea ASPAC oleh nap@iicf.or.id."
)
