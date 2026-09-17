"""
ID Telco Digital AI Assistant - v4 (Supabase Logging)
=====================================================
Changelog vs v3:
- HAPUS semua Google Cloud / gspread / service account
- Logging behavior diganti ke Supabase (free tier) – sangat mudah & gratis
- Soft-fail tetap: aplikasi jalan meski Supabase belum dikonfigurasi
- Sisanya sama: PDF export, Mermaid, specialized search + KEY FACTS, soft error, dll.
"""

import requests
import streamlit as st
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import uuid
import traceback

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
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ============================================================
# API & CONSTANTS
# ============================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"
APP_VERSION = "4.0.0"
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
# UTILS
# ============================================================

def _is_private_ip(ip: str) -> bool:
    """Cek apakah IP termasuk private / internal (10.x, 172.16-31.x, 192.168.x, 127.x)."""
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
    """
    Ambil IP publik client + negara.
    Di Streamlit Cloud sering muncul IP internal (10.x). Kita filter dan fallback ke lookup eksternal.
    """
    ip = "unknown"
    country = "unknown"

    try:
        headers = {}
        if hasattr(st, "context") and st.context:
            headers = dict(st.context.headers or {})

        # Kumpulkan semua kandidat IP dari berbagai header
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
            # X-Forwarded-For bisa berisi rantai: client, proxy1, proxy2
            for part in str(val).replace("for=", "").split(","):
                part = part.strip().strip('"').split(";")[0].strip()
                if part and part not in candidates:
                    candidates.append(part)

        # Pilih IP publik pertama
        for cand in candidates:
            if not _is_private_ip(cand):
                ip = cand
                break

        # Jika masih private / unknown → coba lookup dari sisi server (ipapi melihat IP yang connect)
        # Catatan: di Streamlit Cloud ini sering mengembalikan IP egress Streamlit, bukan user.
        # Tetap dicoba sebagai fallback.
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

        # Lookup negara untuk IP yang sudah publik
        if ip != "unknown" and not _is_private_ip(ip) and country == "unknown":
            try:
                r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    country = data.get("country_name") or data.get("country_code") or country
            except Exception:
                pass

        # Jika tetap private → tandai sebagai internal Streamlit
        if _is_private_ip(ip):
            ip = "internal"
            if country == "unknown":
                country = "Streamlit Cloud"

    except Exception:
        pass

    return ip, country


def get_external_referrer() -> str:
    """
    Deteksi situs eksternal yang mengirimkan user ke app ini (via header Referer).
    - Jika datang dari situs luar → kembalikan URL referrer tersebut.
    - Jika buka langsung dari telco-digital-ai.streamlit.app (atau referrer kosong/internal) → kembalikan string kosong.
    """
    try:
        headers = {}
        if hasattr(st, "context") and st.context:
            headers = st.context.headers or {}

        # Ambil Referer (berbagai kemungkinan kapitalisasi)
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

        # Abaikan jika referrer berasal dari app sendiri
        own_domains = [
            "telco-digital-ai.streamlit.app",
            "localhost",
            "127.0.0.1",
            "streamlit.app",
        ]
        referer_lower = referer.lower()
        if any(d in referer_lower for d in own_domains):
            return ""

        # Batasi panjang agar aman
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


# ============================================================
# SUPABASE LOGGING (pengganti Google Cloud) – soft-fail
# ============================================================

def get_supabase_client() -> Optional["Client"]:
    """Return Supabase client or None if not configured / library missing."""
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")  # anon atau service_role
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def append_behavior_log(row: Dict[str, Any]) -> bool:
    """
    Insert one log row ke tabel 'access_logs' di Supabase.
    Soft-fail total.
    """
    try:
        client = get_supabase_client()
        if client is None:
            return False

        # Pastikan semua value string / serializable
        clean = {k: ("" if v is None else str(v)) for k, v in row.items()}
        client.table("access_logs").insert(clean).execute()
        return True
    except Exception as e:
        if st.session_state.get("debug_mode"):
            st.warning(f"[Log] Gagal tulis ke Supabase: {e}")
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
        }
        append_behavior_log(row)
    except Exception:
        pass


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
    output_parts = []

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
    """Generate PDF as pure bytes. Soft-fail on any error."""
    if not FPDF_AVAILABLE:
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        # fpdf2 baru memakai new_x/new_y, fallback ln=True masih didukung di banyak versi
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
            # Bersihkan karakter di luar latin-1 agar fpdf tidak crash
            content = re.sub(r"[*_`#]", "", str(item.get("content", "")))[:3000]
            content = content.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

        # Pastikan selalu return bytes murni
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

    # Status logging
    supabase_status = "Aktif" if get_supabase_client() else "Belum dikonfigurasi (isi SUPABASE_URL + SUPABASE_KEY di Secrets)"
    st.caption(f"Session ID: `{st.session_state.get('session_id', '-')}` • Logging ke Supabase: **{supabase_status}**")

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

    log_access(
        feature="query_start",
        prompt=prompt.strip(),
        model=model,
        files_count=len(uploaded_files) if uploaded_files else 0,
        web_search=st.session_state["enable_web_search"],
        specialized=st.session_state["enable_specialized_apac"]
    )

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

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Ambil IP, region, dan origin URL untuk ditampilkan di riwayat
        try:
            _ip, _country = safe_get_ip_and_country()
        except Exception:
            _ip, _country = "unknown", "unknown"
        try:
            _origin = get_external_referrer()  # hanya situs eksternal, kosong jika buka langsung
        except Exception:
            _origin = ""

        st.session_state["chat_history"].append({
            "role": "user",
            "content": prompt.strip(),
            "model": model,
            "time": now,
            "ip": _ip,
            "country": _country,
            "origin_url": _origin,
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "model": model,
            "time": now,
            "ip": _ip,
            "country": _country,
            "origin_url": _origin,
        })

        st.markdown("### ✅ Jawaban AI")
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

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

    # Bangun teks untuk download (tanpa HTML)
    history_md = "# Riwayat Chat - ID Telco Digital AI\n\n"
    history_plain = ""
    history_wa = ""

    # Tampilkan riwayat dengan komponen native Streamlit (bersih, tanpa tag HTML)
    for item in st.session_state["chat_history"]:
        role = item["role"]
        ip = item.get("ip", "unknown")
        country = item.get("country", "unknown")
        origin = item.get("origin_url", "") or ""

        session_id = st.session_state.get("session_id", "-")

        if role == "user":
            # Streamlit Cloud sering menyembunyikan IP publik user.
            # Jika IP publik tersedia → tampilkan. Jika tidak → pakai Session ID (lebih berguna).
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
            # Hanya tampilkan Origin jika datang dari situs eksternal
            extra_info = f"🔗 Dari: `{origin}`" if origin else ""
        else:
            role_label = "🤖 AI"
            role_plain = "AI"
            extra_info = ""

        # Header ringkas
        st.markdown(f"**{role_label}**  ·  `{item.get('time', '')}`  ·  `{item.get('model', '')}`")
        if extra_info:
            st.caption(extra_info)

        # Isi pesan (markdown biasa, aman)
        with st.container(border=True):
            st.markdown(item.get("content", ""))

        st.markdown("")  # spasi antar pesan

        # Siapkan file download
        history_md += f"**{role_label}** ({item['time']}) — `{item['model']}`\n"
        if origin:
            history_md += f"Dari: {origin}\n"
        history_md += f"\n{item['content']}\n\n---\n\n"
        history_plain += f"[{item['time']}] {role_plain} ({item['model']}):\n"
        if origin:
            history_plain += f"Dari: {origin}\n"
        history_plain += f"{item['content']}\n\n"
        history_wa += f"*{role_plain}* ({item['time']})\n{item['content']}\n\n"

    # Tombol export
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
            log_access(feature="clear_history")
            st.rerun()

st.markdown("---")
st.caption(
    f"ID Telco Digital AI v{APP_VERSION} • Session: {st.session_state.get('session_id', '-')} • "
    "Logging behavior ke Supabase (jika secrets dikonfigurasi). "
    "Sumber kurasi: Live DC & FO/Subsea ASPAC oleh nap@iicf.or.id."
)
