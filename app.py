import requests
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Digital AI", page_icon="🤖", layout="wide")

# URL API Hugging Face Inference Providers
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Judul & Deskripsi
st.title("🤖 Telco Digital AI")
st.caption("Deep-link: `?prompt=...&model=...` | AI model Qwen2.5-72B, Llama-3.1-8B & gemma-3-4b")

# Inisialisasi Session State untuk Prompt (agar tidak hilang saat klik tombol)
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = ""

# Baca Query Parameter dari URL (Deep Linking)
q = st.query_params
if "prompt" in q:
    st.session_state["prompt_history"] = q["prompt"]

# Input Prompt
prompt = st.text_input(
    "Masukkan Pertanyaan Anda", 
    value=st.session_state["prompt_history"], 
    key="p",
    placeholder="Contoh: Apa itu 5G? Bagaimana cara kerja Fiber Optic?"
)

# Pilihan Model (Sudah diupdate ke model yang lebih pintar)
# Pilihan Model (Hanya yang GRATIS & PASTI JALAN)
model = st.selectbox(
    "Pilih Model AI",
    [
        "Qwen/Qwen2.5-72B-Instruct",              # ⭐ REKOMENDASI: Sangat Bagus untuk Bahasa Indonesia
        "meta-llama/Llama-3.1-8B-Instruct",       # Alternatif: Cepat & Standar
        "google/gemma-2-9b-it",                   # Alternatif: Ringan & Cepat
        "google/gemma-3-4b-it"                    # Alternatif: Model kecil (yang kemarin)
    ],
    index=0  # ⭐ DEFAULT ke Qwen2.5-72B-Instruct
)

# Tombol Kirim
if st.button("Kirim ke AI", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("⚠️ Mohon isi pertanyaan terlebih dahulu.")
    else:
        try:
            with st.spinner("Sedang berpikir dan mencari informasi di web..."):
                # Payload ke API
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,  # Token lebih banyak agar jawaban panjang
                    "temperature": 0.7,  # Kreativitas jawaban
                }
                
                # Header (Gunakan Secret Token)
                headers = {
                    "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
                    "Content-Type": "application/json"
                }
                
                # Request ke API
                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                
                # Ambil Jawaban
                result_data = response.json()
                answer = result_data["choices"][0]["message"]["content"]
                
                # Tampilkan Jawaban
                st.markdown("### ✅ Jawaban:")
                st.markdown(answer)
                
                # Update URL dengan prompt terakhir (agar bisa dibagikan lagi)
                st.query_params["prompt"] = prompt
                
                # Simpan ke history
                st.session_state["prompt_history"] = prompt
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            st.info("💡 Pastikan token HF_TOKEN sudah diset di menu 'Secrets' di Streamlit.")
