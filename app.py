import requests
import streamlit as st

st.set_page_config(page_title="Telco Digital AI")

API_URL = "https://router.huggingface.co/v1/chat/completions"

st.title("Telco Digital AI")
st.caption("Deep-link: `?prompt=...&model=...`")

q = st.query_params
prompt = st.text_input("Prompt", value=q.get("prompt", ""), key="p")
model = st.selectbox(
    "Model",
    ["google/gemma-3-4b-it", "meta-llama/Llama-3.1-8B-Instruct"],
)

if st.button("Kirim ke AI", type="primary"):
    try:
        with st.spinner("Memproses..."):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            }
            headers = {
                "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
            }
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            st.markdown(result)
            st.query_params["prompt"] = prompt
    except Exception as e:
        st.error(f"Error: {e}")
