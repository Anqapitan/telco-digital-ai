import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Telco Digital AI")
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

st.title("Telco Digital AI")
st.caption("Deep-link: `?prompt=...&model=...`")

q = st.query_params
prompt = st.text_input("Prompt", value=q.get("prompt", ""), key="p")
model = st.selectbox(
    "Model",
    ["google/gemma-3-4b-it", "meta-llama/Llama-3.1-8B-Instruct"],
    index=0 if q.get("model", "").startswith("google") else 1,
)

if st.button("Kirim ke AI", type="primary"):
    try:
        with st.spinner("Memproses..."):
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
        st.markdown(r.choices[0].message.content)
        st.query_params["prompt"] = prompt   # URL ikut ter-update
    except Exception as e:
        st.error(f"Error: {e}")
