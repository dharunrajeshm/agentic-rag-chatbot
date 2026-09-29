import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers are grounded strictly in 'Agentic AI for Executives' (Konverge AI).")

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.progress(m["meta"]["confidence"], text=f"Confidence: {m['meta']['confidence']:.0%}")
            with st.expander("Retrieved context"):
                for c in m["meta"]["retrieved_chunks"]:
                    st.markdown(f"**Page {c['page']} · score {c['score']}**\n\n{c['text']}")

if q := st.chat_input("Ask something about the eBook..."):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            r = requests.post(API_URL, json={"question": q, "top_k": 5}, timeout=60)
            r.raise_for_status()
            data = r.json()
            st.markdown(data["answer"])
            st.progress(data["confidence"], text=f"Confidence: {data['confidence']:.0%}")
            with st.expander("Retrieved context"):
                for c in data["retrieved_chunks"]:
                    st.markdown(f"**Page {c['page']} · score {c['score']}**\n\n{c['text']}")
            st.session_state.history.append({"role": "assistant", "content": data["answer"], "meta": data})
        except Exception as e:
            st.error(f"API error: {e}")
