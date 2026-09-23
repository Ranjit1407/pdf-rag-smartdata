"""Streamlit UI for querying the indexed PDF."""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from rag import RAGPipeline
from build_index import PERSIST_DIR

st.set_page_config(page_title="SmartData PDF RAG", page_icon="📄")
st.title("PDF RAG: text, tables & figures")
st.caption("Ask questions about the indexed PDF (Apple Q3 2022 Form 10-Q by default).")

if not os.path.exists(PERSIST_DIR):
    st.error(f"No index found at {PERSIST_DIR}. Run `python src/build_index.py` first.")
    st.stop()


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()

question = st.text_input("Your question", placeholder="e.g. What was gross margin for the nine months ended June 25, 2022?")

if question:
    with st.spinner("Retrieving and generating answer..."):
        answer, sources = pipeline.answer(question)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources")
    for s in sources:
        st.markdown(f"- **{s['type']}** — page {s['page']} — _{s['section']}_")
