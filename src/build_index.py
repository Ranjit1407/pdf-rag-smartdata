"""Builds (or rebuilds) the Chroma vector index from a PDF."""
import argparse
import os
import shutil

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import ingest_pdf

EMBED_MODEL = "bge-m3"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION = "pdf_rag"


def build_index(pdf_path, figures_dir, persist_dir=PERSIST_DIR, caption_figures=True, reset=True):
    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    chunks = ingest_pdf(pdf_path, figures_dir, caption_figures=caption_figures)
    chunks = [c for c in chunks if c["content"].strip()]

    docs = [
        Document(
            page_content=c["content"],
            metadata={
                "type": c["type"],
                "page": c["page"],
                "section": c.get("section", ""),
                "source": os.path.basename(pdf_path),
                **({"image_path": c["image_path"]} if "image_path" in c else {}),
            },
        )
        for c in chunks
    ]

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=persist_dir,
    )

    counts = {}
    for c in chunks:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    print(f"Indexed {len(docs)} chunks -> {persist_dir}")
    print("By type:", counts)
    return vectordb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "data", "2022_Q3_AAPL.pdf"))
    parser.add_argument("--figures-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "figures"))
    parser.add_argument("--no-caption", action="store_true", help="skip VLM captioning of figures")
    args = parser.parse_args()
    build_index(args.pdf, args.figures_dir, caption_figures=not args.no_caption)
