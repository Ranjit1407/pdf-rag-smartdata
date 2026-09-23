"""Retrieval-augmented answering over the indexed PDF."""
import os

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from build_index import PERSIST_DIR, COLLECTION, EMBED_MODEL

LLM_MODEL = "qwen2.5:7b"
TOP_K = 6

SYSTEM_PROMPT = """You are an assistant answering questions about a single PDF document (an SEC Form 10-Q filing).
Answer ONLY using the context chunks below. Each chunk is tagged with its type (text/table/figure) and page number.
When a table chunk answers the question, read it carefully as tabular data before answering.
Always cite the page number(s) you used, like (p. 4).
If the context does not contain the answer, say so plainly instead of guessing."""


def _format_context(docs):
    parts = []
    for d in docs:
        meta = d.metadata
        tag = f"[{meta.get('type', 'text').upper()} | page {meta.get('page')} | {meta.get('section', '')}]"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self, persist_dir=PERSIST_DIR, top_k=TOP_K):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.vectordb = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.top_k = top_k

    def retrieve(self, question, k=None):
        return self.vectordb.similarity_search(question, k=k or self.top_k)

    def answer(self, question, k=None):
        docs = self.retrieve(question, k=k)
        context = _format_context(docs)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self.llm.invoke(prompt)
        sources = [
            {"page": d.metadata.get("page"), "type": d.metadata.get("type"), "section": d.metadata.get("section")}
            for d in docs
        ]
        return response.content, sources


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What were total net sales for the three months ended June 25, 2022?"
    pipeline = RAGPipeline()
    answer, sources = pipeline.answer(question)
    print("Q:", question)
    print("\nA:", answer)
    print("\nSources:")
    for s in sources:
        print(" -", s)
