# SmartData — PDF RAG (text, tables, figures)

Local RAG system that answers questions over a PDF — text, tables, and figures. No API keys; runs entirely on Ollama.

Test document: `data/2022_Q3_AAPL.pdf` (Apple Form 10-Q, Q3 2022).

## Stack

| Component | Model |
|---|---|
| Parsing | PyMuPDF (text, tables, embedded images) |
| Embeddings | `bge-m3` |
| Vector store | Chroma (local, persisted) |
| Generation | `qwen2.5:7b` |
| Figure captioning | `qwen3-vl:8b` (vision) |
| UI | Streamlit |

## Setup

```
pip install -r requirements.txt
ollama pull bge-m3 qwen2.5:7b qwen3-vl:8b
```

## Usage

```
python src/build_index.py              # build the vector index
python src/rag.py "your question"       # query via CLI
streamlit run src/app.py                # or via web UI
```

Different PDF: `python src/build_index.py --pdf path/to/file.pdf`

## How it works

1. `src/ingest.py` — extracts text (chunked, with page + section metadata), tables (as markdown), and figures (embedded images above a size threshold, captioned by the vision model).
2. `src/build_index.py` — embeds all chunks and stores them in Chroma, tagged by type/page/section.
3. `src/rag.py` — retrieves top-k relevant chunks for a question and prompts the LLM to answer only from that context, citing pages.

Full write-up: `docs/Approach_and_Methodology.pdf`

## Tests

```
python tests/test_ingest.py
```
