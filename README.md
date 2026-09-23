# SmartData — PDF RAG (text, tables, figures)

Fully local Retrieval-Augmented Generation system for answering questions over a PDF, including its text, tables, and figures. Test document: Apple Inc. Form 10-Q, Q3 2022 (`data/2022_Q3_AAPL.pdf`).

## Stack (all local via [Ollama](https://ollama.com), no API keys)

- Parsing: PyMuPDF (text, table detection, embedded images)
- Embeddings: `bge-m3` (Ollama)
- Vector store: Chroma (persisted locally)
- Generation: `qwen2.5:7b` (Ollama)
- Figure captioning: `qwen3-vl:8b`, a vision-language model (Ollama)
- UI: Streamlit

## Setup

```
pip install -r requirements.txt
ollama pull bge-m3 qwen2.5:7b qwen3-vl:8b
```

## Usage

```
python src/build_index.py        # parses the PDF and builds the vector index
python src/rag.py "your question"  # ask a question from the CLI
streamlit run src/app.py         # or use the web UI
```

To index a different PDF: `python src/build_index.py --pdf path/to/file.pdf`.

## How it works

1. **Ingest** (`src/ingest.py`): extracts text per page (chunked with overlap), detects and extracts tables as markdown, and pulls embedded images above a size threshold (filters out logos/icons), captioning each with a vision model so it becomes searchable text.
2. **Index** (`src/build_index.py`): embeds all chunks (text/table/figure) with `bge-m3` and stores them in Chroma, tagged with page number, section heading, and chunk type.
3. **Answer** (`src/rag.py`): retrieves the top-k most relevant chunks (regardless of type) for a question, and prompts `qwen2.5:7b` to answer strictly from that context, citing page numbers.

See `docs/Approach_and_Methodology.pdf` for the full write-up.

## Tests

```
python tests/test_ingest.py
```
