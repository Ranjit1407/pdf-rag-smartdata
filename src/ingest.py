"""PDF ingestion: pulls text, tables and figures out of a PDF as retrievable chunks."""
import os
import re
import pymupdf
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_FIGURE_PIXELS = 10_000  # filters out logos/icons/decorative marks
VLM_MODEL = "qwen3-vl:8b"

_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,.&'()\-]{6,80}$")


def _current_heading(line, prev_heading):
    line = line.strip()
    if line and _HEADING_RE.match(line) and not line[0].isdigit():
        return line
    return prev_heading


def extract_text_chunks(pdf_path, chunk_size=800, chunk_overlap=120):
    """Splits page text into overlapping chunks tagged with page number and nearest heading."""
    doc = pymupdf.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    heading = None
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
        for line in text.splitlines():
            heading = _current_heading(line, heading)
        for piece in splitter.split_text(text):
            chunks.append({
                "type": "text",
                "page": page_num,
                "section": heading or "",
                "content": piece,
            })
    doc.close()
    return chunks


def extract_table_chunks(pdf_path, rows_per_chunk=20):
    """Extracts each detected table as markdown, chunked by row count for very long tables."""
    doc = pymupdf.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        tables = page.find_tables()
        for t_idx, table in enumerate(tables.tables, start=1):
            rows = table.extract()
            if not rows:
                continue
            header = rows[0]
            body = rows[1:]
            for start in range(0, max(len(body), 1), rows_per_chunk):
                sub = body[start:start + rows_per_chunk]
                sep = "| " + " | ".join(["---"] * len(header)) + " |"
                content = (
                    f"| {' | '.join('' if c is None else str(c) for c in header)} |\n"
                    f"{sep}\n" +
                    "\n".join(
                        "| " + " | ".join("" if c is None else str(c) for c in row) + " |"
                        for row in sub
                    )
                )
                chunks.append({
                    "type": "table",
                    "page": page_num,
                    "section": f"Table {t_idx} (page {page_num})",
                    "content": content,
                })
    doc.close()
    return chunks


def _caption_image(image_path):
    resp = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Describe this figure/image from a financial filing factually and concisely. "
                "Mention any labeled numbers, axis labels, or captions visible in it."
            ),
            "images": [image_path],
        }],
    )
    return resp["message"]["content"].strip()


def extract_figure_chunks(pdf_path, out_dir, caption=True):
    """Extracts embedded images above a size threshold and captions them with a vision model."""
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base = doc.extract_image(xref)
            if base["width"] * base["height"] < MIN_FIGURE_PIXELS:
                continue
            fname = f"page{page_num}_img{img_idx}.{base['ext']}"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "wb") as f:
                f.write(base["image"])
            description = _caption_image(fpath) if caption else ""
            chunks.append({
                "type": "figure",
                "page": page_num,
                "section": f"Figure {img_idx} (page {page_num})",
                "content": description,
                "image_path": fpath,
            })
    doc.close()
    return chunks


def ingest_pdf(pdf_path, figures_dir, caption_figures=True):
    chunks = []
    chunks += extract_text_chunks(pdf_path)
    chunks += extract_table_chunks(pdf_path)
    chunks += extract_figure_chunks(pdf_path, figures_dir, caption=caption_figures)
    return chunks
