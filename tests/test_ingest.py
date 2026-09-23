import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ingest import extract_text_chunks, extract_table_chunks

PDF = os.path.join(os.path.dirname(__file__), "..", "data", "2022_Q3_AAPL.pdf")


def test_extract_text_chunks_nonempty():
    chunks = extract_text_chunks(PDF)
    assert len(chunks) > 0
    assert all(c["type"] == "text" and c["page"] >= 1 for c in chunks)


def test_extract_table_chunks_nonempty_and_markdown():
    chunks = extract_table_chunks(PDF)
    assert len(chunks) > 0
    assert all(c["type"] == "table" and c["content"].startswith("|") for c in chunks)


if __name__ == "__main__":
    test_extract_text_chunks_nonempty()
    test_extract_table_chunks_nonempty_and_markdown()
    print("All tests passed.")
