"""Extract text from the Highland Green brochure PDF for use as RAG knowledge base."""
import sys
from pathlib import Path

PDF_PATH = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\DSR Highland Green brochure 14 x 10.5 inches June.pdf"


def extract_text() -> str:
    from pypdf import PdfReader
    reader = PdfReader(PDF_PATH)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def extract_text_chunked(chunk_size: int = 10) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(PDF_PATH)
    chunks = []
    current = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            current.append(f"[PAGE {i+1}]\n{text}")
        if len(current) >= chunk_size:
            chunks.append("\n\n".join(current))
            current = []
    if current:
        chunks.append("\n\n".join(current))
    return chunks


if __name__ == "__main__":
    out = Path("eval_harness/brochure_text.txt")
    text = extract_text()
    out.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text):,} chars from {PDF_PATH}")
    print(f"Saved to {out}")
