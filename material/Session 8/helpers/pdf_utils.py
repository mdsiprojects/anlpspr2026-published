from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: str | Path) -> list[dict]:
    """Return extracted text page by page with simple metadata."""
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    records: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        records.append(
            {
                "source_file": path.name,
                "page_number": page_number,
                "text": text,
            }
        )

    return records


def join_pages(page_records: list[dict]) -> str:
    """Join non-empty extracted pages into one string for chunking."""
    return "\n\n".join(record["text"] for record in page_records if record["text"].strip())
