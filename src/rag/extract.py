"""File to text.

PDF extraction uses PyMuPDF, which is materially better than pypdf on real
layouts. It is AGPL; irrelevant for a service that is never distributed, but a
deliberate choice rather than an accident.
"""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type: {path.name}")

    if suffix == ".pdf":
        return path.stem, _extract_pdf(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".md", ".markdown"}:
        if m := _H1.search(text):
            return m.group(1), text
    return path.stem, text


def _extract_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        # Page-joined rather than block-joined: chunking re-flows this anyway,
        # and page order is the only structure a PDF reliably carries.
        return "\n\n".join(page.get_text() for page in doc)
