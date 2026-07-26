from pathlib import Path
import pytest
from rag.extract import extract, SUPPORTED_SUFFIXES


def test_markdown_title_comes_from_first_h1(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# Memory fit\n\nThe repack buffer swaps.\n")
    title, text = extract(p)
    assert title == "Memory fit"
    assert "repack buffer" in text


def test_markdown_without_h1_falls_back_to_filename(tmp_path):
    p = tmp_path / "my-note.md"
    p.write_text("no heading here\n")
    title, _ = extract(p)
    assert title == "my-note"


def test_plain_text_uses_filename(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("plain content")
    title, text = extract(p)
    assert title == "log"
    assert text == "plain content"


def test_unsupported_suffix_raises(tmp_path):
    p = tmp_path / "x.docx"
    p.write_text("x")
    with pytest.raises(ValueError):
        extract(p)


def test_supported_suffixes_include_pdf():
    assert ".pdf" in SUPPORTED_SUFFIXES


def test_pdf_extraction(tmp_path):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "d.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "tailscale serve terminates TLS")
    doc.save(p)
    doc.close()
    title, text = extract(p)
    assert "tailscale serve" in text
