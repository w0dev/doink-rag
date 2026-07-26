"""Text to chunks.

Structure first, size second. A chunk that is one coherent section retrieves
far better than a fixed window straddling two topics, so headings are the
primary boundary and the size budget only splits what is left over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Settings

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: str
    ordinal: int


def chunk_text(text: str, settings: Settings) -> list[Chunk]:
    sections = _split_sections(text)
    out: list[Chunk] = []
    for heading_path, body in sections:
        body = body.strip()
        if not body:
            continue
        for piece in _split_to_budget(body, settings):
            out.append(Chunk(piece, heading_path, len(out)))
    return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Walk headings, tracking the active path at each level."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []
    path = ""

    for line in text.splitlines():
        if m := _HEADING.match(line):
            if buf:
                sections.append((path, "\n".join(buf)))
                buf = []
            level, title = len(m.group(1)), m.group(2)
            del stack[level - 1:]
            stack.append(title)
            path = " > ".join(stack)
        else:
            buf.append(line)
    if buf:
        sections.append((path, "\n".join(buf)))
    return sections


def _split_to_budget(body: str, settings: Settings) -> list[str]:
    """Paragraph-greedy packing, with a character overlap between pieces.

    Overlap exists so a sentence spanning a boundary is still retrievable from
    at least one chunk. A paragraph that alone exceeds the budget (e.g. one
    long unbroken block of prose) is word-wrapped rather than left oversized,
    since one chunk several times the budget would drown its own heading path
    in unrelated text.
    """
    if len(body) <= settings.chunk_chars:
        return [body]

    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    units = [
        piece
        for para in paragraphs
        for piece in (
            [para] if len(para) <= settings.chunk_chars else _wrap_words(para, settings)
        )
    ]
    return _pack(units, settings, joiner="\n\n")


def _wrap_words(para: str, settings: Settings) -> list[str]:
    """Split one oversized paragraph on whitespace, same greedy-pack shape."""
    return _pack(para.split(), settings, joiner=" ")


def _pack(units: list[str], settings: Settings, joiner: str) -> list[str]:
    """Greedily accumulate units up to the budget, carrying a char overlap
    into the next piece so a boundary word/sentence is never orphaned.
    """
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{joiner}{unit}" if current else unit
        if len(candidate) <= settings.chunk_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-settings.chunk_overlap_chars:]
            current = f"{tail}{joiner}{unit}"
    if current:
        pieces.append(current)
    return pieces
