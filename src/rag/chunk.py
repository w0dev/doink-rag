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
    return _pack(paragraphs, settings, joiner="\n\n", expand=_wrap_words)


def _wrap_words(para: str, settings: Settings) -> list[str]:
    """Split one oversized paragraph on whitespace, same greedy-pack shape."""
    return _pack(para.split(), settings, joiner=" ", expand=_hard_split)


def _hard_split(token: str, settings: Settings) -> list[str]:
    """Slice a single oversized token (URL, hash, base64 blob — no whitespace
    to wrap on) into budget-sized pieces with a character overlap.

    Mid-token slicing is accepted here: an unbounded chunk is worse than a
    split URL, and there is no boundary left to prefer.
    """
    stride = settings.chunk_chars - settings.chunk_overlap_chars
    if stride <= 0:
        stride = settings.chunk_chars  # degenerate settings: no overlap
    return [
        token[i : i + settings.chunk_chars]
        for i in range(0, len(token), stride)
    ]


def _pack(units: list[str], settings: Settings, joiner: str, expand) -> list[str]:
    """Greedily accumulate units up to the budget, carrying a char overlap
    into the next piece so a boundary word/sentence is never orphaned.

    A unit that alone exceeds the budget is handed to `expand`, which returns
    already-bounded, already-overlapped pieces. Those are appended directly
    rather than fed back through this same packing loop — re-packing
    pre-packed pieces would join them with another overlap-and-joiner pass on
    top of the one `expand` already applied, inflating every piece after the
    first past the budget.
    """
    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > settings.chunk_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(expand(unit, settings))
            continue
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
