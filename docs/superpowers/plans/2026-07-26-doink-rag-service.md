# doink-rag Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A hybrid keyword + dense retrieval service over personal notes and PDFs, backed by one SQLite file, with embeddings fetched over HTTP from a `llama-server --embedding` instance.

**Architecture:** Python package in `~/git/doink-rag`. One SQLite file holds `documents`, `chunks`, an FTS5 index and a `sqlite-vec` vector table. Ingestion extracts and chunks files, embeds them via HTTP, and writes all three. Search runs both arms and fuses with Reciprocal Rank Fusion in a single SQL statement. A CLI drives indexing; FastAPI exposes search.

**Tech Stack:** Python 3.12 (uv-managed), `sqlite-vec`, `httpx`, `fastapi`, `uvicorn`, `pymupdf`, `pytest`. No ML libraries — embeddings are an HTTP call.

**Spec:** `doink-brain-playbooks/docs/superpowers/specs/2026-07-26-rag-retrieval-design.md`

## Global Constraints

- **Python MUST come from uv, not pyenv.** Verified 2026-07-26: pyenv's 3.11.15 and 3.14.6 on this Mac both lack `sqlite3.Connection.enable_load_extension`, so `sqlite-vec` cannot load. uv-managed 3.12.13 has it. Pin with `.python-version` containing `3.12` and always run via `uv run`. This is not a style preference — the dense arm does not function otherwise.
- **Verified working versions:** `sqlite-vec` v0.1.9, SQLite 3.53.1 (Mac, uv) and 3.46.1 (box). FTS5 with `porter unicode61` present on both.
- **Embedding model:** Qwen3-Embedding-0.6B, **1024 dimensions**, served by `llama-server --embedding --pooling last` at `http://127.0.0.1:8092`. Model name and dimension are config, never literals.
- **Query/document asymmetry is mandatory.** Queries get `Instruct: {task}\nQuery: {text}`; documents get no prefix. Wrong prefixing does not error, it silently degrades retrieval.
- **FTS5 tokenizer is `porter unicode61 remove_diacritics 2`.** Not `trigram`.
- **RRF constant k=60**, top-50 per arm, both configurable.
- **FTS5 `rank` is negative** (more negative = better); `ORDER BY rank` ascending yields best-first.
- **Chunk budget ~512 tokens, ~64 overlap**, approximated as characters at 4 chars/token to avoid a tokenizer dependency. Documented as an approximation, not a measurement.
- **No network calls in tests.** The embedding client is faked; one optional integration test is marked and skipped by default.
- Comment style: explain *why*, not *what*. Terse and direct, matching `doink-brain-playbooks`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | deps, entry point, pytest config |
| `.python-version` | `3.12` — forces a uv-managed interpreter |
| `src/rag/config.py` | `Settings` dataclass; env overrides; model name + dimension |
| `src/rag/db.py` | connection, extension loading, schema DDL, migrations |
| `src/rag/extract.py` | file → `(title, text)` for md / txt / pdf |
| `src/rag/chunk.py` | text → `list[Chunk]` with heading paths |
| `src/rag/embed.py` | HTTP client for `/v1/embeddings`; batching; asymmetric prefixes |
| `src/rag/index.py` | ingestion orchestration, change detection, deletion reconciliation |
| `src/rag/search.py` | keyword arm, dense arm, RRF fusion |
| `src/rag/cli.py` | `index`, `search`, `stats`, `rebuild-vectors` |
| `src/rag/api.py` | FastAPI app: `POST /search`, `GET /healthz` |
| `eval/questions.yaml` | eval set |
| `eval/run_eval.py` | recall@k and MRR per arm and fused |

---

## Task 1: Repo scaffold and the extension-loading gate

The first test exists to fail loudly on the wrong interpreter. Everything else depends on it.

**Files:**
- Create: `~/git/doink-rag/{pyproject.toml,.python-version,.gitignore,README.md}`
- Create: `src/rag/__init__.py`, `src/rag/config.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `Settings` dataclass with fields `db_path: Path`, `corpus_dir: Path`, `embed_url: str`, `embed_model: str`, `embed_dim: int`, `chunk_chars: int`, `chunk_overlap_chars: int`, `rrf_k: int`, `arm_depth: int`, `query_instruction: str`; classmethod `Settings.from_env() -> Settings`.

- [ ] **Step 1: Initialise the repo**

```bash
cd ~/git/doink-rag
git init
printf '3.12\n' > .python-version
printf '__pycache__/\n*.pyc\n.venv/\n*.db\n*.db-*\n.pytest_cache/\n' > .gitignore
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "doink-rag"
version = "0.1.0"
description = "Hybrid keyword + dense retrieval over personal notes"
requires-python = ">=3.12"
dependencies = [
    "sqlite-vec>=0.1.9",
    "httpx>=0.27",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "pymupdf>=1.24",
    "pyyaml>=6.0",
]

[project.scripts]
rag = "rag.cli:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["integration: needs a live embedding server; deselected by default"]
addopts = "-m 'not integration'"
```

- [ ] **Step 3: Write the failing environment test**

`tests/test_environment.py`:

```python
"""The whole dense arm depends on loadable SQLite extensions.

pyenv-built Pythons on macOS ship without them, and the failure mode is a
confusing AttributeError deep inside db.connect(). Fail here instead, with a
message that says what to do.
"""
import sqlite3


def test_sqlite_supports_loadable_extensions():
    assert hasattr(sqlite3.Connection, "enable_load_extension"), (
        "This interpreter cannot load SQLite extensions, so sqlite-vec will not "
        "work. pyenv builds Python without them by default. Run under uv: "
        "`uv run pytest`."
    )
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)


def test_fts5_with_porter_is_available():
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')"
    )


def test_sqlite_vec_loads():
    import sqlite_vec

    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    (version,) = con.execute("SELECT vec_version()").fetchone()
    assert version.startswith("v")
```

- [ ] **Step 4: Run it — expect failure before deps are installed**

Run: `uv run pytest tests/test_environment.py -v`
Expected: `sqlite_vec` import error, then passing once uv resolves dependencies.

- [ ] **Step 5: Write `src/rag/config.py`**

```python
"""Settings. Every tunable lives here; nothing downstream hardcodes a value.

embed_dim in particular is load-bearing: it is baked into the vec0 table's DDL,
so changing the model means rebuilding that table and re-embedding everything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("rag.db")
    corpus_dir: Path = Path("corpus")

    embed_url: str = "http://127.0.0.1:8092"
    embed_model: str = "Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_batch: int = 16

    # ~512 tokens at ~4 chars/token. An approximation, deliberately: a real
    # tokenizer would be a dependency and a network hop for a sizing decision
    # that only needs to be roughly right.
    chunk_chars: int = 2048
    chunk_overlap_chars: int = 256

    rrf_k: int = 60
    arm_depth: int = 50

    # Qwen3 embedding is asymmetric: queries carry an instruction, documents
    # never do. Getting this wrong degrades silently.
    query_instruction: str = (
        "Given a search query, retrieve relevant passages from personal notes"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        d = {}
        if v := os.getenv("RAG_DB_PATH"):
            d["db_path"] = Path(v)
        if v := os.getenv("RAG_CORPUS_DIR"):
            d["corpus_dir"] = Path(v)
        if v := os.getenv("RAG_EMBED_URL"):
            d["embed_url"] = v
        if v := os.getenv("RAG_EMBED_MODEL"):
            d["embed_model"] = v
        if v := os.getenv("RAG_EMBED_DIM"):
            d["embed_dim"] = int(v)
        return cls(**d)
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold with an environment gate for loadable sqlite extensions"
```

---

## Task 2: Database schema

**Files:**
- Create: `src/rag/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: `connect(settings: Settings) -> sqlite3.Connection` (extension loaded, foreign keys on, `Row` factory); `create_schema(con, settings) -> None`; `schema_dim(con) -> int | None` returning the dimension the vec table was built with.

- [ ] **Step 1: Write the failing test**

`tests/test_db.py`:

```python
import sqlite3
import pytest
from rag.config import Settings
from rag.db import connect, create_schema, schema_dim


@pytest.fixture
def con(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", embed_dim=8)
    c = connect(s)
    create_schema(c, s)
    return c


def test_schema_creates_all_tables(con):
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"documents", "chunks", "chunks_fts", "chunks_vec"} <= names


def test_schema_dim_reports_configured_dimension(con):
    assert schema_dim(con) == 8


def test_fts_trigger_mirrors_inserts(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'the memory fit problem','Notes')")
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'memory'").fetchall()
    assert [r[0] for r in rows] == [1]


def test_fts_trigger_handles_delete(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'swapping is bad','Notes')")
    con.execute("DELETE FROM chunks WHERE id=1")
    assert con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'swapping'"
    ).fetchone()[0] == 0


def test_porter_stemming_is_active(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'the process was swapping badly','Notes')")
    # 'swap' must match 'swapping'; this is the whole reason for porter over trigram
    assert con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'swap'"
    ).fetchone()[0] == 1


def test_deleting_document_cascades_to_chunks(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'x','y')")
    con.execute("DELETE FROM documents WHERE id=1")
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'rag.db'`

- [ ] **Step 3: Write `src/rag/db.py`**

```python
"""Schema and connections.

One file holds everything: documents, chunks, the FTS5 index and the vectors.
That is the entire reason this design needs no vector database — sqlite-vec is
an extension, not a service.
"""
from __future__ import annotations

import sqlite3

from .config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  content_hash  TEXT NOT NULL,
  mtime         REAL NOT NULL,
  size          INTEGER NOT NULL,
  title         TEXT,
  indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id           INTEGER PRIMARY KEY,
  document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal      INTEGER NOT NULL,
  text         TEXT NOT NULL,
  heading_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- external-content FTS5: the index stores no copy of the text, so chunks is
-- the single source of truth and the triggers below keep them in step.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading_path,
  content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading_path)
  VALUES (new.id, new.text, new.heading_path);
END;

-- External-content tables need the 'delete' command with the OLD values,
-- otherwise the index keeps a phantom entry that matches queries forever.
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
  VALUES ('delete', old.id, old.text, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
  VALUES ('delete', old.id, old.text, old.heading_path);
  INSERT INTO chunks_fts(rowid, text, heading_path)
  VALUES (new.id, new.text, new.heading_path);
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(settings: Settings) -> sqlite3.Connection:
    con = sqlite3.connect(settings.db_path)
    con.row_factory = sqlite3.Row
    try:
        con.enable_load_extension(True)
    except AttributeError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "This Python cannot load SQLite extensions, so sqlite-vec is "
            "unavailable. pyenv builds Python without them. Run under uv."
        ) from exc
    import sqlite_vec

    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def create_schema(con: sqlite3.Connection, settings: Settings) -> None:
    con.executescript(SCHEMA)
    # vec0 bakes the dimension into its DDL, so it cannot be created from a
    # constant. Changing models means dropping this table and re-embedding.
    con.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{settings.embed_dim}])"
    )
    con.execute(
        "INSERT INTO meta(key,value) VALUES('embed_dim',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(settings.embed_dim),),
    )
    con.commit()


def schema_dim(con: sqlite3.Connection) -> int | None:
    row = con.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone()
    return int(row[0]) if row else None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_db.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(db): schema with external-content FTS5 and vec0"
```

---

## Task 3: Extraction

**Files:**
- Create: `src/rag/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Produces: `extract(path: Path) -> tuple[str, str]` returning `(title, text)`; `SUPPORTED_SUFFIXES: set[str]` = `{".md", ".markdown", ".txt", ".pdf"}`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL, no module `rag.extract`.

- [ ] **Step 3: Write `src/rag/extract.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_extract.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(extract): markdown, text and pdf extraction"
```

---

## Task 4: Chunking

**Files:**
- Create: `src/rag/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Chunk` with `text: str`, `heading_path: str`, `ordinal: int`; `chunk_text(text: str, settings: Settings) -> list[Chunk]`.

- [ ] **Step 1: Write the failing test**

```python
from rag.chunk import chunk_text, Chunk
from rag.config import Settings

S = Settings(chunk_chars=200, chunk_overlap_chars=40)


def test_splits_on_markdown_headings():
    text = "# A\n\nalpha content\n\n# B\n\nbeta content\n"
    chunks = chunk_text(text, S)
    paths = [c.heading_path for c in chunks]
    assert "A" in paths and "B" in paths


def test_heading_path_is_hierarchical():
    text = "# Top\n\nintro\n\n## Inner\n\ndetail here\n"
    chunks = chunk_text(text, S)
    assert any(c.heading_path == "Top > Inner" for c in chunks)


def test_long_section_is_split_with_overlap():
    body = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(f"# H\n\n{body}\n", S)
    assert len(chunks) > 1
    assert all(len(c.text) <= S.chunk_chars * 1.5 for c in chunks)
    # overlap means the tail of one chunk reappears at the head of the next
    assert chunks[0].text[-20:] in chunks[1].text


def test_ordinals_are_sequential_from_zero():
    text = "# A\n\n" + " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, S)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_plain_text_without_headings_still_chunks():
    chunks = chunk_text("just prose, no headings at all", S)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""


def test_empty_text_produces_no_chunks():
    assert chunk_text("   \n\n  ", S) == []
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: FAIL, no module `rag.chunk`.

- [ ] **Step 3: Write `src/rag/chunk.py`**

```python
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
    at least one chunk.
    """
    if len(body) <= settings.chunk_chars:
        return [body]

    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= settings.chunk_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-settings.chunk_overlap_chars:]
            current = f"{tail}\n\n{para}"
    if current:
        pieces.append(current)

    # A single paragraph longer than the budget is still one piece; splitting
    # mid-sentence to satisfy an approximate budget trades real meaning for a
    # number that was a guess anyway.
    return pieces
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(chunk): heading-aware chunking with overlap"
```

---

## Task 5: Embedding client

**Files:**
- Create: `src/rag/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Produces: `class Embedder` with `__init__(self, settings: Settings, client: httpx.Client | None = None)`, `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`, and `health() -> bool`.

- [ ] **Step 1: Write the failing test**

```python
import httpx
import pytest
from rag.config import Settings
from rag.embed import Embedder

S = Settings(embed_dim=4, embed_batch=2)


def _client(capture: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        capture.append(body)
        inputs = body["input"]
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0, 0.0, 0.0, 0.0]} for _ in inputs]
        })
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_documents_get_no_instruction_prefix():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_documents(["plain body text"])
    assert seen[0]["input"] == ["plain body text"]


def test_query_gets_the_instruction_prefix():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_query("how does swapping happen")
    sent = seen[0]["input"][0]
    assert sent.startswith("Instruct: ")
    assert "Query: how does swapping happen" in sent


def test_batching_respects_embed_batch():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_documents(["a", "b", "c"])
    assert [len(c["input"]) for c in seen] == [2, 1]


def test_dimension_mismatch_is_rejected():
    def handler(request):
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})
    e = Embedder(S, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError, match="dimension"):
        e.embed_documents(["x"])


def test_empty_input_makes_no_request():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    assert e.embed_documents([]) == []
    assert seen == []
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL, no module `rag.embed`.

- [ ] **Step 3: Write `src/rag/embed.py`**

```python
"""Embeddings over HTTP.

Deliberately the only thing that talks to a model, and deliberately not an ML
library: llama-server already runs on this box, so embedding is an HTTP call and
this package needs no torch, no onnx and no transformers.
"""
from __future__ import annotations

import httpx

from .config import Settings


class Embedder:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._client = client or httpx.Client(
            base_url=settings.embed_url, timeout=120.0
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Documents carry no instruction prefix. Adding one here would make
        document vectors incomparable with query vectors."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embed_batch):
            batch = texts[start:start + self.settings.embed_batch]
            out.extend(self._post(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Queries MUST carry the instruction. Qwen3 embedding is asymmetric and
        omitting this silently costs retrieval quality."""
        prompt = f"Instruct: {self.settings.query_instruction}\nQuery: {text}"
        return self._post([prompt])[0]

    def health(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def _post(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        response = self._client.post(
            "/v1/embeddings",
            json={"model": self.settings.embed_model, "input": inputs},
        )
        response.raise_for_status()
        vectors = [row["embedding"] for row in response.json()["data"]]
        for vector in vectors:
            if len(vector) != self.settings.embed_dim:
                raise ValueError(
                    f"embedding dimension {len(vector)} does not match "
                    f"configured {self.settings.embed_dim}; the served model is "
                    f"not the one this database was built for"
                )
        return vectors
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embed.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(embed): http embedding client with asymmetric prompting"
```

---

## Task 6: Ingestion

**Files:**
- Create: `src/rag/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `connect`, `create_schema`, `extract`, `chunk_text`, `Embedder`.
- Produces: `@dataclass IndexResult` with `added: int`, `updated: int`, `removed: int`, `unchanged: int`; `index_corpus(con, settings, embedder) -> IndexResult`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus


class FakeEmbedder:
    """Returns a deterministic vector per text so tests never touch a network."""
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4,
                 chunk_chars=400, chunk_overlap_chars=40)
    con = connect(s)
    create_schema(con, s)
    return con, s, corpus


def test_indexes_a_new_file(env):
    con, s, corpus = env
    (corpus / "a.md").write_text("# A\n\nthe memory fit problem\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 1
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] >= 1
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] >= 1


def test_unchanged_file_is_not_re_embedded(env):
    con, s, corpus = env
    (corpus / "a.md").write_text("# A\n\nstable content\n")
    index_corpus(con, s, FakeEmbedder())
    e = FakeEmbedder()
    r = index_corpus(con, s, e)
    assert r.unchanged == 1 and r.added == 0
    assert e.calls == 0, "re-embedding unchanged content wastes the whole budget"


def test_modified_file_is_reindexed(env):
    con, s, corpus = env
    p = corpus / "a.md"
    p.write_text("# A\n\noriginal\n")
    index_corpus(con, s, FakeEmbedder())
    p.write_text("# A\n\ncompletely different words\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.updated == 1
    hits = con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'original'"
    ).fetchone()[0]
    assert hits == 0, "stale chunks must not survive a reindex"


def test_deleted_file_is_removed_from_index(env):
    con, s, corpus = env
    p = corpus / "a.md"
    p.write_text("# A\n\ndoomed content\n")
    index_corpus(con, s, FakeEmbedder())
    p.unlink()
    r = index_corpus(con, s, FakeEmbedder())
    assert r.removed == 1
    assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0


def test_unsupported_files_are_skipped(env):
    con, s, corpus = env
    (corpus / "x.docx").write_text("nope")
    (corpus / "a.md").write_text("# A\n\nyes\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 1
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL, no module `rag.index`.

- [ ] **Step 3: Write `src/rag/index.py`**

```python
"""Ingestion.

A library function, not a CLI command: the CLI calls it today and the planned
`POST /documents` upload endpoint will call the same thing.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunk import chunk_text
from .config import Settings
from .extract import SUPPORTED_SUFFIXES, extract


@dataclass
class IndexResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def index_corpus(con: sqlite3.Connection, settings: Settings, embedder) -> IndexResult:
    result = IndexResult()
    on_disk: dict[str, Path] = {
        str(p.relative_to(settings.corpus_dir)): p
        for p in sorted(settings.corpus_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    }
    known = {
        row["path"]: (row["id"], row["content_hash"])
        for row in con.execute("SELECT id, path, content_hash FROM documents")
    }

    for rel, path in on_disk.items():
        digest = _hash(path)
        if rel in known and known[rel][1] == digest:
            result.unchanged += 1
            continue
        if rel in known:
            # Delete and reinsert rather than diff chunks: content changed, so
            # chunk boundaries may have moved and old chunks are meaningless.
            con.execute("DELETE FROM documents WHERE id=?", (known[rel][0],))
            result.updated += 1
        else:
            result.added += 1
        _index_file(con, settings, embedder, rel, path, digest)

    for rel, (doc_id, _) in known.items():
        if rel not in on_disk:
            con.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            result.removed += 1

    con.commit()
    return result


def _index_file(con, settings, embedder, rel: str, path: Path, digest: str) -> None:
    title, text = extract(path)
    chunks = chunk_text(text, settings)
    stat = path.stat()

    cur = con.execute(
        "INSERT INTO documents(path,content_hash,mtime,size,title,indexed_at) "
        "VALUES (?,?,?,?,?,?)",
        (rel, digest, stat.st_mtime, stat.st_size, title,
         datetime.now(timezone.utc).isoformat()),
    )
    doc_id = cur.lastrowid
    if not chunks:
        return

    vectors = embedder.embed_documents([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        cur = con.execute(
            "INSERT INTO chunks(document_id,ordinal,text,heading_path) "
            "VALUES (?,?,?,?)",
            (doc_id, chunk.ordinal, chunk.text, chunk.heading_path),
        )
        con.execute(
            "INSERT INTO chunks_vec(chunk_id,embedding) VALUES (?,?)",
            (cur.lastrowid, _pack(vector)),
        )
```

- [ ] **Step 4: Handle the vec cascade**

`chunks_vec` is a virtual table and is not covered by `ON DELETE CASCADE`. Add this trigger to `SCHEMA` in `src/rag/db.py`, immediately after `chunks_au`:

```sql
CREATE TRIGGER IF NOT EXISTS chunks_ad_vec AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_index.py -v`
Expected: 5 passed. `test_deleted_file_is_removed_from_index` fails without the Step 4 trigger — confirm it passes now.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(index): ingestion with hash change detection and deletion reconciliation"
```

---

## Task 7: Hybrid search

**Files:**
- Create: `src/rag/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `@dataclass Hit` with `chunk_id: int`, `score: float`, `text: str`, `heading_path: str`, `path: str`, `title: str`; `search(con, settings, embedder, query: str, limit: int = 10) -> list[Hit]`; also `keyword_arm(con, settings, query) -> list[int]` and `vector_arm(con, settings, vector) -> list[int]` returning ranked chunk ids.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus
from rag.search import search, keyword_arm


class FakeEmbedder:
    """Vector arm returns [1,0,0,0] for everything except 'tls', which is
    orthogonal — so the two arms disagree and RRF has something to fuse."""
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(t):
        return [0.0, 1.0, 0.0, 0.0] if "tls" in t.lower() else [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mem.md").write_text("# Memory\n\nthe repack buffer was swapping badly\n")
    (corpus / "net.md").write_text("# Network\n\ntls termination happens in tailscale\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, FakeEmbedder())
    return con, s


def test_keyword_arm_finds_stemmed_match(env):
    con, s = env
    ids = keyword_arm(con, s, "swap")
    assert len(ids) == 1


def test_search_returns_hits_with_provenance(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "swapping")
    assert hits
    assert hits[0].path == "mem.md"
    assert hits[0].title == "Memory"
    assert "repack" in hits[0].text


def test_fusion_ranks_documents_matched_by_both_arms_first(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "tls")
    assert hits[0].path == "net.md", "tls matches both arms and must rank first"


def test_query_with_no_keyword_match_still_returns_vector_hits(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "zzzznotawordanywhere")
    assert hits, "the dense arm must still contribute when BM25 finds nothing"


def test_limit_is_respected(env):
    con, s = env
    assert len(search(con, s, FakeEmbedder(), "the", limit=1)) <= 1


def test_fts_special_characters_do_not_crash(env):
    con, s = env
    # Unescaped, this is an FTS5 syntax error rather than zero results.
    search(con, s, FakeEmbedder(), 'what is "memory fit" AND (swap)')
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL, no module `rag.search`.

- [ ] **Step 3: Write `src/rag/search.py`**

```python
"""Hybrid retrieval.

The two arms fail differently: BM25 misses vocabulary mismatch ("why is it slow"
vs a note titled "the memory-fit problem"), and dense retrieval blurs exact
identifiers like b10075 or --no-repack. RRF is how both get a vote.
"""
from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass

from .config import Settings

_FTS_SAFE = re.compile(r'[^\w\s]')


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    heading_path: str
    path: str
    title: str


def _sanitise(query: str) -> str:
    """FTS5 MATCH takes a query language, not a string. User text containing
    quotes, parentheses or bare AND/OR raises rather than returning nothing, so
    reduce it to bare terms."""
    return " ".join(_FTS_SAFE.sub(" ", query).split())


def keyword_arm(con: sqlite3.Connection, settings: Settings, query: str) -> list[int]:
    terms = _sanitise(query)
    if not terms:
        return []
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY rank LIMIT ?",
        (terms, settings.arm_depth),
    ).fetchall()
    return [r[0] for r in rows]


def vector_arm(con: sqlite3.Connection, settings: Settings,
               vector: list[float]) -> list[int]:
    blob = struct.pack(f"{len(vector)}f", *vector)
    rows = con.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (blob, settings.arm_depth),
    ).fetchall()
    return [r[0] for r in rows]


def search(con: sqlite3.Connection, settings: Settings, embedder,
           query: str, limit: int = 10) -> list[Hit]:
    keyword_ids = keyword_arm(con, settings, query)
    vector_ids = vector_arm(con, settings, embedder.embed_query(query))

    # Reciprocal Rank Fusion: score = sum over arms of 1/(k + rank). Computed in
    # Python rather than SQL because both arms are already materialised as small
    # ranked lists, and the SQL version is harder to read for no gain.
    scores: dict[int, float] = {}
    for ids in (keyword_ids, vector_ids):
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    if not ordered:
        return []

    placeholders = ",".join("?" * len(ordered))
    rows = {
        r["id"]: r
        for r in con.execute(
            f"SELECT c.id, c.text, c.heading_path, d.path, d.title "
            f"FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({placeholders})",
            [cid for cid, _ in ordered],
        )
    }
    return [
        Hit(cid, score, rows[cid]["text"], rows[cid]["heading_path"] or "",
            rows[cid]["path"], rows[cid]["title"] or "")
        for cid, score in ordered
        if cid in rows
    ]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_search.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(search): hybrid retrieval with reciprocal rank fusion"
```

---

## Task 8: CLI

**Files:**
- Create: `src/rag/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; subcommands `index`, `search`, `stats`, `rebuild-vectors`.

- [ ] **Step 1: Write the failing test**

```python
from rag.cli import main


def test_stats_on_empty_db(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(tmp_path))
    assert main(["stats"]) == 0
    assert "documents" in capsys.readouterr().out


def test_index_reports_counts(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nhello world\n")
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("RAG_EMBED_DIM", "4")
    monkeypatch.setattr("rag.cli._embedder", lambda s: _Fake())
    assert main(["index"]) == 0
    assert "added=1" in capsys.readouterr().out


def test_unknown_command_returns_nonzero(capsys):
    assert main(["nope"]) != 0


class _Fake:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL, no module `rag.cli`.

- [ ] **Step 3: Write `src/rag/cli.py`**

```python
"""Command line entry point.

`rebuild-vectors` is first-class rather than an afterthought: the vec0 table
bakes its dimension into DDL, so changing embedding model requires dropping and
rebuilding it, and that will happen every time a model is evaluated.
"""
from __future__ import annotations

import argparse
import sys

from .config import Settings
from .db import connect, create_schema, schema_dim
from .embed import Embedder
from .index import index_corpus
from .search import search


def _embedder(settings: Settings):
    return Embedder(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="sync the index with the corpus directory")
    sub.add_parser("stats", help="show index contents")
    sub.add_parser("rebuild-vectors", help="drop and re-embed every chunk")
    p_search = sub.add_parser("search", help="hybrid search")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--limit", type=int, default=10)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)

    settings = Settings.from_env()
    con = connect(settings)
    create_schema(con, settings)

    if args.command == "index":
        result = index_corpus(con, settings, _embedder(settings))
        print(f"added={result.added} updated={result.updated} "
              f"removed={result.removed} unchanged={result.unchanged}")
        return 0

    if args.command == "stats":
        docs = con.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vectors = con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]
        print(f"documents={docs} chunks={chunks} vectors={vectors} "
              f"dim={schema_dim(con)} model={settings.embed_model}")
        if chunks != vectors:
            print("WARNING: chunk and vector counts differ; the dense arm is "
                  "missing entries. Run `rag rebuild-vectors`.", file=sys.stderr)
        return 0

    if args.command == "rebuild-vectors":
        embedder = _embedder(settings)
        con.execute("DELETE FROM chunks_vec")
        rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        import struct
        for start in range(0, len(rows), settings.embed_batch):
            batch = rows[start:start + settings.embed_batch]
            vectors = embedder.embed_documents([r["text"] for r in batch])
            for row, vector in zip(batch, vectors):
                con.execute(
                    "INSERT INTO chunks_vec(chunk_id,embedding) VALUES (?,?)",
                    (row["id"], struct.pack(f"{len(vector)}f", *vector)),
                )
        con.commit()
        print(f"re-embedded {len(rows)} chunks")
        return 0

    if args.command == "search":
        for hit in search(con, settings, _embedder(settings), args.query, args.limit):
            location = f"{hit.path}" + (f" :: {hit.heading_path}" if hit.heading_path else "")
            print(f"{hit.score:.4f}  {location}")
            print(f"        {hit.text[:160].strip()}")
        return 0

    return 2
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(cli): index, search, stats and rebuild-vectors"
```

---

## Task 9: HTTP API

**Files:**
- Create: `src/rag/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `create_app(settings: Settings | None = None, embedder=None) -> FastAPI` with `POST /search` (body `{"query": str, "limit": int}`) and `GET /healthz`.

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient
from rag.api import create_app
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus


class Fake:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def health(self):
        return True


def _client(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nthe memory fit problem\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())
    con.close()
    return TestClient(create_app(s, Fake()))


def test_search_returns_hits(tmp_path):
    r = _client(tmp_path).post("/search", json={"query": "memory"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["path"] == "a.md"
    assert "score" in body["hits"][0]


def test_healthz_reports_index_and_embedder(tmp_path):
    r = _client(tmp_path).get("/healthz")
    assert r.status_code == 200
    assert r.json()["embedder"] is True
    assert r.json()["documents"] == 1


def test_empty_query_is_rejected(tmp_path):
    assert _client(tmp_path).post("/search", json={"query": "  "}).status_code == 422
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL, no module `rag.api`.

- [ ] **Step 3: Write `src/rag/api.py`**

```python
"""HTTP surface.

Binds loopback in deployment; nothing off-box reaches it directly. Every planned
consumer — chat context injection, the family agent harness — talks to this same
endpoint, which is why it exists before any of them do.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI

from .config import Settings
from .db import connect, create_schema
from .embed import Embedder
from .search import search


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


def create_app(settings: Settings | None = None, embedder=None) -> FastAPI:
    settings = settings or Settings.from_env()
    embedder = embedder or Embedder(settings)
    app = FastAPI(title="doink-rag")

    def _con():
        # A connection per request: SQLite connections are not thread-safe
        # across FastAPI's threadpool, and opening one is microseconds.
        con = connect(settings)
        create_schema(con, settings)
        return con

    @app.post("/search")
    def do_search(request: SearchRequest):
        con = _con()
        try:
            hits = search(con, settings, embedder, request.query, request.limit)
            return {"query": request.query, "hits": [vars(h) for h in hits]}
        finally:
            con.close()

    @app.get("/healthz")
    def healthz():
        con = _con()
        try:
            docs = con.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            vectors = con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]
            return {
                "documents": docs,
                "chunks": chunks,
                "vectors": vectors,
                "vectors_complete": chunks == vectors,
                "embedder": embedder.health(),
            }
        finally:
            con.close()

    return app


app = create_app  # uvicorn factory target: uvicorn rag.api:app --factory
```

- [ ] **Step 4: Add the test client dependency**

```bash
uv add --dev httpx  # already present as a runtime dep; TestClient needs it
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(api): search and health endpoints"
```

---

## Task 10: Evaluation harness

**Files:**
- Create: `eval/questions.yaml`, `eval/run_eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `load_questions(path) -> list[Question]` where `Question` has `question: str` and `expect_paths: list[str]`; `evaluate(con, settings, embedder, questions, k: int = 10) -> dict[str, ArmScore]` where `ArmScore` has `recall: float` and `mrr: float`, keyed `"keyword"`, `"vector"`, `"fused"`.

- [ ] **Step 1: Write the seed question file**

`eval/questions.yaml` — replace these with real questions once a corpus exists; the format is what matters:

```yaml
# Each question names the document(s) that should be retrieved. Recall@k asks
# whether any expected path appears in the top k; MRR rewards ranking it first.
- question: why does the box get slow under memory pressure
  expect_paths: [architecture.md]
- question: what terminates TLS for the inference API
  expect_paths: [architecture.md]
```

- [ ] **Step 2: Write the failing test**

```python
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval"))
from run_eval import evaluate, Question


class Fake:
    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    @staticmethod
    def _v(t):
        return [0.0, 1.0, 0.0, 0.0] if "tls" in t.lower() else [1.0, 0.0, 0.0, 0.0]


def test_evaluate_scores_all_three_arms(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "mem.md").write_text("# Memory\n\nthe repack buffer was swapping\n")
    (corpus / "net.md").write_text("# Net\n\ntls termination in tailscale\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    scores = evaluate(con, s, Fake(),
                      [Question("swapping", ["mem.md"])], k=5)
    assert set(scores) == {"keyword", "vector", "fused"}
    assert scores["keyword"].recall == 1.0
    assert 0.0 <= scores["fused"].mrr <= 1.0
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL, no module `run_eval`.

- [ ] **Step 4: Write `eval/run_eval.py`**

```python
"""Measure the arms separately and fused.

Hybrid was chosen without a bake-off, so this is not a gate. It is the
instrument that says whether rrf_k suits this corpus, whether one arm is dead
weight, and whether a larger embedding model earns its RAM on real questions
rather than on MTEB's average.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.config import Settings          # noqa: E402
from rag.db import connect, create_schema  # noqa: E402
from rag.embed import Embedder            # noqa: E402
from rag.search import keyword_arm, search, vector_arm  # noqa: E402


@dataclass
class Question:
    question: str
    expect_paths: list[str]


@dataclass
class ArmScore:
    recall: float = 0.0
    mrr: float = 0.0


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text()) or []
    return [Question(d["question"], d["expect_paths"]) for d in data]


def _paths_for(con, chunk_ids: list[int]) -> list[str]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = {
        r["id"]: r["path"]
        for r in con.execute(
            f"SELECT c.id, d.path FROM chunks c JOIN documents d "
            f"ON d.id=c.document_id WHERE c.id IN ({placeholders})", chunk_ids)
    }
    return [rows[cid] for cid in chunk_ids if cid in rows]


def _score(ranked_paths: list[str], expected: list[str]) -> tuple[float, float]:
    for position, path in enumerate(ranked_paths, start=1):
        if path in expected:
            return 1.0, 1.0 / position
    return 0.0, 0.0


def evaluate(con, settings: Settings, embedder, questions: list[Question],
             k: int = 10) -> dict[str, ArmScore]:
    totals = {name: [0.0, 0.0] for name in ("keyword", "vector", "fused")}
    for q in questions:
        runs = {
            "keyword": _paths_for(con, keyword_arm(con, settings, q.question)[:k]),
            "vector": _paths_for(
                con, vector_arm(con, settings, embedder.embed_query(q.question))[:k]),
            "fused": [h.path for h in search(con, settings, embedder, q.question, k)],
        }
        for name, paths in runs.items():
            recall, mrr = _score(paths, q.expect_paths)
            totals[name][0] += recall
            totals[name][1] += mrr

    n = max(len(questions), 1)
    return {name: ArmScore(r / n, m / n) for name, (r, m) in totals.items()}


def main() -> int:
    settings = Settings.from_env()
    con = connect(settings)
    create_schema(con, settings)
    questions = load_questions(Path(__file__).parent / "questions.yaml")
    scores = evaluate(con, settings, Embedder(settings), questions)
    print(f"{len(questions)} questions, model={settings.embed_model}, "
          f"rrf_k={settings.rrf_k}\n")
    print(f"{'arm':<10} {'recall@10':>10} {'MRR':>8}")
    for name, score in scores.items():
        print(f"{name:<10} {score.recall:>10.3f} {score.mrr:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -v`
Expected: all tests pass across every file.

- [ ] **Step 6: Write the README**

Cover: the uv-only Python constraint and *why* (pyenv lacks loadable extensions), quickstart, env vars, how to point at the box's embedding endpoint over the tailnet, the model/dimension coupling to `chunks_vec`, and how to run the eval.

- [ ] **Step 7: Commit and tag**

```bash
git add -A
git commit -m "feat(eval): recall and MRR per arm and fused"
git tag v0.1.0
```

The tag matters: `doink-brain-playbooks` deploys by pinning `rag_version`, so there must be something to pin.

---

## Task 11: Live integration check

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the marked test**

```python
"""Deselected by default (addopts = -m 'not integration').

Run against a live embedding server once llama_embed is deployed:
    RAG_EMBED_URL=http://doink-brain:8092 uv run pytest -m integration -v
"""
import pytest
from rag.config import Settings
from rag.embed import Embedder


@pytest.mark.integration
def test_live_embedding_dimension_matches_config():
    settings = Settings.from_env()
    embedder = Embedder(settings)
    assert embedder.health(), f"no embedding server at {settings.embed_url}"
    vector = embedder.embed_query("does the model swap under load")
    assert len(vector) == settings.embed_dim, (
        f"served model returns {len(vector)} dims but config says "
        f"{settings.embed_dim}; chunks_vec was built for the configured value"
    )


@pytest.mark.integration
def test_query_and_document_embeddings_differ():
    """If prefixing is broken, these collapse to the same vector and retrieval
    quietly degrades."""
    settings = Settings.from_env()
    embedder = Embedder(settings)
    text = "tailscale serve terminates TLS"
    assert embedder.embed_query(text) != embedder.embed_documents([text])[0]
```

- [ ] **Step 2: Confirm it is deselected**

Run: `uv run pytest -v`
Expected: integration tests deselected, everything else passes.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: live embedding integration checks, deselected by default"
```

---

## Self-Review

**Spec coverage:** schema (Task 2), chunking (4), retrieval + RRF (7), API (9), eval (10), model asymmetry (5), `porter` tokenizer (2), config-driven dimension (1, 2, 8). The corpus-sync play, `llamacpp_build`, `llama_embed` and `rag_service` belong to the second plan in `doink-brain-playbooks` and are deliberately absent here.

**Placeholder scan:** none. `eval/questions.yaml` ships two seed entries that are explicitly labelled as replaceable — the format is the deliverable, not the content, since real questions require a real corpus.

**Type consistency:** `Settings` field names are identical across Tasks 1–10. `Chunk(text, heading_path, ordinal)` is produced in Task 4 and consumed in Task 6. `Hit` is produced in Task 7 and consumed in Tasks 8, 9 and 10. `embed_documents`/`embed_query`/`health` appear with the same signatures in Tasks 5, 6, 7, 9 and 10, and every fake in the tests implements them.

**Verified before writing, not assumed:**
1. pyenv Python cannot load SQLite extensions; uv-managed 3.12.13 can — hence the Task 1 gate.
2. `sqlite-vec` v0.1.9 loads, and `vec0` KNN works with `MATCH ? AND k = ?`.
3. FTS5 `porter` stems "swapping" → "swap".
4. RRF over both arms ranks a both-arms match above a single-arm match.

**Known risks left to implementation:**
1. `chunks_vec` is a virtual table, so `ON DELETE CASCADE` does not reach it. Task 6 Step 4 adds the trigger; the deletion test fails without it, deliberately.
2. FTS5 `MATCH` takes a query language, not a string — unescaped user punctuation raises. `_sanitise` strips it; the Task 7 test covers it.
3. External-content FTS5 needs the `'delete'` command carrying OLD values on delete/update, or the index keeps phantom rows. Covered by the Task 2 triggers and tested.
