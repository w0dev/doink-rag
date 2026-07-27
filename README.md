# doink-rag

Hybrid keyword + dense retrieval over personal notes. BM25 (SQLite FTS5) and a
dense-vector arm (sqlite-vec) are fused with Reciprocal Rank Fusion.

## Why `uv`, and only `uv`

This project must be run through `uv`, not a bare `pyenv`-managed interpreter,
and not by accident:

Standard `pyenv`-built Pythons are compiled **without loadable extension
support** (no `sqlite3.enable_load_extension`). `sqlite-vec` ships as a
loadable SQLite extension, not a Python wheel that does its own linking — it
has to be loaded into the SQLite C library at runtime via
`enable_load_extension(True)` + `.load_extension(...)`. Without that support
compiled in, `sqlite_vec.load(con)` cannot ever work, no matter what package
is installed. `uv`'s managed Python builds include loadable extension
support; `pyenv`'s do not. This is a fact about how each Python was compiled,
not a preference — do not "fix" this by pointing the project back at a
`pyenv` interpreter.

**It is not a version question.** pyenv's 3.14.6 and uv's 3.14.6 are the same
CPython release and behave differently — only the uv build can load
extensions. A corollary worth knowing: `uv run --python 3.14` will happily
*reuse* a pyenv 3.14 it finds on PATH and hand you a broken interpreter. Pass
`--managed-python` when it matters.

`tests/test_environment.py` asserts this precondition directly
(`test_sqlite_supports_loadable_extensions`, `test_sqlite_vec_loads`) so the
failure mode is a clear assertion, not a confusing error deep inside `vec0`.

Run everything through `uv run`, e.g.:

```bash
uv run --managed-python pytest -v
uv run --managed-python rag index
```

`.python-version` pins **3.14**, matching the Python the box runs
(`/usr/bin/python3`, 3.14.4), so development and production are the same major
version. The box needs no `uv` — its system interpreter already supports
loadable extensions.

If the bare `uv` on your `PATH` is broken or shadowed, invoke it by absolute
path instead of changing `.python-version`, `.gitignore`, or `pyproject.toml`
to work around it.

## Quickstart

```bash
# 1. Point RAG_CORPUS_DIR at a directory of markdown/text/PDF notes.
export RAG_CORPUS_DIR=~/notes

# 2. Index it (also creates the sqlite db + schema on first run).
uv run rag index

# 3. Search it.
uv run rag search "why is the box slow under memory pressure"

# 4. Inspect what's indexed.
uv run rag stats
```

## What gets indexed

`SUPPORTED_SUFFIXES` in `src/rag/extract.py` is the single source of truth:

| Extension | Title from | Text extraction |
|---|---|---|
| `.md`, `.markdown` | first `# H1`, else filename stem | UTF-8, `errors="replace"` |
| `.txt` | filename stem | UTF-8, `errors="replace"` |
| `.pdf` | filename stem | PyMuPDF, pages joined by blank lines |

Suffix matching is **case-insensitive** — `.MD`, `.PDF`, `.TXT` all index.
Subdirectories are walked recursively, and `documents.path` stores the path
relative to `RAG_CORPUS_DIR`.

### Markdown retrieves better than the same content as `.txt` or PDF

This is worth knowing before you decide how to store notes. Chunking splits on
`#` headings and carries a hierarchical `heading_path` — `Architecture > The
memory-fit problem` — which is indexed by FTS5 as its own searchable column and
shown on every hit.

`.txt` and `.pdf` have no headings, so they fall back to paragraph packing at
`chunk_chars` with an empty `heading_path`. They still work; they just retrieve
with less signal and give you no provenance beyond the filename.

### Unsupported files are skipped silently

`rag index` filters on the suffix set before extraction, so a `.docx`, `.org` or
`.png` sitting in your corpus directory is ignored — not an error, and not
reported. If a document you expected is missing from `rag stats`, check its
extension first.

(`extract()` itself raises `ValueError` on an unsupported type, but that is a
programming-error guard for direct callers, not a path `rag index` can reach.)

### PDF caveats

PyMuPDF is materially better than pypdf on real layouts, but extraction is
page-joined, not layout-aware. Multi-column papers, tables and figure captions
come out reflowed or interleaved. **There is no OCR**, so a scanned PDF yields
nothing at all and will index as an empty document.

## Environment variables

All configuration lives in `src/rag/config.py` (`Settings`); nothing
downstream hardcodes a value. `Settings.from_env()` reads:

| Variable          | Default                    | Meaning                                   |
|-------------------|-----------------------------|--------------------------------------------|
| `RAG_DB_PATH`      | `rag.db`                   | SQLite database file                      |
| `RAG_CORPUS_DIR`   | `corpus`                   | Directory of notes to index                |
| `RAG_EMBED_URL`    | `http://127.0.0.1:8092`     | Base URL of the embedding server           |
| `RAG_EMBED_MODEL`  | `Qwen3-Embedding-0.6B`      | Model name sent in embedding requests      |
| `RAG_EMBED_DIM`    | `1024`                      | Embedding vector width                     |
| `RAG_RRF_K`        | `60`                        | RRF fusion constant. Flat for k ≥ 10       |
| `RAG_ARM_DEPTH`    | `10`                        | Candidates per arm before fusion           |

`RAG_RRF_K` and `RAG_ARM_DEPTH` are retrieval-quality knobs rather than
per-deployment configuration, but they are env-overridable so the eval harness
can sweep them without editing source. `arm_depth` is the setting that controls
fusion dilution; [docs/TUNING.md](docs/TUNING.md) covers how the defaults were
measured and when to revisit them. All the integer variables reject empty,
non-numeric and sub-1 values loudly — a sweep that silently used a different
value than it printed would be worse than no sweep.

The remaining tunables (chunking sizes, the query instruction prefix) are
constructor defaults on `Settings` and are not env vars.

## Pointing at the box's embedding endpoint over the tailnet

`llama-server` (or whatever serves embeddings) runs on a box reachable over
Tailscale. Point at it with `RAG_EMBED_URL`, e.g.:

```bash
export RAG_EMBED_URL=http://doink-box.tailnet-name.ts.net:8092
```

Use the box's tailnet hostname or Tailscale IP, not a LAN address — this is
what lets indexing and search work identically whether you're on the same
LAN as the box or not. Confirm reachability before indexing a large corpus:

```bash
curl "$RAG_EMBED_URL/health"
```

`Embedder.health()` performs the same check programmatically and is used by
the API's `/healthz` endpoint.

## The embedding model and `chunks_vec` are coupled

`chunks_vec` is a `vec0` virtual table whose vector width is baked into its
`CREATE VIRTUAL TABLE ... FLOAT[N]` DDL at creation time — it cannot be
resized in place. `create_schema()` stores the dimension it was built with in
`meta` and **refuses to proceed** if `RAG_EMBED_DIM` (or a code default) no
longer matches what's on disk, rather than silently corrupting the index or
failing deep inside `vec0` with an opaque error.

Practically: **switching embedding models means rebuilding vectors.** There
is no partial-migration path.

```bash
# After changing RAG_EMBED_MODEL / RAG_EMBED_URL / RAG_EMBED_DIM to a
# different model:
rm rag.db                 # chunks_vec's old dimension can't be reused
uv run rag index           # re-extracts, re-chunks, and re-embeds everything
```

or, if the chunk boundaries in the database are still valid and only the
vectors need to change:

```bash
uv run rag rebuild-vectors
```

`rebuild-vectors` re-embeds every existing chunk without touching `documents`
or `chunks`; it is the operation you'll run every time you evaluate a
candidate embedding model, which is also why it is a first-class CLI command
rather than a manual script. Note it does not resize `chunks_vec` itself —
if the new model's dimension differs from the old one, `create_schema` will
refuse to start until `chunks_vec` is dropped and recreated at the new
width, which for now means rebuilding the whole database.

## Running the HTTP service

`src/rag/api.py` exposes the same hybrid search as a small FastAPI service.
Every planned consumer (chat context injection, the family agent harness)
talks to this one endpoint.

```bash
uv run uvicorn rag.api:app --factory --host 127.0.0.1 --port 8093
```

**Pass `--factory`.** `app` in `rag.api` is `create_app`, a function that
builds and returns the `FastAPI` instance (it reads `Settings`, opens a
connection, and runs schema setup at call time) -- it is not an app object
itself. Passing `--factory` tells uvicorn to call it with no arguments and
use the return value as the ASGI app, which is what `create_app`'s
`(settings=None, embedder=None)` signature expects.

Recent uvicorn (this project pins `uvicorn>=0.30`) will *auto-detect* a
zero-arg-callable factory and fall back to using it correctly even without
`--factory`, only logging a warning
(`ASGI app factory detected. Using it, but please consider setting the
--factory flag explicitly.`) -- so omitting the flag is not guaranteed to be
silently broken on every uvicorn version. Pass it anyway: it is the
documented, tested contract for running this service, it fails loudly at
startup instead of relying on a fallback path, and it keeps the systemd unit
correct if the pinned uvicorn version ever changes.

### `POST /search`

Request body:

```json
{"query": "why is the box slow under memory pressure", "limit": 10}
```

`query` is required and must not be blank; `limit` is optional (default 10,
must be between 1 and 100). A blank or missing `query` returns `422`.

Response body:

```json
{
  "query": "why is the box slow under memory pressure",
  "hits": [
    {
      "chunk_id": 42,
      "score": 0.0328,
      "text": "full chunk text ...",
      "heading_path": "Architecture > Memory",
      "path": "architecture.md",
      "title": "Architecture"
    }
  ]
}
```

`hits` is ordered by fused RRF score, descending. `text` is the full chunk
body, not a snippet.

### `GET /healthz`

```json
{
  "documents": 12,
  "chunks": 340,
  "vectors": 340,
  "vectors_complete": true,
  "embedder": true
}
```

`vectors_complete` is `chunks == vectors`; if it is `false`, `rag
rebuild-vectors` needs to be run (same check as the `stats` CLI command's
warning). `embedder` is whatever `Embedder.health()` reports for the
configured embedding server.

### `GET /documents`

Lists indexed documents: `{"documents": [{"path", "title", "size",
"indexed_at", "chunks"}, ...]}`.

### `POST /documents`

Multipart upload (`file` field). The file is written flat into `corpus_dir`
(client-supplied directories are stripped) and the same sync that `rag index`
runs picks it up, so an uploaded file is indistinguishable from an rsync'd
one. Returns `{"path", "index": {added, updated, removed, unchanged}}` with
status 201. Unsupported suffixes are rejected with 422; a file that fails
indexing (e.g. an unparseable PDF) is removed again so it cannot poison later
index runs.

### `DELETE /documents/{path}`

Deletes the file from `corpus_dir` (nested paths from rsync are allowed;
escaping `corpus_dir` is not) and re-syncs the index. 404 if the path does not
exist.

### Access control is the bind address, not auth

`/search` returns full note text, and there is no authentication in front of
it. The bind address is the only access control this service has: it must
stay on `127.0.0.1` (loopback) and never be given a `--host` that exposes it
on a LAN or public interface. Reach it from other machines over the tailnet
via `tailscale serve` (or an equivalent reverse-proxy-over-Tailscale setup)
pointed at the loopback port, not by rebinding the service itself.

## Running the evaluation harness

The shipped `eval/questions.yaml` holds 28 questions over this project's own
documentation, each with a verified chunk-level ground-truth phrase. It is the
baseline the current `rrf_k` / `arm_depth` defaults were measured against —
see [docs/TUNING.md](docs/TUNING.md) for the method, the numbers, and the
limitations.

Watch the **chunk** columns, not the document ones: document-level recall on a
small corpus is nearly free and hides a degenerate fused list.


`eval/run_eval.py` measures Recall@k and MRR for the keyword arm, the vector
arm, and the fused result — separately, so that a dead arm or a bad `rrf_k`
is visible rather than averaged away. This is deliberately not a merge gate:
hybrid retrieval was chosen directly, without a keyword-vs-dense bake-off.
The harness is what answers, on your own corpus, whether that choice and its
defaults (`rrf_k=60`, the current embedding model) are actually earning their
keep.

```bash
uv run python eval/run_eval.py
```

This indexes nothing itself — it queries whatever `RAG_DB_PATH` already
points at, so run `rag index` first. It reads questions from
`eval/questions.yaml`:

```yaml
- question: why does the box get slow under memory pressure
  expect_paths: [architecture.md]
```

Each entry is a natural-language question and the path(s) of the document(s)
that should answer it. Recall@k asks whether any `expect_paths` entry
appears anywhere in the top `k` results; MRR is `1 / rank` of the first such
hit (0 if it never appears).

**The seed `eval/questions.yaml` is a placeholder.** Its two entries exist to
pin down the file's *format*, not to evaluate anything real — there is no
real corpus indexed yet. Replace them with actual questions and real
document paths once notes are indexed, and grow the set over time as you
notice retrieval failures worth guarding against.

Programmatic use (e.g. from a notebook or another script):

```python
from rag.config import Settings
from rag.db import connect, create_schema
from rag.embed import Embedder
import sys, pathlib
sys.path.insert(0, "eval")
from run_eval import evaluate, load_questions

settings = Settings.from_env()
con = connect(settings)
create_schema(con, settings)
questions = load_questions(pathlib.Path("eval/questions.yaml"))
scores = evaluate(con, settings, Embedder(settings), questions, k=10)
# scores == {"keyword": ArmScore(recall, mrr, chunk_recall, chunk_mrr), "vector": ..., "fused": ...}
```

## Tests

```bash
uv run pytest -v
```

Tests marked `integration` need a live embedding server and are deselected
by default (`addopts = "-m 'not integration'"` in `pyproject.toml`). No test
in the default run makes a network call.
