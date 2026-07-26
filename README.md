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

`tests/test_environment.py` asserts this precondition directly
(`test_sqlite_supports_loadable_extensions`, `test_sqlite_vec_loads`) so the
failure mode is a clear assertion, not a confusing error deep inside `vec0`.

Run everything through `uv run`, e.g.:

```bash
uv run pytest -v
uv run rag index
```

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

Other tunables (`rrf_k`, `arm_depth`, chunking sizes, the query instruction
prefix) are constructor defaults on `Settings` rather than env vars — they
are retrieval-quality knobs to tune deliberately via the eval harness below,
not per-deployment configuration.

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

## Running the evaluation harness

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
# scores == {"keyword": ArmScore(recall=..., mrr=...), "vector": ..., "fused": ...}
```

## Tests

```bash
uv run pytest -v
```

Tests marked `integration` need a live embedding server and are deselected
by default (`addopts = "-m 'not integration'"` in `pyproject.toml`). No test
in the default run makes a network call.
