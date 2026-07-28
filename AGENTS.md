# AGENTS.md — doink-rag

Agent instructions for the **hybrid retrieval service** (BM25 + dense vectors,
RRF fusion) and its FastAPI HTTP API. This is the search and corpus layer used
by the doink-brain harness.

## Sibling repos (read these when changing deploy or consumer APIs)

| Repo | Role | Local path (control node) |
|---|---|---|
| **doink-rag** (this) | Retrieval library, CLI, and `:8093` HTTP service | `~/git/doink-rag` |
| [doink-brain](https://github.com/w0dev/doink-brain) | Harness chat UI + agent; calls `/search` and `/documents` | `~/git/doink-brain` |
| [doink-brain-playbooks](https://github.com/w0dev/doink-brain-playbooks) | Ansible: builds the wheel from a git tag (`rag_version`), ships `doink-rag.service` | `~/git/doink-brain-playbooks` |

Runtime place in the stack:

```
doink-harness :8094  ──▶  doink-rag :8093  ──▶  llama-embed :8092
                              │
                              └── /var/lib/rag (db + corpus)
```

doink-rag binds **loopback only**. Off-box access is via Tailscale to the
harness (or SSH tunnels for local harness/RAG work), not by rebinding this
service. There is **no application auth** — `/search` returns full note text;
the bind address is the access control.

## What lives here vs elsewhere

- **Here:** indexing, chunking, embedding client, FTS5 + sqlite-vec, RRF
  search, `/search` / `/healthz` / `/documents` HTTP API, CLI (`rag index`,
  `rag search`, …), eval harness scaffolding.
- **Not here:** chat/agent UI, llama.cpp builds, systemd units, Tailscale
  serve. Corpus *sync from a control-node notes tree* is
  `playbooks/corpus-sync.yml` in playbooks; uploads through the harness hit
  this service’s `POST /documents` and land in the same corpus dir.

## Corpus sync vs uploads

`corpus-sync.yml` still uses **rsync with delete** (`rag_corpus_delete: true`
by default): files removed from the control-node source are removed on the box
and reindexed away. Harness/`POST /documents` uploads write into the **same**
`rag_corpus_dir`. A later deleting sync will wipe uploads that are not also
present in the sync source. When changing corpus flows, document that conflict
explicitly — do not assume uploads and deleting sync compose safely.

## Release and deploy (only when asked)

**Do not** bump versions, tag, or run deploy playbooks unless the user
explicitly asks to deploy/release.

When they do: the playbooks `rag_service` role builds a wheel from a
**detached checkout of the git tag** named by `rag_version` in
`doink-brain-playbooks/inventory/group_vars/inference.yml`
(`rag_source_repo` defaults to `~/git/doink-rag`). Dirty working-tree changes
are never what production runs.

Ship steps (only on deploy/release request):

1. Bump `pyproject.toml` version; keep the git tag equal (tests enforce
   tag ↔ package version).
2. Commit, tag `vX.Y.Z`, push commit and tag.
3. In playbooks: set `rag_version: vX.Y.Z`, then run the retrieval /
   inference-stack playbook (e.g. `playbooks/retrieval.yml` for RAG-only, or
   `inference-stack.yml` when embeddings / chat also need to move).
4. If the harness corpus UI or tools depend on new endpoints, bump and deploy
   **doink-rag before** doink-brain (`harness_version`).

## Python compatibility

- `requires-python = ">=3.14"` in `pyproject.toml` is the supported floor.
  Document and respect that annotation; do not silently target older Pythons.
- sqlite-vec needs a Python build with **SQLite loadable extensions**. That is
  a property of how the interpreter was compiled, not of the version number
  alone. `uv`’s managed CPython builds typically have it; many pyenv builds do
  not. Prefer whatever working interpreter the environment already uses that
  satisfies the pin and extension support — **do not force a specific binary
  discovery path** (`PATH` vs absolute `uv` vs system `/usr/bin/python3` on the
  box). On the box, system Python is what the venv/role uses; locally, `uv run`
  is the usual path.
- `tests/test_environment.py` asserts loadable-extension support — keep that
  green rather than papering over a bad interpreter.

## Eval questions

`eval/questions.yaml` is a **format reference / example**, not a complete
working eval set over a real corpus. **Leave it alone unless the user asks** to
change it. Do not grow it “helpfully” from retrieval bugs.

## Hard constraints

- Prefer running via `uv` / the project’s configured environment so deps and
  the Python floor match `pyproject.toml`.
- **Always start uvicorn with `--factory`:**  
  `uv run uvicorn rag.api:app --factory --host 127.0.0.1 --port 8093`
- Embedding dimension is baked into `chunks_vec`. Changing
  `RAG_EMBED_DIM` / model means rebuild (`rm` db + reindex, or
  `rag rebuild-vectors` only when dimension is unchanged).
- Do not expose the service off-loopback or add Funnel in designs.

## Local development

```bash
export RAG_CORPUS_DIR=~/notes   # or a test corpus
uv run rag index
uv run rag search "…"
uv run pytest -v   # integration tests need a live embedder; deselected by default
```

Point `RAG_EMBED_URL` at the box’s embedder over Tailscale or
`doink-brain`’s `scripts/dev-tunnels.sh` (`:8092`).

## Agent conventions for this repo

- Prefer small, test-backed changes. Default pytest excludes `integration`.
- `SUPPORTED_SUFFIXES` in `src/rag/extract.py` is the source of truth for
  indexable types; unsupported uploads must stay 422, not silent poison.
- Markdown chunks better than `.txt`/PDF (heading_path). Document that when
  changing extract/chunk behavior.
- When changing HTTP contracts the harness depends on (`/search`,
  `/documents`, `/healthz`), update doink-brain and note the pin order in
  playbooks / sibling `AGENTS.md`.
- Config is `Settings` in `src/rag/config.py` via env (`RAG_*`); do not
  hardcode paths or URLs downstream.
