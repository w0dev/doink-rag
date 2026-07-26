# Hybrid retrieval service — design

Date: 2026-07-26
Status: approved design, not yet implemented

A local retrieval service for personal notes and documents, served from the
inference box. Hybrid keyword + dense retrieval over one SQLite file, embeddings
from a second `llama-server` instance, deployed by `doink-brain-playbooks`.

This document records *why*; the code and roles are the source of truth for
*what*.

---

## Context

The box (`doink-brain`, Ubuntu 26.04) currently runs one service: `llama-server`
with a resident ~21 GB Qwen3.6-35B-A3B, loopback-bound, reached through
`tailscale serve`. Baseline hardening, patching, firewall and SSH are under
Ansible as of 2026-07-26.

`docs/architecture.md` already recorded a position on this work:

> **Embedding sidecar** — `llama-server` is one model per process, so retrieval
> needs a second unit. Deferred pending a proper look at RAG / hybrid retrieval.
> Worth checking whether SQLite FTS5 full-text search is sufficient before adding
> vector infrastructure; on a personal-scale corpus it often is.

That caution targets **vector infrastructure** — Qdrant, Weaviate, pgvector — each
a second service to run, patch and firewall. This design does not add any of
those. `sqlite-vec` is an extension loaded into a database file: no daemon, no
port, no unit. The warning is honoured by not building what it warned against,
not by forgoing dense retrieval.

### The binding constraint: memory

| | |
|---|---|
| Usable RAM | ~29 GiB |
| `llama-server` RSS | 21.2 GiB (mmap'd, file-backed) |
| System | ~2 GiB |
| **Headroom** | **~5.8 GiB** |

`free` reports ~26 GiB "available", which is misleading: the model is mmap'd, so
its 21 GiB *is* the page cache. Anonymous allocations do not come from free
memory — they evict model pages, which then re-read from NVMe mid-generation.
That is exactly the failure `llama_server/tasks/verify.yml` detects with its
`VmSwap` assert.

Every sizing decision below is bounded by that ~5.8 GiB, not by 26.

---

## Decisions

| # | Decision | Rejected |
|---|---|---|
| 1 | Separate `doink-rag` repo; playbooks deploy it pinned by tag | Monorepo; conflates infra and application lifecycles |
| 2 | v1 is local files only | Google Drive in v1 — OAuth, sync semantics and a secret before any evidence retrieval works |
| 3 | Hybrid BM25 + dense with RRF from the start | FTS5-first bake-off; user chose to skip straight to the known-good baseline |
| 4 | Hand-rolled Python + SQLite | LlamaIndex/Haystack — abstracts exactly the retrieval logic that needs tuning |
| 5 | Embeddings from a second `llama-server --embedding` | In-process ONNX or sentence-transformers; both add an ML dependency stack to a box that already has an inference engine |
| 6 | Qwen3-Embedding-0.6B Q8_0, model and dimension config-driven | 8B at any quant does not fit; 4B is a measured upgrade path |
| 7 | Extract `llamacpp_build`; `llama_server` and `llama_embed` as siblings | Generalising `llama_server`; ~60% of a merged role would be conditionals |
| 8 | Ansible play pushes the corpus from the Mac, then triggers indexing | Syncthing — a daemon, a web UI and a patch surface, to replace one rsync task |
| 9 | FTS5 `porter unicode61` tokenizer | `trigram` — no stemming, 2–3× index size; wrong default for English prose |
| 10 | Qwen3-Embedding over EmbeddingGemma, Jina v5, Harrier-OSS | see "Embedding model" — the deciding factors are the upgrade ladder and licensing, not quality |

### Why not llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) is healthy and well-made
(MIT, Go, actively released). It solves loading and unloading models to fit
limited hardware. This box's central commitment is the opposite: one 21 GB model,
permanently resident, never evicted — unloading it costs a 15–20 s cold reload.

Adopting it here means running a swapping tool with swapping disabled, as a
static supervisor for two models that both stay resident forever, in exchange for
a Go binary to pin and patch, a second configuration language beside Ansible, a
proxy hop on every embedding call, and a failure mode where the proxy takes out
both models at once. It would replace roughly 40 lines of systemd unit template.

It would genuinely win with several models that cannot be co-resident — vision,
code models, size experiments, a reranker. **That future stays open:** llama-swap
launches `llama-server` processes, so adopting it later changes what starts the
services, not the roles that install them. A cheap decision to reverse is a poor
one to pre-pay.

---

## Repo boundary

**`doink-rag`** (new) — the service. Python, own tests, own release tags.
**`doink-brain-playbooks`** — deployment only, pinned by `rag_version`, exactly as
`llamacpp_version: b10075` pins llama.cpp today.

The seam is a git tag. The playbook repo never sees Python source, only a version
to check out and build. This mirrors a pattern the repo already runs: a role that
deploys an upstream project at a pinned tag into a versioned directory behind a
stable symlink, with rollback by repointing the symlink.

Justification for the split: the lifecycles genuinely differ (infra changes are
rare and carry lockout risk; the service changes hourly during development), the
tooling does not overlap (`ansible-playbook --syntax-check` vs pytest/ruff), and
the playbooks repo was *just* given the identity "how the box is built."

---

## Architecture

```
Mac ──ansible-playbook corpus-sync.yml (rsync over tailnet)──> /var/lib/rag/corpus/
                                       │  …then the same play triggers:
                                 ┌─────▼─────┐
                                 │  ingest   │  extract → chunk → hash → embed
                                 └─────┬─────┘
                                       │ POST /v1/embeddings
                                 ┌─────▼──────────────┐
                                 │ llama-server :8092 │  Qwen3-Embedding-0.6B
                                 │ --embedding        │  --pooling last
                                 └─────┬──────────────┘
                                       ▼
                                    rag.db
                         documents │ chunks │ chunks_fts │ chunks_vec
                                       │
                                 ┌─────▼─────┐
                                 │ FastAPI   │  :8093, loopback
                                 └───────────┘
```

Ports 8092 and 8093 bind loopback only, like 8091. **`roles/firewall` requires no
changes** — nothing new is externally reachable, and anything that should be
reachable goes through `tailscale serve`.

---

## The service

### Layout

```
pyproject.toml
src/rag/
  config.py     settings; model name and dimension live here
  db.py         schema creation, migrations, connection handling
  extract.py    file -> (text, title); markdown, txt, PDF
  chunk.py      text -> chunks with heading paths
  embed.py      HTTP client for /v1/embeddings; batching, retries
  index.py      ingestion orchestration; change detection
  search.py     keyword arm, dense arm, RRF fusion
  api.py        FastAPI app
  cli.py        index / search / stats / rebuild-vectors
tests/
eval/
  questions.yaml
  run_eval.py
```

### Schema

Three tables sharing `rowid`, following the structure described in
[hybrid-rag-200-lines](https://media.patentllm.org/blog/database/hybrid-rag-200-lines),
with the tokenizer changed:

```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  content_hash TEXT NOT NULL,     -- sha256; unchanged files cost a stat and a hash
  mtime REAL, size INTEGER, title TEXT, indexed_at TEXT
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER, text TEXT NOT NULL,
  heading_path TEXT               -- "Architecture > The memory-fit problem"
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, heading_path,
  content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
  chunk_id INTEGER PRIMARY KEY,
  embedding FLOAT[1024]           -- generated from the configured model's dimension
);
```

Triggers keep `chunks_fts` synchronised on insert/update/delete.

`heading_path` is indexed alongside text in both arms: it is high-signal and free
to carry.

**The vector table's dimension is baked into DDL.** Switching models means
dropping and rebuilding `chunks_vec` and re-embedding everything. The schema is
generated from config and `rag rebuild-vectors` is a first-class command, not an
afterthought.

The blog post indexes whole documents and specifies no chunking. That is
insufficient for PDFs, so chunking is designed below rather than inherited.

### Chunking

Structure first, size second. Markdown splits on heading boundaries; sections
longer than the budget split by paragraph. PDFs extract per page, then merge
paragraphs to the same budget. Every chunk carries its heading path.

Budget: ~512 tokens with ~64 token overlap, both config values. The exact numbers
matter less than the principle — a chunk that is one coherent section retrieves
far better than a fixed window straddling two topics.

PDF extraction uses **PyMuPDF**, materially better than pypdf on real-world
layouts. It is AGPL, which is irrelevant for a personal service that is never
distributed, but it is a deliberate choice rather than an accident.

### Retrieval

Both arms retrieve 50; results are fused in one SQL statement:

```
score(chunk) = Σ 1 / (k + rank_in_arm)      k = 60
```

`k`, per-arm weights and arm depth are configuration, not literals. Setting them
is what the eval harness is for.

**Qwen3 embedding is asymmetric.** Queries carry an instruction prefix; documents
must not:

```
Instruct: Given a search query, retrieve relevant passages from personal notes
Query: {text}
```

Getting this wrong does not error. It silently returns worse results, which is
the worst kind of bug to have in a retrieval system.

### API

- `POST /search` — `{query, k}` → ranked chunks with scores and source paths
- `GET /healthz` — liveness, plus embedding-server reachability
- Indexing is CLI and timer driven, not an API endpoint

### Evaluation

`eval/questions.yaml` holds 20–30 real questions with known-correct source
documents. `run_eval.py` reports recall@k and MRR for keyword-only, dense-only
and fused.

Hybrid was chosen directly rather than after a bake-off, so this is not a gate.
It is the instrument that says whether `k=60` suits this corpus, whether one arm
is dead weight, and later whether the 4B model's benchmark advantage appears on
real data. Without it, every tuning decision is guesswork.

---

## Embedding model

Measured GGUF sizes and MTEB English v2 scores:

| Model | Dim | Quant | File | Est. total footprint | MTEB | Verdict |
|---|---|---|---|---|---|---|
| 0.6B | 1024 | Q8_0 | 0.64 GB | ~1.0 GB | 70.70 | **chosen** |
| 4B | 2560 | Q4_K_M | 2.50 GB | ~3.5 GB | 74.60 | measured upgrade path |
| 4B | 2560 | Q8_0 | 4.28 GB | ~5.0 GB | 74.60 | too tight |
| 8B | 4096 | Q4_K_M | 4.68 GB | ~5.9 GB | 75.22 | **exceeds headroom** |
| 8B | 4096 | Q8_0 | 8.05 GB | — | 75.22 | impossible |

Totals include llama.cpp context overhead, the Python service (~0.4 GB) and a
flat `vec0` index (~82 MB at 1024 dims / 20k chunks).

**8B is ruled out on memory**, and would be a poor trade regardless: 1.9× the RAM
of the 4B for **+0.62 MTEB**.

**0.6B is chosen to start** for two reasons beyond fit. Embedding is compute-bound
on 6 cores and the 4B is ~7× the work, which makes a full re-index a multi-hour
job — punishing while chunking is still changing and everything gets re-embedded
repeatedly. And MTEB is an average over benchmarks that are not these notes; the
4B's +3.9 should be demonstrated on this corpus before it is paid for. Model and
dimension are configuration, so that test is a config change plus a re-index.

`--pooling last` is **mandatory**: Qwen3 embedding uses last-token pooling and
llama.cpp defaults to mean. Wrong pooling does not error, it silently degrades.
`llama_embed` asserts the flag exists before writing the unit, in the same spirit
as `llama_server/tasks/flag_gate.yml`.

### Why Qwen3 over the alternatives

The field was re-examined on 2026-07-26 rather than assumed:

| Model | Q8 GGUF | Dims | Context | License | MTEB Eng v2 | dl/30d |
|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 639 MB | 1024 | 32k | Apache-2.0 | 70.70 | 10.7M |
| EmbeddingGemma-300M | 334 MB | 768 | 2k | Gemma Terms | 69.67 | 1.7M |
| Jina v5 small-retrieval | 639 MB | — | — | CC-BY-NC-4.0 | ~71.7 (family) | 1.7k |
| Jina v5 nano-retrieval | 233 MB | — | — | CC-BY-NC-4.0 | — | 652 |
| Harrier-OSS-v1-0.6b | 639 MB | — | — | MIT | — | 669 |

Jina v5 small-retrieval and Harrier-OSS 0.6b are both **639.4 MB at Q8** — the
same as Qwen3-Embedding-0.6B's 639.2 MB. They are near-certainly Qwen3-0.6B
fine-tunes; the ecosystem converged on this base.

Quality is not the deciding factor: 70.70 vs 69.67 is roughly a point, and
third-party retrieval comparisons favouring EmbeddingGemma contradict the
official MTEB figures. Neither settles anything about this corpus. The reasons
that actually decide it:

- **The upgrade ladder.** This design plans to measure 0.6B and possibly move to
  4B. Qwen3 offers 0.6B → 4B → 8B with the same prompt format, pooling and code.
  Every alternative is a single size, so upgrading means changing families —
  pooling, prompts and dimensions all at once.
- **Licensing.** Apache-2.0 is unambiguous for something the family will use.
  Jina v5 is non-commercial; Gemma Terms carry restrictions that must be passed
  along.
- **Context.** 32k versus EmbeddingGemma's 2k. Moot at 512-token chunks, but 2k
  forecloses ever embedding larger units.
- **Adoption.** 10.7M downloads a month means pitfalls like the pooling flag are
  already documented by someone else.

**What would change this:** EmbeddingGemma is half the compute. If CPU indexing
throughput becomes the binding pain on six shared cores, it is the right switch,
and it is a config change plus a re-index.

The eval harness makes this empirical rather than a judgment call. Its first real
job should be a three-way comparison — Qwen3-0.6B, EmbeddingGemma-300M,
Qwen3-4B — on real questions from this corpus.

---

## Deployment

Roles added to `doink-brain-playbooks`:

| Role | Responsibility |
|---|---|
| `llamacpp_build` | **extracted** from `llama_server`: clone, cmake, versioned dir, stable symlink |
| `llama_server` | modified: keeps chat model, MTP, sampling, smoke test; depends on the build |
| `llama_embed` | new: embedding model, `--pooling last`, own unit and port |
| `rag_service` | new: clone pinned tag, venv, systemd unit, verify |

Plus `playbooks/corpus-sync.yml` — pushes the corpus from the Mac and triggers
indexing. A play, not a role: it has no convergent state to maintain, and it is
deliberately not part of `site.yml`.

### Why the build is extracted

An embedding server uses none of `llama_spec_type`, `llama_spec_draft_n_max`, the
sampling parameters, `llama_reasoning_format`, `llama_context_shift`,
`llama_parallel`, the MTP flag gate, or the chat smoke test. It needs
`--embedding`, `--pooling last` and possibly `--ubatch-size` tuning that the chat
server never wants.

Roughly 40% shared surface, 60% conditionals — the signal that these are two
roles, not one parameterised role. What they actually share is the build, so that
is what becomes a role. It also makes the dependency explicit rather than hiding
it behind an assert, which is what `README.md`'s "no cross-role variable
references" rule is really protecting.

The refactor is contained: the build tasks are already self-contained, the
versioned-directory-plus-symlink design makes rollback trivial, and
`--tags llamacpp_build` verifies it in isolation before anything depends on it.

### Corpus delivery: a play, not a daemon

`playbooks/corpus-sync.yml` pushes from the Mac with `ansible.posix.synchronize`
and then runs `rag index` on the box. No daemon, no web UI, no port, no extra
patch surface — one task and a command, in a repo whose whole point is that
changes to the box are deliberate and recorded.

Both halves are naturally idempotent and compose well: rsync ships only changed
files, and content-hash change detection means only changed chunks are
re-embedded. Running it twice is nearly free.

`delete: true` — a note deleted on the Mac is removed from the box and therefore
from the index. Ingestion reconciles `documents` against what is on disk, so a
removed file's chunks are deleted rather than lingering as unreachable results.

**The trigger is explicit, and that is the point.** The corpus is only as fresh
as the last run. This was chosen over Syncthing deliberately: background
reconciliation would keep it fresher, at the cost of a daemon on a box kept
otherwise spare.

**Forward path.** Ingestion is a library function in `index.py`, called by the
CLI today. The planned family agent harness wants a `+` button that uploads and
ingests immediately; that becomes a thin `POST /documents` wrapper over the same
function, not a redesign. The genuinely new question family use raises is
multi-tenancy — who may search what — which v1 does not attempt and does not
foreclose.

---

## Explicitly not doing

- **Google Drive** — OAuth, rate limits, sync semantics and Google-native export
  quirks, plus a stored refresh token on a box deliberately kept free of
  `ansible-vault`. A v2 subsystem, not a v1 folder.
- **Code repositories** — needs syntax-aware chunking, a second strategy. v3.
- **A reranker** — Qwen3-Reranker would improve precision, but adds a third
  resident model and a second inference hop. Revisit when the eval harness shows
  ranking, not recall, is the bottleneck.
- **A vector database** — see the memory constraint and the architecture doc's
  own warning.
- **llama-swap** — see Decisions.
- **An agent harness** — every consumer of retrieval needs the same HTTP
  interface, so the harness decision is deferrable at zero cost. Build the API;
  decide the consumer when there is something worth consuming.
- **File watching** — a full re-index at this scale is minutes, and content-hash
  change detection makes re-runs cheap. A timer avoids a stateful daemon,
  debouncing and the half-written-file race.

---

## Open questions

1. **Chunk size and overlap are guesses** (~512/64) until the eval set exists.
   They are configuration precisely because the first real measurement will
   probably move them.
2. **Whether `porter` beats `trigram` on this corpus** is asserted, not measured.
   It is the right default for English prose; if code repos arrive in v3,
   substring matching on identifiers may argue for a second FTS table rather than
   changing this one.
3. **`--ubatch-size` for long chunks.** llama.cpp has required `n_ubatch >=
   n_tokens` for embeddings in some versions. To be verified against the pinned
   build rather than assumed.
4. **Indexing contends with inference for the same 6 cores.** A large re-index
   while serving chat will slow both. Whether that needs `nice`, a cgroup slice,
   or simply running overnight is deferred until it is observed.

---

## Implementation sequence

Two plans, two repos, in this order — the playbooks cannot deploy a tag that does
not exist:

1. **`doink-rag`** — schema, extraction, chunking, embed client, both
   retrieval arms, RRF, CLI, API, eval harness. Testable entirely on the Mac
   against a local SQLite file and the box's embedding endpoint over the tailnet.
2. **`doink-brain-playbooks`** — extract `llamacpp_build`, add `llama_embed` and
   `rag_service`, plus `corpus-sync.yml`. Staged like the baseline rollout: the build
   refactor verified alone before anything depends on it.
