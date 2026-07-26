# Known follow-ups

Deliberate deferrals with reasoning, not forgotten work. Last reconciled against
the code 2026-07-26.

Everything that can be settled without your real notes **has been**. What remains
below needs data this corpus cannot provide.

---

## Open — needs the real corpus

These are not oversights. Each one's correct answer depends on what your notes
actually look like, and guessing would be worse than waiting.

### 1. Re-run the tuning at real scale

`arm_depth=10` was measured over 156 chunks, where it is 6% of the corpus. On
20,000 chunks the same absolute number is 0.05%, which may be far too few. The
*mechanism* that made 10 right — truncating away matches BM25's IDF had already
ranked worthless — does not pin an absolute value to it.

`docs/TUNING.md` has a step-by-step revisit checklist. Start there.

### 2. Per-document diversity cap

Results are chunk-level with no per-document limit, so one long note can
monopolise the result set — measured at **9 of 10 slots** from a single
document. Whether this matters depends on your notes: a corpus of many short
notes may never hit it; one with a few very long documents will hit it
constantly.

It is also invisible to the eval harness, which stops scoring at the first
correct chunk — so document-level metrics look perfect while the fused list is
degenerate. Measure deliberately before building.

### 3. Relevance floor

`vector_arm` returns `arm_depth` nearest neighbours unconditionally, and nearest
is not the same as near. An out-of-domain question still gets confident-looking
results, and `Hit.score` is an unnormalised RRF value (~0.016–0.033) that a
caller cannot threshold on.

Surfacing raw vector distance on `Hit` is the enabling change. Choosing a
threshold needs real queries — including ones that *should* return nothing.

### 4. Re-test the embedding model

Qwen3-Embedding-0.6B was chosen over 4B on measured grounds: 8B does not fit the
RAM headroom at all, and 4B costs ~7x the CPU per index run for a benchmark gain
unproven on this corpus. Once real questions exist, `RAG_EMBED_MODEL` plus
`rag rebuild-vectors` turns that from a judgment into a measurement.

---

## Measured and rejected

Recorded so they are not re-litigated. Each has numbers in `docs/TUNING.md`.

- **Embedding `heading_path`** — improves the dense arm 15% overall
  (0.448 -> 0.514) but makes it *worse* on exact identifiers (0.300 -> 0.232),
  already its weakest category, and leaves fused output unchanged at the shipped
  settings while making h@3 worse. The arm-level gain never reaches the user
  because RRF ranks rather than scores. Revisit only if arm weighting is
  introduced or a real corpus proves far more heading-dense.
- **Stopword filtering** — a stopword-filtered keyword arm is 0.008 *worse* at
  `arm_depth=10`. BM25's IDF plus arm truncation already does the job.
- **Weighted RRF** — `w_vec=1.5` buys +0.019 MRR and costs h@5. Not worth the
  complexity.
- **`arm_depth=5`** — scores better at MRR and at k<=5, but caps the candidate
  pool at 10 and makes three questions unreachable at any depth. A close call,
  documented in TUNING Result 3. `RAG_ARM_DEPTH=5` if your consumer only ever
  reads the top few chunks.

---

## Resolved

- **Retrieval tuning** — `arm_depth` 50 -> 10, measured. See TUNING.md.
- **Sweeping `rrf_k` / `arm_depth`** — `RAG_RRF_K` and `RAG_ARM_DEPTH`, with
  empty, non-numeric and sub-1 values all rejected loudly.
- **Sweep tooling unshipped** — `eval/sweep.py` now lives in the repo and
  imports production fusion, so TUNING's tables can be regenerated. Previously a
  scratchpad script with its own tie-break measured a different ranker.
- **Eval could silently shrink its sample** — `evaluate()` raises when a
  question's phrase no longer resolves to a chunk.
- **No explicit tie-break in fusion** — `rag.search.fuse()` sorts on
  `(-score, chunk_id)`. This was the root cause of the sweep/production drift
  above, not merely a wart.
- **`rebuild-vectors` unguarded `zip`** — guarded, matching `index.py`.
- **Fence tracking ignored delimiter type and length** — a ``` block containing
  `~~~`, or a ````-delimited block containing ```, no longer closes early.
  CommonMark requires the closer to match character and be at least as long. An
  unterminated fence swallowing the rest of the document remains correct.
- **`db.py` dimension guard read only `meta.embed_dim`** — now cross-checks the
  declared width in `chunks_vec`'s own DDL from `sqlite_master`, so a deleted
  `meta` row no longer lets a mismatched config through to an opaque `vec0`
  error.
- **Missing tests** — `Settings.from_env()` coercion, `Embedder.health()`,
  uppercase suffixes, nested subdirectory corpora.

---

## Process note

The v0.1.1 tuning fix wave was reviewed, its findings were fixed, and those
fixes were **not** re-reviewed — every other change in this project got a scoped
re-review. Worth closing before building further on the retrieval core.
