# Known follow-ups

Recorded 2026-07-26, after the v0.1.0 whole-branch review. These are deliberate
deferrals with reasoning, not forgotten work. Nothing here blocks deployment;
the first one blocks *trusting the defaults*.

---

## 1. Retrieval tuning — RESOLVED 2026-07-26, with residue

Measured against 28 verified questions over a 156-chunk corpus of real technical
notes, using the real embedding model. Full method and numbers in
[TUNING.md](TUNING.md).

Outcome: `arm_depth` 50 -> **10**; `rrf_k` stays 60, measured flat for k >= 10.
`arm_depth` was the lever all along — BM25's IDF already ranks stopword-only
matches low, so truncating the arm removes them before they can vote, and no
stopword list is needed. Chunk-level fused MRR@10 rose 0.492 -> 0.529 and hit
rate improved at every k measured. Document-level scoring was 1.000 for every
arm and every setting on this corpus, so it discriminated nothing.

`arm_depth=5` scores better still at MRR and at hit rate for k <= 5, but caps
the candidate pool at 10 and makes three questions unreachable at any depth.
Picking 10 over 5 is a judgment call at the margin, documented as such.

**Still open from that work:**

- **Per-document diversity cap.** Results are chunk-level with no cap, so one
  long note can monopolise the result set — measured at **9 of 10 slots** from a
  single document. This matters most for the planned chat-context-injection
  consumer, and it is invisible to document-level eval scoring.
- **Relevance floor.** `vector_arm` returns `arm_depth` nearest neighbours
  unconditionally, and nearest is not near. An out-of-domain query still gets
  confident-looking results, and `Hit.score` is an unnormalised RRF value
  (~0.016-0.033) a caller cannot threshold on. Surfacing raw vector distance on
  `Hit` would let the caller decide.
- **Re-run on the real corpus.** `arm_depth=10` is 6% of 156 chunks; on a
  20,000-chunk corpus the same absolute value is 0.05%, which may be too few.
  The measurement does not automatically transfer.

## 2. Sweeping `rrf_k` / `arm_depth` — RESOLVED 2026-07-26

Both are now overridable via `RAG_RRF_K` and `RAG_ARM_DEPTH`, specifically so
the harness can sweep them without editing source. A non-numeric value fails
loudly rather than falling back to the default, so a sweep cannot silently
report numbers for settings it never used.

## 3. `rebuild-vectors` has an unguarded `zip`

`cli.py` zips chunks against vectors without a length check — the same failure
mode that was guarded in `index.py`. Lower severity here: this path deletes all
vectors first, so a short batch leaves `vectors < chunks`, which `rag stats`
does warn about. In the index path the drop happened in pairs, which defeated
that warning entirely; that is why only the index path was fixed.

## 4. Fence tracking ignores delimiter type and length

`chunk.py` toggles on any ``` or ~~~ without matching the opening delimiter's
type or length. A ``` block containing a `~~~` line closes early and can produce
a phantom heading; likewise a ````-delimited block containing ``` lines.
CommonMark requires matching type and length.

Narrower than the bug it replaced — it needs a document that mixes or nests
fence styles. An unterminated fence swallowing subsequent headings is *correct*
per CommonMark and is not a defect.

## 5. Smaller items

- `Embedder.health()` has no test, and it backs `/healthz`, which monitoring
  will consume.
- No test pins uppercase suffix handling (`.MD`, `.PDF`), though both
  `index.py` and `extract.py` do lowercase correctly.
- No test for a nested subdirectory corpus, though nesting was verified working.
- `db.py`'s dimension guard reads only `meta.embed_dim`; if that row is deleted
  while `chunks_vec` still exists, a mismatched config is accepted and fails
  later as an opaque vec0 error. Reading the declared width from `sqlite_master`
  as a second source would close it.

---

## Measured and rejected

- **Embedding `heading_path`** — improves the dense arm 15% overall but makes it
  worse at exact identifiers, and leaves fused output unchanged at the shipped
  settings while making h@3 worse. See TUNING.md Result 4. Revisit only if arm
  weighting is introduced or a real corpus proves far more heading-dense.

## Resolved since this list was written

- **`Settings.from_env()` coercion untested** — `tests/test_config.py` now covers
  coercion, empty strings, non-numeric values and sub-1 values, all of which
  raise.
- **No explicit tie-break in fusion** — `rag.search.fuse()` sorts on
  `(-score, chunk_id)`. This also removed a class of bug rather than just a
  wart: the previous implicit tie-break was why an offline sweep and the live
  service ranked differently.
- **Sweep tooling was unshipped** — `eval/sweep.py` now lives in the repo and
  imports production fusion, so `docs/TUNING.md` can actually be regenerated.
- **Eval could silently shrink its sample** — `evaluate()` raises when a
  question's phrase no longer resolves to a chunk.
