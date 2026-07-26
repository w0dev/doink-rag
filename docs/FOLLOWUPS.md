# Known follow-ups

Recorded 2026-07-26, after the v0.1.0 whole-branch review. These are deliberate
deferrals with reasoning, not forgotten work. Nothing here blocks deployment;
the first one blocks *trusting the defaults*.

---

## 1. Retrieval tuning is unvalidated — run the eval harness before trusting `rrf_k` / `arm_depth`

**This is the one that matters.**

The keyword arm originally joined query terms with implicit AND, which meant any
multi-word question returned zero BM25 results — the hybrid retriever was
silently dense-only. That was fixed by OR-joining, and the correct document now
ranks first for both seed questions.

But OR-joining has its own cost, and it was measured rather than assumed. On a
30-chunk fixture with a deliberately vocabulary-mismatched target (dense rank 1,
zero query terms in common), query `"why is the system slow"`:

| Join | Fused rank of the correct document |
|---|---|
| AND (before) | 1 — but only because the keyword arm returned nothing at all |
| OR (now) | **4**, behind three chunks that matched only on stopwords |

The mechanism is the flatness of the RRF curve at these constants. With
`rrf_k=60` and `arm_depth=50`, keyword rank 50 scores `1/110 = 0.0091` — more
than half of rank 1's `0.0164`. A weak keyword match plus a mediocre dense rank
therefore outranks a strong dense-only hit.

Neither AND nor OR is right in the abstract. The correct settings depend on the
real corpus, which is exactly what `eval/run_eval.py` exists to measure and why
it reports each arm separately.

**Before trusting retrieval quality:** write 20–30 real questions into
`eval/questions.yaml`, run the harness, and look at the per-arm recall. If the
keyword arm's recall is high but fused recall is lower than dense-only, dilution
is real on this corpus and the mitigations below become worth building.

Mitigations deliberately not built, because the right values are unknowable
without that data:

- **Relevance floor** — `vector_arm` returns `arm_depth` nearest neighbours
  unconditionally, and nearest is not the same as near. An out-of-domain query
  gets confident-looking results. `Hit.score` is an unnormalised RRF value
  (~0.016–0.033) that a caller cannot threshold on either. Surfacing raw vector
  distance on `Hit` would let the caller decide.
- **Per-document diversity cap** — results are chunk-level with no cap, so one
  long note can monopolise the result set. Measured: a 12-section document took
  **9 of 10 slots**. This matters most for the planned chat-context-injection
  consumer, and it is invisible to the eval harness, which scores at document
  level and stops at the first match.
- **Tuning `rrf_k` / `arm_depth`** — currently not env-overridable (see below).

## 2. `rrf_k` and `arm_depth` cannot be swept without editing code

`Settings.from_env()` reads the paths, embedding URL, model and dimension, but
not the retrieval constants. The eval harness exists to answer "does `rrf_k=60`
suit this corpus", and answering it currently means editing `config.py`.

A three-line script using `dataclasses.replace` works today. Making them
env-overridable would be a small change, deferred only because the deployment
config surface should stay minimal.

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

- No unit test for `Settings.from_env()` coercion — env vars are the entire
  deployment surface Ansible drives, and `int()` coercion is untested.
- `Embedder.health()` has no test, and it backs `/healthz`, which monitoring
  will consume.
- No test pins uppercase suffix handling (`.MD`, `.PDF`), though both
  `index.py` and `extract.py` do lowercase correctly.
- No test for a nested subdirectory corpus, though nesting was verified working.
- `db.py`'s dimension guard reads only `meta.embed_dim`; if that row is deleted
  while `chunks_vec` still exists, a mismatched config is accepted and fails
  later as an opaque vec0 error. Reading the declared width from `sqlite_master`
  as a second source would close it.
- `heading_path` is indexed by FTS5 but not included in the embedded text.
  Prepending it is a one-line change, but it requires a full re-embed, so it
  belongs with the next re-index rather than as its own event.
- No explicit tie-break in fusion; ordering falls back to dict insertion order.
  More likely to bite now that the keyword arm returns wider result sets.
