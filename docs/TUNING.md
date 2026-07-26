# Retrieval tuning — the 2026-07-26 measurement

Why `rrf_k` and `arm_depth` are set where they are, and how to redo this when
the corpus changes.

## Setup

- **Corpus:** 8 real technical documents (~24,800 words) from the
  `doink-brain-playbooks` and `doink-rag` repositories — architecture notes,
  design specs, implementation plans, READMEs. 156 chunks, 218 fenced code
  blocks. Genre-matched to the intended corpus: markdown, shell fences, flag
  names, build tags.
- **Questions:** 28, in `eval/questions.yaml`. Every ground-truth phrase was
  verified to exist in the named document before use — the first draft had 3
  wrong and 1 too broad, all caught by that check.
- **Embeddings:** Qwen3-Embedding-0.6B Q8_0 via `llama-server --embedding
  --pooling last`, 1024 dims, L2-normalised. Real model, not a fake.
- **Scoring:** chunk level. Document-level recall over 8 documents is 1.000 for
  both arms and tells you nothing.

Questions are tagged by the arm they should favour, so per-arm behaviour can be
read rather than assumed: `vocab` (vocabulary mismatch), `exact` (literal
identifier), `natural` (ordinary multi-word phrasing).

## Result 1 — the AND join was catastrophic, and this quantifies it

MRR@10, chunk level, each arm alone:

| arm | vocab | exact | natural | ALL |
|---|---|---|---|---|
| keyword, AND-joined | **0.000** | 0.473 | **0.077** | **0.154** |
| keyword, OR-joined | 0.375 | 0.487 | 0.511 | 0.466 |
| dense | 0.333 | **0.300** | 0.615 | 0.456 |

AND-joining required a chunk to contain *every* query term, scoring exactly zero
on vocabulary-mismatch questions. The keyword arm was dead for everything except
literal identifiers, and RRF kept producing plausible scores throughout.

The table also confirms the premise of the two-arm design: the dense arm is
**worst** on exact identifiers (0.300, below keyword's 0.487), because embeddings
blur precisely the tokens that need matching literally.

## Result 2 — `arm_depth` is the lever; `rrf_k` is inert

Fused MRR@10 across the grid (OR join):

| rrf_k \ arm_depth | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| 5 | 0.555 | 0.532 | 0.509 | 0.509 | 0.513 | 0.513 |
| 10 | 0.555 | 0.529 | 0.508 | 0.510 | 0.513 | 0.513 |
| 60 | 0.555 | 0.529 | 0.495 | 0.490 | 0.495 | 0.495 |
| 100 | 0.555 | 0.529 | 0.495 | 0.490 | 0.495 | 0.495 |

Reading down a column, `rrf_k` barely moves anything — at depth 10, k=10 and
k=60 are identical to three decimals. Reading across a row, depth moves
everything.

The mechanism: BM25's IDF already ranks stopword-only matches near the bottom.
At `arm_depth=50` those low-value candidates were still admitted to fusion and
still got a vote worth `1/(60+50) = 0.0091` — more than half of a rank-1 vote's
`0.0164`. Truncating the arm at 10 removes them before they can vote. `rrf_k`
only reshapes the curve *among candidates already admitted*, so it cannot undo
what `arm_depth` lets in.

**Consequence: no stopword list is needed.** That was considered and is
unnecessary — IDF plus truncation already does the job.

## Result 3 — why 10 and not 5

`arm_depth=5` scores the highest MRR, but it is a trap:

| config | MRR@10 | r@1 | r@3 | r@5 | r@10 | r@20 |
|---|---|---|---|---|---|---|
| OR / 60 / 50 (old) | 0.495 | 0.357 | 0.571 | 0.679 | 0.786 | 0.893 |
| **OR / 60 / 10 (now)** | **0.529** | **0.393** | **0.607** | **0.714** | **0.821** | **0.929** |
| OR / 60 / 5 | 0.555 | 0.393 | 0.750 | 0.786 | 0.786 | 0.786 |

Two arms at depth 5 can produce at most 10 unique candidates, so recall is
**frozen at 0.786** however deep you look — r@10 and r@20 cannot improve. Depth
10 keeps the ceiling open and still beats the old setting at every k.

Per-question against the old setting, depth 10 is **7 better, 1 worse, 20
unchanged**; the single regression is small (0.250 → 0.200). Depth 5 is also 7/1
but its regression is severe (1.000 → 0.333).

One honest cost: document-level recall@10 slips from 1.000 to 0.964 — one
question's correct *document* leaves the top 10 while chunk-level results
improve. A net win, not a free one.

## Result 4 — weighted RRF is not worth it

At depth 20, `w_vec=1.5` buys +0.019 MRR and costs r@5 (0.643 vs 0.679).
`w_kw=2` is worse on MRR. Equal votes are fine; the complexity is not justified.

## Settings that came out of this

```python
rrf_k: int = 60      # measured inert; the original RRF paper's value
arm_depth: int = 10  # was 50
```

Both are now overridable — `RAG_RRF_K` and `RAG_ARM_DEPTH` — specifically so
this can be re-swept without editing source.

## Limitations — read before trusting these numbers

- **28 questions over 156 chunks is small.** Differences under ~0.03 MRR are
  within noise. The depth 50 → 10 change is worth acting on because it moves
  *every* chunk-level metric in the same direction and wins 7 questions to 1,
  not because 0.529 beats 0.495.
- **`arm_depth` is corpus-size dependent.** 10 of 156 chunks is 6%; on a 20,000
  chunk corpus the same absolute value is 0.05%, which may be too few. Re-run
  rather than assuming this transfers.
- **The corpus is this project's own documentation.** Genre-matched to the
  target, but not the actual notes, and written by the same author as the
  questions.
- **Not measured:** per-document diversity (one long note can still dominate the
  result set) and any relevance floor. Both remain open in `FOLLOWUPS.md`.

## Redoing this

```bash
export RAG_CORPUS_DIR=/path/to/corpus RAG_DB_PATH=/path/to/rag.db
export RAG_EMBED_URL=http://127.0.0.1:8092
rag index
python eval/run_eval.py                      # current settings
RAG_ARM_DEPTH=20 python eval/run_eval.py     # sweep without editing source
```

Watch the **chunk** columns. If keyword chunk MRR collapses toward zero, the FTS
query semantics have regressed — that is the failure this whole exercise exists
to prevent.
