# Retrieval tuning — the 2026-07-26 measurement

Why `rrf_k` and `arm_depth` are set where they are, and how to redo this when
the corpus changes.

**Every number here comes from `eval/sweep.py`, which imports `rag.search.fuse`
rather than reimplementing RRF.** The first version of this document did not:
its sweep script had its own copy of the fusion arithmetic with a different
tie-break, so every figure described a ranker that was not being shipped. That
is why `fuse()` is a separate function now.

## Setup

- **Corpus:** 8 real technical documents (~24,800 words) from the
  `doink-brain-playbooks` and `doink-rag` repositories — architecture notes,
  design specs, implementation plans, READMEs. 156 chunks, ~109 fenced code
  blocks. Genre-matched to the intended corpus: markdown, shell fences, flag
  names, build tags.
- **Questions:** 28, in `eval/questions.yaml`, each with a chunk-level
  ground-truth phrase. `evaluate()` now **raises** if a phrase resolves to no
  chunk, so the corpus and the questions cannot drift apart silently — an early
  version of this measurement lost a question that way when a corpus document
  was edited, while still printing "28 questions".
- **Embeddings:** Qwen3-Embedding-0.6B Q8_0 via `llama-server --embedding
  --pooling last`, 1024 dims, L2-normalised. Real model, not a fake.
- **Scoring:** chunk level. Document-level scoring is **1.000 for every arm and
  every setting tested** on this corpus — it cannot discriminate at all.

"Hit rate at k" below is the fraction of questions with a gold chunk in the top
k. It is deliberately not called recall: gold sets are 1–4 chunks, so recall —
the fraction of gold chunks retrieved — is a different number.

## Result 1 — the AND join was catastrophic

Chunk MRR@10, each arm alone. The AND-join column is from before the fix:

| arm | vocab | exact | natural | ALL |
|---|---|---|---|---|
| keyword, AND-joined | **0.000** | 0.473 | **0.077** | **0.154** |
| keyword, OR-joined | 0.375 | 0.487 | 0.507 | 0.464 |
| dense | 0.333 | **0.300** | 0.599 | 0.448 |

AND-joining required a chunk to contain *every* query term, scoring exactly zero
on vocabulary-mismatch questions. The keyword arm was dead for everything except
literal identifiers, and RRF kept producing plausible scores throughout.

The table also confirms the premise of the two-arm design: the dense arm is
**worst** on exact identifiers (0.300 vs keyword's 0.487), because embeddings
blur precisely the tokens that need matching literally.

Subgroup sizes are small — 8 vocab, 7 exact, 13 natural — so a single question
moves a per-kind cell by up to 0.14. Read the directions, not the decimals.

## Result 2 — `arm_depth` is the lever; `rrf_k` matters only below ~10

Fused chunk MRR@10:

| rrf_k \ arm_depth | 3 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| 1 | 0.517 | 0.551 | 0.512 | 0.493 | 0.489 | 0.489 | 0.489 |
| 5 | 0.517 | 0.560 | 0.515 | 0.495 | 0.491 | 0.494 | 0.496 |
| 10 | 0.517 | 0.560 | **0.529** | 0.508 | 0.510 | 0.512 | 0.513 |
| 60 | 0.517 | 0.560 | **0.529** | 0.499 | 0.492 | **0.492** | 0.496 |
| 100 | 0.517 | 0.560 | 0.529 | 0.499 | 0.493 | 0.493 | 0.497 |

Reading down a column: `rrf_k` is flat from 10 upward — k=10, 60 and 100 are
identical to three decimals at every depth. It is **not** inert below that;
k=1 and k=5 are measurably worse. "Inert" applies to k ≥ 10 only.

Reading across a row: depth moves everything.

The mechanism, confirmed directly: for 21 of the 28 questions the OR-joined
keyword arm matches 112–155 of the 156 chunks. BM25's IDF ranks those
stopword-only matches near the bottom, but at `arm_depth=50` they were still
admitted to fusion and still got a vote worth `1/(60+50) = 0.0091` — more than
half of a rank-1 vote's `0.0164`. Truncating the arm removes them before they
can vote.

**Consequence: no stopword list is needed.** This was checked, not assumed — a
stopword-filtered keyword arm scores −0.008 MRR at depth 10 (slightly worse) and
+0.006 at depth 50. IDF plus truncation already does the job.

Note also that depth 20, 50 and 100 all score within 0.007 of each other. The
entire effect lives between depth 3 and 20; "50" is not special.

## Result 3 — depth 5 vs 10 is a genuine judgment call

| depth | MRR@10 | h@1 | h@3 | h@5 | h@10 | h@20 |
|---|---|---|---|---|---|---|
| 3 | 0.517 | 0.357 | 0.679 | 0.786 | 0.786 | 0.786 |
| **5** | **0.560** | 0.393 | **0.750** | **0.786** | 0.821 | 0.821 |
| **10** | 0.529 | 0.393 | 0.607 | 0.714 | 0.821 | **0.929** |
| 20 | 0.499 | 0.357 | 0.607 | 0.679 | 0.786 | 0.929 |
| 50 (old) | 0.492 | 0.357 | 0.607 | 0.643 | 0.750 | 0.893 |

Both 5 and 10 beat the old 50 decisively. Between them the data does **not** pick
a winner — they trade precision at small k against reach:

- **depth 5** is better at MRR, h@3 and h@5. Its ceiling is real but higher than
  first thought: two arms × 5 gives at most 10 candidates, so h@10 and h@20 are
  both frozen at 0.821. Three questions become unreachable no matter how many
  results are requested.
- **depth 10** is better at h@20 (0.929) and keeps the ceiling open.

Per-question against depth 50: depth 5 is **8 better, 1 worse**; depth 10 is
**7 better, 1 worse**. Depth 5's single regression is severe (1.000 → 0.333);
depth 10's is mild (0.333 → 0.200).

**Chosen: 10.** Three reasons, and it is a judgment call at the margin rather
than a result:
1. Depth 5's ceiling is structural, not statistical. No amount of extra
   requested results recovers those three questions.
2. Depth 5's single regression is three times worse.
3. Depth 5 of a much larger corpus is very few candidates; 10 degrades more
   gracefully as the corpus grows.

If the primary consumer turns out to be context injection that only ever reads
the top 3–5 chunks, **depth 5 is the better setting** and this should be
revisited. `RAG_ARM_DEPTH=5` needs no code change.

## Result 4 — weighted RRF is not worth it

At depth 20, `w_vec=1.5` buys +0.019 MRR and costs h@5 (0.643 vs 0.679).
`w_kw=2` is worse on MRR. Equal votes are fine; the complexity is not justified.

## Settings that came out of this

```python
rrf_k: int = 60      # flat for k >= 10; the original RRF paper's value
arm_depth: int = 10  # was 50
```

Both are overridable — `RAG_RRF_K`, `RAG_ARM_DEPTH` — so this can be re-swept
without editing source. Non-numeric, empty, and sub-1 values all raise, because
a sweep that silently used a different value than it printed would be worse than
no sweep.

## Limitations — read before trusting these numbers

- **28 questions over 156 chunks is small.** Differences under ~0.03 MRR are
  within noise, which means depth 5 vs depth 10 (0.560 vs 0.529) is **inside**
  the noise band. Depth 50 → 10 is worth acting on because it moves every
  chunk-level metric in the same direction and wins 7 questions to 1, not
  because 0.529 beats 0.492.
- **`arm_depth` is corpus-size dependent.** 10 of 156 chunks is 6%; on a 20,000
  chunk corpus the same absolute value is 0.05%, which may be too few. Re-run
  rather than assuming this transfers.
- **The corpus is this project's own documentation**, and the same author wrote
  the corpus, the questions and the fix. That inflates absolute levels. It
  applies equally to every configuration compared, so the *comparison* survives
  — but 0.529 is not a quality figure to quote out of context.
- **A random ranker scores 0.033** on this setup, so the absolute numbers are
  far above chance.
- **Not measured:** per-document diversity (one long note can still dominate the
  result set) and any relevance floor. Both remain open in `FOLLOWUPS.md`.

## Reconstructing the measurement corpus

The corpus is 8 documents copied out of the two repositories, so it is
reproducible without shipping a duplicate copy:

```bash
mkdir -p /tmp/eval-corpus && cd /tmp/eval-corpus
P=~/git/doink-brain-playbooks; R=~/git/doink-rag
cp $P/docs/architecture.md                                          architecture.md
cp $P/README.md                                                     playbooks-readme.md
cp $P/docs/superpowers/specs/2026-07-26-baseline-hardening-design.md baseline-hardening-spec.md
cp $P/docs/superpowers/plans/2026-07-26-baseline-hardening.md        baseline-hardening-plan.md
cp $R/docs/superpowers/specs/2026-07-26-rag-retrieval-design.md      rag-design-spec.md
cp $R/docs/superpowers/plans/2026-07-26-doink-rag-service.md         rag-plan.md
cp $R/README.md                                                     rag-readme.md
cp $R/docs/FOLLOWUPS.md                                             followups.md
```

Those documents keep changing, so questions will drift out of date. `evaluate()`
raises rather than silently dropping a question whose phrase no longer resolves
— if that fires, fix the phrase, do not delete the question.

## Redoing this

```bash
export RAG_CORPUS_DIR=/tmp/eval-corpus RAG_DB_PATH=/tmp/eval.db
export RAG_EMBED_URL=http://127.0.0.1:8092
rag index

python eval/run_eval.py                    # current settings, both scoring levels
python eval/sweep.py                       # regenerates every table above
RAG_ARM_DEPTH=5 python eval/run_eval.py    # spot-check one setting
```

Watch the **chunk** columns. If keyword chunk MRR collapses toward zero, the FTS
query semantics have regressed — that is the failure this whole exercise exists
to prevent.
