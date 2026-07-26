"""Sweep rrf_k and arm_depth against a real corpus, using production fusion.

Ships in the repo, and imports `rag.search.fuse` rather than reimplementing RRF,
because a sweep with its own copy of the arithmetic measures a different ranker
than the one being shipped. That is not hypothetical — it happened during the
2026-07-26 tuning run and put wrong numbers in docs/TUNING.md.

Each query is embedded once and each arm queried once, deep; every parameter
combination is then scored over those cached rankings, so the grid is free.

    export RAG_CORPUS_DIR=... RAG_DB_PATH=... RAG_EMBED_URL=...
    rag index
    python eval/sweep.py
"""
from __future__ import annotations

import struct
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.config import Settings                    # noqa: E402
from rag.db import connect                         # noqa: E402
from rag.embed import Embedder                     # noqa: E402
from rag.search import _sanitise, fuse             # noqa: E402
from run_eval import _gold_chunks, load_questions  # noqa: E402

DEEP = 200          # pulled once; arm_depth truncates from here
KS = (1, 5, 10, 20, 30, 60, 100)
DEPTHS = (3, 5, 10, 20, 30, 50, 100)


def gather(con, settings, embedder, questions):
    """One embedding call and two deep queries per question."""
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"  {i}/{len(questions)} {q.question[:56]}", file=sys.stderr)
        vec = embedder.embed_query(q.question)
        blob = struct.pack(f"<{len(vec)}f", *vec)
        dense = [r[0] for r in con.execute(
            "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance", (blob, DEEP))]
        terms = _sanitise(q.question)
        kw = [r[0] for r in con.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?", (terms, DEEP))] if terms else []
        gold = _gold_chunks(con, q.expect_paths, q.expect_phrase) \
            if q.expect_phrase else set()
        if q.expect_phrase and not gold:
            raise ValueError(
                f"no chunk contains {q.expect_phrase!r} for {q.question!r}; "
                f"corpus and questions have drifted apart")
        rows.append({"q": q, "kw": kw, "vec": dense, "gold": gold})
    return rows


def mrr(ranked: list[int], gold: set[int], at: int = 10) -> float:
    for pos, cid in enumerate(ranked[:at], start=1):
        if cid in gold:
            return 1.0 / pos
    return 0.0


def hit_rate(ranked: list[int], gold: set[int], at: int) -> float:
    """Fraction of questions with a gold chunk in the top k.

    Named hit rate, not recall: with gold sets of 1-4 chunks, recall would be
    the fraction of gold chunks retrieved, which is a different number.
    """
    return 1.0 if gold & set(ranked[:at]) else 0.0


def ranked_for(row, settings, k, depth) -> list[int]:
    s = replace(settings, rrf_k=k, arm_depth=depth)
    return [cid for cid, _ in fuse(row["kw"][:depth], row["vec"][:depth], s)]


def main() -> int:
    settings = Settings.from_env()
    con = connect(settings)
    questions = load_questions(Path(__file__).parent / "questions.yaml")
    rows = gather(con, settings, Embedder(settings), questions)
    n = len(rows)

    print("\n=== arm health (chunk MRR@10) ===")
    kinds = sorted({r["q"].kind for r in rows if r["q"].kind})
    print(f"{'arm':<10}" + "".join(f"{k:>10}" for k in kinds) + f"{'ALL':>10}")
    for arm in ("kw", "vec"):
        line = f"{'keyword' if arm == 'kw' else 'dense':<10}"
        for kind in kinds + [None]:
            sel = [r for r in rows if kind is None or r["q"].kind == kind]
            line += f"{sum(mrr(r[arm], r['gold']) for r in sel) / len(sel):>10.3f}"
        print(line)

    print("\n=== fused chunk MRR@10 by (rrf_k, arm_depth) ===")
    print("   k\\depth" + "".join(f"{d:>8}" for d in DEPTHS))
    best = []
    for k in KS:
        line = f"{k:>8} "
        for depth in DEPTHS:
            m = sum(mrr(ranked_for(r, settings, k, depth), r["gold"])
                    for r in rows) / n
            best.append((m, k, depth))
            line += f"{m:>8.3f}"
        print(line)

    print("\n=== hit rate at k, by arm_depth (rrf_k=%d) ===" % settings.rrf_k)
    print(f"{'depth':>6}{'MRR@10':>9}" + "".join(f"{'h@%d' % a:>8}"
                                                 for a in (1, 3, 5, 10, 20)))
    for depth in DEPTHS:
        ranked = [ranked_for(r, settings, settings.rrf_k, depth) for r in rows]
        line = f"{depth:>6}{sum(mrr(x, r['gold']) for x, r in zip(ranked, rows)) / n:>9.3f}"
        for at in (1, 3, 5, 10, 20):
            line += f"{sum(hit_rate(x, r['gold'], at) for x, r in zip(ranked, rows)) / n:>8.3f}"
        print(line)

    print("\n=== per-question change vs arm_depth=50 ===")
    base = [ranked_for(r, settings, settings.rrf_k, 50) for r in rows]
    for depth in (5, 10):
        cand = [ranked_for(r, settings, settings.rrf_k, depth) for r in rows]
        win = loss = 0
        detail = []
        for r, b, c in zip(rows, base, cand):
            mb, mc = mrr(b, r["gold"]), mrr(c, r["gold"])
            if mc > mb:
                win += 1; detail.append(("+", r["q"], mb, mc))
            elif mc < mb:
                loss += 1; detail.append(("-", r["q"], mb, mc))
        print(f"\n  depth {depth}: {win} better, {loss} worse, "
              f"{n - win - loss} unchanged (n={n})")
        for sign, q, mb, mc in detail:
            print(f"    {sign} [{q.kind or '?':<7}] {q.question[:46]:<48} "
                  f"{mb:.3f} -> {mc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
