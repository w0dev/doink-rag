"""Measure the arms separately and fused.

Hybrid was chosen without a bake-off, so this is not a gate. It is the
instrument that says whether rrf_k suits this corpus, whether one arm is dead
weight, and whether a larger embedding model earns its RAM on real questions
rather than on MTEB's average.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.config import Settings          # noqa: E402
from rag.db import connect, create_schema  # noqa: E402
from rag.embed import Embedder            # noqa: E402
from rag.search import keyword_arm, search, vector_arm  # noqa: E402


@dataclass
class Question:
    question: str
    expect_paths: list[str]
    # A distinctive string that must appear in the CORRECT chunk. Document-level
    # recall on a small corpus is nearly free and hides a degenerate fused list;
    # chunk-level is where fusion dilution actually shows.
    expect_phrase: str | None = None
    kind: str | None = None


@dataclass
class ArmScore:
    recall: float = 0.0
    mrr: float = 0.0
    chunk_recall: float = 0.0
    chunk_mrr: float = 0.0


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text()) or []
    return [Question(d["question"], d["expect_paths"],
                     d.get("expect_phrase"), d.get("kind")) for d in data]


def _paths_for(con, chunk_ids: list[int]) -> list[str]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = {
        r["id"]: r["path"]
        for r in con.execute(
            f"SELECT c.id, d.path FROM chunks c JOIN documents d "
            f"ON d.id=c.document_id WHERE c.id IN ({placeholders})", chunk_ids)
    }
    return [rows[cid] for cid in chunk_ids if cid in rows]


def _gold_chunks(con, expect_paths: list[str], phrase: str) -> set[int]:
    """Chunk ids in an expected document that contain the distinguishing phrase."""
    placeholders = ",".join("?" * len(expect_paths))
    rows = con.execute(
        f"SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE d.path IN ({placeholders}) "
        f"AND (c.text LIKE ? OR c.heading_path LIKE ?)",
        [*expect_paths, f"%{phrase}%", f"%{phrase}%"])
    return {r["id"] for r in rows}


def _score_ids(ranked_ids: list[int], gold: set[int]) -> tuple[float, float]:
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0, 1.0 / position
    return 0.0, 0.0


def _score(ranked_paths: list[str], expected: list[str]) -> tuple[float, float]:
    for position, path in enumerate(ranked_paths, start=1):
        if path in expected:
            return 1.0, 1.0 / position
    return 0.0, 0.0


def evaluate(con, settings: Settings, embedder, questions: list[Question],
             k: int = 10) -> dict[str, ArmScore]:
    """Score each arm and the fusion, at document level and — where the question
    supplies expect_phrase — at chunk level too.

    Both are reported because they answer different questions. Document recall
    says "did we find the right note"; chunk MRR says "is the right passage near
    the top", which is what a context window actually consumes and what fusion
    dilution degrades first.
    """
    names = ("keyword", "vector", "fused")
    totals = {n_: [0.0, 0.0] for n_ in names}
    chunk_totals = {n_: [0.0, 0.0] for n_ in names}
    chunk_n = 0

    for q in questions:
        kw_ids = keyword_arm(con, settings, q.question)[:k]
        vec_ids = vector_arm(con, settings, embedder.embed_query(q.question))[:k]
        hits = search(con, settings, embedder, q.question, k)

        id_runs = {"keyword": kw_ids, "vector": vec_ids,
                   "fused": [h.chunk_id for h in hits]}
        path_runs = {"keyword": _paths_for(con, kw_ids),
                     "vector": _paths_for(con, vec_ids),
                     "fused": [h.path for h in hits]}

        for name in names:
            recall, mrr = _score(path_runs[name], q.expect_paths)
            totals[name][0] += recall
            totals[name][1] += mrr

        if q.expect_phrase:
            gold = _gold_chunks(con, q.expect_paths, q.expect_phrase)
            if gold:
                chunk_n += 1
                for name in names:
                    recall, mrr = _score_ids(id_runs[name], gold)
                    chunk_totals[name][0] += recall
                    chunk_totals[name][1] += mrr

    n = max(len(questions), 1)
    cn = max(chunk_n, 1)
    return {
        name: ArmScore(totals[name][0] / n, totals[name][1] / n,
                       chunk_totals[name][0] / cn, chunk_totals[name][1] / cn)
        for name in names
    }


def main() -> int:
    settings = Settings.from_env()
    con = connect(settings)
    create_schema(con, settings)
    questions = load_questions(Path(__file__).parent / "questions.yaml")
    scores = evaluate(con, settings, Embedder(settings), questions)
    print(f"{len(questions)} questions | model={settings.embed_model} | "
          f"rrf_k={settings.rrf_k} arm_depth={settings.arm_depth}\n")
    print(f"{'arm':<10} {'doc r@10':>10} {'doc MRR':>9} "
          f"{'chunk r@10':>12} {'chunk MRR':>11}")
    for name, score in scores.items():
        print(f"{name:<10} {score.recall:>10.3f} {score.mrr:>9.3f} "
              f"{score.chunk_recall:>12.3f} {score.chunk_mrr:>11.3f}")
    print("\nchunk-level is the one to watch: document recall on a small corpus "
          "is nearly free\nand hides a degenerate fused list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
