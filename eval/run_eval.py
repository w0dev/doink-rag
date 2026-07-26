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


@dataclass
class ArmScore:
    recall: float = 0.0
    mrr: float = 0.0


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text()) or []
    return [Question(d["question"], d["expect_paths"]) for d in data]


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


def _score(ranked_paths: list[str], expected: list[str]) -> tuple[float, float]:
    for position, path in enumerate(ranked_paths, start=1):
        if path in expected:
            return 1.0, 1.0 / position
    return 0.0, 0.0


def evaluate(con, settings: Settings, embedder, questions: list[Question],
             k: int = 10) -> dict[str, ArmScore]:
    totals = {name: [0.0, 0.0] for name in ("keyword", "vector", "fused")}
    for q in questions:
        runs = {
            "keyword": _paths_for(con, keyword_arm(con, settings, q.question)[:k]),
            "vector": _paths_for(
                con, vector_arm(con, settings, embedder.embed_query(q.question))[:k]),
            "fused": [h.path for h in search(con, settings, embedder, q.question, k)],
        }
        for name, paths in runs.items():
            recall, mrr = _score(paths, q.expect_paths)
            totals[name][0] += recall
            totals[name][1] += mrr

    n = max(len(questions), 1)
    return {name: ArmScore(r / n, m / n) for name, (r, m) in totals.items()}


def main() -> int:
    settings = Settings.from_env()
    con = connect(settings)
    create_schema(con, settings)
    questions = load_questions(Path(__file__).parent / "questions.yaml")
    scores = evaluate(con, settings, Embedder(settings), questions)
    print(f"{len(questions)} questions, model={settings.embed_model}, "
          f"rrf_k={settings.rrf_k}\n")
    print(f"{'arm':<10} {'recall@10':>10} {'MRR':>8}")
    for name, score in scores.items():
        print(f"{name:<10} {score.recall:>10.3f} {score.mrr:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
