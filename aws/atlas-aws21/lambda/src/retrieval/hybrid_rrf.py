from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HybridHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]

    dense_rank: int | None
    lexical_rank: int | None

    dense_score: float | None
    lexical_score: float | None


def reciprocal_rank_fusion(
    *,
    dense_hits: list[Any],
    lexical_hits: list[Any],
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[HybridHit]:
    """
    Fuse dense and lexical rankings with weighted Reciprocal Rank Fusion.

    The fused score is intentionally based on rank, not raw dense/BM25
    scores, because the two retrieval systems use incomparable score
    scales.
    """

    merged: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(
        dense_hits,
        start=1,
    ):
        payload = hit.payload or {}

        chunk_id = payload.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        item = merged.setdefault(
            str(chunk_id),
            {
                "payload": dict(payload),
                "score": 0.0,
                "dense_rank": None,
                "lexical_rank": None,
                "dense_score": None,
                "lexical_score": None,
            },
        )

        item["score"] += (
            dense_weight
            / (
                rrf_k
                + rank
            )
        )

        item["dense_rank"] = rank
        item["dense_score"] = float(
            hit.score
        )

    for rank, hit in enumerate(
        lexical_hits,
        start=1,
    ):
        chunk_id = str(
            hit.chunk_id
        )

        item = merged.setdefault(
            chunk_id,
            {
                "payload": dict(
                    hit.payload
                ),
                "score": 0.0,
                "dense_rank": None,
                "lexical_rank": None,
                "dense_score": None,
                "lexical_score": None,
            },
        )

        item["score"] += (
            lexical_weight
            / (
                rrf_k
                + rank
            )
        )

        item["lexical_rank"] = rank
        item["lexical_score"] = float(
            hit.score
        )

    hits = [
        HybridHit(
            chunk_id=chunk_id,
            score=float(
                item["score"]
            ),
            payload=item[
                "payload"
            ],
            dense_rank=item[
                "dense_rank"
            ],
            lexical_rank=item[
                "lexical_rank"
            ],
            dense_score=item[
                "dense_score"
            ],
            lexical_score=item[
                "lexical_score"
            ],
        )
        for chunk_id, item in merged.items()
    ]

    hits.sort(
        key=lambda hit: (
            -hit.score,
            hit.dense_rank
            if hit.dense_rank is not None
            else 10**9,
            hit.lexical_rank
            if hit.lexical_rank is not None
            else 10**9,
            hit.chunk_id,
        )
    )

    return hits[:top_k]
