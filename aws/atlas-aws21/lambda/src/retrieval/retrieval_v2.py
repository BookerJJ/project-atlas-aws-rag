from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


from src.retrieval.hybrid_rrf import reciprocal_rank_fusion
from src.retrieval.intent_reranker import (
    Candidate,
    RerankedHit,
    analyze_query,
    rerank_candidates,
)
from src.retrieval.query_router import route_query


@dataclass(frozen=True)
class RetrievalHit:
    payload: dict[str, Any]
    score: float
    source_rank: int
    route: str
    reranked: bool
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    identifier_bonus: float = 0.0
    section_bonus: float = 0.0
    status_bonus: float = 0.0
    date_bonus: float = 0.0
    coverage_identifiers: tuple[str, ...] = ()
    coverage_injected: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    route: str
    identifiers: list[str]
    reranked: bool
    query_signals: dict[str, Any]
    hits: list[RetrievalHit]
    latency_ms: dict[str, float]


def _normalize_identifier(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _direct_document_match(
    payload: dict[str, Any],
    identifier: str,
) -> bool:
    """
    Return True only when the chunk belongs to the identifier's own
    document, rather than merely mentioning the identifier in body text.

    Examples:
      AUTH-342 -> engineering_issue_auth342
      CR-014   -> change_request_cr014

    This deliberately avoids using payload text for direct-entity coverage.
    A CR-017 chunk that mentions AUTH-342 is supporting evidence, not the
    primary AUTH-342 entity document.
    """

    needle = _normalize_identifier(
        identifier
    )

    if not needle:
        return False

    document_id = _normalize_identifier(
        payload.get(
            "document_id",
            "",
        )
    )

    chunk_id = str(
        payload.get(
            "chunk_id",
            "",
        )
    ).split("::", 1)[0]

    chunk_document = _normalize_identifier(
        chunk_id
    )

    return (
        needle in document_id
        or needle in chunk_document
    )


class RetrievalV2:
    """
    Retrieval v0.4 policy with a caller-supplied vector and dense backend.

    v0.4 preserves v0.3 behavior for zero- and single-identifier queries.
    For queries containing two or more explicit technical identifiers, it
    adds deterministic direct-entity evidence coverage before final Top-K
    selection.

    Flow:
      1. Dense BGE-M3 default retrieval.
      2. Explicit identifiers -> Dense + BM25 + weighted RRF.
      3. Existing v0.3 intent/metadata/temporal reranking.
      4. Multi-identifier only: augment the rerank pool with chunks from
         each identifier's own document.
      5. Ensure at least one direct entity hit per identifier survives in
         the final Top-K when such evidence exists in the indexed corpus.

    The policy is generic. No Project Atlas identifiers or chunk IDs are
    hard-coded.
    """

    def __init__(self, *, experiment, store, lexical_index) -> None:
        self.experiment = experiment
        self.collection_name = "project_atlas_snapshot_v01"
        self.store = store
        self.lexical_index = lexical_index

    def _dense_candidates(
        self,
        dense_hits,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []

        for rank, hit in enumerate(
            dense_hits,
            start=1,
        ):
            payload = dict(
                hit.payload or {}
            )
            chunk_id = payload.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            candidates.append(
                Candidate(
                    chunk_id=str(
                        chunk_id
                    ),
                    payload=payload,
                    source_rank=rank,
                    source_score=float(
                        hit.score
                    ),
                    dense_rank=rank,
                    dense_score=float(
                        hit.score
                    ),
                )
            )

        return candidates

    def _hybrid_candidates(
        self,
        hybrid_hits,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []

        for rank, hit in enumerate(
            hybrid_hits,
            start=1,
        ):
            candidates.append(
                Candidate(
                    chunk_id=hit.chunk_id,
                    payload=dict(
                        hit.payload or {}
                    ),
                    source_rank=rank,
                    source_score=float(
                        hit.score
                    ),
                    dense_rank=hit.dense_rank,
                    lexical_rank=hit.lexical_rank,
                    dense_score=hit.dense_score,
                    lexical_score=hit.lexical_score,
                )
            )

        return candidates

    def _coverage_enabled(
        self,
        identifiers: list[str],
    ) -> bool:
        cfg = self.experiment.get(
            "multi_identifier_coverage",
            {},
        )

        if not cfg.get(
            "enabled",
            False,
        ):
            return False

        minimum_count = int(
            cfg.get(
                "minimum_identifier_count",
                2,
            )
        )

        return len(
            identifiers
        ) >= minimum_count

    def _augment_direct_entity_candidates(
        self,
        *,
        candidates: list[Candidate],
        identifiers: list[str],
    ) -> tuple[
        list[Candidate],
        dict[str, list[str]],
    ]:
        """
        Add direct chunks from each identifier's own document to the
        reranking pool. Existing candidates are never duplicated.

        The lexical index already contains the exact Qdrant payload set,
        so this is a deterministic metadata scan over the indexed corpus,
        not a second semantic retrieval model.
        """

        cfg = self.experiment[
            "multi_identifier_coverage"
        ]

        max_per_identifier = int(
            cfg.get(
                "max_direct_candidates_per_identifier",
                8,
            )
        )

        augmented = list(
            candidates
        )

        existing_chunk_ids = {
            candidate.chunk_id
            for candidate in augmented
        }

        direct_candidates: dict[
            str,
            list[str],
        ] = {
            identifier: []
            for identifier in identifiers
        }

        next_source_rank = (
            max(
                (
                    candidate.source_rank
                    for candidate in augmented
                ),
                default=0,
            )
            + 1
        )

        for identifier in identifiers:
            matched_payloads = [
                payload
                for payload in self.lexical_index.payloads
                if _direct_document_match(
                    payload,
                    identifier,
                )
            ]

            matched_payloads.sort(
                key=lambda payload: str(
                    payload.get(
                        "chunk_id",
                        "",
                    )
                )
            )

            matched_payloads = matched_payloads[
                :max_per_identifier
            ]

            for payload in matched_payloads:
                chunk_id = str(
                    payload.get(
                        "chunk_id",
                        "",
                    )
                )

                if not chunk_id:
                    continue

                direct_candidates[
                    identifier
                ].append(
                    chunk_id
                )

                if chunk_id in existing_chunk_ids:
                    continue

                augmented.append(
                    Candidate(
                        chunk_id=chunk_id,
                        payload=dict(
                            payload
                        ),
                        source_rank=next_source_rank,
                        source_score=None,
                        dense_rank=None,
                        lexical_rank=None,
                        dense_score=None,
                        lexical_score=None,
                    )
                )

                existing_chunk_ids.add(
                    chunk_id
                )
                next_source_rank += 1

        return (
            augmented,
            direct_candidates,
        )

    @staticmethod
    def _coverage_map(
        hits: list[RerankedHit],
        identifiers: list[str],
    ) -> dict[str, list[str]]:
        return {
            identifier: [
                hit.chunk_id
                for hit in hits
                if _direct_document_match(
                    hit.payload,
                    identifier,
                )
            ]
            for identifier in identifiers
        }

    def _select_with_minimum_coverage(
        self,
        *,
        reranked_hits: list[RerankedHit],
        identifiers: list[str],
        final_top_k: int,
    ) -> tuple[
        list[RerankedHit],
        dict[str, Any],
    ]:
        cfg = self.experiment[
            "multi_identifier_coverage"
        ]

        minimum_hits = int(
            cfg.get(
                "minimum_direct_hits_per_identifier",
                1,
            )
        )

        selected = list(
            reranked_hits[
                :final_top_k
            ]
        )

        initial_chunk_ids = {
            hit.chunk_id
            for hit in selected
        }

        injected_chunk_ids: list[str] = []
        injected_for: dict[
            str,
            list[str],
        ] = {
            identifier: []
            for identifier in identifiers
        }

        def coverage_counts() -> dict[str, int]:
            return {
                identifier: sum(
                    1
                    for hit in selected
                    if _direct_document_match(
                        hit.payload,
                        identifier,
                    )
                )
                for identifier in identifiers
            }

        def removable(
            candidate: RerankedHit,
            counts: dict[str, int],
        ) -> bool:
            for identifier in identifiers:
                if not _direct_document_match(
                    candidate.payload,
                    identifier,
                ):
                    continue

                if counts[
                    identifier
                ] <= minimum_hits:
                    return False

            return True

        for identifier in identifiers:
            counts = coverage_counts()

            if counts[
                identifier
            ] >= minimum_hits:
                continue

            direct_pool = [
                hit
                for hit in reranked_hits
                if _direct_document_match(
                    hit.payload,
                    identifier,
                )
                and hit.chunk_id
                not in {
                    item.chunk_id
                    for item in selected
                }
            ]

            needed = (
                minimum_hits
                - counts[
                    identifier
                ]
            )

            for direct_hit in direct_pool[
                :needed
            ]:
                if len(
                    selected
                ) < final_top_k:
                    selected.append(
                        direct_hit
                    )
                else:
                    counts = coverage_counts()
                    replace_index = None

                    for index in range(
                        len(selected) - 1,
                        -1,
                        -1,
                    ):
                        if removable(
                            selected[index],
                            counts,
                        ):
                            replace_index = index
                            break

                    if replace_index is None:
                        break

                    selected[
                        replace_index
                    ] = direct_hit

                if direct_hit.chunk_id not in initial_chunk_ids:
                    injected_chunk_ids.append(
                        direct_hit.chunk_id
                    )
                    injected_for[
                        identifier
                    ].append(
                        direct_hit.chunk_id
                    )

        selected = list(
            {
                hit.chunk_id: hit
                for hit in selected
            }.values()
        )

        selected.sort(
            key=lambda hit: (
                -hit.score,
                hit.source_rank,
                hit.chunk_id,
            )
        )

        selected = selected[
            :final_top_k
        ]

        final_coverage = self._coverage_map(
            selected,
            identifiers,
        )

        coverage_complete = all(
            len(
                final_coverage[
                    identifier
                ]
            ) >= minimum_hits
            for identifier in identifiers
        )

        return (
            selected,
            {
                "coverage_complete": (
                    coverage_complete
                ),
                "minimum_direct_hits_per_identifier": (
                    minimum_hits
                ),
                "identifier_coverage": (
                    final_coverage
                ),
                "coverage_injected_chunk_ids": list(
                    dict.fromkeys(
                        injected_chunk_ids
                    )
                ),
                "coverage_injected_for": {
                    identifier: list(
                        dict.fromkeys(
                            chunk_ids
                        )
                    )
                    for identifier, chunk_ids in injected_for.items()
                    if chunk_ids
                },
            },
        )

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
    ) -> RetrievalResult:
        import time

        routing_cfg = self.experiment[
            "routing"
        ]
        hybrid_cfg = self.experiment[
            "hybrid"
        ]
        rerank_cfg = self.experiment[
            "reranking"
        ]

        route_decision = route_query(
            query
        )

        signals = analyze_query(
            query,
            identifiers=route_decision.identifiers,
            default_year=rerank_cfg[
                "default_year"
            ],
        )

        coverage_applied = (
            self._coverage_enabled(
                route_decision.identifiers
            )
        )

        total_start = time.perf_counter()

        lexical_ms = 0.0
        fusion_ms = 0.0
        rerank_ms = 0.0
        coverage_ms = 0.0

        if route_decision.route == "identifier_hybrid":
            dense_top_k = hybrid_cfg[
                "dense_candidate_k"
            ]
        elif signals.should_rerank:
            dense_top_k = routing_cfg[
                "rerank_candidate_k"
            ]
        else:
            dense_top_k = routing_cfg[
                "dense_top_k"
            ]

        dense_start = time.perf_counter()

        dense_hits = self.store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=dense_top_k,
        )

        dense_ms = (
            time.perf_counter()
            - dense_start
        ) * 1000

        if route_decision.route == "identifier_hybrid":
            lexical_start = time.perf_counter()

            lexical_hits = self.lexical_index.search(
                query,
                top_k=hybrid_cfg[
                    "lexical_candidate_k"
                ],
            )

            lexical_ms = (
                time.perf_counter()
                - lexical_start
            ) * 1000

            fusion_start = time.perf_counter()

            hybrid_hits = reciprocal_rank_fusion(
                dense_hits=dense_hits,
                lexical_hits=lexical_hits,
                top_k=hybrid_cfg[
                    "fusion_candidate_k"
                ],
                rrf_k=hybrid_cfg[
                    "rrf_k"
                ],
                dense_weight=hybrid_cfg[
                    "dense_weight"
                ],
                lexical_weight=hybrid_cfg[
                    "lexical_weight"
                ],
            )

            fusion_ms = (
                time.perf_counter()
                - fusion_start
            ) * 1000

            candidates = self._hybrid_candidates(
                hybrid_hits
            )
        else:
            candidates = self._dense_candidates(
                dense_hits
            )

        direct_candidate_inventory: dict[
            str,
            list[str],
        ] = {}

        if coverage_applied:
            coverage_start = time.perf_counter()

            (
                candidates,
                direct_candidate_inventory,
            ) = self._augment_direct_entity_candidates(
                candidates=candidates,
                identifiers=route_decision.identifiers,
            )

            coverage_ms += (
                time.perf_counter()
                - coverage_start
            ) * 1000

        coverage_diagnostics: dict[
            str,
            Any,
        ] = {
            "coverage_complete": None,
            "minimum_direct_hits_per_identifier": None,
            "identifier_coverage": {},
            "coverage_injected_chunk_ids": [],
            "coverage_injected_for": {},
        }

        if signals.should_rerank:
            rerank_start = time.perf_counter()

            rerank_top_k = (
                len(candidates)
                if coverage_applied
                else hybrid_cfg[
                    "final_top_k"
                ]
            )

            reranked_hits = rerank_candidates(
                candidates=candidates,
                signals=signals,
                final_top_k=rerank_top_k,
                rank_step=rerank_cfg[
                    "rank_step"
                ],
                minimum_base_score=rerank_cfg[
                    "minimum_base_score"
                ],
                identifier_document_bonus=rerank_cfg[
                    "identifier_document_bonus"
                ],
                section_intent_bonus=rerank_cfg[
                    "section_intent_bonus"
                ],
                max_section_bonus=rerank_cfg[
                    "max_section_bonus"
                ],
                current_status_bonus=rerank_cfg[
                    "current_status_bonus"
                ],
                superseded_penalty=rerank_cfg[
                    "superseded_penalty"
                ],
                final_approved_bonus=rerank_cfg[
                    "final_approved_bonus"
                ],
                exact_date_bonus=rerank_cfg[
                    "exact_date_bonus"
                ],
                future_date_penalty=rerank_cfg[
                    "future_date_penalty"
                ],
                near_past_bonus=rerank_cfg[
                    "near_past_bonus"
                ],
            )

            rerank_ms = (
                time.perf_counter()
                - rerank_start
            ) * 1000

            if coverage_applied:
                coverage_start = time.perf_counter()

                (
                    final_hits,
                    coverage_diagnostics,
                ) = self._select_with_minimum_coverage(
                    reranked_hits=reranked_hits,
                    identifiers=route_decision.identifiers,
                    final_top_k=hybrid_cfg[
                        "final_top_k"
                    ],
                )

                coverage_ms += (
                    time.perf_counter()
                    - coverage_start
                ) * 1000
            else:
                final_hits = reranked_hits[
                    :hybrid_cfg[
                        "final_top_k"
                    ]
                ]

            injected_set = set(
                coverage_diagnostics[
                    "coverage_injected_chunk_ids"
                ]
            )

            hits = []

            for hit in final_hits:
                matched_identifiers = tuple(
                    identifier
                    for identifier in route_decision.identifiers
                    if _direct_document_match(
                        hit.payload,
                        identifier,
                    )
                )

                hits.append(
                    RetrievalHit(
                        payload=dict(
                            hit.payload or {}
                        ),
                        score=float(
                            hit.score
                        ),
                        source_rank=hit.source_rank,
                        route=route_decision.route,
                        reranked=True,
                        dense_rank=hit.dense_rank,
                        lexical_rank=hit.lexical_rank,
                        dense_score=hit.dense_score,
                        lexical_score=hit.lexical_score,
                        identifier_bonus=hit.identifier_bonus,
                        section_bonus=hit.section_bonus,
                        status_bonus=hit.status_bonus,
                        date_bonus=hit.date_bonus,
                        coverage_identifiers=(
                            matched_identifiers
                        ),
                        coverage_injected=(
                            hit.chunk_id
                            in injected_set
                        ),
                    )
                )
        else:
            final_candidates = candidates[
                :hybrid_cfg[
                    "final_top_k"
                ]
            ]

            hits = [
                RetrievalHit(
                    payload=dict(
                        candidate.payload
                    ),
                    score=float(
                        candidate.source_score
                        or 0.0
                    ),
                    source_rank=candidate.source_rank,
                    route=route_decision.route,
                    reranked=False,
                    dense_rank=candidate.dense_rank,
                    lexical_rank=candidate.lexical_rank,
                    dense_score=candidate.dense_score,
                    lexical_score=candidate.lexical_score,
                )
                for candidate in final_candidates
            ]

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        query_signals = {
            "current_intent": (
                signals.current_intent
            ),
            "final_decision_intent": (
                signals.final_decision_intent
            ),
            "reference_date": (
                signals.reference_date.isoformat()
                if signals.reference_date
                else None
            ),
            "section_intents": list(
                signals.section_intents
            ),
            "multi_identifier_coverage_applied": (
                coverage_applied
            ),
            "direct_candidate_inventory": (
                direct_candidate_inventory
            ),
            **coverage_diagnostics,
        }

        return RetrievalResult(
            query=query,
            route=route_decision.route,
            identifiers=list(
                route_decision.identifiers
            ),
            reranked=signals.should_rerank,
            query_signals=query_signals,
            hits=hits,
            latency_ms={
                "dense_search": round(
                    dense_ms,
                    2,
                ),
                "lexical_search": round(
                    lexical_ms,
                    2,
                ),
                "fusion": round(
                    fusion_ms,
                    2,
                ),
                "rerank": round(
                    rerank_ms,
                    2,
                ),
                "coverage": round(
                    coverage_ms,
                    2,
                ),
                "retrieval_total": round(
                    total_ms,
                    2,
                ),
            },
        )
