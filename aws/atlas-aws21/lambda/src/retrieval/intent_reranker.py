from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any


EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


CURRENT_MARKERS = (
    "current",
    "currently",
    "latest",
    "now",
    "as of",
    "현재",
    "최신",
    "지금",
    "기준",
)

FINAL_DECISION_MARKERS = (
    "final",
    "finally",
    "finalized",
    "approved",
    "approval",
    "agreed",
    "최종",
    "승인",
    "확정",
    "합의",
)

REFERENCE_DATE_ANCHORS = (
    "as of",
    "meeting",
    "review",
    " on ",
    "기준",
    "미팅",
    "리뷰",
)


SECTION_INTENT_QUERY_TERMS = {
    "current_state": (
        "current state",
        "current status",
        "implementation state",
        "status",
        "현재 상태",
        "상태",
    ),
    "dependency": (
        "dependency",
        "dependencies",
        "depend on",
        "depends on",
        "blocker",
        "막고",
        "의존",
    ),
    "ownership": (
        "who owns",
        "owner",
        "ownership",
        " owns ",
        "누구",
        "담당",
        "소유",
    ),
    "impact": (
        "impact",
        "영향",
        "blocker",
    ),
    "decision": (
        "decision",
        "approved",
        "approval",
        "final",
        "finalized",
        "agreed",
        "결정",
        "승인",
        "최종",
        "확정",
        "합의",
    ),
    "release": (
        "production launch",
        "launch target",
        "release target",
        "release date",
        "launch date",
        "목표일",
        "출시일",
        "launch 날짜",
        "release 날짜",
    ),
}


SECTION_INTENT_PAYLOAD_TERMS = {
    "current_state": (
        "current state",
        "status",
        "state",
        "상태",
    ),
    "dependency": (
        "dependency",
        "dependencies",
        "dep-",
        "risk-",
        "의존",
    ),
    "ownership": (
        "ownership",
        "owner",
        "responsibility",
        "customer-owned",
        "담당",
        "소유",
    ),
    "impact": (
        "impact",
        "영향",
    ),
    "decision": (
        "decision",
        "decision status",
        "change",
        "결정",
        "운영 정책 결정",
    ),
    "release": (
        "production launch",
        "release target",
        "milestone",
        "schedule",
        "일정",
    ),
}


@dataclass(frozen=True)
class QuerySignals:
    current_intent: bool
    final_decision_intent: bool
    reference_date: date | None
    section_intents: tuple[str, ...]
    identifiers: tuple[str, ...]

    @property
    def should_rerank(self) -> bool:
        if self.identifiers:
            return True

        if self.current_intent:
            return True

        if self.final_decision_intent:
            return True

        if self.reference_date is not None:
            return True

        if (
            "ownership" in self.section_intents
            or "dependency" in self.section_intents
        ):
            return True

        return False


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    payload: dict[str, Any]
    source_rank: int
    source_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: str
    payload: dict[str, Any]
    score: float
    source_rank: int
    source_score: float | None
    dense_rank: int | None
    lexical_rank: int | None
    dense_score: float | None
    lexical_score: float | None
    base_rank_score: float
    identifier_bonus: float
    section_bonus: float
    status_bonus: float
    date_bonus: float


def _normalize(value: str) -> str:
    return re.sub(
        r"[^a-z0-9가-힣]+",
        "",
        value.lower(),
    )


def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    lowered = text.lower()

    return any(
        term.lower() in lowered
        for term in terms
    )


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None

    try:
        return datetime.strptime(
            str(value),
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return None


def _has_reference_date_anchor(
    query: str,
) -> bool:
    lowered = f" {query.lower()} "

    return any(
        marker in lowered
        for marker in REFERENCE_DATE_ANCHORS
    )


def extract_reference_date(
    query: str,
    *,
    default_year: int,
) -> date | None:
    """
    Extract an explicit as-of / meeting / review date.

    We intentionally require a temporal anchor. This prevents target dates
    in questions such as "October 15 or October 29?" from being mistaken
    for an as-of date.
    """

    if not _has_reference_date_anchor(
        query
    ):
        return None

    # ISO date: 2026-09-10
    iso_match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        query,
    )

    if iso_match:
        return date(
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
        )

    # Korean date: 9월 10일
    ko_match = re.search(
        r"(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        query,
    )

    if ko_match:
        return date(
            default_year,
            int(ko_match.group(1)),
            int(ko_match.group(2)),
        )

    # English date: September 10 / August 13
    en_pattern = (
        r"\b("
        + "|".join(
            EN_MONTHS.keys()
        )
        + r")\s+(\d{1,2})\b"
    )

    en_match = re.search(
        en_pattern,
        query.lower(),
    )

    if en_match:
        return date(
            default_year,
            EN_MONTHS[
                en_match.group(1)
            ],
            int(en_match.group(2)),
        )

    return None


def detect_section_intents(
    query: str,
) -> tuple[str, ...]:
    lowered = f" {query.lower()} "

    intents: list[str] = []

    for (
        intent,
        terms,
    ) in SECTION_INTENT_QUERY_TERMS.items():
        if any(
            term.lower() in lowered
            for term in terms
        ):
            intents.append(
                intent
            )

    return tuple(
        intents
    )


def analyze_query(
    query: str,
    *,
    identifiers: list[str],
    default_year: int,
) -> QuerySignals:
    lowered = f" {query.lower()} "

    return QuerySignals(
        current_intent=any(
            marker in lowered
            for marker in CURRENT_MARKERS
        ),
        final_decision_intent=any(
            marker in lowered
            for marker in FINAL_DECISION_MARKERS
        ),
        reference_date=extract_reference_date(
            query,
            default_year=default_year,
        ),
        section_intents=detect_section_intents(
            query
        ),
        identifiers=tuple(
            identifiers
        ),
    )


def _identifier_bonus(
    *,
    candidate: Candidate,
    identifiers: tuple[str, ...],
    bonus: float,
) -> float:
    if not identifiers:
        return 0.0

    searchable = " ".join(
        [
            str(
                candidate.payload.get(
                    "document_id",
                    "",
                )
            ),
            str(
                candidate.payload.get(
                    "section",
                    "",
                )
            ),
            str(
                candidate.payload.get(
                    "text",
                    "",
                )
            ),
        ]
    )

    normalized_searchable = _normalize(
        searchable
    )

    for identifier in identifiers:
        if _normalize(
            identifier
        ) in normalized_searchable:
            return bonus

    return 0.0


def _section_bonus(
    *,
    candidate: Candidate,
    section_intents: tuple[str, ...],
    per_intent_bonus: float,
    max_bonus: float,
) -> float:
    if not section_intents:
        return 0.0

    searchable = " ".join(
        [
            str(
                candidate.payload.get(
                    "section",
                    "",
                )
            ),
            str(
                candidate.payload.get(
                    "document_id",
                    "",
                )
            ),
            str(
                candidate.payload.get(
                    "document_type",
                    "",
                )
            ),
        ]
    ).lower()

    matched = 0

    for intent in section_intents:
        terms = SECTION_INTENT_PAYLOAD_TERMS.get(
            intent,
            (),
        )

        if _contains_any(
            searchable,
            terms,
        ):
            matched += 1

    return min(
        matched
        * per_intent_bonus,
        max_bonus,
    )


def _status_bonus(
    *,
    candidate: Candidate,
    signals: QuerySignals,
    current_positive_bonus: float,
    superseded_penalty: float,
    final_approved_bonus: float,
) -> float:
    status = str(
        candidate.payload.get(
            "status",
            "",
        )
    ).lower()

    score = 0.0

    if signals.current_intent:
        if status in {
            "active",
            "open",
            "approved",
        }:
            score += (
                current_positive_bonus
            )

        if status == "superseded":
            score -= abs(
                superseded_penalty
            )

    if signals.final_decision_intent:
        if status == "approved":
            score += (
                final_approved_bonus
            )

        if status == "superseded":
            score -= (
                abs(
                    superseded_penalty
                )
                / 2.0
            )

    return score


def _date_bonus(
    *,
    candidate: Candidate,
    reference_date: date | None,
    exact_date_bonus: float,
    future_date_penalty: float,
    near_past_bonus: float,
) -> float:
    if reference_date is None:
        return 0.0

    candidate_date = _parse_iso_date(
        candidate.payload.get(
            "date"
        )
    )

    if candidate_date is None:
        return 0.0

    delta_days = (
        candidate_date
        - reference_date
    ).days

    if delta_days == 0:
        return exact_date_bonus

    if delta_days > 0:
        return -abs(
            future_date_penalty
        )

    if abs(delta_days) <= 7:
        return near_past_bonus

    return 0.0


def rerank_candidates(
    *,
    candidates: list[Candidate],
    signals: QuerySignals,
    final_top_k: int,
    rank_step: float,
    minimum_base_score: float,
    identifier_document_bonus: float,
    section_intent_bonus: float,
    max_section_bonus: float,
    current_status_bonus: float,
    superseded_penalty: float,
    final_approved_bonus: float,
    exact_date_bonus: float,
    future_date_penalty: float,
    near_past_bonus: float,
) -> list[RerankedHit]:
    reranked: list[RerankedHit] = []

    for candidate in candidates:
        base_rank_score = max(
            minimum_base_score,
            1.0
            - rank_step
            * (
                candidate.source_rank
                - 1
            ),
        )

        identifier_bonus = _identifier_bonus(
            candidate=candidate,
            identifiers=signals.identifiers,
            bonus=identifier_document_bonus,
        )

        section_bonus = _section_bonus(
            candidate=candidate,
            section_intents=signals.section_intents,
            per_intent_bonus=section_intent_bonus,
            max_bonus=max_section_bonus,
        )

        status_bonus = _status_bonus(
            candidate=candidate,
            signals=signals,
            current_positive_bonus=current_status_bonus,
            superseded_penalty=superseded_penalty,
            final_approved_bonus=final_approved_bonus,
        )

        date_bonus = _date_bonus(
            candidate=candidate,
            reference_date=signals.reference_date,
            exact_date_bonus=exact_date_bonus,
            future_date_penalty=future_date_penalty,
            near_past_bonus=near_past_bonus,
        )

        total_score = (
            base_rank_score
            + identifier_bonus
            + section_bonus
            + status_bonus
            + date_bonus
        )

        reranked.append(
            RerankedHit(
                chunk_id=candidate.chunk_id,
                payload=candidate.payload,
                score=total_score,
                source_rank=candidate.source_rank,
                source_score=candidate.source_score,
                dense_rank=candidate.dense_rank,
                lexical_rank=candidate.lexical_rank,
                dense_score=candidate.dense_score,
                lexical_score=candidate.lexical_score,
                base_rank_score=base_rank_score,
                identifier_bonus=identifier_bonus,
                section_bonus=section_bonus,
                status_bonus=status_bonus,
                date_bonus=date_bonus,
            )
        )

    reranked.sort(
        key=lambda hit: (
            -hit.score,
            hit.source_rank,
            hit.chunk_id,
        )
    )

    return reranked[
        :final_top_k
    ]
