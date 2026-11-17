from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9가-힣]+(?:-[A-Za-z0-9]+)*"
)

IDENTIFIER_PATTERN = re.compile(
    r"\b([A-Za-z]{2,12})[-_]?(\d{2,8})\b"
)


def _identifier_aliases(text: str) -> list[str]:
    aliases: list[str] = []

    for prefix, digits in IDENTIFIER_PATTERN.findall(text):
        prefix = prefix.lower()

        aliases.append(
            f"{prefix}{digits}"
        )
        aliases.append(
            f"{prefix}-{digits}"
        )

    return aliases


def tokenize(text: str) -> list[str]:
    if not text:
        return []

    lowered = text.lower().replace("_", " ")

    tokens: list[str] = []

    for token in TOKEN_PATTERN.findall(lowered):
        tokens.append(token)

        compact = token.replace("-", "")

        if compact != token:
            tokens.append(compact)

    tokens.extend(
        _identifier_aliases(lowered)
    )

    return tokens


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class LexicalBM25Index:
    """
    Small, deterministic BM25 index for the local Project Atlas corpus.

    The index is built from Qdrant payloads so the lexical experiment
    uses the exact same indexed chunks as dense retrieval.

    Metadata fields are deliberately weighted because exact technical
    identifiers often live in document_id/title/section even when they
    are not repeated inside every chunk body.
    """

    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        document_id_weight: int = 4,
        title_weight: int = 2,
        section_weight: int = 2,
        text_weight: int = 1,
    ) -> None:
        if not payloads:
            raise ValueError(
                "Cannot build lexical index from an empty payload list."
            )

        self.k1 = k1
        self.b = b

        self.payloads = payloads

        self.doc_tokens: list[list[str]] = []
        self.term_frequencies: list[dict[str, int]] = []
        self.document_frequencies: dict[str, int] = {}

        for payload in payloads:
            tokens: list[str] = []

            weighted_fields = [
                (
                    str(
                        payload.get(
                            "document_id",
                            "",
                        )
                    ),
                    document_id_weight,
                ),
                (
                    str(
                        payload.get(
                            "title",
                            "",
                        )
                    ),
                    title_weight,
                ),
                (
                    str(
                        payload.get(
                            "section",
                            "",
                        )
                    ),
                    section_weight,
                ),
                (
                    str(
                        payload.get(
                            "text",
                            payload.get(
                                "content",
                                payload.get(
                                    "chunk_text",
                                    "",
                                ),
                            ),
                        )
                    ),
                    text_weight,
                ),
            ]

            for field_text, weight in weighted_fields:
                field_tokens = tokenize(field_text)

                for _ in range(max(weight, 0)):
                    tokens.extend(field_tokens)

            self.doc_tokens.append(tokens)

            tf: dict[str, int] = {}

            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            self.term_frequencies.append(tf)

            for token in set(tokens):
                self.document_frequencies[token] = (
                    self.document_frequencies.get(
                        token,
                        0,
                    )
                    + 1
                )

        self.document_count = len(
            self.doc_tokens
        )

        self.doc_lengths = [
            len(tokens)
            for tokens in self.doc_tokens
        ]

        self.avg_doc_length = (
            sum(self.doc_lengths)
            / self.document_count
        )

    def _idf(
        self,
        token: str,
    ) -> float:
        df = self.document_frequencies.get(
            token,
            0,
        )

        return math.log(
            1.0
            + (
                self.document_count
                - df
                + 0.5
            )
            / (
                df
                + 0.5
            )
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[LexicalHit]:
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        unique_query_tokens = list(
            dict.fromkeys(
                query_tokens
            )
        )

        scored: list[LexicalHit] = []

        for index, payload in enumerate(
            self.payloads
        ):
            tf = self.term_frequencies[
                index
            ]

            doc_length = self.doc_lengths[
                index
            ]

            score = 0.0

            for token in unique_query_tokens:
                frequency = tf.get(
                    token,
                    0,
                )

                if frequency <= 0:
                    continue

                idf = self._idf(
                    token
                )

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * (
                            doc_length
                            / max(
                                self.avg_doc_length,
                                1e-9,
                            )
                        )
                    )
                )

                score += (
                    idf
                    * (
                        frequency
                        * (
                            self.k1
                            + 1.0
                        )
                    )
                    / denominator
                )

            if score <= 0.0:
                continue

            scored.append(
                LexicalHit(
                    chunk_id=str(
                        payload[
                            "chunk_id"
                        ]
                    ),
                    score=score,
                    payload=payload,
                )
            )

        scored.sort(
            key=lambda hit: (
                -hit.score,
                hit.chunk_id,
            )
        )

        return scored[:top_k]
