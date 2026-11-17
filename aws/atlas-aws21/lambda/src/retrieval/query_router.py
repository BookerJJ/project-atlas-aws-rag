from __future__ import annotations

import re
from dataclasses import dataclass


# Match enterprise-style technical identifiers while remaining compatible
# with Korean grammatical suffixes attached directly after the identifier.
#
# Examples:
# - AUTH-342
# - NET-217이
# - OBS-119는
# - CR-014에서
# - RISK-007
# - DEP-SEC-01
# - AUTH342
#
# We intentionally avoid \b because Python treats Hangul letters as Unicode
# word characters. A pattern such as r"\bNET-217\b" therefore fails for
# "NET-217이": there is no Unicode word boundary between "7" and "이".
#
# Instead, the identifier is bounded only by ASCII letters/digits.
TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"("
    r"[A-Za-z]{2,12}(?:-[A-Za-z]{2,12})?-\d{2,8}"
    r"|"
    r"[A-Za-z]{2,12}\d{2,8}"
    r")"
    r"(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    identifiers: list[str]
    reason: str


def extract_technical_identifiers(query: str) -> list[str]:
    if not query:
        return []

    identifiers = [
        match.group(1)
        for match in TECHNICAL_IDENTIFIER_PATTERN.finditer(query)
    ]

    # Stable de-duplication while preserving query order.
    return list(dict.fromkeys(identifiers))


def route_query(query: str) -> RouteDecision:
    identifiers = extract_technical_identifiers(query)

    if identifiers:
        return RouteDecision(
            route="identifier_hybrid",
            identifiers=identifiers,
            reason=(
                "Explicit technical identifier detected; "
                "use dense + lexical hybrid retrieval."
            ),
        )

    return RouteDecision(
        route="dense",
        identifiers=[],
        reason=(
            "No explicit technical identifier detected; "
            "preserve the frozen dense baseline behavior."
        ),
    )
