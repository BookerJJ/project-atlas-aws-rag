from __future__ import annotations

import re


CITATION_GROUP_PATTERN = re.compile(
    r"\[(S\d+(?:\s*,\s*S\d+)*)\]"
)

LABEL_PATTERN = re.compile(
    r"S\d+"
)


def extract_citations(
    answer: str,
) -> list[str]:
    labels: list[str] = []

    groups = CITATION_GROUP_PATTERN.findall(answer)

    for group in groups:
        found_labels = LABEL_PATTERN.findall(group)

        for label in found_labels:
            if label not in labels:
                labels.append(label)

    return labels


def validate_citations(
    answer: str,
    valid_labels: set[str],
) -> dict:
    cited_labels = extract_citations(answer)

    invalid_labels = [
        label
        for label in cited_labels
        if label not in valid_labels
    ]

    has_citations = len(cited_labels) > 0
    labels_exist = len(invalid_labels) == 0

    return {
        "cited_labels": cited_labels,
        "invalid_labels": invalid_labels,
        "has_citations": has_citations,
        "labels_exist": labels_exist,
        "valid": (
            has_citations
            and labels_exist
        ),
    }
