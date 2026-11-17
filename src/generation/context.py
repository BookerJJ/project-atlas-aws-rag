from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextSource:
    label: str
    chunk_id: str
    document_id: str
    title: str
    section: str
    date: str
    version: str
    status: str
    language: str
    text: str
    score: float


def build_context(
    hits,
) -> tuple[str, list[ContextSource]]:

    sources: list[ContextSource] = []
    blocks: list[str] = []

    for index, hit in enumerate(hits, start=1):
        payload = hit.payload or {}

        label = f"S{index}"

        source = ContextSource(
            label=label,
            chunk_id=payload.get("chunk_id", ""),
            document_id=payload.get("document_id", ""),
            title=payload.get("title", ""),
            section=payload.get("section", ""),
            date=str(payload.get("date", "")),
            version=str(payload.get("version", "")),
            status=payload.get("status", ""),
            language=payload.get("language", ""),
            text=payload.get("text", ""),
            score=float(hit.score),
        )

        sources.append(source)

        blocks.append(
            "\n".join(
                [
                    f"[{label}]",
                    f"Document ID: {source.document_id}",
                    f"Title: {source.title}",
                    f"Section: {source.section}",
                    f"Date: {source.date}",
                    f"Version: {source.version}",
                    f"Status: {source.status}",
                    f"Language: {source.language}",
                    f"Evidence:",
                    source.text,
                ]
            )
        )

    context = "\n\n---\n\n".join(blocks)

    return context, sources
