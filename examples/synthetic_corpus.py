"""Publishable fixture corpus. It is not the historical evaluation snapshot."""

from __future__ import annotations


def synthetic_chunks():
    def row(document, index, section, status, date, title, text):
        return {
            "chunk_id": f"{document}::chunk::{index:03}",
            "document_id": document, "chunk_index": index,
            "section": section, "title": title, "text": text, "date": date,
            "status": status, "version": "1.0", "language": "en",
            "project": "atlas_demo", "document_type": "demo_note",
        }

    highlights = [
        row("issue_demo101", 0, "Current State", "open", "2026-09-01", "DEMO-101",
            "DEMO-101 implementation has not started. The customer security board has not approved the production OAuth design."),
        row("issue_demo202", 0, "Current State", "open", "2026-09-05", "DEMO-202",
            "DEMO-202 passed staging validation. Production configuration is ready, but final production validation is pending."),
        row("demo_release_readiness", 0, "Engineering", "active", "2026-09-12", "Demo review",
            "DEMO-101 and DEMO-202 are tracked in the release readiness review."),
        row("issue_demo101", 1, "Dependencies", "open", "2026-09-01", "DEMO-101",
            "Approval from the customer security board is an external dependency for DEMO-101."),
        row("issue_demo202", 1, "Ownership", "open", "2026-09-05", "DEMO-202",
            "The observability team owns the remaining production validation for DEMO-202."),
        row("demo_archived_readiness", 0, "Decision", "superseded", "2026-08-02", "Archived readiness",
            "This early planning note was replaced by the September review and should not be used as current evidence."),
    ]
    return highlights + [
        row(f"demo_note_{i:03}", 0, "Notes", "active", "2026-08-01", f"Fixture {i:03}",
            f"Synthetic background note {i:03} for the Atlas demonstration corpus.")
        for i in range(122)
    ]
