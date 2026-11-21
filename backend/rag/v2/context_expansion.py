"""Resolve parent context from canonical storage without widening retrieval scope."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any, Dict, Optional

from backend.rag.v2.models import DocStatus


def bounded_excerpt(text: str, max_chars: int) -> str:
    """Keep a share of each section; explicitly mark omitted text."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n[上下文已截断]"
    if max_chars <= len(marker):
        return marker[:max_chars]
    sections = re.split(
        r"\n(?=(?:#{1,6}\s|(?:事实|经过|背景|案情|处置|措施|行动|方案|结果|效果|总结|经验)[：:]))",
        text,
    )
    if len(sections) == 1:
        sections = [part for part in text.split("\n\n") if part.strip()] or [text]
    # Keep the first and last sections when an unusually large outline is used.
    if len(sections) > 8:
        sections = sections[:4] + sections[-4:]
    budget = max_chars - len(marker) - max(0, len(sections) - 1)
    per_section = max(0, budget // len(sections))
    return "\n".join(section[:per_section] for section in sections) + marker


def resolve_parent_context(
    document_id: str, chunk_id: str, *, parent_chunk_id: Optional[str] = None,
    version: Optional[int] = None, expected_content: Optional[str] = None,
    expected_scope: Optional[Dict[str, Any]] = None, max_chars: int = 2200,
) -> Optional[Dict[str, Any]]:
    """Read only an active, same-document, same-version parent.

    Existing indexes without parent IDs may use a unique matching stored parent
    (a case parent or the exact section). Ambiguous or stale links fail closed.
    No database, embedding index or source document is modified here.
    """
    from backend.rag.v2.document_repository import (
        get_chunk, get_chunks_by_document, get_document,
    )

    try:
        doc, child = get_document(document_id), get_chunk(chunk_id)
        if (doc is None or child is None or doc.status != DocStatus.ACTIVE.value
                or child.document_id != document_id or child.version != doc.version):
            return None
        if version is not None and version != child.version:
            return None
        if expected_content is not None and not any(
            content.startswith(expected_content) for content in
            (child.raw_content, child.contextual_content)
        ):
            return None
        if any(getattr(child, key, None) != value
               for key, value in (expected_scope or {}).items()):
            return None
        target = parent_chunk_id or child.parent_chunk_id
        if parent_chunk_id and child.parent_chunk_id and parent_chunk_id != child.parent_chunk_id:
            return None
        if target:
            parent = get_chunk(target)
        else:
            # Compatibility for pre-change rows, not a guess across documents.
            parents = [
                item for item in get_chunks_by_document(document_id)
                if item.parent_chunk_id is None and item.chunk_id != child.chunk_id
                and item.chunk_id.startswith(document_id + "_p")
                and (doc.doc_type in {"case", "event_report"}
                     or item.section_path == child.section_path)
            ]
            parent = parents[0] if len(parents) == 1 else None
        if (parent is None or parent.chunk_id == child.chunk_id
                or parent.document_id != document_id or parent.version != child.version):
            return None
        # Expanding must not add a different region, event type or time window.
        scope = ("region_id", "road_id", "intersection_id", "grounding_scope",
                 "event_type", "effective_from", "effective_to")
        if any(getattr(parent, key) != getattr(child, key)
               or getattr(child, key) != getattr(doc, key) for key in scope):
            return None
        text = bounded_excerpt(parent.raw_content, max_chars)
        if not text:
            return None
        return {
            "documentId": document_id, "parentChunkId": parent.chunk_id,
            "documentVersion": doc.version, "chunkVersion": child.version,
            "parentVersion": parent.version, "sectionPath": parent.section_path,
            "checksum": hashlib.sha256(parent.raw_content.encode("utf-8")).hexdigest(),
            "content": text, "truncated": text != parent.raw_content,
        }
    except (sqlite3.Error, OSError):
        # Optional expansion cannot break a valid child-only retrieval result.
        return None
