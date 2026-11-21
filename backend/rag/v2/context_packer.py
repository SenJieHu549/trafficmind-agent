"""
RAG V2 Context Packer — Parent Context Pack + token-aware evidence formatting.

For each selected child chunk, include its parent chunk for context.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from backend.rag.v2.config import RAG_MAX_CONTEXT_TOKENS
from backend.rag.v2.models import EvidenceItem
from backend.rag.v2.context_expansion import bounded_excerpt, resolve_parent_context


class ContextPacker:
    """上下文打包器。"""

    def __init__(self, max_tokens: int = RAG_MAX_CONTEXT_TOKENS, max_evidence_chars: int = 3200):
        self.max_tokens = max_tokens
        self.max_evidence_chars = max(0, max_evidence_chars)

    def pack(
        self,
        evidence_items: List[EvidenceItem],
        include_parents: bool = True,
    ) -> str:
        """打包证据为 LLM prompt 上下文。

        Args:
            evidence_items: 选中的证据列表
            include_parents: 是否包含父 chunk 上下文

        Returns:
            格式化的上下文字符串
        """
        if not evidence_items:
            return "（无有效证据）"

        parts = []
        total_chars = 0
        char_budget = self.max_tokens * 2  # approximate: 2 chars per token
        expanded_parents = set()

        for e in evidence_items:
            section_header = f"\n[{e.evidence_id}] {e.title or '知识片段'}"
            if e.section_path:
                section_header += f" — {e.section_path}"
            section_header += f"\n类型: {e.doc_type} | 权威等级: {e.authority_level}"

            if e.effective_from or e.effective_to:
                eff = f"{e.effective_from.strftime('%Y-%m-%d') if e.effective_from else '无'}"
                exp = f"{e.effective_to.strftime('%Y-%m-%d') if e.effective_to else '无'}"
                section_header += f" | 有效期: {eff} ~ {exp}"

            section_header += f"\n来源: {e.document_id}/{e.chunk_id}"
            if e.version is not None:
                section_header += f" | 版本: v{e.version}"
            available = min(self.max_evidence_chars, char_budget - total_chars - len(section_header) - 2)
            if available <= 0:
                break
            body = (e.contextual_content or e.content) if include_parents else e.content
            body = body[:800]
            if include_parents:
                expansion = resolve_parent_context(
                    e.document_id, e.chunk_id, parent_chunk_id=e.parent_chunk_id,
                    version=e.version, expected_content=e.content or None,
                    expected_scope={key: getattr(e, key) for key in (
                        "region_id", "road_id", "intersection_id", "event_type",
                        "effective_from", "effective_to",
                    ) if getattr(e, key) is not None},
                )
                if expansion:
                    identity = (expansion["documentId"], expansion["parentChunkId"],
                                expansion["parentVersion"])
                    if identity not in expanded_parents:
                        header = (f"\n父块上下文: {expansion['parentChunkId']}"
                                  f" | 版本: v{expansion['parentVersion']}\n")
                        parent_budget = available - len(body) - len(header)
                        if parent_budget > 64:
                            body += header + bounded_excerpt(expansion["content"], parent_budget)
                            expanded_parents.add(identity)
            block = f"{section_header}\n{bounded_excerpt(body, available)}"
            parts.append(block)
            total_chars += len(block) + (2 if len(parts) > 1 else 0)

        return "\n\n".join(parts)

    def format_evidence_for_display(self, evidence_items: List[EvidenceItem]) -> List[Dict]:
        """格式化为前端可用的结构化证据展示。"""
        return [
            {
                "evidenceId": e.evidence_id,
                "title": e.title,
                "sectionPath": e.section_path,
                "documentId": e.document_id,
                "chunkId": e.chunk_id,
                "parentChunkId": e.parent_chunk_id,
                "version": e.version,
                "docType": e.doc_type,
                "authorityLevel": e.authority_level,
                "effectiveFrom": e.effective_from.isoformat() if e.effective_from else None,
                "effectiveTo": e.effective_to.isoformat() if e.effective_to else None,
                "retrievalChannels": e.retrieval_channels,
                "rrfScore": e.rrf_score,
                "rerankScore": e.rerank_score,
                "content": e.content[:300],
                "sourceUri": e.source_uri,
            }
            for e in evidence_items
        ]
