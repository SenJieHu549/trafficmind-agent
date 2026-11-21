"""Chinese query labels must retrieve canonical English and legacy documents."""

from types import SimpleNamespace
import uuid

import pytest


@pytest.fixture
def retriever(monkeypatch):
    import chromadb
    from backend.rag.v2 import dense_index
    from backend.rag.v2.hybrid_retriever import HybridRetriever

    collection = chromadb.EphemeralClient().create_collection(
        name="event_alias_" + uuid.uuid4().hex,
        embedding_function=None,
    )
    collection.add(
        ids=["canonical", "legacy", "other"],
        embeddings=[[1.0, 0.0]] * 3,
        metadatas=[{"event_type": value} for value in ("accident", "事故", "congestion")],
    )

    def search(query_embedding, top_k=10, where=None, collection_name=None):
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        return [{"chunk_id": item} for item in results["ids"][0]]

    monkeypatch.setattr(dense_index, "is_available", lambda: True)
    monkeypatch.setattr(dense_index, "get_active_collection_name", lambda: collection.name)
    monkeypatch.setattr(dense_index, "search_dense", search)
    return HybridRetriever(SimpleNamespace(embed_query=lambda text: [1.0, 0.0]))


@pytest.mark.parametrize("event_type", ["事故", "accident"])
def test_dense_retrieval_accepts_event_type_aliases(retriever, event_type):
    from backend.rag.v2.models import QueryAnalysis

    results = retriever._dense_retrieve(
        "事故处置", 10, QueryAnalysis(filters={"event_type": event_type}), None
    )

    assert {item["chunk_id"] for item in results} == {"canonical", "legacy"}
