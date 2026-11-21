"""Regression tests for offline embeddings and split SQLite FTS retrieval."""

import math
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def unavailable_embedding(monkeypatch):
    from backend.rag.v2 import providers

    def missing_model(*args, **kwargs):
        assert kwargs.get("local_files_only") is True
        raise OSError("test model is not available locally")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing_model)
    )
    monkeypatch.setattr(providers, "RAG_ALLOW_MODEL_DOWNLOAD", False)
    return providers


def test_explicit_hash_fallback_handles_missing_model(unavailable_embedding, monkeypatch):
    providers = unavailable_embedding
    monkeypatch.setattr(providers, "RAG_ALLOW_HASH_FALLBACK", True)
    provider = providers.SentenceTransformersEmbeddingProvider(device="cpu")

    query_vector = provider.embed_query("accident 122")
    document_vectors = provider.embed_documents(["accident 122", "traffic 120"])

    assert len(query_vector) == provider.get_dimension() == 1024
    assert query_vector == document_vectors[0]
    assert query_vector != document_vectors[1]
    assert math.isclose(sum(value * value for value in query_vector), 1.0)
    assert provider.is_degraded()
    assert provider.get_resolved_model_name() == "hash-fallback"
    assert "not available locally" in provider.get_degraded_reason()


def test_hash_fallback_remains_opt_in(unavailable_embedding, monkeypatch):
    providers = unavailable_embedding
    monkeypatch.setattr(providers, "RAG_ALLOW_HASH_FALLBACK", False)
    provider = providers.SentenceTransformersEmbeddingProvider(device="cpu")

    with pytest.raises(RuntimeError, match="Embedding model unavailable"):
        provider.embed_query("accident 122")


def test_explicit_hash_fallback_handles_encoding_failure(unavailable_embedding, monkeypatch):
    providers = unavailable_embedding
    monkeypatch.setattr(providers, "RAG_ALLOW_HASH_FALLBACK", True)
    provider = providers.SentenceTransformersEmbeddingProvider(device="cpu")

    def failed_encode(*args, **kwargs):
        raise RuntimeError("test encoding failure")

    provider._loaded = True
    provider._model = SimpleNamespace(encode=failed_encode)

    assert len(provider.embed_text("accident 122")) == 1024
    assert provider.is_degraded()
    assert "test encoding failure" in provider.get_degraded_reason()


@pytest.fixture(params=["split", "shared"])
def sparse_stores(request, tmp_path, monkeypatch):
    from backend.rag.v2 import config

    # Include spaces and apostrophes to exercise bound database filenames.
    directory = tmp_path / "knowledge store's files"
    directory.mkdir()
    document_db = directory / "documents.db"
    fts_db = directory / "fts.db" if request.param == "split" else document_db
    monkeypatch.setattr(config, "RAG_V2_DB_PATH", str(document_db))
    monkeypatch.setattr(config, "RAG_V2_FTS_PATH", str(fts_db))

    from backend.rag.v2 import document_repository, sparse_index
    from backend.rag.v2.models import DocType, RagChunk, RagDocument

    monkeypatch.setattr(document_repository, "RAG_V2_DB_PATH", str(document_db))
    monkeypatch.setattr(sparse_index, "RAG_V2_FTS_PATH", str(fts_db))
    document_repository.init_db()
    sparse_index.init_fts()

    for document_id, doc_type in [
        ("rule_active", DocType.RULE),
        ("case_active", DocType.EVENT_REPORT),
        ("rule_deleted", DocType.RULE),
    ]:
        document_repository.upsert_document(
            RagDocument(
                document_id=document_id,
                source_id="test:" + document_id,
                title="Title " + document_id,
                content="accident 122 response 120",
                doc_type=doc_type,
                source_uri="local-test://" + document_id,
            )
        )
        chunks = [RagChunk(
            chunk_id="chunk_" + document_id,
            document_id=document_id,
            raw_content="accident 122 response 120",
            contextual_content="accident 122 response 120",
            doc_type=doc_type,
            region_id="region_test",
        )]
        document_repository.upsert_chunks(chunks)
        assert sparse_index.upsert_chunks_fts(chunks)

    document_repository.soft_delete_document("rule_deleted")
    return sparse_index


def test_fts_joins_document_metadata_across_stores(sparse_stores, caplog):
    results = sparse_stores._search_fts5("122", 10, None)

    assert {item["document_id"] for item in results} == {"rule_active", "case_active"}
    for item in results:
        assert item["title"] == "Title " + item["document_id"]
        assert item["source_uri"] == "local-test://" + item["document_id"]
        assert item["region_id"] == "region_test"
    assert "FTS5 search failed" not in caplog.text


def test_sparse_retrieval_does_not_hide_fts_failure(sparse_stores, monkeypatch):
    def unexpected_fallback(*args, **kwargs):
        pytest.fail("FTS should return results without a Python BM25 fallback")

    monkeypatch.setattr(sparse_stores, "_search_bm25_python", unexpected_fallback)
    results = sparse_stores.search_sparse("122", top_k=10)

    assert {item["document_id"] for item in results} == {"rule_active", "case_active"}


def test_fts_applies_document_filters_before_limit(sparse_stores):
    results = sparse_stores._search_fts5("122", 1, None, {"case_active"})
    assert [item["document_id"] for item in results] == ["case_active"]

    results = sparse_stores._search_fts5("122", 1, "rule")
    assert [item["document_id"] for item in results] == ["rule_active"]
    assert sparse_stores._search_fts5("122", 1, None, set()) == []
