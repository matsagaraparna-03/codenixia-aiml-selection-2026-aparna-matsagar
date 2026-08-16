"""
Basic automated test suite (Milestone 10).

Covers: health check, input validation, RAG retrieval quality, and data pipeline
correctness. LLM/agent calls that need a live API key are NOT exercised here by
default (see test_agent_requires_api_key) so the suite can run in CI without secrets.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from main import app
from data_pipeline import build_processed_chunks, clean_text, validate_document
import rag


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- API tests ---

def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_ask_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic validation error


def test_ask_rejects_missing_question(client):
    r = client.post("/ask", json={})
    assert r.status_code == 422


def test_ask_rejects_overlong_question(client):
    r = client.post("/ask", json={"question": "a" * 5000})
    assert r.status_code == 422

def test_ask_without_api_key_fails_gracefully(client, monkeypatch):
    # Use setenv to an empty string rather than delenv: on some machines the key may
    # also be present as a persistent OS-level environment variable outside pytest's
    # control, which delenv alone wouldn't override for the duration of this test.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    r = client.post("/ask", json={"question": "How many sick days do I get?"})
    # Should return a clean 500 with a helpful message, not crash the server
    assert r.status_code == 500
    assert "GEMINI_API_KEY" in r.json()["detail"]

def test_data_pipeline_produces_chunks():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data", "policies")
    chunks = build_processed_chunks(data_dir)
    assert len(chunks) > 0
    assert all(c.text.strip() for c in chunks)


def test_clean_text_normalizes_whitespace():
    dirty = "Hello    world\r\n\n\n\nBye"
    cleaned = clean_text(dirty)
    assert "\r\n" not in cleaned
    assert "\n\n\n" not in cleaned


def test_validate_document_rejects_empty():
    assert validate_document({"filename": "empty.md", "raw_text": ""}) is False


def test_validate_document_rejects_no_headings():
    assert validate_document({"filename": "bad.md", "raw_text": "just plain text, no markdown"}) is False


def test_validate_document_accepts_valid_doc():
    assert validate_document({"filename": "ok.md", "raw_text": "# Title\n## Section\nSome text."}) is True


# --- RAG retrieval tests ---

def test_retrieval_returns_relevant_chunk():
    rag.build_index()
    results = rag.retrieve("how many sick leave days", top_k=3)
    assert len(results) > 0
    assert any("sick" in r["text"].lower() for r in results)


def test_retrieval_returns_empty_for_irrelevant_query():
    rag.build_index()
    results = rag.retrieve("what is the airspeed velocity of an unladen swallow", top_k=3, min_score=0.3)
    # An out-of-domain query should not force-match unrelated policy text
    assert len(results) == 0
