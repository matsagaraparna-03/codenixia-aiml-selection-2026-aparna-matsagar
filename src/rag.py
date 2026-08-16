"""
RAG - Knowledge Intelligence Layer (Milestone 6)

Architecture: Documents -> Chunking (data_pipeline.py) -> TF-IDF vectorization
              -> in-memory vector store -> retrieval (cosine similarity, top-k)
              -> passed to the LLM as grounding context

Why TF-IDF instead of a downloaded neural embedding model:
  - Zero external downloads -> fully reproducible in Docker / offline environments,
    no dependency on a model hub being reachable at runtime.
  - The knowledge base is small (a few dozen policy chunks) and domain-specific
    (HR/IT vocabulary), which is exactly the regime where sparse lexical retrieval
    (TF-IDF + cosine similarity) performs comparably to dense embeddings.
  - This trade-off (and when you WOULD switch to dense embeddings - larger,
    more varied corpora, or queries that are lexically very different from the
    source text) is documented in DECISION_LOG.md.

The index (vectorizer + matrix + chunk metadata) is built once and persisted to disk
so the app doesn't recompute it on every restart.
"""

from __future__ import annotations
import os
import json
import pickle
import logging
from dataclasses import asdict
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_pipeline import build_processed_chunks, Chunk

logger = logging.getLogger("rag")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "policies")
INDEX_DIR = os.path.join(HERE, "..", "data", "index")
VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.pkl")
MATRIX_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_metadata.json")


def build_index(force_rebuild: bool = False) -> None:
    """Build (or rebuild) the TF-IDF vector store from the processed document chunks."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not force_rebuild and all(os.path.exists(p) for p in
                                  [VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH]):
        logger.info("Index already exists, skipping build. Use force_rebuild=True to rebuild.")
        return

    chunks: List[Chunk] = build_processed_chunks(DATA_DIR)
    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams capture short policy phrases better
        max_df=0.9,
    )
    matrix = vectorizer.fit_transform(texts)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(matrix, f)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)

    logger.info("Built TF-IDF index: %d chunks, vocabulary size %d",
                matrix.shape[0], len(vectorizer.vocabulary_))


def load_index():
    if not all(os.path.exists(p) for p in [VECTORIZER_PATH, MATRIX_PATH, METADATA_PATH]):
        build_index()
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        matrix = pickle.load(f)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return vectorizer, matrix, metadata


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05) -> List[dict]:
    """Return the top_k most relevant chunks for a query, each with a similarity score.
    Chunks below min_score are dropped - this lets the caller detect 'no relevant
    knowledge found' rather than always returning something, however irrelevant."""
    vectorizer, matrix, metadata = load_index()

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        chunk = metadata[idx]
        results.append({
            "source_file": chunk["source_file"],
            "section_title": chunk["section_title"],
            "text": chunk["text"],
            "score": score,
        })
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index(force_rebuild=True)
    test_query = "how many days of sick leave do I get"
    results = retrieve(test_query, top_k=3)
    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(f"[{r['score']:.3f}] {r['source_file']} - {r['section_title']}")
        print(f"  {r['text'][:120]}...\n")
