# Debugging Report

## Issue 1: Embedding model download failed in a network-restricted environment

**What failed:** The original RAG implementation used `sentence-transformers`
(`all-MiniLM-L6-v2`) for dense embeddings, loaded via Hugging Face Hub at runtime.

**Error:** `huggingface_hub.errors.HfHubHTTPError: 403 Forbidden` /
`OSError: We couldn't connect to 'https://huggingface.co' to load the files`.

**Investigation:** The development/build environment only allows outbound network
access to a small allow-list of package registries (PyPI, npm, GitHub) — Hugging Face
Hub was not reachable. This is realistic: many corporate or CI environments restrict
outbound network access for security reasons, and a production deployment might face
the same restriction.

**Solution implemented:** Replaced the dense-embedding approach with
`scikit-learn`'s `TfidfVectorizer` + cosine similarity. This runs entirely locally with
no model download required — the "embedding" step is just fitting a vectorizer on the
local document corpus.

**How it was verified:** Reran the retrieval test (`rag.py` main block and
`test_retrieval_returns_relevant_chunk`) — a query like "how many days of sick leave do
I get" correctly retrieves the Sick Leave section with a high similarity score
(0.71), with no external network calls at runtime.

**Trade-off accepted:** TF-IDF is lexical, not semantic, so it's more sensitive to
vocabulary mismatch between the query and the source text than a neural embedding would
be. Documented as a known limitation in `README.md` and as a decision in
`DECISION_LOG.md`.

---

## Issue 2: Title-only chunks were polluting retrieval results

**What failed:** After implementing section-based chunking (splitting on `##`
headings), a generic query like "policy" returned mostly document titles
(e.g. `"# Leave Policy"`) instead of the actual relevant policy content.

**What happened:** Every markdown document starts with a level-1 `# Title` heading
before its first `##` section. The chunking logic split on `##` headings, so the
title line became its own tiny chunk. Because TF-IDF similarity is influenced by
document length (shorter documents/chunks concentrate their term frequency on fewer
words), these near-empty title chunks scored deceptively high on generic queries and
crowded out the top-k results.

**How I investigated:** Ran `rag.retrieve("policy", top_k=5)` directly and printed
the chunk text alongside its score — the top 3 results were literally just document
titles (`# Leave Policy`, `# Expense Reimbursement Policy`, `# Work From Home (WFH)
Policy`), confirming the chunker, not the vectorizer, was the source of the problem.

**Solution implemented:** Added a minimum chunk-length filter (`MIN_CHUNK_CHARS = 40`)
in `data_pipeline.py::build_processed_chunks`, dropping near-empty chunks after
chunking, with the drop count logged for visibility.

**How it was verified:** Rebuilt the index and reran the same query — the title-only
chunks no longer appear in results, and the full test suite (`pytest tests/`) still
passes with the fix applied.
