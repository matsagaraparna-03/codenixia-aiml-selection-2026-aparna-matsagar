"""
Data / Processing Pipeline (Milestone 3)

Flow: Data Source (markdown policy docs) -> Load -> Validate -> Clean -> Chunk -> Processed chunks

This module is intentionally framework-free (pure Python) so it's easy to test,
reason about, and explain in a live technical defense.
"""

from __future__ import annotations
import re
import os
import logging
from dataclasses import dataclass, asdict
from typing import List

logger = logging.getLogger("data_pipeline")


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    section_title: str
    text: str


def load_documents(data_dir: str) -> List[dict]:
    """Load all markdown files from a directory. Returns list of {filename, raw_text}."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        docs.append({"filename": fname, "raw_text": raw_text})

    if not docs:
        raise ValueError(f"No .md documents found in {data_dir}")

    logger.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs


def validate_document(doc: dict) -> bool:
    """Basic data-quality validation: non-empty, has at least one heading."""
    text = doc.get("raw_text", "")
    if not text or not text.strip():
        logger.warning("Rejected empty document: %s", doc.get("filename"))
        return False
    if "#" not in text:
        logger.warning("Document has no markdown headings: %s", doc.get("filename"))
        return False
    return True


def clean_text(text: str) -> str:
    """Normalize whitespace, strip stray characters."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(doc: dict, max_chunk_chars: int = 800) -> List[Chunk]:
    """
    Chunk a markdown document by section (## headings), then further split
    any section that's too long. Section-based chunking keeps semantically
    related content together, which improves retrieval quality.
    """
    text = clean_text(doc["raw_text"])
    filename = doc["filename"]

    # Split on level-2 headings ("## Something"), keep the heading with its body
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks: List[Chunk] = []
    chunk_counter = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        title_match = re.match(r"^##\s+(.+)$", section, flags=re.MULTILINE)
        section_title = title_match.group(1).strip() if title_match else filename

        if len(section) <= max_chunk_chars:
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"{filename}::{chunk_counter}",
                source_file=filename,
                section_title=section_title,
                text=section,
            ))
        else:
            # Sub-split long sections by sentence boundaries to stay under the char limit
            sentences = re.split(r"(?<=[.!?])\s+", section)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) > max_chunk_chars and buffer:
                    chunk_counter += 1
                    chunks.append(Chunk(
                        chunk_id=f"{filename}::{chunk_counter}",
                        source_file=filename,
                        section_title=section_title,
                        text=buffer.strip(),
                    ))
                    buffer = ""
                buffer += sentence + " "
            if buffer.strip():
                chunk_counter += 1
                chunks.append(Chunk(
                    chunk_id=f"{filename}::{chunk_counter}",
                    source_file=filename,
                    section_title=section_title,
                    text=buffer.strip(),
                ))

    return chunks


MIN_CHUNK_CHARS = 40  # drop near-empty chunks (e.g. a lone "# Title" line with no body)


def build_processed_chunks(data_dir: str) -> List[Chunk]:
    """Full pipeline: load -> validate -> clean -> chunk. This is the single entry point
    used by the RAG layer."""
    raw_docs = load_documents(data_dir)
    valid_docs = [d for d in raw_docs if validate_document(d)]

    if len(valid_docs) < len(raw_docs):
        logger.warning("%d documents failed validation and were skipped",
                        len(raw_docs) - len(valid_docs))

    all_chunks: List[Chunk] = []
    for doc in valid_docs:
        all_chunks.extend(chunk_document(doc))

    # Drop title-only / near-empty chunks - see DEBUGGING_REPORT.md. These carry almost
    # no information but their short length inflates their TF-IDF similarity to generic
    # queries (e.g. a query containing the word "policy" matches "# Leave Policy" with a
    # deceptively high score), crowding out genuinely useful chunks from the top-k results.
    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if len(c.text) >= MIN_CHUNK_CHARS]
    dropped = before - len(all_chunks)
    if dropped:
        logger.info("Dropped %d near-empty chunks (< %d chars)", dropped, MIN_CHUNK_CHARS)

    logger.info("Produced %d chunks from %d documents", len(all_chunks), len(valid_docs))
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data", "policies")
    chunks = build_processed_chunks(data_dir)
    for c in chunks[:5]:
        print(asdict(c))
    print(f"\nTotal chunks: {len(chunks)}")
