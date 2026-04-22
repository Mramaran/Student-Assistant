"""Build the ChromaDB vector store for the UC4 RAG pipeline.

``build_db(data_dir, db_dir, manifest_path)`` is idempotent via a sha1-per-file
manifest at ``data_dir/manifest.json``. When every file's checksum matches the
manifest, the function short-circuits and returns zeros for all counters.
Any new / changed / removed file triggers a full rebuild (simple MVP —
per-file delta is out of scope for Phase 1).

Returned result dict:
    new_files:       int — count of files that are new or changed this run
    new_embeddings:  int — total chunks embedded this run
    skipped_files:   int — files whose checksum matched the manifest
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable, Optional

from ingestion.chunker import split_into_chunks
from ingestion.cleaner import clean_text
from ingestion.loader import _compute_checksum, load_documents, SUPPORTED_EXTENSIONS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DB_DIR,
    EMBEDDING_MODEL,
    INSTITUTION_NAME,
)


BM25_CACHE = "bm25_docs.pkl"


def _scan_corpus(data_dir: Path, exclude_dirs: Iterable[str]) -> dict[str, str]:
    """Walk ``data_dir`` recursively and return ``{relative_posix_path: sha1}``."""
    excluded = {e.strip("/").lower() for e in exclude_dirs}
    checksums: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel = path.relative_to(data_dir).as_posix()
        # Skip excluded directories (any path segment matches).
        parts = {p.lower() for p in path.relative_to(data_dir).parts[:-1]}
        if parts & excluded:
            continue
        if path.name == "manifest.json":
            continue
        checksums[rel] = _compute_checksum(path)
    return checksums


def build_db(
    data_dir: str = DATA_DIR,
    db_dir: str = DB_DIR,
    manifest_path: Optional[str] = None,
    exclude_dirs: Iterable[str] = ("redteam", "extended"),
) -> dict:
    """Ingest ``data_dir`` into a Chroma vector store at ``db_dir``.

    Skips the full pipeline when every file's sha1 matches the saved manifest.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Created '{data_dir}'. Add your files, then run again.")
        return {"new_files": 0, "new_embeddings": 0, "skipped_files": 0}

    manifest_file = Path(manifest_path) if manifest_path else data_path / "manifest.json"

    current = _scan_corpus(data_path, exclude_dirs)
    previous: dict[str, str] = {}
    if manifest_file.exists():
        try:
            previous = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}

    new_or_changed = [rel for rel, sha in current.items() if previous.get(rel) != sha]
    removed = [rel for rel in previous if rel not in current]

    if not new_or_changed and not removed:
        print(f"No changes detected — reusing existing DB at {db_dir}.")
        return {
            "new_files":       0,
            "new_embeddings":  0,
            "skipped_files":   len(current),
        }

    print(
        f"Detected {len(new_or_changed)} new/changed file(s) and "
        f"{len(removed)} removed file(s); rebuilding..."
    )

    raw_docs = load_documents(str(data_path), exclude_dirs=tuple(exclude_dirs))
    if not raw_docs:
        print("No ingestable documents found.")
        return {"new_files": 0, "new_embeddings": 0, "skipped_files": 0}

    cleaned = clean_text(raw_docs)
    chunks = split_into_chunks(cleaned, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    doc_objects = [
        Document(
            page_content=c["text"],
            metadata={
                "source":      c["source"],
                "page":        c.get("page"),  # present only for PDF inputs
                "section":     c.get("section", "General"),
                "doc_type":    c.get("doc_type", "other"),
                "checksum":    c.get("checksum", ""),
                "ingested_at": c.get("ingested_at", ""),
                "risk_flag":   bool(c.get("risk_flag", False)),
                "department":  c.get("dept", INSTITUTION_NAME),
            },
        )
        for c in chunks
    ]

    # Filter out None values from metadata (Chroma rejects None).
    for doc in doc_objects:
        doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None}

    if db_dir == DB_DIR:
        bm25_path = Path(BM25_CACHE)
    else:
        bm25_path = Path(db_dir) / "bm25_docs.pkl"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as handle:
        pickle.dump(doc_objects, handle)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except OSError as exc:
            print(f"Could not delete old DB at {db_dir}: {exc}")

    Chroma.from_documents(
        documents=doc_objects,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(current, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    return {
        "new_files":       len(new_or_changed),
        "new_embeddings":  len(doc_objects),
        "skipped_files":   len(current) - len(new_or_changed),
    }


if __name__ == "__main__":
    result = build_db()
    print(
        f"Ingest complete: {result['new_files']} new/changed file(s), "
        f"{result['new_embeddings']} embeddings, "
        f"{result['skipped_files']} skipped."
    )
