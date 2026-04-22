"""Semantic & sentence-boundary aware chunker for the UC4 pipeline.

Every chunk returned by ``split_into_chunks`` carries this metadata contract:
  text:        enriched chunk text
  source:      document filename (carried from loader)
  doc_type:    inferred category (carried from loader)
  checksum:    sha1 of the source file (carried from loader)
  section:     nearest section heading (defaults to 'General')
  ingested_at: ISO 8601 UTC timestamp when the chunker ran
  risk_flag:   bool — set True when ``services.safety_handler.scan_document``
               flags the chunk at ingest time (wired up in Task 7.2). Default
               False until then.
  page:        integer; ONLY PRESENT when the source document had a page
               attribute (i.e. PDF inputs).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.safety_handler import scan_document as _scan_document

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_DEPT_MAP = {
    "Computer Science": "CSE",
    "Information Technology": "IT",
    "Artificial Intelligence": "AI & DS",
}


def _extract_block_meta(text: str) -> dict:
    """Pull SECTION/Department hints out of the raw block text."""
    meta: dict = {}
    section_match = re.search(r"SECTION:\s*(.*)", text, re.IGNORECASE)
    if section_match:
        meta["section"] = section_match.group(1).strip()
    dept_match = re.search(r"Department:\s*(.*)", text, re.IGNORECASE)
    if dept_match:
        dept = dept_match.group(1).strip()
        for long, short in _DEPT_MAP.items():
            if long in dept:
                dept = short
                break
        meta["dept"] = dept
        meta["type"] = "department"
    return meta


def _nearest_markdown_heading(full_text: str, block_start: int) -> str | None:
    """Return the most recent Markdown heading text preceding ``block_start``."""
    last: str | None = None
    for match in _MARKDOWN_HEADING_RE.finditer(full_text, 0, block_start + 1):
        last = match.group(1).strip()
    return last


def _fallback_section(chunk_text: str) -> str:
    """Heuristic section label when no heading is available."""
    first_line = chunk_text.strip().splitlines()[0] if chunk_text.strip() else ""
    return first_line[:80] if first_line else "General"


def _split_sentences(text: str) -> list[str]:
    if _HAS_NLTK:
        return sent_tokenize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_into_chunks(
    docs: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split loader output into metadata-rich chunks.

    Parameters
    ----------
    docs : list of loader-output dicts (must contain 'text' and 'source').
    chunk_size / chunk_overlap : soft thresholds for the semantic splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    ingested_at = _utc_now_iso()
    chunks: list[dict] = []

    for doc in docs:
        filename = doc.get("source", "unknown")
        full_text = doc.get("text", "") or ""
        if not full_text:
            continue

        doc_type = doc.get("doc_type", "other")
        checksum = doc.get("checksum", "")
        page = doc.get("page")  # None for non-PDF; int for PDF

        # Track offsets so we can resolve the nearest preceding Markdown heading.
        block_pattern = re.compile(r"(?:\n\s*\n|[-=]{5,})")
        cursor = 0
        raw_blocks: list[tuple[int, str]] = []  # (start_offset, text)
        for m in block_pattern.finditer(full_text):
            if m.start() > cursor:
                raw_blocks.append((cursor, full_text[cursor:m.start()]))
            cursor = m.end()
        if cursor < len(full_text):
            raw_blocks.append((cursor, full_text[cursor:]))

        for start, block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            block_meta = _extract_block_meta(block)

            # Keep short "entity blocks" (dept/HOD cards) whole so retrieval
            # returns them as a single unit.
            is_entity_block = "Department:" in block or "HOD:" in block

            if is_entity_block or len(block) < chunk_size:
                sub_blocks = [block]
            else:
                try:
                    sub_blocks = splitter.split_text(block)
                except Exception:
                    sub_blocks = [block]

            heading = (
                block_meta.get("section")
                or _nearest_markdown_heading(full_text, start)
                or _fallback_section(block)
            )

            for sub in sub_blocks:
                sub = sub.strip()
                if not sub:
                    continue

                header = ""
                if block_meta.get("section"):
                    header += f"[SECTION: {block_meta['section']}] "
                if block_meta.get("dept"):
                    header += f"[Dept: {block_meta['dept']}] "
                enriched_text = f"{header}\n{sub}" if header else sub

                record: dict = {
                    "text":        enriched_text,
                    "source":      filename,
                    "doc_type":    doc_type,
                    "checksum":    checksum,
                    "section":     heading,
                    "ingested_at": ingested_at,
                    "risk_flag":   bool(_scan_document(sub)),
                }
                if page is not None:
                    record["page"] = int(page)
                if block_meta.get("dept"):
                    record["dept"] = block_meta["dept"]
                chunks.append(record)

    return chunks
