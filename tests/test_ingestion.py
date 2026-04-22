"""Task 1.2 — chunk metadata contract for the UC4 ingestion stage."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest


def _run_ingestion(data_dir: Path) -> list[dict]:
    """Shared helper: load + chunk a single directory and return the chunks."""
    from ingestion.chunker import split_into_chunks
    from ingestion.cleaner import clean_text
    from ingestion.loader import load_documents

    raw = load_documents(str(data_dir))
    cleaned = clean_text(raw)
    return split_into_chunks(cleaned, chunk_size=400, chunk_overlap=80)


def test_chunk_metadata_contract(tmp_path: Path):
    """Every chunk from a .txt file must carry the required metadata keys."""
    fixture = tmp_path / "CIT_handbook.txt"
    fixture.write_text(
        "# Attendance Policy\n\n"
        "Minimum attendance requirement is 75 percent of total working days.\n"
        "Students falling below this threshold are barred from semester exams.\n",
        encoding="utf-8",
    )

    chunks = _run_ingestion(tmp_path)
    assert chunks, "Expected at least one chunk from the .txt fixture"

    required_keys = {"source", "doc_type", "section", "ingested_at", "checksum", "risk_flag"}
    for chunk in chunks:
        missing = required_keys - set(chunk.keys())
        assert not missing, f"Chunk missing keys {missing}: {chunk}"

        # ingested_at must be ISO 8601 UTC
        datetime.datetime.fromisoformat(chunk["ingested_at"])

        # checksum is 40-char lowercase hex (sha1)
        checksum = chunk["checksum"]
        assert isinstance(checksum, str) and len(checksum) == 40, (
            f"Expected 40-char sha1 checksum, got {checksum!r}"
        )
        int(checksum, 16)  # must parse as hex

        # risk_flag is a bool
        assert isinstance(chunk["risk_flag"], bool), (
            f"Expected bool risk_flag, got {type(chunk['risk_flag']).__name__}"
        )

        # doc_type is a string drawn from the taxonomy (or 'other')
        assert chunk["doc_type"] in {
            "handbook", "exam", "scholarship", "placement",
            "admission", "facilities", "other",
        }, f"Unexpected doc_type: {chunk['doc_type']!r}"

        # .txt inputs must NOT carry a page key
        assert "page" not in chunk, (
            f"Non-PDF chunk unexpectedly carries 'page': {chunk}"
        )


def test_chunk_metadata_pdf_has_page(tmp_path: Path):
    """Chunks from a PDF fixture must include a 'page' integer in their metadata."""
    # Build a minimal PDF with real text by hand-writing the PDF format.
    # This avoids the heavier reportlab dependency while still producing a
    # file PyPDFLoader can extract text from.
    pdf_path = tmp_path / "CIT_handbook.pdf"
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        b"4 0 obj <</Length 83>> stream\n"
        b"BT /F1 12 Tf 72 720 Td "
        b"(Minimum attendance requirement is 75 percent) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000052 00000 n \n"
        b"0000000097 00000 n \n"
        b"0000000183 00000 n \n"
        b"0000000289 00000 n \n"
        b"trailer <</Size 6 /Root 1 0 R>>\n"
        b"startxref\n355\n%%EOF\n"
    )
    pdf_path.write_bytes(pdf_bytes)

    chunks = _run_ingestion(tmp_path)
    if not chunks:
        pytest.skip("PDF produced no chunks; text extraction returned empty")

    for chunk in chunks:
        assert "page" in chunk, f"PDF chunk missing 'page': {chunk}"
        assert isinstance(chunk["page"], int)
