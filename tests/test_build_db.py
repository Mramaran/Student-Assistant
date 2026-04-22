"""Task 1.5 — build_db must be idempotent via a checksum manifest."""

from __future__ import annotations

import json
from pathlib import Path


def test_idempotent_reingest(tmp_path: Path):
    """Second run on an unchanged fixture dir must report 0 new embeddings."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "CIT_handbook.txt").write_text(
        "CIT handbook. Minimum attendance requirement is 75 percent.",
        encoding="utf-8",
    )
    db_dir = str(tmp_path / "db")

    from build_db import build_db

    result1 = build_db(data_dir=str(data_dir), db_dir=db_dir)
    assert result1["new_files"] >= 1, f"Run 1 must ingest new files: {result1}"
    assert result1["new_embeddings"] >= 1, f"Run 1 must create embeddings: {result1}"

    result2 = build_db(data_dir=str(data_dir), db_dir=db_dir)
    assert result2["new_files"] == 0, f"Expected 0 new files on reingest: {result2}"
    assert result2["new_embeddings"] == 0, (
        f"Expected 0 new embeddings on reingest: {result2}"
    )

    manifest_path = data_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must be written alongside the corpus"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "CIT_handbook.txt" in manifest, f"Manifest missing fixture file: {manifest}"
    assert len(manifest["CIT_handbook.txt"]) == 40, "Manifest value must be a sha1 hex"


def test_changed_file_triggers_reingest(tmp_path: Path):
    """When any file's checksum changes, build_db must re-ingest."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture = data_dir / "CIT_handbook.txt"
    fixture.write_text("Minimum attendance is 75 percent.", encoding="utf-8")
    db_dir = str(tmp_path / "db")

    from build_db import build_db

    build_db(data_dir=str(data_dir), db_dir=db_dir)  # initial build
    fixture.write_text("Minimum attendance is 80 percent.", encoding="utf-8")

    result = build_db(data_dir=str(data_dir), db_dir=db_dir)
    assert result["new_files"] >= 1 or result["new_embeddings"] >= 1, (
        f"Expected re-ingest after file change: {result}"
    )
