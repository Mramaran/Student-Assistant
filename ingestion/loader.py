"""Document loading utilities for ingestion.

Output contract (one dict per logical document page):
  {
    "text":     "...",               # raw text
    "source":   "filename.ext",      # basename
    "doc_type": "handbook" | "exam" | ...,  # inferred category
    "checksum": "<sha1 of file bytes>",     # 40-char hex
    "page":     int                  # PRESENT ONLY FOR PDFs
  }
"""

from __future__ import annotations

import hashlib
import os
from html.parser import HTMLParser
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)


# OCR dependencies are loaded lazily to prevent errors on systems without Tesseract
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg"}


DOC_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "handbook":    ("handbook", "academic_info", "regulation", "conduct", "basic_facts"),
    "exam":        ("exam", "grading", "grade_system"),
    "scholarship": ("scholarship",),
    "placement":   ("placement",),
    "admission":   ("admission", "brochure"),
    "facilities":  ("facilities", "hostel", "lab_facilities", "infrastructure", "library"),
}


def _infer_doc_type(filename: str) -> str:
    """Map a filename to a UC4 taxonomy category; default 'other'."""
    lower = filename.lower()
    for doc_type, keywords in DOC_TYPE_PATTERNS.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return "other"


def _compute_checksum(path: Path) -> str:
    """Return sha1 hex of the file's bytes (40 chars)."""
    hasher = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class _SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self.parts)


def _load_one_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    doc_type = _infer_doc_type(path.name)
    checksum = _compute_checksum(path)

    # ── Image OCR ────────────────────────────────────────────────────────────
    if suffix in {".png", ".jpg", ".jpeg"}:
        if not HAS_OCR:
            print(f"⚠️  Skipping {path.name}: OCR dependencies not found.")
            return []
        try:
            text = pytesseract.image_to_string(Image.open(path))
            return [{
                "text":     text,
                "source":   path.name,
                "doc_type": doc_type,
                "checksum": checksum,
            }]
        except Exception as exc:
            print(f"❌ OCR failed for {path.name}: {exc}")
            return []

    # ── HTML / Markdown plain read ───────────────────────────────────────────
    if suffix in {".html", ".htm"}:
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        parser = _SimpleHTMLTextExtractor()
        parser.feed(html_text)
        return [{
            "text":     parser.get_text(),
            "source":   path.name,
            "doc_type": doc_type,
            "checksum": checksum,
        }]

    if suffix == ".md":
        return [{
            "text":     path.read_text(encoding="utf-8", errors="ignore"),
            "source":   path.name,
            "doc_type": doc_type,
            "checksum": checksum,
        }]

    # ── LangChain loaders (.txt, .pdf, .docx) ────────────────────────────────
    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        return []

    is_pdf = suffix == ".pdf"
    results: list[dict] = []
    for idx, doc in enumerate(loader.load()):
        source_name = os.path.basename(doc.metadata.get("source", str(path)))
        entry: dict = {
            "text":     doc.page_content or "",
            "source":   source_name,
            "doc_type": doc_type,
            "checksum": checksum,
        }
        if is_pdf:
            entry["page"] = int(doc.metadata.get("page", idx + 1))
        results.append(entry)
    return results


def load_documents(data_dir: str, exclude_dirs: tuple[str, ...] = ()) -> list[dict]:
    """Load all supported documents from ``data_dir`` (recursively).

    Any file whose relative path contains a directory segment listed in
    ``exclude_dirs`` (case-insensitive match) is skipped — used to keep
    adversarial corpora like ``data/redteam/`` out of the default build.
    """
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        return []

    excluded = {e.strip("/").lower() for e in exclude_dirs}
    documents: list[dict] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel_parts = {p.lower() for p in path.relative_to(directory).parts[:-1]}
        if rel_parts & excluded:
            continue
        try:
            documents.extend(_load_one_file(path))
        except Exception as exc:
            print(f"[loader] Skipping {path.name}: {exc}")
    return documents
