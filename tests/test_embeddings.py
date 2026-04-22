"""Task 1.1 — assert embeddings use HuggingFace (not Ollama)."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_embedder_is_huggingface():
    """Factory returns a HuggingFace embedder that produces EMBEDDING_DIM-dim vectors."""
    from core.config import EMBEDDING_DIM
    from embeddings.embedder import get_embedding_function

    embed = get_embedding_function()
    cls_name = type(embed).__name__
    assert "HuggingFace" in cls_name, f"Expected HuggingFace embedder, got {cls_name}"
    assert not cls_name.startswith("Ollama"), f"Unexpected Ollama embedder: {cls_name}"

    vec = embed.embed_query("cit handbook")
    assert isinstance(vec, list), f"Expected list, got {type(vec).__name__}"
    assert vec, "Embedding vector must be non-empty"
    assert all(isinstance(x, float) for x in vec), "Vector must be floats"
    assert len(vec) == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim vector, got {len(vec)}"
    )


def test_no_ollama_embeddings_in_rag_layer():
    """No module under embeddings/, rag/, or build_db.py may use OllamaEmbeddings."""
    targets = ["embeddings", "rag", "build_db.py"]
    offenders = []
    for target in targets:
        path = REPO_ROOT / target
        if not path.exists():
            continue
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "OllamaEmbeddings" in text or "langchain_ollama" in text:
                offenders.append(file.relative_to(REPO_ROOT).as_posix())
    assert offenders == [], f"Still references Ollama embeddings: {offenders}"
