"""rag/generator.py — legacy-name shim.

The implementation lives in ``rag.chain.answer`` now (LCEL + Gemini). This
module is retained so legacy imports (``from rag.generator import generate_answer``)
keep working. It delegates to ``rag.chain.answer`` and returns the chain's
contract dict unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from rag.chain import answer as _chain_answer


def generate_answer(
    query: str,
    docs: Any = None,  # kept for signature back-compat; ignored by the chain
    stream_placeholder: Optional[Any] = None,  # retained for legacy callers
    **_unused,
) -> dict:
    """Delegate to ``rag.chain.answer`` and return the contract dict.

    ``docs`` and ``stream_placeholder`` are accepted to preserve the legacy
    call shape; they are not forwarded because the LCEL chain runs its own
    retriever and does not stream via Streamlit placeholders.
    """
    return _chain_answer(query)
