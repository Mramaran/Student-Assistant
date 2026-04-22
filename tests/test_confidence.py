"""Task 3.1 — 4-factor confidence breakdown."""

from __future__ import annotations

from langchain_core.documents import Document


EXPECTED_FACTOR_KEYS = {"cosine", "evidence_alignment", "corroboration", "availability"}


def _docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "Minimum attendance requirement is 75 percent. "
                "Students falling below this threshold are barred from semester exams."
            ),
            metadata={"source": "attendance_rules.txt"},
        ),
        Document(
            page_content=(
                "To appear in university examinations, a minimum of 75 percent "
                "attendance is mandatory in each course."
            ),
            metadata={"source": "exam_policy.txt"},
        ),
    ]


def test_score_contract_shape():
    from verification.confidence import score

    answer = (
        "Minimum attendance is 75 percent [Source 1]. "
        "Students below 75 percent are barred from exams [Source 2]."
    )
    result = score(answer, _docs(), verifier_output={"verdict": "Supported"})

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"composite", "factors", "weights", "label"}, (
        f"Missing required top-level keys: {sorted(set(result.keys()))}"
    )

    composite = result["composite"]
    assert isinstance(composite, float) and 0.0 <= composite <= 1.0

    factors = result["factors"]
    assert isinstance(factors, dict)
    assert set(factors.keys()) == EXPECTED_FACTOR_KEYS, (
        f"factors must be exactly {EXPECTED_FACTOR_KEYS}; got {set(factors.keys())}"
    )
    for key, value in factors.items():
        assert isinstance(value, float), f"factor {key!r} must be float"
        assert 0.0 <= value <= 1.0, f"factor {key!r} must be in [0, 1]; got {value}"

    weights = result["weights"]
    assert set(weights.keys()) == EXPECTED_FACTOR_KEYS
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0 (got {total})"

    assert result["label"] in {"High", "Medium", "Low"}, (
        f"label must be one of High/Medium/Low; got {result['label']!r}"
    )


def test_composite_equals_weighted_sum():
    from verification.confidence import score

    result = score(
        "Minimum attendance is 75 percent [Source 1].",
        _docs(),
        verifier_output={"verdict": "Supported"},
    )
    recomputed = sum(result["factors"][k] * result["weights"][k] for k in result["factors"])
    assert abs(recomputed - result["composite"]) < 1e-6, (
        f"Composite must equal sum(factor*weight): got {recomputed}, expected {result['composite']}"
    )


def test_availability_scales_with_k():
    """availability = min(n_retrieved / TOP_K, 1.0)."""
    from verification.confidence import score
    from core.config import TOP_K

    docs = _docs()  # 2 docs
    result = score("any answer", docs)
    expected = min(len(docs) / TOP_K, 1.0)
    assert abs(result["factors"]["availability"] - expected) < 1e-6


def test_label_thresholds():
    """Labels fall out of CONF_HIGH (0.75) and CONF_MEDIUM (0.45)."""
    from verification.confidence import _classify
    from core.config import CONF_HIGH, CONF_MEDIUM

    assert _classify(CONF_HIGH) == "High"
    assert _classify(CONF_HIGH - 0.01) == "Medium"
    assert _classify(CONF_MEDIUM) == "Medium"
    assert _classify(CONF_MEDIUM - 0.01) == "Low"
    assert _classify(0.0) == "Low"


def test_empty_docs_degrades_gracefully():
    from verification.confidence import score

    result = score("Something the model made up.", [], verifier_output={"verdict": "Not Supported"})
    assert result["label"] == "Low"
    assert result["factors"]["availability"] == 0.0
    assert result["factors"]["corroboration"] == 0.0
