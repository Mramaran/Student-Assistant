"""Task 1.7 — slim core/config.py and core/feature_flags.py."""

from __future__ import annotations


REQUIRED_CONFIG_KEYS = {
    "DB_DIR",
    "TOP_K",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "COLLECTION_NAME",
    "GEMINI_MODEL",
    "ESCALATION_EMAIL",
    "CONF_HIGH",
    "CONF_MEDIUM",
    "CACHE_TTL_DAYS",
}

FORBIDDEN_SUBSTRINGS = ("TUTOR", "MATCH", "COMPETENCY", "STUDY_PLAN")


def test_config_keys_minimal():
    import core.config as cfg

    names = {n for n in vars(cfg) if not n.startswith("_")}

    missing = REQUIRED_CONFIG_KEYS - names
    assert missing == set(), f"Missing required config keys: {sorted(missing)}"

    offenders = [
        name
        for name in names
        if any(forbidden in name.upper() for forbidden in FORBIDDEN_SUBSTRINGS)
    ]
    assert offenders == [], f"Forbidden config keys present: {offenders}"


def test_feature_flags_only_five():
    from core.feature_flags import DEFAULTS

    expected = {
        "confidence_scoring",
        "contradiction_alerts",
        "human_escalation",
        "benchmark_suite",
        "safety_guardrails",
    }
    actual = set(DEFAULTS.keys())
    assert actual == expected, (
        f"DEFAULTS must be exactly the 5 UC4 flags. "
        f"Missing={sorted(expected - actual)}, Extra={sorted(actual - expected)}"
    )
    # All flags default True per the plan.
    for flag, value in DEFAULTS.items():
        assert value is True, f"Flag {flag!r} must default True, got {value!r}"
