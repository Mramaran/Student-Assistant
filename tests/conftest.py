"""Test-session configuration.

Responsibilities:
  1. Force ``GEMINI_MODEL`` to a lighter variant for pytest — production
     defaults in ``core/config.py`` and ``.env`` are untouched.
  2. Export a ``skip_on_quota`` helper that converts Gemini
     ``RESOURCE_EXHAUSTED`` (429) failures into ``pytest.skip``, so hitting
     the free-tier daily cap does not masquerade as a regression.
"""

from __future__ import annotations

import functools
import os
from typing import Callable

import pytest


os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash-lite")


def _looks_like_quota_error(exc: BaseException) -> bool:
    text = repr(exc) + " " + str(exc)
    return "RESOURCE_EXHAUSTED" in text or "quota" in text.lower() or " 429" in text


def skip_on_quota(fn: Callable) -> Callable:
    """Wrap a test so free-tier 429 responses become ``pytest.skip``."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — broad on purpose
            if _looks_like_quota_error(exc):
                pytest.skip(f"Gemini free-tier quota exhausted: {exc!s}")
            raise

    return wrapper


@pytest.fixture(scope="session")
def skip_on_quota_fixture():
    """Fixture form of the decorator for tests that prefer fixture injection."""
    return skip_on_quota
