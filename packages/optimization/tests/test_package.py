"""Smoke tests for ldai_optimizer."""

import pytest

from ldai_optimizer import OptimizationClient, __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_client_requires_ldai_client():
    with pytest.raises(TypeError):
        OptimizationClient()  # type: ignore[call-arg]
