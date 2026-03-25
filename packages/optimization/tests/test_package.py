"""Smoke tests for ldai_optimization."""

import pytest

from ldai_optimization import ApiAgentOptimizationClient, __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_optimize_not_implemented():
    client = ApiAgentOptimizationClient()
    with pytest.raises(NotImplementedError):
        client.optimize("example", {})
