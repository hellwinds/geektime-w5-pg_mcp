"""Pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import os
import tempfile

import pytest

from pg_mcp.config.settings import reset_settings


@pytest.fixture(autouse=True)
def isolate_dotenv() -> None:
    """Prevent settings classes from reading the developer's .env file.

    Nested config classes resolve ".env" relative to the working directory.
    Tests assert against defaults, so tests run from a temp directory where
    no .env exists.
    """
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="pg_mcp_tests_")
    os.chdir(tmp)
    yield  # type: ignore[misc]
    os.chdir(old_cwd)


@pytest.fixture(autouse=True)
def reset_config() -> None:
    """Reset global settings before each test."""
    reset_settings()


@pytest.fixture(autouse=True)
def disable_metrics_for_tests():
    """Disable metrics for tests to avoid port conflicts."""
    os.environ["OBSERVABILITY_METRICS_ENABLED"] = "false"
    yield
    # Clean up
    if "OBSERVABILITY_METRICS_ENABLED" in os.environ:
        del os.environ["OBSERVABILITY_METRICS_ENABLED"]
