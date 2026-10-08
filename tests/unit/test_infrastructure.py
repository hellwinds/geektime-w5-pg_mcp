"""Unit tests for connection pool management, logging, and tracing utilities.

These modules were previously untested; these tests cover the main code
paths without requiring a live PostgreSQL server or HTTP endpoint.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from pg_mcp.config.settings import DatabaseProfile
from pg_mcp.db.pool import close_pools
from pg_mcp.observability.logging import configure_logging, get_logger
from pg_mcp.observability.tracing import (
    clear_request_id,
    generate_request_id,
    get_request_id,
    request_context,
    trace_sync,
)


class TestClosePools:
    """Tests for graceful pool shutdown."""

    async def test_close_pools_graceful(self) -> None:
        """Pools close gracefully when close() completes in time."""
        pool = MagicMock()
        pool.close = AsyncMock(return_value=None)

        await close_pools({"db1": pool}, timeout=1.0)

        pool.close.assert_awaited_once()
        pool.terminate.assert_not_called()

    async def test_close_pools_timeout_terminates(self) -> None:
        """Slow pools are force-terminated after timeout."""
        pool = MagicMock()

        async def slow_close() -> None:
            await asyncio.sleep(5)

        pool.close = slow_close

        await close_pools({"db1": pool}, timeout=0.05)

        pool.terminate.assert_called_once()

    async def test_close_pools_error_terminates(self) -> None:
        """Pools raising errors during close are force-terminated."""
        pool = MagicMock()
        pool.close = AsyncMock(side_effect=RuntimeError("boom"))

        await close_pools({"db1": pool}, timeout=1.0)

        pool.terminate.assert_called_once()

    async def test_close_pools_multiple(self) -> None:
        """All pools are closed even if some fail."""
        good = MagicMock()
        good.close = AsyncMock(return_value=None)
        bad = MagicMock()
        bad.close = AsyncMock(side_effect=RuntimeError("x"))

        await close_pools({"good": good, "bad": bad}, timeout=1.0)

        good.terminate.assert_not_called()
        bad.terminate.assert_called_once()


class TestLoggingConfiguration:
    """Tests for observability logging setup."""

    def test_configure_logging_text(self) -> None:
        """Text logging configures without errors."""
        configure_logging(level="DEBUG", log_format="text", enable_sensitive_filter=False)
        logger = get_logger("test_text")
        logger.info("hello text")

    def test_configure_logging_json(self) -> None:
        """JSON logging configures without errors."""
        configure_logging(level="INFO", log_format="json", enable_sensitive_filter=True)
        logger = get_logger("test_json")
        logger.info("hello json")

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test_return")
        assert isinstance(logger, logging.Logger)


class TestTracing:
    """Tests for request tracing context."""

    def test_generate_request_id_unique(self) -> None:
        """Request IDs are unique."""
        id1 = generate_request_id()
        id2 = generate_request_id()
        assert id1 != id2
        assert len(id1) == 36

    async def test_request_context_sets_and_clears(self) -> None:
        """request_context sets ID inside and restores outside."""
        assert get_request_id() is None
        async with request_context() as req_id:
            assert req_id == get_request_id()
            assert get_request_id() is not None
        assert get_request_id() is None

    async def test_request_context_custom_id(self) -> None:
        """Custom request IDs are honored."""
        async with request_context("custom-id") as req_id:
            assert req_id == "custom-id"
            assert get_request_id() == "custom-id"

    def test_clear_request_id(self) -> None:
        """clear_request_id resets the context."""
        from pg_mcp.observability.tracing import set_request_id

        set_request_id("temp")
        clear_request_id()
        assert get_request_id() is None

    def test_trace_sync_decorator(self) -> None:
        """trace_sync passes through return values inside a context."""

        @trace_sync(operation="op")
        def compute(x: int) -> int:
            return x * 2

        async def run() -> int:
            async with request_context():
                return compute(21)

        assert asyncio.run(run()) == 42


class TestDatabaseProfileDefaults:
    """Quick sanity tests for DatabaseProfile defaults."""

    def test_defaults(self) -> None:
        """Profile defaults match DatabaseConfig legacy defaults."""
        profile = DatabaseProfile()
        assert profile.host == "localhost"
        assert profile.port == 5432
        assert profile.min_pool_size == 5
        assert profile.max_pool_size == 20
        assert profile.blocked_tables is None
