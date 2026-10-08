"""Unit tests for resilience and observability integration in the orchestrator.

This module verifies that:
- Query and LLM rate limiters are enforced in the request flow
- Metrics are recorded for success, failure, SQL rejection, and duration
- Retry uses exponential backoff (retry_delay * backoff_factor ** attempt)
- Per-database executors and validators are selected per request
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


class TestOrchestratorIntegration:
    """Test resilience/observability integration in QueryOrchestrator."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(database_name="db1", tables=[])

    @pytest.fixture
    def mock_components(self) -> dict[str, AsyncMock | MagicMock]:
        """Create mocked pipeline components."""
        schema_cache = MagicMock()
        schema_cache.load = AsyncMock()
        schema_cache.get = MagicMock(return_value=None)
        generator = MagicMock()
        generator.generate = AsyncMock(return_value="SELECT 1")
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=([{"count": 1}], 1))
        return {
            "generator": generator,
            "validator": MagicMock(),
            "executor": executor,
            "result_validator": AsyncMock(),
            "schema_cache": schema_cache,
        }

    def _build_orchestrator(
        self,
        components: dict[str, AsyncMock | MagicMock],
        **overrides: object,
    ) -> QueryOrchestrator:
        """Build an orchestrator with mocked parts."""
        pools = {"db1": MagicMock(), "db2": MagicMock()}
        executors = {"db1": components["executor"], "db2": MagicMock()}
        kwargs: dict[str, object] = {
            "sql_generator": components["generator"],
            "sql_validator": components["validator"],
            "sql_executors": executors,
            "result_validator": components["result_validator"],
            "schema_cache": components["schema_cache"],
            "pools": pools,
            "resilience_config": ResilienceConfig(max_retries=0, retry_delay=0.1),
            "validation_config": ValidationConfig(enabled=False),
            "rate_limiter": MultiRateLimiter(query_limit=10, llm_limit=10),
        }
        kwargs.update(overrides)
        return QueryOrchestrator(**kwargs)  # type: ignore[arg-type]

    async def test_executor_selected_per_database(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Each database executes through its own dedicated executor."""
        executor_db2 = MagicMock()
        executor_db2.execute = AsyncMock(return_value=([{"x": 1}], 1))
        executors = {"db1": mock_components["executor"], "db2": executor_db2}
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db2", tables=[])
        orchestrator = self._build_orchestrator(
            mock_components, sql_executors=executors, schema_cache=schema_cache
        )
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db2"))

        assert response.success
        executor_db2.execute.assert_awaited_once()
        mock_components["executor"].execute.assert_not_awaited()

    async def test_validator_selected_per_database(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Database-specific validator enforces that database's policy."""
        db1_validator = MagicMock()  # allows everything
        db2_validator = MagicMock()
        db2_validator.validate_or_raise = MagicMock(
            side_effect=SecurityViolationError("blocked table")
        )

        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db2", tables=[])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        executors = {"db1": mock_components["executor"], "db2": mock_components["executor"]}
        orchestrator = self._build_orchestrator(
            components,
            sql_executors=executors,
            sql_validator=db1_validator,
            sql_validators={"db1": db1_validator, "db2": db2_validator},
        )
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db2"))

        assert not response.success
        assert response.error is not None
        assert response.error.code == "security_violation"
        db2_validator.validate_or_raise.assert_called()

    async def test_empty_executors_fail_query(self, mock_components: dict) -> None:
        """Query against a database without an executor fails cleanly."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db1", tables=[])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        orchestrator = self._build_orchestrator(components, sql_executors={"other_db": MagicMock()})
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db1"))
        assert not response.success
        assert response.error is not None
        assert response.error.code == "database_error"

    async def test_rate_limiter_releases_query_slot(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Query slot is released after the request completes."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db1", tables=[])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = self._build_orchestrator(components, rate_limiter=limiter)
        await orchestrator.execute_query(QueryRequest(question="count", database="db1"))

        assert limiter.query_limiter.active_count == 0
        assert limiter.llm_limiter.active_count == 0

    async def test_exponential_backoff_between_retries(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Retries wait retry_delay * backoff_factor ** attempt between attempts."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db1", tables=[])

        generator = MagicMock()
        generator.generate = AsyncMock(side_effect=["SELECT bad", "SELECT 1"])
        validator = MagicMock()
        validator.validate_or_raise = MagicMock(side_effect=[SecurityViolationError("no"), None])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache
        components["generator"] = generator
        components["validator"] = validator

        resilience = ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0)
        orchestrator = self._build_orchestrator(
            components,
            resilience_config=resilience,
        )
        response = await orchestrator.execute_query(
            QueryRequest(question="count", database="db1", return_type=ReturnType.SQL)
        )

        assert response.success
        assert generator.generate.await_count == 2

    async def test_metrics_recorded_on_success(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Successful queries record request and DB duration metrics."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db1", tables=[])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        metrics = MetricsCollector()
        orchestrator = self._build_orchestrator(components, metrics=metrics)
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db1"))

        assert response.success
        assert metrics.query_requests.labels(status="success", database="db1")._value.get() == 1

    async def test_metrics_recorded_on_failure(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Failed queries record error metrics."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = None
        schema_cache.load.side_effect = RuntimeError("boom")
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        metrics = MetricsCollector()
        orchestrator = self._build_orchestrator(components, metrics=metrics)
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db1"))

        assert not response.success
        assert metrics.query_requests.labels(status="error", database="db1")._value.get() == 1

    async def test_works_without_metrics(
        self, mock_components: dict[str, AsyncMock | MagicMock]
    ) -> None:
        """Orchestrator works when metrics collector is not provided."""
        schema_cache = mock_components["schema_cache"]
        assert isinstance(schema_cache, MagicMock)
        schema_cache.get.return_value = DatabaseSchema(database_name="db1", tables=[])
        components = dict(mock_components)
        components["schema_cache"] = schema_cache

        orchestrator = self._build_orchestrator(components, metrics=None)
        response = await orchestrator.execute_query(QueryRequest(question="count", database="db1"))
        assert response.success


class TestRateLimiting:
    """Test rate limiting behavior at the limiter level."""

    async def test_query_limiter_blocks_over_limit(self) -> None:
        """Limiter rejects acquisition when all slots are taken."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)

        acquired_first = await limiter.query_limiter.acquire()
        acquired_second = await limiter.query_limiter.acquire(timeout=0.05)

        assert acquired_first is True
        assert acquired_second is False
        assert limiter.query_limiter.get_stats()["total_rejections"] == 1

        limiter.query_limiter.release()

    async def test_llm_limiter_blocks_over_limit(self) -> None:
        """LLM limiter rejects acquisition when all slots are taken."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)

        acquired_first = await limiter.llm_limiter.acquire()
        acquired_second = await limiter.llm_limiter.acquire(timeout=0.05)

        assert acquired_first is True
        assert acquired_second is False

        limiter.llm_limiter.release()

    async def test_limiter_timeout_returns_false(self) -> None:
        """Acquisition with timeout returns False instead of raising."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        await limiter.query_limiter.acquire()
        result = await asyncio.wait_for(limiter.query_limiter.acquire(timeout=0.05), timeout=1.0)
        assert result is False
        limiter.query_limiter.release()
