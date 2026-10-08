"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
metrics collection, circuit breaker pattern for fault tolerance, and per-database
executor selection for multi-database security isolation.
"""

import asyncio
import logging
import uuid
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with exponential backoff, per-database
    executor selection for security isolation, rate limiting for queries and LLM
    calls, Prometheus metrics collection, circuit breaker pattern for fault
    tolerance, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"db1": executor1, "db2": executor2},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"db1": pool1, "db2": pool2},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="db1"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
        sql_validators: dict[str, SQLValidator] | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: Default SQL validation service (fallback when no
                database-specific validator is registered).
            sql_executors: Dictionary mapping database names to their executors.
                Each executor may enforce database-specific security policies.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries,
                exponential backoff, and circuit breaker.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional rate limiter for queries and LLM calls.
                When None, a MultiRateLimiter is built from resilience_config.
            metrics: Optional Prometheus metrics collector.
            sql_validators: Optional per-database validators keyed by database
                name. Used to enforce database-specific security policies
                (blocked tables/columns, EXPLAIN strategy).
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.sql_validators = sql_validators or {}
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter or MultiRateLimiter(
            query_limit=resilience_config.max_concurrent_queries,
            llm_limit=resilience_config.max_concurrent_llm_calls,
        )
        self.metrics = metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    def _get_validator_for_database(self, database_name: str) -> SQLValidator:
        """Get the validator enforcing the security policy of a database.

        Args:
            database_name: Target database name.

        Returns:
            SQLValidator: Database-specific validator if registered,
                otherwise the default validator.
        """
        return self.sql_validators.get(database_name, self.sql_validator)

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracking
        2. Resolve and validate database name
        3. Enforce query rate limits
        4. Load schema from cache
        5. Generate and validate SQL with retry logic (rate-limited LLM calls)
        6. Execute SQL via the target database's dedicated executor (if RESULT)
        7. Validate results (optional)
        8. Record metrics and return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        # Generate request_id for full-chain tracing
        request_id = str(uuid.uuid4())
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        start_time = self._get_current_time_ms()
        database_name: str | None = None

        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Enforce query rate limit
            acquired = await self.rate_limiter.query_limiter.acquire()
            if not acquired:
                raise RateLimitExceededError(
                    message="Query rate limit exceeded, please retry later",
                    details={"database": database_name},
                )
            try:
                return await self._process_query(
                    request=request,
                    database_name=database_name,
                    request_id=request_id,
                )
            finally:
                self.rate_limiter.query_limiter.release()

        except PgMcpError as e:
            # Handle known application errors
            self._record_failure_metrics(e, database_name)
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            self._record_failure_metrics(None, database_name)
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )
        finally:
            if self.metrics is not None:
                duration_seconds = (self._get_current_time_ms() - start_time) / 1000
                self.metrics.query_duration.observe(duration_seconds)

    async def _process_query(
        self,
        request: QueryRequest,
        database_name: str,
        request_id: str,
    ) -> QueryResponse:
        """Process the query after rate limiting and database resolution.

        Args:
            request: Query request containing question and parameters.
            database_name: Resolved target database name.
            request_id: Request ID for tracking.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        # Resolve the security policy for this database once
        validator = self._get_validator_for_database(database_name)

        # Step 1: Get schema from cache
        schema = self.schema_cache.get(database_name)
        if schema is None:
            # Schema not in cache, load it
            pool = self.pools.get(database_name)
            if pool is None:
                raise DatabaseError(
                    message=f"No connection pool available for database '{database_name}'",
                    details={"database": database_name},
                )
            try:
                schema = await self.schema_cache.load(database_name, pool)
            except Exception as e:
                raise SchemaLoadError(
                    message=f"Failed to load schema for database '{database_name}': {e!s}",
                    details={"database": database_name, "error": str(e)},
                ) from e

        logger.debug(
            "Schema loaded",
            extra={
                "request_id": request_id,
                "database": database_name,
                "tables": len(schema.tables),
            },
        )

        # Step 2: Generate and validate SQL with retry logic
        generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
            question=request.question,
            schema=schema,
            request_id=request_id,
            validator=validator,
        )

        # Step 3: If return_type is SQL, return early
        if request.return_type == ReturnType.SQL:
            logger.info(
                "Returning SQL only",
                extra={"request_id": request_id, "sql_length": len(generated_sql)},
            )
            if self.metrics is not None:
                self.metrics.increment_query_request(status="sql_only", database=database_name)
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=None,
                error=None,
                confidence=100,
                tokens_used=tokens_used,
            )

        # Step 4: Execute SQL using this database's dedicated executor.
        # This enforces per-database security policies (blocked tables/columns,
        # EXPLAIN strategy) and prevents cross-database access.
        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No executor configured for database '{database_name}'",
                details={"database": database_name},
            )

        logger.debug("Executing SQL", extra={"request_id": request_id})
        execution_start = self._get_current_time_ms()

        results, total_count = await executor.execute(generated_sql)

        execution_time_ms = self._get_current_time_ms() - execution_start
        logger.info(
            "SQL executed successfully",
            extra={
                "request_id": request_id,
                "database": database_name,
                "row_count": total_count,
                "execution_time_ms": execution_time_ms,
            },
        )
        if self.metrics is not None:
            self.metrics.observe_db_query_duration(execution_time_ms / 1000)
            self.metrics.increment_query_request(status="success", database=database_name)

        # Step 5: Validate results (non-blocking, failures don't fail the request)
        result_confidence = await self._validate_results_safely(
            question=request.question,
            sql=generated_sql,
            results=results,
            row_count=total_count,
            request_id=request_id,
        )

        # Step 6: Build successful response
        query_result = QueryResult(
            columns=list(results[0].keys()) if results else [],
            rows=results,
            row_count=len(results),  # Limited row count (after max_rows applied)
            execution_time_ms=execution_time_ms,
        )

        return QueryResponse(
            success=True,
            generated_sql=generated_sql,
            validation=validation_result,
            data=query_result,
            error=None,
            confidence=result_confidence,
            tokens_used=tokens_used,
        )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
        validator: SQLValidator | None = None,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (subject to LLM rate limiting)
        3. Validates the generated SQL against the database security policy
        4. On validation failure, retries with error feedback and exponential backoff
        5. Records success/failure to circuit breaker

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.
            validator: Validator enforcing the target database's security policy.
                Defaults to the orchestrator's default validator.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        effective_validator = validator or self.sql_validator
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        retry_delay = self.resilience_config.retry_delay
        backoff_factor = self.resilience_config.backoff_factor
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Enforce LLM rate limit (non-blocking acquisition)
                llm_acquired = await self.rate_limiter.llm_limiter.acquire()
                if not llm_acquired:
                    raise LLMError(
                        message="LLM rate limit exceeded, please retry later",
                        details={"active": self.rate_limiter.llm_limiter.active_count},
                    )
                try:
                    generated_sql = await self.sql_generator.generate(
                        question=question,
                        schema=schema,
                        previous_attempt=previous_sql,
                        error_feedback=error_feedback,
                    )
                finally:
                    self.rate_limiter.llm_limiter.release()

                if self.metrics is not None:
                    self.metrics.increment_llm_call("generate_sql")

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL against the database's security policy
                try:
                    effective_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if attempt < max_retries:
                        # Record as failure and retry with feedback (exponential backoff)
                        delay = retry_delay * (backoff_factor**attempt)
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "retry_delay_seconds": delay,
                                "error": str(validation_error),
                            },
                        )
                        if self.metrics is not None:
                            self.metrics.increment_sql_rejected("validation_failed")
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        if self.metrics is not None:
                            self.metrics.increment_sql_rejected("validation_failed")
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        import time

        return time.time() * 1000

    def _record_failure_metrics(self, error: PgMcpError | None, database: str | None) -> None:
        """Record failure metrics for a failed query.

        Args:
            error: The application error that caused the failure, or None for
                unexpected errors.
            database: Target database name if it was resolved, None otherwise.
        """
        if self.metrics is None:
            return

        reason = error.code.value if error is not None else "internal_error"
        self.metrics.increment_sql_rejected(reason)
        self.metrics.increment_query_request(status="error", database=database or "unknown")
