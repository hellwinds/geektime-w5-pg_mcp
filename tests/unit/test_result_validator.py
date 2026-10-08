"""Unit tests for the LLM-based result validator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.services.result_validator import ResultValidator


def _make_completion(content: str | None) -> MagicMock:
    """Build a fake ChatCompletion with a single choice."""
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def _make_validator(
    validation_config: ValidationConfig | None = None,
) -> tuple[ResultValidator, MagicMock]:
    """Create a ResultValidator with a mocked OpenAI client."""
    config = OpenAIConfig(api_key="sk-test-key")
    vc = validation_config or ValidationConfig()
    validator = ResultValidator(openai_config=config, validation_config=vc)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    validator.client = client  # type: ignore[assignment]
    return validator, client


class TestResultValidator:
    """Tests for ResultValidator behavior."""

    async def test_disabled_returns_high_confidence(self) -> None:
        """When disabled, validation short-circuits with confidence 100."""
        validator, client = _make_validator(ValidationConfig(enabled=False))
        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"a": 1}], row_count=1
        )
        assert result.confidence == 100
        assert result.is_acceptable
        client.chat.completions.create.assert_not_awaited()

    async def test_successful_validation(self) -> None:
        """Valid JSON response is parsed into the result."""
        validator, client = _make_validator(ValidationConfig(confidence_threshold=70))
        payload = {
            "confidence": 85,
            "explanation": "matches",
            "suggestion": None,
        }
        client.chat.completions.create.return_value = _make_completion(json.dumps(payload))

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"a": 1}], row_count=1
        )
        assert result.confidence == 85
        assert result.is_acceptable is True

    async def test_below_threshold_not_acceptable(self) -> None:
        """Confidence below threshold marks result as not acceptable."""
        validator, client = _make_validator(ValidationConfig(confidence_threshold=90))
        payload = {"confidence": 60, "explanation": "weak match"}
        client.chat.completions.create.return_value = _make_completion(json.dumps(payload))

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
        assert result.confidence == 60
        assert result.is_acceptable is False

    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Non-JSON response yields confidence 60 and not acceptable."""
        validator, client = _make_validator()
        client.chat.completions.create.return_value = _make_completion("not json")

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
        assert result.confidence == 60
        assert result.is_acceptable is False

    async def test_out_of_range_confidence_clamped(self) -> None:
        """Float confidence is clamped to 0-100 bounds."""
        validator, client = _make_validator()
        payload = {"confidence": 250, "explanation": "x"}
        client.chat.completions.create.return_value = _make_completion(json.dumps(payload))
        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
        assert result.confidence == 100

    async def test_non_numeric_confidence_defaults(self) -> None:
        """Non-numeric confidence falls back to 50."""
        validator, client = _make_validator()
        payload = {"confidence": "high", "explanation": "x"}
        client.chat.completions.create.return_value = _make_completion(json.dumps(payload))
        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)
        assert result.confidence == 50

    async def test_empty_response_raises_llm_error(self) -> None:
        """Empty choices raise LLMError."""
        validator, client = _make_validator()
        completion = MagicMock()
        completion.choices = []
        client.chat.completions.create.return_value = completion

        with pytest.raises(LLMError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_empty_content_raises_llm_error(self) -> None:
        """Empty message content raises LLMError."""
        validator, client = _make_validator()
        client.chat.completions.create.return_value = _make_completion(None)

        with pytest.raises(LLMError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_timeout_raises_llm_timeout(self) -> None:
        """TimeoutError from the client maps to LLMTimeoutError."""
        validator, client = _make_validator(ValidationConfig(timeout_seconds=5.0))
        client.chat.completions.create.side_effect = TimeoutError

        with pytest.raises(LLMTimeoutError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_auth_error_raises_llm_unavailable(self) -> None:
        """Authentication errors map to LLMUnavailableError."""
        validator, client = _make_validator()
        client.chat.completions.create.side_effect = Exception("Error: incorrect api_key provided")

        with pytest.raises(LLMUnavailableError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_rate_limit_error_raises_llm_unavailable(self) -> None:
        """Rate limit errors map to LLMUnavailableError."""
        validator, client = _make_validator()
        client.chat.completions.create.side_effect = Exception("rate_limit exceeded")

        with pytest.raises(LLMUnavailableError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_other_errors_raise_llm_error(self) -> None:
        """Generic exceptions map to LLMError."""
        validator, client = _make_validator()
        client.chat.completions.create.side_effect = Exception("network down")

        with pytest.raises(LLMError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    async def test_results_sampled_to_configured_rows(self) -> None:
        """Only sample_rows results are sent to the LLM."""
        validator, client = _make_validator(ValidationConfig(sample_rows=2))
        captured_prompt: list[str] = []

        async def capture(**kwargs: object) -> MagicMock:
            captured_prompt.append(str(kwargs.get("messages")))
            return _make_completion(json.dumps({"confidence": 80, "explanation": "x"}))

        client.chat.completions.create.side_effect = capture

        rows = [{"id": i} for i in range(10)]
        await validator.validate(question="q", sql="SELECT 1", results=rows, row_count=10)

        assert len(captured_prompt) == 1
        # id 2..9 should not appear in the prompt (only first 2 sampled)
        assert '"id": 2' not in captured_prompt[0]
