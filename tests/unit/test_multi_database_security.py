"""Unit tests for multi-database security configuration.

This module tests:
- SecurityConfig blocked_tables / blocked_columns / allow_explain fields
- DatabaseProfile per-database security overrides
- Settings.all_databases aggregation (primary + extra databases)
- Extra databases parsing from JSON
"""

import json

import pytest
from pydantic import ValidationError

from pg_mcp.config.settings import DatabaseProfile, SecurityConfig, Settings


class TestSecurityConfigNewFields:
    """Tests for new SecurityConfig security control fields."""

    def test_defaults(self) -> None:
        """By default no tables/columns blocked and EXPLAIN disallowed."""
        config = SecurityConfig()
        assert config.blocked_tables == []
        assert config.blocked_columns == []
        assert config.allow_explain is False

    def test_blocked_tables_from_comma_string(self) -> None:
        """Blocked tables can be configured via comma-separated string."""
        config = SecurityConfig(blocked_tables="users, api_keys,sessions")
        assert config.blocked_tables == ["users", "api_keys", "sessions"]

    def test_blocked_columns_from_comma_string(self) -> None:
        """Blocked columns can be configured via comma-separated string."""
        config = SecurityConfig(blocked_columns="password, users.ssn")
        assert config.blocked_columns == ["password", "users.ssn"]

    def test_allow_explain_enabled(self) -> None:
        """EXPLAIN can be explicitly permitted."""
        config = SecurityConfig(allow_explain=True)
        assert config.allow_explain is True

    def test_blocked_functions_still_parsed(self) -> None:
        """Original blocked_functions parsing still works."""
        config = SecurityConfig(blocked_functions="pg_sleep, lo_import")
        assert config.blocked_functions == ["pg_sleep", "lo_import"]


class TestDatabaseProfile:
    """Tests for DatabaseProfile per-database security overrides."""

    def test_defaults_fall_back_to_global(self) -> None:
        """None overrides mean global policy should be used."""
        profile = DatabaseProfile(name="db1")
        assert profile.blocked_tables is None
        assert profile.blocked_columns is None
        assert profile.allow_explain is None

    def test_per_database_overrides(self) -> None:
        """Per-database overrides can be set."""
        profile = DatabaseProfile(
            name="db1",
            blocked_tables=["secrets"],
            blocked_columns="users.ssn, api_keys.key",
            allow_explain=True,
        )
        assert profile.blocked_tables == ["secrets"]
        assert profile.blocked_columns == ["users.ssn", "api_keys.key"]
        assert profile.allow_explain is True

    def test_dsn_property(self) -> None:
        """DSN is built from connection fields."""
        profile = DatabaseProfile(
            host="localhost", port=5433, name="mydb", user="alice", password="secret"
        )
        assert profile.dsn == "postgresql://alice:secret@localhost:5433/mydb"
        assert profile.safe_dsn == "postgresql://alice:***@localhost:5433/mydb"


class TestSettingsMultiDatabase:
    """Tests for multi-database Settings aggregation."""

    @staticmethod
    def _make_settings(**kwargs: object) -> Settings:
        """Build Settings with a valid API key for testing."""
        return Settings(
            openai={"api_key": "sk-test-key-123"},
            _env_file=None,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_all_databases_single(self) -> None:
        """With no extra databases, only the primary is returned."""
        settings = self._make_settings(database={"name": "primary_db"})
        databases = settings.all_databases
        assert list(databases.keys()) == ["primary_db"]

    def test_all_databases_with_extra(self) -> None:
        """Extra databases are merged with the primary."""
        settings = self._make_settings(
            database={"name": "primary_db"},
            extra_databases={
                "analytics": DatabaseProfile(host="remote", name="analytics_db", user="reader")
            },
        )
        databases = settings.all_databases
        assert set(databases.keys()) == {"primary_db", "analytics"}
        assert databases["analytics"].host == "remote"

    def test_primary_takes_precedence(self) -> None:
        """Primary database wins on name conflict with extras."""
        settings = self._make_settings(
            database={"name": "db1", "host": "primary-host"},
            extra_databases={"db1": DatabaseProfile(name="db1", host="extra-host")},
        )
        assert settings.all_databases["db1"].host == "primary-host"

    def test_extra_databases_from_json_string(self) -> None:
        """Extra databases can be loaded from JSON env var."""
        payload = {
            "analytics": {
                "host": "remote",
                "name": "analytics_db",
                "user": "reader",
                "blocked_tables": "audit_logs",
            }
        }
        settings = self._make_settings(extra_databases=json.dumps(payload))
        analytics = settings.all_databases["analytics"]
        assert analytics.host == "remote"
        assert analytics.blocked_tables == ["audit_logs"]

    def test_extra_databases_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValidationError."""
        with pytest.raises(ValidationError):
            self._make_settings(extra_databases="{invalid json")

    def test_extra_databases_non_object_json_raises(self) -> None:
        """JSON that is not an object raises ValidationError."""
        with pytest.raises(ValidationError):
            self._make_settings(extra_databases="[1, 2, 3]")
