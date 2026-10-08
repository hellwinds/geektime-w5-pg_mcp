"""Unit tests for schema introspection using mocked database connections.

These tests verify the SQL introspection logic without requiring a live
PostgreSQL database by mocking asyncpg connection responses.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.db.introspection import SchemaIntrospector


def _make_conn() -> MagicMock:
    """Create a mock asyncpg connection with canned catalog responses."""
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value="PostgreSQL 15.0, compiler")
    conn.fetch = AsyncMock(return_value=[])

    def fetch_side_effect(query: str, *args: object) -> object:
        q = " ".join(query.split())
        if "relkind = 'r'" in q:
            return [
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "comment": "User accounts",
                }
            ]
        if "relkind = 'v'" in q:
            return []
        if "typtype = 'e'" in q:
            return []
        if "indexrelid" in q:
            return []  # non-primary indexes
        if "indisprimary" in q:
            return [{"column_name": "id"}]
        if "contype = 'f'" in q:
            return []
        if "attnum > 0" in q:
            return [
                {
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "default_value": None,
                    "comment": None,
                }
            ]
        return []

    conn.fetch.side_effect = fetch_side_effect

    def fetchval_side_effect(query: str, *args: object) -> object:
        q = " ".join(query.split())
        if "version()" in q:
            return "PostgreSQL 15.0, compiler"
        if "contype = 'u'" in q:
            return True
        if "reltuples" in q:
            return 42
        return None

    conn.fetchval.side_effect = fetchval_side_effect
    return conn


def _make_pool(conn: MagicMock) -> MagicMock:
    """Create a mock pool that yields the given connection."""
    pool = MagicMock()

    class _AcquireCtx:
        async def __aenter__(self) -> MagicMock:
            return conn

        async def __aexit__(self, *args: object) -> None:
            return None

    pool.acquire.return_value = _AcquireCtx()
    return pool


class TestSchemaIntrospector:
    """Tests for SchemaIntrospector with mocked connections."""

    @pytest.fixture
    def introspector(self) -> SchemaIntrospector:
        """Create introspector with mocked pool."""
        return SchemaIntrospector(_make_pool(_make_conn()), "test_db")

    async def test_introspect_returns_schema(self, introspector: SchemaIntrospector) -> None:
        """Full introspection returns a populated DatabaseSchema."""
        schema = await introspector.introspect()

        assert schema.database_name == "test_db"
        assert schema.version == "PostgreSQL 15.0"
        assert len(schema.tables) == 1

        users = schema.tables[0]
        assert users.table_name == "users"
        assert users.comment == "User accounts"
        assert len(users.columns) == 1
        assert users.columns[0].name == "id"
        # Primary key marked
        assert users.columns[0].is_primary_key is True

    async def test_introspect_empty_database(self) -> None:
        """Introspection of an empty database returns empty schema."""
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value="PostgreSQL 15.0, x")
        conn.fetch = AsyncMock(return_value=[])

        introspector = SchemaIntrospector(_make_pool(conn), "empty_db")
        schema = await introspector.introspect()

        assert schema.database_name == "empty_db"
        assert schema.tables == []
        assert schema.enum_types == []

    async def test_introspect_with_views_and_enums(self) -> None:
        """Views and enum types are collected alongside tables."""
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value="PostgreSQL 16.2 (Debian)")

        tables = [{"schema_name": "public", "table_name": "orders", "comment": None}]
        views = [{"schema_name": "public", "table_name": "order_summary", "comment": "v"}]
        enums = [
            {
                "schema_name": "public",
                "type_name": "order_status",
                "values": ["pending", "shipped"],
            }
        ]

        async def fetch(query: str, *args: object) -> list[dict[str, object]]:
            q = " ".join(query.split())
            if "relkind = 'r'" in q:
                return tables
            if "relkind = 'v'" in q:
                return views
            if "typtype = 'e'" in q:
                return enums
            if "indexrelid" in q:
                return []
            if "indisprimary" in q:
                return [{"column_name": "id"}]
            if "contype = 'f'" in q:
                return []
            if "attnum > 0" in q:
                return [
                    {
                        "column_name": "id",
                        "data_type": "integer",
                        "is_nullable": False,
                        "default_value": None,
                        "comment": None,
                    }
                ]
            return []

        conn.fetch = AsyncMock(side_effect=fetch)

        async def fetchval(query: str, *args: object) -> object:
            q = " ".join(query.split())
            if "version()" in q:
                return "PostgreSQL 16.2 (Debian)"
            if "reltuples" in q:
                return 10
            if "contype = 'u'" in q:
                return False
            return None

        conn.fetchval = AsyncMock(side_effect=fetchval)

        introspector = SchemaIntrospector(_make_pool(conn), "rich_db")
        schema = await introspector.introspect()

        assert schema.version == "PostgreSQL 16.2 (Debian)"
        assert len(schema.tables) == 2  # 1 table + 1 view
        assert len(schema.enum_types) == 1
        assert schema.enum_types[0].values == ["pending", "shipped"]

    async def test_row_count_estimate_none_defaults_to_zero(self) -> None:
        """Missing statistics produce a zero row count estimate."""
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value="PostgreSQL 15.0")
        conn.fetch = AsyncMock(return_value=[])

        async def fetchval(query: str, *args: object) -> object:
            if "reltuples" in query:
                return None
            return None

        # fetch returns one table, reltuples is None
        async def fetch(query: str, *args: object) -> list[dict[str, object]]:
            q = " ".join(query.split())
            if "relkind = 'r'" in q:
                return [{"schema_name": "public", "table_name": "t1", "comment": None}]
            if "attnum > 0" in q:
                return []
            return []

        conn.fetch = AsyncMock(side_effect=fetch)
        conn.fetchval = AsyncMock(side_effect=fetchval)

        introspector = SchemaIntrospector(_make_pool(conn), "db")
        schema = await introspector.introspect()

        assert schema.tables[0].row_count_estimate == 0

    async def test_version_none_when_unavailable(self) -> None:
        """Version is None when the server returns nothing."""
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=None)
        conn.fetch = AsyncMock(return_value=[])

        introspector = SchemaIntrospector(_make_pool(conn), "db")
        schema = await introspector.introspect()

        assert schema.version is None
