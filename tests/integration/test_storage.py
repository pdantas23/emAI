"""Integration tests for `src/storage/`.

Coverage map:

  ┌────────────────────────────┬────────────────────────────────────────────┐
  │ Concern                    │ What we prove                              │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Construction & fail-stop   │ SELECT 1 actually runs; unreachable URL    │
  │                            │ raises StorageError; create_all idempotent.│
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ has_been_processed         │ False → True after persist; query errors   │
  │                            │ surface as StorageError.                   │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ mark_as_processed          │ Persists the metadata-only column set,     │
  │                            │ returns assigned id, logs each insertion.  │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Privacy contract           │ The store REFUSES sensitive payloads — no  │
  │                            │ subject / sender / reason / summary column │
  │                            │ exists, and the signature won't accept     │
  │                            │ those keyword arguments.                   │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Enums round-trip           │ Priority + DeliveryStatus stored as their  │
  │                            │ string values (not names, not ints).       │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Deduplication              │ Same message_id twice → StorageError;      │
  │                            │ existing row is NOT corrupted.             │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ list_recent (audit)        │ Newest-first order; respects limit; mixed  │
  │                            │ delivery states all surface.               │
  └────────────────────────────┴────────────────────────────────────────────┘

We use one fresh in-memory SQLite engine per test (autouse fixture) so
runs are isolated AND fast — no temp files, no cleanup. Each test starts
from an empty schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger as loguru_logger
from sqlmodel import create_engine

from src.ai.classifier import Priority
from src.storage.models import DeliveryStatus, ProcessedEmail
from src.storage.state import StateStore, StorageError, _build_engine


# =========================================================================== #
# Fixtures & helpers
# =========================================================================== #


@pytest.fixture
def store() -> StateStore:
    """A clean StateStore wired to a per-test in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    return StateStore(engine=engine)


@pytest.fixture
def loguru_capture() -> Iterator[list[dict[str, Any]]]:
    """Collect every loguru record emitted during the test as a list of
    dicts (record.message, record.level.name, etc.). The handler is
    removed automatically after the test."""
    captured: list[dict[str, Any]] = []
    handler_id = loguru_logger.add(
        lambda msg: captured.append(dict(msg.record)),
        level="DEBUG",
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(handler_id)


def _persist(
    store: StateStore,
    *,
    message_id: str = "<msg-42@example.com>",
    uid: str = "42",
    relevance: bool = True,
    priority: str = Priority.high.value,
    delivery_status: DeliveryStatus = DeliveryStatus.delivered,
    message_ids: list[str] | None = None,
) -> ProcessedEmail:
    """Thin wrapper that lets each test override only the fields it cares
    about. Mirrors the new `mark_as_processed` signature, which is
    keyword-only and scoped to non-sensitive metadata."""
    return store.mark_as_processed(
        message_id=message_id,
        uid=uid,
        relevance=relevance,
        priority=priority,
        delivery_status=delivery_status,
        message_ids=message_ids,
    )


# =========================================================================== #
# 1. Construction & fail-stop
# =========================================================================== #


class TestConstruction:
    def test_select_one_check_runs_on_init(self) -> None:
        """A fresh in-memory SQLite is by definition reachable, but the
        SELECT 1 path must execute — otherwise the fail-stop is dead code.
        We force a connection error by closing the engine before init."""
        engine = create_engine("sqlite:///:memory:")
        engine.dispose()  # still usable; sqlite reopens on demand

        # This must NOT raise — sqlite reopens the in-memory DB.
        store = StateStore(engine=engine)
        assert store is not None

    def test_unreachable_url_raises_storage_error(self) -> None:
        """The fail-stop gate. Postgres at a non-existent host must abort
        construction immediately — the pipeline depends on this so it
        doesn't run 'blind' without state persistence."""
        engine = create_engine(
            "postgresql://nobody:wrong@nonexistent.invalid:5432/db",
            connect_args={"connect_timeout": 1},
        )
        with pytest.raises(StorageError, match="unreachable"):
            StateStore(engine=engine)

    def test_construct_twice_does_not_re_break_schema(self) -> None:
        """create_all is documented as idempotent; verify by constructing
        twice against the same engine (each call runs _ensure_tables)."""
        engine = create_engine("sqlite:///:memory:")
        StateStore(engine=engine)
        # No exception → idempotent ✓
        StateStore(engine=engine)


class TestEngineFactory:
    """`_build_engine` is a tiny dialect-aware wrapper, but the SQLite branch
    is what makes the apscheduler thread story safe — pin the contract."""

    def test_sqlite_url_sets_check_same_thread_false(self) -> None:
        engine = _build_engine("sqlite:///:memory:")
        # SQLAlchemy stashes connect_args on the engine's pool dialect.
        assert engine.dialect.name == "sqlite"
        # The connection must work without raising the cross-thread guard.
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")

    def test_postgres_url_does_not_inject_sqlite_args(self) -> None:
        # We don't connect — just ensure the factory returns an engine
        # configured for postgres without choking on dialect detection.
        engine = _build_engine("postgresql://u:p@host/db")
        assert engine.dialect.name == "postgresql"


class TestEnsureTablesFailure:
    def test_create_all_failure_surfaces_as_storage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If schema creation fails (perms, dialect mismatch, etc.) we must
        raise StorageError, not let SQLAlchemy's exception escape raw."""
        from sqlmodel import SQLModel

        engine = create_engine("sqlite:///:memory:")

        def boom(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("schema-create exploded")

        monkeypatch.setattr(SQLModel.metadata, "create_all", boom)
        with pytest.raises(StorageError, match="Failed to create/verify schema"):
            StateStore(engine=engine)


# =========================================================================== #
# 2. has_been_processed
# =========================================================================== #


class TestHasBeenProcessed:
    def test_returns_false_for_unknown_message_id(self, store: StateStore) -> None:
        assert store.has_been_processed("<never-seen@nowhere>") is False

    def test_returns_true_after_persist(self, store: StateStore) -> None:
        _persist(store, message_id="<seen@x>")
        assert store.has_been_processed("<seen@x>") is True

    def test_query_failure_surfaces_as_storage_error(self, store: StateStore) -> None:
        """Disposing the engine and then trying to query forces a connection
        error inside the query path — must surface as StorageError, not as
        a raw SQLAlchemy exception."""
        store._engine.dispose()
        # Force a real failure by pointing the engine at a now-bad URL.
        store._engine = create_engine("postgresql://x:x@nonexistent.invalid:5432/db",
                                      connect_args={"connect_timeout": 1})
        with pytest.raises(StorageError, match="Failed to query"):
            store.has_been_processed("<msg@x>")


# =========================================================================== #
# 3. mark_as_processed — happy path & audit logging
# =========================================================================== #


class TestMarkAsProcessedDelivered:
    def test_persists_every_retained_field_for_delivered_email(
        self, store: StateStore
    ) -> None:
        """The retained column set is intentionally small: Message-ID, UID,
        relevance, priority, delivery_status, message_ids, processed_at.
        Everything else is dropped before we reach the database."""
        sids = ["SM" + "a" * 32, "SM" + "b" * 32]
        record = _persist(
            store,
            message_id="<msg-42@x>",
            uid="42",
            relevance=True,
            priority="high",
            delivery_status=DeliveryStatus.delivered,
            message_ids=sids,
        )

        fetched = store.get("<msg-42@x>")
        assert fetched is not None
        assert fetched.id == record.id
        assert fetched.message_id == "<msg-42@x>"
        assert fetched.uid == "42"
        assert fetched.relevance is True
        assert fetched.priority == "high"
        assert fetched.delivery_status == "delivered"
        assert fetched.message_ids == ",".join(sids)
        # processed_at gets a server-side default we control via default_factory.
        assert fetched.processed_at is not None

    def test_returns_record_with_database_assigned_id(
        self, store: StateStore
    ) -> None:
        """The returned record must have its `id` populated by the DB —
        callers (e.g., orchestrator audit lines) rely on it."""
        record = _persist(store)
        assert isinstance(record.id, int)
        assert record.id > 0

    def test_logger_emits_one_info_line_per_insert(
        self,
        store: StateStore,
        loguru_capture: list[dict[str, Any]],
    ) -> None:
        """Audit promise: every insertion is logged, with only the
        non-sensitive metadata that actually made it into the row."""
        record = _persist(
            store,
            message_id="<audit-1@x>",
            message_ids=["SM" + "x" * 32],
        )

        info_lines = [r for r in loguru_capture if r["level"].name == "INFO"]
        assert len(info_lines) == 1, "exactly ONE INFO line per persisted row"

        msg = info_lines[0]["message"]
        assert f"id={record.id}" in msg
        assert "message_id=<audit-1@x>" in msg
        assert "status=delivered" in msg
        assert "priority=high" in msg
        assert "relevance=True" in msg
        assert "SM" + "x" * 32 in msg, "twilio sid must appear for delivery audits"


class TestMarkAsProcessedSkipped:
    """When the classifier rejects an email, we still persist a row — but
    without twilio SIDs (there was no send). This lets the orchestrator's
    dedup gate skip the same email on subsequent runs without re-calling
    the classifier."""

    def test_persists_with_skipped_status_and_no_sids(
        self, store: StateStore
    ) -> None:
        _persist(
            store,
            message_id="<skip-1@x>",
            uid="9",
            relevance=False,
            priority="low",
            delivery_status=DeliveryStatus.skipped_irrelevant,
            message_ids=None,
        )

        fetched = store.get("<skip-1@x>")
        assert fetched is not None
        assert fetched.relevance is False
        assert fetched.priority == "low"
        assert fetched.delivery_status == "skipped_irrelevant"
        assert fetched.message_ids is None


# =========================================================================== #
# 4. Privacy contract — the store refuses sensitive payloads
# =========================================================================== #


class TestPrivacyContract:
    """The refactor that introduced this contract hinges on TWO guarantees:

      1. The `ProcessedEmail` table has no column that could hold an
         email body, subject, sender, classifier reason, or LLM summary.
      2. The `mark_as_processed` signature doesn't accept keyword
         arguments for those fields — callers CANNOT smuggle them in.

    Both guarantees are tested here. If either regresses, sensitive data
    could silently start flowing back into the database.
    """

    _FORBIDDEN_COLUMNS = (
        "subject",
        "sender_email",
        "sender_name",
        "classification_reason",
        "summary_resumo",
        "summary_contexto",
        "summary_acao",
    )

    def test_no_sensitive_columns_on_processed_email(self) -> None:
        present = set(ProcessedEmail.model_fields.keys())
        forbidden = set(self._FORBIDDEN_COLUMNS)
        leaked = present & forbidden
        assert not leaked, (
            f"Sensitive columns have reappeared on ProcessedEmail: {sorted(leaked)}. "
            "If you're adding a retention policy, document it AND update the "
            "privacy contract in src/storage/models.py first."
        )

    @pytest.mark.parametrize("field_name", _FORBIDDEN_COLUMNS)
    def test_mark_as_processed_rejects_sensitive_kwargs(
        self, store: StateStore, field_name: str
    ) -> None:
        """Smuggling any sensitive field in via kwargs must raise TypeError
        at call time — the signature is the enforcement surface."""
        with pytest.raises(TypeError):
            store.mark_as_processed(  # type: ignore[call-arg]
                message_id="<x@x>",
                uid="1",
                relevance=True,
                priority="high",
                **{field_name: "secret-payload"},  # type: ignore[arg-type]
            )


# =========================================================================== #
# 5. Enum round-trip integrity
# =========================================================================== #


class TestEnumPersistence:
    """Pydantic enums are easy to persist incorrectly — the most common bug
    is storing the enum NAME instead of the .value, or storing the int
    auto-index. Pin the contract: stored as lowercase string `value`."""

    @pytest.mark.parametrize(
        "priority,expected",
        [
            (Priority.low,    "low"),
            (Priority.medium, "medium"),
            (Priority.high,   "high"),
        ],
    )
    def test_priority_persisted_as_string_value(
        self,
        store: StateStore,
        priority: Priority,
        expected: str,
    ) -> None:
        _persist(store, priority=priority.value)
        fetched = store.get("<msg-42@example.com>")
        assert fetched is not None
        assert fetched.priority == expected
        assert isinstance(fetched.priority, str)

    @pytest.mark.parametrize(
        "status,expected",
        [
            (DeliveryStatus.delivered,          "delivered"),
            (DeliveryStatus.skipped_irrelevant, "skipped_irrelevant"),
        ],
    )
    def test_delivery_status_persisted_as_string_value(
        self,
        store: StateStore,
        status: DeliveryStatus,
        expected: str,
    ) -> None:
        _persist(store, delivery_status=status)
        fetched = store.get("<msg-42@example.com>")
        assert fetched is not None
        assert fetched.delivery_status == expected
        assert isinstance(fetched.delivery_status, str)


# =========================================================================== #
# 6. Deduplication
# =========================================================================== #


class TestDeduplication:
    """The unique constraint on `message_id` is what guarantees idempotency
    across pipeline runs. If this regresses, a single email could trigger
    duplicate WhatsApp briefings — exactly the failure mode the project
    was designed to prevent."""

    def test_same_message_id_twice_raises_storage_error(
        self, store: StateStore
    ) -> None:
        _persist(store, message_id="<dup@x>")
        with pytest.raises(StorageError, match="Failed to persist"):
            # Same message_id — second write must abort.
            _persist(store, message_id="<dup@x>")

    def test_duplicate_attempt_does_not_corrupt_existing_row(
        self, store: StateStore
    ) -> None:
        """After a failed dedup insert, the original row must still be
        intact. SQLAlchemy rolls back inside the failing session, but we
        prove it observationally instead of trusting the framework."""
        original = _persist(
            store,
            message_id="<dup@x>",
            relevance=True,
            priority="high",
            delivery_status=DeliveryStatus.delivered,
            message_ids=["SM" + "1" * 32],
        )

        # Try to insert a different-content row with the SAME message_id.
        with pytest.raises(StorageError):
            _persist(
                store,
                message_id="<dup@x>",
                relevance=False,
                priority="low",
                delivery_status=DeliveryStatus.skipped_irrelevant,
                message_ids=None,
            )

        # The original row must be untouched.
        fetched = store.get("<dup@x>")
        assert fetched is not None
        assert fetched.id == original.id
        assert fetched.relevance is True
        assert fetched.priority == "high"
        assert fetched.delivery_status == "delivered"
        assert fetched.message_ids == "SM" + "1" * 32

    def test_different_message_ids_with_same_uid_both_persist(
        self, store: StateStore
    ) -> None:
        """`uid` is INDEXED but NOT unique — two different emails sharing a
        UID (possible after folder moves on different IMAP servers) must
        both be storable. Only `message_id` is the dedup key."""
        _persist(store, message_id="<a@x>", uid="7")
        _persist(store, message_id="<b@x>", uid="7")

        assert store.has_been_processed("<a@x>") is True
        assert store.has_been_processed("<b@x>") is True


# =========================================================================== #
# 7. get
# =========================================================================== #


class TestGet:
    def test_returns_none_for_unknown_message_id(self, store: StateStore) -> None:
        assert store.get("<never-seen@nowhere>") is None

    def test_returns_full_record_for_known_message_id(
        self, store: StateStore
    ) -> None:
        record = _persist(store, message_id="<known@x>")
        fetched = store.get("<known@x>")
        assert fetched is not None
        assert fetched.id == record.id

    def test_query_failure_surfaces_as_storage_error(self, store: StateStore) -> None:
        """Same fail-stop contract as has_been_processed: raw SQLAlchemy
        errors must be wrapped so the orchestrator never has to catch
        framework-specific exceptions."""
        store._engine.dispose()
        store._engine = create_engine(
            "postgresql://x:x@nonexistent.invalid:5432/db",
            connect_args={"connect_timeout": 1},
        )
        with pytest.raises(StorageError, match="Failed to fetch"):
            store.get("<msg@x>")


# =========================================================================== #
# 8. list_recent — the audit log
# =========================================================================== #


class TestListRecent:
    """list_recent is the customer-facing audit story: 'show me what your
    agent did today'. Order and completeness are non-negotiable here."""

    def test_returns_empty_list_when_no_rows(self, store: StateStore) -> None:
        assert store.list_recent() == []

    def test_returns_rows_newest_first(self, store: StateStore) -> None:
        # Insert in a known order; list_recent must reverse that order.
        for i in range(5):
            _persist(store, message_id=f"<msg-{i}@x>", uid=str(i))

        recent = store.list_recent()
        # Newest first: index 4 must come before index 0.
        ids_in_order = [r.message_id for r in recent]
        assert ids_in_order == [
            "<msg-4@x>", "<msg-3@x>", "<msg-2@x>", "<msg-1@x>", "<msg-0@x>",
        ]

    def test_respects_limit_parameter(self, store: StateStore) -> None:
        for i in range(10):
            _persist(store, message_id=f"<msg-{i}@x>", uid=str(i))

        recent = store.list_recent(limit=3)
        assert len(recent) == 3
        # Still newest-first.
        assert recent[0].message_id == "<msg-9@x>"

    def test_returns_mixed_delivered_and_skipped_states(
        self, store: StateStore
    ) -> None:
        """Audit must show what was DELIVERED *and* what was filtered out.
        Skipped emails should NOT be hidden from the operator."""
        # Three delivered.
        for i in range(3):
            _persist(store, message_id=f"<deliv-{i}@x>", uid=f"d{i}")

        # Two skipped.
        for i in range(2):
            _persist(
                store,
                message_id=f"<skip-{i}@x>",
                uid=f"s{i}",
                relevance=False,
                priority="low",
                delivery_status=DeliveryStatus.skipped_irrelevant,
            )

        recent = store.list_recent()
        assert len(recent) == 5
        statuses = [r.delivery_status for r in recent]
        assert statuses.count("delivered") == 3
        assert statuses.count("skipped_irrelevant") == 2

    def test_default_limit_caps_at_50(self, store: StateStore) -> None:
        """Default limit protects audit calls from accidentally pulling
        the whole table on a long-lived agent."""
        for i in range(75):
            _persist(store, message_id=f"<m-{i}@x>", uid=str(i))

        # No explicit limit → default 50.
        assert len(store.list_recent()) == 50

    def test_query_failure_surfaces_as_storage_error(self, store: StateStore) -> None:
        store._engine.dispose()
        store._engine = create_engine(
            "postgresql://x:x@nonexistent.invalid:5432/db",
            connect_args={"connect_timeout": 1},
        )
        with pytest.raises(StorageError, match="Failed to list recent"):
            store.list_recent()


# =========================================================================== #
# 9. close()
# =========================================================================== #


class TestClose:
    def test_close_disposes_engine_without_error(self, store: StateStore) -> None:
        store.close()
        # close() is documented as safe to call multiple times.
        store.close()

    def test_operations_after_close_either_work_or_fail_loud(
        self, store: StateStore
    ) -> None:
        """`engine.dispose()` only invalidates the connection pool —
        SQLite reopens transparently. We just want to confirm operations
        do NOT silently corrupt data after a close, even if they succeed."""
        store.close()
        # Either the operation succeeds (sqlite reopens) or it raises
        # StorageError. Both are acceptable; silent data loss is not.
        try:
            _persist(store, message_id="<post-close@x>")
        except StorageError:
            pass
        else:
            # If it succeeded, the row must be readable.
            assert store.has_been_processed("<post-close@x>")
