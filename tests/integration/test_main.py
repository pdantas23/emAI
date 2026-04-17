"""Integration tests for `src/main.py` — the CLI entry point.

We test main.py the same way an operator interacts with it: by calling
`run(argv)` with a list of CLI tokens and asserting on the return code,
the side effects (IMAP/DB closed?), and the log output.

Coverage map:

  ┌────────────────────────────┬────────────────────────────────────────────┐
  │ Concern                    │ What we prove                              │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Argument parsing           │ --once / --interval N / --skip-preflight   │
  │                            │ each toggle the right flag.                │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Preflight check            │ Catches every missing credential we said   │
  │                            │ we'd catch (LLM, Twilio, IMAP, DB URL).    │
  │                            │ Returns OK when fully configured.          │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Exit codes                 │ Preflight failure → 1; StorageError at     │
  │                            │ boot → 1; StorageError in run → 2;         │
  │                            │ Ctrl-C → 130; happy path → 0.              │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Graceful shutdown          │ IMAP.disconnect AND state.close called;    │
  │                            │ idempotent across repeated SIGINT;         │
  │                            │ failure in one cleanup does not block the  │
  │                            │ next.                                      │
  ├────────────────────────────┼────────────────────────────────────────────┤
  │ Scheduled mode             │ Job registered with the right interval;    │
  │                            │ next_run_time fires immediately;           │
  │                            │ scheduler shutdown wired to coordinator.   │
  └────────────────────────────┴────────────────────────────────────────────┘
"""

from __future__ import annotations

import signal
from typing import Any
from unittest.mock import MagicMock

import pytest

from config.settings import LLMProvider
from src.core.orchestrator import RunStats
from src.main import (
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PREFLIGHT_FAILURE,
    EXIT_RUN_FAILURE,
    PreflightResult,
    _parse_args,
    _ShutdownCoordinator,
    preflight_check,
    run,
    run_scheduled,
)
from src.storage.state import StorageError


# =========================================================================== #
# 1. CLI argument parsing
# =========================================================================== #


class TestParseArgs:
    def test_no_args_defaults_to_scheduled_mode(self) -> None:
        args = _parse_args([])
        assert args.once is False
        assert args.interval is None
        assert args.skip_preflight is False

    def test_once_flag_is_recognized(self) -> None:
        args = _parse_args(["--once"])
        assert args.once is True

    def test_interval_accepts_minutes(self) -> None:
        args = _parse_args(["--interval", "15"])
        assert args.interval == 15

    def test_skip_preflight_flag(self) -> None:
        args = _parse_args(["--skip-preflight"])
        assert args.skip_preflight is True

    def test_combined_flags(self) -> None:
        args = _parse_args(["--once", "--interval", "5", "--skip-preflight"])
        assert args.once and args.interval == 5 and args.skip_preflight

    def test_invalid_interval_value_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--interval", "not-a-number"])


# =========================================================================== #
# 2. Preflight check
# =========================================================================== #


class TestPreflightCheck:
    def test_fully_configured_settings_pass(self) -> None:
        """The conftest.py sets every required env var to a valid stub —
        a clean settings object must come back with zero errors."""
        from config.settings import settings
        result = preflight_check(settings)
        assert result.ok is True
        assert result.errors == []

    def test_missing_anthropic_key_caught_when_provider_anthropic(self) -> None:
        from config.settings import settings
        # Build a settings clone where the Anthropic key has been wiped.
        cfg = settings.model_copy(update={"anthropic_api_key": None})
        result = preflight_check(cfg)
        assert result.ok is False
        assert any("ANTHROPIC_API_KEY" in e for e in result.errors)

    def test_missing_openai_key_caught_when_provider_openai(self) -> None:
        from config.settings import settings
        cfg = settings.model_copy(
            update={
                "openai_api_key": None,
                "llm": settings.llm.model_copy(update={"provider": LLMProvider.openai}),
            },
        )
        result = preflight_check(cfg)
        assert any("OPENAI_API_KEY" in e for e in result.errors)

    def test_missing_twilio_credentials_caught(self) -> None:
        from pydantic import SecretStr
        from config.settings import settings
        cfg = settings.model_copy(
            update={
                "whatsapp": settings.whatsapp.model_copy(
                    update={
                        "account_sid": SecretStr(""),
                        "auth_token":  SecretStr(""),
                    },
                ),
            },
        )
        result = preflight_check(cfg)
        joined = " | ".join(result.errors)
        assert "TWILIO_ACCOUNT_SID" in joined
        assert "TWILIO_AUTH_TOKEN" in joined

    def test_missing_imap_credentials_caught(self) -> None:
        from pydantic import SecretStr
        from config.settings import settings
        cfg = settings.model_copy(
            update={
                "imap": settings.imap.model_copy(
                    update={"host": "", "username": "", "password": SecretStr("")},
                ),
            },
        )
        result = preflight_check(cfg)
        joined = " | ".join(result.errors)
        assert "IMAP_HOST" in joined
        assert "IMAP_USERNAME" in joined
        assert "IMAP_PASSWORD" in joined

    def test_missing_database_url_caught(self) -> None:
        from config.settings import settings
        cfg = settings.model_copy(update={"database_url": ""})
        result = preflight_check(cfg)
        assert any("DATABASE_URL" in e for e in result.errors)

    def test_preflight_result_ok_property(self) -> None:
        assert PreflightResult(errors=[]).ok is True
        assert PreflightResult(errors=["x"]).ok is False


# =========================================================================== #
# 3. Shutdown coordinator
# =========================================================================== #


class TestShutdownCoordinator:
    def test_shutdown_closes_imap_and_state(self) -> None:
        imap = MagicMock()
        state = MagicMock()
        coord = _ShutdownCoordinator(email_client=imap, state=state)

        coord.shutdown()

        imap.disconnect.assert_called_once()
        state.close.assert_called_once()

    def test_shutdown_with_no_collaborators_is_a_noop(self) -> None:
        # Should not raise even when nothing is wired (e.g. early failure).
        _ShutdownCoordinator().shutdown()

    def test_imap_disconnect_failure_does_not_block_db_close(self) -> None:
        """Cleanup is best-effort. A broken IMAP socket must NOT prevent
        the DB pool from being disposed — that's how connections leak."""
        imap = MagicMock()
        imap.disconnect.side_effect = RuntimeError("socket already closed")
        state = MagicMock()
        coord = _ShutdownCoordinator(email_client=imap, state=state)

        coord.shutdown()  # must not raise

        state.close.assert_called_once()

    def test_db_close_failure_is_swallowed_with_warning(self) -> None:
        imap = MagicMock()
        state = MagicMock()
        state.close.side_effect = RuntimeError("pool already disposed")
        coord = _ShutdownCoordinator(email_client=imap, state=state)
        coord.shutdown()  # must not raise

    def test_request_shutdown_is_idempotent(self) -> None:
        """A second Ctrl-C from an impatient operator must not re-trigger
        the cleanup chain — that could double-close a socket and raise."""
        scheduler = MagicMock()
        scheduler.running = True
        coord = _ShutdownCoordinator(scheduler=scheduler)

        coord.request_shutdown(signal.SIGINT, None)
        coord.request_shutdown(signal.SIGINT, None)

        # Scheduler.shutdown only called once despite two signals.
        assert scheduler.shutdown.call_count == 1

    def test_request_shutdown_skips_scheduler_when_not_running(self) -> None:
        scheduler = MagicMock()
        scheduler.running = False
        coord = _ShutdownCoordinator(scheduler=scheduler)
        coord.request_shutdown(signal.SIGTERM, None)
        scheduler.shutdown.assert_not_called()


# =========================================================================== #
# 4. run() — top-level entry point
# =========================================================================== #


class _FakeOrchestrator:
    """Minimal stand-in for the real Orchestrator. Records `run` calls and
    optionally raises when the test wants to exercise an error path."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.raises = raises
        self.run_calls: int = 0

    def run(self) -> RunStats:
        self.run_calls += 1
        if self.raises is not None:
            raise self.raises
        return RunStats(fetched=2, delivered=1, skipped_irrelevant=1)


@pytest.fixture
def patched_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `build_orchestrator` with a fake that returns mocks for the
    IMAP client and state store. Returns a dict of the wired pieces so
    each test can grab the one it needs to assert against."""
    fake_imap = MagicMock()
    fake_state = MagicMock()
    fake_orch = _FakeOrchestrator()

    handle = {"orch": fake_orch, "imap": fake_imap, "state": fake_state}

    def fake_build() -> tuple[Any, Any, Any]:
        return fake_orch, fake_imap, fake_state

    monkeypatch.setattr("src.main.build_orchestrator", fake_build)
    return handle


class TestRunOnce:
    def test_happy_path_returns_exit_ok_and_runs_once(
        self, patched_build: dict[str, Any]
    ) -> None:
        rc = run(["--once"])
        assert rc == EXIT_OK
        assert patched_build["orch"].run_calls == 1
        # Cleanup ran on exit.
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_storage_error_during_run_returns_exit_run_failure(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patched_build["orch"].raises = StorageError("DB lost connection")
        rc = run(["--once"])
        assert rc == EXIT_RUN_FAILURE
        # Cleanup still ran — that's the load-bearing guarantee.
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_keyboard_interrupt_during_run_returns_130(
        self, patched_build: dict[str, Any]
    ) -> None:
        patched_build["orch"].raises = KeyboardInterrupt()
        rc = run(["--once"])
        assert rc == EXIT_INTERRUPTED
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_unexpected_exception_during_run_returns_exit_run_failure(
        self, patched_build: dict[str, Any]
    ) -> None:
        patched_build["orch"].raises = RuntimeError("something else broke")
        rc = run(["--once"])
        assert rc == EXIT_RUN_FAILURE


class TestRunPreflightFailure:
    def test_missing_credential_returns_exit_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.main.preflight_check",
            lambda cfg: PreflightResult(errors=["TEST_KEY missing"]),
        )
        rc = run(["--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_skip_preflight_bypasses_the_check(
        self, monkeypatch: pytest.MonkeyPatch, patched_build: dict[str, Any]
    ) -> None:
        """If the operator explicitly skips preflight, even broken creds
        must not abort startup. Useful for local debugging where partial
        config is intentional."""
        called = {"n": 0}

        def fail_if_called(cfg: Any) -> PreflightResult:
            called["n"] += 1
            return PreflightResult(errors=["should not run"])

        monkeypatch.setattr("src.main.preflight_check", fail_if_called)
        rc = run(["--once", "--skip-preflight"])
        assert rc == EXIT_OK
        assert called["n"] == 0, "preflight_check must not be invoked"

    def test_storage_error_at_startup_returns_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `StateStore()` raises during build_orchestrator (Supabase
        unreachable), we must report it as a STARTUP problem, not a runtime
        one — the operator should treat it like a missing credential."""

        def fail_build() -> Any:
            raise StorageError("Supabase down at boot")

        monkeypatch.setattr("src.main.build_orchestrator", fail_build)
        rc = run(["--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_unexpected_startup_exception_returns_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> Any:
            raise RuntimeError("pyproject is broken")

        monkeypatch.setattr("src.main.build_orchestrator", boom)
        rc = run(["--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE


class TestRunScheduled:
    def test_invalid_interval_returns_preflight_failure(
        self, patched_build: dict[str, Any]
    ) -> None:
        rc = run(["--interval", "0"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_scheduled_mode_registers_job_with_correct_interval(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replace the scheduler with a stub that records `add_job` and
        immediately returns from `start()` so the test never blocks."""
        fake_scheduler = MagicMock()
        fake_scheduler.running = False

        # `BlockingScheduler()` is constructed inside run() — patch it.
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--interval", "7"])
        assert rc == EXIT_OK

        # The job was registered with interval=7.
        fake_scheduler.add_job.assert_called_once()
        kwargs = fake_scheduler.add_job.call_args.kwargs
        assert kwargs["minutes"] == 7
        assert kwargs["max_instances"] == 1
        assert kwargs["coalesce"] is True
        assert kwargs["next_run_time"] is not None  # fires immediately
        assert kwargs["id"] == "emai-pipeline"
        # And start was called → cleanup ran.
        fake_scheduler.start.assert_called_once()
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_keyboard_interrupt_in_scheduled_mode_returns_130(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        fake_scheduler.start.side_effect = KeyboardInterrupt()
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--interval", "10"])
        assert rc == EXIT_INTERRUPTED
        patched_build["imap"].disconnect.assert_called_once()

    def test_storage_error_inside_scheduler_returns_run_failure(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        fake_scheduler.start.side_effect = StorageError("supabase died mid-run")
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--interval", "5"])
        assert rc == EXIT_RUN_FAILURE
        patched_build["state"].close.assert_called_once()

    def test_unexpected_exception_in_scheduler_returns_run_failure(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        fake_scheduler.start.side_effect = RuntimeError("apscheduler bug")
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--interval", "3"])
        assert rc == EXIT_RUN_FAILURE


# =========================================================================== #
# 5. run_scheduled internals — via the scheduler_factory injection seam
# =========================================================================== #


class TestRunScheduledFunction:
    def test_scheduler_factory_seam_lets_us_inject_a_fake(self) -> None:
        """`run_scheduled` accepts a scheduler factory specifically so we
        can drive it without actually blocking. Pin that contract."""
        fake_orch = _FakeOrchestrator()
        fake_scheduler = MagicMock()
        fake_scheduler.running = False

        run_scheduled(
            fake_orch,  # type: ignore[arg-type]
            interval_minutes=2,
            scheduler_factory=lambda: fake_scheduler,
        )

        fake_scheduler.add_job.assert_called_once()
        fake_scheduler.start.assert_called_once()


# =========================================================================== #
# 6. Safe pipeline pass — wraps orch.run() with selective error handling.
# =========================================================================== #


class TestSafePipelinePass:
    """`_safe_pipeline_pass` is what APScheduler invokes every tick.
    StorageError must propagate (kills the loop intentionally), every
    other exception must be logged + swallowed (so a transient blip
    doesn't kill the long-running scheduler)."""

    def test_storage_error_propagates_out_of_safe_wrapper(self) -> None:
        from src.main import _safe_pipeline_pass

        orch = _FakeOrchestrator(raises=StorageError("DB blew up"))
        with pytest.raises(StorageError):
            _safe_pipeline_pass(orch)  # type: ignore[arg-type]

    def test_other_exceptions_are_swallowed_so_scheduler_keeps_running(
        self,
    ) -> None:
        from src.main import _safe_pipeline_pass

        orch = _FakeOrchestrator(raises=RuntimeError("twilio glitch"))
        # Must NOT raise — the scheduler would be killed otherwise.
        _safe_pipeline_pass(orch)  # type: ignore[arg-type]
        assert orch.run_calls == 1

    def test_clean_pass_returns_normally(self) -> None:
        from src.main import _safe_pipeline_pass
        orch = _FakeOrchestrator()
        _safe_pipeline_pass(orch)  # type: ignore[arg-type]
        assert orch.run_calls == 1
