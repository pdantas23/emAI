"""Integration tests for `src/main.py` — the CLI entry point.

Updated for the admin-provisioned architecture: `--user-id` is required,
credentials are loaded from the database, and `preflight_check` validates
a `UserRuntimeConfig` instead of reading from `.env`.
"""

from __future__ import annotations

import signal
from typing import Any
from unittest.mock import MagicMock

import pytest

from config.runtime_settings import UserRuntimeConfig
from src.ai.llm_client import LLMProvider
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _valid_config(**overrides: Any) -> UserRuntimeConfig:
    """Build a fully-valid UserRuntimeConfig for tests."""
    defaults = dict(
        user_id="test-user",
        anthropic_api_key="sk-ant-fake",
        twilio_account_sid="ACxxx",
        twilio_auth_token="token",
        twilio_whatsapp_from="whatsapp:+14155238886",
        imap_username="test@example.com",
        imap_password="test-password",
        whatsapp_to="whatsapp:+5511999999999",
        database_url="sqlite:///",
    )
    defaults.update(overrides)
    return UserRuntimeConfig(**defaults)


# =========================================================================== #
# 1. CLI argument parsing
# =========================================================================== #


class TestParseArgs:
    def test_user_id_is_required(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_user_id_is_captured(self) -> None:
        args = _parse_args(["--user-id", "philip"])
        assert args.user_id == "philip"

    def test_once_flag_is_recognized(self) -> None:
        args = _parse_args(["--user-id", "x", "--once"])
        assert args.once is True

    def test_interval_accepts_minutes(self) -> None:
        args = _parse_args(["--user-id", "x", "--interval", "15"])
        assert args.interval == 15

    def test_skip_preflight_flag(self) -> None:
        args = _parse_args(["--user-id", "x", "--skip-preflight"])
        assert args.skip_preflight is True

    def test_combined_flags(self) -> None:
        args = _parse_args(["--user-id", "x", "--once", "--interval", "5", "--skip-preflight"])
        assert args.once and args.interval == 5 and args.skip_preflight


# =========================================================================== #
# 2. Preflight check (now validates UserRuntimeConfig)
# =========================================================================== #


class TestPreflightCheck:
    def test_fully_configured_passes(self) -> None:
        result = preflight_check(_valid_config())
        assert result.ok is True
        assert result.errors == []

    def test_missing_llm_key_caught(self) -> None:
        result = preflight_check(_valid_config(anthropic_api_key=None, openai_api_key=None))
        assert not result.ok
        assert any("LLM" in e for e in result.errors)

    def test_missing_twilio_sid_caught(self) -> None:
        result = preflight_check(_valid_config(twilio_account_sid=None))
        assert any("twilio_sid" in e for e in result.errors)

    def test_missing_twilio_token_caught(self) -> None:
        result = preflight_check(_valid_config(twilio_auth_token=None))
        assert any("twilio_token" in e for e in result.errors)

    def test_missing_twilio_number_caught(self) -> None:
        result = preflight_check(_valid_config(twilio_whatsapp_from=None))
        assert any("twilio_number" in e for e in result.errors)

    def test_missing_imap_username_caught(self) -> None:
        result = preflight_check(_valid_config(imap_username=""))
        assert any("email" in e.lower() for e in result.errors)

    def test_missing_imap_password_caught(self) -> None:
        result = preflight_check(_valid_config(imap_password=""))
        assert any("gmail_app_password" in e for e in result.errors)

    def test_missing_whatsapp_to_caught(self) -> None:
        result = preflight_check(_valid_config(whatsapp_to=""))
        assert any("whatsapp_to" in e for e in result.errors)

    def test_missing_database_url_caught(self) -> None:
        result = preflight_check(_valid_config(database_url=""))
        assert any("database_url" in e for e in result.errors)

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
        _ShutdownCoordinator().shutdown()

    def test_imap_disconnect_failure_does_not_block_db_close(self) -> None:
        imap = MagicMock()
        imap.disconnect.side_effect = RuntimeError("socket already closed")
        state = MagicMock()
        coord = _ShutdownCoordinator(email_client=imap, state=state)
        coord.shutdown()
        state.close.assert_called_once()

    def test_db_close_failure_is_swallowed_with_warning(self) -> None:
        imap = MagicMock()
        state = MagicMock()
        state.close.side_effect = RuntimeError("pool already disposed")
        coord = _ShutdownCoordinator(email_client=imap, state=state)
        coord.shutdown()

    def test_request_shutdown_is_idempotent(self) -> None:
        scheduler = MagicMock()
        scheduler.running = True
        coord = _ShutdownCoordinator(scheduler=scheduler)
        coord.request_shutdown(signal.SIGINT, None)
        coord.request_shutdown(signal.SIGINT, None)
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
    """Replace `load_user_config` and `build_orchestrator` with fakes."""
    fake_imap = MagicMock()
    fake_state = MagicMock()
    fake_orch = _FakeOrchestrator()

    handle = {"orch": fake_orch, "imap": fake_imap, "state": fake_state}

    # Bypass DB load — return a valid config directly
    monkeypatch.setattr(
        "src.main.load_user_config",
        lambda user_id: _valid_config(user_id=user_id),
    )

    def fake_build(config: Any) -> tuple[Any, Any, Any]:
        return fake_orch, fake_imap, fake_state

    monkeypatch.setattr("src.main.build_orchestrator", fake_build)
    return handle


class TestRunOnce:
    def test_happy_path_returns_exit_ok_and_runs_once(
        self, patched_build: dict[str, Any]
    ) -> None:
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_OK
        assert patched_build["orch"].run_calls == 1
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_storage_error_during_run_returns_exit_run_failure(
        self, patched_build: dict[str, Any]
    ) -> None:
        patched_build["orch"].raises = StorageError("DB lost connection")
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_RUN_FAILURE
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_keyboard_interrupt_during_run_returns_130(
        self, patched_build: dict[str, Any]
    ) -> None:
        patched_build["orch"].raises = KeyboardInterrupt()
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_INTERRUPTED
        patched_build["imap"].disconnect.assert_called_once()
        patched_build["state"].close.assert_called_once()

    def test_unexpected_exception_during_run_returns_exit_run_failure(
        self, patched_build: dict[str, Any]
    ) -> None:
        patched_build["orch"].raises = RuntimeError("something else broke")
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_RUN_FAILURE


class TestRunPreflightFailure:
    def test_missing_credential_returns_exit_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.main.load_user_config",
            lambda uid: _valid_config(user_id=uid),
        )
        monkeypatch.setattr(
            "src.main.preflight_check",
            lambda cfg: PreflightResult(errors=["TEST_KEY missing"]),
        )
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_skip_preflight_bypasses_the_check(
        self, monkeypatch: pytest.MonkeyPatch, patched_build: dict[str, Any]
    ) -> None:
        called = {"n": 0}

        def fail_if_called(cfg: Any) -> PreflightResult:
            called["n"] += 1
            return PreflightResult(errors=["should not run"])

        monkeypatch.setattr("src.main.preflight_check", fail_if_called)
        rc = run(["--user-id", "test", "--once", "--skip-preflight"])
        assert rc == EXIT_OK
        assert called["n"] == 0

    def test_storage_error_at_startup_returns_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.main.load_user_config",
            lambda uid: _valid_config(user_id=uid),
        )

        def fail_build(config: Any) -> Any:
            raise StorageError("Supabase down at boot")

        monkeypatch.setattr("src.main.build_orchestrator", fail_build)
        rc = run(["--user-id", "test", "--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_load_user_config_failure_returns_preflight_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_load(uid: str) -> Any:
            raise StorageError("No credentials for this user")

        monkeypatch.setattr("src.main.load_user_config", fail_load)
        rc = run(["--user-id", "ghost", "--once"])
        assert rc == EXIT_PREFLIGHT_FAILURE


class TestRunScheduled:
    def test_invalid_interval_returns_preflight_failure(
        self, patched_build: dict[str, Any]
    ) -> None:
        rc = run(["--user-id", "test", "--interval", "0"])
        assert rc == EXIT_PREFLIGHT_FAILURE

    def test_scheduled_mode_registers_job_with_correct_interval(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--user-id", "test", "--interval", "7"])
        assert rc == EXIT_OK

        fake_scheduler.add_job.assert_called_once()
        kwargs = fake_scheduler.add_job.call_args.kwargs
        assert kwargs["minutes"] == 7
        assert kwargs["max_instances"] == 1
        assert kwargs["coalesce"] is True
        assert kwargs["next_run_time"] is not None
        assert kwargs["id"] == "emai-pipeline"
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

        rc = run(["--user-id", "test", "--interval", "10"])
        assert rc == EXIT_INTERRUPTED
        patched_build["imap"].disconnect.assert_called_once()

    def test_storage_error_inside_scheduler_returns_run_failure(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        fake_scheduler.start.side_effect = StorageError("supabase died mid-run")
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--user-id", "test", "--interval", "5"])
        assert rc == EXIT_RUN_FAILURE
        patched_build["state"].close.assert_called_once()

    def test_unexpected_exception_in_scheduler_returns_run_failure(
        self, patched_build: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scheduler = MagicMock()
        fake_scheduler.running = False
        fake_scheduler.start.side_effect = RuntimeError("apscheduler bug")
        monkeypatch.setattr("src.main.BlockingScheduler", lambda: fake_scheduler)

        rc = run(["--user-id", "test", "--interval", "3"])
        assert rc == EXIT_RUN_FAILURE


# =========================================================================== #
# 5. run_scheduled internals
# =========================================================================== #


class TestRunScheduledFunction:
    def test_scheduler_factory_seam_lets_us_inject_a_fake(self) -> None:
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
