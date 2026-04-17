"""emAI command-line entrypoint.

This is the **only** module the operator actually launches. Everything below
it is a library — the orchestrator, the IA layers, the messaging client, the
state store. `main.py` exists to:

  1. Parse CLI arguments and pick a run mode (`--once` vs scheduled).
  2. Run a **pre-flight check** so a misconfigured deployment fails LOUD at
     startup instead of halfway through processing the first email.
  3. Wire all collaborators from `settings` and hand them to the orchestrator.
  4. Trap SIGINT / SIGTERM so a Ctrl-C (or `kill` from a process supervisor)
     drains the in-flight job, closes the IMAP socket, and disposes the DB
     pool BEFORE the process exits.

----------------------------------------------------------------------------
CLI
----------------------------------------------------------------------------

    emai                          # scheduled mode using settings.run_interval_minutes
    emai --once                   # one pass, then exit
    emai --interval 15            # scheduled, every 15 minutes (overrides settings)
    emai --once --skip-preflight  # rare: useful for debugging in CI

The `emai` console script is wired in `pyproject.toml` ([project.scripts])
to call the `run()` function defined at the bottom of this module.

----------------------------------------------------------------------------
Exit codes
----------------------------------------------------------------------------

    0   success — `--once` ran cleanly, OR scheduler shut down gracefully.
    1   pre-flight failure (missing credentials, unreachable DB at startup).
    2   pipeline crashed mid-run (storage error, unhandled exception).
    130 process was interrupted (SIGINT, by convention 128 + signal number).

These codes matter because operators wire the binary into systemd /
docker-compose / a cron wrapper and use them to decide whether to alert.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import Settings, settings
from src.ai.classifier import EmailClassifier
from src.ai.summarizer import EmailSummarizer
from src.core.orchestrator import Orchestrator, RunStats
from src.email_client.gmail_imap import GmailIMAPClient
from src.messaging.whatsapp_twilio import WhatsAppClient
from src.storage.state import StateStore, StorageError
from src.utils.logger import log


# --------------------------------------------------------------------------- #
# Exit codes — kept as named constants so logs and tests stay readable.
# --------------------------------------------------------------------------- #


EXIT_OK: int = 0
EXIT_PREFLIGHT_FAILURE: int = 1
EXIT_RUN_FAILURE: int = 2
EXIT_INTERRUPTED: int = 130  # POSIX convention: 128 + SIGINT(2)


# --------------------------------------------------------------------------- #
# Pre-flight check — fail LOUD before we even touch IMAP.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PreflightResult:
    """Outcome of the credential / dependency sanity check.

    A non-empty `errors` list means we MUST NOT proceed: the operator either
    forgot a `.env` value or the deployment got the wrong image. Either way
    a half-running pipeline is worse than a hard failure with a clear list
    of what's missing.
    """

    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight_check(cfg: Settings) -> PreflightResult:
    """Verify that every credential the pipeline needs is actually present.

    We do NOT call the remote services here (that would slow startup and
    hide real outages behind preflight noise). We only check that the
    Pydantic Settings model carries non-empty values for the things we
    can't run without:

      • Active LLM provider key (Anthropic OR OpenAI, depending on
        `LLM_PROVIDER`).
      • Twilio credentials + sender + recipient.
      • IMAP host / username / password.
      • A database URL — the StateStore will probe the actual connection
        on construction (that's where Supabase reachability is verified).

    Pydantic itself already enforces "required" at import time for the
    most critical fields; this function exists to surface the **logical**
    requirements that depend on configuration choices (e.g. an Anthropic
    key is only required when provider=anthropic).
    """
    errors: list[str] = []

    # ---- LLM provider key matches the chosen provider ----
    provider = cfg.llm.provider.value
    if provider == "anthropic" and not cfg.anthropic_api_key:
        errors.append(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty"
        )
    if provider == "openai" and not cfg.openai_api_key:
        errors.append(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is empty"
        )

    # ---- Twilio credentials ----
    if not cfg.whatsapp.account_sid.get_secret_value():
        errors.append("TWILIO_ACCOUNT_SID is empty")
    if not cfg.whatsapp.auth_token.get_secret_value():
        errors.append("TWILIO_AUTH_TOKEN is empty")
    if not cfg.whatsapp.whatsapp_from:
        errors.append("TWILIO_WHATSAPP_FROM is empty")
    if not cfg.whatsapp_to:
        errors.append("WHATSAPP_TO is empty")

    # ---- IMAP credentials ----
    if not cfg.imap.host:
        errors.append("IMAP_HOST is empty")
    if not cfg.imap.username:
        errors.append("IMAP_USERNAME is empty")
    if not cfg.imap.password.get_secret_value():
        errors.append("IMAP_PASSWORD is empty")

    # ---- Database URL ----
    if not cfg.database_url:
        errors.append("DATABASE_URL is empty")

    return PreflightResult(errors=errors)


def _log_preflight(result: PreflightResult) -> None:
    """Emit a single, scannable log block summarizing the preflight result."""
    if result.ok:
        log.info("[PREFLIGHT] all credentials present — proceeding to startup")
        return
    log.error("[PREFLIGHT] FAILED — {} issue(s) found:", len(result.errors))
    for err in result.errors:
        log.error("[PREFLIGHT]   • {}", err)
    log.error(
        "[PREFLIGHT] fix the .env file (see .env.example) and try again"
    )


# --------------------------------------------------------------------------- #
# Wiring — one place that turns Settings into a fully-built Orchestrator.
# --------------------------------------------------------------------------- #


def build_orchestrator() -> tuple[Orchestrator, GmailIMAPClient, StateStore]:
    """Construct every collaborator and return the wired orchestrator.

    Returns the IMAP client and state store alongside so the signal handler
    can close them on shutdown without going through the orchestrator.
    Constructing the StateStore here (rather than lazily on first run)
    means a Supabase outage at startup raises StorageError NOW, where the
    operator is watching the logs — not 30 minutes later.
    """
    state = StateStore()  # SELECT 1 happens here — fail-stop if DB is down
    email_client = GmailIMAPClient()
    classifier = EmailClassifier()
    summarizer = EmailSummarizer()
    whatsapp = WhatsAppClient()

    orch = Orchestrator(
        email_client=email_client,
        classifier=classifier,
        summarizer=summarizer,
        whatsapp=whatsapp,
        state=state,
    )
    return orch, email_client, state


# --------------------------------------------------------------------------- #
# Run modes
# --------------------------------------------------------------------------- #


def run_once(orch: Orchestrator) -> RunStats:
    """Execute a single pipeline pass. Used both by `--once` and as the
    APScheduler job body in scheduled mode."""
    return orch.run()


def run_scheduled(
    orch: Orchestrator,
    *,
    interval_minutes: int,
    scheduler_factory: Callable[[], BlockingScheduler] = BlockingScheduler,
) -> None:
    """Run the pipeline every `interval_minutes` until the process is
    interrupted (Ctrl-C / SIGTERM). Blocks the calling thread.

    `scheduler_factory` is the dependency-injection seam that makes this
    function testable: tests pass a fake scheduler that records the job
    registration without ever actually blocking.
    """
    scheduler = scheduler_factory()
    scheduler.add_job(
        lambda: _safe_pipeline_pass(orch),
        trigger="interval",
        minutes=interval_minutes,
        id="emai-pipeline",
        next_run_time=_now(),  # fire immediately on startup, then every N minutes
        max_instances=1,        # never overlap two pipeline passes
        coalesce=True,          # if we missed runs (laptop sleep), run once not N times
    )
    log.info(
        "[SCHEDULER] starting — first run NOW, then every {} minute(s)",
        interval_minutes,
    )
    scheduler.start()  # blocks until shutdown() or KeyboardInterrupt


def _safe_pipeline_pass(orch: Orchestrator) -> None:
    """Wrap `orch.run()` so a single failed pass doesn't kill the scheduler.

    StorageError still propagates intentionally: if the DB is gone we want
    APScheduler to surface the exception and (at the operator's discretion)
    bring the whole process down rather than silently grinding on a broken
    state store. Other exceptions are logged and swallowed so the next
    scheduled tick gets a clean attempt.
    """
    try:
        orch.run()
    except StorageError:
        log.critical("[SCHEDULER] storage failure — aborting scheduler loop")
        raise
    except Exception as exc:  # noqa: BLE001 — last-ditch guard for the loop
        log.exception(
            "[SCHEDULER] unhandled error during pipeline pass: {}: {}",
            type(exc).__name__, exc,
        )


def _now() -> "datetime":  # pragma: no cover — trivial wrapper
    """Indirection so tests can patch `time.now` without freezing the clock."""
    from datetime import datetime
    return datetime.now()


# --------------------------------------------------------------------------- #
# Signal handling — graceful shutdown.
# --------------------------------------------------------------------------- #


class _ShutdownCoordinator:
    """Centralizes the cleanup actions a SIGINT/SIGTERM needs to perform.

    The handler is intentionally tiny: do the minimum work that lets the
    process exit cleanly. We MUST NOT do any I/O that could itself raise
    inside a signal handler — instead we delegate to `shutdown()` which
    runs in the main thread after `scheduler.start()` returns.
    """

    def __init__(
        self,
        *,
        scheduler: BlockingScheduler | None = None,
        email_client: GmailIMAPClient | None = None,
        state: StateStore | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.email_client = email_client
        self.state = state
        self._shutdown_initiated: bool = False

    def request_shutdown(self, signum: int, _frame: FrameType | None) -> None:
        """Signal handler. Idempotent — a second Ctrl-C from an impatient
        operator must NOT trigger a double shutdown."""
        if self._shutdown_initiated:
            log.warning("[SIGNAL] shutdown already in progress — be patient")
            return
        self._shutdown_initiated = True
        signal_name = signal.Signals(signum).name
        log.info("[SIGNAL] received {} — initiating graceful shutdown", signal_name)

        # Asking APScheduler to stop is safe inside a signal handler —
        # internally it just sets a flag and unblocks the main thread.
        if self.scheduler is not None and self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def shutdown(self) -> None:
        """Close every external resource. Called from the main thread,
        AFTER `scheduler.start()` has returned (or the `--once` block
        exits). Each step is wrapped so a failure in one cleanup does
        not prevent the next from running."""
        log.info("[SHUTDOWN] closing IMAP connection")
        if self.email_client is not None:
            try:
                self.email_client.disconnect()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                log.warning("[SHUTDOWN] IMAP disconnect failed: {}", exc)

        log.info("[SHUTDOWN] disposing database pool")
        if self.state is not None:
            try:
                self.state.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                log.warning("[SHUTDOWN] DB close failed: {}", exc)

        log.info("[SHUTDOWN] complete")


def _install_signal_handlers(coordinator: _ShutdownCoordinator) -> None:
    """Wire SIGINT (Ctrl-C) and SIGTERM (process supervisor) into the
    coordinator. SIGTERM is what `docker stop` and `systemctl stop` send
    by default — we need to honor it for clean container shutdowns."""
    signal.signal(signal.SIGINT, coordinator.request_shutdown)
    signal.signal(signal.SIGTERM, coordinator.request_shutdown)


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments. Exposed (not _underscored) so tests
    can call it directly with a list of fake argv tokens."""
    parser = argparse.ArgumentParser(
        prog="emai",
        description=(
            "emAI — agente que lê emails não-lidos e envia um briefing "
            "executivo para o WhatsApp."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="executa exatamente um ciclo do pipeline e encerra",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        metavar="MINUTES",
        help=(
            "intervalo (em minutos) entre execuções no modo agendado. "
            "Sobrescreve RUN_INTERVAL_MINUTES do .env"
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="pula a verificação de credenciais (uso restrito a debug local)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Entry point — wired to `emai` via [project.scripts] in pyproject.toml.
# --------------------------------------------------------------------------- #


def run(argv: list[str] | None = None) -> int:
    """Top-level entry point. Returns the exit code (0 / 1 / 2 / 130).

    `argv` defaults to `sys.argv[1:]` (argparse standard); the parameter
    exists so the test suite can drive this without `sys.argv` games.
    """
    args = _parse_args(argv)

    # ---- Preflight ----
    if not args.skip_preflight:
        result = preflight_check(settings)
        _log_preflight(result)
        if not result.ok:
            return EXIT_PREFLIGHT_FAILURE

    # ---- Wire everything ----
    try:
        orch, email_client, state = build_orchestrator()
    except StorageError as exc:
        log.error("[STARTUP] storage unreachable at boot: {}", exc)
        return EXIT_PREFLIGHT_FAILURE
    except Exception as exc:  # noqa: BLE001 — startup must not leak exceptions
        log.exception("[STARTUP] failed to construct orchestrator: {}", exc)
        return EXIT_PREFLIGHT_FAILURE

    coordinator = _ShutdownCoordinator(
        email_client=email_client, state=state,
    )

    # ---- Run ----
    if args.once:
        # In `--once` mode we don't need APScheduler, but we still install
        # the signal handlers so a Ctrl-C mid-pipeline closes the IMAP/DB.
        _install_signal_handlers(coordinator)
        try:
            stats = run_once(orch)
        except StorageError as exc:
            log.error("[RUN] storage failure: {}", exc)
            coordinator.shutdown()
            return EXIT_RUN_FAILURE
        except KeyboardInterrupt:
            log.info("[RUN] interrupted by user")
            coordinator.shutdown()
            return EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001 — top-level safety net
            log.exception("[RUN] unhandled error: {}: {}", type(exc).__name__, exc)
            coordinator.shutdown()
            return EXIT_RUN_FAILURE
        coordinator.shutdown()
        log.info(
            "[RUN] one-shot complete — delivered={} irrelevant={} failed={}",
            stats.delivered, stats.skipped_irrelevant, stats.failed,
        )
        return EXIT_OK

    # Scheduled mode. APScheduler's BlockingScheduler intercepts SIGINT on
    # its own, but we install our coordinator first so we know exactly
    # which cleanup ran and in what order.
    # Explicit None-check, NOT `or`: `--interval 0` is meaningful here
    # (it must trigger the validation below), and `0 or X` would silently
    # fall back to X.
    interval = (
        args.interval if args.interval is not None
        else settings.run_interval_minutes
    )
    if interval < 1:
        log.error("[STARTUP] --interval must be >= 1 minute (got {})", interval)
        return EXIT_PREFLIGHT_FAILURE

    scheduler = BlockingScheduler()
    coordinator.scheduler = scheduler
    _install_signal_handlers(coordinator)

    try:
        run_scheduled(
            orch,
            interval_minutes=interval,
            scheduler_factory=lambda: scheduler,
        )
    except (KeyboardInterrupt, SystemExit):
        log.info("[RUN] interrupted by user")
        coordinator.shutdown()
        return EXIT_INTERRUPTED
    except StorageError:
        coordinator.shutdown()
        return EXIT_RUN_FAILURE
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        log.exception("[RUN] unhandled error in scheduler: {}: {}",
                      type(exc).__name__, exc)
        coordinator.shutdown()
        return EXIT_RUN_FAILURE

    coordinator.shutdown()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover — exercised by `emai` script
    sys.exit(run())
