"""emAI command-line entrypoint.

This is the **only** module the operator actually launches. Everything below
it is a library — the orchestrator, the IA layers, the messaging client, the
state store. `main.py` exists to:

  1. Parse CLI arguments and pick a run mode (`--once` vs scheduled).
  2. Load the user's credentials + settings from the database.
  3. Run a **pre-flight check** so a misconfigured user fails LOUD at
     startup instead of halfway through processing the first email.
  4. Wire all collaborators from the loaded config and hand them to the
     orchestrator.
  5. Trap SIGINT / SIGTERM for graceful shutdown.

----------------------------------------------------------------------------
CLI
----------------------------------------------------------------------------

    emai --user-id philip                    # scheduled mode
    emai --user-id philip --once             # one pass, then exit
    emai --user-id philip --interval 15      # override interval
    emai --user-id philip --once --skip-preflight  # debug only

----------------------------------------------------------------------------
Exit codes
----------------------------------------------------------------------------

    0   success
    1   pre-flight failure (missing credentials, unreachable DB)
    2   pipeline crashed mid-run (storage error, unhandled exception)
    130 process was interrupted (SIGINT)
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlmodel import SQLModel

from config.runtime_settings import UserRuntimeConfig
from config.settings import settings
from src.ai.classifier import EmailClassifier
from src.ai.llm_client import LLMClient
from src.ai.summarizer import EmailSummarizer
from src.core.orchestrator import Orchestrator, RunStats
from src.email_client.gmail_imap import GmailIMAPClient
from src.messaging.whatsapp_twilio import WhatsAppClient
from src.storage.credentials import CredentialStore
from src.storage.state import StateStore, StorageError, _build_engine
from src.storage.user_settings import UserSettingsStore
from src.utils.logger import log


# --------------------------------------------------------------------------- #
# Exit codes
# --------------------------------------------------------------------------- #

EXIT_OK: int = 0
EXIT_PREFLIGHT_FAILURE: int = 1
EXIT_RUN_FAILURE: int = 2
EXIT_INTERRUPTED: int = 130


# --------------------------------------------------------------------------- #
# Load user config from database
# --------------------------------------------------------------------------- #


def load_user_config(user_id: str) -> UserRuntimeConfig:
    """Query `user_credentials` + `user_settings` and build a runtime config.

    Decrypted secrets live only in the returned `UserRuntimeConfig` object
    and are garbage-collected when the caller drops the reference.
    """
    engine = _build_engine(settings.database_url)

    # Ensure tables exist (for dev/SQLite; Supabase uses migrations)
    from src.storage.credentials import UserCredential
    from src.storage.user_settings import UserSetting

    SQLModel.metadata.create_all(engine)

    cred_store = CredentialStore(engine)
    settings_store = UserSettingsStore(engine)

    creds = cred_store.get_decrypted(user_id)
    user_settings = settings_store.get(user_id)

    if not creds:
        raise StorageError(
            f"No credentials found for user_id={user_id!r}. "
            "Ask the Admin to provision this user first."
        )
    if not user_settings:
        raise StorageError(
            f"No settings found for user_id={user_id!r}. "
            "The user must configure email/WhatsApp in the User tab first."
        )

    return UserRuntimeConfig(
        user_id=user_id,
        anthropic_api_key=creds.get("anthropic_key"),
        openai_api_key=creds.get("openai_key"),
        twilio_account_sid=creds.get("twilio_sid"),
        twilio_auth_token=creds.get("twilio_token"),
        twilio_whatsapp_from=creds.get("twilio_number"),
        supabase_url=creds.get("supabase_url"),
        supabase_key=creds.get("supabase_key"),
        imap_host=str(user_settings.get("imap_host", "imap.gmail.com")),
        imap_port=int(user_settings.get("imap_port", 993)),
        imap_username=str(user_settings.get("email", "")),
        imap_password=creds.get("gmail_app_password") or "",
        whatsapp_to=str(user_settings.get("whatsapp_to", "")),
        run_interval_minutes=int(user_settings.get("run_interval_minutes", 30)),
        database_url=settings.database_url,
    )


# --------------------------------------------------------------------------- #
# Pre-flight check
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PreflightResult:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight_check(config: UserRuntimeConfig) -> PreflightResult:
    """Verify that every credential the pipeline needs is present."""
    return PreflightResult(errors=config.validate())


def _log_preflight(result: PreflightResult) -> None:
    if result.ok:
        log.info("[PREFLIGHT] all credentials present — proceeding to startup")
        return
    log.error("[PREFLIGHT] FAILED — {} issue(s) found:", len(result.errors))
    for err in result.errors:
        log.error("[PREFLIGHT]   * {}", err)
    log.error(
        "[PREFLIGHT] provision credentials via the Admin panel and configure "
        "email/WhatsApp in the User panel, then try again"
    )


# --------------------------------------------------------------------------- #
# Wiring — build orchestrator from UserRuntimeConfig
# --------------------------------------------------------------------------- #


def build_orchestrator(
    config: UserRuntimeConfig,
) -> tuple[Orchestrator, GmailIMAPClient, StateStore]:
    """Construct every collaborator from the per-user runtime config."""
    state = StateStore(engine=_build_engine(config.database_url), user_id=config.user_id)

    email_client = GmailIMAPClient(
        host=config.imap_host,
        port=config.imap_port,
        username=config.imap_username,
        password=config.imap_password,
    )

    llm = LLMClient(
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
    )
    classifier = EmailClassifier(llm=llm)
    summarizer = EmailSummarizer(llm=llm)

    whatsapp = WhatsAppClient(
        account_sid=config.twilio_account_sid or "",
        auth_token=config.twilio_auth_token or "",
        whatsapp_from=config.twilio_whatsapp_from or "",
        whatsapp_to=config.whatsapp_to,
    )

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
    return orch.run()


def run_scheduled(
    orch: Orchestrator,
    *,
    interval_minutes: int,
    scheduler_factory: Callable[[], BlockingScheduler] = BlockingScheduler,
) -> None:
    scheduler = scheduler_factory()
    scheduler.add_job(
        lambda: _safe_pipeline_pass(orch),
        trigger="interval",
        minutes=interval_minutes,
        id="emai-pipeline",
        next_run_time=_now(),
        max_instances=1,
        coalesce=True,
    )
    log.info(
        "[SCHEDULER] starting — first run NOW, then every {} minute(s)",
        interval_minutes,
    )
    scheduler.start()


def _safe_pipeline_pass(orch: Orchestrator) -> None:
    try:
        orch.run()
    except StorageError:
        log.critical("[SCHEDULER] storage failure — aborting scheduler loop")
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "[SCHEDULER] unhandled error during pipeline pass: {}: {}",
            type(exc).__name__, exc,
        )


def _now() -> "datetime":  # pragma: no cover
    from datetime import datetime
    return datetime.now()


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #


class _ShutdownCoordinator:
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
        if self._shutdown_initiated:
            log.warning("[SIGNAL] shutdown already in progress — be patient")
            return
        self._shutdown_initiated = True
        signal_name = signal.Signals(signum).name
        log.info("[SIGNAL] received {} — initiating graceful shutdown", signal_name)
        if self.scheduler is not None and self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def shutdown(self) -> None:
        log.info("[SHUTDOWN] closing IMAP connection")
        if self.email_client is not None:
            try:
                self.email_client.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.warning("[SHUTDOWN] IMAP disconnect failed: {}", exc)

        log.info("[SHUTDOWN] disposing database pool")
        if self.state is not None:
            try:
                self.state.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("[SHUTDOWN] DB close failed: {}", exc)

        log.info("[SHUTDOWN] complete")


def _install_signal_handlers(coordinator: _ShutdownCoordinator) -> None:
    signal.signal(signal.SIGINT, coordinator.request_shutdown)
    signal.signal(signal.SIGTERM, coordinator.request_shutdown)


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="emai",
        description=(
            "emAI — agente que le emails nao-lidos e envia um briefing "
            "executivo para o WhatsApp."
        ),
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="ID do usuario cujas credenciais serao carregadas do banco",
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
        help="intervalo (em minutos) entre execucoes no modo agendado",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="pula a verificacao de credenciais (uso restrito a debug local)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(argv: list[str] | None = None) -> int:
    """Top-level entry point. Returns the exit code (0 / 1 / 2 / 130)."""
    args = _parse_args(argv)

    # ---- Load user config from DB ----
    try:
        config = load_user_config(args.user_id)
    except StorageError as exc:
        log.error("[STARTUP] failed to load config for user '{}': {}", args.user_id, exc)
        return EXIT_PREFLIGHT_FAILURE
    except Exception as exc:  # noqa: BLE001
        log.exception("[STARTUP] unexpected error loading user config: {}", exc)
        return EXIT_PREFLIGHT_FAILURE

    # ---- Preflight ----
    if not args.skip_preflight:
        result = preflight_check(config)
        _log_preflight(result)
        if not result.ok:
            return EXIT_PREFLIGHT_FAILURE

    # ---- Wire everything ----
    try:
        orch, email_client, state = build_orchestrator(config)
    except StorageError as exc:
        log.error("[STARTUP] storage unreachable at boot: {}", exc)
        return EXIT_PREFLIGHT_FAILURE
    except Exception as exc:  # noqa: BLE001
        log.exception("[STARTUP] failed to construct orchestrator: {}", exc)
        return EXIT_PREFLIGHT_FAILURE

    coordinator = _ShutdownCoordinator(
        email_client=email_client, state=state,
    )

    # ---- Run ----
    if args.once:
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
        except Exception as exc:  # noqa: BLE001
            log.exception("[RUN] unhandled error: {}: {}", type(exc).__name__, exc)
            coordinator.shutdown()
            return EXIT_RUN_FAILURE
        coordinator.shutdown()
        log.info(
            "[RUN] one-shot complete — delivered={} irrelevant={} failed={}",
            stats.delivered, stats.skipped_irrelevant, stats.failed,
        )
        return EXIT_OK

    # Scheduled mode
    interval = (
        args.interval if args.interval is not None
        else config.run_interval_minutes
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
    except Exception as exc:  # noqa: BLE001
        log.exception("[RUN] unhandled error in scheduler: {}: {}",
                      type(exc).__name__, exc)
        coordinator.shutdown()
        return EXIT_RUN_FAILURE

    coordinator.shutdown()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
