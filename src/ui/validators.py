"""Live validators for API credentials.

Each validator makes a lightweight, real call to the respective service to
confirm the key is valid. Used by the Admin tab before saving credentials.

All validators return `(ok: bool, message: str)` so the UI can show
green/red feedback inline.
"""

from __future__ import annotations


def validate_anthropic_key(api_key: str) -> tuple[bool, str]:
    """Send a tiny completion to verify the Anthropic key works."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=10.0)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, f"OK — model responded ({resp.usage.output_tokens} tokens)"
    except anthropic.AuthenticationError:
        return False, "Invalid API key (401 Unauthorized)"
    except Exception as exc:
        return False, f"Connection failed: {type(exc).__name__}: {exc}"


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Send a tiny completion to verify the OpenAI key works."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=10.0)
        client.models.list()
        return True, "OK — key accepted"
    except openai.AuthenticationError:
        return False, "Invalid API key (401 Unauthorized)"
    except Exception as exc:
        return False, f"Connection failed: {type(exc).__name__}: {exc}"


def validate_twilio(account_sid: str, auth_token: str) -> tuple[bool, str]:
    """Verify Twilio credentials by fetching the account info."""
    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        account = client.api.accounts(account_sid).fetch()
        return True, f"OK — account '{account.friendly_name}' ({account.status})"
    except Exception as exc:
        return False, f"Twilio validation failed: {type(exc).__name__}: {exc}"


def validate_imap(
    host: str, port: int, username: str, password: str
) -> tuple[bool, str]:
    """Test an IMAP login to verify Gmail App Password works."""
    try:
        from imap_tools import MailBox, MailboxLoginError

        mb = MailBox(host, port=port)
        mb.login(username, password, initial_folder="INBOX")
        mb.logout()
        return True, f"OK — logged in to {host} as {username}"
    except MailboxLoginError as exc:
        return False, f"IMAP login failed: {exc}"
    except Exception as exc:
        return False, f"IMAP connection failed: {type(exc).__name__}: {exc}"
