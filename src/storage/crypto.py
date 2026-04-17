"""AES-256-GCM encryption for credential storage.

Every API key stored in `user_credentials` passes through this module before
touching the database and again on the way out. The encryption key itself
(`ENCRYPTION_KEY`) is the ONE secret that remains as an environment variable —
it never reaches the database.

Design decisions:
  - AES-256-GCM: authenticated encryption — tampered ciphertext is detected.
  - Random 12-byte nonce per encryption: stored alongside the ciphertext as
    `nonce || ciphertext || tag` (12 + N + 16 bytes). No nonce reuse.
  - The `cryptography` library is NIST-blessed and audited. We use the
    high-level `AESGCM` interface, not the hazmat primitives.

The ENCRYPTION_KEY must be exactly 32 bytes (256 bits), provided as a
64-character hex string in the environment.
"""

from __future__ import annotations

import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --------------------------------------------------------------------------- #
# Key loading
# --------------------------------------------------------------------------- #

_NONCE_BYTES = 12  # GCM standard nonce length


def _load_key() -> bytes:
    """Load the 256-bit encryption key from the environment.

    The key MUST be a 64-character hex string (32 bytes decoded).
    This is the only secret that lives in the environment — everything
    else comes from the database.
    """
    raw = os.environ.get("ENCRYPTION_KEY", "")
    if not raw:
        raise EnvironmentError(
            "ENCRYPTION_KEY is not set. This 64-char hex string is required "
            "to encrypt/decrypt API credentials in the database."
        )
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise EnvironmentError(
            "ENCRYPTION_KEY must be a valid hex string (64 hex characters = 32 bytes)."
        ) from exc
    if len(key) != 32:
        raise EnvironmentError(
            f"ENCRYPTION_KEY must be exactly 32 bytes (64 hex chars), got {len(key)} bytes."
        )
    return key


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def encrypt(plaintext: str) -> bytes:
    """Encrypt a plaintext string and return `nonce || ciphertext || tag`.

    The returned bytes are safe to store in a BYTEA column. The decrypted
    value lives in memory only for the duration of the pipeline run.
    """
    key = _load_key()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # ct includes the 16-byte GCM tag appended by the library
    return nonce + ct


def decrypt(blob: bytes) -> str:
    """Decrypt a `nonce || ciphertext || tag` blob back to a plaintext string.

    Raises `cryptography.exceptions.InvalidTag` if the blob was tampered
    with or the wrong ENCRYPTION_KEY is used — this is intentional.
    """
    key = _load_key()
    aesgcm = AESGCM(key)
    nonce = blob[:_NONCE_BYTES]
    ct = blob[_NONCE_BYTES:]
    plaintext = aesgcm.decrypt(nonce, ct, None)
    return plaintext.decode("utf-8")


def generate_key() -> str:
    """Generate a new random 256-bit key as a hex string.

    Convenience for operators bootstrapping a new deployment:
        python -c "from src.storage.crypto import generate_key; print(generate_key())"
    """
    return secrets.token_hex(32)
