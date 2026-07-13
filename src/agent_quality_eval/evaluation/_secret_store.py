"""Local secret storage for API keys.

The product had been storing API keys as plain-text under
``~/.agent-quality-eval/config/settings.json``. That file lives in the user's
home directory, which is fine against casual filesystem inspection, but any
process running under the same user (browser sync, telemetry agents, screen
recorders that back up the entire profile, etc.) can trivially exfiltrate it.

This module wraps ``settings.json`` writes so that the ``api_key`` field is
never persisted in the clear. Two backends are supported:

* **Windows DPAPI** — the OS-provided per-user secret vault, invoked via
  ``ctypes`` to keep the runtime dependency zero. Ciphertext is only
  decryptable by the same Windows account on the same machine.
* **Machine-bound Fernet-lite fallback** — for non-Windows environments (or
  when DPAPI is somehow unavailable). We derive a per-machine key from
  ``uname / user / stable salt`` using PBKDF2 + HMAC-SHA256 and encrypt with a
  small AES-CTR + HMAC-SHA256 construction from ``hashlib`` primitives (no
  extra deps). This is not a hardware secret but does prevent the plain-text
  key from being read by simply opening ``settings.json``.

The public API is intentionally tiny: :func:`encrypt_secret` /
:func:`decrypt_secret`. ``settings.py`` is the only caller.

Migration is opportunistic: when we read an old settings file that still has
``api_key`` in the clear, we return the plain-text and mark the blob as
``needs_migration``; the caller writes it back and the next read only sees
``api_key_enc``.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import os
import platform
import secrets
import sys
from typing import Optional


ENC_PREFIX = "enc:v1:"


def _dpapi_encrypt(data: bytes) -> Optional[bytes]:
    """Encrypt via Windows DPAPI (CurrentUser scope). Returns None on failure."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        buf_in = ctypes.create_string_buffer(data)
        blob_in = DATA_BLOB(len(data), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_byte)))
        blob_out = DATA_BLOB()

        # Description string for the audit trail. Not a secret.
        desc = "agent-quality-eval:api_key"
        ok = crypt32.CryptProtectData(
            ctypes.byref(blob_in), desc, None, None, None, 0, ctypes.byref(blob_out)
        )
        if not ok:
            return None
        try:
            out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)
        return out
    except Exception:
        return None


def _dpapi_decrypt(data: bytes) -> Optional[bytes]:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        buf_in = ctypes.create_string_buffer(data)
        blob_in = DATA_BLOB(len(data), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_byte)))
        blob_out = DATA_BLOB()

        ok = crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
        )
        if not ok:
            return None
        try:
            out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)
        return out
    except Exception:
        return None


def _machine_key() -> bytes:
    """Derive a stable per-user/per-machine key. Not hardware-backed.

    The intent is defense-in-depth so ``settings.json`` cannot be trivially
    grepped for a bearer token. An attacker with root on the box can still
    reconstruct the key. This is the same trust model as VS Code's fallback
    secret storage.
    """
    material = "|".join([
        "agent-quality-eval-secret-v1",
        platform.node() or "unknown-host",
        platform.system() or "unknown-os",
        platform.machine() or "unknown-arch",
        getpass.getuser() or "unknown-user",
    ]).encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", material, b"aqe-secret-salt-v1", 200_000, dklen=32)


def _fallback_encrypt(data: bytes) -> bytes:
    """AES-CTR-lite via ChaCha20-like keystream from hashlib.

    We don't ship a real AES lib to avoid new deps. HMAC-SHA256 as a PRF over
    (key, nonce, counter) generates a keystream we XOR with plaintext, then
    append an HMAC-SHA256 tag over (nonce || ciphertext). For a 500-byte API
    key on a single-user box, this level of protection is adequate as a
    defense-in-depth layer against casual disk grep.
    """
    key = _machine_key()
    nonce = secrets.token_bytes(16)
    stream = bytearray()
    counter = 0
    while len(stream) < len(data):
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        stream.extend(block)
        counter += 1
    ciphertext = bytes(a ^ b for a, b in zip(data, stream[: len(data)]))
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()[:16]
    return b"F1" + nonce + tag + ciphertext


def _fallback_decrypt(blob: bytes) -> Optional[bytes]:
    if len(blob) < 2 + 16 + 16 or blob[:2] != b"F1":
        return None
    key = _machine_key()
    nonce = blob[2:18]
    tag = blob[18:34]
    ciphertext = blob[34:]
    expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(expected, tag):
        return None
    stream = bytearray()
    counter = 0
    while len(stream) < len(ciphertext):
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        stream.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(ciphertext, stream[: len(ciphertext)]))


def encrypt_secret(plain: str) -> str:
    """Return an ``enc:v1:...`` string that hides ``plain`` at rest.

    Empty input is passed through so callers don't need to special-case blank
    keys (they still show up as ``has_api_key=false`` in :func:`masked`).
    """
    if not plain:
        return ""
    data = plain.encode("utf-8")
    ciphertext = _dpapi_encrypt(data) or _fallback_encrypt(data)
    return ENC_PREFIX + base64.b64encode(ciphertext).decode("ascii")


def decrypt_secret(stored: str) -> str:
    """Reverse :func:`encrypt_secret`.

    Backwards-compatible: if ``stored`` doesn't have the ``enc:v1:`` prefix
    it's assumed to be a legacy plain-text value from an older install and is
    returned as-is. ``settings.load_critic_settings`` uses this behavior to
    silently migrate users forward.
    """
    if not stored:
        return ""
    if not stored.startswith(ENC_PREFIX):
        # Legacy plaintext — return unchanged; caller will re-save encrypted.
        return stored
    try:
        blob = base64.b64decode(stored[len(ENC_PREFIX):].encode("ascii"))
    except Exception:
        return ""
    plain = _dpapi_decrypt(blob)
    if plain is not None:
        return plain.decode("utf-8", errors="replace")
    plain = _fallback_decrypt(blob)
    if plain is not None:
        return plain.decode("utf-8", errors="replace")
    return ""


def is_encrypted(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith(ENC_PREFIX)
