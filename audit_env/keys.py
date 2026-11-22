"""Keyed randomness: public IDs and data never reveal the seed without the key.

Open train/dev episodes use the published OPEN_KEY so anyone can regenerate them.
Hidden episodes use a secret key held by the evaluator (env var, key file, or a
per-process fallback). An agent that knows the generator code and guesses the
seed still cannot reproduce a secret-key episode.
"""
import hashlib
import hmac
import os
import sys
import time
from pathlib import Path
from random import Random

OPEN_KEY = b"fin-case-1/open-episodes/v2"
ENV_VAR = "AUDIT_ENV_SECRET"
KEY_FILE_VAR = "AUDIT_ENV_KEY_FILE"
MIN_KEY_BYTES = 16
_process_key = None


def _key_file():
    configured = os.environ.get(KEY_FILE_VAR)
    return Path(configured) if configured else Path.home() / ".audit_env" / "secret.key"


def secret_key():
    value = os.environ.get(ENV_VAR)
    if value:
        encoded = value.encode("utf-8")
        if len(encoded) >= MIN_KEY_BYTES:
            return encoded
        print(f"audit_env: {ENV_VAR} has fewer than {MIN_KEY_BYTES} bytes, ignoring", file=sys.stderr)
    path = _key_file()
    try:
        # A short file is never a key: it is either being written by a parallel process
        # (wait for it) or damaged (fall back below rather than use a guessable key).
        for _ in range(20):
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                data = os.urandom(32).hex().encode("ascii")
                try:
                    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
                except FileExistsError:
                    continue
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data)
                return data
            data = path.read_bytes().strip()
            if len(data) >= MIN_KEY_BYTES:
                return data
            time.sleep(0.025)
        reason = f"key file {path} holds fewer than {MIN_KEY_BYTES} bytes"
    except OSError:
        reason = "no writable key file"
    global _process_key
    if _process_key is None:
        _process_key = os.urandom(32)
        print(f"audit_env: {reason}, using a per-process secret key; "
              f"set {ENV_VAR} for reproducible hidden episodes", file=sys.stderr)
    return _process_key


def resolve(keyspace):
    return OPEN_KEY if keyspace == "open" else secret_key()


def fingerprint(key):
    return hashlib.sha256(b"audit-env-key:" + key).hexdigest()[:16]


def digest(key, *parts):
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hmac.digest(key, message, "sha256")


def stream(key, *parts):
    """Independent deterministic random stream for one labelled purpose."""
    return Random(int.from_bytes(digest(key, "stream", *parts), "big"))


def opaque(key, kind, *parts):
    return f"{kind}_{digest(key, 'id', kind, *parts).hex()[:24]}"


def table_key(seed):
    """An evaluator key fixed by a printed --seed, so a published table is reproducible byte for byte.

    Only for reproducing benchmark tables; a real closed evaluation uses secret_key() or a fresh key.
    """
    return hashlib.sha256(f"red-team:{seed}".encode()).digest()
