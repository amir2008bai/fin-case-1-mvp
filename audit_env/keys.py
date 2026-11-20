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
from pathlib import Path
from random import Random

OPEN_KEY = b"fin-case-1/open-episodes/v2"
ENV_VAR = "AUDIT_ENV_SECRET"
KEY_FILE_VAR = "AUDIT_ENV_KEY_FILE"
_process_key = None


def _key_file():
    configured = os.environ.get(KEY_FILE_VAR)
    return Path(configured) if configured else Path.home() / ".audit_env" / "secret.key"


def secret_key():
    value = os.environ.get(ENV_VAR)
    if value:
        return value.encode("utf-8")
    path = _key_file()
    try:
        if path.is_file():
            data = path.read_bytes().strip()
            if len(data) >= 16:
                return data
        path.parent.mkdir(parents=True, exist_ok=True)
        data = os.urandom(32).hex().encode("ascii")
        with open(path, "xb") as stream:
            stream.write(data)
        return data
    except FileExistsError:
        return path.read_bytes().strip()
    except OSError:
        global _process_key
        if _process_key is None:
            _process_key = os.urandom(32)
            print(f"audit_env: no writable key file, using a per-process secret key; "
                  f"set {ENV_VAR} for reproducible hidden episodes", file=sys.stderr)
        return _process_key


def resolve(keyspace):
    return OPEN_KEY if keyspace == "open" else secret_key()


def fingerprint(key):
    return hashlib.sha256(b"audit-env-key:" + key).hexdigest()[:16]


def digest(key, *parts):
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).digest()


def stream(key, *parts):
    """Independent deterministic random stream for one labelled purpose."""
    return Random(int.from_bytes(digest(key, "stream", *parts), "big"))


def opaque(key, kind, *parts):
    return f"{kind}_{digest(key, 'id', kind, *parts).hex()[:24]}"
