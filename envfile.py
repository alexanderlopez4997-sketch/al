#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny .env loader shared by mailer.py and discord_notify.py — avoids a
python-dotenv dependency just for local credential storage."""
import os


def load_dotenv(path=None):
    """Read KEY=VALUE lines from a .env file (default: next to this module)
    into os.environ, without overriding vars already set."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def save_dotenv_var(key, value, path=None):
    """Append KEY=VALUE to the .env file (default: next to this module), creating it
    owner-only (0600) if missing. Doesn't replace an existing line — callers only
    save keys they've just found unset. Raises OSError if the file isn't writable."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    prefix = ""
    if os.path.isfile(path):
        with open(path) as f:
            text = f.read()
        if text and not text.endswith("\n"):
            prefix = "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(f"{prefix}{key}={value}\n")
