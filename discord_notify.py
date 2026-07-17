#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord notify — post alerts to a Discord channel via an incoming webhook.

Create one in Discord: Server Settings -> Integrations -> Webhooks ->
New Webhook -> Copy Webhook URL, then set:

    export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

(or drop it in a local .env file — see envfile.py). No account credentials
involved; the webhook can be regenerated/revoked from the same settings page.
"""
import json
import os
import urllib.error
import urllib.request

from envfile import load_dotenv

DISCORD_CONTENT_LIMIT = 2000
_FENCE = "```\n"
_FENCE_END = "\n```"

load_dotenv()


def _chunks(text, limit):
    """Split text into <=limit-char pieces on line boundaries where possible."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > limit:
            if cur:
                parts.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        parts.append(cur)
    return parts


def send_discord(content, webhook_url=None):
    """POST content to a Discord incoming webhook, code-fenced for monospace
    formatting and split across messages if it exceeds Discord's 2000-char
    limit. Raises RuntimeError if no webhook URL is configured, or
    urllib.error.URLError/HTTPError on send failure."""
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError(
            "set DISCORD_WEBHOOK_URL to send Discord alerts "
            "(Server Settings -> Integrations -> Webhooks -> New Webhook)")
    budget = DISCORD_CONTENT_LIMIT - len(_FENCE) - len(_FENCE_END)
    for chunk in _chunks(content, budget):
        body = json.dumps({"content": _FENCE + chunk + _FENCE_END}).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST",
                                      headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=15).close()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise urllib.error.HTTPError(e.url, e.code, f"{e.reason}: {detail}", e.headers, e.fp)
