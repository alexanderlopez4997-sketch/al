#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mailer — send plain-text/HTML email through Gmail's SMTP relay.

Requires a Gmail App Password (not your account password): generate one at
https://myaccount.google.com/apppasswords (needs 2-Step Verification enabled
on the account), then set:

    export GMAIL_ADDRESS="you@gmail.com"
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"

Recipient defaults to GMAIL_ADDRESS itself (mail yourself); override with
GMAIL_TO or the to_addr argument.
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def _load_dotenv():
    """Load KEY=VALUE lines from a .env file next to this module into
    os.environ (without overriding vars already set). Avoids needing a
    python-dotenv dependency just for local credential storage."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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


_load_dotenv()


def send_email(subject, text_body, html_body=None, to_addr=None):
    """Send an email via Gmail SMTP. Raises RuntimeError if credentials are
    missing, or smtplib.SMTPException on send failure."""
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not password:
        raise RuntimeError(
            "set GMAIL_ADDRESS and GMAIL_APP_PASSWORD to send email "
            "(generate an app password at https://myaccount.google.com/apppasswords)")
    recipient = to_addr or os.environ.get("GMAIL_TO") or sender

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(text_body, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.login(sender, password)
        server.sendmail(sender, [recipient], msg.as_string())
    return recipient
