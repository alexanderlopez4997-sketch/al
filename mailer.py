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
