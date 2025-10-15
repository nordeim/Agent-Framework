"""Simple chat-triggered email summary agent (IMAP + OpenAI)

Usage:
  Set environment variables:
    EMAIL_HOST, EMAIL_USER, EMAIL_PASSWORD (or use OAuth token)
    OPENAI_API_KEY

  python agent.py

This script connects to IMAP, fetches up to 10 unread messages, and sends
their snippets to OpenAI for a concise summary.
"""
import os
import imaplib
import email
from email.header import decode_header
import datetime
import time
import json
from typing import List, Dict, Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_HOST = os.getenv("EMAIL_HOST", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def connect_imap(host: str, user: str, password: str, timeout: int = 10):
    """Connect to IMAP server and return IMAP4_SSL instance."""
    imap = imaplib.IMAP4_SSL(host)
    imap.login(user, password)
    return imap


def fetch_unread(imap, limit: int = 10) -> List[Dict[str, Any]]:
    imap.select("INBOX")
    status, data = imap.search(None, "UNSEEN")
    if status != "OK":
        return []
    ids = data[0].split()
    ids = ids[-limit:]
    emails = []
    for eid in reversed(ids):
        status, msg_data = imap.fetch(eid, "(RFC822)")
        if status != "OK":
            continue
        msg = email.message_from_bytes(msg_data[0][1])
        subject, encoding = decode_header(msg.get("Subject", ""))[0]
        if isinstance(subject, bytes):
            try:
                subject = subject.decode(encoding or "utf-8", errors="ignore")
            except Exception:
                subject = subject.decode(errors="ignore")
        from_ = msg.get("From", "")
        date_ = msg.get("Date", "")
        snippet = extract_snippet(msg)
        emails.append({"id": eid.decode() if isinstance(eid, bytes) else str(eid), "subject": subject, "from": from_, "date": date_, "snippet": snippet})
    return emails


def extract_snippet(msg) -> str:
    # Prefer text/plain part
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore").strip()
                except Exception:
                    continue
    else:
        try:
            return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return ""


def build_prompt(emails: List[Dict[str, Any]]) -> str:
    system = (
        "You are an assistant that summarizes unread email messages. Redact PII and do not repeat full email bodies. "
        "Provide a one-sentence TL;DR, then 2-3 sentence summaries per email with the subject and urgency flag (urgent/normal)."
    )
    user = "Here are the unread emails:\n\n"
    for i, e in enumerate(emails, 1):
        user += f"Email {i}: Subject: {e['subject']}\nFrom: {e['from']}\nDate: {e['date']}\nSnippet: {e['snippet'][:1000]}\n\n"
    user += "\nSummary:" 
    return system + "\n\n" + user


def summarize_with_openai(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY is not set"}
    prompt = build_prompt(emails)
    # Use OpenAI chat completions via openai python package or modern client
    try:
        if OpenAI is None:
            import openai
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"system","content": "You are a concise summarizer."}, {"role":"user","content": prompt}], max_tokens=600)
            summary = resp["choices"][0]["message"]["content"].strip()
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are a concise summarizer."},{"role":"user","content":prompt}], max_tokens=600)
            summary = resp.choices[0].message.content
        return {"summary": summary, "emails": emails}
    except Exception as ex:
        return {"error": str(ex)}


def main():
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print("Set EMAIL_USER and EMAIL_PASSWORD (or provide OAuth). Exiting.")
        return
    try:
        imap = connect_imap(EMAIL_HOST, EMAIL_USER, EMAIL_PASSWORD)
    except Exception as ex:
        print("IMAP connection failed:", ex)
        return
    try:
        emails = fetch_unread(imap, limit=10)
        if not emails:
            print(json.dumps({"summary": "No unread messages found.", "emails": []}, indent=2))
            return
        result = summarize_with_openai(emails)
        print(json.dumps(result, indent=2))
    finally:
        try:
            imap.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
