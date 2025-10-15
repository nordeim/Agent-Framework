"""Gmail API fetcher: list and retrieve unread messages using the Gmail REST API."""
from typing import List, Dict, Any
import os
from googleapiclient.discovery import build

try:
    from .gmail_oauth import get_credentials
except (ImportError, SystemError):
    try:
        from gmail_oauth import get_credentials
    except ImportError:
        import sys, importlib
        get_credentials = importlib.import_module('gmail_oauth').get_credentials


def _get_service():
    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    return service


def get_unread_messages(max_results: int = 10) -> List[Dict[str, Any]]:
    service = _get_service()
    results = service.users().messages().list(userId="me", q="is:unread", maxResults=max_results).execute()
    messages = results.get("messages", [])
    out = []
    for msg in messages:
        m = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        payload = m.get("payload", {})
        headers = payload.get("headers", [])
        subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(no subject)")
        from_ = next((h["value"] for h in headers if h["name"].lower() == "from"), "")
        date_ = next((h["value"] for h in headers if h["name"].lower() == "date"), "")
        snippet = m.get("snippet", "")
        out.append({"id": m.get("id"), "subject": subject, "from": from_, "date": date_, "snippet": snippet})
    return out


if __name__ == "__main__":
    msgs = get_unread_messages(5)
    import json
    print(json.dumps(msgs, indent=2))
