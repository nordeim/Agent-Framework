"""Gmail OAuth helper: performs InstalledAppFlow and stores credentials to token.json

Usage:
  Place your `client_secrets.json` (from Google Cloud Console) in this folder or set
  the environment variable `GOOGLE_CLIENT_SECRETS_PATH` to its path.

  Then run:
    python gmail_oauth.py

This will open a browser, ask you to consent, and create `token.json` with refresh tokens.
"""
import os
import json
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CLIENT_SECRETS = os.getenv("GOOGLE_CLIENT_SECRETS_PATH", os.path.join(os.path.dirname(__file__), "client_secrets.json"))
TOKEN_PATH = os.getenv("TOKEN_JSON_PATH", os.path.join(os.path.dirname(__file__), "token.json"))


def load_credentials() -> Optional[Credentials]:
    """Load credentials from token.json if available and valid; otherwise return None."""
    token_file = Path(TOKEN_PATH)
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                save_credentials(creds)
                return creds
            except Exception:
                return None
    return None


def save_credentials(creds: Credentials):
    token_file = Path(TOKEN_PATH)
    token_file.write_text(creds.to_json())
    try:
        # Try to set secure permissions (POSIX only). On Windows, user should run icacls as documented.
        token_file.chmod(0o600)
    except Exception:
        pass


def run_local_oauth() -> Credentials:
    """Run InstalledAppFlow to obtain credentials and save them."""
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
    creds = flow.run_local_server(port=0)
    save_credentials(creds)
    return creds


def get_credentials() -> Credentials:
    creds = load_credentials()
    if creds:
        return creds
    return run_local_oauth()


if __name__ == "__main__":
    print("Starting OAuth flow. A browser will open for consent.")
    creds = get_credentials()
    print(f"Saved credentials to {TOKEN_PATH}")
