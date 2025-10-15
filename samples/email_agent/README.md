# Email Agent (sample)

This folder contains a simple example that fetches unread emails via IMAP and summarizes them using OpenAI.

Quick start

1. Create a Python venv and install requirements:

   python -m venv .venv
   .venv\Scripts\Activate.ps1; pip install -r requirements.txt

2. Set environment variables (PowerShell example):

   $env:EMAIL_HOST = "imap.gmail.com"
   $env:EMAIL_USER = "you@example.com"
   $env:EMAIL_PASSWORD = "app-password-or-token"
   $env:OPENAI_API_KEY = "sk-..."

3. Run:

   python agent.py

Notes
- For Gmail, prefer OAuth and the Gmail API for production. This example uses IMAP for simplicity.

Gmail API (OAuth) and agent-framework

1) Create `client_secrets.json` as documented in `GOOGLE_CLOUD_SETUP.md`.

2) Run the OAuth dance to create `token.json`:

```powershell
python .\samples\email_agent\gmail_oauth.py
```

3) Use the agent-framework ChatAgent example (installs `agent-framework` from pip):

```powershell
python .\samples\email_agent\agent_framework_agent.py
```

If you prefer not to install `agent-framework`, continue to use `agent.py` which uses IMAP.
