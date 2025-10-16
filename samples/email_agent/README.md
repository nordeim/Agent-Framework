# Email Agent (sample)

This folder contains a simple example that fetches unread emails via IMAP and summarizes them using OpenAI.


# Quick Start

## 1. Create a Python venv and install requirements

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For agent-framework sample, install pre-release:
pip install --pre agent-framework
```

## 2. Gmail OAuth setup

Place your `client_secrets.json` (from Google Cloud Console) in this folder.
Run:
```powershell
python gmail_oauth.py
```
This will open a browser, ask for consent, and create `token.json`.

## 3. Run the IMAP sample

Set environment variables (PowerShell):
```powershell
$env:EMAIL_HOST = "imap.gmail.com"
$env:EMAIL_USER = "you@example.com"
$env:EMAIL_PASSWORD = "app-password-or-token"
$env:OPENAI_API_KEY = "sk-..."
python agent.py
```

## 4. Run the Gmail API sample

```powershell
python gmail_fetcher.py
```

## 5. Run the agent-framework ChatAgent sample

```powershell
python agent_framework_agent.py
```

## Security notes
- For Windows, restrict `token.json` to your user:
```powershell
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
icacls .\token.json /inheritance:r
icacls .\token.json /grant:r "$user:(R,W)"
```
- For production, use a secrets manager and OAuth web flow.


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
