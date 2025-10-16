# Programming Guide — Email Summary Agent (Issues, Troubleshooting, Resolutions)

This document captures everything we encountered while building the sample email-summary agent (IMAP + Gmail API + Agent Framework integration). It is designed to help a new developer avoid common pitfalls when creating an AI Agent Application with the Microsoft Agent Framework.

Contents
- Goals & audience
- High-level architecture
- Prerequisites
- Step-by-step implementation summary
- Detailed issues encountered (symptoms, root cause, fix, test)
- Troubleshooting recipes (copyable commands and checks)
- Testing strategy and mocking patterns
- Security & production considerations
- Developer runbook (quick start checklist)
- Further reading and resources

Goals & audience
- Audience: Python developers new to the Microsoft Agent Framework and to integrating LLMs with external tools (Gmail API). Basic Python and virtualenv familiarity assumed.
- Goal: Provide a clear, practical guide describing the agent we built, record the real issues we hit, and give robust, copy-pasteable fixes so a new developer can reproduce and extend the sample with confidence.

High-level architecture
- Components:
  - Gmail OAuth helper (`gmail_oauth.py`) — performs InstalledAppFlow and persists refresh tokens.
  - Gmail fetcher (`gmail_fetcher.py`) — lists unread messages and fetches metadata/snippet.
  - IMAP sample (`agent.py`) — quick demo using IMAP (fallback).
  - Agent Framework sample (`agent_framework_agent.py`) — registers `get_unread_emails` as a tool and runs a ChatAgent.
  - Tests under `tests/` — unit tests that mock Gmail API and Agent Framework primitives.
- Data flow: Chat trigger -> ChatAgent -> Tool call (get_unread_emails) -> Gmail API -> formatted prompt -> LLM -> summary returned to user.

Prerequisites
- Python 3.10+ (sample used Python 3.12 in the local venv)
- An OpenAI API key in `OPENAI_API_KEY` for LLM calls (or configure agent-framework chat client)
- For Gmail API: Google Cloud Console project with OAuth credentials (`client_secrets.json`) and `token.json` will be created by the OAuth flow.
- Windows PowerShell familiarity (commands provided below).

Step-by-step implementation summary
1. Added IMAP-based quick sample `agent.py` to fetch unread messages and call OpenAI.
2. Implemented Gmail OAuth flow with `google-auth-oauthlib` (InstalledAppFlow) and `gmail_oauth.py` to persist tokens in `token.json`.
3. Implemented `gmail_fetcher.py` that uses `google-api-python-client` to list unread messages and fetch metadata/snippets.
4. Created `agent_framework_agent.py` to register the fetcher as a tool with `ChatAgent` and `OpenAIChatClient` (from `agent-framework`).
5. Wrote isolated unit tests that mock Gmail fetcher and the Agent Framework so tests run without network or pre-release dependency.

Detailed issues encountered, with root cause and resolution

Issue 1 — ImportError: attempted relative import with no known parent package
- Symptom: When importing modules directly (via `importlib.util.spec_from_file_location` or running the script), Python raised "attempted relative import with no known parent package".
- Root cause: Using relative imports (e.g. `from .gmail_oauth import get_credentials`) only works when the package is imported as a package (i.e., `python -m pkg.module` or when package root is on sys.path). Running a module as a script or loading it directly via spec loader doesn't set a package context.
- Fix applied:
  - Use absolute imports where possible (e.g., `from samples.email_agent.gmail_oauth import get_credentials`) wrapped in a try/except fallback that adds the local `samples/email_agent` path to `sys.path` and imports the module directly. Example pattern used:

```python
try:
    from samples.email_agent.gmail_oauth import get_credentials
except ImportError:
    # fallback for direct script/test runs
    import sys, os
    sample_dir = os.path.abspath(os.path.dirname(__file__))
    if sample_dir not in sys.path:
        sys.path.insert(0, sample_dir)
    import gmail_oauth
    get_credentials = gmail_oauth.get_credentials
```

  - This keeps imports robust when running as a script, from tests, and when installed as a package.

Issue 2 — ModuleNotFoundError: No module named 'samples' in tests
- Symptom: Tests using `importlib.import_module('samples.email_agent.agent_framework_agent')` failed with ModuleNotFoundError.
- Root cause: The repo root wasn't on `sys.path` as a package root during pytest collection, so `samples` wasn't recognized as a top-level package.
- Fix applied:
  - Switch tests to load the module by filename using `importlib.util.spec_from_file_location` and executing the module. This ensures tests don't rely on `sys.path` or package layout.
  - Alternatively you could configure pytest `pythonpath` or `conftest.py` to add repo root to `sys.path`, but the direct load approach isolates tests and avoids global changes.

Example test loader used:

```python
import importlib.util, os
agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','samples','email_agent','agent_framework_agent.py'))
spec = importlib.util.spec_from_file_location('agent_framework_agent', agent_path)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
```

Issue 3 — Tests initiating OAuth (FileNotFoundError: client_secrets.json)
- Symptom: A test unexpectedly invoked the real Gmail OAuth flow and failed because `client_secrets.json` was missing.
- Root cause: Tests indirectly called the real `gmail_fetcher.get_unread_messages`, which triggered `get_credentials()` and attempted to run InstalledAppFlow.
- Fix applied:
  - Ensure all tests that exercise agent logic mock `get_unread_messages` or monkeypatch `gmail_oauth.get_credentials` / `gmail_fetcher._get_service` to return a fake service. Tests now always mock external APIs and the OAuth flow.
  - Example monkeypatch in tests:

```python
monkeypatch.setattr(agent_mod, 'get_unread_messages', lambda max_results=10: [ ... ])
```

Issue 4 — Linter/IDE shows unresolved `agent-framework` import
- Symptom: The editor (and static checkers) showed "Import 'agent_framework' could not be resolved" even though the package can be installed via pip.
- Root cause: `agent-framework` is a pre-release package in this repo sample and may not be present in the developer environment.
- Fix / guidance:
  - Document in `README.md` to install `agent-framework` using `pip install --pre agent-framework` after activating venv.
  - In tests, avoid depending on the real package by injecting fake modules into `sys.modules` or use dependency injection to pass in fake `ChatAgent` and `OpenAIChatClient` instances.

Issue 5 — Tests that import package run different module contexts
- Symptom: Tests sometimes saw different module-level state when modules were loaded twice via different import mechanisms.
- Root cause: Loading the same module via `importlib.import_module('samples...')` and via `spec_from_file_location` creates different module objects and duplicate state.
- Fix / guidance:
  - Standardize on one approach in tests (we used the file-load approach) and avoid mixing import mechanisms for the same module.
  - When writing code, prefer idempotent module-level initialization and avoid side-effects at import time.

Issue 6 — Windows file permission guidance for `token.json`
- Symptom: Developer concerned `token.json` readable by other users.
- Guidance / Fix:
  - POSIX: use `chmod 600 token.json`.
  - Windows: provide PowerShell/icacls snippet to remove inheritance and grant current user exclusive access. Example (PowerShell):

```powershell
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
icacls .\token.json /inheritance:r
icacls .\token.json /grant:r "$user:(R,W)"
```

Troubleshooting recipes (copyable checks and fixes)

- If you see "attempted relative import with no known parent package":
  1. Check if the import uses a leading dot (e.g. `from .module import X`).
  2. Change to an absolute import; or add a try/except fallback that inserts local folder into `sys.path` before importing.

- If pytest can't find your sample package (`ModuleNotFoundError: No module named 'samples'`):
  1. Prefer `importlib.util.spec_from_file_location` in tests to load a module by path.
  2. Or add the repo root to `PYTHONPATH` in your test environment.

- If tests unexpectedly open a browser / invoke OAuth:
  1. Search tests for unmocked calls to `get_credentials` / `InstalledAppFlow`.
  2. Monkeypatch those functions to return a fake `Credentials` object or make `gmail_fetcher._get_service` return a fake service.

Testing strategy & mocking patterns
- Unit tests should never call external services. Use these patterns:
  - Replace `get_unread_messages` with a lambda that returns sample dictionaries.
  - For Gmail API shape tests, create a `DummyService` object that matches the subset of `googleapiclient` used (`users().messages().list(...).execute()` and `users().messages().get(...).execute()`).
  - For Agent Framework, create a small fake `ChatAgent` and `OpenAIChatClient` that implement the minimal API (e.g., `async run(self, message)`, `summarize(emails)`) and inject them via `sys.modules` or dependency injection.

Example test monkeypatch (pytest):

```python
import types, sys
sys.modules['agent_framework'] = types.SimpleNamespace(ChatAgent=FakeChatAgent)
sys.modules['agent_framework.openai'] = types.SimpleNamespace(OpenAIChatClient=FakeChatClient)
```

Security & production considerations
- Never commit `client_secrets.json` or `token.json` to source control. Add them to `.gitignore`.
- For production, use a secrets manager for OAuth credentials and refresh tokens (Azure Key Vault, AWS Secrets Manager, Google Secret Manager).
- Use least-privilege OAuth scopes (e.g. `https://www.googleapis.com/auth/gmail.readonly`).
- Consider running the agent's mailbox access in a separate background worker, and only expose a small, authenticated API to the chat front-end.

Developer runbook (quick start checklist)
1. Clone repo and open PowerShell.
2. Create and activate venv:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install requirements:

```powershell
pip install -r .\samples\email_agent\requirements.txt
pip install --pre agent-framework
```

4. Prepare Google OAuth credentials (optional for Gmail API):
  - Create OAuth 2.0 Client ID in Google Cloud Console (Desktop/Other).
  - Download `client_secrets.json` to `samples/email_agent` or set `GOOGLE_CLIENT_SECRETS_PATH`.

5. Run OAuth flow once interactively:

```powershell
python .\samples\email_agent\gmail_oauth.py
```

6. Run tests:

```powershell
pytest -q
```

7. Run agent-framework sample (after installing agent-framework and doing OAuth):

```powershell
python .\samples\email_agent\agent_framework_agent.py
```

Recommended next steps (for a new developer)
- Read the ms-agent-framework README and samples (the repo's `python` folder) to understand `ChatAgent` and `OpenAIChatClient` usage patterns.
- Extend the tool: add more granular extractors (thread detection, attachments) and a secondary model call to classify urgency/action items.
- Add CI to run tests on push and enforce secret scanning (prevent accidental commits of `client_secrets.json`).

Further reading & resources
- Microsoft Agent Framework (Python): https://github.com/microsoft/agent-framework (samples & docs)
- Google APIs Python client: https://developers.google.com/api-client-library/python
- Google OAuth guides: https://developers.google.com/identity/protocols/oauth2
- Testing/mocking patterns: pytest docs on monkeypatch and unittest.mock

---
If you'd like, I can now:
- Commit minor follow-ups: add a `conftest.py` to standardize sample path handling for tests, or
- Generate a short troubleshooting cheat-sheet card extracted from this guide suitable for developers onboarding.

I can now save this guide into `samples/email_agent/programming_guide_email_agent.md`. Let me know if you'd like adjustments or want me to add the `conftest.py` as a next step.
