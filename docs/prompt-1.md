I'll create a short todo plan and then open the GitHub page (via chrome-devtools-mcp) and fetch its content so I can inspect the `python` folder — outcome: gather repository context to inform the programming guide. I'll run two read-only tools in parallel: open the page and fetch its HTML content. 

Created 5 todos

Why/what/outcome: I'll open the GitHub repo page and fetch it so I can inspect the `python` directory contents and look for relevant patterns or helpers to reuse. I'll run two parallel read-only calls: one to open the page and one to fetch content from the same URL. 

Using "Fetch Web Page"

Fetched https://github.com/nordeim/ms-agent-framework/tree/main/python

Opened Simple Browser at https://github.com/nordeim/ms-agent-framework/tree/main/python

Progress update: I inspected the repo's `python` folder and README (agent-framework) to learn patterns for ChatAgent and chat clients. Next I'll mark the discovery task done and start designing the agent contract and guide. I'm updating the todo list to reflect that. 

Starting (2/5) *Design agent contract*

I'll mark the "Design agent contract" task complete and set "Create implementation guide" to in-progress. Expected outcome: todo list reflects current progress. 

Starting (3/5) *Create implementation guide*

Why/what/outcome: I'll add a programming guide and a runnable sample agent (plus requirements and a simple test) to the workspace so you can run and adapt it locally; expected outcome: new guide at `docs/email_agent_guide.md` and sample code in `samples/email_agent/` with README, script, requirements, and a test. 

Using "Apply Patch"

I'll add the guide and sample files to the workspace under absolute paths so you can run them locally. Expecting to create five new files: the guide, agent script, requirements, README, and a unit test. I'll apply the patch now. 

Using "Apply Patch"


```

```
Next I'll add the sample agent script, requirements, README, and a simple unit test under `samples/email_agent` (absolute paths). This will be one patch creating multiple files. 

Using "Apply Patch"


```

```
Now I'll add requirements, README for the sample, and a basic unit test. 

Using "Apply Patch"


```

```


Using "Apply Patch"


```

```


Using "Apply Patch"


```

```
Progress update: I added the guide (email_agent_guide.md), sample agent (agent.py), `requirements.txt`, README.md, and a unit test (test_summarize.py). Next I'll update the todo list marking the implementation guide and runnable sample tasks as completed, and move to deliver final notes. 

Starting (5/5) *Deliver final notes and next steps*

I'll deliver a concise programming guide and a runnable sample for a chat-triggered email-summary agent that uses OpenAI and IMAP (plus notes for Gmail/OAuth and Agent Framework integration). I inspected the `python` folder in the ms-agent-framework repo to align patterns (ChatAgent, OpenAIChatClient) and then added the guide and sample files to your workspace.

What I added (files and purpose)
- email_agent_guide.md — High-level programming guide: contract, data shapes, auth, security, prompting, and testing notes.
- agent.py — Runnable example (IMAP + OpenAI). Fetches up to 10 unread emails and summarizes them with OpenAI.
- requirements.txt — Minimal dependencies (openai, requests).
- README.md — Quick start (PowerShell commands).
- test_summarize.py — Basic unit test that verifies prompt building and exercises the summarization wrapper with a mocked OpenAI.

Where I placed them
- docs: email_agent_guide.md
- sample code: agent.py
- sample README & requirements: README.md and requirements.txt
- test: test_summarize.py

Short usage instructions (PowerShell)
- Create a virtual env and install requirements:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1; pip install -r .\samples\email_agent\requirements.txt
```

- Configure environment variables (PowerShell example):
```powershell
$env:EMAIL_HOST = "imap.gmail.com"
$env:EMAIL_USER = "you@example.com"
$env:EMAIL_PASSWORD = "app-password-or-token"
$env:OPENAI_API_KEY = "sk-..."
```

- Run the sample:
```powershell
python .\samples\email_agent\agent.py
```

Notes on design and choices
- Quick demo uses IMAP over TLS (IMAPS) to fetch unread messages. This is easy to run locally but for Google/Gmail production you should use Gmail API + OAuth2 with the `gmail.readonly` scope.
- The sample uses OpenAI ChatCompletions via the `openai` package if available. It tries to be compatible with both the older `openai` usage and a modern-style client. Replace `gpt-4o-mini` with your preferred model or configured `OPENAI_CHAT_MODEL_ID`.
- Security: do not log email bodies or secrets. Use environment variables or a secret manager. Prefer OAuth for Gmail and refresh tokens for long-running services.
- Privacy: The prompt instructs the model to redact PII and avoid printing full email text.

Integration with `agent-framework`
- If you use the Agent Framework present in the repo, you can create a `ChatAgent` and register an extractor tool (get_unread_emails) for the agent to call. The README in the repo demonstrates `ChatAgent` + `OpenAIChatClient` patterns already present in the `python` folder.

Edge cases & production recommendations
- No unread messages -> return a friendly message.
- Auth failure -> clear actionable error message for the user (re-auth, reset app password).
- Transient network failures -> retry with exponential backoff.
- Email-only attachments -> indicate non-text-only content and optionally summarize metadata only.
- Cache short-term results: avoid repeated LLM calls for the same unread set.

Testing & validation
- I added test_summarize.py which mocks a simple OpenAI response and checks prompt contents and wrapper behavior.
- For integration tests, set up a test mailbox with seeded unread messages and run the agent against it.

Quality gates (quick check)
- I added code files and a test. I haven't run the test or executed the script (I cannot run the local environment from here). To validate locally:
  - Install requirements (PowerShell steps above).
  - Run the test (if using pytest):
    ```powershell
    .venv\Scripts\Activate.ps1; pip install pytest
    pytest -q
    ```
  - Run the agent script with a real test mailbox configured.

Next steps I can take for you (pick one or more)
- Implement a Gmail API-based sample with OAuth flow (recommended for production).
- Integrate the sample into the `agent-framework` ChatAgent pattern and provide a fully working sample using `OpenAIChatClient` from that framework.
- Run tests locally and fix any environment-specific issues (if you allow me to run commands in your terminal).
- Add more unit tests (edge cases: no unread, attachment-only, large bodies) and continuous integration config.

If you want, I can:
- Add a Gmail API (OAuth) implementation and a short script to perform the OAuth dance and persist refresh tokens.
- Convert the sample to an `agent-framework` ChatAgent example and show how to register it as a tool for chat triggers.

Which of the next steps would you like me to implement now?

Made changes.
