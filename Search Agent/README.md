# Brave MCP Agent — Repo Scaffold

This scaffold provides a minimal, **Meticulous Approach**-driven Python project that uses the Agent Framework to create a chat agent integrated with your Smithery/Brave MCP server.

It contains:

* `README.md` — setup & run instructions
* `requirements.txt` — Python dependencies
* `.env.example` — environment variables
* `brave_mcp_agent.py` — main example agent (streaming REPL)
* `tests/test_tools_list.py` — minimal integration test that calls `tools/list` on the MCP endpoint
* `.gitignore` — basic ignores
* `run.sh` — convenience script for running the agent

---

Below are the files. Copy each file into your repo (same filenames).

---

```md
# README.md
```

````md
# Brave MCP Agent — Minimal Scaffold

This repository is a minimal scaffold to run a Python Agent (using the Agent Framework) that integrates a Smithery/Brave MCP server as a tool.

> This scaffold follows the user's requested **Meticulous Approach**: clear structure, validation, and a simple test that calls `tools/list` on the MCP.

## Prerequisites
- Python 3.10+ (3.11 recommended)
- A running Smithery MCP server (e.g. via Smithery CLI). Example dev endpoints from previous logs:
  - Local: `http://localhost:6969/mcp`
  - Remote/ngrok: `https://78643eb4.ngrok.smithery.ai/mcp`
- (Optional) An OpenAI API key or whatever chat LLM the Agent Framework requires if you run the ChatAgent. See `SMITHERY_MCP_URL`.

## Setup
1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

2. Copy `.env.example` to `.env` and edit values:

```bash
cp .env.example .env
# edit .env and set SMITHERY_MCP_URL, and optionally OPENAI_API_KEY
```

3. Run the quick tools-list test to validate the MCP endpoint:

```bash
python -m tests.test_tools_list
```

4. Run the interactive agent (streaming REPL):

```bash
./run.sh
```

## Files

* `brave_mcp_agent.py`: Agent example using the Agent Framework's `ChatAgent` and `MCPStreamableHTTPTool` (streaming). Edit `SMITHERY_MCP_URL` via env.
* `tests/test_tools_list.py`: Minimal test that posts a JSON-RPC `tools/list` to the MCP endpoint and prints the response.

## Notes & Next Steps

* This scaffold intentionally keeps the Agent example minimal to help you validate the MCP integration first.
* For production: add logging, secrets management, timeouts, retries, and automated CI tests.

---

````

```text
# requirements.txt
````

```text
# Core dependencies
aiohttp>=3.8.5
python-dotenv>=1.0.0

# Agent Framework (as used in the programming guide)
# The package name in the guide is 'agent-framework' (installable name may vary).
# If the package name differs in your environment, adjust accordingly.
agent-framework>=0.1.0

# Optional: OpenAI client if your agent uses OpenAIChatClient
openai>=0.27.0
```

```text
# .gitignore
```

```text
.venv/
__pycache__/
*.pyc
.env
```

```bash
# run.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

# Load environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
fi

python brave_mcp_agent.py
```

```python
# brave_mcp_agent.py
```

```python
"""brave_mcp_agent.py

Minimal streaming ChatAgent that integrates a local Smithery/Brave MCP server using
MCPStreamableHTTPTool as recommended by the Agent Framework Programming Guide.

Usage:
    - Set SMITHERY_MCP_URL in your environment (or .env)
    - Run: python brave_mcp_agent.py

This script provides a simple REPL that streams agent output as it arrives.
"""

import asyncio
import os
from dotenv import load_dotenv

# NOTE: import names below match the Agent Framework examples in the guide. If your
# installed package exposes different names, adjust imports accordingly.
try:
    from agent_framework import ChatAgent, MCPStreamableHTTPTool
    from agent_framework.openai import OpenAIChatClient
except Exception as e:
    raise ImportError(
        "Could not import Agent Framework. Ensure 'agent-framework' is installed and imports match your version."
    ) from e

load_dotenv()

SMITHERY_MCP_URL = os.environ.get("SMITHERY_MCP_URL", "http://localhost:6969/mcp")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


async def run_agent_with_local_mcp():
    instructions = (
        """
You are BraveSearch Assistant. Use the available MCP tools to perform web, image, video,
news, and local searches when the user asks for current or external information.
When using a tool, include short provenance (tool name and URL) in your answer.
        """
    )

    # Create client for the agent's chat model. If the OpenAIChatClient requires an API key
    # via env var, set it in OPENAI_API_KEY (or adjust according to your provider).
    chat_client = OpenAIChatClient()

    # Build MCP tool. This matches the Agent Framework's Local MCP example in the guide.
    mcp_tool = MCPStreamableHTTPTool(name="Brave MCP", url=SMITHERY_MCP_URL)

    async with ChatAgent(
        chat_client=chat_client,
        name="BraveSearchAgent",
        instructions=instructions,
        tools=mcp_tool,
    ) as agent:
        print(f"Agent ready — MCP URL: {SMITHERY_MCP_URL}\n")

        thread = agent.get_new_thread()

        # Simple REPL
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                break

            print("Agent:", end=" ", flush=True)
            try:
                async for chunk in agent.run_stream(query, thread=thread):
                    if getattr(chunk, "text", None):
                        print(chunk.text, end="", flush=True)
                print("\n" + "-" * 60 + "\n")
            except Exception as ex:
                print(f"\n[Error running agent]: {ex}\n")


if __name__ == "__main__":
    asyncio.run(run_agent_with_local_mcp())
```

```python
# tests/test_tools_list.py
```

```python
"""Simple test script that POSTs a JSON-RPC tools/list to the Smithery MCP endpoint
and prints the response. Run as a module:

    python -m tests.test_tools_list

This file intentionally does not depend on the Agent Framework package so it can
validate the MCP endpoint independently.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

SMITHERY_MCP_URL = os.environ.get("SMITHERY_MCP_URL", "http://localhost:6969/mcp")

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp (pip install aiohttp)")
    sys.exit(1)


async def call_tools_list(url: str):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except Exception:
                    print("Non-JSON response:\n", text)
                    return
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error calling tools/list on {url}: {e}")


if __name__ == "__main__":
    asyncio.run(call_tools_list(SMITHERY_MCP_URL))
```

---

## What I created

* A single, copyable scaffold (above) with files you can drop into a repo and run.
* The `brave_mcp_agent.py` follows the Agent Framework examples (ChatAgent + MCPStreamableHTTPTool) and uses streaming.
* The test script `tests/test_tools_list.py` posts a JSON-RPC `tools/list` to the MCP and prints the response.

If you'd like, I can:

* Convert the scaffold into a downloadable zip, or
* Create expanded unit tests (mocking MCP responses), or
* Replace the `OpenAIChatClient()` placeholder with an explicit configuration (e.g., provide instructions for using `OPENAI_API_KEY`) and a `requirements.txt` pinset.

Which would you like next?


https://chatgpt.com/share/68f82d3d-0918-8010-b619-762d46e2667a
