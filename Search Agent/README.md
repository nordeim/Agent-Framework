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

https://chatgpt.com/share/68f82d3d-0918-8010-b619-762d46e2667a
