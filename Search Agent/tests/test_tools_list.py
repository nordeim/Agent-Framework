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
