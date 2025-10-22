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
