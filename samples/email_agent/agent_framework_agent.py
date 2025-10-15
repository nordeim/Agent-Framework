"""Example: integrate Gmail fetcher as a tool into agent-framework ChatAgent.

This file demonstrates how to register `get_unread_emails` as a tool and run a
ChatAgent that summarizes unread emails. It is written to be testable: the
async function `run_agent()` returns the result so tests can call it.
"""
import asyncio
from typing import List, Dict, Any

try:
    from samples.email_agent.gmail_fetcher import get_unread_messages
except Exception:
    # Allow importing when running from the samples directory
    from gmail_fetcher import get_unread_messages


def sanitize_email_for_prompt(email_item: Dict[str, Any]) -> Dict[str, Any]:
    # Redact long bodies and PII; only keep subject/from/date/snippet preview
    return {
        "id": email_item.get("id"),
        "subject": email_item.get("subject", "(no subject)"),
        "from": email_item.get("from", ""),
        "date": email_item.get("date", ""),
        "snippet": (email_item.get("snippet") or "")[:1000],
    }


def get_unread_emails_tool() -> List[Dict[str, Any]]:
    """Tool function that the agent will call to retrieve unread emails.

    Returns a list of email metadata dictionaries.
    """
    messages = get_unread_messages(max_results=10)
    return [sanitize_email_for_prompt(m) for m in messages]


async def run_agent() -> Dict[str, Any]:
    """Create a ChatAgent, register the tool, and run a summary task.

    Returns the agent result (should be a dict with `summary` and `emails`).
    """
    # Import agent-framework lazily so tests can inject a fake module before import
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient

    instructions = (
        "You are a helpful assistant that summarizes unread emails. Redact PII and do not repeat full email bodies. "
        "Provide a one-sentence TL;DR, then short per-email bullets with urgency flags."
    )

    agent = ChatAgent(chat_client=OpenAIChatClient(), instructions=instructions, tools=[get_unread_emails_tool])
    result = await agent.run("Summarize my unread emails")
    return result


def main():
    res = asyncio.run(run_agent())
    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
"""Example: integrate Gmail fetcher with agent-framework ChatAgent as a tool."""
import asyncio
from typing import List, Dict

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from .gmail_fetcher import get_unread_messages


def get_unread_emails_tool() -> List[Dict]:
    """Tool function that returns unread emails. Matches agent-framework tool signature."""
    return get_unread_messages(10)


async def main():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions=("You are an assistant that summarizes unread email messages. Redact PII and return a TL;DR and brief per-email lines."),
        tools=[get_unread_emails_tool],
    )

    resp = await agent.run("Summarize my unread emails")
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
