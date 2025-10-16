# agents/openai/README.md
```md
# OpenAI Agent Framework Examples

This folder contains examples demonstrating different ways to create and use agents with the OpenAI Assistants client from the `agent_framework.openai` package.

## Examples

| File | Description |
|------|-------------|
| [`openai_assistants_basic.py`](openai_assistants_basic.py) | The simplest way to create an agent using `ChatAgent` with `OpenAIAssistantsClient`. Shows both streaming and non-streaming responses with automatic assistant creation and cleanup. |
| [`openai_assistants_with_existing_assistant.py`](openai_assistants_with_existing_assistant.py) | Shows how to work with a pre-existing assistant by providing the assistant ID to the OpenAI Assistants client. Demonstrates proper cleanup of manually created assistants. |
| [`openai_assistants_with_explicit_settings.py`](openai_assistants_with_explicit_settings.py) | Shows how to initialize an agent with a specific assistants client, configuring settings explicitly including API key and model ID. |
| [`openai_assistants_with_function_tools.py`](openai_assistants_with_function_tools.py) | Demonstrates how to use function tools with agents. Shows both agent-level tools (defined when creating the agent) and query-level tools (provided with specific queries). |
| [`openai_assistants_with_code_interpreter.py`](openai_assistants_with_code_interpreter.py) | Shows how to use the HostedCodeInterpreterTool with OpenAI agents to write and execute Python code. Includes helper methods for accessing code interpreter data from response chunks. |
| [`openai_assistants_with_file_search.py`](openai_assistants_with_file_search.py) | Demonstrates how to use file search capabilities with OpenAI agents, allowing the agent to search through uploaded files to answer questions. |
| [`openai_assistants_with_thread.py`](openai_assistants_with_thread.py) | Demonstrates thread management with OpenAI agents, including automatic thread creation for stateless conversations and explicit thread management for maintaining conversation context across multiple interactions. |
| [`openai_chat_client_basic.py`](openai_chat_client_basic.py) | The simplest way to create an agent using `ChatAgent` with `OpenAIChatClient`. Shows both streaming and non-streaming responses for chat-based interactions with OpenAI models. |
| [`openai_chat_client_with_explicit_settings.py`](openai_chat_client_with_explicit_settings.py) | Shows how to initialize an agent with a specific chat client, configuring settings explicitly including API key and model ID. |
| [`openai_chat_client_with_function_tools.py`](openai_chat_client_with_function_tools.py) | Demonstrates how to use function tools with agents. Shows both agent-level tools (defined when creating the agent) and query-level tools (provided with specific queries). |
| [`openai_chat_client_with_local_mcp.py`](openai_chat_client_with_local_mcp.py) | Shows how to integrate OpenAI agents with local Model Context Protocol (MCP) servers for enhanced functionality and tool integration. |
| [`openai_chat_client_with_thread.py`](openai_chat_client_with_thread.py) | Demonstrates thread management with OpenAI agents, including automatic thread creation for stateless conversations and explicit thread management for maintaining conversation context across multiple interactions. |
| [`openai_chat_client_with_web_search.py`](openai_chat_client_with_web_search.py) | Shows how to use web search capabilities with OpenAI agents to retrieve and use information from the internet in responses. |
| [`openai_responses_client_basic.py`](openai_responses_client_basic.py) | The simplest way to create an agent using `ChatAgent` with `OpenAIResponsesClient`. Shows both streaming and non-streaming responses for structured response generation with OpenAI models. |
| [`openai_responses_client_image_analysis.py`](openai_responses_client_image_analysis.py) | Demonstrates how to use vision capabilities with agents to analyze images. |
| [`openai_responses_client_reasoning.py`](openai_responses_client_reasoning.py) | Demonstrates how to use reasoning capabilities with OpenAI agents, showing how the agent can provide detailed reasoning for its responses. |
| [`openai_responses_client_with_code_interpreter.py`](openai_responses_client_with_code_interpreter.py) | Shows how to use the HostedCodeInterpreterTool with OpenAI agents to write and execute Python code. Includes helper methods for accessing code interpreter data from response chunks. |
| [`openai_responses_client_with_explicit_settings.py`](openai_responses_client_with_explicit_settings.py) | Shows how to initialize an agent with a specific responses client, configuring settings explicitly including API key and model ID. |
| [`openai_responses_client_with_file_search.py`](openai_responses_client_with_file_search.py) | Demonstrates how to use file search capabilities with OpenAI agents, allowing the agent to search through uploaded files to answer questions. |
| [`openai_responses_client_with_function_tools.py`](openai_responses_client_with_function_tools.py) | Demonstrates how to use function tools with agents. Shows both agent-level tools (defined when creating the agent) and run-level tools (provided with specific queries). |
| [`openai_responses_client_with_hosted_mcp.py`](openai_responses_client_with_hosted_mcp.py) | Shows how to integrate OpenAI agents with hosted Model Context Protocol (MCP) servers, including approval workflows and tool management for remote MCP services. |
| [`openai_responses_client_with_local_mcp.py`](openai_responses_client_with_local_mcp.py) | Shows how to integrate OpenAI agents with local Model Context Protocol (MCP) servers for enhanced functionality and tool integration. |
| [`openai_responses_client_with_structured_output.py`](openai_responses_client_with_structured_output.py) | Demonstrates how to use structured outputs with OpenAI agents to get structured data responses in predefined formats. |
| [`openai_responses_client_with_thread.py`](openai_responses_client_with_thread.py) | Demonstrates thread management with OpenAI agents, including automatic thread creation for stateless conversations and explicit thread management for maintaining conversation context across multiple interactions. |
| [`openai_responses_client_with_web_search.py`](openai_responses_client_with_web_search.py) | Shows how to use web search capabilities with OpenAI agents to retrieve and use information from the internet in responses. |

## Environment Variables

Make sure to set the following environment variables before running the examples:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_CHAT_MODEL_ID`: The OpenAI model to use (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`)
- `OPENAI_RESPONSES_MODEL_ID`: The OpenAI model to use (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`)
- For image processing examples, use a vision-capable model like `gpt-4o` or `gpt-4o-mini`

Optionally, you can set:
- `OPENAI_ORG_ID`: Your OpenAI organization ID (if applicable)
- `OPENAI_API_BASE_URL`: Your OpenAI base URL (if using a different base URL)

## Optional Dependencies

Some examples require additional dependencies:

- **Image Generation Example**: The `openai_responses_client_image_generation.py` example requires PIL (Pillow) for image display. Install with:
  ```bash
  # Using uv
  uv add pillow

  # Or using pip
  pip install pillow
  ```

```

# agents/openai/openai_assistants_basic.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants Basic Example

This sample demonstrates basic usage of OpenAIAssistantsClient with automatic
assistant lifecycle management, showing both streaming and non-streaming responses.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def non_streaming_example() -> None:
    """Example of non-streaming response (get the complete result at once)."""
    print("=== Non-streaming Response Example ===")

    # Since no assistant ID is provided, the assistant will be automatically created
    # and deleted after getting a response
    async with OpenAIAssistantsClient().create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n")


async def streaming_example() -> None:
    """Example of streaming response (get results as they are generated)."""
    print("=== Streaming Response Example ===")

    # Since no assistant ID is provided, the assistant will be automatically created
    # and deleted after getting a response
    async with OpenAIAssistantsClient().create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        query = "What's the weather like in Portland?"
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n")


async def main() -> None:
    print("=== Basic OpenAI Assistants Chat Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_code_interpreter.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunResponseUpdate, ChatAgent, ChatResponseUpdate, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIAssistantsClient
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

"""
OpenAI Assistants with Code Interpreter Example

This sample demonstrates using HostedCodeInterpreterTool with OpenAI Assistants
for Python code execution and mathematical problem solving.
"""


def get_code_interpreter_chunk(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter data."""
    if (
        isinstance(chunk.raw_representation, ChatResponseUpdate)
        and isinstance(chunk.raw_representation.raw_representation, RunStepDeltaEvent)
        and isinstance(chunk.raw_representation.raw_representation.delta, RunStepDelta)
        and isinstance(chunk.raw_representation.raw_representation.delta.step_details, ToolCallDeltaObject)
        and chunk.raw_representation.raw_representation.delta.step_details.tool_calls
    ):
        for tool_call in chunk.raw_representation.raw_representation.delta.step_details.tool_calls:
            if (
                isinstance(tool_call, CodeInterpreterToolCallDelta)
                and isinstance(tool_call.code_interpreter, CodeInterpreter)
                and tool_call.code_interpreter.input is not None
            ):
                return tool_call.code_interpreter.input
    return None


async def main() -> None:
    """Example showing how to use the HostedCodeInterpreterTool with OpenAI Assistants."""
    print("=== OpenAI Assistants Agent with Code Interpreter Example ===")

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code to solve problems.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        query = "Use code to get the factorial of 100?"
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk

        print(f"\nGenerated code:\n{generated_code}")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_existing_assistant.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIAssistantsClient
from openai import AsyncOpenAI
from pydantic import Field

"""
OpenAI Assistants with Existing Assistant Example

This sample demonstrates working with pre-existing OpenAI Assistants
using existing assistant IDs rather than creating new ones.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    print("=== OpenAI Assistants Chat Client with Existing Assistant ===")

    # Create the client
    client = AsyncOpenAI()

    # Create an assistant that will persist
    created_assistant = await client.beta.assistants.create(
        model=os.environ["OPENAI_CHAT_MODEL_ID"], name="WeatherAssistant"
    )

    try:
        async with ChatAgent(
            chat_client=OpenAIAssistantsClient(async_client=client, assistant_id=created_assistant.id),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent:
            result = await agent.run("What's the weather like in Tokyo?")
            print(f"Result: {result}\n")
    finally:
        # Clean up the assistant manually
        await client.beta.assistants.delete(created_assistant.id)


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_explicit_settings.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants with Explicit Settings Example

This sample demonstrates creating OpenAI Assistants with explicit configuration
settings rather than relying on environment variable defaults.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    print("=== OpenAI Assistants Client with Explicit Settings ===")

    async with OpenAIAssistantsClient(
        model_id=os.environ["OPENAI_CHAT_MODEL_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    ).create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        result = await agent.run("What's the weather like in New York?")
        print(f"Result: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_file_search.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

"""
OpenAI Assistants with File Search Example

This sample demonstrates using HostedFileSearchTool with OpenAI Assistants
for document-based question answering and information retrieval.
"""

# Helper functions


async def create_vector_store(client: OpenAIAssistantsClient) -> tuple[str, HostedVectorStoreContent]:
    """Create a vector store with sample documents."""
    file = await client.client.files.create(
        file=("todays_weather.txt", b"The weather today is sunny with a high of 75F."), purpose="user_data"
    )
    vector_store = await client.client.vector_stores.create(
        name="knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    result = await client.client.vector_stores.files.create_and_poll(vector_store_id=vector_store.id, file_id=file.id)
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")

    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)


async def delete_vector_store(client: OpenAIAssistantsClient, file_id: str, vector_store_id: str) -> None:
    """Delete the vector store after using it."""

    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)


async def main() -> None:
    client = OpenAIAssistantsClient()
    async with ChatAgent(
        chat_client=client,
        instructions="You are a helpful assistant that searches files in a knowledge base.",
        tools=HostedFileSearchTool(),
    ) as agent:
        query = "What is the weather today? Do a file search to find the answer."
        file_id, vector_store = await create_vector_store(client)

        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(
            query, tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        await delete_vector_store(client, file_id, vector_store.vector_store_id)


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_function_tools.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants with Function Tools Example

This sample demonstrates function tool integration with OpenAI Assistants,
showing both agent-level and query-level tool configuration patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def get_time() -> str:
    """Get the current UTC time."""
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."


async def tools_on_agent_level() -> None:
    """Example showing tools defined when creating the agent."""
    print("=== Tools Defined on Agent Level ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    ) as agent:
        # First query - agent can use weather tool
        query1 = "What's the weather like in New York?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1}\n")

        # Second query - agent can use time tool
        query2 = "What's the current UTC time?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2}\n")

        # Third query - agent can use both tools if needed
        query3 = "What's the weather in London and what's the current UTC time?"
        print(f"User: {query3}")
        result3 = await agent.run(query3)
        print(f"Agent: {result3}\n")


async def tools_on_run_level() -> None:
    """Example showing tools passed to the run method."""
    print("=== Tools Passed to Run Method ===")

    # Agent created without tools
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant.",
        # No tools defined here
    ) as agent:
        # First query with weather tool
        query1 = "What's the weather like in Seattle?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, tools=[get_weather])  # Tool passed to run method
        print(f"Agent: {result1}\n")

        # Second query with time tool
        query2 = "What's the current UTC time?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=[get_time])  # Different tool for this query
        print(f"Agent: {result2}\n")

        # Third query with multiple tools
        query3 = "What's the weather in Chicago and what's the current UTC time?"
        print(f"User: {query3}")
        result3 = await agent.run(query3, tools=[get_weather, get_time])  # Multiple tools
        print(f"Agent: {result3}\n")


async def mixed_tools_example() -> None:
    """Example showing both agent-level tools and run-method tools."""
    print("=== Mixed Tools Example (Agent + Run Method) ===")

    # Agent created with some base tools
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a comprehensive assistant that can help with various information requests.",
        tools=[get_weather],  # Base tool available for all queries
    ) as agent:
        # Query using both agent tool and additional run-method tools
        query = "What's the weather in Denver and what's the current UTC time?"
        print(f"User: {query}")

        # Agent has access to get_weather (from creation) + additional tools from run method
        result = await agent.run(
            query,
            tools=[get_time],  # Additional tools for this specific query
        )
        print(f"Agent: {result}\n")


async def main() -> None:
    print("=== OpenAI Assistants Chat Client Agent with Function Tools Examples ===\n")

    await tools_on_agent_level()
    await tools_on_run_level()
    await mixed_tools_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_assistants_with_thread.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants with Thread Management Example

This sample demonstrates thread management with OpenAI Assistants, showing
persistent conversation threads and context preservation across interactions.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """Example showing automatic thread creation (service-managed thread)."""
    print("=== Automatic Thread Creation Example ===")

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # First conversation - no thread provided, will be created automatically
        query1 = "What's the weather like in Seattle?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1.text}")

        # Second conversation - still no thread provided, will create another new thread
        query2 = "What was the last city I asked about?"
        print(f"\nUser: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2.text}")
        print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence() -> None:
    """Example showing thread persistence across multiple conversations."""
    print("=== Thread Persistence Example ===")
    print("Using the same thread across multiple conversations to maintain context.\n")

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # Create a new thread that will be reused
        thread = agent.get_new_thread()

        # First conversation
        query1 = "What's the weather like in Tokyo?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, thread=thread)
        print(f"Agent: {result1.text}")

        # Second conversation using the same thread - maintains context
        query2 = "How about London?"
        print(f"\nUser: {query2}")
        result2 = await agent.run(query2, thread=thread)
        print(f"Agent: {result2.text}")

        # Third conversation - agent should remember both previous cities
        query3 = "Which of the cities I asked about has better weather?"
        print(f"\nUser: {query3}")
        result3 = await agent.run(query3, thread=thread)
        print(f"Agent: {result3.text}")
        print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """Example showing how to work with an existing thread ID from the service."""
    print("=== Existing Thread ID Example ===")
    print("Using a specific thread ID to continue an existing conversation.\n")

    # First, create a conversation and capture the thread ID
    existing_thread_id = None

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # Start a conversation and get the thread ID
        thread = agent.get_new_thread()
        query1 = "What's the weather in Paris?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, thread=thread)
        print(f"Agent: {result1.text}")

        # The thread ID is set after the first response
        existing_thread_id = thread.service_thread_id
        print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        # Create a new agent instance but use the existing thread ID
        async with ChatAgent(
            chat_client=OpenAIAssistantsClient(thread_id=existing_thread_id),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent:
            # Create a thread with the existing ID
            thread = AgentThread(service_thread_id=existing_thread_id)

            query2 = "What was the last city I asked about?"
            print(f"User: {query2}")
            result2 = await agent.run(query2, thread=thread)
            print(f"Agent: {result2.text}")
            print("Note: The agent continues the conversation from the previous thread.\n")


async def main() -> None:
    print("=== OpenAI Assistants Chat Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_basic.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient

"""
OpenAI Chat Client Basic Example

This sample demonstrates basic usage of OpenAIChatClient for direct chat-based
interactions, showing both streaming and non-streaming responses.
"""


def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def non_streaming_example() -> None:
    """Example of non-streaming response (get the complete result at once)."""
    print("=== Non-streaming Response Example ===")

    agent = OpenAIChatClient().create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Seattle?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Result: {result}\n")


async def streaming_example() -> None:
    """Example of streaming response (get results as they are generated)."""
    print("=== Streaming Response Example ===")

    agent = OpenAIChatClient().create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Portland?"
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


async def main() -> None:
    print("=== Basic OpenAI Chat Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_with_explicit_settings.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
OpenAI Chat Client with Explicit Settings Example

This sample demonstrates creating OpenAI Chat Client with explicit configuration
settings rather than relying on environment variable defaults.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    print("=== OpenAI Chat Client with Explicit Settings ===")

    agent = OpenAIChatClient(
        model_id=os.environ["OPENAI_CHAT_MODEL_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    ).create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    result = await agent.run("What's the weather like in New York?")
    print(f"Result: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_with_function_tools.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
OpenAI Chat Client with Function Tools Example

This sample demonstrates function tool integration with OpenAI Chat Client,
showing both agent-level and query-level tool configuration patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def get_time() -> str:
    """Get the current UTC time."""
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."


async def tools_on_agent_level() -> None:
    """Example showing tools defined when creating the agent."""
    print("=== Tools Defined on Agent Level ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    )

    # First query - agent can use weather tool
    query1 = "What's the weather like in New York?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}\n")

    # Second query - agent can use time tool
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}\n")

    # Third query - agent can use both tools if needed
    query3 = "What's the weather in London and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3)
    print(f"Agent: {result3}\n")


async def tools_on_run_level() -> None:
    """Example showing tools passed to the run method."""
    print("=== Tools Passed to Run Method ===")

    # Agent created without tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        # No tools defined here
    )

    # First query with weather tool
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, tools=[get_weather])  # Tool passed to run method
    print(f"Agent: {result1}\n")

    # Second query with time tool
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, tools=[get_time])  # Different tool for this query
    print(f"Agent: {result2}\n")

    # Third query with multiple tools
    query3 = "What's the weather in Chicago and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3, tools=[get_weather, get_time])  # Multiple tools
    print(f"Agent: {result3}\n")


async def mixed_tools_example() -> None:
    """Example showing both agent-level tools and run-method tools."""
    print("=== Mixed Tools Example (Agent + Run Method) ===")

    # Agent created with some base tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a comprehensive assistant that can help with various information requests.",
        tools=[get_weather],  # Base tool available for all queries
    )

    # Query using both agent tool and additional run-method tools
    query = "What's the weather in Denver and what's the current UTC time?"
    print(f"User: {query}")

    # Agent has access to get_weather (from creation) + additional tools from run method
    result = await agent.run(
        query,
        tools=[get_time],  # Additional tools for this specific query
    )
    print(f"Agent: {result}\n")


async def main() -> None:
    print("=== OpenAI Chat Client Agent with Function Tools Examples ===\n")

    await tools_on_agent_level()
    await tools_on_run_level()
    await mixed_tools_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_with_local_mcp.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

"""
OpenAI Chat Client with Local MCP Example

This sample demonstrates integrating Model Context Protocol (MCP) tools with
OpenAI Chat Client for extended functionality and external service access.
"""


async def mcp_tools_on_run_level() -> None:
    """Example showing MCP tools defined when running the agent."""
    print("=== Tools Defined on Run Level ===")

    # Tools are provided when running the agent
    # This means we have to ensure we connect to the MCP server before running the agent
    # and pass the tools to the run method.
    async with (
        MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ) as mcp_server,
        ChatAgent(
            chat_client=OpenAIChatClient(),
            name="DocsAgent",
            instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        ) as agent,
    ):
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, tools=mcp_server)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=mcp_server)
        print(f"{agent.name}: {result2}\n")


async def mcp_tools_on_agent_level() -> None:
    """Example showing tools defined when creating the agent."""
    print("=== Tools Defined on Agent Level ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    # The agent will connect to the MCP server through its context manager.
    async with OpenAIChatClient().create_agent(
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=MCPStreamableHTTPTool(  # Tools defined at agent creation
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ),
    ) as agent:
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"{agent.name}: {result2}\n")


async def main() -> None:
    print("=== OpenAI Chat Client Agent with MCP Tools Examples ===\n")

    await mcp_tools_on_agent_level()
    await mcp_tools_on_run_level()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_with_thread.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent, ChatMessageStore
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
OpenAI Chat Client with Thread Management Example

This sample demonstrates thread management with OpenAI Chat Client, showing
conversation threads and message history preservation across interactions.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """Example showing automatic thread creation (service-managed thread)."""
    print("=== Automatic Thread Creation Example ===")

    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # First conversation - no thread provided, will be created automatically
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1.text}")

    # Second conversation - still no thread provided, will create another new thread
    query2 = "What was the last city I asked about?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2.text}")
    print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence() -> None:
    """Example showing thread persistence across multiple conversations."""
    print("=== Thread Persistence Example ===")
    print("Using the same thread across multiple conversations to maintain context.\n")

    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Create a new thread that will be reused
    thread = agent.get_new_thread()

    # First conversation
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # Second conversation using the same thread - maintains context
    query2 = "How about London?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")

    # Third conversation - agent should remember both previous cities
    query3 = "Which of the cities I asked about has better weather?"
    print(f"\nUser: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")
    print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_messages() -> None:
    """Example showing how to work with existing thread messages for OpenAI."""
    print("=== Existing Thread Messages Example ===")

    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Start a conversation and build up message history
    thread = agent.get_new_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # The thread now contains the conversation history in memory
    if thread.message_store:
        messages = await thread.message_store.list_messages()
        print(f"Thread contains {len(messages or [])} messages")

    print("\n--- Continuing with the same thread in a new agent instance ---")

    # Create a new agent instance but use the existing thread with its message history
    new_agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Use the same thread object which contains the conversation history
    query2 = "What was the last city I asked about?"
    print(f"User: {query2}")
    result2 = await new_agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")
    print("Note: The agent continues the conversation using the local message history.\n")

    print("\n--- Alternative: Creating a new thread from existing messages ---")

    # You can also create a new thread from existing messages
    messages = await thread.message_store.list_messages() if thread.message_store else []

    new_thread = AgentThread(message_store=ChatMessageStore(messages))

    query3 = "How does the Paris weather compare to London?"
    print(f"User: {query3}")
    result3 = await new_agent.run(query3, thread=new_thread)
    print(f"Agent: {result3.text}")
    print("Note: This creates a new thread with the same conversation history.\n")


async def main() -> None:
    print("=== OpenAI Chat Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence()
    await example_with_existing_thread_messages()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_chat_client_with_web_search.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIChatClient

"""
OpenAI Chat Client with Web Search Example

This sample demonstrates using HostedWebSearchTool with OpenAI Chat Client
for real-time information retrieval and current data access.
"""


async def main() -> None:
    client = OpenAIChatClient(model_id="gpt-4o-search-preview")

    message = "What is the current weather? Do not ask for my current location."
    # Test that the client will use the web search tool with location
    additional_properties = {
        "user_location": {
            "country": "US",
            "city": "Seattle",
        }
    }
    stream = False
    print(f"User: {message}")
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(
            message,
            tools=[HostedWebSearchTool(additional_properties=additional_properties)],
            tool_choice="auto",
        ):
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        response = await client.get_response(
            message,
            tools=[HostedWebSearchTool(additional_properties=additional_properties)],
            tool_choice="auto",
        )
        print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_basic.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client Basic Example

This sample demonstrates basic usage of OpenAIResponsesClient for structured
response generation, showing both streaming and non-streaming responses.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def non_streaming_example() -> None:
    """Example of non-streaming response (get the complete result at once)."""
    print("=== Non-streaming Response Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Seattle?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Result: {result}\n")


async def streaming_example() -> None:
    """Example of streaming response (get results as they are generated)."""
    print("=== Streaming Response Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Portland?"
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


async def main() -> None:
    print("=== Basic OpenAI Responses Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_image_analysis.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client Image Analysis Example

This sample demonstrates using OpenAI Responses Client for image analysis and vision tasks,
showing multi-modal content handling with text and images.
"""


async def main():
    print("=== OpenAI Responses Agent with Image Analysis ===")

    # 1. Create an OpenAI Responses agent with vision capabilities
    agent = OpenAIResponsesClient().create_agent(
        name="VisionAgent",
        instructions="You are a helpful agent that can analyze images.",
    )

    # 2. Create a simple message with both text and image content
    user_message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="What do you see in this image?"),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                media_type="image/jpeg",
            ),
        ],
    )

    # 3. Get the agent's response
    print("User: What do you see in this image? [Image provided]")
    result = await agent.run(user_message)
    print(f"Agent: {result.text}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_image_generation.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import base64

from agent_framework import DataContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client Image Generation Example

This sample demonstrates how to generate images using OpenAI's DALL-E models
through the Responses Client. Image generation capabilities enable AI to create visual content from text,
making it ideal for creative applications, content creation, design prototyping,
and automated visual asset generation.
"""


def show_image_info(data_uri: str) -> None:
    """Display information about the generated image."""
    try:
        # Extract format and size info from data URI
        if data_uri.startswith("data:image/"):
            format_info = data_uri.split(";")[0].split("/")[1]
            base64_data = data_uri.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
            size_kb = len(image_bytes) / 1024

            print(" Image successfully generated!")
            print(f"   Format: {format_info.upper()}")
            print(f"   Size: {size_kb:.1f} KB")
            print(f"   Data URI length: {len(data_uri)} characters")
            print("")
            print(" To save and view the image:")
            print('   1. Install Pillow: "pip install pillow" or "uv add pillow"')
            print("   2. Use the data URI in your code to save/display the image")
            print("   3. Or copy the base64 data to an online base64 image decoder")
        else:
            print(f" Image URL generated: {data_uri}")
            print(" You can open this URL in a browser to view the image")

    except Exception as e:
        print(f" Error processing image data: {e}")
        print(" Image generated but couldn't parse details")


async def main() -> None:
    print("=== OpenAI Responses Image Generation Agent Example ===")

    # Create an agent with customized image generation options
    agent = OpenAIResponsesClient().create_agent(
        instructions="You are a helpful AI that can generate images.",
        tools=[
            {
                "type": "image_generation",
                # Core parameters
                "size": "1024x1024",
                "background": "transparent",
                "quality": "low",
                "format": "webp",
            }
        ],
    )

    query = "Generate a nice beach scenery with blue skies in summer time."
    print(f"User: {query}")
    print("Generating image with parameters: 1024x1024 size, transparent background, low quality, WebP format...")

    result = await agent.run(query)
    print(f"Agent: {result.text}")

    # Show information about the generated image
    for message in result.messages:
        for content in message.contents:
            if isinstance(content, (DataContent, UriContent)) and content.uri:
                show_image_info(content.uri)
                break


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_reasoning.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client Reasoning Example

This sample demonstrates advanced reasoning capabilities using OpenAI's gpt-5 models,
showing step-by-step reasoning process visualization and complex problem-solving.

This uses the additional_chat_options parameter to enable reasoning with high effort and detailed summaries.
You can also set these options at the run level, since they are api and/or provider specific, you will need to lookup
the correct values for your provider, since these are passed through as-is.

In this case they are here: https://platform.openai.com/docs/api-reference/responses/create#responses-create-reasoning
"""


agent = OpenAIResponsesClient(model_id="gpt-5").create_agent(
    name="MathHelper",
    instructions="You are a personal math tutor. When asked a math question, "
    "reason over how best to approach the problem and share your thought process.",
    additional_chat_options={"reasoning": {"effort": "high", "summary": "detailed"}},
)


async def reasoning_example() -> None:
    """Example of reasoning response (get results as they are generated)."""
    print("\033[92m=== Reasoning Example ===\033[0m")

    query = "I need to solve the equation 3x + 11 = 14 and I need to prove the pythagorean theorem. Can you help me?"
    print(f"User: {query}")
    print(f"{agent.name}: ", end="", flush=True)
    response = await agent.run(query)
    for msg in response.messages:
        if msg.contents:
            for content in msg.contents:
                if content.type == "text_reasoning":
                    print(f"\033[94m{content.text}\033[0m", end="", flush=True)
                elif content.type == "text":
                    print(content.text, end="", flush=True)
    print("\n")
    if response.usage_details:
        print(f"Usage: {response.usage_details}")


async def streaming_reasoning_example() -> None:
    """Example of reasoning response (get results as they are generated)."""
    print("\033[92m=== Streaming Reasoning Example ===\033[0m")

    query = "I need to solve the equation 3x + 11 = 14 and I need to prove the pythagorean theorem. Can you help me?"
    print(f"User: {query}")
    print(f"{agent.name}: ", end="", flush=True)
    usage = None
    async for chunk in agent.run_stream(query):
        if chunk.contents:
            for content in chunk.contents:
                if content.type == "text_reasoning":
                    print(f"\033[94m{content.text}\033[0m", end="", flush=True)
                elif content.type == "text":
                    print(content.text, end="", flush=True)
                elif content.type == "usage":
                    usage = content
    print("\n")
    if usage:
        print(f"Usage: {usage.details}")


async def main() -> None:
    print("\033[92m=== Basic OpenAI Responses Reasoning Agent Example ===\033[0m")

    await reasoning_example()
    await streaming_reasoning_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_code_interpreter.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, ChatResponse, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

"""
OpenAI Responses Client with Code Interpreter Example

This sample demonstrates using HostedCodeInterpreterTool with OpenAI Responses Client
for Python code execution and mathematical problem solving.
"""


async def main() -> None:
    """Example showing how to use the HostedCodeInterpreterTool with OpenAI Responses."""
    print("=== OpenAI Responses Agent with Code Interpreter Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant that can write and execute Python code to solve problems.",
        tools=HostedCodeInterpreterTool(),
    )

    query = "Use code to get the factorial of 100?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Result: {result}\n")

    if (
        isinstance(result.raw_representation, ChatResponse)
        and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
        and len(result.raw_representation.raw_representation.output) > 0
        and isinstance(result.raw_representation.raw_representation.output[0], ResponseCodeInterpreterToolCall)
    ):
        generated_code = result.raw_representation.raw_representation.output[0].code

        print(f"Generated code:\n{generated_code}")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_explicit_settings.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Explicit Settings Example

This sample demonstrates creating OpenAI Responses Client with explicit configuration
settings rather than relying on environment variable defaults.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    print("=== OpenAI Responses Client with Explicit Settings ===")

    agent = OpenAIResponsesClient(
        model_id=os.environ["OPENAI_RESPONSES_MODEL_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    ).create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    result = await agent.run("What's the weather like in New York?")
    print(f"Result: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_file_search.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client with File Search Example

This sample demonstrates using HostedFileSearchTool with OpenAI Responses Client
for direct document-based question answering and information retrieval.
"""

# Helper functions


async def create_vector_store(client: OpenAIResponsesClient) -> tuple[str, HostedVectorStoreContent]:
    """Create a vector store with sample documents."""
    file = await client.client.files.create(
        file=("todays_weather.txt", b"The weather today is sunny with a high of 75F."), purpose="user_data"
    )
    vector_store = await client.client.vector_stores.create(
        name="knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    result = await client.client.vector_stores.files.create_and_poll(vector_store_id=vector_store.id, file_id=file.id)
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")

    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)


async def delete_vector_store(client: OpenAIResponsesClient, file_id: str, vector_store_id: str) -> None:
    """Delete the vector store after using it."""

    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)


async def main() -> None:
    client = OpenAIResponsesClient()

    message = "What is the weather today? Do a file search to find the answer."

    stream = False
    print(f"User: {message}")
    file_id, vector_store = await create_vector_store(client)
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(
            message,
            tools=[HostedFileSearchTool(inputs=vector_store)],
            tool_choice="auto",
        ):
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        response = await client.get_response(
            message,
            tools=[HostedFileSearchTool(inputs=vector_store)],
            tool_choice="auto",
        )
        print(f"Assistant: {response}")
    await delete_vector_store(client, file_id, vector_store.vector_store_id)


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_function_tools.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Function Tools Example

This sample demonstrates function tool integration with OpenAI Responses Client,
showing both agent-level and query-level tool configuration patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def get_time() -> str:
    """Get the current UTC time."""
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."


async def tools_on_agent_level() -> None:
    """Example showing tools defined when creating the agent."""
    print("=== Tools Defined on Agent Level ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    )

    # First query - agent can use weather tool
    query1 = "What's the weather like in New York?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}\n")

    # Second query - agent can use time tool
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}\n")

    # Third query - agent can use both tools if needed
    query3 = "What's the weather in London and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3)
    print(f"Agent: {result3}\n")


async def tools_on_run_level() -> None:
    """Example showing tools passed to the run method."""
    print("=== Tools Passed to Run Method ===")

    # Agent created without tools
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant.",
        # No tools defined here
    )

    # First query with weather tool
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, tools=[get_weather])  # Tool passed to run method
    print(f"Agent: {result1}\n")

    # Second query with time tool
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, tools=[get_time])  # Different tool for this query
    print(f"Agent: {result2}\n")

    # Third query with multiple tools
    query3 = "What's the weather in Chicago and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3, tools=[get_weather, get_time])  # Multiple tools
    print(f"Agent: {result3}\n")


async def mixed_tools_example() -> None:
    """Example showing both agent-level tools and run-method tools."""
    print("=== Mixed Tools Example (Agent + Run Method) ===")

    # Agent created with some base tools
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a comprehensive assistant that can help with various information requests.",
        tools=[get_weather],  # Base tool available for all queries
    )

    # Query using both agent tool and additional run-method tools
    query = "What's the weather in Denver and what's the current UTC time?"
    print(f"User: {query}")

    # Agent has access to get_weather (from creation) + additional tools from run method
    result = await agent.run(
        query,
        tools=[get_time],  # Additional tools for this specific query
    )
    print(f"Agent: {result}\n")


async def main() -> None:
    print("=== OpenAI Responses Client Agent with Function Tools Examples ===\n")

    await tools_on_agent_level()
    await tools_on_run_level()
    await mixed_tools_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_hosted_mcp.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import TYPE_CHECKING, Any

from agent_framework import ChatAgent, HostedMCPTool
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client with Hosted MCP Example

This sample demonstrates integrating hosted Model Context Protocol (MCP) tools with
OpenAI Responses Client, including user approval workflows for function call security.
"""

if TYPE_CHECKING:
    from agent_framework import AgentProtocol, AgentThread


async def handle_approvals_without_thread(query: str, agent: "AgentProtocol"):
    """When we don't have a thread, we need to ensure we return with the input, approval request and approval."""
    from agent_framework import ChatMessage

    result = await agent.run(query)
    while len(result.user_input_requests) > 0:
        new_inputs: list[Any] = [query]
        for user_input_needed in result.user_input_requests:
            print(
                f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                f" with arguments: {user_input_needed.function_call.arguments}"
            )
            new_inputs.append(ChatMessage(role="assistant", contents=[user_input_needed]))
            user_approval = input("Approve function call? (y/n): ")
            new_inputs.append(
                ChatMessage(role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")])
            )

        result = await agent.run(new_inputs)
    return result


async def handle_approvals_with_thread(query: str, agent: "AgentProtocol", thread: "AgentThread"):
    """Here we let the thread deal with the previous responses, and we just rerun with the approval."""
    from agent_framework import ChatMessage

    result = await agent.run(query, thread=thread, store=True)
    while len(result.user_input_requests) > 0:
        new_input: list[Any] = []
        for user_input_needed in result.user_input_requests:
            print(
                f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                f" with arguments: {user_input_needed.function_call.arguments}"
            )
            user_approval = input("Approve function call? (y/n): ")
            new_input.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(user_approval.lower() == "y")],
                )
            )
        result = await agent.run(new_input, thread=thread, store=True)
    return result


async def handle_approvals_with_thread_streaming(query: str, agent: "AgentProtocol", thread: "AgentThread"):
    """Here we let the thread deal with the previous responses, and we just rerun with the approval."""
    from agent_framework import ChatMessage

    new_input: list[ChatMessage] = []
    new_input_added = True
    while new_input_added:
        new_input_added = False
        new_input.append(ChatMessage(role="user", text=query))
        async for update in agent.run_stream(new_input, thread=thread, store=True):
            if update.user_input_requests:
                for user_input_needed in update.user_input_requests:
                    print(
                        f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                        f" with arguments: {user_input_needed.function_call.arguments}"
                    )
                    user_approval = input("Approve function call? (y/n): ")
                    new_input.append(
                        ChatMessage(
                            role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                        )
                    )
                    new_input_added = True
            else:
                yield update


async def run_hosted_mcp_without_thread_and_specific_approval() -> None:
    """Example showing Mcp Tools with approvals without using a thread."""
    print("=== Mcp with approvals and without thread ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # we don't require approval for microsoft_docs_search tool calls
            # but we do for any other tool
            approval_mode={"never_require_approval": ["microsoft_docs_search"]},
        ),
    ) as agent:
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_without_thread(query1, agent)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_without_thread(query2, agent)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_without_approval() -> None:
    """Example showing Mcp Tools without approvals."""
    print("=== Mcp without approvals ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # we don't require approval for any function calls
            # this means we will not see the approval messages,
            # it is fully handled by the service and a final response is returned.
            approval_mode="never_require",
        ),
    ) as agent:
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_without_thread(query1, agent)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_without_thread(query2, agent)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_with_thread() -> None:
    """Example showing Mcp Tools with approvals using a thread."""
    print("=== Mcp with approvals and with thread ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # we require approval for all function calls
            approval_mode="always_require",
        ),
    ) as agent:
        # First query
        thread = agent.get_new_thread()
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_with_thread(query1, agent, thread)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_with_thread(query2, agent, thread)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_with_thread_streaming() -> None:
    """Example showing Mcp Tools with approvals using a thread."""
    print("=== Mcp with approvals and with thread ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # we require approval for all function calls
            approval_mode="always_require",
        ),
    ) as agent:
        # First query
        thread = agent.get_new_thread()
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        print(f"{agent.name}: ", end="")
        async for update in handle_approvals_with_thread_streaming(query1, agent, thread):
            print(update, end="")
        print("\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        print(f"{agent.name}: ", end="")
        async for update in handle_approvals_with_thread_streaming(query2, agent, thread):
            print(update, end="")
        print("\n")


async def main() -> None:
    print("=== OpenAI Responses Client Agent with Hosted Mcp Tools Examples ===\n")

    await run_hosted_mcp_without_approval()
    await run_hosted_mcp_without_thread_and_specific_approval()
    await run_hosted_mcp_with_thread()
    await run_hosted_mcp_with_thread_streaming()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_local_mcp.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client with Local MCP Example

This sample demonstrates integrating local Model Context Protocol (MCP) tools with
OpenAI Responses Client for direct response generation with external capabilities.
"""


async def streaming_with_mcp(show_raw_stream: bool = False) -> None:
    """Example showing tools defined when creating the agent.

    If you want to access the full stream of events that has come from the model, you can access it,
    through the raw_representation. You can view this, by setting the show_raw_stream parameter to True.
    """
    print("=== Tools Defined on Agent Level ===")
    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=MCPStreamableHTTPTool(  # Tools defined at agent creation
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ),
    ) as agent:
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        print(f"{agent.name}: ", end="")
        async for chunk in agent.run_stream(query1):
            if show_raw_stream:
                print("Streamed event: ", chunk.raw_representation.raw_representation)  # type:ignore
            elif chunk.text:
                print(chunk.text, end="")
        print("")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        print(f"{agent.name}: ", end="")
        async for chunk in agent.run_stream(query2):
            if show_raw_stream:
                print("Streamed event: ", chunk.raw_representation.raw_representation)  # type:ignore
            elif chunk.text:
                print(chunk.text, end="")
        print("\n\n")


async def run_with_mcp() -> None:
    """Example showing tools defined when creating the agent."""
    print("=== Tools Defined on Agent Level ===")

    # Tools are provided when creating the agent
    # The agent can use these tools for any query during its lifetime
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=MCPStreamableHTTPTool(  # Tools defined at agent creation
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ),
    ) as agent:
        # First query
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # Second query
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"{agent.name}: {result2}\n")


async def main() -> None:
    print("=== OpenAI Responses Client Agent with Function Tools Examples ===\n")

    await run_with_mcp()
    await streaming_with_mcp()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_structured_output.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunResponse
from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

"""
OpenAI Responses Client with Structured Output Example

This sample demonstrates using structured output capabilities with OpenAI Responses Client,
showing Pydantic model integration for type-safe response parsing and data extraction.
"""


class OutputStruct(BaseModel):
    """A structured output for testing purposes."""

    city: str
    description: str


async def non_streaming_example() -> None:
    print("=== Non-streaming example ===")

    # Create an OpenAI Responses agent
    agent = OpenAIResponsesClient().create_agent(
        name="CityAgent",
        instructions="You are a helpful agent that describes cities in a structured format.",
    )

    # Ask the agent about a city
    query = "Tell me about Paris, France"
    print(f"User: {query}")

    # Get structured response from the agent using response_format parameter
    result = await agent.run(query, response_format=OutputStruct)

    # Access the structured output directly from the response value
    if result.value:
        structured_data: OutputStruct = result.value  # type: ignore
        print("Structured Output Agent (from result.value):")
        print(f"City: {structured_data.city}")
        print(f"Description: {structured_data.description}")
    else:
        print("Error: No structured data found in result.value")


async def streaming_example() -> None:
    print("=== Streaming example ===")

    # Create an OpenAI Responses agent
    agent = OpenAIResponsesClient().create_agent(
        name="CityAgent",
        instructions="You are a helpful agent that describes cities in a structured format.",
    )

    # Ask the agent about a city
    query = "Tell me about Tokyo, Japan"
    print(f"User: {query}")

    # Get structured response from streaming agent using AgentRunResponse.from_agent_response_generator
    # This method collects all streaming updates and combines them into a single AgentRunResponse
    result = await AgentRunResponse.from_agent_response_generator(
        agent.run_stream(query, response_format=OutputStruct),
        output_format_type=OutputStruct,
    )

    # Access the structured output directly from the response value
    if result.value:
        structured_data: OutputStruct = result.value  # type: ignore
        print("Structured Output (from streaming with AgentRunResponse.from_agent_response_generator):")
        print(f"City: {structured_data.city}")
        print(f"Description: {structured_data.description}")
    else:
        print("Error: No structured data found in result.value")


async def main() -> None:
    print("=== OpenAI Responses Agent with Structured Output ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_thread.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Thread Management Example

This sample demonstrates thread management with OpenAI Responses Client, showing
persistent conversation context and simplified response handling.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """Example showing automatic thread creation."""
    print("=== Automatic Thread Creation Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # First conversation - no thread provided, will be created automatically
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1.text}")

    # Second conversation - still no thread provided, will create another new thread
    query2 = "What was the last city I asked about?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2.text}")
    print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence_in_memory() -> None:
    """
    Example showing thread persistence across multiple conversations.
    In this example, messages are stored in-memory.
    """
    print("=== Thread Persistence Example (In-Memory) ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Create a new thread that will be reused
    thread = agent.get_new_thread()

    # First conversation
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # Second conversation using the same thread - maintains context
    query2 = "How about London?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")

    # Third conversation - agent should remember both previous cities
    query3 = "Which of the cities I asked about has better weather?"
    print(f"\nUser: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")
    print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """
    Example showing how to work with an existing thread ID from the service.
    In this example, messages are stored on the server using OpenAI conversation state.
    """
    print("=== Existing Thread ID Example ===")

    # First, create a conversation and capture the thread ID
    existing_thread_id = None

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Start a conversation and get the thread ID
    thread = agent.get_new_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    # Enable OpenAI conversation state by setting `store` parameter to True
    result1 = await agent.run(query1, thread=thread, store=True)
    print(f"Agent: {result1.text}")

    # The thread ID is set after the first response
    existing_thread_id = thread.service_thread_id
    print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        agent = ChatAgent(
            chat_client=OpenAIResponsesClient(),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        )

        # Create a thread with the existing ID
        thread = AgentThread(service_thread_id=existing_thread_id)

        query2 = "What was the last city I asked about?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, thread=thread, store=True)
        print(f"Agent: {result2.text}")
        print("Note: The agent continues the conversation from the previous thread by using thread ID.\n")


async def main() -> None:
    print("=== OpenAI Response Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence_in_memory()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())

```

# agents/openai/openai_responses_client_with_web_search.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client with Web Search Example

This sample demonstrates using HostedWebSearchTool with OpenAI Responses Client
for direct real-time information retrieval and current data access.
"""


async def main() -> None:
    client = OpenAIResponsesClient()

    message = "What is the current weather? Do not ask for my current location."
    # Test that the client will use the web search tool with location
    additional_properties = {
        "user_location": {
            "country": "US",
            "city": "Seattle",
        }
    }
    stream = False
    print(f"User: {message}")
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(
            message,
            tools=[HostedWebSearchTool(additional_properties=additional_properties)],
            tool_choice="auto",
        ):
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        response = await client.get_response(
            message,
            tools=[HostedWebSearchTool(additional_properties=additional_properties)],
            tool_choice="auto",
        )
        print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

```

# chat_client/README.md
```md
# Chat Client Examples

This folder contains simple examples demonstrating direct usage of various chat clients.

## Examples

| File | Description |
|------|-------------|
| [`azure_assistants_client.py`](azure_assistants_client.py) | Direct usage of Azure Assistants Client for basic chat interactions with Azure OpenAI assistants. |
| [`azure_chat_client.py`](azure_chat_client.py) | Direct usage of Azure Chat Client for chat interactions with Azure OpenAI models. |
| [`azure_responses_client.py`](azure_responses_client.py) | Direct usage of Azure Responses Client for structured response generation with Azure OpenAI models. |
| [`chat_response_cancellation.py`](chat_response_cancellation.py) | Demonstrates how to cancel chat responses during streaming, showing proper cancellation handling and cleanup. |
| [`azure_ai_chat_client.py`](azure_ai_chat_client.py) | Direct usage of Azure AI Chat Client for chat interactions with Azure AI models. |
| [`openai_assistants_client.py`](openai_assistants_client.py) | Direct usage of OpenAI Assistants Client for basic chat interactions with OpenAI assistants. |
| [`openai_chat_client.py`](openai_chat_client.py) | Direct usage of OpenAI Chat Client for chat interactions with OpenAI models. |
| [`openai_responses_client.py`](openai_responses_client.py) | Direct usage of OpenAI Responses Client for structured response generation with OpenAI models. |

## Environment Variables

Depending on which client you're using, set the appropriate environment variables:

**For Azure clients:**
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`: The name of your Azure OpenAI chat deployment
- `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`: The name of your Azure OpenAI responses deployment

**For Azure AI client:**
- `AZURE_AI_PROJECT_ENDPOINT`: Your Azure AI project endpoint
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`: The name of your model deployment

**For OpenAI clients:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_CHAT_MODEL_ID`: The OpenAI model to use for chat clients (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`)
- `OPENAI_RESPONSES_MODEL_ID`: The OpenAI model to use for responses clients (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`)

```

# chat_client/openai_assistants_client.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants Client Direct Usage Example

Demonstrates direct OpenAIAssistantsClient usage for chat interactions with OpenAI assistants.
Shows function calling capabilities and automatic assistant creation.

"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    async with OpenAIAssistantsClient() as client:
        message = "What's the weather in Amsterdam and in Paris?"
        stream = False
        print(f"User: {message}")
        if stream:
            print("Assistant: ", end="")
            async for chunk in client.get_streaming_response(message, tools=get_weather):
                if str(chunk):
                    print(str(chunk), end="")
            print("")
        else:
            response = await client.get_response(message, tools=get_weather)
            print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

```

# chat_client/openai_chat_client.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
OpenAI Chat Client Direct Usage Example

Demonstrates direct OpenAIChatClient usage for chat interactions with OpenAI models.
Shows function calling capabilities with custom business logic.

"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    client = OpenAIChatClient()
    message = "What's the weather in Amsterdam and in Paris?"
    stream = True
    print(f"User: {message}")
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(message, tools=get_weather):
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        response = await client.get_response(message, tools=get_weather)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

```

# chat_client/openai_responses_client.py
```py
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client Direct Usage Example

Demonstrates direct OpenAIResponsesClient usage for structured response generation with OpenAI models.
Shows function calling capabilities with custom business logic.

"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    client = OpenAIResponsesClient()
    message = "What's the weather in Amsterdam and in Paris?"
    stream = False
    print(f"User: {message}")
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(message, tools=get_weather):
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        response = await client.get_response(message, tools=get_weather)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

```

