Microsoft Agent Framework: Comprehensive Programming Guide
Table of Contents
Introduction
Core Concepts
Agent Type 1: Basic Conversational Agent
Agent Type 2: Function-Calling Agent
Agent Type 3: RAG Agent (Knowledge Retrieval)
Agent Type 4: Code Execution Agent
Agent Type 5: Multi-Modal Agent
Advanced Topics
Best Practices & Patterns
Troubleshooting Guide
Quick Reference
Glossary
1. Introduction
1.1 What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, production-ready toolkit for building AI agents that can interact with users, execute code, retrieve knowledge, call functions, and handle multi-modal content. The framework abstracts the complexity of working with different AI providers (OpenAI, Azure OpenAI, Azure AI) while providing a consistent, Pythonic interface for building sophisticated AI applications.

Key Benefits:

Unified Interface: Work with OpenAI, Azure OpenAI, and Azure AI using the same programming model
Production-Ready: Built-in support for streaming, error handling, resource management, and context managers
Flexible Tool Integration: Easily add custom functions, code execution, file search, web search, and MCP tools
Thread Management: Built-in conversation state management with both in-memory and service-based persistence
Type Safety: Full Pydantic integration for structured outputs and parameter validation
1.2 Architecture Overview
The framework is organized around three core abstractions:

text

┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                             │
│  (ChatAgent - High-level interface for agent interactions)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Chat Client Layer                          │
│  (OpenAIChatClient, OpenAIResponsesClient,                  │
│   OpenAIAssistantsClient, AzureAI, etc.)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Provider API Layer                          │
│  (OpenAI API, Azure OpenAI API, Azure AI API)               │
└─────────────────────────────────────────────────────────────┘
1.3 Client Types Comparison
The framework provides three primary client types, each optimized for different use cases:

Client Type	Best For	Key Features	Thread Support
Chat Client	Direct conversational interactions, maximum flexibility	In-memory message history, full control over conversation flow, local tools	In-memory
Responses Client	Structured outputs, multi-modal content, advanced features	Structured responses, reasoning, image generation/analysis, server-side thread storage	In-memory + Server-side
Assistants Client	Persistent assistants, server-managed state	Automatic assistant lifecycle, server-side threads, file storage	Server-side
1.4 Environment Setup
Installation:

Bash

# Using pip
pip install agent-framework

# Using uv (recommended)
uv add agent-framework
Environment Variables:

For OpenAI:

Bash

export OPENAI_API_KEY="your-api-key"
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"
For Azure OpenAI:

Bash

export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-chat-deployment"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="your-responses-deployment"
For Azure AI:

Bash

export AZURE_AI_PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="your-model-deployment"
1.5 Quick Start Example
Here's a minimal example to verify your setup:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create a simple chat agent
    agent = OpenAIChatClient().create_agent(
        name="HelloAgent",
        instructions="You are a helpful assistant."
    )
    
    # Get a response
    result = await agent.run("Hello! What can you help me with?")
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
This simple example demonstrates the framework's clean, intuitive API. You create a client, create an agent with instructions, and start interacting with a single run() call.

2. Core Concepts
2.1 Understanding Client Types
2.1.1 Chat Client - Maximum Flexibility
The Chat Client is ideal when you need full control over conversation flow and want to manage message history in your application:

Python

from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent

# Create agent with chat client
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant."
)
When to use:

You need to manage conversation state in your application
You want to implement custom message filtering or transformation
You're building stateless APIs where each request is independent
You need maximum control over the conversation flow
2.1.2 Responses Client - Advanced Features
The Responses Client provides access to cutting-edge features like structured outputs, reasoning, and multi-modal capabilities:

Python

from agent_framework.openai import OpenAIResponsesClient
from agent_framework import ChatAgent

# Create agent with responses client
agent = ChatAgent(
    chat_client=OpenAIResponsesClient(),
    instructions="You are a helpful assistant."
)
When to use:

You need structured outputs with Pydantic models
You want to leverage reasoning capabilities
You're working with images or multi-modal content
You need server-side thread persistence
2.1.3 Assistants Client - Managed Persistence
The Assistants Client leverages OpenAI's Assistants API for fully managed, persistent assistants:

Python

from agent_framework.openai import OpenAIAssistantsClient

# Create agent with assistants client (automatic lifecycle management)
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant."
) as agent:
    result = await agent.run("Hello!")
    print(result)
# Assistant automatically cleaned up when context exits
When to use:

You want OpenAI to manage assistant state and lifecycle
You need persistent assistants across application restarts
You want server-side file storage and retrieval
You prefer managed infrastructure over custom state management
2.2 Agent Lifecycle Management
The framework provides elegant lifecycle management using Python's context manager protocol:

Python

# Automatic cleanup with context manager
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant."
) as agent:
    result = await agent.run("What's the weather?")
    # Agent and associated resources automatically cleaned up

# Manual management (when needed)
agent = OpenAIChatClient().create_agent(
    name="MyAgent",
    instructions="You are a helpful assistant."
)
try:
    result = await agent.run("Hello!")
finally:
    # Manual cleanup if needed
    pass
Best Practice: Always use context managers (async with) when working with Assistants Client to ensure proper resource cleanup.

2.3 Thread Management Patterns
Threads represent conversation contexts and message history. The framework supports three thread management patterns:

2.3.1 Stateless (No Thread)
Each interaction is independent with no memory:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant."
)

# First query - no context
result1 = await agent.run("My name is Alice")

# Second query - agent doesn't remember Alice
result2 = await agent.run("What's my name?")
# Agent: "I don't have access to your name."
2.3.2 In-Memory Thread Persistence
Maintain conversation context in application memory:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant."
)

# Create a thread for this conversation
thread = agent.get_new_thread()

# First query with thread
result1 = await agent.run("My name is Alice", thread=thread)

# Second query with same thread - agent remembers
result2 = await agent.run("What's my name?", thread=thread)
# Agent: "Your name is Alice."
2.3.3 Server-Side Thread Persistence
Store conversation history on the server:

Python

agent = ChatAgent(
    chat_client=OpenAIResponsesClient(),
    instructions="You are a helpful assistant."
)

thread = agent.get_new_thread()

# Enable server-side storage with store=True
result1 = await agent.run("My name is Alice", thread=thread, store=True)

# Thread ID is now available
thread_id = thread.service_thread_id

# Later, reconstruct thread from ID
from agent_framework import AgentThread
restored_thread = AgentThread(service_thread_id=thread_id)
result2 = await agent.run("What's my name?", thread=restored_thread, store=True)
2.4 Tool Integration Approaches
The framework supports two primary patterns for adding tools to agents:

2.4.1 Agent-Level Tools
Tools defined at agent creation are available for all queries:

Python

def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=[get_weather]  # Available for all queries
)

result = await agent.run("What's the weather in Paris?")
2.4.2 Run-Level Tools
Tools provided per query for fine-grained control:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant."
    # No tools defined here
)

# Provide tool only for this specific query
result = await agent.run(
    "What's the weather in Paris?",
    tools=[get_weather]  # Tool only for this query
)
2.4.3 Mixed Approach
Combine base tools with query-specific tools:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=[get_weather]  # Base tool
)

# Add additional tools for specific query
result = await agent.run(
    "What's the weather and what time is it?",
    tools=[get_current_time]  # Additional tool
)
# Agent has access to both get_weather and get_current_time
2.5 Streaming vs Non-Streaming
2.5.1 Non-Streaming (Complete Response)
Get the entire response at once:

Python

result = await agent.run("Tell me a story")
print(result.text)  # Complete story
Use when:

You need the complete response before processing
You're implementing batch processing
You want to analyze the full response structure
2.5.2 Streaming (Real-Time Response)
Receive response chunks as they're generated:

Python

async for chunk in agent.run_stream("Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
Use when:

Building chat interfaces with real-time display
Implementing user-facing applications
You want to reduce perceived latency
Processing can start before response completion
2.6 Error Handling Patterns
The framework follows Python's exception handling conventions:

Python

from openai import APIError, RateLimitError
import asyncio

async def robust_agent_call(agent, query, max_retries=3):
    """Make agent calls with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            return result
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except APIError as e:
            print(f"API error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

# Usage
agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant."
)

try:
    result = await robust_agent_call(agent, "Hello!")
    print(result)
except Exception as e:
    print(f"Failed after retries: {e}")
3. Basic Conversational Agent
3.1 Use Case
Basic conversational agents are the foundation of AI applications. They handle straightforward question-answering, provide information, assist with writing tasks, and engage in general dialogue without needing external tools or data sources. This is the simplest agent type and perfect for:

Customer support chatbots answering FAQs
Writing assistants for content creation
Educational tutors for concept explanations
Personal assistants for scheduling and reminders
General-purpose chat interfaces
3.2 Architecture
A basic conversational agent consists of:

Chat Client: Handles communication with the AI provider
Instructions/System Prompt: Defines the agent's personality and behavior
Message History: Maintains conversation context (optional)
Response Handler: Processes and displays agent responses
text

User Input → Agent → Chat Client → AI Provider
                ↓
         Message History (optional)
                ↓
         Response → User
3.3 Complete Implementation
Here's a production-ready implementation of a basic conversational agent with both streaming and non-streaming support:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent

class BasicConversationalAgent:
    """A production-ready basic conversational agent with streaming support."""
    
    def __init__(self, name: str, instructions: str, model_id: str = "gpt-4o"):
        """
        Initialize the conversational agent.
        
        Args:
            name: Agent name for identification
            instructions: System prompt defining agent behavior
            model_id: OpenAI model to use
        """
        self.agent = ChatAgent(
            chat_client=OpenAIChatClient(model_id=model_id),
            name=name,
            instructions=instructions
        )
        self.thread = self.agent.get_new_thread()  # Maintain conversation context
    
    async def chat(self, message: str, stream: bool = False) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: User message
            stream: Whether to stream the response
            
        Returns:
            Agent's response text
        """
        if stream:
            return await self._chat_streaming(message)
        else:
            return await self._chat_non_streaming(message)
    
    async def _chat_non_streaming(self, message: str) -> str:
        """Get complete response at once."""
        result = await self.agent.run(message, thread=self.thread)
        return result.text
    
    async def _chat_streaming(self, message: str) -> str:
        """Stream response in real-time."""
        full_response = ""
        print(f"{self.agent.name}: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(message, thread=self.thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print()  # New line after streaming completes
        return full_response
    
    def reset_conversation(self):
        """Clear conversation history and start fresh."""
        self.thread = self.agent.get_new_thread()


async def main():
    """Demonstration of basic conversational agent."""
    print("=== Basic Conversational Agent Demo ===\n")
    
    # Create a friendly customer support agent
    agent = BasicConversationalAgent(
        name="SupportBot",
        instructions=(
            "You are a friendly and helpful customer support agent. "
            "Provide clear, concise answers and always maintain a positive tone. "
            "If you don't know something, admit it honestly and offer to help find the information."
        )
    )
    
    # Example conversation with non-streaming
    print("--- Non-Streaming Example ---")
    print("User: What are your business hours?")
    response1 = await agent.chat(
        "What are your business hours?",
        stream=False
    )
    print(f"{agent.agent.name}: {response1}\n")
    
    # Follow-up question (agent remembers context)
    print("User: Are you open on holidays?")
    response2 = await agent.chat(
        "Are you open on holidays?",
        stream=False
    )
    print(f"{agent.agent.name}: {response2}\n")
    
    # Example conversation with streaming
    print("\n--- Streaming Example ---")
    print("User: Can you help me understand your return policy?")
    await agent.chat(
        "Can you help me understand your return policy?",
        stream=True
    )
    
    # Reset and start new conversation
    print("\n--- After Reset ---")
    agent.reset_conversation()
    print("User: What was the last question I asked?")
    response3 = await agent.chat(
        "What was the last question I asked?",
        stream=False
    )
    print(f"{agent.agent.name}: {response3}")
    # Agent will not remember previous conversation after reset


if __name__ == "__main__":
    asyncio.run(main())
3.4 Configuration Options
Fine-tune your conversational agent with these configuration options:

Python

# Temperature control (creativity vs consistency)
agent = ChatAgent(
    chat_client=OpenAIChatClient(model_id="gpt-4o"),
    instructions="You are a helpful assistant.",
    additional_chat_options={
        "temperature": 0.7,  # 0.0 = deterministic, 2.0 = very creative
        "max_tokens": 500,   # Limit response length
        "top_p": 0.9,        # Nucleus sampling
    }
)

# Model selection based on use case
# gpt-4o: Best quality, most capable
# gpt-4o-mini: Fast and cost-effective
# gpt-3.5-turbo: Budget-friendly option

fast_agent = ChatAgent(
    chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    instructions="You are a quick response assistant."
)
3.5 Best Practices
1. Craft Clear Instructions:

Python

# ❌ Vague instructions
instructions = "You are helpful."

# ✅ Clear, specific instructions
instructions = """
You are a technical support specialist for a SaaS product.
- Always ask clarifying questions before providing solutions
- Provide step-by-step instructions with numbered lists
- Use simple language, avoiding jargon unless necessary
- End responses by asking if the user needs further help
"""
2. Manage Conversation Length:

Python

from agent_framework import ChatMessageStore

# Limit context window to last N messages
async def get_limited_thread(agent, max_messages=10):
    """Create thread with limited message history."""
    thread = agent.get_new_thread()
    
    # Get existing messages
    messages = await thread.message_store.list_messages()
    
    # Keep only recent messages
    if messages and len(messages) > max_messages:
        recent_messages = messages[-max_messages:]
        thread = agent.get_new_thread()
        thread.message_store = ChatMessageStore(recent_messages)
    
    return thread
3. Handle Edge Cases:

Python

async def safe_chat(agent, message: str, timeout: float = 30.0) -> str:
    """Chat with timeout and error handling."""
    import asyncio
    
    try:
        # Add timeout to prevent hanging
        result = await asyncio.wait_for(
            agent.run(message),
            timeout=timeout
        )
        return result.text
    except asyncio.TimeoutError:
        return "I'm taking longer than expected. Please try again."
    except Exception as e:
        print(f"Error: {e}")
        return "I encountered an error. Please try rephrasing your question."
3.6 Common Pitfalls
Pitfall 1: Forgetting Thread Management

Python

# ❌ No thread - agent forgets context
result1 = await agent.run("My name is Alice")
result2 = await agent.run("What's my name?")  # Agent doesn't remember

# ✅ Use thread for context
thread = agent.get_new_thread()
result1 = await agent.run("My name is Alice", thread=thread)
result2 = await agent.run("What's my name?", thread=thread)  # Remembers!
Pitfall 2: Unbounded Context Growth

Python

# ❌ Context grows indefinitely, causing errors
thread = agent.get_new_thread()
for i in range(1000):
    await agent.run(f"Message {i}", thread=thread)  # Eventually hits token limit

# ✅ Periodically trim or reset context
if len(await thread.message_store.list_messages()) > 50:
    thread = agent.get_new_thread()  # Start fresh
Pitfall 3: Ignoring Streaming Errors

Python

# ❌ No error handling in streaming
async for chunk in agent.run_stream(message):
    print(chunk.text, end="")

# ✅ Handle streaming errors
try:
    async for chunk in agent.run_stream(message):
        if chunk.text:
            print(chunk.text, end="", flush=True)
except Exception as e:
    print(f"\nError during streaming: {e}")
4. Function-Calling Agent
4.1 Use Case
Function-calling agents extend conversational capabilities by integrating with external systems, APIs, databases, and custom business logic. They can:

Query databases for real-time information
Call REST APIs to fetch data or trigger actions
Integrate with enterprise systems (CRM, ERP, etc.)
Perform calculations or data transformations
Execute business workflows
This makes them ideal for building AI assistants that can take actions, not just provide information.

4.2 Function Tool Patterns
The framework supports Python functions as tools with automatic schema generation from type hints:

Python

from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(description="The city name, e.g., 'Paris' or 'New York'")],
    units: Annotated[str, Field(description="Temperature units: 'celsius' or 'fahrenheit'")] = "celsius"
) -> str:
    """
    Get current weather for a location.
    
    The docstring becomes the function description shown to the AI model.
    """
    # Your implementation here
    return f"Weather in {location}: Sunny, 22°C"
Key Components:

Type Hints: Required for parameter types
Annotated: Provides rich parameter descriptions
Pydantic Field: Adds validation and constraints
Docstring: Describes function purpose to the AI
Return Type: Can be string, dict, or Pydantic model
4.3 Complete Implementation
Here's a production-ready function-calling agent with multiple tools:

Python

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Literal
from random import randint
from pydantic import Field
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def get_weather(
    location: Annotated[str, Field(description="City name, e.g., 'London' or 'Tokyo'")],
    units: Annotated[
        Literal["celsius", "fahrenheit"],
        Field(description="Temperature units")
    ] = "celsius"
) -> str:
    """
    Get current weather conditions for a specified location.
    
    Returns weather description and temperature.
    """
    conditions = ["sunny", "cloudy", "rainy", "stormy", "snowy"]
    temp = randint(10, 30) if units == "celsius" else randint(50, 86)
    unit_symbol = "°C" if units == "celsius" else "°F"
    
    return f"Weather in {location}: {conditions[randint(0, 4)]}, {temp}{unit_symbol}"


def get_current_time(
    timezone_name: Annotated[
        str,
        Field(description="Timezone name, e.g., 'UTC', 'America/New_York', 'Europe/London'")
    ] = "UTC"
) -> str:
    """
    Get current time in specified timezone.
    
    Returns formatted time string.
    """
    # Simplified implementation - in production, use pytz or zoneinfo
    current_time = datetime.now(timezone.utc)
    return f"Current time in {timezone_name}: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"


def calculate(
    expression: Annotated[str, Field(description="Mathematical expression to evaluate, e.g., '2 + 2' or '10 * 5'")]
) -> str:
    """
    Safely evaluate mathematical expressions.
    
    Supports basic arithmetic operations: +, -, *, /, **, ()
    """
    try:
        # Security note: In production, use a safe math parser, not eval()
        # This is simplified for demonstration
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def search_database(
    query: Annotated[str, Field(description="Search query for the product database")],
    category: Annotated[
        Literal["electronics", "clothing", "books", "all"],
        Field(description="Product category to search within")
    ] = "all"
) -> str:
    """
    Search product database for items matching query.
    
    Returns list of matching products with prices.
    """
    # Simulated database search
    products = {
        "electronics": ["Laptop - $999", "Phone - $699", "Tablet - $499"],
        "clothing": ["Jacket - $89", "Shoes - $129", "Shirt - $39"],
        "books": ["Python Guide - $29", "AI Handbook - $49", "Fiction Novel - $15"]
    }
    
    if category == "all":
        results = [item for items in products.values() for item in items]
    else:
        results = products.get(category, [])
    
    matching = [item for item in results if query.lower() in item.lower()]
    
    if matching:
        return f"Found {len(matching)} products:\n" + "\n".join(f"- {item}" for item in matching)
    else:
        return f"No products found matching '{query}' in category '{category}'"


# ============================================================================
# AGENT IMPLEMENTATION
# ============================================================================

class FunctionCallingAgent:
    """Production-ready function-calling agent with multiple tools."""
    
    def __init__(
        self,
        name: str = "AssistantAgent",
        instructions: str = "You are a helpful assistant with access to various tools.",
        agent_level_tools: list = None,
        model_id: str = "gpt-4o"
    ):
        """
        Initialize function-calling agent.
        
        Args:
            name: Agent identifier
            instructions: System prompt
            agent_level_tools: Tools available for all queries
            model_id: Model to use
        """
        self.agent = ChatAgent(
            chat_client=OpenAIChatClient(model_id=model_id),
            name=name,
            instructions=instructions,
            tools=agent_level_tools or []
        )
        self.thread = self.agent.get_new_thread()
    
    async def run(
        self,
        message: str,
        additional_tools: list = None,
        stream: bool = False
    ) -> str:
        """
        Execute query with optional additional tools.
        
        Args:
            message: User query
            additional_tools: Extra tools for this specific query
            stream: Whether to stream response
            
        Returns:
            Agent response text
        """
        if stream:
            return await self._run_streaming(message, additional_tools)
        else:
            return await self._run_non_streaming(message, additional_tools)
    
    async def _run_non_streaming(self, message: str, additional_tools: list = None) -> str:
        """Execute query and return complete response."""
        result = await self.agent.run(
            message,
            thread=self.thread,
            tools=additional_tools or []
        )
        return result.text
    
    async def _run_streaming(self, message: str, additional_tools: list = None) -> str:
        """Execute query with streaming response."""
        full_response = ""
        print(f"{self.agent.name}: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(
            message,
            thread=self.thread,
            tools=additional_tools or []
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print()
        return full_response


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

async def demo_agent_level_tools():
    """Demonstrate tools defined at agent creation."""
    print("=== Agent-Level Tools Demo ===\n")
    
    # Create agent with base tools available for all queries
    agent = FunctionCallingAgent(
        name="MultiToolAgent",
        instructions=(
            "You are a helpful assistant with access to weather, time, and calculation tools. "
            "Always use the appropriate tool to provide accurate information."
        ),
        agent_level_tools=[get_weather, get_current_time, calculate]
    )
    
    # Query 1: Uses weather tool
    print("User: What's the weather like in Paris?")
    response1 = await agent.run("What's the weather like in Paris?")
    print(f"Agent: {response1}\n")
    
    # Query 2: Uses time tool
    print("User: What time is it in New York?")
    response2 = await agent.run("What time is it in New York?")
    print(f"Agent: {response2}\n")
    
    # Query 3: Uses multiple tools
    print("User: What's the weather in London and what time is it there?")
    response3 = await agent.run(
        "What's the weather in London and what time is it there?",
        stream=True
    )


async def demo_run_level_tools():
    """Demonstrate tools provided per query."""
    print("\n=== Run-Level Tools Demo ===\n")
    
    # Create agent without predefined tools
    agent = FunctionCallingAgent(
        name="FlexibleAgent",
        instructions="You are a versatile assistant that can help with various tasks."
    )
    
    # Query 1: Provide weather tool only for this query
    print("User: What's the weather in Tokyo?")
    response1 = await agent.run(
        "What's the weather in Tokyo?",
        additional_tools=[get_weather]
    )
    print(f"Agent: {response1}\n")
    
    # Query 2: Provide different tool for this query
    print("User: Search for laptops in electronics")
    response2 = await agent.run(
        "Search for laptops in electronics",
        additional_tools=[search_database]
    )
    print(f"Agent: {response2}\n")


async def demo_mixed_tools():
    """Demonstrate combining agent-level and run-level tools."""
    print("\n=== Mixed Tools Demo ===\n")
    
    # Agent has base weather tool
    agent = FunctionCallingAgent(
        name="HybridAgent",
        instructions="You are a comprehensive assistant with various capabilities.",
        agent_level_tools=[get_weather]  # Base tool
    )
    
    # Add calculation tool for this specific query
    print("User: What's the weather in Seattle and what's 15% of 200?")
    response = await agent.run(
        "What's the weather in Seattle and what's 15% of 200?",
        additional_tools=[calculate],  # Additional tool for this query
        stream=True
    )


async def main():
    """Run all function-calling demonstrations."""
    print("=== Function-Calling Agent Examples ===\n")
    
    await demo_agent_level_tools()
    await demo_run_level_tools()
    await demo_mixed_tools()


if __name__ == "__main__":
    asyncio.run(main())
4.4 Parameter Handling and Validation
Use Pydantic for robust parameter validation:

Python

from pydantic import Field, validator
from typing import Annotated

def book_flight(
    origin: Annotated[str, Field(min_length=3, max_length=3, description="Origin airport code (e.g., 'JFK')")],
    destination: Annotated[str, Field(min_length=3, max_length=3, description="Destination airport code")],
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Travel date in YYYY-MM-DD format")],
    passengers: Annotated[int, Field(ge=1, le=9, description="Number of passengers (1-9)")] = 1
) -> str:
    """
    Book a flight with validated parameters.
    
    Pydantic automatically validates:
    - origin/destination are exactly 3 characters
    - date matches YYYY-MM-DD pattern
    - passengers is between 1 and 9
    """
    return f"Booking flight: {origin} → {destination} on {date} for {passengers} passenger(s)"

# The agent will receive clear error messages if validation fails
4.5 Best Practices for Tool Design
1. Clear, Descriptive Names:

Python

# ❌ Vague function name
def get_data(id: str) -> str:
    pass

# ✅ Descriptive function name
def get_customer_order_details(order_id: str) -> str:
    """Retrieve detailed information about a specific customer order."""
    pass
2. Comprehensive Docstrings:

Python

def send_email(
    recipient: Annotated[str, Field(description="Email address of the recipient")],
    subject: Annotated[str, Field(description="Email subject line")],
    body: Annotated[str, Field(description="Email message body")]
) -> str:
    """
    Send an email to a specified recipient.
    
    This function sends an email via the company SMTP server.
    
    Important: 
    - Emails are sent asynchronously and may take a few minutes to deliver
    - Maximum body length is 10,000 characters
    - HTML is not supported, use plain text only
    
    Returns:
        Success message with email ID or error description
    """
    # Implementation
    pass
3. Structured Error Handling:

Python

def query_api(endpoint: str) -> str:
    """Query external API with comprehensive error handling."""
    try:
        # API call logic
        response = make_api_call(endpoint)
        return f"Success: {response}"
    except TimeoutError:
        return "Error: API request timed out. Please try again."
    except ValueError as e:
        return f"Error: Invalid data received - {str(e)}"
    except Exception as e:
        # Log error for debugging
        print(f"Unexpected error: {e}")
        return "Error: An unexpected error occurred. Support has been notified."
4. Return Structured Data:

Python

from pydantic import BaseModel

class ProductInfo(BaseModel):
    """Structured product information."""
    name: str
    price: float
    in_stock: bool
    description: str

def get_product_info(product_id: str) -> str:
    """
    Get product information.
    
    Returns JSON string of ProductInfo for easy parsing.
    """
    product = ProductInfo(
        name="Laptop",
        price=999.99,
        in_stock=True,
        description="High-performance laptop"
    )
    return product.model_dump_json()
4.6 Common Pitfalls
Pitfall 1: Missing Type Hints

Python

# ❌ No type hints - agent doesn't understand parameters
def get_weather(location):
    return f"Weather in {location}"

# ✅ Proper type hints
def get_weather(
    location: Annotated[str, Field(description="City name")]
) -> str:
    return f"Weather in {location}"
Pitfall 2: Unsafe Code Execution

Python

# ❌ Dangerous - allows arbitrary code execution
def calculate(expression: str) -> str:
    return str(eval(expression))  # Security risk!

# ✅ Safe calculation with validation
def calculate(expression: str) -> str:
    import ast
    import operator
    
    # Whitelist of safe operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    
    try:
        node = ast.parse(expression, mode='eval').body
        # Validate and execute safely
        # (simplified - use a proper math parser in production)
        return str(eval(expression, {"__builtins__": {}}, {}))
    except:
        return "Error: Invalid expression"
Pitfall 3: Poor Error Messages

Python

# ❌ Unhelpful error message
def get_order(order_id: str) -> str:
    if not order_exists(order_id):
        return "Error"  # Agent doesn't know what went wrong

# ✅ Descriptive error messages
def get_order(order_id: str) -> str:
    if not order_exists(order_id):
        return f"Error: Order {order_id} not found. Please verify the order ID and try again."
5. RAG Agent (Knowledge Retrieval)
5.1 Use Case
Retrieval-Augmented Generation (RAG) agents combine AI language models with document search capabilities to provide accurate, source-based answers from your knowledge base. They excel at:

Customer support with documentation search
Internal knowledge base querying
Legal document analysis
Technical documentation assistance
Research and information retrieval
Compliance and policy question answering
RAG agents solve the hallucination problem by grounding responses in actual documents rather than relying solely on the model's training data.

5.2 Architecture
A RAG agent architecture involves:

text

User Query → Agent → Vector Search → Relevant Documents
                ↓                           ↓
         AI Model ← Context + Documents ←────┘
                ↓
         Response with Sources
Components:

Vector Store: Stores document embeddings for semantic search
File Search Tool: Retrieves relevant documents
Chat Client: Processes query with retrieved context
Response Generator: Produces answer grounded in documents
5.3 Complete Implementation
Here's a production-ready RAG agent with file upload, indexing, and retrieval:

Python

import asyncio
from pathlib import Path
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIResponsesClient


class RAGAgent:
    """
    Production-ready RAG agent with document management and retrieval.
    
    Supports file upload, vector indexing, and semantic search.
    """
    
    def __init__(
        self,
        name: str = "KnowledgeAgent",
        instructions: str = None,
        model_id: str = "gpt-4o"
    ):
        """
        Initialize RAG agent.
        
        Args:
            name: Agent identifier
            instructions: System prompt (defaults to RAG-optimized prompt)
            model_id: Model to use
        """
        self.client = OpenAIResponsesClient(model_id=model_id)
        
        default_instructions = """
        You are a knowledgeable assistant with access to a document database.
        
        When answering questions:
        1. Always search the knowledge base first using the file search tool
        2. Base your answers on the retrieved documents
        3. Cite specific information from the documents when possible
        4. If the information isn't in the knowledge base, clearly state that
        5. Don't make up information - stick to what's in the documents
        """
        
        self.agent = ChatAgent(
            chat_client=self.client,
            name=name,
            instructions=instructions or default_instructions,
            tools=[HostedFileSearchTool()]
        )
        
        self.vector_store_id = None
        self.file_ids = []
        self.thread = self.agent.get_new_thread()
    
    async def create_knowledge_base(
        self,
        documents: list[tuple[str, bytes]],
        knowledge_base_name: str = "knowledge_base"
    ) -> str:
        """
        Create vector store and upload documents.
        
        Args:
            documents: List of (filename, content) tuples
            knowledge_base_name: Name for the vector store
            
        Returns:
            Vector store ID
        """
        print(f"Creating knowledge base '{knowledge_base_name}'...")
        
        # Create vector store
        vector_store = await self.client.client.vector_stores.create(
            name=knowledge_base_name,
            expires_after={"anchor": "last_active_at", "days": 7}
        )
        self.vector_store_id = vector_store.id
        print(f"✓ Vector store created: {self.vector_store_id}")
        
        # Upload and index documents
        for filename, content in documents:
            print(f"  Uploading {filename}...")
            
            # Upload file
            file = await self.client.client.files.create(
                file=(filename, content),
                purpose="user_data"
            )
            self.file_ids.append(file.id)
            
            # Add to vector store and wait for processing
            result = await self.client.client.vector_stores.files.create_and_poll(
                vector_store_id=self.vector_store_id,
                file_id=file.id
            )
            
            if result.last_error:
                raise Exception(f"Failed to process {filename}: {result.last_error.message}")
            
            print(f"  ✓ {filename} indexed")
        
        print(f"✓ Knowledge base ready with {len(documents)} documents\n")
        return self.vector_store_id
    
    async def query(self, question: str, stream: bool = False) -> str:
        """
        Query the knowledge base.
        
        Args:
            question: User question
            stream: Whether to stream response
            
        Returns:
            Answer based on retrieved documents
        """
        if not self.vector_store_id:
            return "Error: No knowledge base created. Please create one first."
        
        # Configure tool to use our vector store
        tool_resources = {
            "file_search": {
                "vector_store_ids": [self.vector_store_id]
            }
        }
        
        if stream:
            return await self._query_streaming(question, tool_resources)
        else:
            return await self._query_non_streaming(question, tool_resources)
    
    async def _query_non_streaming(self, question: str, tool_resources: dict) -> str:
        """Execute query and return complete response."""
        result = await self.agent.run(
            question,
            thread=self.thread,
            tool_resources=tool_resources
        )
        return result.text
    
    async def _query_streaming(self, question: str, tool_resources: dict) -> str:
        """Execute query with streaming response."""
        full_response = ""
        print(f"{self.agent.name}: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(
            question,
            thread=self.thread,
            tool_resources=tool_resources
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print()
        return full_response
    
    async def cleanup(self):
        """Clean up vector store and uploaded files."""
        print("\nCleaning up resources...")
        
        if self.vector_store_id:
            await self.client.client.vector_stores.delete(self.vector_store_id)
            print(f"✓ Deleted vector store: {self.vector_store_id}")
        
        for file_id in self.file_ids:
            await self.client.client.files.delete(file_id)
        print(f"✓ Deleted {len(self.file_ids)} files")


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def main():
    """Demonstrate RAG agent with document-based Q&A."""
    print("=== RAG Agent Demo ===\n")
    
    # Create RAG agent
    agent = RAGAgent(
        name="DocAssistant",
        instructions=(
            "You are a helpful documentation assistant. "
            "Search the knowledge base to answer questions accurately. "
            "Always cite the source documents when providing information."
        )
    )
    
    # Prepare sample documents
    documents = [
        (
            "company_policy.txt",
            b"""
            Company Vacation Policy
            
            All full-time employees are entitled to:
            - 15 days of paid vacation per year
            - 10 sick days per year
            - 8 public holidays
            
            Vacation requests must be submitted at least 2 weeks in advance.
            Unused vacation days can be carried over to the next year, up to a maximum of 5 days.
            """
        ),
        (
            "product_info.txt",
            b"""
            Product Information - Premium Laptop
            
            Model: TechPro X1
            Price: $1,299
            Specifications:
            - Processor: Intel Core i7-12th Gen
            - RAM: 16GB DDR4
            - Storage: 512GB SSD
            - Display: 15.6" 4K OLED
            - Battery Life: Up to 12 hours
            
            Warranty: 2 years parts and labor
            Return Policy: 30-day money-back guarantee
            """
        ),
        (
            "support_faq.txt",
            b"""
            Frequently Asked Questions
            
            Q: How do I reset my password?
            A: Click "Forgot Password" on the login page and follow the email instructions.
            
            Q: What are your support hours?
            A: Monday-Friday, 9 AM - 6 PM EST. Emergency support available 24/7.
            
            Q: How long does shipping take?
            A: Standard shipping: 5-7 business days. Express shipping: 2-3 business days.
            
            Q: Do you ship internationally?
            A: Yes, we ship to over 50 countries. Additional customs fees may apply.
            """
        )
    ]
    
    try:
        # Create knowledge base
        await agent.create_knowledge_base(
            documents=documents,
            knowledge_base_name="company_docs"
        )
        
        # Example queries
        print("=== Query 1: Vacation Policy ===")
        print("User: How many vacation days do employees get?")
        answer1 = await agent.query(
            "How many vacation days do employees get?",
            stream=False
        )
        print(f"{agent.agent.name}: {answer1}\n")
        
        print("=== Query 2: Product Information ===")
        print("User: What are the specs of the TechPro X1 laptop?")
        answer2 = await agent.query(
            "What are the specs of the TechPro X1 laptop?",
            stream=True
        )
        
        print("\n=== Query 3: Support Information ===")
        print("User: What are your support hours and do you ship internationally?")
        answer3 = await agent.query(
            "What are your support hours and do you ship internationally?",
            stream=False
        )
        print(f"{agent.agent.name}: {answer3}\n")
        
        print("=== Query 4: Information Not in Knowledge Base ===")
        print("User: What's your office address?")
        answer4 = await agent.query(
            "What's your office address?",
            stream=False
        )
        print(f"{agent.agent.name}: {answer4}\n")
        
    finally:
        # Clean up resources
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
5.4 Query Optimization Techniques
1. Query Reformulation:

Python

async def optimized_query(agent: RAGAgent, user_question: str) -> str:
    """
    Optimize queries for better retrieval using query expansion.
    """
    # First, ask the agent to reformulate the query for better search
    reformulation_prompt = f"""
    Given this user question: "{user_question}"
    
    Generate 2-3 alternative phrasings or related questions that would help
    find relevant information in a knowledge base. Return only the questions,
    one per line.
    """
    
    # Get reformulated queries (you might use a separate agent for this)
    # Then search with all variations and combine results
    
    # For simplicity, here we just execute the original query
    return await agent.query(user_question)
2. Metadata Filtering:

Python

# When creating vector store, you can add metadata to filter searches
async def create_knowledge_base_with_metadata(self, documents_with_metadata):
    """
    Create knowledge base with document metadata for filtering.
    
    Args:
        documents_with_metadata: List of (filename, content, metadata) tuples
    """
    for filename, content, metadata in documents_with_metadata:
        file = await self.client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        
        # Add file with metadata to enable filtering
        await self.client.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file.id,
            chunking_strategy={"type": "auto"},
            # Metadata can be used for filtering in queries
            # metadata=metadata  # Feature availability depends on API version
        )
5.5 Best Practices for Knowledge Bases
1. Document Preparation:

Python

def prepare_document(raw_text: str, filename: str) -> bytes:
    """
    Prepare documents for optimal indexing.
    
    Best practices:
    - Clean formatting and remove excessive whitespace
    - Add clear section headers
    - Include metadata at the top
    - Keep paragraphs focused and coherent
    """
    # Add metadata header
    prepared = f"""
    Document: {filename}
    Last Updated: {datetime.now().strftime('%Y-%m-%d')}
    
    {raw_text.strip()}
    """
    
    return prepared.encode('utf-8')
2. Chunking Strategy:

Python

# For long documents, consider chunking for better retrieval
def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split document into overlapping chunks for better context preservation.
    
    Args:
        text: Document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks to maintain context
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # If we found a reasonable break point
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks
3. Vector Store Management:

Python

async def update_knowledge_base(agent: RAGAgent, new_documents: list):
    """
    Add new documents to existing knowledge base without recreating.
    """
    for filename, content in new_documents:
        # Upload file
        file = await agent.client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        
        # Add to existing vector store
        result = await agent.client.client.vector_stores.files.create_and_poll(
            vector_store_id=agent.vector_store_id,
            file_id=file.id
        )
        
        if result.status == "completed":
            print(f"✓ Added {filename} to knowledge base")
        else:
            print(f"✗ Failed to add {filename}: {result.last_error}")
5.6 Common Pitfalls
Pitfall 1: Not Waiting for Indexing

Python

# ❌ Query before indexing completes
await client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)
result = await agent.query("question")  # May not find document!

# ✅ Wait for indexing to complete
await client.vector_stores.files.create_and_poll(vector_store_id=vs_id, file_id=file_id)
result = await agent.query("question")  # Document is indexed and searchable
Pitfall 2: Poor Document Quality

Python

# ❌ Unstructured, unclear document
document = b"stuff about products and policies mixed together..."

# ✅ Well-structured, focused documents
product_doc = b"""
Product Information

Name: Widget Pro
Price: $99
Features:
- Feature 1
- Feature 2
"""

policy_doc = b"""
Return Policy

- 30-day returns
- Full refund for unopened items
"""
Pitfall 3: Forgetting to Clean Up

Python

# ❌ Resources left behind
agent = RAGAgent()
await agent.create_knowledge_base(docs)
# Exit without cleanup - vector store persists and incurs costs

# ✅ Always clean up
try:
    agent = RAGAgent()
    await agent.create_knowledge_base(docs)
    # Use agent...
finally:
    await agent.cleanup()  # Ensure cleanup happens
6. Code Execution Agent
6.1 Use Case
Code execution agents can write and run Python code dynamically to solve problems, perform calculations, analyze data, and generate visualizations. They're ideal for:

Data analysis and statistical computations
Mathematical problem solving
Chart and graph generation
File processing and transformations
Algorithm implementations
Scientific computing tasks
This capability transforms the agent from a text processor into a computational problem solver.

6.2 Architecture
The code execution flow:

text

User Request → Agent → Generates Code → HostedCodeInterpreterTool
                ↓                              ↓
         Analyzes Result ← Code Output ←───────┘
                ↓
         Final Response with Results
Security Note: The hosted code interpreter runs in a sandboxed environment, making it safe for untrusted code execution.

6.3 Complete Implementation
Here's a production-ready code execution agent:

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool, ChatResponse
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall


class CodeExecutionAgent:
    """
    Production-ready agent with Python code execution capabilities.
    
    Can write and execute Python code to solve computational problems,
    analyze data, and generate visualizations.
    """
    
    def __init__(
        self,
        name: str = "CodeAgent",
        instructions: str = None,
        model_id: str = "gpt-4o"
    ):
        """
        Initialize code execution agent.
        
        Args:
            name: Agent identifier
            instructions: System prompt (defaults to code-optimized prompt)
            model_id: Model to use
        """
        default_instructions = """
        You are an expert Python programmer and data analyst.
        
        When solving problems:
        1. Write clean, well-commented Python code
        2. Use appropriate libraries (numpy, pandas, matplotlib, etc.)
        3. Handle edge cases and validate inputs
        4. Provide clear explanations of your approach
        5. Show intermediate steps in calculations
        6. Format output for readability
        
        Always use the code interpreter tool for:
        - Mathematical calculations
        - Data analysis
        - File processing
        - Algorithm implementations
        """
        
        self.agent = ChatAgent(
            chat_client=OpenAIResponsesClient(model_id=model_id),
            name=name,
            instructions=instructions or default_instructions,
            tools=[HostedCodeInterpreterTool()]
        )
        self.thread = self.agent.get_new_thread()
    
    async def execute(self, task: str, stream: bool = False) -> tuple[str, str]:
        """
        Execute a computational task.
        
        Args:
            task: Description of the task to perform
            stream: Whether to stream the response
            
        Returns:
            Tuple of (response_text, generated_code)
        """
        if stream:
            return await self._execute_streaming(task)
        else:
            return await self._execute_non_streaming(task)
    
    async def _execute_non_streaming(self, task: str) -> tuple[str, str]:
        """Execute task and return complete response."""
        result = await self.agent.run(task, thread=self.thread)
        
        # Extract generated code from response
        code = self._extract_code(result)
        
        return result.text, code
    
    async def _execute_streaming(self, task: str) -> tuple[str, str]:
        """Execute task with streaming response."""
        full_response = ""
        code_parts = []
        
        print(f"{self.agent.name}: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(task, thread=self.thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            
            # Extract code from streaming chunks
            code_chunk = self._extract_code_from_chunk(chunk)
            if code_chunk:
                code_parts.append(code_chunk)
        
        print()
        return full_response, "".join(code_parts)
    
    def _extract_code(self, result) -> str:
        """Extract generated code from response."""
        if (
            isinstance(result.raw_representation, ChatResponse)
            and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
            and len(result.raw_representation.raw_representation.output) > 0
            and isinstance(result.raw_representation.raw_representation.output[0], ResponseCodeInterpreterToolCall)
        ):
            return result.raw_representation.raw_representation.output[0].code
        return ""
    
    def _extract_code_from_chunk(self, chunk) -> str:
        """Extract code from streaming chunk."""
        # Implementation for extracting code from streaming updates
        # This would inspect chunk.raw_representation for code interpreter data
        return ""


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

async def demo_mathematical_computation():
    """Demonstrate mathematical problem solving."""
    print("=== Mathematical Computation Demo ===\n")
    
    agent = CodeExecutionAgent(name="MathSolver")
    
    task = """
    Calculate the factorial of 100 and express it in scientific notation.
    Also calculate how many digits it has.
    """
    
    print(f"User: {task}")
    response, code = await agent.execute(task, stream=False)
    print(f"{agent.agent.name}: {response}\n")
    
    if code:
        print("Generated Code:")
        print("```python")
        print(code)
        print("```\n")


async def demo_data_analysis():
    """Demonstrate data analysis capabilities."""
    print("=== Data Analysis Demo ===\n")
    
    agent = CodeExecutionAgent(name="DataAnalyst")
    
    task = """
    Create a dataset of 100 random numbers from a normal distribution (mean=50, std=10).
    Calculate:
    1. Mean
    2. Median
    3. Standard deviation
    4. 95th percentile
    
    Present the results in a formatted table.
    """
    
    print(f"User: {task}")
    response, code = await agent.execute(task, stream=True)
    
    if code:
        print("\nGenerated Code:")
        print("```python")
        print(code)
        print("```\n")


async def demo_algorithm_implementation():
    """Demonstrate algorithm implementation."""
    print("=== Algorithm Implementation Demo ===\n")
    
    agent = CodeExecutionAgent(name="AlgoExpert")
    
    task = """
    Implement the Fibonacci sequence using dynamic programming.
    Calculate the 50th Fibonacci number.
    Show the execution time and compare it with a naive recursive approach
    (but don't actually run the recursive version for large n).
    """
    
    print(f"User: {task}")
    response, code = await agent.execute(task, stream=False)
    print(f"{agent.agent.name}: {response}\n")


async def demo_statistical_analysis():
    """Demonstrate statistical computations."""
    print("=== Statistical Analysis Demo ===\n")
    
    agent = CodeExecutionAgent(name="StatExpert")
    
    task = """
    Generate two datasets:
    - Dataset A: 1000 random numbers from normal distribution (mean=100, std=15)
    - Dataset B: 1000 random numbers from normal distribution (mean=105, std=15)
    
    Perform a t-test to determine if the means are significantly different.
    Report the t-statistic, p-value, and conclusion (at α=0.05).
    """
    
    print(f"User: {task}")
    response, code = await agent.execute(task, stream=True)


async def main():
    """Run all code execution demonstrations."""
    print("=== Code Execution Agent Examples ===\n")
    
    await demo_mathematical_computation()
    await demo_data_analysis()
    await demo_algorithm_implementation()
    await demo_statistical_analysis()


if __name__ == "__main__":
    asyncio.run(main())
6.4 Security Considerations
The hosted code interpreter provides a sandboxed environment with important security properties:

1. Isolation: Code runs in a secure, isolated environment separate from your application.

2. Limited File Access: The interpreter has restricted filesystem access.

3. Network Restrictions: Outbound network access is controlled.

4. Resource Limits: CPU time, memory, and execution time are limited.

However, you should still:

Python

# Best practice: Validate user inputs before code execution
async def safe_execute(agent: CodeExecutionAgent, user_task: str) -> str:
    """
    Execute code with input validation.
    """
    # Check for suspicious patterns
    suspicious_keywords = ["import os", "import subprocess", "eval(", "exec("]
    
    if any(keyword in user_task.lower() for keyword in suspicious_keywords):
        print("Warning: Task contains potentially unsafe operations")
        # You might want to add additional approval workflow here
    
    # Execute with timeout
    try:
        result, code = await asyncio.wait_for(
            agent.execute(user_task),
            timeout=60.0  # 60 second timeout
        )
        return result
    except asyncio.TimeoutError:
        return "Error: Code execution timed out"
6.5 Output Handling Best Practices
1. Structured Results:

Python

async def execute_with_structured_output(agent: CodeExecutionAgent, task: str):
    """
    Request structured output from code execution.
    """
    structured_task = f"""
    {task}
    
    Format your response as:
    1. Approach: Brief explanation of your method
    2. Code: The Python code you'll execute
    3. Results: The output and interpretation
    4. Conclusion: Summary of findings
    """
    
    return await agent.execute(structured_task)
2. Error Handling in Generated Code:

Instruct the agent to include error handling:

Python

agent = CodeExecutionAgent(
    instructions="""
    You are an expert Python programmer.
    
    Always include proper error handling in your code:
    - Use try/except blocks
    - Validate inputs
    - Provide informative error messages
    - Handle edge cases gracefully
    
    Example structure:
    ```python
    try:
        # Main logic here
        result = perform_calculation()
        print(f"Success: {result}")
    except ValueError as e:
        print(f"Input error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    ```
    """
)
6.6 Common Pitfalls
Pitfall 1: Vague Task Descriptions

Python

# ❌ Vague task
task = "Analyze some data"

# ✅ Specific task with clear requirements
task = """
Given a list of 100 random numbers between 1-1000:
1. Calculate mean, median, mode
2. Identify outliers (values > 2 standard deviations from mean)
3. Create a histogram showing the distribution
4. Report findings in a formatted table
"""
Pitfall 2: Not Handling Long-Running Operations

Python

# ❌ No timeout protection
result = await agent.execute("Calculate pi to 1 million digits")

# ✅ Use timeout for potentially long operations
try:
    result = await asyncio.wait_for(
        agent.execute("Calculate pi to 1 million digits"),
        timeout=30.0
    )
except asyncio.TimeoutError:
    print("Operation exceeded time limit")
Pitfall 3: Ignoring Code Extraction

Python

# ❌ Not capturing generated code
response, _ = await agent.execute(task)  # Lost the code!

# ✅ Capture and log code for debugging/auditing
response, code = await agent.execute(task)
if code:
    # Log for audit trail
    print(f"Generated code:\n{code}")
    # Could save to database or file for review
7. Multi-Modal Agent
7.1 Use Case
Multi-modal agents can process and generate multiple types of content beyond text: images, reasoning traces, web search results, and integration with external tools via MCP. They excel at:

Image analysis and description
Visual content generation
Real-time information retrieval
Complex reasoning visualization
Multi-step problem solving with explanations
Integration with enterprise systems
These agents represent the cutting edge of AI capabilities, combining vision, reasoning, and action.

7.2 Capabilities Overview
Capability	Purpose	Use Cases
Image Analysis	Understand visual content	Product identification, document analysis, accessibility
Web Search	Access current information	News, weather, real-time data, fact-checking
Reasoning	Show thought process	Education, debugging, transparency, complex problem-solving
MCP Integration	Connect to external systems	Enterprise tools, databases, custom APIs
Multi-modal Combination	Use multiple capabilities together	Comprehensive analysis, research, automation
7.3 Complete Implementation
Here's a comprehensive multi-modal agent showcasing all capabilities:

Python

import asyncio
from agent_framework import (
    ChatAgent,
    ChatMessage,
    TextContent,
    UriContent,
    HostedWebSearchTool,
    MCPStreamableHTTPTool
)
from agent_framework.openai import OpenAIResponsesClient


class MultiModalAgent:
    """
    Production-ready multi-modal agent with vision, web search, reasoning, and MCP integration.
    
    Supports:
    - Image analysis
    - Web search for current information
    - Step-by-step reasoning
    - External tool integration via MCP
    """
    
    def __init__(
        self,
        name: str = "MultiModalAgent",
        instructions: str = None,
        enable_reasoning: bool = False,
        model_id: str = "gpt-4o"
    ):
        """
        Initialize multi-modal agent.
        
        Args:
            name: Agent identifier
            instructions: System prompt
            enable_reasoning: Whether to enable reasoning mode
            model_id: Model to use (must support vision for image analysis)
        """
        default_instructions = """
        You are a versatile AI assistant with multiple capabilities:
        
        1. Image Analysis: Describe images in detail, identify objects, read text, analyze scenes
        2. Web Search: Find current information from the internet
        3. Problem Solving: Break down complex problems step-by-step
        4. Tool Integration: Use external tools when needed
        
        Always:
        - Use the appropriate capability for each task
        - Provide detailed, accurate information
        - Cite sources when using web search
        - Explain your reasoning when solving complex problems
        """
        
        additional_options = {}
        if enable_reasoning:
            additional_options["reasoning"] = {
                "effort": "high",
                "summary": "detailed"
            }
        
        self.client = OpenAIResponsesClient(model_id=model_id)
        self.agent = ChatAgent(
            chat_client=self.client,
            name=name,
            instructions=instructions or default_instructions,
            additional_chat_options=additional_options
        )
        self.thread = self.agent.get_new_thread()
    
    async def analyze_image(
        self,
        image_url: str,
        question: str = "What do you see in this image?",
        stream: bool = False
    ) -> str:
        """
        Analyze an image and answer questions about it.
        
        Args:
            image_url: URL of the image to analyze
            question: Question about the image
            stream: Whether to stream response
            
        Returns:
            Analysis of the image
        """
        # Create message with both text and image
        message = ChatMessage(
            role="user",
            contents=[
                TextContent(text=question),
                UriContent(uri=image_url, media_type="image/jpeg")
            ]
        )
        
        if stream:
            return await self._run_streaming(message)
        else:
            result = await self.agent.run(message, thread=self.thread)
            return result.text
    
    async def search_web(
        self,
        query: str,
        location: dict = None,
        stream: bool = False
    ) -> str:
        """
        Search the web for current information.
        
        Args:
            query: Search query
            location: Optional location dict with 'country' and 'city' keys
            stream: Whether to stream response
            
        Returns:
            Answer based on web search results
        """
        additional_properties = {}
        if location:
            additional_properties["user_location"] = location
        
        web_tool = HostedWebSearchTool(additional_properties=additional_properties)
        
        if stream:
            return await self._run_streaming(query, tools=[web_tool])
        else:
            result = await self.agent.run(query, thread=self.thread, tools=[web_tool])
            return result.text
    
    async def solve_with_reasoning(
        self,
        problem: str,
        stream: bool = True
    ) -> str:
        """
        Solve a problem with step-by-step reasoning visible.
        
        Args:
            problem: Problem to solve
            stream: Whether to stream response (recommended for reasoning)
            
        Returns:
            Solution with reasoning steps
        """
        if stream:
            full_response = ""
            print(f"{self.agent.name}: ", end="", flush=True)
            
            async for chunk in self.agent.run_stream(problem, thread=self.thread):
                if chunk.contents:
                    for content in chunk.contents:
                        if content.type == "text_reasoning":
                            # Highlight reasoning in different color
                            print(f"\033[94m{content.text}\033[0m", end="", flush=True)
                            full_response += f"[REASONING: {content.text}]"
                        elif content.type == "text":
                            print(content.text, end="", flush=True)
                            full_response += content.text
            
            print()
            return full_response
        else:
            result = await self.agent.run(problem, thread=self.thread)
            return result.text
    
    async def use_mcp_tool(
        self,
        query: str,
        mcp_tool: MCPStreamableHTTPTool,
        stream: bool = False
    ) -> str:
        """
        Execute query using MCP tool integration.
        
        Args:
            query: User query
            mcp_tool: MCP tool to use
            stream: Whether to stream response
            
        Returns:
            Response using MCP tool
        """
        # MCP tools need to be connected
        async with mcp_tool:
            if stream:
                return await self._run_streaming(query, tools=[mcp_tool])
            else:
                result = await self.agent.run(query, thread=self.thread, tools=[mcp_tool])
                return result.text
    
    async def _run_streaming(self, message, tools=None) -> str:
        """Execute query with streaming."""
        full_response = ""
        print(f"{self.agent.name}: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(message, thread=self.thread, tools=tools):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print()
        return full_response


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

async def demo_image_analysis():
    """Demonstrate image analysis capabilities."""
    print("=== Image Analysis Demo ===\n")
    
    agent = MultiModalAgent(name="VisionAgent")
    
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    # Analyze image with specific question
    print("User: What do you see in this image? Describe the scene in detail.")
    analysis = await agent.analyze_image(
        image_url=image_url,
        question="What do you see in this image? Describe the scene in detail including colors, objects, and atmosphere.",
        stream=False
    )
    print(f"{agent.agent.name}: {analysis}\n")


async def demo_web_search():
    """Demonstrate web search capabilities."""
    print("=== Web Search Demo ===\n")
    
    agent = MultiModalAgent(name="SearchAgent")
    
    # Search with location context
    print("User: What's the current weather? [Location: Seattle, US]")
    weather = await agent.search_web(
        query="What's the current weather? Do not ask for my location.",
        location={"country": "US", "city": "Seattle"},
        stream=True
    )
    
    print("\nUser: What are the latest developments in AI?")
    news = await agent.search_web(
        query="What are the latest developments in AI and machine learning?",
        stream=False
    )
    print(f"{agent.agent.name}: {news}\n")


async def demo_reasoning():
    """Demonstrate reasoning capabilities."""
    print("=== Reasoning Demo ===\n")
    
    # Create agent with reasoning enabled
    agent = MultiModalAgent(
        name="ReasoningAgent",
        enable_reasoning=True,
        model_id="gpt-5"  # Reasoning requires gpt-5 model
    )
    
    problem = """
    I need to solve this problem:
    
    A train leaves Station A at 2:00 PM traveling at 60 mph.
    Another train leaves Station B at 3:00 PM traveling at 80 mph toward Station A.
    The stations are 300 miles apart.
    
    At what time will the trains meet?
    """
    
    print(f"User: {problem}")
    solution = await agent.solve_with_reasoning(problem, stream=True)


async def demo_mcp_integration():
    """Demonstrate MCP tool integration."""
    print("\n=== MCP Integration Demo ===\n")
    
    agent = MultiModalAgent(name="DocAgent")
    
    # Create MCP tool for Microsoft Learn
    mcp_tool = MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp"
    )
    
    print("User: How do I create an Azure storage account using Azure CLI?")
    response = await agent.use_mcp_tool(
        query="How do I create an Azure storage account using Azure CLI?",
        mcp_tool=mcp_tool,
        stream=True
    )


async def demo_combined_capabilities():
    """Demonstrate using multiple capabilities together."""
    print("\n=== Combined Capabilities Demo ===\n")
    
    agent = MultiModalAgent(name="ComprehensiveAgent")
    
    # First, analyze an image
    image_url = "https://example.com/product.jpg"
    print("User: [Uploads product image] What product is this?")
    # Simulate: analysis = await agent.analyze_image(image_url)
    print(f"{agent.agent.name}: [Analyzes image]\n")
    
    # Then search web for current price
    print("User: What's the current market price for this product?")
    # Simulate: price = await agent.search_web("current price for [product]")
    print(f"{agent.agent.name}: [Searches web for pricing]\n")
    
    print("Demonstration: This shows how an agent can combine vision and web search")
    print("to identify a product from an image and find current pricing information.\n")


async def main():
    """Run all multi-modal demonstrations."""
    print("=== Multi-Modal Agent Examples ===\n")
    
    await demo_image_analysis()
    await demo_web_search()
    # await demo_reasoning()  # Requires gpt-5 model
    await demo_mcp_integration()
    await demo_combined_capabilities()


if __name__ == "__main__":
    asyncio.run(main())
7.4 Image Analysis Best Practices
1. Provide Context in Queries:

Python

# ❌ Vague question
await agent.analyze_image(image_url, "What is this?")

# ✅ Specific, contextual question
await agent.analyze_image(
    image_url,
    """
    This is a product photo. Please identify:
    1. The product type and model
    2. Visible features and specifications
    3. Condition (new/used)
    4. Any text or labels visible
    5. Estimated quality/value based on appearance
    """
)
2. Handle Multiple Images:

Python

async def analyze_multiple_images(agent: MultiModalAgent, image_urls: list[str], question: str):
    """
    Analyze multiple images in a single query.
    """
    contents = [TextContent(text=question)]
    
    for url in image_urls:
        contents.append(UriContent(uri=url, media_type="image/jpeg"))
    
    message = ChatMessage(role="user", contents=contents)
    result = await agent.agent.run(message)
    return result.text
7.5 Web Search Optimization
1. Location-Aware Searches:

Python

# For location-specific queries
location = {
    "country": "US",
    "city": "New York"
}

result = await agent.search_web(
    "Best restaurants nearby",
    location=location
)
2. Temporal Queries:

Python

# For time-sensitive information
queries = [
    "What's the weather today?",
    "Latest news about [topic]",
    "Current stock price of [company]",
    "Today's sports scores"
]

for query in queries:
    result = await agent.search_web(query)
    print(f"{query}: {result}\n")
7.6 Common Pitfalls
Pitfall 1: Wrong Model for Vision

Python

# ❌ Using non-vision model
agent = MultiModalAgent(model_id="gpt-3.5-turbo")  # Doesn't support images
await agent.analyze_image(url)  # Will fail

# ✅ Use vision-capable model
agent = MultiModalAgent(model_id="gpt-4o")  # Supports vision
await agent.analyze_image(url)
Pitfall 2: Not Handling MCP Connection Lifecycle

Python

# ❌ Using MCP tool without connection
mcp_tool = MCPStreamableHTTPTool(url="...")
await agent.use_mcp_tool(query, mcp_tool)  # May fail

# ✅ Proper connection management
async with MCPStreamableHTTPTool(url="...") as mcp_tool:
    await agent.use_mcp_tool(query, mcp_tool)
Pitfall 3: Ignoring Reasoning Output Format

Python

# ❌ Not differentiating reasoning from final answer
async for chunk in agent.run_stream(problem):
    print(chunk.text)  # Mixes reasoning and answer

# ✅ Handle reasoning and text separately
async for chunk in agent.run_stream(problem):
    for content in chunk.contents:
        if content.type == "text_reasoning":
            print(f"[Thinking: {content.text}]")
        elif content.type == "text":
            print(content.text)
8. Advanced Topics
8.1 Thread Persistence Strategies
Strategy 1: In-Memory with Serialization

Python

import json
from agent_framework import ChatMessageStore

async def save_thread_to_disk(thread, filename: str):
    """Save thread message history to disk."""
    messages = await thread.message_store.list_messages()
    
    serializable = []
    for msg in messages:
        serializable.append({
            "role": msg.role,
            "content": str(msg.contents[0].text) if msg.contents else ""
        })
    
    with open(filename, 'w') as f:
        json.dump(serializable, f)

async def load_thread_from_disk(agent, filename: str):
    """Load thread message history from disk."""
    from agent_framework import ChatMessage, TextContent
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    messages = [
        ChatMessage(role=msg["role"], contents=[TextContent(text=msg["content"])])
        for msg in data
    ]
    
    thread = agent.get_new_thread()
    thread.message_store = ChatMessageStore(messages)
    return thread
Strategy 2: Database Persistence

Python

import sqlite3
from agent_framework import ChatMessage, TextContent, ChatMessageStore

class DatabaseThreadStore:
    """Persist threads in SQLite database."""
    
    def __init__(self, db_path: str = "threads.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
            )
        """)
        conn.commit()
        conn.close()
    
    async def save_message(self, thread_id: str, role: str, content: str):
        """Save a single message."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content)
        )
        conn.commit()
        conn.close()
    
    async def load_thread(self, agent, thread_id: str):
        """Load thread from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY timestamp",
            (thread_id,)
        )
        
        messages = [
            ChatMessage(role=row[0], contents=[TextContent(text=row[1])])
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        thread = agent.get_new_thread()
        thread.message_store = ChatMessageStore(messages)
        return thread
8.2 Custom Message Stores
Implement custom message storage logic:

Python

from agent_framework import MessageStore, ChatMessage

class RedisMessageStore(MessageStore):
    """Custom message store using Redis."""
    
    def __init__(self, redis_client, thread_key: str):
        self.redis = redis_client
        self.thread_key = thread_key
    
    async def add_message(self, message: ChatMessage):
        """Add message to Redis."""
        import json
        serialized = json.dumps({
            "role": message.role,
            "content": str(message.contents[0].text) if message.contents else ""
        })
        await self.redis.rpush(self.thread_key, serialized)
    
    async def list_messages(self) -> list[ChatMessage]:
        """Retrieve all messages from Redis."""
        import json
        from agent_framework import TextContent
        
        raw_messages = await self.redis.lrange(self.thread_key, 0, -1)
        
        messages = []
        for raw in raw_messages:
            data = json.loads(raw)
            messages.append(
                ChatMessage(
                    role=data["role"],
                    contents=[TextContent(text=data["content"])]
                )
            )
        
        return messages

# Usage
import redis.asyncio as redis

async def use_redis_store():
    redis_client = redis.Redis(host='localhost', port=6379)
    store = RedisMessageStore(redis_client, "thread:user123:conv1")
    
    # Use with agent thread
    from agent_framework import AgentThread
    thread = AgentThread(message_store=store)
8.3 Approval Workflows for Hosted MCP
Implement user approval for function calls:

Python

from agent_framework import HostedMCPTool, ChatMessage

async def handle_approvals(agent, query: str, thread):
    """
    Execute query with user approval for function calls.
    """
    result = await agent.run(query, thread=thread, store=True)
    
    while len(result.user_input_requests) > 0:
        new_inputs = []
        
        for user_input_needed in result.user_input_requests:
            # Display function call details
            print(f"\nFunction Call Request:")
            print(f"  Function: {user_input_needed.function_call.name}")
            print(f"  Arguments: {user_input_needed.function_call.arguments}")
            
            # Get user approval
            approval = input("Approve this function call? (y/n): ")
            
            # Create approval response
            new_inputs.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(approval.lower() == "y")]
                )
            )
        
        # Re-run with approvals
        result = await agent.run(new_inputs, thread=thread, store=True)
    
    return result

# Create agent with MCP tool requiring approval
async def demo_approval_workflow():
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        tools=HostedMCPTool(
            name="Enterprise API",
            url="https://api.example.com/mcp",
            approval_mode="always_require"  # Always ask for approval
        )
    )
    
    thread = agent.get_new_thread()
    result = await handle_approvals(
        agent,
        "Create a new customer record for John Doe",
        thread
    )
    print(f"\nFinal Result: {result}")
8.4 Structured Outputs with Pydantic
Define response schemas for type-safe data extraction:

Python

from pydantic import BaseModel, Field
from typing import List

class CustomerInfo(BaseModel):
    """Structured customer information."""
    name: str = Field(description="Customer full name")
    email: str = Field(description="Customer email address")
    phone: str = Field(description="Customer phone number")
    preferences: List[str] = Field(description="Customer preferences or interests")

class OrderSummary(BaseModel):
    """Structured order summary."""
    order_id: str
    customer: CustomerInfo
    items: List[str]
    total: float
    status: str

async def extract_structured_data():
    """Extract structured data from conversation."""
    agent = OpenAIResponsesClient().create_agent(
        instructions="You are a data extraction assistant. Extract information into the requested format."
    )
    
    conversation = """
    Customer called: John Smith
    Email: john.smith@example.com
    Phone: 555-0123
    Interested in: Python programming, AI, Cloud computing
    """
    
    result = await agent.run(
        f"Extract customer information from this conversation:\n{conversation}",
        response_format=CustomerInfo
    )
    
    if result.value:
        customer: CustomerInfo = result.value
        print(f"Name: {customer.name}")
        print(f"Email: {customer.email}")
        print(f"Phone: {customer.phone}")
        print(f"Preferences: {', '.join(customer.preferences)}")
8.5 Performance Optimization
1. Parallel Queries:

Python

import asyncio

async def parallel_agent_queries(agent, queries: list[str]):
    """
    Execute multiple queries in parallel for faster processing.
    """
    tasks = [agent.run(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

# Usage
agent = OpenAIChatClient().create_agent(instructions="You are helpful.")
queries = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]
results = await parallel_agent_queries(agent, queries)
2. Response Caching:

Python

from functools import lru_cache
import hashlib

class CachedAgent:
    """Agent with response caching."""
    
    def __init__(self, agent):
        self.agent = agent
        self.cache = {}
    
    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def run(self, query: str, use_cache: bool = True):
        """Run query with optional caching."""
        cache_key = self._cache_key(query)
        
        if use_cache and cache_key in self.cache:
            print("Cache hit!")
            return self.cache[cache_key]
        
        result = await self.agent.run(query)
        self.cache[cache_key] = result
        return result
8.6 Testing Strategies
Unit Testing Agents:

Python

import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_weather_agent():
    """Test weather function calling."""
    # Mock the chat client
    mock_client = AsyncMock()
    mock_client.get_response.return_value = "The weather in Paris is sunny, 22°C"
    
    agent = ChatAgent(
        chat_client=mock_client,
        instructions="You are a weather assistant.",
        tools=[get_weather]
    )
    
    result = await agent.run("What's the weather in Paris?")
    
    assert "Paris" in str(result)
    assert mock_client.get_response.called

@pytest.mark.asyncio
async def test_tool_integration():
    """Test that tools are properly called."""
    call_count = 0
    
    def mock_tool(location: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Result for {location}"
    
    agent = OpenAIChatClient().create_agent(
        instructions="Use the tool",
        tools=[mock_tool]
    )
    
    # This would actually call the API in a real test
    # For true unit testing, mock the OpenAI client as well
9. Best Practices & Patterns
9.1 Resource Management with Context Managers
Always use context managers for automatic cleanup:

Python

# ✅ Best Practice: Use context managers
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful"
) as agent:
    result = await agent.run("Hello")
# Agent automatically cleaned up

# ✅ For multiple resources
async with (
    MCPStreamableHTTPTool(url="...") as mcp_tool,
    OpenAIChatClient().create_agent(tools=mcp_tool) as agent
):
    result = await agent.run("Query")
# Both resources cleaned up
9.2 Error Handling and Retry Logic
Implement robust error handling:

Python

from openai import APIError, RateLimitError
import asyncio
from typing import Optional

async def robust_agent_call(
    agent,
    query: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 30.0
) -> Optional[str]:
    """
    Make agent calls with comprehensive error handling.
    
    Features:
    - Exponential backoff for rate limits
    - Timeout protection
    - Specific error handling
    - Logging
    """
    for attempt in range(max_retries):
        try:
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                agent.run(query),
                timeout=timeout
            )
            return result.text
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached for rate limit")
                raise
                
        except asyncio.TimeoutError:
            print(f"Request timed out after {timeout}s")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            raise
            
        except APIError as e:
            print(f"API error: {e.message}")
            # Don't retry on client errors (4xx)
            if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                raise
            # Retry on server errors (5xx)
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay)
                continue
            raise
            
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise
    
    return None
9.3 Logging and Debugging
Implement comprehensive logging:

Python

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LoggedAgent:
    """Agent wrapper with comprehensive logging."""
    
    def __init__(self, agent, logger_name: str = "agent"):
        self.agent = agent
        self.logger = logging.getLogger(logger_name)
    
    async def run(self, query: str, **kwargs):
        """Run query with logging."""
        self.logger.info(f"Query: {query}")
        start_time = datetime.now()
        
        try:
            result = await self.agent.run(query, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Success in {duration:.2f}s")
            self.logger.debug(f"Response: {result.text[:100]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error: {type(e).__name__}: {e}")
            raise

# Usage
base_agent = OpenAIChatClient().create_agent(instructions="You are helpful")
agent = LoggedAgent(base_agent, logger_name="my_agent")
result = await agent.run("Hello")
9.4 Production Considerations
1. Configuration Management:

Python

from pydantic_settings import BaseSettings

class AgentConfig(BaseSettings):
    """Centralized configuration management."""
    
    openai_api_key: 
