Microsoft Agent Framework: Comprehensive Programming Guide
Table of Contents
Introduction

What is the Microsoft Agent Framework?
Framework Architecture
Client Types Overview
Environment Setup
Quick Start Example
Core Concepts

Understanding Client Types
Agent Lifecycle Management
Thread Management
Tool Integration Approaches
Streaming vs Non-Streaming Responses
Error Handling Patterns
Agent Type 1: Basic Conversational Agent

Use Case & Overview
Architecture
Implementation
Configuration Options
Best Practices
Common Pitfalls
Agent Type 2: Function-Calling Agent

Use Case & Overview
Function Tool Patterns
Implementation Examples
Parameter Handling
Best Practices for Tool Design
Error Handling in Functions
Agent Type 3: RAG Agent (Knowledge Retrieval)

Use Case & Overview
Vector Store Setup
Implementation
Query Optimization
Best Practices
Agent Type 4: Code Execution Agent

Use Case & Overview
Security Considerations
Implementation
Output Handling
Best Practices & Limitations
Agent Type 5: Multi-Modal Agent

Use Case & Overview
Image Analysis
Web Search Integration
MCP Tool Integration
Complete Multi-Capability Example
Best Practices
Advanced Topics

Thread Persistence Strategies
Approval Workflows for Hosted MCP
Structured Outputs with Pydantic
Performance Optimization
Best Practices & Patterns

Resource Management
Production Considerations
Security Best Practices
Troubleshooting Guide

Common Errors
Debugging Techniques
Quick Reference

Next Steps

1. Introduction
What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, production-ready Python framework designed to simplify the creation and deployment of AI agents. It provides a unified, consistent interface for building intelligent agents that can interact with users, call functions, retrieve knowledge, execute code, and process multi-modal content.

Key Benefits:

Unified Interface: Work with OpenAI, Azure OpenAI, and Azure AI models through a consistent API
Multiple Client Types: Choose from Assistants, Chat, or Responses clients based on your needs
Rich Tool Ecosystem: Built-in support for function calling, file search, code execution, web search, and MCP integration
Thread Management: Maintain conversation context across multiple interactions
Async-First Design: Built on modern Python async/await patterns for high performance
Production Ready: Comprehensive error handling, context managers, and resource cleanup
Framework Architecture
The Microsoft Agent Framework follows a layered architecture:

text

┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  ChatAgent (High-Level)                  │
│         Provides simplified agent interactions           │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│              Chat Clients (Mid-Level)                    │
│   OpenAIChatClient │ OpenAIResponsesClient │ Others      │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│            Provider SDKs (Low-Level)                     │
│        OpenAI SDK │ Azure SDK │ Other SDKs              │
└─────────────────────────────────────────────────────────┘
Component Overview:

Chat Clients: Abstraction layer over provider-specific SDKs
ChatAgent: High-level agent class for common use cases
Tools: Function tools, hosted tools (code interpreter, file search, web search, MCP)
Threads: Message history and conversation context management
Message Stores: In-memory or persistent storage for conversation history
Client Types Overview
The framework provides three primary client types, each optimized for different use cases:

Client Type	Best For	Key Features	Response Format
Assistants Client	Stateful, persistent agents with automatic lifecycle	Built-in thread management, automatic assistant creation/cleanup	Text responses
Chat Client	Flexible chat interactions with manual control	Full control over conversation flow, in-memory threads	Text responses
Responses Client	Structured outputs, advanced features	Reasoning models, structured output, image generation	Structured or text
When to Use Each Client:

OpenAIAssistantsClient / AzureAssistantsClient: Use when you want OpenAI/Azure to manage assistant state, threads are persisted on the service, and you need automatic lifecycle management.

OpenAIChatClient / AzureChatClient: Use when you want full control over the conversation, prefer in-memory thread management, or need maximum flexibility in message handling.

OpenAIResponsesClient / AzureResponsesClient: Use when you need advanced features like reasoning capabilities, structured outputs with Pydantic models, or image generation.

Environment Setup
Step 1: Install the Framework

Bash

# Using pip
pip install agent-framework

# Using uv (recommended for faster installs)
uv add agent-framework
Step 2: Configure Environment Variables

For OpenAI:

Bash

# Required
export OPENAI_API_KEY="your-api-key"

# Optional - defaults to gpt-4o if not set
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"
For Azure OpenAI:

Bash

# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-chat-deployment"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="your-responses-deployment"

# Authentication (choose one)
# Option 1: API Key
export AZURE_OPENAI_API_KEY="your-api-key"

# Option 2: Azure AD (uses DefaultAzureCredential)
# No additional env vars needed - authenticate via Azure CLI or managed identity
For Azure AI:

Bash

export AZURE_AI_PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="your-model-deployment"
Step 3: Verify Setup

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def verify_setup():
    client = OpenAIChatClient()
    response = await client.get_response("Hello! Can you confirm you're working?")
    print(f"Response: {response}")

asyncio.run(verify_setup())
Quick Start Example
Here's a minimal example to get you started immediately:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create a client - uses environment variables for configuration
    client = OpenAIChatClient()
    
    # Create an agent with instructions
    agent = client.create_agent(
        name="HelperAgent",
        instructions="You are a helpful assistant."
    )
    
    # Get a response
    response = await agent.run("What is the capital of France?")
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
Expected Output:

text

Agent: The capital of France is Paris.
This simple example demonstrates the core framework pattern: create a client, create an agent, and run queries. The following sections will build upon this foundation to create increasingly sophisticated agents.

2. Core Concepts
Understanding Client Types
The framework's three client types serve different architectural patterns:

1. Assistants Client Pattern

The Assistants Client delegates state management to the service provider (OpenAI/Azure). This is ideal when:

You want the service to manage conversation threads
You need assistants to persist between application restarts
You prefer automatic resource cleanup
You're building stateful, long-running assistants
Python

from agent_framework.openai import OpenAIAssistantsClient

# Assistant is created on the service and automatically managed
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_function
) as agent:
    # Assistant is created when entering context
    result = await agent.run("Hello")
    # Assistant is automatically deleted when exiting context
2. Chat Client Pattern

The Chat Client gives you full control over conversation management. Best for:

Applications needing fine-grained control over message flow
In-memory conversation management
Stateless or session-based interactions
Maximum flexibility in message handling
Python

from agent_framework.openai import OpenAIChatClient

# Agent exists only in your application's memory
client = OpenAIChatClient()
agent = client.create_agent(
    instructions="You are a helpful assistant."
)

# Full control over threads and message storage
thread = agent.get_new_thread()
result = await agent.run("Hello", thread=thread)
3. Responses Client Pattern

The Responses Client provides access to advanced features. Use when you need:

Reasoning capabilities (e.g., GPT-5 models)
Structured outputs with Pydantic schemas
Image generation
Advanced content handling
Python

from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int
    country: str

client = OpenAIResponsesClient()
agent = client.create_agent(
    instructions="You are a geography expert."
)

# Get structured output
result = await agent.run(
    "Tell me about Paris",
    response_format=CityInfo
)
city_data: CityInfo = result.value  # Type-safe access
Agent Lifecycle Management
Proper agent lifecycle management ensures efficient resource usage and prevents memory leaks.

Using Context Managers (Recommended)

Python

# Automatic cleanup with context manager
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    result = await agent.run("Hello")
    # Agent and associated resources automatically cleaned up
Manual Lifecycle Management

Python

# When you need more control
client = OpenAIAssistantsClient()
agent = client.create_agent(instructions="You are helpful.")

try:
    await agent.__aenter__()  # Initialize resources
    result = await agent.run("Hello")
finally:
    await agent.__aexit__(None, None, None)  # Cleanup resources
Best Practice: Always use context managers (async with) unless you have a specific reason to manage lifecycle manually.

Thread Management
Threads represent conversation history and context. The framework supports multiple thread management patterns:

Pattern 1: Stateless (No Thread)

Python

# Each call is independent - no conversation memory
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

await agent.run("My name is Alice")
await agent.run("What's my name?")  # Agent won't remember
Pattern 2: In-Memory Thread

Python

# Conversation context maintained in memory
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

thread = agent.get_new_thread()  # Create new thread

await agent.run("My name is Alice", thread=thread)
result = await agent.run("What's my name?", thread=thread)
# Result: "Your name is Alice"
Pattern 3: Service-Persisted Thread (Assistants/Responses)

Python

# Thread persisted on the service
client = OpenAIResponsesClient()
agent = client.create_agent(instructions="You are helpful.")

thread = agent.get_new_thread()

# Enable service storage with store=True
await agent.run("My name is Alice", thread=thread, store=True)

# Thread ID can be saved and reused later
thread_id = thread.service_thread_id

# Later, recreate thread from ID
new_thread = AgentThread(service_thread_id=thread_id)
await agent.run("What's my name?", thread=new_thread, store=True)
Pattern 4: Custom Message Store

Python

from agent_framework import ChatMessageStore, ChatMessage

# Create thread with custom message store
messages = [
    ChatMessage(role="user", text="My name is Alice"),
    ChatMessage(role="assistant", text="Nice to meet you, Alice!")
]

thread = AgentThread(message_store=ChatMessageStore(messages))
result = await agent.run("What's my name?", thread=thread)
Tool Integration Approaches
Tools extend agent capabilities with custom functions, external APIs, and service-hosted features.

Approach 1: Agent-Level Tools

Tools defined when creating the agent are available for all queries:

Python

def get_current_time() -> str:
    return datetime.now().isoformat()

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_current_time]  # Available for all queries
)

await agent.run("What time is it?")  # Uses tool
await agent.run("What's the weather?")  # Tool still available
Approach 2: Run-Level Tools

Tools passed to run() are available only for that specific query:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
    # No tools at agent level
)

# Tool only available for this specific call
await agent.run("What time is it?", tools=[get_current_time])
Approach 3: Mixed Tools

Combine agent-level and run-level tools:

Python

def base_function():
    return "base"

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[base_function]  # Always available
)

def special_function():
    return "special"

# This query has access to both base_function and special_function
await agent.run("Help me", tools=[special_function])
Hosted Tools

The framework provides pre-built hosted tools:

Python

from agent_framework import (
    HostedCodeInterpreterTool,
    HostedFileSearchTool,
    HostedWebSearchTool,
    HostedMCPTool
)

agent = OpenAIResponsesClient().create_agent(
    instructions="You are helpful.",
    tools=[
        HostedCodeInterpreterTool(),  # Python code execution
        HostedFileSearchTool(),  # Document search
        HostedWebSearchTool(),  # Real-time web search
    ]
)
Streaming vs Non-Streaming Responses
Non-Streaming (Default)

Receive the complete response at once:

Python

# Wait for complete response
result = await agent.run("Write a story about a robot")
print(result.text)  # Full story printed at once
Advantages:

Simpler code
Easier error handling
Complete response available immediately
Use when:

Response latency isn't critical
You need the full response before processing
Building batch processing systems
Streaming

Receive response chunks as they're generated:

Python

# Stream response in real-time
async for chunk in agent.run_stream("Write a story about a robot"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
Advantages:

Lower perceived latency
Better user experience for long responses
Can process or display content as it arrives
Use when:

Building interactive UIs
User experience is priority
Processing streaming data
Advanced Streaming with Metadata

Python

async for chunk in agent.run_stream("Analyze this data"):
    # Access different content types in the stream
    for content in chunk.contents:
        if content.type == "text":
            print(content.text, end="")
        elif content.type == "text_reasoning":
            print(f"[Reasoning: {content.text}]")
        elif content.type == "usage":
            print(f"[Tokens used: {content.details}]")
Error Handling Patterns
Pattern 1: Basic Try-Except

Python

try:
    result = await agent.run("Query that might fail")
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
Pattern 2: Specific Exception Handling

Python

from openai import APIError, RateLimitError, APITimeoutError

try:
    result = await agent.run(query)
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
    result = await agent.run(query)
except APITimeoutError:
    # Handle timeout
    print("Request timed out")
except APIError as e:
    # Handle API errors
    print(f"API error: {e}")
Pattern 3: Retry Logic with Backoff

Python

import asyncio
from typing import Optional

async def run_with_retry(
    agent, 
    query: str, 
    max_retries: int = 3
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            return await agent.run(query)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None
Pattern 4: Context Manager Error Handling

Python

async with OpenAIChatClient().create_agent(
    instructions="You are helpful."
) as agent:
    try:
        result = await agent.run(query)
    except Exception as e:
        print(f"Error during execution: {e}")
        # Agent resources still cleaned up automatically
Best Practices:

Always handle RateLimitError with retry logic
Log errors with sufficient context for debugging
Use specific exception types when possible
Implement timeouts for long-running operations
Clean up resources even when errors occur (use context managers)
3. Agent Type 1: Basic Conversational Agent
Use Case & Overview
Basic Conversational Agents are the foundation of AI interaction. They excel at:

Customer support and FAQ handling
General information queries
Conversational interfaces
Tutorial and onboarding assistants
Personal productivity helpers
Key Characteristics:

No external tools or API calls
Relies on model's built-in knowledge
Simple request-response pattern
Minimal configuration required
Fast response times
Real-World Examples:

FAQ chatbot for a website
Personal journaling assistant
Study buddy for students
General knowledge helper
Architecture
text

User Query → Agent (with Instructions) → LLM → Response → User
The agent acts as a configured interface to the language model, with:

Instructions: Define the agent's personality, role, and behavior
Model: The underlying LLM (GPT-4, GPT-3.5, etc.)
Context: Optional conversation history for multi-turn interactions
Implementation
Example 1: Simple Q&A Agent

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def create_simple_agent():
    """
    Creates a basic Q&A agent with no special capabilities.
    Perfect for general knowledge queries.
    """
    # Create the client
    client = OpenAIChatClient()
    
    # Create agent with clear instructions
    agent = client.create_agent(
        name="KnowledgeHelper",
        instructions="""You are a knowledgeable and friendly assistant.
        Provide clear, concise answers to user questions.
        If you don't know something, admit it honestly."""
    )
    
    # Single query
    response = await agent.run("What is quantum computing?")
    print(f"Agent: {response}")
    
    return agent

asyncio.run(create_simple_agent())
Example 2: Conversational Agent with Memory

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def create_conversational_agent():
    """
    Creates an agent that maintains conversation context.
    Useful for multi-turn interactions.
    """
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="ConversationalHelper",
        instructions="""You are a friendly conversational assistant.
        Remember context from previous messages in the conversation.
        Be warm and engaging while remaining helpful."""
    )
    
    # Create a thread to maintain conversation history
    thread = agent.get_new_thread()
    
    # First interaction
    response1 = await agent.run(
        "Hi! My name is Sarah and I'm learning Python.",
        thread=thread
    )
    print(f"Agent: {response1.text}\n")
    
    # Second interaction - agent remembers the name
    response2 = await agent.run(
        "What topics should I focus on first?",
        thread=thread
    )
    print(f"Agent: {response2.text}\n")
    
    # Third interaction - agent remembers both name and context
    response3 = await agent.run(
        "Can you remind me what we were discussing?",
        thread=thread
    )
    print(f"Agent: {response3.text}")

asyncio.run(create_conversational_agent())
Example 3: Streaming Conversational Agent

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def create_streaming_agent():
    """
    Creates an agent with streaming responses for better UX.
    Shows response as it's generated, reducing perceived latency.
    """
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="StreamingAssistant",
        instructions="You are a helpful assistant. Provide detailed, thoughtful responses."
    )
    
    query = "Explain the concept of machine learning to a beginner."
    print(f"User: {query}\n")
    print("Agent: ", end="", flush=True)
    
    # Stream the response
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    
    print("\n")  # New line after response completes

asyncio.run(create_streaming_agent())
Example 4: Specialized Domain Agent

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def create_specialized_agent():
    """
    Creates an agent specialized in a specific domain.
    Instructions define expertise area and response style.
    """
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="MedicalAdvisor",
        instructions="""You are a medical information assistant.
        
        Guidelines:
        - Provide general health information only
        - Always recommend consulting healthcare professionals for medical decisions
        - Use clear, accessible language
        - Be empathetic and supportive
        - Never diagnose or prescribe treatments
        
        Your role is educational support, not medical advice."""
    )
    
    thread = agent.get_new_thread()
    
    # Query about symptoms
    query1 = "What are common causes of headaches?"
    print(f"User: {query1}")
    response1 = await agent.run(query1, thread=thread)
    print(f"Agent: {response1.text}\n")
    
    # Follow-up query
    query2 = "Should I be worried?"
    print(f"User: {query2}")
    response2 = await agent.run(query2, thread=thread)
    print(f"Agent: {response2.text}")

asyncio.run(create_specialized_agent())
Configuration Options
Model Selection

Python

from agent_framework.openai import OpenAIChatClient

# Specify model explicitly
client = OpenAIChatClient(model_id="gpt-4o")  # High capability
# OR
client = OpenAIChatClient(model_id="gpt-3.5-turbo")  # Faster, cheaper

agent = client.create_agent(
    instructions="You are helpful."
)
Temperature Control

Python

# Control randomness/creativity of responses
agent = client.create_agent(
    instructions="You are a creative writing assistant.",
    additional_chat_options={
        "temperature": 0.9  # Higher = more creative (0.0 to 2.0)
    }
)

# For factual, deterministic responses
agent = client.create_agent(
    instructions="You are a factual information assistant.",
    additional_chat_options={
        "temperature": 0.1  # Lower = more focused and deterministic
    }
)
Response Length Control

Python

agent = client.create_agent(
    instructions="You are a concise assistant.",
    additional_chat_options={
        "max_tokens": 150  # Limit response length
    }
)
Top-P Sampling

Python

agent = client.create_agent(
    instructions="You are helpful.",
    additional_chat_options={
        "top_p": 0.9  # Nucleus sampling (alternative to temperature)
    }
)
Best Practices
1. Clear, Specific Instructions

❌ Vague:

Python

instructions = "Help users."
✅ Clear:

Python

instructions = """You are a customer support assistant for TechCorp.

Your responsibilities:
- Answer questions about our products
- Help troubleshoot common issues
- Direct complex problems to human support

Communication style:
- Professional but friendly
- Clear and concise
- Patient and empathetic

When you don't know: Admit it and offer to connect them with a specialist."""
2. Thread Management for Conversations

❌ Without thread - no memory:

Python

await agent.run("My order number is 12345")
await agent.run("What's the status?")  # Agent doesn't remember order number
✅ With thread - maintains context:

Python

thread = agent.get_new_thread()
await agent.run("My order number is 12345", thread=thread)
await agent.run("What's the status?", thread=thread)  # Agent remembers
3. Use Streaming for Better UX

✅ For user-facing applications:

Python

# User sees response immediately as it generates
async for chunk in agent.run_stream(query):
    if chunk.text:
        display_to_user(chunk.text)  # Update UI incrementally
4. Implement Proper Error Handling

Python

from openai import APIError

async def safe_agent_call(agent, query, thread=None):
    try:
        return await agent.run(query, thread=thread)
    except APIError as e:
        print(f"API Error: {e}")
        return "I'm having trouble connecting. Please try again."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."
5. Resource Cleanup

✅ Use context managers:

Python

async with client.create_agent(instructions="You are helpful.") as agent:
    result = await agent.run(query)
    # Automatic cleanup when done
Common Pitfalls
Pitfall 1: Not Using Threads for Multi-Turn Conversations

Problem:

Python

agent = client.create_agent(instructions="You are helpful.")

await agent.run("My favorite color is blue")
result = await agent.run("What's my favorite color?")
# Agent responds: "I don't know your favorite color"
Solution:

Python

agent = client.create_agent(instructions="You are helpful.")
thread = agent.get_new_thread()

await agent.run("My favorite color is blue", thread=thread)
result = await agent.run("What's my favorite color?", thread=thread)
# Agent responds: "Your favorite color is blue"
Pitfall 2: Overly Long Instructions

Problem:

Python

instructions = """[5 pages of instructions]..."""  # Too long
Solution:

Python

# Keep instructions focused and concise (a few paragraphs max)
instructions = """You are a friendly customer service agent.
Be helpful, concise, and professional.
If you can't help, escalate to human support."""
Pitfall 3: Not Handling API Limits

Problem:

Python

# No error handling - app crashes on rate limits
result = await agent.run(query)
Solution:

Python

from openai import RateLimitError
import asyncio

try:
    result = await agent.run(query)
except RateLimitError:
    await asyncio.sleep(60)  # Wait before retrying
    result = await agent.run(query)
Pitfall 4: Blocking the Event Loop

Problem:

Python

# Using synchronous sleep in async code
import time
time.sleep(5)  # Blocks entire event loop!
Solution:

Python

# Use async sleep
import asyncio
await asyncio.sleep(5)  # Non-blocking
4. Agent Type 2: Function-Calling Agent
Use Case & Overview
Function-calling agents bridge AI with your business logic, APIs, and external systems. They enable agents to:

Call custom functions to retrieve real-time data
Integrate with databases and external APIs
Perform calculations and data processing
Execute business logic based on user queries
Orchestrate complex workflows
Key Characteristics:

Dynamically determine when to call functions
Parse function parameters from natural language
Handle multiple function calls in a single query
Combine function results with natural language responses
Real-World Examples:

Weather agent accessing weather APIs
Order management system checking order status
Calendar assistant scheduling meetings
E-commerce agent checking inventory and prices
Financial advisor calculating investment returns
Function Tool Patterns
The framework supports three patterns for providing tools to agents:

Pattern 1: Agent-Level Tools

Tools defined during agent creation are available for all queries:

Python

def get_stock_price(symbol: str) -> float:
    # Your implementation
    return 150.25

agent = client.create_agent(
    instructions="You are a financial assistant.",
    tools=[get_stock_price]  # Available for all queries
)
Use when:

Tools are core to the agent's purpose
Same tools needed across all interactions
Building specialized agents
Pattern 2: Run-Level Tools

Tools passed to specific run() calls:

Python

agent = client.create_agent(
    instructions="You are a general assistant."
)

# Tool only available for this specific query
result = await agent.run(
    "What's the stock price of AAPL?",
    tools=[get_stock_price]
)
Use when:

Tools change based on user context
Different users have different available tools
Security requires per-query tool authorization
Pattern 3: Mixed Tools

Combine base tools with query-specific tools:

Python

def base_tool():
    return "base functionality"

agent = client.create_agent(
    instructions="You are helpful.",
    tools=[base_tool]  # Always available
)

def special_tool():
    return "special functionality"

# This query has both base_tool and special_tool
result = await agent.run(
    query,
    tools=[special_tool]  # Additional tool for this query
)
Use when:

Some tools are universal, others are contextual
Building extensible agent systems
Role-based access control needed
Implementation Examples
Example 1: Simple Function-Calling Agent

Python

import asyncio
from typing import Annotated
from datetime import datetime
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

def get_current_time(
    timezone: Annotated[str, Field(description="Timezone name (e.g., 'UTC', 'US/Pacific')")] = "UTC"
) -> str:
    """Get the current time in the specified timezone."""
    # Simplified - use pytz for real implementation
    current_time = datetime.now()
    return f"The current time in {timezone} is {current_time.strftime('%Y-%m-%d %H:%M:%S')}"

def get_current_date() -> str:
    """Get today's date."""
    return datetime.now().strftime('%Y-%m-%d')

async def main():
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="TimeAgent",
        instructions="You are a helpful time assistant. Use the available functions to provide accurate time information.",
        tools=[get_current_time, get_current_date]
    )
    
    # Agent will automatically call the appropriate function
    query = "What time is it in UTC?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Agent: {result}\n")
    
    # Agent can call multiple functions
    query2 = "What's today's date and the current time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}")

asyncio.run(main())
Example 2: Database Integration Agent

Python

import asyncio
from typing import Annotated, Optional
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

# Simulated database
ORDERS_DB = {
    "12345": {"status": "shipped", "items": ["Laptop", "Mouse"], "total": 1299.99},
    "67890": {"status": "processing", "items": ["Keyboard"], "total": 89.99}
}

def get_order_status(
    order_id: Annotated[str, Field(description="The order ID to look up")]
) -> str:
    """Retrieve the status of an order."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return f"Order {order_id} not found."
    return f"Order {order_id} status: {order['status']}"

def get_order_details(
    order_id: Annotated[str, Field(description="The order ID to look up")]
) -> str:
    """Get detailed information about an order."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return f"Order {order_id} not found."
    
    items_list = ", ".join(order['items'])
    return f"Order {order_id}: Status={order['status']}, Items=[{items_list}], Total=${order['total']}"

def search_orders_by_status(
    status: Annotated[str, Field(description="Order status to search for (e.g., 'shipped', 'processing')")]
) -> str:
    """Find all orders with a specific status."""
    matching_orders = [
        order_id for order_id, details in ORDERS_DB.items()
        if details['status'].lower() == status.lower()
    ]
    
    if not matching_orders:
        return f"No orders found with status '{status}'"
    
    return f"Orders with status '{status}': {', '.join(matching_orders)}"

async def main():
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="OrderAssistant",
        instructions="""You are a customer service assistant for an e-commerce platform.
        
        Help customers:
        - Check order status
        - View order details
        - Find orders by status
        
        Be friendly and helpful. Use the available functions to provide accurate information.""",
        tools=[get_order_status, get_order_details, search_orders_by_status]
    )
    
    thread = agent.get_new_thread()
    
    # Query 1: Simple status check
    query1 = "What's the status of order 12345?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1}\n")
    
    # Query 2: Detailed information
    query2 = "Can you give me all the details for that order?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2}\n")
    
    # Query 3: Search by status
    query3 = "Show me all shipped orders"
    print(f"User: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3}")

asyncio.run(main())
Example 3: External API Integration

Python

import asyncio
from typing import Annotated
from agent_framework.openai import OpenAIChatClient
from pydantic import Field
import httpx  # Install: pip install httpx

async def get_github_user_info(
    username: Annotated[str, Field(description="GitHub username to lookup")]
) -> str:
    """Retrieve public information about a GitHub user."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"https://api.github.com/users/{username}")
            response.raise_for_status()
            data = response.json()
            
            return f"""GitHub User: {data['login']}
Name: {data.get('name', 'Not provided')}
Bio: {data.get('bio', 'No bio')}
Public Repos: {data['public_repos']}
Followers: {data['followers']}
Following: {data['following']}
Profile: {data['html_url']}"""
        except httpx.HTTPError as e:
            return f"Error fetching user info: {str(e)}"

async def get_github_repo_info(
    owner: Annotated[str, Field(description="Repository owner username")],
    repo: Annotated[str, Field(description="Repository name")]
) -> str:
    """Get information about a GitHub repository."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
            response.raise_for_status()
            data = response.json()
            
            return f"""Repository: {data['full_name']}
Description: {data.get('description', 'No description')}
Stars: {data['stargazers_count']}
Forks: {data['forks_count']}
Open Issues: {data['open_issues_count']}
Language: {data.get('language', 'Not specified')}
URL: {data['html_url']}"""
        except httpx.HTTPError as e:
            return f"Error fetching repo info: {str(e)}"

async def main():
    client = OpenAIChatClient()
    
    agent = client.create_agent(
        name="GitHubAssistant",
        instructions="""You are a GitHub information assistant.
        Help users learn about GitHub users and repositories.
        Use the available functions to fetch real-time data from GitHub.""",
        tools=[get_github_user_info, get_github_repo_info]
    )
    
    # Query about a user
    query1 = "Tell me about the GitHub user 'torvalds'"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}\n")
    
    # Query about a repository
    query2 = "What can you tell me about the microsoft/autogen repository?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}")

asyncio.run(main())
Parameter Handling
Type Annotations and Pydantic Fields

The framework uses Python type hints and Pydantic Field for parameter validation:

Python

from typing import Annotated, Optional, Literal
from pydantic import Field

def book_flight(
    departure: Annotated[str, Field(description="Departure city")],
    destination: Annotated[str, Field(description="Destination city")],
    date: Annotated[str, Field(description="Travel date in YYYY-MM-DD format")],
    passengers: Annotated[int, Field(description="Number of passengers", ge=1, le=9)] = 1,
    class_type: Annotated[
        Literal["economy", "business", "first"],
        Field(description="Cabin class")
    ] = "economy"
) -> str:
    """Book a flight with specified parameters."""
    return f"Booking {passengers} passenger(s) from {departure} to {destination} on {date} in {class_type} class"
Key annotations:

Annotated[type, Field(...)]: Provides type and description
Field(description="..."): Helps the model understand parameter purpose
ge=, le=: Validation constraints (greater/less than or equal)
Literal[...]: Restricts to specific values
Default values: Make parameters optional
Complex Parameter Types

Python

from typing import Annotated, List, Dict
from pydantic import Field

def create_report(
    metrics: Annotated[
        List[str],
        Field(description="List of metrics to include (e.g., ['sales', 'revenue', 'users'])")
    ],
    filters: Annotated[
        Dict[str, str],
        Field(description="Filters to apply as key-value pairs")
    ] = None,
    format: Annotated[
        Literal["pdf", "csv", "json"],
        Field(description="Output format")
    ] = "pdf"
) -> str:
    """Generate a report with specified metrics and filters."""
    filters = filters or {}
    return f"Generating {format} report with metrics {metrics} and filters {filters}"
Best Practices for Tool Design
1. Clear, Descriptive Function Names

❌ Vague:

Python

def get_data(id: str) -> str:
    pass
✅ Clear:

Python

def get_customer_order_details(order_id: str) -> str:
    """Retrieve detailed information about a customer's order."""
    pass
2. Comprehensive Docstrings

✅ Good practice:

Python

def calculate_shipping_cost(
    weight: Annotated[float, Field(description="Package weight in kilograms")],
    destination: Annotated[str, Field(description="Destination country code (e.g., 'US', 'UK')")]
) -> str:
    """
    Calculate shipping cost based on package weight and destination.
    
    Args:
        weight: Package weight in kilograms
        destination: ISO country code for destination
    
    Returns:
        Formatted string with shipping cost and estimated delivery time
    """
    # Implementation
    pass
3. Return Structured, Informative Results

❌ Minimal information:

Python

def get_weather(city: str) -> str:
    return "72°F"
✅ Rich information:

Python

def get_weather(city: str) -> str:
    return """Weather in New York:
Temperature: 72°F (22°C)
Conditions: Partly Cloudy
Humidity: 65%
Wind: 8 mph NE
Forecast: Sunny tomorrow"""
4. Handle Errors Gracefully

Python

def get_user_profile(user_id: str) -> str:
    """Retrieve user profile information."""
    try:
        # Your database call
        profile = database.get_user(user_id)
        if not profile:
            return f"User {user_id} not found."
        return format_profile(profile)
    except DatabaseError as e:
        return f"Unable to retrieve user profile due to a database error."
    except Exception as e:
        return "An unexpected error occurred while fetching user profile."
5. Keep Functions Focused

❌ Too many responsibilities:

Python

def do_everything(user_id: str, action: str, data: dict) -> str:
    # Handles user creation, updates, deletion, and queries
    pass
✅ Focused functions:

Python

def create_user(username: str, email: str) -> str:
    """Create a new user account."""
    pass

def update_user_email(user_id: str, new_email: str) -> str:
    """Update a user's email address."""
    pass

def delete_user(user_id: str) -> str:
    """Delete a user account."""
    pass
Error Handling in Functions
Pattern 1: Return Error Messages

Python

def get_product_info(product_id: str) -> str:
    """Get product information from database."""
    try:
        product = database.query(product_id)
        if not product:
            return f"Product {product_id} not found in our catalog."
        return format_product_info(product)
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        return "Unable to retrieve product information at this time. Please try again later."
Pattern 2: Validation Before Processing

Python

def schedule_meeting(
    date: Annotated[str, Field(description="Meeting date in YYYY-MM-DD format")],
    time: Annotated[str, Field(description="Meeting time in HH:MM format")],
    duration: Annotated[int, Field(description="Duration in minutes")]
) -> str:
    """Schedule a meeting."""
    # Validate date format
    try:
        from datetime import datetime
        meeting_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD format."
    
    # Validate time format
    try:
        meeting_time = datetime.strptime(time, "%H:%M")
    except ValueError:
        return "Invalid time format. Please use HH:MM format."
    
    # Validate duration
    if duration < 15 or duration > 480:
        return "Meeting duration must be between 15 minutes and 8 hours."
    
    # Process the meeting
    return f"Meeting scheduled for {date} at {time} for {duration} minutes."
Pattern 3: Async Error Handling

Python

async def fetch_external_data(api_endpoint: str) -> str:
    """Fetch data from external API."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_endpoint)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutError:
        return "Request timed out. The external service is not responding."
    except httpx.HTTPStatusError as e:
        return f"External API error: {e.response.status_code}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Unable to fetch data from external service."
5. Agent Type 3: RAG Agent (Knowledge Retrieval)
Use Case & Overview
Retrieval-Augmented Generation (RAG) agents combine the power of large language models with your specific knowledge base. They excel at:

Document-based question answering
Internal knowledge base queries
Technical documentation assistance
Policy and procedure guidance
Research paper analysis
Legal document review
Key Characteristics:

Upload and index documents into vector stores
Semantic search across your knowledge base
Cite sources in responses
Keep information up-to-date without retraining models
Scale to thousands of documents
Real-World Examples:

Company policy chatbot
Technical support assistant with product documentation
Legal research assistant
Academic research helper
Customer support with knowledge base integration
Advantages Over Fine-Tuning:

No retraining required for new information
Easy to update knowledge base
More cost-effective
Transparent source attribution
Works with any LLM
Vector Store Setup
Vector stores enable semantic search by converting documents into embeddings. The framework integrates with hosted vector stores:

Step 1: Upload Documents

Python

from agent_framework.openai import OpenAIResponsesClient

async def upload_document(client: OpenAIResponsesClient, file_path: str) -> str:
    """Upload a document to the service."""
    with open(file_path, 'rb') as f:
        file = await client.client.files.create(
            file=(file_path, f),
            purpose="user_data"
        )
    return file.id
Step 2: Create Vector Store

Python

async def create_vector_store(
    client: OpenAIResponsesClient,
    name: str,
    file_ids: list[str]
) -> str:
    """Create a vector store and add files to it."""
    # Create the vector store
    vector_store = await client.client.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": 7}  # Auto-cleanup
    )
    
    # Add files to vector store
    for file_id in file_ids:
        result = await client.client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file_id
        )
        
        # Check for errors
        if result.last_error:
            raise Exception(f"File processing failed: {result.last_error.message}")
    
    return vector_store.id
Step 3: Cleanup

Python

async def cleanup_vector_store(
    client: OpenAIResponsesClient,
    vector_store_id: str,
    file_ids: list[str]
):
    """Clean up vector store and files."""
    await client.client.vector_stores.delete(vector_store_id)
    for file_id in file_ids:
        await client.client.files.delete(file_id)
Implementation
Example 1: Basic RAG Agent

Python

import asyncio
from agent_framework import (
    ChatAgent,
    HostedFileSearchTool,
    HostedVectorStoreContent
)
from agent_framework.openai import OpenAIResponsesClient

async def create_knowledge_base(client: OpenAIResponsesClient):
    """Create a simple knowledge base from text content."""
    # Create a document
    file = await client.client.files.create(
        file=("company_policies.txt", b"""
Company Vacation Policy

Full-time employees receive:
- 15 days vacation per year (first 3 years)
- 20 days vacation per year (3-7 years)
- 25 days vacation per year (7+ years)

Part-time employees receive pro-rated vacation based on hours worked.

Vacation requests must be submitted at least 2 weeks in advance.
Maximum consecutive vacation days: 10 days.
Unused vacation days can be rolled over up to 5 days per year.

Remote Work Policy

Employees may work remotely up to 3 days per week with manager approval.
Core hours (10 AM - 3 PM) must be available for meetings.
Home office must meet security and ergonomic requirements.
        """),
        purpose="user_data"
    )
    
    # Create vector store
    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7}
    )
    
    # Add file to vector store
    result = await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file.id
    )
    
    if result.last_error:
        raise Exception(f"Failed to process file: {result.last_error.message}")
    
    return file.id, vector_store.id

async def main():
    client = OpenAIResponsesClient()
    
    # Create knowledge base
    file_id, vector_store_id = await create_knowledge_base(client)
    
    try:
        # Create RAG agent
        agent = ChatAgent(
            chat_client=client,
            instructions="""You are a helpful HR assistant.
            Answer questions about company policies using the knowledge base.
            Always cite specific policies when answering questions.""",
            tools=[HostedFileSearchTool()]
        )
        
        # Create vector store content reference
        vector_store = HostedVectorStoreContent(vector_store_id=vector_store_id)
        
        # Query the knowledge base
        queries = [
            "How many vacation days do I get after 5 years?",
            "What's the remote work policy?",
            "Can I roll over unused vacation days?"
        ]
        
        for query in queries:
            print(f"\nUser: {query}")
            print("Agent: ", end="", flush=True)
            
            async for chunk in agent.run_stream(
                query,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
    
    finally:
        # Cleanup
        await client.client.vector_stores.delete(vector_store_id)
        await client.client.files.delete(file_id)

asyncio.run(main())
Example 2: Multi-Document RAG Agent

Python

import asyncio
from pathlib import Path
from agent_framework import ChatAgent, HostedFileSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def create_multi_doc_knowledge_base(
    client: OpenAIResponsesClient,
    documents: dict[str, bytes]
) -> tuple[list[str], str]:
    """Create a knowledge base from multiple documents."""
    file_ids = []
    
    # Upload all documents
    for filename, content in documents.items():
        file = await client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        file_ids.append(file.id)
    
    # Create vector store
    vector_store = await client.client.vector_stores.create(
        name="technical_docs",
        expires_after={"anchor": "last_active_at", "days": 30}
    )
    
    # Add all files to vector store
    for file_id in file_ids:
        result = await client.client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file_id
        )
        if result.last_error:
            print(f"Warning: File {file_id} had an error: {result.last_error}")
    
    return file_ids, vector_store.id

async def main():
    client = OpenAIResponsesClient()
    
    # Sample documents (in practice, load from files)
    documents = {
        "api_documentation.txt": b"""
API Documentation

Authentication:
All API requests require an API key in the header: Authorization: Bearer <API_KEY>

Endpoints:
GET /api/users - List all users
POST /api/users - Create a new user
GET /api/users/{id} - Get user details
PUT /api/users/{id} - Update user
DELETE /api/users/{id} - Delete user

Rate Limits:
- Free tier: 100 requests/hour
- Pro tier: 1000 requests/hour
- Enterprise: Unlimited
        """,
        
        "troubleshooting_guide.txt": b"""
Troubleshooting Guide

Error: "Authentication Failed"
Solution: Verify your API key is correct and has not expired.

Error: "Rate Limit Exceeded"
Solution: Wait for the rate limit window to reset, or upgrade your plan.

Error: "Invalid Request Format"
Solution: Ensure your request body is valid JSON and includes all required fields.

Error: "Resource Not Found"
Solution: Verify the resource ID exists and you have permission to access it.
        """
    }
    
    file_ids, vector_store_id = await create_multi_doc_knowledge_base(
        client,
        documents
    )
    
    try:
        agent = ChatAgent(
            chat_client=client,
            instructions="""You are a technical support assistant.
            Help developers use our API by answering questions from the documentation.
            Provide clear, code examples when relevant.
            If information isn't in the docs, say so.""",
            tools=[HostedFileSearchTool()]
        )
        
        thread = agent.get_new_thread()
        
        queries = [
            "How do I authenticate with the API?",
            "What should I do if I get a rate limit error?",
            "What are the available user endpoints?"
        ]
        
        for query in queries:
            print(f"\nUser: {query}")
            result = await agent.run(
                query,
                thread=thread,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            print(f"Agent: {result.text}")
    
    finally:
        # Cleanup
        await client.client.vector_stores.delete(vector_store_id)
        for file_id in file_ids:
            await client.client.files.delete(file_id)

asyncio.run(main())
Example 3: Dynamic Knowledge Base Updates

Python

import asyncio
from agent_framework import ChatAgent, HostedFileSearchTool
from agent_framework.openai import OpenAIResponsesClient

class KnowledgeBaseManager:
    """Manages a dynamic knowledge base with add/remove capabilities."""
    
    def __init__(self, client: OpenAIResponsesClient, name: str):
        self.client = client
        self.name = name
        self.vector_store_id = None
        self.file_ids = []
    
    async def initialize(self):
        """Create the vector store."""
        vector_store = await self.client.client.vector_stores.create(
            name=self.name,
            expires_after={"anchor": "last_active_at", "days": 30}
        )
        self.vector_store_id = vector_store.id
    
    async def add_document(self, filename: str, content: bytes):
        """Add a document to the knowledge base."""
        # Upload file
        file = await self.client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        self.file_ids.append(file.id)
        
        # Add to vector store
        result = await self.client.client.vector_stores.files.create_and_poll(
            vector_store_id=self.vector_store_id,
            file_id=file.id
        )
        
        if result.last_error:
            raise Exception(f"Failed to add document: {result.last_error.message}")
        
        print(f"✓ Added document: {filename}")
    
    async def remove_document(self, file_id: str):
        """Remove a document from the knowledge base."""
        await self.client.client.vector_stores.files.delete(
            vector_store_id=self.vector_store_id,
            file_id=file_id
        )
        await self.client.client.files.delete(file_id)
        self.file_ids.remove(file_id)
        print(f"✓ Removed document: {file_id}")
    
    async def cleanup(self):
        """Clean up all resources."""
        if self.vector_store_id:
            await self.client.client.vector_stores.delete(self.vector_store_id)
        for file_id in self.file_ids:
            try:
                await self.client.client.files.delete(file_id)
            except:
                pass  # May have been deleted already

async def main():
    client = OpenAIResponsesClient()
    kb_manager = KnowledgeBaseManager(client, "dynamic_kb")
    
    await kb_manager.initialize()
    
    try:
        # Add initial documents
        await kb_manager.add_document(
            "product_info.txt",
            b"Our flagship product is the SuperWidget 3000, priced at $299."
        )
        
        # Create agent
        agent = ChatAgent(
            chat_client=client,
            instructions="You are a product specialist. Answer questions using the knowledge base.",
            tools=[HostedFileSearchTool()]
        )
        
        # Query
        result = await agent.run(
            "What's the price of the SuperWidget 3000?",
            tool_resources={"file_search": {"vector_store_ids": [kb_manager.vector_store_id]}}
        )
        print(f"Agent: {result.text}\n")
        
        # Update knowledge base
        await kb_manager.add_document(
            "pricing_update.txt",
            b"PRICE UPDATE: SuperWidget 3000 is now on sale for $249!"
        )
        
        # Query again
        result = await agent.run(
            "What's the current price of the SuperWidget 3000?",
            tool_resources={"file_search": {"vector_store_ids": [kb_manager.vector_store_id]}}
        )
        print(f"Agent: {result.text}")
    
    finally:
        await kb_manager.cleanup()

asyncio.run(main())
Query Optimization
Technique 1: Provide Context in Queries

❌ Vague query:

Python

result = await agent.run("What's the policy?")
✅ Specific query:

Python

result = await agent.run("What's the company's remote work policy for full-time employees?")
Technique 2: Use Conversation History

Python

thread = agent.get_new_thread()

# Build context
await agent.run("I'm a full-time employee with 5 years at the company", thread=thread)
await agent.run("How many vacation days do I get?", thread=thread)  # Uses context
Technique 3: Metadata Filtering (Advanced)

Python

# When creating vector store, add metadata to files
file = await client.client.files.create(
    file=("2024_policies.txt", content),
    purpose="user_data",
    # Note: Metadata support varies by provider
)
Best Practices
1. Document Preparation

✅ Clean, structured documents:

Use clear headings and sections
Remove unnecessary formatting
Include relevant metadata
Keep documents focused on specific topics
2. Chunking Strategy

For large documents, consider splitting into smaller, focused files:

Python

# Instead of one huge file
huge_policy_doc.txt  # 500 pages

# Split into logical sections
vacation_policy.txt
remote_work_policy.txt
benefits_policy.txt
3. Regular Knowledge Base Updates

Python

# Implement a refresh strategy
async def refresh_knowledge_base(kb_manager, new_docs):
    """Replace outdated documents with current versions."""
    # Remove old versions
    for old_file_id in outdated_files:
        await kb_manager.remove_document(old_file_id)
    
    # Add new versions
    for filename, content in new_docs.items():
        await kb_manager.add_document(filename, content)
4. Monitor Vector Store Size

Python

async def check_vector_store_status(client, vector_store_id):
    """Check vector store file count and status."""
    vector_store = await client.client.vector_stores.retrieve(vector_store_id)
    print(f"Files in vector store: {vector_store.file_counts.total}")
    print(f"Status: {vector_store.status}")
5. Error Handling for File Processing

Python

async def add_document_with_retry(kb_manager, filename, content, max_retries=3):
    """Add document with retry logic for processing failures."""
    for attempt in range(max_retries):
        try:
            await kb_manager.add_document(filename, content)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to add {filename} after {max_retries} attempts: {e}")
                return False
6. Agent Type 4: Code Execution Agent
Use Case & Overview
Code Execution Agents can write and execute Python code dynamically to solve problems. They excel at:

Mathematical calculations and data analysis
Data visualization and plotting
File manipulation and processing
Scientific computing
Algorithm implementation and testing
Data transformation and cleaning
Key Characteristics:

Write Python code on-the-fly
Execute code in a sandboxed environment
Handle numerical computations
Process and analyze data
Generate visualizations
Real-World Examples:

Data analysis assistant
Math tutor solving complex equations
Statistical analysis tool
Financial modeling assistant
Scientific research helper
Security Note: Code execution happens in a hosted, sandboxed environment. The agent cannot access your local filesystem or make network requests (unless explicitly configured).

Security Considerations
Hosted Code Interpreter Safety:

The framework uses hosted code interpreter services that provide:

Sandboxed Execution: Code runs in isolated containers
No Network Access: Cannot make external API calls (by default)
Limited File System: Cannot access your local files
Resource Limits: CPU and memory constraints prevent abuse
Timeout Protection: Long-running code is automatically terminated
Best Practices:

Python

# ✅ Safe: Using hosted code interpreter
from agent_framework import HostedCodeInterpreterTool

agent = client.create_agent(
    instructions="You are a data analysis assistant.",
    tools=[HostedCodeInterpreterTool()]  # Runs in secure sandbox
)
User Input Validation:

Python

# When accepting user data for code execution
def validate_user_input(user_data: str) -> bool:
    """Validate user input before processing."""
    # Check for reasonable data size
    if len(user_data) > 1_000_000:  # 1MB limit
        return False
    
    # Add other validation as needed
    return True

# In your agent workflow
if validate_user_input(user_provided_data):
    result = await agent.run(f"Analyze this data: {user_provided_data}")
Implementation
Example 1: Basic Math and Calculation Agent

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient

async def main():
    """Create an agent that can solve mathematical problems."""
    client = OpenAIResponsesClient()
    
    agent = ChatAgent(
        chat_client=client,
        instructions="""You are a mathematics assistant that can solve complex problems.
        
        When solving problems:
        1. Write clear, well-commented Python code
        2. Show your work step-by-step
        3. Explain the approach before coding
        4. Present results in a readable format
        
        You have access to Python with libraries like math, numpy, and scipy.""",
        tools=[HostedCodeInterpreterTool()]
    )
    
    queries = [
        "Calculate the factorial of 50",
        "What's the 100th Fibonacci number?",
        "Solve the quadratic equation: 2x² + 5x - 3 = 0",
        "Calculate the compound interest on $10,000 at 5% annual rate for 10 years, compounded monthly"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")
        print("Agent: ", end="", flush=True)
        
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n")

asyncio.run(main())
Example 2: Data Analysis Agent

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool, ChatResponse
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall
)

async def analyze_data_example():
    """Agent that analyzes data using Python code."""
    client = OpenAIResponsesClient()
    
    agent = ChatAgent(
        chat_client=client,
        instructions="""You are a data analysis expert.
        
        When analyzing data:
        - Use pandas for data manipulation
        - Calculate relevant statistics
        - Identify patterns and insights
        - Present findings clearly
        
        Libraries available: pandas, numpy, matplotlib, scipy""",
        tools=[HostedCodeInterpreterTool()]
    )
    
    # Sample data scenario
    query = """I have sales data for the last 6 months:
    January: $45,000
    February: $52,000
    March: $48,000
    April: $61,000
    May: $58,000
    June: $64,000
    
    Please analyze this data and tell me:
    1. The average monthly sales
    2. The growth rate from January to June
    3. The month-over-month growth percentage
    4. Projected sales for July if the trend continues"""
    
    print(f"User: {query}\n")
    print("Agent Analysis:")
    print("-" * 60)
    
    result = await agent.run(query)
    print(result.text)
    
    # Extract the generated code if available
    if (
        isinstance(result.raw_representation, ChatResponse)
        and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
        and len(result.raw_representation.raw_representation.output) > 0
    ):
        for output in result.raw_representation.raw_representation.output:
            if isinstance(output, ResponseCodeInterpreterToolCall):
                print("\n" + "="*60)
                print("Generated Python Code:")
                print("="*60)
                print(output.code)

asyncio.run(analyze_data_example())
Example 3: Scientific Computing Agent

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient

async def scientific_computing_example():
    """Agent for scientific calculations and simulations."""
    client = OpenAIResponsesClient()
    
    agent = ChatAgent(
        chat_client=client,
        instructions="""You are a scientific computing assistant.
        
        Capabilities:
        - Physics calculations
        - Statistical analysis
        - Numerical simulations
        - Scientific visualizations
        
        Always:
        - Show units in calculations
        - Explain scientific concepts
        - Validate results for reasonableness
        - Use appropriate precision""",
        tools=[HostedCodeInterpreterTool()]
    )
    
    queries = [
        """Calculate the escape velocity from Earth.
        Use: Earth's radius = 6,371 km, Earth's mass = 5.972 × 10^24 kg, 
        Gravitational constant G = 6.674 × 10^-11 N⋅m²/kg²""",
        
        """A projectile is launched at 45 degrees with initial velocity 50 m/s.
        Calculate: maximum height, time of flight, and horizontal range.
        Use g = 9.8 m/s²""",
        
        """Calculate the half-life of a radioactive substance if 75% of the original 
        sample decays in 120 days."""
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Problem {i}:")
        print(f"{'='*70}")
        print(f"{query}\n")
        print("Solution:")
        print("-" * 70)
        
        result = await agent.run(query)
        print(result.text)

asyncio.run(scientific_computing_example())
Example 4: Interactive Problem-Solving Agent

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient

async def interactive_problem_solver():
    """Multi-turn conversation for complex problem solving."""
    client = OpenAIResponsesClient()
    
    agent = ChatAgent(
        chat_client=client,
        instructions="""You are an expert problem solver and teacher.
        
        Teaching approach:
        - Break down complex problems
        - Explain each step
        - Use code to verify solutions
        - Encourage understanding, not just answers
        
        Be patient and thorough in explanations.""",
        tools=[HostedCodeInterpreterTool()]
    )
    
    thread = agent.get_new_thread()
    
    # Simulating a multi-turn conversation
    conversation = [
        "I need to understand the probability of getting at least one 6 when rolling a dice 4 times.",
        "Can you show me the calculation step by step?",
        "What if I roll the dice 10 times instead?",
        "Can you create a simulation to verify this?"
    ]
    
    for user_message in conversation:
        print(f"\n{'='*60}")
        print(f"User: {user_message}")
        print(f"{'='*60}")
        print("Agent: ", end="", flush=True)
        
        async for chunk in agent.run_stream(user_message, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n")
        await asyncio.sleep(0.5)  # Pause between turns for readability

asyncio.run(interactive_problem_solver())
Output Handling
Accessing Generated Code

Python

from agent_framework import ChatResponse
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall
)

result = await agent.run("Calculate factorial of 100")

# Extract code from response
if (
    isinstance(result.raw_representation, ChatResponse)
    and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
):
    for output in result.raw_representation.raw_representation.output:
        if isinstance(output, ResponseCodeInterpreterToolCall):
            generated_code = output.code
            print(f"Code executed:\n{generated_code}")
Handling Code Output

Python

# The agent's response includes both code execution results and explanation
result = await agent.run("Create a list of prime numbers under 100")

# The text contains the formatted response
print(result.text)  # Includes both explanation and results

# For structured access to outputs
if isinstance(result.raw_representation, ChatResponse):
    for message in result.messages:
        for content in message.contents:
            if content.type == "text":
                print(f"Explanation: {content.text}")
Error Handling in Code Execution

Python

# Agent handles code errors gracefully
query = "Calculate 1/0"  # Will cause division by zero

result = await agent.run(query)
# Agent will catch the error and explain it to the user
print(result.text)
# Output: "I encountered a division by zero error. Let me explain why..."
Best Practices & Limitations
Best Practices:

1. Provide Clear Problem Specifications

✅ Good:

Python

query = """Calculate the volume of a sphere with radius 5 cm.
Use the formula V = (4/3)πr³
Express the answer in cubic centimeters with 2 decimal places."""
❌ Vague:

Python

query = "What's the volume of a sphere?"
2. Request Step-by-Step Solutions

Python

query = """Solve this problem step by step:
1. First, explain the approach
2. Then write the code
3. Finally, explain the result

Problem: Find all prime numbers between 1 and 100."""
3. Verify Critical Calculations

Python

# For important calculations, ask for verification
query = """Calculate the loan payment for:
- Principal: $250,000
- Annual interest rate: 3.5%
- Term: 30 years
- Monthly payments

Please verify the calculation using the standard mortgage formula."""
Limitations:

1. No Persistent State Between Runs

Python

# Each run starts fresh - no persistent variables
result1 = await agent.run("Create a variable x = 10")
result2 = await agent.run("What is x + 5?")  # Won't remember x

# Solution: Include all context in one query
result = await agent.run("""
Create a variable x = 10
Then calculate x + 5
""")
2. Limited External Libraries

Python

# Available: numpy, pandas, matplotlib, scipy, math
# Not available: requests, custom packages

# ✅ Works
await agent.run("Use numpy to create an array")

# ❌ Won't work
await agent.run("Use requests to fetch data from an API")
3. No File Persistence

Python

# Files created during execution don't persist between runs
# Each execution starts with a clean environment
4. Execution Time Limits

Python

# Very long-running computations may time out
# Break down large problems into smaller chunks

# ❌ May timeout
query = "Calculate prime numbers up to 10 million"

# ✅ Better approach
query = "Calculate prime numbers up to 10,000 and show the pattern"
5. No Interactive Input

Python

# Code cannot use input() or other interactive features
# All data must be provided in the query
Optimization Tips:

Python

# 1. Batch related calculations
query = """Calculate for me:
1. Factorial of 20
2. Factorial of 30
3. Factorial of 40
Compare the growth rate."""

# 2. Use efficient algorithms
query = """Find prime numbers under 1000.
Use the Sieve of Eratosthenes algorithm for efficiency."""

# 3. Request specific output formats
query = """Analyze this data and present results as:
- Summary statistics in a table
- Key insights in bullet points
- Specific recommendations"""
7. Agent Type 5: Multi-Modal Agent
Use Case & Overview
Multi-Modal Agents integrate multiple capabilities - vision, web search, reasoning, and MCP tools - to handle complex, diverse tasks. They excel at:

Image analysis and interpretation
Real-time information retrieval
Complex reasoning tasks
Integration with external tools and services
Cross-domain problem solving
Key Characteristics:

Process multiple input types (text, images, URLs)
Access real-time information via web search
Use advanced reasoning capabilities
Connect to external services via MCP
Combine multiple tools in a single workflow
Real-World Examples:

Visual customer support (analyze product images)
Research assistant (web search + document analysis)
Content creation (image analysis + web trends)
Technical documentation helper (screenshots + code search)
Market research analyst (web data + reasoning)
Image Analysis
Example 1: Basic Image Analysis

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def analyze_image_from_url():
    """Analyze an image from a URL."""
    client = OpenAIResponsesClient()
    
    agent = client.create_agent(
        name="VisionAgent",
        instructions="""You are an expert image analyst.
        
        When analyzing images:
        - Describe what you see in detail
        - Identify key elements and their relationships
        - Note colors, composition, and style
        - Provide relevant context or insights"""
    )
    
    # Create message with text and image
    message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="Please analyze this image and describe what you see."),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                media_type="image/jpeg"
            )
        ]
    )
    
    print("User: [Provided image] Please analyze this image\n")
    print("Agent: ", end="", flush=True)
    
    async for chunk in agent.run_stream(message):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    
    print("\n")

asyncio.run(analyze_image_from_url())
Example 2: Multi-Image Comparison

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def compare_images():
    """Compare multiple images."""
    client = OpenAIResponsesClient()
    
    agent = client.create_agent(
        name="ImageComparison",
        instructions="""You are an expert at comparing and analyzing multiple images.
        
        When comparing images:
        - Identify similarities and differences
        - Note style, composition, and subject matter
        - Provide detailed comparative analysis
        - Draw insights from the comparison"""
    )
    
    message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="Compare these two landscape images. What are the key differences in composition and mood?"),
            UriContent(
                uri="https://example.com/landscape1.jpg",
                media_type="image/jpeg"
            ),
            UriContent(
                uri="https://example.com/landscape2.jpg",
                media_type="image/jpeg"
            )
        ]
    )
    
    result = await agent.run(message)
    print(f"Analysis:\n{result.text}")

# asyncio.run(compare_images())
Example 3: Visual Problem Solving

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def visual_problem_solving():
    """Solve problems based on visual input."""
    client = OpenAIResponsesClient()
    
    agent = client.create_agent(
        name="VisualProblemSolver",
        instructions="""You are a visual problem-solving assistant.
        
        Capabilities:
        - Analyze diagrams and charts
        - Extract text from images (OCR)
        - Interpret screenshots
        - Solve visual puzzles
        - Explain visual data"""
    )
    
    # Example: Analyzing a chart or diagram
    message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="What insights can you derive from this sales chart? What trends do you notice?"),
            UriContent(
                uri="https://example.com/sales_chart.png",
                media_type="image/png"
            )
        ]
    )
    
    result = await agent.run(message)
    print(result.text)

# asyncio.run(visual_problem_solving())
Web Search Integration
Example 1: Real-Time Information Retrieval

Python

import asyncio
from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def web_search_agent():
    """Agent with web search capabilities."""
    client = OpenAIResponsesClient()
    
    agent = client.create_agent(
        name="ResearchAgent",
        instructions="""You are a research assistant with access to real-time web search.
        
        When answering questions:
        - Use web search for current information
        - Cite your sources
        - Verify information from multiple sources when possible
        - Distinguish between factual information and opinions"""
    )
    
    # Configure web search with user location
    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "user_location": {
                "country": "US",
                "city": "Seattle"
            }
        }
    )
    
    queries = [
        "What's the current weather in Seattle?",
        "What are the latest developments in quantum computing?",
        "Who won the most recent Nobel Prize in Physics?"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        print("Agent: ", end="", flush=True)
        
        async for chunk in agent.run_stream(query, tools=[web_search_tool]):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n")

asyncio.run(web_search_agent())
Example 2: Research Assistant with Web Search

Python

import asyncio
from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def research_assistant():
    """Comprehensive research assistant."""
    client = OpenAIResponsesClient()
    
    agent = client.create_agent(
        name="ResearchAssistant",
        instructions="""You are an advanced research assistant.
        
        Research methodology:
        - Search for current, authoritative sources
        - Cross-reference information
        - Provide balanced perspectives
        - Cite all sources
        - Distinguish facts from analysis
        
        Present findings in a structured, academic format.""",
        tools=[HostedWebSearchTool()]
    )
    
    research_query = """Research the impact of artificial intelligence on healthcare.
    
    Please provide:
    1. Current applications of AI in healthcare
    2. Recent breakthroughs or developments
    3. Challenges and limitations
    4. Future outlook
    
    Include specific examples and cite your sources."""
    
    print(f"Research Query:\n{research_query}\n")
    print("="*70)
    print("Research Report:")
    print("="*70)
    
    result = await agent.run(research_query)
    print(result.text)

asyncio.run(research_assistant())
MCP Tool Integration
Example 1: Local MCP Server

Python

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIResponsesClient

async def mcp_documentation_agent():
    """Agent with MCP tool for documentation search."""
    client = OpenAIResponsesClient()
    
    async with ChatAgent(
        chat_client=client,
        name="DocsAgent",
        instructions="""You are a helpful documentation assistant.
        Use the Microsoft Learn MCP server to find accurate, up-to-date documentation.
        
        When answering:
        - Search the documentation first
        - Provide specific, actionable guidance
        - Include relevant links
        - Offer code examples when available""",
        tools=MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp"
        )
    ) as agent:
        
        queries = [
            "How do I create an Azure storage account using Azure CLI?",
            "What is the Microsoft Agent Framework and how do I get started?",
            "Show me how to deploy a web app to Azure App Service"
        ]
        
        thread = agent.get_new_thread()
        
        for query in queries:
            print(f"\n{'='*70}")
            print(f"User: {query}")
            print(f"{'='*70}")
            print("Agent: ", end="", flush=True)
            
            async for chunk in agent.run_stream(query, thread=thread):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            
            print("\n")

asyncio.run(mcp_documentation_agent())
Example 2: Hosted MCP with Approval Workflow

Python

import asyncio
from typing import Any
from agent_framework import ChatAgent, HostedMCPTool, ChatMessage
from agent_framework.openai import OpenAIResponsesClient

async def handle_approvals(query: str, agent, thread):
    """Handle approval workflow for MCP function calls."""
    result = await agent.run(query, thread=thread, store=True)
    
    while len(result.user_input_requests) > 0:
        new_inputs = []
        
        for user_input_needed in result.user_input_requests:
            print(f"\n🔔 Approval Required:")
            print(f"Function: {user_input_needed.function_call.name}")
            print(f"Arguments: {user_input_needed.function_call.arguments}")
            
            user_approval = input("\nApprove this function call? (y/n): ")
            
            new_inputs.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                )
            )
        
        result = await agent.run(new_inputs, thread=thread, store=True)
    
    return result

async def mcp_with_approvals():
    """MCP agent with user approval workflow."""
    client = OpenAIResponsesClient()
    
    async with ChatAgent(
        chat_client=client,
        name="SecureDocsAgent",
        instructions="You are a documentation assistant with approval-gated tool access.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            approval_mode="always_require"  # Require approval for all calls
        )
    ) as agent:
        
        thread = agent.get_new_thread()
        
        query = "Search for information about Azure Functions"
        print(f"User: {query}")
        
        result = await handle_approvals(query, agent, thread)
        print(f"\nAgent: {result.text}")

asyncio.run(mcp_with_approvals())
Complete Multi-Capability Example
Example: Advanced Multi-Modal Research Agent

Python

import asyncio
from agent_framework import (
    ChatAgent,
    ChatMessage,
    TextContent,
    UriContent,
    HostedWebSearchTool,
    HostedCodeInterpreterTool,
    MCPStreamableHTTPTool
)
from agent_framework.openai import OpenAIResponsesClient

async def multi_modal_research_agent():
    """
    Comprehensive agent combining:
    - Image analysis
    - Web search
    - Code execution
    - MCP tool integration
    """
    client = OpenAIResponsesClient()
    
    async with ChatAgent(
        chat_client=client,
        name="AdvancedResearchAgent",
        instructions="""You are an advanced research and analysis agent.
        
        Capabilities:
        - Analyze images and visual content
        - Search the web for current information
        - Execute Python code for data analysis
        - Access documentation via MCP tools
        
        Approach:
        - Use the most appropriate tool for each task
        - Combine information from multiple sources
        - Provide comprehensive, well-reasoned answers
        - Show your work and cite sources""",
        tools=[
            HostedWebSearchTool(),
            HostedCodeInterpreterTool(),
            MCPStreamableHTTPTool(
                name="Microsoft Learn MCP",
                url="https://learn.microsoft.com/api/mcp"
            )
        ]
    ) as agent:
        
        thread = agent.get_new_thread()
        
        # Task 1: Image analysis with web context
        print("\n" + "="*70)
        print("TASK 1: Image Analysis with Web Context")
        print("="*70)
        
        task1 = ChatMessage(
            role="user",
            contents=[
                TextContent(text="Analyze this architectural structure and search for information about similar buildings."),
                UriContent(
                    uri="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_York_City_view.jpg/1200px-New_York_City_view.jpg",
                    media_type="image/jpeg"
                )
            ]
        )
        
        result1 = await agent.run(task1, thread=thread)
        print(f"\nAgent: {result1.text}")
        
        # Task 2: Data analysis with web research
        print("\n" + "="*70)
        print("TASK 2: Data Analysis with Web Research")
        print("="*70)
        
        task2 = """What's the current market cap of the top 5 tech companies?
        Then calculate the total market cap and each company's percentage of the total.
        Present the results in a formatted table."""
        
        print(f"\nUser: {task2}")
        print("\nAgent: ", end="", flush=True)
        
        async for chunk in agent.run_stream(task2, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n")
        
        # Task 3: Technical documentation query
        print("\n" + "="*70)
        print("TASK 3: Technical Documentation Query")
        print("="*70)
        
        task3 = """How would I deploy a Python web application to Azure?
        Provide step-by-step instructions with CLI commands."""
        
        print(f"\nUser: {task3}")
        print("\nAgent: ", end="", flush=True)
        
        async for chunk in agent.run_stream(task3, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n")

asyncio.run(multi_modal_research_agent())
Best Practices
1. Choose the Right Tool for Each Task

Python

# ✅ Use appropriate tools for specific tasks
agent = client.create_agent(
    instructions="Use the best tool for each task.",
    tools=[
        HostedWebSearchTool(),  # For current events, real-time data
        HostedCodeInterpreterTool(),  # For calculations, data analysis
        HostedFileSearchTool(),  # For document-based questions
    ]
)
2. Provide Clear Instructions for Tool Usage

Python

instructions = """You are a multi-modal research assistant.

Tool usage guidelines:
- Use web search for: current events, recent data, fact-checking
- Use code interpreter for: calculations, data analysis, visualizations
- Use file search for: company policies, internal documentation
