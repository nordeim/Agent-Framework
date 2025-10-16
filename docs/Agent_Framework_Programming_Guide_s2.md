Microsoft Agent Framework: Comprehensive Programming Guide
A practical guide to building 5 types of AI agents with the Microsoft Agent Framework

Table of Contents
Introduction

What is the Microsoft Agent Framework?
Framework Architecture
Client Types Comparison
Environment Setup
Your First Agent
Core Concepts

Understanding Client Types
Agent Lifecycle Management
Thread Management Patterns
Tool Integration Approaches
Streaming vs Non-Streaming Responses
Error Handling Patterns
Agent Type 1: Basic Conversational Agent

Use Cases
Implementation
Configuration Options
Best Practices
Agent Type 2: Function-Calling Agent

Use Cases
Tool Definition Patterns
Implementation Examples
Advanced Function Patterns
Best Practices
Agent Type 3: RAG Agent (Knowledge Retrieval)

Use Cases
Vector Store Setup
Complete Implementation
Query Optimization
Best Practices
Agent Type 4: Code Execution Agent

Use Cases
Security Considerations
Implementation
Output Handling
Best Practices
Agent Type 5: Multi-Modal Agent

Use Cases
Image Analysis
Web Search Integration
MCP Tool Integration
Complete Multi-Capability Example
Best Practices
Advanced Topics

Thread Persistence Strategies
Structured Outputs with Pydantic
Approval Workflows
Performance Optimization
Best Practices & Patterns

Resource Management
Error Handling & Retry Logic
Production Considerations
Security Best Practices
Troubleshooting Guide

Common Errors
Debugging Techniques
Performance Issues
Quick Reference

Next Steps

Introduction
What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, flexible Python framework for building AI agents that can interact with users, call functions, access knowledge bases, execute code, and integrate with external services. It provides a unified interface for working with various AI models from OpenAI and Azure, enabling developers to build sophisticated conversational AI applications quickly and reliably.

Key capabilities include:

Multiple Client Types: Choose from Assistants, Chat, or Responses clients based on your needs
Tool Integration: Seamlessly connect agents to custom functions, APIs, and external services
Knowledge Retrieval: Build RAG (Retrieval-Augmented Generation) systems with file search capabilities
Code Execution: Enable agents to write and run Python code dynamically
Multi-Modal Support: Process images, search the web, and combine multiple capabilities
Thread Management: Maintain conversation context across multiple interactions
Streaming Support: Get real-time responses as they're generated
Production-Ready: Built-in error handling, resource management, and best practices
Framework Architecture
The framework follows a layered architecture that separates concerns and provides flexibility:

text

┌─────────────────────────────────────────────────────┐
│              Your Application Layer                  │
│        (Business Logic, UI, Orchestration)          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│               Agent Layer (ChatAgent)                │
│    (Conversation Management, Tool Orchestration)    │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              Client Layer                            │
│  (OpenAIAssistantsClient | OpenAIChatClient |       │
│   OpenAIResponsesClient | Azure Equivalents)        │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│           AI Service Layer                           │
│      (OpenAI API | Azure OpenAI Service)            │
└─────────────────────────────────────────────────────┘
This architecture allows you to:

Swap AI providers without changing agent code
Test agents with mock clients
Add custom middleware and logging
Scale from prototype to production seamlessly
Client Types Comparison
The framework offers three main client types, each optimized for different scenarios:

Feature	AssistantsClient	ChatClient	ResponsesClient
Best For	Stateful conversations with built-in thread management	Lightweight chat interactions with full control	Structured outputs and advanced features
Thread Storage	Server-side (automatic)	Client-side (in-memory)	Client-side or server-side
State Management	Automatic	Manual	Manual or automatic
Function Calling	✅ Yes	✅ Yes	✅ Yes
File Search	✅ Yes	❌ No	✅ Yes
Code Interpreter	✅ Yes	❌ No	✅ Yes
Structured Outputs	❌ Limited	❌ Limited	✅ Full support
Image Analysis	❌ Limited	✅ Yes	✅ Yes
Web Search	❌ No	✅ Yes	✅ Yes
Reasoning Models	❌ No	❌ No	✅ Yes (gpt-5)
Setup Complexity	Medium	Low	Medium
Resource Cleanup	Automatic	Manual	Manual
Use Case	Customer support, long conversations	Chatbots, quick Q&A	Data extraction, complex workflows
Selection Guide:

Use AssistantsClient when you need automatic conversation state management and built-in tools
Use ChatClient for simple, stateless interactions or when you need full control over message history
Use ResponsesClient for advanced features like structured outputs, reasoning models, or complex multi-tool scenarios
Environment Setup
Before building agents, configure your environment with the necessary credentials:

For OpenAI:
Bash

# Required
export OPENAI_API_KEY="sk-..."
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"

# Optional
export OPENAI_ORG_ID="org-..."
export OPENAI_API_BASE_URL="https://api.openai.com/v1"
For Azure OpenAI:
Bash

# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="gpt-4o"

# Authentication (choose one)
# Option 1: API Key
export AZURE_OPENAI_API_KEY="your-key"

# Option 2: Azure AD (Managed Identity)
# No additional environment variables needed
Installation:
Bash

# Using pip
pip install agent-framework

# Using uv (recommended for faster installs)
uv pip install agent-framework

# For specific providers
pip install agent-framework[openai]  # OpenAI support
pip install agent-framework[azure]   # Azure support
Your First Agent
Let's create a simple weather agent to understand the basics:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

# Define a simple function the agent can call
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # In a real application, this would call a weather API
    return f"The weather in {location} is sunny with a high of 75°F."

async def main():
    # Create an agent with the chat client
    agent = OpenAIChatClient().create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather assistant.",
        tools=[get_weather]
    )
    
    # Ask a question
    query = "What's the weather like in Seattle?"
    result = await agent.run(query)
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
Output:

text

Agent: The weather in Seattle is sunny with a high of 75°F.
What happened here?

We created an agent using OpenAIChatClient
We provided a custom get_weather function as a tool
The agent understood the user's question
It called the get_weather function with "Seattle" as the location
It formatted the response naturally
This simple example demonstrates the core power of the framework: seamless integration between AI reasoning and custom business logic.

Core Concepts
Understanding Client Types
AssistantsClient Deep Dive
The AssistantsClient wraps the OpenAI Assistants API, providing server-side conversation management:

Python

from agent_framework.openai import OpenAIAssistantsClient

async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=[my_function]
) as agent:
    # Agent automatically creates an assistant on the server
    result = await agent.run("Hello!")
    # Assistant is automatically deleted when context exits
Key characteristics:

Automatic lifecycle: Assistants are created and cleaned up automatically
Server-side threads: Conversation history stored on OpenAI servers
Built-in tools: Code interpreter and file search available out-of-the-box
Persistent state: Threads can be referenced by ID across sessions
When to use:

Multi-session conversations (customer support)
When you need built-in code execution
When server-side state management is preferred
ChatClient Deep Dive
The ChatClient provides direct access to chat completion models with full control:

Python

from agent_framework.openai import OpenAIChatClient

agent = OpenAIChatClient().create_agent(
    name="MyAgent",
    instructions="You are a helpful assistant.",
    tools=[my_function]
)

# Messages stored in memory, you control the history
result = await agent.run("Hello!")
Key characteristics:

Lightweight: No server-side resources created
Full control: You manage message history
Flexible: Easy to customize and extend
In-memory: Threads stored locally by default
When to use:

Simple Q&A interactions
When you need custom message storage
Stateless applications
Maximum flexibility required
ResponsesClient Deep Dive
The ResponsesClient leverages advanced OpenAI features for structured interactions:

Python

from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

class WeatherData(BaseModel):
    location: str
    temperature: int
    condition: str

agent = OpenAIResponsesClient().create_agent(
    instructions="Extract weather data from user queries.",
    tools=[my_function]
)

# Get structured output
result = await agent.run("It's 75°F and sunny in Seattle", 
                         response_format=WeatherData)
structured_data: WeatherData = result.value
Key characteristics:

Structured outputs: Native Pydantic model support
Advanced features: Reasoning, image generation, detailed control
Flexible storage: Client or server-side threads
Rich tool support: MCP, web search, file search, code interpreter
When to use:

Data extraction pipelines
Complex multi-tool workflows
When you need structured, validated outputs
Advanced reasoning tasks
Agent Lifecycle Management
Proper lifecycle management ensures resources are cleaned up and connections are handled correctly:

Python

# ✅ RECOMMENDED: Using context managers
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    result = await agent.run("Hello")
    # Agent and resources automatically cleaned up

# ✅ ALSO GOOD: Manual cleanup
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)
try:
    result = await agent.run("Hello")
finally:
    # Cleanup if needed (ChatClient doesn't require it)
    pass

# ❌ AVOID: No cleanup for AssistantsClient
# This leaves resources on the server
agent = OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
)
result = await agent.run("Hello")
# Assistant not deleted!
Best practices:

Always use async with for AssistantsClient
ChatClient and ResponsesClient don't require cleanup but support it
For long-running applications, implement proper shutdown handlers
Thread Management Patterns
Threads manage conversation context. The framework supports multiple patterns:

Pattern 1: Automatic Thread Creation (Stateless)
Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Each call creates a new thread (no shared context)
result1 = await agent.run("My name is Alice")
result2 = await agent.run("What's my name?")
# Agent won't know the name (different threads)
Pattern 2: Explicit Thread (In-Memory State)
Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Create a thread to maintain context
thread = agent.get_new_thread()

result1 = await agent.run("My name is Alice", thread=thread)
result2 = await agent.run("What's my name?", thread=thread)
# Agent responds: "Your name is Alice" (shared context)
Pattern 3: Server-Side Thread (Persistent State)
Python

async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    thread = agent.get_new_thread()
    
    result1 = await agent.run("My name is Alice", thread=thread)
    
    # Save thread ID for later
    thread_id = thread.service_thread_id
    print(f"Save this ID: {thread_id}")

# Later, in a different session:
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    # Restore thread from ID
    from agent_framework import AgentThread
    thread = AgentThread(service_thread_id=thread_id)
    
    result2 = await agent.run("What's my name?", thread=thread)
    # Agent responds: "Your name is Alice" (persistent state)
Pattern 4: ResponsesClient with Server Storage
Python

agent = OpenAIResponsesClient().create_agent(
    instructions="You are helpful."
)

thread = agent.get_new_thread()

# Enable server-side storage with store=True
result1 = await agent.run("My name is Alice", thread=thread, store=True)

# Thread now has server-side ID
thread_id = thread.service_thread_id

# Later session
thread = AgentThread(service_thread_id=thread_id)
result2 = await agent.run("What's my name?", thread=thread, store=True)
Thread management best practices:

Use in-memory threads for temporary conversations
Use server-side threads for multi-session persistence
Always save thread IDs when you need to resume conversations
Consider implementing custom message stores for database persistence
Tool Integration Approaches
Tools extend agent capabilities by connecting to external functions and services. The framework supports two integration patterns:

Agent-Level Tools
Tools defined when creating the agent are available for all queries:

Python

def get_weather(location: str) -> str:
    return f"Weather in {location}: sunny"

def get_time() -> str:
    return "Current time: 10:00 AM"

# Tools available for all agent queries
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather, get_time]  # Defined at agent level
)

# Agent can use both tools in any query
await agent.run("What's the weather and time?")
Use when:

Tools are core to the agent's purpose
Tools should always be available
Building specialized agents (e.g., always needs database access)
Run-Level Tools
Tools can also be provided per query for dynamic capabilities:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    # No tools defined here
)

# Provide tools for specific queries
await agent.run(
    "What's the weather?", 
    tools=[get_weather]  # Only available for this query
)

await agent.run(
    "What's the time?",
    tools=[get_time]  # Different tool for this query
)
Use when:

Tools vary per query
Implementing dynamic tool selection
Security: limiting tool access per request
Multi-tenant systems with different capabilities
Combining Both Patterns
Python

# Base tools always available
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather]  # Always available
)

# Add additional tools per query
await agent.run(
    "Weather and time?",
    tools=[get_time]  # Added for this query
)
# Agent has access to both get_weather and get_time
Streaming vs Non-Streaming Responses
The framework supports both streaming and non-streaming response patterns:

Non-Streaming (Complete Response)
Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Wait for complete response
result = await agent.run("Tell me a story")
print(result.text)  # Full story printed at once
Advantages:

Simpler code
Easier error handling
Better for short responses
Use when:

Responses are short
You need the complete result before processing
Building batch processing systems
Streaming (Real-Time Response)
Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Stream response as it's generated
print("Agent: ", end="", flush=True)
async for chunk in agent.run_stream("Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()  # New line when done
Advantages:

Better user experience (immediate feedback)
Lower perceived latency
Can process response incrementally
Use when:

Building interactive UIs
Responses are long
User experience is critical
You want to display progress
Streaming with Structured Outputs
Python

from agent_framework import AgentRunResponse
from pydantic import BaseModel

class Story(BaseModel):
    title: str
    content: str

agent = OpenAIResponsesClient().create_agent(
    instructions="You are a storyteller."
)

# Stream and collect into structured format
result = await AgentRunResponse.from_agent_response_generator(
    agent.run_stream("Tell me a story", response_format=Story),
    output_format_type=Story
)

story: Story = result.value
print(f"Title: {story.title}")
print(f"Content: {story.content}")
Error Handling Patterns
Robust error handling is crucial for production agents:

Python

from openai import APIError, RateLimitError
import asyncio

async def run_agent_with_retry(agent, query, max_retries=3):
    """Run agent with automatic retry on rate limits."""
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            return result
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
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
    instructions="You are helpful."
)

try:
    result = await run_agent_with_retry(agent, "Hello")
except Exception as e:
    print(f"Failed after retries: {e}")
Best practices:

Implement exponential backoff for rate limits
Log errors with context for debugging
Use specific exception types for different error scenarios
Always have a fallback strategy
Consider circuit breaker patterns for production
Agent Type 1: Basic Conversational Agent
Conversational Agent Use Cases
Basic conversational agents are perfect for:

Customer support: Answer FAQs, provide information
Virtual assistants: Help with scheduling, reminders, general queries
Educational tutors: Explain concepts, answer questions
Information retrieval: Search documentation, explain features
Simple chatbots: Engage users, provide guidance
Conversational Agent Implementation
Let's build a customer support agent for a fictional online bookstore:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create a customer support agent
    agent = OpenAIChatClient().create_agent(
        name="BookstoreSupportAgent",
        instructions="""You are a helpful customer support agent for "PageTurner Books," 
        an online bookstore. You can help customers with:
        - Book recommendations
        - Order status questions
        - Return and refund policies
        - Store hours and location information
        
        Always be polite, professional, and helpful. If you don't know something,
        admit it and offer to connect the customer with a human agent."""
    )
    
    # Simulate a customer conversation
    print("=== PageTurner Books Customer Support ===\n")
    
    # Create a thread to maintain conversation context
    thread = agent.get_new_thread()
    
    # Conversation flow
    queries = [
        "Hi, I'm looking for a good science fiction book",
        "Something like Dune",
        "What's your return policy?",
        "Thanks for your help!"
    ]
    
    for query in queries:
        print(f"Customer: {query}")
        result = await agent.run(query, thread=thread)
        print(f"Agent: {result.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
Key features demonstrated:

Context maintenance: Using threads to remember conversation history
Clear instructions: Defining agent behavior and boundaries
Natural conversation: Multi-turn dialogue that flows naturally
Conversational Agent Configuration
You can customize agent behavior through various configuration options:

Python

from agent_framework.openai import OpenAIChatClient

# Basic configuration
agent = OpenAIChatClient(
    model_id="gpt-4o",  # Specify model
    api_key="your-api-key",  # Override environment variable
).create_agent(
    name="ConfiguredAgent",
    instructions="You are a helpful assistant.",
)

# Advanced configuration with additional options
agent = OpenAIChatClient().create_agent(
    name="AdvancedAgent",
    instructions="""You are a technical documentation assistant.
    - Provide code examples when helpful
    - Explain concepts clearly
    - Ask clarifying questions if needed""",
    
    # Additional chat-specific options
    additional_chat_options={
        "temperature": 0.7,  # Control randomness (0.0-2.0)
        "max_tokens": 1000,  # Limit response length
        "top_p": 0.9,  # Nucleus sampling
    }
)
Configuration parameters:

temperature: Controls randomness (0.0 = deterministic, 1.0+ = creative)
max_tokens: Limits response length
top_p: Controls diversity via nucleus sampling
model_id: Specify which model to use
Conversational Agent Best Practices
1. Write Clear, Specific Instructions
Python

# ❌ Too vague
instructions = "You are helpful."

# ✅ Clear and specific
instructions = """You are a financial advisor assistant.

Your role:
- Help users understand investment concepts
- Explain financial terms in simple language
- Provide general education, NOT specific investment advice

Guidelines:
- Always include disclaimers for financial information
- Ask clarifying questions before giving detailed explanations
- Suggest consulting certified financial advisors for specific decisions
- Be clear, accurate, and unbiased"""
2. Implement Proper Thread Management
Python

# For stateless interactions (each query independent)
async def stateless_chat():
    agent = OpenAIChatClient().create_agent(
        instructions="You are helpful."
    )
    
    # Each call is independent
    await agent.run("Question 1")
    await agent.run("Question 2")

# For conversations with context
async def stateful_chat():
    agent = OpenAIChatClient().create_agent(
        instructions="You are helpful."
    )
    
    thread = agent.get_new_thread()
    
    # Maintain context across calls
    await agent.run("My name is Alice", thread=thread)
    await agent.run("What's my name?", thread=thread)
3. Handle Edge Cases
Python

async def robust_conversation():
    agent = OpenAIChatClient().create_agent(
        instructions="""You are helpful. If the user's message is unclear,
        ask for clarification. If you don't know something, admit it."""
    )
    
    thread = agent.get_new_thread()
    
    try:
        result = await agent.run("", thread=thread)
    except ValueError:
        print("Empty message not allowed")
        return
    
    # Check for ambiguous responses
    if result.text and len(result.text) < 10:
        print("Warning: Very short response, may need clarification")
4. Use Streaming for Better UX
Python

async def streaming_conversation():
    agent = OpenAIChatClient().create_agent(
        name="StreamingAgent",
        instructions="You are a storytelling assistant."
    )
    
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream("Tell me a short story"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # Newline when complete
Common pitfalls to avoid:

Not maintaining threads for multi-turn conversations
Overly generic instructions that don't guide behavior
Ignoring error cases and edge conditions
Not using streaming for long responses
Hardcoding API keys instead of using environment variables
Agent Type 2: Function-Calling Agent
Function-Calling Use Cases
Function-calling agents bridge AI reasoning with your business logic:

E-commerce: Check inventory, process orders, update cart
CRM systems: Retrieve customer data, update records, create tickets
IoT control: Turn on lights, adjust thermostats, check sensors
Database queries: Search records, generate reports, update entries
API integration: Call external services, process data, orchestrate workflows
Tool Definition Patterns
The framework supports multiple ways to define tools for agents:

Pattern 1: Simple Function with Type Annotations
Python

from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(description="The city name, e.g., 'Seattle'")],
    unit: Annotated[str, Field(description="Temperature unit: 'celsius' or 'fahrenheit'")] = "fahrenheit"
) -> str:
    """Get the current weather for a given location.
    
    This function retrieves weather data including temperature and conditions.
    """
    # In production, call a real weather API
    return f"The weather in {location} is sunny, 72°{unit[0].upper()}"

# The framework automatically converts this to a tool schema
Key points:

Use Annotated with Field for parameter descriptions
Docstring becomes the tool description
Default values are supported
Return type should be string or JSON-serializable
Pattern 2: Pydantic Models for Complex Parameters
Python

from pydantic import BaseModel, Field
from typing import Optional

class OrderRequest(BaseModel):
    """Request to create a new order."""
    customer_id: str = Field(description="Unique customer identifier")
    product_id: str = Field(description="Product to order")
    quantity: int = Field(description="Number of items", ge=1)
    priority: Optional[str] = Field(description="Order priority: normal, high, urgent", default="normal")

def create_order(request: OrderRequest) -> str:
    """Create a new order in the system."""
    # Process order
    return f"Order created for customer {request.customer_id}: {request.quantity}x {request.product_id}"

# Agent can call this with structured data
Pattern 3: Multiple Return Types
Python

import json
from typing import Dict, Any

def search_products(query: str, max_results: int = 5) -> str:
    """Search for products matching the query."""
    # In production, query your database
    results = [
        {"id": "123", "name": "Python Programming Book", "price": 29.99},
        {"id": "456", "name": "Python for Data Science", "price": 39.99}
    ]
    
    # Return JSON string for complex data
    return json.dumps(results[:max_results])

# Agent receives JSON string, can parse and use data
Function-Calling Implementation
Let's build a complete e-commerce assistant with multiple tools:

Python

import asyncio
from typing import Annotated, Optional
from pydantic import Field
import json
from agent_framework.openai import OpenAIChatClient

# Simulated database
PRODUCTS = {
    "p001": {"name": "Laptop", "price": 999.99, "stock": 5},
    "p002": {"name": "Mouse", "price": 29.99, "stock": 50},
    "p003": {"name": "Keyboard", "price": 79.99, "stock": 0},
}

ORDERS = {}

def search_products(
    query: Annotated[str, Field(description="Product search query")],
    max_results: Annotated[int, Field(description="Maximum results to return")] = 5
) -> str:
    """Search for products in the catalog."""
    results = []
    query_lower = query.lower()
    
    for pid, product in PRODUCTS.items():
        if query_lower in product["name"].lower():
            results.append({
                "id": pid,
                "name": product["name"],
                "price": product["price"],
                "in_stock": product["stock"] > 0
            })
    
    return json.dumps(results[:max_results])

def check_stock(
    product_id: Annotated[str, Field(description="Product ID to check")]
) -> str:
    """Check inventory stock for a product."""
    if product_id not in PRODUCTS:
        return f"Product {product_id} not found"
    
    stock = PRODUCTS[product_id]["stock"]
    name = PRODUCTS[product_id]["name"]
    
    if stock > 10:
        return f"{name} is in stock ({stock} units available)"
    elif stock > 0:
        return f"{name} has limited stock ({stock} units remaining)"
    else:
        return f"{name} is out of stock"

def create_order(
    customer_id: Annotated[str, Field(description="Customer ID")],
    product_id: Annotated[str, Field(description="Product ID to order")],
    quantity: Annotated[int, Field(description="Quantity to order", ge=1)]
) -> str:
    """Create a new order for a customer."""
    if product_id not in PRODUCTS:
        return "Error: Product not found"
    
    product = PRODUCTS[product_id]
    
    if product["stock"] < quantity:
        return f"Error: Insufficient stock. Only {product['stock']} units available"
    
    # Create order
    order_id = f"ORD{len(ORDERS) + 1:04d}"
    ORDERS[order_id] = {
        "customer_id": customer_id,
        "product_id": product_id,
        "quantity": quantity,
        "total": product["price"] * quantity
    }
    
    # Update stock
    PRODUCTS[product_id]["stock"] -= quantity
    
    return f"Order {order_id} created successfully! Total: ${ORDERS[order_id]['total']:.2f}"

def get_order_status(
    order_id: Annotated[str, Field(description="Order ID to check")]
) -> str:
    """Get the status of an order."""
    if order_id not in ORDERS:
        return f"Order {order_id} not found"
    
    order = ORDERS[order_id]
    product = PRODUCTS[order["product_id"]]
    
    return json.dumps({
        "order_id": order_id,
        "product": product["name"],
        "quantity": order["quantity"],
        "total": order["total"],
        "status": "confirmed"
    })

async def main():
    # Create agent with all tools
    agent = OpenAIChatClient().create_agent(
        name="ShopAssistant",
        instructions="""You are a helpful e-commerce shopping assistant.

        You can help customers:
        - Search for products
        - Check product availability
        - Place orders
        - Check order status
        
        Always confirm details before placing orders. Be helpful and friendly.""",
        tools=[
            search_products,
            check_stock,
            create_order,
            get_order_status
        ]
    )
    
    # Simulate shopping conversation
    thread = agent.get_new_thread()
    
    print("=== E-Commerce Shopping Assistant ===\n")
    
    queries = [
        "I'm looking for a laptop",
        "Is it in stock?",
        "I'd like to order one. My customer ID is CUST001",
        "What's my order status?"
    ]
    
    for query in queries:
        print(f"Customer: {query}")
        result = await agent.run(query, thread=thread)
        print(f"Assistant: {result.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
This example demonstrates:

Multiple tool integration
Stateful operations (inventory management)
Error handling in functions
Complex multi-step workflows
JSON return values for structured data
Advanced Function Patterns
Dynamic Tool Selection
Python

def get_tools_for_user(user_role: str) -> list:
    """Return different tools based on user role."""
    base_tools = [search_products, check_stock]
    
    if user_role == "customer":
        return base_tools + [create_order]
    elif user_role == "admin":
        return base_tools + [create_order, cancel_order, update_inventory]
    else:
        return base_tools

# Create agent with role-specific tools
user_role = "customer"
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=get_tools_for_user(user_role)
)
Async Functions
Python

import httpx

async def fetch_live_weather(
    location: Annotated[str, Field(description="City name")]
) -> str:
    """Fetch real-time weather data from an API."""
    async with httpx.AsyncClient() as client:
        # Example API call (replace with real API)
        response = await client.get(
            f"https://api.weather.com/v1/current",
            params={"location": location}
        )
        data = response.json()
        return f"Temperature: {data['temp']}°F, Conditions: {data['conditions']}"

# Agent can call async functions
agent = OpenAIChatClient().create_agent(
    instructions="You provide weather information.",
    tools=[fetch_live_weather]
)
Function Chaining
Python

def validate_customer(
    customer_id: Annotated[str, Field(description="Customer ID")]
) -> str:
    """Validate that a customer exists."""
    # Check database
    if customer_id in CUSTOMERS:
        return json.dumps({"valid": True, "customer": CUSTOMERS[customer_id]})
    return json.dumps({"valid": False})

def calculate_shipping(
    customer_id: Annotated[str, Field(description="Customer ID")],
    product_id: Annotated[str, Field(description="Product ID")]
) -> str:
    """Calculate shipping cost for a customer and product."""
    # Agent might call validate_customer first, then use result here
    customer = CUSTOMERS.get(customer_id)
    product = PRODUCTS.get(product_id)
    
    if not customer or not product:
        return "Error: Invalid customer or product"
    
    # Calculate based on customer location and product weight
    base_cost = 5.99
    if customer.get("premium"):
        return "Free shipping (Premium member)"
    
    return f"Shipping: ${base_cost:.2f}"

# Agent can chain: validate → calculate shipping → create order
agent = OpenAIChatClient().create_agent(
    instructions="""When processing orders:
    1. First validate the customer
    2. Calculate shipping costs
    3. Create the order
    4. Confirm with user""",
    tools=[validate_customer, calculate_shipping, create_order]
)
Function-Calling Best Practices
1. Comprehensive Parameter Descriptions
Python

# ❌ Poor descriptions
def update_record(id: str, data: dict) -> str:
    """Update a record."""
    pass

# ✅ Detailed descriptions
def update_record(
    record_id: Annotated[str, Field(
        description="Unique record identifier in format REC-XXXXX"
    )],
    field_name: Annotated[str, Field(
        description="Field to update. Valid values: 'status', 'priority', 'assignee'"
    )],
    new_value: Annotated[str, Field(
        description="New value for the field. Must match field type constraints."
    )]
) -> str:
    """Update a specific field of a record in the database.
    
    Returns confirmation message if successful, error message if failed.
    """
    pass
2. Robust Error Handling
Python

def safe_database_query(
    query: Annotated[str, Field(description="Search query")]
) -> str:
    """Query the database safely."""
    try:
        # Validate input
        if not query or len(query) < 2:
            return json.dumps({
                "error": "Query too short",
                "message": "Please provide at least 2 characters"
            })
        
        # Execute query
        results = database.search(query)
        
        return json.dumps({
            "success": True,
            "results": results,
            "count": len(results)
        })
        
    except DatabaseError as e:
        return json.dumps({
            "error": "Database error",
            "message": str(e)
        })
    except Exception as e:
        return json.dumps({
            "error": "Unexpected error",
            "message": "Please try again later"
        })
3. Logging and Monitoring
Python

import logging

logger = logging.getLogger(__name__)

def tracked_function(
    param: Annotated[str, Field(description="Parameter")]
) -> str:
    """Function with logging and monitoring."""
    logger.info(f"Function called with param: {param}")
    
    try:
        result = process(param)
        logger.info(f"Function succeeded: {result}")
        return result
    except Exception as e:
        logger.error(f"Function failed: {e}", exc_info=True)
        return f"Error: {str(e)}"
4. Testing Tools Independently
Python

import pytest

def test_search_products():
    """Test product search function."""
    result = search_products("laptop", max_results=5)
    data = json.loads(result)
    
    assert isinstance(data, list)
    assert len(data) <= 5
    assert all("name" in item for item in data)

def test_create_order_insufficient_stock():
    """Test order creation with insufficient stock."""
    result = create_order("CUST001", "p003", 10)
    assert "Error" in result
    assert "Insufficient stock" in result
Agent Type 3: RAG Agent (Knowledge Retrieval)
RAG Agent Use Cases
RAG (Retrieval-Augmented Generation) agents combine AI reasoning with knowledge retrieval:

Documentation assistants: Search technical docs, API references
Customer support: Access knowledge bases, FAQs, policy documents
Research assistants: Search papers, articles, internal documents
Legal/compliance: Query contracts, regulations, case law
HR assistants: Search employee handbooks, policies, procedures
Vector Store Setup
RAG agents use vector stores to search through documents efficiently:

Python

from agent_framework.openai import OpenAIAssistantsClient
from agent_framework import HostedFileSearchTool, HostedVectorStoreContent

async def create_knowledge_base(client: OpenAIAssistantsClient) -> tuple[str, str]:
    """Create a vector store with documents."""
    
    # Step 1: Upload files
    file1 = await client.client.files.create(
        file=("company_policy.txt", b"""
        Company Policy Document
        
        Work Hours: Standard work hours are 9 AM to 5 PM, Monday through Friday.
        Remote Work: Employees may work remotely up to 3 days per week.
        Vacation: All employees receive 15 days of paid vacation annually.
        Sick Leave: Unlimited sick leave with manager approval.
        """),
        purpose="user_data"
    )
    
    file2 = await client.client.files.create(
        file=("benefits.txt", b"""
        Employee Benefits Guide
        
        Health Insurance: Comprehensive health, dental, and vision coverage.
        401(k): Company matches up to 6% of salary.
        Gym Membership: $50/month gym membership reimbursement.
        Learning Budget: $1000/year for professional development.
        """),
        purpose="user_data"
    )
    
    # Step 2: Create vector store
    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7}  # Auto-cleanup
    )
    
    # Step 3: Add files to vector store
    await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file1.id
    )
    
    await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file2.id
    )
    
    return vector_store.id, [file1.id, file2.id]

async def cleanup_knowledge_base(
    client: OpenAIAssistantsClient, 
    vector_store_id: str, 
    file_ids: list[str]
):
    """Clean up vector store and files."""
    await client.client.vector_stores.delete(vector_store_id)
    for file_id in file_ids:
        await client.client.files.delete(file_id)
Vector store features:

Automatic indexing: Files are automatically embedded and indexed
Efficient search: Vector similarity search for relevant content
Multi-file support: Combine multiple documents
Auto-expiration: Automatically cleanup unused stores
RAG Agent Implementation
Complete implementation of a company HR assistant:

Python

import asyncio
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

async def main():
    """HR Assistant with knowledge base."""
    
    client = OpenAIAssistantsClient()
    
    # Create knowledge base
    print("Setting up knowledge base...")
    vector_store_id, file_ids = await create_knowledge_base(client)
    print(f"Knowledge base created: {vector_store_id}\n")
    
    try:
        # Create RAG agent
        async with ChatAgent(
            chat_client=client,
            instructions="""You are an HR assistant with access to company policies and benefits.
            
            When answering questions:
            - Search the knowledge base using the file_search tool
            - Provide accurate information based on company documents
            - Cite which document the information comes from
            - If information isn't in the knowledge base, say so clearly
            - Be helpful and professional""",
            tools=[HostedFileSearchTool()]
        ) as agent:
            
            print("=== Company HR Assistant ===\n")
            
            # Questions about company policies
            queries = [
                "What are the company's work hours?",
                "How many vacation days do I get?",
                "What health benefits are available?",
                "Do you offer 401(k) matching?",
                "What's the policy on working from home?"
            ]
            
            for query in queries:
                print(f"Employee: {query}")
                
                # Search knowledge base
                result = await agent.run(
                    query,
                    tool_resources={
                        "file_search": {
                            "vector_store_ids": [vector_store_id]
                        }
                    }
                )
                
                print(f"HR Assistant: {result.text}\n")
    
    finally:
        # Cleanup
        print("Cleaning up knowledge base...")
        await cleanup_knowledge_base(client, vector_store_id, file_ids)

if __name__ == "__main__":
    asyncio.run(main())
Key components:

HostedFileSearchTool: Enables knowledge base search
tool_resources: Specifies which vector store to search
Automatic retrieval: Agent automatically searches when needed
Query Optimization
Improve RAG performance with these techniques:

1. Chunk Size Optimization
Python

# When creating files, structure content for better retrieval
document = """
# Section: Vacation Policy

Vacation Days: All full-time employees receive 15 days of paid vacation annually.

Accrual: Vacation days accrue at a rate of 1.25 days per month.

Carryover: Up to 5 unused vacation days may be carried over to the next year.

# Section: Sick Leave Policy

Sick Leave: Unlimited sick leave with manager approval.

Documentation: Medical documentation required for absences over 3 consecutive days.
"""

# Clear sections help vector search find relevant chunks
2. Metadata Enhancement
Python

# Add metadata to help categorize and search
file_with_metadata = await client.client.files.create(
    file=("policies.txt", document.encode()),
    purpose="user_data"
    # Note: OpenAI currently doesn't support custom metadata in API,
    # but you can structure filenames meaningfully
)
3. Multi-Step Retrieval
Python

async def enhanced_rag_query(agent, query, vector_store_id):
    """Perform query with follow-up for better results."""
    
    # First query
    result = await agent.run(
        f"Search the knowledge base for: {query}",
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )
    
    # If answer seems incomplete, ask follow-up
    if len(result.text) < 50 or "I don't" in result.text:
        follow_up = f"Can you search for more details about {query}?"
        result = await agent.run(
            follow_up,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
    
    return result
RAG Agent Best Practices
1. Document Structure
Python

# ✅ GOOD: Well-structured documents
good_doc = """
# Employee Benefits Guide

## Health Insurance
- Coverage: Comprehensive medical, dental, and vision
- Cost: Company pays 80% of premiums
- Enrollment: Annual enrollment period in November

## Retirement Benefits
- 401(k): Available to all employees
- Matching: Company matches up to 6% of salary
- Vesting: Immediate vesting of all contributions

## Professional Development
- Budget: $1,000 per year per employee
- Eligible: Courses, conferences, certifications
- Approval: Manager approval required
"""

# ❌ POOR: Unstructured, hard to search
poor_doc = """
We offer health insurance and the company pays most of it and you can sign up 
in November. Also we have 401k and match some of your contributions. There's 
also money for training if your manager says ok.
"""
2. Keep Documents Updated
Python

async def update_knowledge_base(
    client: OpenAIAssistantsClient,
    vector_store_id: str,
    updated_file_path: str
):
    """Update a document in the knowledge base."""
    
    # Upload new version
    with open(updated_file_path, 'rb') as f:
        new_file = await client.client.files.create(
            file=f,
            purpose="user_data"
        )
    
    # Add to vector store (OpenAI handles updates)
    await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store_id,
        file_id=new_file.id
    )
    
    return new_file.id
3. Implement Source Citations
Python

instructions = """You are a research assistant with access to a document library.

When answering questions:
1. Always use the file_search tool to find relevant information
2. Include citations showing which document you found the information in
3. Format citations as [Source: document_name.pdf]
4. If multiple sources contain relevant info, cite all of them
5. Never make up information - only use what's in the documents"""
4. Handle Missing Information Gracefully
Python

instructions = """You are a helpful assistant with access to company documents.

If the information is not in the knowledge base:
- Clearly state "I don't have information about that in my knowledge base"
- Suggest who the user might contact for that information
- Offer to help with related topics you do have information about
- Never guess or make up information"""
5. Monitor Vector Store Usage
Python

async def get_vector_store_stats(client: OpenAIAssistantsClient, vector_store_id: str):
    """Get information about vector store usage."""
    
    vector_store = await client.client.vector_stores.retrieve(vector_store_id)
    
    print(f"Vector Store: {vector_store.name}")
    print(f"File count: {vector_store.file_counts.total}")
    print(f"Status: {vector_store.status}")
    print(f"Created: {vector_store.created_at}")
    print(f"Expires: {vector_store.expires_at}")
Agent Type 4: Code Execution Agent
Code Execution Use Cases
Code execution agents can write and run Python code dynamically:

Data analysis: Process datasets, generate statistics, create visualizations
Mathematical computation: Solve complex equations, perform calculations
Format conversion: Transform data between formats (CSV, JSON, XML)
Report generation: Create summaries, charts, formatted outputs
Algorithm execution: Run sorting, searching, optimization algorithms
Security Considerations
CRITICAL SECURITY NOTES:

Code execution agents run in sandboxed environments, but you should:

✅ Use only with trusted clients (OpenAI's hosted sandbox)
✅ Understand code will execute on OpenAI's servers
✅ Never pass sensitive credentials or secrets
✅ Review generated code before relying on results
❌ Never run code execution agents with untrusted user input in production without additional safeguards
Code Execution Implementation
Building a data analysis assistant:

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIAssistantsClient

async def main():
    """Data analysis agent with code execution."""
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="""You are a data analysis assistant with Python code execution.
        
        When users ask for calculations or data processing:
        - Write clear, well-commented Python code
        - Execute the code to get results
        - Explain what the code does and what the results mean
        - Use appropriate libraries (numpy, pandas, matplotlib)
        - Show your work step-by-step""",
        tools=[HostedCodeInterpreterTool()]
    ) as agent:
        
        print("=== Data Analysis Assistant ===\n")
        
        # Mathematical calculations
        query1 = "What is the factorial of 100?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1.text}\n")
        
        # Statistical analysis
        query2 = """I have sales data: [120, 135, 148, 122, 155, 167, 143, 139]
        What is the mean, median, and standard deviation?"""
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2.text}\n")
        
        # Data transformation
        query3 = """Convert this data to a percentage of total:
        Product A: 45 sales
        Product B: 30 sales  
        Product C: 25 sales"""
        print(f"User: {query3}")
        result3 = await agent.run(query3)
        print(f"Agent: {result3.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
Output Handling
Accessing the generated code and results:

Python

from agent_framework import AgentRunResponseUpdate, ChatResponse, ChatResponseUpdate
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

def extract_code_from_response(chunk: AgentRunResponseUpdate) -> str | None:
    """Extract generated code from response chunks."""
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

async def run_with_code_extraction():
    """Run agent and capture generated code."""
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You can write and execute Python code.",
        tools=[HostedCodeInterpreterTool()]
    ) as agent:
        
        query = "Calculate the first 10 Fibonacci numbers"
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        
        generated_code = ""
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Extract code as it's generated
            code_chunk = extract_code_from_response(chunk)
            if code_chunk:
                generated_code += code_chunk
        
        print(f"\n\nGenerated Code:\n{generated_code}")

# Run it
asyncio.run(run_with_code_extraction())
Code Execution Best Practices
1. Clear Task Instructions
Python

# ✅ GOOD: Specific instructions
instructions = """You are a financial calculator that writes Python code.

When performing calculations:
- Always show your code before executing it
- Add comments explaining each step
- Round monetary values to 2 decimal places
- Format large numbers with commas for readability
- Show intermediate steps in multi-step calculations"""

# ❌ POOR: Vague instructions
instructions = "You can run code to help users."
2. Guide Code Quality
Python

instructions = """You are a data analysis assistant.

Code quality standards:
- Use descriptive variable names
- Add docstrings for functions
- Include error handling where appropriate
- Import only necessary libraries
- Write efficient, readable code
- Add comments for complex logic"""
3. Validate Results
Python

async def validated_calculation(agent, query):
    """Run calculation and validate results."""
    result = await agent.run(query)
    
    # Check if code was executed
    if "Error" in result.text:
        print("⚠️ Calculation had errors")
        return None
    
    # Ask agent to verify
    verification = await agent.run(
        "Please verify the calculation is correct and show your reasoning"
    )
    
    return result
4. Handle Limitations
Python

instructions = """You are a code execution assistant.

Important limitations:
- Maximum execution time: 120 seconds
- No network access in code environment
- Cannot install additional packages
- Limited to standard Python libraries
- Cannot save files permanently

If a task requires these features, explain the limitation to the user."""
5. Use for Appropriate Tasks
Python

# ✅ GOOD uses of code execution
good_uses = [
    "Calculate compound interest over 20 years",
    "Find prime numbers between 1 and 1000",
    "Analyze this dataset: [1,2,3,4,5,6,7,8,9,10]",
    "Convert these temperatures from Celsius to Fahrenheit",
]

# ❌ POOR uses (don't need code execution)
poor_uses = [
    "What is 2 + 2?",  # Simple calculation
    "Tell me about Python",  # Doesn't need code
    "What's the weather?",  # Needs external API
]
Agent Type 5: Multi-Modal Agent
Multi-Modal Use Cases
Multi-modal agents combine multiple capabilities for complex tasks:

Content moderation: Analyze images and text for policy violations
Research assistants: Search web, analyze images, retrieve documents
Creative tools: Generate images, analyze designs, provide feedback
Technical support: Analyze screenshots, search documentation, execute diagnostic code
Market research: Web search, data analysis, report generation
Image Analysis
Analyzing images with vision-capable models:

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def main():
    """Image analysis example."""
    
    # Use a vision-capable model
    agent = OpenAIResponsesClient(model_id="gpt-4o").create_agent(
        name="VisionAgent",
        instructions="""You are an image analysis expert.
        
        When analyzing images:
        - Describe what you see in detail
        - Identify key objects, people, and elements
        - Note colors, composition, and style
        - Provide insights about the context
        - Answer specific questions about the image"""
    )
    
    # Create message with image
    message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="What do you see in this image? Describe it in detail."),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                media_type="image/jpeg"
            )
        ]
    )
    
    print("Analyzing image...")
    result = await agent.run(message)
    print(f"Analysis: {result.text}")

if __name__ == "__main__":
    asyncio.run(main())
Image analysis capabilities:

Object detection and identification
Scene understanding
Text extraction (OCR)
Composition analysis
Contextual understanding
Web Search Integration
Adding real-time web search capabilities:

Python

import asyncio
from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIChatClient

async def main():
    """Web search enabled agent."""
    
    # Note: Requires gpt-4o-search-preview or similar model
    agent = OpenAIChatClient(model_id="gpt-4o-search-preview").create_agent(
        name="ResearchAgent",
        instructions="""You are a research assistant with web search capabilities.
        
        When users ask questions:
        - Use web search for current information
        - Cite your sources
        - Provide accurate, up-to-date information
        - Distinguish between search results and your knowledge"""
    )
    
    # Provide user location for better search results
    additional_properties = {
        "user_location": {
            "country": "US",
            "city": "Seattle"
        }
    }
    
    query = "What is the current weather? Do not ask for my location."
    print(f"User: {query}")
    
    result = await agent.run(
        query,
        tools=[HostedWebSearchTool(additional_properties=additional_properties)],
        tool_choice="auto"
    )
    
    print(f"Agent: {result.text}")

if __name__ == "__main__":
    asyncio.run(main())
MCP Tool Integration
Model Context Protocol (MCP) enables integration with external tools and services:

Python

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIResponsesClient

async def main():
    """MCP tool integration example."""
    
    # Create agent with MCP tool
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="""You are a Microsoft documentation expert.
        
        Use the documentation search tool to:
        - Find relevant documentation
        - Provide accurate, official information
        - Include links to sources
        - Explain technical concepts clearly""",
        tools=[
            MCPStreamableHTTPTool(
                name="Microsoft Learn MCP",
                url="https://learn.microsoft.com/api/mcp"
            )
        ]
    ) as agent:
        
        queries = [
            "How do I create an Azure storage account using Azure CLI?",
            "What is Microsoft Agent Framework?"
        ]
        
        for query in queries:
            print(f"User: {query}")
            result = await agent.run(query)
            print(f"Agent: {result.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
MCP benefits:

Access to external APIs and services
Standardized tool integration
Local or hosted tool servers
Extensible architecture
Multi-Modal Complete Example
Complete example combining multiple capabilities:

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

async def main():
    """Complete multi-modal agent example."""
    
    # Create comprehensive agent with multiple tools
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(model_id="gpt-4o"),
        name="MultiModalAssistant",
        instructions="""You are a comprehensive AI assistant with multiple capabilities:
        
        - Image analysis: Describe and analyze images in detail
        - Web search: Find current information from the internet
        - Code execution: Write and run Python code for calculations
        - Documentation search: Access Microsoft documentation
        
        Choose the appropriate tool for each task. You can combine tools when needed.""",
        tools=[
            HostedCodeInterpreterTool(),
            MCPStreamableHTTPTool(
                name="Microsoft Learn",
                url="https://learn.microsoft.com/api/mcp"
            )
        ]
    ) as agent:
        
        print("=== Multi-Modal AI Assistant ===\n")
        
        # Task 1: Code execution
        print("Task 1: Mathematical Calculation")
        query1 = "Calculate the first 20 prime numbers"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Assistant: {result1.text}\n")
        
        # Task 2: Documentation search
        print("\nTask 2: Documentation Search")
        query2 = "How do I authenticate with Azure using Python?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"Assistant: {result2.text}\n")
        
        # Task 3: Image analysis
        print("\nTask 3: Image Analysis")
        message = ChatMessage(
            role="user",
            contents=[
                TextContent(text="Analyze this image and describe what you see"),
                UriContent(
                    uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    media_type="image/jpeg"
                )
            ]
        )
        result3 = await agent.run(message)
        print(f"Assistant: {result3.text}\n")
        
        # Task 4: Web search (would require search-enabled model)
        print("\nTask 4: Combining Tools")
        query4 = """Using the documentation search, find information about 
        Azure Functions, then write Python code to calculate the cost of 
        running 1 million function executions."""
        print(f"User: {query4}")
        result4 = await agent.run(query4)
        print(f"Assistant: {result4.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
Multi-Modal Best Practices
1. Tool Selection Logic
Python

instructions = """You are a multi-modal assistant with several tools:

Tool selection guidelines:
- Use CODE EXECUTION for: math, data analysis, calculations
- Use WEB SEARCH for: current events, weather, real-time data
- Use IMAGE ANALYSIS for: describing, analyzing visual content
- Use DOCUMENTATION SEARCH for: technical information, how-tos
- Use MULTIPLE TOOLS when: a task requires combining capabilities

Always choose the most appropriate tool for the task."""
2. Error Recovery
Python

async def resilient_multi_modal_query(agent, query):
    """Query with fallback strategies."""
    try:
        # Try primary approach
        result = await agent.run(query)
        return result
    except Exception as e:
        print(f"Primary approach failed: {e}")
        
        # Try alternative approach
        fallback_query = f"Without using external tools, {query}"
        try:
            result = await agent.run(fallback_query)
            return result
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return "I encountered errors processing your request."
3. Rate Limiting Multiple APIs
Python

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Simple rate limiter for multi-tool agents."""
    
    def __init__(self, calls_per_minute: int = 10):
        self.calls_per_minute = calls_per_minute
        self.calls = defaultdict(list)
    
    async def acquire(self, tool_name: str):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old calls
        self.calls[tool_name] = [
            call_time for call_time in self.calls[tool_name]
            if call_time > minute_ago
        ]
        
        # Check limit
        if len(self.calls[tool_name]) >= self.calls_per_minute:
            wait_time = (self.calls[tool_name][0] - minute_ago).total_seconds()
            await asyncio.sleep(wait_time)
        
        self.calls[tool_name].append(now)

# Usage
limiter = RateLimiter(calls_per_minute=5)
await limiter.acquire("web_search")
result = await agent.run(query)
4. Cost Optimization
Python

# ✅ Cost-effective approach: Use appropriate models
vision_agent = OpenAIResponsesClient(
    model_id="gpt-4o-mini"  # Cheaper for simple image tasks
).create_agent(
    instructions="Describe images briefly"
)

reasoning_agent = OpenAIResponsesClient(
    model_id="gpt-4o"  # More capable for complex tasks
).create_agent(
    instructions="Solve complex problems"
)

# Route queries to appropriate agents
if "image" in query.lower():
    result = await vision_agent.run(query)
else:
    result = await reasoning_agent.run(query)
5. Monitoring and Logging
Python

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def logged_multi_modal_run(agent, query, **kwargs):
    """Run agent with comprehensive logging."""
    start_time = datetime.now()
    
    logger.info(f"Query: {query}")
    logger.info(f"Tools available: {[tool.name for tool in kwargs.get('tools', [])]}")
    
    try:
        result = await agent.run(query, **kwargs)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Success in {duration:.2f}s")
        logger.info(f"Response length: {len(result.text)} chars")
        
        if result.usage_details:
            logger.info(f"Tokens used: {result.usage_details}")
        
        return result
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed after {duration:.2f}s: {e}")
        raise
Advanced Topics
Thread Persistence Strategies
Different strategies for maintaining conversation state:

In-Memory Persistence (Default)
Python

from agent_framework import ChatAgent, AgentThread
from agent_framework.openai import OpenAIChatClient

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Thread stored in memory
thread = agent.get_new_thread()

# Conversation history maintained in memory
await agent.run("My name is Alice", thread=thread)
await agent.run("What's my name?", thread=thread)

# Access message history
if thread.message_store:
    messages = await thread.message_store.list_messages()
    print(f"Conversation has {len(messages)} messages")
Server-Side Persistence (OpenAI)
Python

from agent_framework import ChatAgent, AgentThread
from agent_framework.openai import OpenAIResponsesClient

agent = OpenAIResponsesClient().create_agent(
    instructions="You are helpful."
)

thread = agent.get_new_thread()

# Enable server-side storage
await agent.run("My name is Alice", thread=thread, store=True)

# Thread ID can be saved and restored later
thread_id = thread.service_thread_id
print(f"Save this thread ID: {thread_id}")

# Later, restore the thread
restored_thread = AgentThread(service_thread_id=thread_id)
await agent.run("What's my name?", thread=restored_thread, store=True)
Database Persistence (Custom)
Python

from agent_framework import ChatMessageStore, ChatMessage, AgentThread
import json

class DatabaseMessageStore(ChatMessageStore):
    """Custom message store backed by a database."""
    
    def __init__(self, thread_id: str, db_connection):
        super().__init__()
        self.thread_id = thread_id
        self.db = db_connection
    
    async def add_message(self, message: ChatMessage):
        """Save message to database."""
        await self.db.execute(
            "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (self.thread_id, message.role, json.dumps(message.contents), datetime.now())
        )
    
    async def list_messages(self) -> list[ChatMessage]:
        """Load messages from database."""
        rows = await self.db.fetch_all(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY timestamp",
            (self.thread_id,)
        )
        return [
            ChatMessage(role=row['role'], contents=json.loads(row['content']))
            for row in rows
        ]

# Usage
db = await connect_to_database()
custom_store = DatabaseMessageStore("thread-123", db)
thread = AgentThread(message_store=custom_store)

await agent.run("Hello", thread=thread)
# Messages automatically persisted to database
Structured Outputs with Pydantic
Extract structured data from agent responses:

Python

import asyncio
from agent_framework import AgentRunResponse
from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel, Field
from typing import List, Optional

# Define output schema
class ContactInfo(BaseModel):
    """Structured contact information."""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number", default=None)
    company: Optional[str] = Field(description="Company name", default=None)

class MeetingExtraction(BaseModel):
    """Extract meeting details from text."""
    date: str = Field(description="Meeting date")
    time: str = Field(description="Meeting time")
    attendees: List[str] = Field(description="List of attendees")
    topic: str = Field(description="Meeting topic")
    location: Optional[str] = Field(description="Meeting location", default=None)

async def extract_contact_info():
    """Extract contact information with structured output."""
    agent = OpenAIResponsesClient().create_agent(
        instructions="Extract contact information from the user's message."
    )
    
    user_text = """
    Please add John Smith to my contacts. His email is john.smith@example.com
    and he works at TechCorp. His phone number is 555-0123.
    """
    
    print("Extracting contact info...")
    result = await agent.run(user_text, response_format=ContactInfo)
    
    # Access structured data
    contact: ContactInfo = result.value
    print(f"Name: {contact.name}")
    print(f"Email: {contact.email}")
    print(f"Phone: {contact.phone}")
    print(f"Company: {contact.company}")

async def extract_meeting_details():
    """Extract meeting details with streaming."""
    agent = OpenAIResponsesClient().create_agent(
        instructions="Extract meeting information from the message."
    )
    
    user_text = """
    Let's schedule a project review meeting for next Tuesday at 2 PM.
    Attendees will be Alice Johnson, Bob Chen, and Carol Williams.
    We'll meet in Conference Room B to discuss the Q4 roadmap.
    """
    
    print("Extracting meeting details...")
    
    # Stream and collect into structured format
    result = await AgentRunResponse.from_agent_response_generator(
        agent.run_stream(user_text, response_format=MeetingExtraction),
        output_format_type=MeetingExtraction
    )
    
    meeting: MeetingExtraction = result.value
    print(f"Date: {meeting.date}")
    print(f"Time: {meeting.time}")
    print(f"Topic: {meeting.topic}")
    print(f"Attendees: {', '.join(meeting.attendees)}")
    print(f"Location: {meeting.location}")

async def main():
    await extract_contact_info()
    print()
    await extract_meeting_details()

if __name__ == "__main__":
    asyncio.run(main())
Structured output benefits:

Type-safe data extraction
Automatic validation
Easy integration with data pipelines
Consistent output format
Approval Workflows
Implement user approval for sensitive operations:

Python

import asyncio
from typing import Any
from agent_framework import ChatAgent, ChatMessage, HostedMCPTool
from agent_framework.openai import OpenAIResponsesClient

async def handle_approvals_with_thread(query: str, agent, thread):
    """Handle approval requests in a conversation."""
    
    result = await agent.run(query, thread=thread, store=True)
    
    # Keep running while there are approval requests
    while len(result.user_input_requests) > 0:
        new_input: list[ChatMessage] = []
        
        for approval_request in result.user_input_requests:
            # Show user what the agent wants to do
            print(f"\n⚠️ Approval Required:")
            print(f"Function: {approval_request.function_call.name}")
            print(f"Arguments: {approval_request.function_call.arguments}")
            
            # Get user approval
            user_response = input("\nApprove this action? (y/n): ")
            approved = user_response.lower() == 'y'
            
            # Create approval response
            new_input.append(
                ChatMessage(
                    role="user",
                    contents=[approval_request.create_response(approved)]
                )
            )
        
        # Continue conversation with approval responses
        result = await agent.run(new_input, thread=thread, store=True)
    
    return result

async def main():
    """Example with approval workflow."""
    
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="SecureAgent",
        instructions="You are a helpful assistant with documentation access.",
        tools=[
            HostedMCPTool(
                name="Microsoft Learn",
                url="https://learn.microsoft.com/api/mcp",
                # Require approval for all function calls
                approval_mode="always_require"
            )
        ]
    ) as agent:
        
        thread = agent.get_new_thread()
        
        query = "Search for Azure storage documentation"
        print(f"User: {query}")
        
        result = await handle_approvals_with_thread(query, agent, thread)
        print(f"\nAgent: {result.text}")

if __name__ == "__main__":
    asyncio.run(main())
Approval mode options:

"always_require": Approval needed for all function calls
"never_require": No approval needed (auto-approve)
{"never_require_approval": ["function1", "function2"]}: Selective approval
Performance Optimization
Techniques for optimizing agent performance:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

# 1. Use appropriate models
fast_agent = OpenAIChatClient(
    model_id="gpt-4o-mini"  # Faster, cheaper for simple tasks
).create_agent(instructions="Answer briefly")

smart_agent = OpenAIChatClient(
    model_id="gpt-4o"  # More capable for complex tasks
).create_agent(instructions="Provide detailed analysis")

# 2. Optimize token usage
efficient_agent = OpenAIChatClient().create_agent(
    instructions="""Be concise. Provide direct answers without unnecessary elaboration.""",
    additional_chat_options={
        "max_tokens": 500,  # Limit response length
        "temperature": 0.3,  # More focused responses
    }
)

# 3. Parallel processing
async def process_queries_parallel(agent, queries):
    """Process multiple queries in parallel."""
    tasks = [agent.run(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

# 4. Caching for repeated queries
from functools import lru_cache
import hashlib

class CachedAgent:
    """Agent wrapper with response caching."""
    
    def __init__(self, agent):
        self.agent = agent
        self.cache = {}
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    async def run(self, query: str):
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            print("Cache hit!")
            return self.cache[query_hash]
        
        result = await self.agent.run(query)
        self.cache[query_hash] = result
        return result

# Usage
base_agent = OpenAIChatClient().create_agent(instructions="You are helpful.")
cached_agent = CachedAgent(base_agent)

# First call - hits API
await cached_agent.run("What is AI?")

# Second call - uses cache
await cached_agent.run("What is AI?")
Best Practices & Patterns
Resource Management
Always use context managers for proper cleanup:

Python

# ✅ RECOMMENDED: Context managers
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    result = await agent.run("Hello")
# Resources automatically cleaned up

# ✅ ALSO GOOD: Explicit cleanup
agent = OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
)
try:
    result = await agent.run("Hello")
finally:
    await agent.__aexit__(None, None, None)

# ✅ BEST: Multiple resources
async with (
    OpenAIAssistantsClient().create_agent(instructions="Agent 1") as agent1,
    OpenAIAssistantsClient().create_agent(instructions="Agent 2") as agent2,
):
    result1 = await agent1.run("Hello")
    result2 = await agent2.run("Hi")
Error Handling & Retry Logic
Implement robust error handling:

Python

import asyncio
from openai import APIError, RateLimitError, APITimeoutError
from typing import Optional

async def run_with_retry(
    agent,
    query: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[str]:
    """Run agent with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            return result.text
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print("Max retries exceeded for rate limit")
                raise
                
        except APITimeoutError as e:
            if attempt < max_retries - 1:
                print(f"Timeout. Retrying...")
                await asyncio.sleep(base_delay)
            else:
                print("Max retries exceeded for timeout")
                raise
                
        except APIError as e:
            print(f"API error: {e}")
            raise
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
    return None
Production Considerations
Checklist for production deployments:

Python

import os
import logging
from agent_framework.openai import OpenAIChatClient

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Use environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable required")

# 3. Set reasonable limits
agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
    additional_chat_options={
        "max_tokens": 1000,  # Prevent runaway costs
        "temperature": 0.7,  # Consistent behavior
    }
)

# 4. Implement monitoring
class MonitoredAgent:
    """Agent wrapper with monitoring."""
    
    def __init__(self, agent):
        self.agent = agent
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_tokens": 0
        }
    
    async def run(self, query: str):
        self.metrics["total_queries"] += 1
        
        try:
            result = await self.agent.run(query)
            self.metrics["successful_queries"] += 1
            
            if result.usage_details:
                self.metrics["total_tokens"] += result.usage_details.get("total_tokens", 0)
            
            return result
            
        except Exception as e:
            self.metrics["failed_queries"] += 1
            logger.error(f"Query failed: {e}")
            raise
    
    def get_metrics(self):
        return self.metrics

# 5. Implement rate limiting
from asyncio import Semaphore

class RateLimitedAgent:
    """Agent with concurrent request limiting."""
    
    def __init__(self, agent, max_concurrent: int = 5):
        self.agent = agent
        self.semaphore = Semaphore(max_concurrent)
    
    async def run(self, query: str):
        async with self.semaphore:
            return await self.agent.run(query)

# 6. Health checks
async def health_check(agent) -> bool:
    """Verify agent is functioning."""
    try:
        result = await asyncio.wait_for(
            agent.run("ping"),
            timeout=10.0
        )
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
Security Best Practices
Protect your agents and data:

Python

# 1. Never hardcode credentials
# ❌ BAD
client = OpenAIChatClient(api_key="sk-...")

# ✅ GOOD
client = OpenAIChatClient()  # Uses environment variable

# 2. Validate user input
def sanitize_input(user_input: str) -> str:
    """Sanitize user input before processing."""
    # Remove potential injection attempts
    sanitized = user_input.strip()
    
    # Limit length
    if len(sanitized) > 1000:
        raise ValueError("Input too long")
    
    # Check for suspicious patterns
    if "ignore previous instructions" in sanitized.lower():
        raise ValueError("Suspicious input detected")
    
    return sanitized

# 3. Limit tool access
def get_safe_tools(user_role: str):
    """Provide tools based on user role."""
    if user_role == "admin":
        return [search_db, update_db, delete_db]
    elif user_role == "user":
        return [search_db]  # Read-only
    else:
        return []  # No tools

# 4. Implement approval for sensitive operations
sensitive_operations = ["delete", "update", "transfer"]

def requires_approval(function_name: str) -> bool:
    """Check if function requires approval."""
    return any(op in function_name.lower() for op in sensitive_operations)

# 5. Audit logging
def audit_log(user_id: str, action: str, details: dict):
    """Log all sensitive operations."""
    logger.info(f"AUDIT: User {user_id} performed {action}: {details}")
    # Also write to secure audit database
Troubleshooting Guide
Common Errors
Error: "Rate limit exceeded"
Python

# Problem: Too many requests
# Solution: Implement exponential backoff

from openai import RateLimitError
import asyncio

async def handle_rate_limit():
    try:
        result = await agent.run(query)
    except RateLimitError:
        print("Rate limited. Waiting...")
        await asyncio.sleep(60)  # Wait 1 minute
        result = await agent.run(query)
    return result
Error: "Invalid authentication"
Python

# Problem: API key not set or invalid
# Solution: Check environment variables

import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY='your-key'")
else:
    print(f"API key found: {api_key[:8]}...")
Error: "Model not found"
Python

# Problem: Model ID incorrect or not available
# Solution: Use correct model name

# ❌ Wrong
client = OpenAIChatClient(model_id="gpt4")

# ✅ Correct
client = OpenAIChatClient(model_id="gpt-4o")
Error: "Thread not found"
Python

# Problem: Trying to use invalid thread ID
# Solution: Validate thread ID exists

from agent_framework import AgentThread

async def validate_thread(thread_id: str, client):
    """Check if thread exists."""
    try:
        thread = AgentThread(service_thread_id=thread_id)
        result = await agent.run("test", thread=thread, store=True)
        return True
    except Exception as e:
        print(f"Thread {thread_id} invalid: {e}")
        return False
Debugging Techniques
Enable Verbose Logging
Python

import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger("agent_framework").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
Inspect Raw Responses
Python

# Access raw API responses
async for chunk in agent.run_stream(query):
    print(f"Raw chunk: {chunk.raw_representation}")
    if chunk.text:
        print(f"Text: {chunk.text}")
Print Message History
Python

# Examine conversation history
if thread.message_store:
    messages = await thread.message_store.list_messages()
    for i, msg in enumerate(messages):
        print(f"{i}: {msg.role}: {msg.contents}")
Performance Issues
Slow Response Times
Python

# 1. Use faster models
fast_client = OpenAIChatClient(model_id="gpt-4o-mini")

# 2. Reduce max_tokens
agent = client.create_agent(
    instructions="Be concise.",
    additional_chat_options={"max_tokens": 500}
)

# 3. Use streaming for perceived speed
async for chunk in agent.run_stream(query):
    print(chunk.text, end="", flush=True)

# 4. Monitor with timing
import time
start = time.time()
result = await agent.run(query)
print(f"Took {time.time() - start:.2f}s")
High Costs
Python

# 1. Track token usage
if result.usage_details:
    tokens = result.usage_details.get("total_tokens", 0)
    print(f"Used {tokens} tokens")

# 2. Set budget limits
class BudgetTracker:
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.used_tokens = 0
    
    def check(self, tokens: int):
        self.used_tokens += tokens
        if self.used_tokens > self.max_tokens:
            raise Exception("Budget exceeded")

# 3. Use cheaper models for simple tasks
# gpt-4o-mini is ~10x cheaper than 
