Microsoft Agent Framework: Comprehensive Programming Guide
A Complete Guide to Building 5 Types of AI Agents

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

Use Case & Architecture
Implementation
Configuration Options
Best Practices
Agent Type 2: Function-Calling Agent

Use Case & Architecture
Function Tool Patterns
Implementation Examples
Best Practices for Tool Design
Agent Type 3: RAG Agent (Knowledge Retrieval)

Use Case & Architecture
Vector Store Setup
Implementation
Query Optimization
Agent Type 4: Code Execution Agent

Use Case & Architecture
Implementation
Security Considerations
Output Handling
Agent Type 5: Multi-Modal Agent

Use Case & Architecture
Image Analysis
Web Search Integration
MCP Tool Integration
Complete Multi-Capability Example
Advanced Topics

Thread Persistence Strategies
Approval Workflows
Structured Outputs
Performance Optimization
Best Practices & Production Patterns

Resource Management
Error Handling & Retry Logic
Security Best Practices
Testing Strategies
Troubleshooting Guide

Common Errors & Solutions
Debugging Techniques
Quick Reference

Glossary
Additional Resources
1. Introduction
What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, flexible Python library designed to simplify the creation of AI agents that can interact with users, call functions, access knowledge bases, execute code, and integrate with external services. Built on top of OpenAI's API (and compatible with Azure OpenAI), the framework abstracts away much of the complexity involved in building production-ready AI applications.

Key Benefits:

Multiple Client Types: Choose between Assistants, Chat, or Responses clients based on your use case
Unified API: Consistent interface across different AI providers and capabilities
Built-in Tool Support: Easy integration of function calling, file search, code interpretation, and web search
Thread Management: Sophisticated conversation state management with both in-memory and service-hosted options
Production Ready: Context managers, error handling, and resource cleanup built-in
Extensible: Easy integration with Model Context Protocol (MCP) servers and custom tools
Framework Architecture
The framework is built on a layered architecture:

text

┌─────────────────────────────────────────┐
│         Your Application                │
│  (Business Logic & User Interface)      │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│      Agent Framework Layer              │
│  • ChatAgent                            │
│  • AgentThread                          │
│  • Tool Integration                     │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│         Client Layer                    │
│  • OpenAIAssistantsClient               │
│  • OpenAIChatClient                     │
│  • OpenAIResponsesClient                │
│  • AzureAI variants                     │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│       AI Provider APIs                  │
│  • OpenAI API                           │
│  • Azure OpenAI API                     │
│  • Azure AI Services                    │
└─────────────────────────────────────────┘
Client Types Comparison
The framework provides three main client types, each optimized for different use cases:

Feature	Assistants Client	Chat Client	Responses Client
Primary Use Case	Persistent assistants with managed state	Stateless or custom-managed conversations	Structured output generation
State Management	Service-hosted threads	Client-managed or in-memory	Client-managed or service-hosted
Assistant Lifecycle	Can be pre-created and reused	N/A (stateless model)	N/A (stateless model)
Thread Storage	Service-managed	In-memory by default	In-memory or service (with store=True)
Best For	Multi-session apps, customer support	Chatbots, simple Q&A	APIs, data extraction, reasoning tasks
Streaming Support	✓	✓	✓
Function Calling	✓	✓	✓
File Search	✓	Limited	✓
Code Interpreter	✓	Limited	✓
Structured Output	Limited	Limited	✓ (Pydantic models)
Image Analysis	Limited	✓	✓
Reasoning	Limited	Limited	✓ (with gpt-5 models)
Environment Setup
Before you start building agents, you need to configure your environment with the necessary credentials.

For OpenAI:

Bash

# Required
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_CHAT_MODEL_ID="gpt-4o"  # or gpt-4o-mini, gpt-3.5-turbo
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"

# Optional
export OPENAI_ORG_ID="your-org-id"  # if applicable
export OPENAI_API_BASE_URL="custom-url"  # if using custom endpoint
For Azure OpenAI:

Bash

# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-chat-deployment"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="your-responses-deployment"

# Authentication (choose one)
export AZURE_OPENAI_API_KEY="your-api-key"
# OR use Azure AD authentication (recommended for production)
For Azure AI:

Bash

export AZURE_AI_PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="your-model-deployment"
Install the Framework:

Bash

# Using pip
pip install agent-framework

# Using uv (recommended)
uv add agent-framework
Your First Agent
Let's create a simple conversational agent to verify your setup:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create a simple agent
    agent = OpenAIChatClient().create_agent(
        name="GreeterAgent",
        instructions="You are a friendly assistant that greets users warmly."
    )
    
    # Get a response
    query = "Hello! What can you help me with?"
    print(f"User: {query}")
    
    result = await agent.run(query)
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
Expected Output:

text

User: Hello! What can you help me with?
Agent: Hello! I'm here to assist you with any questions or tasks you might have...
If this runs successfully, you're ready to build more sophisticated agents!

2. Core Concepts
Understanding Client Types
The framework provides three primary client types, each wrapping different OpenAI API endpoints with distinct capabilities:

OpenAIAssistantsClient

This client uses the OpenAI Assistants API, which provides service-managed persistent assistants. When you create an agent with this client, OpenAI maintains the assistant configuration on their servers.

Python

from agent_framework.openai import OpenAIAssistantsClient

# Assistant is created on OpenAI's servers and can be reused
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_function
) as agent:
    result = await agent.run("Hello")
    # Assistant is automatically deleted when context exits
Key characteristics:

Assistants are persistent server-side entities
Threads are managed by OpenAI's service
Excellent for applications where you need to maintain assistant state across application restarts
Supports automatic cleanup with context managers
OpenAIChatClient

This client uses the standard Chat Completions API, providing maximum flexibility with minimal server-side state management.

Python

from agent_framework.openai import OpenAIChatClient

# Stateless chat interactions
agent = OpenAIChatClient().create_agent(
    name="ChatBot",
    instructions="You are a helpful assistant."
)

# Each call is independent unless you manage threads
result = await agent.run("Hello")
Key characteristics:

Lightweight and stateless by default
You control all state management
Perfect for simple chatbots or applications with custom storage
Lower overhead than Assistants API
OpenAIResponsesClient

This client uses the newer Responses API, optimized for structured output generation and advanced features like reasoning.

Python

from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

class WeatherData(BaseModel):
    city: str
    temperature: float
    conditions: str

agent = OpenAIResponsesClient().create_agent(
    instructions="Extract weather information."
)

# Get structured output
result = await agent.run(
    "It's 72°F and sunny in Seattle",
    response_format=WeatherData
)

weather: WeatherData = result.value  # Type-safe!
Key characteristics:

First-class support for structured outputs
Advanced reasoning capabilities (with gpt-5 models)
Can use service-hosted threads with store=True
Ideal for API backends and data extraction
Agent Lifecycle Management
Proper agent lifecycle management ensures resources are properly allocated and cleaned up. The framework provides several patterns:

Context Manager Pattern (Recommended):

Python

# Automatic resource cleanup
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    result = await agent.run("Hello")
    # Resources automatically cleaned up on exit
Manual Management:

Python

# When you need more control
client = OpenAIAssistantsClient()
agent = client.create_agent(instructions="You are helpful.")

try:
    result = await agent.run("Hello")
finally:
    # Manual cleanup if needed
    await agent.close()  # Cleanup resources
Working with Existing Assistants:

Python

from openai import AsyncOpenAI

# Create assistant separately
client = AsyncOpenAI()
assistant = await client.beta.assistants.create(
    model="gpt-4o",
    name="PersistentAssistant"
)

try:
    # Use existing assistant
    agent = OpenAIAssistantsClient(
        assistant_id=assistant.id
    ).create_agent(tools=my_tool)
    
    result = await agent.run("Hello")
finally:
    # Clean up manually
    await client.beta.assistants.delete(assistant.id)
Thread Management Patterns
Threads represent conversation history and state. The framework supports multiple thread management patterns:

Pattern 1: Stateless (No Thread)

Each interaction is independent:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Each call is independent
result1 = await agent.run("What's 2+2?")  # "4"
result2 = await agent.run("What was my last question?")  # Doesn't remember
Pattern 2: In-Memory Thread

Maintain conversation history in your application's memory:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Create a thread to maintain context
thread = agent.get_new_thread()

# Conversation with context
result1 = await agent.run("My name is Alice", thread=thread)
result2 = await agent.run("What's my name?", thread=thread)  # "Alice"

# Thread data is in memory only
Pattern 3: Service-Hosted Thread

Let the service manage conversation state:

Python

agent = OpenAIResponsesClient().create_agent(
    instructions="You are helpful."
)

thread = agent.get_new_thread()

# First message - creates thread on service
result1 = await agent.run(
    "My name is Alice", 
    thread=thread,
    store=True  # Enable service storage
)

# Thread now has a service_thread_id
print(f"Thread ID: {thread.service_thread_id}")

# Continue conversation - state maintained by service
result2 = await agent.run(
    "What's my name?", 
    thread=thread,
    store=True
)  # "Alice"
Pattern 4: Thread Persistence Across Sessions

Resume conversations using thread IDs:

Python

# Session 1: Start conversation
agent1 = OpenAIResponsesClient().create_agent(
    instructions="You are helpful."
)
thread = agent1.get_new_thread()
await agent1.run("My favorite color is blue", thread=thread, store=True)
thread_id = thread.service_thread_id

# Save thread_id to database/storage
save_to_database(user_id, thread_id)

# Session 2: Resume conversation (later, different process)
thread_id = load_from_database(user_id)
agent2 = OpenAIResponsesClient().create_agent(
    instructions="You are helpful."
)
thread = AgentThread(service_thread_id=thread_id)
result = await agent2.run("What's my favorite color?", thread=thread, store=True)
# "Your favorite color is blue"
Tool Integration Approaches
The framework supports multiple approaches for integrating tools (functions, file search, code execution, etc.):

Approach 1: Agent-Level Tools

Tools defined at agent creation are available for all interactions:

Python

def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")

# Tools available for all queries
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather, get_time]  # Available for all queries
)

await agent.run("What's the weather in Seattle?")  # Uses get_weather
await agent.run("What time is it?")  # Uses get_time
Approach 2: Run-Level Tools

Tools specified per query for fine-grained control:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
    # No tools defined here
)

# Different tools for different queries
await agent.run("Weather in NYC?", tools=[get_weather])
await agent.run("Current time?", tools=[get_time])
Approach 3: Mixed Approach

Combine base tools with query-specific tools:

Python

# Base tool available for all queries
agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather]  # Always available
)

# Add additional tools for specific queries
await agent.run(
    "Weather and time in NYC?",
    tools=[get_time]  # get_weather + get_time for this query
)
Streaming vs Non-Streaming Responses
The framework supports both streaming and non-streaming response patterns:

Non-Streaming (Buffered):

Get the complete response at once:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Wait for complete response
result = await agent.run("Tell me a story")
print(result)  # Complete story printed at once
Pros:

Simpler code
Can process complete response together
Easier error handling
Cons:

Higher latency (user waits for full response)
No progressive feedback
Streaming:

Process response as it's generated:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

# Stream response chunks
print("Agent: ", end="", flush=True)
async for chunk in agent.run_stream("Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()  # New line at end
Pros:

Lower perceived latency
Progressive user feedback
Better user experience for long responses
Cons:

More complex code
Harder to handle errors mid-stream
Streaming with Structured Output:

Even with streaming, you can get structured output:

Python

from agent_framework import AgentRunResponse
from pydantic import BaseModel

class Story(BaseModel):
    title: str
    content: str

agent = OpenAIResponsesClient().create_agent(
    instructions="You are a storyteller."
)

# Collect streaming chunks into structured output
result = await AgentRunResponse.from_agent_response_generator(
    agent.run_stream("Tell me a story", response_format=Story),
    output_format_type=Story
)

story: Story = result.value
print(f"Title: {story.title}")
print(f"Content: {story.content}")
Error Handling Patterns
Robust error handling is crucial for production applications:

Pattern 1: Basic Try-Except:

Python

from openai import APIError, RateLimitError

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

try:
    result = await agent.run("Hello")
except RateLimitError as e:
    print("Rate limit exceeded, please try again later")
except APIError as e:
    print(f"API error occurred: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
Pattern 2: Retry with Exponential Backoff:

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
                print(f"Rate limited. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except APIError as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
    return None
Pattern 3: Streaming Error Handling:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful."
)

partial_response = ""
try:
    async for chunk in agent.run_stream("Hello"):
        if chunk.text:
            partial_response += chunk.text
            print(chunk.text, end="", flush=True)
except Exception as e:
    print(f"\nError during streaming: {e}")
    print(f"Partial response received: {partial_response}")
3. Agent Type 1: Basic Conversational Agent
Use Case & Architecture
Basic conversational agents are the foundation of most AI applications. They handle straightforward question-answering, provide information, and engage in natural dialogue without requiring external tools or complex capabilities.

Ideal Use Cases:

Customer support chatbots
FAQ assistants
General information providers
Conversational interfaces
Educational tutors
Architecture Overview:

text

User Input → Agent → LLM → Response → User
     ↑                                   ↓
     └─────── Thread (Optional) ─────────┘
The basic conversational agent maintains context through threads (optional) and relies solely on the LLM's training data and provided instructions.

Implementation
Let's build a complete customer support agent:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import AgentThread

class CustomerSupportAgent:
    """A basic conversational agent for customer support."""
    
    def __init__(self):
        self.agent = OpenAIChatClient().create_agent(
            name="SupportAgent",
            instructions="""You are a friendly and helpful customer support agent.
            
            Your responsibilities:
            - Answer customer questions clearly and concisely
            - Be empathetic and professional
            - If you don't know something, admit it honestly
            - Guide users to appropriate resources when needed
            
            Always maintain a positive, helpful tone."""
        )
    
    async def handle_query(self, query: str, thread: AgentThread = None) -> str:
        """Handle a single customer query."""
        try:
            result = await self.agent.run(query, thread=thread)
            return result.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    async def handle_conversation(self):
        """Handle a multi-turn conversation with context."""
        print("Customer Support Agent (type 'exit' to quit)")
        print("=" * 50)
        
        # Create thread to maintain conversation context
        thread = self.agent.get_new_thread()
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Agent: Thank you for contacting support. Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Get response with context
            response = await self.handle_query(user_input, thread=thread)
            print(f"Agent: {response}")


async def main():
    support_agent = CustomerSupportAgent()
    
    # Example 1: Single query without context
    print("=== Single Query Example ===")
    response = await support_agent.handle_query(
        "What are your business hours?"
    )
    print(f"Response: {response}\n")
    
    # Example 2: Interactive conversation with context
    print("\n=== Interactive Conversation ===")
    await support_agent.handle_conversation()


if __name__ == "__main__":
    asyncio.run(main())
Streaming Version for Better UX:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def streaming_conversation():
    """Conversational agent with streaming responses."""
    agent = OpenAIChatClient().create_agent(
        name="StreamingBot",
        instructions="You are a helpful assistant. Provide clear, concise answers."
    )
    
    thread = agent.get_new_thread()
    
    print("Streaming Chatbot (type 'exit' to quit)")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Stream the response
        print("Bot: ", end="", flush=True)
        async for chunk in agent.run_stream(user_input, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()  # New line after response


if __name__ == "__main__":
    asyncio.run(streaming_conversation())
Configuration Options
Customize your agent's behavior through various configuration options:

Python

from agent_framework.openai import OpenAIChatClient

# Basic configuration
agent = OpenAIChatClient(
    model_id="gpt-4o",  # Model selection
    api_key="your-key",  # Explicit API key
).create_agent(
    name="ConfiguredAgent",
    instructions="You are helpful.",
    
    # Additional chat options (provider-specific)
    additional_chat_options={
        "temperature": 0.7,  # Creativity (0.0-2.0)
        "max_tokens": 500,   # Response length limit
        "top_p": 0.9,        # Nucleus sampling
        "presence_penalty": 0.0,  # Penalize repetition
        "frequency_penalty": 0.0,  # Penalize frequent tokens
    }
)
Temperature Guide:

0.0-0.3: Deterministic, factual responses (customer support, data extraction)
0.4-0.7: Balanced creativity (general conversation)
0.8-1.0: Creative responses (storytelling, brainstorming)
1.0+: Highly creative but potentially less coherent
Best Practices
1. Write Clear, Specific Instructions:

Python

# ❌ Vague
instructions = "Be helpful."

# ✅ Specific
instructions = """You are a technical support agent for a SaaS product.

Guidelines:
- Always ask clarifying questions before providing solutions
- Provide step-by-step instructions
- Use simple, non-technical language unless the user demonstrates technical expertise
- If a problem requires escalation, clearly state that and explain why
- Never make promises about features or timelines"""
2. Use Thread Management Appropriately:

Python

# For stateless Q&A (no context needed)
agent = OpenAIChatClient().create_agent(
    instructions="Answer questions about products."
)
result = await agent.run("What's your return policy?")  # No thread

# For conversations requiring context
thread = agent.get_new_thread()
await agent.run("I bought a laptop last week", thread=thread)
await agent.run("Can I return it?", thread=thread)  # Uses context
3. Handle Errors Gracefully:

Python

async def safe_query(agent, query: str, thread=None) -> str:
    """Query with graceful error handling."""
    try:
        result = await agent.run(query, thread=thread)
        return result.text
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Agent error: {e}", exc_info=True)
        
        # Return user-friendly message
        return (
            "I apologize, but I'm having trouble processing your request. "
            "Please try rephrasing your question or contact support if the issue persists."
        )
4. Implement Rate Limiting:

Python

import time
from collections import deque

class RateLimitedAgent:
    """Agent with client-side rate limiting."""
    
    def __init__(self, max_requests_per_minute: int = 10):
        self.agent = OpenAIChatClient().create_agent(
            instructions="You are helpful."
        )
        self.request_times = deque(maxlen=max_requests_per_minute)
        self.max_requests = max_requests_per_minute
    
    async def run(self, query: str, thread=None):
        # Check rate limit
        now = time.time()
        if len(self.request_times) >= self.max_requests:
            oldest = self.request_times[0]
            if now - oldest < 60:
                wait_time = 60 - (now - oldest)
                raise Exception(f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.")
        
        self.request_times.append(now)
        return await self.agent.run(query, thread=thread)
5. Monitor and Log Interactions:

Python

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def logged_query(agent, query: str, user_id: str = None):
    """Run query with comprehensive logging."""
    start_time = datetime.now()
    
    logger.info(f"Query from user {user_id}: {query[:100]}...")
    
    try:
        result = await agent.run(query)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Successful response for user {user_id} "
            f"(duration: {duration:.2f}s, length: {len(result.text)} chars)"
        )
        
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Error for user {user_id} after {duration:.2f}s: {e}",
            exc_info=True
        )
        raise
4. Agent Type 2: Function-Calling Agent
Use Case & Architecture
Function-calling agents extend conversational capabilities by integrating with external systems, APIs, databases, and custom business logic. The agent intelligently determines when to call functions based on user queries and context.

Ideal Use Cases:

E-commerce assistants (check inventory, place orders)
Travel booking agents (search flights, hotels)
Enterprise automation (query databases, trigger workflows)
IoT control (adjust thermostats, control lights)
Financial assistants (check balances, transfer funds)
Architecture Overview:

text

User Query → Agent → LLM Decision
                      ├─→ Call Function(s) → Execute Business Logic
                      └─→ Synthesize Results → Final Response
Function Tool Patterns
The framework supports multiple patterns for defining and using function tools:

Pattern 1: Simple Function with Type Hints

Python

from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(description="City name or ZIP code")],
    units: Annotated[str, Field(description="'celsius' or 'fahrenheit'")] = "fahrenheit"
) -> str:
    """Get current weather for a location.
    
    This docstring is sent to the LLM to help it understand when to use this function.
    """
    # Your business logic here
    # In production, call actual weather API
    return f"The weather in {location} is sunny and 72°{units[0].upper()}"

# Use with agent
agent = OpenAIChatClient().create_agent(
    instructions="You are a weather assistant.",
    tools=[get_weather]  # Framework automatically generates schema
)
Pattern 2: Multiple Related Functions

Python

from datetime import datetime
from typing import Annotated
from pydantic import Field

class TravelAssistant:
    """Collection of travel-related functions."""
    
    @staticmethod
    def search_flights(
        origin: Annotated[str, Field(description="Departure airport code")],
        destination: Annotated[str, Field(description="Arrival airport code")],
        date: Annotated[str, Field(description="Travel date (YYYY-MM-DD)")]
    ) -> str:
        """Search for available flights."""
        # Business logic: query flight database/API
        return f"Found 5 flights from {origin} to {destination} on {date}"
    
    @staticmethod
    def search_hotels(
        location: Annotated[str, Field(description="City or hotel name")],
        check_in: Annotated[str, Field(description="Check-in date (YYYY-MM-DD)")],
        check_out: Annotated[str, Field(description="Check-out date (YYYY-MM-DD)")]
    ) -> str:
        """Search for available hotels."""
        # Business logic: query hotel database/API
        return f"Found 12 hotels in {location} from {check_in} to {check_out}"
    
    @staticmethod
    def get_booking(
        booking_id: Annotated[str, Field(description="Booking confirmation number")]
    ) -> str:
        """Retrieve booking details."""
        # Business logic: query booking database
        return f"Booking {booking_id}: Flight SEA→NYC, Hotel Marriott (confirmed)"

# Use all functions
assistant = TravelAssistant()
agent = OpenAIChatClient().create_agent(
    instructions="You are a travel booking assistant.",
    tools=[
        assistant.search_flights,
        assistant.search_hotels,
        assistant.get_booking
    ]
)
Implementation Examples
Example 1: E-Commerce Agent

Python

import asyncio
from typing import Annotated, Dict, List
from pydantic import Field
from agent_framework.openai import OpenAIChatClient

# Simulated database
INVENTORY = {
    "laptop-001": {"name": "Pro Laptop 15", "price": 1299.99, "stock": 5},
    "phone-001": {"name": "SmartPhone X", "price": 899.99, "stock": 12},
    "tablet-001": {"name": "Tablet Pro", "price": 599.99, "stock": 0},
}

CART: Dict[str, int] = {}

def search_products(
    query: Annotated[str, Field(description="Search terms for products")]
) -> str:
    """Search for products in the inventory."""
    results = []
    query_lower = query.lower()
    
    for product_id, details in INVENTORY.items():
        if query_lower in details["name"].lower():
            stock_status = "In Stock" if details["stock"] > 0 else "Out of Stock"
            results.append(
                f"{details['name']} (${details['price']}) - {stock_status} [ID: {product_id}]"
            )
    
    if not results:
        return "No products found matching your search."
    
    return "Available products:\n" + "\n".join(results)

def add_to_cart(
    product_id: Annotated[str, Field(description="Product ID to add")],
    quantity: Annotated[int, Field(description="Quantity to add")] = 1
) -> str:
    """Add a product to the shopping cart."""
    if product_id not in INVENTORY:
        return f"Product {product_id} not found."
    
    product = INVENTORY[product_id]
    
    if product["stock"] < quantity:
        return f"Sorry, only {product['stock']} units of {product['name']} available."
    
    CART[product_id] = CART.get(product_id, 0) + quantity
    return f"Added {quantity} x {product['name']} to cart."

def view_cart() -> str:
    """View current shopping cart contents."""
    if not CART:
        return "Your cart is empty."
    
    total = 0
    items = []
    
    for product_id, quantity in CART.items():
        product = INVENTORY[product_id]
        subtotal = product["price"] * quantity
        total += subtotal
        items.append(
            f"{quantity} x {product['name']} - ${subtotal:.2f}"
        )
    
    return "Your cart:\n" + "\n".join(items) + f"\n\nTotal: ${total:.2f}"

def checkout() -> str:
    """Process checkout for items in cart."""
    if not CART:
        return "Your cart is empty. Add items before checking out."
    
    # Validate stock and update inventory
    for product_id, quantity in CART.items():
        product = INVENTORY[product_id]
        if product["stock"] < quantity:
            return f"Sorry, {product['name']} is no longer available in requested quantity."
        product["stock"] -= quantity
    
    total = sum(INVENTORY[pid]["price"] * qty for pid, qty in CART.items())
    order_id = f"ORD-{len(CART)}-{total:.0f}"
    
    CART.clear()
    
    return f"Order placed successfully! Order ID: {order_id}, Total: ${total:.2f}"


async def main():
    # Create e-commerce agent
    agent = OpenAIChatClient().create_agent(
        name="ShoppingAssistant",
        instructions="""You are a helpful e-commerce shopping assistant.
        
        Help customers:
        - Find products they're looking for
        - Add items to their cart
        - Review their cart
        - Complete checkout
        
        Be friendly and guide them through the shopping process.""",
        tools=[search_products, add_to_cart, view_cart, checkout]
    )
    
    # Simulate a shopping conversation
    thread = agent.get_new_thread()
    
    queries = [
        "I'm looking for a laptop",
        "Add the Pro Laptop to my cart",
        "What's in my cart?",
        "Proceed to checkout"
    ]
    
    for query in queries:
        print(f"\nCustomer: {query}")
        print("Assistant: ", end="", flush=True)
        
        async for chunk in agent.run_stream(query, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())
Example 2: Database Query Agent

Python

import asyncio
from typing import Annotated, List, Dict
from pydantic import Field
from agent_framework.openai import OpenAIChatClient

# Simulated database
EMPLOYEES = [
    {"id": 1, "name": "Alice Johnson", "department": "Engineering", "salary": 120000},
    {"id": 2, "name": "Bob Smith", "department": "Sales", "salary": 85000},
    {"id": 3, "name": "Carol White", "department": "Engineering", "salary": 115000},
    {"id": 4, "name": "David Brown", "department": "Marketing", "salary": 75000},
]

def query_employees_by_department(
    department: Annotated[str, Field(description="Department name to filter by")]
) -> str:
    """Get all employees in a specific department."""
    results = [emp for emp in EMPLOYEES if emp["department"].lower() == department.lower()]
    
    if not results:
        return f"No employees found in {department} department."
    
    return "\n".join([
        f"{emp['name']} (ID: {emp['id']}) - ${emp['salary']:,}"
        for emp in results
    ])

def get_employee_details(
    employee_id: Annotated[int, Field(description="Employee ID number")]
) -> str:
    """Get detailed information about a specific employee."""
    employee = next((emp for emp in EMPLOYEES if emp["id"] == employee_id), None)
    
    if not employee:
        return f"Employee with ID {employee_id} not found."
    
    return (
        f"Name: {employee['name']}\n"
        f"Department: {employee['department']}\n"
        f"Salary: ${employee['salary']:,}"
    )

def calculate_average_salary(
    department: Annotated[str, Field(description="Department name, or 'all' for company-wide")] = "all"
) -> str:
    """Calculate average salary for a department or entire company."""
    if department.lower() == "all":
        employees = EMPLOYEES
        dept_name = "company-wide"
    else:
        employees = [emp for emp in EMPLOYEES if emp["department"].lower() == department.lower()]
        dept_name = department
    
    if not employees:
        return f"No employees found in {department} department."
    
    avg_salary = sum(emp["salary"] for emp in employees) / len(employees)
    return f"Average salary for {dept_name}: ${avg_salary:,.2f}"


async def main():
    agent = OpenAIChatClient().create_agent(
        name="HRAssistant",
        instructions="""You are an HR data assistant. Help users query employee information.
        
        You can:
        - Look up employees by department
        - Get individual employee details
        - Calculate salary averages
        
        Always protect sensitive data and only provide information the user is authorized to see.""",
        tools=[
            query_employees_by_department,
            get_employee_details,
            calculate_average_salary
        ]
    )
    
    # Interactive queries
    thread = agent.get_new_thread()
    
    queries = [
        "How many people work in Engineering?",
        "What's the average salary in Engineering?",
        "Tell me about employee 2"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        result = await agent.run(query, thread=thread)
        print(f"Assistant: {result}")

if __name__ == "__main__":
    asyncio.run(main())
Best Practices for Tool Design
1. Clear Function Names and Docstrings:

Python

# ❌ Unclear
def get_data(id: str) -> str:
    """Gets data."""
    pass

# ✅ Clear and descriptive
def get_customer_order_history(
    customer_id: Annotated[str, Field(description="Unique customer identifier")]
) -> str:
    """Retrieve complete order history for a specific customer.
    
    Returns a formatted list of all orders including order IDs, dates,
    items, and total amounts.
    """
    pass
2. Use Type Hints and Pydantic Fields:

Python

from typing import Annotated, Literal
from pydantic import Field

def book_appointment(
    date: Annotated[str, Field(description="Appointment date in YYYY-MM-DD format")],
    time: Annotated[str, Field(description="Appointment time in HH:MM format (24-hour)")],
    service_type: Annotated[
        Literal["haircut", "coloring", "styling"],
        Field(description="Type of service requested")
    ]
) -> str:
    """Book a salon appointment."""
    # Validation is automatic thanks to type hints
    pass
3. Handle Errors Gracefully in Functions:

Python

def transfer_funds(
    from_account: Annotated[str, Field(description="Source account number")],
    to_account: Annotated[str, Field(description="Destination account number")],
    amount: Annotated[float, Field(description="Amount to transfer")]
) -> str:
    """Transfer funds between accounts."""
    try:
        # Validate inputs
        if amount <= 0:
            return "Error: Transfer amount must be positive."
        
        if from_account == to_account:
            return "Error: Cannot transfer to the same account."
        
        # Check balance (simulated)
        balance = get_account_balance(from_account)
        if balance < amount:
            return f"Error: Insufficient funds. Available balance: ${balance:.2f}"
        
        # Process transfer (simulated)
        process_transfer(from_account, to_account, amount)
        
        return f"Successfully transferred ${amount:.2f} from {from_account} to {to_account}"
        
    except Exception as e:
        # Log error for debugging
        logger.error(f"Transfer error: {e}", exc_info=True)
        return "Error: Transfer failed. Please contact support."
4. Implement Tool-Level Authorization:

Python

from typing import Annotated
from pydantic import Field

class SecureAgent:
    def __init__(self, user_role: str):
        self.user_role = user_role
    
    def delete_user_account(
        self,
        user_id: Annotated[str, Field(description="User ID to delete")]
    ) -> str:
        """Delete a user account (admin only)."""
        if self.user_role != "admin":
            return "Error: Insufficient permissions. This action requires admin access."
        
        # Proceed with deletion
        return f"User {user_id} deleted successfully."
    
    def view_user_profile(
        self,
        user_id: Annotated[str, Field(description="User ID to view")]
    ) -> str:
        """View user profile information."""
        # All roles can view profiles
        return f"Profile for user {user_id}: ..."

# Create agent with role-based tools
admin_agent = SecureAgent(user_role="admin")
agent = OpenAIChatClient().create_agent(
    instructions="You are a user management assistant.",
    tools=[admin_agent.delete_user_account, admin_agent.view_user_profile]
)
5. Provide Rich, Structured Return Values:

Python

import json
from typing import Annotated
from pydantic import Field

def get_product_details(
    product_id: Annotated[str, Field(description="Product SKU or ID")]
) -> str:
    """Get comprehensive product information."""
    # Query database (simulated)
    product_data = {
        "name": "Wireless Headphones",
        "price": 199.99,
        "in_stock": True,
        "stock_count": 45,
        "rating": 4.5,
        "reviews_count": 1203,
        "features": ["Noise cancelling", "30hr battery", "Bluetooth 5.0"],
        "shipping": {
            "free_shipping": True,
            "estimated_days": "2-3"
        }
    }
    
    # Return structured data that LLM can parse and present nicely
    return json.dumps(product_data, indent=2)
5. Agent Type 3: RAG Agent (Knowledge Retrieval)
Use Case & Architecture
Retrieval-Augmented Generation (RAG) agents combine the power of LLMs with external knowledge bases, enabling them to provide accurate, up-to-date information from your documents, manuals, wikis, or databases.

Ideal Use Cases:

Technical documentation assistants
Legal document analysis
Medical knowledge bases
Customer support with product manuals
Research assistants
Corporate knowledge management
Architecture Overview:

text

User Query → Agent → Vector Search
                      ↓
              Retrieve Relevant Chunks
                      ↓
              LLM + Context → Response
Vector Store Setup
The framework integrates with OpenAI's vector store service for document indexing and retrieval:

Python

import asyncio
from agent_framework import HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIResponsesClient

async def create_knowledge_base(client: OpenAIResponsesClient):
    """Create and populate a vector store."""
    
    # Step 1: Upload files
    file1 = await client.client.files.create(
        file=("product_manual.txt", b"""
        Product: SmartHome Hub X1000
        
        Setup Instructions:
        1. Download the SmartHome app from your app store
        2. Create an account or sign in
        3. Plug in the hub and wait for the blue light
        4. In the app, tap 'Add Device' and select 'Hub'
        5. Follow the on-screen pairing instructions
        
        Troubleshooting:
        - If the light is red, check power connection
        - If pairing fails, ensure phone and hub are on same WiFi network
        - For persistent issues, press reset button for 10 seconds
        """),
        purpose="user_data"
    )
    
    file2 = await client.client.files.create(
        file=("faq.txt", b"""
        Frequently Asked Questions
        
        Q: Does the hub work with Alexa?
        A: Yes, the SmartHome Hub X1000 is compatible with Alexa, Google Assistant, and Siri.
        
        Q: How many devices can I connect?
        A: You can connect up to 100 smart devices to a single hub.
        
        Q: What's the warranty period?
        A: All SmartHome products come with a 2-year limited warranty.
        
        Q: Can I use it without internet?
        A: Basic functions work offline, but remote access and voice control require internet.
        """),
        purpose="user_data"
    )
    
    # Step 2: Create vector store
    vector_store = await client.client.vector_stores.create(
        name="SmartHome Knowledge Base",
        expires_after={"anchor": "last_active_at", "days": 7}  # Auto-cleanup
    )
    
    # Step 3: Add files to vector store (with polling for completion)
    await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file1.id
    )
    
    await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file2.id
    )
    
    print(f"Vector store created: {vector_store.id}")
    return vector_store.id, [file1.id, file2.id]
Implementation
Complete RAG Agent Example:

Python

import asyncio
from typing import Optional
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIResponsesClient

class DocumentAssistant:
    """RAG agent for document-based Q&A."""
    
    def __init__(self):
        self.client = OpenAIResponsesClient()
        self.vector_store_id: Optional[str] = None
        self.file_ids: list[str] = []
        self.agent: Optional[ChatAgent] = None
    
    async def setup(self):
        """Initialize vector store and agent."""
        # Create vector store with documents
        self.vector_store_id, self.file_ids = await self._create_vector_store()
        
        # Create agent with file search capability
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions="""You are a helpful product support assistant.
            
            Use the file search tool to find relevant information from the product documentation.
            
            Guidelines:
            - Always search the documentation before answering
            - If information isn't in the docs, say so clearly
            - Provide specific, accurate answers based on the documentation
            - Include relevant details like page numbers or section names when available
            """,
            tools=HostedFileSearchTool()
        )
    
    async def _create_vector_store(self):
        """Create and populate vector store with sample documents."""
        # Upload documentation files
        files_data = [
            ("setup_guide.txt", b"""
            SmartHome Hub X1000 - Quick Setup Guide
            
            What's in the box:
            - SmartHome Hub X1000
            - Power adapter (12V, 2A)
            - Ethernet cable
            - Quick start guide
            
            Initial Setup:
            1. Connect the hub to power using the included adapter
            2. Wait 30-60 seconds for the hub to boot (blue LED indicates ready)
            3. Download the SmartHome app from Apple App Store or Google Play
            4. Create an account or log in to existing account
            5. In the app, tap the '+' icon and select 'Add Hub'
            6. Follow the in-app pairing instructions
            
            The hub will automatically update its firmware on first boot.
            This may take 5-10 minutes. Do not unplug during this process.
            """),
            
            ("troubleshooting.txt", b"""
            SmartHome Hub X1000 - Troubleshooting Guide
            
            Hub won't power on:
            - Verify power adapter is properly connected
            - Try a different power outlet
            - Check that the power adapter LED is lit
            - If LED on adapter is off, contact support for replacement
            
            Hub won't connect to WiFi:
            - Ensure WiFi is 2.4GHz (5GHz not supported)
            - Verify WiFi password is correct
            - Move hub closer to router
            - Restart router and hub
            
            Pairing with app fails:
            - Ensure phone and hub are on same WiFi network
            - Close and reopen the app
            - Disable VPN on your phone
            - Try resetting hub (hold reset button 10 seconds)
            
            Devices not responding:
            - Check if devices have power
            - Verify devices are within 30 feet of hub
            - Try removing and re-adding the device
            - Update device firmware in the app
            """),
            
            ("warranty.txt", b"""
            SmartHome Hub X1000 - Warranty Information
            
            Limited Warranty Period: 2 years from date of purchase
            
            What's Covered:
            - Manufacturing defects
            - Hardware failures under normal use
            - Firmware issues
            
            What's NOT Covered:
            - Physical damage (drops, water damage, etc.)
            - Unauthorized modifications
            - Normal wear and tear
            - Damage from power surges (use surge protector)
            
            To make a warranty claim:
            1. Visit support.smarthome.com/warranty
            2. Provide proof of purchase
            3. Describe the issue
            4. Follow the support team's instructions
            
            Contact: support@smarthome.com | 1-800-SMART-HUB
            """)
        ]
        
        # Upload files
        file_ids = []
        for filename, content in files_data:
            file = await self.client.client.files.create(
                file=(filename, content),
                purpose="user_data"
            )
            file_ids.append(file.id)
        
        # Create vector store
        vector_store = await self.client.client.vector_stores.create(
            name="SmartHome Documentation",
            expires_after={"anchor": "last_active_at", "days": 7}
        )
        
        # Add files to vector store
        for file_id in file_ids:
            result = await self.client.client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
            
            if result.last_error:
                raise Exception(f"Failed to add file: {result.last_error.message}")
        
        return vector_store.id, file_ids
    
    async def query(self, question: str) -> str:
        """Ask a question about the documentation."""
        if not self.agent or not self.vector_store_id:
            raise Exception("Agent not initialized. Call setup() first.")
        
        # Run query with file search tool
        result = await self.agent.run(
            question,
            tool_resources={
                "file_search": {"vector_store_ids": [self.vector_store_id]}
            }
        )
        
        return result.text
    
    async def cleanup(self):
        """Clean up resources."""
        if self.vector_store_id:
            await self.client.client.vector_stores.delete(self.vector_store_id)
        
        for file_id in self.file_ids:
            await self.client.client.files.delete(file_id)


async def main():
    assistant = DocumentAssistant()
    
    try:
        # Initialize
        print("Setting up document assistant...")
        await assistant.setup()
        print("Ready!\n")
        
        # Example queries
        questions = [
            "How do I set up the hub?",
            "The hub won't connect to WiFi. What should I do?",
            "What's covered under warranty?",
            "How long does the initial firmware update take?"
        ]
        
        for question in questions:
            print(f"Q: {question}")
            answer = await assistant.query(question)
            print(f"A: {answer}\n")
            print("-" * 60 + "\n")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        await assistant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
Query Optimization
Best Practices for RAG Performance:

1. Document Chunking Strategy:

Python

def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split document into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap to preserve context
    
    return chunks

# Use when uploading large documents
document_text = read_large_document()
chunks = chunk_document(document_text)

for i, chunk in enumerate(chunks):
    file = await client.client.files.create(
        file=(f"document_chunk_{i}.txt", chunk.encode()),
        purpose="user_data"
    )
    # Add to vector store...
2. Metadata for Better Retrieval:

Python

# Add metadata to help with retrieval
await client.client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id,
    metadata={
        "source": "product_manual",
        "version": "2.1",
        "category": "troubleshooting",
        "last_updated": "2024-01-15"
    }
)
3. Multiple Vector Stores for Organization:

Python

class MultiDomainAssistant:
    """Agent with multiple specialized knowledge bases."""
    
    async def setup(self):
        # Create separate vector stores for different domains
        self.product_docs_store = await create_vector_store(
            name="Product Documentation"
        )
        
        self.troubleshooting_store = await create_vector_store(
            name="Troubleshooting Guides"
        )
        
        self.legal_store = await create_vector_store(
            name="Legal & Compliance"
        )
    
    async def query(self, question: str, domain: str = "all"):
        """Query specific or all knowledge bases."""
        vector_store_ids = []
        
        if domain in ["all", "products"]:
            vector_store_ids.append(self.product_docs_store)
        if domain in ["all", "troubleshooting"]:
            vector_store_ids.append(self.troubleshooting_store)
        if domain in ["all", "legal"]:
            vector_store_ids.append(self.legal_store)
        
        result = await self.agent.run(
            question,
            tool_resources={"file_search": {"vector_store_ids": vector_store_ids}}
        )
        
        return result.text
4. Query Rewriting for Better Retrieval:

Python

async def query_with_expansion(agent, user_query: str, vector_store_id: str):
    """Expand query for better retrieval."""
    
    # First, use LLM to expand the query
    expansion_prompt = f"""Given this user question: "{user_query}"
    
    Generate 2-3 alternative phrasings that might help find relevant information.
    Return only the alternative phrasings, one per line."""
    
    expansion_agent = OpenAIChatClient().create_agent(
        instructions="You expand search queries."
    )
    
    expanded = await expansion_agent.run(expansion_prompt)
    alternative_queries = expanded.text.strip().split('\n')
    
    # Search with original + expanded queries
    all_queries = [user_query] + alternative_queries
    combined_query = " OR ".join(all_queries)
    
    result = await agent.run(
        combined_query,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )
    
    return result.text
6. Agent Type 4: Code Execution Agent
Use Case & Architecture
Code execution agents can write and run Python code to solve problems, perform calculations, analyze data, and generate visualizations. This capability is powered by the HostedCodeInterpreterTool.

Ideal Use Cases:

Data analysis and visualization
Mathematical problem solving
Scientific computations
Report generation
Educational tutoring (showing work)
Financial calculations
Architecture Overview:

text

User Query → Agent → Generate Code
                      ↓
              Execute in Sandbox
                      ↓
              Return Results → Response
Implementation
Basic Code Execution Agent:

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient

async def basic_math_assistant():
    """Simple code execution agent for math problems."""
    
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="""You are a mathematics tutor that solves problems using Python code.
        
        For each problem:
        1. Explain your approach
        2. Write and execute Python code
        3. Show the result
        4. Explain the answer in simple terms
        """,
        tools=HostedCodeInterpreterTool()
    )
    
    problems = [
        "What is the factorial of 100?",
        "Calculate the first 20 Fibonacci numbers",
        "What is the compound interest on $10,000 invested at 5% annually for 10 years?"
    ]
    
    for problem in problems:
        print(f"\nProblem: {problem}")
        print("Solution: ", end="", flush=True)
        
        async for chunk in agent.run_stream(problem):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(basic_math_assistant())
Advanced: Data Analysis Agent:

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool, ChatResponse
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

class DataAnalysisAgent:
    """Agent that can analyze data and generate insights."""
    
    def __init__(self):
        self.client = OpenAIResponsesClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions="""You are a data analysis expert.
            
            When analyzing data:
            1. First, explore the data structure
            2. Calculate relevant statistics
            3. Identify patterns and insights
            4. Generate visualizations when appropriate
            5. Provide clear, actionable insights
            
            Always write clean, well-commented Python code.""",
            tools=HostedCodeInterpreterTool()
        )
    
    async def analyze_sales_data(self):
        """Analyze sample sales data."""
        query = """I have sales data for Q1 2024:
        January: $45,000
        February: $52,000
        March: $48,500
        
        Please analyze this data and tell me:
        1. Total sales
        2. Average monthly sales
        3. Month-over-month growth rates
        4. Any trends or insights
        """
        
        print("Query:", query)
        print("\nAnalysis:\n")
        
        result = await self.agent.run(query)
        
        print(result.text)
        
        # Extract generated code if available
        if (
            isinstance(result.raw_representation, ChatResponse)
            and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
            and len(result.raw_representation.raw_representation.output) > 0
        ):
            for output in result.raw_representation.raw_representation.output:
                if isinstance(output, ResponseCodeInterpreterToolCall):
                    print("\n" + "=" * 60)
                    print("Generated Code:")
                    print("=" * 60)
                    print(output.code)
    
    async def statistical_analysis(self):
        """Perform statistical analysis."""
        query = """Generate 1000 random samples from a normal distribution 
        with mean=100 and std=15. Then:
        1. Calculate mean, median, mode, and standard deviation
        2. Show the distribution
        3. Calculate the 95% confidence interval
        """
        
        print("Query:", query)
        print("\nAnalysis:\n")
        
        async for chunk in self.agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()


async def main():
    agent = DataAnalysisAgent()
    
    print("=== Sales Data Analysis ===\n")
    await agent.analyze_sales_data()
    
    print("\n\n=== Statistical Analysis ===\n")
    await agent.statistical_analysis()


if __name__ == "__main__":
    asyncio.run(main())
Security Considerations
The code interpreter runs in a sandboxed environment, but you should still be aware of security considerations:

1. Input Validation:

Python

def validate_query(query: str) -> bool:
    """Validate user input before processing."""
    # Block potentially dangerous patterns
    dangerous_patterns = [
        "import os",
        "import subprocess",
        "exec(",
        "eval(",
        "__import__",
    ]
    
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return False
    
    return True

# Use in agent
async def safe_query(agent, user_input: str):
    if not validate_query(user_input):
        return "Error: Query contains potentially unsafe operations."
    
    return await agent.run(user_input)
2. Timeout Limits:

Python

import asyncio

async def query_with_timeout(agent, query: str, timeout: int = 30):
    """Execute query with timeout to prevent hanging."""
    try:
        return await asyncio.wait_for(
            agent.run(query),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return "Error: Query execution timeout. Please simplify your request."
3. Resource Monitoring:

Python

class MonitoredCodeAgent:
    """Agent with execution monitoring."""
    
    def __init__(self):
        self.execution_count = 0
        self.max_executions_per_session = 50
        
        self.agent = ChatAgent(
            chat_client=OpenAIResponsesClient(),
            instructions="You are a code execution assistant.",
            tools=HostedCodeInterpreterTool()
        )
    
    async def run(self, query: str):
        if self.execution_count >= self.max_executions_per_session:
            return "Error: Maximum executions reached for this session."
        
        self.execution_count += 1
        return await self.agent.run(query)
Output Handling
Extracting Generated Code:

Python

from agent_framework import ChatResponse, AgentRunResponse
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

def extract_code(result: AgentRunResponse) -> str | None:
    """Extract generated code from agent response."""
    if (
        isinstance(result.raw_representation, ChatResponse)
        and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
    ):
        for output in result.raw_representation.raw_representation.output:
            if isinstance(output, ResponseCodeInterpreterToolCall):
                return output.code
    return None

# Use
result = await agent.run("Calculate factorial of 50")
code = extract_code(result)
if code:
    print("Generated code:")
    print(code)
7. Agent Type 5: Multi-Modal Agent
Use Case & Architecture
Multi-modal agents combine multiple capabilities—vision, web search, reasoning, code execution, and more—to handle complex, diverse tasks.

Ideal Use Cases:

Research assistants
Content creation
Educational platforms
Complex problem-solving
Cross-domain queries
Architecture Overview:

text

User Input (text/image/etc.)
         ↓
    Agent Decision
    ├─→ Image Analysis
    ├─→ Web Search
    ├─→ Code Execution
    ├─→ Knowledge Retrieval
    └─→ Reasoning
         ↓
   Synthesize → Response
Image Analysis
Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def image_analysis_example():
    """Analyze images with vision capabilities."""
    
    # Use a vision-capable model
    agent = OpenAIResponsesClient(model_id="gpt-4o").create_agent(
        name="VisionAgent",
        instructions="""You are an expert image analyst.
        
        When analyzing images:
        - Describe what you see in detail
        - Identify objects, people, text, and scenes
        - Note colors, composition, and style
        - Answer specific questions about the image
        """
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
    
    result = await agent.run(message)
    print("Image Analysis:")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(image_analysis_example())
Web Search Integration
Python

import asyncio
from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIChatClient

async def web_search_example():
    """Agent with web search capability."""
    
    # Use search-enabled model
    agent = OpenAIChatClient(model_id="gpt-4o-search-preview").create_agent(
        name="ResearchAgent",
        instructions="""You are a research assistant with internet access.
        
        When answering questions:
        - Search the web for current information
        - Cite your sources
        - Provide up-to-date, accurate information
        - Distinguish between factual data and opinions
        """
    )
    
    # Query with location context
    query = "What is the current weather in Seattle?"
    
    result = await agent.run(
        query,
        tools=[HostedWebSearchTool(
            additional_properties={
                "user_location": {
                    "country": "US",
                    "city": "Seattle"
                }
            }
        )]
    )
    
    print("Web Search Result:")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(web_search_example())
MCP Tool Integration
Python

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIResponsesClient

async def mcp_integration_example():
    """Integrate with Model Context Protocol servers."""
    
    # Connect to MCP server (Microsoft Learn documentation)
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="""You are a helpful documentation assistant.
        
        You have access to Microsoft Learn documentation through MCP.
        Use this to provide accurate, up-to-date information about Microsoft products and services.
        """,
        tools=MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp"
        )
    ) as agent:
        
        queries = [
            "How do I create an Azure storage account using Azure CLI?",
            "What is the Microsoft Agent Framework?"
        ]
        
        for query in queries:
            print(f"\nUser: {query}")
            print("Agent: ", end="", flush=True)
            
            async for chunk in agent.run_stream(query):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(mcp_integration_example())
Complete Multi-Capability Example
Python

import asyncio
from agent_framework import (
    ChatAgent, 
    ChatMessage,
    TextContent,
    UriContent,
    HostedCodeInterpreterTool,
    HostedWebSearchTool,
    MCPStreamableHTTPTool
)
from agent_framework.openai import OpenAIResponsesClient
from typing import Annotated
from pydantic import Field

# Custom function tool
def save_research_note(
    title: Annotated[str, Field(description="Note title")],
    content: Annotated[str, Field(description="Note content")]
) -> str:
    """Save a research note."""
    # In production, save to database
    print(f"\n[Saved Note: {title}]")
    return f"Note '{title}' saved successfully."


class UniversalResearchAgent:
    """Multi-modal agent with all capabilities."""
    
    def __init__(self):
        self.agent = None
    
    async def __aenter__(self):
        """Setup agent with all tools."""
        self.agent = ChatAgent(
            chat_client=OpenAIResponsesClient(model_id="gpt-4o"),
            name="UniversalAgent",
            instructions="""You are a universal research and analysis assistant.
            
            Your capabilities:
            - Analyze images and visual content
            - Search the web for current information
            - Execute Python code for calculations and data analysis
            - Access Microsoft documentation via MCP
            - Save research notes
            
            Use the appropriate tools based on the user's needs.
            Combine capabilities when necessary for comprehensive answers.
            """,
            tools=[
                HostedCodeInterpreterTool(),
                HostedWebSearchTool(),
                MCPStreamableHTTPTool(
                    name="Microsoft Learn",
                    url="https://learn.microsoft.com/api/mcp"
                ),
                save_research_note
            ]
        )
        
        # Start MCP connection
        await self.agent.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        """Cleanup."""
        if self.agent:
            await self.agent.__aexit__(*args)
    
    async def process_query(self, query, image_url=None):
        """Process a query with optional image."""
        if image_url:
            message = ChatMessage(
                role="user",
                contents=[
                    TextContent(text=query),
                    UriContent(uri=image_url, media_type="image/jpeg")
                ]
            )
        else:
            message = query
        
        print(f"\nUser: {query}")
        if image_url:
            print(f"[Image: {image_url}]")
        
        print("Agent: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(message):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n" + "=" * 80)


async def main():
    """Demonstrate multi-modal capabilities."""
    
    async with UniversalResearchAgent() as agent:
        
        # Task 1: Code execution
        await agent.process_query(
            "Calculate the compound annual growth rate (CAGR) if an investment "
            "grows from $10,000 to $25,000 over 7 years. Show your calculation."
        )
        
        # Task 2: Web search with current data
        await agent.process_query(
            "What are the latest developments in AI agent frameworks? "
            "Save a summary as a research note titled 'AI Agent Trends 2024'."
        )
        
        # Task 3: Documentation lookup
        await agent.process_query(
            "How do I deploy an Azure Function using the Azure CLI? "
            "Provide the exact commands."
        )
        
        # Task 4: Image analysis
        await agent.process_query(
            "Analyze this nature scene and describe the composition, "
            "colors, and mood it conveys.",
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
        
        # Task 5: Complex multi-tool query
        await agent.process_query(
            "Research the current price of Bitcoin, calculate what a $1000 "
            "investment made 5 years ago would be worth today (assuming you bought "
            "at the price from 5 years ago), and save the analysis as a note."
        )


if __name__ == "__main__":
    asyncio.run(main())
8. Advanced Topics
Thread Persistence Strategies
Strategy 1: Database-Backed Thread Storage:

Python

import json
from typing import Optional
from agent_framework import AgentThread, ChatMessageStore, ChatMessage

class DatabaseThreadManager:
    """Manage threads with database persistence."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def save_thread(self, user_id: str, thread: AgentThread):
        """Save thread to database."""
        if thread.message_store:
            messages = await thread.message_store.list_messages()
            messages_json = json.dumps([msg.model_dump() for msg in messages])
        else:
            messages_json = "[]"
        
        await self.db.execute(
            """INSERT INTO conversation_threads (user_id, service_thread_id, messages)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   service_thread_id = excluded.service_thread_id,
                   messages = excluded.messages
            """,
            (user_id, thread.service_thread_id, messages_json)
        )
    
    async def load_thread(self, user_id: str) -> Optional[AgentThread]:
        """Load thread from database."""
        row = await self.db.fetch_one(
            "SELECT service_thread_id, messages FROM conversation_threads WHERE user_id = ?",
            (user_id,)
        )
        
        if not row:
            return None
        
        messages_data = json.loads(row["messages"])
        messages = [ChatMessage(**msg) for msg in messages_data]
        
        return AgentThread(
            service_thread_id=row["service_thread_id"],
            message_store=ChatMessageStore(messages)
        )
Approval Workflows
For hosted MCP tools, implement approval workflows for security:

Python

import asyncio
from agent_framework import ChatAgent, HostedMCPTool, ChatMessage
from agent_framework.openai import OpenAIResponsesClient

async def approval_workflow_example():
    """Demonstrate approval workflow for function calls."""
    
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are helpful.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            approval_mode="always_require"  # Require approval for all calls
        )
    )
    
    thread = agent.get_new_thread()
    query = "Search Microsoft documentation for Azure Functions deployment."
    
    # Run and handle approvals
    result = await agent.run(query, thread=thread, store=True)
    
    while len(result.user_input_requests) > 0:
        for approval_needed in result.user_input_requests:
            print(f"\nFunction Call Approval Needed:")
            print(f"Function: {approval_needed.function_call.name}")
            print(f"Arguments: {approval_needed.function_call.arguments}")
            
            user_approval = input("Approve? (y/n): ")
            
            # Send approval response
            approval_response = ChatMessage(
                role="user",
                contents=[approval_needed.create_response(user_approval.lower() == "y")]
            )
            
            result = await agent.run([approval_response], thread=thread, store=True)
    
    print(f"\nFinal result: {result.text}")


if __name__ == "__main__":
    asyncio.run(approval_workflow_example())
Structured Outputs
Use Pydantic models for type-safe structured outputs:

Python

import asyncio
from agent_framework import AgentRunResponse
from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    """Structured product information."""
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")
    in_stock: bool = Field(description="Availability status")
    features: List[str] = Field(description="Key features")

class ProductCatalog(BaseModel):
    """Catalog of products."""
    products: List[Product]
    total_count: int

async def structured_output_example():
    """Extract structured data from unstructured text."""
    
    agent = OpenAIResponsesClient().create_agent(
        instructions="Extract product information into structured format."
    )
    
    text = """
    Our store has three great laptops:
    
    1. ProBook X15 - A professional laptop for $1299. It's currently in stock 
       and features a 15" display, 16GB RAM, and 512GB SSD.
    
    2. Student Lite - Perfect for students at just $599. In stock now.
       Features include 13" screen, 8GB RAM, 256GB storage, and all-day battery.
    
    3. Gaming Beast - The ultimate gaming machine at $2499. Currently out of stock.
       Boasts RTX 4080 graphics, 32GB RAM, 1TB SSD, and RGB everything.
    """
    
    result = await agent.run(
        f"Extract product information from this text:\n\n{text}",
        response_format=ProductCatalog
    )
    
    # Type-safe access to structured data
    catalog: ProductCatalog = result.value
    
    print(f"Found {catalog.total_count} products:\n")
    for product in catalog.products:
        status = "✓ In Stock" if product.in_stock else "✗ Out of Stock"
        print(f"{product.name} ({product.category}) - ${product.price:.2f} {status}")
        print(f"  Features: {', '.join(product.features)}")
        print()


if __name__ == "__main__":
    asyncio.run(structured_output_example())
Performance Optimization
1. Concurrent Requests:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def parallel_queries():
    """Process multiple queries concurrently."""
    agent = OpenAIChatClient().create_agent(
        instructions="You are helpful."
    )
    
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]
    
    # Process all queries concurrently
    results = await asyncio.gather(*[
        agent.run(query) for query in queries
    ])
    
    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result.text[:100]}...\n")
2. Caching Responses:

Python

import hashlib
import json
from typing import Optional

class CachedAgent:
    """Agent with response caching."""
    
    def __init__(self):
        self.agent = OpenAIChatClient().create_agent(
            instructions="You are helpful."
        )
        self.cache = {}
    
    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def run(self, query: str, use_cache: bool = True) -> str:
        """Run query with optional caching."""
        cache_key = self._cache_key(query)
        
        # Check cache
        if use_cache and cache_key in self.cache:
            print("[Cache hit]")
            return self.cache[cache_key]
        
        # Get fresh response
        result = await self.agent.run(query)
        response_text = result.text
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = response_text
        
        return response_text
3. Batch Processing:

Python

async def batch_process(agent, queries: List[str], batch_size: int = 5):
    """Process queries in batches to avoid rate limits."""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}...")
        batch_results = await asyncio.gather(*[
            agent.run(query) for query in batch
        ])
        
        results.extend(batch_results)
        
        # Pause between batches
        if i + batch_size < len(queries):
            await asyncio.sleep(1)
    
    return results
9. Best Practices & Production Patterns
Resource Management
Always use context managers for automatic cleanup:

Python

# ✅ Good: Automatic cleanup
async with OpenAIAssistantsClient().create_agent(
    instructions="You are helpful."
) as agent:
    result = await agent.run("Hello")
# Agent and resources automatically cleaned up

# ✅ Good: Multiple resources
async with (
    MCPStreamableHTTPTool(url="...") as mcp_server,
    OpenAIChatClient().create_agent(tools=mcp_server) as agent
):
    result = await agent.run("Hello")
# Both cleaned up automatically

# ❌ Avoid: Manual management (error-prone)
agent = OpenAIAssistantsClient().create_agent(instructions="...")
result = await agent.run("Hello")
# Resource may not be cleaned up if error occurs
Error Handling & Retry Logic
Implement comprehensive error handling:

Python

import asyncio
from openai import APIError, RateLimitError, APIConnectionError
from typing import Optional

class RobustAgent:
    """Agent with comprehensive error handling."""
    
    def __init__(self):
        self.agent = OpenAIChatClient().create_agent(
            instructions="You are helpful."
        )
        self.max_retries = 3
        self.base_delay = 1
    
    async def run_with_retry(self, query: str) -> Optional[str]:
        """Run query with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await self.agent.run(query)
                return result.text
                
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    
            except APIConnectionError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Connection error. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    
            except APIError as e:
                # Don't retry on API errors (likely client issue)
                print(f"API error: {e}")
                return None
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
        
        print(f"Failed after {self.max_retries} attempts: {last_error}")
        return None
Security Best Practices
1. Never Expose API Keys:

Python

# ✅ Good: Use environment variables
import os
api_key = os.environ.get("OPENAI_API_KEY")

# ❌ Never: Hardcode keys
api_key = "sk-proj-..."  # DON'T DO THIS

# ✅ Good: Validate environment
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")
2. Input Sanitization:

Python

def sanitize_input(user_input: str) -> str:
    """Sanitize user input."""
    # Remove excessive whitespace
    sanitized = " ".join(user_input.split())
    
    # Limit length
    max_length = 4000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove potentially dangerous patterns (context-dependent)
    # This is just an example - adjust based on your use case
    dangerous = ["<script>", "javascript:", "onerror="]
    for pattern in dangerous:
        sanitized = sanitized.replace(pattern, "")
    
    return sanitized
3. Rate Limiting:

Python

from collections import deque
import time

class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    async def acquire(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Check limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        self.requests.append(now)
Testing Strategies
1. Unit Testing Agent Functions:

Python

import pytest
from your_module import get_weather, search_products

def test_get_weather():
    """Test weather function."""
    result = get_weather(location="Seattle")
    assert "Seattle" in result
    assert "°" in result  # Contains temperature

def test_search_products():
    """Test product search."""
    result = search_products(query="laptop")
    assert "laptop" in result.lower()
    assert len(result) > 0

@pytest.mark.asyncio
async def test_agent_integration():
    """Test agent with mock client."""
    from unittest.mock import AsyncMock
    
    mock_client = AsyncMock()
    mock_client.get_response.return_value = "Test response"
    
    # Test your agent with mock
    # ...
2. Integration Testing:

Python

import pytest
from agent_framework.openai import OpenAIChatClient

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_basic_query():
    """Integration test with real API (mark as integration)."""
    agent = OpenAIChatClient().create_agent(
        instructions="You are a test assistant. Always respond with 'Test OK'."
    )
    
    result = await agent.run("Hello")
    assert result.text is not None
    assert len(result.text) > 0
10. Troubleshooting Guide
Common Errors & Solutions
Error: "No API key provided"

Python

# Problem: API key not set
# Solution:
import os

if not os.environ.get("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-key-here'")
Error: "Rate limit exceeded"

Python

# Problem: Too many requests
# Solution: Implement retry with backoff
import asyncio
from openai import RateLimitError

async def handle_rate_limit(agent, query):
    for attempt in range(3):
        try:
            return await agent.run(query)
        except RateLimitError:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
Error: "Thread not found"

Python

# Problem: Using invalid thread ID
# Solution: Check thread ID validity
async def safe_thread_usage(agent, thread_id: str):
    try:
        from agent_framework import AgentThread
        thread = AgentThread(service_thread_id=thread_id)
        return await agent.run("Hello", thread=thread, store=True)
    except Exception as e:
        print(f"Invalid thread: {e}")
        # Create new thread
        thread = agent.get_new_thread()
        return await agent.run("Hello", thread=thread, store=True)
Error: "Model not found"

Python

# Problem: Invalid model ID
# Solution: Use valid model IDs
from agent_framework.openai import OpenAIChatClient

# ✅ Valid model IDs
valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

agent = OpenAIChatClient(model_id="gpt-4o").create_agent(
    instructions="You are helpful."
)
Debugging Techniques
1. Enable Verbose Logging:

Python

import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent_framework")
logger.setLevel(logging.DEBUG)
2. Inspect Raw Responses:

Python

# Access raw API responses
result = await agent.run("Hello")

# Check raw representation
print("Raw response:", result.raw_representation)

# For streaming
async for chunk in agent.run_stream("Hello"):
    print("Raw chunk:", chunk.raw_representation)
3. Track Token Usage:

Python

result = await agent.run("Explain quantum computing")

if result.usage_details:
    print(f"Prompt tokens: {result.usage_details.prompt_tokens}")
    print(f"Completion tokens: {result.usage_details.completion_tokens}")
    print(f"Total tokens: {result.usage_details.total_tokens}")
11. Quick Reference
Glossary
Agent: An AI-powered entity that can interact with users, call functions, and use tools to accomplish tasks.

Assistant: In the context of OpenAI Assistants API, a persistent server-side entity with configured behavior and capabilities.

Thread: A conversation context that maintains message history across multiple interactions.

Tool: A capability that an agent can use, such as function calling, file search, code interpretation, or web search.

Streaming: A response pattern where output is delivered incrementally as it's generated, rather than all at once.

RAG (Retrieval-Augmented Generation): A pattern where an agent retrieves relevant information from a knowledge base before generating responses.

MCP (Model Context Protocol): A protocol for integrating external tools and services with AI agents.

Vector Store: A searchable database of document embeddings used for semantic search and retrieval.

Context Manager: A Python pattern using async with for automatic resource setup and cleanup.

Additional Resources
Official Documentation:

Microsoft Agent Framework: GitHub Repository
OpenAI API Reference:
So