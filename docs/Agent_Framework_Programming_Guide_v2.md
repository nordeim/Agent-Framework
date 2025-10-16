# Comprehensive Programming Guide for Microsoft Agent Framework

## Table of Contents
1. [Introduction to Microsoft Agent Framework](#introduction)
2. [Core Concepts and Architecture](#core-concepts)
3. [Agent Type 1: Basic Conversational Agent](#basic-conversational-agent)
4. [Agent Type 2: Function-Calling Agent](#function-calling-agent)
5. [Agent Type 3: RAG Agent (Retrieval-Augmented Generation)](#rag-agent)
6. [Agent Type 4: Code Execution Agent](#code-execution-agent)
7. [Agent Type 5: Multi-Modal Agent](#multi-modal-agent)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices and Patterns](#best-practices)
10. [Troubleshooting Guide](#troubleshooting)

## Introduction to Microsoft Agent Framework {#introduction}

The Microsoft Agent Framework is a powerful toolkit designed to simplify the creation of AI-powered agents that can interact with users through natural language. It provides a unified interface to various AI models and services, enabling developers to build sophisticated conversational experiences without dealing with the complexity of direct API interactions.

### Framework Overview

The framework abstracts away the complexities of working with different AI services by providing three main client types:

1. **OpenAIAssistantsClient** - For building agents using OpenAI's Assistants API
2. **OpenAIChatClient** - For direct chat-based interactions with OpenAI models
3. **OpenAIResponsesClient** - For structured response generation with OpenAI models

Each client type is optimized for specific use cases while maintaining a consistent interface for agent creation, configuration, and interaction.

### Value Proposition

The Microsoft Agent Framework offers several key benefits:

- **Simplified Integration**: Abstracts away the complexity of direct API interactions
- **Consistent Interface**: Provides a unified approach across different AI services
- **Rich Tool Support**: Enables integration with custom functions, file search, code execution, and more
- **Thread Management**: Handles conversation state and context preservation
- **Streaming Support**: Supports both streaming and non-streaming responses
- **Resource Management**: Provides automatic cleanup of resources through context managers

### Architecture

At its core, the framework follows a client-agent pattern:

```
Client (OpenAIAssistantsClient/OpenAIChatClient/OpenAIResponsesClient)
  ↓
Agent (ChatAgent with specific configuration)
  ↓
Tools (Function tools, file search, code interpreter, etc.)
  ↓
Responses (Streaming or non-streaming)
```

The framework handles the lifecycle management of agents, threads, and tools, allowing developers to focus on the core functionality of their applications.

## Core Concepts and Architecture {#core-concepts}

### Client Types Deep Dive

The Microsoft Agent Framework provides three main client types, each optimized for different scenarios:

#### OpenAIAssistantsClient

The AssistantsClient is designed for building agents using OpenAI's Assistants API, which provides advanced features like thread management, tool integration, and persistent conversation state.

```python
from agent_framework.openai import OpenAIAssistantsClient

# Create a client with automatic assistant lifecycle management
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful weather assistant.",
    tools=get_weather,
) as agent:
    result = await agent.run("What's the weather like in Seattle?")
    print(result)
```

Key features:
- Automatic assistant creation and cleanup
- Thread management for conversation persistence
- Rich tool integration (functions, file search, code interpreter)
- Support for both streaming and non-streaming responses

#### OpenAIChatClient

The ChatClient provides direct access to OpenAI's chat models for simpler conversational experiences.

```python
from agent_framework.openai import OpenAIChatClient

# Create a simple chat agent
agent = OpenAIChatClient().create_agent(
    name="WeatherAgent",
    instructions="You are a helpful weather assistant.",
    tools=get_weather,
)

result = await agent.run("What's the weather like in Seattle?")
print(result)
```

Key features:
- Direct chat model interaction
- Simpler setup for basic conversations
- Tool support for function calling
- Thread management for conversation context

#### OpenAIResponsesClient

The ResponsesClient is optimized for structured response generation with OpenAI models.

```python
from agent_framework.openai import OpenAIResponsesClient

# Create an agent for structured responses
agent = ChatAgent(
    chat_client=OpenAIResponsesClient(),
    instructions="You are a helpful weather assistant.",
    tools=get_weather,
)

result = await agent.run("What's the weather like in Seattle?")
print(result)
```

Key features:
- Structured response generation
- Support for vision capabilities
- Advanced reasoning capabilities
- Tool integration for complex workflows

### Agent Lifecycle Management

The framework provides robust lifecycle management for agents through context managers:

```python
# Automatic lifecycle management
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_tools,
) as agent:
    # Agent is automatically created and configured
    result = await agent.run("Help me with a task.")
    # Agent is automatically cleaned up when exiting the context
```

This approach ensures that resources are properly managed and cleaned up, preventing memory leaks and unnecessary API charges.

### Thread Management Patterns

Threads are essential for maintaining conversation context across multiple interactions. The framework provides several patterns for thread management:

#### Automatic Thread Creation

```python
# Each call creates a new thread (stateless)
async with ChatAgent(chat_client=OpenAIChatClient()) as agent:
    result1 = await agent.run("What's the weather in Seattle?")
    result2 = await agent.run("What was the last city I asked about?")  # Won't remember
```

#### Thread Persistence

```python
# Reusing the same thread across calls (stateful)
async with ChatAgent(chat_client=OpenAIChatClient()) as agent:
    thread = agent.get_new_thread()
    
    result1 = await agent.run("What's the weather in Seattle?", thread=thread)
    result2 = await agent.run("What was the last city I asked about?", thread=thread)  # Will remember
```

#### Existing Thread ID

```python
# Continuing an existing conversation
async with ChatAgent(
    chat_client=OpenAIChatClient(thread_id="existing_thread_id")
) as agent:
    thread = AgentThread(service_thread_id="existing_thread_id")
    result = await agent.run("Continue our conversation", thread=thread)
```

### Tool Integration Approaches

The framework supports multiple approaches to tool integration:

#### Agent-Level Tools

```python
# Tools available for all queries during the agent's lifetime
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tools=[get_weather, get_time],  # Tools defined at agent creation
) as agent:
    result1 = await agent.run("What's the weather in Seattle?")  # Can use get_weather
    result2 = await agent.run("What time is it?")  # Can use get_time
```

#### Run-Level Tools

```python
# Tools provided for specific queries
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
) as agent:
    result1 = await agent.run("What's the weather in Seattle?", tools=[get_weather])
    result2 = await agent.run("What time is it?", tools=[get_time])
```

#### Mixed Approach

```python
# Combining agent-level and run-level tools
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tools=[get_weather],  # Base tool available for all queries
) as agent:
    result = await agent.run(
        "What's the weather in Seattle and what time is it?",
        tools=[get_time]  # Additional tool for this specific query
    )
```

### Streaming vs Non-Streaming Responses

The framework supports both streaming and non-streaming responses:

#### Non-Streaming Responses

```python
# Get the complete result at once
result = await agent.run("What's the weather in Seattle?")
print(result)
```

#### Streaming Responses

```python
# Get results as they are generated
async for chunk in agent.run_stream("What's the weather in Seattle?"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

### Error Handling Patterns

The framework provides several patterns for error handling:

#### Basic Error Handling

```python
try:
    result = await agent.run("What's the weather in Seattle?")
except Exception as e:
    print(f"Error: {e}")
```

#### Context Manager Error Handling

```python
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_tools,
) as agent:
    try:
        result = await agent.run("Help me with a task.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
# Agent is automatically cleaned up even if an error occurs
```

#### Retry Logic

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_agent_with_retry(agent, query):
    return await agent.run(query)

try:
    result = await run_agent_with_retry(agent, "What's the weather in Seattle?")
except Exception as e:
    print(f"Failed after retries: {e}")
```

## Agent Type 1: Basic Conversational Agent {#basic-conversational-agent}

### Use Case Description

A Basic Conversational Agent is the simplest type of AI agent that can engage in natural language conversations with users. These agents are ideal for customer support, general information queries, and simple task automation. They don't require specialized tools or complex integrations, making them perfect for getting started with the framework.

### Architecture Explanation

Basic Conversational Agents are built using the ChatAgent class with either OpenAIChatClient or OpenAIAssistantsClient. The architecture is straightforward:

1. **Client**: Handles communication with the AI service
2. **Agent**: Manages the conversation flow and context
3. **Instructions**: Define the agent's behavior and personality
4. **Response**: Generates natural language responses to user queries

### Complete Code Example with Annotations

```python
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
Basic Conversational Agent Example

This sample demonstrates a simple conversational agent that can engage
in natural language conversations without specialized tools.
"""

async def basic_conversational_agent():
    """Example of a basic conversational agent."""
    print("=== Basic Conversational Agent Example ===")
    
    # Create a simple chat agent with OpenAIChatClient
    agent = OpenAIChatClient().create_agent(
        name="BasicAssistant",
        instructions="You are a helpful assistant that provides clear, concise answers to questions.",
        # No specialized tools needed for basic conversation
    )
    
    # Example 1: Non-streaming response
    print("\n--- Non-streaming Response ---")
    query1 = "What is artificial intelligence?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}")
    
    # Example 2: Streaming response
    print("\n--- Streaming Response ---")
    query2 = "Can you explain machine learning in simple terms?"
    print(f"User: {query2}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query2):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # New line after streaming
    
    # Example 3: Follow-up question
    print("\n--- Follow-up Question ---")
    query3 = "How is that different from deep learning?"
    print(f"User: {query3}")
    result3 = await agent.run(query3)
    print(f"Agent: {result3}")

async def threaded_conversational_agent():
    """Example of a conversational agent with thread management."""
    print("\n=== Threaded Conversational Agent Example ===")
    
    # Create a chat agent
    agent = OpenAIChatClient().create_agent(
        name="ThreadedAssistant",
        instructions="You are a helpful assistant that remembers previous parts of our conversation.",
    )
    
    # Create a thread to maintain conversation context
    thread = agent.get_new_thread()
    
    # First question
    print("\n--- First Question ---")
    query1 = "I'm planning a trip to Japan. What are some must-visit places?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")
    
    # Follow-up question that references previous context
    print("\n--- Follow-up Question ---")
    query2 = "Which of those places would be best for someone interested in technology?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")
    
    # Another follow-up question
    print("\n--- Another Follow-up Question ---")
    query3 = "What's the best time of year to visit those places?"
    print(f"User: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")

async def main():
    """Main function to run all examples."""
    await basic_conversational_agent()
    await threaded_conversational_agent()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Options

Basic Conversational Agents can be configured with several options:

#### Model Selection

```python
# Using a specific model
agent = OpenAIChatClient(
    model_id="gpt-4o-mini",  # Use a specific model
    api_key="your-api-key",   # Explicit API key
).create_agent(
    name="BasicAssistant",
    instructions="You are a helpful assistant.",
)
```

#### Temperature and Other Parameters

```python
# Using explicit settings
agent = OpenAIChatClient(
    model_id="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,  # Controls randomness (0-1)
    max_tokens=1000,  # Maximum response length
).create_agent(
    name="BasicAssistant",
    instructions="You are a helpful assistant.",
)
```

#### System Instructions

```python
# Detailed system instructions
agent = OpenAIChatClient().create_agent(
    name="CustomerSupportAgent",
    instructions="""
    You are a customer support agent for a tech company.
    - Always be polite and professional
    - Provide clear, step-by-step solutions
    - If you don't know something, admit it and offer to escalate
    - Keep responses concise but thorough
    """,
)
```

### Best Practices

1. **Clear Instructions**: Provide specific, detailed instructions that define the agent's behavior and limitations.

2. **Appropriate Model Selection**: Choose models based on your requirements (speed vs. capability).

3. **Thread Management**: Use threads for conversations that require context preservation.

4. **Error Handling**: Implement proper error handling to manage API failures gracefully.

5. **Response Validation**: Validate responses when necessary, especially for critical applications.

### Common Pitfalls

1. **Vague Instructions**: Unclear instructions can lead to inconsistent agent behavior.

2. **Missing Context**: Not using threads for conversations that require context can result in disjointed interactions.

3. **Ignoring Rate Limits**: Failing to account for API rate limits can cause service interruptions.

4. **No Error Handling**: Not implementing error handling can cause applications to crash unexpectedly.

5. **Excessive Token Usage**: Not monitoring token usage can lead to unexpected costs.

## Agent Type 2: Function-Calling Agent {#function-calling-agent}

### Use Case Description

Function-Calling Agents extend basic conversational capabilities by integrating custom functions and APIs. These agents can perform actions, retrieve data from external systems, and execute complex workflows based on user requests. They're ideal for applications that need to interact with existing systems, perform calculations, or access real-time data.

### Function Tool Patterns

The framework supports two main patterns for function tools:

#### Agent-Level Tools

Tools defined when creating the agent are available for all queries during the agent's lifetime:

```python
async def agent_level_tools_example():
    """Example showing tools defined when creating the agent."""
    print("=== Agent-Level Tools Example ===")
    
    # Define tools that will be available for all queries
    def get_weather(location: str) -> str:
        """Get the weather for a given location."""
        conditions = ["sunny", "cloudy", "rainy", "stormy"]
        return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."
    
    def get_time() -> str:
        """Get the current UTC time."""
        current_time = datetime.now(timezone.utc)
        return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."
    
    # Create agent with tools
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    ) as agent:
        # First query - agent can use weather tool
        query1 = "What's the weather like in New York?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1}")
        
        # Second query - agent can use time tool
        query2 = "What's the current UTC time?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2}")
        
        # Third query - agent can use both tools
        query3 = "What's the weather in London and what's the current UTC time?"
        print(f"User: {query3}")
        result3 = await agent.run(query3)
        print(f"Agent: {result3}")
```

#### Run-Level Tools

Tools provided when running the agent are available only for that specific query:

```python
async def run_level_tools_example():
    """Example showing tools passed to the run method."""
    print("=== Run-Level Tools Example ===")
    
    # Define tools
    def get_weather(location: str) -> str:
        """Get the weather for a given location."""
        conditions = ["sunny", "cloudy", "rainy", "stormy"]
        return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."
    
    def get_time() -> str:
        """Get the current UTC time."""
        current_time = datetime.now(timezone.utc)
        return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."
    
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a given symbol."""
        prices = {"AAPL": 150.25, "GOOGL": 2750.50, "MSFT": 300.75}
        if symbol in prices:
            return f"The current price of {symbol} is ${prices[symbol]}."
        else:
            return f"Sorry, I don't have price information for {symbol}."
    
    # Create agent without tools
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    ) as agent:
        # First query with weather tool
        query1 = "What's the weather like in Seattle?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, tools=[get_weather])
        print(f"Agent: {result1}")
        
        # Second query with time tool
        query2 = "What's the current UTC time?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=[get_time])
        print(f"Agent: {result2}")
        
        # Third query with stock tool
        query3 = "What's the current price of Apple stock?"
        print(f"User: {query3}")
        result3 = await agent.run(query3, tools=[get_stock_price])
        print(f"Agent: {result3}")
```

### Complete Implementation Example

```python
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
from datetime import datetime, timezone
from random import randint
from typing import Annotated, Dict, List

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
Function-Calling Agent Example

This sample demonstrates a function-calling agent that can interact with
external systems through custom functions.
"""

# Define function tools with proper type annotations
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")]
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

def get_time() -> str:
    """Get the current UTC time."""
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."

def calculate_expression(
    expression: Annotated[str, Field(description="The mathematical expression to evaluate.")]
) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Simple expression evaluation (in production, use a safer approach)
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def search_database(
    query: Annotated[str, Field(description="The search query for the database.")],
    table: Annotated[str, Field(description="The table to search in.")] = "products"
) -> str:
    """Search a database for information."""
    # Simulated database search
    if "laptop" in query.lower():
        return "Found 3 laptops: Dell XPS ($1200), MacBook Pro ($1500), HP Spectre ($1100)."
    elif "phone" in query.lower():
        return "Found 5 phones: iPhone 14 ($999), Samsung Galaxy S23 ($799), Google Pixel 7 ($599)."
    else:
        return f"No results found for '{query}' in {table} table."

def create_calendar_event(
    title: Annotated[str, Field(description="The title of the event.")],
    date: Annotated[str, Field(description="The date of the event (YYYY-MM-DD).")],
    time: Annotated[str, Field(description="The time of the event (HH:MM).")],
    duration: Annotated[int, Field(description="Duration in minutes.")] = 60
) -> str:
    """Create a calendar event."""
    # Simulated calendar event creation
    event_id = randint(1000, 9999)
    return f"Created event '{title}' for {date} at {time} (ID: {event_id})."

async def function_calling_agent_example():
    """Example of a function-calling agent with multiple tools."""
    print("=== Function-Calling Agent Example ===")
    
    # Create agent with multiple tools
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can get weather information, current time, evaluate expressions, search databases, and create calendar events.",
        tools=[get_weather, get_time, calculate_expression, search_database, create_calendar_event],
    ) as agent:
        # Example 1: Weather query
        print("\n--- Weather Query ---")
        query1 = "What's the weather like in Tokyo?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1}")
        
        # Example 2: Time query
        print("\n--- Time Query ---")
        query2 = "What time is it now in UTC?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2}")
        
        # Example 3: Math calculation
        print("\n--- Math Calculation ---")
        query3 = "What is 15% of 250?"
        print(f"User: {query3}")
        result3 = await agent.run(query3)
        print(f"Agent: {result3}")
        
        # Example 4: Database search
        print("\n--- Database Search ---")
        query4 = "Can you find laptops under $1300?"
        print(f"User: {query4}")
        result4 = await agent.run(query4)
        print(f"Agent: {result4}")
        
        # Example 5: Calendar event creation
        print("\n--- Calendar Event Creation ---")
        query5 = "Schedule a team meeting for next Friday at 2 PM for 90 minutes."
        print(f"User: {query5}")
        result5 = await agent.run(query5)
        print(f"Agent: {result5}")

async def dynamic_tool_selection_example():
    """Example showing dynamic tool selection based on query."""
    print("\n=== Dynamic Tool Selection Example ===")
    
    # Define specialized tools
    def get_weather(location: str) -> str:
        """Get the weather for a given location."""
        conditions = ["sunny", "cloudy", "rainy", "stormy"]
        return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."
    
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a given symbol."""
        prices = {"AAPL": 150.25, "GOOGL": 2750.50, "MSFT": 300.75}
        if symbol in prices:
            return f"The current price of {symbol} is ${prices[symbol]}."
        else:
            return f"Sorry, I don't have price information for {symbol}."
    
    def get_news(category: str = "technology") -> str:
        """Get the latest news for a given category."""
        headlines = {
            "technology": "New AI breakthrough announced by research team",
            "sports": "Local team wins championship in thrilling final",
            "business": "Stock market reaches all-time high"
        }
        return f"Latest {category} news: {headlines.get(category, 'No news available for this category.')}"
    
    # Create agent without predefined tools
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can provide information on various topics.",
    ) as agent:
        # Query 1: Weather - provide weather tool
        print("\n--- Weather Query ---")
        query1 = "What's the weather like in Paris?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, tools=[get_weather])
        print(f"Agent: {result1}")
        
        # Query 2: Stock - provide stock tool
        print("\n--- Stock Query ---")
        query2 = "What's the current price of Microsoft stock?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=[get_stock_price])
        print(f"Agent: {result2}")
        
        # Query 3: News - provide news tool
        print("\n--- News Query ---")
        query3 = "What's the latest technology news?"
        print(f"User: {query3}")
        result3 = await agent.run(query3, tools=[get_news])
        print(f"Agent: {result3}")

async def main():
    """Main function to run all examples."""
    await function_calling_agent_example()
    await dynamic_tool_selection_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Parameter Handling and Validation

Proper parameter handling is crucial for function tools:

```python
from pydantic import BaseModel, Field, validator

class WeatherParams(BaseModel):
    location: str = Field(..., description="The location to get the weather for.")
    units: str = Field("metric", description="The units for temperature (metric or imperial).")
    
    @validator('units')
    def validate_units(cls, v):
        if v not in ['metric', 'imperial']:
            raise ValueError('Units must be either "metric" or "imperial"')
        return v

def get_weather(params: WeatherParams) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    unit_symbol = "°C" if params.units == "metric" else "°F"
    temp = randint(10, 30) if params.units == "metric" else randint(50, 86)
    return f"The weather in {params.location} is {conditions[randint(0, 3)]} with a high of {temp}{unit_symbol}."

# Usage in agent
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
) as agent:
    result = await agent.run("What's the weather like in New York in Fahrenheit?")
    print(result)
```

### Best Practices for Tool Design

1. **Clear Descriptions**: Provide clear descriptions for functions and parameters to help the AI understand when and how to use them.

2. **Type Annotations**: Use proper type annotations to ensure correct parameter handling.

3. **Error Handling**: Implement robust error handling within functions to provide meaningful feedback.

4. **Input Validation**: Validate inputs to prevent errors and security issues.

5. **Idempotency**: Design functions to be idempotent when possible, meaning they produce the same result when called multiple times with the same inputs.

### Error Handling in Functions

```python
def safe_divide(a: float, b: float) -> str:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return "Error: Cannot divide by zero."
        result = a / b
        return f"The result of {a} divided by {b} is {result}."
    except Exception as e:
        return f"Error performing division: {str(e)}"

def api_request(endpoint: str) -> str:
    """Make a request to an API endpoint."""
    try:
        # Simulated API request
        if "invalid" in endpoint.lower():
            return "Error: Invalid endpoint."
        return f"Successfully retrieved data from {endpoint}."
    except Exception as e:
        return f"Error making API request: {str(e)}"
```

## Agent Type 3: RAG Agent (Retrieval-Augmented Generation) {#rag-agent}

### Use Case Description

RAG (Retrieval-Augmented Generation) Agents combine the power of language models with information retrieval from external knowledge bases. These agents can search through documents, databases, and other knowledge sources to provide accurate, context-aware responses. They're ideal for customer support, knowledge management, research assistance, and any application that requires access to specific information.

### Vector Store Setup and Management

RAG Agents rely on vector stores to efficiently search through large amounts of text:

```python
async def create_vector_store(client: OpenAIAssistantsClient) -> tuple[str, HostedVectorStoreContent]:
    """Create a vector store with sample documents."""
    # Create a file with content
    file = await client.client.files.create(
        file=("company_policy.txt", b"Our company policy states that all employees must complete security training annually. Remote work is allowed with manager approval. Vacation requests must be submitted 2 weeks in advance."),
        purpose="user_data"
    )
    
    # Create a vector store
    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    
    # Add the file to the vector store
    result = await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id, 
        file_id=file.id
    )
    
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed: {result.last_error.message}")
    
    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)

async def delete_vector_store(client: OpenAIAssistantsClient, file_id: str, vector_store_id: str) -> None:
    """Delete the vector store after using it."""
    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)
```

### File Upload and Indexing

```python
async def upload_multiple_documents(client: OpenAIAssistantsClient) -> tuple[List[str], HostedVectorStoreContent]:
    """Upload multiple documents to a vector store."""
    documents = [
        ("product_manual.txt", b"Product X Setup Guide: 1. Unbox the device. 2. Connect to power. 3. Follow on-screen instructions. 4. Connect to Wi-Fi. 5. Complete registration."),
        ("faq.txt", b"FAQ: Q: How long is the warranty? A: 1 year from purchase date. Q: Can I use it internationally? A: Yes, with the appropriate power adapter. Q: Is it water-resistant? A: No, keep away from water."),
        ("troubleshooting.txt", b"Troubleshooting: If device won't turn on, check power connection. If Wi-Fi fails, restart router. If screen is frozen, hold power button for 10 seconds.")
    ]
    
    # Create files
    file_ids = []
    for filename, content in documents:
        file = await client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        file_ids.append(file.id)
    
    # Create vector store
    vector_store = await client.client.vector_stores.create(
        name="product_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7},
    )
    
    # Add all files to the vector store
    batch = await client.client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store.id,
        file_ids=file_ids
    )
    
    if batch.last_error is not None:
        raise Exception(f"Vector store batch processing failed: {batch.last_error.message}")
    
    return file_ids, HostedVectorStoreContent(vector_store_id=vector_store.id)
```

### Complete Implementation Example

```python
# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

"""
RAG Agent (Retrieval-Augmented Generation) Example

This sample demonstrates a RAG agent that can search through uploaded documents
to provide accurate, context-aware responses.
"""

async def rag_agent_example():
    """Example of a RAG agent with document search."""
    print("=== RAG Agent Example ===")
    
    client = OpenAIAssistantsClient()
    
    async with ChatAgent(
        chat_client=client,
        instructions="You are a helpful assistant that searches through knowledge bases to answer questions. Always cite your sources when providing information.",
        tools=HostedFileSearchTool(),
    ) as agent:
        # Create a vector store with sample documents
        file_id, vector_store = await create_vector_store(client)
        
        try:
            # Example 1: Simple document search
            print("\n--- Simple Document Search ---")
            query1 = "What is the company policy on remote work?"
            print(f"User: {query1}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query1, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
            # Example 2: Complex query requiring synthesis
            print("\n--- Complex Query ---")
            query2 = "What are the requirements for taking a vacation?"
            print(f"User: {query2}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query2, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
            # Example 3: Query with no answer in documents
            print("\n--- Query with No Answer ---")
            query3 = "What is the company's policy on pet insurance?"
            print(f"User: {query3}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query3, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
        finally:
            # Clean up resources
            await delete_vector_store(client, file_id, vector_store.vector_store_id)

async def multi_document_rag_example():
    """Example of a RAG agent with multiple documents."""
    print("\n=== Multi-Document RAG Agent Example ===")
    
    client = OpenAIAssistantsClient()
    
    async with ChatAgent(
        chat_client=client,
        instructions="You are a product support assistant that searches through product documentation to help customers. Provide clear, step-by-step solutions.",
        tools=HostedFileSearchTool(),
    ) as agent:
        # Upload multiple documents
        file_ids, vector_store = await upload_multiple_documents(client)
        
        try:
            # Example 1: Setup question
            print("\n--- Setup Question ---")
            query1 = "How do I set up my Product X?"
            print(f"User: {query1}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query1, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
            # Example 2: Troubleshooting question
            print("\n--- Troubleshooting Question ---")
            query2 = "My device won't turn on, what should I do?"
            print(f"User: {query2}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query2, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
            # Example 3: Cross-document question
            print("\n--- Cross-Document Question ---")
            query3 = "Can I use my Product X internationally and what's the warranty?"
            print(f"User: {query3}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query3, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            
        finally:
            # Clean up resources
            await client.client.vector_stores.delete(vector_store.vector_store_id)
            for file_id in file_ids:
                await client.client.files.delete(file_id=file_id)

async def main():
    """Main function to run all examples."""
    await rag_agent_example()
    await multi_document_rag_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Query Optimization Techniques

1. **Specific Queries**: Encourage users to ask specific questions that can be answered with the available documents.

2. **Query Expansion**: Implement query expansion to improve search results:

```python
def expand_query(query: str) -> str:
    """Expand a query with related terms."""
    expansions = {
        "setup": ["installation", "configuration", "getting started"],
        "troubleshoot": ["fix", "repair", "problem", "issue"],
        "warranty": ["guarantee", "protection", "coverage"]
    }
    
    for term, synonyms in expansions.items():
        if term in query.lower():
            for synonym in synonyms:
                query += f" {synonym}"
    
    return query
```

3. **Result Filtering**: Filter search results based on relevance:

```python
def filter_results(results: List[Dict], threshold: float = 0.7) -> List[Dict]:
    """Filter search results based on relevance threshold."""
    return [result for result in results if result.get("relevance", 0) >= threshold]
```

### Best Practices for Knowledge Bases

1. **Document Quality**: Ensure documents are well-structured, accurate, and up-to-date.

2. **Chunking Strategy**: Break large documents into smaller, coherent chunks for better search results.

3. **Regular Updates**: Regularly update the knowledge base with new information.

4. **Access Control**: Implement proper access controls for sensitive information.

5. **Monitoring**: Monitor search quality and user feedback to continuously improve the system.

## Agent Type 4: Code Execution Agent {#code-execution-agent}

### Use Case Description

Code Execution Agents can write, execute, and analyze code dynamically based on user requests. These agents are ideal for mathematical calculations, data analysis, automation tasks, and educational purposes. They can solve complex problems by generating and executing code in a controlled environment.

### Code Interpreter Setup

The framework provides the HostedCodeInterpreterTool for code execution:

```python
from agent_framework import HostedCodeInterpreterTool

async with ChatAgent(
    chat_client=OpenAIAssistantsClient(),
    instructions="You are a helpful assistant that can write and execute Python code to solve problems.",
    tools=HostedCodeInterpreterTool(),
) as agent:
    result = await agent.run("Calculate the factorial of 10 using Python.")
    print(result)
```

### Security Considerations

When implementing code execution agents, security is paramount:

1. **Sandboxed Environment**: Code should be executed in a sandboxed environment with limited resources.

2. **Input Validation**: Validate user inputs to prevent injection attacks.

3. **Resource Limits**: Set limits on execution time, memory usage, and file access.

4. **Permission Restrictions**: Restrict access to sensitive system resources.

### Complete Implementation Example

```python
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
Code Execution Agent Example

This sample demonstrates a code execution agent that can write and execute
Python code to solve mathematical problems and analyze data.
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

async def mathematical_calculations_example():
    """Example showing mathematical calculations with code execution."""
    print("=== Mathematical Calculations Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code to solve mathematical problems. Show your work and explain each step.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        # Example 1: Factorial calculation
        print("\n--- Factorial Calculation ---")
        query1 = "Calculate the factorial of 10 using Python."
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query1):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 2: Fibonacci sequence
        print("\n--- Fibonacci Sequence ---")
        query2 = "Generate the first 15 numbers in the Fibonacci sequence and plot them."
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query2):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 3: Statistical analysis
        print("\n--- Statistical Analysis ---")
        query3 = "Generate 100 random numbers from a normal distribution and calculate mean, median, and standard deviation."
        print(f"User: {query3}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query3):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")

async def data_analysis_example():
    """Example showing data analysis with code execution."""
    print("\n=== Data Analysis Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a data analyst that can write and execute Python code to analyze data. Use pandas, matplotlib, and other libraries as needed.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        # Example 1: Data visualization
        print("\n--- Data Visualization ---")
        query1 = "Create a bar chart showing the population of the 5 largest countries in the world."
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query1):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 2: Data processing
        print("\n--- Data Processing ---")
        query2 = "Create a DataFrame with student names and grades, then calculate the average grade for each student."
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query2):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")

async def automation_example():
    """Example showing automation tasks with code execution."""
    print("\n=== Automation Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code to automate tasks. Be efficient and provide clear explanations.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        # Example 1: File operations
        print("\n--- File Operations ---")
        query1 = "Write a Python script that creates 10 text files with names 'file1.txt' to 'file10.txt', each containing a random number."
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query1):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 2: Web scraping simulation
        print("\n--- Web Scraping Simulation ---")
        query2 = "Write a Python function that simulates fetching data from a website and parsing it."
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query2):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        print(f"\nGenerated code:\n{generated_code}")

async def main():
    """Main function to run all examples."""
    await mathematical_calculations_example()
    await data_analysis_example()
    await automation_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Output Handling

The framework provides helper methods to access code interpreter output:

```python
def get_code_output(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter output."""
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
                and tool_call.code_interpreter.outputs is not None
            ):
                return "\n".join([output.get("text", "") for output in tool_call.code_interpreter.outputs])
    return None

# Usage in streaming
async for chunk in agent.run_stream(query):
    if chunk.text:
        print(chunk.text, end="", flush=True)
    
    code_output = get_code_output(chunk)
    if code_output:
        print(f"\nCode output:\n{code_output}")
```

### Best Practices and Limitations

1. **Clear Instructions**: Provide clear instructions about what kind of code the agent should generate.

2. **Resource Management**: Be mindful of resource usage when executing code.

3. **Error Handling**: Implement proper error handling for code execution failures.

4. **Security**: Never execute untrusted code without proper sandboxing.

5. **Limitations**: Be aware of limitations such as execution time, memory constraints, and available libraries.

## Agent Type 5: Multi-Modal Agent {#multi-modal-agent}

### Use Case Description

Multi-Modal Agents can process and generate content across multiple modalities, including text, images, and web content. These agents are ideal for applications that need to analyze images, search the web for current information, or combine multiple types of data in their responses. They provide a comprehensive solution for complex, multi-faceted queries.

### Image Analysis Implementation

The framework supports image analysis through vision-capable models:

```python
from agent_framework.openai import OpenAIResponsesClient

async with ChatAgent(
    chat_client=OpenAIResponsesClient(model_id="gpt-4o"),  # Vision-capable model
    instructions="You are a helpful assistant that can analyze images and answer questions about them.",
) as agent:
    # Analyze an image from a file
    with open("image.jpg", "rb") as image_file:
        result = await agent.run(
            "What's in this image?",
            images=[image_file.read()]
        )
    print(result)
```

### Web Search Integration

Web search capabilities enable agents to access current information:

```python
from agent_framework import HostedWebSearchTool

async with ChatAgent(
    chat_client=OpenAIChatClient(model_id="gpt-4o-search-preview"),
    instructions="You are a helpful assistant that can search the web for current information.",
    tools=HostedWebSearchTool(),
) as agent:
    result = await agent.run("What's the current weather in Seattle?")
    print(result)
```

### MCP Tool Integration

Model Context Protocol (MCP) tools enable integration with external services:

```python
from agent_framework import MCPStreamableHTTPTool

async with (
    MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
    ) as mcp_server,
    ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that can access Microsoft documentation.",
    ) as agent,
):
    result = await agent.run("How to create an Azure storage account?", tools=mcp_server)
    print(result)
```

### Complete Multi-Capability Example

```python
# Copyright (c) Microsoft. All rights reserved.

import asyncio
import base64
from io import BytesIO

from agent_framework import ChatAgent, HostedWebSearchTool, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient

"""
Multi-Modal Agent Example

This sample demonstrates a multi-modal agent that can analyze images,
search the web, and integrate with external services through MCP.
"""

async def image_analysis_example():
    """Example showing image analysis capabilities."""
    print("=== Image Analysis Example ===")
    
    # Create a simple test image (in a real application, you would load from file)
    # This is just a placeholder for demonstration
    test_image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(model_id="gpt-4o"),  # Vision-capable model
        instructions="You are a helpful assistant that can analyze images and provide detailed descriptions.",
    ) as agent:
        # Example 1: Basic image analysis
        print("\n--- Basic Image Analysis ---")
        query1 = "What do you see in this image?"
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query1, images=[test_image_data]):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        
        # Example 2: Specific question about image
        print("\n--- Specific Image Question ---")
        query2 = "Is there any text in this image? If so, what does it say?"
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query2, images=[test_image_data]):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

async def web_search_example():
    """Example showing web search capabilities."""
    print("\n=== Web Search Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(model_id="gpt-4o-search-preview"),
        instructions="You are a helpful assistant that can search the web for current information. Always cite your sources.",
        tools=HostedWebSearchTool(),
    ) as agent:
        # Example 1: Current events query
        print("\n--- Current Events Query ---")
        query1 = "What are the latest developments in artificial intelligence?"
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query1):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        
        # Example 2: Specific information query
        print("\n--- Specific Information Query ---")
        query2 = "What's the current stock price of Microsoft?"
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query2):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

async def mcp_integration_example():
    """Example showing MCP tool integration."""
    print("\n=== MCP Integration Example ===")
    
    async with (
        MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ) as mcp_server,
        ChatAgent(
            chat_client=OpenAIChatClient(),
            instructions="You are a helpful assistant that can access Microsoft documentation and provide technical guidance.",
        ) as agent,
    ):
        # Example 1: Documentation query
        print("\n--- Documentation Query ---")
        query1 = "How to create an Azure storage account using the Azure CLI?"
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query1, tools=mcp_server):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        
        # Example 2: Technical guidance
        print("\n--- Technical Guidance ---")
        query2 = "What are the best practices for securing a web application?"
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query2, tools=mcp_server):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

async def multi_modal_combination_example():
    """Example combining multiple modalities."""
    print("\n=== Multi-Modal Combination Example ===")
    
    # Create a simple test image (placeholder)
    test_image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    
    async with (
        MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ) as mcp_server,
        ChatAgent(
            chat_client=OpenAIResponsesClient(model_id="gpt-4o"),
            instructions="You are a helpful assistant that can analyze images, search the web, and access documentation. Use all available tools to provide comprehensive answers.",
            tools=HostedWebSearchTool(),
        ) as agent,
    ):
        # Example 1: Complex query requiring multiple modalities
        print("\n--- Complex Multi-Modal Query ---")
        query1 = "I have this image of a computer setup. Can you identify the components and then search for the latest prices for similar equipment?"
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(
            query1, 
            images=[test_image_data],
            tools=[mcp_server, HostedWebSearchTool()]
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        
        # Example 2: Technical problem solving
        print("\n--- Technical Problem Solving ---")
        query2 = "I'm trying to set up a development environment for Python on Windows. Can you search for the latest best practices and also provide documentation from Microsoft?"
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query2, tools=[mcp_server, HostedWebSearchTool()]):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

async def main():
    """Main function to run all examples."""
    await image_analysis_example()
    await web_search_example()
    await mcp_integration_example()
    await multi_modal_combination_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Best Practices for Complex Agents

1. **Clear Instructions**: Provide detailed instructions that guide the agent on when and how to use different modalities.

2. **Tool Selection**: Carefully select and configure tools to ensure they work well together.

3. **Error Handling**: Implement robust error handling for each modality.

4. **Resource Management**: Be mindful of resource usage, especially when processing images or making web requests.

5. **User Experience**: Design the interaction flow to provide clear feedback to users about what the agent is doing.

## Advanced Topics {#advanced-topics}

### Thread Persistence Strategies

Maintaining conversation state across sessions is crucial for many applications:

```python
# Save thread state to database
async def save_thread_state(thread_id: str, user_id: str):
    """Save thread state to persistent storage."""
    # Implementation depends on your database
    pass

# Load thread state from database
async def load_thread_state(user_id: str):
    """Load thread state from persistent storage."""
    # Implementation depends on your database
    pass

# Usage in application
async def continue_conversation(user_id: str, message: str):
    """Continue a conversation for a user."""
    # Load existing thread state
    thread_id = await load_thread_state(user_id)
    
    if thread_id:
        # Continue existing conversation
        async with ChatAgent(
            chat_client=OpenAIChatClient(thread_id=thread_id),
            instructions="You are a helpful assistant.",
        ) as agent:
            thread = AgentThread(service_thread_id=thread_id)
            result = await agent.run(message, thread=thread)
    else:
        # Start new conversation
        async with ChatAgent(
            chat_client=OpenAIChatClient(),
            instructions="You are a helpful assistant.",
        ) as agent:
            thread = agent.get_new_thread()
            result = await agent.run(message, thread=thread)
            thread_id = thread.service_thread_id
            await save_thread_state(thread_id, user_id)
    
    return result
```

### Custom Message Stores

For advanced use cases, you might want to implement custom message stores:

```python
from agent_framework import ChatMessageStore, ChatMessage

class DatabaseMessageStore(ChatMessageStore):
    """Custom message store that uses a database."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def add_message(self, message: ChatMessage) -> None:
        """Add a message to the database."""
        # Implementation depends on your database
        pass
    
    async def list_messages(self) -> List[ChatMessage]:
        """List all messages from the database."""
        # Implementation depends on your database
        pass
    
    async def clear_messages(self) -> None:
        """Clear all messages from the database."""
        # Implementation depends on your database
        pass

# Usage with custom message store
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
) as agent:
    thread = agent.get_new_thread()
    thread.message_store = DatabaseMessageStore(db_connection)
    
    result = await agent.run("Hello", thread=thread)
    # Messages are now stored in the database
```

### Approval Workflows for Hosted MCP

For security-sensitive applications, you might want to implement approval workflows:

```python
async def approve_mcp_tool_use(tool_name: str, parameters: dict) -> bool:
    """Approve or deny the use of an MCP tool."""
    # Implement your approval logic here
    # This could involve sending a notification to an administrator
    # or checking against a policy engine
    return True  # or False

# Usage with approval workflow
async with ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
) as agent:
    # Override the tool execution to include approval
    original_run = agent.run
    
    async def approved_run(query, **kwargs):
        # Check if any MCP tools are being used
        if "tools" in kwargs:
            for tool in kwargs["tools"]:
                if isinstance(tool, MCPStreamableHTTPTool):
                    # Get tool parameters (this is simplified)
                    if await approve_mcp_tool_use(tool.name, {}):
                        return await original_run(query, **kwargs)
                    else:
                        return "Tool use was not approved."
        
        # No MCP tools or all approved
        return await original_run(query, **kwargs)
    
    agent.run = approved_run
    
    result = await agent.run("Execute a sensitive operation", tools=[mcp_tool])
    print(result)
```

### Structured Outputs with Pydantic

For applications that require structured data, you can use Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: float
    wind_speed: float

class TravelRecommendation(BaseModel):
    destination: str
    activities: List[str]
    best_time_to_visit: str
    estimated_cost: float

# Usage with structured outputs
async with ChatAgent(
    chat_client=OpenAIResponsesClient(),
    instructions="You are a helpful travel assistant that provides structured recommendations.",
) as agent:
    # Request structured output
    query = "Provide a travel recommendation for a weekend trip to Seattle."
    print(f"User: {query}")
    
    # This is a simplified example - actual implementation may vary
    result = await agent.run(query, response_format=TravelRecommendation)
    
    # Parse the structured response
    recommendation = TravelRecommendation.parse_raw(result)
    print(f"Destination: {recommendation.destination}")
    print(f"Activities: {', '.join(recommendation.activities)}")
    print(f"Best time to visit: {recommendation.best_time_to_visit}")
    print(f"Estimated cost: ${recommendation.estimated_cost}")
```

### Performance Optimization

To optimize performance for production applications:

1. **Caching**: Implement caching for frequently accessed data:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_weather_cached(location: str) -> str:
    """Cached version of weather function."""
    return get_weather(location)
```

2. **Batch Processing**: Process multiple requests in batches when possible:

```python
async def batch_process(queries: List[str]) -> List[str]:
    """Process multiple queries in batch."""
    results = []
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    ) as agent:
        for query in queries:
            result = await agent.run(query)
            results.append(result)
    return results
```

3. **Connection Pooling**: Reuse connections when possible:

```python
# Create a single client instance and reuse it
client = OpenAIChatClient()

async def process_query(query: str) -> str:
    """Process a single query using the shared client."""
    agent = client.create_agent(
        instructions="You are a helpful assistant.",
    )
    return await agent.run(query)
```

### Testing Strategies

Implement comprehensive testing for your agents:

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_weather_agent():
    """Test the weather agent."""
    with patch('agent_framework.openai.OpenAIChatClient') as mock_client:
        # Setup mock
        mock_agent = AsyncMock()
        mock_agent.run.return_value = "The weather in Seattle is sunny with a high of 25°C."
        mock_client.return_value.create_agent.return_value = mock_agent
        
        # Test
        agent = OpenAIChatClient().create_agent(
            instructions="You are a helpful weather assistant.",
            tools=[get_weather],
        )
        
        result = await agent.run("What's the weather like in Seattle?")
        
        # Assertions
        assert "sunny" in result
        mock_agent.run.assert_called_once_with("What's the weather like in Seattle?")

@pytest.mark.asyncio
async def test_function_calling():
    """Test function calling."""
    with patch('agent_framework.openai.OpenAIChatClient') as mock_client:
        # Setup mock
        mock_agent = AsyncMock()
        mock_agent.run.return_value = "The current time is 12:00 PM."
        mock_client.return_value.create_agent.return_value = mock_agent
        
        # Test
        agent = OpenAIChatClient().create_agent(
            instructions="You are a helpful assistant.",
            tools=[get_time],
        )
        
        result = await agent.run("What time is it?")
        
        # Assertions
        assert "12:00 PM" in result
        mock_agent.run.assert_called_once_with("What time is it?")
```

## Best Practices and Patterns {#best-practices}

### Resource Management with Context Managers

Always use context managers to ensure proper resource cleanup:

```python
# Good: Using context manager
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_tools,
) as agent:
    result = await agent.run("Help me with a task.")
    # Agent is automatically cleaned up

# Bad: Manual resource management
agent = OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_tools,
)
result = await agent.run("Help me with a task.")
# Agent is not cleaned up, may cause resource leaks
```

### Error Handling and Retry Logic

Implement robust error handling and retry logic:

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_agent_with_retry(agent, query):
    """Run an agent with retry logic."""
    try:
        return await agent.run(query)
    except Exception as e:
        print(f"Error running agent: {e}")
        raise

# Usage
try:
    result = await run_agent_with_retry(agent, "What's the weather like?")
except Exception as e:
    print(f"Failed after retries: {e}")
    # Handle the error appropriately
```

### Logging and Debugging

Implement comprehensive logging for debugging and monitoring:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_agent_with_logging(agent, query):
    """Run an agent with logging."""
    logger.info(f"Running agent with query: {query}")
    try:
        result = await agent.run(query)
        logger.info(f"Agent response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise

# Usage
result = await run_agent_with_logging(agent, "What's the weather like?")
```

### Production Considerations

When deploying to production:

1. **Environment Variables**: Use environment variables for configuration:

```python
import os

# Load configuration from environment variables
api_key = os.environ.get("OPENAI_API_KEY")
model_id = os.environ.get("OPENAI_MODEL_ID", "gpt-4o-mini")

# Create client with configuration
client = OpenAIChatClient(
    api_key=api_key,
    model_id=model_id,
)
```

2. **Rate Limiting**: Implement rate limiting to avoid API limits:

```python
import asyncio
from asyncio import Semaphore

# Create a semaphore to limit concurrent requests
semaphore = Semaphore(5)  # Allow 5 concurrent requests

async def run_agent_with_rate_limit(agent, query):
    """Run an agent with rate limiting."""
    async with semaphore:
        return await agent.run(query)
```

3. **Monitoring**: Implement monitoring to track performance and errors:

```python
import time
from prometheus_client import Counter, Histogram

# Define metrics
request_count = Counter('agent_requests_total', 'Total agent requests', ['agent_type'])
request_duration = Histogram('agent_request_duration_seconds', 'Agent request duration')

async def run_agent_with_monitoring(agent, query, agent_type):
    """Run an agent with monitoring."""
    start_time = time.time()
    request_count.labels(agent_type=agent_type).inc()
    
    try:
        result = await agent.run(query)
        return result
    finally:
        request_duration.observe(time.time() - start_time)
```

### Security Best Practices

1. **API Key Management**: Never hardcode API keys in your code:

```python
# Bad: Hardcoded API key
client = OpenAIChatClient(api_key="sk-1234567890")

# Good: Environment variable
client = OpenAIChatClient(api_key=os.environ.get("OPENAI_API_KEY"))
```

2. **Input Validation**: Validate user inputs to prevent injection attacks:

```python
def validate_query(query: str) -> str:
    """Validate user query."""
    # Remove potentially harmful characters
    query = query.replace("<", "").replace(">", "")
    
    # Limit length
    if len(query) > 1000:
        raise ValueError("Query too long")
    
    return query

# Usage
validated_query = validate_query(user_input)
result = await agent.run(validated_query)
```

3. **Content Filtering**: Implement content filtering for inappropriate content:

```python
def filter_content(content: str) -> str:
    """Filter inappropriate content."""
    # Implement your content filtering logic
    # This could involve using a content moderation API
    return content

# Usage
filtered_result = filter_content(result)
print(filtered_result)
```

## Troubleshooting Guide {#troubleshooting}

### Common Errors and Solutions

1. **API Key Error**:
   ```
   Error: Invalid API key
   ```
   **Solution**: Check that your API key is correct and properly set in environment variables.

2. **Model Not Found Error**:
   ```
   Error: Model not found
   ```
   **Solution**: Verify that the model ID is correct and available in your region.

3. **Rate Limit Error**:
   ```
   Error: Rate limit exceeded
   ```
   **Solution**: Implement rate limiting and retry logic with exponential backoff.

4. **Thread Not Found Error**:
   ```
   Error: Thread not found
   ```
   **Solution**: Ensure the thread ID is correct and hasn't expired.

5. **Tool Execution Error**:
   ```
   Error: Tool execution failed
   ```
   **Solution**: Check the tool implementation and ensure it's properly defined.

### Debugging Techniques

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Inspect Agent State**:
   ```python
   print(f"Agent instructions: {agent.instructions}")
   print(f"Agent tools: {agent.tools}")
   ```

3. **Check Thread State**:
   ```python
   print(f"Thread ID: {thread.service_thread_id}")
   print(f"Thread messages: {await thread.message_store.list_messages()}")
   ```

### API Rate Limiting Handling

Implement proper rate limiting to avoid API limits:

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(lambda e: "rate limit" in str(e).lower())
)
async def run_agent_with_rate_limit_retry(agent, query):
    """Run an agent with rate limit retry logic."""
    return await agent.run(query)
```

### Thread State Issues

To troubleshoot thread state issues:

1. **Check Thread Existence**:
   ```python
   try:
       result = await agent.run(query, thread=thread)
   except Exception as e:
       if "not found" in str(e).lower():
           # Thread doesn't exist, create a new one
           thread = agent.get_new_thread()
           result = await agent.run(query, thread=thread)
       else:
           raise
   ```

2. **Reset Thread State**:
   ```python
   # Clear thread messages
   await thread.message_store.clear_messages()
   ```

3. **Verify Thread Continuity**:
   ```python
   # Check if thread is still valid
   messages = await thread.message_store.list_messages()
   if not messages:
       print("Thread appears to be empty or invalid")
   ```

---

## Conclusion

This comprehensive guide has covered five distinct types of AI agents that can be built using the Microsoft Agent Framework:

1. **Basic Conversational Agents** - For simple chat interactions
2. **Function-Calling Agents** - For integrating custom functions and APIs
3. **RAG Agents** - For knowledge retrieval with file search
4. **Code Execution Agents** - For dynamic code generation and execution
5. **Multi-Modal Agents** - For image analysis, web search, and complex integrations

Each agent type serves different use cases and can be customized to meet specific requirements. By following the examples and best practices outlined in this guide, developers can quickly become productive with the framework and build sophisticated AI-powered applications.

The Microsoft Agent Framework provides a powerful, flexible foundation for building AI agents, with support for various client types, tool integrations, and advanced features. Whether you're building a simple chatbot or a complex multi-modal system, the framework offers the tools and capabilities you need.

As you continue to explore the framework, remember to follow the best practices for security, performance, and maintainability. And don't hesitate to experiment with different agent types and configurations to find the best solution for your specific use case.

https://chat.z.ai/s/31ff1d36-66c8-4a40-8d2f-bb424fa0d300