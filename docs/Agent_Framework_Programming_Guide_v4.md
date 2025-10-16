# Comprehensive Guide to Building AI Agents with Microsoft Agent Framework

## Table of Contents

1. [Introduction & Getting Started](#introduction--getting-started)
2. [Core Concepts](#core-concepts)
3. [Agent Type #1: Basic Conversational Agent](#agent-type-1-basic-conversational-agent)
4. [Agent Type #2: RAG Agent (Retrieval-Augmented Generation)](#agent-type-2-rag-agent-retrieval-augmented-generation)
5. [Agent Type #3: Code Execution Agent](#agent-type-3-code-execution-agent)
6. [Agent Type #4: Multi-Modal Agent](#agent-type-4-multi-modal-agent)
7. [Agent Type #5: MCP-Integrated Agent](#agent-type-5-mcp-integrated-agent)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices & Production Considerations](#best-practices--production-considerations)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Quick Reference & Next Steps](#quick-reference--next-steps)

---

## Introduction & Getting Started

### Framework Overview

The Microsoft Agent Framework is a powerful, unified interface for building sophisticated AI agents across different OpenAI and Azure services. It abstracts away the complexity of working with various AI services while providing a consistent developer experience regardless of the underlying client type.

The framework enables developers to create agents that can:
- Engage in natural language conversations
- Execute code and perform calculations
- Search and retrieve information from documents
- Analyze images and other media
- Integrate with external systems through Model Context Protocol (MCP)
- Maintain conversation context across multiple interactions

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Microsoft Agent Framework                │
├─────────────────────────────────────────────────────────────┤
│  ChatAgent (High-level abstraction)                        │
│  ├── Thread Management                                     │
│  ├── Tool Integration                                      │
│  ├── Streaming Support                                     │
│  └── Lifecycle Management                                  │
├─────────────────────────────────────────────────────────────┤
│  Client Types (Abstraction Layer)                          │
│  ├── OpenAIAssistantsClient                                │
│  ├── OpenAIChatClient                                      │
│  ├── OpenAIResponsesClient                                 │
│  └── (Azure variants for each)                             │
├─────────────────────────────────────────────────────────────┤
│  External Services                                          │
│  ├── OpenAI Assistants API                                 │
│  ├── OpenAI Chat API                                       │
│  ├── OpenAI Responses API                                  │
│  └── Azure OpenAI Services                                 │
└─────────────────────────────────────────────────────────────┘
```

### Client Types Comparison

| Client Type | Best For | Key Features | State Management |
|-------------|----------|--------------|------------------|
| **Assistants** | Complex, multi-turn conversations | Built-in thread management, file search, code interpreter | Service-managed |
| **Chat** | Direct chat interactions | Simple API, streaming support, function calling | Client-managed |
| **Responses** | Structured output generation | Precise output control, reasoning capabilities | Stateless |

### Environment Setup

Before building agents with the framework, ensure you have the following:

1. **Install the required packages**:
   ```bash
   pip install agent-framework openai pydantic
   ```

2. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export OPENAI_CHAT_MODEL_ID="gpt-4o"  # or gpt-4o-mini, gpt-3.5-turbo
   export OPENAI_RESPONSES_MODEL_ID="gpt-4o"  # or gpt-4o-mini, gpt-3.5-turbo
   ```

3. **Optional environment variables**:
   ```bash
   export OPENAI_ORG_ID="your-organization-id"  # if applicable
   export OPENAI_API_BASE_URL="your-custom-base-url"  # if using a different base URL
   ```

### Quick Start Example

Let's start with a simple "Hello World" example to demonstrate the basic usage:

```python
import asyncio
from agent_framework.openai import OpenAIChatClient

async def hello_world():
    # Create a simple agent
    agent = OpenAIChatClient().create_agent(
        name="HelloAgent",
        instructions="You are a friendly assistant that greets people.",
    )
    
    # Send a message and get a response
    response = await agent.run("Hello! Can you introduce yourself?")
    print(response)
    
    # Or use streaming for real-time responses
    print("Streaming response: ", end="", flush=True)
    async for chunk in agent.run_stream("Tell me a fun fact."):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # New line after streaming

if __name__ == "__main__":
    asyncio.run(hello_world())
```

This simple example demonstrates the core pattern of the framework:
1. Create a client
2. Create an agent with instructions
3. Run queries with either streaming or non-streaming responses

---

## Core Concepts

### Client Types Deep Dive

#### OpenAIAssistantsClient

The Assistants client is designed for complex, multi-turn conversations where state needs to be maintained across interactions. It leverages OpenAI's Assistants API which provides built-in thread management, file search capabilities, and code execution.

Key characteristics:
- Service-managed conversation threads
- Built-in file search and code interpreter tools
- Supports complex tool workflows
- Automatic thread lifecycle management

Example usage:
```python
from agent_framework.openai import OpenAIAssistantsClient

# Create an assistant with automatic lifecycle management
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful research assistant.",
    tools=[get_weather, get_time],
) as agent:
    response = await agent.run("What's the weather in Tokyo and what time is it?")
    print(response)
```

#### OpenAIChatClient

The Chat client provides a direct interface to OpenAI's Chat Completions API. It's ideal for simpler, stateless interactions where you want more control over the conversation flow.

Key characteristics:
- Client-managed conversation state
- Simpler API compared to Assistants
- Faster response times
- More granular control over messages

Example usage:
```python
from agent_framework.openai import OpenAIChatClient

# Create a chat agent
agent = OpenAIChatClient().create_agent(
    name="ChatAgent",
    instructions="You are a helpful assistant.",
    tools=[get_weather],
)

# Send a message and get a response
response = await agent.run("What's the weather in New York?")
print(response)
```

#### OpenAIResponsesClient

The Responses client is designed for structured output generation where you need precise control over the format of the response. It's particularly useful when you need the AI to generate data in a specific format.

Key characteristics:
- Precise output control
- Structured data generation
- Reasoning capabilities
- Stateless interactions

Example usage:
```python
from agent_framework.openai import OpenAIResponsesClient

# Create a responses agent
agent = ChatAgent(
    chat_client=OpenAIResponsesClient(),
    instructions="You are a helpful assistant that provides structured responses.",
    tools=[get_weather],
)

# Send a message and get a structured response
response = await agent.run("What's the weather in London?")
print(response)
```

### Agent Lifecycle Management

The framework provides clear patterns for managing agent lifecycles:

1. **Automatic Lifecycle Management** (Recommended for most cases):
   ```python
   # Using context managers for automatic cleanup
   async with OpenAIAssistantsClient().create_agent(
       instructions="You are a helpful assistant.",
   ) as agent:
       response = await agent.run("Hello!")
       print(response)
   # Agent and resources are automatically cleaned up
   ```

2. **Manual Lifecycle Management** (For advanced scenarios):
   ```python
   # Creating and managing agents manually
   client = OpenAIAssistantsClient()
   agent = client.create_agent(
       instructions="You are a helpful assistant.",
   )
   
   try:
       response = await agent.run("Hello!")
       print(response)
   finally:
       # Manual cleanup
       await agent.cleanup()
   ```

### Thread Management Patterns

Thread management is crucial for maintaining conversation context across multiple interactions. The framework offers several patterns:

| Pattern | Description | Use Case | Example |
|---------|-------------|----------|---------|
| **Automatic Thread Creation** | Framework creates a new thread for each interaction | Stateless conversations, simple Q&A | `await agent.run("What's the weather?")` |
| **Thread Persistence** | Reuse the same thread across multiple interactions | Conversations that need context | `await agent.run("How about London?", thread=thread)` |
| **Thread ID Reuse** | Continue a conversation using an existing thread ID | Resuming previous conversations | `agent = ChatAgent(chat_client=OpenAIAssistantsClient(thread_id="thread_123"))` |

Example of thread persistence:
```python
from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIChatClient

# Create an agent
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful weather agent.",
    tools=get_weather,
)

# Create a new thread that will be reused
thread = agent.get_new_thread()

# First conversation
query1 = "What's the weather like in Tokyo?"
result1 = await agent.run(query1, thread=thread)
print(f"Agent: {result1.text}")

# Second conversation using the same thread - maintains context
query2 = "How about London?"
result2 = await agent.run(query2, thread=thread)
print(f"Agent: {result2.text}")

# Third conversation - agent should remember both previous cities
query3 = "Which of the cities I asked about has better weather?"
result3 = await agent.run(query3, thread=thread)
print(f"Agent: {result3.text}")
```

### Tool Integration Approaches

The framework supports several approaches to tool integration:

1. **Agent-Level Tools** (Available for all queries during the agent's lifetime):
   ```python
   # Tools defined when creating the agent
   agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="You are a helpful assistant.",
       tools=[get_weather, get_time],  # Tools available for all queries
   )
   
   # Agent can use any of these tools for any query
   response = await agent.run("What's the weather in New York and what time is it?")
   ```

2. **Run-Level Tools** (Specific to a single query):
   ```python
   # Agent created without tools
   agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="You are a helpful assistant.",
   )
   
   # Tools provided for a specific query
   response = await agent.run(
       "What's the weather in Seattle?", 
       tools=[get_weather]  # Tool only available for this query
   )
   ```

3. **Mixed Approach** (Combining agent-level and run-level tools):
   ```python
   # Agent created with some base tools
   agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="You are a comprehensive assistant.",
       tools=[get_weather],  # Base tool available for all queries
   )
   
   # Query using both agent tool and additional run-method tools
   response = await agent.run(
       "What's the weather in Denver and what's the current UTC time?",
       tools=[get_time],  # Additional tool for this specific query
   )
   ```

### Streaming vs Non-Streaming Responses

The framework supports both streaming and non-streaming responses:

**Non-Streaming** (Get the complete result at once):
```python
# Non-streaming response
response = await agent.run("What's the weather like in Seattle?")
print(f"Agent: {response}")
```

**Streaming** (Get results as they are generated):
```python
# Streaming response
print("Agent: ", end="", flush=True)
async for chunk in agent.run_stream("What's the weather like in Portland?"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()  # New line after streaming
```

### Error Handling

Proper error handling is crucial for building robust agents:

```python
import asyncio
from agent_framework.openai import OpenAIChatClient

async def robust_agent_example():
    try:
        agent = OpenAIChatClient().create_agent(
            name="RobustAgent",
            instructions="You are a helpful assistant.",
        )
        
        try:
            response = await agent.run("Tell me a joke.")
            print(response)
        except Exception as e:
            print(f"Error during agent execution: {e}")
            # Implement retry logic or fallback behavior
            
    except Exception as e:
        print(f"Error creating agent: {e}")
        # Handle initialization errors
    finally:
        # Cleanup resources if needed
        pass

if __name__ == "__main__":
    asyncio.run(robust_agent_example())
```

---

## Agent Type #1: Basic Conversational Agent

### Use Case & Architecture

Basic conversational agents are the foundation of many AI applications. They're designed to engage in natural language conversations, answer questions, and perform simple tasks through function calling. These agents are ideal for customer service, personal assistants, and information retrieval systems.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Basic Conversational Agent               │
├─────────────────────────────────────────────────────────────┤
│  Components:                                               │
│  ├── ChatAgent (Core abstraction)                          │
│  ├── OpenAIChatClient (Interface to OpenAI)                │
│  ├── Function Tools (For task execution)                   │
│  └── Thread Management (For conversation context)          │
├─────────────────────────────────────────────────────────────┤
│  Capabilities:                                             │
│  ├── Natural language understanding                        │
│  ├── Function calling for task execution                   │
│  ├── Context maintenance across turns                      │
│  └── Streaming and non-streaming responses                 │
└─────────────────────────────────────────────────────────────┘
```

### Complete Implementation

Let's build a complete basic conversational agent that can answer questions and perform tasks:

```python
import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

"""
Basic Conversational Agent Example

This sample demonstrates a complete conversational agent with function tools,
thread management, and both streaming and non-streaming responses.
"""

# Define function tools that the agent can use
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

def get_date() -> str:
    """Get the current date."""
    current_date = datetime.now(timezone.utc)
    return f"Today's date is {current_date.strftime('%Y-%m-%d')}."

async def basic_conversational_agent():
    """Example of a basic conversational agent with function tools."""
    print("=== Basic Conversational Agent Example ===")
    
    # Create the agent with function tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="ConversationalAgent",
        instructions="You are a helpful assistant that can provide weather information, time, and date. Be friendly and conversational.",
        tools=[get_weather, get_time, get_date],
    )
    
    # Example 1: Non-streaming response
    print("\n--- Non-streaming Response Example ---")
    query1 = "What's the weather like in Tokyo and what time is it now?"
    print(f"User: {query1}")
    response1 = await agent.run(query1)
    print(f"Agent: {response1}")
    
    # Example 2: Streaming response
    print("\n--- Streaming Response Example ---")
    query2 = "Tell me about the weather in three different cities and today's date."
    print(f"User: {query2}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query2):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # New line after streaming
    
    # Example 3: Conversation with context
    print("\n--- Conversation with Context Example ---")
    from agent_framework import AgentThread
    
    # Create a thread for conversation context
    thread = agent.get_new_thread()
    
    # First message
    query3 = "What's the weather like in New York?"
    print(f"User: {query3}")
    response3 = await agent.run(query3, thread=thread)
    print(f"Agent: {response3.text}")
    
    # Follow-up message (agent should remember the previous context)
    query4 = "How does that compare to the weather in London?"
    print(f"User: {query4}")
    response4 = await agent.run(query4, thread=thread)
    print(f"Agent: {response4.text}")
    
    # Another follow-up (agent should remember both cities)
    query5 = "Which city would you recommend for a vacation today?"
    print(f"User: {query5}")
    response5 = await agent.run(query5, thread=thread)
    print(f"Agent: {response5.text}")

if __name__ == "__main__":
    asyncio.run(basic_conversational_agent())
```

### Function Tool Patterns

Function tools are a key capability of conversational agents, allowing them to perform actions and retrieve information. Let's explore different patterns for using function tools:

#### 1. Agent-Level Tools (Available for all queries)

```python
# Tools defined when creating the agent
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant that can provide weather and time information.",
    tools=[get_weather, get_time],  # Tools available for all queries
)

# The agent can use any of these tools for any query
response = await agent.run("What's the weather in New York and what time is it?")
```

#### 2. Run-Level Tools (Specific to a single query)

```python
# Agent created without tools
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
)

# Tools provided for a specific query
response = await agent.run(
    "What's the weather in Seattle?", 
    tools=[get_weather]  # Tool only available for this query
)
```

#### 3. Mixed Approach (Combining agent-level and run-level tools)

```python
# Agent created with some base tools
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a comprehensive assistant.",
    tools=[get_weather],  # Base tool available for all queries
)

# Query using both agent tool and additional run-method tools
response = await agent.run(
    "What's the weather in Denver and what's the current UTC time?",
    tools=[get_time],  # Additional tool for this specific query
)
```

### Best Practices

1. **Design Clear Function Signatures**:
   ```python
   # Good: Clear parameter descriptions
   def get_weather(
       location: Annotated[str, Field(description="The location to get the weather for.")],
   ) -> str:
       """Get the weather for a given location."""
       # Implementation...
   
   # Avoid: Vague parameter descriptions
   def get_weather(loc: str) -> str:
       """Get weather."""
       # Implementation...
   ```

2. **Provide Comprehensive Instructions**:
   ```python
   # Good: Detailed instructions
   agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="You are a helpful assistant that can provide weather information, time, and date. "
                   "When providing weather information, always include the temperature and conditions. "
                   "Be friendly and conversational, but keep your responses concise.",
       tools=[get_weather, get_time, get_date],
   )
   
   # Avoid: Vague instructions
   agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="Help with weather.",
       tools=[get_weather, get_time, get_date],
   )
   ```

3. **Use Thread Management for Context**:
   ```python
   # For conversations that need context
   thread = agent.get_new_thread()
   
   # First message
   response1 = await agent.run("What's the weather in Tokyo?", thread=thread)
   
   # Follow-up message (agent remembers previous context)
   response2 = await agent.run("How does that compare to London?", thread=thread)
   ```

4. **Handle Errors Gracefully**:
   ```python
   try:
       response = await agent.run("What's the weather in Tokyo?")
       print(response)
   except Exception as e:
       print(f"Sorry, I encountered an error: {e}")
       # Implement fallback behavior
   ```

5. **Choose the Right Response Type**:
   ```python
   # For quick, complete responses
   response = await agent.run("What's the weather in Tokyo?")
   
   # For longer responses or real-time interaction
   print("Agent: ", end="", flush=True)
   async for chunk in agent.run_stream("Tell me about the weather in Asia."):
       if chunk.text:
           print(chunk.text, end="", flush=True)
   print()
   ```

---

## Agent Type #2: RAG Agent (Retrieval-Augmented Generation)

### Use Case & Architecture

Retrieval-Augmented Generation (RAG) agents combine the power of large language models with external knowledge retrieval. These agents can search through documents, knowledge bases, and other data sources to provide accurate, up-to-date information. They're ideal for customer support, research assistants, and knowledge management systems.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                        RAG Agent                            │
├─────────────────────────────────────────────────────────────┤
│  Components:                                               │
│  ├── ChatAgent (Core abstraction)                          │
│  ├── OpenAIAssistantsClient (For file search)              │
│  ├── HostedFileSearchTool (For document retrieval)         │
│  ├── Vector Store (For document indexing)                  │
│  └── File Management (For document upload)                 │
├─────────────────────────────────────────────────────────────┤
│  Capabilities:                                             │
│  ├── Document upload and indexing                          │
│  ├── Vector-based similarity search                        │
│  ├── Knowledge retrieval and synthesis                     │
│  └── Context-aware responses based on documents            │
└─────────────────────────────────────────────────────────────┘
```

### Vector Store Setup

Before building a RAG agent, we need to set up a vector store to index our documents:

```python
import asyncio
from agent_framework import HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

async def create_vector_store(client: OpenAIAssistantsClient) -> tuple[str, HostedVectorStoreContent]:
    """Create a vector store with sample documents."""
    # Create a file with sample content
    file = await client.client.files.create(
        file=("company_policy.txt", 
              b"Our company policy states that all employees must complete security training within 30 days of hire. "
              b"Remote work is allowed with manager approval. "
              b"Vacation requests must be submitted at least 2 weeks in advance. "
              b"The standard work week is Monday to Friday, 9 AM to 5 PM."),
        purpose="user_data"
    )
    
    # Create a vector store
    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    
    # Add the file to the vector store and wait for processing
    result = await client.client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id, 
        file_id=file.id
    )
    
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")

    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)

async def delete_vector_store(client: OpenAIAssistantsClient, file_id: str, vector_store_id: str) -> None:
    """Delete the vector store after using it."""
    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)
```

### File Upload and Indexing

For a RAG agent to work effectively, we need to upload and index documents:

```python
async def upload_and_index_documents(client: OpenAIAssistantsClient):
    """Upload and index multiple documents for the RAG agent."""
    documents = [
        ("product_manual.txt", 
         b"Product X is a smart home device that can control lights, thermostats, and security systems. "
         b"Installation requires a stable Wi-Fi connection and the companion mobile app. "
         b"The device supports voice commands through Alexa and Google Assistant. "
         b"Battery life is approximately 6 months on a single charge."),
        
        ("faq.txt", 
         b"Q: How do I reset Product X? A: Press and hold the reset button for 10 seconds. "
         b"Q: Is Product X compatible with older homes? A: Yes, but may require a Wi-Fi extender. "
         b"Q: Can I control Product X when away from home? A: Yes, through the mobile app. "
         b"Q: What is the warranty period? A: Product X comes with a 2-year limited warranty."),
        
        ("troubleshooting.txt", 
         b"If Product X is unresponsive, check the following: 1) Ensure it has power. "
         b"2) Verify Wi-Fi connectivity. 3) Restart the device. "
         b"4) Update firmware through the app. "
         b"If issues persist, contact customer support at support@example.com or 1-800-123-4567.")
    ]
    
    # Create a vector store
    vector_store = await client.client.vector_stores.create(
        name="product_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7},
    )
    
    file_ids = []
    
    # Upload and index each document
    for filename, content in documents:
        file = await client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        
        file_ids.append(file.id)
        
        # Add the file to the vector store
        result = await client.client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file.id
        )
        
        if result.last_error is not None:
            print(f"Warning: File {filename} processing failed with status: {result.last_error.message}")
    
    return file_ids, HostedVectorStoreContent(vector_store_id=vector_store.id)
```

### Complete RAG Implementation

Now let's build a complete RAG agent that can search through documents and answer questions based on the retrieved information:

```python
import asyncio
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

async def rag_agent_example():
    """Example of a RAG agent that can search through documents."""
    print("=== RAG Agent Example ===")
    
    client = OpenAIAssistantsClient()
    
    # Create and upload documents to the vector store
    print("Uploading and indexing documents...")
    file_ids, vector_store = await upload_and_index_documents(client)
    
    try:
        # Create the RAG agent
        async with ChatAgent(
            chat_client=client,
            instructions="You are a helpful product support assistant. Search through the provided documents to answer questions accurately. "
                       "If the information is not available in the documents, say so clearly. "
                       "Always cite the source document when providing information.",
            tools=HostedFileSearchTool(),
        ) as agent:
            
            # Example 1: Product information query
            print("\n--- Product Information Query ---")
            query1 = "What is Product X and what can it control?"
            print(f"User: {query1}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query1, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()  # New line after streaming
            
            # Example 2: Troubleshooting query
            print("\n--- Troubleshooting Query ---")
            query2 = "My Product X is unresponsive. What should I do?"
            print(f"User: {query2}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query2, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()  # New line after streaming
            
            # Example 3: Information not in documents
            print("\n--- Information Not in Documents ---")
            query3 = "When will the next version of Product X be released?"
            print(f"User: {query3}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query3, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()  # New line after streaming
            
            # Example 4: Complex query requiring synthesis
            print("\n--- Complex Query Requiring Synthesis ---")
            query4 = "Compare the installation requirements with the troubleshooting steps for connectivity issues."
            print(f"User: {query4}")
            print("Agent: ", end="", flush=True)
            async for chunk in agent.run_stream(
                query4, 
                tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}}
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()  # New line after streaming
    
    finally:
        # Clean up resources
        print("\nCleaning up resources...")
        await client.client.vector_stores.delete(vector_store.vector_store_id)
        for file_id in file_ids:
            await client.client.files.delete(file_id=file_id)
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(rag_agent_example())
```

### Query Optimization Tips

1. **Specific Queries Work Better**:
   ```python
   # Good: Specific query
   query = "What are the steps to reset Product X if it's unresponsive?"
   
   # Avoid: Vague query
   query = "Tell me about Product X problems."
   ```

2. **Include Context in Instructions**:
   ```python
   # Good: Detailed instructions
   instructions = "You are a helpful product support assistant. Search through the provided documents to answer questions accurately. "
                  "If the information is not available in the documents, say so clearly. "
                  "Always cite the source document when providing information. "
                  "For troubleshooting steps, present them as a numbered list."
   
   # Avoid: Vague instructions
   instructions = "Answer questions based on documents."
   ```

3. **Use Multiple Vector Stores for Different Topics**:
   ```python
   # For different types of documents
   product_vector_store = HostedVectorStoreContent(vector_store_id="product_docs")
   policy_vector_store = HostedVectorStoreContent(vector_store_id="policy_docs")
   
   # Use the appropriate vector store based on the query
   if "product" in query.lower():
       tool_resources = {"file_search": {"vector_store_ids": [product_vector_store.vector_store_id]}}
   elif "policy" in query.lower():
       tool_resources = {"file_search": {"vector_store_ids": [policy_vector_store.vector_store_id]}}
   else:
       tool_resources = {"file_search": {"vector_store_ids": [product_vector_store.vector_store_id, policy_vector_store.vector_store_id]}}
   ```

### Best Practices for Knowledge Bases

1. **Chunk Documents Appropriately**:
   - Break large documents into smaller, focused chunks
   - Ensure each chunk contains complete information on a topic
   - Avoid chunks that are too small (lack context) or too large (reduce relevance)

2. **Maintain Document Metadata**:
   ```python
   # Include metadata when uploading files
   file = await client.client.files.create(
       file=("product_manual.txt", content),
       purpose="user_data",
       metadata={
           "category": "product_manual",
           "product": "Product X",
           "version": "1.2",
           "last_updated": "2023-11-15"
       }
   )
   ```

3. **Regularly Update Knowledge Base**:
   ```python
   # Create a new vector store for updated content
   new_vector_store = await client.client.vector_stores.create(
       name="updated_knowledge_base",
       expires_after={"anchor": "last_active_at", "days": 7},
   )
   
   # Upload updated documents
   # ...
   
   # Update agent to use the new vector store
   tool_resources = {"file_search": {"vector_store_ids": [new_vector_store.id]}}
   ```

4. **Monitor Vector Store Usage**:
   ```python
   # Check vector store status
   vector_store_status = await client.client.vector_stores.retrieve(vector_store_id)
   print(f"Vector store status: {vector_store_status.status}")
   print(f"File count: {vector_store_status.file_counts}")
   ```

### Cleanup Patterns

Proper cleanup is important for managing resources and costs:

```python
async def cleanup_vector_store(client: OpenAIAssistantsClient, vector_store_id: str, file_ids: list[str]):
    """Clean up vector store and associated files."""
    try:
        # Delete the vector store
        await client.client.vector_stores.delete(vector_store_id)
        print(f"Deleted vector store: {vector_store_id}")
        
        # Delete all associated files
        for file_id in file_ids:
            await client.client.files.delete(file_id=file_id)
            print(f"Deleted file: {file_id}")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Usage in a try-finally block
try:
    # Use the vector store
    # ...
finally:
    # Clean up resources
    await cleanup_vector_store(client, vector_store_id, file_ids)
```

---

## Agent Type #3: Code Execution Agent

### Use Case & Architecture

Code execution agents can write and execute code to solve problems, perform calculations, and analyze data. These agents are invaluable for data analysis, mathematical computations, automation tasks, and any scenario where dynamic code execution is needed. They use the HostedCodeInterpreterTool to securely execute Python code in a sandboxed environment.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Code Execution Agent                     │
├─────────────────────────────────────────────────────────────┤
│  Components:                                               │
│  ├── ChatAgent (Core abstraction)                          │
│  ├── OpenAIAssistantsClient (For code execution)           │
│  ├── HostedCodeInterpreterTool (For Python execution)      │
│  └── Code Output Processing (For handling results)         │
├─────────────────────────────────────────────────────────────┤
│  Capabilities:                                             │
│  ├── Dynamic Python code generation                        │
│  ├── Secure code execution in sandboxed environment         │
│  ├── Mathematical computations                             │
│  ├── Data analysis and visualization                       │
│  └── File operations within the execution environment      │
└─────────────────────────────────────────────────────────────┘
```

### Security Considerations

Code execution agents run code in a sandboxed environment, but security is still a critical consideration:

1. **Sandboxed Execution**: Code runs in an isolated environment with limited access to system resources.
2. **Time Limits**: Execution is automatically terminated after a set time to prevent infinite loops.
3. **Resource Constraints**: Memory and CPU usage are limited to prevent resource exhaustion.
4. **No Internet Access**: The execution environment typically doesn't have internet access for security reasons.

### Complete Implementation

Let's build a complete code execution agent that can solve mathematical problems and perform data analysis:

```python
import asyncio
from agent_framework import AgentRunResponseUpdate, ChatAgent, HostedCodeInterpreterTool
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

This sample demonstrates a code execution agent that can write and execute Python code
to solve mathematical problems and perform data analysis.
"""

def get_code_interpreter_chunk(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter data from response chunks."""
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

async def code_execution_agent():
    """Example of a code execution agent that can solve mathematical problems."""
    print("=== Code Execution Agent Example ===")
    
    # Create the code execution agent
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code to solve problems. "
                   "When solving mathematical problems, show your work by explaining your approach before writing code. "
                   "After executing code, explain the results in a clear, understandable way.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        
        # Example 1: Mathematical calculation
        print("\n--- Mathematical Calculation Example ---")
        query1 = "Calculate the factorial of 100 and explain how large it is compared to the number of atoms in the observable universe."
        print(f"User: {query1}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query1):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Capture the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 2: Data analysis
        print("\n--- Data Analysis Example ---")
        query2 = "Create a sample dataset of monthly sales figures for a year, calculate basic statistics, and identify the best and worst performing months."
        print(f"User: {query2}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query2):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Capture the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 3: Algorithm implementation
        print("\n--- Algorithm Implementation Example ---")
        query3 = "Implement a sorting algorithm and test it with a random list of 1000 numbers. Measure the execution time."
        print(f"User: {query3}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query3):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Capture the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        
        print(f"\nGenerated code:\n{generated_code}")
        
        # Example 4: File operations
        print("\n--- File Operations Example ---")
        query4 = "Create a CSV file with sample employee data, read it back, and calculate the average salary by department."
        print(f"User: {query4}")
        print("Agent: ", end="", flush=True)
        generated_code = ""
        async for chunk in agent.run_stream(query4):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Capture the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        
        print(f"\nGenerated code:\n{generated_code}")

if __name__ == "__main__":
    asyncio.run(code_execution_agent())
```

### Output Handling Code Examples

The code interpreter generates both code and output. Here's how to handle both:

```python
async def code_with_output_handling():
    """Example showing how to handle both code and output from the code interpreter."""
    print("=== Code and Output Handling Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        
        query = "Generate a list of 10 random numbers, calculate their mean and standard deviation, and create a simple visualization."
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        
        generated_code = ""
        code_outputs = []
        
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Capture the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
                print("\n[CODE EXECUTION]\n", code_interpreter_chunk, "\n[/CODE EXECUTION]\n", end="", flush=True)
            
            # Capture code outputs (if available)
            # Note: Accessing outputs may require additional processing depending on the framework version
            # This is a conceptual example
            if hasattr(chunk, 'code_output') and chunk.code_output:
                code_outputs.append(chunk.code_output)
                print(f"[OUTPUT]\n{chunk.code_output}\n[/OUTPUT]\n", end="", flush=True)
        
        print(f"\nComplete generated code:\n{generated_code}")
        
        if code_outputs:
            print("\nCode outputs:")
            for i, output in enumerate(code_outputs, 1):
                print(f"Output {i}:\n{output}")
```

### Accessing Generated Code from Responses

To access the code generated by the agent:

```python
async def extract_generated_code():
    """Example showing how to extract the code generated by the agent."""
    print("=== Extracting Generated Code Example ===")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code.",
        tools=HostedCodeInterpreterTool(),
    ) as agent:
        
        query = "Write a Python function to calculate the Fibonacci sequence and test it with the first 20 numbers."
        print(f"User: {query}")
        
        generated_code = ""
        
        # Non-streaming approach
        response = await agent.run(query)
        print(f"Agent: {response}")
        
        # To extract the code, we need to use streaming
        print("\nExtracting generated code:")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            
            # Extract the generated code
            code_interpreter_chunk = get_code_interpreter_chunk(chunk)
            if code_interpreter_chunk is not None:
                generated_code += code_interpreter_chunk
        
        print(f"\n\nExtracted code:\n{generated_code}")
        
        # Save the extracted code to a file
        with open("fibonacci.py", "w") as f:
            f.write(generated_code)
        print("Code saved to fibonacci.py")
```

### Best Practices and Limitations

1. **Be Specific in Your Requests**:
   ```python
   # Good: Specific request
   query = "Write a Python function to calculate the factorial of a number using recursion. Test it with the number 10."
   
   # Avoid: Vague request
   query = "Write some code about factorials."
   ```

2. **Provide Clear Context**:
   ```python
   # Good: Clear context
   query = "I'm analyzing sales data for a retail store. Write Python code to calculate the month-over-month growth rate from the following data: [1000, 1200, 1100, 1300, 1500]"
   
   # Avoid: Unclear context
   query = "Analyze this data: [1000, 1200, 1100, 1300, 1500]"
   ```

3. **Break Complex Tasks into Steps**:
   ```python
   # Good: Step-by-step approach
   query = "First, create a list of 100 random numbers between 1 and 1000. Then, sort the list using bubble sort. Finally, measure how long the sorting takes."
   
   # Avoid: All at once
   query = "Create, sort, and time a list of random numbers."
   ```

4. **Understand Limitations**:
   - No internet access
   - Limited execution time
   - Limited memory and CPU
   - No access to local files (except within the sandboxed environment)

### Use Case Matrix

| Use Case | Example Query | Considerations |
|----------|---------------|----------------|
| **Mathematical Calculations** | "Calculate the compound interest on a $10,000 investment at 5% annual rate over 10 years." | Ensure the agent shows the formula before calculation |
| **Data Analysis** | "Analyze this dataset and identify trends: [data]" | Provide clear instructions on what analysis to perform |
| **Algorithm Implementation** | "Implement a binary search algorithm and test it with a sorted list." | Specify the requirements and test cases |
| **Data Visualization** | "Create a bar chart showing monthly sales data for the past year." | Specify the type of visualization and data format |
| **File Processing** | "Read a CSV file, filter rows based on a condition, and save the result." | Provide the file structure or sample data |

---

## Agent Type #4: Multi-Modal Agent

### Use Case & Architecture

Multi-modal agents can process and analyze various types of data, including text, images, and information from the web. These agents are ideal for applications that need to understand visual content, retrieve current information, and perform complex reasoning across different data types. They combine vision capabilities, web search, and reasoning to provide comprehensive responses.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     Multi-Modal Agent                       │
├─────────────────────────────────────────────────────────────┤
│  Components:                                               │
│  ├── ChatAgent (Core abstraction)                          │
│  ├── OpenAIResponsesClient (For vision capabilities)        │
│  ├── HostedWebSearchTool (For web search)                  │
│  ├── Image Processing (For visual analysis)                │
│  └── Reasoning Engine (For complex reasoning)              │
├─────────────────────────────────────────────────────────────┤
│  Capabilities:                                             │
│  ├── Image analysis and understanding                      │
│  ├── Real-time web search for current information          │
│  ├── Complex reasoning across multiple data types          │
│  ├── Structured output generation                          │
│  └── Integration of multiple information sources           │
└─────────────────────────────────────────────────────────────┘
```

### Image Analysis Implementation

Let's implement image analysis capabilities using the OpenAIResponsesClient:

```python
import asyncio
import base64
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient

async def image_analysis_agent():
    """Example of an agent that can analyze images."""
    print("=== Image Analysis Agent Example ===")
    
    # Create the image analysis agent
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant that can analyze images. "
                   "Provide detailed descriptions of what you see in images, "
                   "identify objects, people, and scenes, and answer questions about the visual content.",
    )
    
    # Example 1: Analyze an image from a file
    print("\n--- Image Analysis from File ---")
    
    # Read and encode the image
    with open("example_image.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    query = "Describe this image in detail."
    print(f"User: {query} (with image)")
    
    # Include the image in the query
    response = await agent.run(
        query,
        images=[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        }]
    )
    
    print(f"Agent: {response}")
    
    # Example 2: Answer questions about an image
    print("\n--- Question Answering about Image ---")
    query2 = "What colors are most prominent in this image? Are there any people in it?"
    print(f"User: {query2} (with image)")
    
    response2 = await agent.run(
        query2,
        images=[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        }]
    )
    
    print(f"Agent: {response2}")
    
    # Example 3: Compare two images
    print("\n--- Image Comparison ---")
    
    # Read and encode a second image
    with open("example_image2.jpg", "rb") as image_file:
        image_data2 = base64.b64encode(image_file.read()).decode("utf-8")
    
    query3 = "Compare these two images. What are the main similarities and differences?"
    print(f"User: {query3} (with two images)")
    
    response3 = await agent.run(
        query3,
        images=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data2
                }
            }
        ]
    )
    
    print(f"Agent: {response3}")

if __name__ == "__main__":
    asyncio.run(image_analysis_agent())
```

### Web Search Integration

Now let's add web search capabilities to get current information:

```python
import asyncio
from agent_framework import ChatAgent, HostedWebSearchTool
from agent_framework.openai import OpenAIChatClient

async def web_search_agent():
    """Example of an agent that can search the web for current information."""
    print("=== Web Search Agent Example ===")
    
    # Create the web search agent
    agent = ChatAgent(
        chat_client=OpenAIChatClient(model_id="gpt-4o-search-preview"),
        instructions="You are a helpful assistant that can search the web for current information. "
                   "Always cite your sources when providing information from the web. "
                   "If you can't find relevant information, say so clearly.",
    )
    
    # Example 1: Current events query
    print("\n--- Current Events Query ---")
    query1 = "What are the latest developments in artificial intelligence this week?"
    print(f"User: {query1}")
    
    response1 = await agent.run(
        query1,
        tools=[HostedWebSearchTool()],
        tool_choice="auto"
    )
    
    print(f"Agent: {response1}")
    
    # Example 2: Location-specific query
    print("\n--- Location-Specific Query ---")
    query2 = "What's the weather forecast for New York City for the next 3 days?"
    print(f"User: {query2}")
    
    # Provide location context for the web search
    additional_properties = {
        "user_location": {
            "country": "US",
            "city": "New York",
        }
    }
    
    response2 = await agent.run(
        query2,
        tools=[HostedWebSearchTool(additional_properties=additional_properties)],
        tool_choice="auto"
    )
    
    print(f"Agent: {response2}")
    
    # Example 3: Product comparison
    print("\n--- Product Comparison ---")
    query3 = "Compare the latest iPhone models based on reviews and specifications."
    print(f"User: {query3}")
    
    response3 = await agent.run(
        query3,
        tools=[HostedWebSearchTool()],
        tool_choice="auto"
    )
    
    print(f"Agent: {response3}")

if __name__ == "__main__":
    asyncio.run(web_search_agent())
```

### Reasoning Capabilities

Let's implement reasoning capabilities to analyze complex problems:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient

async def reasoning_agent():
    """Example of an agent with reasoning capabilities."""
    print("=== Reasoning Agent Example ===")
    
    # Create the reasoning agent
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant with strong reasoning capabilities. "
                   "When answering questions, show your step-by-step thinking process. "
                   "Consider multiple perspectives and evaluate different options before reaching a conclusion. "
                   "Acknowledge any assumptions you're making and any limitations in your reasoning.",
    )
    
    # Example 1: Logical reasoning
    print("\n--- Logical Reasoning ---")
    query1 = "If all cats are animals, and some animals are pets, can we conclude that some cats are pets? Explain your reasoning."
    print(f"User: {query1}")
    
    response1 = await agent.run(query1)
    print(f"Agent: {response1}")
    
    # Example 2: Ethical reasoning
    print("\n--- Ethical Reasoning ---")
    query2 = "A self-driving car is about to hit five pedestrians. It can swerve to avoid them, but would hit one person on the sidewalk. What should it do? Analyze this ethical dilemma from multiple perspectives."
    print(f"User: {query2}")
    
    response2 = await agent.run(query2)
    print(f"Agent: {response2}")
    
    # Example 3: Mathematical reasoning
    print("\n--- Mathematical Reasoning ---")
    query3 = "If a train travels 120 miles in 2 hours, and another train travels 180 miles in 3 hours, which train is faster and by how much? Show your calculations."
    print(f"User: {query3}")
    
    response3 = await agent.run(query3)
    print(f"Agent: {response3}")
    
    # Example 4: Creative reasoning
    print("\n--- Creative Reasoning ---")
    query4 = "If you could design a new public transportation system for a city of 1 million people, what factors would you consider and what would your system look like?"
    print(f"User: {query4}")
    
    response4 = await agent.run(query4)
    print(f"Agent: {response4}")

if __name__ == "__main__":
    asyncio.run(reasoning_agent())
```

### Combined Multi-Capability Example

Now let's combine all these capabilities into a single multi-modal agent:

```python
import asyncio
import base64
from agent_framework import ChatAgent, HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def multi_modal_agent():
    """Example of a multi-modal agent that combines vision, web search, and reasoning."""
    print("=== Multi-Modal Agent Example ===")
    
    # Create the multi-modal agent
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(model_id="gpt-4o"),
        instructions="You are a helpful multi-modal assistant with vision, web search, and reasoning capabilities. "
                   "When analyzing images, provide detailed descriptions. "
                   "When searching the web, always cite your sources. "
                   "When reasoning, show your step-by-step thinking process. "
                   "For complex queries, use all your capabilities to provide comprehensive answers.",
    )
    
    # Example 1: Visual analysis with web search
    print("\n--- Visual Analysis with Web Search ---")
    
    # Read and encode an image
    with open("landmark.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    query1 = "Identify this landmark and provide current information about visiting hours and ticket prices."
    print(f"User: {query1} (with image)")
    
    response1 = await agent.run(
        query1,
        images=[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        }],
        tools=[HostedWebSearchTool()],
        tool_choice="auto"
    )
    
    print(f"Agent: {response1}")
    
    # Example 2: Complex reasoning with web search
    print("\n--- Complex Reasoning with Web Search ---")
    query2 = "Analyze the current state of renewable energy adoption in three different countries. Consider economic, environmental, and political factors. Provide a reasoned recommendation for which approach is most promising."
    print(f"User: {query2}")
    
    response2 = await agent.run(
        query2,
        tools=[HostedWebSearchTool()],
        tool_choice="auto"
    )
    
    print(f"Agent: {response2}")
    
    # Example 3: Visual analysis with reasoning
    print("\n--- Visual Analysis with Reasoning ---")
    
    # Read and encode a second image
    with open("room_design.jpg", "rb") as image_file:
        image_data2 = base64.b64encode(image_file.read()).decode("utf-8")
    
    query3 = "Analyze this room design and suggest three improvements based on principles of interior design. Explain your reasoning for each suggestion."
    print(f"User: {query3} (with image)")
    
    response3 = await agent.run(
        query3,
        images=[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data2
            }
        }]
    )
    
    print(f"Agent: {response3}")
    
    # Example 4: Multi-modal reasoning with web search
    print("\n--- Multi-Modal Reasoning with Web Search ---")
    query4 = "Based on current market trends, analyze whether this product would be successful. Consider factors like target audience, pricing, and competition."
    print(f"User: {query4} (with image)")
    
    response4 = await agent.run(
        query4,
        images=[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data2
            }
        }],
        tools=[HostedWebSearchTool()],
        tool_choice="auto"
    )
    
    print(f"Agent: {response4}")

if __name__ == "__main__":
    asyncio.run(multi_modal_agent())
```

### Best Practices for Complex Agents

1. **Provide Clear Instructions**:
   ```python
   # Good: Detailed instructions
   instructions = "You are a helpful multi-modal assistant with vision, web search, and reasoning capabilities. "
                  "When analyzing images, provide detailed descriptions. "
                  "When searching the web, always cite your sources. "
                  "When reasoning, show your step-by-step thinking process. "
                  "For complex queries, use all your capabilities to provide comprehensive answers."
   
   # Avoid: Vague instructions
   instructions = "You are a helpful assistant."
   ```

2. **Choose the Right Model**:
   ```python
   # For vision capabilities
   agent = ChatAgent(
       chat_client=OpenAIResponsesClient(model_id="gpt-4o"),  # Vision-capable model
       instructions=instructions,
   )
   
   # For web search
   agent = ChatAgent(
       chat_client=OpenAIChatClient(model_id="gpt-4o-search-preview"),  # Search-capable model
       instructions=instructions,
   )
   ```

3. **Handle Different Input Types Appropriately**:
   ```python
   # For image analysis
   response = await agent.run(
       query,
       images=[{
           "type": "image",
           "source": {
               "type": "base64",
               "media_type": "image/jpeg",
               "data": image_data
           }
       }]
   )
   
   # For web search
   response = await agent.run(
       query,
       tools=[HostedWebSearchTool()],
       tool_choice="auto"
   )
   ```

### Performance Considerations

1. **Model Selection**: Choose the appropriate model based on the task requirements. Vision models may be slower and more expensive than text-only models.

2. **Image Size**: Large images may take longer to process. Consider resizing images before sending them to the agent.

3. **Web Search Limits**: Be aware of rate limits for web search queries. Implement caching if you're frequently searching for the same information.

4. **Tool Usage**: Only use the tools necessary for the task. For example, don't include web search if you're only analyzing an image.

---

## Agent Type #5: MCP-Integrated Agent

### Use Case & Architecture

Model Context Protocol (MCP) integrated agents can connect to external systems and services, extending their capabilities beyond what's built into the framework. These agents are ideal for enterprise integrations, accessing specialized tools, and connecting to custom APIs. They can work with both local MCP servers (running on the same machine) and hosted MCP servers (running remotely).

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                   MCP-Integrated Agent                      │
├─────────────────────────────────────────────────────────────┤
│  Components:                                               │
│  ├── ChatAgent (Core abstraction)                          │
│  ├── OpenAIChatClient (For MCP integration)                │
│  ├── MCPStreamableHTTPTool (For MCP connections)           │
│  ├── Approval Workflows (For hosted MCP)                   │
│  └── External Systems (Databases, APIs, etc.)              │
├─────────────────────────────────────────────────────────────┤
│  Capabilities:                                             │
│  ├── Integration with external systems                     │
│  ├── Access to specialized tools and services              │
│  ├── Approval workflows for sensitive operations            │
│  ├── Enterprise system connectivity                        │
│  └── Custom tool integration                               │
└─────────────────────────────────────────────────────────────┘
```

### Local MCP Implementation

Let's implement an agent that connects to a local MCP server:

```python
import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

async def local_mcp_agent():
    """Example of an agent connected to a local MCP server."""
    print("=== Local MCP Agent Example ===")
    
    # Create the MCP tool for connecting to a local server
    mcp_tool = MCPStreamableHTTPTool(
        name="Local Database MCP",
        url="http://localhost:8080/mcp",  # Local MCP server URL
    )
    
    # Create the agent with the MCP tool
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="DatabaseAgent",
        instructions="You are a helpful assistant that can access a local database through an MCP server. "
                   "When retrieving data, present it in a clear, organized way. "
                   "If you encounter any errors, explain them clearly to the user.",
        tools=mcp_tool,
    ) as agent:
        
        # Example 1: Query a database
        print("\n--- Database Query Example ---")
        query1 = "Show me the top 5 customers by total purchase amount."
        print(f"User: {query1}")
        
        response1 = await agent.run(query1)
        print(f"Agent: {response1}")
        
        # Example 2: Data analysis
        print("\n--- Data Analysis Example ---")
        query2 = "Analyze the sales data for the last quarter and identify any trends."
        print(f"User: {query2}")
        
        response2 = await agent.run(query2)
        print(f"Agent: {response2}")
        
        # Example 3: Complex query with joins
        print("\n--- Complex Query Example ---")
        query3 = "Find all customers who purchased product X in the last month and haven't been contacted in the last 30 days."
        print(f"User: {query3}")
        
        response3 = await agent.run(query3)
        print(f"Agent: {response3}")

if __name__ == "__main__":
    asyncio.run(local_mcp_agent())
```

### Hosted MCP with Approval Workflow

For hosted MCP servers, we need to implement approval workflows for security:

```python
import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

async def hosted_mcp_agent():
    """Example of an agent connected to a hosted MCP server with approval workflow."""
    print("=== Hosted MCP Agent Example ===")
    
    # Create the MCP tool for connecting to a hosted server
    mcp_tool = MCPStreamableHTTPTool(
        name="Enterprise CRM MCP",
        url="https://crm.example.com/api/mcp",  # Hosted MCP server URL
        headers={
            "Authorization": "Bearer your-api-token",
            "X-Tenant-ID": "your-tenant-id"
        }
    )
    
    # Create the agent with the MCP tool
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="CRMAgent",
        instructions="You are a helpful assistant that can access a hosted CRM system through an MCP server. "
                   "When accessing customer data, be mindful of privacy and security. "
                   "For any operations that modify data, explain what you're doing and why before proceeding.",
        tools=mcp_tool,
    ) as agent:
        
        # Example 1: Read-only operation
        print("\n--- Read-Only Operation Example ---")
        query1 = "Find customer John Doe's contact information and recent purchase history."
        print(f"User: {query1}")
        
        response1 = await agent.run(query1)
        print(f"Agent: {response1}")
        
        # Example 2: Operation requiring approval
        print("\n--- Operation Requiring Approval Example ---")
        query2 = "Update John Doe's phone number to +1-555-123-4567."
        print(f"User: {query2}")
        
        # This would typically trigger an approval workflow
        print("Agent: I need to update John Doe's phone number. This operation requires approval.")
        print("Agent: Sending approval request...")
        
        # Simulate approval process
        approved = input("Approve this operation? (y/n): ").lower() == 'y'
        
        if approved:
            response2 = await agent.run(query2)
            print(f"Agent: {response2}")
        else:
            print("Agent: Operation not approved. No changes were made.")
        
        # Example 3: Complex operation with multiple steps
        print("\n--- Complex Operation Example ---")
        query3 = "Create a new customer account for Jane Smith with email jane.smith@example.com, then add her to the VIP customer group."
        print(f"User: {query3}")
        
        print("Agent: I need to perform two operations: 1) Create a new customer account, and 2) Add the customer to the VIP group. Both operations require approval.")
        print("Agent: Sending approval requests...")
        
        # Simulate approval process for each step
        step1_approved = input("Approve creating customer account? (y/n): ").lower() == 'y'
        step2_approved = input("Approve adding to VIP group? (y/n): ").lower() == 'y'
        
        if step1_approved and step2_approved:
            response3 = await agent.run(query3)
            print(f"Agent: {response3}")
        else:
            print("Agent: One or more operations were not approved. No changes were made.")

if __name__ == "__main__":
    asyncio.run(hosted_mcp_agent())
```

### Approval Handling Patterns

There are several approaches to handling approval workflows:

#### 1. Manual Approval (Interactive)

```python
async def manual_approval_workflow():
    """Example of a manual approval workflow where the user approves each operation."""
    print("=== Manual Approval Workflow Example ===")
    
    mcp_tool = MCPStreamableHTTPTool(
        name="Secure Database MCP",
        url="https://secure-db.example.com/api/mcp",
        headers={"Authorization": "Bearer your-api-token"}
    )
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="SecureDBAgent",
        instructions="You are a helpful assistant that can access a secure database. "
                   "For any operations that modify data, explain what you're doing and wait for approval before proceeding.",
        tools=mcp_tool,
    ) as agent:
        
        query = "Delete all records older than 5 years from the audit log."
        print(f"User: {query}")
        
        print("Agent: I need to delete records older than 5 years from the audit log. This is a destructive operation.")
        print("Agent: The following records will be deleted:")
        
        # First, show what will be affected
        preview_query = "Show me a sample of records that will be deleted"
        preview_response = await agent.run(preview_query)
        print(f"Agent: {preview_response}")
        
        # Request approval
        approved = input("Do you approve this operation? (y/n): ").lower() == 'y'
        
        if approved:
            response = await agent.run(query)
            print(f"Agent: {response}")
        else:
            print("Agent: Operation not approved. No changes were made.")
```

#### 2. Automatic Approval (Pre-approved Operations)

```python
async def automatic_approval_workflow():
    """Example of an automatic approval workflow for pre-approved operations."""
    print("=== Automatic Approval Workflow Example ===")
    
    mcp_tool = MCPStreamableHTTPTool(
        name="Analytics MCP",
        url="https://analytics.example.com/api/mcp",
        headers={"Authorization": "Bearer your-api-token"}
    )
    
    # Define pre-approved operations
    pre_approved_operations = [
        "SELECT",  # Read operations
        "SHOW",    # Metadata operations
        "DESCRIBE" # Schema operations
    ]
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="AnalyticsAgent",
        instructions="You are a helpful assistant that can access an analytics database. "
                   "Read operations are automatically approved. "
                   "For any operations that modify data, explain what you're doing and wait for approval before proceeding.",
        tools=mcp_tool,
    ) as agent:
        
        # Read operation (automatically approved)
        query1 = "SELECT COUNT(*) FROM users WHERE signup_date > '2023-01-01'"
        print(f"User: {query1}")
        
        response1 = await agent.run(query1)
        print(f"Agent: {response1}")
        
        # Write operation (requires approval)
        query2 = "UPDATE users SET status = 'active' WHERE last_login > '2023-06-01'"
        print(f"User: {query2}")
        
        print("Agent: I need to update user statuses. This operation modifies data and requires approval.")
        approved = input("Do you approve this operation? (y/n): ").lower() == 'y'
        
        if approved:
            response2 = await agent.run(query2)
            print(f"Agent: {response2}")
        else:
            print("Agent: Operation not approved. No changes were made.")
```

#### 3. Role-Based Approval (Different Approval Requirements Based on User Role)

```python
async def role_based_approval_workflow():
    """Example of a role-based approval workflow."""
    print("=== Role-Based Approval Workflow Example ===")
    
    mcp_tool = MCPStreamableHTTPTool(
        name="HR System MCP",
        url="https://hr.example.com/api/mcp",
        headers={"Authorization": "Bearer your-api-token"}
    )
    
    # Define user role and permissions
    user_role = "manager"  # Could be "employee", "manager", "admin", etc.
    
    # Define approval requirements based on role
    approval_requirements = {
        "employee": ["SELECT"],  # Employees can only read data
        "manager": ["SELECT", "UPDATE"],  # Managers can read and update
        "admin": ["SELECT", "UPDATE", "INSERT", "DELETE"]  # Admins can do everything
    }
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="HRAgent",
        instructions=f"You are a helpful assistant that can access the HR system. "
                   f"Your current role is {user_role}. "
                   f"You can perform the following operations without approval: {', '.join(approval_requirements[user_role])}. "
                   f"For any other operations, explain what you're doing and wait for approval before proceeding.",
        tools=mcp_tool,
    ) as agent:
        
        # Operation within role permissions
        query1 = "SELECT * FROM employees WHERE department = 'Sales'"
        print(f"User: {query1}")
        
        response1 = await agent.run(query1)
        print(f"Agent: {response1}")
        
        # Operation requiring approval (based on role)
        if user_role == "manager":
            query2 = "UPDATE employees SET salary = salary * 1.05 WHERE performance_rating > 8"
            print(f"User: {query2}")
            
            print("Agent: I need to update employee salaries. This operation is within your role permissions as a manager.")
            
            response2 = await agent.run(query2)
            print(f"Agent: {response2}")
        elif user_role == "employee":
            query2 = "UPDATE employees SET salary = salary * 1.05 WHERE performance_rating > 8"
            print(f"User: {query2}")
            
            print("Agent: I need to update employee salaries. This operation requires admin privileges and is not within your role permissions.")
            print("Agent: Please contact an administrator to perform this operation.")

if __name__ == "__main__":
    asyncio.run(role_based_approval_workflow())
```

### Security Considerations

1. **Authentication and Authorization**:
   ```python
   # Always include proper authentication headers
   mcp_tool = MCPStreamableHTTPTool(
       name="Secure MCP",
       url="https://secure.example.com/api/mcp",
       headers={
           "Authorization": "Bearer your-api-token",
           "X-API-Key": "your-api-key"
       }
   )
   ```

2. **Input Validation**:
   ```python
   # Validate inputs before sending to MCP
   def validate_customer_id(customer_id):
       if not customer_id or not customer_id.isalnum():
           raise ValueError("Invalid customer ID")
       return customer_id
   
   # Usage
   customer_id = validate_customer_id(user_input)
   query = f"SELECT * FROM customers WHERE id = '{customer_id}'"
   ```

3. **Error Handling**:
   ```python
   try:
       response = await agent.run(query)
       print(f"Agent: {response}")
   except Exception as e:
       print(f"Error: {e}")
       # Implement appropriate error handling
   ```

### Best Practices for MCP Integration

1. **Use Connection Pooling**: For high-traffic applications, consider implementing connection pooling to improve performance.

2. **Implement Caching**: Cache frequently accessed data to reduce the number of MCP calls.

3. **Monitor Usage**: Track MCP usage to identify potential issues or optimization opportunities.

4. **Secure Sensitive Data**: Ensure sensitive data is properly encrypted both in transit and at rest.

5. **Implement Rate Limiting**: Prevent abuse by implementing rate limiting for MCP operations.

---

## Advanced Topics

### Thread Persistence Strategies

Thread persistence is crucial for maintaining conversation context across multiple interactions. Let's explore three different approaches:

#### 1. Service-Managed Thread Persistence

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIAssistantsClient

async def service_managed_thread_persistence():
    """Example of service-managed thread persistence using OpenAI Assistants."""
    print("=== Service-Managed Thread Persistence Example ===")
    
    # Create an agent with OpenAI Assistants
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that remembers previous conversations.",
    ) as agent:
        
        # Create a new thread
        thread = agent.get_new_thread()
        
        # First conversation
        query1 = "My name is Alice and I'm planning a trip to Japan. Can you recommend some places to visit?"
        print(f"User: {query1}")
        response1 = await agent.run(query1, thread=thread)
        print(f"Agent: {response1.text}")
        
        # The thread ID is now set and can be retrieved
        thread_id = thread.service_thread_id
        print(f"Thread ID: {thread_id}")
        
        # Second conversation (agent should remember the user's name and trip plans)
        query2 = "What's the weather like in Japan in May?"
        print(f"User: {query2}")
        response2 = await agent.run(query2, thread=thread)
        print(f"Agent: {response2.text}")
        
        # Later, we can resume the conversation using the thread ID
        print("\n--- Resuming Conversation Later ---")
        
        # Create a new agent instance with the existing thread ID
        async with ChatAgent(
            chat_client=OpenAIAssistantsClient(thread_id=thread_id),
            instructions="You are a helpful assistant that remembers previous conversations.",
        ) as new_agent:
            
            # Create a thread with the existing ID
            resumed_thread = AgentThread(service_thread_id=thread_id)
            
            # The agent should still remember the previous context
            query3 = "Based on my previous questions, what would you recommend I pack for my trip?"
            print(f"User: {query3}")
            response3 = await new_agent.run(query3, thread=resumed_thread)
            print(f"Agent: {response3.text}")

if __name__ == "__main__":
    asyncio.run(service_managed_thread_persistence())
```

#### 2. Client-Side Thread Persistence

```python
import asyncio
import json
from agent_framework import AgentThread, ChatAgent, ChatMessageStore
from agent_framework.openai import OpenAIChatClient

async def client_side_thread_persistence():
    """Example of client-side thread persistence using message stores."""
    print("=== Client-Side Thread Persistence Example ===")
    
    # Create an agent with OpenAI Chat Client
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that remembers previous conversations.",
    )
    
    # Create a new thread with a message store
    thread = agent.get_new_thread()
    
    # First conversation
    query1 = "I'm interested in learning about renewable energy. Can you explain the basics?"
    print(f"User: {query1}")
    response1 = await agent.run(query1, thread=thread)
    print(f"Agent: {response1.text}")
    
    # Second conversation
    query2 = "What are the advantages of solar power compared to wind power?"
    print(f"User: {query2}")
    response2 = await agent.run(query2, thread=thread)
    print(f"Agent: {response2.text}")
    
    # Save the conversation history to a file
    if thread.message_store:
        messages = await thread.message_store.list_messages()
        with open("conversation_history.json", "w") as f:
            # Convert messages to a serializable format
            serializable_messages = []
            for message in messages or []:
                serializable_messages.append({
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat() if hasattr(message, 'timestamp') else None
                })
            json.dump(serializable_messages, f, indent=2)
        print("Conversation history saved to conversation_history.json")
    
    # Later, we can resume the conversation by loading the history
    print("\n--- Resuming Conversation Later ---")
    
    # Load the conversation history
    with open("conversation_history.json", "r") as f:
        loaded_messages = json.load(f)
    
    # Create a new message store with the loaded messages
    message_store = ChatMessageStore(loaded_messages)
    
    # Create a new thread with the loaded message store
    resumed_thread = AgentThread(message_store=message_store)
    
    # Create a new agent instance
    new_agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that remembers previous conversations.",
    )
    
    # The agent should still remember the previous context
    query3 = "Based on our previous discussion, which renewable energy source would be best for a small home in a cloudy region?"
    print(f"User: {query3}")
    response3 = await new_agent.run(query3, thread=resumed_thread)
    print(f"Agent: {response3.text}")

if __name__ == "__main__":
    asyncio.run(client_side_thread_persistence())
```

#### 3. Hybrid Thread Persistence

```python
import asyncio
import json
from agent_framework import AgentThread, ChatAgent, ChatMessageStore
from agent_framework.openai import OpenAIAssistantsClient

async def hybrid_thread_persistence():
    """Example of hybrid thread persistence combining service and client-side storage."""
    print("=== Hybrid Thread Persistence Example ===")
    
    # Create an agent with OpenAI Assistants
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that remembers previous conversations.",
    ) as agent:
        
        # Create a new thread
        thread = agent.get_new_thread()
        
        # First conversation
        query1 = "I'm starting a new business selling handmade crafts online. Can you give me some advice?"
        print(f"User: {query1}")
        response1 = await agent.run(query1, thread=thread)
        print(f"Agent: {response1.text}")
        
        # Save both the service thread ID and the conversation history
        thread_id = thread.service_thread_id
        
        if thread.message_store:
            messages = await thread.message_store.list_messages()
            with open("business_advice_history.json", "w") as f:
                # Convert messages to a serializable format
                serializable_messages = []
                for message in messages or []:
                    serializable_messages.append({
                        "role": message.role,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat() if hasattr(message, 'timestamp') else None
                    })
                json.dump({
                    "thread_id": thread_id,
                    "messages": serializable_messages
                }, f, indent=2)
            print("Conversation history and thread ID saved to business_advice_history.json")
    
    # Later, we can try to resume the conversation using the thread ID first
    print("\n--- Resuming Conversation Later ---")
    
    # Load the saved conversation data
    with open("business_advice_history.json", "r") as f:
        saved_data = json.load(f)
    
    saved_thread_id = saved_data["thread_id"]
    saved_messages = saved_data["messages"]
    
    try:
        # Try to resume using the service thread ID
        async with ChatAgent(
            chat_client=OpenAIAssistantsClient(thread_id=saved_thread_id),
            instructions="You are a helpful assistant that remembers previous conversations.",
        ) as new_agent:
            
            # Create a thread with the existing ID
            resumed_thread = AgentThread(service_thread_id=saved_thread_id)
            
            # The agent should still remember the previous context
            query3 = "Based on my previous questions, what's the best platform for selling handmade crafts?"
            print(f"User: {query3}")
            response3 = await new_agent.run(query3, thread=resumed_thread)
            print(f"Agent: {response3.text}")
            
    except Exception as e:
        print(f"Failed to resume using service thread ID: {e}")
        print("Falling back to client-side message store...")
        
        # Fall back to client-side message store
        message_store = ChatMessageStore(saved_messages)
        fallback_thread = AgentThread(message_store=message_store)
        
        # Create a new agent instance
        fallback_agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            instructions="You are a helpful assistant that remembers previous conversations.",
        )
        
        # The agent should still remember the previous context from the message store
        query3 = "Based on my previous questions, what's the best platform for selling handmade crafts?"
        print(f"User: {query3}")
        response3 = await fallback_agent.run(query3, thread=fallback_thread)
        print(f"Agent: {response3.text}")

if __name__ == "__main__":
    asyncio.run(hybrid_thread_persistence())
```

### Custom Message Stores

For more advanced use cases, you can implement custom message stores:

```python
import asyncio
from typing import List, Optional
from agent_framework import ChatMessage, ChatMessageStore, ChatAgent
from agent_framework.openai import OpenAIChatClient

class DatabaseMessageStore(ChatMessageStore):
    """Custom message store that saves messages to a database."""
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        # Initialize database connection
        # This is a simplified example
        self.messages = []
    
    async def add_message(self, message: ChatMessage) -> None:
        """Add a message to the database."""
        # In a real implementation, this would save to a database
        self.messages.append(message)
        print(f"Saved message to database: {message.role}: {message.content[:50]}...")
    
    async def list_messages(self) -> Optional[List[ChatMessage]]:
        """Retrieve all messages from the database."""
        # In a real implementation, this would retrieve from a database
        return self.messages
    
    async def clear_messages(self) -> None:
        """Clear all messages from the database."""
        # In a real implementation, this would clear the database
        self.messages = []
        print("Cleared all messages from database")

async def custom_message_store_example():
    """Example using a custom message store."""
    print("=== Custom Message Store Example ===")
    
    # Create a custom message store
    message_store = DatabaseMessageStore("sqlite:///conversation_history.db")
    
    # Create an agent with the custom message store
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant that remembers previous conversations.",
    )
    
    # Create a thread with the custom message store
    thread = AgentThread(message_store=message_store)
    
    # First conversation
    query1 = "I'm learning to cook. Can you recommend some easy recipes for beginners?"
    print(f"User: {query1}")
    response1 = await agent.run(query1, thread=thread)
    print(f"Agent: {response1.text}")
    
    # Second conversation
    query2 = "What ingredients do I need for the first recipe you mentioned?"
    print(f"User: {query2}")
    response2 = await agent.run(query2, thread=thread)
    print(f"Agent: {response2.text}")
    
    # Retrieve messages from the custom store
    messages = await message_store.list_messages()
    print(f"Retrieved {len(messages or [])} messages from custom store")

if __name__ == "__main__":
    asyncio.run(custom_message_store_example())
```

### Performance Optimization

1. **Use Streaming for Long Responses**:
   ```python
   # For long responses, use streaming to improve perceived performance
   print("Agent: ", end="", flush=True)
   async for chunk in agent.run_stream("Explain quantum computing in detail."):
       if chunk.text:
           print(chunk.text, end="", flush=True)
   print()
   ```

2. **Cache Frequently Used Data**:
   ```python
   # Cache responses to frequently asked questions
   response_cache = {}
   
   async def cached_agent_run(agent, query, thread=None):
       cache_key = f"{query}_{thread.id if thread else 'no_thread'}"
       
       if cache_key in response_cache:
           print("Agent: [Cached response] " + response_cache[cache_key])
           return response_cache[cache_key]
       
       response = await agent.run(query, thread=thread)
       response_cache[cache_key] = response
       return response
   ```

3. **Optimize Tool Usage**:
   ```python
   # Only use tools when necessary
   if "weather" in query.lower():
       response = await agent.run(query, tools=[get_weather])
   else:
       response = await agent.run(query)
   ```

### Testing Strategies

Testing is crucial for ensuring your agents work as expected:

```python
import asyncio
import pytest
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

class TestChatAgent:
    """Test suite for a chat agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return ChatAgent(
            chat_client=OpenAIChatClient(),
            instructions="You are a helpful assistant for testing.",
        )
    
    @pytest.mark.asyncio
    async def test_simple_query(self, agent):
        """Test a simple query."""
        response = await agent.run("What is 2 + 2?")
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_streaming_query(self, agent):
        """Test a streaming query."""
        response_chunks = []
        async for chunk in agent.run_stream("Tell me a joke."):
            if chunk.text:
                response_chunks.append(chunk.text)
        
        response = "".join(response_chunks)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_thread_persistence(self, agent):
        """Test thread persistence."""
        thread = agent.get_new_thread()
        
        # First query
        response1 = await agent.run("My name is Alice.", thread=thread)
        
        # Second query (should remember the name)
        response2 = await agent.run("What's my name?", thread=thread)
        assert "Alice" in response2

# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
```

---

## Best Practices & Production Considerations

### Resource Management

Proper resource management is crucial for production applications:

```python
import asyncio
from contextlib import asynccontextmanager
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

@asynccontextmanager
async def managed_agent():
    """Context manager for proper agent lifecycle management."""
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )
    
    try:
        yield agent
    finally:
        # Perform any necessary cleanup
        pass

async def production_agent_example():
    """Example of proper resource management in production."""
    async with managed_agent() as agent:
        response = await agent.run("Hello!")
        print(response)
    # Agent is automatically cleaned up

if __name__ == "__main__":
    asyncio.run(production_agent_example())
```

### Error Handling and Retry Logic

Implement robust error handling and retry logic:

```python
import asyncio
import random
from typing import Callable, Any

async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (exponential_base ** attempt) + random.uniform(0, 1), max_delay)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

async def robust_agent_example():
    """Example of an agent with robust error handling."""
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
    
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )
    
    try:
        response = await retry_with_backoff(
            agent.run,
            max_retries=3,
            query="What's the weather like today?"
        )
        print(response)
    except Exception as e:
        print(f"Failed after multiple retries: {e}")
        # Implement fallback behavior
        print("I'm sorry, I'm having trouble accessing that information right now.")

if __name__ == "__main__":
    asyncio.run(robust_agent_example())
```

### Logging and Debugging

Implement comprehensive logging and debugging:

```python
import asyncio
import logging
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def logging_agent_example():
    """Example of an agent with comprehensive logging."""
    logger.info("Creating agent")
    
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )
    
    try:
        query = "What's the weather like today?"
        logger.info(f"Sending query: {query}")
        
        response = await agent.run(query)
        logger.info(f"Received response: {response[:100]}...")
        
        print(response)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(logging_agent_example())
```

### Production Deployment Checklist

- [ ] **Environment Variables**: Ensure all required environment variables are set
- [ ] **Error Handling**: Implement comprehensive error handling and retry logic
- [ ] **Logging**: Set up detailed logging for monitoring and debugging
- [ ] **Resource Management**: Use context managers for proper resource cleanup
- [ ] **Rate Limiting**: Implement rate limiting to avoid API quota issues
- [ ] **Security**: Secure API keys and sensitive information
- [ ] **Testing**: Implement comprehensive tests for all functionality
- [ ] **Monitoring**: Set up monitoring for performance and error rates
- [ ] **Scaling**: Design for horizontal scaling if needed
- [ ] **Backup**: Implement backup and recovery procedures

### Security Best Practices

1. **Secure API Keys**:
   ```python
   import os
   from agent_framework.openai import OpenAIChatClient
   
   # Use environment variables for API keys
   api_key = os.environ.get("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("OPENAI_API_KEY environment variable not set")
   
   client = OpenAIChatClient(api_key=api_key)
   ```

2. **Input Validation**:
   ```python
   def validate_query(query: str) -> str:
       """Validate user input to prevent injection attacks."""
       if len(query) > 1000:
           raise ValueError("Query too long")
       
       # Remove any potentially harmful content
       # This is a simplified example
       sanitized_query = query.replace("<script>", "").replace("</script>", "")
       
       return sanitized_query
   ```

3. **Output Sanitization**:
   ```python
   def sanitize_output(output: str) -> str:
       """Sanitize agent output to prevent XSS attacks."""
       # This is a simplified example
       sanitized_output = output.replace("<", "&lt;").replace(">", "&gt;")
       return sanitized_output
   ```

### Cost Optimization

1. **Choose the Right Model**:
   ```python
   # Use smaller models for simple tasks
   simple_agent = ChatAgent(
       chat_client=OpenAIChatClient(model_id="gpt-3.5-turbo"),
       instructions="You are a helpful assistant for simple tasks.",
   )
   
   # Use larger models for complex tasks
   complex_agent = ChatAgent(
       chat_client=OpenAIChatClient(model_id="gpt-4o"),
       instructions="You are a helpful assistant for complex tasks.",
   )
   ```

2. **Implement Caching**:
   ```python
   # Cache responses to frequently asked questions
   response_cache = {}
   
   async def cached_agent_run(agent, query, thread=None):
       cache_key = f"{query}_{thread.id if thread else 'no_thread'}"
       
       if cache_key in response_cache:
           return response_cache[cache_key]
       
       response = await agent.run(query, thread=thread)
       response_cache[cache_key] = response
       return response
   ```

3. **Use Streaming for Long Responses**:
   ```python
   # Streaming can be more cost-effective for long responses
   print("Agent: ", end="", flush=True)
   async for chunk in agent.run_stream("Explain quantum computing in detail."):
       if chunk.text:
           print(chunk.text, end="", flush=True)
   print()
   ```

---

## Troubleshooting Guide

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `AuthenticationError` | Invalid API key | Check your API key and ensure it's set correctly |
| `RateLimitError` | Too many API requests | Implement rate limiting and retry logic |
| `TimeoutError` | Request took too long | Increase timeout or break down the request |
| `InvalidRequestError` | Invalid request format | Check the request format and parameters |
| `ConnectionError` | Network issues | Check network connection and implement retry logic |

### Debugging Techniques

1. **Enable Verbose Logging**:
   ```python
   import logging
   
   # Enable verbose logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use a Test Agent**:
   ```python
   # Create a simple test agent to isolate issues
   test_agent = ChatAgent(
       chat_client=OpenAIChatClient(),
       instructions="You are a test assistant.",
   )
   
   response = await test_agent.run("Hello!")
   print(response)
   ```

3. **Check API Status**:
   ```python
   import requests
   
   # Check OpenAI API status
   response = requests.get("https://status.openai.com/api/v2/status.json")
   print(response.json())
   ```

### Performance Issues

1. **Slow Response Times**:
   - Check network latency
   - Use streaming for long responses
   - Consider using a faster model

2. **High Memory Usage**:
   - Limit conversation history
   - Implement message pruning
   - Use smaller models when possible

3. **API Rate Limiting**:
   - Implement exponential backoff
   - Use caching for frequently requested data
   - Consider upgrading your API plan

---

## Quick Reference & Next Steps

### Client Type Selection Flowchart

```
Need to process images or structured output?
├─ Yes → Use OpenAIResponsesClient with gpt-4o
└─ No
   Need complex multi-turn conversations with file search?
   ├─ Yes → Use OpenAIAssistantsClient
   └─ No
      Need simple chat interactions?
      ├─ Yes → Use OpenAIChatClient
      └─ No → Consider a different approach
```

### Common Patterns Cheat Sheet

1. **Basic Agent Creation**:
   ```python
   from agent_framework.openai import OpenAIChatClient
   
   agent = OpenAIChatClient().create_agent(
       name="MyAgent",
       instructions="You are a helpful assistant.",
   )
   ```

2. **Adding Function Tools**:
   ```python
   def my_tool(param: str) -> str:
       """Tool description."""
       return f"Result: {param}"
   
   agent = OpenAIChatClient().create_agent(
       instructions="You are a helpful assistant.",
       tools=[my_tool],
   )
   ```

3. **Streaming Responses**:
   ```python
   async for chunk in agent.run_stream("Tell me a story."):
       if chunk.text:
           print(chunk.text, end="", flush=True)
   ```

4. **Thread Management**:
   ```python
   thread = agent.get_new_thread()
   response = await agent.run("Hello!", thread=thread)
   ```

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `OPENAI_CHAT_MODEL_ID` | Model for chat completions | `gpt-4o` |
| `OPENAI_RESPONSES_MODEL_ID` | Model for structured responses | `gpt-4o` |
| `OPENAI_ORG_ID` | Your OpenAI organization ID | `org-...` |
| `OPENAI_API_BASE_URL` | Custom base URL | `https://api.openai.com/v1` |

### Glossary of Terms

- **Agent**: An AI entity that can process inputs and generate responses
- **Client**: Interface to a specific AI service (OpenAI, Azure, etc.)
- **Thread**: A sequence of messages in a conversation
- **Tool**: A function or capability that an agent can use
- **Streaming**: Receiving response chunks as they are generated
- **MCP**: Model Context Protocol, for integrating with external systems

### Additional Resources

- [Microsoft Agent Framework Documentation](https://docs.microsoft.com/azure/ai-services/agent-framework)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai)

### Next Steps

1. **Experiment with Different Agent Types**: Try building each of the 5 agent types described in this guide.

2. **Explore Advanced Features**: Dive deeper into features like custom message stores, advanced tool integration, and performance optimization.

3. **Build a Real Application**: Apply what you've learned to build a real application that solves a specific problem.

4. **Join the Community**: Participate in forums, discussions, and contribute to the framework.

5. **Stay Updated**: Keep up with the latest developments and updates to the framework.

---

# Conclusion

This comprehensive guide has covered the Microsoft Agent Framework in detail, explaining how to build 5 different types of AI agents:

1. **Basic Conversational Agent**: For natural language conversations and simple task execution
2. **RAG Agent**: For knowledge retrieval and document-based question answering
3. **Code Execution Agent**: For dynamic code execution and computational tasks
4. **Multi-Modal Agent**: For processing text, images, and web search with complex reasoning
5. **MCP-Integrated Agent**: For connecting to external systems and enterprise integrations

We've explored core concepts, implementation details, best practices, and production considerations. With this knowledge, you should be well-equipped to build sophisticated AI agents using the Microsoft Agent Framework.

Remember that the framework is designed to be flexible and extensible, so don't hesitate to experiment with different approaches and patterns. As you gain experience, you'll discover new ways to leverage the framework's capabilities to solve complex problems.
