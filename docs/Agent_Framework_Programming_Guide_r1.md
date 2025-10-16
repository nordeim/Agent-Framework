Microsoft Agent Framework: Comprehensive Programming Guide
Build Production-Ready AI Agents in Minutes
Version 1.0 | Target Framework: Microsoft Agent Framework for Python

Table of Contents
Introduction & Getting Started
Core Concepts
Agent Type #1: Basic Conversational Agent
Agent Type #2: RAG Agent
Agent Type #3: Code Execution Agent
Agent Type #4: Multi-Modal Agent
Agent Type #5: MCP-Integrated Agent
Advanced Topics
Best Practices & Production Considerations
Troubleshooting Guide
Quick Reference & Next Steps
1. Introduction & Getting Started
What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, production-ready Python library that simplifies building AI agents with advanced capabilities. Whether you need a simple chatbot, a knowledge-retrieval system, a code-executing assistant, or a multi-modal agent that can analyze images and search the web, this framework provides the abstractions and tools to build it quickly and reliably.

Key Value Propositions:

🚀 Rapid Development: Build sophisticated AI agents in minutes, not days
🔧 Multiple Client Types: Support for OpenAI and Azure OpenAI with Assistants, Chat, and Responses APIs
🧩 Extensible Tool System: Integrate custom functions, hosted tools, and external services via MCP
💾 Thread Management: Built-in conversation state management with flexible persistence options
🔄 Streaming Support: Real-time response streaming for better user experience
🏗️ Production-Ready: Context managers, error handling, and resource cleanup built-in
Framework Architecture
The framework is built on three foundational layers:

text

┌─────────────────────────────────────────────────────────┐
│                    Agent Layer                          │
│  (ChatAgent - High-level abstraction for AI agents)     │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Client Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Assistants  │  │     Chat     │  │  Responses   │  │
│  │    Client    │  │    Client    │  │    Client    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Provider Layer                         │
│         OpenAI          │        Azure OpenAI           │
└─────────────────────────────────────────────────────────┘
Layer Responsibilities:

Agent Layer: Manages lifecycle, threading, tool orchestration, and conversation flow
Client Layer: Handles API communication, response formatting, and tool execution
Provider Layer: Connects to OpenAI or Azure OpenAI services
Client Types Comparison
Choosing the right client type is crucial for your use case:

Client Type	Best For	Key Features	Service-Managed State	Complexity
Assistants	Long-running conversations, complex workflows	Full assistant lifecycle, service-managed threads, tool persistence	✅ Yes	Medium
Chat	Simple chat applications, stateless interactions	Lightweight, fast, local message management	❌ No	Low
Responses	Structured outputs, function-heavy apps, modern API	Structured response format, advanced tools, reasoning support	⚡ Optional	Medium
Decision Guide:

Choose Assistants when you need persistent assistants with service-managed state
Choose Chat for simple, fast, stateless chat applications
Choose Responses for advanced features like structured outputs, reasoning, or when using the latest OpenAI models
Environment Setup
Step 1: Install the Framework
Bash

# Using pip
pip install agent-framework

# Using uv (recommended for faster installs)
uv pip install agent-framework
Step 2: Set Environment Variables
For OpenAI:

Bash

export OPENAI_API_KEY="your-api-key-here"
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"
For Azure OpenAI:

Bash

export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-chat-deployment"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="your-responses-deployment"
Step 3: Verify Installation
Create a file named test_setup.py:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    response = await client.get_response("Hello! Can you confirm you're working?")
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
Run it:

Bash

python test_setup.py
If you see a response from the agent, you're all set! 🎉

Quick Start: Your First Agent in 10 Lines
Let's build a simple weather agent that can answer questions about weather in different cities:

Python

import asyncio
from random import randint
from typing import Annotated
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

# Define a custom function the agent can call
def get_weather(
    location: Annotated[str, Field(description="The location to get weather for")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

async def main():
    # Create an agent with the weather function
    agent = OpenAIChatClient().create_agent(
        instructions="You are a helpful weather assistant.",
        tools=get_weather,
    )
    
    # Ask a question
    result = await agent.run("What's the weather in Paris and London?")
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
What just happened?

We defined a Python function get_weather() using type annotations
We created an agent with instructions and provided our function as a tool
The agent automatically determined it needed to call our function to answer the question
The framework handled the function call, got the results, and generated a natural response
This is the core pattern you'll use throughout the framework! 🚀

2. Core Concepts
Before diving into specific agent types, let's understand the fundamental concepts that power the framework.

Understanding Client Types
Assistants Client
The Assistants Client maps to OpenAI's Assistants API, which provides persistent assistant entities on the server.

When to use:

You need assistants that persist across sessions
You want the service to manage conversation threads
You're building complex, stateful applications
Key characteristics:

Assistants are created and stored on the server
Threads are service-managed with automatic persistence
Ideal for long-running conversations
Requires cleanup (deletion) when done
Example:

Python

from agent_framework.openai import OpenAIAssistantsClient

async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_function,
) as agent:
    # Assistant is automatically created and will be deleted on exit
    result = await agent.run("Hello!")
Chat Client
The Chat Client provides a lightweight interface to OpenAI's Chat Completions API.

When to use:

You need fast, stateless responses
You want to manage message history yourself
You're building simple chat interfaces
Key characteristics:

No persistent assistants or threads on the server
Message history managed locally or in your own storage
Faster and simpler than Assistants
Great for microservices and stateless architectures
Example:

Python

from agent_framework.openai import OpenAIChatClient

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=my_function,
)

result = await agent.run("Hello!")
Responses Client
The Responses Client uses OpenAI's newer Responses API with advanced features.

When to use:

You need structured outputs (Pydantic models)
You want reasoning capabilities (with gpt-5 models)
You need advanced features like image generation or vision
You want optional service-managed state
Key characteristics:

Supports structured response formats
Enables reasoning with detailed thought process
Optional conversation state persistence
Most feature-rich client type
Example:

Python

from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

class WeatherData(BaseModel):
    city: str
    temperature: int
    condition: str

agent = OpenAIResponsesClient().create_agent(
    instructions="You are a weather assistant.",
)

result = await agent.run(
    "What's the weather in Tokyo?",
    response_format=WeatherData,  # Get structured output!
)
Agent Lifecycle Management
Agents in the framework follow a clear lifecycle pattern:

text

Create Agent → Configure Tools → Run Queries → Cleanup Resources
Using Context Managers (Recommended):

Python

async with OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=[tool1, tool2],
) as agent:
    # Agent is ready to use
    result = await agent.run("Query here")
    # Automatic cleanup happens here
Manual Management (Advanced):

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are a helpful assistant.",
)

try:
    result = await agent.run("Query here")
finally:
    # Manual cleanup if needed
    if hasattr(agent, 'cleanup'):
        await agent.cleanup()
Best Practice: Always use context managers (async with) to ensure proper resource cleanup, especially with Assistants clients.

Thread Management Patterns
Threads represent conversation contexts. The framework supports three thread management patterns:

Pattern 1: Automatic Thread Creation (Stateless)
Each run() call creates a new, isolated thread:

Python

agent = OpenAIChatClient().create_agent(instructions="You are helpful.")

# Each call is independent
result1 = await agent.run("What's 2+2?")
result2 = await agent.run("What was my last question?")  # Agent won't remember!
Use when: You need stateless, independent queries.

Pattern 2: Explicit Thread Management (In-Memory State)
Create a thread and reuse it to maintain conversation context:

Python

agent = OpenAIChatClient().create_agent(instructions="You are helpful.")

# Create a thread to maintain context
thread = agent.get_new_thread()

result1 = await agent.run("What's 2+2?", thread=thread)
result2 = await agent.run("What was my last question?", thread=thread)  # Remembers!
Use when: You need conversation context but want local control over message storage.

Pattern 3: Service-Managed Threads (Persistent State)
Use service-managed thread IDs for server-side persistence:

Python

agent = OpenAIResponsesClient().create_agent(instructions="You are helpful.")

thread = agent.get_new_thread()

# Enable service storage with store=True
result1 = await agent.run("What's 2+2?", thread=thread, store=True)

# Get the service thread ID
thread_id = thread.service_thread_id

# Later, in a different session:
thread = AgentThread(service_thread_id=thread_id)
result2 = await agent.run("What was my last question?", thread=thread, store=True)
Use when: You need conversation persistence across sessions or server instances.

Tool Integration Approaches
The framework supports multiple ways to integrate tools:

Approach 1: Agent-Level Tools
Tools defined when creating the agent are available for all queries:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather, get_time],  # Available for all queries
)

await agent.run("What's the weather?")  # Can use get_weather
await agent.run("What time is it?")     # Can use get_time
Approach 2: Run-Level Tools
Tools passed to specific run() calls are only available for that query:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
)

await agent.run("What's the weather?", tools=[get_weather])  # Only this query can use get_weather
await agent.run("What time is it?", tools=[get_time])        # Only this query can use get_time
Approach 3: Mixed Tools
Combine both approaches for base tools + query-specific tools:

Python

agent = OpenAIChatClient().create_agent(
    instructions="You are helpful.",
    tools=[get_weather],  # Always available
)

await agent.run("Weather and time?", tools=[get_time])  # Has both get_weather and get_time
Best Practice: Use agent-level tools for core capabilities and run-level tools for query-specific needs.

Streaming vs Non-Streaming Responses
Non-Streaming (Simple)
Get the complete response at once:

Python

result = await agent.run("Tell me a story")
print(f"Agent: {result}")
Use when:

You need the complete response before processing
You're batch processing or logging
Latency isn't a concern
Streaming (Real-Time)
Get response chunks as they're generated:

Python

print("Agent: ", end="", flush=True)
async for chunk in agent.run_stream("Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()  # New line at the end
Use when:

You want to show real-time progress to users
You're building chat UIs
You want to reduce perceived latency
Performance Tip: Streaming doesn't make responses faster, but it improves perceived performance by showing progress immediately.

Error Handling Patterns
Robust error handling is critical for production agents:

Python

from openai import APIError, RateLimitError
import asyncio

async def robust_agent_call(agent, query, max_retries=3):
    """Call agent with retry logic and error handling."""
    
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            return result
            
        except RateLimitError as e:
            # Exponential backoff for rate limits
            wait_time = 2 ** attempt
            print(f"Rate limited. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        except APIError as e:
            # Log and retry on API errors
            print(f"API error: {e}. Retrying...")
            await asyncio.sleep(1)
            
        except Exception as e:
            # Log unexpected errors and fail
            print(f"Unexpected error: {e}")
            raise
    
    raise Exception(f"Failed after {max_retries} attempts")
Key Error Types:

RateLimitError: API rate limits hit (implement backoff)
APIError: Service errors (retry with caution)
ValidationError: Invalid inputs (fix and retry)
TimeoutError: Request timeout (increase timeout or retry)
3. Agent Type #1: Basic Conversational Agent
Use Case: Customer Support Chatbot
Imagine you're building a customer support chatbot for an e-commerce company. The bot needs to:

Answer questions about order status
Check product availability
Handle returns and refunds
Escalate complex issues to humans
This is the perfect use case for a Basic Conversational Agent with function calling capabilities.

Architecture Overview
A Basic Conversational Agent consists of:

text

User Query → Agent (with Instructions) → LLM Decision → Function Calls → Response
                        ↓
                  Available Tools:
                  - get_order_status()
                  - check_inventory()
                  - process_refund()
                  - escalate_to_human()
The agent receives natural language queries, decides which functions to call based on the query, executes them, and formulates a natural language response.

Complete Implementation
Let's build a complete customer support agent:

Python

import asyncio
from typing import Annotated, Literal
from datetime import datetime
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

# Define business logic functions
def get_order_status(
    order_id: Annotated[str, Field(description="The order ID to check")],
) -> str:
    """Get the current status of an order."""
    # In production, this would query your database
    statuses = {
        "ORD123": "Shipped - Expected delivery: Dec 25",
        "ORD456": "Processing - Will ship within 24 hours",
        "ORD789": "Delivered on Dec 20",
    }
    return statuses.get(order_id, "Order not found")

def check_inventory(
    product_id: Annotated[str, Field(description="The product ID to check")],
) -> str:
    """Check if a product is in stock."""
    # In production, this would query your inventory system
    inventory = {
        "PROD001": "In stock - 45 units available",
        "PROD002": "Low stock - 3 units remaining",
        "PROD003": "Out of stock - Expected restock: Jan 5",
    }
    return inventory.get(product_id, "Product not found")

def process_refund(
    order_id: Annotated[str, Field(description="The order ID to refund")],
    reason: Annotated[str, Field(description="Reason for the refund")],
) -> str:
    """Process a refund for an order."""
    # In production, this would integrate with your payment processor
    return f"Refund initiated for order {order_id}. Reason: {reason}. You'll receive confirmation via email within 24 hours."

def escalate_to_human(
    issue_description: Annotated[str, Field(description="Description of the issue")],
    priority: Annotated[Literal["low", "medium", "high"], Field(description="Priority level")],
) -> str:
    """Escalate a complex issue to a human agent."""
    # In production, this would create a ticket in your support system
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Your issue has been escalated to our support team. Ticket ID: {ticket_id}. Priority: {priority}. A human agent will contact you within 2 hours."

async def main():
    # Create the customer support agent
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="""You are a helpful customer support agent for an e-commerce company.
        
        Your responsibilities:
        - Help customers check order status
        - Check product availability
        - Process refunds when appropriate
        - Escalate complex issues to human agents
        
        Be polite, professional, and empathetic. Always confirm order/product IDs before taking action.""",
        tools=[
            get_order_status,
            check_inventory,
            process_refund,
            escalate_to_human,
        ],
    )
    
    # Simulate a customer conversation
    print("=== Customer Support Chat ===\n")
    
    # Maintain conversation context with a thread
    thread = agent.get_new_thread()
    
    # Query 1: Order status
    query1 = "Hi! Can you check the status of my order ORD123?"
    print(f"Customer: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1}\n")
    
    # Query 2: Product availability
    query2 = "Great! I also want to order product PROD002. Is it in stock?"
    print(f"Customer: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2}\n")
    
    # Query 3: Refund request
    query3 = "Actually, I need to return order ORD456. The product arrived damaged."
    print(f"Customer: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3}\n")

if __name__ == "__main__":
    asyncio.run(main())
Function Tools: Agent-Level vs Run-Level
You have flexibility in how you provide tools to your agent:

Agent-Level Tools (Recommended for Core Features)
Python

# Tools available for ALL queries during agent lifetime
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a customer support agent.",
    tools=[get_order_status, check_inventory, process_refund],
)

# All queries can use all three tools
await agent.run("Check order ORD123")
await agent.run("Is PROD001 in stock?")
await agent.run("Process refund for ORD456")
Run-Level Tools (For Query-Specific Capabilities)
Python

# Agent created without tools
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a customer support agent.",
)

# Provide tools per query
await agent.run("Check order ORD123", tools=[get_order_status])
await agent.run("Is PROD001 in stock?", tools=[check_inventory])
Mixed Approach (Best of Both Worlds)
Python

# Core tools at agent level
agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a customer support agent.",
    tools=[get_order_status, check_inventory],  # Core features
)

# Add specialized tools for specific queries
await agent.run(
    "I need to return this order",
    tools=[process_refund],  # Add refund capability for this query
)
Streaming Responses for Better UX
For customer-facing applications, streaming provides better user experience:

Python

async def chat_with_streaming():
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a customer support agent.",
        tools=[get_order_status, check_inventory],
    )
    
    query = "Can you check the status of order ORD123 and tell me if PROD001 is available?"
    
    print("Customer:", query)
    print("Agent: ", end="", flush=True)
    
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    
    print()  # New line
Best Practices for Conversational Agents
✅ DO:

Write Clear Instructions: Be specific about the agent's role, capabilities, and tone
Use Type Annotations: Properly annotate function parameters for better function calling
Validate Inputs: Add validation logic in your functions before performing actions
Maintain Context: Use threads for multi-turn conversations
Handle Errors Gracefully: Wrap function calls in try-except blocks
Log Function Calls: Track what functions are called for debugging and analytics
❌ DON'T:

Don't Expose Dangerous Functions: Never give agents unrestricted access to destructive operations
Don't Trust User Input Blindly: Always validate and sanitize data from user queries
Don't Forget Rate Limits: Implement backoff strategies for API calls
Don't Ignore Thread Cleanup: Clean up threads when conversations end
Don't Hardcode Secrets: Use environment variables for API keys and credentials
Common Pitfalls
Pitfall 1: Missing Function Descriptions
Python

# ❌ Bad: No description
def get_order_status(order_id: str) -> str:
    return "Order status"

# ✅ Good: Clear docstring and parameter descriptions
def get_order_status(
    order_id: Annotated[str, Field(description="The order ID to check (e.g., ORD123)")],
) -> str:
    """Get the current status and tracking information for an order."""
    return "Order status"
Pitfall 2: Not Maintaining Context
Python

# ❌ Bad: Each query loses context
await agent.run("What's my order status?")
await agent.run("When will it arrive?")  # Agent doesn't know which order!

# ✅ Good: Use threads to maintain context
thread = agent.get_new_thread()
await agent.run("What's my order status for ORD123?", thread=thread)
await agent.run("When will it arrive?", thread=thread)  # Agent remembers!
Pitfall 3: Blocking Function Calls
Python

# ❌ Bad: Synchronous blocking call
def get_order_status(order_id: str) -> str:
    response = requests.get(f"https://api.example.com/orders/{order_id}")  # Blocks!
    return response.json()

# ✅ Good: Async function for I/O operations
async def get_order_status(order_id: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/orders/{order_id}")
        return response.json()
4. Agent Type #2: RAG Agent (Retrieval-Augmented Generation)
Use Case: Enterprise Knowledge Base Q&A
Imagine you're building an internal knowledge assistant for a company with thousands of pages of documentation, policies, and procedures. Employees need to quickly find accurate information without manually searching through documents.

Business Requirements:

Answer questions based on company documents
Cite sources for answers
Handle updates to the knowledge base
Support multiple document types (PDFs, Word docs, text files)
This is the perfect scenario for a RAG (Retrieval-Augmented Generation) Agent.

What is RAG?
RAG combines:

Retrieval: Finding relevant information from a knowledge base
Augmentation: Adding that information to the agent's context
Generation: Using the LLM to generate answers based on retrieved information
text

User Query → Vector Search → Retrieve Relevant Docs → LLM + Context → Response
                 ↓
          Knowledge Base
        (Vector Store)
Key Benefits:

Grounds responses in your actual documents (reduces hallucination)
Automatically cites sources
Easily updatable (just add/remove documents)
Works with large document collections
Architecture Overview
A RAG Agent in the Microsoft Agent Framework uses:

HostedFileSearchTool: Built-in vector search capability
Vector Store: Document embeddings for semantic search
File Upload: Your knowledge base documents
Agent with Instructions: Orchestrates retrieval and generation
text

┌─────────────────────────────────────────────┐
│            User Query                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    Agent with HostedFileSearchTool          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Vector Store (Embedded Documents)          │
│  - company_policy.pdf                       │
│  - employee_handbook.docx                   │
│  - technical_docs.txt                       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Retrieved Relevant Chunks + LLM → Answer   │
└─────────────────────────────────────────────┘
Complete Implementation
Let's build a complete RAG agent for a company knowledge base:

Python

import asyncio
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

# Helper functions for vector store management
async def create_knowledge_base(client: OpenAIAssistantsClient) -> tuple[list[str], HostedVectorStoreContent]:
    """
    Create a vector store with company knowledge base documents.
    
    Returns:
        tuple: (list of file IDs, HostedVectorStoreContent object)
    """
    print("📚 Creating knowledge base...")
    
    # Sample documents (in production, these would be real files)
    documents = [
        (
            "company_policy.txt",
            b"""Company Vacation Policy
            
            All full-time employees receive:
            - 15 days paid vacation per year
            - 10 sick days per year
            - 12 public holidays
            
            Vacation days must be requested at least 2 weeks in advance through the HR portal.
            Unused vacation days carry over up to 5 days maximum.
            """
        ),
        (
            "expense_policy.txt",
            b"""Expense Reimbursement Policy
            
            Employees can be reimbursed for:
            - Business travel (flights, hotels, meals)
            - Client entertainment (up to $500 per event)
            - Office supplies (up to $200 per month)
            - Professional development (courses, books, conferences)
            
            Submit expense reports within 30 days with receipts attached.
            Approval required for expenses over $1000.
            """
        ),
        (
            "remote_work_policy.txt",
            b"""Remote Work Policy
            
            Employees may work remotely up to 3 days per week.
            Remote work requirements:
            - Stable internet connection (minimum 25 Mbps)
            - Dedicated workspace
            - Available during core hours (10 AM - 3 PM local time)
            - Attend all required meetings via video
            
            Remote work days must be coordinated with your team manager.
            """
        ),
    ]
    
    # Upload files to OpenAI
    file_ids = []
    for filename, content in documents:
        file = await client.client.files.create(
            file=(filename, content),
            purpose="user_data"
        )
        file_ids.append(file.id)
        print(f"  ✅ Uploaded: {filename}")
    
    # Create vector store
    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7},  # Auto-cleanup after 7 days inactive
    )
    print(f"  ✅ Created vector store: {vector_store.id}")
    
    # Add files to vector store and wait for processing
    for file_id in file_ids:
        result = await client.client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file_id
        )
        
        if result.status == "completed":
            print(f"  ✅ Indexed file: {file_id}")
        else:
            print(f"  ❌ Failed to index file: {file_id} - {result.last_error}")
            raise Exception(f"Vector store file processing failed: {result.last_error}")
    
    print("✅ Knowledge base ready!\n")
    
    return file_ids, HostedVectorStoreContent(vector_store_id=vector_store.id)

async def cleanup_knowledge_base(
    client: OpenAIAssistantsClient,
    file_ids: list[str],
    vector_store_id: str
) -> None:
    """Clean up vector store and files."""
    print("\n🧹 Cleaning up resources...")
    
    # Delete vector store
    await client.client.vector_stores.delete(vector_store_id)
    print(f"  ✅ Deleted vector store: {vector_store_id}")
    
    # Delete files
    for file_id in file_ids:
        await client.client.files.delete(file_id)
        print(f"  ✅ Deleted file: {file_id}")
    
    print("✅ Cleanup complete!")

async def main():
    # Create the RAG agent
    client = OpenAIAssistantsClient()
    
    async with client.create_agent(
        instructions="""You are a helpful HR assistant that answers questions about company policies.
        
        When answering questions:
        1. Search the knowledge base using the file search tool
        2. Provide accurate information based on the documents
        3. Cite which policy document you're referencing
        4. If information isn't in the knowledge base, say so clearly
        5. Be concise but complete in your answers""",
        tools=HostedFileSearchTool(),
    ) as agent:
        
        # Create knowledge base
        file_ids, vector_store = await create_knowledge_base(client)
        
        try:
            # Simulate employee queries
            print("=== Employee Self-Service Q&A ===\n")
            
            queries = [
                "How many vacation days do I get per year?",
                "What are the requirements for working remotely?",
                "Can I expense client dinners? What's the limit?",
                "Do unused vacation days carry over to next year?",
            ]
            
            for query in queries:
                print(f"👤 Employee: {query}")
                print("🤖 HR Assistant: ", end="", flush=True)
                
                # Run query with file search against the vector store
                async for chunk in agent.run_stream(
                    query,
                    tool_resources={
                        "file_search": {
                            "vector_store_ids": [vector_store.vector_store_id]
                        }
                    }
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                
                print("\n")  # Spacing between queries
        
        finally:
            # Clean up resources
            await cleanup_knowledge_base(client, file_ids, vector_store.vector_store_id)

if __name__ == "__main__":
    asyncio.run(main())
Vector Store Setup and Management
Creating a Vector Store
Vector stores hold embedded versions of your documents for semantic search:

Python

# Create a vector store with expiration
vector_store = await client.client.vector_stores.create(
    name="my_knowledge_base",
    expires_after={
        "anchor": "last_active_at",  # Start counting from last use
        "days": 7,                    # Delete after 7 days of inactivity
    },
)
Expiration Options:

anchor: When to start counting ("last_active_at" or "created_at")
days: Number of days until automatic deletion (1-365)
File Upload and Indexing
Files must be uploaded and indexed before they can be searched:

Python

# Upload file
file = await client.client.files.create(
    file=("document.pdf", open("document.pdf", "rb")),
    purpose="user_data"  # Required for vector stores
)

# Add to vector store and wait for indexing
result = await client.client.vector_stores.files.create_and_poll(
    vector_store_id=vector_store.id,
    file_id=file.id
)

# Check indexing status
if result.status == "completed":
    print("File indexed successfully!")
elif result.status == "failed":
    print(f"Indexing failed: {result.last_error.message}")
Supported File Types:

.pdf, .doc, .docx
.txt, .md
.html
.json, .csv
File Size Limits:

Maximum file size: 512 MB
Maximum characters per file: 5,000,000
Batch File Operations
For multiple files, batch operations are more efficient:

Python

# Create multiple file uploads
file_ids = []
for filepath in ["doc1.pdf", "doc2.txt", "doc3.md"]:
    file = await client.client.files.create(
        file=(filepath, open(filepath, "rb")),
        purpose="user_data"
    )
    file_ids.append(file.id)

# Batch add to vector store
batch = await client.client.vector_stores.file_batches.create_and_poll(
    vector_store_id=vector_store.id,
    file_ids=file_ids
)

print(f"Indexed {batch.file_counts.completed} files")
print(f"Failed: {batch.file_counts.failed} files")
Query Optimization Techniques
Technique 1: Provide Context in Instructions
Python

# ❌ Generic instructions
instructions = "You are a helpful assistant."

# ✅ Domain-specific instructions
instructions = """You are an HR policy expert for Acme Corp.
When answering questions:
- Always search the knowledge base first
- Reference specific policy documents in your answers
- If information is outdated or unclear, recommend contacting HR
- Use formal but friendly tone"""
Technique 2: Use Hybrid Search (Keyword + Semantic)
The file search tool automatically uses hybrid search, but you can optimize queries:

Python

# ❌ Vague query
query = "time off"

# ✅ Specific query with keywords
query = "How many paid vacation days are full-time employees entitled to per year?"
Technique 3: Chunk Large Documents
For better search accuracy, break large documents into smaller, focused sections:

Python

# Instead of one 100-page employee handbook:
handbook = "100_page_handbook.pdf"

# Split into focused documents:
documents = [
    "vacation_policy.txt",
    "expense_policy.txt",
    "code_of_conduct.txt",
    "benefits_overview.txt",
]
Best Practices for Knowledge Bases
✅ DO:

Organize Documents Logically: Use clear filenames and structure documents by topic
Keep Documents Updated: Regularly refresh the knowledge base with current information
Use Expiration Policies: Set appropriate expiration times to avoid stale data
Monitor Search Quality: Track which queries don't return good results
Provide Source Citations: Instruct agents to cite which documents they reference
Handle "Not Found" Gracefully: Tell the agent what to do when information isn't available
Python

instructions = """You are an HR assistant.

If you find information in the knowledge base, cite the document name.
If you cannot find relevant information, say:
'I don't have that information in my current knowledge base. Please contact HR at hr@company.com for assistance.'
"""
❌ DON'T:

Don't Upload Sensitive Data Unnecessarily: Only upload what's needed for the use case
Don't Ignore File Size Limits: Large files may fail to index
Don't Forget Cleanup: Delete vector stores and files when they're no longer needed
Don't Mix Unrelated Content: Keep knowledge bases focused on specific domains
Don't Skip Error Handling: Always check file indexing status
Cleanup Patterns
Always clean up resources to avoid unnecessary costs:

Python

async def safe_cleanup(client, file_ids, vector_store_id):
    """Safely cleanup resources with error handling."""
    errors = []
    
    # Try to delete vector store
    try:
        await client.client.vector_stores.delete(vector_store_id)
        print(f"✅ Deleted vector store: {vector_store_id}")
    except Exception as e:
        errors.append(f"Vector store deletion failed: {e}")
    
    # Try to delete each file
    for file_id in file_ids:
        try:
            await client.client.files.delete(file_id)
            print(f"✅ Deleted file: {file_id}")
        except Exception as e:
            errors.append(f"File {file_id} deletion failed: {e}")
    
    if errors:
        print("⚠️ Cleanup warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ All resources cleaned up successfully!")
Advanced: Custom Document Processing
For production systems, you might want custom preprocessing:

Python

import pypdf
from typing import List

async def process_and_upload_pdf(
    client: OpenAIAssistantsClient,
    pdf_path: str,
    chunk_size: int = 1000
) -> List[str]:
    """Process PDF into chunks and upload."""
    
    # Extract text from PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Split into chunks (simple example - use better chunking in production)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Upload chunks
    file_ids = []
    for i, chunk in enumerate(chunks):
        file = await client.client.files.create(
            file=(f"{pdf_path}_chunk_{i}.txt", chunk.encode()),
            purpose="user_data"
        )
        file_ids.append(file.id)
    
    return file_ids
5. Agent Type #3: Code Execution Agent
Use Case: Data Analysis Assistant
Imagine you're building an AI assistant for data analysts who need to:

Perform statistical calculations on datasets
Generate data visualizations
Analyze CSV files and extract insights
Perform mathematical computations
This is where a Code Execution Agent shines—it can write and execute Python code dynamically to solve computational problems.

What is Code Execution?
The HostedCodeInterpreterTool allows agents to write and run Python code in a secure, sandboxed environment. The agent:

Receives a query requiring computation
Writes Python code to solve the problem
Executes the code in a secure sandbox
Returns the output and/or files generated
text

User: "Calculate the factorial of 100"
        ↓
Agent writes Python code:
        import math
        result = math.factorial(100)
        print(result)
        ↓
Code Executor runs it → Returns result
        ↓
Agent formulates response with the result
Architecture Overview
text

┌─────────────────────────────────────────────┐
│         User Query (requires computation)    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Agent decides code execution is needed      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Agent generates Python code                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  HostedCodeInterpreterTool                   │
│  - Executes code in sandbox                  │
│  - Returns output/errors                     │
│  - Can generate files (images, CSVs, etc.)   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Agent formats final response                │
└─────────────────────────────────────────────┘
Complete Implementation
Let's build a data analysis assistant with code execution:

Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool, ChatResponse
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

async def main():
    print("=== Data Analysis Assistant with Code Execution ===\n")
    
    # Create agent with code interpreter capability
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="""You are a data analysis assistant with Python coding capabilities.
        
        When users ask for calculations, data analysis, or visualizations:
        1. Write clear, efficient Python code to solve the problem
        2. Use appropriate libraries (numpy, pandas, matplotlib, etc.)
        3. Explain what your code does
        4. Show the results clearly
        
        For complex calculations, always use code rather than manual computation.""",
        tools=HostedCodeInterpreterTool(),
    )
    
    # Example 1: Mathematical computation
    print("📊 Example 1: Complex Calculation")
    query1 = "Calculate the factorial of 100 and show it in scientific notation"
    print(f"User: {query1}")
    print("Assistant: ", end="", flush=True)
    
    result1 = await agent.run(query1)
    print(result1)
    
    # Extract and display generated code
    if (
        isinstance(result1.raw_representation, ChatResponse)
        and isinstance(result1.raw_representation.raw_representation, OpenAIResponse)
        and len(result1.raw_representation.raw_representation.output) > 0
        and isinstance(result1.raw_representation.raw_representation.output[0], ResponseCodeInterpreterToolCall)
    ):
        generated_code = result1.raw_representation.raw_representation.output[0].code
        print(f"\n💻 Generated Code:\n{generated_code}\n")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Statistical analysis
    print("📊 Example 2: Statistical Analysis")
    query2 = """I have these test scores: 85, 92, 78, 95, 88, 76, 90, 84, 89, 91.
    Calculate the mean, median, standard deviation, and identify outliers."""
    print(f"User: {query2}")
    print("Assistant: ", end="", flush=True)
    
    result2 = await agent.run(query2)
    print(result2)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Data transformation
    print("📊 Example 3: Data Processing")
    query3 = """Generate a list of the first 20 Fibonacci numbers and calculate:
    1. The sum of all numbers
    2. The ratio of consecutive numbers (should approach golden ratio)
    3. Show the last 5 ratios"""
    print(f"User: {query3}")
    print("Assistant: ", end="", flush=True)
    
    result3 = await agent.run(query3)
    print(result3)
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Streaming response
    print("📊 Example 4: Streaming Response")
    query4 = "Calculate the sum of squares for numbers 1 to 100 and explain the formula"
    print(f"User: {query4}")
    print("Assistant: ", end="", flush=True)
    
    async for chunk in agent.run_stream(query4):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
Code Interpreter Capabilities
The code interpreter supports a rich Python environment:

Pre-installed Libraries
Python

# Data Analysis
import numpy as np
import pandas as pd
import scipy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Math and Statistics
import math
import statistics
from decimal import Decimal

# File Processing
import csv
import json
import xml.etree.ElementTree as ET

# Date and Time
from datetime import datetime, timedelta
import calendar
File Operations
The code interpreter can create and work with files:

Python

# Agent can generate CSV files
query = "Create a CSV with sales data for Q1 2024 and calculate total sales"

# Agent might write code like:
"""
import pandas as pd

data = {
    'Month': ['Jan', 'Feb', 'Mar'],
    'Sales': [15000, 18000, 22000]
}
df = pd.DataFrame(data)
df.to_csv('q1_sales.csv', index=False)

total = df['Sales'].sum()
print(f"Total Q1 Sales: ${total:,}")
"""
Limitations
🚫 What Code Interpreter CANNOT Do:

Network Access: Cannot make HTTP requests or access external APIs
File System Access: Cannot access files outside the sandbox
Long-Running Tasks: Execution timeout (typically 60-120 seconds)
Large Memory Operations: Limited memory (typically ~512MB)
Install New Packages: Cannot use pip or install additional libraries
✅ What Code Interpreter CAN Do:

Complex Calculations: Mathematical computations, statistical analysis
Data Processing: CSV/JSON parsing, data transformations
Generate Visualizations: Create charts and graphs (as image files)
Text Processing: String manipulation, regex, parsing
File Generation: Create CSV, JSON, text files, images
Security Considerations
Sandboxing
The code interpreter runs in a completely isolated sandbox:

No access to your local file system
No network connectivity
No access to your environment variables or secrets
Code cannot escape the sandbox
Input Validation
Even though code runs in a sandbox, validate inputs:

Python

# ❌ Don't blindly trust user input
query = "Calculate result for user_input = 999999999999999999"  # Could cause performance issues

# ✅ Set expectations in instructions
instructions = """You are a data assistant.
When writing code:
- Validate input ranges (e.g., factorials only up to 1000)
- Handle edge cases gracefully
- Set timeouts for potentially long operations
- Avoid memory-intensive operations with user-provided large numbers"""
Monitoring Generated Code
For production use, monitor what code is being generated:

Python

async def run_with_code_monitoring(agent, query):
    """Run agent and log generated code for security review."""
    result = await agent.run(query)
    
    # Extract generated code
    if (
        isinstance(result.raw_representation, ChatResponse)
        and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
        and result.raw_representation.raw_representation.output
    ):
        for output in result.raw_representation.raw_representation.output:
            if isinstance(output, ResponseCodeInterpreterToolCall):
                generated_code = output.code
                
                # Log for security review
                print(f"[SECURITY LOG] Generated Code:\n{generated_code}\n")
                
                # Could add additional checks here
                if "os.system" in generated_code or "subprocess" in generated_code:
                    print("[WARNING] Potentially unsafe code detected!")
    
    return result
Accessing Generated Code from Responses
To extract the actual code that was generated:

Python

from agent_framework import ChatResponse
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

async def get_generated_code(result):
    """Extract generated Python code from agent response."""
    
    if not isinstance(result.raw_representation, ChatResponse):
        return None
    
    if not isinstance(result.raw_representation.raw_representation, OpenAIResponse):
        return None
    
    outputs = result.raw_representation.raw_representation.output
    if not outputs:
        return None
    
    codes = []
    for output in outputs:
        if isinstance(output, ResponseCodeInterpreterToolCall):
            codes.append(output.code)
    
    return "\n\n".join(codes) if codes else None

# Usage
result = await agent.run("Calculate factorial of 50")
code = await get_generated_code(result)
if code:
    print(f"Generated Code:\n{code}")
Best Practices and Use Case Matrix
When to Use Code Execution
Use Case	Good Fit?	Why
Mathematical Calculations	✅ Excellent	Precise, handles large numbers, shows work
Statistical Analysis	✅ Excellent	Access to numpy/pandas/scipy
Data Transformation	✅ Excellent	Clean data processing with Python
Chart Generation	✅ Good	Can create matplotlib/seaborn visualizations
CSV/JSON Processing	✅ Excellent	Built-in libraries for data formats
Web Scraping	❌ No	No network access
API Calls	❌ No	No network access
Real-time Data	❌ No	No external data sources
Large File Processing	⚠️ Limited	Memory and time constraints
Best Practices
✅ DO:

Set Clear Expectations: Tell the agent when to use code vs. when to calculate mentally
Handle Timeouts: Be aware of execution time limits
Validate Inputs: Use instructions to set boundaries on computations
Extract Results: Parse code output from responses when needed
Log for Debugging: Keep track of generated code for troubleshooting
Python

instructions = """You are a calculation assistant.

For simple math (2+2, 10*5): Calculate mentally.
For complex math (factorials, statistics, data processing): Write Python code.

When writing code:
- Keep it simple and readable
- Add comments explaining key steps
- Handle potential errors (division by zero, etc.)
- Limit loops to reasonable sizes (< 1 million iterations)"""
❌ DON'T:

Don't Assume Network Access: Code interpreter cannot make API calls
Don't Process Sensitive Data: Avoid sending confidential data to be processed
Don't Expect Persistence: Code execution is stateless; files don't persist between runs
Don't Ignore Errors: Always check execution results
Don't Overuse for Simple Tasks: Simple calculations don't need code execution
Error Handling
Code execution can fail—handle it gracefully:

Python

async def safe_code_execution(agent, query, max_retries=2):
    """Execute query with retry logic for code execution errors."""
    
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            
            # Check if code execution succeeded
            if result.text and "error" not in result.text.lower():
                return result
            else:
                print(f"Attempt {attempt + 1} had issues, retrying...")
                
        except Exception as e:
            print(f"Execution error: {e}")
            if attempt == max_retries - 1:
                raise
    
    return result
6. Agent Type #4: Multi-Modal Agent
Use Case: Visual Content Analyzer & Research Assistant
Imagine building an AI assistant for a marketing team that needs to:

Analyze product images and suggest improvements
Research current market trends using web search
Provide detailed reasoning for strategic recommendations
Process both visual and textual information
This requires a Multi-Modal Agent with vision, web search, and reasoning capabilities.

What is Multi-Modal?
Multi-modal agents can process and understand multiple types of input:

Text: Natural language queries and documents
Images: Photos, screenshots, diagrams, charts
Real-time Data: Web search for current information
Reasoning: Step-by-step logical thinking for complex problems
text

┌─────────────────────────────────────────────┐
│     Multi-Modal Input                        │
│  - Text Query                                │
│  - Image URL/Data                            │
│  - Web Search Context                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Multi-Modal Agent                        │
│  - Vision Model (analyzes images)            │
│  - Web Search Tool (finds current info)      │
│  - Reasoning (thinks through problems)       │
│  - Language Model (generates responses)      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Integrated Response                      │
│  - Combines visual analysis                  │
│  - Incorporates web research                 │
│  - Shows reasoning process                   │
│  - Natural language output                   │
└─────────────────────────────────────────────┘
Architecture Overview
Multi-modal agents combine multiple capabilities:

Vision: Analyze images using vision-capable models (gpt-4o, gpt-4o-mini)
Web Search: Retrieve current information from the internet
Reasoning: Show step-by-step thinking process (gpt-5 models)
Function Calling: Execute custom tools and APIs
Complete Implementation: Image Analysis
Let's build an agent that can analyze images:

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.openai import OpenAIResponsesClient

async def analyze_image_example():
    """Example of image analysis with vision-capable agent."""
    print("=== 🎨 Image Analysis Agent ===\n")
    
    # Create agent with vision capabilities (requires gpt-4o or gpt-4o-mini)
    agent = OpenAIResponsesClient(model_id="gpt-4o").create_agent(
        name="VisionAnalyst",
        instructions="""You are a professional image analyst.
        
        When analyzing images:
        1. Describe what you see in detail
        2. Identify key elements, colors, composition
        3. Suggest improvements if asked
        4. Be specific and actionable in your feedback""",
    )
    
    # Create message with both text and image
    user_message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="Analyze this image. What do you see? What's the mood and composition like?"),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                media_type="image/jpeg",
            ),
        ],
    )
    
    print("👤 User: [Provided image of nature boardwalk]")
    print("     Query: Analyze this image. What do you see?\n")
    print("🤖 VisionAnalyst: ", end="", flush=True)
    
    # Get analysis
    result = await agent.run(user_message)
    print(result.text)
    print("\n" + "="*60 + "\n")
    
    # Follow-up question with context
    followup = ChatMessage(
        role="user",
        contents=[
            TextContent(text="What photography techniques were used here?"),
        ],
    )
    
    print("👤 User: What photography techniques were used here?\n")
    print("🤖 VisionAnalyst: ", end="", flush=True)
    
    # Create thread to maintain context
    thread = agent.get_new_thread()
    await agent.run(user_message, thread=thread)  # First message
    result2 = await agent.run(followup, thread=thread)  # Follow-up
    print(result2.text)

if __name__ == "__main__":
    asyncio.run(analyze_image_example())
Web Search Integration
Enable agents to access current, real-time information:

Python

import asyncio
from agent_framework import HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def web_search_example():
    """Example of agent with web search capabilities."""
    print("=== 🔍 Web Search Research Agent ===\n")
    
    # Create agent with web search (requires gpt-4o-search-preview or similar)
    agent = OpenAIResponsesClient(model_id="gpt-4o-search-preview").create_agent(
        name="ResearchAssistant",
        instructions="""You are a research assistant with web search capabilities.
        
        When answering questions:
        1. Use web search to find current, accurate information
        2. Cite your sources
        3. Synthesize information from multiple sources
        4. Distinguish between facts and opinions
        5. Note when information might be time-sensitive""",
        tools=[
            HostedWebSearchTool(
                additional_properties={
                    "user_location": {
                        "country": "US",
                        "city": "Seattle",
                    }
                }
            )
        ],
    )
    
    queries = [
        "What's the current weather in Seattle? I'm located there.",
        "What are the latest developments in AI agents in 2024?",
        "Compare the top 3 cloud providers for AI workloads today",
    ]
    
    for query in queries:
        print(f"👤 User: {query}\n")
        print("🤖 ResearchAssistant: ", end="", flush=True)
        
        result = await agent.run(query)
        print(result.text)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(web_search_example())
Reasoning Capabilities
Show the agent's thought process with reasoning models:

Python

import asyncio
from agent_framework.openai import OpenAIResponsesClient

async def reasoning_example():
    """Example of agent with visible reasoning process."""
    print("=== 🧠 Reasoning Agent ===\n")
    
    # Create agent with reasoning enabled (requires gpt-5 or o1 models)
    agent = OpenAIResponsesClient(model_id="gpt-5").create_agent(
        name="MathTutor",
        instructions="""You are a patient math tutor.
        
        When solving problems:
        1. Think through the problem step-by-step
        2. Explain your reasoning clearly
        3. Show your work
        4. Verify your answer""",
        additional_chat_options={
            "reasoning": {
                "effort": "high",      # How much reasoning to apply
                "summary": "detailed"  # How detailed to show reasoning
            }
        },
    )
    
    query = "I need to solve the equation 3x + 11 = 14. Can you help me understand how to solve it?"
    
    print(f"👤 Student: {query}\n")
    print("🤖 MathTutor:\n")
    
    # Stream the response to see reasoning and answer
    async for chunk in agent.run_stream(query):
        if chunk.contents:
            for content in chunk.contents:
                # Reasoning appears as text_reasoning type
                if content.type == "text_reasoning":
                    print(f"💭 [Reasoning]: {content.text}", end="", flush=True)
                # Final answer appears as text type
                elif content.type == "text":
                    print(f"📝 [Answer]: {content.text}", end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    asyncio.run(reasoning_example())
Combined Multi-Capability Example
Here's a comprehensive example combining vision, web search, and reasoning:

Python

import asyncio
from agent_framework import ChatMessage, TextContent, UriContent, HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

async def multi_modal_agent():
    """Comprehensive multi-modal agent with vision, web search, and reasoning."""
    print("=== 🚀 Multi-Modal Marketing Analysis Agent ===\n")
    
    agent = OpenAIResponsesClient(model_id="gpt-5").create_agent(
        name="MarketingStrategist",
        instructions="""You are a senior marketing strategist with visual analysis and research capabilities.
        
        Your workflow:
        1. If given images, analyze them for marketing effectiveness
        2. Use web search to research current trends and competitors
        3. Think through strategic recommendations carefully
        4. Provide actionable, data-driven advice
        5. Support recommendations with both visual analysis and market research""",
        tools=[
            HostedWebSearchTool(
                additional_properties={
                    "user_location": {"country": "US", "city": "New York"}
                }
            )
        ],
        additional_chat_options={
            "reasoning": {"effort": "high", "summary": "detailed"}
        },
    )
    
    # Scenario: Analyzing a product image and providing strategic recommendations
    task = ChatMessage(
        role="user",
        contents=[
            TextContent(
                text="""Analyze this product packaging and:
                1. Evaluate its visual appeal and brand positioning
                2. Research current trends in sustainable packaging
                3. Recommend improvements based on market analysis"""
            ),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Product_packaging.jpg/800px-Product_packaging.jpg",
                media_type="image/jpeg",
            ),
        ],
    )
    
    print("👤 Marketing Manager: [Provided product packaging image]")
    print("     Task: Analyze and provide strategic recommendations\n")
    print("🤖 MarketingStrategist:\n")
    
    # Stream the comprehensive analysis
    async for chunk in agent.run_stream(task):
        if chunk.contents:
            for content in chunk.contents:
                if content.type == "text_reasoning":
                    print(f"💭 {content.text}", end="", flush=True)
                elif content.type == "text":
                    print(f"{content.text}", end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    asyncio.run(multi_modal_agent())
Best Practices for Complex Agents
✅ DO:

Choose the Right Model:

Vision tasks: gpt-4o, gpt-4o-mini
Web search: gpt-4o-search-preview
Reasoning: gpt-5, o1-preview
Structure Instructions Clearly:

Python

instructions = """You are a [ROLE].

Capabilities:
- Vision: [when and how to use]
- Web Search: [when and how to use]
- Reasoning: [when and how to use]

Workflow:
1. [First step]
2. [Second step]
3. [Final step]

Output Format:
- [How to structure responses]"""
Handle Different Content Types:
Python

from agent_framework import ChatMessage, TextContent, UriContent, DataContent

# Image from URL
msg_url = ChatMessage(
    role="user",
    contents=[
        TextContent(text="Analyze this"),
        UriContent(uri="https://example.com/image.jpg", media_type="image/jpeg"),
    ],
)

# Image from base64 data
msg_data = ChatMessage(
    role="user",
    contents=[
        TextContent(text="Analyze this"),
        DataContent(uri="data:image/jpeg;base64,/9j/4AAQ...", media_type="image/jpeg"),
    ],
)
Optimize Reasoning Settings:
Python

# For complex strategic decisions
additional_chat_options = {
    "reasoning": {
        "effort": "high",        # Maximum reasoning depth
        "summary": "detailed"    # Show full thought process
    }
}

# For simple tasks
additional_chat_options = {
    "reasoning": {
        "effort": "low",         # Minimal reasoning
        "summary": "concise"     # Brief summary only
    }
}
Stream Multi-Modal Responses:
Python

async for chunk in agent.run_stream(query):
    if chunk.contents:
        for content in chunk.contents:
            # Handle different content types
            if content.type == "text":
                print(content.text, end="")
            elif content.type == "text_reasoning":
                print(f"[Thinking: {content.text}]", end="")
            elif content.type == "usage":
                # Track token usage
                print(f"\n[Tokens used: {content.details}]")
❌ DON'T:

Don't Use Vision Models for Text-Only Tasks: Vision models cost more
Don't Over-Use Reasoning: High reasoning effort increases latency and cost
Don't Send Large Images: Optimize image sizes (< 2MB recommended)
Don't Ignore Token Costs: Multi-modal inputs use more tokens
Don't Mix Incompatible Features: Check model capabilities (not all models support all features)
Performance Considerations
Image Optimization
Python

from PIL import Image
import base64
from io import BytesIO

def optimize_image_for_vision(image_path: str, max_size: tuple = (800, 800)) -> str:
    """Optimize image for vision API to reduce costs and latency."""
    
    # Open and resize image
    img = Image.open(image_path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_base64}"

# Usage
optimized_uri = optimize_image_for_vision("large_image.jpg")
message = ChatMessage(
    role="user",
    contents=[
        TextContent(text="Analyze this"),
        DataContent(uri=optimized_uri, media_type="image/jpeg"),
    ],
)
Cost Management
Multi-modal features have different costs:

Feature	Relative Cost	When to Use
Text-only	1x (baseline)	Default for text tasks
Vision	2-3x	When image analysis is needed
Web Search	1.5-2x	When current information required
High Reasoning	3-4x	Complex strategic decisions
Combined	5-10x	Critical multi-modal analysis
Cost Optimization Tips:

Python

# ✅ Good: Only use vision when needed
if user_uploaded_image:
    agent = OpenAIResponsesClient(model_id="gpt-4o").create_agent(...)
else:
    agent = OpenAIResponsesClient(model_id="gpt-4o-mini").create_agent(...)

# ✅ Good: Adjust reasoning effort based on task complexity
if complex_strategic_decision:
    options = {"reasoning": {"effort": "high"}}
else:
    options = {"reasoning": {"effort": "low"}}
7. Agent Type #5: MCP-Integrated Agent
Use Case: Enterprise Tool Integration
Imagine you're building an AI assistant that needs to integrate with your company's internal systems:

Query your CRM for customer data
Access your knowledge base (SharePoint, Confluence)
Trigger workflows in your project management system
Search enterprise documentation
The Model Context Protocol (MCP) provides a standardized way to connect agents to external tools and services. This is where MCP-Integrated Agents excel.

What is MCP?
The Model Context Protocol (MCP) is an open standard for connecting AI agents to external tools, data sources, and services. Think of it as a universal adapter for your AI agent.

text

┌─────────────────────────────────────────────┐
│         AI Agent                             │
└─────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   MCP Interface       │
        └───────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
┌─────────┐                   ┌─────────┐
│  Local  │                   │ Hosted  │
│  MCP    │                   │  MCP    │
│ Server  │                   │ Server  │
└─────────┘                   └─────────┘
    ↓                               ↓
┌─────────┐                   ┌─────────┐
│ Your    │                   │ Cloud   │
│ Tools   │                   │ Service │
└─────────┘                   └─────────┘
Key Concepts:

Local MCP: Runs in your environment (full control, no approval needed)
Hosted MCP: Runs as a service (requires approval workflow for security)
MCP Tools: Standardized interface to external capabilities
Approval Workflow: Security mechanism for hosted MCPs
Architecture Overview
Local MCP Architecture
text

Agent → MCPStreamableHTTPTool → Local MCP Server → Your Services
No approval required
Direct execution
Full control over security
Hosted MCP Architecture
text

Agent → HostedMCPTool → Service → Approval Request → User Approves → Execution
Service-managed
Approval workflow for security
Reduced infrastructure management
Complete Implementation: Local MCP
Let's build an agent that connects to a local MCP server (Microsoft Learn documentation):

Python

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIResponsesClient

async def local_mcp_example():
    """Example using local MCP server for documentation search."""
    print("=== 📚 Documentation Assistant with Local MCP ===\n")
    
    # Create agent with local MCP tool
    # The agent connects to the MCP server which provides documentation search
    async with OpenAIResponsesClient().create_agent(
        name="DocsAssistant",
        instructions="""You are a helpful documentation assistant.
        
        When users ask questions:
        1. Search the documentation using the available tools
        2. Provide accurate, cited answers
        3. Include links to relevant documentation
        4. If you can't find information, say so clearly""",
        tools=MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ),
    ) as agent:
        
        # Example queries
        queries = [
            "How do I create an Azure storage account using Azure CLI?",
            "What is the Microsoft Agent Framework?",
            "How do I deploy a container to Azure Container Apps?",
        ]
        
        # Maintain conversation context
        thread = agent.get_new_thread()
        
        for query in queries:
            print(f"👤 Developer: {query}\n")
            print("🤖 DocsAssistant: ", end="", flush=True)
            
            async for chunk in agent.run_stream(query, thread=thread):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(local_mcp_example())
Hosted MCP with Approval Workflow
Hosted MCPs require user approval before executing functions for security:

Python

import asyncio
from typing import Any, TYPE_CHECKING
from agent_framework import ChatAgent, HostedMCPTool, ChatMessage
from agent_framework.openai import OpenAIResponsesClient

if TYPE_CHECKING:
    from agent_framework import AgentProtocol, AgentThread

async def handle_approvals_with_thread(
    query: str,
    agent: "AgentProtocol",
    thread: "AgentThread"
):
    """Handle approval workflow for hosted MCP function calls."""
    
    # Initial run - may return approval requests
    result = await agent.run(query, thread=thread, store=True)
    
    # Loop until no more approvals needed
    while len(result.user_input_requests) > 0:
        new_inputs: list[Any] = []
        
        for user_input_needed in result.user_input_requests:
            # Display approval request
            print(f"\n⚠️  Approval Request:")
            print(f"   Function: {user_input_needed.function_call.name}")
            print(f"   Arguments: {user_input_needed.function_call.arguments}")
            
            # Get user approval
            user_approval = input("   Approve this function call? (y/n): ")
            approved = user_approval.lower() == "y"
            
            # Create approval response
            new_inputs.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(approved)]
                )
            )
            
            if approved:
                print("   ✅ Approved")
            else:
                print("   ❌ Denied")
        
        # Re-run with approval responses
        result = await agent.run(new_inputs, thread=thread, store=True)
    
    return result

async def hosted_mcp_example():
    """Example using hosted MCP with approval workflow."""
    print("=== 🔐 Secure Hosted MCP with Approvals ===\n")
    
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="SecureDocsAgent",
        instructions="You are a documentation assistant with secure tool access.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # Require approval for all function calls
            approval_mode="always_require",
        ),
    ) as agent:
        
        thread = agent.get_new_thread()
        
        query = "Search for information about Azure Functions"
        print(f"👤 User: {query}\n")
        print("🤖 SecureDocsAgent: ", end="", flush=True)
        
        result = await handle_approvals_with_thread(query, agent, thread)
        print(result.text)

if __name__ == "__main__":
    asyncio.run(hosted_mcp_example())
Approval Modes
The framework supports three approval strategies:

1. Never Require Approval (Automatic Execution)
Python

tools=HostedMCPTool(
    name="My MCP",
    url="https://api.example.com/mcp",
    approval_mode="never_require",  # Auto-approve all calls
)
Use when:

Tool is completely trusted
Functions are read-only
Running in secure environment
⚠️ Security Risk: Functions execute without user confirmation

2. Always Require Approval (Maximum Security)
Python

tools=HostedMCPTool(
    name="My MCP",
    url="https://api.example.com/mcp",
    approval_mode="always_require",  # User approves every call
)
Use when:

Functions modify data
Running in production
Security is critical
✅ Most Secure: User reviews every function call

3. Selective Approval (Balanced Approach)
Python

tools=HostedMCPTool(
    name="My MCP",
    url="https://api.example.com/mcp",
    approval_mode={
        "never_require_approval": ["search_docs", "get_info"],  # Auto-approve these
        # All other functions require approval
    },
)
Use when:

Mix of safe and sensitive functions
Balance security and UX
Read operations are safe, writes need approval
✅ Recommended: Best balance of security and usability

Three Approval Handling Patterns
Pattern 1: Without Thread (Stateless)
Python

async def handle_approvals_without_thread(query: str, agent: "AgentProtocol"):
    """Handle approvals by resending full context each time."""
    from agent_framework import ChatMessage
    
    result = await agent.run(query)
    
    while len(result.user_input_requests) > 0:
        # Build full message history
        new_inputs: list[Any] = [query]
        
        for user_input_needed in result.user_input_requests:
            # Add approval request to history
            new_inputs.append(
                ChatMessage(role="assistant", contents=[user_input_needed])
            )
            
            # Get approval
            user_approval = input("Approve? (y/n): ")
            
            # Add approval response to history
            new_inputs.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                )
            )
        
        # Re-run with full history
        result = await agent.run(new_inputs)
    
    return result
Pattern 2: With Thread (Stateful)
Python

async def handle_approvals_with_thread(
    query: str,
    agent: "AgentProtocol",
    thread: "AgentThread"
):
    """Handle approvals using thread to maintain state."""
    from agent_framework import ChatMessage
    
    # Thread stores history, so we only send new messages
    result = await agent.run(query, thread=thread, store=True)
    
    while len(result.user_input_requests) > 0:
        new_inputs: list[ChatMessage] = []
        
        for user_input_needed in result.user_input_requests:
            user_approval = input("Approve? (y/n): ")
            
            # Only send approval response (thread has the rest)
            new_inputs.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                )
            )
        
        result = await agent.run(new_inputs, thread=thread, store=True)
    
    return result
Pattern 3: With Streaming
Python

async def handle_approvals_streaming(
    query: str,
    agent: "AgentProtocol",
    thread: "AgentThread"
):
    """Handle approvals in streaming mode."""
    from agent_framework import ChatMessage
    
    new_input: list[ChatMessage] = [ChatMessage(role="user", text=query)]
    new_input_added = True
    
    while new_input_added:
        new_input_added = False
        
        async for update in agent.run_stream(new_input, thread=thread, store=True):
            # Stream text content
            if update.text:
                yield update
            
            # Handle approval requests
            if update.user_input_requests:
                for user_input_needed in update.user_input_requests:
                    user_approval = input("\nApprove? (y/n): ")
                    
                    new_input.append(
                        ChatMessage(
                            role="user",
                            contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                        )
                    )
                    new_input_added = True
Best Practices for MCP Integration
✅ DO:

Choose the Right Approval Mode:
Python

# Production system with data modification
approval_mode = "always_require"

# Read-only documentation search
approval_mode = "never_require"

# Mixed operations
approval_mode = {
    "never_require_approval": ["read_only_func1", "read_only_func2"],
}
Provide Clear Instructions:
Python

instructions = """You are a helpful assistant with access to [TOOL NAME].

Available actions:
- [Action 1]: [When to use]
- [Action 2]: [When to use]

Important:
- Always explain what you're about to do before calling tools
- If a function call is denied, explain alternatives
- Respect user's approval decisions"""
Handle Approval Denials Gracefully:
Python

async def smart_approval_handler(agent, query, thread):
    result = await agent.run(query, thread=thread, store=True)
    
    while result.user_input_requests:
        for request in result.user_input_requests:
            print(f"Function: {request.function_call.name}")
            print(f"Purpose: {request.function_call.arguments}")
            
            approval = input("Approve? (y/n): ")
            approved = approval.lower() == "y"
            
            if not approved:
                print("💡 Tip: I can try alternative approaches!")
            
            response = ChatMessage(
                role="user",
                contents=[request.create_response(approved)]
            )
            result = await agent.run([response], thread=thread, store=True)
    
    return result
Log MCP Interactions:
Python

import logging

logger = logging.getLogger("mcp_agent")

async def logged_mcp_agent(agent, query):
    logger.info(f"Query: {query}")
    
    result = await agent.run(query)
    
    for request in result.user_input_requests:
        logger.info(f"MCP call requested: {request.function_call.name}")
        logger.info(f"Arguments: {request.function_call.arguments}")
        
        # Handle approval...
    
    logger.info(f"Response: {result.text}")
    return result
Use Context Managers for Connection Management:
Python

# ✅ Good: Automatic cleanup
async with OpenAIResponsesClient().create_agent(
    tools=MCPStreamableHTTPTool(name="My MCP", url="..."),
) as agent:
    result = await agent.run(query)
    # MCP connection automatically closed

# ❌ Avoid: Manual management
tool = MCPStreamableHTTPTool(name="My MCP", url="...")
await tool.connect()
agent = OpenAIResponsesClient().create_agent(tools=tool)
result = await agent.run(query)
await tool.disconnect()  # Easy to forget!
❌ DON'T:

Don't Use "never_require" for Destructive Operations
Don't Ignore Approval Responses: Always handle denials gracefully
Don't Hardcode MCP URLs: Use environment variables
Don't Skip Error Handling: MCP servers can fail
Don't Forget Connection Cleanup: Use context managers
Troubleshooting MCP Connections
Common Issues
Issue 1: Connection Timeout
Python

# Problem: MCP server not responding
async with MCPStreamableHTTPTool(
    name="My MCP",
    url="https://slow-server.com/mcp",
) as mcp:
    # Times out...
    pass

# Solution: Check server health first
import httpx

async def check_mcp_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health", timeout=5.0)
            return response.status_code == 200
    except:
        return False

# Usage
if await check_mcp_health("https://my-server.com/mcp"):
    async with MCPStreamableHTTPTool(...) as mcp:
        # Safe to proceed
        pass
else:
    print("MCP server unavailable")
Issue 2: Approval Loop
Python

# Problem: Infinite approval loop
while result.user_input_requests:
    # Forgot to send approval response!
    result = await agent.run(query, thread=thread)

# Solution: Always send approval responses
while result.user_input_requests:
    responses = []
    for request in result.user_input_requests:
        approval = input("Approve? (y/n): ")
        responses.append(
            ChatMessage(
                role="user",
                contents=[request.create_response(approval.lower() == "y")]
            )
        )
    
    # Send approval responses
    result = await agent.run(responses, thread=thread, store=True)
Issue 3: Missing Tools
Python

# Problem: Agent can't find MCP tools
agent = OpenAIResponsesClient().create_agent(
    tools=HostedMCPTool(name="My MCP", url="..."),
)
result = await agent.run("Use the search tool")  # "I don't have that tool"

# Solution: Check tool availability
async with agent:
    # Tools are loaded when agent context is entered
    result = await agent.run("Use the search tool")  # Works!
8. Advanced Topics
Thread Persistence Strategies
The framework supports multiple strategies for persisting conversation state:

Strategy 1: In-Memory (Default)
Conversation stored in application memory:

Python

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

agent = ChatAgent(chat_client=OpenAIChatClient(), instructions="You are helpful.")

thread = agent.get_new_thread()

# Messages stored in memory
await agent.run("Hello!", thread=thread)
await agent.run("How are you?", thread=thread)

# Access message history
if thread.message_store:
    messages = await thread.message_store.list_messages()
    print(f"Thread has {len(messages or [])} messages")
Pros:

Fast access
No external dependencies
Full control
Cons:

Lost when application restarts
Doesn't scale across servers
Memory consumption grows with long conversations
Use when: Single-server applications, short-lived sessions
