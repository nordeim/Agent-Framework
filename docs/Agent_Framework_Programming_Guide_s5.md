Microsoft Agent Framework: Comprehensive Programming Guide
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

Agent Type 2: Function-Calling Agent

Agent Type 3: RAG Agent (Retrieval-Augmented Generation)

Agent Type 4: Code Execution Agent

Agent Type 5: Multi-Modal Agent

Advanced Topics

Best Practices & Patterns

Troubleshooting Guide

Quick Reference

Glossary

Introduction
What is the Microsoft Agent Framework?
The Microsoft Agent Framework is a powerful, flexible Python framework designed to simplify the creation of AI-powered agents. Whether you're building a simple chatbot, a complex multi-agent system, or an AI assistant with specialized capabilities, this framework provides the abstractions and tools you need to build production-ready solutions quickly.

The framework supports multiple AI providers including OpenAI, Azure OpenAI, and Azure AI, offering a unified interface that makes it easy to switch between providers or leverage provider-specific features. With built-in support for function calling, file search, code interpretation, web search, and the Model Context Protocol (MCP), you can create sophisticated AI agents that interact with external systems, process documents, execute code, and much more.

Key Benefits:

Provider Agnostic: Work with OpenAI, Azure OpenAI, or Azure AI using the same patterns
Rich Tool Ecosystem: Built-in support for function calling, file search, code execution, web search, and MCP
Thread Management: Sophisticated conversation state management with both in-memory and service-based persistence
Streaming Support: Real-time response generation for better user experience
Production Ready: Comprehensive error handling, async/await patterns, and context managers for resource safety
Type Safe: Full typing support with Pydantic models for structured outputs
Framework Architecture
The Microsoft Agent Framework is organized around several core abstractions that work together to provide a flexible, powerful development experience:

text

┌─────────────────────────────────────────────────────────┐
│                     Agent Layer                          │
│  (ChatAgent, custom agents implementing AgentProtocol)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Client Layer                            │
│  (OpenAIChatClient, OpenAIResponsesClient,              │
│   OpenAIAssistantsClient, Azure variants, etc.)         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   Tool Layer                             │
│  (Function tools, HostedFileSearchTool,                 │
│   HostedCodeInterpreterTool, MCPTools, etc.)            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Message Layer                            │
│  (ChatMessage, ChatMessageStore, AgentThread)           │
└─────────────────────────────────────────────────────────┘
Agent Layer: High-level abstractions like ChatAgent that orchestrate the entire conversation flow. Agents manage the lifecycle of interactions, coordinate tool usage, and maintain conversation context.

Client Layer: Provider-specific clients that handle communication with AI services. Each client type (Chat, Responses, Assistants) offers different capabilities and interaction patterns.

Tool Layer: Extensible system for providing agents with additional capabilities like function calling, file search, code execution, and external integrations.

Message Layer: Core data structures for representing conversations, managing message history, and maintaining state across interactions.

Client Types Comparison
The framework provides three primary client types, each optimized for different use cases:

Feature	Chat Client	Responses Client	Assistants Client
Use Case	Standard conversational AI	Structured outputs, advanced features	Long-running assistants with state
State Management	In-memory by default	In-memory or service-based	Service-managed threads
Best For	Quick integrations, prototypes	Production apps, structured data	Complex multi-turn conversations
Streaming	✅ Full support	✅ Full support	✅ Full support
Function Calling	✅ Yes	✅ Yes	✅ Yes
File Search	❌ No	✅ Yes	✅ Yes
Code Interpreter	❌ No	✅ Yes	✅ Yes
Structured Outputs	❌ Limited	✅ Full Pydantic support	❌ Limited
Image Analysis	❌ No	✅ Yes	✅ Yes
Web Search	✅ Yes	✅ Yes	❌ No
Reasoning Models	❌ No	✅ Yes (GPT-5)	❌ No
Thread Persistence	In-memory	Optional service-based	Service-managed
Setup Complexity	Low	Low	Medium (assistant creation)
Chat Client is ideal for straightforward conversational interfaces, real-time interactions, and scenarios where you want full control over message history. It's the simplest to set up and perfect for getting started.

Responses Client offers the most comprehensive feature set, including structured outputs with Pydantic models, advanced reasoning capabilities, and full support for all tool types. It's the best choice for production applications requiring sophisticated AI capabilities.

Assistants Client provides service-managed conversation state and is best suited for long-running conversations where you want the AI service to handle thread persistence. It automatically manages assistant lifecycle and supports the full range of OpenAI Assistant features.

Environment Setup
Before building your first agent, you'll need to configure your environment with the appropriate credentials and settings.

Installation:

Bash

# Using pip
pip install agent-framework-openai

# Using uv (recommended for faster installs)
uv add agent-framework-openai
Environment Variables for OpenAI:

Bash

# Required
export OPENAI_API_KEY="sk-..."

# Optional - Model Configuration
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"

# Optional - Organization
export OPENAI_ORG_ID="org-..."
Environment Variables for Azure OpenAI:

Bash

# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="gpt-4o"

# Authentication (choose one)
export AZURE_OPENAI_API_KEY="..."
# OR use Azure AD authentication (recommended for production)
Environment Variables for Azure AI:

Bash

# Required
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.api.azureml.ms"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4o"
Verifying Your Setup:

Python

import os
from agent_framework.openai import OpenAIChatClient

# Quick verification
async def verify_setup():
    client = OpenAIChatClient()
    response = await client.get_response("Hello! Can you hear me?")
    print(f"Agent: {response}")

# Run verification
import asyncio
asyncio.run(verify_setup())
Your First Agent
Let's create a simple conversational agent to ensure everything is working correctly. This example demonstrates the minimal code needed to get started:

Python

import asyncio
from agent_framework.openai import OpenAIChatClient

async def main():
    # Create a simple agent with basic instructions
    agent = OpenAIChatClient().create_agent(
        name="GreeterAgent",
        instructions="You are a friendly assistant who greets users warmly."
    )
    
    # Send a message and get a response
    query = "Hello! What can you help me with?"
    print(f"User: {query}")
    
    result = await agent.run(query)
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
Output:

text

User: Hello! What can you help me with?
Agent: Hello! I'm here to help you with any questions or tasks you might have...
What's Happening Here:

Client Creation: OpenAIChatClient() initializes a client that will communicate with OpenAI's API
Agent Creation: .create_agent() creates an agent with a specific personality defined by the instructions
Running the Agent: .run(query) sends the message and waits for a complete response
Async Pattern: The framework uses async/await for efficient I/O operations
This simple example is the foundation for all agent interactions. Next, we'll explore the core concepts that will help you build more sophisticated agents.

Core Concepts
Understanding Client Types
The Microsoft Agent Framework provides three distinct client types, each designed for specific interaction patterns and capabilities. Understanding when to use each client is crucial for building effective AI agents.

OpenAIChatClient - Conversational Simplicity

The Chat Client is your go-to choice for standard conversational interfaces. It provides a straightforward request-response pattern where you send messages and receive replies. The client maintains conversation history in memory, giving you full control over what context the agent sees.

Python

from agent_framework.openai import OpenAIChatClient

# Initialize with automatic configuration from environment variables
client = OpenAIChatClient()

# Or with explicit settings
client = OpenAIChatClient(
    model_id="gpt-4o",
    api_key="sk-...",
    temperature=0.7
)
The Chat Client is ideal for:

Chatbots and virtual assistants
Real-time customer support interfaces
Prototyping and development
Scenarios requiring custom message history management
OpenAIResponsesClient - Advanced Capabilities

The Responses Client unlocks the full power of modern AI models, including structured outputs, reasoning capabilities, and comprehensive tool support. It's designed for production applications that need reliability, type safety, and advanced features.

Python

from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

# Standard initialization
client = OpenAIResponsesClient()

# With reasoning capabilities (GPT-5 models)
client = OpenAIResponsesClient(
    model_id="gpt-5",
    additional_chat_options={
        "reasoning": {
            "effort": "high",
            "summary": "detailed"
        }
    }
)

# Example: Structured output
class CityInfo(BaseModel):
    name: str
    population: int
    country: str

agent = client.create_agent(
    instructions="Extract city information from user queries."
)

result = await agent.run(
    "Tell me about Paris",
    response_format=CityInfo
)

# result.value is a validated CityInfo instance
city_data: CityInfo = result.value
The Responses Client is ideal for:

Production applications requiring structured data
Document analysis and knowledge extraction
Multi-modal applications (text, images, files)
Complex reasoning tasks
Applications using file search or code execution
OpenAIAssistantsClient - Service-Managed State

The Assistants Client leverages OpenAI's Assistants API, where conversation threads are managed by the service. This is particularly useful for applications where you want persistent conversation history without managing storage yourself.

Python

from agent_framework.openai import OpenAIAssistantsClient

# Automatic assistant creation and cleanup
async with OpenAIAssistantsClient().create_agent(
    instructions="You are a helpful assistant.",
    tools=[my_function]
) as agent:
    result = await agent.run("Hello!")
    # Assistant is automatically deleted when exiting the context

# Working with existing assistant
async with OpenAIAssistantsClient(
    assistant_id="asst_..."
).create_agent() as agent:
    result = await agent.run("Continue our conversation")
The Assistants Client is ideal for:

Multi-session applications where users return over time
Scenarios requiring service-managed conversation history
Applications leveraging OpenAI's assistant features
Long-running conversations with complex state
Agent Lifecycle Management
Proper lifecycle management ensures efficient resource usage and prevents memory leaks. The framework provides context managers for safe resource handling.

Context Manager Pattern (Recommended):

Python

# Automatic cleanup when done
async with client.create_agent(
    instructions="You are helpful.",
    tools=[my_tool]
) as agent:
    result = await agent.run("Hello")
    # Tool connections, threads, and resources automatically cleaned up

# Agent is no longer usable here - resources released
Manual Management Pattern:

Python

# Create agent
agent = client.create_agent(
    instructions="You are helpful.",
    tools=[my_tool]
)

try:
    result = await agent.run("Hello")
finally:
    # Manually clean up if needed
    if hasattr(agent, 'cleanup'):
        await agent.cleanup()
Lifecycle Events:

Initialization: Client creates connection to AI service
Agent Creation: Agent configured with instructions and tools
Tool Preparation: Tools initialized and connected (e.g., MCP servers)
Execution: Agent processes queries and generates responses
Cleanup: Resources released, connections closed, temporary assistants deleted
Thread Management Patterns
Threads represent conversation contexts, containing the message history that the agent uses to generate responses. The framework supports multiple thread management patterns.

Pattern 1: Stateless Interactions (No Thread)

Python

agent = client.create_agent(instructions="You are helpful.")

# Each call is independent - no memory between calls
response1 = await agent.run("What's the weather in Paris?")
response2 = await agent.run("What city did I just ask about?")
# Agent won't remember Paris - this is a new conversation
Pattern 2: In-Memory Thread Persistence

Python

agent = client.create_agent(instructions="You are helpful.")

# Create a thread to maintain conversation context
thread = agent.get_new_thread()

# All calls using this thread share context
response1 = await agent.run("What's the weather in Paris?", thread=thread)
response2 = await agent.run("What city did I just ask about?", thread=thread)
# Agent remembers Paris from the thread history
Pattern 3: Service-Managed Threads (Responses/Assistants Client)

Python

agent = client.create_agent(instructions="You are helpful.")

thread = agent.get_new_thread()

# Enable service-side persistence with store=True
response1 = await agent.run(
    "What's the weather in Paris?",
    thread=thread,
    store=True
)

# Thread ID is now set - can be persisted to your database
thread_id = thread.service_thread_id  # Save this

# Later, restore the conversation
restored_thread = AgentThread(service_thread_id=thread_id)
response2 = await agent.run(
    "What was the weather like?",
    thread=restored_thread,
    store=True
)
# Agent retrieves history from service
Pattern 4: Custom Message Storage

Python

from agent_framework import ChatMessageStore

# Initialize with existing messages
messages = [
    ChatMessage(role="user", text="Hello"),
    ChatMessage(role="assistant", text="Hi there!")
]

thread = AgentThread(
    message_store=ChatMessageStore(messages)
)

# Continue the conversation
response = await agent.run("What did I just say?", thread=thread)
Tool Integration Approaches
Tools extend agent capabilities, allowing them to call functions, search files, execute code, and more. The framework provides flexible patterns for tool integration.

Agent-Level Tools (Permanent Capabilities):

Python

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

def get_time() -> str:
    """Get current time."""
    return "2024-01-15 10:30 AM"

# Tools available for all queries
agent = client.create_agent(
    instructions="You can check weather and time.",
    tools=[get_weather, get_time]  # Available throughout agent lifetime
)

# Agent can use either tool
result1 = await agent.run("What's the weather in Tokyo?")
result2 = await agent.run("What time is it?")
Run-Level Tools (Dynamic Capabilities):

Python

agent = client.create_agent(
    instructions="You are a flexible assistant."
    # No tools defined at creation
)

# Provide tools for specific queries
result1 = await agent.run(
    "What's the weather in London?",
    tools=[get_weather]  # Only weather tool available
)

result2 = await agent.run(
    "What time is it?",
    tools=[get_time]  # Only time tool available
)
Mixed Approach (Base + Dynamic Tools):

Python

# Base capabilities always available
agent = client.create_agent(
    instructions="You have weather and can get additional tools.",
    tools=[get_weather]  # Always available
)

# Add tools for specific queries
result = await agent.run(
    "What's the weather in NYC and what time is it?",
    tools=[get_time]  # Agent has both get_weather and get_time
)
Tool Definition Best Practices:

Python

from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(
        description="The city and country, e.g., 'London, UK'"
    )],
    units: Annotated[str, Field(
        description="Temperature units: 'celsius' or 'fahrenheit'",
        default="celsius"
    )]
) -> str:
    """
    Get the current weather for a specific location.
    
    This function retrieves real-time weather data including
    temperature, conditions, and forecast.
    """
    # Implementation here
    pass

# The framework uses:
# - Function docstring for tool description
# - Parameter annotations for type information
# - Field descriptions for parameter guidance
# - Default values for optional parameters
Streaming vs Non-Streaming Responses
The framework supports both streaming and non-streaming response patterns, each suited for different scenarios.

Non-Streaming (Complete Response):

Python

# Wait for complete response
result = await agent.run("Write a long story about space exploration.")

# Get full text at once
print(result.text)  # Complete story

# Access all metadata
print(f"Tokens used: {result.usage_details}")
print(f"Messages: {result.messages}")
Advantages:

Simpler to implement
Complete metadata available immediately
Easier error handling
Better for batch processing
Streaming (Real-Time Response):

Python

# Stream response as it's generated
print("Agent: ", end="", flush=True)

async for chunk in agent.run_stream("Write a long story."):
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()  # New line after streaming completes
Advantages:

Better user experience for long responses
Lower perceived latency
Real-time progress indication
Can process data before completion
Streaming with Metadata Collection:

Python

from agent_framework import AgentRunResponse

# Collect streaming chunks into complete response
result = await AgentRunResponse.from_agent_response_generator(
    agent.run_stream("Tell me about AI"),
    output_format_type=MyOutputType  # Optional for structured output
)

# Now you have both streaming benefits and complete metadata
print(result.text)
print(result.usage_details)
Choosing Between Streaming and Non-Streaming:

Use non-streaming when:

Building APIs that return complete responses
Processing responses programmatically
Simplicity is more important than responsiveness
You need complete metadata before proceeding
Use streaming when:

Building interactive chat interfaces
Responses might be long
User experience is critical
You want to show progress
Error Handling Patterns
Robust error handling is essential for production applications. The framework provides several patterns for handling errors gracefully.

Basic Try-Catch Pattern:

Python

try:
    result = await agent.run("Hello")
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
Specific Error Handling:

Python

from openai import APIError, RateLimitError, APITimeoutError

try:
    result = await agent.run(query)
except RateLimitError:
    print("Rate limit exceeded. Waiting before retry...")
    await asyncio.sleep(60)
    # Implement retry logic
except APITimeoutError:
    print("Request timed out. Using fallback response...")
    # Provide fallback
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
    # Log and handle
except Exception as e:
    print(f"Unexpected error: {e}")
    # General error handling
Retry Pattern with Exponential Backoff:

Python

import asyncio
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
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Rate limited. Retrying in {delay}s...")
            await asyncio.sleep(delay)
        except APITimeoutError:
            if attempt == max_retries - 1:
                raise
            print(f"Timeout. Retrying attempt {attempt + 1}...")
    return None
Resource Cleanup with Error Handling:

Python

agent = None
try:
    agent = client.create_agent(
        instructions="You are helpful.",
        tools=[complex_tool]
    )
    result = await agent.run(query)
except Exception as e:
    print(f"Error during execution: {e}")
    # Log error for debugging
finally:
    if agent and hasattr(agent, 'cleanup'):
        await agent.cleanup()
Context Manager for Automatic Cleanup:

Python

# Recommended: Let context manager handle cleanup
async with client.create_agent(
    instructions="You are helpful.",
    tools=[tool]
) as agent:
    try:
        result = await agent.run(query)
    except Exception as e:
        print(f"Error: {e}")
        # Error handling logic
# Cleanup happens automatically, even if error occurred
Agent Type 1: Basic Conversational Agent
Use Case & Overview
A Basic Conversational Agent is the foundation of AI-powered chat applications. This agent type excels at natural language understanding and generation, making it perfect for customer support bots, virtual assistants, FAQ systems, and interactive chat interfaces. Unlike specialized agents with tools or advanced capabilities, the conversational agent focuses purely on understanding user intent and providing helpful, contextual responses.

Ideal Scenarios:

Customer service chatbots answering common questions
Educational tutors providing explanations
Personal assistants for scheduling and reminders
Information retrieval from trained knowledge
Creative writing assistance
General Q&A systems
Architecture & Components
A basic conversational agent consists of three primary components:

Client: Handles communication with the AI service
Agent: Orchestrates conversation flow and maintains instructions
Thread (optional): Manages conversation history and context
text

User Query → Agent (with instructions) → Client → AI Service → Response
                ↓                                              ↓
            Thread (conversation context) ←──────────────────┘
Complete Implementation Example
Here's a production-ready conversational agent with both streaming and non-streaming support:

Python

import asyncio
from typing import Optional
from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIChatClient

class ConversationalAssistant:
    """A production-ready conversational agent with context management."""
    
    def __init__(
        self,
        name: str = "Assistant",
        instructions: str = "You are a helpful assistant.",
        model_id: str = "gpt-4o"
    ):
        """
        Initialize the conversational assistant.
        
        Args:
            name: The agent's name
            instructions: System instructions defining agent behavior
            model_id: The model to use (gpt-4o, gpt-4o-mini, etc.)
        """
        self.client = OpenAIChatClient(model_id=model_id)
        self.agent = self.client.create_agent(
            name=name,
            instructions=instructions
        )
        self.thread: Optional[AgentThread] = None
        
    def start_conversation(self) -> None:
        """Start a new conversation with fresh context."""
        self.thread = self.agent.get_new_thread()
        
    async def chat(self, message: str, stream: bool = False) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: User's message
            stream: Whether to stream the response
            
        Returns:
            The agent's response text
        """
        if stream:
            return await self._chat_streaming(message)
        else:
            return await self._chat_complete(message)
    
    async def _chat_complete(self, message: str) -> str:
        """Get complete response at once."""
        result = await self.agent.run(message, thread=self.thread)
        return result.text
    
    async def _chat_streaming(self, message: str) -> str:
        """Stream response in real-time."""
        full_response = ""
        print("Assistant: ", end="", flush=True)
        
        async for chunk in self.agent.run_stream(message, thread=self.thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        
        print()  # New line after streaming
        return full_response


async def demo_basic_conversation():
    """Demonstrate basic conversational agent usage."""
    
    # Create a friendly customer support agent
    assistant = ConversationalAssistant(
        name="SupportBot",
        instructions="""You are a friendly customer support agent for TechCorp.
        Be helpful, professional, and empathetic. Answer questions about our 
        products, help troubleshoot issues, and provide general information.
        If you don't know something, admit it and offer to escalate."""
    )
    
    # Start a new conversation
    assistant.start_conversation()
    
    # Example interaction 1: Non-streaming
    query1 = "Hi! I'm having trouble logging into my account."
    print(f"User: {query1}")
    response1 = await assistant.chat(query1, stream=False)
    print(f"Assistant: {response1}\n")
    
    # Example interaction 2: Streaming (better UX for long responses)
    query2 = "Can you walk me through the password reset process step by step?"
    print(f"User: {query2}")
    await assistant.chat(query2, stream=True)
    print()
    
    # Example interaction 3: Context awareness
    query3 = "Thanks! One more question - do you offer two-factor authentication?"
    print(f"User: {query3}")
    response3 = await assistant.chat(query3, stream=False)
    print(f"Assistant: {response3}\n")


async def demo_specialized_assistant():
    """Demonstrate a specialized tutor agent."""
    
    tutor = ConversationalAssistant(
        name="MathTutor",
        instructions="""You are an encouraging math tutor for high school students.
        Explain concepts clearly with examples. When a student makes a mistake,
        gently correct them and explain why. Use analogies to make complex topics
        more relatable. Encourage questions and celebrate progress."""
    )
    
    tutor.start_conversation()
    
    questions = [
        "Can you explain what derivatives are?",
        "How do I find the derivative of x²?",
        "What about more complex functions?"
    ]
    
    for question in questions:
        print(f"Student: {question}")
        await tutor.chat(question, stream=True)
        print()


async def main():
    print("=== Basic Conversational Agent Demo ===\n")
    print("--- Customer Support Bot ---")
    await demo_basic_conversation()
    
    print("\n--- Specialized Math Tutor ---")
    await demo_specialized_assistant()


if __name__ == "__main__":
    asyncio.run(main())
Configuration Options
System Instructions: The instructions parameter is crucial for defining agent behavior:

Python

# Personality and tone
instructions = """You are a cheerful, encouraging fitness coach. 
Use motivating language and celebrate user progress. Be supportive 
but also provide honest, realistic advice."""

# Role and boundaries
instructions = """You are a medical information assistant. Provide 
general health information but ALWAYS remind users to consult 
healthcare professionals for medical advice. Never diagnose conditions."""

# Format and style
instructions = """You are a technical documentation assistant. 
Provide concise, accurate answers. Use bullet points for lists, 
code blocks for examples, and avoid unnecessary elaboration."""
Temperature Control (affects response randomness):

Python

# Creative writing (higher temperature)
client = OpenAIChatClient(
    model_id="gpt-4o",
    temperature=0.9  # More creative, varied responses
)

# Factual Q&A (lower temperature)
client = OpenAIChatClient(
    model_id="gpt-4o",
    temperature=0.3  # More focused, consistent responses
)
Context Window Management:

Python

# For long conversations, you may want to limit history
from agent_framework import ChatMessageStore

# Keep only last N messages
def get_limited_thread(full_thread: AgentThread, keep_last: int = 10):
    if full_thread.message_store:
        messages = await full_thread.message_store.list_messages()
        recent_messages = messages[-keep_last:] if messages else []
        return AgentThread(message_store=ChatMessageStore(recent_messages))
    return full_thread
Best Practices
1. Clear Instructions: Be specific about agent behavior, tone, and boundaries.

2. Context Management: Use threads for multi-turn conversations but monitor context length.

3. Error Handling: Always wrap agent calls in try-except blocks for production.

4. Streaming for UX: Use streaming for better user experience in interactive applications.

5. Testing Instructions: Test your instructions with various inputs to ensure desired behavior.

Common Pitfalls
❌ Overly Complex Instructions: Keep instructions focused. Too much detail can confuse the model.

❌ No Error Handling: Network issues, rate limits, and API errors will occur in production.

❌ Ignoring Context Limits: Very long conversations can exceed token limits.

❌ Inconsistent Thread Usage: Either use threads consistently or not at all per conversation.

✅ Good Practice:

Python

async def safe_chat(agent, message, thread=None):
    try:
        result = await agent.run(message, thread=thread)
        return result.text
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "I apologize, but I'm having trouble responding right now. Please try again."
Agent Type 2: Function-Calling Agent
Use Case & Overview
Function-Calling Agents bridge AI capabilities with your business logic, APIs, databases, and external services. By defining functions that the agent can invoke, you create agents that can perform actions, retrieve real-time data, and integrate with existing systems. This is one of the most powerful patterns in the framework, enabling agents to move beyond pure text generation to become active participants in your application workflows.

Ideal Scenarios:

E-commerce agents that check inventory and process orders
Booking systems that check availability and make reservations
Customer service bots that query databases for account information
Smart home assistants that control devices
Financial agents that retrieve stock prices and execute trades
DevOps assistants that query system status and trigger deployments
How Function Calling Works
The function-calling flow involves several steps orchestrated by the framework:

text

1. User Query → Agent
2. Agent (AI) decides which function(s) to call and with what parameters
3. Framework executes the function(s) with provided parameters
4. Function results returned to Agent
5. Agent formulates natural language response using function results
6. Response delivered to user
The AI model intelligently determines when to call functions, which ones to call, and what parameters to use based on the user's query and the function descriptions you provide.

Complete Implementation Example
Here's a comprehensive example building a travel assistant with multiple functions:

Python

import asyncio
from typing import Annotated, Literal
from datetime import datetime, timedelta
from pydantic import Field
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# ============================================================================
# Function Definitions (Business Logic)
# ============================================================================

def search_flights(
    origin: Annotated[str, Field(description="Departure airport code (e.g., 'LAX')")],
    destination: Annotated[str, Field(description="Arrival airport code (e.g., 'JFK')")],
    date: Annotated[str, Field(description="Departure date in YYYY-MM-DD format")],
    passengers: Annotated[int, Field(description="Number of passengers", ge=1, le=9)] = 1
) -> str:
    """
    Search for available flights between two airports.
    
    Returns flight options with prices, times, and airlines.
    In a real application, this would query a flight booking API.
    """
    # Simulated flight search (replace with real API call)
    flights = [
        {
            "airline": "SkyWings",
            "flight_number": "SW123",
            "departure": "08:00 AM",
            "arrival": "04:30 PM",
            "price": 299 * passengers,
            "stops": 0
        },
        {
            "airline": "AirConnect",
            "flight_number": "AC456",
            "departure": "11:30 AM",
            "arrival": "08:15 PM",
            "price": 249 * passengers,
            "stops": 1
        }
    ]
    
    result = f"Found {len(flights)} flights from {origin} to {destination} on {date}:\n\n"
    for i, flight in enumerate(flights, 1):
        result += f"{i}. {flight['airline']} {flight['flight_number']}\n"
        result += f"   Departure: {flight['departure']} → Arrival: {flight['arrival']}\n"
        result += f"   Price: ${flight['price']} ({passengers} passenger(s))\n"
        result += f"   Stops: {flight['stops']}\n\n"
    
    return result


def check_hotel_availability(
    city: Annotated[str, Field(description="City name (e.g., 'New York')")],
    check_in: Annotated[str, Field(description="Check-in date (YYYY-MM-DD)")],
    check_out: Annotated[str, Field(description="Check-out date (YYYY-MM-DD)")],
    guests: Annotated[int, Field(description="Number of guests", ge=1)] = 2
) -> str:
    """
    Search for available hotels in a city.
    
    Returns hotel options with prices, ratings, and amenities.
    """
    # Calculate nights
    check_in_date = datetime.strptime(check_in, "%Y-%m-%d")
    check_out_date = datetime.strptime(check_out, "%Y-%m-%d")
    nights = (check_out_date - check_in_date).days
    
    # Simulated hotel search
    hotels = [
        {
            "name": "Grand Plaza Hotel",
            "rating": 4.5,
            "price_per_night": 180,
            "amenities": ["Pool", "Gym", "Free WiFi"]
        },
        {
            "name": "City View Inn",
            "rating": 4.0,
            "price_per_night": 120,
            "amenities": ["Free Breakfast", "WiFi"]
        }
    ]
    
    result = f"Available hotels in {city} ({check_in} to {check_out}, {nights} nights):\n\n"
    for i, hotel in enumerate(hotels, 1):
        total_price = hotel['price_per_night'] * nights
        result += f"{i}. {hotel['name']} ⭐ {hotel['rating']}/5\n"
        result += f"   ${hotel['price_per_night']}/night × {nights} nights = ${total_price}\n"
        result += f"   Amenities: {', '.join(hotel['amenities'])}\n\n"
    
    return result


def get_weather_forecast(
    city: Annotated[str, Field(description="City name")],
    days: Annotated[int, Field(description="Number of days to forecast", ge=1, le=7)] = 3
) -> str:
    """
    Get weather forecast for a city.
    
    Returns temperature, conditions, and precipitation chance.
    """
    # Simulated weather data
    import random
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
    
    result = f"Weather forecast for {city} (next {days} days):\n\n"
    base_date = datetime.now()
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        temp = random.randint(60, 85)
        condition = random.choice(conditions)
        precipitation = random.randint(0, 60)
        
        result += f"{date.strftime('%A, %B %d')}:\n"
        result += f"  {temp}°F, {condition}\n"
        result += f"  Precipitation: {precipitation}%\n\n"
    
    return result


def calculate_trip_budget(
    flights_cost: Annotated[float, Field(description="Total cost of flights")],
    hotel_cost: Annotated[float, Field(description="Total cost of hotel")],
    days: Annotated[int, Field(description="Number of days")],
    daily_food_budget: Annotated[float, Field(description="Daily food budget")] = 50.0,
    daily_activities_budget: Annotated[float, Field(description="Daily activities budget")] = 100.0
) -> str:
    """
    Calculate total trip budget including all expenses.
    """
    food_total = daily_food_budget * days
    activities_total = daily_activities_budget * days
    total = flights_cost + hotel_cost + food_total + activities_total
    
    result = "Trip Budget Breakdown:\n\n"
    result += f"Flights: ${flights_cost:.2f}\n"
    result += f"Hotel: ${hotel_cost:.2f}\n"
    result += f"Food ({days} days × ${daily_food_budget}): ${food_total:.2f}\n"
    result += f"Activities ({days} days × ${daily_activities_budget}): ${activities_total:.2f}\n"
    result += f"─────────────────────\n"
    result += f"Total Budget: ${total:.2f}\n"
    
    return result

# ============================================================================
# Agent Implementation
# ============================================================================

class TravelAssistant:
    """A travel planning agent with multiple function calling capabilities."""
    
    def __init__(self):
        # Define all tools the agent can use
        self.tools = [
            search_flights,
            check_hotel_availability,
            get_weather_forecast,
            calculate_trip_budget
        ]
        
        # Create agent with tools
        self.agent = ChatAgent(
            chat_client=OpenAIChatClient(model_id="gpt-4o"),
            name="TravelAgent",
            instructions="""You are a helpful travel planning assistant.
            
            You can help users:
            - Search for flights between cities
            - Find hotel accommodations
            - Check weather forecasts
            - Calculate trip budgets
            
            When users ask about travel plans, use the available functions to 
            provide accurate, real-time information. Always confirm important 
            details like dates and locations before searching.
            
            Be proactive in suggesting complete travel plans, not just answering 
            individual questions.""",
            tools=self.tools  # Agent-level tools available for all queries
        )
        
        self.thread = self.agent.get_new_thread()
    
    async def chat(self, message: str, stream: bool = True) -> str:
        """Chat with the travel assistant."""
        if stream:
            print("TravelAgent: ", end="", flush=True)
            full_response = ""
            async for chunk in self.agent.run_stream(message, thread=self.thread):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            print()
            return full_response
        else:
            result = await self.agent.run(message, thread=self.thread)
            return result.text


# ============================================================================
# Demo Usage
# ============================================================================

async def demo_agent_level_tools():
    """Demonstrate agent-level tools (tools defined at agent creation)."""
    print("=== Agent-Level Tools Demo ===\n")
    
    assistant = TravelAssistant()
    
    # The agent can intelligently chain multiple function calls
    queries = [
        "I want to plan a trip from LAX to JFK on 2024-03-15, returning 2024-03-20",
        "What's the weather going to be like in New York during my trip?",
        "Can you help me calculate a total budget? I found flights for $598 and hotels for $600"
    ]
    
    for query in queries:
        print(f"User: {query}")
        await assistant.chat(query, stream=True)
        print()


async def demo_run_level_tools():
    """Demonstrate run-level tools (tools provided per query)."""
    print("=== Run-Level Tools Demo ===\n")
    
    # Agent created without tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful assistant."
    )
    
    # Provide specific tools for specific queries
    query1 = "Search for flights from SFO to BOS on 2024-04-01 for 2 passengers"
    print(f"User: {query1}")
    result1 = await agent.run(
        query1,
        tools=[search_flights]  # Only flight search available
    )
    print(f"Agent: {result1}\n")
    
    query2 = "What's the weather forecast for Boston for the next 5 days?"
    print(f"User: {query2}")
    result2 = await agent.run(
        query2,
        tools=[get_weather_forecast]  # Only weather available
    )
    print(f"Agent: {result2}\n")


async def demo_mixed_tools():
    """Demonstrate mixing agent-level and run-level tools."""
    print("=== Mixed Tools Demo ===\n")
    
    # Base agent with some always-available tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a travel assistant.",
        tools=[search_flights, check_hotel_availability]  # Always available
    )
    
    thread = agent.get_new_thread()
    
    # Use base tools
    query1 = "Find me flights from ORD to MIA on 2024-05-10"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1}\n")
    
    # Add additional tools for specific query
    query2 = "Also show me the weather and calculate the budget with $500 flights and $400 hotel for 5 days"
    print(f"User: {query2}")
    result2 = await agent.run(
        query2,
        thread=thread,
        tools=[get_weather_forecast, calculate_trip_budget]  # Additional tools
    )
    print(f"Agent: {result2}\n")


async def main():
    print("=== Function-Calling Agent Comprehensive Demo ===\n")
    
    await demo_agent_level_tools()
    await demo_run_level_tools()
    await demo_mixed_tools()


if __name__ == "__main__":
    asyncio.run(main())
Function Definition Best Practices
1. Clear Descriptions: Use comprehensive docstrings and Field descriptions:

Python

def book_appointment(
    patient_id: Annotated[str, Field(
        description="Unique patient identifier from the database"
    )],
    doctor_id: Annotated[str, Field(
        description="Doctor's ID. Use 'search_doctors' function to find available doctors"
    )],
    date: Annotated[str, Field(
        description="Appointment date in YYYY-MM-DD format"
    )],
    time: Annotated[str, Field(
        description="Appointment time in HH:MM format (24-hour)"
    )],
    reason: Annotated[str, Field(
        description="Brief reason for visit (e.g., 'annual checkup', 'flu symptoms')"
    )]
) -> str:
    """
    Book a medical appointment for a patient.
    
    This function creates a new appointment in the system after validating
    doctor availability and patient eligibility. Returns confirmation details
    or error message if booking fails.
    
    Important: Always verify the date is in the future and during business hours.
    """
    pass
2. Type Safety: Use proper type annotations and validation:

Python

from typing import Literal
from pydantic import Field, validator

def set_thermostat(
    temperature: Annotated[int, Field(
        description="Target temperature in Fahrenheit",
        ge=60,  # Minimum 60°F
        le=85   # Maximum 85°F
    )],
    mode: Annotated[Literal["heat", "cool", "auto"], Field(
        description="Thermostat mode"
    )] = "auto"
) -> str:
    """Set thermostat temperature and mode."""
    return f"Thermostat set to {temperature}°F in {mode} mode"
3. Error Handling in Functions:

Python

def process_refund(
    order_id: Annotated[str, Field(description="Order ID to refund")],
    amount: Annotated[float, Field(description="Refund amount")]
) -> str:
    """Process a customer refund."""
    try:
        # Validate order exists
        order = get_order(order_id)
        if not order:
            return f"Error: Order {order_id} not found"
        
        # Validate refund amount
        if amount > order['total']:
            return f"Error: Refund amount ${amount} exceeds order total ${order['total']}"
        
        # Process refund (actual business logic)
        refund_id = process_payment_refund(order_id, amount)
        
        return f"Refund processed successfully. Refund ID: {refund_id}. Amount: ${amount}"
        
    except Exception as e:
        # Log error for debugging
        logger.error(f"Refund processing error: {e}")
        return f"Error processing refund: {str(e)}"
Configuration Options
Tool Choice Control:

Python

# Let the agent decide when to use tools (default)
result = await agent.run(query, tool_choice="auto")

# Force the agent to use a specific tool
result = await agent.run(
    query,
    tool_choice={"type": "function", "function": {"name": "search_flights"}}
)

# Prevent tool usage (text-only response)
result = await agent.run(query, tool_choice="none")
Common Pitfalls
❌ Insufficient Function Descriptions: The AI relies on descriptions to understand when and how to call functions.

❌ No Error Handling in Functions: Functions should handle errors gracefully and return meaningful messages.

❌ Functions That Take Too Long: Long-running operations can cause timeouts.

❌ Side Effects Without Confirmation: Destructive operations should require explicit confirmation.

✅ Best Practice: For destructive operations, use a two-step pattern:

Python

# Step 1: Preview
def preview_delete_account(account_id: str) -> str:
    """Preview what will be deleted."""
    return f"This will delete account {account_id} and all associated data..."

# Step 2: Execute (requires explicit parameter)
def delete_account(
    account_id: str,
    confirmed: Annotated[bool, Field(
        description="Must be True to proceed with deletion"
    )] = False
) -> str:
    """Delete account after confirmation."""
    if not confirmed:
        return "Deletion not confirmed. Use preview_delete_account first."
    # Proceed with deletion
    pass
Agent Type 3: RAG Agent (Retrieval-Augmented Generation)
Use Case & Overview
RAG (Retrieval-Augmented Generation) Agents combine the power of AI language models with your organization's specific knowledge base, documents, and data. By uploading files to a vector store, the agent can search through your documents to find relevant information and use it to answer questions accurately. This is essential for building agents that need to work with company-specific information, technical documentation, legal documents, or any domain-specific knowledge.

Ideal Scenarios:

Customer support bots answering from product documentation
Legal assistants searching through case law and contracts
HR assistants providing policy information
Technical support using troubleshooting guides
Research assistants working with academic papers
Compliance bots answering regulatory questions
Key Advantages:

Answers grounded in your specific documents
Reduces hallucinations by using factual sources
Automatically cites sources
Handles large document collections
Updates easily by adding new documents
Architecture & Components
A RAG agent involves several components working together:

text

User Query
    ↓
Agent with File Search Tool
    ↓
Vector Store Search → Relevant Document Chunks Retrieved
    ↓
Chunks + Query → AI Model → Response with Citations
    ↓
User receives answer with source references
Components:

Vector Store: Stores document embeddings for semantic search
File Search Tool: Enables the agent to search the vector store
Documents: Your knowledge base (PDFs, text files, markdown, etc.)
Agent: Orchestrates search and response generation
Complete Implementation Example
Here's a comprehensive example building a technical support agent with document search capabilities:

Python

import asyncio
from typing import List, Optional
from pathlib import Path
from agent_framework import (
    ChatAgent,
    HostedFileSearchTool,
    HostedVectorStoreContent
)
from agent_framework.openai import OpenAIResponsesClient

class DocumentKnowledgeBase:
    """Manages vector store creation and document uploads."""
    
    def __init__(self, client: OpenAIResponsesClient):
        self.client = client
        self.vector_store_id: Optional[str] = None
        self.file_ids: List[str] = []
    
    async def create_from_texts(
        self,
        documents: List[tuple[str, str]],
        name: str = "knowledge_base"
    ) -> HostedVectorStoreContent:
        """
        Create a vector store from text documents.
        
        Args:
            documents: List of (filename, content) tuples
            name: Name for the vector store
            
        Returns:
            HostedVectorStoreContent for use with file search tool
        """
        # Upload files
        for filename, content in documents:
            file = await self.client.client.files.create(
                file=(filename, content.encode('utf-8')),
                purpose="user_data"
            )
            self.file_ids.append(file.id)
            print(f"✓ Uploaded: {filename} (ID: {file.id})")
        
        # Create vector store
        vector_store = await self.client.client.vector_stores.create(
            name=name,
            expires_after={"anchor": "last_active_at", "days": 7}
        )
        self.vector_store_id = vector_store.id
        print(f"✓ Created vector store: {name} (ID: {vector_store.id})")
        
        # Add files to vector store and wait for processing
        for file_id in self.file_ids:
            result = await self.client.client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
            
            if result.last_error:
                raise Exception(f"File processing failed: {result.last_error.message}")
            
            print(f"✓ Processed file {file_id} in vector store")
        
        return HostedVectorStoreContent(vector_store_id=vector_store.id)
    
    async def create_from_files(
        self,
        file_paths: List[Path],
        name: str = "knowledge_base"
    ) -> HostedVectorStoreContent:
        """
        Create a vector store from file paths.
        
        Args:
            file_paths: List of Path objects to upload
            name: Name for the vector store
        """
        documents = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                documents.append((path.name, f.read()))
        
        return await self.create_from_texts(documents, name)
    
    async def cleanup(self):
        """Clean up vector store and files."""
        if self.vector_store_id:
            await self.client.client.vector_stores.delete(self.vector_store_id)
            print(f"✓ Deleted vector store: {self.vector_store_id}")
        
        for file_id in self.file_ids:
            await self.client.client.files.delete(file_id)
            print(f"✓ Deleted file: {file_id}")


class RAGAgent:
    """A RAG-enabled agent that searches documents to answer questions."""
    
    def __init__(
        self,
        client: OpenAIResponsesClient,
        vector_store: HostedVectorStoreContent,
        instructions: str
    ):
        self.client = client
        self.vector_store = vector_store
        
        # Create agent with file search capability
        self.agent = ChatAgent(
            chat_client=client,
            instructions=instructions,
            tools=[HostedFileSearchTool()]
        )
        
        self.thread = self.agent.get_new_thread()
    
    async def ask(
        self,
        question: str,
        stream: bool = False
    ) -> str:
        """
        Ask a question and get an answer based on the knowledge base.
        
        Args:
            question: User's question
            stream: Whether to stream the response
        """
        # Provide vector store in tool_resources
        tool_resources = {
            "file_search": {
                "vector_store_ids": [self.vector_store.vector_store_id]
            }
        }
        
        if stream:
            return await self._ask_streaming(question, tool_resources)
        else:
            return await self._ask_complete(question, tool_resources)
    
    async def _ask_complete(self, question: str, tool_resources: dict) -> str:
        """Get complete response."""
        result = await self.agent.run(
            question,
            thread=self.thread,
            tool_resources=tool_resources
        )
        return result.text
    
    async def _ask_streaming(self, question: str, tool_resources: dict) -> str:
        """Stream response in real-time."""
        print("Agent: ", end="", flush=True)
        full_response = ""
        
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


# ============================================================================
# Sample Knowledge Base Documents
# ============================================================================

SAMPLE_DOCS = [
    ("product_overview.txt", """
TechCorp CloudSync Pro - Product Overview

CloudSync Pro is our flagship cloud storage and synchronization solution designed 
for businesses of all sizes. 

KEY FEATURES:
- 10TB storage per user
- Real-time file synchronization across all devices
- Advanced sharing and collaboration tools
- Enterprise-grade encryption (AES-256)
- Version history with unlimited retention
- Mobile apps for iOS and Android
- Desktop clients for Windows, macOS, and Linux

PRICING:
- Starter Plan: $10/user/month (1TB storage)
- Professional Plan: $25/user/month (5TB storage)
- Enterprise Plan: $50/user/month (10TB storage, priority support)

SYSTEM REQUIREMENTS:
- Internet connection (minimum 5 Mbps recommended)
- Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- Minimum 4GB RAM
- 500MB free disk space for client application
"""),

    ("troubleshooting.txt", """
CloudSync Pro - Common Issues and Solutions

SYNC ISSUES:

Problem: Files not syncing
Solution:
1. Check internet connection
2. Verify you're logged in to the correct account
3. Check if file size exceeds plan limits
4. Restart the CloudSync Pro application
5. Check firewall settings - CloudSync requires ports 443 and 8443

Problem: Slow sync speeds
Solution:
1. Close other applications using bandwidth
2. Check if bandwidth throttling is enabled in Settings > Network
3. Verify your internet speed meets minimum requirements (5 Mbps)
4. Contact support if issue persists

LOGIN ISSUES:

Problem: Cannot log in
Solution:
1. Verify email and password are correct
2. Check for account suspension notification in email
3. Try password reset at cloudsync.com/reset
4. Clear browser cache if using web interface
5. Ensure account is active (check billing status)

STORAGE ISSUES:

Problem: "Storage full" error
Solution:
1. Check current usage in Settings > Account
2. Delete unnecessary files or purchase additional storage
3. Check if shared folders count against your quota
4. Empty trash to permanently delete files

For issues not covered here, contact support@techcorp.com or call 1-800-TECH-CORP
"""),

    ("api_documentation.txt", """
CloudSync Pro API Documentation

AUTHENTICATION:
All API requests require an API key passed in the X-API-Key header.

Generate API keys in your dashboard: Settings > Developer > API Keys

BASE URL: https://api.cloudsync.com/v1

ENDPOINTS:

GET /files
List all files in account
Parameters:
  - folder_id (optional): Filter by folder
  - limit (optional): Number of results (default: 100)
  - offset (optional): Pagination offset
Response: JSON array of file objects

POST /files/upload
Upload a new file
Headers:
  - Content-Type: multipart/form-data
Body:
  - file: File to upload
  - folder_id (optional): Destination folder
Response: File object with ID and metadata

GET /files/{file_id}
Get file metadata
Response: File object

DELETE /files/{file_id}
Delete a file
Response: 204 No Content

POST /sharing/create
Create a share link
Body:
  - file_id: File to share
  - expiration (optional): Link expiration date
  - password (optional): Password protection
Response: Share link object

RATE LIMITS:
- 1000 requests per hour for Starter plan
- 5000 requests per hour for Professional plan
- 20000 requests per hour for Enterprise plan

ERROR CODES:
- 400: Bad Request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 403: Forbidden (insufficient permissions)
- 404: Not Found
- 429: Rate Limit Exceeded
- 500: Internal Server Error
"""),

    ("security_policies.txt", """
CloudSync Pro - Security and Compliance

DATA ENCRYPTION:
- All data encrypted in transit using TLS 1.3
- All data encrypted at rest using AES-256
- Zero-knowledge encryption option available for Enterprise plan
- Encryption keys rotated every 90 days

ACCESS CONTROLS:
- Two-factor authentication (2FA) available
- Single Sign-On (SSO) supported for Enterprise
- Role-based access control (RBAC)
- IP whitelisting for Enterprise plan
- Session timeout after 12 hours of inactivity

COMPLIANCE:
- SOC 2 Type II certified
- GDPR compliant
- HIPAA compliant (Enterprise plan with BAA)
- ISO 27001 certified

DATA RETENTION:
- Deleted files retained in trash for 30 days
- Version history: Unlimited for Enterprise, 90 days for Professional
- Account data deleted 30 days after account closure
- Backups retained for 90 days

AUDIT LOGS:
- All file access logged (Enterprise plan)
- Admin actions logged
- API usage logged
- Logs retained for 1 year
- Logs available for export in JSON format

DATA LOCATION:
- US customers: Data stored in US data centers (AWS us-east-1)
- EU customers: Data stored in EU data centers (AWS eu-west-1)
- Data residency guarantees available for Enterprise

For security inquiries: security@techcorp.com
To report vulnerabilities: security-reports@techcorp.com
""")
]


# ============================================================================
# Demo Usage
# ============================================================================

async def demo_rag_agent():
    """Demonstrate RAG agent with comprehensive knowledge base."""
    print("=== RAG Agent Demo: Technical Support Assistant ===\n")
    
    # Initialize client
    client = OpenAIResponsesClient()
    
    # Create knowledge base
    print("📚 Setting up knowledge base...")
    kb = DocumentKnowledgeBase(client)
    vector_store = await kb.create_from_texts(
        SAMPLE_DOCS,
        name="TechCorp Support Docs"
    )
    print()
    
    try:
        # Create RAG agent
        agent = RAGAgent(
            client=client,
            vector_store=vector_store,
            instructions="""You are a technical support agent for CloudSync Pro.
            
            Use the file search tool to find relevant information from our 
            documentation to answer customer questions accurately. Always cite 
            the specific documents you reference.
            
            If the information isn't in the documentation, say so clearly rather 
            than making up information. For issues not covered in the docs, 
            advise customers to contact support@techcorp.com.
            
            Be professional, helpful, and concise."""
        )
        
        # Example questions demonstrating RAG capabilities
        questions = [
            "What are the storage limits for different plans?",
            "My files aren't syncing. What should I check?",
            "How do I authenticate with the CloudSync API?",
            "Is CloudSync HIPAA compliant?",
            "What ports does CloudSync use?"
        ]
        
        for question in questions:
            print(f"💬 Customer: {question}")
            await agent.ask(question, stream=True)
            print()
        
    finally:
        # Cleanup resources
        print("🧹 Cleaning up...")
        await kb.cleanup()


async def demo_rag_with_citations():
    """Demonstrate RAG with explicit citation tracking."""
    print("=== RAG with Citation Tracking ===\n")
    
    client = OpenAIResponsesClient()
    kb = DocumentKnowledgeBase(client)
    vector_store = await kb.create_from_texts(
        SAMPLE_DOCS,
        name="Support Docs with Citations"
    )
    
    try:
        agent = RAGAgent(
            client=client,
            vector_store=vector_store,
            instructions="""You are a support agent. When answering questions,
            always explicitly cite which document you found the information in.
            Format citations like [Source: filename.txt]"""
        )
        
        question = "What's the pricing for the Enterprise plan?"
        print(f"Question: {question}\n")
        
        answer = await agent.ask(question, stream=False)
        print(f"Answer: {answer}\n")
        
    finally:
        await kb.cleanup()


async def main():
    print("=== RAG Agent Comprehensive Demo ===\n")
    
    await demo_rag_agent()
    print("\n" + "="*60 + "\n")
    await demo_rag_with_citations()


if __name__ == "__main__":
    asyncio.run(main())
Vector Store Management
Creating from Different Sources:

Python

# From text strings
documents = [
    ("doc1.txt", "Content of document 1..."),
    ("doc2.txt", "Content of document 2...")
]
vector_store = await kb.create_from_texts(documents)

# From files on disk
from pathlib import Path
file_paths = [
    Path("docs/manual.pdf"),
    Path("docs/faq.md"),
    Path("docs/policies.txt")
]
vector_store = await kb.create_from_files(file_paths)

# Supported file types: .txt, .md, .pdf, .docx, .html, and more
Updating Vector Stores:

Python

# Add new file to existing vector store
new_file = await client.client.files.create(
    file=("new_doc.txt", b"New content..."),
    purpose="user_data"
)

await client.client.vector_stores.files.create_and_poll(
    vector_store_id=vector_store_id,
    file_id=new_file.id
)
Best Practices
1. Document Preparation: Optimize documents for better search results:

Python

# Good: Well-structured with clear sections
good_doc = """
# Database Connection Issues

## Symptoms
- Application cannot connect to database
- Timeout errors in logs

## Solution
1. Check database server is running
2. Verify connection string in config.yaml
3. Ensure firewall allows port 5432
"""

# Less effective: Unstructured wall of text
bad_doc = "If you can't connect to the database check if it's running..."
2. Meaningful Filenames: Use descriptive names that help with retrieval:

Python

# Good
("troubleshooting_database_connection.txt", content)
("api_authentication_guide.txt", content)

# Less helpful
("doc1.txt", content)
("temp.txt", content)
3. Chunk Size Awareness: Keep documents focused - very large documents should be split:

Python

# Instead of one massive document
# Split into focused documents
docs = [
    ("product_features_storage.txt", storage_content),
    ("product_features_sync.txt", sync_content),
    ("product_features_sharing.txt", sharing_content)
]
4. Expiration Management: Set appropriate expiration for vector stores:

Python

# Short-lived for testing
vector_store = await client.client.vector_stores.create(
    name="test_store",
    expires_after={"anchor": "last_active_at", "days": 1}
)

# Longer retention for production
vector_store = await client.client.vector_stores.create(
    name="prod_knowledge_base",
    expires_after={"anchor": "last_active_at", "days": 30}
)
Common Pitfalls
❌ Not Waiting for Processing: Files must finish processing before search works:

Python

# Wrong: May search before files are indexed
result = await client.vector_stores.files.create(...)
# Immediately try to search - might fail

# Correct: Wait for processing
result = await client.vector_stores.files.create_and_poll(...)
if result.last_error:
    raise Exception(f"Processing failed: {result.last_error}")
# Now safe to search
❌ Forgetting Cleanup: Always clean up resources:

Python

# Use try/finally
try:
    # Create and use vector store
    pass
finally:
    await kb.cleanup()

# Or context managers (if available)
❌ Poor Quality Documents: Garbage in, garbage out:

Python

# Bad: No structure, hard to search
"stuff about things and other stuff..."

# Good: Clear, structured, specific
"Product Name: CloudSync\nFeature: File Sharing\nHow to share a file:..."
Agent Type 4: Code Execution Agent
Use Case & Overview
Code Execution Agents can write and run Python code dynamically to solve problems, perform calculations, analyze data, create visualizations, and more. This capability transforms your agent from a text-based assistant into a powerful computational tool that can handle complex mathematical problems, data analysis tasks, and algorithmic challenges that would be difficult to solve through natural language alone.

Ideal Scenarios:

Mathematical problem solving (complex calculations, proofs)
Data analysis and statistics
Creating plots and visualizations
File format conversions
Algorithm implementation and testing
Scientific computing
Financial calculations and modeling
Educational tools for teaching programming
Key Capabilities:

Executes Python code in a secure sandbox
Access to common libraries (NumPy, Pandas, Matplotlib, etc.)
Can generate and save files (images, CSVs, etc.)
Handles iterative problem-solving
Shows step-by-step reasoning through code
Architecture & Components
The code execution flow involves:

text

User Query → Agent decides to write code → Code generated
    ↓
Code sent to secure execution environment
    ↓
Code executed (with timeout and resource limits)
    ↓
Results (output, files, errors) returned to agent
    ↓
Agent interprets results and formulates response
    ↓
Response delivered to user
Security Features:

Isolated execution environment
No network access from code
Resource limits (CPU, memory, time)
No access to host filesystem
Limited to safe libraries
Complete Implementation Example
Python

import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool, ChatResponse
from agent_framework.openai import OpenAIResponsesClient
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall
)

class CodeExecutionAgent:
    """Agent capable of writing and executing Python code."""
    
    def __init__(self, model_id: str = "gpt-4o"):
        self.client = OpenAIResponsesClient(model_id=model_id)
        
        # Create agent with code interpreter capability
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions="""You are a helpful assistant that can write and execute 
            Python code to solve problems.
            
            When faced with mathematical, data analysis, or computational problems:
            1. Explain your approach
            2. Write clear, well-commented Python code
            3. Execute the code to get results
            4. Interpret and explain the results
            
            Available libraries include: numpy, pandas, matplotlib, scipy, and more.
            
            Always show your code and explain what it does.""",
            tools=[HostedCodeInterpreterTool()]
        )
        
        self.thread = self.agent.get_new_thread()
    
    async def solve(
        self,
        problem: str,
        stream: bool = False,
        show_code: bool = True
    ) -> tuple[str, str]:
        """
        Solve a problem using code execution.
        
        Args:
            problem: Problem description
            stream: Whether to stream the response
            show_code: Whether to print generated code
            
        Returns:
            Tuple of (response_text, generated_code)
        """
        if stream:
            return await self._solve_streaming(problem, show_code)
        else:
            return await self._solve_complete(problem, show_code)
    
    async def _solve_complete(
        self,
        problem: str,
        show_code: bool
    ) -> tuple[str, str]:
        """Get complete solution."""
        result = await self.agent.run(problem, thread=self.thread)
        
        # Extract generated code from response
        code = self._extract_code(result)
        
        if show_code and code:
            print("\n📝 Generated Code:")
            print("─" * 60)
            print(code)
            print("─" * 60)
        
        return result.text, code
    
    async def _solve_streaming(
        self,
        problem: str,
        show_code: bool
    ) -> tuple[str, str]:
        """Stream solution in real-time."""
        print("Agent: ", end="", flush=True)
        full_response = ""
        code_shown = False
        
        async for chunk in self.agent.run_stream(problem, thread=self.thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            
            # Show code as it's being generated
            if show_code and not code_shown:
                code = self._extract_code_from_chunk(chunk)
                if code:
                    print(f"\n\n📝 Generated Code:\n{code}", end="", flush=True)
                    code_shown = True
        
        print("\n")
        
        # Get final code
        final_code = ""
        # You'd extract this from the full response or stored state
        
        return full_response, final_code
    
    def _extract_code(self, result) -> str:
        """Extract generated code from result."""
        if (
            isinstance(result.raw_representation, ChatResponse)
            and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
            and len(result.raw_representation.raw_representation.output) > 0
            and isinstance(
                result.raw_representation.raw_representation.output[0],
                ResponseCodeInterpreterToolCall
            )
        ):
            return result.raw_representation.raw_representation.output[0].code
        return ""
    
    def _extract_code_from_chunk(self, chunk) -> str:
        """Extract code from streaming chunk."""
        # Implementation would depend on chunk structure
        # This is a simplified version
        return ""


# ============================================================================
# Demo Usage
# ============================================================================

async def demo_math_problems():
    """Demonstrate solving mathematical problems."""
    print("=== Code Execution Agent: Math Problems ===\n")
    
    agent = CodeExecutionAgent()
    
    problems = [
        "Calculate the factorial of 100. What's the result?",
        
        "Generate the first 20 Fibonacci numbers and show me the sequence.",
        
        """Solve this system of equations:
        2x + 3y = 13
        x - y = -1""",
        
        "What is the sum of all prime numbers between 1 and 100?"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*60}")
        print(f"Problem {i}: {problem}")
        print('='*60)
        
        response, code = await agent.solve(problem, stream=False, show_code=True)
        print(f"\n✅ Answer:\n{response}\n")


async def demo_data_analysis():
    """Demonstrate data analysis capabilities."""
    print("=== Code Execution Agent: Data Analysis ===\n")
    
    agent = CodeExecutionAgent()
    
    problem = """I have the following sales data for the past week:
    Monday: $1200, Tuesday: $1450, Wednesday: $980, Thursday: $1670, 
    Friday: $2100, Saturday: $2400, Sunday: $1890
    
    Please:
    1. Calculate the total sales
    2. Find the average daily sales
    3. Identify the best and worst days
    4. Calculate the percentage difference between the best and worst days"""
    
    print(f"Problem:\n{problem}\n")
    response, code = await agent.solve(problem, stream=False, show_code=True)
    print(f"\n✅ Analysis:\n{response}\n")


async def demo_statistics():
    """Demonstrate statistical calculations."""
    print("=== Code Execution Agent: Statistics ===\n")
    
    agent = CodeExecutionAgent()
    
    problem = """I measured the heights (in cm) of 15 students:
    165, 170, 168, 172, 169, 171, 167, 173, 170, 168, 169, 171, 172, 170, 169
    
    Calculate:
    1. Mean, median, and mode
    2. Standard deviation
    3. Create a simple frequency distribution"""
    
    print(f"Problem:\n{problem}\n")
    response, code = await agent.solve(problem, stream=False, show_code=True)
    print(f"\n✅ Results:\n{response}\n")


async def demo_algorithm_implementation():
    """Demonstrate algorithm implementation."""
    print("=== Code Execution Agent: Algorithm Implementation ===\n")
    
    agent = CodeExecutionAgent()
    
    problem = """Implement the Sieve of Eratosthenes algorithm to find all 
    prime numbers up to 200. Show me the algorithm in action and list all the primes found."""
    
    print(f"Problem:\n{problem}\n")
    response, code = await agent.solve(problem, stream=False, show_code=True)
    print(f"\n✅ Results:\n{response}\n")


async def demo_conversational_coding():
    """Demonstrate iterative problem solving with code."""
    print("=== Code Execution Agent: Conversational Coding ===\n")
    
    agent = CodeExecutionAgent()
    
    # Multi-turn conversation with context
    exchanges = [
        "Create a list of the first 10 perfect squares.",
        "Now filter that list to only include even numbers.",
        "Calculate the sum of those even perfect squares."
    ]
    
    for query in exchanges:
        print(f"\n💬 User: {query}")
        response, code = await agent.solve(query, stream=False, show_code=True)
        print(f"✅ Agent: {response}\n")


async def main():
    print("=== Code Execution Agent Comprehensive Demo ===\n")
    
    await demo_math_problems()
    print("\n" + "="*70 + "\n")
    
    await demo_data_analysis()
    print("\n" + "="*70 + "\n")
    
    await demo_statistics()
    print("\n" + "="*70 + "\n")
    
    await demo_algorithm_implementation()
    print("\n" + "="*70 + "\n")
    
    await demo_conversational_coding()


if __name__ == "__main__":
    asyncio.run(main())
Working with Code Output
Accessing Generated Code:

Python

# The code interpreter stores generated code in the response
result = await agent.run("Calculate factorial of 50")

# Extract code from response
if (
    isinstance(result.raw_representation, ChatResponse)
    and isinstance(result.raw_representation.raw_representation, OpenAIResponse)
):
    for output in result.raw_representation.raw_representation.output:
        if isinstance(output, ResponseCodeInterpreterToolCall):
            generated_code = output.code
            print(f"Code:\n{generated_code}")
Handling Files Generated by Code:

Python

# Agent can generate files (plots, CSVs, etc.)
problem = """Create a bar chart showing sales data:
Product A: 100, Product B: 150, Product C: 75
Save it as a PNG file."""

result = await agent.run(problem)

# Files are available in the response
# The exact structure depends on the client implementation
Best Practices
1. Clear Problem Descriptions:

Python

# Good: Specific requirements
"Calculate the compound interest on $10,000 at 5% annual rate for 10 years, compounded monthly"

# Less effective: Vague
"Do some interest calculations"
2. Specify Output Format:

Python

# Good: Clear expectations
"Find all prime numbers under 100 and format the output as a comma-separated list"

# Less effective: No format guidance
"Find all prime numbers under 100"
3. Encourage Explanation:

Python

instructions = """When solving problems with code:
1. Explain your approach before coding
2. Write clean, commented code
3. Show the output
4. Explain the results in plain language"""
Common Pitfalls
❌ Expecting Network Access: Code cannot make HTTP requests or access external resources.

❌ Very Long Computations: There are timeout limits for code execution.

❌ File System Access: Code cannot read/write to arbitrary locations.

✅ Best Practices:

Python

# Instead of: "Download this CSV from URL and analyze it"
# Do this: "Here's the CSV data: [paste data]. Please analyze it."

# Instead of: Long-running computation
# Break it down: "Let's solve this step by step, starting with..."
Agent Type 5: Multi-Modal Agent
Use Case & Overview
Multi-Modal Agents represent the cutting edge of AI capabilities, combining multiple modalities and advanced features into a single, powerful agent. These agents can process and generate text, analyze images, search the web for current information, reason through complex problems step-by-step, and integrate with external services through the Model Context Protocol (MCP). This makes them ideal for sophisticated applications requiring comprehensive AI capabilities.

Ideal Scenarios:

Content creation combining text and images
Visual question answering and analysis
Research assistants with real-time information access
Complex problem-solving requiring deep reasoning
Applications needing external service integration
Multi-capability chatbots for diverse tasks
Educational applications with visual learning
E-commerce with product image analysis
Key Capabilities:

Image understanding and analysis
Web search for current information
Advanced reasoning (GPT-5 models)
MCP integration for external tools
Structured outputs with Pydantic
Code execution and data analysis
File search and document understanding
Function calling for custom logic
Architecture & Components
A multi-modal agent orchestrates various capabilities:

text

User Query (text + optional images)
    ↓
Agent analyzes query and determines needed capabilities
    ↓
    ├→ Image Analysis (if images present)
    ├→ Web Search (if current info needed)
    ├→ Code Execution (if computation needed)
    ├→ File Search (if document lookup needed)
    ├→ MCP Tools (if external service needed)
    ├→ Function Calling (if custom logic needed)
    └→ Reasoning (if complex problem)
    ↓
Results integrated and formatted
    ↓
Response delivered (text + optional structured data)
Complete Implementation Example
Python

import asyncio
from typing import Optional
from pathlib import Path
from agent_framework import (
    ChatAgent,
    ChatMessage,
    TextContent,
    UriContent,
    HostedWebSearchTool,
    HostedFileSearchTool,
    HostedCodeInterpreterTool,
    HostedVectorStoreContent,
    MCPStreamableHTTPTool
)
from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

class MultiModalAgent:
    """
    A comprehensive multi-modal agent with multiple capabilities.
    """
    
    def __init__(
        self,
        enable_web_search: bool = True,
        enable_code_execution: bool = True,
        enable_file_search: bool = False,
        enable_mcp: bool = False,
        enable_reasoning: bool = False,
        vector_store: Optional[HostedVectorStoreContent] = None,
        mcp_url: Optional[str] = None
    ):
        """
        Initialize multi-modal agent with desired capabilities.
        
        Args:
            enable_web_search: Enable real-time web search
            enable_code_execution: Enable Python code execution
            enable_file_search: Enable document search
            enable_mcp: Enable MCP tool integration
            enable_reasoning: Enable advanced reasoning (requires GPT-5)
            vector_store: Vector store for file search
            mcp_url: MCP server URL
        """
        # Select model based on requirements
        model_id = "gpt-5" if enable_reasoning else "gpt-4o"
        
        # Configure client
        client_options = {}
        if enable_reasoning:
            client_options["additional_chat_options"] = {
                "reasoning": {
                    "effort": "high",
                    "summary": "detailed"
                }
            }
        
        self.client = OpenAIResponsesClient(
            model_id=model_id,
            **client_options
        )
        
        # Build tools list
        self.tools = []
        
        if enable_web_search:
            self.tools.append(HostedWebSearchTool())
        
        if enable_code_execution:
            self.tools.append(HostedCodeInterpreterTool())
        
        if enable_file_search and vector_store:
            self.tools.append(HostedFileSearchTool())
        
        if enable_mcp and mcp_url:
            self.tools.append(MCPStreamableHTTPTool(
                name="External Services",
                url=mcp_url
            ))
        
        self.vector_store = vector_store
        self.enable_reasoning = enable_reasoning
        
        # Create agent
        self.agent = self._create_agent()
        self.thread = self.agent.get_new_thread()
    
    def _create_agent(self) -> ChatAgent:
        """Create agent with comprehensive instructions."""
        instructions = """You are an advanced AI assistant with multiple capabilities:
        
        CAPABILITIES:
        """
        
        if any(isinstance(t, HostedWebSearchTool) for t in self.tools):
            instructions += "\n- Web Search: Access current, real-time information from the internet"
        
        if any(isinstance(t, HostedCodeInterpreterTool) for t in self.tools):
            instructions += "\n- Code Execution: Write and run Python code for calculations and data analysis"
        
        if any(isinstance(t, HostedFileSearchTool) for t in self.tools):
            instructions += "\n- Document Search: Search through uploaded documents for information"
        
        if any(isinstance(t, MCPStreamableHTTPTool) for t in self.tools):
            instructions += "\n- External Services: Access external tools and services via MCP"
        
        instructions += """
        
        - Image Analysis: Understand and describe images provided by users
        """
        
        if self.enable_reasoning:
            instructions += "\n- Advanced Reasoning: Think through complex problems step-by-step"
        
        instructions += """
        
        GUIDELINES:
        1. Choose the right tool for each task
        2. Combine multiple capabilities when needed
        3. Explain your approach before taking action
        4. Provide clear, well-formatted responses
        5. Cite sources when using web search or documents
        6. Show your work when performing calculations
        """
        
        if self.enable_reasoning:
            instructions += """
        7. Use detailed reasoning for complex problems
        8. Break down multi-step problems clearly
            """
        
        return ChatAgent(
            chat_client=self.client,
            name="MultiModalAssistant",
            instructions=instructions,
            tools=self.tools if self.tools else None
        )
    
    async def chat(
        self,
        message: str,
        image_url: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Chat with the agent, optionally including an image.
        
        Args:
            message: Text message
            image_url: Optional URL to an image
            stream: Whether to stream the response
        """
        # Build message with optional image
        if image_url:
            chat_message = ChatMessage(
                role="user",
                contents=[
                    TextContent(text=message),
                    UriContent(uri=image_url, media_type="image/jpeg")
                ]
            )
        else:
            chat_message = message
        
        # Prepare tool resources if needed
        tool_resources = None
        if self.vector_store:
            
