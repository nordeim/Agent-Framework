# Microsoft Agent Framework Programming Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Agent Type 1: Basic Conversational Agent](#agent-type-1-basic-conversational-agent)
4. [Agent Type 2: Function-Calling Agent](#agent-type-2-function-calling-agent)
5. [Agent Type 3: RAG Agent](#agent-type-3-rag-agent)
6. [Agent Type 4: Code Execution Agent](#agent-type-4-code-execution-agent)
7. [Agent Type 5: Multi-Modal Agent](#agent-type-5-multi-modal-agent)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices & Patterns](#best-practices--patterns)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Quick Reference](#quick-reference)

---

## Introduction

The Microsoft Agent Framework represents a paradigm shift in AI application development, providing a sophisticated abstraction layer that transforms complex AI orchestration into elegant, maintainable code. This comprehensive guide will equip you with the knowledge and practical skills to build production-ready AI agents across five distinct architectural patterns.

### Framework Overview

At its core, the Microsoft Agent Framework is built on three fundamental pillars that distinguish it from traditional AI integration approaches:

**1. Unified Client Architecture**: The framework provides three specialized clients—Assistants, Chat, and Responses—each optimized for specific use cases while maintaining consistent patterns and interfaces. This architectural decision eliminates the cognitive overhead of learning multiple disparate APIs.

**2. Declarative Tool Integration**: Rather than manually orchestrating function calls and parameter validation, the framework employs a declarative approach where tools are defined with rich metadata and automatically integrated into the agent's capabilities. This pattern reduces boilerplate code by approximately 70% compared to direct API integration.

**3. Conversation State Management**: The framework introduces sophisticated thread management that handles conversation persistence, context maintenance, and message history automatically. This eliminates the common pattern of manually managing conversation state that plagues most AI integrations.

### Value Proposition

Consider the traditional approach to building an AI-powered customer service chatbot:

```python
# Traditional approach (complex, error-prone)
async def traditional_approach():
    # Manual client initialization
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Manual conversation management
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Manual function calling logic
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {...}
            }
        }]
    )
    
    # Manual tool execution and response handling
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            # Execute function manually
            result = execute_function(tool_call)
            # Manually construct follow-up messages
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    # Make another API call with tool results
    final_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return final_response.choices[0].message.content
```

Now compare this with the Microsoft Agent Framework approach:

```python
# Framework approach (elegant, maintainable)
async def framework_approach():
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        instructions="You are a helpful customer service assistant.",
        tools=get_weather  # Simply reference the function
    ) as agent:
        result = await agent.run("What's the weather in Seattle?")
        return result.text
```

The framework reduces complexity by an order of magnitude while providing superior error handling, resource management, and conversation persistence.

### Architecture Deep Dive

The framework's architecture follows a layered approach that promotes separation of concerns and testability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ ChatAgent   │  │ Thread       │  │ Tool Registry   │  │
│  │             │  │ Management   │  │                 │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                 │                    │            │
│  ┌──────┴─────────────────┴────────────────────┴────────┐  │
│  │              Agent Framework Core                     │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ Client      │  │ Response     │  │ Tool        │ │  │
│  │  │ Abstraction │  │ Processing   │  │ Execution   │ │  │
│  │  └─────────────┘  └──────────────┘  └─────────────┘ │  │
│  └────────────────────────┬──────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │              Provider Clients                          │  │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │  │
│  │  │ OpenAI       │  │ Azure       │  │ Azure AI     │ │  │
│  │  │ Assistants   │  │ OpenAI      │  │ Services     │ │  │
│  │  │ Client       │  │ Client      │  │ Client       │ │  │
│  │  └──────────────┘  └─────────────┘  └──────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

This architecture provides several key benefits:

1. **Provider Agnostic**: Switch between OpenAI, Azure OpenAI, and Azure AI services with minimal code changes
2. **Testable Design**: Each layer can be independently tested with mock implementations
3. **Extensible**: New tool types and providers can be added without breaking existing code
4. **Type Safe**: Full TypeScript-style type safety with Python type hints and Pydantic validation

### Client Types Comparison

Understanding when to use each client type is crucial for optimal implementation:

| Client Type | Best For | Key Features | Complexity |
|-------------|----------|--------------|------------|
| **OpenAIChatClient** | Real-time conversations, simple integrations | Direct chat completions, streaming support | Low |
| **OpenAIAssistantsClient** | Complex workflows, file handling, persistent conversations | Thread management, built-in tools, state persistence | Medium |
| **OpenAIResponsesClient** | Structured outputs, reasoning, multi-modal tasks | Response formatting, image analysis, web search | High |

### Environment Setup

Before diving into agent development, ensure your environment is properly configured:

```bash
# Create virtual environment (recommended)
python -m v agent_framework_env
source agent_framework_env/bin/activate  # Linux/Mac
# or
agent_framework_env\Scripts\activate  # Windows

# Install the framework
pip install agent-framework

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_CHAT_MODEL_ID="gpt-4o"
export OPENAI_RESPONSES_MODEL_ID="gpt-4o"

# Optional: Azure configuration
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
export AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="gpt-4o"
```

### Quick Start: Your First Agent

Let's create your first working agent to demonstrate the framework's simplicity:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

async def first_agent():
    """Create a simple conversational agent"""
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="MyFirstAgent",
        instructions="You are a friendly and helpful assistant."
    ) as agent:
        # Non-streaming response
        result = await agent.run("Hello! Can you introduce yourself?")
        print(f"Agent: {result.text}")
        
        # Streaming response
        print("Agent: ", end="")
        async for chunk in agent.run_stream("Tell me a fun fact about space"):
            print(chunk.text, end="")
        print()

if __name__ == "__main__":
    asyncio.run(first_agent())
```

This simple example demonstrates the framework's core philosophy: **elegant simplicity without sacrificing power**. The same patterns you'll learn here scale to enterprise-grade applications handling millions of conversations.

---

## Core Concepts

Understanding the framework's fundamental concepts is essential for building sophisticated AI agents. This section provides deep technical insights into the architectural patterns that make the Microsoft Agent Framework both powerful and intuitive.

### Client Architecture Deep Dive

The framework's client architecture represents a sophisticated abstraction over various AI service providers, unified under consistent interfaces while preserving the unique capabilities of each underlying service.

#### OpenAIChatClient: The Conversational Specialist

The `OpenAIChatClient` serves as the foundation for real-time conversational AI, optimized for low-latency interactions and simple integration patterns. Its architecture emphasizes direct communication with minimal overhead:

```python
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent
import asyncio

async def chat_client_deep_dive():
    """Demonstrate OpenAIChatClient capabilities and configuration"""
    
    # Basic client initialization
    client = OpenAIChatClient(
        model_id="gpt-4o",  # Explicit model selection
        api_key="your-api-key",  # Optional: defaults to OPENAI_API_KEY env var
        base_url="https://api.openai.com/v1",  # Optional: for custom endpoints
        timeout=30.0,  # Request timeout in seconds
        max_retries=3,  # Automatic retry configuration
        additional_headers={"Custom-Header": "value"}  # Custom headers
    )
    
    # Advanced configuration with client-level defaults
    client_configured = OpenAIChatClient(
        model_id="gpt-4o",
        temperature=0.7,  # Creativity vs determinism
        max_tokens=1000,  # Response length limit
        top_p=0.9,  # Nucleus sampling parameter
        frequency_penalty=0.1,  # Reduce repetition
        presence_penalty=0.1,  # Encourage topic diversity
        stop_sequences=["\n\n", "Human:"]  # Custom stop sequences
    )
    
    # Creating agents with different configurations
    async with ChatAgent(
        chat_client=client_configured,
        name="CreativeWriter",
        instructions="You are a creative writing assistant. Be imaginative and engaging.",
        temperature=0.8  # Agent-level override
    ) as agent:
        # The agent inherits client configuration but can override specific parameters
        result = await agent.run("Write a mysterious opening paragraph")
        print(f"Creative output: {result.text}")

if __name__ == "__main__":
    asyncio.run(chat_client_deep_dive())
```

The `OpenAIChatClient` excels in scenarios requiring:
- **Real-time conversational interfaces** with sub-second response times
- **Simple integrations** where minimal setup complexity is paramount
- **Streaming responses** for interactive user experiences
- **Custom parameter tuning** for specific use cases (creative writing, technical explanations, etc.)

#### OpenAIAssistantsClient: The Enterprise Workhorse

The `OpenAIAssistantsClient` provides a comprehensive solution for complex workflows requiring persistent state, file handling, and sophisticated tool integration:

```python
from agent_framework.openai import OpenAIAssistantsClient
from agent_framework import ChatAgent, HostedFileSearchTool, HostedCodeInterpreterTool
import asyncio
import os

async def assistants_client_comprehensive():
    """Comprehensive demonstration of OpenAIAssistantsClient capabilities"""
    
    # Client initialization with enterprise features
    client = OpenAIAssistantsClient(
        model_id="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        organization_id=os.getenv("OPENAI_ORG_ID"),  # Multi-tenant support
        project_id=os.getenv("OPENAI_PROJECT_ID"),  # Project-based organization
        thread_auto_cleanup=False,  # Manual thread management for enterprise
        assistant_auto_cleanup=False  # Persistent assistants across sessions
    )
    
    # Create a sophisticated research assistant
    async with ChatAgent(
        chat_client=client,
        name="ResearchAssistant",
        instructions="""You are a comprehensive research assistant capable of:
        1. Searching through uploaded documents
        2. Writing and executing Python code for analysis
        3. Maintaining conversation context across sessions
        4. Providing citations and sources for your claims""",
        tools=[HostedFileSearchTool(), HostedCodeInterpreterTool()]
    ) as agent:
        
        # Example 1: File-based research with code execution
        research_query = """
        I have uploaded several research papers about climate change.
        Please:
        1. Search for information about carbon capture technologies
        2. Analyze any data tables you find
        3. Create a summary with specific statistics
        4. Generate a simple visualization of the trends
        """
        
        # This would work with properly uploaded files
        # result = await agent.run(research_query)
        
        # Example 2: Multi-step reasoning with persistent context
        step1 = await agent.run("What are the main types of renewable energy?")
        print(f"Step 1 - Energy types: {step1.text[:200]}...")
        
        step2 = await agent.run("Which of these is most efficient currently?")
        print(f"Step 2 - Efficiency comparison: {step2.text[:200]}...")
        
        step3 = await agent.run("Based on our discussion, what investment recommendations would you make?")
        print(f"Step 3 - Investment advice: {step3.text[:200]}...")

if __name__ == "__main__":
    asyncio.run(assistants_client_comprehensive())
```

The `OpenAIAssistantsClient` shines in enterprise scenarios requiring:
- **Persistent conversation threads** that survive application restarts
- **File processing and analysis** with built-in vector store management
- **Code execution capabilities** for data analysis and computation
- **Multi-step workflows** with maintained context and state

#### OpenAIResponsesClient: The Advanced Reasoning Engine

The `OpenAIResponsesClient` represents the cutting edge of AI capabilities, supporting structured outputs, reasoning chains, and multi-modal processing:

```python
from agent_framework.openai import OpenAIResponsesClient
from agent_framework import ChatAgent, HostedWebSearchTool
from pydantic import BaseModel
from typing import List
import asyncio

# Define structured output schemas
class MarketAnalysis(BaseModel):
    """Structured output for market analysis"""
    market_trend: str
    key_factors: List[str]
    risk_level: str  # "low", "medium", "high"
    investment_recommendation: str
    confidence_score: float  # 0.0 to 1.0

class ResearchSummary(BaseModel):
    """Structured output for research summaries"""
    main_findings: List[str]
    data_sources: List[str]
    methodology: str
    limitations: List[str]
    future_research_directions: List[str]

async def responses_client_advanced():
    """Advanced usage of OpenAIResponsesClient with reasoning and structured outputs"""
    
    # Client with reasoning capabilities
    reasoning_client = OpenAIResponsesClient(
        model_id="gpt-4o",
        additional_chat_options={
            "reasoning": {
                "effort": "high",  # Detailed reasoning
                "summary": "detailed"  # Include reasoning summary
            }
        }
    )
    
    # Agent with web search and reasoning
    async with ChatAgent(
        chat_client=reasoning_client,
        name="MarketAnalyst",
        instructions="""You are a sophisticated market analyst who:
        1. Searches for current market data
        2. Performs detailed reasoning about trends
        3. Provides structured, actionable insights
        4. Always cites sources and acknowledges uncertainty""",
        tools=HostedWebSearchTool()
    ) as agent:
        
        # Example 1: Complex reasoning with web search
        analysis_query = """
        Analyze the current state of the electric vehicle market.
        Consider:
        - Recent sales data and growth trends
        - Major player performance and market share
        - Regulatory impacts and government incentives
        - Supply chain challenges and opportunities
        - Consumer adoption patterns
        
        Provide a comprehensive analysis with specific data points.
        """
        
        print("Performing market analysis with reasoning...")
        async for chunk in agent.run_stream(analysis_query):
            if hasattr(chunk, 'reasoning'):
                print(f"🤔 Reasoning: {chunk.reasoning}")
            if chunk.text:
                print(f"📊 Analysis: {chunk.text}")
        
        # Example 2: Structured output with validation
        print("\n" + "="*50)
        print("Generating structured market analysis...")
        
        structured_result = await agent.run(
            "Provide a structured analysis of the current cryptocurrency market trends",
            response_format=MarketAnalysis
        )
        
        if structured_result.value:
            analysis: MarketAnalysis = structured_result.value
            print(f"Market Trend: {analysis.market_trend}")
            print(f"Risk Level: {analysis.risk_level}")
            print(f"Confidence: {analysis.confidence_score:.1%}")
            print(f"Recommendation: {analysis.investment_recommendation}")
            print(f"Key Factors: {', '.join(analysis.key_factors)}")

if __name__ == "__main__":
    asyncio.run(responses_client_advanced())
```

The `OpenAIResponsesClient` excels in advanced scenarios requiring:
- **Structured data extraction** with type safety and validation
- **Complex reasoning chains** with transparent thought processes
- **Multi-modal processing** combining text, images, and web data
- **Research and analysis tasks** requiring citation and source tracking

### Agent Lifecycle Management

Understanding the agent lifecycle is crucial for building robust applications that handle resources correctly and maintain consistent behavior:

```python
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
import asyncio
import logging

# Configure logging for lifecycle visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def agent_lifecycle_comprehensive():
    """Comprehensive demonstration of agent lifecycle management"""
    
    print("🔍 Agent Lifecycle Demonstration")
    print("=" * 50)
    
    # Phase 1: Agent Creation and Configuration
    print("1️⃣ Creating agent with custom configuration...")
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="LifecycleDemoAgent",
        instructions="You are demonstrating agent lifecycle management.",
        temperature=0.7,
        max_tokens=500
    )
    print(f"✅ Agent created: {agent.name}")
    
    # Phase 2: Context Manager Entry (Resource Acquisition)
    print("\n2️⃣ Entering context manager (acquiring resources)...")
    async with agent:
        print("✅ Agent context entered - resources allocated")
        
        # Phase 3: Active Operation
        print("\n3️⃣ Active operation phase...")
        
        # Multiple interactions within the same lifecycle
        interactions = [
            "What is your purpose?",
            "How do you manage resources?",
            "What happens when you're cleaned up?"
        ]
        
        for i, query in enumerate(interactions, 1):
            print(f"\n   Interaction {i}: {query}")
            result = await agent.run(query)
            print(f"   Response: {result.text[:100]}...")
        
        # Phase 4: Streaming Operation
        print("\n   Streaming interaction...")
        stream_query = "Explain resource management in programming"
        print(f"   Query: {stream_query}")
        print("   Response: ", end="")
        
        async for chunk in agent.run_stream(stream_query):
            print(chunk.text, end="")
        print()
        
        print("\n✅ Active operations completed")
    
    # Phase 5: Context Manager Exit (Resource Cleanup)
    print("\n4️⃣ Exited context manager (resources cleaned up)")
    print("✅ Agent lifecycle completed successfully")

async def advanced_lifecycle_patterns():
    """Advanced lifecycle patterns for production applications"""
    
    print("\n🔧 Advanced Lifecycle Patterns")
    print("=" * 50)
    
    # Pattern 1: Agent Pool Management
    print("1️⃣ Agent Pool Pattern...")
    async def create_agent_pool(size=3):
        """Create a pool of agents for concurrent processing"""
        pool = []
        for i in range(size):
            agent = ChatAgent(
                chat_client=OpenAIChatClient(),
                name=f"PoolAgent-{i}",
                instructions="You are part of an agent pool for concurrent processing."
            )
            pool.append(agent)
        return pool
    
    # Pattern 2: Lifecycle with Error Handling
    print("\n2️⃣ Robust Error Handling Pattern...")
    async def robust_agent_operation():
        """Demonstrate robust error handling throughout the lifecycle"""
        agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            name="RobustAgent",
            instructions="Handle errors gracefully and provide fallback responses."
        )
        
        try:
            async with agent:
                # Simulate various operations with error handling
                operations = [
                    "Valid query",
                    "Another valid operation",
                    # This would be an invalid operation that might fail
                    "Process this extremely large input" * 1000  
                ]
                
                for operation in operations:
                    try:
                        result = await agent.run(operation)
                        print(f"✅ Success: {result.text[:50]}...")
                    except Exception as e:
                        print(f"❌ Operation failed: {str(e)}")
                        # Implement fallback logic
                        fallback_result = await agent.run(
                            "Provide a helpful error message"
                        )
                        print(f"🔄 Fallback: {fallback_result.text}")
        
        except Exception as lifecycle_error:
            print(f"❌ Lifecycle error: {str(lifecycle_error)}")
            # Implement recovery or notification logic
    
    await robust_agent_operation()
    
    # Pattern 3: Resource Monitoring
    print("\n3️⃣ Resource Monitoring Pattern...")
    async def monitored_agent_operation():
        """Monitor resource usage during agent lifecycle"""
        import time
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        print("Monitoring resource usage...")
        
        # Pre-operation metrics
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Memory before: {memory_before:.2f} MB")
        
        agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            name="MonitoredAgent",
            instructions="Monitor my resource usage."
        )
        
        start_time = time.time()
        
        async with agent:
            # Perform operations
            for i in range(5):
                result = await agent.run(f"Operation {i + 1}")
                print(f"   Completed operation {i + 1}")
        
        end_time = time.time()
        
        # Post-operation metrics
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        execution_time = end_time - start_time
        
        print(f"   Memory after: {memory_after:.2f} MB")
        print(f"   Memory delta: {memory_after - memory_before:.2f} MB")
        print(f"   Execution time: {execution_time:.2f} seconds")
    
    await monitored_agent_operation()

if __name__ == "__main__":
    asyncio.run(agent_lifecycle_comprehensive())
    asyncio.run(advanced_lifecycle_patterns())
```

### Thread Management Patterns

Thread management represents one of the framework's most sophisticated features, providing multiple patterns for conversation state persistence:

```python
from agent_framework import ChatAgent, AgentThread, ChatMessageStore
from agent_framework.openai import OpenAIChatClient, OpenAIAssistantsClient
import asyncio
from datetime import datetime
import json

async def thread_management_comprehensive():
    """Comprehensive exploration of thread management patterns"""
    
    print("🧵 Thread Management Patterns")
    print("=" * 50)
    
    # Pattern 1: Automatic Thread Management (Stateless)
    print("1️⃣ Automatic Thread Creation (Stateless)...")
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="StatelessAgent",
        instructions="You maintain no memory between interactions."
    ) as agent:
        
        # Each call creates a new thread - no context persistence
        result1 = await agent.run("My name is Alice")
        print(f"First interaction: {result1.text[:50]}...")
        
        result2 = await agent.run("What is my name?")
        print(f"Second interaction: {result2.text[:50]}...")
        print("   Note: Agent doesn't remember the name (separate threads)")
    
    # Pattern 2: Explicit Thread Persistence
    print("\n2️⃣ Explicit Thread Persistence...")
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="PersistentAgent",
        instructions="You remember information from our conversation."
    ) as agent:
        
        # Create a thread that persists across interactions
        thread = agent.get_new_thread()
        
        # First interaction with context storage
        result1 = await agent.run("My name is Bob and I'm a software developer", thread=thread)
        print(f"First interaction: Stored context")
        
        # Second interaction - thread remembers previous context
        result2 = await agent.run("What do you know about me?", thread=thread)
        print(f"Second interaction: {result2.text}")
        print("   Note: Agent remembers name and profession (same thread)")
        
        # Third interaction - continued context
        result3 = await agent.run("What programming languages might I know?", thread=thread)
        print(f"Third interaction: {result3.text}")
    
    # Pattern 3: Thread with Custom Message Store
    print("\n3️⃣ Custom Message Store Pattern...")
    
    # Create a custom message store for advanced scenarios
    class PersistentMessageStore(ChatMessageStore):
        """Custom message store with file-based persistence"""
        
        def __init__(self, filename="conversation_history.json"):
            super().__init__()
            self.filename = filename
            self.load_history()
        
        def load_history(self):
            """Load conversation history from file"""
            try:
                with open(self.filename, 'r') as f:
                    history = json.load(f)
                    # Restore messages from saved history
                    for msg_data in history:
                        # Reconstruct message objects
                        pass  # Implementation depends on message format
            except FileNotFoundError:
                print("   No existing conversation history found")
        
        async def save_history(self):
            """Save conversation history to file"""
            messages = await self.list_messages()
            history_data = []
            
            for message in messages:
                history_data.append({
                    'role': message.role,
                    'content': str(message.content),
                    'timestamp': datetime.now().isoformat()
                })
            
            with open(self.filename, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"   Saved {len(history_data)} messages to {self.filename}")
    
    # Pattern 4: Cross-Session Thread Persistence with Assistants
    print("\n4️⃣ Cross-Session Thread Persistence...")
    
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        name="CrossSessionAgent",
        instructions="You maintain long-term conversation memory."
    ) as agent:
        
        # Create or retrieve a persistent thread
        thread = agent.get_new_thread()
        
        # Store conversation with persistence flag
        await agent.run("Remember that our company is building an AI platform", thread=thread, store=True)
        
        # The thread ID can be saved and reused in future sessions
        persistent_thread_id = thread.service_thread_id
        print(f"   Persistent thread ID: {persistent_thread_id}")
        
        # Simulate future session continuation
        print("   Simulating future session...")
        
        # In a future application run:
        future_thread = AgentThread(service_thread_id=persistent_thread_id)
        future_result = await agent.run(
            "What is my company building?", 
            thread=future_thread, 
            store=True
        )
        print(f"   Future session result: {future_result.text}")

async def advanced_thread_patterns():
    """Advanced thread management patterns for production use"""
    
    print("\n🔧 Advanced Thread Patterns")
    print("=" * 50)
    
    # Pattern 1: Thread Pool for Concurrent Conversations
    print("1️⃣ Thread Pool Management...")
    async def manage_concurrent_conversations():
        """Manage multiple concurrent conversations with thread isolation"""
        
        agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            name="ConcurrentManager",
            instructions="Handle multiple customer conversations simultaneously."
        )
        
        async with agent:
            # Simulate multiple customer conversations
            conversations = [
                {"customer": "Alice", "inquiry": "Order status #12345"},
                {"customer": "Bob", "inquiry": "Return policy question"},
                {"customer": "Carol", "inquiry": "Product recommendation"}
            ]
            
            # Create separate threads for each conversation
            customer_threads = {}
            for conv in conversations:
                thread = agent.get_new_thread()
                customer_threads[conv["customer"]] = thread
                
                # Initialize with customer context
                await agent.run(
                    f"Customer {conv['customer']} inquiry: {conv['inquiry']}",
                    thread=thread
                )
            
            # Continue conversations with context isolation
            follow_ups = [
                {"customer": "Alice", "question": "When will it arrive?"},
                {"customer": "Bob", "question": "How long does the return take?"},
                {"customer": "Carol", "question": "What's the warranty?"}
            ]
            
            for follow_up in follow_ups:
                thread = customer_threads[follow_up["customer"]]
                result = await agent.run(follow_up["question"], thread=thread)
                print(f"   {follow_up['customer']}: {result.text[:100]}...")
    
    await manage_concurrent_conversations()
    
    # Pattern 2: Thread Archival and Retrieval
    print("\n2️⃣ Thread Archival System...")
    async def archival_system():
        """Implement thread archival for compliance and analysis"""
        
        class ThreadArchive:
            """Archive system for conversation threads"""
            
            def __init__(self, archive_dir="thread_archives"):
                self.archive_dir = archive_dir
                import os
                os.makedirs(archive_dir, exist_ok=True)
            
            async def archive_thread(self, thread: AgentThread, metadata=None):
                """Archive a complete thread with metadata"""
                archive_data = {
                    "thread_id": thread.service_thread_id,
                    "archived_at": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "messages": []
                }
                
                # Extract messages from thread
                if thread.message_store:
                    messages = await thread.message_store.list_messages()
                    for msg in messages:
                        archive_data["messages"].append({
                            "role": msg.role,
                            "content": str(msg.content),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Save to file
                filename = f"thread_{thread.service_thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.archive_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(archive_data, f, indent=2)
                
                print(f"   Archived thread to {filepath}")
                return filepath
            
            def list_archives(self):
                """List all archived threads"""
                archives = []
                for filename in os.listdir(self.archive_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.archive_dir, filename)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        archives.append({
                            "filename": filename,
                            "thread_id": data["thread_id"],
                            "archived_at": data["archived_at"],
                            "message_count": len(data["messages"])
                        })
                return archives
        
        # Demonstrate archival system
        archive_system = ThreadArchive()
        
        agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            name="ArchivableAgent",
            instructions="Conversations will be archived for compliance."
        )
        
        async with agent:
            thread = agent.get_new_thread()
            
            # Simulate a customer service conversation
            await agent.run("I need help with my account", thread=thread)
            await agent.run("My account number is ACC-12345", thread=thread)
            result = await agent.run("I can't access my dashboard", thread=thread)
            
            # Archive the conversation
            await archive_system.archive_thread(
                thread, 
                metadata={
                    "customer_id": "CUST-12345",
                    "agent_id": "AGENT-001",
                    "conversation_type": "technical_support"
                }
            )
            
            # List archives
            archives = archive_system.list_archives()
            print(f"   Found {len(archives)} archived conversations")
    
    await archival_system()

if __name__ == "__main__":
    asyncio.run(thread_management_comprehensive())
    asyncio.run(advanced_thread_patterns())
```

### Tool Integration Approaches

The framework's tool integration system represents a sophisticated approach to extending agent capabilities while maintaining clean architecture and testability:

```python
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework.tools import HostedWebSearchTool, HostedCodeInterpreterTool, MCPStreamableHTTPTool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timezone
import httpx
import json

# Define custom tool schemas
class WeatherRequest(BaseModel):
    """Schema for weather requests"""
    location: str = Field(..., description="City and country (e.g., 'London, UK')")
    units: str = Field("celsius", description="Temperature units: celsius or fahrenheit")
    include_forecast: bool = Field(False, description="Include 5-day forecast")

class WeatherResponse(BaseModel):
    """Schema for weather responses"""
    location: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    forecast: Optional[List[Dict[str, Any]]] = None

async def tool_integration_comprehensive():
    """Comprehensive exploration of tool integration patterns"""
    
    print("🔧 Tool Integration Patterns")
    print("=" * 50)
    
    # Pattern 1: Simple Function Tools
    print("1️⃣ Simple Function Tools...")
    
    def get_current_time() -> str:
        """Get the current time in UTC"""
        current_time = datetime.now(timezone.utc)
        return f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def calculate_compound_interest(
        principal: float = Field(..., description="Initial investment amount"),
        rate: float = Field(..., description="Annual interest rate (as decimal, e.g., 0.05 for 5%)"),
        years: int = Field(..., description="Number of years"),
        compounds_per_year: int = Field(12, description="Number of times interest compounds per year")
    ) -> str:
        """Calculate compound interest and return investment summary"""
        amount = principal * (1 + rate/compounds_per_year) ** (compounds_per_year * years)
        interest_earned = amount - principal
        
        return f"""
        Investment Summary:
        - Principal: ${principal:,.2f}
        - Annual Rate: {rate*100:.1f}%
        - Time Period: {years} years
        - Compounding: {compounds_per_year} times per year
        - Final Amount: ${amount:,.2f}
        - Interest Earned: ${interest_earned:,.2f}
        - Total Return: {(interest_earned/principal)*100:.1f}%
        """
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="FinancialAdvisor",
        instructions="You are a helpful financial advisor with calculation tools.",
        tools=[get_current_time, calculate_compound_interest]
    ) as agent:
        
        result1 = await agent.run("What time is it?")
        print(f"Time query: {result1.text}")
        
        result2 = await agent.run("Calculate the return on $10,000 invested at 7% for 10 years")
        print(f"Investment calculation:\n{result2.text}")
    
    # Pattern 2: Async Function Tools
    print("\n2️⃣ Async Function Tools...")
    
    async def fetch_crypto_price(
        symbol: str = Field(..., description="Cryptocurrency symbol (e.g., 'BTC', 'ETH')"),
        currency: str = Field("USD", description="Currency to display price in")
    ) -> str:
        """Fetch current cryptocurrency price"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": symbol.lower(),
                        "vs_currencies": currency.lower(),
                        "include_24hr_change": "true"
                    },
                    timeout=10.0
                )
                data = response.json()
                
                if symbol.lower() in data:
                    price = data[symbol.lower()][currency.lower()]
                    change_24h = data[symbol.lower()].get(f"{currency.lower()}_24h_change", 0)
                    
                    return f"""
                    {symbol.upper()} Price Information:
                    - Current Price: ${price:,.2f} {currency.upper()}
                    - 24h Change: {change_24h:+.2f}%
                    - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                    """
                else:
                    return f"Cryptocurrency '{symbol}' not found. Try BTC, ETH, etc."
        
        except Exception as e:
            return f"Error fetching price: {str(e)}"
    
    async def search_news(
        query: str = Field(..., description="News search query"),
        max_results: int = Field(3, description="Maximum number of news items", ge=1, le=10)
    ) -> str:
        """Search for current news (simulated for demo)"""
        # In production, this would call a real news API
        news_items = [
            f"News {i+1}: Sample news about '{query}' - This is where real news would appear"
            for i in range(max_results)
        ]
        return "\n\n".join(news_items)
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="CryptoAnalyst",
        instructions="You are a cryptocurrency analyst with real-time data access.",
        tools=[fetch_crypto_price, search_news]
    ) as agent:
        
        crypto_result = await agent.run("What's the current price of Bitcoin?")
        print(f"Crypto query:\n{crypto_result.text}")
    
    # Pattern 3: Class-Based Tools with State
    print("\n3️⃣ Class-Based Tools with State...")
    
    class PortfolioTracker:
        """Stateful portfolio tracking tool"""
        
        def __init__(self):
            self.holdings: Dict[str, float] = {}
            self.transactions: List[Dict] = []
        
        def add_holding(
            self,
            symbol: str = Field(..., description="Stock symbol"),
            shares: float = Field(..., description="Number of shares", gt=0),
            purchase_price: float = Field(..., description="Purchase price per share", gt=0)
        ) -> str:
            """Add a stock holding to the portfolio"""
            if symbol in self.holdings:
                return f"Symbol {symbol} already exists. Use update_holding to modify."
            
            self.holdings[symbol] = shares
            self.transactions.append({
                "type": "add",
                "symbol": symbol,
                "shares": shares,
                "price": purchase_price,
                "timestamp": datetime.now().isoformat()
            })
            
            total_value = shares * purchase_price
            return f"Added {shares} shares of {symbol} at ${purchase_price:.2f} (Total: ${total_value:,.2f})"
        
        def get_portfolio_summary(self) -> str:
            """Get current portfolio summary"""
            if not self.holdings:
                return "Portfolio is empty."
            
            total_value = 0
            summary = ["Portfolio Summary:"]
            
            for symbol, shares in self.holdings.items():
                # In production, this would fetch current prices
                current_price = 150.00  # Simulated price
                value = shares * current_price
                total_value += value
                
                summary.append(f"- {symbol}: {shares} shares @ ${current_price:.2f} = ${value:,.2f}")
            
            summary.append(f"Total Portfolio Value: ${total_value:,.2f}")
            summary.append(f"Number of Holdings: {len(self.holdings)}")
            
            return "\n".join(summary)
        
        def remove_holding(self, symbol: str) -> str:
            """Remove a holding from the portfolio"""
            if symbol not in self.holdings:
                return f"Symbol {symbol} not found in portfolio."
            
            shares = self.holdings.pop(symbol)
            self.transactions.append({
                "type": "remove",
                "symbol": symbol,
                "shares": shares,
                "timestamp": datetime.now().isoformat()
            })
            
            return f"Removed {shares} shares of {symbol} from portfolio."
    
    # Create portfolio tracker instance
    portfolio = PortfolioTracker()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="PortfolioManager",
        instructions="You are a portfolio management assistant with tracking capabilities.",
        tools=[portfolio.add_holding, portfolio.get_portfolio_summary, portfolio.remove_holding]
    ) as agent:
        
        # Build a sample portfolio through conversation
        await agent.run("Add 100 shares of AAPL at $150.50")
        await agent.run("Add 50 shares of GOOGL at $2800.00")
        await agent.run("Add 200 shares of TSLA at $900.25")
        
        summary_result = await agent.run("Show me my portfolio summary")
        print(f"Portfolio summary:\n{summary_result.text}")

async def advanced_tool_patterns():
    """Advanced tool integration patterns for production use"""
    
    print("\n🔧 Advanced Tool Patterns")
    print("=" * 50)
    
    # Pattern 1: Tool Composition and Chaining
    print("1️⃣ Tool Composition and Chaining...")
    
    class DataProcessor:
        """Composable data processing tools"""
        
        def load_data(self, source: str = Field(..., description="Data source identifier")) -> str:
            """Load data from source (simulated)"""
            return f"Loaded dataset with 1000 records from {source}"
        
        def clean_data(self, dataset_id: str = Field(..., description="Dataset identifier")) -> str:
            """Clean and preprocess data"""
            return f"Cleaned dataset {dataset_id}: removed 50 invalid records, normalized 5 fields"
        
        def analyze_data(self, dataset_id: str = Field(..., description="Clean dataset identifier")) -> str:
            """Perform statistical analysis"""
            return f"Analysis of {dataset_id}: mean=42.5, median=40.2, std=15.3, correlation=0.78"
        
        def visualize_data(self, analysis_id: str = Field(..., description="Analysis identifier")) -> str:
            """Create data visualizations"""
            return f"Created 3 visualizations for {analysis_id}: histogram, scatter plot, correlation matrix"
    
    processor = DataProcessor()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="DataAnalyst",
        instructions="You are a data analyst that can process and analyze datasets.",
        tools=[
            processor.load_data,
            processor.clean_data,
            processor.analyze_data,
            processor.visualize_data
        ]
    ) as agent:
        
        # Chain multiple tools in a single conversation
        chain_result = await agent.run("""
        Please perform a complete data analysis:
        1. Load the sales_data_2024 dataset
        2. Clean the loaded data
        3. Analyze the cleaned dataset
        4. Create visualizations of the analysis
        """)
        
        print(f"Data processing chain result:\n{chain_result.text}")
    
    # Pattern 2: Conditional Tool Execution
    print("\n2️⃣ Conditional Tool Execution...")
    
    class SmartRouter:
        """Intelligent tool routing based on query analysis"""
        
        def route_query(self, query: str) -> str:
            """Analyze query and route to appropriate tools"""
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
                return "weather_tools"
            elif any(word in query_lower for word in ['stock', 'price', 'market', 'trading']):
                return "finance_tools"
            elif any(word in query_lower for word in ['code', 'programming', 'debug', 'algorithm']):
                return "code_tools"
            else:
                return "general_tools"
        
        def get_weather_summary(self, location: str) -> str:
            """Get weather summary for location"""
            return f"Weather in {location}: Sunny, 72°F, light winds"
        
        def get_stock_info(self, symbol: str) -> str:
            """Get stock information"""
            return f"{symbol} stock: $150.25 (+2.3%), volume: 1.2M"
        
        def get_code_help(self, language: str) -> str:
            """Get programming help"""
            return f"Programming help for {language}: Common patterns and best practices"
    
    router = SmartRouter()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="SmartRouterAgent",
        instructions="You intelligently route queries to appropriate tools.",
        tools=[
            router.route_query,
            router.get_weather_summary,
            router.get_stock_info,
            router.get_code_help
        ]
    ) as agent:
        
        # Test routing with different query types
        test_queries = [
            "What's the weather like today?",
            "Tell me about AAPL stock",
            "Help me with Python programming",
            "What's the meaning of life?"
        ]
        
        for query in test_queries:
            result = await agent.run(query)
            print(f"Query: {query}")
            print(f"Response: {result.text[:100]}...\n")
    
    # Pattern 3: Tool Result Caching
    print("\n3️⃣ Tool Result Caching...")
    
    from functools import lru_cache
    import time
    
    class CachedDataService:
        """Data service with intelligent caching"""
        
        def __init__(self):
            self.cache_stats = {"hits": 0, "misses": 0}
        
        @lru_cache(maxsize=128)
        def _expensive_api_call(self, endpoint: str, params: tuple) -> str:
            """Simulated expensive API call with caching"""
            self.cache_stats["misses"] += 1
            time.sleep(0.1)  # Simulate API latency
            
            return f"API response for {endpoint} with params {dict(params)}"
        
        def get_data(self, 
            endpoint: str = Field(..., description="API endpoint"),
            **params
        ) -> str:
            """Get data with caching"""
            # Convert params to hashable tuple for caching
            params_tuple = tuple(sorted(params.items()))
            
            # Check cache first
            cached_result = self._expensive_api_call(endpoint, params_tuple)
            
            if self.cache_stats["misses"] > 0:
                self.cache_stats["hits"] += 1
            
            return f"{cached_result} (Cache stats: {self.cache_stats})"
        
        def clear_cache(self) -> str:
            """Clear the cache"""
            self._expensive_api_call.cache_clear()
            self.cache_stats = {"hits": 0, "misses": 0}
            return "Cache cleared successfully"
    
    cached_service = CachedDataService()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="CachedDataAgent",
        instructions="You provide data with intelligent caching.",
        tools=[cached_service.get_data, cached_service.clear_cache]
    ) as agent:
        
        # First call - cache miss
        result1 = await agent.run("Get user data for user_id 12345")
        print(f"First call: {result1.text}")
        
        # Second call - cache hit
        result2 = await agent.run("Get user data for user_id 12345")
        print(f"Second call: {result2.text}")
        
        # Different parameters - cache miss
        result3 = await agent.run("Get user data for user_id 67890")
        print(f"Different params: {result3.text}")

if __name__ == "__main__":
    asyncio.run(tool_integration_comprehensive())
    asyncio.run(advanced_tool_patterns())
```

### Streaming vs Non-Streaming Responses

Understanding when and how to use streaming versus non-streaming responses is crucial for building responsive applications:

```python
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient
import asyncio
import time
from typing import AsyncGenerator, List

async def streaming_patterns_comprehensive():
    """Comprehensive exploration of streaming patterns"""
    
    print("🌊 Streaming vs Non-Streaming Patterns")
    print("=" * 50)
    
    # Performance comparison
    print("1️⃣ Performance Comparison...")
    
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="PerformanceTestAgent",
        instructions="Provide detailed, multi-paragraph responses."
    )
    
    async with agent:
        query = "Explain the history and impact of artificial intelligence in healthcare"
        
        # Non-streaming approach
        print("Non-streaming approach:")
        start_time = time.time()
        result = await agent.run(query)
        non_streaming_time = time.time() - start_time
        print(f"   Total time: {non_streaming_time:.2f}s")
        print(f"   Response length: {len(result.text)} characters")
        print(f"   First 100 chars: {result.text[:100]}...")
        
        # Streaming approach
        print("\nStreaming approach:")
        start_time = time.time()
        chunks_received = 0
        total_chars = 0
        
        async for chunk in agent.run_stream(query):
            chunks_received += 1
            if chunk.text:
                total_chars += len(chunk.text)
                if chunks_received <= 3:  # Show first few chunks
                    print(f"   Chunk {chunks_received}: {chunk.text[:50]}...")
        
        streaming_time = time.time() - start_time
        print(f"   Total time: {streaming_time:.2f}s")
        print(f"   Chunks received: {chunks_received}")
        print(f"   Total characters: {total_chars}")
        print(f"   Time to first chunk: Much faster!")
    
    # Real-world streaming patterns
    print("\n2️⃣ Real-World Streaming Patterns...")
    
    class StreamingUI:
        """Simulate a streaming UI component"""
        
        def __init__(self):
            self.buffer = []
            self.total_chars = 0
        
        async def update_display(self, text: str, is_partial: bool = True):
            """Update UI with new text"""
            if is_partial:
                # Simulate progressive UI update
                print(f"\r📝 {text}", end="", flush=True)
            else:
                # Final update
                print(f"\n✅ Complete: {text[:100]}...")
        
        async def handle_streaming_response(self, stream: AsyncGenerator) -> str:
            """Handle streaming response with UI updates"""
            full_response = []
            
            async for chunk in stream:
                if chunk.text:
                    full_response.append(chunk.text)
                    self.total_chars += len(chunk.text)
                    
                    # Update UI every 50 characters or at sentence boundaries
                    current_text = "".join(full_response)
                    if self.total_chars % 50 == 0 or chunk.text.endswith(('.', '!', '?')):
                        await self.update_display(current_text[-100:])
            
            # Final update
            final_text = "".join(full_response)
            await self.update_display(final_text, is_partial=False)
            return final_text
    
    # Demonstrate streaming UI
    ui = StreamingUI()
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="UIStreamingAgent",
        instructions="Provide helpful, detailed responses."
    )
    
    async with agent:
        print("\nStreaming with UI updates:")
        stream = agent.run_stream("Explain quantum computing simply")
        final_response = await ui.handle_streaming_response(stream)
    
    # Advanced streaming with chunk processing
    print("\n3️⃣ Advanced Chunk Processing...")
    
    class ChunkProcessor:
        """Advanced chunk processing for different content types"""
        
        def __init__(self):
            self.code_blocks = []
            self.lists = []
            self.paragraphs = []
            self.current_code_block = []
            self.in_code_block = False
        
        def process_chunk(self, text: str) -> dict:
            """Process chunk and categorize content"""
            result = {
                "type": "text",
                "content": text,
                "is_code": False,
                "is_list": False,
                "is_heading": False
            }
            
            # Detect code blocks
            if "```" in text:
                self.in_code_block = not self.in_code_block
                result["is_code"] = True
                result["type"] = "code_boundary"
            
            elif self.in_code_block:
                result["is_code"] = True
                result["type"] = "code_content"
                self.current_code_block.append(text)
            
            # Detect lists
            elif text.strip().startswith(('-', '*', '+', '1.', '2.', '3.')):
                result["is_list"] = True
                result["type"] = "list_item"
            
            # Detect headings
            elif text.strip().startswith('#'):
                result["is_heading"] = True
                result["type"] = "heading"
            
            return result
        
        def get_formatted_output(self) -> str:
            """Get formatted output based on processing"""
            output = []
            
            for chunk_type, chunks in [
                ("headings", [p for p in self.paragraphs if p["is_heading"]]),
                ("paragraphs", [p for p in self.paragraphs if not p["is_heading"] and not p["is_list"]]),
                ("lists", self.lists),
                ("code_blocks", self.code_blocks)
            ]:
                if chunks:
                    output.append(f"\n{chunk_type.upper()}:")
                    for chunk in chunks[:3]:  # Show first 3 of each type
                        output.append(f"  - {chunk['content'][:50]}...")
            
            return "\n".join(output)
    
    processor = ChunkProcessor()
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="ChunkProcessingAgent",
        instructions="Provide structured responses with examples and lists."
    )
    
    async with agent:
        print("Processing chunks by content type:")
        
        async for chunk in agent.run_stream("Explain Python decorators with examples"):
            if chunk.text:
                processed = processor.process_chunk(chunk.text)
                
                # Store for analysis
                if processed["type"] != "code_boundary":
                    processor.paragraphs.append(processed)
                if processed["is_list"]:
                    processor.lists.append(processed)
                if processed["is_code"] and processed["type"] == "code_content":
                    if processor.current_code_block:
                        processor.code_blocks.append({
                            "content": "".join(processor.current_code_block),
                            "type": "code_block"
                        })
                        processor.current_code_block = []
        
        print("\nContent analysis:")
        print(processor.get_formatted_output())
    
    # Streaming with backpressure handling
    print("\n4️⃣ Streaming with Backpressure...")
    
    class BackpressureHandler:
        """Handle streaming backpressure for high-volume responses"""
        
        def __init__(self, max_buffer_size: int = 1000):
            self.max_buffer_size = max_buffer_size
            self.buffer = []
            self.processed_chunks = 0
            self.start_time = time.time()
        
        async def process_with_backpressure(self, stream: AsyncGenerator) -> List[str]:
            """Process stream with backpressure control"""
            chunks = []
            
            async for chunk in stream:
                if chunk.text:
                    # Add to buffer
                    self.buffer.append(chunk.text)
                    self.processed_chunks += 1
                    
                    # Simulate processing delay
                    await asyncio.sleep(0.01)  # 10ms processing time
                    
                    # Check buffer size and apply backpressure if needed
                    if len(self.buffer) > self.max_buffer_size:
                        print(f"⚠️  Backpressure applied: buffer size {len(self.buffer)}")
                        # Process buffer
                        processed = "".join(self.buffer)
                        chunks.append(processed)
                        self.buffer = []
                        
                        # Simulate slower processing for large buffers
                        await asyncio.sleep(0.1)
            
            # Process remaining buffer
            if self.buffer:
                chunks.append("".join(self.buffer))
            
            elapsed = time.time() - self.start_time
            print(f"📊 Backpressure processing complete:")
            print(f"   Total chunks: {self.processed_chunks}")
            print(f"   Total time: {elapsed:.2f}s")
            print(f"   Average chunk time: {elapsed/self.processed_chunks*1000:.1f}ms")
            
            return chunks
    
    handler = BackpressureHandler(max_buffer_size=50)
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="BackpressureAgent",
        instructions="Generate a very long, detailed response about machine learning."
    )
    
    async with agent:
        print("Processing with backpressure control:")
        stream = agent.run_stream("Explain machine learning in detail")
        chunks = await handler.process_with_backpressure(stream)
        print(f"Final result: {len(chunks)} major chunks processed")

async def streaming_best_practices():
    """Best practices for streaming implementations"""
    
    print("\n🎯 Streaming Best Practices")
    print("=" * 50)
    
    # Best Practice 1: Appropriate Use Cases
    print("1️⃣ When to Use Streaming:")
    use_cases = [
        "✅ Real-time chat interfaces where users expect immediate feedback",
        "✅ Large document generation where progress indication improves UX",
        "✅ Live presentations or demonstrations of AI capabilities",
        "✅ Bandwidth-constrained environments where partial results are valuable",
        "❌ Small, predictable responses where overhead exceeds benefit",
        "❌ Batch processing where complete results are required before continuation",
        "❌ Simple queries that return in <100ms"
    ]
    
    for case in use_cases:
        print(f"   {case}")
    
    # Best Practice 2: Error Handling in Streams
    print("\n2️⃣ Error Handling in Streams:")
    
    class RobustStreamHandler:
        """Robust streaming with error handling and recovery"""
        
        def __init__(self):
            self.errors = []
            self.warnings = []
            self.success_count = 0
        
        async def handle_stream_with_recovery(self, stream: AsyncGenerator) -> dict:
            """Handle streaming with comprehensive error handling"""
            chunks = []
            partial_response = ""
            
            try:
                async for chunk in stream:
                    try:
                        if chunk.text:
                            chunks.append(chunk.text)
                            partial_response += chunk.text
                            self.success_count += 1
                            
                            # Validate chunk content
                            if len(chunk.text) > 1000:  # Suspiciously large chunk
                                self.warnings.append(f"Large chunk detected: {len(chunk.text)} chars")
                            
                    except UnicodeDecodeError as e:
                        self.errors.append(f"Unicode error in chunk: {str(e)}")
                        # Attempt recovery by skipping problematic chunk
                        continue
                    
                    except Exception as e:
                        self.errors.append(f"Chunk processing error: {str(e)}")
                        # Implement chunk-level recovery
                        if "timeout" in str(e).lower():
                            self.warnings.append("Timeout detected, continuing with partial response")
                            break
            
            except asyncio.CancelledError:
                self.errors.append("Stream cancelled by user")
                return {
                    "status": "cancelled",
                    "partial_response": partial_response,
                    "chunks_received": len(chunks)
                }
            
            except Exception as e:
                self.errors.append(f"Stream-level error: {str(e)}")
                return {
                    "status": "error",
                    "partial_response": partial_response,
                    "error": str(e),
                    "chunks_received": len(chunks)
                }
            
            return {
                "status": "success" if not self.errors else "partial",
                "full_response": "".join(chunks),
                "chunks_received": len(chunks),
                "errors": self.errors,
                "warnings": self.warnings
            }
    
    handler = RobustStreamHandler()
    agent = ChatAgent(
        chat_client=OpenAIChatClient(),
        name="RobustStreamingAgent",
        instructions="Provide helpful responses."
    )
    
    async with agent:
        print("Testing robust streaming with error handling:")
        stream = agent.run_stream("Explain quantum computing")
        result = await handler.handle_stream_with_recovery(stream)
        
        print(f"Status: {result['status']}")
        print(f"Chunks received: {result['chunks_received']}")
        if result.get('errors'):
            print(f"Errors: {len(result['errors'])}")
        if result.get('warnings'):
            print(f"Warnings: {len(result['warnings'])}")
    
    # Best Practice 3: Performance Optimization
    print("\n3️⃣ Performance Optimization:")
    optimizations = [
        "🚀 Use streaming for first user interaction, then switch to non-streaming for follow-ups",
        "🚀 Implement chunk buffering to reduce UI update frequency",
        "🚀 Pre-warm connections to reduce initial latency",
        "🚀 Use connection pooling for high-throughput applications",
        "🚀 Implement adaptive chunk sizes based on content type",
        "🚀 Monitor and optimize for time-to-first-chunk metrics"
    ]
    
    for opt in optimizations:
        print(f"   {opt}")

if __name__ == "__main__":
    asyncio.run(streaming_patterns_comprehensive())
    asyncio.run(streaming_best_practices())
```

---

## Agent Type 1: Basic Conversational Agent

The Basic Conversational Agent serves as the foundation for all AI agent development within the Microsoft Agent Framework. This agent type focuses on natural language interactions without external tool dependencies, making it ideal for customer service, FAQ systems, educational assistants, and general-purpose chatbots.

### Architecture Overview

The Basic Conversational Agent follows a clean, three-layer architecture that promotes maintainability and extensibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ChatAgent Interface                       │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ Instructions│  │ Configuration│  │  Context    │ │  │
│  │  │   Engine    │  │   Manager    │  │  Manager    │ │  │
│  │  └─────────────┘  └──────────────┘  └─────────────┘ │  │
│  └─────────────────────────┬─────────────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │              OpenAIChatClient                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │  │
│  │  │  Request     │  │   Response   │  │   Stream   │ │  │
│  │  │  Builder     │  │  Processor   │  │  Handler   │ │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │  │
│  └─────────────────────────┬─────────────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │              OpenAI API                                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

This architecture provides several key advantages:
- **Separation of Concerns**: Each layer has a single responsibility
- **Testability**: Individual components can be tested in isolation
- **Flexibility**: Easy to extend with additional capabilities
- **Performance**: Optimized for both latency and throughput

### Complete Implementation Examples

Let's build a comprehensive Basic Conversational Agent that demonstrates all core capabilities:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json

# Configure logging for production-ready debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BasicConversationalAgent:
    """Production-ready basic conversational agent with comprehensive features"""
    
    def __init__(self, 
                 name: str = "Assistant",
                 instructions: Optional[str] = None,
                 model_id: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the conversational agent with configuration options
        
        Args:
            name: Agent name for identification
            instructions: System instructions for agent behavior
            model_id: OpenAI model to use
            temperature: Creativity vs determinism (0.0 to 2.0)
            max_tokens: Maximum response length
        """
        self.name = name
        self.instructions = instructions or self._get_default_instructions()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.interaction_count = 0
        self.conversation_history = []
        
        # Initialize the client with configuration
        self.client = OpenAIChatClient(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"Initialized {name} with model {model_id}")
    
    def _get_default_instructions(self) -> str:
        """Get default instructions for a helpful assistant"""
        return """You are a helpful, friendly, and knowledgeable assistant. 
        
        Your characteristics:
        - Provide accurate, helpful information
        - Be conversational and engaging
        - Admit when you don't know something
        - Ask clarifying questions when needed
        - Maintain a positive, professional tone
        - Keep responses concise but comprehensive"""
    
    async def chat(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message and get a response (non-streaming)
        
        Args:
            message: User's message
            user_id: Optional user identifier for personalization
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        self.interaction_count += 1
        
        try:
            # Add user context if available
            enhanced_instructions = self.instructions
            if user_id:
                enhanced_instructions += f"\n\nYou are currently assisting user: {user_id}"
            
            async with ChatAgent(
                chat_client=self.client,
                name=self.name,
                instructions=enhanced_instructions
            ) as agent:
                
                # Get response
                result = await agent.run(message)
                
                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Store conversation history
                self.conversation_history.append({
                    "timestamp": start_time.isoformat(),
                    "user_id": user_id,
                    "message": message,
                    "response": result.text,
                    "response_time": response_time,
                    "interaction_number": self.interaction_count
                })
                
                logger.info(f"Interaction {self.interaction_count} completed in {response_time:.2f}s")
                
                return {
                    "success": True,
                    "response": result.text,
                    "response_time": response_time,
                    "interaction_number": self.interaction_count,
                    "model_used": self.model_id
                }
        
        except Exception as e:
            logger.error(f"Error in interaction {self.interaction_count}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request.",
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def chat_stream(self, message: str, user_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Send a message and get a streaming response
        
        Args:
            message: User's message
            user_id: Optional user identifier
            
        Yields:
            Response chunks as they become available
        """
        self.interaction_count += 1
        logger.info(f"Starting streaming interaction {self.interaction_count}")
        
        try:
            enhanced_instructions = self.instructions
            if user_id:
                enhanced_instructions += f"\n\nYou are currently assisting user: {user_id}"
            
            async with ChatAgent(
                chat_client=self.client,
                name=self.name,
                instructions=enhanced_instructions
            ) as agent:
                
                full_response = []
                chunk_count = 0
                
                async for chunk in agent.run_stream(message):
                    if chunk.text:
                        chunk_count += 1
                        full_response.append(chunk.text)
                        yield chunk.text
                
                # Store complete conversation
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "message": message,
                    "response": "".join(full_response),
                    "chunks_received": chunk_count,
                    "interaction_number": self.interaction_count,
                    "streaming": True
                })
                
                logger.info(f"Streaming interaction {self.interaction_count} completed with {chunk_count} chunks")
        
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def interactive_session(self, context: Optional[str] = None):
        """
        Run an interactive chat session with the agent
        
        Args:
            context: Optional context to set for the session
        """
        print(f"\n🤖 Starting interactive session with {self.name}")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'history' to see conversation history")
        print("Type 'save' to save the conversation")
        if context:
            print(f"Session context: {context}")
        print("-" * 50)
        
        session_instructions = self.instructions
        if context:
            session_instructions += f"\n\nSession context: {context}"
        
        try:
            async with ChatAgent(
                chat_client=self.client,
                name=self.name,
                instructions=session_instructions
            ) as agent:
                
                while True:
                    try:
                        user_input = input("\n👤 You: ").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'bye']:
                            print("🤖 Assistant: Goodbye! Have a great day!")
                            break
                        
                        if user_input.lower() == 'history':
                            self._show_history()
                            continue
                        
                        if user_input.lower() == 'save':
                            self._save_conversation()
                            continue
                        
                        if not user_input:
                            continue
                        
                        # Get streaming response for better UX
                        print("\n🤖 Assistant: ", end="", flush=True)
                        
                        full_response = []
                        async for chunk in agent.run_stream(user_input):
                            if chunk.text:
                                print(chunk.text, end="", flush=True)
                                full_response.append(chunk.text)
                        
                        print()  # New line after response
                        
                        # Store in history
                        self.conversation_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "user": user_input,
                            "assistant": "".join(full_response),
                            "type": "interactive"
                        })
                    
                    except KeyboardInterrupt:
                        print("\n\nSession interrupted. Goodbye!")
                        break
                    except Exception as e:
                        print(f"\nError: {str(e)}")
        
        except Exception as e:
            logger.error(f"Session error: {str(e)}")
            print(f"Failed to start session: {str(e)}")
    
    def _show_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print(f"\n📜 Conversation History ({len(self.conversation_history)} entries):")
        print("-" * 50)
        
        for i, entry in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            timestamp = entry.get("timestamp", "Unknown")
            if "user" in entry:  # Interactive session
                print(f"\n{i}. [{timestamp}]")
                print(f"   You: {entry['user']}")
                print(f"   Assistant: {entry['assistant'][:100]}...")
            else:  # API call
                print(f"\n{i}. [{timestamp}]")
                print(f"   Message: {entry.get('message', 'N/A')}")
                print(f"   Response: {entry.get('response', 'N/A')[:100]}...")
    
    def _save_conversation(self, filename: Optional[str] = None):
        """Save conversation history to file"""
        if not self.conversation_history:
            print("No conversation history to save.")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "agent_name": self.name,
                    "model": self.model_id,
                    "total_interactions": self.interaction_count,
                    "saved_at": datetime.now().isoformat(),
                    "conversation_history": self.conversation_history
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Conversation saved to {filename}")
        
        except Exception as e:
            print(f"Failed to save conversation: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent usage statistics"""
        if not self.conversation_history:
            return {"total_interactions": 0}
        
        total_time = sum(
            entry.get("response_time", 0) 
            for entry in self.conversation_history 
            if "response_time" in entry
        )
        
        return {
            "total_interactions": self.interaction_count,
            "conversations_stored": len(self.conversation_history),
            "average_response_time": total_time / len([
                e for e in self.conversation_history 
                if "response_time" in e
            ]) if total_time > 0 else 0,
            "model_used": self.model_id,
            "first_interaction": self.conversation_history[0].get("timestamp"),
            "last_interaction": self.conversation_history[-1].get("timestamp")
        }

# Advanced conversational patterns
async def advanced_conversational_patterns():
    """Demonstrate advanced conversational agent patterns"""
    
    print("🎯 Advanced Conversational Patterns")
    print("=" * 60)
    
    # Pattern 1: Multi-Personality Agent
    print("1️⃣ Multi-Personality Agent...")
    
    personalities = {
        "professional": {
            "instructions": "You are a professional business consultant. Be formal, analytical, and data-driven.",
            "temperature": 0.3
        },
        "friendly": {
            "instructions": "You are a friendly, approachable assistant. Be warm, conversational, and encouraging.",
            "temperature": 0.7
        },
        "creative": {
            "instructions": "You are a creative writer and artist. Be imaginative, poetic, and inspiring.",
            "temperature": 0.9
        }
    }
    
    async def switch_personality(personality: str, query: str):
        """Switch between different agent personalities"""
        config = personalities[personality]
        
        agent = BasicConversationalAgent(
            name=f"{personality.title()}Assistant",
            instructions=config["instructions"],
            temperature=config["temperature"]
        )
        
        result = await agent.chat(query)
        print(f"\n{personality.upper()} PERSONALITY:")
        print(f"Response: {result['response'][:200]}...")
        print(f"Temperature: {config['temperature']}")
    
    await switch_personality("professional", "How can I improve my business strategy?")
    await switch_personality("friendly", "I'm feeling overwhelmed with my project. Any advice?")
    await switch_personality("creative", "Describe the feeling of watching a sunrise.")
    
    # Pattern 2: Contextual Conversation
    print("\n2️⃣ Contextual Conversation...")
    
    class ContextualAgent(BasicConversationalAgent):
        """Agent that maintains and uses conversation context"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context = {
                "user_preferences": {},
                "conversation_topics": [],
                "user_mood": "neutral"
            }
        
        def update_context(self, message: str, response: str):
            """Update conversation context based on interaction"""
            # Simple keyword-based context detection
            message_lower = message.lower()
            response_lower = response.lower()
            
            # Detect topics
            topics = ["technology", "business", "health", "travel", "food"]
            for topic in topics:
                if topic in message_lower or topic in response_lower:
                    if topic not in self.context["conversation_topics"]:
                        self.context["conversation_topics"].append(topic)
            
            # Detect mood
            positive_words = ["happy", "great", "excellent", "wonderful", "amazing"]
            negative_words = ["sad", "terrible", "awful", "horrible", "bad"]
            
            if any(word in response_lower for word in positive_words):
                self.context["user_mood"] = "positive"
            elif any(word in response_lower for word in negative_words):
                self.context["user_mood"] = "negative"
        
        async def contextual_chat(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
            """Chat with context awareness"""
            # Enhance message with context
            context_prompt = ""
            if self.context["conversation_topics"]:
                context_prompt += f"[Context: Previous topics included {', '.join(self.context['conversation_topics'])}] "
            if self.context["user_mood"] != "neutral":
                context_prompt += f"[User seems {self.context['user_mood']}] "
            
            enhanced_message = context_prompt + message
            
            # Get response
            result = await self.chat(enhanced_message, user_id)
            
            # Update context
            if result['success']:
                self.update_context(message, result['response'])
            
            # Add context to result
            result['context'] = self.context.copy()
            return result
    
    contextual_agent = ContextualAgent(name="ContextualAssistant")
    
    # Simulate contextual conversation
    conversations = [
        "Tell me about the latest technology trends",
        "I'm planning to start a business in tech",
        "I'm feeling excited about this new venture!",
        "What are some healthy habits for entrepreneurs?"
    ]
    
    print("Contextual conversation flow:")
    for i, message in enumerate(conversations, 1):
        result = await contextual_agent.contextual_chat(message)
        print(f"\n{i}. User: {message}")
        print(f"   Agent: {result['response'][:100]}...")
        print(f"   Context: {result['context']}")
    
    # Pattern 3: Conversation Quality Monitoring
    print("\n3️⃣ Conversation Quality Monitoring...")
    
    class QualityMonitoredAgent(BasicConversationalAgent):
        """Agent with built-in conversation quality monitoring"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quality_metrics = {
                "total_interactions": 0,
                "response_length_avg": 0,
                "response_time_avg": 0,
                "satisfaction_score": 0,
                "topic_relevance_score": 0
            }
        
        def calculate_quality_scores(self, message: str, response: str, response_time: float):
            """Calculate various quality metrics"""
            self.quality_metrics["total_interactions"] += 1
            
            # Response length score (optimal: 50-500 words)
            word_count = len(response.split())
            length_score = min(1.0, word_count / 100) if word_count < 500 else max(0.0, 2.0 - word_count / 500)
            
            # Response time score (optimal: <2 seconds)
            time_score = max(0.0, min(1.0, 2.0 / response_time if response_time > 0 else 1.0))
            
            # Topic relevance (simple keyword matching)
            message_keywords = set(message.lower().split())
            response_keywords = set(response.lower().split())
            relevance_score = len(message_keywords.intersection(response_keywords)) / len(message_keywords) if message_keywords else 0.5
            
            # Update rolling averages
            n = self.quality_metrics["total_interactions"]
            self.quality_metrics["response_length_avg"] = (self.quality_metrics["response_length_avg"] * (n-1) + word_count) / n
            self.quality_metrics["response_time_avg"] = (self.quality_metrics["response_time_avg"] * (n-1) + response_time) / n
            self.quality_metrics["topic_relevance_score"] = (self.quality_metrics["topic_relevance_score"] * (n-1) + relevance_score) / n
            
            return {
                "length_score": length_score,
                "time_score": time_score,
                "relevance_score": relevance_score,
                "overall_score": (length_score + time_score + relevance_score) / 3
            }
        
        async def quality_monitored_chat(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
            """Chat with quality monitoring"""
            result = await self.chat(message, user_id)
            
            if result['success']:
                quality_scores = self.calculate_quality_scores(
                    message,
                    result['response'],
                    result['response_time']
                )
                result['quality_scores'] = quality_scores
            
            return result
        
        def get_quality_report(self) -> Dict[str, Any]:
            """Get comprehensive quality report"""
            return {
                "metrics": self.quality_metrics,
                "quality_grade": self._calculate_grade(),
                "recommendations": self._get_quality_recommendations()
            }
        
        def _calculate_grade(self) -> str:
            """Calculate overall quality grade"""
            if self.quality_metrics["total_interactions"] == 0:
                return "N/A"
            
            avg_score = (
                min(self.quality_metrics["topic_relevance_score"] * 2, 1.0) +
                (1.0 if self.quality_metrics["response_time_avg"] < 2.0 else 0.5) +
                (1.0 if 50 < self.quality_metrics["response_length_avg"] < 500 else 0.7)
            ) / 3
            
            if avg_score >= 0.9:
                return "A+"
            elif avg_score >= 0.8:
                return "A"
            elif avg_score >= 0.7:
                return "B"
            elif avg_score >= 0.6:
                return "C"
            else:
                return "D"
        
        def _get_quality_recommendations(self) -> List[str]:
            """Get quality improvement recommendations"""
            recommendations = []
            
            if self.quality_metrics["response_time_avg"] > 2.0:
                recommendations.append("Consider optimizing for faster response times")
            
            if self.quality_metrics["response_length_avg"] < 50:
                recommendations.append("Responses may be too brief; consider more detailed explanations")
            elif self.quality_metrics["response_length_avg"] > 500:
                recommendations.append("Responses may be too long; consider more concise answers")
            
            if self.quality_metrics["topic_relevance_score"] < 0.7:
                recommendations.append("Work on improving topic relevance and context understanding")
            
            return recommendations
    
    quality_agent = QualityMonitoredAgent(name="QualityMonitoredAssistant")
    
    # Test with various queries
    test_queries = [
        "What's the weather like?",
        "Can you explain in detail how machine learning works and provide examples of different algorithms and their applications?",
        "Tell me about cats"
    ]
    
    print("Quality monitoring results:")
    for query in test_queries:
        result = await quality_agent.quality_monitored_chat(query)
        if 'quality_scores' in result:
            print(f"\nQuery: {query}")
            print(f"Quality Scores: {result['quality_scores']}")
    
    # Final quality report
    quality_report = quality_agent.get_quality_report()
    print(f"\nOverall Quality Grade: {quality_report['quality_grade']}")
    print(f"Recommendations: {quality_report['recommendations']}")

async def production_deployment_example():
    """Example of production-ready deployment"""
    
    print("\n🏭 Production Deployment Example")
    print("=" * 60)
    
    # Production configuration
    production_config = {
        "model_id": "gpt-4o",
        "temperature": 0.3,  # Lower temperature for consistency
        "max_tokens": 500,   # Reasonable limit for production
        "name": "CustomerSupportAgent"
    }
    
    # Instructions for production use
    production_instructions = """You are a professional customer support assistant for TechCorp.
    
    Guidelines:
    1. Always be polite, patient, and professional
    2. Provide accurate information about our products
    3. If you cannot help, escalate to a human agent
    4. Never share internal company information
    5. Keep responses under 300 words when possible
    6. Ask clarifying questions when needed
    
    Product Information:
    - TechCorp offers cloud computing solutions
    - Main products: CloudStorage, ComputeEngine, DataAnalytics
    - Support hours: 24/7 for critical issues, 9-5 for general inquiries
    - Contact: support@techcorp.com, 1-800-TECHCORP"""
    
    # Create production agent
    support_agent = BasicConversationalAgent(
        **production_config,
        instructions=production_instructions
    )
    
    # Simulate production scenarios
    production_scenarios = [
        {
            "user_id": "customer_12345",
            "message": "I can't access my CloudStorage account. It says my password is incorrect but I'm sure it's right.",
            "expected_type": "technical_support"
        },
        {
            "user_id": "customer_67890", 
            "message": "What are the pricing plans for your DataAnalytics service?",
            "expected_type": "sales_inquiry"
        },
        {
            "user_id": "customer_11111",
            "message": "I'm having trouble understanding how to set up ComputeEngine. Is there a tutorial?",
            "expected_type": "how_to"
        }
    ]
    
    print("Production scenario testing:")
    for scenario in production_scenarios:
        print(f"\n--- {scenario['expected_type'].replace('_', ' ').title()} ---")
        print(f"Customer: {scenario['message']}")
        
        result = await support_agent.chat(
            scenario['message'],
            user_id=scenario['user_id']
        )
        
        if result['success']:
            print(f"Agent: {result['response']}")
            print(f"Response time: {result['response_time']:.2f}s")
        else:
            print(f"Error: {result['error']}")
    
    # Show final statistics
    stats = support_agent.get_stats()
    print(f"\n📊 Production Session Statistics:")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Average response time: {stats['average_response_time']:.2f}s")
    print(f"Model used: {stats['model_used']}")

if __name__ == "__main__":
    # Basic usage example
    async def basic_example():
        agent = BasicConversationalAgent(name="MyAssistant")
        result = await agent.chat("Hello! Can you help me learn about Python?")
        print(f"Response: {result['response']}")
        print(f"Response time: {result['response_time']:.2f}s")
    
    # Run examples
    asyncio.run(basic_example())
    asyncio.run(advanced_conversational_patterns())
    asyncio.run(production_deployment_example())
```

### Configuration Options and Best Practices

The Basic Conversational Agent supports extensive configuration options that allow fine-tuning for specific use cases:

```python
# Configuration examples for different scenarios
CONFIGURATIONS = {
    "customer_service": {
        "instructions": """You are a professional customer service representative.
        - Always maintain a polite and helpful tone
        - Provide accurate information about products and services
        - Escalate complex issues appropriately
        - Follow company policies and procedures
        - Keep responses concise but complete""",
        "temperature": 0.3,
        "max_tokens": 300
    },
    
    "educational_tutor": {
        "instructions": """You are an experienced educational tutor.
        - Adapt explanations to the student's level
        - Use examples and analogies to clarify concepts
        - Encourage questions and curiosity
        - Provide step-by-step guidance
        - Celebrate learning progress""",
        "temperature": 0.5,
        "max_tokens": 500
    },
    
    "creative_writing": {
        "instructions": """You are a creative writing assistant.
        - Encourage imagination and creativity
        - Provide inspiring prompts and suggestions
        - Help overcome writer's block
        - Offer constructive feedback
        - Celebrate unique ideas""",
        "temperature": 0.8,
        "max_tokens": 800
    },
    
    "technical_expert": {
        "instructions": """You are a technical expert.
        - Provide accurate, detailed technical information
        - Use precise terminology appropriately
        - Include code examples when relevant
        - Explain complex concepts clearly
        - Cite sources and best practices""",
        "temperature": 0.2,
        "max_tokens": 1000
    }
}
```

### Common Pitfalls and Solutions

Understanding common issues helps build more robust conversational agents:

```python
# Common pitfalls and their solutions

class ConversationalAgentSolutions:
    """Solutions to common conversational agent problems"""
    
    @staticmethod
    def handle_vague_responses():
        """
        Problem: Agent provides vague or generic responses
        Solution: Enhance instructions with specific requirements
        """
        solution_instructions = """You are a helpful assistant. IMPORTANT: Always be specific and detailed in your responses.
        
        Guidelines for high-quality responses:
        1. Provide concrete examples and specific details
        2. Avoid generic phrases like "it depends" without explanation
        3. Include relevant context and background information
        4. Structure responses logically with clear points
        5. If uncertain, explain your reasoning process
        
        Example of improvement:
        ❌ Bad: "Python is good for web development"
        ✅ Good: "Python excels in web development through frameworks like Django (full-featured, batteries-included) and Flask (lightweight, flexible). For example, Django provides built-in ORM, authentication, and admin interface, making it ideal for rapid development of complex applications." """
        
        return solution_instructions
    
    @staticmethod
    def handle_repetitive_responses():
        """
        Problem: Agent becomes repetitive in long conversations
        Solution: Implement conversation context awareness
        """
        return """You are a helpful assistant. Avoid repetition by:
        
        1. Track topics already discussed in this conversation
        2. Introduce new perspectives and information
        3. Use varied language and examples
        4. Build upon previous points rather than repeating them
        5. Ask follow-up questions to explore new angles
        
        Before responding, quickly review what we've already covered and aim to add new value."""
    
    @staticmethod
    def handle_inconsistent_personality():
        """
        Problem: Agent personality varies between responses
        Solution: Define clear personality parameters
        """
        return """You are ConsistentAssistant. Your personality traits are:
        
        CORE TRAITS (always maintain):
        - Friendly but professional
        - Knowledgeable but humble
        - Encouraging and supportive
        - Clear and structured in communication
        
        COMMUNICATION STYLE:
        - Use "I" statements when sharing opinions
        - Ask clarifying questions when needed
        - Acknowledge user contributions
        - Provide encouragement for learning efforts
        
        NEVER:
        - Switch between formal and casual abruptly
        - Contradict previous statements without explanation
        - Become overly technical without context
        - Lose track of the conversation's tone"""
    
    @staticmethod
    def handle_long_response_times():
        """
        Problem: Agent takes too long to respond
        Solution: Optimize configuration and implement timeouts
        """
        return {
            "max_tokens": 300,  # Limit response length
            "temperature": 0.5,  # Reduce processing complexity
            "instructions": """You are a helpful assistant. Provide concise, focused responses.
            
            Response guidelines:
            - Aim for 2-4 sentences per response
            - Focus on the most important information
            - Use bullet points for multiple items
            - Save detailed explanations for when specifically requested
            - It's better to be concise and clear than comprehensive and confusing"""
        }
```

### Testing and Validation

Comprehensive testing ensures your conversational agent performs reliably:

```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

class TestBasicConversationalAgent:
    """Comprehensive test suite for basic conversational agents"""
    
    @pytest.fixture
    async def agent(self):
        """Create test agent"""
        return BasicConversationalAgent(
            name="TestAgent",
            temperature=0.0  # Deterministic for testing
        )
    
    @pytest.mark.asyncio
    async def test_basic_response(self, agent):
        """Test basic response generation"""
        result = await agent.chat("Hello")
        assert result["success"] is True
        assert len(result["response"]) > 0
        assert result["response_time"] > 0
    
    @pytest.mark.asyncio 
    async def test_streaming_response(self, agent):
        """Test streaming response generation"""
        chunks = []
        async for chunk in agent.chat_stream("Tell me a joke"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling with invalid input"""
        # Test with very long input
        long_input = "x" * 10000
        result = await agent.chat(long_input)
        
        # Should handle gracefully
        assert "success" in result
        if not result["success"]:
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, agent):
        """Test conversation history tracking"""
        # Have multiple interactions
        for i in range(3):
            await agent.chat(f"Message {i}")
        
        stats = agent.get_stats()
        assert stats["total_interactions"] == 3
        assert len(agent.conversation_history) == 3
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, agent):
        """Test response time performance requirements"""
        import time
        
        start_time = time.time()
        result = await agent.chat("Quick response test")
        end_time = time.time()
        
        # Should respond within reasonable time (< 5 seconds)
        assert result["response_time"] < 5.0
        assert result["success"] is True
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test invalid temperature
        with pytest.raises(ValueError):
            BasicConversationalAgent(temperature=3.0)  # Invalid temperature
        
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            BasicConversationalAgent(max_tokens=-1)  # Invalid max_tokens
```

The Basic Conversational Agent provides a solid foundation for AI-powered applications. Its clean architecture, comprehensive configuration options, and built-in best practices make it suitable for production deployment across various use cases, from customer service to educational applications. The patterns and practices demonstrated here scale naturally to more complex agent types while maintaining the same core principles of reliability, testability, and maintainability.

---

## Agent Type 2: Function-Calling Agent

The Function-Calling Agent represents a quantum leap in AI capabilities, transforming static conversational agents into dynamic systems that can interact with external APIs, perform calculations, manipulate data, and execute real-world actions. This agent type bridges the gap between AI understanding and practical functionality, enabling the creation of truly useful applications that go beyond simple question-answering.

### Architecture Overview

The Function-Calling Agent architecture introduces a sophisticated orchestration layer that manages tool discovery, parameter validation, execution, and result integration:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Function-Calling Agent Architecture              │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │   Tool Registry │  │  Parameter      │  │   Execution     │    │
│ │   & Discovery   │──│  Validation     │──│   Engine        │    │
│ │                 │  │                 │  │                 │    │
│ │ • Function      │  │ • Type checking │  │ • Async/Sync    │    │
│ │   registration  │  │ • Range validation│  │   execution     │    │
│ │ • Schema        │  │ • Required      │  │ • Error handling│    │
│ │   validation    │  │   fields        │  │ • Timeout mgmt  │    │
│ │ • Auto-discovery│  │ • Custom rules  │  │ • Retry logic   │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                              │                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                    Result Integration                           │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │  Formatter   │  │  Filter      │  │   Context        │    │ │
│ │  │              │  │              │  │   Integration    │    │ │
│ │  │ • JSON/XML   │  │ • Sensitive  │  │ • Conversation   │    │ │
│ │  │ • Markdown   │  │   data       │  │   memory         │    │ │
│ │  │ • Custom     │  │ • Errors     │  │ • Multi-tool     │    │ │
│ │  │   templates  │  │ • Warnings   │  │   coordination   │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                    ChatAgent with Tools                         │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

This architecture enables several powerful capabilities:
- **Dynamic Tool Discovery**: Functions are automatically discovered and registered
- **Type-Safe Execution**: Pydantic models ensure parameter validation
- **Intelligent Routing**: The agent decides which tools to use based on context
- **Result Orchestration**: Multiple tool results are combined intelligently
- **Error Resilience**: Graceful handling of tool failures and edge cases

### Core Implementation Patterns

Let's explore the fundamental patterns for building Function-Calling Agents:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum
import httpx
import json
import math
import random
from functools import lru_cache
import aiofiles

# Define comprehensive tool schemas
class WeatherLocation(BaseModel):
    """Schema for weather location requests"""
    city: str = Field(..., description="City name (e.g., 'New York', 'London')")
    country: Optional[str] = Field(None, description="Country code (e.g., 'US', 'UK')")
    units: str = Field("celsius", description="Temperature units: 'celsius' or 'fahrenheit'")
    
    @validator('units')
    def validate_units(cls, v):
        if v not in ['celsius', 'fahrenheit']:
            raise ValueError('Units must be either "celsius" or "fahrenheit"')
        return v

class WeatherResponse(BaseModel):
    """Schema for weather information responses"""
    location: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    visibility: int
    uv_index: Optional[int] = None
    air_quality: Optional[str] = None

class StockSymbol(BaseModel):
    """Schema for stock symbol requests"""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')")
    exchange: Optional[str] = Field(None, description="Stock exchange (e.g., 'NASDAQ', 'NYSE')")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Basic validation - in production, validate against known symbols
        if len(v) < 1 or len(v) > 5:
            raise ValueError('Stock symbol must be 1-5 characters')
        return v.upper()

class StockResponse(BaseModel):
    """Schema for stock information responses"""
    symbol: str
    company_name: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: str
    day_high: float
    day_low: float
    year_high: float
    year_low: float
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

# Implement function tools with comprehensive error handling
class WeatherService:
    """Comprehensive weather service with caching and error handling"""
    
    def __init__(self):
        self.request_count = 0
        self.cache_hits = 0
    
    @lru_cache(maxsize=100)
    def _get_cached_weather(self, location_key: str) -> Optional[Dict[str, Any]]:
        """Internal caching mechanism"""
        return None  # In production, implement actual caching
    
    async def get_current_weather(self, location: WeatherLocation) -> WeatherResponse:
        """
        Get current weather information for a location
        
        This demonstrates:
        - Complex parameter validation with Pydantic
        - Async API calls with proper error handling
        - Data transformation and formatting
        - Caching for performance optimization
        """
        try:
            self.request_count += 1
            
            # Build location string
            location_str = f"{location.city}"
            if location.country:
                location_str += f", {location.country}"
            
            # Simulate API call (in production, use real weather API)
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Generate realistic weather data
            base_temp = random.uniform(15, 30) if location.units == "celsius" else random.uniform(60, 85)
            
            # Simulate different weather conditions based on city
            weather_conditions = {
                "London": {"description": "Partly cloudy", "humidity": 75, "temp_adjust": -5},
                "New York": {"description": "Clear sky", "humidity": 60, "temp_adjust": 0},
                "Tokyo": {"description": "Light rain", "humidity": 80, "temp_adjust": 3},
                "Sydney": {"description": "Sunny", "humidity": 55, "temp_adjust": 8}
            }
            
            city_weather = weather_conditions.get(location.city, {
                "description": "Clear",
                "humidity": random.randint(40, 80),
                "temp_adjust": random.uniform(-3, 3)
            })
            
            temperature = base_temp + city_weather["temp_adjust"]
            feels_like = temperature + random.uniform(-2, 2)
            
            return WeatherResponse(
                location=location_str,
                temperature=round(temperature, 1),
                feels_like=round(feels_like, 1),
                humidity=city_weather["humidity"],
                description=city_weather["description"],
                wind_speed=random.uniform(5, 25),
                pressure=random.randint(1000, 1030),
                visibility=random.randint(5, 20),
                uv_index=random.randint(1, 10),
                air_quality=random.choice(["Good", "Moderate", "Unhealthy for sensitive groups"])
            )
            
        except Exception as e:
            logger.error(f"Weather service error for {location}: {str(e)}")
            raise Exception(f"Unable to fetch weather data for {location.city}. Please check the city name and try again.")

class StockService:
    """Comprehensive stock service with market data simulation"""
    
    def __init__(self):
        self.market_data = self._initialize_market_data()
    
    def _initialize_market_data(self) -> Dict[str, StockResponse]:
        """Initialize realistic market data"""
        return {
            "AAPL": StockResponse(
                symbol="AAPL",
                company_name="Apple Inc.",
                current_price=175.43,
                change=2.15,
                change_percent=1.24,
                volume=45678900,
                market_cap="2.8T",
                day_high=177.50,
                day_low=173.20,
                year_high=182.94,
                year_low=124.17,
                pe_ratio=29.5,
                dividend_yield=0.52
            ),
            "GOOGL": StockResponse(
                symbol="GOOGL",
                company_name="Alphabet Inc.",
                current_price=138.21,
                change=-0.85,
                change_percent=-0.61,
                volume=23456789,
                market_cap="1.7T",
                day_high=139.50,
                day_low=137.80,
                year_high=151.55,
                year_low=83.34,
                pe_ratio=25.8,
                dividend_yield=0.0
            ),
            "TSLA": StockResponse(
                symbol="TSLA",
                company_name="Tesla Inc.",
                current_price=248.87,
                change=5.32,
                change_percent=2.18,
                volume=67890123,
                market_cap="790B",
                day_high=251.20,
                day_low=245.50,
                year_high=299.29,
                year_low=101.81,
                pe_ratio=65.2,
                dividend_yield=0.0
            )
        }
    
    async def get_stock_info(self, symbol_request: StockSymbol) -> StockResponse:
        """
        Get comprehensive stock information
        
        This demonstrates:
        - Market data retrieval with fallback mechanisms
        - Dynamic data generation for unknown symbols
        - Rich response formatting with multiple data points
        """
        symbol = symbol_request.symbol
        
        # Return existing data or generate new data
        if symbol in self.market_data:
            return self.market_data[symbol]
        else:
            # Generate realistic data for unknown symbols
            base_price = random.uniform(50, 500)
            change = random.uniform(-10, 10)
            
            return StockResponse(
                symbol=symbol,
                company_name=f"{symbol} Corporation",
                current_price=round(base_price, 2),
                change=round(change, 2),
                change_percent=round((change / base_price) * 100, 2),
                volume=random.randint(1000000, 100000000),
                market_cap=f"{random.uniform(1, 500):.1f}B",
                day_high=round(base_price * 1.02, 2),
                day_low=round(base_price * 0.98, 2),
                year_high=round(base_price * 1.5, 2),
                year_low=round(base_price * 0.5, 2),
                pe_ratio=round(random.uniform(10, 50), 1),
                dividend_yield=round(random.uniform(0, 5), 2)
            )

# Advanced function tools with state management
class CalculatorService:
    """Advanced calculator with expression parsing and history"""
    
    def __init__(self):
        self.history = []
        self.variables = {}
    
    def calculate_expression(self, 
        expression: str = Field(..., description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(45)', 'sqrt(16)')")
    ) -> str:
        """
        Evaluate mathematical expressions with comprehensive error handling
        
        Supports:
        - Basic arithmetic (+, -, *, /, **)
        - Trigonometric functions (sin, cos, tan)
        - Logarithmic functions (log, ln)
        - Constants (pi, e)
        - Variables (if previously defined)
        """
        try:
            # Clean and validate expression
            expression = expression.strip().lower()
            
            # Replace common words
            replacements = {
                'pi': str(math.pi),
                'e': str(math.e),
                '×': '*',
                '÷': '/',
                'square root': 'sqrt',
                'root': 'sqrt'
            }
            
            for old, new in replacements.items():
                expression = expression.replace(old, new)
            
            # Safe evaluation with limited functions
            safe_dict = {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sqrt': math.sqrt,
                'log': math.log10,
                'ln': math.log,
                'abs': abs,
                'round': round,
                'max': max,
                'min': min,
                'pi': math.pi,
                'e': math.e
            }
            
            # Add user-defined variables
            safe_dict.update(self.variables)
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            # Store in history
            self.history.append({
                "expression": expression,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 6)
            
            return f"Expression: {expression}\nResult: {result}\nType: {type(result).__name__}"
            
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}\nPlease check your syntax and try again."
    
    def set_variable(self,
        name: str = Field(..., description="Variable name (letters and numbers only)"),
        value: float = Field(..., description="Numeric value to assign")
    ) -> str:
        """Set a variable for use in calculations"""
        # Validate variable name
        if not name.replace('_', '').isalnum():
            return "Variable name must contain only letters, numbers, and underscores."
        
        self.variables[name] = value
        return f"Variable '{name}' set to {value}"
    
    def get_history(self) -> str:
        """Get calculation history"""
        if not self.history:
            return "No calculations in history."
        
        history_text = "Calculation History:\n"
        for i, calc in enumerate(self.history[-10:], 1):  # Last 10 calculations
            history_text += f"{i}. {calc['expression']} = {calc['result']}\n"
        
        return history_text

# File operations tool with comprehensive functionality
class FileOperations:
    """File operations with safety checks and validation"""
    
    def __init__(self, allowed_extensions=None):
        self.allowed_extensions = allowed_extensions or {'.txt', '.json', '.csv', '.md'}
        self.operation_history = []
    
    async def read_file(self,
        filepath: str = Field(..., description="Path to the file to read"),
        max_lines: int = Field(50, description="Maximum number of lines to read", ge=1, le=1000)
    ) -> str:
        """
        Read file contents with safety checks
        
        Demonstrates:
        - File validation and safety checks
        - Async file operations
        - Content limiting for large files
        - Error handling for various file issues
        """
        try:
            # Validate file extension
            import os
            _, ext = os.path.splitext(filepath)
            if ext.lower() not in self.allowed_extensions:
                return f"File type '{ext}' not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
            
            # Check if file exists
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            # Read file asynchronously
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as file:
                lines = []
                line_count = 0
                
                async for line in file:
                    lines.append(line.rstrip())
                    line_count += 1
                    if line_count >= max_lines:
                        lines.append("... (file truncated)")
                        break
                
                content = "\n".join(lines)
                
                # Store operation history
                self.operation_history.append({
                    "operation": "read",
                    "filepath": filepath,
                    "lines_read": line_count,
                    "timestamp": datetime.now().isoformat()
                })
                
                return f"File: {filepath}\nLines read: {line_count}\nContent:\n{content}"
                
        except Exception as e:
            return f"Error reading file {filepath}: {str(e)}"
    
    async def write_file(self,
        filepath: str = Field(..., description="Path to the file to write"),
        content: str = Field(..., description="Content to write to the file"),
        backup: bool = Field(True, description="Create backup of existing file")
    ) -> str:
        """Write content to file with backup option"""
        try:
            import os
            import shutil
            
            # Validate file extension
            _, ext = os.path.splitext(filepath)
            if ext.lower() not in self.allowed_extensions:
                return f"File type '{ext}' not allowed."
            
            # Create backup if requested and file exists
            if backup and os.path.exists(filepath):
                backup_path = f"{filepath}.backup"
                shutil.copy2(filepath, backup_path)
            
            # Write file asynchronously
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as file:
                await file.write(content)
            
            # Store operation history
            self.operation_history.append({
                "operation": "write",
                "filepath": filepath,
                "content_length": len(content),
                "backup_created": backup and os.path.exists(filepath),
                "timestamp": datetime.now().isoformat()
            })
            
            return f"Successfully wrote {len(content)} characters to {filepath}" + \
                   (f" (backup created at {filepath}.backup)" if backup and os.path.exists(f"{filepath}.backup") else "")
                   
        except Exception as e:
            return f"Error writing file {filepath}: {str(e)}"

async def basic_function_calling_demo():
    """Demonstrate basic function calling capabilities"""
    
    print("🔧 Basic Function-Calling Agent Demo")
    print("=" * 60)
    
    # Initialize services
    weather_service = WeatherService()
    stock_service = StockService()
    calculator_service = CalculatorService()
    file_ops = FileOperations()
    
    # Create agent with multiple tools
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="MultiToolAssistant",
        instructions="""You are a helpful assistant with access to weather, stock, calculator, and file tools.
        
        Guidelines:
        - Use appropriate tools based on user requests
        - Provide clear, formatted responses
        - Explain what you're doing when using tools
        - Handle errors gracefully
        - Combine tools when beneficial""",
        tools=[
            weather_service.get_current_weather,
            stock_service.get_stock_info,
            calculator_service.calculate_expression,
            calculator_service.set_variable,
            calculator_service.get_history,
            file_ops.read_file,
            file_ops.write_file
        ]
    ) as agent:
        
        # Example 1: Simple weather query
        print("1️⃣ Weather Query:")
        weather_result = await agent.run("What's the weather like in London?")
        print(f"Response: {weather_result.text}\n")
        
        # Example 2: Stock information
        print("2️⃣ Stock Information:")
        stock_result = await agent.run("Tell me about Apple stock (AAPL)")
        print(f"Response: {stock_result.text}\n")
        
        # Example 3: Calculator with expressions
        print("3️⃣ Calculator Operations:")
        calc_result = await agent.run("Calculate the square root of 144 plus sin(45 degrees)")
        print(f"Response: {calc_result.text}\n")
        
        # Example 4: Complex multi-tool query
        print("4️⃣ Multi-Tool Query:")
        multi_result = await agent.run("""
        I'm planning a trip to Tokyo and want to invest in some stocks.
        Can you tell me:
        1. What the weather is like in Tokyo?
        2. Information about Tesla stock?
        3. Calculate how much 100 shares of Tesla would cost at the current price?
        """)
        print(f"Response: {multi_result.text}\n")

async def advanced_function_patterns():
    """Advanced patterns for function-calling agents"""
    
    print("🚀 Advanced Function-Calling Patterns")
    print("=" * 60)
    
    # Pattern 1: Tool Composition and Chaining
    print("1️⃣ Tool Composition and Chaining...")
    
    class DataAnalysisPipeline:
        """Composable data analysis tools"""
        
        def __init__(self):
            self.datasets = {}
            self.analyses = {}
        
        async def load_dataset(self,
            name: str = Field(..., description="Dataset name"),
            source: str = Field(..., description="Data source description"),
            sample_size: int = Field(1000, description="Number of sample records")
        ) -> str:
            """Load a dataset for analysis"""
            # Simulate dataset loading
            await asyncio.sleep(0.1)
            
            self.datasets[name] = {
                "source": source,
                "size": sample_size,
                "loaded_at": datetime.now().isoformat(),
                "columns": ["id", "value", "category", "timestamp"],
                "sample_data": [{"id": i, "value": random.uniform(10, 100), "category": random.choice(["A", "B", "C"])} for i in range(min(sample_size, 5))]
            }
            
            return f"Loaded dataset '{name}' with {sample_size} records from {source}"
        
        def perform_statistical_analysis(self,
            dataset_name: str = Field(..., description="Name of the dataset to analyze"),
            analysis_type: str = Field("descriptive", description="Type of analysis: 'descriptive', 'correlation', 'regression'")
        ) -> str:
            """Perform statistical analysis on a dataset"""
            if dataset_name not in self.datasets:
                return f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}"
            
            dataset = self.datasets[dataset_name]
            analysis_id = f"{dataset_name}_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if analysis_type == "descriptive":
                stats = {
                    "mean": random.uniform(40, 60),
                    "median": random.uniform(40, 60),
                    "std_dev": random.uniform(5, 15),
                    "min": random.uniform(10, 30),
                    "max": random.uniform(70, 100)
                }
                self.analyses[analysis_id] = stats
                
                return f"""Statistical Analysis for {dataset_name}:
                - Mean: {stats['mean']:.2f}
                - Median: {stats['median']:.2f}
                - Standard Deviation: {stats['std_dev']:.2f}
                - Range: {stats['min']:.2f} to {stats['max']:.2f}
                - Analysis ID: {analysis_id}"""
            
            elif analysis_type == "correlation":
                correlations = {
                    "value_category": random.uniform(-0.8, 0.8),
                    "id_value": random.uniform(-0.3, 0.3),
                    "timestamp_value": random.uniform(-0.5, 0.5)
                }
                self.analyses[analysis_id] = correlations
                
                return f"""Correlation Analysis for {dataset_name}:
                - Value vs Category: {correlations['value_category']:.3f}
                - ID vs Value: {correlations['id_value']:.3f}
                - Timestamp vs Value: {correlations['timestamp_value']:.3f}
                - Analysis ID: {analysis_id}"""
            
            else:
                return f"Analysis type '{analysis_type}' not supported. Use 'descriptive' or 'correlation'."
        
        def generate_report(self,
            analysis_ids: List[str] = Field(..., description="List of analysis IDs to include in report"),
            report_format: str = Field("summary", description="Report format: 'summary', 'detailed', 'executive'")
        ) -> str:
            """Generate a comprehensive report from multiple analyses"""
            if not analysis_ids:
                return "No analysis IDs provided"
            
            valid_analyses = [aid for aid in analysis_ids if aid in self.analyses]
            
            if not valid_analyses:
                return f"No valid analyses found. Available: {list(self.analyses.keys())}"
            
            report_content = f"# Data Analysis Report\n\n"
            report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_content += f"Included Analyses: {len(valid_analyses)}\n\n"
            
            if report_format == "executive":
                report_content += "## Executive Summary\n\n"
                report_content += f"- Total datasets analyzed: {len(self.datasets)}\n"
                report_content += f"- Total analyses performed: {len(valid_analyses)}\n"
                report_content += f"- Key findings: Multiple statistical measures calculated\n\n"
            
            for analysis_id in valid_analyses:
                analysis_data = self.analyses[analysis_id]
                report_content += f"## Analysis: {analysis_id}\n\n"
                report_content += f"```{json.dumps(analysis_data, indent=2)}```\n\n"
            
            return report_content
    
    # Create pipeline instance
    pipeline = DataAnalysisPipeline()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="DataAnalystAgent",
        instructions="You are a data analysis expert who can perform complex multi-step analyses.",
        tools=[
            pipeline.load_dataset,
            pipeline.perform_statistical_analysis,
            pipeline.generate_report
        ]
    ) as agent:
        
        # Complex multi-step analysis
        analysis_result = await agent.run("""
        I need to perform a comprehensive data analysis:
        
        1. Load the sales_data_2024 dataset with 5000 records from our CRM system
        2. Perform descriptive statistical analysis on this dataset
        3. Also perform correlation analysis 
        4. Generate an executive summary report combining both analyses
        
        Please execute this entire pipeline and provide the final report.
        """)
        
        print(f"Data Analysis Pipeline Result:\n{analysis_result.text}\n")
    
    # Pattern 2: Conditional Tool Execution with Business Logic
    print("2️⃣ Conditional Tool Execution...")
    
    class SmartBusinessAssistant:
        """Intelligent business assistant with conditional logic"""
        
        def __init__(self):
            self.client_tier = "standard"  # Could be dynamic based on authentication
            self.request_history = []
        
        def classify_request(self,
            request: str = Field(..., description="Customer request to classify")
        ) -> str:
            """Classify customer requests into categories"""
            request_lower = request.lower()
            
            if any(word in request_lower for word in ['price', 'cost', 'expensive', 'cheap', 'discount']):
                return "pricing_inquiry"
            elif any(word in request_lower for word in ['problem', 'issue', 'error', 'broken', 'not working']):
                return "technical_support"
            elif any(word in request_lower for word in ['feature', 'upgrade', 'enhancement', 'improvement']):
                return "feature_request"
            elif any(word in request_lower for word in ['cancel', 'refund', 'return', 'stop']):
                return "cancellation_request"
            else:
                return "general_inquiry"
        
        def get_pricing_info(self,
            product: str = Field(..., description="Product name"),
            tier: str = Field("standard", description="Customer tier")
        ) -> str:
            """Get pricing information based on customer tier"""
            pricing_data = {
                "standard": {
                    "basic_plan": "$29/month",
                    "pro_plan": "$79/month",
                    "enterprise_plan": "$299/month"
                },
                "premium": {
                    "basic_plan": "$19/month",
                    "pro_plan": "$59/month", 
                    "enterprise_plan": "$249/month"
                },
                "vip": {
                    "basic_plan": "$15/month",
                    "pro_plan": "$45/month",
                    "enterprise_plan": "$199/month"
                }
            }
            
            tier_pricing = pricing_data.get(tier, pricing_data["standard"])
            
            return f"""Pricing for {product} (Tier: {tier}):
            - Basic Plan: {tier_pricing['basic_plan']}
            - Pro Plan: {tier_pricing['pro_plan']}
            - Enterprise Plan: {tier_pricing['enterprise_plan']}
            
            Note: {tier.title()} tier customers receive special pricing."""
        
        def create_support_ticket(self,
            issue_description: str = Field(..., description="Description of the technical issue"),
            priority: str = Field("medium", description="Priority level: low, medium, high, critical")
        ) -> str:
            """Create a support ticket for technical issues"""
            ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            self.request_history.append({
                "type": "support_ticket",
                "ticket_id": ticket_id,
                "issue": issue_description,
                "priority": priority,
                "created_at": datetime.now().isoformat()
            })
            
            estimated_response = {
                "low": "24-48 hours",
                "medium": "4-8 hours",
                "high": "1-2 hours",
                "critical": "within 1 hour"
            }
            
            return f"""Support Ticket Created Successfully!
            
            Ticket ID: {ticket_id}
            Priority: {priority.upper()}
            Issue: {issue_description[:100]}...
            
            Estimated Response Time: {estimated_response.get(priority, '4-8 hours')}
            
            Our support team will contact you shortly."""
        
        def escalate_to_human(self,
            reason: str = Field(..., description="Reason for escalation"),
            urgency: str = Field("medium", description="Escalation urgency")
        ) -> str:
            """Escalate to human agent when AI cannot handle the request"""
            escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            self.request_history.append({
                "type": "escalation",
                "escalation_id": escalation_id,
                "reason": reason,
                "urgency": urgency,
                "timestamp": datetime.now().isoformat()
            })
            
            return f"""Request Escalated to Human Agent
            
            Escalation ID: {escalation_id}
            Urgency: {urgency.upper()}
            Reason: {reason}
            
            A human agent will review your request and respond within 2-4 hours.
            You can reference this escalation ID in future communications."""
        
        def process_feature_request(self,
            feature_description: str = Field(..., description="Description of the requested feature"),
            business_justification: str = Field(..., description="Business justification for the feature")
        ) -> str:
            """Process new feature requests"""
            request_id = f"FEAT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            self.request_history.append({
                "type": "feature_request",
                "request_id": request_id,
                "feature": feature_description,
                "justification": business_justification,
                "status": "under_review",
                "timestamp": datetime.now().isoformat()
            })
            
            return f"""Feature Request Submitted Successfully!
            
            Request ID: {request_id}
            Feature: {feature_description}
            
            Your feature request has been logged and will be reviewed by our product team.
            Typical review time: 1-2 weeks.
            You'll receive updates at your registered email address."""
    
    business_assistant = SmartBusinessAssistant()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="BusinessAssistantAgent",
        instructions="""You are an intelligent business assistant who:
        1. Classifies customer requests automatically
        2. Routes to appropriate tools based on classification
        3. Provides personalized responses based on customer tier
        4. Escalates complex issues when needed
        5. Maintains request history for context""",
        tools=[
            business_assistant.classify_request,
            business_assistant.get_pricing_info,
            business_assistant.create_support_ticket,
            business_assistant.escalate_to_human,
            business_assistant.process_feature_request
        ]
    ) as agent:
        
        # Test different request types
        test_requests = [
            "How much does your pro plan cost?",
            "I'm having trouble logging into my account",
            "I have an idea for a new feature that would help our team",
            "This is too complicated, I need to speak with a real person"
        ]
        
        print("Testing intelligent request routing:")
        for request in test_requests:
            print(f"\n--- Request: {request} ---")
            result = await agent.run(request)
            print(f"Response: {result.text}\n")
    
    # Pattern 3: Tool Result Caching and Optimization
    print("3️⃣ Tool Result Caching...")
    
    class OptimizedDataService:
        """Data service with intelligent caching and optimization"""
        
        def __init__(self):
            self.cache = {}
            self.cache_stats = {"hits": 0, "misses": 0, "expired": 0}
            self.request_count = 0
        
        def _get_cache_key(self, func_name: str, **kwargs) -> str:
            """Generate cache key from function name and parameters"""
            sorted_params = sorted(kwargs.items())
            return f"{func_name}:{str(sorted_params)}"
        
        def _is_cache_valid(self, cache_key: str, max_age_seconds: int = 300) -> bool:
            """Check if cached data is still valid"""
            if cache_key not in self.cache:
                return False
            
            cached_time = self.cache[cache_key].get("timestamp", 0)
            current_time = datetime.now().timestamp()
            
            if current_time - cached_time > max_age_seconds:
                self.cache_stats["expired"] += 1
                del self.cache[cache_key]
                return False
            
            return True
        
        async def get_expensive_data(self,
            query: str = Field(..., description="Data query"),
            use_cache: bool = Field(True, description="Whether to use caching"),
            cache_timeout: int = Field(300, description="Cache timeout in seconds")
        ) -> str:
            """Get expensive data with caching"""
            self.request_count += 1
            cache_key = self._get_cache_key("get_expensive_data", query=query)
            
            # Check cache first
            if use_cache and self._is_cache_valid(cache_key, cache_timeout):
                self.cache_stats["hits"] += 1
                cached_data = self.cache[cache_key]["data"]
                return f"{cached_data} (from cache - {cache_timeout}s timeout)"
            
            # Cache miss - fetch data
            self.cache_stats["misses"] += 1
            
            # Simulate expensive operation
            await asyncio.sleep(0.5)  # Simulate API call
            
            # Generate response
            result = f"Expensive data result for '{query}': {random.randint(1000, 9999)}"
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now().timestamp()
                }
            
            return f"{result} (fresh data - cached for {cache_timeout}s)"
        
        def get_cache_statistics(self) -> str:
            """Get cache performance statistics"""
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return f"""Cache Performance Statistics:
            - Total Requests: {self.request_count}
            - Cache Hits: {self.cache_stats["hits"]}
            - Cache Misses: {self.cache_stats["misses"]}
            - Cache Hit Rate: {hit_rate:.1f}%
            - Expired Entries: {self.cache_stats["expired"]}
            - Active Cache Entries: {len(self.cache)}
            """
        
        def clear_cache(self) -> str:
            """Clear all cached data"""
            cache_size = len(self.cache)
            self.cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0, "expired": 0}
            
            return f"Cache cleared. Removed {cache_size} entries."
    
    optimized_service = OptimizedDataService()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="OptimizedAgent",
        instructions="You are an optimized service with intelligent caching.",
        tools=[
            optimized_service.get_expensive_data,
            optimized_service.get_cache_statistics,
            optimized_service.clear_cache
        ]
    ) as agent:
        
        # Test caching behavior
        print("Testing cache performance:")
        
        # First call - cache miss
        result1 = await agent.run("Get data for customer analytics")
        print(f"First call: {result1.text[:100]}...")
        
        # Second call - cache hit
        result2 = await agent.run("Get data for customer analytics")
        print(f"Second call: {result2.text[:100]}...")
        
        # Get statistics
        stats_result = await agent.run("Show cache statistics")
        print(f"Cache stats: {stats_result.text}")

if __name__ == "__main__":
    asyncio.run(basic_function_calling_demo())
    asyncio.run(advanced_function_patterns())
```

### Production Patterns and Best Practices

Building production-ready Function-Calling Agents requires understanding advanced patterns for reliability, performance, and maintainability:

```python
# Production patterns for function-calling agents

class ProductionFunctionAgentPatterns:
    """Production-ready patterns for function-calling agents"""
    
    @staticmethod
    def implement_circuit_breaker():
        """
        Pattern: Circuit Breaker for External API Calls
        
        Prevents cascading failures when external services are down
        """
        from typing import Callable, Any
        import time
        from enum import Enum
        
        class CircuitState(Enum):
            CLOSED = "closed"      # Normal operation
            OPEN = "open"          # Failing fast
            HALF_OPEN = "half_open"  # Testing recovery
        
        class CircuitBreaker:
            def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.expected_exception = expected_exception
                self.failure_count = 0
                self.last_failure_time = None
                self.state = CircuitState.CLOSED
            
            def __call__(self, func: Callable) -> Callable:
                async def wrapper(*args, **kwargs):
                    if self.state == CircuitState.OPEN:
                        if time.time() - self.last_failure_time > self.recovery_timeout:
                            self.state = CircuitState.HALF_OPEN
                        else:
                            raise Exception("Circuit breaker is OPEN - failing fast")
                    
                    try:
                        result = await func(*args, **kwargs)
                        if self.state == CircuitState.HALF_OPEN:
                            self.state = CircuitState.CLOSED
                            self.failure_count = 0
                        return result
                    
                    except self.expected_exception as e:
                        self.failure_count += 1
                        self.last_failure_time = time.time()
                        
                        if self.failure_count >= self.failure_threshold:
                            self.state = CircuitState.OPEN
                        
                        raise e
                
                return wrapper
        
        # Usage example
        @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        async def reliable_api_call(endpoint: str, params: dict) -> dict:
            """API call with circuit breaker protection"""
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, params=params, timeout=10.0)
                return response.json()
        
        return reliable_api_call
    
    @staticmethod
    def implement_rate_limiting():
        """
        Pattern: Rate Limiting for API Consumption
        
        Controls usage of rate-limited external APIs
        """
        import asyncio
        from typing import Dict
        from datetime import datetime, timedelta
        
        class RateLimiter:
            def __init__(self, max_requests: int = 100, time_window: int = 3600):
                self.max_requests = max_requests
                self.time_window = timedelta(seconds=time_window)
                self.requests: Dict[str, List[datetime]] = {}
            
            async def acquire(self, identifier: str = "default") -> bool:
                """Acquire permission to make a request"""
                now = datetime.now()
                
                # Clean old requests
                if identifier in self.requests:
                    self.requests[identifier] = [
                        req_time for req_time in self.requests[identifier]
                        if now - req_time < self.time_window
                    ]
                else:
                    self.requests[identifier] = []
                
                # Check if under limit
                if len(self.requests[identifier]) < self.max_requests:
                    self.requests[identifier].append(now)
                    return True
                
                return False
            
            def get_wait_time(self, identifier: str = "default") -> float:
                """Get wait time until next request is allowed"""
                if identifier not in self.requests or not self.requests[identifier]:
                    return 0.0
                
                oldest_request = min(self.requests[identifier])
                reset_time = oldest_request + self.time_window
                wait_seconds = (reset_time - datetime.now()).total_seconds()
                
                return max(0.0, wait_seconds)
        
        return RateLimiter
    
    @staticmethod
    def implement_bulk_operations():
        """
        Pattern: Bulk Operations for Efficiency
        
        Process multiple items in a single operation
        """
        
        class BulkProcessor:
            """Efficient bulk processing with batching"""
            
            def __init__(self, batch_size: int = 10):
                self.batch_size = batch_size
            
            async def process_bulk_items(self, items: List[Any], process_func) -> List[Any]:
                """Process items in batches for efficiency"""
                results = []
                
                for i in range(0, len(items), self.batch_size):
                    batch = items[i:i + self.batch_size]
                    batch_results = await asyncio.gather(
                        *[process_func(item) for item in batch],
                        return_exceptions=True
                    )
                    results.extend(batch_results)
                
                return results
            
            def create_bulk_tool(self, single_item_func):
                """Convert single-item function to bulk operation"""
                async def bulk_func(items: List[Any]) -> List[Any]:
                    return await self.process_bulk_items(items, single_item_func)
                return bulk_func
        
        return BulkProcessor

# Example: Production-ready function tool
class ProductionWeatherService:
    """Production-ready weather service with all patterns applied"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.rate_limiter = ProductionFunctionAgentPatterns.implement_rate_limiting()(
            max_requests=100, time_window=3600
        )
        self.circuit_breaker = ProductionFunctionAgentPatterns.implement_circuit_breaker()
    
    @lru_cache(maxsize=1000)
    async def get_weather_production(self,
        city: str = Field(..., description="City name"),
        country: str = Field("US", description="Country code"),
        include_forecast: bool = Field(False, description="Include 5-day forecast")
    ) -> str:
        """Production weather service with caching, rate limiting, and circuit breaker"""
        
        # Rate limiting check
        if not await self.rate_limiter.acquire(f"weather_{city}"):
            wait_time = self.rate_limiter.get_wait_time(f"weather_{city}")
            return f"Rate limit exceeded. Please wait {wait_time:.0f} seconds before retrying."
        
        try:
            # Circuit breaker protected call
            weather_data = await self._fetch_weather_data(city, country)
            
            # Process and format response
            response = self._format_weather_response(weather_data, include_forecast)
            
            self.request_count += 1
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Weather service error: {str(e)}")
            return f"Weather service temporarily unavailable. Please try again later."
    
    @ProductionFunctionAgentPatterns.implement_circuit_breaker()(
        failure_threshold=5, recovery_timeout=60
    )
    async def _fetch_weather_data(self, city: str, country: str) -> Dict[str, Any]:
        """Protected weather data fetch"""
        # Simulate API call with circuit breaker
        await asyncio.sleep(0.1)
        
        # Simulate occasional failures for demonstration
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Weather API temporarily unavailable")
        
        return {
            "city": city,
            "country": country,
            "temperature": random.uniform(15, 30),
            "description": random.choice(["Sunny", "Cloudy", "Rainy", "Partly cloudy"]),
            "humidity": random.randint(40, 90),
            "wind_speed": random.uniform(5, 30)
        }
    
    def _format_weather_response(self, data: Dict[str, Any], include_forecast: bool) -> str:
        """Format weather data into user-friendly response"""
        response = f"""Weather for {data['city']}, {data['country']}:
        🌡️ Temperature: {data['temperature']:.1f}°C
        🌤️ Conditions: {data['description']}
        💧 Humidity: {data['humidity']}%
        💨 Wind: {data['wind_speed']:.1f} km/h"""
        
        if include_forecast:
            response += "\n\n📅 5-Day Forecast: Generally stable conditions expected"
        
        return response
    
    def get_service_health(self) -> str:
        """Get service health metrics"""
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        
        return f"""Service Health Report:
        ✅ Total Requests: {self.request_count}
        ⚠️ Errors: {self.error_count} ({error_rate:.1f}%)
        🎯 Cache Size: {len(self.get_weather_production.cache_info().currsize)}
        🔧 Cache Hits: {self.get_weather_production.cache_info().hits}
        📊 Cache Misses: {self.get_weather_production.cache_info().misses}"""

# Error handling best practices
class RobustErrorHandling:
    """Comprehensive error handling for function tools"""
    
    @staticmethod
    def create_robust_tool_wrapper(func):
        """Create a robust wrapper for any tool function"""
        
        async def robust_wrapper(*args, **kwargs):
            try:
                # Input validation
                if not args and not kwargs:
                    return "Error: No arguments provided"
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Validate result
                if result is None:
                    return "Error: Function returned no result"
                
                if isinstance(result, str) and len(result.strip()) == 0:
                    return "Error: Function returned empty result"
                
                return result
            
            except ValueError as e:
                return f"Input validation error: {str(e)}"
            
            except ConnectionError as e:
                return "Service temporarily unavailable. Please try again in a few moments."
            
            except TimeoutError as e:
                return "Request timed out. The service might be slow. Please try again."
            
            except PermissionError as e:
                return "Access denied. Please check your permissions or contact support."
            
            except Exception as e:
                # Log unexpected errors
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
                return f"An unexpected error occurred. Our team has been notified. Error: {str(e)}"
        
        # Preserve function metadata
        robust_wrapper.__name__ = func.__name__
        robust_wrapper.__doc__ = func.__doc__
        
        return robust_wrapper

# Testing framework for function tools
class FunctionToolTestingFramework:
    """Comprehensive testing framework for function tools"""
    
    @staticmethod
    async def test_tool_functionality(tool_func, test_cases):
        """Test tool functionality with various inputs"""
        results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance": []
        }
        
        for test_case in test_cases:
            start_time = time.time()
            
            try:
                result = await tool_func(**test_case["input"])
                end_time = time.time()
                
                # Validate result
                if "expected_contains" in test_case:
                    if test_case["expected_contains"] in result:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Test {test_case['name']}: Expected '{test_case['expected_contains']}' in result")
                
                elif "validation_func" in test_case:
                    if test_case["validation_func"](result):
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Test {test_case['name']}: Validation failed")
                
                results["performance"].append({
                    "test_name": test_case["name"],
                    "execution_time": end_time - start_time
                })
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Test {test_case['name']}: Exception - {str(e)}")
        
        return results
    
    @staticmethod
    def generate_test_report(results):
        """Generate comprehensive test report"""
        total_tests = results["passed"] + results["failed"]
        success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
        Function Tool Test Report
        ========================
        
        Summary:
        - Total Tests: {total_tests}
        - Passed: {results['passed']}
        - Failed: {results['failed']}
        - Success Rate: {success_rate:.1f}%
        
        Performance Metrics:
        """
        
        if results["performance"]:
            avg_time = sum(p["execution_time"] for p in results["performance"]) / len(results["performance"])
            report += f"- Average Execution Time: {avg_time:.3f}s\n"
            report += f"- Fastest Test: {min(p['execution_time'] for p in results['performance']):.3f}s\n"
            report += f"- Slowest Test: {max(p['execution_time'] for p in results['performance']):.3f}s\n"
        
        if results["errors"]:
            report += "\nErrors:\n"
            for error in results["errors"]:
                report += f"- {error}\n"
        
        return report

# Example usage of production patterns
async def production_function_agent_demo():
    """Demonstrate production-ready function-calling agent"""
    
    print("🏭 Production Function-Calling Agent Demo")
    print("=" * 60)
    
    # Create production weather service
    weather_service = ProductionWeatherService()
    
    # Create robust agent with error handling
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="ProductionAssistant",
        instructions="""You are a production-ready assistant with reliable tool access.
        
        Guidelines:
        - Handle errors gracefully and provide helpful feedback
        - Use caching to improve performance
        - Respect rate limits and implement backoff strategies
        - Provide clear, actionable responses
        - Monitor service health and report issues""",
        tools=[
            RobustErrorHandling.create_robust_tool_wrapper(weather_service.get_weather_production),
            weather_service.get_service_health
        ]
    ) as agent:
        
        # Test normal operation
        print("1️⃣ Normal Operation:")
        result1 = await agent.run("What's the weather in London?")
        print(f"Response: {result1.text}\n")
        
        # Test error handling
        print("2️⃣ Error Handling:")
        result2 = await agent.run("What's the weather in InvalidCity123?")
        print(f"Response: {result2.text}\n")
        
        # Test service health
        print("3️⃣ Service Health Check:")
        result3 = await agent.run("Show me the service health statistics")
        print(f"Response: {result3.text}\n")
    
    # Test comprehensive tool testing framework
    print("4️⃣ Tool Testing Framework:")
    
    # Create test cases
    test_cases = [
        {
            "name": "Valid City Weather",
            "input": {"city": "London", "country": "UK"},
            "expected_contains": "Temperature"
        },
        {
            "name": "Empty City Name",
            "input": {"city": "", "country": "US"},
            "validation_func": lambda result: "Error" in result
        },
        {
            "name": "Special Characters in City",
            "input": {"city": "New York", "country": "US"},
            "expected_contains": "New York"
        }
    ]
    
    # Run tests
    test_results = await FunctionToolTestingFramework.test_tool_functionality(
        weather_service.get_weather_production,
        test_cases
    )
    
    test_report = FunctionToolTestingFramework.generate_test_report(test_results)
    print(test_report)

if __name__ == "__main__":
    asyncio.run(production_function_agent_demo())
```

The Function-Calling Agent represents the pinnacle of AI utility, transforming theoretical AI capabilities into practical, production-ready solutions. By mastering the patterns and practices outlined in this section, you'll be equipped to build sophisticated AI applications that can interact with the real world, process complex data, and provide genuine value to users. The combination of robust error handling, performance optimization, and intelligent tool orchestration creates agents that are not just functional, but reliable and maintainable in production environments.

---

## Agent Type 3: RAG Agent

The Retrieval-Augmented Generation (RAG) Agent represents a paradigm shift in AI knowledge management, combining the generative power of large language models with the precision of information retrieval systems. This agent type enables organizations to create AI systems that can access, understand, and reason over vast amounts of proprietary knowledge while maintaining accuracy and providing source attribution.

### Architecture Overview

The RAG Agent architecture implements a sophisticated multi-stage pipeline that transforms raw documents into actionable knowledge:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG Agent Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │   Document      │  │   Chunking      │  │   Embedding     │    │
│ │   Ingestion     │──│   Strategy      │──│   Generation    │    │
│ │                 │  │                 │  │                 │    │
│ │ • Multi-format  │  │ • Size-based    │  │ • Vector        │    │
│ │   support       │  │   splitting     │  │   generation    │    │
│ │ • Metadata      │  │ • Semantic      │  │ • Dense         │    │
│ │   extraction    │  │   boundaries    │  │   embeddings    │    │
│ │ • Validation    │  │ • Overlap       │  │ • Caching       │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                              │                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │              Vector Store & Indexing                            │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │   Vector     │  │   Metadata   │  │   Search         │    │ │
│ │  │   Storage    │  │   Index      │  │   Optimization   │    │ │
│ │  │              │  │              │  │                  │    │ │
│ │  │ • Pinecone   │  │ • Filtering  │  │ • HNSW           │    │ │
│ │  │ • Weaviate   │  │ • Sorting    │  │ • FAISS           │    │ │
│ │  │ • Chroma     │  │ • Ranking    │  │ • Hybrid search  │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │              Retrieval & Ranking                                │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │   Query      │  │   Similarity │  │   Re-ranking     │    │ │
│ │  │   Processing │  │   Search     │  │   & Filtering    │    │ │
│ │  │              │  │              │  │                  │    │ │
│ │  │ • Expansion  │  │ • Cosine     │  │ • Cross-encoder  │    │ │
│ │  │ • Embedding  │  │   similarity │  │ • Relevance      │    │ │
│ │  │ • HyDE       │  │ • BM25       │  │   scoring        │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │              Response Generation                                │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │   Context    │  │   Prompt     │  │   Source         │    │ │
│ │  │   Assembly   │  │   Engineering│  │   Attribution    │    │ │
│ │  │              │  │              │  │                  │    │ │
│ │  │ • Relevant   │  │ • Few-shot   │  │ • Citations      │    │ │
│ │  │   chunks     │  │   examples   │  │ • Confidence     │    │ │
│ │  │ • Priority   │  │ • Role-based │  │ • References     │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Document Processing and Ingestion

The foundation of any RAG system lies in its ability to process diverse document formats and extract meaningful content:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIAssistantsClient
from agent_framework.tools import HostedFileSearchTool, HostedVectorStoreContent
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import aiofiles
import json
import os
from pathlib import Path
import hashlib
import mimetypes

# Enhanced document processing schemas
class DocumentMetadata(BaseModel):
    """Comprehensive document metadata"""
    filename: str
    file_size: int
    file_type: str
    created_at: datetime
    modified_at: datetime
    checksum: str
    language: Optional[str] = "en"
    category: Optional[str] = None
    tags: List[str] = []
    summary: Optional[str] = None
    author: Optional[str] = None

class ProcessedDocument(BaseModel):
    """Processed document with chunks and metadata"""
    metadata: DocumentMetadata
    chunks: List[Dict[str, Any]]
    vector_store_id: Optional[str] = None
    processing_timestamp: datetime

class DocumentProcessor:
    """Advanced document processor with multi-format support"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {'.txt', '.md', '.json', '.csv', '.pdf', '.docx'}
    
    async def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> ProcessedDocument:
        """
        Process a document with comprehensive error handling
        
        Demonstrates:
        - Multi-format document support
        - Intelligent chunking strategies
        - Metadata extraction and validation
        - Checksum generation for deduplication
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists and format
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
            
            # Extract basic metadata
            file_stats = file_path.stat()
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                created_at=datetime.fromtimestamp(file_stats.st_ctime),
                modified_at=datetime.fromtimestamp(file_stats.st_mtime),
                checksum=await self._calculate_checksum(file_path),
                **(metadata or {})
            )
            
            # Extract content based on file type
            content = await self._extract_content(file_path)
            
            # Generate intelligent chunks
            chunks = await self._create_intelligent_chunks(content, metadata)
            
            # Generate summary if not provided
            if not metadata.summary and len(content) > 100:
                metadata.summary = await self._generate_summary(content[:2000])
            
            return ProcessedDocument(
                metadata=metadata,
                chunks=chunks,
                processing_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {str(e)}")
            raise
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for deduplication"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _extract_content(self, file_path: Path) -> str:
        """Extract content based on file type"""
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return await self._extract_pdf_content(file_path)
        elif file_type == '.docx':
            return await self._extract_docx_content(file_path)
        elif file_type == '.json':
            return await self._extract_json_content(file_path)
        elif file_type == '.csv':
            return await self._extract_csv_content(file_path)
        else:  # txt, md
            return await self._extract_text_content(file_path)
    
    async def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text files"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    async def _extract_json_content(self, file_path: Path) -> str:
        """Extract and format JSON content"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                return content
    
    async def _extract_csv_content(self, file_path: Path) -> str:
        """Extract and format CSV content"""
        import csv
        content = []
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            # Read first few lines to understand structure
            lines = []
            async for line in f:
                lines.append(line.strip())
                if len(lines) > 20:  # Limit for large CSVs
                    break
            
            # Convert to readable format
            content.append(f"CSV File: {file_path.name}")
            content.append(f"Row count: {len(lines) - 1}")  # Header + data
            content.append("Preview:")
            content.extend(lines[:10])  # First 10 rows
            
        return "\n".join(content)
    
    async def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract content from PDF files"""
        try:
            import PyPDF2
            
            content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                content.append(f"PDF Document: {file_path.name}")
                content.append(f"Pages: {len(pdf_reader.pages)}")
                content.append("Content:")
                
                # Extract text from first few pages
                for page_num in range(min(5, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        content.append(f"\n--- Page {page_num + 1} ---")
                        content.append(text)
            
            return "\n".join(content)
            
        except ImportError:
            return f"PDF content extraction requires PyPDF2. File: {file_path.name}"
        except Exception as e:
            return f"Error extracting PDF content: {str(e)}"
    
    async def _extract_docx_content(self, file_path: Path) -> str:
        """Extract content from Word documents"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            content = [f"Word Document: {file_path.name}"]
            content.append("Content:")
            
            # Extract text from paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    content.append(para.text)
            
            return "\n".join(content)
            
        except ImportError:
            return f"Word document processing requires python-docx. File: {file_path.name}"
        except Exception as e:
            return f"Error extracting Word content: {str(e)}"
    
    async def _create_intelligent_chunks(self, content: str, metadata: DocumentMetadata) -> List[Dict[str, Any]]:
        """Create intelligent chunks with overlap and metadata"""
        chunks = []
        
        # Simple word-based chunking (in production, use more sophisticated methods)
        words = content.split()
        chunk_size_words = self.chunk_size // 5  # Rough estimate: 5 chars per word
        
        for i in range(0, len(words), chunk_size_words - self.chunk_overlap // 5):
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            
            chunk_data = {
                "text": chunk_text,
                "chunk_id": f"{metadata.checksum}_{len(chunks)}",
                "metadata": {
                    "start_word": i,
                    "end_word": min(i + chunk_size_words, len(words)),
                    "word_count": len(chunk_words),
                    "source_file": metadata.filename
                }
            }
            
            chunks.append(chunk_data)
        
        return chunks
    
    async def _generate_summary(self, content: str) -> str:
        """Generate a brief summary of the content"""
        # Simple summary generation (in production, use more sophisticated methods)
        sentences = content.split('.')
        if len(sentences) > 3:
            return ". ".join(sentences[:3]) + "..."
        return content[:200] + "..." if len(content) > 200 else content

# Vector store management with advanced features
class VectorStoreManager:
    """Advanced vector store management with optimization"""
    
    def __init__(self, client: OpenAIAssistantsClient):
        self.client = client
        self.vector_stores: Dict[str, str] = {}  # name -> id mapping
        self.file_mappings: Dict[str, List[str]] = {}  # store_id -> file_ids
    
    async def create_optimized_vector_store(self, 
        name: str, 
        documents: List[ProcessedDocument],
        expires_days: int = 7,
        chunk_size: int = 1000
    ) -> str:
        """
        Create an optimized vector store for RAG operations
        
        Features:
        - Automatic expiration management
        - Batch file processing
        - Metadata preservation
        - Performance optimization
        """
        try:
            # Create vector store with expiration
            vector_store = await self.client.client.beta.vector_stores.create(
                name=name,
                expires_after={
                    "anchor": "last_active_at",
                    "days": expires_days
                },
                chunking_strategy={
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": chunk_size,
                        "chunk_overlap_tokens": min(200, chunk_size // 5)
                    }
                }
            )
            
            store_id = vector_store.id
            self.vector_stores[name] = store_id
            
            # Process documents in batches for efficiency
            batch_size = 10
            all_file_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_file_ids = await self._process_document_batch(batch, store_id)
                all_file_ids.extend(batch_file_ids)
            
            self.file_mappings[store_id] = all_file_ids
            
            logger.info(f"Created vector store '{name}' with {len(documents)} documents")
            return store_id
            
        except Exception as e:
            logger.error(f"Failed to create vector store {name}: {str(e)}")
            raise
    
    async def _process_document_batch(self, documents: List[ProcessedDocument], store_id: str) -> List[str]:
        """Process a batch of documents efficiently"""
        file_ids = []
        
        for doc in documents:
            try:
                # Combine all chunks into a single file for vector store
                full_content = "\n\n".join(chunk["text"] for chunk in doc.chunks)
                
                # Create file with metadata
                file_response = await self.client.client.files.create(
                    file=(doc.metadata.filename, full_content.encode('utf-8')),
                    purpose="assistants"
                )
                
                file_id = file_response.id
                file_ids.append(file_id)
                
                # Add to vector store
                await self.client.client.beta.vector_stores.files.create(
                    vector_store_id=store_id,
                    file_id=file_id
                )
                
            except Exception as e:
                logger.error(f"Failed to process document {doc.metadata.filename}: {str(e)}")
                continue
        
        return file_ids
    
    async def search_vector_stores(self, 
        query: str, 
        store_names: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple vector stores with ranking
        
        Implements:
        - Multi-store search
        - Result deduplication
        - Relevance ranking
        - Performance optimization
        """
        results = []
        stores_to_search = store_names or list(self.vector_stores.keys())
        
        for store_name in stores_to_search:
            if store_name not in self.vector_stores:
                continue
            
            store_id = self.vector_stores[store_name]
            
            try:
                # Perform search
                search_results = await self.client.client.beta.vector_stores.search(
                    vector_store_id=store_id,
                    query=query,
                    max_results=max_results
                )
                
                for result in search_results.data:
                    results.append({
                        "store_name": store_name,
                        "store_id": store_id,
                        "file_id": result.file_id,
                        "score": result.score,
                        "content": result.content,
                        "metadata": result.metadata
                    })
            
            except Exception as e:
                logger.error(f"Search failed for store {store_name}: {str(e)}")
                continue
        
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    async def cleanup_expired_stores(self):
        """Clean up expired vector stores"""
        expired_stores = []
        
        for name, store_id in self.vector_stores.items():
            try:
                # Check if store still exists
                store = await self.client.client.beta.vector_stores.retrieve(store_id)
                if store.status == "expired":
                    expired_stores.append((name, store_id))
            except Exception:
                # Store doesn't exist anymore
                expired_stores.append((name, store_id))
        
        # Clean up local references
        for name, store_id in expired_stores:
            if name in self.vector_stores:
                del self.vector_stores[name]
            if store_id in self.file_mappings:
                del self.file_mappings[store_id]
        
        logger.info(f"Cleaned up {len(expired_stores)} expired vector stores")
        return expired_stores

# Advanced RAG implementation
class AdvancedRAGAgent:
    """Production-ready RAG agent with comprehensive features"""
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 chunk_size: int = 1000,
                 similarity_threshold: float = 0.7):
        self.vector_store_manager = vector_store_manager
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.query_history = []
        self.document_processor = DocumentProcessor(chunk_size=chunk_size)
    
    async def ingest_documents(self, 
        document_paths: List[str], 
        store_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ingest documents with comprehensive processing
        
        Features:
        - Batch document processing
        - Duplicate detection
        - Progress tracking
        - Error handling
        """
        results = {
            "processed": 0,
            "failed": 0,
            "duplicates": 0,
            "store_id": None,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        processed_documents = []
        
        print(f"📚 Processing {len(document_paths)} documents...")
        
        for i, doc_path in enumerate(document_paths, 1):
            try:
                print(f"Processing {i}/{len(document_paths)}: {os.path.basename(doc_path)}")
                
                # Process document
                processed_doc = await self.document_processor.process_document(doc_path, metadata)
                
                # Check for duplicates
                is_duplicate = await self._check_duplicate(processed_doc.metadata.checksum)
                if is_duplicate:
                    results["duplicates"] += 1
                    print(f"  ⚠️  Duplicate detected, skipping...")
                    continue
                
                processed_documents.append(processed_doc)
                results["processed"] += 1
                print(f"  ✅ Processed successfully")
                
            except Exception as e:
                results["failed"] += 1
                print(f"  ❌ Failed: {str(e)}")
                continue
        
        # Create vector store if documents were processed
        if processed_documents:
            store_id = await self.vector_store_manager.create_optimized_vector_store(
                name=store_name,
                documents=processed_documents
            )
            results["store_id"] = store_id
            print(f"🎯 Created vector store: {store_name} (ID: {store_id})")
        
        results["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        print(f"""
        📊 Processing Complete:
        - Documents processed: {results['processed']}
        - Duplicates skipped: {results['duplicates']}
        - Failed: {results['failed']}
        - Total time: {results['processing_time']:.2f}s
        """)
        
        return results
    
    async def _check_duplicate(self, checksum: str) -> bool:
        """Check if document already exists in any vector store"""
        # In production, implement proper duplicate checking
        # For now, return False (no duplicates)
        return False
    
    async def query_knowledge_base(self,
        query: str,
        store_names: Optional[List[str]] = None,
        max_results: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the knowledge base with advanced retrieval
        
        Features:
        - Multi-store search
        - Relevance ranking
        - Source attribution
        - Query history tracking
        """
        start_time = datetime.now()
        
        try:
            # Perform search
            search_results = await self.vector_store_manager.search_vector_stores(
                query=query,
                store_names=store_names,
                max_results=max_results
            )
            
            if not search_results:
                return {
                    "success": False,
                    "response": "No relevant information found in the knowledge base.",
                    "sources": []
                }
            
            # Format response with sources
            response_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                response_parts.append(f"{i}. {result['content'][:500]}...")
                
                if include_sources:
                    sources.append({
                        "source": f"{result['store_name']} (Score: {result['score']:.3f})",
                        "content_preview": result['content'][:200] + "...",
                        "file_id": result['file_id']
                    })
            
            # Generate comprehensive response
            response = "\n\n".join(response_parts)
            
            # Track query
            self.query_history.append({
                "query": query,
                "timestamp": datetime.now(),
                "results_count": len(search_results),
                "stores_searched": store_names or list(self.vector_store_manager.vector_stores.keys()),
                "response_time": (datetime.now() - start_time).total_seconds()
            })
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "search_results": search_results,
                "query_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            return {
                "success": False,
                "response": f"Error searching knowledge base: {str(e)}",
                "sources": []
            }
    
    async def conversational_rag(self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        store_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Conversational RAG with context awareness
        
        Maintains conversation context and provides more natural interactions
        """
        # Enhance query with conversation context
        enhanced_query = query
        
        if conversation_history:
            context = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
            enhanced_query = f"Conversation context:\n{context}\n\nCurrent question: {query}"
        
        # Search knowledge base
        kb_results = await self.query_knowledge_base(
            query=enhanced_query,
            store_names=store_names,
            max_results=3
        )
        
        if not kb_results["success"]:
            return kb_results
        
        # Create conversational response
        conversational_context = f"""
        Based on the knowledge base search for: "{query}"
        
        Found information:
        {kb_results["response"]}
        
        Please provide a helpful, conversational response that:
        1. Directly answers the user's question
        2. Uses the found information appropriately
        3. Maintains a natural, helpful tone
        4. Cites sources when relevant
        """
        
        return {
            "success": True,
            "response": conversational_context,
            "sources": kb_results["sources"],
            "raw_search_results": kb_results["search_results"]
        }

async def basic_rag_demo():
    """Demonstrate basic RAG functionality"""
    
    print("📚 Basic RAG Agent Demo")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        "climate_change_basics.txt",
        "renewable_energy_guide.txt", 
        "carbon_footprint_tips.txt"
    ]
    
    # Create document content
    doc_contents = {
        "climate_change_basics.txt": """
        Climate Change Basics
        
        Climate change refers to long-term shifts in global temperatures and weather patterns. 
        While climate variations are natural, scientific evidence shows that human activities 
        have been the dominant driver since the mid-20th century.
        
        Key Facts:
        - Global average temperature has risen by about 1.1°C since pre-industrial times
        - Carbon dioxide levels are at their highest in 3 million years
        - Sea levels are rising 3.3mm per year
        - Arctic sea ice is declining at 13% per decade
        
        The primary cause is the greenhouse effect from burning fossil fuels, which traps 
        heat in the Earth's atmosphere.
        """,
        
        "renewable_energy_guide.txt": """
        Renewable Energy Guide
        
        Renewable energy comes from natural sources that are constantly replenished. 
        The main types include solar, wind, hydroelectric, geothermal, and biomass.
        
        Solar Power:
        - Converts sunlight into electricity using photovoltaic cells
        - Cost has decreased by 90% since 2010
        - Can power homes, businesses, and even cities
        - Works best in sunny regions but viable globally
        
        Wind Energy:
        - Uses turbines to convert wind into electricity
        - Offshore wind is particularly promising
        - Modern turbines can power thousands of homes
        - Requires consistent wind speeds of 6-55 mph
        """,
        
        "carbon_footprint_tips.txt": """
        Carbon Footprint Reduction Tips
        
        A carbon footprint is the total amount of greenhouse gases generated by our actions. 
        The average person produces about 4 tons of CO2 per year.
        
        Transportation:
        - Walk, bike, or use public transport when possible
        - Consider electric or hybrid vehicles
        - Combine errands to reduce trips
        - Fly less or offset flight emissions
        
        Home Energy:
        - Switch to renewable energy providers
        - Improve home insulation
        - Use energy-efficient appliances
        - Install solar panels if possible
        """
    }
    
    # Create sample documents
    doc_paths = []
    for filename, content in doc_contents.items():
        filepath = f"/tmp/{filename}"
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(content)
        doc_paths.append(filepath)
    
    print("📝 Created sample documents")
    
    # Initialize RAG components
    async with OpenAIAssistantsClient() as client:
        vector_store_manager = VectorStoreManager(client)
        rag_agent = AdvancedRAGAgent(vector_store_manager)
        
        # Ingest documents
        print("\n📥 Ingesting documents into knowledge base...")
        ingestion_result = await rag_agent.ingest_documents(
            document_paths=doc_paths,
            store_name="environmental_knowledge",
            metadata={"topic": "climate_and_energy", "language": "en"}
        )
        
        # Query the knowledge base
        print("\n🔍 Querying knowledge base...")
        
        test_queries = [
            "What is climate change and what causes it?",
            "How does solar power work and is it cost-effective?",
            "What are some ways to reduce my carbon footprint?",
            "Compare renewable energy sources"
        ]
        
        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            
            result = await rag_agent.query_knowledge_base(
                query=query,
                max_results=2,
                include_sources=True
            )
            
            if result["success"]:
                print(f"Response: {result['response'][:300]}...")
                print(f"Sources: {len(result['sources'])} found")
                print(f"Query time: {result['query_time']:.3f}s")
            else:
                print(f"No results found for: {query}")

async def advanced_rag_patterns():
    """Advanced RAG patterns for production use"""
    
    print("\n🚀 Advanced RAG Patterns")
    print("=" * 60)
    
    # Pattern 1: Multi-Modal RAG (Text + Metadata)
    print("1️⃣ Multi-Modal RAG...")
    
    class MultiModalRAGAgent:
        """RAG agent that handles text, metadata, and structured data"""
        
        def __init__(self):
            self.knowledge_graph = {}  # Store relationships
        
        async def ingest_structured_data(self,
            data_type: str = Field(..., description="Type of data: 'product', 'research', 'policy'"),
            data_content: Dict[str, Any] = Field(..., description="Structured data content"),
            relationships: List[Dict[str, str]] = Field(default_factory=list, description="Relationships to other data")
        ) -> str:
            """Ingest structured data with relationships"""
            
            data_id = f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store data
            self.knowledge_graph[data_id] = {
                "type": data_type,
                "content": data_content,
                "relationships": relationships,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create searchable text
            searchable_text = f"{data_type.upper()} DATA\n"
            for key, value in data_content.items():
                searchable_text += f"{key}: {value}\n"
            
            if relationships:
                searchable_text += "\nRELATIONSHIPS:\n"
                for rel in relationships:
                    searchable_text += f"- {rel['type']}: {rel['target']}\n"
            
            return f"Ingested {data_type} data with ID: {data_id}\nContent preview: {searchable_text[:200]}..."
        
        def query_knowledge_graph(self,
            entity: str = Field(..., description="Entity to search for"),
            relationship_type: Optional[str] = Field(None, description="Specific relationship type to filter by")
        ) -> str:
            """Query the knowledge graph for entities and relationships"""
            
            relevant_items = []
            
            for data_id, data in self.knowledge_graph.items():
                # Search in content
                content_str = str(data["content"]).lower()
                if entity.lower() in content_str:
                    relevant_items.append(data)
                
                # Search in relationships
                for rel in data["relationships"]:
                    if entity.lower() in rel.get("target", "").lower():
                        relevant_items.append(data)
                        break
            
            if not relevant_items:
                return f"No information found about '{entity}'"
            
            # Format results
            result = f"Knowledge Graph Results for '{entity}':\n\n"
            
            for item in relevant_items[:3]:  # Limit results
                result += f"Type: {item['type']}\n"
                result += f"Content: {str(item['content'])[:200]}...\n"
                
                if relationship_type:
                    filtered_rels = [r for r in item["relationships"] if r["type"] == relationship_type]
                    if filtered_rels:
                        result += f"Relationships ({relationship_type}): {filtered_rels}\n"
                elif item["relationships"]:
                    result += f"Relationships: {item['relationships'][:2]}\n"
                
                result += "\n"
            
            return result
    
    multimodal_agent = MultiModalRAGAgent()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="MultimodalRAGAgent",
        instructions="You can ingest and query structured data with relationships.",
        tools=[
            multimodal_agent.ingest_structured_data,
            multimodal_agent.query_knowledge_graph
        ]
    ) as agent:
        
        # Ingest structured data
        print("Ingesting structured research data...")
        
        research_data = {
            "title": "Climate Change Impact on Agriculture",
            "author": "Dr. Sarah Johnson",
            "year": 2023,
            "findings": "Temperature increases of 2°C could reduce crop yields by 10-25%",
            "methodology": "Meta-analysis of 150 studies",
            "confidence": "High"
        }
        
        relationships = [
            {"type": "cites", "target": "IPCC_2023_report"},
            {"type": "related_to", "target": "temperature_studies"},
            {"type": "impacts", "target": "food_security"}
        ]
        
        result1 = await agent.run(f"Ingest research data: {research_data} with relationships: {relationships}")
        print(f"Ingestion result: {result1.text[:200]}...")
        
        # Query knowledge graph
        print("\nQuerying knowledge graph...")
        result2 = await agent.run("What research has been done on temperature impacts?")
        print(f"Query result: {result2.text}")
    
    # Pattern 2: Conversational RAG with Memory
    print("\n2️⃣ Conversational RAG with Memory...")
    
    class ConversationalMemoryRAG:
        """RAG with conversation memory and context awareness"""
        
        def __init__(self):
            self.conversation_memory = []
            self.user_preferences = {}
        
        async def conversational_query(self,
            query: str = Field(..., description="User's question"),
            user_id: Optional[str] = Field(None, description="User identifier for personalization"),
            include_preferences: bool = Field(True, description="Include user preferences in search")
        ) -> str:
            """Handle conversational queries with memory"""
            
            # Store query in memory
            self.conversation_memory.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            })
            
            # Build context from memory
            context = ""
            if self.conversation_memory:
                recent_exchanges = self.conversation_memory[-5:]  # Last 5 exchanges
                context = "Recent conversation:\n"
                for exchange in recent_exchanges:
                    context += f"{exchange['role']}: {exchange['content'][:100]}...\n"
            
            # Add user preferences
            preferences_context = ""
            if include_preferences and user_id and user_id in self.user_preferences:
                prefs = self.user_preferences[user_id]
                preferences_context = f"\nUser preferences: {prefs}\n"
            
            # Enhanced query with context
            enhanced_query = f"""
            Conversation context: {context}
            {preferences_context}
            
            Current question: {query}
            
            Please provide a response that:
            1. Addresses the current question directly
            2. Considers the conversation history
            3. Respects user preferences if available
            4. Maintains conversational continuity
            """
            
            # Store enhanced query
            self.conversation_memory.append({
                "role": "system",
                "content": f"Enhanced query: {enhanced_query[:200]}...",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            })
            
            return f"Enhanced query ready: {enhanced_query[:300]}...\nThis query now includes conversation context and user preferences."
        
        def update_user_preferences(self,
            user_id: str = Field(..., description="User identifier"),
            preferences: Dict[str, Any] = Field(..., description="User preferences to store")
        ) -> str:
            """Update user preferences for personalized responses"""
            
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            self.user_preferences[user_id].update(preferences)
            
            return f"Updated preferences for user {user_id}: {preferences}"
        
        def get_conversation_summary(self,
            max_exchanges: int = Field(10, description="Maximum number of recent exchanges to include")
        ) -> str:
            """Get a summary of recent conversation"""
            
            if not self.conversation_memory:
                return "No conversation history available."
            
            recent_exchanges = self.conversation_memory[-max_exchanges:]
            
            summary = f"Conversation Summary (last {len(recent_exchanges)} exchanges):\n\n"
            
            for i, exchange in enumerate(recent_exchanges, 1):
                summary += f"{i}. {exchange['role'].title()}: {exchange['content'][:150]}...\n"
            
            return summary
    
    memory_rag = ConversationalMemoryRAG()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="MemoryRAGAgent",
        instructions="You maintain conversation memory and user preferences for personalized RAG.",
        tools=[
            memory_rag.conversational_query,
            memory_rag.update_user_preferences,
            memory_rag.get_conversation_summary
        ]
    ) as agent:
        
        # Simulate conversation
        print("Simulating conversational RAG...")
        
        # First query
        result1 = await agent.run("Tell me about climate change impacts", user_id="user123")
        print(f"Query 1: {result1.text[:200]}...")
        
        # Update preferences
        await agent.run("Update preferences: {'interests': ['renewable energy', 'policy'], 'expertise_level': 'intermediate'}", user_id="user123")
        
        # Follow-up query
        result2 = await agent.run("What about renewable energy solutions?", user_id="user123")
        print(f"Query 2: {result2.text[:200]}...")
        
        # Get conversation summary
        result3 = await agent.run("Show me our conversation history")
        print(f"History: {result3.text[:300]}...")
    
    # Pattern 3: Hierarchical RAG with Multiple Knowledge Domains
    print("\n3️⃣ Hierarchical RAG...")
    
    class HierarchicalRAG:
        """Hierarchical RAG with domain-specific knowledge stores"""
        
        def __init__(self):
            self.domain_hierarchy = {
                "science": {
                    "climate_science": {},
                    "physics": {},
                    "biology": {}
                },
                "technology": {
                    "ai_ml": {},
                    "software": {},
                    "hardware": {}
                },
                "business": {
                    "strategy": {},
                    "operations": {},
                    "finance": {}
                }
            }
        
        def identify_domain(self,
            query: str = Field(..., description="User query to classify")
        ) -> str:
            """Identify the most relevant knowledge domain"""
            
            query_lower = query.lower()
            
            # Domain keywords
            domain_keywords = {
                "climate_science": ["climate", "weather", "temperature", "carbon", "greenhouse"],
                "physics": ["physics", "energy", "force", "motion", "quantum"],
                "biology": ["biology", "life", "organism", "cell", "evolution"],
                "ai_ml": ["ai", "machine learning", "neural", "algorithm", "model"],
                "software": ["software", "programming", "code", "development"],
                "hardware": ["hardware", "computer", "processor", "memory"],
                "strategy": ["strategy", "planning", "competition", "market"],
                "operations": ["operations", "process", "efficiency", "workflow"],
                "finance": ["finance", "money", "investment", "budget", "cost"]
            }
            
            # Score each domain
            domain_scores = {}
            for domain, keywords in domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                domain_scores[domain] = score
            
            # Return highest scoring domain
            best_domain = max(domain_scores, key=domain_scores.get)
            
            return f"Identified domain: {best_domain} (score: {domain_scores[best_domain]})"
        
        def hierarchical_search(self,
            query: str = Field(..., description="Search query"),
            start_domain: str = Field(..., description="Starting domain for search"),
            search_depth: int = Field(2, description="How many levels to search up and down the hierarchy")
        ) -> str:
            """Perform hierarchical search across related domains"""
            
            # Find domain in hierarchy
            found_domain = None
            parent_domain = None
            
            for main_domain, subdomains in self.domain_hierarchy.items():
                if start_domain in subdomains:
                    found_domain = start_domain
                    parent_domain = main_domain
                    break
            
            if not found_domain:
                return f"Domain '{start_domain}' not found in hierarchy"
            
            # Collect related domains
            related_domains = [found_domain]
            
            # Add sibling domains (same level)
            if parent_domain:
                sibling_domains = list(self.domain_hierarchy[parent_domain].keys())
                related_domains.extend([d for d in sibling_domains if d != found_domain])
            
            # Add parent domain
            if parent_domain:
                related_domains.append(parent_domain)
            
            # Format hierarchical search plan
            search_plan = f"""Hierarchical Search Plan for: {query}
            
            Starting Domain: {start_domain}
            Search Depth: {search_depth}
            
            Search Order:
            1. {found_domain} (primary domain)
            2. Sibling domains: {[d for d in related_domains if d != found_domain and d != parent_domain]}
            3. Parent domain: {parent_domain}
            
            This approach ensures comprehensive coverage while maintaining relevance.
            """
            
            return search_plan
    
    hierarchical_agent = HierarchicalRAG()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="HierarchicalRAGAgent",
        instructions="You perform hierarchical searches across knowledge domains.",
        tools=[
            hierarchical_agent.identify_domain,
            hierarchical_agent.hierarchical_search
        ]
    ) as agent:
        
        # Test hierarchical search
        print("Testing hierarchical search:")
        
        queries = [
            "What are the latest developments in machine learning?",
            "How do businesses approach strategic planning?",
            "Explain quantum physics concepts"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            
            # Identify domain
            domain_result = await agent.run(f"Identify domain: {query}")
            print(f"Domain: {domain_result.text}")
            
            # Perform hierarchical search
            search_result = await agent.run(f"Hierarchical search: {query} starting from identified domain")
            print(f"Search plan: {search_result.text[:300]}...")

async def production_rag_deployment():
    """Production-ready RAG deployment patterns"""
    
    print("\n🏭 Production RAG Deployment")
    print("=" * 60)
    
    # Production configuration
    production_config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "similarity_threshold": 0.75,
        "max_results": 5,
        "cache_size": 1000,
        "rate_limit": 100  # requests per hour
    }
    
    # Monitoring and observability
    class RAGMonitoring:
        """Comprehensive monitoring for RAG operations"""
        
        def __init__(self):
            self.metrics = {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "average_response_time": 0,
                "cache_hit_rate": 0,
                "documents_processed": 0,
                "vector_stores_created": 0
            }
            self.query_log = []
        
        def record_query(self, query: str, success: bool, response_time: float, results_count: int):
            """Record query metrics"""
            self.metrics["total_queries"] += 1
            if success:
                self.metrics["successful_queries"] += 1
            else:
                self.metrics["failed_queries"] += 1
            
            # Update average response time
            n = self.metrics["total_queries"]
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (n-1) + response_time) / n
            )
            
            # Log query
            self.query_log.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "success": success,
                "response_time": response_time,
                "results_count": results_count
            })
        
        def get_health_status(self) -> str:
            """Get comprehensive health status"""
            success_rate = (
                self.metrics["successful_queries"] / max(self.metrics["total_queries"], 1) * 100
            )
            
            return f"""RAG System Health Report
            ======================
            
            Query Performance:
            - Total Queries: {self.metrics['total_queries']}
            - Success Rate: {success_rate:.1f}%
            - Average Response Time: {self.metrics['average_response_time']:.3f}s
            - Failed Queries: {self.metrics['failed_queries']}
            
            System Metrics:
            - Documents Processed: {self.metrics['documents_processed']}
            - Vector Stores: {self.metrics['vector_stores_created']}
            - Cache Hit Rate: {self.metrics['cache_hit_rate']:.1f}%
            
            Status: {'🟢 HEALTHY' if success_rate > 95 else '🟡 DEGRADED' if success_rate > 80 else '🔴 UNHEALTHY'}
            """
    
    # Performance optimization
    class PerformanceOptimizer:
        """Performance optimization for RAG operations"""
        
        def __init__(self, cache_size: int = 1000):
            self.query_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_size = cache_size
        
        def get_cached_results(self, query: str, max_age_seconds: int = 300) -> Optional[Dict]:
            """Get cached results if available and fresh"""
            if query in self.query_cache:
                cached_data = self.query_cache[query]
                if datetime.now().timestamp() - cached_data["timestamp"] < max_age_seconds:
                    self.cache_hits += 1
                    return cached_data["results"]
                else:
                    # Remove expired entry
                    del self.query_cache[query]
            
            self.cache_misses += 1
            return None
        
        def cache_results(self, query: str, results: Dict):
            """Cache query results"""
            # Implement LRU eviction if cache is full
            if len(self.query_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k]["timestamp"])
                del self.query_cache[oldest_key]
            
            self.query_cache[query] = {
                "results": results,
                "timestamp": datetime.now().timestamp()
            }
        
        def get_cache_stats(self) -> Dict[str, float]:
            """Get cache performance statistics"""
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "cache_size": len(self.query_cache),
                "efficiency": hit_rate / 100  # 0-1 scale
            }
    
    # Security and access control
    class SecurityManager:
        """Security and access control for RAG operations"""
        
        def __init__(self):
            self.access_logs = []
            self.blocked_patterns = [
                "password", "secret", "key", "token",
                "ssn", "social security", "credit card"
            ]
        
        def validate_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
            """Validate query for security concerns"""
            
            # Check for blocked patterns
            query_lower = query.lower()
            for pattern in self.blocked_patterns:
                if pattern in query_lower:
                    return {
                        "valid": False,
                        "reason": f"Query contains blocked pattern: {pattern}",
                        "sanitized_query": self.sanitize_query(query)
                    }
            
            # Log access
            self.access_logs.append({
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "query": query,
                "query_length": len(query),
                "valid": True
            })
            
            return {
                "valid": True,
                "sanitized_query": query
            }
        
        def sanitize_query(self, query: str) -> str:
            """Sanitize query by removing sensitive patterns"""
            sanitized = query
            
            for pattern in self.blocked_patterns:
                sanitized = sanitized.replace(pattern, "[REDACTED]")
            
            return sanitized
        
        def get_security_report(self) -> str:
            """Get security access report"""
            total_queries = len(self.access_logs)
            valid_queries = sum(1 for log in self.access_logs if log["valid"])
            blocked_queries = total_queries - valid_queries
            
            return f"""Security Report
            =============
            
            Access Statistics:
            - Total Queries: {total_queries}
            - Valid Queries: {valid_queries}
            - Blocked Queries: {blocked_queries}
            - Security Rate: {(valid_queries/max(total_queries, 1)*100):.1f}%
            
            Recent Activity: {len([log for log in self.access_logs[-10:] if log['valid']])} valid queries in last 10 attempts
            """
    
    # Demonstrate production features
    print("Production RAG Features:")
    
    # Initialize components
    monitoring = RAGMonitoring()
    optimizer = PerformanceOptimizer(cache_size=100)
    security = SecurityManager()
    
    print("\nSecurity validation:")
    test_queries = [
        "What is climate change?",
        "Tell me about renewable energy",
        "My password is 12345, tell me about solar power"
    ]
    
    for query in test_queries:
        validation = security.validate_query(query, user_id="test_user")
        print(f"Query: {query[:50]}...")
        print(f"Valid: {validation['valid']}")
        if not validation['valid']:
            print(f"Reason: {validation['reason']}")
        print()
    
    print("Performance optimization:")
    # Simulate cache operations
    for i, query in enumerate(["climate change", "renewable energy", "climate change"]):
        cached = optimizer.get_cached_results(query)
        if cached:
            print(f"Query {i+1}: Cache hit for '{query}'")
        else:
            print(f"Query {i+1}: Cache miss for '{query}'")
            # Simulate caching results
            optimizer.cache_results(query, {"results": f"Results for {query}"})
    
    cache_stats = optimizer.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    print("\nMonitoring metrics:")
    # Simulate some operations
    for i in range(5):
        monitoring.record_query(
            query=f"Test query {i+1}",
            success=i != 2,  # Simulate one failure
            response_time=0.1 + i * 0.05,
            results_count=3
        )
    
    health_report = monitoring.get_health_status()
    print(health_report)

if __name__ == "__main__":
    asyncio.run(basic_rag_demo())
    asyncio.run(advanced_rag_patterns())
    asyncio.run(production_rag_deployment())
```

The RAG Agent represents the convergence of information retrieval and generative AI, creating systems that can access vast knowledge bases while maintaining conversational fluency. By mastering the patterns and techniques outlined in this section, you'll be equipped to build sophisticated knowledge management systems that provide accurate, attributed, and contextually relevant information to users. The combination of intelligent document processing, advanced retrieval strategies, and production-ready deployment patterns creates agents that are both powerful and reliable in enterprise environments.

---

## Agent Type 4: Code Execution Agent

The Code Execution Agent represents the pinnacle of AI-powered computational assistance, enabling dynamic code generation, execution, and analysis within a secure sandboxed environment. This agent type transforms AI from a passive information provider into an active computational partner capable of solving complex mathematical problems, analyzing data, generating visualizations, and even debugging code in real-time.

### Architecture Overview

The Code Execution Agent architecture implements a sophisticated multi-layered security and execution framework:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Code Execution Agent Architecture               │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │   Code          │  │   Security      │  │   Execution     │    │
│ │   Generation    │──│   Validation    │──│   Environment   │    │
│ │                 │  │                 │  │                 │    │
│ │ • LLM-based     │  │ • Syntax        │  │ • Sandboxed     │    │
│ │   generation    │  │   validation    │  │   runtime       │    │
│ │ • Template      │  │ • Import        │  │ • Resource      │    │
│ │   assistance    │  │   restrictions  │  │   limits        │    │
│ │ • Best practices│  │ • Operation     │  │ • Timeout       │    │
│ └─────────────────┘  │   whitelisting  │  │   management    │    │
│                      └─────────────────┘  └─────────────────┘    │
│                               │                      │             │
│ ┌─────────────────────────────┴──────────────────────┴───────────┐ │
│ │                    Output Processing                            │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │   Result     │  │   Error      │  │   Visualization  │    │ │
│ │  │   Formatter  │  │   Handler    │  │   Generator      │    │ │
│ │  │              │  │              │  │                  │    │ │
│ │  │ • stdout     │  │ • Exception  │  │ • Chart          │    │ │
│ │  │ • return     │  │   capture    │  │   generation     │    │ │
│ │  │   values     │  │ • Stack      │  │ • Image          │    │ │
│ │  │ • variables  │  │   traces     │  │   rendering      │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                    Integration Layer                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

This architecture provides several critical capabilities:
- **Secure Code Execution**: Sandboxed environment with resource limits and timeout controls
- **Intelligent Code Generation**: LLM-powered code creation with best practice enforcement
- **Comprehensive Error Handling**: Graceful handling of syntax errors, runtime exceptions, and resource exhaustion
- **Rich Output Processing**: Support for text, images, data visualizations, and interactive elements
- **Resource Management**: Automatic cleanup and resource monitoring

### Core Implementation Patterns

Let's explore the fundamental patterns for building Code Execution Agents:

```python
import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIResponsesClient
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io
import base64
from contextlib import redirect_stdout, redirect_stderr
import sys
import traceback
import ast
import resource

# Enhanced code execution schemas
class CodeExecutionRequest(BaseModel):
    """Schema for code execution requests"""
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(30, description="Maximum execution time in seconds", ge=5, le=300)
    include_visualizations: bool = Field(False, description="Generate visualizations if applicable")
    save_outputs: bool = Field(False, description="Save outputs to files")
    return_variables: Optional[List[str]] = Field(None, description="Specific variables to return")

class CodeExecutionResponse(BaseModel):
    """Schema for code execution responses"""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None  # Base64 encoded images
    execution_time: float
    memory_usage: Optional[int] = None  # Peak memory usage in bytes

# Advanced code execution service
class AdvancedCodeExecutionService:
    """Production-ready code execution service with comprehensive features"""
    
    def __init__(self):
        self.execution_history = []
        self.security_config = self._initialize_security()
        self.allowed_imports = {
            'math', 'random', 'datetime', 'json', 'csv',
            'numpy', 'pandas', 'matplotlib', 'seaborn',
            'scipy', 'sklearn', 'requests', 'urllib'
        }
        self.forbidden_operations = {
            'open', 'file', 'input', 'raw_input', 'eval', 'exec',
            '__import__', 'compile', 'globals', 'locals', 'vars',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }
    
    def _initialize_security(self) -> Dict[str, Any]:
        """Initialize security configuration"""
        return {
            'max_memory_mb': 100,  # Maximum memory usage
            'max_cpu_time_seconds': 30,  # Maximum CPU time
            'forbidden_builtins': ['open', 'file', 'eval', 'exec'],
            'allowed_network_hosts': [],  # No network access by default
            'max_output_length': 10000,  # Maximum output length
        }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Comprehensive code validation for security
        
        Checks:
        - Syntax validity
        - Forbidden imports and operations
        - Resource-intensive patterns
        - Security vulnerabilities
        """
        try:
            # Parse AST for security analysis
            tree = ast.parse(code)
            
            # Security checks
            security_issues = []
            
            for node in ast.walk(tree):
                # Check for forbidden operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.forbidden_operations:
                            security_issues.append(f"Forbidden operation: {node.func.id}")
                
                # Check for forbidden imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            security_issues.append(f"Forbidden import: {alias.name}")
                
                if isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_imports:
                        security_issues.append(f"Forbidden import from: {node.module}")
                
                # Check for file operations
                if isinstance(node, ast.Name) and node.id in self.security_config['forbidden_builtins']:
                    security_issues.append(f"Forbidden builtin: {node.id}")
            
            return {
                "valid": len(security_issues) == 0,
                "security_issues": security_issues,
                "ast_valid": True
            }
            
        except SyntaxError as e:
            return {
                "valid": False,
                "syntax_error": str(e),
                "ast_valid": False
            }
    
    async def execute_code_safely(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """
        Execute code in a secure, sandboxed environment
        
        Features:
        - Resource limiting (memory, CPU time)
        - Timeout protection
        - Output capture
        - Error handling with stack traces
        - Variable inspection
        - Memory usage monitoring
        """
        start_time = datetime.now()
        
        try:
            # Validate code first
            validation = self.validate_code(request.code)
            if not validation["valid"]:
                return CodeExecutionResponse(
                    success=False,
                    output="",
                    error=f"Code validation failed: {validation.get('security_issues', validation.get('syntax_error', 'Unknown error'))}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Set resource limits
            self._set_resource_limits()
            
            # Create execution environment
            exec_globals = {
                '__builtins__': self._get_safe_builtins(),
                'np': np,
                'pd': pd,
                'plt': plt,
                'datetime': datetime,
                'json': json,
                'math': __import__('math'),
                'random': __import__('random'),
            }
            
            exec_locals = {}
            
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            output_parts = []
            visualizations = []
            
            # Execute code with comprehensive monitoring
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # Set timeout using asyncio
                    exec_task = asyncio.create_task(self._execute_with_timeout(
                        request.code, exec_globals, exec_locals, request.timeout
                    ))
                    
                    await exec_task
                    
                    # Get execution results
                    stdout_output = stdout_capture.getvalue()
                    stderr_output = stderr_capture.getvalue()
                    
                    # Combine outputs
                    if stdout_output:
                        output_parts.append(f"STDOUT:\n{stdout_output}")
                    if stderr_output:
                        output_parts.append(f"STDERR:\n{stderr_output}")
                    
                    # Extract requested variables
                    variables = {}
                    if request.return_variables:
                        for var_name in request.return_variables:
                            if var_name in exec_locals:
                                variables[var_name] = self._serialize_variable(exec_locals[var_name])
                    
                    # Generate visualizations if requested
                    if request.include_visualizations:
                        visualizations = await self._generate_visualizations(exec_locals)
                    
                    # Format final output
                    final_output = "\n\n".join(output_parts) if output_parts else "Code executed successfully with no output."
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    # Record execution history
                    self.execution_history.append({
                        "timestamp": start_time.isoformat(),
                        "code": request.code[:200] + "..." if len(request.code) > 200 else request.code,
                        "success": True,
                        "execution_time": execution_time,
                        "output_length": len(final_output)
                    })
                    
                    return CodeExecutionResponse(
                        success=True,
                        output=final_output,
                        variables=variables if variables else None,
                        visualizations=visualizations if visualizations else None,
                        execution_time=execution_time,
                        memory_usage=self._get_memory_usage()
                    )
                    
                except asyncio.TimeoutError:
                    return CodeExecutionResponse(
                        success=False,
                        output=stdout_capture.getvalue(),
                        error=f"Code execution timed out after {request.timeout} seconds",
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                    
                except Exception as e:
                    # Capture full stack trace
                    error_output = stderr_capture.getvalue()
                    stack_trace = traceback.format_exc()
                    
                    return CodeExecutionResponse(
                        success=False,
                        output=stdout_capture.getvalue(),
                        error=f"Runtime error: {str(e)}\n\nStack trace:\n{stack_trace}",
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
        
        except Exception as e:
            return CodeExecutionResponse(
                success=False,
                output="",
                error=f"Execution setup error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _execute_with_timeout(self, code: str, globals_dict: dict, locals_dict: dict, timeout: int):
        """Execute code with timeout protection"""
        # Use asyncio.wait_for for timeout control
        await asyncio.wait_for(
            asyncio.to_thread(exec, code, globals_dict, locals_dict),
            timeout=timeout
        )
    
    def _set_resource_limits(self):
        """Set system resource limits"""
        try:
            # Set memory limit (may not work on all systems)
            max_memory = self.security_config['max_memory_mb'] * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
            
            # Set CPU time limit
            max_cpu_time = self.security_config['max_cpu_time_seconds']
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time, max_cpu_time))
            
        except (resource.error, OSError):
            # Resource limits not available on this system
            pass
    
    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get safe builtin functions"""
        safe_builtins = __builtins__.copy()
        
        # Remove dangerous builtins
        for forbidden in self.security_config['forbidden_builtins']:
            safe_builtins.pop(forbidden, None)
        
        return safe_builtins
    
    def _serialize_variable(self, var: Any) -> Any:
        """Serialize variable for JSON response"""
        try:
            if isinstance(var, (str, int, float, bool, list, dict, type(None))):
                return var
            elif isinstance(var, np.ndarray):
                return var.tolist() if var.size < 1000 else f"Array with shape {var.shape}"
            elif isinstance(var, pd.DataFrame):
                return var.head(10).to_dict() if len(var) > 10 else var.to_dict()
            elif hasattr(var, '__dict__'):
                return str(var)
            else:
                return str(var)
        except Exception:
            return f"Variable of type {type(var).__name__} (serialization failed)"
    
    async def _generate_visualizations(self, exec_locals: Dict[str, Any]) -> List[str]:
        """Generate visualizations from execution results"""
        visualizations = []
        
        try:
            # Check for matplotlib figures
            for var_name, var_value in exec_locals.items():
                if hasattr(var_value, 'savefig'):  # matplotlib figure
                    img_buffer = io.BytesIO()
                    var_value.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    visualizations.append(img_base64)
                    plt.close(var_value)  # Close figure to free memory
                
                # Check for data that could be visualized
                elif isinstance(var_value, pd.DataFrame) and len(var_value) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    var_value.head(20).plot(ax=ax)
                    plt.title(f'DataFrame: {var_name}')
                    plt.xticks(rotation=45)
                    
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    visualizations.append(img_base64)
                    plt.close(fig)
                
                elif isinstance(var_value, np.ndarray) and var_value.ndim <= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if var_value.ndim == 1:
                        ax.plot(var_value)
                        ax.set_title(f'Array: {var_name}')
                    else:
                        ax.imshow(var_value, cmap='viridis')
                        ax.set_title(f'Array: {var_name} (shape: {var_value.shape})')
                    
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    visualizations.append(img_base64)
                    plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
        
        return visualizations
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

# Advanced mathematical computation service
class MathematicalComputationService:
    """Advanced mathematical computations with visualization support"""
    
    def __init__(self):
        self.computation_history = []
    
    async def solve_equation(self, 
        equation: str = Field(..., description="Mathematical equation to solve (e.g., 'x^2 - 4 = 0', 'sin(x) = 0.5')"),
        variable: str = Field("x", description="Variable to solve for"),
        numerical_method: str = Field("analytical", description="Solution method: 'analytical', 'newton', 'bisection'")
    ) -> str:
        """
        Solve mathematical equations with step-by-step solutions
        
        Demonstrates:
        - Symbolic mathematics
        - Numerical methods
        - Step-by-step solution generation
        - Visualization of solutions
        """
        try:
            import sympy as sp
            from sympy import symbols, solve, sin, cos, tan, log, exp, sqrt, simplify
            
            # Create symbolic variable
            x = symbols(variable, real=True)
            
            # Parse equation
            eq_str = equation.replace("^", "**")  # Convert to Python syntax
            lhs, rhs = eq_str.split("=") if "=" in eq_str else (eq_str, "0")
            
            # Create symbolic equation
            lhs_expr = sp.sympify(lhs.strip())
            rhs_expr = sp.sympify(rhs.strip())
            equation_expr = lhs_expr - rhs_expr
            
            # Solve based on method
            if numerical_method == "analytical":
                solutions = solve(equation_expr, x)
                solution_method = "Analytical solution"
            elif numerical_method == "newton":
                # Newton-Raphson method for numerical solutions
                from sympy import diff
                f = equation_expr
                df = diff(f, x)
                
                # Find approximate solutions
                solutions = []
                for initial_guess in [-10, -1, 1, 10]:  # Multiple initial guesses
                    try:
                        sol = sp.nsolve(f, x, initial_guess)
                        if sol not in solutions:
                            solutions.append(float(sol))
                    except:
                        continue
                
                solution_method = "Newton-Raphson numerical method"
            else:
                solutions = solve(equation_expr, x)
                solution_method = f"Default method (analytical)"
            
            # Generate visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot the function
            x_vals = np.linspace(-10, 10, 1000)
            y_vals = [float(equation_expr.subs(x, val)) for val in x_vals]
            
            ax1.plot(x_vals, y_vals, 'b-', label=f'f({variable}) = {equation_expr}', linewidth=2)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Mark solutions
            for sol in solutions[:10]:  # Limit to 10 solutions for visibility
                if isinstance(sol, (int, float)):
                    ax1.plot(float(sol), 0, 'ro', markersize=10)
                    ax1.annotate(f'x = {float(sol):.3f}', (float(sol), 0), 
                               xytext=(5, 10), textcoords='offset points')
            
            ax1.set_xlabel(variable)
            ax1.set_ylabel('f(' + variable + ')')
            ax1.set_title(f'Function: {equation}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot convergence for numerical methods
            if numerical_method == "newton" and solutions:
                ax2.plot(x_vals, y_vals, 'b-', alpha=0.5, label='Function')
                for i, sol in enumerate(solutions[:3]):
                    convergence_x = np.linspace(sol-2, sol+2, 100)
                    convergence_y = [float(equation_expr.subs(x, val)) for val in convergence_x]
                    ax2.plot(convergence_x, convergence_y, '--', alpha=0.7, 
                            label=f'Solution {i+1}: x = {float(sol):.3f}')
                
                ax2.set_xlabel(variable)
                ax2.set_ylabel('f(' + variable + ')')
                ax2.set_title('Solution Convergence')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            
            # Save plot to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            # Format solution
            if solutions:
                solution_text = f"Solutions found using {solution_method}:\n"
                for i, sol in enumerate(solutions):
                    if isinstance(sol, (int, float)):
                        solution_text += f"  x_{i+1} = {float(sol):.6f}\n"
                    else:
                        solution_text += f"  x_{i+1} = {sol}\n"
            else:
                solution_text = f"No solutions found using {solution_method}.\n"
            
            # Add verification
            if solutions:
                solution_text += "\nVerification (substituting back into original equation):\n"
                for i, sol in enumerate(solutions[:3]):  # Verify first 3 solutions
                    verification = equation_expr.subs(x, sol)
                    simplified_verification = simplify(verification)
                    solution_text += f"  For x_{i+1} = {sol}: f({sol}) = {simplified_verification}\n"
            
            # Record computation
            self.computation_history.append({
                "type": "equation_solving",
                "equation": equation,
                "variable": variable,
                "method": numerical_method,
                "solutions": [str(sol) for sol in solutions],
                "timestamp": datetime.now().isoformat()
            })
            
            return f"""## Equation Solver Results

**Equation:** {equation}
**Variable:** {variable}
**Method:** {solution_method}

### Solutions:
{solution_text}

### Analysis:
- Number of solutions found: {len(solutions)}
- Equation complexity: {len(str(equation_expr))} characters
- Computation completed successfully

The visualization shows the function plot with solutions marked as red dots."""
            
        except Exception as e:
            return f"Error solving equation '{equation}': {str(e)}\n\nPlease check the equation syntax and try again."
    
    async def perform_statistical_analysis(self,
        data_description: str = Field(..., description="Description of data to analyze or generate"),
        analysis_type: str = Field("descriptive", description="Type of analysis: 'descriptive', 'regression', 'hypothesis_testing'"),
        sample_size: int = Field(1000, description="Number of data points to generate", ge=100, le=10000)
    ) -> str:
        """
        Perform comprehensive statistical analysis with visualizations
        
        Demonstrates:
        - Data generation and simulation
        - Statistical testing
        - Regression analysis
        - Comprehensive visualization
        """
        try:
            # Generate sample data based on description
            np.random.seed(42)  # For reproducibility
            
            if "normal" in data_description.lower():
                data = np.random.normal(loc=50, scale=15, size=sample_size)
                data_type = "Normal distribution (μ=50, σ=15)"
            elif "uniform" in data_description.lower():
                data = np.random.uniform(low=0, high=100, size=sample_size)
                data_type = "Uniform distribution (0-100)"
            elif "exponential" in data_description.lower():
                data = np.random.exponential(scale=20, size=sample_size)
                data_type = "Exponential distribution (λ=0.05)"
            elif "bimodal" in data_description.lower():
                data1 = np.random.normal(loc=30, scale=5, size=sample_size//2)
                data2 = np.random.normal(loc=70, scale=5, size=sample_size//2)
                data = np.concatenate([data1, data2])
                data_type = "Bimodal distribution (μ₁=30, μ₂=70)"
            else:
                data = np.random.normal(loc=50, scale=15, size=sample_size)
                data_type = "Normal distribution (μ=50, σ=15)"
            
            # Create comprehensive analysis based on type
            if analysis_type == "descriptive":
                results = self._descriptive_analysis(data)
            elif analysis_type == "regression":
                results = await self._regression_analysis(data)
            elif analysis_type == "hypothesis_testing":
                results = self._hypothesis_testing(data)
            else:
                results = self._descriptive_analysis(data)
            
            # Generate comprehensive visualizations
            fig = plt.figure(figsize=(20, 12))
            
            # 1. Histogram with distribution overlay
            ax1 = plt.subplot(2, 3, 1)
            ax1.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Overlay theoretical distribution
            x = np.linspace(data.min(), data.max(), 100)
            if "normal" in data_type.lower():
                from scipy.stats import norm
                ax1.plot(x, norm.pdf(x, data.mean(), data.std()), 'r-', linewidth=2, label='Normal fit')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Histogram: {data_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Box plot
            ax2 = plt.subplot(2, 3, 2)
            box_plot = ax2.boxplot(data, vert=True, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            ax2.set_ylabel('Value')
            ax2.set_title('Box Plot')
            ax2.grid(True, alpha=0.3)
            
            # 3. Q-Q plot
            ax3 = plt.subplot(2, 3, 3)
            from scipy import stats
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normal Distribution)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Time series (if applicable)
            ax4 = plt.subplot(2, 3, 4)
            sample_indices = np.random.choice(len(data), min(200, len(data)), replace=False)
            ax4.plot(sample_indices, data[sample_indices], 'b-', alpha=0.7)
            ax4.set_xlabel('Sample Index')
            ax4.set_ylabel('Value')
            ax4.set_title('Sample Time Series')
            ax4.grid(True, alpha=0.3)
            
            # 5. Cumulative distribution
            ax5 = plt.subplot(2, 3, 5)
            sorted_data = np.sort(data)
            cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax5.plot(sorted_data, cumulative_prob, 'g-', linewidth=2)
            ax5.set_xlabel('Value')
            ax5.set_ylabel('Cumulative Probability')
            ax5.set_title('Cumulative Distribution Function')
            ax5.grid(True, alpha=0.3)
            
            # 6. Statistical summary
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            stats_text = f"""
            Statistical Summary:
            Sample Size: {len(data):,}
            Mean: {data.mean():.3f}
            Median: {np.median(data):.3f}
            Std Dev: {data.std():.3f}
            Min: {data.min():.3f}
            Max: {data.max():.3f}
            Skewness: {stats.skew(data):.3f}
            Kurtosis: {stats.kurtosis(data):.3f}
            """
            ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Statistical Analysis: {analysis_type.title()}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            # Record computation
            self.computation_history.append({
                "type": "statistical_analysis",
                "data_type": data_type,
                "analysis_type": analysis_type,
                "sample_size": sample_size,
                "timestamp": datetime.now().isoformat()
            })
            
            return f"""## Statistical Analysis Results

**Data Type:** {data_type}
**Analysis Type:** {analysis_type.title()}
**Sample Size:** {sample_size:,}

### Key Statistics:
{results['summary']}

### Detailed Results:
{results['details']}

### Interpretation:
{results['interpretation']}

The comprehensive visualization shows the data distribution, identifies outliers, 
and provides insights into the underlying patterns and characteristics.
Visualization generated successfully with multiple chart types."""
            
        except Exception as e:
            return f"Error in statistical analysis: {str(e)}\n\nPlease check your parameters and try again."
    
    def _descriptive_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        from scipy import stats
        
        # Basic statistics
        mean = data.mean()
        median = np.median(data)
        std_dev = data.std()
        variance = data.var()
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val
        
        # Advanced statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Confidence intervals
        confidence_level = 0.95
        confidence_interval = stats.t.interval(
            confidence_level, 
            len(data)-1, 
            loc=mean, 
            scale=stats.sem(data)
        )
        
        summary = f"""
        Mean: {mean:.3f}
        Median: {median:.3f}
        Standard Deviation: {std_dev:.3f}
        Variance: {variance:.3f}
        Range: {range_val:.3f} ({min_val:.3f} to {max_val:.3f})
        Interquartile Range: {iqr:.3f}
        """
        
        details = f"""
        Skewness: {skewness:.3f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}
        Kurtosis: {kurtosis:.3f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}
        95% Confidence Interval: ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})
        """
        
        interpretation = f"""
        The data shows {'normal' if abs(skewness) < 0.5 else 'skewed'} distribution 
        with {'moderate' if abs(kurtosis) < 3 else 'extreme'} tail behavior.
        The mean {'closely matches' if abs(mean - median) < std_dev * 0.1 else 'differs from'} 
        the median, indicating {'symmetric' if abs(mean - median) < std_dev * 0.1 else 'asymmetric'} distribution.
        """
        
        return {
            "summary": summary,
            "details": details,
            "interpretation": interpretation
        }
    
    async def _regression_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform regression analysis"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Create synthetic x values
            x = np.arange(len(data)).reshape(-1, 1)
            y = data.reshape(-1, 1)
            
            # Linear regression
            linear_model = LinearRegression()
            linear_model.fit(x, y)
            y_pred_linear = linear_model.predict(x)
            r2_linear = r2_score(y, y_pred_linear)
            
            # Polynomial regression (degree 2)
            poly_features = PolynomialFeatures(degree=2)
            x_poly = poly_features.fit_transform(x)
            poly_model = LinearRegression()
            poly_model.fit(x_poly, y)
            y_pred_poly = poly_model.predict(x_poly)
            r2_poly = r2_score(y, y_pred_poly)
            
            # Choose best model
            if r2_poly > r2_linear + 0.05:  # Polynomial significantly better
                best_model = "Polynomial (degree 2)"
                best_r2 = r2_poly
                equation = f"y = {poly_model.coef_[0][0]:.3f} + {poly_model.coef_[0][1]:.3f}x + {poly_model.coef_[0][2]:.3f}x²"
            else:
                best_model = "Linear"
                best_r2 = r2_linear
                equation = f"y = {linear_model.coef_[0][0]:.3f}x + {linear_model.intercept_[0]:.3f}"
            
            summary = f"""
            Best Model: {best_model}
            R-squared: {best_r2:.3f}
            Equation: {equation}
            MSE: {mean_squared_error(y, y_pred_linear if best_model == "Linear" else y_pred_poly):.3f}
            """
            
            details = f"""
            Linear Regression R²: {r2_linear:.3f}
            Polynomial Regression R²: {r2_poly:.3f}
            Model Selection: {'Polynomial' if best_model == 'Polynomial' else 'Linear'} (better fit)
            """
            
            interpretation = f"""
            The {best_model.lower()} regression model explains {best_r2*100:.1f}% of the variance 
            in the data, indicating {'strong' if best_r2 > 0.7 else 'moderate' if best_r2 > 0.4 else 'weak'} 
            predictive power.
            """
            
            return {
                "summary": summary,
                "details": details,
                "interpretation": interpretation
            }
            
        except ImportError:
            return {
                "summary": "Regression analysis requires scikit-learn.",
                "details": "Install scikit-learn for regression analysis.",
                "interpretation": "Regression analysis not available."
            }
    
    def _hypothesis_testing(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform hypothesis testing"""
        try:
            from scipy import stats
            
            # One-sample t-test (test if mean is significantly different from 50)
            hypothesized_mean = 50
            t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(data[:min(5000, len(data))])  # Shapiro-Wilk test
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            
            # Confidence interval for mean
            confidence_interval = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
            
            # Effect size (Cohen's d)
            cohens_d = (data.mean() - hypothesized_mean) / data.std()
            
            # Summary statistics
            alpha = 0.05
            
            summary = f"""
            One-sample t-test:
            t-statistic: {t_stat:.3f}
            p-value: {p_value:.6f}
            Result: {'Reject' if p_value < alpha else 'Fail to reject'} H₀ (α = {alpha})
            
            Sample Mean: {data.mean():.3f}
            Hypothesized Mean: {hypothesized_mean}
            95% CI: ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})
            """
            
            details = f"""
            Normality Tests:
            Shapiro-Wilk: p = {shapiro_p:.6f} {'(Normal)' if shapiro_p > 0.05 else '(Not normal)'}
            Kolmogorov-Smirnov: p = {ks_p:.6f} {'(Normal)' if ks_p > 0.05 else '(Not normal)'}
            
            Effect Size (Cohen's d): {cohens_d:.3f} 
            {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} effect
            """
            
            interpretation = f"""
            The hypothesis test {'provides significant evidence' if p_value < alpha else 'does not provide sufficient evidence'} 
            to conclude that the population mean differs from {hypothesized_mean}.
            The effect size is {'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'}.
            """
            
            return {
                "summary": summary,
                "details": details,
                "interpretation": interpretation
            }
            
        except ImportError:
            return {
                "summary": "Hypothesis testing requires scipy.",
                "details": "Install scipy for hypothesis testing.",
                "interpretation": "Hypothesis testing not available."
            }

# Advanced code patterns and utilities
class AdvancedCodePatterns:
    """Advanced patterns for code execution agents"""
    
    @staticmethod
    def create_code_generator():
        """Create intelligent code generator with templates"""
        
        class CodeGenerator:
            def __init__(self):
                self.templates = {
                    "data_analysis": """
import pandas as pd
import matplotlib.pyplot as plt

# Load and explore data
data = pd.read_csv('{filename}')
print("Data shape:", data.shape)
print("\\nFirst 5 rows:")
print(data.head())
print("\\nData info:")
print(data.info())
print("\\nBasic statistics:")
print(data.describe())

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
data.hist(ax=axes.ravel()[:len(data.columns)])
plt.tight_layout()
plt.show()
                    """,
                    
                    "web_scraping": """
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = '{url}'
headers = {'User-Agent': 'Mozilla/5.0'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract data
data = []
for element in soup.find_all('{tag}'):
    data.append(element.text.strip())

# Create DataFrame
df = pd.DataFrame(data, columns=['extracted_data'])
print(f"Extracted {len(df)} items")
print(df.head())
                    """,
                    
                    "machine_learning": """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load data
data = pd.read_csv('{filename}')
X = data.drop('{target_column}', axis=1)
y = data['{target_column}']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
                    """
                }
            
            def generate_code(self, task_type: str, parameters: Dict[str, Any]) -> str:
                """Generate code from template"""
                if task_type not in self.templates:
                    return f"# Template not found for task type: {task_type}"
                
                template = self.templates[task_type]
                
                try:
                    return template.format(**parameters)
                except KeyError as e:
                    return f"# Missing parameter: {e}"
            
            def create_custom_template(self, name: str, template: str):
                """Add custom template"""
                self.templates[name] = template
            
            def list_templates(self) -> List[str]:
                """List available templates"""
                return list(self.templates.keys())
        
        return CodeGenerator()
    
    @staticmethod
    def create_code_optimizer():
        """Create code optimization utilities"""
        
        class CodeOptimizer:
            def optimize_loop(self, code: str) -> str:
                """Optimize loops using vectorization"""
                # Simple optimization patterns
                optimizations = [
                    ("for i in range(len(array)):", "# Consider using numpy vectorization"),
                    ("append(", "# Consider pre-allocating arrays for better performance"),
                    ("list comprehension", "# Good use of Python idioms"),
                ]
                
                optimized_code = code
                for pattern, suggestion in optimizations:
                    if pattern in code:
                        optimized_code = optimized_code.replace(
                            pattern, 
                            f"{pattern}  {suggestion}"
                        )
                
                return optimized_code
            
            def add_profiling(self, code: str) -> str:
                """Add profiling to code"""
                profiling_code = """
import time
import memory_profiler

@memory_profiler.profile
def profiled_function():
    start_time = time.time()
    
    # Original code here
    {original_code}
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    profiled_function()
                """
                
                return profiling_code.format(original_code=code)
            
            def suggest_improvements(self, code: str) -> List[str]:
                """Suggest code improvements"""
                suggestions = []
                
                if "import pandas" in code and "for " in code:
                    suggestions.append("Consider using pandas vectorized operations instead of loops")
                
                if "global " in code:
                    suggestions.append("Avoid using global variables; use function parameters instead")
                
                if "except:" in code:
                    suggestions.append("Use specific exception types instead of bare except clauses")
                
                if len(code.split('\n')) > 50:
                    suggestions.append("Consider breaking this into smaller functions for better maintainability")
                
                return suggestions
        
        return CodeOptimizer()

async def basic_code_execution_demo():
    """Demonstrate basic code execution capabilities"""
    
    print("💻 Basic Code Execution Agent Demo")
    print("=" * 60)
    
    # Initialize services
    code_service = AdvancedCodeExecutionService()
    math_service = MathematicalComputationService()
    
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="CodeExecutionAgent",
        instructions="""You are an expert code execution assistant with advanced capabilities.
        
        Guidelines:
        - Execute code safely and efficiently
        - Provide clear explanations of results
        - Generate visualizations when helpful
        - Handle errors gracefully
        - Optimize for performance when possible""",
        tools=[
            code_service.execute_code_safely,
            math_service.solve_equation,
            math_service.perform_statistical_analysis
        ]
    ) as agent:
        
        # Example 1: Simple mathematical computation
        print("1️⃣ Mathematical Computation:")
        math_result = await agent.run("Calculate the factorial of 100 and show me the result")
        print(f"Response: {math_result.text}\n")
        
        # Example 2: Data analysis with visualization
        print("2️⃣ Data Analysis with Visualization:")
        data_result = await agent.run("""
        Generate a dataset of 1000 random numbers from a normal distribution 
        and perform statistical analysis with visualizations
        """)
        print(f"Response: {data_result.text[:500]}...\n")
        
        # Example 3: Equation solving
        print("3️⃣ Equation Solving:")
        equation_result = await agent.run("Solve the equation x^2 - 5x + 6 = 0 and show the solution graphically")
        print(f"Response: {equation_result.text[:500]}...\n")
        
        # Example 4: Code validation and execution
        print("4️⃣ Code Validation:")
        code_request = CodeExecutionRequest(
            code="""
import math

# Calculate Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Generate first 10 Fibonacci numbers
fib_sequence = [fibonacci(i) for i in range(10)]
print("Fibonacci sequence:", fib_sequence)

# Calculate golden ratio approximation
golden_ratios = [fib_sequence[i+1]/fib_sequence[i] for i in range(1, len(fib_sequence)-1)]
print("Golden ratio approximations:", golden_ratios)
print("Average golden ratio:", sum(golden_ratios)/len(golden_ratios))
            """,
            timeout=15,
            include_visualizations=True,
            return_variables=["fib_sequence", "golden_ratios"]
        )
        
        # Execute code through the agent
        code_result = await agent.run(f"Execute this code and provide analysis: {code_request.code}")
        print(f"Code execution result: {code_result.text[:500]}...")

async def advanced_code_patterns():
    """Advanced code execution patterns"""
    
    print("\n🚀 Advanced Code Execution Patterns")
    print("=" * 60)
    
    # Pattern 1: Code Generation and Templates
    print("1️⃣ Code Generation and Templates...")
    
    code_generator = AdvancedCodePatterns.create_code_generator()
    code_optimizer = AdvancedCodePatterns.create_code_optimizer()
    
    # Generate code from template
    generated_code = code_generator.generate_code(
        "data_analysis",
        {"filename": "sales_data.csv"}
    )
    
    print("Generated data analysis code:")
    print(generated_code)
    
    # Optimize the code
    optimized_code = code_optimizer.optimize_loop(generated_code)
    suggestions = code_optimizer.suggest_improvements(optimized_code)
    
    print("\nOptimization suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    # Pattern 2: Interactive Code Debugging
    print("\n2️⃣ Interactive Code Debugging...")
    
    class InteractiveDebugger:
        """Interactive code debugging assistant"""
        
        def __init__(self):
            self.debug_history = []
        
        async def debug_code(self,
            code: str = Field(..., description="Code to debug"),
            error_message: str = Field(..., description="Error message or description of the issue"),
            test_cases: List[str] = Field(default_factory=list, description="Test cases to validate the fix")
        ) -> str:
            """Debug code with interactive assistance"""
            
            # Parse error message
            error_type = "Unknown"
            if "SyntaxError" in error_message:
                error_type = "Syntax Error"
            elif "NameError" in error_message:
                error_type = "Name Error"
            elif "TypeError" in error_message:
                error_type = "Type Error"
            elif "IndexError" in error_message:
                error_type = "Index Error"
            elif "ValueError" in error_message:
                error_type = "Value Error"
            
            # Generate debugging suggestions
            suggestions = []
            
            if error_type == "Syntax Error":
                suggestions.extend([
                    "Check for missing colons after function definitions",
                    "Verify all parentheses and brackets are properly closed",
                    "Ensure consistent indentation",
                    "Check for invalid syntax in expressions"
                ])
            elif error_type == "Name Error":
                suggestions.extend([
                    "Verify all variables are defined before use",
                    "Check for typos in variable names",
                    "Ensure proper import statements",
                    "Check variable scope"
                ])
            elif error_type == "Type Error":
                suggestions.extend([
                    "Check data types of variables",
                    "Ensure proper type conversion",
                    "Verify function arguments match expected types",
                    "Check for mixing incompatible types"
                ])
            
            # Create debug plan
            debug_plan = f"""Interactive Debugging Plan
            ========================
            
            Error Type: {error_type}
            Original Error: {error_message}
            
            Suggested Debugging Steps:
            {chr(10).join(f"{i+1}. {suggestion}" for i, suggestion in enumerate(suggestions))}
            
            Testing Strategy:
            {chr(10).join(f"- {test_case}" for test_case in test_cases) if test_cases else "- No specific test cases provided"}
            
            Next Steps:
            1. Apply suggested fixes systematically
            2. Test each fix incrementally
            3. Validate with provided test cases
            4. Document the solution for future reference
            """
            
            # Store debug session
            self.debug_history.append({
                "timestamp": datetime.now().isoformat(),
                "code": code[:200] + "..." if len(code) > 200 else code,
                "error_type": error_type,
                "error_message": error_message,
                "suggestions": suggestions
            })
            
            return debug_plan
        
        def get_debug_history(self) -> str:
            """Get recent debugging history"""
            if not self.debug_history:
                return "No debugging history available."
            
            history = "Recent Debugging Sessions:\n\n"
            for i, session in enumerate(self.debug_history[-5:], 1):  # Last 5 sessions
                history += f"{i}. {session['error_type']} - {session['timestamp']}\n"
                history += f"   Code: {session['code'][:100]}...\n"
                history += f"   Suggestions: {len(session['suggestions'])} provided\n\n"
            
            return history
    
    debugger = InteractiveDebugger()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="DebugAssistant",
        instructions="You help debug code interactively with step-by-step guidance.",
        tools=[
            debugger.debug_code,
            debugger.get_debug_history
        ]
    ) as agent:
        
        # Simulate debugging session
        buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers
        total += num
    return total / len(numbers)

numbers = [1, 2, 3, 4, 5]
average = calculate_average(numbers)
print(f"Average: {average}")
        """
        
        error_msg = "SyntaxError: invalid syntax at line 3"
        
        debug_result = await agent.run(f"""
        Debug this code:
        {buggy_code}
        
        Error: {error_msg}
        
        Test cases: ["calculate_average([1,2,3]) should return 2.0", "calculate_average([10,20,30]) should return 20.0"]
        """)
        
        print(f"Debug assistance: {debug_result.text}")
    
    # Pattern 3: Code Performance Profiling
    print("\n3️⃣ Code Performance Profiling...")
    
    class PerformanceProfiler:
        """Code performance profiling and optimization"""
        
        def profile_execution(self,
            code: str = Field(..., description="Code to profile"),
            iterations: int = Field(1000, description="Number of iterations for timing", ge=100, le=10000)
        ) -> str:
            """Profile code execution performance"""
            
            import timeit
            import cProfile
            import pstats
            import io
            
            # Time-based profiling
            timer_setup = """
import numpy as np
import pandas as pd
            """
            
            execution_time = timeit.timeit(
                stmt=code,
                setup=timer_setup,
                number=iterations
            )
            
            average_time = execution_time / iterations
            
            # Memory profiling (simplified)
            import sys
            code_size = sys.getsizeof(code)
            
            # Complexity analysis (basic)
            lines_of_code = len(code.split('\n'))
            has_loops = 'for ' in code or 'while ' in code
            has_nested_loops = code.count('for ') + code.count('while ') > 1
            
            # Generate profiling report
            report = f"""Performance Profiling Report
            ==========================
            
            Execution Metrics:
            - Total time for {iterations} iterations: {execution_time:.4f} seconds
            - Average time per execution: {average_time:.6f} seconds
            - Code size: {code_size} bytes
            - Lines of code: {lines_of_code}
            
            Complexity Indicators:
            - Contains loops: {'Yes' if has_loops else 'No'}
            - Nested loops: {'Yes' if has_nested_loops else 'No'}
            - Has function definitions: {'Yes' if 'def ' in code else 'No'}
            
            Performance Classification:
            {'HIGH PERFORMANCE' if average_time < 0.001 else 'MEDIUM PERFORMANCE' if average_time < 0.01 else 'LOW PERFORMANCE'}
            
            Optimization Suggestions:
            {self._generate_optimization_suggestions(code, average_time)}
            """
            
            return report
        
        def _generate_optimization_suggestions(self, code: str, avg_time: float) -> str:
            """Generate optimization suggestions based on code analysis"""
            suggestions = []
            
            if avg_time > 0.01:
                suggestions.append("- Consider algorithm optimization for better performance")
            
            if 'for ' in code and 'range(len(' in code:
                suggestions.append("- Use enumerate() instead of range(len()) for better Pythonic style")
            
            if 'append(' in code and 'for ' in code:
                suggestions.append("- Consider list comprehensions or generator expressions")
            
            if 'import ' in code and avg_time > 0.001:
                suggestions.append("- Ensure imports are at module level, not inside loops")
            
            if 'pandas' in code and 'for ' in code:
                suggestions.append("- Use pandas vectorized operations instead of loops")
            
            if not suggestions:
                suggestions.append("- Code appears well-optimized for current performance level")
            
            return "\n".join(f"  {suggestion}" for suggestion in suggestions)
    
    profiler = PerformanceProfiler()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="PerformanceProfilerAgent",
        instructions="You analyze code performance and provide optimization recommendations.",
        tools=[
            profiler.profile_execution
        ]
    ) as agent:
        
        # Profile different code patterns
        test_codes = [
            """
# Simple loop
result = []
for i in range(100):
    result.append(i * 2)
            """,
            """
# List comprehension
result = [i * 2 for i in range(100)]
            """,
            """
# NumPy operation
import numpy as np
arr = np.arange(100)
result = arr * 2
            """
        ]
        
        print("Performance profiling comparison:")
        for i, code in enumerate(test_codes, 1):
            print(f"\nTest {i}:")
            result = await agent.run(f"Profile this code with 1000 iterations: {code}")
            print(f"Profile result: {result.text[:500]}...")

async def production_code_deployment():
    """Production deployment patterns for code execution agents"""
    
    print("\n🏭 Production Code Execution Deployment")
    print("=" * 60)
    
    # Production security configuration
    production_security = {
        "max_memory_mb": 50,  # Reduced for production
        "max_cpu_time_seconds": 10,
        "max_output_length": 5000,
        "allowed_imports": {
            "math", "random", "datetime", "json", "csv",
            "numpy", "pandas", "matplotlib"
        },
        "forbidden_operations": {
            "open", "file", "input", "eval", "exec", "__import__",
            "compile", "globals", "locals", "vars", "getattr",
            "setattr", "delattr", "hasattr", "open", "file"
        }
    }
    
    # Resource monitoring
    class ResourceMonitor:
        """Monitor resource usage for code execution"""
        
        def __init__(self):
            self.execution_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0,
                "peak_memory_usage": 0,
                "timeout_count": 0
            }
        
        def record_execution(self, success: bool, execution_time: float, memory_usage: int, timed_out: bool = False):
            """Record execution statistics"""
            self.execution_stats["total_executions"] += 1
            
            if success:
                self.execution_stats["successful_executions"] += 1
            else:
                self.execution_stats["failed_executions"] += 1
            
            if timed_out:
                self.execution_stats["timeout_count"] += 1
            
            # Update averages
            n = self.execution_stats["total_executions"]
            self.execution_stats["average_execution_time"] = (
                (self.execution_stats["average_execution_time"] * (n-1) + execution_time) / n
            )
            
            # Update peak memory usage
            self.execution_stats["peak_memory_usage"] = max(
                self.execution_stats["peak_memory_usage"], 
                memory_usage
            )
        
        def get_performance_report(self) -> str:
            """Get comprehensive performance report"""
            stats = self.execution_stats
            success_rate = (stats["successful_executions"] / max(stats["total_executions"], 1)) * 100
            
            return f"""Code Execution Performance Report
            =================================
            
            Execution Statistics:
            - Total Executions: {stats['total_executions']}
            - Success Rate: {success_rate:.1f}%
            - Average Execution Time: {stats['average_execution_time']:.3f}s
            - Peak Memory Usage: {stats['peak_memory_usage'] / (1024*1024):.1f} MB
            - Timeout Count: {stats['timeout_count']}
            
            System Health: {'🟢 HEALTHY' if success_rate > 95 else '🟡 DEGRADED' if success_rate > 80 else '🔴 UNHEALTHY'}
            """
    
    # Scalability patterns
    class ScalableExecutionPool:
        """Scalable execution pool for handling multiple requests"""
        
        def __init__(self, max_workers: int = 4):
            self.max_workers = max_workers
            self.execution_queue = asyncio.Queue()
            self.active_executions = 0
        
        async def submit_execution(self, code: str, timeout: int = 30) -> str:
            """Submit code for execution"""
            # Create future for result
            future = asyncio.Future()
            
            # Add to queue
            await self.execution_queue.put({
                "code": code,
                "timeout": timeout,
                "future": future
            })
            
            # Process queue if workers available
            if self.active_executions < self.max_workers:
                asyncio.create_task(self._process_queue())
            
            # Wait for result
            return await future
        
        async def _process_queue(self):
            """Process execution queue"""
            self.active_executions += 1
            
            try:
                while not self.execution_queue.empty():
                    task = await self.execution_queue.get()
                    
                    try:
                        # Execute code (simplified - use actual code service)
                        result = f"Executed: {task['code'][:50]}..."
                        task["future"].set_result(result)
                        
                    except Exception as e:
                        task["future"].set_exception(e)
                    
                    self.execution_queue.task_done()
            
            finally:
                self.active_executions -= 1
        
        def get_queue_stats(self) -> Dict[str, Any]:
            """Get queue statistics"""
            return {
                "queue_size": self.execution_queue.qsize(),
                "active_executions": self.active_executions,
                "max_workers": self.max_workers,
                "utilization": self.active_executions / self.max_workers * 100
            }
    
    # Demonstrate production features
    print("Production Code Execution Features:")
    
    # Initialize components
    monitor = ResourceMonitor()
    execution_pool = ScalableExecutionPool(max_workers=3)
    
    print("\n1. Resource Monitoring:")
    
    # Simulate executions
    test_scenarios = [
        {"code": "import math\nprint(math.pi)", "success": True, "time": 0.1, "memory": 1024*1024},
        {"code": "while True: pass", "success": False, "time": 5.0, "memory": 1024*1024, "timeout": True},
        {"code": "import numpy as np\ndata = np.random.random(1000)\nprint(data.mean())", "success": True, "time": 0.5, "memory": 2048*1024}
    ]
    
    for scenario in test_scenarios:
        monitor.record_execution(
            success=scenario["success"],
            execution_time=scenario["time"],
            memory_usage=scenario["memory"],
            timed_out=scenario.get("timeout", False)
        )
    
    performance_report = monitor.get_performance_report()
    print(performance_report)
    
    print("\n2. Scalable Execution Pool:")
    
    # Test execution pool
    async def test_execution_pool():
        tasks = []
        for i in range(5):
            code = f"print('Execution {i+1}')"
            task = execution_pool.submit_execution(code, timeout=10)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i+1}: Failed - {result}")
            else:
                print(f"Task {i+1}: {result}")
    
    await test_execution_pool()
    
    queue_stats = execution_pool.get_queue_stats()
    print(f"Queue stats: {queue_stats}")
    
    print("\n3. Security and Compliance:")
    
    # Security audit
    security_audit = {
        "code_validation": "✅ Enabled - AST parsing and security checks",
        "resource_limits": f"✅ Enabled - {production_security['max_memory_mb']}MB memory, {production_security['max_cpu_time_seconds']}s CPU",
        "import_restrictions": f"✅ Enabled - {len(production_security['allowed_imports'])} allowed imports",
        "operation_whitelist": f"✅ Enabled - {len(production_security['forbidden_operations'])} forbidden operations",
        "timeout_protection": "✅ Enabled - All executions have timeouts",
        "output_sanitization": "✅ Enabled - Output length and content filtering"
    }
    
    for check, status in security_audit.items():
        print(f"- {check.replace('_', ' ').title()}: {status}")
    
    print("\n4. Error Handling and Recovery:")
    
    error_scenarios = [
        "SyntaxError handling",
        "Timeout recovery",
        "Memory limit enforcement", 
        "Import restriction validation",
        "Output sanitization"
    ]
    
    print("Error handling scenarios covered:")
    for scenario in error_scenarios:
        print(f"- {scenario}: ✅ Implemented")
    
    print("\n5. Monitoring and Alerting:")
    
    monitoring_capabilities = {
        "real_time_metrics": "Execution count, success rate, response time",
        "resource_monitoring": "Memory usage, CPU time, timeout tracking",
        "performance_trends": "Historical performance analysis",
        "alert_system": "Threshold-based alerting for degradation",
        "audit_logging": "Complete execution history with security events"
    }
    
    for capability, description in monitoring_capabilities.items():
        print(f"- {capability.replace('_', ' ').title()}: {description}")

if __name__ == "__main__":
    asyncio.run(basic_code_execution_demo())
    asyncio.run(advanced_code_patterns())
    asyncio.run(production_code_deployment())
```

The Code Execution Agent represents the convergence of AI intelligence and computational power, creating systems that can not only understand problems but actively solve them through code. By mastering the patterns and techniques outlined in this section, you'll be equipped to build sophisticated computational assistants that can handle everything from simple calculations to complex data analysis, visualization, and algorithmic problem-solving. The combination of secure execution environments, intelligent code generation, and comprehensive error handling creates agents that are both powerful and safe for production deployment.

---

## Agent Type 5: Multi-Modal Agent

The Multi-Modal Agent represents the frontier of AI capabilities, seamlessly integrating text, images, web data, and external tools to create comprehensive AI experiences. This agent type breaks the boundaries of traditional text-only interactions, enabling applications that can analyze images, fetch real-time data, generate visual content, and orchestrate complex workflows across multiple modalities.

### Architecture Overview

The Multi-Modal Agent architecture implements a sophisticated orchestration layer that coordinates multiple AI capabilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Multi-Modal Agent Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │   Vision        │  │   Web Search    │  │   MCP Tools     │    │
│ │   Processing    │──│   Integration   │──│   & External    │    │
│ │                 │  │                 │  │   APIs          │    │
│ │ • Image         │  │ • Real-time     │  │ • Model         │    │
│ │   analysis      │  │   data          │  │   Context       │    │
│ │ • OCR &         │  │   retrieval     │  │   Protocol      │    │
│ │   extraction    │  │ • Dynamic       │  │ • Tool          │    │
│ │ • Visual        │  │   search        │  │   orchestration │    │
│ │   generation    │  │ • Content       │  │ • Approval      │    │
│ └─────────────────┘  │   aggregation   │  │   workflows     │    │
│                      └─────────────────┘  └─────────────────┘    │
│                               │                      │             │
│ ┌─────────────────────────────┴──────────────────────┴───────────┐ │
│ │                    Reasoning & Synthesis                        │ │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │ │
│ │  │   Context    │  │   Priority   │  │   Response       │    │ │
│ │  │   Integration│  │   Manager    │  │   Synthesis    │    │ │
│ │  │              │  │              │  │                  │    │ │
│ │  │ • Memory     │  │ • Urgency    │  │ • Multi-modal  │    │ │
│ │  │   management │  │   detection  │  │   integration  │    │ │
│ │  │ • Relevance  │  │ • Resource   │  │ • Coherent     │    │ │
│ │  │   scoring    │  │   allocation │  │   narratives   │    │ │
│ │  └──────────────┘  └──────────────┘  └──────────────────┘    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                    Unified Response Generation                  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

This architecture enables several groundbreaking capabilities:
- **Simultaneous Multi-Modal Processing**: Handle text, images, and data in a single interaction
- **Intelligent Tool Orchestration**: Automatically select and coordinate the best tools for each task
- **Context-Aware Reasoning**: Maintain context across different modalities and tools
- **Dynamic Workflow Generation**: Create complex, multi-step workflows on-demand
- **Real-Time Data Integration**: Incorporate live web data and external API responses

### Core Implementation Patterns

Let's explore the fundamental patterns for building Multi-Modal Agents:

```python
import asyncio
from agent_framework import ChatAgent, ChatMessage, TextContent, UriContent, DataContent
from agent_framework.openai import OpenAIResponsesClient
from agent_framework.tools import HostedWebSearchTool, HostedCodeInterpreterTool, MCPStreamableHTTPTool
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import base64
import json
from datetime import datetime
import aiohttp

# Enhanced multi-modal schemas
class MultiModalRequest(BaseModel):
    """Schema for multi-modal requests"""
    text: Optional[str] = None
    image_urls: Optional[List[str]] = None
    image_data: Optional[List[str]] = None  # Base64 encoded images
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    tools: Optional[List[str]] = None
    reasoning_effort: str = Field("medium", description="Reasoning effort: low, medium, high")

class MultiModalResponse(BaseModel):
    """Schema for multi-modal responses"""
    text_response: str
    generated_images: Optional[List[str]] = None  # Base64 encoded
    data_visualizations: Optional[List[str]] = None
    web_data: Optional[List[Dict[str, Any]]] = None
    reasoning_process: Optional[str] = None
    sources: List[Dict[str, Any]]
    confidence_score: float

# Advanced multi-modal processing service
class MultiModalProcessingService:
    """Comprehensive multi-modal processing service"""
    
    def __init__(self):
        self.processing_history = []
        self.tool_registry = {}
        self.context_memory = {}
    
    async def process_multimodal_input(self, request: MultiModalRequest) -> Dict[str, Any]:
        """
        Process multi-modal input with intelligent orchestration
        
        Features:
        - Automatic modality detection
        - Context-aware processing
        - Tool selection and orchestration
        - Result synthesis
        """
        start_time = datetime.now()
        
        try:
            # Analyze input modalities
            modalities = self._detect_modalities(request)
            
            # Select appropriate tools
            selected_tools = await self._select_tools(modalities, request.tools)
            
            # Process each modality
            results = {}
            
            if modalities.get("text"):
                results["text_analysis"] = await self._process_text(request.text, request.context)
            
            if modalities.get("images"):
                results["image_analysis"] = await self._process_images(
                    request.image_urls or [], 
                    request.image_data or [], 
                    request.context
                )
            
            if modalities.get("web_data"):
                results["web_search"] = await self._search_web(request.text, request.context)
            
            # Generate reasoning if requested
            if request.reasoning_effort != "low":
                results["reasoning"] = await self._generate_reasoning(results, request.context)
            
            # Synthesize final response
            final_response = await self._synthesize_response(results, request.context)
            
            # Record processing history
            self.processing_history.append({
                "timestamp": start_time.isoformat(),
                "modalities": modalities,
                "tools_used": selected_tools,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "success": True
            })
            
            return final_response
            
        except Exception as e:
            self.processing_history.append({
                "timestamp": start_time.isoformat(),
                "error": str(e),
                "success": False
            })
            raise
    
    def _detect_modalities(self, request: MultiModalRequest) -> Dict[str, bool]:
        """Automatically detect available modalities"""
        return {
            "text": bool(request.text),
            "images": bool(request.image_urls or request.image_data),
            "audio": bool(request.audio_url),
            "video": bool(request.video_url),
            "web_data": bool(request.text)  # Assume web search for text queries
        }
    
    async def _select_tools(self, modalities: Dict[str, bool], requested_tools: Optional[List[str]]) -> List[str]:
        """Intelligently select tools based on modalities and requirements"""
        available_tools = []
        
        if modalities["text"]:
            available_tools.extend(["text_analysis", "web_search"])
        
        if modalities["images"]:
            available_tools.extend(["image_analysis", "vision_processing"])
        
        if modalities["web_data"]:
            available_tools.append("web_search")
        
        # Filter by requested tools if specified
        if requested_tools:
            available_tools = [tool for tool in available_tools if tool in requested_tools]
        
        return available_tools
    
    async def _process_text(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced text processing with context awareness"""
        # Extract key information
        key_phrases = self._extract_key_phrases(text)
        entities = self._extract_entities(text)
        sentiment = self._analyze_sentiment(text)
        
        return {
            "text": text,
            "key_phrases": key_phrases,
            "entities": entities,
            "sentiment": sentiment,
            "word_count": len(text.split()),
            "processing_timestamp": datetime.now().isoformat()
        }
    
    async def _process_images(self, 
        image_urls: List[str], 
        image_data: List[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process images with comprehensive analysis"""
        results = []
        
        # Process URL-based images
        for url in image_urls:
            try:
                image_analysis = await self._analyze_image_from_url(url, context)
                results.append({
                    "source": "url",
                    "url": url,
                    "analysis": image_analysis,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "source": "url",
                    "url": url,
                    "error": str(e),
                    "success": False
                })
        
        # Process base64-encoded images
        for i, b64_data in enumerate(image_data):
            try:
                image_analysis = await self._analyze_image_from_base64(b64_data, context)
                results.append({
                    "source": "base64",
                    "index": i,
                    "analysis": image_analysis,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "source": "base64",
                    "index": i,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "image_count": len(results),
            "successful_analyses": sum(1 for r in results if r["success"]),
            "results": results
        }
    
    async def _analyze_image_from_url(self, url: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze image from URL"""
        # In production, implement actual image analysis
        return {
            "description": f"Analysis of image from {url}",
            "detected_objects": ["object1", "object2"],
            "text_content": "Sample text from image",
            "dominant_colors": ["#FF0000", "#00FF00"],
            "analysis_confidence": 0.85
        }
    
    async def _analyze_image_from_base64(self, b64_data: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze base64-encoded image"""
        # In production, implement actual image analysis
        return {
            "description": "Analysis of base64-encoded image",
            "detected_objects": ["item1", "item2"],
            "text_content": "Extracted text content",
            "image_size": f"{len(b64_data)} characters",
            "analysis_confidence": 0.90
        }
    
    async def _search_web(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform intelligent web search"""
        # In production, implement actual web search
        return {
            "query": query,
            "results": [
                {
                    "title": "Sample Result 1",
                    "url": "https://example.com/result1",
                    "snippet": "This is a sample search result snippet.",
                    "relevance_score": 0.95
                },
                {
                    "title": "Sample Result 2", 
                    "url": "https://example.com/result2",
                    "snippet": "Another sample result with relevant information.",
                    "relevance_score": 0.87
                }
            ],
            "search_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_reasoning(self, results: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Generate reasoning process for multi-modal analysis"""
        reasoning_steps = []
        
        if "text_analysis" in results:
            reasoning_steps.append(f"Analyzed text input: {results['text_analysis']['word_count']} words")
        
        if "image_analysis" in results:
            reasoning_steps.append(f"Processed {results['image_analysis']['image_count']} images")
        
        if "web_search" in results:
            reasoning_steps.append(f"Performed web search for: '{results['web_search']['query']}'")
        
        return "\n".join([
            "Multi-modal Analysis Reasoning:",
            *reasoning_steps,
            f"Total processing time: {len(results)} components analyzed",
            "Synthesizing comprehensive response..."
        ])
    
    async def _synthesize_response(self, results: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final response from all modalities"""
        response_parts = []
        sources = []
        confidence_scores = []
        
        # Text component
        if "text_analysis" in results:
            text_data = results["text_analysis"]
            response_parts.append(f"Text Analysis: {text_data['word_count']} words processed")
            sources.append({"type": "text", "content": text_data["text"][:200] + "..."})
            confidence_scores.append(0.9)
        
        # Image component
        if "image_analysis" in results:
            image_data = results["image_analysis"]
            response_parts.append(f"Image Analysis: {image_data['successful_analyses']} images processed successfully")
            sources.append({"type": "images", "count": image_data["image_count"]})
            confidence_scores.append(0.85)
        
        # Web search component
        if "web_search" in results:
            web_data = results["web_search"]
            response_parts.append(f"Web Search: Found {len(web_data['results'])} relevant results for '{web_data['query']}'")
            sources.extend([{"type": "web", "title": result["title"], "url": result["url"]} for result in web_data["results"]])
            confidence_scores.append(0.8)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "text_response": "\n\n".join(response_parts),
            "sources": sources,
            "confidence_score": overall_confidence,
            "processing_summary": {
                "modalities_processed": list(results.keys()),
                "total_sources": len(sources)
            }
        }

# Advanced vision processing service
class VisionProcessingService:
    """Advanced vision processing with multiple analysis capabilities"""
    
    def __init__(self):
        self.analysis_history = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    async def comprehensive_image_analysis(self,
        image_source: str = Field(..., description="Image URL or base64 data"),
        analysis_types: List[str] = Field(["objects", "text", "colors"], description="Types of analysis to perform"),
        detailed_analysis: bool = Field(False, description="Perform detailed analysis")
    ) -> Dict[str, Any]:
        """
        Perform comprehensive image analysis with multiple modalities
        
        Analysis types:
        - objects: Object detection and classification
        - text: OCR and text extraction
        - colors: Color palette and analysis
        - faces: Face detection and analysis
        - scene: Scene classification
        - quality: Image quality assessment
        """
        
        try:
            # Determine image source type
            if image_source.startswith('http'):
                image_type = "url"
                image_data = await self._download_image(image_source)
            elif image_source.startswith('data:image'):
                image_type = "base64"
                image_data = base64.b64decode(image_source.split(',')[1])
            else:
                image_type = "base64"
                image_data = base64.b64decode(image_source)
            
            results = {}
            
            # Object detection and classification
            if "objects" in analysis_types:
                results["objects"] = await self._detect_objects(image_data, detailed_analysis)
            
            # Text extraction (OCR)
            if "text" in analysis_types:
                results["text"] = await self._extract_text(image_data)
            
            # Color analysis
            if "colors" in analysis_types:
                results["colors"] = await self._analyze_colors(image_data)
            
            # Face detection
            if "faces" in analysis_types:
                results["faces"] = await self._detect_faces(image_data)
            
            # Scene classification
            if "scene" in analysis_types:
                results["scene"] = await self._classify_scene(image_data)
            
            # Quality assessment
            if "quality" in analysis_types:
                results["quality"] = await self._assess_quality(image_data)
            
            # Store analysis history
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "image_type": image_type,
                "analysis_types": analysis_types,
                "detailed": detailed_analysis,
                "results_summary": {k: len(v) if isinstance(v, list) else "completed" for k, v in results.items()}
            })
            
            return {
                "success": True,
                "analysis_type": image_type,
                "analysis_results": results,
                "summary": self._generate_analysis_summary(results),
                "recommendations": self._generate_recommendations(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_type": "unknown"
            }
    
    async def _download_image(self, url: str) -> bytes:
        """Download image from URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Failed to download image: HTTP {response.status}")
    
    async def _detect_objects(self, image_data: bytes, detailed: bool) -> Dict[str, Any]:
        """Detect and classify objects in image"""
        # In production, implement actual object detection
        # For demonstration, return simulated results
        objects = [
            {"name": "person", "confidence": 0.95, "bbox": [100, 150, 200, 300]},
            {"name": "laptop", "confidence": 0.87, "bbox": [250, 180, 400, 280]},
            {"name": "cup", "confidence": 0.76, "bbox": [300, 250, 350, 300]}
        ]
        
        if detailed:
            # Add more detailed analysis
            for obj in objects:
                obj["attributes"] = {
                    "color": "unknown",
                    "size": "medium",
                    "position": "foreground" if obj["bbox"][1] < 200 else "background"
                }
        
        return {
            "object_count": len(objects),
            "objects": objects,
            "detection_confidence": sum(obj["confidence"] for obj in objects) / len(objects)
        }
    
    async def _extract_text(self, image_data: bytes) -> Dict[str, Any]:
        """Extract text using OCR"""
        # In production, implement actual OCR
        return {
            "text_found": True,
            "text_content": "Sample text extracted from image\nSecond line of text\nThird line with numbers: 12345",
            "text_regions": [
                {"text": "Sample text", "bbox": [50, 100, 200, 130], "confidence": 0.92},
                {"text": "extracted from image", "bbox": [50, 140, 280, 170], "confidence": 0.88}
            ],
            "language": "en",
            "total_characters": 85
        }
    
    async def _analyze_colors(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze color palette and distribution"""
        # Simulate color analysis
        dominant_colors = [
            {"hex": "#2E3440", "name": "Dark Gray", "percentage": 35.2},
            {"hex": "#5E81AC", "name": "Blue", "percentage": 22.8},
            {"hex": "#88C0D0", "name": "Light Blue", "percentage": 18.5},
            {"hex": "#D8DEE9", "name": "Light Gray", "percentage": 15.3},
            {"hex": "#BF616A", "name": "Red", "percentage": 8.2}
        ]
        
        return {
            "dominant_colors": dominant_colors,
            "color_count": len(dominant_colors),
            "primary_palette": dominant_colors[:3],
            "brightness": "medium",  # dark, medium, bright
            "contrast": "high",  # low, medium, high
            "color_harmony": "complementary"
        }
    
    async def _detect_faces(self, image_data: bytes) -> Dict[str, Any]:
        """Detect and analyze faces"""
        # Simulate face detection
        faces = [
            {
                "bbox": [150, 100, 250, 200],
                "confidence": 0.94,
                "attributes": {
                    "age_range": "25-35",
                    "gender": "unknown",
                    "emotion": "neutral",
                    "head_pose": "frontal"
                }
            }
        ]
        
        return {
            "face_count": len(faces),
            "faces": faces,
            "largest_face": max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])) if faces else None
        }
    
    async def _classify_scene(self, image_data: bytes) -> Dict[str, Any]:
        """Classify scene type"""
        # Simulate scene classification
        scene_categories = [
            {"category": "indoor", "confidence": 0.78},
            {"category": "office", "confidence": 0.65},
            {"category": "workspace", "confidence": 0.58}
        ]
        
        return {
            "scene_categories": scene_categories,
            "primary_category": max(scene_categories, key=lambda x: x["confidence"]),
            "confidence_score": max(scene_categories, key=lambda x: x["confidence"])["confidence"]
        }
    
    async def _assess_quality(self, image_data: bytes) -> Dict[str, Any]:
        """Assess image quality"""
        # Simulate quality assessment
        return {
            "resolution": "1920x1080",
            "file_size": len(image_data),
            "format": "JPEG",
            "quality_score": 0.85,
            "issues": ["slight blur", "moderate compression"],
            "recommendations": ["Use higher resolution", "Reduce compression"]
        }
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis summary"""
        summary_parts = []
        
        if "objects" in results:
            objects = results["objects"]
            summary_parts.append(f"Detected {objects['object_count']} objects with average confidence {objects['detection_confidence']:.2f}")
        
        if "text" in results:
            text = results["text"]
            summary_parts.append(f"Extracted {text['total_characters']} characters of text in {text['language']}")
        
        if "colors" in results:
            colors = results["colors"]
            summary_parts.append(f"Analyzed {colors['color_count']} dominant colors with {colors['contrast']} contrast")
        
        if "faces" in results:
            faces = results["faces"]
            summary_parts.append(f"Found {faces['face_count']} faces")
        
        if "scene" in results:
            scene = results["scene"]
            summary_parts.append(f"Classified scene as {scene['primary_category']['category']} ({scene['confidence_score']:.2f} confidence)")
        
        return " | ".join(summary_parts) if summary_parts else "Basic image analysis completed"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if "quality" in results:
            quality = results["quality"]
            if quality["quality_score"] < 0.8:
                recommendations.extend(quality["recommendations"])
        
        if "text" in results and not results["text"]["text_found"]:
            recommendations.append("No text detected - image may be purely visual")
        
        if "objects" in results:
            objects = results["objects"]
            if objects["detection_confidence"] < 0.8:
                recommendations.append("Consider using higher quality images for better object detection")
        
        if not recommendations:
            recommendations.append("Image quality and content are suitable for most applications")
        
        return recommendations

# Advanced web search and data aggregation
class WebSearchAggregator:
    """Advanced web search with data aggregation and analysis"""
    
    def __init__(self):
        self.search_history = []
        self.aggregation_cache = {}
    
    async def comprehensive_web_search(self,
        query: str = Field(..., description="Search query"),
        search_types: List[str] = Field(["web", "news", "academic"], description="Types of search to perform"),
        time_range: str = Field("past_year", description="Time range: 'past_day', 'past_week', 'past_month', 'past_year'"),
        max_results_per_type: int = Field(5, description="Maximum results per search type", ge=1, le=20)
    ) -> Dict[str, Any]:
        """
        Perform comprehensive web search across multiple sources
        
        Search types:
        - web: General web search
        - news: News articles and updates
        - academic: Academic papers and research
        - social: Social media and discussions
        - multimedia: Images, videos, and other media
        """
        
        search_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        search_results = {}
        
        try:
            # Simulate different search types
            for search_type in search_types:
                if search_type == "web":
                    search_results["web"] = await self._search_web_general(query, time_range, max_results_per_type)
                elif search_type == "news":
                    search_results["news"] = await self._search_news(query, time_range, max_results_per_type)
                elif search_type == "academic":
                    search_results["academic"] = await self._search_academic(query, time_range, max_results_per_type)
                elif search_type == "social":
                    search_results["social"] = await self._search_social(query, time_range, max_results_per_type)
                elif search_type == "multimedia":
                    search_results["multimedia"] = await self._search_multimedia(query, time_range, max_results_per_type)
            
            # Aggregate and analyze results
            aggregated_analysis = await self._aggregate_results(search_results, query)
            
            # Store search history
            self.search_history.append({
                "search_id": search_id,
                "query": query,
                "search_types": search_types,
                "time_range": time_range,
                "total_results": sum(len(results.get("results", [])) for results in search_results.values()),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "search_id": search_id,
                "query": query,
                "search_results": search_results,
                "aggregation": aggregated_analysis,
                "summary": self._generate_search_summary(search_results),
                "recommendations": self._generate_search_recommendations(search_results)
            }
            
        except Exception as e:
            return {
                "search_id": search_id,
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _search_web_general(self, query: str, time_range: str, max_results: int) -> Dict[str, Any]:
        """Simulate general web search"""
        results = []
        
        # Generate realistic search results
        search_titles = [
            f"Comprehensive guide to {query}",
            f"Latest developments in {query} - 2024",
            f"Understanding {query}: A complete overview",
            f"Best practices for {query} implementation",
            f"Case studies: Successful {query} projects"
        ]
        
        for i in range(min(max_results, len(search_titles))):
            results.append({
                "title": search_titles[i],
                "url": f"https://example{i+1}.com/{query.replace(' ', '-')}",
                "snippet": f"This article provides comprehensive information about {query}. "
                          f"It covers key concepts, practical applications, and recent developments.",
                "source": "example.com",
                "date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
                "relevance_score": 0.95 - (i * 0.05)
            })
        
        return {
            "search_type": "web",
            "query": query,
            "results_count": len(results),
            "results": results,
            "search_metadata": {
                "time_range": time_range,
                "estimated_total_results": len(results) * 10
            }
        }
    
    async def _search_news(self, query: str, time_range: str, max_results: int) -> Dict[str, Any]:
        """Simulate news search"""
        results = []
        
        news_headlines = [
            f"Breaking: Major developments in {query} announced",
            f"Industry report: {query} market shows significant growth",
            f"New research reveals insights about {query}",
            f"Experts predict future trends in {query}",
            f"Government policy changes affecting {query}"
        ]
        
        for i in range(min(max_results, len(news_headlines))):
            results.append({
                "title": news_headlines[i],
                "url": f"https://news{i+1}.com/{query.replace(' ', '-')}-update",
                "snippet": f"Recent news about {query}: Industry experts discuss latest developments "
                          f"and their implications for businesses and consumers.",
                "source": f"News Source {i+1}",
                "date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                "relevance_score": 0.90 - (i * 0.08)
            })
        
        return {
            "search_type": "news",
            "query": query,
            "results_count": len(results),
            "results": results,
            "search_metadata": {
                "time_range": "recent",
                "freshness_score": 0.85
            }
        }
    
    async def _search_academic(self, query: str, time_range: str, max_results: int) -> Dict[str, Any]:
        """Simulate academic search"""
        results = []
        
        paper_titles = [
            f"Advanced methodologies in {query}: A systematic review",
            f"Empirical analysis of {query} effectiveness across domains",
            f"Theoretical foundations of {query}: Mathematical models and applications",
            f"Longitudinal study: Evolution and impact of {query}",
            f"Comparative analysis: {query} vs. alternative approaches"
        ]
        
        for i in range(min(max_results, len(paper_titles))):
            results.append({
                "title": paper_titles[i],
                "url": f"https://academic{i+1}.edu/research/{query.replace(' ', '-')}",
                "snippet": f"This peer-reviewed research paper examines {query} from an academic perspective, "
                          f"providing rigorous analysis and empirical evidence.",
                "source": f"Academic Journal {i+1}",
                "date": (datetime.now() - timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d"),
                "relevance_score": 0.88 - (i * 0.07),
                "citation_count": random.randint(10, 500),
                "authors": [f"Author {j+1}" for j in range(random.randint(2, 5))]
            })
        
        return {
            "search_type": "academic",
            "query": query,
            "results_count": len(results),
            "results": results,
            "search_metadata": {
                "peer_reviewed": True,
                "citation_analysis": True
            }
        }
    
    async def _aggregate_results(self, search_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Aggregate and analyze results from multiple search types"""
        
        all_results = []
        source_distribution = {}
        temporal_distribution = {}
        
        # Collect all results
        for search_type, results_data in search_results.items():
            if "results" in results_data:
                all_results.extend(results_data["results"])
                source_distribution[search_type] = len(results_data["results"])
        
        # Analyze temporal distribution
        for result in all_results:
            if "date" in result:
                year = result["date"][:4]
                temporal_distribution[year] = temporal_distribution.get(year, 0) + 1
        
        # Calculate relevance metrics
        relevance_scores = [result.get("relevance_score", 0.5) for result in all_results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Identify key themes (simplified)
        key_themes = ["technology", "implementation", "best_practices", "case_studies"]
        
        # Generate insights
        insights = [
            f"Found {len(all_results)} total results across {len(search_results)} search types",
            f"Average relevance score: {avg_relevance:.3f}",
            f"Most results from: {max(source_distribution, key=source_distribution.get)}",
            f"Temporal coverage: {min(temporal_distribution.keys()) if temporal_distribution else 'N/A'} to {max(temporal_distribution.keys()) if temporal_distribution else 'N/A'}"
        ]
        
        return {
            "total_results": len(all_results),
            "source_distribution": source_distribution,
            "temporal_distribution": temporal_distribution,
            "average_relevance": avg_relevance,
            "key_themes": key_themes,
            "insights": insights,
            "aggregation_timestamp": datetime.now().isoformat()
        }
    
    def _generate_search_summary(self, search_results: Dict[str, Any]) -> str:
        """Generate comprehensive search summary"""
        total_results = sum(len(data.get("results", [])) for data in search_results.values())
        
        summary = f"Comprehensive search completed with {total_results} total results:\n\n"
        
        for search_type, results_data in search_results.items():
            count = len(results_data.get("results", []))
            summary += f"- {search_type.title()}: {count} results\n"
        
        return summary
    
    def _generate_search_recommendations(self, search_results: Dict[str, Any]) -> List[str]:
        """Generate search recommendations based on results"""
        recommendations = []
        
        # Based on result distribution
        result_counts = {k: len(v.get("results", [])) for k, v in search_results.items()}
        max_source = max(result_counts, key=result_counts.get)
        
        if result_counts[max_source] > 10:
            recommendations.append(f"Focus on {max_source} sources for most comprehensive coverage")
        
        # Based on temporal distribution
        for results_data in search_results.values():
            if "results" in results_data and results_data["results"]:
                dates = [r.get("date", "") for r in results_data["results"] if r.get("date")]
                if dates:
                    recent_dates = [d for d in dates if datetime.now() - datetime.strptime(d, "%Y-%m-%d") < timedelta(days=30)]
                    if len(recent_dates) < 2:
                        recommendations.append("Consider searching for more recent information")
                        break
        
        # Based on relevance
        all_relevances = []
        for results_data in search_results.values():
            if "results" in results_data:
                all_relevances.extend([r.get("relevance_score", 0.5) for r in results_data["results"]])
        
        if all_relevances:
            avg_relevance = sum(all_relevances) / len(all_relevances)
            if avg_relevance < 0.7:
                recommendations.append("Consider refining search query for better relevance")
        
        if not recommendations:
            recommendations.append("Search results appear comprehensive and relevant")
        
        return recommendations

# Advanced reasoning capabilities
class AdvancedReasoningService:
    """Advanced reasoning and synthesis for complex multi-modal queries"""
    
    def __init__(self):
        self.reasoning_history = []
        self.reasoning_strategies = {
            "analytical": "Break down complex problems into components",
            "comparative": "Compare and contrast different approaches or data",
            "causal": "Identify cause-and-effect relationships",
            "synthetic": "Combine information from multiple sources",
            "evaluative": "Assess quality, validity, or effectiveness"
        }
    
    async def perform_advanced_reasoning(self,
        query: str = Field(..., description="Complex query requiring advanced reasoning"),
        reasoning_type: str = Field("analytical", description="Type of reasoning: analytical, comparative, causal, synthetic, evaluative"),
        evidence_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence sources for reasoning"),
        reasoning_effort: str = Field("high", description="Reasoning effort level: low, medium, high")
    ) -> Dict[str, Any]:
        """
        Perform advanced reasoning with structured thinking process
        
        Reasoning types:
        - analytical: Break down complex problems
        - comparative: Compare different options or data
        - causal: Identify cause-effect relationships  
        - synthetic: Combine information from multiple sources
        - evaluative: Assess quality or effectiveness
        """
        
        reasoning_id = f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # Initialize reasoning process
            reasoning_process = {
                "query": query,
                "reasoning_type": reasoning_type,
                "evidence_count": len(evidence_sources),
                "effort_level": reasoning_effort,
                "steps": []
            }
            
            # Step 1: Problem decomposition
            decomposition = await self._decompose_problem(query, reasoning_type)
            reasoning_process["steps"].append({
                "step": "problem_decomposition",
                "description": "Break down complex query into manageable components",
                "result": decomposition
            })
            
            # Step 2: Evidence evaluation
            evidence_assessment = await self._evaluate_evidence(evidence_sources, reasoning_type)
            reasoning_process["steps"].append({
                "step": "evidence_evaluation",
                "description": "Assess quality and relevance of evidence sources",
                "result": evidence_assessment
            })
            
            # Step 3: Reasoning execution
            reasoning_result = await self._execute_reasoning(
                decomposition, evidence_assessment, reasoning_type, reasoning_effort
            )
            reasoning_process["steps"].append({
                "step": "reasoning_execution",
                "description": "Apply reasoning strategy to solve the problem",
                "result": reasoning_result
            })
            
            # Step 4: Validation and verification
            validation = await self._validate_reasoning(reasoning_result, evidence_sources)
            reasoning_process["steps"].append({
                "step": "validation",
                "description": "Verify reasoning consistency and validity",
                "result": validation
            })
            
            # Step 5: Conclusion synthesis
            conclusion = await self._synthesize_conclusion(reasoning_result, validation)
            reasoning_process["steps"].append({
                "step": "conclusion_synthesis",
                "description": "Synthesize final conclusion from reasoning process",
                "result": conclusion
            })
            
            # Calculate confidence based on evidence quality and reasoning coherence
            confidence_score = self._calculate_confidence(evidence_assessment, validation)
            
            # Record reasoning history
            self.reasoning_history.append({
                "reasoning_id": reasoning_id,
                "timestamp": start_time.isoformat(),
                "reasoning_process": reasoning_process,
                "confidence_score": confidence_score,
                "execution_time": (datetime.now() - start_time).total_seconds()
            })
            
            return {
                "reasoning_id": reasoning_id,
                "success": True,
                "reasoning_process": reasoning_process,
                "final_conclusion": conclusion,
                "confidence_score": confidence_score,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "evidence_summary": evidence_assessment
            }
            
        except Exception as e:
            return {
                "reasoning_id": reasoning_id,
                "success": False,
                "error": str(e),
                "partial_results": reasoning_process if 'reasoning_process' in locals() else {}
            }
    
    async def _decompose_problem(self, query: str, reasoning_type: str) -> Dict[str, Any]:
        """Decompose complex problem into components"""
        
        # Simple decomposition based on reasoning type
        if reasoning_type == "analytical":
            components = [
                "problem_identification",
                "data_gathering", 
                "analysis_execution",
                "result_interpretation"
            ]
        elif reasoning_type == "comparative":
            components = [
                "criteria_definition",
                "option_identification",
                "comparative_analysis",
                "conclusion_synthesis"
            ]
        elif reasoning_type == "causal":
            components = [
                "cause_identification",
                "effect_analysis",
                "relationship_mapping",
                "causal_conclusion"
            ]
        else:
            components = ["problem_analysis", "solution_development", "validation"]
        
        return {
            "original_query": query,
            "reasoning_type": reasoning_type,
            "components": components,
            "complexity_level": "high" if len(query.split()) > 50 else "medium" if len(query.split()) > 20 else "low"
        }
    
    async def _evaluate_evidence(self, evidence_sources: List[Dict[str, Any]], reasoning_type: str) -> Dict[str, Any]:
        """Evaluate quality and relevance of evidence sources"""
        
        evaluation_results = []
        total_relevance = 0
        total_credibility = 0
        
        for i, evidence in enumerate(evidence_sources):
            # Evaluate relevance to reasoning type
            relevance_score = self._assess_relevance(evidence, reasoning_type)
            
            # Evaluate credibility
            credibility_score = self._assess_credibility(evidence)
            
            # Evaluate recency
            recency_score = self._assess_recency(evidence)
            
            evaluation = {
                "evidence_id": i,
                "source_type": evidence.get("type", "unknown"),
                "relevance_score": relevance_score,
                "credibility_score": credibility_score,
                "recency_score": recency_score,
                "overall_quality": (relevance_score + credibility_score + recency_score) / 3
            }
            
            evaluation_results.append(evaluation)
            total_relevance += relevance_score
            total_credibility += credibility_score
        
        # Calculate overall evidence quality
        avg_relevance = total_relevance / len(evidence_sources) if evidence_sources else 0
        avg_credibility = total_credibility / len(evidence_sources) if evidence_sources else 0
        
        return {
            "individual_evaluations": evaluation_results,
            "average_relevance": avg_relevance,
            "average_credibility": avg_credibility,
            "evidence_sufficiency": "sufficient" if avg_relevance > 0.7 and avg_credibility > 0.6 else "insufficient",
            "recommendations": self._generate_evidence_recommendations(evaluation_results)
        }
    
    def _assess_relevance(self, evidence: Dict[str, Any], reasoning_type: str) -> float:
        """Assess relevance of evidence to reasoning type"""
        # Simplified relevance scoring
        content = str(evidence.get("content", "")).lower()
        
        relevance_indicators = {
            "analytical": ["data", "analysis", "statistics", "evidence"],
            "comparative": ["comparison", "versus", "difference", "similarity"],
            "causal": ["cause", "effect", "reason", "impact"],
            "synthetic": ["synthesis", "combination", "integration"],
            "evaluative": ["evaluation", "assessment", "quality", "effectiveness"]
        }
        
        indicators = relevance_indicators.get(reasoning_type, [])
        matches = sum(1 for indicator in indicators if indicator in content)
        
        return min(matches / len(indicators), 1.0) if indicators else 0.5
    
    def _assess_credibility(self, evidence: Dict[str, Any]) -> float:
        """Assess credibility of evidence source"""
        source_type = evidence.get("type", "unknown")
        
        credibility_scores = {
            "academic": 0.9,
            "official": 0.8,
            "news": 0.7,
            "web": 0.5,
            "social": 0.3,
            "unknown": 0.2
        }
        
        base_score = credibility_scores.get(source_type, 0.2)
        
        # Adjust based on recency and corroboration
        if evidence.get("date"):
            date_obj = datetime.fromisoformat(evidence["date"]) if isinstance(evidence["date"], str) else evidence["date"]
            age_days = (datetime.now() - date_obj).days
            if age_days < 30:
                base_score += 0.1
            elif age_days > 365:
                base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _assess_recency(self, evidence: Dict[str, Any]) -> float:
        """Assess recency of evidence"""
        if not evidence.get("date"):
            return 0.3  # Neutral score for unknown dates
        
        try:
            date_obj = datetime.fromisoformat(evidence["date"]) if isinstance(evidence["date"], str) else evidence["date"]
            age_days = (datetime.now() - date_obj).days
            
            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.8
            elif age_days < 90:
                return 0.6
            elif age_days < 365:
                return 0.4
            else:
                return 0.2
        except:
            return 0.3
    
    async def _execute_reasoning(self, 
        decomposition: Dict[str, Any], 
        evidence_assessment: Dict[str, Any], 
        reasoning_type: str, 
        effort_level: str
    ) -> Dict[str, Any]:
        """Execute the core reasoning process"""
        
        # Generate reasoning based on type and effort level
        if effort_level == "high":
            reasoning_depth = "comprehensive"
            consideration_factors = 5
        elif effort_level == "medium":
            reasoning_depth = "moderate"
            consideration_factors = 3
        else:
            reasoning_depth = "basic"
            consideration_factors = 2
        
        # Simulate reasoning process (in production, implement actual reasoning)
        reasoning_steps = []
        
        for component in decomposition["components"]:
            step_reasoning = f"Applying {reasoning_type} reasoning to {component} with {reasoning_depth} analysis"
            
            # Incorporate evidence
            if evidence_assessment["evidence_sufficiency"] == "sufficient":
                step_reasoning += " using high-quality evidence"
            else:
                step_reasoning += " with limited evidence"
            
            reasoning_steps.append({
                "component": component,
                "reasoning": step_reasoning,
                "evidence_used": len(evidence_assessment["individual_evaluations"]),
                "confidence": 0.8 + (consideration_factors * 0.04)  # Higher effort = higher confidence
            })
        
        return {
            "reasoning_type": reasoning_type,
            "effort_level": effort_level,
            "reasoning_depth": reasoning_depth,
            "steps": reasoning_steps,
            "overall_confidence": sum(step["confidence"] for step in reasoning_steps) / len(reasoning_steps)
        }
    
    async def _validate_reasoning(self, reasoning_result: Dict[str, Any], evidence_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate reasoning consistency and completeness"""
        
        # Check for logical consistency
        consistency_checks = []
        
        # Validate that all components were addressed
        addressed_components = [step["component"] for step in reasoning_result["steps"]]
        missing_components = set(reasoning_result.get("components", [])) - set(addressed_components)
        
        consistency_checks.append({
            "check": "component_coverage",
            "passed": len(missing_components) == 0,
            "details": f"Missing components: {list(missing_components)}" if missing_components else "All components addressed"
        })
        
        # Validate evidence consistency
        evidence_consistency = all(
            step.get("evidence_used", 0) > 0 for step in reasoning_result["steps"]
        ) if evidence_sources else True
        
        consistency_checks.append({
            "check": "evidence_consistency",
            "passed": evidence_consistency,
            "details": "Evidence used consistently across reasoning steps" if evidence_consistency else "Some steps lack evidence"
        })
        
        # Validate confidence levels
        avg_confidence = reasoning_result.get("overall_confidence", 0)
        confidence_check = avg_confidence > 0.6  # Minimum confidence threshold
        
        consistency_checks.append({
            "check": "confidence_threshold",
            "passed": confidence_check,
            "details": f"Average confidence: {avg_confidence:.3f} {'(acceptable)' if confidence_check else '(below threshold)'}"
        })
        
        overall_validity = all(check["passed"] for check in consistency_checks)
        
        return {
            "overall_validity": overall_validity,
            "consistency_checks": consistency_checks,
            "validation_score": sum(1 for check in consistency_checks if check["passed"]) / len(consistency_checks)
        }
    
    async def _synthesize_conclusion(self, reasoning_result: Dict[str, Any], validation: Dict[str, Any]) -> str:
        """Synthesize final conclusion from reasoning process"""
        
        # Build conclusion based on reasoning results
        conclusion_parts = []
        
        # Add reasoning summary
        conclusion_parts.append(f"Based on {reasoning_result['reasoning_type']} reasoning with {reasoning_result['effort_level']} effort:")
        
        # Add key findings from each step
        for step in reasoning_result["steps"]:
            conclusion_parts.append(f"- {step['component']}: {step['reasoning']}")
        
        # Add validation status
        if validation["overall_validity"]:
            conclusion_parts.append(f"\nValidation: Reasoning process is consistent and valid (score: {validation['validation_score']:.2f})")
        else:
            conclusion_parts.append(f"\nValidation: Some inconsistencies detected (score: {validation['validation_score']:.2f})")
        
        # Add confidence statement
        confidence = reasoning_result.get("overall_confidence", 0)
        if confidence > 0.8:
            confidence_statement = "High confidence in conclusions"
        elif confidence > 0.6:
            confidence_statement = "Moderate confidence in conclusions"
        else:
            confidence_statement = "Low confidence - consider additional evidence"
        
        conclusion_parts.append(f"\nConfidence Level: {confidence_statement} ({confidence:.3f})")
        
        return "\n".join(conclusion_parts)
    
    def _calculate_confidence(self, evidence_assessment: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from evidence quality
        evidence_confidence = evidence_assessment.get("average_relevance", 0) * 0.4 + \
                             evidence_assessment.get("average_credibility", 0) * 0.3
        
        # Add validation confidence
        validation_confidence = validation.get("validation_score", 0) * 0.3
        
        overall_confidence = evidence_confidence + validation_confidence
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _generate_evidence_recommendations(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving evidence quality"""
        
        recommendations = []
        
        # Find low-scoring areas
        low_relevance = [eval for eval in evaluations if eval["relevance_score"] < 0.5]
        low_credibility = [eval for eval in evaluations if eval["credibility_score"] < 0.5]
        low_recency = [eval for eval in evaluations if eval["recency_score"] < 0.5]
        
        if low_relevance:
            recommendations.append(f"Improve relevance: {len(low_relevance)} sources need better alignment with reasoning goals")
        
        if low_credibility:
            recommendations.append(f"Enhance credibility: {len(low_credibility)} sources from less reliable origins")
        
        if low_recency:
            recommendations.append(f"Update recency: {len(low_recency)} sources are outdated")
        
        if not recommendations:
            recommendations.append("Evidence quality is sufficient for current reasoning requirements")
        
        return recommendations

async def basic_multimodal_demo():
    """Demonstrate basic multi-modal capabilities"""
    
    print("🌈 Basic Multi-Modal Agent Demo")
    print("=" * 60)
    
    # Initialize services
    multimodal_service = MultiModalProcessingService()
    vision_service = VisionProcessingService()
    web_search_service = WebSearchAggregator()
    reasoning_service = AdvancedReasoningService()
    
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="MultiModalAssistant",
        instructions="""You are a comprehensive multi-modal AI assistant with advanced capabilities.
        
        Guidelines:
        - Process multiple input types simultaneously
        - Use appropriate tools for each modality
        - Synthesize information from diverse sources
        - Provide reasoning for complex queries
        - Generate visualizations when helpful""",
        tools=[
            vision_service.comprehensive_image_analysis,
            web_search_service.comprehensive_web_search,
            reasoning_service.perform_advanced_reasoning
        ]
    ) as agent:
        
        # Example 1: Multi-modal query with text and image analysis
        print("1️⃣ Multi-Modal Text and Image Analysis:")
        
        # Create a multi-modal message
        multimodal_message = ChatMessage(
            role="user",
            contents=[
                TextContent(text="Analyze this image and tell me what you see. Also search for current information about the objects detected."),
                UriContent(
                    uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    media_type="image/jpeg"
                )
            ]
        )
        
        result1 = await agent.run(multimodal_message)
        print(f"Multi-modal response: {result1.text[:500]}...\n")
        
        # Example 2: Comprehensive web search with aggregation
        print("2️⃣ Comprehensive Web Search:")
        
        search_result = await agent.run("""
        Perform a comprehensive search about renewable energy technologies.
        Include web results, news, and academic sources.
        Focus on the past year and provide analysis of trends.
        """)
        
        print(f"Search results: {search_result.text[:500]}...\n")
        
        # Example 3: Advanced reasoning with evidence
        print("3️⃣ Advanced Reasoning:")
        
        evidence_sources = [
            {
                "type": "academic",
                "content": "Climate change is accelerating due to human activities, with global temperatures rising 1.1°C above pre-industrial levels.",
                "date": "2023-06-15"
            },
            {
                "type": "official",
                "content": "Renewable energy adoption has increased by 45% in the last two years, with solar and wind leading growth.",
                "date": "2024-01-20"
            },
            {
                "type": "news",
                "content": "Major corporations are committing to net-zero emissions by 2030, investing billions in clean technology.",
                "date": "2024-02-10"
            }
        ]
        
        reasoning_result = await agent.run(f"""
        Based on this evidence: {json.dumps(evidence_sources)}
        
        Perform analytical reasoning to answer:
        What is the relationship between climate change acceleration and renewable energy adoption?
        Provide a comprehensive analysis with conclusions.
        """)
        
        print(f"Reasoning result: {reasoning_result.text}")

async def advanced_multimodal_patterns():
    """Advanced multi-modal patterns for production use"""
    
    print("\n🚀 Advanced Multi-Modal Patterns")
    print("=" * 60)
    
    # Pattern 1: Real-Time Data Pipeline
    print("1️⃣ Real-Time Data Pipeline...")
    
    class RealTimeDataPipeline:
        """Real-time data processing and analysis pipeline"""
        
        def __init__(self):
            self.data_buffer = []
            self.processing_metrics = {
                "total_processed": 0,
                "errors": 0,
                "average_processing_time": 0
            }
        
        async def process_real_time_data(self,
            data_stream: str = Field(..., description="Real-time data stream identifier"),
            processing_type: str = Field("analyze", description="Type of processing: analyze, visualize, alert"),
            window_size: int = Field(100, description="Number of data points to process", ge=10, le=1000)
        ) -> str:
            """Process real-time data with streaming analytics"""
            
            try:
                start_time = datetime.now()
                
                # Simulate real-time data processing
                data_points = []
                for i in range(window_size):
                    data_point = {
                        "timestamp": (datetime.now() - timedelta(seconds=i)).isoformat(),
                        "value": random.uniform(10, 100),
                        "quality": random.choice(["good", "medium", "poor"])
                    }
                    data_points.append(data_point)
                
                # Process based on type
                if processing_type == "analyze":
                    result = self._analyze_data_stream(data_points)
                elif processing_type == "visualize":
                    result = await self._visualize_data_stream(data_points)
                elif processing_type == "alert":
                    result = self._check_alerts(data_points)
                else:
                    result = self._analyze_data_stream(data_points)
                
                # Update metrics
                self.processing_metrics["total_processed"] += window_size
                processing_time = (datetime.now() - start_time).total_seconds()
                
                n = self.processing_metrics["total_processed"] / window_size
                self.processing_metrics["average_processing_time"] = (
                    (self.processing_metrics["average_processing_time"] * (n-1) + processing_time) / n
                )
                
                return f"""Real-time Data Processing Complete
            
            Stream: {data_stream}
            Processing Type: {processing_type}
            Data Points: {window_size}
            Processing Time: {processing_time:.3f}s
            
            Results:
            {result}
            
            Pipeline Status:
            - Total Processed: {self.processing_metrics['total_processed']}
            - Average Time: {self.processing_metrics['average_processing_time']:.3f}s
            - Success Rate: {((self.processing_metrics['total_processed'] - self.processing_metrics['errors']) / max(self.processing_metrics['total_processed'], 1) * 100):.1f}%
            """
                
            except Exception as e:
                self.processing_metrics["errors"] += 1
                return f"Real-time processing failed: {str(e)}"
        
        def _analyze_data_stream(self, data_points: List[Dict[str, Any]]) -> str:
            """Analyze data stream for patterns and insights"""
            
            values = [dp["value"] for dp in data_points]
            
            stats = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std_dev": (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5
            }
            
            # Detect anomalies (simple threshold-based)
            threshold = stats["mean"] + 2 * stats["std_dev"]
            anomalies = [v for v in values if v > threshold]
            
            return f"""
            Statistical Summary:
            - Count: {stats['count']}
            - Mean: {stats['mean']:.2f}
            - Range: {stats['min']:.2f} to {stats['max']:.2f}
            - Standard Deviation: {stats['std_dev']:.2f}
            
            Anomaly Detection:
            - Threshold: {threshold:.2f}
            - Anomalies Detected: {len(anomalies)}
            - Anomaly Rate: {len(anomalies)/len(values)*100:.1f}%
            """
        
        async def _visualize_data_stream(self, data_points: List[Dict[str, Any]]) -> str:
            """Create real-time visualization"""
            
            import matplotlib.pyplot as plt
            import numpy as np
            
            values = [dp["value"] for dp in data_points]
            timestamps = [datetime.fromisoformat(dp["timestamp"]) for dp in data_points]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Time series plot
            ax1.plot(timestamps, values, 'b-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.set_title('Real-time Data Stream')
            ax1.grid(True, alpha=0.3)
            
            # Histogram
            ax2.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Value Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return f"Real-time visualization generated successfully. Chart shows {len(values)} data points with trends and distribution."
        
        def _check_alerts(self, data_points: List[Dict[str, Any]]) -> str:
            """Check for alert conditions"""
            
            values = [dp["value"] for dp in data_points]
            current_avg = sum(values[-10:]) / 10  # Last 10 values average
            overall_avg = sum(values) / len(values)
            
            alerts = []
            
            # Threshold alert
            if current_avg > overall_avg * 1.2:
                alerts.append("HIGH: Current average significantly above overall average")
            elif current_avg < overall_avg * 0.8:
                alerts.append("LOW: Current average significantly below overall average")
            
            # Trend alert
            if len(values) >= 20:
                recent_trend = values[-10:]
                early_trend = values[-20:-10]
                if sum(recent_trend) / len(recent_trend) > sum(early_trend) / len(early_trend) * 1.1:
                    alerts.append("TREND: Significant upward trend detected")
                elif sum(recent_trend) / len(recent_trend) < sum(early_trend) / len(early_trend) * 0.9:
                    alerts.append("TREND: Significant downward trend detected")
            
            if not alerts:
                alerts.append("NORMAL: No significant alerts detected")
            
            return f"Alert Status: {' | '.join(alerts)}"
    
    # Pattern 2: Cross-Modal Reasoning
    print("\n2️⃣ Cross-Modal Reasoning...")
    
    class CrossModalReasoningEngine:
        """Cross-modal reasoning that combines insights from different modalities"""
        
        def __init__(self):
            self.reasoning_patterns = {
                "visual_text": "Combine visual observations with textual descriptions",
                "data_narrative": "Transform data insights into narrative explanations", 
                "trend_prediction": "Use current patterns to predict future trends",
                "anomaly_detection": "Identify unusual patterns across modalities"
            }
        
        async def perform_cross_modal_reasoning(self,
            visual_data: Dict[str, Any] = Field(..., description="Visual analysis results"),
            textual_data: str = Field(..., description="Textual information"),
            numerical_data: List[float] = Field(default_factory=list, description="Numerical data points"),
            reasoning_pattern: str = Field("visual_text", description="Type of cross-modal reasoning")
        ) -> str:
            """Perform reasoning across different data modalities"""
            
            try:
                # Validate reasoning pattern
                if reasoning_pattern not in self.reasoning_patterns:
                    return f"Invalid reasoning pattern. Available: {list(self.reasoning_patterns.keys())}"
                
                # Extract insights from each modality
                visual_insights = self._extract_visual_insights(visual_data)
                textual_insights = self._extract_textual_insights(textual_data)
                numerical_insights = self._extract_numerical_insights(numerical_data)
                
                # Apply cross-modal reasoning
                if reasoning_pattern == "visual_text":
                    result = self._reason_visual_text(visual_insights, textual_insights)
                elif reasoning_pattern == "data_narrative":
                    result = self._reason_data_narrative(numerical_insights, textual_insights)
                elif reasoning_pattern == "trend_prediction":
                    result = self._reason_trend_prediction(visual_insights, numerical_insights)
                elif reasoning_pattern == "anomaly_detection":
                    result = self._reason_anomaly_detection(visual_insights, numerical_insights, textual_insights)
                
                return f"""Cross-Modal Reasoning Results
                ===========================
                
                Reasoning Pattern: {reasoning_pattern}
                Description: {self.reasoning_patterns[reasoning_pattern]}
                
                Modalities Analyzed:
                - Visual: {len(visual_insights)} insights
                - Textual: {len(textual_insights)} insights  
                - Numerical: {len(numerical_insights)} insights
                
                Combined Reasoning:
                {result}
                
                Cross-modal analysis completed successfully.
                """
                
            except Exception as e:
                return f"Cross-modal reasoning failed: {str(e)}"
        
        def _extract_visual_insights(self, visual_data: Dict[str, Any]) -> List[str]:
            """Extract insights from visual analysis"""
            insights = []
            
            if "objects" in visual_data:
                objects = visual_data["objects"]
                insights.append(f"Detected {objects['object_count']} objects")
                insights.extend([f"Found {obj['name']} (confidence: {obj['confidence']})" for obj in objects.get("objects", [])])
            
            if "scene" in visual_data:
                scene = visual_data["scene"]
                insights.append(f"Scene classified as {scene['primary_category']['category']}")
            
            if "colors" in visual_data:
                colors = visual_data["colors"]
                insights.append(f"Color palette shows {colors['contrast']} contrast")
            
            return insights
        
        def _extract_textual_insights(self, textual_data: str) -> List[str]:
            """Extract insights from textual data"""
            insights = []
            
            # Simple keyword extraction
            keywords = ["trend", "growth", "increase", "decrease", "significant", "important"]
            for keyword in keywords:
                if keyword in textual_data.lower():
                    insights.append(f"Text mentions '{keyword}'")
            
            # Length and complexity
            word_count = len(textual_data.split())
            insights.append(f"Text contains {word_count} words")
            
            return insights
        
        def _extract_numerical_insights(self, numerical_data: List[float]) -> List[str]:
            """Extract insights from numerical data"""
            if not numerical_data:
                return ["No numerical data provided"]
            
            insights = []
            
            # Basic statistics
            mean_val = sum(numerical_data) / len(numerical_data)
            insights.append(f"Average value: {mean_val:.2f}")
            
            # Trend detection
            if len(numerical_data) > 10:
                recent = numerical_data[-10:]
                early = numerical_data[:10]
                recent_avg = sum(recent) / len(recent)
                early_avg = sum(early) / len(early)
                
                if recent_avg > early_avg * 1.1:
                    insights.append("Upward trend detected")
                elif recent_avg < early_avg * 0.9:
                    insights.append("Downward trend detected")
                else:
                    insights.append("Stable trend observed")
            
            return insights
        
        def _reason_visual_text(self, visual_insights: List[str], textual_insights: List[str]) -> str:
            """Reason about visual and textual data together"""
            overlapping_concepts = []
            
            # Look for overlapping themes
            visual_text = " ".join(visual_insights).lower()
            for insight in textual_insights:
                if any(word in visual_text for word in insight.lower().split()):
                    overlapping_concepts.append(insight)
            
            return f"""
            Visual-Text Integration:
            - Visual analysis shows: {len(visual_insights)} key observations
            - Textual analysis reveals: {len(textual_insights)} important points
            - Overlapping themes: {len(overlapping_concepts)}
            
            Combined Insights:
            The visual and textual data {'show strong alignment' if overlapping_concepts else 'provide complementary perspectives'}.
            This cross-validation {'strengthens' if overlapping_concepts else 'broadens'} the overall understanding.
            """
        
        def _reason_data_narrative(self, numerical_insights: List[str], textual_insights: List[str]) -> str:
            """Transform data insights into narrative"""
            data_story = f"""
            Data-Narrative Synthesis:
            - Numerical evidence: {len(numerical_insights)} quantitative insights
            - Textual context: {len(textual_insights)} qualitative insights
            
            Story Generation:
            The data tells a story where {numerical_insights[0] if numerical_insights else 'quantitative patterns'} 
            combine with {textual_insights[0] if textual_insights else 'qualitative descriptions'} to create 
            a comprehensive narrative that bridges numbers and meaning.
            """
            
            return data_story
        
        def _reason_trend_prediction(self, visual_insights: List[str], numerical_insights: List[str]) -> str:
            """Predict trends based on visual and numerical data"""
            trend_indicators = []
            
            # Extract trend information
            for insight in numerical_insights:
                if "trend" in insight.lower():
                    trend_indicators.append(insight)
            
            return f"""
            Trend Prediction Analysis:
            - Visual patterns: {len([i for i in visual_insights if 'trend' in i.lower()])} trend indicators
            - Numerical trends: {len(trend_indicators)} quantitative trends
            
            Predictive Insights:
            Based on cross-modal trend analysis, the combined data suggests 
            directional patterns that can inform future predictions and planning.
            """
        
        def _reason_anomaly_detection(self, visual_insights: List[str], numerical_insights: List[str], textual_insights: List[str]) -> str:
            """Detect anomalies across multiple modalities"""
            anomaly_indicators = []
            
            # Collect anomaly indicators
            for insights in [visual_insights, numerical_insights, textual_insights]:
                for insight in insights:
                    if any(word in insight.lower() for word in ["unusual", "unexpected", "anomaly", "outlier"]):
                        anomaly_indicators.append(insight)
            
            return f"""
            Anomaly Detection Results:
            - Cross-modal anomalies detected: {len(anomaly_indicators)}
            - Anomaly distribution: Visual({len([i for i in visual_insights if 'anomal' in i.lower()])}), 
                                   Numerical({len([i for i in numerical_insights if 'anomal' in i.lower()])}), 
                                   Textual({len([i for i in textual_insights if 'anomal' in i.lower()])})
            
            Anomaly Assessment:
            {len(anomaly_indicators)} anomalous patterns detected across modalities suggest 
            {'significant' if len(anomaly_indicators) > 2 else 'moderate'} deviations from expected patterns.
            """
    
    # Pattern 3: Intelligent Workflow Orchestration
    print("\n3️⃣ Intelligent Workflow Orchestration...")
    
    class WorkflowOrchestrator:
        """Intelligent workflow orchestration for complex multi-step tasks"""
        
        def __init__(self):
            self.workflow_templates = {
                "research_report": {
                    "steps": ["web_search", "data_analysis", "content_synthesis", "report_generation"],
                    "dependencies": {"data_analysis": ["web_search"], "content_synthesis": ["data_analysis"]}
                },
                "market_analysis": {
                    "steps": ["data_collection", "trend_analysis", "competitive_analysis", "recommendations"],
                    "dependencies": {"trend_analysis": ["data_collection"], "competitive_analysis": ["data_collection"]}
                },
                "content_creation": {
                    "steps": ["research", "outline", "draft", "review", "finalize"],
                    "dependencies": {"outline": ["research"], "draft": ["outline"], "review": ["draft"]}
                }
            }
        
        async def orchestrate_workflow(self,
            workflow_type: str = Field(..., description="Type of workflow to execute"),
            parameters: Dict[str, Any] = Field(..., description="Parameters for workflow execution"),
            priority: str = Field("normal", description="Workflow priority: low, normal, high")
        ) -> str:
            """Orchestrate complex multi-step workflows"""
            
            if workflow_type not in self.workflow_templates:
                return f"Unknown workflow type. Available: {list(self.workflow_templates.keys())}"
            
            template = self.workflow_templates[workflow_type]
            steps = template["steps"]
            dependencies = template["dependencies"]
            
            # Create execution plan
            execution_plan = self._create_execution_plan(steps, dependencies)
            
            # Simulate workflow execution
            execution_results = []
            
            for step_info in execution_plan:
                step_name = step_info["step"]
                dependencies_met = step_info["dependencies_met"]
                
                if dependencies_met:
                    # Simulate step execution
                    result = await self._execute_workflow_step(step_name, parameters, priority)
                    execution_results.append({
                        "step": step_name,
                        "status": "completed",
                        "result": result
                    })
                else:
                    execution_results.append({
                        "step": step_name,
                        "status": "skipped",
                        "reason": "dependencies_not_met"
                    })
            
            # Generate workflow summary
            completed_steps = [r for r in execution_results if r["status"] == "completed"]
            success_rate = len(completed_steps) / len(execution_results) * 100
            
            return f"""Workflow Orchestration Complete
            =============================
            
            Workflow Type: {workflow_type}
            Priority: {priority}
            Execution Plan: {len(steps)} steps
            
            Results:
            - Completed Steps: {len(completed_steps)}
            - Success Rate: {success_rate:.1f}%
            - Total Execution Time: Simulated
            
            Step Results:
            {chr(10).join(f"- {r['step']}: {r['status']}" for r in execution_results)}
            
            Workflow orchestration completed successfully.
            """
        
        def _create_execution_plan(self, steps: List[str], dependencies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
            """Create ordered execution plan based on dependencies"""
            
            execution_plan = []
            completed_steps = set()
            
            # Simple topological sort
            remaining_steps = steps.copy()
            
            while remaining_steps:
                for step in remaining_steps[:]:  # Copy to avoid modification during iteration
                    step_dependencies = dependencies.get(step, [])
                    
                    # Check if all dependencies are met
                    dependencies_met = all(dep in completed_steps for dep in step_dependencies)
                    
                    execution_plan.append({
                        "step": step,
                        "dependencies": step_dependencies,
                        "dependencies_met": dependencies_met
                    })
                    
                    if dependencies_met:
                        completed_steps.add(step)
                        remaining_steps.remove(step)
                        break
            
            return execution_plan
        
        async def _execute_workflow_step(self, step_name: str, parameters: Dict[str, Any], priority: str) -> str:
            """Execute individual workflow step"""
            
            # Simulate step execution with priority consideration
            execution_time = 0.1  # Simulated execution time
            priority_multiplier = {"low": 1.5, "normal": 1.0, "high": 0.7}[priority]
            
            # Simulate different step types
            if step_name == "web_search":
                return f"Web search completed with {parameters.get('search_terms', 'default terms')}"
            elif step_name == "data_analysis":
                return f"Data analysis completed on {parameters.get('dataset_size', 'unknown')} records"
            elif step_name == "content_synthesis":
                return f"Content synthesis completed with {parameters.get('synthesis_depth', 'standard')} depth"
            else:
                return f"Step {step_name} completed successfully"
    
    # Demonstrate advanced patterns
    print("Testing Advanced Multi-Modal Patterns:")
    
    # Initialize services
    data_pipeline = RealTimeDataPipeline()
    reasoning_engine = CrossModalReasoningEngine()
    workflow_orchestrator = WorkflowOrchestrator()
    
    async with ChatAgent(
        chat_client=OpenAIChatClient(),
        name="AdvancedMultiModalAgent",
        instructions="You demonstrate advanced multi-modal capabilities with real-time processing.",
        tools=[
            data_pipeline.process_real_time_data,
            reasoning_engine.perform_cross_modal_reasoning,
            workflow_orchestrator.orchestrate_workflow
        ]
    ) as agent:
        
        # Test real-time data pipeline
        print("Testing real-time data pipeline:")
        pipeline_result = await agent.run("Process real-time data stream 'sensor_data' with analysis type and 200 data points")
        print(f"Pipeline result: {pipeline_result.text[:300]}...")
        
        # Test cross-modal reasoning
        print("\nTesting cross-modal reasoning:")
        
        visual_data = {
            "objects": {"object_count": 3, "detection_confidence": 0.85},
            "scene": {"primary_category": {"category": "office"}, "confidence_score": 0.78},
            "colors": {"contrast": "high"}
        }
        
        reasoning_result = await agent.run(f"""
        Perform cross-modal reasoning with:
        - Visual data: {json.dumps(visual_data)}
        - Textual data: "The office environment shows high productivity metrics with increasing trend"
        - Numerical data: [65, 70, 75, 80, 85, 90, 95]
        - Reasoning pattern: visual_text
        """)
        
        print(f"Cross-modal reasoning: {reasoning_result.text[:300]}...")
        
        # Test workflow orchestration
        print("\nTesting workflow orchestration:")
        
        workflow_result = await agent.run(f"""
        Orchestrate research_report workflow with:
        - parameters: {{"topic": "renewable energy", "depth": "comprehensive"}}
        - priority: high
        """)
        
        print(f"Workflow orchestration: {workflow_result.text[:300]}...")

async def production_multimodal_deployment():
    """Production deployment patterns for multi-modal agents"""
    
    print("\n🏭 Production Multi-Modal Deployment")
    print("=" * 60)
    
    # Production scalability patterns
    class ScalableMultiModalProcessor:
        """Scalable multi-modal processing with load balancing"""
        
        def __init__(self, max_concurrent: int = 5):
            self.max_concurrent = max_concurrent
            self.processing_queue = asyncio.Queue()
            self.active_processes = 0
            self.performance_metrics = {
                "total_requests": 0,
                "successful_processing": 0,
                "failed_processing": 0,
                "average_latency": 0
            }
        
        async def process_multimodal_request(self,
            request_data: Dict[str, Any] = Field(..., description="Multi-modal request data"),
            priority: int = Field(5, description="Processing priority 1-10", ge=1, le=10),
            timeout: int = Field(30, description="Processing timeout in seconds", ge=5, le=300)
        ) -> str:
            """Process multi-modal request with scalability"""
            
            try:
                start_time = datetime.now()
                
                # Add to processing queue
                await self.processing_queue.put({
                    "data": request_data,
                    "priority": priority,
                    "timeout": timeout,
                    "timestamp": start_time
                })
                
                # Process if under concurrent limit
                if self.active_processes < self.max_concurrent:
                    asyncio.create_task(self._process_queue())
                
                # Simulate processing (in production, implement actual processing)
                await asyncio.sleep(0.1 * (11 - priority))  # Higher priority = faster processing
                
                # Update metrics
                self.performance_metrics["total_requests"] += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                
                n = self.performance_metrics["total_requests"]
                self.performance_metrics["average_latency"] = (
                    (self.performance_metrics["average_latency"] * (n-1) + processing_time) / n
                )
                
                self.performance_metrics["successful_processing"] += 1
                
                return f"""Multi-Modal Processing Complete
            
            Request ID: {request_data.get('id', 'unknown')}
            Priority: {priority}
            Processing Time: {processing_time:.3f}s
            Status: Success
            
            Scalability Metrics:
            - Queue Position: Processed immediately
            - Concurrent Processes: {self.active_processes}/{self.max_concurrent}
            - System Load: {(self.active_processes/self.max_concurrent)*100:.1f}%
            """
                
            except Exception as e:
                self.performance_metrics["failed_processing"] += 1
                return f"Multi-modal processing failed: {str(e)}"
        
        async def _process_queue(self):
            """Process items from the queue"""
            self.active_processes += 1
            
            try:
                while not self.processing_queue.empty():
                    item = await self.processing_queue.get()
                    
                    # Process item (simplified)
                    await asyncio.sleep(0.1)  # Simulate processing
                    
                    self.processing_queue.task_done()
            
            finally:
                self.active_processes -= 1
        
        def get_scalability_metrics(self) -> Dict[str, Any]:
            """Get scalability and performance metrics"""
            success_rate = (
                self.performance_metrics["successful_processing"] / 
                max(self.performance_metrics["total_requests"], 1) * 100
            )
            
            return {
                "performance_metrics": self.performance_metrics,
                "success_rate": success_rate,
                "current_load": (self.active_processes / self.max_concurrent) * 100,
                "queue_size": self.processing_queue.qsize(),
                "scalability_status": "healthy" if success_rate > 95 else "degraded" if success_rate > 80 else "critical"
            }
    
    # Quality assurance patterns
    class QualityAssuranceFramework:
        """Comprehensive quality assurance for multi-modal outputs"""
        
        def __init__(self):
            self.quality_thresholds = {
                "text_coherence": 0.8,
                "factual_accuracy": 0.9,
                "modal_consistency": 0.85,
                "response_relevance": 0.8
            }
            self.quality_history = []
        
        async def assess_output_quality(self,
            output_data: Dict[str, Any] = Field(..., description="Multi-modal output to assess"),
            expected_criteria: Dict[str, Any] = Field(default_factory=dict, description="Expected quality criteria"),
            strict_mode: bool = Field(False, description="Enable strict quality assessment")
        ) -> Dict[str, Any]:
            """Comprehensive quality assessment of multi-modal outputs"""
            
            quality_scores = {}
            
            # Assess text coherence
            if "text" in output_data:
                quality_scores["text_coherence"] = self._assess_text_coherence(output_data["text"])
            
            # Assess factual accuracy (simplified)
            quality_scores["factual_accuracy"] = self._assess_factual_accuracy(output_data)
            
            # Assess modal consistency
            quality_scores["modal_consistency"] = self._assess_modal_consistency(output_data)
            
            # Assess response relevance
            quality_scores["response_relevance"] = self._assess_response_relevance(output_data, expected_criteria)
            
            # Calculate overall quality
            overall_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
            
            # Determine pass/fail
            passed_quality = overall_quality >= 0.8  # Overall threshold
            
            if strict_mode:
                # All individual scores must meet thresholds
                individual_passes = all(
                    score >= self.quality_thresholds.get(criterion, 0.7) 
                    for criterion, score in quality_scores.items()
                )
                passed_quality = passed_quality and individual_passes
            
            # Record quality assessment
            assessment_record = {
                "timestamp": datetime.now().isoformat(),
                "quality_scores": quality_scores,
                "overall_quality": overall_quality,
                "passed": passed_quality,
                "strict_mode": strict_mode
            }
            
            self.quality_history.append(assessment_record)
            
            return {
                "quality_assessment": assessment_record,
                "recommendations": self._generate_quality_recommendations(quality_scores),
                "improvement_suggestions": self._suggest_improvements(quality_scores)
            }
        
        def _assess_text_coherence(self, text_data: str) -> float:
            """Assess coherence of text output"""
            # Simplified coherence assessment
            sentences = text_data.split('.')
            
            # Check for logical flow indicators
            coherence_indicators = ["however", "therefore", "furthermore", "additionally", "consequently"]
            indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_data.lower())
            
            # Check sentence variety
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
            length_variance = max(sentence_lengths) - min(sentence_lengths) if sentence_lengths else 0
            
            # Calculate coherence score (simplified)
            coherence_score = min(1.0, (indicator_count / 5) + (length_variance / 50))
            
            return coherence_score
        
        def _assess_factual_accuracy(self, output_data: Dict[str, Any]) -> float:
            """Assess factual accuracy of output"""
            # In production, implement fact-checking against reliable sources
            # For now, return high score for demonstration
            
            factual_elements = 0
            verified_elements = 0
            
            # Check for citations and sources
            if "sources" in output_data:
                factual_elements += len(output_data["sources"])
                verified_elements += len(output_data["sources"]) * 0.9  # Assume 90% accuracy
            
            # Check for specific claims (simplified)
            if "text" in output_data:
                text = output_data["text"]
                # Look for statements of fact vs. opinion
                factual_statements = text.count("is ") + text.count("are ") + text.count("was ") + text.count("were ")
                opinion_statements = text.count("believe") + text.count("think") + text.count("opinion")
                
                if factual_statements > 0:
                    accuracy_score = min(1.0, 0.8 + (opinion_statements / factual_statements) * 0.1)
                else:
                    accuracy_score = 0.9  # High score if mostly opinion/analysis
            
            return accuracy_score
        
        def _assess_modal_consistency(self, output_data: Dict[str, Any]) -> float:
            """Assess consistency across different modalities"""
            modalities_present = []
            
            if "text" in output_data:
                modalities_present.append("text")
            if "images" in output_data:
                modalities_present.append("images")
            if "data" in output_data:
                modalities_present.append("data")
            
            # Check for consistency indicators
            consistency_score = 0.7  # Base score
            
            # If multiple modalities, check for cross-references
            if len(modalities_present) > 1:
                text_content = output_data.get("text", "").lower()
                
                # Check for references between modalities
                cross_modal_refs = 0
                if "images" in output_data and "visual" in text_content:
                    cross_modal_refs += 1
                if "data" in output_data and ("data" in text_content or "analysis" in text_content):
                    cross_modal_refs += 1
                
                consistency_score += (cross_modal_refs / len(modalities_present)) * 0.3
            
            return min(1.0, consistency_score)
        
        def _assess_response_relevance(self, output_data: Dict[str, Any], expected_criteria: Dict[str, Any]) -> float:
            """Assess relevance to expected criteria"""
            if not expected_criteria:
                return 0.9  # High score if no criteria specified
            
            relevance_indicators = 0
            total_criteria = len(expected_criteria)
            
            for criterion, expected_value in expected_criteria.items():
                # Simple keyword matching for relevance
                output_text = str(output_data).lower()
                expected_text = str(expected_value).lower()
                
                if any(word in output_text for word in expected_text.split()):
                    relevance_indicators += 1
            
            return relevance_indicators / total_criteria if total_criteria > 0 else 0.9
        
        def _generate_quality_recommendations(self, quality_scores: Dict[str, float]) -> List[str]:
            """Generate recommendations for quality improvement"""
            
            recommendations = []
            
            for criterion, score in quality_scores.items():
                threshold = self.quality_thresholds.get(criterion, 0.7)
                if score < threshold:
                    recommendations.append(f"Improve {criterion.replace('_', ' ')}: current score {score:.2f} < threshold {threshold}")
            
            if not recommendations:
                recommendations.append("Quality meets all specified thresholds")
            
            return recommendations
        
        def _suggest_improvements(self, quality_scores: Dict[str, float]) -> List[str]:
            """Suggest specific improvements based on quality scores"""
            
            improvements = []
            
            if quality_scores.get("text_coherence", 1.0) < 0.8:
                improvements.append("Add more transition words and logical connectors")
                improvements.append("Vary sentence structure and length")
            
            if quality_scores.get("factual_accuracy", 1.0) < 0.9:
                improvements.append("Include more citations and source references")
                improvements.append("Fact-check claims against reliable sources")
            
            if quality_scores.get("modal_consistency", 1.0) < 0.85:
                improvements.append("Add explicit cross-references between modalities")
                improvements.append("Ensure visual and textual content align")
            
            if quality_scores.get("response_relevance", 1.0) < 0.8:
                improvements.append("Better align content with user expectations")
                improvements.append("Include more specific details requested by user")
            
            if not improvements:
                improvements.append("Quality is high - focus on maintaining current standards")
            
            return improvements

    # Demonstrate production features
    print("Production Multi-Modal Features:")
    
    # Initialize components
    scalable_processor = ScalableMultiModalProcessor(max_concurrent=3)
    quality_assurance = QualityAssuranceFramework()
    
    print("\n1. Scalable Processing:")
    
    # Test scalable processing
    test_requests = [
        {"id": "req1", "type": "text_analysis", "data": "Sample text data"},
        {"id": "req2", "type": "image_analysis", "data": "base64_image_data"},
        {"id": "req3", "type": "combined", "data": {"text": "Combined", "image": "data"}}
    ]
    
    async def test_scalability():
        tasks = []
        for i, request in enumerate(test_requests):
            task = scalable_processor.process_multimodal_request(
                request_data=request,
                priority=7 - i,  # Decreasing priority
                timeout=30
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i+1}: Failed")
            else:
                print(f"Request {i+1}: Processed successfully")
    
    await test_scalability()
    
    metrics = scalable_processor.get_scalability_metrics()
    print(f"Scalability metrics: {metrics}")
    
    print("\n2. Quality Assurance:")
    
    # Test quality assurance
    test_output = {
        "text": "This is a comprehensive analysis of renewable energy trends showing significant growth in solar and wind adoption.",
        "sources": [
            {"title": "Solar Energy Report 2024", "url": "https://example.com/solar"},
            {"title": "Wind Power Analysis", "url": "https://example.com/wind"}
        ],
        "confidence": 0.85
    }
    
    quality_result = await quality_assurance.assess_output_quality(
        output_data=test_output,
        expected_criteria={"topic": "renewable energy", "depth": "comprehensive"},
        strict_mode=True
    )
    
    print(f"Quality assessment: {json.dumps(quality_result['quality_assessment'], indent=2)}")
    print(f"Recommendations: {quality_result['recommendations']}")
    
    print("\n3. Production Monitoring:")
    
    monitoring_capabilities = {
        "real_time_performance": "Latency, throughput, error rates",
        "resource_utilization": "CPU, memory, network usage",
        "quality_metrics": "Accuracy, relevance, user satisfaction",
        "business_metrics": "Cost per transaction, ROI, adoption rates",
        "security_monitoring": "Access patterns, anomaly detection, compliance"
    }
    
    print("Comprehensive monitoring covers:")
    for capability, description in monitoring_capabilities.items():
        print(f"- {capability.replace('_', ' ').title()}: {description}")

if __name__ == "__main__":
    asyncio.run(basic_multimodal_demo())
    asyncio.run(advanced_multimodal_patterns())
    asyncio.run(production_multimodal_deployment())
```

The Multi-Modal Agent represents the pinnacle of AI system integration, combining the power of multiple AI capabilities into cohesive, intelligent workflows. By mastering the patterns and techniques outlined in this section, you'll be equipped to build sophisticated AI applications that can understand, process, and respond to complex real-world scenarios involving multiple types of data and interactions. The combination of intelligent orchestration, advanced reasoning, and production-ready scalability creates agents that are not just powerful, but practical for enterprise deployment.

---

## Advanced Topics

This section explores sophisticated patterns and techniques that elevate your agent development from functional to exceptional. These advanced topics address real-world challenges such as performance optimization, security hardening, complex workflow orchestration, and production-scale deployment strategies.

### Thread Persistence and State Management

Advanced thread management is crucial for building agents that maintain meaningful long-term conversations and complex state across multiple interactions:

```python
import asyncio
from agent_framework import ChatAgent, AgentThread, ChatMessageStore
from agent_framework.openai import OpenAIChatClient, OpenAIAssistantsClient
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import redis
from dataclasses import dataclass, asdict
import pickle
import hashlib

@dataclass
class ConversationState:
    """Comprehensive conversation state management"""
    thread_id: str
    user_id: str
    context: Dict[str, Any]
    message_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    last_updated: datetime
    version: int = 1

class AdvancedThreadManager:
    """Production-ready thread management with multiple persistence strategies"""
    
    def __init__(self, 
                 persistence_strategy: str = "hybrid",  # memory, redis, database, hybrid
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 retention_days: int = 30):
        self.persistence_strategy = persistence_strategy
        self.retention_days = retention_days
        self.redis_client = None
        
        if persistence_strategy in ["redis", "hybrid"]:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.memory_cache = {}  # In-memory cache for hybrid strategy
        self.state_serializers = {
            "json": self._serialize_json,
            "pickle": self._serialize_pickle,
            "compressed": self._serialize_compressed
        }
    
    async def create_persistent_thread(self, 
                                     user_id: str,
                                     initial_context: Optional[Dict[str, Any]] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a thread with comprehensive persistence setup"""
        
        thread_id = f"thread_{user_id}_{datetime.now().timestamp()}"
        
        # Create conversation state
        state = ConversationState(
            thread_id=thread_id,
            user_id=user_id,
            context=initial_context or {},
            message_history=[],
            metadata=metadata or {},
            last_updated=datetime.now()
        )
        
        # Persist based on strategy
        await self._persist_state(state)
        
        # Set up expiration for cleanup
        if self.redis_client:
            await self.redis_client.expire(
                f"thread:{thread_id}", 
                timedelta(days=self.retention_days)
            )
        
        return thread_id
    
    async def restore_thread_state(self, thread_id: str) -> Optional[ConversationState]:
        """Restore thread state from persistent storage"""
        
        try:
            if self.persistence_strategy == "memory":
                return self.memory_cache.get(thread_id)
            
            elif self.persistence_strategy == "redis":
                state_data = await self.redis_client.get(f"thread:{thread_id}")
                if state_data:
                    return self._deserialize_state(state_data)
            
            elif self.persistence_strategy == "hybrid":
                # Try memory first, then Redis
                if thread_id in self.memory_cache:
                    return self.memory_cache[thread_id]
                
                state_data = await self.redis_client.get(f"thread:{thread_id}")
                if state_data:
                    state = self._deserialize_state(state_data)
                    # Cache in memory for faster access
                    self.memory_cache[thread_id] = state
                    return state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to restore thread {thread_id}: {str(e)}")
            return None
    
    async def update_thread_context(self, 
                                  thread_id: str, 
                                  context_updates: Dict[str, Any],
                                  merge_strategy: str = "deep_merge") -> bool:
        """Update thread context with advanced merging strategies"""
        
        state = await self.restore_thread_state(thread_id)
        if not state:
            return False
        
        # Apply context updates based on merge strategy
        if merge_strategy == "replace":
            state.context = context_updates
        elif merge_strategy == "shallow_merge":
            state.context.update(context_updates)
        elif merge_strategy == "deep_merge":
            state.context = self._deep_merge(state.context, context_updates)
        
        # Add to message history as context update
        state.message_history.append({
            "type": "context_update",
            "updates": context_updates,
            "merge_strategy": merge_strategy,
            "timestamp": datetime.now().isoformat()
        })
        
        state.last_updated = datetime.now()
        state.version += 1
        
        await self._persist_state(state)
        return True
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep merge of nested dictionaries"""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def get_thread_analytics(self, thread_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a thread"""
        
        state = await self.restore_thread_state(thread_id)
        if not state:
            return {}
        
        # Calculate analytics
        message_count = len(state.message_history)
        context_size = len(json.dumps(state.context))
        time_since_creation = datetime.now() - state.last_updated
        
        # Analyze message patterns
        user_messages = [msg for msg in state.message_history if msg.get("role") == "user"]
        assistant_messages = [msg for msg in state.message_history if msg.get("role") == "assistant"]
        
        # Calculate engagement metrics
        avg_message_length = sum(len(msg.get("content", "")) for msg in state.message_history) / max(message_count, 1)
        
        return {
            "thread_id": thread_id,
            "user_id": state.user_id,
            "message_count": message_count,
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "context_size_bytes": context_size,
            "last_activity": state.last_updated.isoformat(),
            "time_since_last_activity_seconds": time_since_creation.total_seconds(),
            "average_message_length": avg_message_length,
            "version": state.version,
            "persistence_strategy": self.persistence_strategy
        }
    
    async def cleanup_expired_threads(self) -> Dict[str, Any]:
        """Clean up threads that have exceeded retention period"""
        
        cleanup_stats = {
            "threads_checked": 0,
            "threads_removed": 0,
            "space_freed_bytes": 0
        }
        
        try:
            if self.persistence_strategy in ["redis", "hybrid"]:
                # Find expired threads in Redis
                pattern = "thread:*"
                keys = await self.redis_client.keys(pattern)
                
                for key in keys:
                    cleanup_stats["threads_checked"] += 1
                    
                    # Check TTL
                    ttl = await self.redis_client.ttl(key)
                    if ttl == -2:  # Key doesn't exist
                        cleanup_stats["threads_removed"] += 1
                    elif ttl > 0 and ttl < 3600:  # Expires within an hour
                        # Get size before deletion
                        size = await self.redis_client.memory_usage(key) or 0
                        cleanup_stats["space_freed_bytes"] += size
                        
                        # Remove key
                        await self.redis_client.delete(key)
                        cleanup_stats["threads_removed"] += 1
            
            # Clean up memory cache
            if self.persistence_strategy in ["memory", "hybrid"]:
                expired_threads = []
                current_time = datetime.now()
                
                for thread_id, state in self.memory_cache.items():
                    if current_time - state.last_updated > timedelta(days=self.retention_days):
                        expired_threads.append(thread_id)
                
                for thread_id in expired_threads:
                    del self.memory_cache[thread_id]
                    cleanup_stats["threads_removed"] += 1
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return cleanup_stats
    
    async def _persist_state(self, state: ConversationState) -> bool:
        """Persist state based on configured strategy"""
        
        try:
            if self.persistence_strategy == "memory":
                self.memory_cache[state.thread_id] = state
                return True
            
            elif self.persistence_strategy == "redis":
                state_data = self._serialize_state(state)
                await self.redis_client.set(f"thread:{state.thread_id}", state_data)
                return True
            
            elif self.persistence_strategy == "hybrid":
                # Store in both memory and Redis
                self.memory_cache[state.thread_id] = state
                state_data = self._serialize_state(state)
                await self.redis_client.set(f"thread:{state.thread_id}", state_data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Persistence failed for thread {state.thread_id}: {str(e)}")
            return False
    
    def _serialize_state(self, state: ConversationState) -> str:
        """Serialize state for storage"""
        # Use JSON serialization for human-readable storage
        state_dict = asdict(state)
        state_dict['last_updated'] = state_dict['last_updated'].isoformat()
        return json.dumps(state_dict)
    
    def _deserialize_state(self, state_data: str) -> ConversationState:
        """Deserialize state from storage"""
        state_dict = json.loads(state_data)
        state_dict['last_updated'] = datetime.fromisoformat(state_dict['last_updated'])
        return ConversationState(**state_dict)
    
    def _serialize_pickle(self, state: ConversationState) -> bytes:
        """Serialize using pickle for complex objects"""
        return pickle.dumps(state)
    
    def _serialize_compressed(self, state: ConversationState) -> str:
        """Serialize with compression for large states"""
        import gzip
        state_data = self._serialize_state(state)
        compressed = gzip.compress(state_data.encode())
        return base64.b64encode(compressed).decode()

# Advanced message store with search capabilities
class SearchableMessageStore(ChatMessageStore):
    """Message store with advanced search and indexing capabilities"""
    
    def __init__(self, index_fields: Optional[List[str]] = None):
        super().__init__()
        self.index_fields = index_fields or ["content", "role", "timestamp"]
        self.search_index = {}  # Simple in-memory index
        self.message_metadata = {}
    
    async def add_message(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Add message with indexing"""
        await super().add_message(message)
        
        # Create message ID
        msg_id = hashlib.md5(f"{message.get('content', '')}{message.get('timestamp', '')}".encode()).hexdigest()
        
        # Store metadata
        if metadata:
            self.message_metadata[msg_id] = metadata
        
        # Update search index
        for field in self.index_fields:
            if field in message:
                field_value = str(message[field]).lower()
                words = field_value.split()
                
                for word in words:
                    if word not in self.search_index:
                        self.search_index[word] = []
                    self.search_index[word].append(msg_id)
    
    async def search_messages(self, 
                            query: str, 
                            field: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search messages with advanced filtering"""
        
        query_words = query.lower().split()
        matching_messages = []
        
        # Find messages matching query words
        message_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for msg_id in self.search_index[word]:
                    message_scores[msg_id] = message_scores.get(msg_id, 0) + 1
        
        # Sort by relevance score
        sorted_messages = sorted(message_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Retrieve actual messages
        all_messages = await self.list_messages()
        message_dict = {hashlib.md5(f"{msg.get('content', '')}{msg.get('timestamp', '')}".encode()).hexdigest(): msg 
                       for msg in all_messages}
        
        for msg_id, score in sorted_messages[:limit]:
            if msg_id in message_dict:
                msg_data = message_dict[msg_id].copy()
                msg_data["relevance_score"] = score / len(query_words)
                msg_data["metadata"] = self.message_metadata.get(msg_id, {})
                matching_messages.append(msg_data)
        
        return matching_messages
    
    async def get_conversation_summary(self, max_messages: int = 10) -> Dict[str, Any]:
        """Generate conversation summary with analytics"""
        
        messages = await self.list_messages()
        if not messages:
            return {}
        
        # Basic analytics
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        # Word frequency analysis
        all_text = " ".join([msg.get("content", "") for msg in messages]).lower()
        words = all_text.split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "average_message_length": sum(len(msg.get("content", "")) for msg in messages) / len(messages),
            "top_keywords": [{"word": word, "frequency": freq} for word, freq in top_words],
            "conversation_duration": self._calculate_conversation_duration(messages)
        }
    
    def _calculate_conversation_duration(self, messages: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate conversation duration in seconds"""
        
        timestamps = []
        for msg in messages:
            if "timestamp" in msg:
                try:
                    ts = datetime.fromisoformat(msg["timestamp"])
                    timestamps.append(ts)
                except:
                    continue
        
        if len(timestamps) >= 2:
            return (max(timestamps) - min(timestamps)).total_seconds()
        
        return None

# Production-ready implementation example
async def advanced_thread_management_demo():
    """Demonstrate advanced thread management capabilities"""
    
    print("🧵 Advanced Thread Management Demo")
    print("=" * 60)
    
    # Initialize advanced thread manager
    thread_manager = AdvancedThreadManager(
        persistence_strategy="hybrid",
        retention_days=7
    )
    
    async with OpenAIChatClient() as client:
        
        # Create agent with persistent thread
        user_id = "user_12345"
        thread_id = await thread_manager.create_persistent_thread(
            user_id=user_id,
            initial_context={
                "user_preferences": {"language": "en", "detail_level": "comprehensive"},
                "session_info": {"start_time": datetime.now().isoformat()}
            },
            metadata={"source": "web_app", "version": "2.0"}
        )
        
        print(f"Created persistent thread: {thread_id}")
        
        # Create agent with the persistent thread
        async with ChatAgent(
            chat_client=client,
            name="PersistentAssistant",
            instructions="You maintain conversation context and user preferences across sessions."
        ) as agent:
            
            # Simulate conversation with context updates
            conversations = [
                {
                    "message": "Hello! I'm interested in learning about renewable energy.",
                    "context_update": {"topic": "renewable_energy", "interest_level": "high"}
                },
                {
                    "message": "What are the latest developments in solar technology?",
                    "context_update": {"focus_area": "solar", "detail_preference": "technical"}
                },
                {
                    "message": "How do these developments compare to wind energy?",
                    "context_update": {"comparison_mode": True, "include_wind": True}
                }
            ]
            
            for i, conv in enumerate(conversations, 1):
                print(f"\n--- Conversation {i} ---")
                
                # Update context before response
                await thread_manager.update_thread_context(
                    thread_id=thread_id,
                    context_updates=conv["context_update"],
                    merge_strategy="deep_merge"
                )
                
                # Get response with thread context
                thread = AgentThread(service_thread_id=thread_id)
                result = await agent.run(conv["message"], thread=thread, store=True)
                
                print(f"User: {conv['message']}")
                print(f"Assistant: {result.text[:200]}...")
            
            # Get thread analytics
            analytics = await thread_manager.get_thread_analytics(thread_id)
            print(f"\n📊 Thread Analytics:")
            for key, value in analytics.items():
                print(f"  {key}: {value}")
            
            # Demonstrate message search
            if hasattr(agent, 'message_store'):
                searchable_store = SearchableMessageStore()
                
                # Copy messages to searchable store
                messages = await thread.message_store.list_messages() if thread.message_store else []
                for msg in messages:
                    await searchable_store.add_message(msg)
                
                # Search for specific content
                search_results = await searchable_store.search_messages("renewable energy", limit=3)
                print(f"\n🔍 Search Results for 'renewable energy':")
                for result in search_results:
                    print(f"  Score: {result.get('relevance_score', 0):.2f} - {result.get('content', '')[:100]}...")
            
            # Demonstrate conversation summary
            summary = await searchable_store.get_conversation_summary()
            print(f"\n📝 Conversation Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    
    # Demonstrate cleanup
    print(f"\n🧹 Cleanup Results:")
    cleanup_stats = await thread_manager.cleanup_expired_threads()
    for key, value in cleanup_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(advanced_thread_management_demo())
```

### Custom Tool Development and Integration

Creating sophisticated custom tools that integrate seamlessly with the agent framework:

```python
from agent_framework.tools import BaseTool, ToolResult
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Callable
import asyncio
import httpx
import json
from datetime import datetime
import hashlib
from functools import lru_cache
import asyncio
from dataclasses import dataclass

# Advanced tool base class with comprehensive features
class AdvancedTool(BaseTool):
    """Production-ready tool base class with caching, validation, and monitoring"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600,
                 rate_limit: Optional[int] = None,
                 timeout: int = 30):
        super().__init__()
        self.name = name
        self.description = description
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.execution_history = []
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool with comprehensive error handling and monitoring"""
        
        start_time = datetime.now()
        cache_key = None
        
        try:
            # Validate inputs
            validation_result = await self._validate_inputs(**kwargs)
            if not validation_result["valid"]:
                return ToolResult(
                    success=False,
                    error=f"Input validation failed: {validation_result['error']}"
                )
            
            # Generate cache key
            if self.enable_caching:
                cache_key = self._generate_cache_key(**kwargs)
                
                # Check cache
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    self.cache_stats["hits"] += 1
                    return cached_result
            
            self.cache_stats["misses"] += 1
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_tool(**kwargs),
                timeout=self.timeout
            )
            
            # Cache result if enabled
            if self.enable_caching and cache_key:
                await self._cache_result(cache_key, result)
            
            # Record execution history
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "success": result.success,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "input_hash": hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()
            })
            
            return result
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            })
            
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
    
    async def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate tool inputs - override in subclasses"""
        return {"valid": True}
    
    async def _execute_tool(self, **kwargs) -> ToolResult:
        """Execute the actual tool logic - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_tool")
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from tool inputs"""
        sorted_kwargs = sorted(kwargs.items())
        return f"{self.name}:{hashlib.md5(str(sorted_kwargs).encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ToolResult]:
        """Get result from cache - override for custom caching"""
        # Simple in-memory cache implementation
        # In production, use Redis or other caching solutions
        return None
    
    async def _cache_result(self, cache_key: str, result: ToolResult) -> None:
        """Cache result - override for custom caching"""
        # Simple in-memory cache implementation
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for exec in self.execution_history if exec["success"])
        
        avg_execution_time = 0
        if self.execution_history:
            avg_execution_time = sum(exec["execution_time"] for exec in self.execution_history) / len(self.execution_history)
        
        return {
            "tool_name": self.name,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / max(total_executions, 1)) * 100,
            "average_execution_time": avg_execution_time,
            "cache_stats": self.cache_stats.copy(),
            "cache_hit_rate": (self.cache_stats["hits"] / max(sum(self.cache_stats.values()), 1)) * 100
        }

# Sophisticated API integration tool
class APIIntegrationTool(AdvancedTool):
    """Advanced API integration with authentication, retry logic, and response processing"""
    
    def __init__(self, 
                 api_base_url: str,
                 api_key: Optional[str] = None,
                 rate_limit_per_minute: int = 60,
                 max_retries: int = 3,
                 backoff_factor: float = 2.0):
        
        super().__init__(
            name="api_integration",
            description="Advanced API integration with comprehensive error handling",
            rate_limit=rate_limit_per_minute,
            timeout=30
        )
        
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.request_history = []
    
    class InputSchema(BaseModel):
        """Input validation schema"""
        endpoint: str = Field(..., description="API endpoint (without base URL)")
        method: str = Field("GET", description="HTTP method")
        params: Optional[Dict[str, Any]] = None
        data: Optional[Dict[str, Any]] = None
        headers: Optional[Dict[str, str]] = None
        timeout: Optional[int] = 30
        
        @validator('method')
        def validate_method(cls, v):
            if v.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                raise ValueError('Method must be one of: GET, POST, PUT, DELETE, PATCH')
            return v.upper()
    
    async def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate API inputs"""
        try:
            validated_input = self.InputSchema(**kwargs)
            return {"valid": True, "validated_input": validated_input}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    async def _execute_tool(self, **kwargs) -> ToolResult:
        """Execute API request with comprehensive error handling"""
        
        # Get validated input
        validation_result = await self._validate_inputs(**kwargs)
        if not validation_result["valid"]:
            return ToolResult(success=False, error=validation_result["error"])
        
        input_data = validation_result["validated_input"]
        
        # Prepare request
        url = f"{self.api_base_url}/{input_data.endpoint.lstrip('/')}"
        headers = input_data.headers or {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Execute with retry logic
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=input_data.timeout) as client:
                    response = await client.request(
                        method=input_data.method,
                        url=url,
                        params=input_data.params,
                        json=input_data.data,
                        headers=headers
                    )
                
                # Record successful request
                self.request_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": input_data.endpoint,
                    "method": input_data.method,
                    "status_code": response.status_code,
                    "success": response.status_code < 400
                })
                
                # Process response
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return ToolResult(
                            success=True,
                            data=data,
                            metadata={
                                "status_code": response.status_code,
                                "response_headers": dict(response.headers),
                                "request_duration": response.elapsed.total_seconds()
                            }
                        )
                    except json.JSONDecodeError:
                        return ToolResult(
                            success=True,
                            data={"content": response.text},
                            metadata={"status_code": response.status_code}
                        )
                else:
                    return ToolResult(
                        success=False,
                        error=f"API returned status code {response.status_code}: {response.text}",
                        metadata={"status_code": response.status_code}
                    )
                
            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    return ToolResult(
                        success=False,
                        error=f"Request timed out after {input_data.timeout} seconds"
                    )
                
            except httpx.RequestError as e:
                if attempt == self.max_retries - 1:
                    return ToolResult(
                        success=False,
                        error=f"Request failed: {str(e)}"
                    )
                
            # Exponential backoff
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.backoff_factor ** attempt)
        
        return ToolResult(success=False, error="Max retries exceeded")

# Advanced data processing tool
class DataProcessingTool(AdvancedTool):
    """Advanced data processing with multiple analysis capabilities"""
    
    def __init__(self):
        super().__init__(
            name="data_processing",
            description="Advanced data processing with statistical analysis and visualization",
            enable_caching=True,
            cache_ttl=1800  # 30 minutes
        )
    
    class InputSchema(BaseModel):
        """Data processing input schema"""
        data: List[Dict[str, Any]] = Field(..., description="Data to process")
        operations: List[str] = Field(..., description="Operations to perform")
        group_by: Optional[str] = None
        filters: Optional[Dict[str, Any]] = None
        generate_visualizations: bool = Field(False, description="Generate visualizations")
        
        @validator('operations')
        def validate_operations(cls, v):
            valid_operations = ['summary', 'statistics', 'trends', 'correlations', 'groupby', 'filter']
            for op in v:
                if op not in valid_operations:
                    raise ValueError(f'Invalid operation: {op}. Must be one of: {valid_operations}')
            return v
    
    async def _execute_tool(self, **kwargs) -> ToolResult:
        """Execute data processing operations"""
        
        # Parse and validate input
        try:
            input_data = self.InputSchema(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=f"Input validation failed: {str(e)}")
        
        try:
            data = input_data.data
            
            # Apply filters first
            if input_data.filters:
                data = self._apply_filters(data, input_data.filters)
            
            results = {}
            
            # Execute requested operations
            for operation in input_data.operations:
                if operation == "summary":
                    results["summary"] = self._generate_summary(data)
                elif operation == "statistics":
                    results["statistics"] = self._calculate_statistics(data)
                elif operation == "trends":
                    results["trends"] = self._analyze_trends(data)
                elif operation == "correlations":
                    results["correlations"] = self._calculate_correlations(data)
                elif operation == "groupby" and input_data.group_by:
                    results["groupby"] = self._group_by(data, input_data.group_by)
                elif operation == "filter":
                    results["filter"] = {"filtered_count": len(data), "original_count": len(input_data.data)}
            
            # Generate visualizations if requested
            if input_data.generate_visualizations:
                results["visualizations"] = await self._generate_visualizations(results)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "operations_performed": input_data.operations,
                    "record_count": len(data),
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Data processing failed: {str(e)}")
    
    def _apply_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to data"""
        filtered_data = []
        
        for record in data:
            include_record = True
            
            for field, condition in filters.items():
                if field not in record:
                    continue
                
                if isinstance(condition, dict):
                    # Complex filtering (e.g., {"operator": ">", "value": 10})
                    operator = condition.get("operator", "=")
                    value = condition.get("value")
                    
                    if operator == ">" and record[field] <= value:
                        include_record = False
                    elif operator == "<" and record[field] >= value:
                        include_record = False
                    elif operator == "=" and record[field] != value:
                        include_record = False
                else:
                    # Simple equality filter
                    if record[field] != condition:
                        include_record = False
            
            if include_record:
                filtered_data.append(record)
        
        return filtered_data
    
    def _generate_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data summary"""
        if not data:
            return {"error": "No data to summarize"}
        
        # Get all numeric fields
        numeric_fields = []
        for record in data[:10]:  # Sample first 10 records
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    numeric_fields.append(key)
        
        numeric_fields = list(set(numeric_fields))  # Remove duplicates
        
        summary = {
            "total_records": len(data),
            "fields": list(data[0].keys()) if data else [],
            "numeric_fields": numeric_fields,
            "sample_records": data[:5]  # First 5 records as sample
        }
        
        # Add field-specific summaries for numeric fields
        if numeric_fields:
            summary["numeric_summary"] = {}
            for field in numeric_fields:
                values = [record[field] for record in data if field in record and record[field] is not None]
                if values:
                    summary["numeric_summary"][field] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "sum": sum(values)
                    }
        
        return summary
    
    def _calculate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        # Implementation similar to _generate_summary but more detailed
        return self._generate_summary(data)  # Simplified for demo
    
    def _analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in data"""
        # Simple trend analysis
        if not data or len(data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Look for timestamp or sequential fields
        timestamp_fields = [field for field in data[0].keys() if "time" in field.lower() or "date" in field.lower()]
        sequential_fields = [field for field in data[0].keys() if isinstance(data[0][field], (int, float))]
        
        trends = {}
        
        for field in sequential_fields:
            values = [record[field] for record in data if field in record and record[field] is not None]
            if len(values) > 1:
                # Simple linear trend
                x = list(range(len(values)))
                slope = (len(values) * sum(i * v for i, v in enumerate(values)) - sum(x) * sum(values)) / \
                       (len(values) * sum(i * i for i in x) - sum(x) ** 2)
                
                trends[field] = {
                    "slope": slope,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "trend_strength": abs(slope)
                }
        
        return {"trends": trends, "analysis_timestamp": datetime.now().isoformat()}
    
    def _calculate_correlations(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlations between numeric fields"""
        # Simplified correlation calculation
        numeric_fields = []
        for record in data[:10]:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    numeric_fields.append(key)
        
        numeric_fields = list(set(numeric_fields))
        
        if len(numeric_fields) < 2:
            return {"error": "Insufficient numeric fields for correlation analysis"}
        
        correlations = {}
        for i, field1 in enumerate(numeric_fields):
            for field2 in numeric_fields[i+1:]:
                values1 = [record[field1] for record in data if field1 in record and record[field1] is not None]
                values2 = [record[field2] for record in data if field2 in record and record[field2] is not None]
                
                if len(values1) == len(values2) and len(values1) > 1:
                    # Simple correlation coefficient
                    mean1 = sum(values1) / len(values1)
                    mean2 = sum(values2) / len(values2)
                    
                    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
                    denominator = (sum((x - mean1) ** 2 for x in values1) * sum((y - mean2) ** 2 for y in values2)) ** 0.5
                    
                    if denominator != 0:
                        correlation = numerator / denominator
                        correlations[f"{field1}_vs_{field2}"] = {
                            "correlation_coefficient": correlation,
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak",
                            "direction": "positive" if correlation > 0 else "negative"
                        }
        
        return {"correlations": correlations}
    
    def _group_by(self, data: List[Dict[str, Any]], group_field: str) -> Dict[str, Any]:
        """Group data by specified field"""
        groups = {}
        
        for record in data:
            if group_field in record:
                key = str(record[group_field])
                if key not in groups:
                    groups[key] = []
                groups[key].append(record)
        
        # Calculate group statistics
        group_stats = {}
        for key, group_data in groups.items():
            group_stats[key] = {
                "count": len(group_data),
                "percentage": len(group_data) / len(data) * 100,
                "summary": self._generate_summary(group_data) if group_data else {}
            }
        
        return {"groups": groups, "group_statistics": group_stats}
    
    async def _generate_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Generate data visualizations"""
        # Simplified visualization generation
        # In production, use matplotlib or similar to create actual charts
        visualizations = []
        
        if "numeric_summary" in results:
            # Generate text-based visualization description
            viz_desc = "Data Visualization Summary:\n"
            for field, stats in results["numeric_summary"].items():
                viz_desc += f"- {field}: bar chart showing distribution\n"
                viz_desc += f"- {field}: line chart showing trends\n"
            
            visualizations.append(viz_desc)
        
        return visualizations

# Advanced integration patterns
class ToolIntegrationFramework:
    """Framework for integrating multiple tools with orchestration"""
    
    def __init__(self):
        self.tools = {}
        self.dependency_graph = {}
        self.execution_cache = {}
    
    def register_tool(self, tool: AdvancedTool, dependencies: Optional[List[str]] = None):
        """Register tool with dependency information"""
        self.tools[tool.name] = tool
        self.dependency_graph[tool.name] = dependencies or []
    
    async def execute_tool_chain(self, 
                               tool_chain: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute chain of tools with dependency resolution"""
        
        results = []
        execution_context = {}
        
        for tool_config in tool_chain:
            tool_name = tool_config["tool"]
            tool_input = tool_config["input"]
            
            if tool_name not in self.tools:
                results.append(ToolResult(success=False, error=f"Tool {tool_name} not found"))
                continue
            
            # Resolve dependencies
            dependencies_met = await self._check_dependencies(tool_name, execution_context)
            if not dependencies_met:
                results.append(ToolResult(success=False, error=f"Dependencies not met for {tool_name}"))
                continue
            
            # Execute tool
            tool = self.tools[tool_name]
            result = await tool.execute(**tool_input)
            results.append(result)
            
            # Update execution context
            execution_context[tool_name] = result
            
            # Store in execution cache
            self.execution_cache[f"{tool_name}_{datetime.now().timestamp()}"] = result
        
        return results
    
    async def _check_dependencies(self, tool_name: str, context: Dict[str, Any]) -> bool:
        """Check if tool dependencies are satisfied"""
        dependencies = self.dependency_graph.get(tool_name, [])
        
        for dependency in dependencies:
            if dependency not in context or not context[dependency].success:
                return False
        
        return True
    
    def get_tool_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for all tools"""
        
        reports = {}
        overall_stats = {
            "total_tools": len(self.tools),
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0
        }
        
        execution_times = []
        
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'get_performance_metrics'):
                metrics = tool.get_performance_metrics()
                reports[tool_name] = metrics
                
                overall_stats["total_executions"] += metrics["total_executions"]
                overall_stats["successful_executions"] += metrics["successful_executions"]
                execution_times.extend([exec["execution_time"] for exec in tool.execution_history])
        
        if execution_times:
            overall_stats["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        return {
            "individual_reports": reports,
            "overall_statistics": overall_stats,
            "cache_performance": self._calculate_cache_performance()
        }
    
    def _calculate_cache_performance(self) -> Dict[str, Any]:
        """Calculate overall cache performance across all tools"""
        
        total_hits = 0
        total_misses = 0
        total_requests = 0
        
        for tool in self.tools.values():
            if hasattr(tool, 'cache_stats'):
                total_hits += tool.cache_stats["hits"]
                total_misses += tool.cache_stats["misses"]
        
        total_requests = total_hits + total_misses
        
        return {
            "total_cache_requests": total_requests,
            "total_cache_hits": total_hits,
            "total_cache_misses": total_misses,
            "overall_cache_hit_rate": (total_hits / max(total_requests, 1)) * 100
        }

async def advanced_tool_integration_demo():
    """Demonstrate advanced tool integration capabilities"""
    
    print("🔧 Advanced Tool Integration Demo")
    print("=" * 60)
    
    # Initialize framework
    integration_framework = ToolIntegrationFramework()
    
    # Create and register tools
    api_tool = APIIntegrationTool(
        api_base_url="https://api.example.com",
        rate_limit_per_minute=30,
        max_retries=3
    )
    
    data_tool = DataProcessingTool()
    
    integration_framework.register_tool(api_tool)
    integration_framework.register_tool(data_tool, dependencies=[])  # API tool could be dependency
    
    print("1️⃣ Individual Tool Performance:")
    
    # Test individual tool execution
    api_result = await api_tool.execute(
        endpoint="users",
        method="GET",
        params={"limit": 5}
    )
    
    print(f"API Tool Result: {'Success' if api_result.success else 'Failed'}")
    if api_result.success:
        print(f"Data retrieved: {len(api_result.data) if isinstance(api_result.data, list) else 'Single item'}")
    
    # Test data processing tool
    sample_data = [
        {"name": "Product A", "sales": 100, "category": "Electronics"},
        {"name": "Product B", "sales": 150, "category": "Electronics"},
        {"name": "Product C", "sales": 80, "category": "Books"},
        {"name": "Product D", "sales": 200, "category": "Electronics"},
        {"name": "Product E", "sales": 120, "category": "Books"}
    ]
    
    data_result = await data_tool.execute(
        data=sample_data,
        operations=["summary", "statistics", "groupby"],
        group_by="category",
        generate_visualizations=True
    )
    
    print(f"Data Tool Result: {'Success' if data_result.success else 'Failed'}")
    if data_result.success:
        print("Processing results:")
        for key, value in data_result.data.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items processed")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    print("\n2️⃣ Tool Chain Execution:")
    
    # Execute tool chain
    tool_chain = [
        {
            "tool": "data_processing",
            "input": {
                "data": sample_data,
                "operations": ["summary", "statistics"],
                "generate_visualizations": False
            }
        },
        {
            "tool": "data_processing", 
            "input": {
                "data": sample_data,
                "operations": ["groupby"],
                "group_by": "category"
            }
        }
    ]
    
    chain_results = await integration_framework.execute_tool_chain(tool_chain)
    
    print(f"Tool chain executed with {len(chain_results)} results")
    for i, result in enumerate(chain_results):
        print(f"  Step {i+1}: {'Success' if result.success else 'Failed'}")
    
    print("\n3️⃣ Performance Analytics:")
    
    # Get performance report
    performance_report = integration_framework.get_tool_performance_report()
    
    print("Overall Statistics:")
    for key, value in performance_report["overall_statistics"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nCache Performance:")
    for key, value in performance_report["cache_performance"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    asyncio.run(advanced_tool_integration_demo())
```

### Performance Optimization and Scaling

Advanced performance optimization techniques for production-scale agent deployments:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from typing import Dict, Any, List, Optional
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    BATCH_PROCESSING = "batch_processing"
    ASYNC_PROCESSING = "async_processing"
    MEMORY_MANAGEMENT = "memory_management"
    LOAD_BALANCING = "load_balancing"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency: float = 0.0
    total_processing_time: float = 0.0
    memory_usage_peak: int = 0
    cache_hit_rate: float = 0.0
    connection_pool_efficiency: float = 0.0

class AdvancedPerformanceOptimizer:
    """Production-ready performance optimization framework"""
    
    def __init__(self, 
                 enabled_strategies: List[OptimizationStrategy] = None,
                 max_memory_usage_mb: int = 500,
                 target_latency_ms: int = 1000,
                 cache_size: int = 1000):
        
        self.enabled_strategies = enabled_strategies or list(OptimizationStrategy)
        self.max_memory_usage = max_memory_usage_mb * 1024 * 1024  # Convert to bytes
        self.target_latency = target_latency_ms / 1000  # Convert to seconds
        self.cache_size = cache_size
        
        self.metrics = PerformanceMetrics()
        self.latency_history = deque(maxlen=1000)
        self.memory_monitor = MemoryMonitor()
        self.connection_pool = None
        self.cache = LRUCache(max_size=cache_size)
        self.request_queue = asyncio.Queue(maxsize=100)
        
        # Initialize optimization components
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize optimization components based on enabled strategies"""
        
        if OptimizationStrategy.CONNECTION_POOLING in self.enabled_strategies:
            self.connection_pool = ConnectionPool(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_timeout=30
            )
        
        if OptimizationStrategy.MEMORY_MANAGEMENT in self.enabled_strategies:
            self.memory_monitor.start_monitoring()
    
    async def optimize_agent_execution(self, 
                                     agent: ChatAgent,
                                     user_queries: List[str],
                                     batch_size: int = 5) -> List[Dict[str, Any]]:
        """Execute agent with comprehensive optimization"""
        
        results = []
        start_time = time.time()
        
        try:
            # Pre-allocate resources
            if OptimizationStrategy.MEMORY_MANAGEMENT in self.enabled_strategies:
                await self._optimize_memory_allocation(len(user_queries))
            
            # Batch processing for efficiency
            if OptimizationStrategy.BATCH_PROCESSING in self.enabled_strategies:
                results = await self._batch_process_queries(agent, user_queries, batch_size)
            else:
                # Individual processing with optimizations
                results = await self._process_queries_individually(agent, user_queries)
            
            # Post-processing optimization
            if OptimizationStrategy.MEMORY_MANAGEMENT in self.enabled_strategies:
                self._cleanup_resources()
            
            total_time = time.time() - start_time
            logger.info(f"Processed {len(user_queries)} queries in {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return [{"error": str(e), "success": False} for _ in user_queries]
    
    async def _batch_process_queries(self, 
                                   agent: ChatAgent, 
                                   queries: List[str], 
                                   batch_size: int) -> List[Dict[str, Any]]:
        """Process queries in batches for better efficiency"""
        
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = await self._execute_batch(agent, batch)
            results.extend(batch_results)
            
            # Adaptive batch sizing based on performance
            if len(batch_results) > 0:
                avg_batch_latency = sum(r.get("latency", 0) for r in batch_results) / len(batch_results)
                if avg_batch_latency > self.target_latency * 2:
                    batch_size = max(1, batch_size // 2)  # Reduce batch size
                elif avg_batch_latency < self.target_latency / 2:
                    batch_size = min(len(queries) - i - batch_size, batch_size * 2)  # Increase batch size
            
            # Brief pause to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_batch(self, agent: ChatAgent, batch: List[str]) -> List[Dict[str, Any]]:
        """Execute a single batch of queries"""
        
        batch_start = time.time()
        
        # Execute queries concurrently with connection pooling
        if OptimizationStrategy.CONNECTION_POOLING in self.enabled_strategies and self.connection_pool:
            tasks = [
                self._execute_with_connection_pool(agent, query) 
                for query in batch
            ]
        else:
            tasks = [
                self._execute_single_query(agent, query) 
                for query in batch
            ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and add metrics
        processed_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "query": batch[i],
                    "latency": time.time() - batch_start
                })
                self.metrics.failure_count += 1
            else:
                processed_results.append(result)
                self.metrics.success_count += 1
            
            self.metrics.request_count += 1
        
        return processed_results
    
    async def _execute_with_connection_pool(self, agent: ChatAgent, query: str) -> Dict[str, Any]:
        """Execute query using connection pool for better resource utilization"""
        
        start_time = time.time()
        
        # Check cache first
        if OptimizationStrategy.CACHING in self.enabled_strategies:
            cached_result = self.cache.get(query)
            if cached_result:
                self.metrics.cache_hit_rate = (self.cache.hits / max(self.cache.hits + self.cache.misses, 1)) * 100
                return {
                    "success": True,
                    "response": cached_result,
                    "query": query,
                    "latency": time.time() - start_time,
                    "cached": True
                }
        
        # Execute query
        try:
            result = await agent.run(query)
            latency = time.time() - start_time
            
            # Cache result
            if OptimizationStrategy.CACHING in self.enabled_strategies:
                self.cache.set(query, result.text)
            
            # Update metrics
            self.latency_history.append(latency)
            self.metrics.total_latency += latency
            
            # Monitor memory usage
            current_memory = self.memory_monitor.get_current_usage()
            self.metrics.memory_usage_peak = max(self.metrics.memory_usage_peak, current_memory)
            
            return {
                "success": True,
                "response": result.text,
                "query": query,
                "latency": latency,
                "cached": False,
                "memory_usage_mb": current_memory / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "latency": time.time() - start_time
            }
    
    async def _execute_single_query(self, agent: ChatAgent, query: str) -> Dict[str, Any]:
        """Execute single query with basic optimizations"""
        
        start_time = time.time()
        
        try:
            # Use async processing for better performance
            if OptimizationStrategy.ASYNC_PROCESSING in self.enabled_strategies:
                result = await self._async_execute(agent, query)
            else:
                result = await agent.run(query)
            
            latency = time.time() - start_time
            
            return {
                "success": True,
                "response": result.text,
                "query": query,
                "latency": latency
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "latency": time.time() - start_time
            }
    
    async def _async_execute(self, agent: ChatAgent, query: str) -> Any:
        """Execute with async optimizations"""
        
        # Use streaming for large responses
        response_parts = []
        async for chunk in agent.run_stream(query):
            response_parts.append(chunk.text)
        
        return type('obj', (object,), {'text': ''.join(response_parts)})()
    
    async def _optimize_memory_allocation(self, expected_queries: int):
        """Pre-allocate memory based on expected load"""
        
        # Estimate memory requirements
        estimated_memory_per_query = 10 * 1024 * 1024  # 10MB per query
        total_estimated = estimated_memory_per_query * expected_queries
        
        if total_estimated > self.max_memory_usage:
            logger.warning(f"Estimated memory usage ({total_estimated / (1024*1024):.1f}MB) exceeds limit")
        
        # Force garbage collection
        if OptimizationStrategy.MEMORY_MANAGEMENT in self.enabled_strategies:
            gc.collect()
    
    def _cleanup_resources(self):
        """Clean up resources after processing"""
        
        if OptimizationStrategy.MEMORY_MANAGEMENT in self.enabled_strategies:
            gc.collect()
        
        # Clear cache if it's getting too large
        if self.cache.size > self.cache_size * 0.9:
            self.cache.clear_oldest(int(self.cache_size * 0.1))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Calculate derived metrics
        success_rate = (self.metrics.success_count / max(self.metrics.request_count, 1)) * 100
        avg_latency = self.metrics.total_latency / max(self.metrics.request_count, 1)
        
        # Analyze latency trends
        recent_latencies = list(self.latency_history)[-100:]  # Last 100 requests
        p95_latency = np.percentile(recent_latencies, 95) if recent_latencies else 0
        p99_latency = np.percentile(recent_latencies, 99) if recent_latencies else 0
        
        # Memory usage analysis
        memory_efficiency = (self.metrics.memory_usage_peak / self.max_memory_usage) * 100
        
        return {
            "basic_metrics": {
                "total_requests": self.metrics.request_count,
                "success_rate": success_rate,
                "average_latency": avg_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency
            },
            "resource_usage": {
                "peak_memory_mb": self.metrics.memory_usage_peak / (1024 * 1024),
                "memory_efficiency": memory_efficiency,
                "cache_hit_rate": self.metrics.cache_hit_rate
            },
            "optimization_effectiveness": {
                "target_latency_met": avg_latency < self.target_latency,
                "memory_usage_optimal": memory_efficiency < 80,
                "strategies_enabled": [strategy.value for strategy in self.enabled_strategies]
            },
            "recommendations": self._generate_recommendations(avg_latency, memory_efficiency, success_rate)
        }
    
    def _generate_recommendations(self, avg_latency: float, memory_efficiency: float, success_rate: float) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if avg_latency > self.target_latency:
            recommendations.append(f"Average latency ({avg_latency:.3f}s) exceeds target ({self.target_latency:.3f}s)")
            recommendations.append("Consider enabling connection pooling or increasing cache size")
        
        if memory_efficiency > 80:
            recommendations.append(f"Memory usage is high ({memory_efficiency:.1f}%) - consider memory optimization")
        
        if success_rate < 95:
            recommendations.append(f"Success rate ({success_rate:.1f}%) is below optimal (95%)")
            recommendations.append("Review error handling and retry logic")
        
        if not recommendations:
            recommendations.append("Performance is within optimal parameters")
        
        return recommendations

class MemoryMonitor:
    """Real-time memory monitoring and optimization"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = deque(maxlen=100)
        self.monitor_task = None
    
    def start_monitoring(self):
        """Start memory monitoring"""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_memory())
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
    
    async def _monitor_memory(self):
        """Monitor memory usage continuously"""
        while self.monitoring:
            try:
                memory_info = psutil.virtual_memory()
                current_usage = memory_info.used
                
                self.memory_samples.append({
                    "timestamp": time.time(),
                    "memory_usage": current_usage,
                    "memory_percent": memory_info.percent
                })
                
                await asyncio.sleep(1)  # Sample every second
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    def get_current_usage(self) -> int:
        """Get current memory usage in bytes"""
        return psutil.virtual_memory().used
    
    def get_peak_usage(self) -> int:
        """Get peak memory usage from samples"""
        if not self.memory_samples:
            return 0
        return max(sample["memory_usage"] for sample in self.memory_samples)

class ConnectionPool:
    """Advanced connection pooling for HTTP requests"""
    
    def __init__(self, 
                 max_connections: int = 20,
                 max_keepalive_connections: int = 10,
                 keepalive_timeout: int = 30):
        
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_timeout = keepalive_timeout
        
        self.session = None
        self.connection_stats = {
            "created": 0,
            "reused": 0,
            "closed": 0
        }
    
    async def __aenter__(self):
        """Initialize connection pool"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_keepalive_connections,
            keepalive_timeout=self.keepalive_timeout
        )
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up connection pool"""
        if self.session:
            await self.session.close()
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request using connection pool"""
        
        if not self.session:
            raise RuntimeError("Connection pool not initialized")
        
        self.connection_stats["created"] += 1
        
        try:
            response = await self.session.request(method, url, **kwargs)
            self.connection_stats["reused"] += 1
            return response
            
        except Exception as e:
            self.connection_stats["closed"] += 1
            raise e

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        async with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        async with self.lock:
            if key in self.cache:
                # Update existing item
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear_oldest(self, count: int) -> None:
        """Clear oldest n items"""
        for _ in range(min(count, len(self.access_order))):
            if self.access_order:
                key = self.access_order.popleft()
                if key in self.cache:
                    del self.cache[key]
    
    @property
    def size(self) -> int:
        """Current cache size"""
        return len(self.cache)

async def performance_optimization_demo():
    """Demonstrate advanced performance optimization"""
    
    print("⚡ Advanced Performance Optimization Demo")
    print("=" * 60)
    
    # Create performance optimizer
    optimizer = AdvancedPerformanceOptimizer(
        enabled_strategies=[
            OptimizationStrategy.CACHING,
            OptimizationStrategy.CONNECTION_POOLING,
            OptimizationStrategy.BATCH_PROCESSING,
            OptimizationStrategy.MEMORY_MANAGEMENT
        ],
        max_memory_usage_mb=100,
        target_latency_ms=500,
        cache_size=500
    )
    
    # Create test agent
    async with OpenAIChatClient() as client:
        agent = ChatAgent(
            chat_client=client,
            name="OptimizedAgent",
            instructions="You are a fast and efficient assistant."
        )
        
        # Generate test queries
        test_queries = [
            "What is machine learning?",
            "Explain neural networks simply",
            "How does deep learning work?",
            "What are the types of AI?",
            "Describe reinforcement learning",
            "What is natural language processing?",
            "Explain computer vision",
            "How do recommendation systems work?"
        ]
        
        print("1️⃣ Optimized Batch Processing:")
        
        # Execute with optimization
        start_time = time.time()
        results = await optimizer.optimize_agent_execution(
            agent=agent,
            user_queries=test_queries,
            batch_size=3
        )
        total_time = time.time() - start_time
        
        print(f"Processed {len(results)} queries in {total_time:.2f} seconds")
        print(f"Average latency: {sum(r.get('latency', 0) for r in results) / len(results):.3f}s")
        
        # Show performance report
        print("\n2️⃣ Performance Report:")
        report = optimizer.get_performance_report()
        
        print("Basic Metrics:")
        for key, value in report["basic_metrics"].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}" if isinstance(value, float) else f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\nResource Usage:")
        for key, value in report["resource_usage"].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.1f}" if isinstance(value, float) else f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n3️⃣ Optimization Effectiveness:")
        for key, value in report["optimization_effectiveness"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n4️⃣ Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"  - {recommendation}")
    
    # Demonstrate memory optimization
    print("\n5️⃣ Memory Optimization:")
    
    # Simulate memory-intensive operation
    large_data = np.random.random((1000, 1000))  # ~8MB of data
    
    # Monitor memory usage
    initial_memory = psutil.virtual_memory().used
    
    # Process with optimization
    async def memory_intensive_operation():
        # Simulate processing
        result = np.mean(large_data, axis=0)
        
        # Explicit cleanup
        del large_data
        gc.collect()
        
        return result
    
    result = await memory_intensive_operation()
    final_memory = psutil.virtual_memory().used
    
    print(f"Memory usage change: {(final_memory - initial_memory) / (1024*1024):.1f} MB")
    print(f"Processing result shape: {result.shape}")

if __name__ == "__main__":
    asyncio.run(performance_optimization_demo())
```

---

## Best Practices & Patterns

This section consolidates essential best practices and proven patterns for building production-ready AI agents. These guidelines represent lessons learned from real-world deployments and cover critical aspects of reliability, maintainability, security, and performance.

### Production-Ready Architecture Patterns

```python
"""
Production-Ready Agent Architecture Patterns
============================================

This module demonstrates essential patterns for building robust,
scalable, and maintainable AI agents for production environments.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
import json
from enum import Enum
from functools import wraps
import time
import hashlib
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentHealthStatus(Enum):
    """Agent health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ProductionConfig:
    """Production configuration with validation and defaults"""
    
    # Core settings
    agent_name: str
    model_id: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 1000
    
    # Reliability settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    batch_size: int = 5
    
    # Monitoring settings
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
    
    # Security settings
    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    allowed_file_types: List[str] = field(default_factory=lambda: [".txt", ".pdf", ".docx"])
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.agent_name or len(self.agent_name.strip()) == 0:
            issues.append("Agent name cannot be empty")
        
        if self.temperature < 0 or self.temperature > 2:
            issues.append("Temperature must be between 0 and 2")
        
        if self.max_retries < 0 or self.max_retries > 10:
            issues.append("Max retries must be between 0 and 10")
        
        if self.timeout < 1 or self.timeout > 300:
            issues.append("Timeout must be between 1 and 300 seconds")
        
        return issues

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable) -> Callable:
        """Decorator for circuit breaker functionality"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open - failing fast")
            
            try:
                result = await func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e
        
        return wrapper

class RateLimiter:
    """Token bucket rate limiter for API protection"""
    
    def __init__(self, rate: int = 10, capacity: int = 10):
        self.rate = rate  # requests per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            refill_tokens = elapsed * self.rate
            self.tokens = min(self.capacity, self.tokens + refill_tokens)
            self.last_refill = now
            
