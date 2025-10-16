# Comprehensive Programming Guide: Building AI Agents with the Microsoft Agent Framework

## Table of Contents
1. [Introduction to the Microsoft Agent Framework](#introduction)
2. [Setting Up Your Development Environment](#setup)
3. [Agent Type 1: Basic Conversational Agent](#agent1)
4. [Agent Type 2: Function-Calling Agent](#agent2)
5. [Agent Type 3: Code Execution Agent](#agent3)
6. [Agent Type 4: Knowledge Retrieval Agent](#agent4)
7. [Agent Type 5: Multimodal Agent](#agent5)
8. [Advanced Topics and Best Practices](#advanced)
9. [Conclusion and Next Steps](#conclusion)

<a name="introduction"></a>
## 1. Introduction to the Microsoft Agent Framework

The Microsoft Agent Framework is a powerful and flexible toolkit designed to simplify the creation of AI-powered agents. It provides a unified interface for interacting with various AI models and services, with a focus on making complex AI capabilities accessible to developers.

### Key Components of the Framework

The framework is built around three main client types, each serving different use cases:

1. **OpenAIChatClient**: Provides direct chat-based interactions with OpenAI models like GPT-4. This is ideal for creating conversational agents that need to respond to user input in real-time.

2. **OpenAIAssistantsClient**: Integrates with OpenAI's Assistants API, offering more advanced capabilities like thread management, function calling, and tool integration. This is suitable for building more sophisticated agents with persistent memory and external tool access.

3. **OpenAIResponsesClient**: Focuses on structured response generation, allowing developers to get responses in predefined formats. This is useful when you need consistent, structured data from the AI model.

### Core Features

The framework supports a rich set of features that enable the creation of diverse AI agents:

- **Function Tools**: Allow agents to call external functions and APIs, extending their capabilities beyond text generation.
- **Code Interpreter**: Enables agents to write and execute Python code, useful for mathematical calculations, data analysis, and more.
- **File Search**: Allows agents to search through uploaded documents, making them capable of answering questions based on specific knowledge bases.
- **Thread Management**: Maintains conversation context across multiple interactions, enabling more coherent and context-aware conversations.
- **Web Search**: Integrates with web search capabilities, allowing agents to access up-to-date information from the internet.
- **Model Context Protocol (MCP)**: Facilitates integration with external services and tools through a standardized protocol.
- **Structured Outputs**: Ensures responses from the AI model follow predefined schemas, making it easier to parse and use the data.
- **Vision Capabilities**: Enables agents to analyze and interpret images, expanding their understanding beyond text.

### Architecture Overview

The Microsoft Agent Framework follows a modular architecture that separates concerns and allows for flexible composition:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                    Agent Framework                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ ChatAgent   │  │ Assistants  │  │ Responses           │  │
│  │             │  │ Agent       │  │ Agent               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Client Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ OpenAIChat  │  │ OpenAI      │  │ OpenAI              │  │
│  │ Client      │  │ Assistants  │  │ Responses           │  │
│  │             │  │ Client      │  │ Client              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Tool Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Function    │  │ Code        │  │ File                │  │
│  │ Tools       │  │ Interpreter │  │ Search              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

This architecture allows developers to choose the appropriate client and tools for their specific use case, while maintaining a consistent interface across different implementations.

<a name="setup"></a>
## 2. Setting Up Your Development Environment

Before we dive into building our first agent, let's ensure your development environment is properly configured.

### Prerequisites

1. **Python 3.8 or higher**: The framework is built on Python, so ensure you have a compatible version installed.

2. **Microsoft Agent Framework**: Install the framework using pip:
   ```bash
   pip install agent-framework
   ```

3. **OpenAI API Key**: Most examples in this guide use OpenAI's models, so you'll need an API key. You can obtain one from the [OpenAI platform](https://platform.openai.com/).

### Environment Variables

The framework relies on several environment variables for configuration. Create a `.env` file in your project root and add the following:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL_ID=gpt-4o  # or gpt-4o-mini, gpt-3.5-turbo
OPENAI_RESPONSES_MODEL_ID=gpt-4o  # or gpt-4o-mini, gpt-3.5-turbo

# Optional
OPENAI_ORG_ID=your_openai_organization_id  # if applicable
OPENAI_API_BASE_URL=https://api.openai.com/v1  # if using a different base URL
```

### Optional Dependencies

Some agent types require additional dependencies:

- **For image processing**: Install Pillow for image display:
  ```bash
  pip install pillow
  ```

- **For web search**: No additional dependencies required, but ensure your API key has access to web search capabilities.

### Basic Project Structure

Here's a recommended project structure for your agent development:

```
my_agent_project/
├── .env
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── basic_conversational_agent.py
│   ├── function_calling_agent.py
│   ├── code_execution_agent.py
│   ├── knowledge_retrieval_agent.py
│   └── multimodal_agent.py
├── tools/
│   ├── __init__.py
│   ├── weather_tools.py
│   ├── calculation_tools.py
│   └── custom_tools.py
├── data/
│   └── sample_documents/
└── tests/
    ├── __init__.py
    └── test_agents.py
```

### Verifying Your Setup

Let's create a simple script to verify that your environment is correctly configured:

```python
# verify_setup.py
import os
import asyncio
from agent_framework.openai import OpenAIChatClient

async def verify_setup():
    try:
        # Check if environment variables are set
        api_key = os.getenv("OPENAI_API_KEY")
        model_id = os.getenv("OPENAI_CHAT_MODEL_ID")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not model_id:
            raise ValueError("OPENAI_CHAT_MODEL_ID environment variable is not set")
            
        print("Environment variables are properly configured.")
        
        # Test a simple connection to OpenAI
        client = OpenAIChatClient()
        agent = client.create_agent(
            name="TestAgent",
            instructions="You are a helpful assistant.",
        )
        
        response = await agent.run("Say 'Hello, world!'")
        print(f"OpenAI connection successful. Response: {response}")
        
        print("Setup verification complete. You're ready to start building agents!")
        
    except Exception as e:
        print(f"Setup verification failed: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    asyncio.run(verify_setup())
```

Run this script to ensure everything is working correctly:

```bash
python verify_setup.py
```

If you see the "Setup verification complete" message, you're ready to start building agents with the Microsoft Agent Framework!

<a name="agent1"></a>
## 3. Agent Type 1: Basic Conversational Agent

### Purpose and Use Cases

A Basic Conversational Agent is the simplest type of AI agent you can build with the Microsoft Agent Framework. Its primary purpose is to engage in text-based conversations with users, providing responses based on its training and any instructions you provide.

Common use cases include:
- Customer service chatbots
- Information retrieval assistants
- Personal productivity helpers
- Educational tutors
- Entertainment and companionship bots

### Implementation Using OpenAIChatClient

For our Basic Conversational Agent, we'll use the OpenAIChatClient, which provides direct access to OpenAI's chat models. This client is ideal for simple conversational agents that don't need advanced features like persistent memory or tool integration.

Let's create our first agent:

```python
# agents/basic_conversational_agent.py
import asyncio
import os
from agent_framework.openai import OpenAIChatClient

class BasicConversationalAgent:
    def __init__(self, name="Assistant", instructions="You are a helpful assistant."):
        """
        Initialize the Basic Conversational Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = self.client.create_agent(
            name=name,
            instructions=instructions,
        )
    
    async def chat(self, message, stream=False):
        """
        Send a message to the agent and get a response.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(message):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(message)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
                
            await self.chat(user_input)

async def main():
    # Create a basic conversational agent
    agent = BasicConversationalAgent(
        name="ChatBot",
        instructions="You are a friendly and helpful assistant. Be concise in your responses."
    )
    
    # Example 1: Single interaction
    print("=== Example 1: Single Interaction ===")
    await agent.chat("Hello! Can you tell me a fun fact about space?")
    
    # Example 2: Streaming response
    print("\n=== Example 2: Streaming Response ===")
    await agent.chat("Explain quantum computing in simple terms.", stream=True)
    
    # Example 3: Interactive conversation
    print("\n=== Example 3: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming vs. Non-Streaming Responses

The framework supports both streaming and non-streaming responses. Streaming responses are useful when you want to display the response as it's being generated, providing a more interactive experience. Non-streaming responses wait until the entire response is generated before returning it.

In our example, the `chat` method accepts a `stream` parameter that determines which approach to use. For streaming responses, we use the `run_stream` method and iterate over the response chunks. For non-streaming responses, we use the `run` method, which returns the complete response at once.

### Thread Management for Conversation Context

By default, each interaction with the agent is independent, meaning the agent doesn't remember previous conversations. To maintain context across multiple interactions, we can use thread management:

```python
# agents/basic_conversational_agent_with_thread.py
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent, AgentThread

class ConversationalAgentWithMemory:
    def __init__(self, name="Assistant", instructions="You are a helpful assistant."):
        """
        Initialize the Conversational Agent with memory.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
        )
        self.thread = self.agent.get_new_thread()
    
    async def chat(self, message, stream=False):
        """
        Send a message to the agent and get a response, maintaining conversation context.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(message, thread=self.thread):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(message, thread=self.thread)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
                
            await self.chat(user_input)

async def main():
    # Create a conversational agent with memory
    agent = ConversationalAgentWithMemory(
        name="MemoryBot",
        instructions="You are a friendly assistant with a good memory. Remember details from our conversation."
    )
    
    # Example: Context-aware conversation
    print("=== Context-Aware Conversation ===")
    await agent.chat("My name is Alex and I love hiking.")
    await agent.chat("What's my name?")
    await agent.chat("What hobbies do you think I would enjoy?")

if __name__ == "__main__":
    asyncio.run(main())
```

In this enhanced version, we use the `ChatAgent` class and create a thread using `get_new_thread()`. By passing this thread to each interaction, the agent maintains context across the conversation, allowing it to remember previous messages and provide more coherent responses.

### Testing and Validation

To ensure our Basic Conversational Agent is working correctly, let's create a simple test:

```python
# tests/test_basic_conversational_agent.py
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.basic_conversational_agent import BasicConversationalAgent

async def test_basic_conversational_agent():
    """Test the Basic Conversational Agent."""
    agent = BasicConversationalAgent(
        name="TestBot",
        instructions="You are a test assistant. Always respond with 'Test successful!'"
    )
    
    # Test non-streaming response
    response = await agent.chat("Hello")
    assert "Test successful!" in response, "Non-streaming response test failed"
    
    # Test streaming response
    response = await agent.chat("Hello again", stream=True)
    assert "Test successful!" in response, "Streaming response test failed"
    
    print("All tests passed for Basic Conversational Agent!")

if __name__ == "__main__":
    asyncio.run(test_basic_conversational_agent())
```

### Common Issues and Troubleshooting

1. **API Key Issues**: Ensure your OpenAI API key is correctly set in the environment variables. If you're getting authentication errors, double-check that the key is valid and has the necessary permissions.

2. **Model Availability**: Some models might not be available in your region or with your API plan. If you encounter model-related errors, try switching to a different model like `gpt-3.5-turbo`.

3. **Rate Limiting**: OpenAI APIs have rate limits. If you're making too many requests in a short period, you might encounter rate limit errors. Consider implementing exponential backoff for production applications.

4. **Context Length**: Each model has a maximum context length. If your conversation becomes too long, you might encounter context length errors. For long conversations, consider implementing context summarization or limiting the conversation history.

### Variations and Extensions

1. **Persona-based Agents**: You can create agents with specific personas by customizing the instructions:

```python
# Create a pirate-themed agent
pirate_agent = BasicConversationalAgent(
    name="Captain AI",
    instructions="You are a friendly pirate. Speak in pirate slang and always talk about treasure and adventures."
)
```

2. **Multi-language Support**: Create agents that can converse in different languages:

```python
# Create a Spanish-speaking agent
spanish_agent = BasicConversationalAgent(
    name="Asistente Español",
    instructions="Eres un asistente útil que responde en español. Sé amable y conciso."
)
```

3. **Response Filtering**: Add filtering to ensure the agent's responses adhere to certain guidelines:

```python
class FilteredConversationalAgent(BasicConversationalAgent):
    def __init__(self, *args, forbidden_words=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden_words = forbidden_words or []
    
    async def chat(self, message, stream=False):
        response = await super().chat(message, stream)
        
        # Check for forbidden words
        for word in self.forbidden_words:
            if word.lower() in response.lower():
                return "I'm sorry, I can't provide that response."
        
        return response
```

With this foundation, you now have a Basic Conversational Agent that can engage in text-based conversations. In the next section, we'll explore how to extend this agent with function-calling capabilities.

<a name="agent2"></a>
## 4. Agent Type 2: Function-Calling Agent

### Purpose and Use Cases

A Function-Calling Agent extends the basic conversational capabilities by allowing the AI to call external functions and APIs. This enables the agent to perform actions, retrieve real-time data, and interact with external systems, greatly expanding its utility.

Common use cases include:
- Weather information retrieval
- Database querying
- API integration for services like calendars, email, or CRM
- E-commerce operations like checking inventory or placing orders
- IoT device control
- Data analysis and visualization

### Defining and Registering Functions

The Microsoft Agent Framework makes it easy to define and register functions that the agent can call. Functions are defined as regular Python functions with type hints and descriptions, which the framework uses to generate the necessary schema for the AI model.

Let's create a Function-Calling Agent that can retrieve weather information and perform calculations:

```python
# agents/function_calling_agent.py
import asyncio
import json
from datetime import datetime, timezone
from random import randint
from typing import Annotated, List, Dict, Any
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

# Define the functions the agent can call
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
    unit: Annotated[str, Field(description="The temperature unit, either 'celsius' or 'fahrenheit'.")] = "celsius",
) -> str:
    """
    Get the weather for a given location.
    
    Args:
        location: The location to get the weather for.
        unit: The temperature unit, either 'celsius' or 'fahrenheit'.
        
    Returns:
        A string describing the weather conditions.
    """
    # In a real implementation, you would call a weather API here
    conditions = ["sunny", "cloudy", "rainy", "stormy", "snowy"]
    temp = randint(10, 30) if unit == "celsius" else randint(50, 86)
    condition = conditions[randint(0, len(conditions) - 1)]
    
    return f"The weather in {location} is {condition} with a high of {temp}°{unit[0].upper()}."

def get_time(
    timezone_str: Annotated[str, Field(description="The timezone to get the current time for, e.g., 'UTC', 'America/New_York'.")] = "UTC",
) -> str:
    """
    Get the current time in a specific timezone.
    
    Args:
        timezone_str: The timezone to get the current time for.
        
    Returns:
        A string describing the current time.
    """
    try:
        import pytz
        tz = pytz.timezone(timezone_str)
        current_time = datetime.now(tz)
        return f"The current time in {timezone_str} is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."
    except ImportError:
        # Fallback if pytz is not installed
        current_time = datetime.now(timezone.utc)
        return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."
    except Exception as e:
        return f"Error getting time for timezone {timezone_str}: {str(e)}"

def calculate(
    expression: Annotated[str, Field(description="A mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'.")],
) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate.
        
    Returns:
        The result of the mathematical expression.
    """
    try:
        # In a real implementation, you might want to use a safer evaluation method
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error evaluating expression {expression}: {str(e)}"

def search_database(
    query: Annotated[str, Field(description="The search query for the database.")],
    table: Annotated[str, Field(description="The table to search in.")] = "products",
    limit: Annotated[int, Field(description="The maximum number of results to return.")] = 5,
) -> str:
    """
    Search a database for information.
    
    Args:
        query: The search query for the database.
        table: The table to search in.
        limit: The maximum number of results to return.
        
    Returns:
        A JSON string containing the search results.
    """
    # In a real implementation, you would query an actual database
    # For this example, we'll return mock data
    mock_data = {
        "products": [
            {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
            {"id": 2, "name": "Smartphone", "price": 699.99, "category": "Electronics"},
            {"id": 3, "name": "Headphones", "price": 149.99, "category": "Electronics"},
            {"id": 4, "name": "Coffee Maker", "price": 79.99, "category": "Appliances"},
            {"id": 5, "name": "Desk Chair", "price": 199.99, "category": "Furniture"},
        ],
        "customers": [
            {"id": 1, "name": "John Doe", "email": "john@example.com", "join_date": "2023-01-15"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "join_date": "2023-02-20"},
            {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "join_date": "2023-03-10"},
        ]
    }
    
    if table not in mock_data:
        return json.dumps({"error": f"Table '{table}' not found."})
    
    # Simple mock search - in reality, you'd use SQL or another query language
    results = []
    query_lower = query.lower()
    
    for item in mock_data[table]:
        # Check if the query matches any field value
        for key, value in item.items():
            if query_lower in str(value).lower():
                results.append(item)
                break
        
        if len(results) >= limit:
            break
    
    return json.dumps({"results": results, "count": len(results)})

class FunctionCallingAgent:
    def __init__(self, name="FunctionBot", instructions="You are a helpful assistant with access to various tools."):
        """
        Initialize the Function-Calling Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = self.client.create_agent(
            name=name,
            instructions=instructions,
            tools=[get_weather, get_time, calculate, search_database],
        )
    
    async def chat(self, message, stream=False):
        """
        Send a message to the agent and get a response.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(message):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(message)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
                
            await self.chat(user_input)

async def main():
    # Create a function-calling agent
    agent = FunctionCallingAgent(
        name="ToolBot",
        instructions="You are a helpful assistant with access to weather information, time data, calculation capabilities, and a product database. Use the appropriate tools to answer questions."
    )
    
    # Example 1: Weather information
    print("=== Example 1: Weather Information ===")
    await agent.chat("What's the weather like in New York?")
    
    # Example 2: Time information
    print("\n=== Example 2: Time Information ===")
    await agent.chat("What time is it in Tokyo?")
    
    # Example 3: Calculation
    print("\n=== Example 3: Calculation ===")
    await agent.chat("What is 15% of 250?")
    
    # Example 4: Database search
    print("\n=== Example 4: Database Search ===")
    await agent.chat("Find electronics products under $1000.")
    
    # Example 5: Complex query requiring multiple tools
    print("\n=== Example 5: Complex Query ===")
    await agent.chat("I'm planning a trip to London. What's the weather like there and what time is it now?")

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent-Level vs. Query-Level Tools

The framework supports two approaches to providing tools to the agent:

1. **Agent-Level Tools**: Tools are provided when creating the agent and are available for all queries during the agent's lifetime.

2. **Query-Level Tools**: Tools are provided with specific queries, allowing you to customize the available tools for each interaction.

Let's modify our example to demonstrate both approaches:

```python
# agents/function_calling_agent_with_tool_levels.py
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent
from pydantic import Field
from typing import Annotated
from random import randint

# Define the functions
def get_weather(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

def get_time() -> str:
    """Get the current UTC time."""
    from datetime import datetime, timezone
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."

def calculate(expression: Annotated[str, Field(description="A mathematical expression to evaluate.")]) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

class FunctionCallingAgentWithToolLevels:
    def __init__(self, name="FunctionBot", instructions="You are a helpful assistant."):
        """
        Initialize the Function-Calling Agent with different tool levels.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        
        # Create an agent with agent-level tools
        self.agent_with_agent_level_tools = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=[get_weather],  # Agent-level tool
        )
        
        # Create an agent without agent-level tools
        self.agent_without_agent_level_tools = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            # No agent-level tools
        )
    
    async def chat_with_agent_level_tools(self, message):
        """
        Chat using an agent with agent-level tools.
        
        Args:
            message (str): The message to send to the agent.
            
        Returns:
            str: The agent's response.
        """
        response = await self.agent_with_agent_level_tools.run(message)
        print(f"{self.name} (Agent-Level Tools): {response}")
        return response
    
    async def chat_with_query_level_tools(self, message, tools=None):
        """
        Chat using an agent with query-level tools.
        
        Args:
            message (str): The message to send to the agent.
            tools (list): The tools to provide for this specific query.
            
        Returns:
            str: The agent's response.
        """
        response = await self.agent_without_agent_level_tools.run(message, tools=tools)
        print(f"{self.name} (Query-Level Tools): {response}")
        return response
    
    async def chat_with_mixed_tools(self, message, additional_tools=None):
        """
        Chat using an agent with both agent-level and query-level tools.
        
        Args:
            message (str): The message to send to the agent.
            additional_tools (list): Additional tools to provide for this specific query.
            
        Returns:
            str: The agent's response.
        """
        response = await self.agent_with_agent_level_tools.run(message, tools=additional_tools)
        print(f"{self.name} (Mixed Tools): {response}")
        return response

async def main():
    # Create a function-calling agent with different tool levels
    agent = FunctionCallingAgentWithToolLevels(
        name="ToolBot",
        instructions="You are a helpful assistant with access to various tools."
    )
    
    # Example 1: Using agent-level tools
    print("=== Example 1: Using Agent-Level Tools ===")
    await agent.chat_with_agent_level_tools("What's the weather like in Paris?")
    
    # Example 2: Using query-level tools
    print("\n=== Example 2: Using Query-Level Tools ===")
    await agent.chat_with_query_level_tools("What time is it now?", tools=[get_time])
    
    # Example 3: Using different query-level tools
    print("\n=== Example 3: Using Different Query-Level Tools ===")
    await agent.chat_with_query_level_tools("What is 25 * 4?", tools=[calculate])
    
    # Example 4: Using mixed tools
    print("\n=== Example 4: Using Mixed Tools ===")
    await agent.chat_with_mixed_tools(
        "What's the weather in Tokyo and what time is it now?",
        additional_tools=[get_time]
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling for Function Calls

When working with function calls, it's important to handle errors gracefully. The framework provides mechanisms to catch and handle errors that occur during function execution:

```python
# agents/function_calling_agent_with_error_handling.py
import asyncio
import logging
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent
from pydantic import Field
from typing import Annotated
from random import randint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the functions with error handling
def get_weather(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
    """Get the weather for a given location."""
    try:
        # Simulate a potential error
        if location.lower() == "error":
            raise ValueError("Simulated error in weather service")
            
        conditions = ["sunny", "cloudy", "rainy", "stormy"]
        return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."
    except Exception as e:
        logger.error(f"Error getting weather for {location}: {str(e)}")
        return f"Sorry, I couldn't retrieve the weather information for {location} due to an error: {str(e)}"

def get_time() -> str:
    """Get the current UTC time."""
    try:
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc)
        return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."
    except Exception as e:
        logger.error(f"Error getting current time: {str(e)}")
        return f"Sorry, I couldn't retrieve the current time due to an error: {str(e)}"

def divide_numbers(
    a: Annotated[float, Field(description="The first number.")],
    b: Annotated[float, Field(description="The second number to divide by.")]
) -> str:
    """Divide two numbers."""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        return f"The result of {a} divided by {b} is {result}."
    except Exception as e:
        logger.error(f"Error dividing {a} by {b}: {str(e)}")
        return f"Sorry, I couldn't perform the division due to an error: {str(e)}"

class FunctionCallingAgentWithErrorHandling:
    def __init__(self, name="ErrorBot", instructions="You are a helpful assistant with error handling capabilities."):
        """
        Initialize the Function-Calling Agent with error handling.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=[get_weather, get_time, divide_numbers],
        )
    
    async def chat(self, message):
        """
        Send a message to the agent and get a response with error handling.
        
        Args:
            message (str): The message to send to the agent.
            
        Returns:
            str: The agent's response.
        """
        try:
            response = await self.agent.run(message)
            print(f"{self.name}: {response}")
            return response
        except Exception as e:
            error_message = f"An error occurred while processing your request: {str(e)}"
            logger.error(error_message)
            print(f"{self.name}: {error_message}")
            return error_message

async def main():
    # Create a function-calling agent with error handling
    agent = FunctionCallingAgentWithErrorHandling(
        name="SafeBot",
        instructions="You are a helpful assistant with access to weather information, time data, and calculation capabilities. Handle errors gracefully and inform the user when something goes wrong."
    )
    
    # Example 1: Normal function call
    print("=== Example 1: Normal Function Call ===")
    await agent.chat("What's the weather like in London?")
    
    # Example 2: Function call with error
    print("\n=== Example 2: Function Call with Error ===")
    await agent.chat("What's the weather like in Error?")
    
    # Example 3: Division by zero error
    print("\n=== Example 3: Division by Zero Error ===")
    await agent.chat("What is 10 divided by 0?")
    
    # Example 4: Successful function call after error
    print("\n=== Example 4: Successful Function Call After Error ===")
    await agent.chat("What time is it now?")

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing and Validation

Let's create a test for our Function-Calling Agent:

```python
# tests/test_function_calling_agent.py
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.function_calling_agent import FunctionCallingAgent

async def test_function_calling_agent():
    """Test the Function-Calling Agent."""
    agent = FunctionCallingAgent(
        name="TestBot",
        instructions="You are a test assistant. Always use the appropriate tools to answer questions."
    )
    
    # Test weather function
    response = await agent.chat("What's the weather like in Test City?")
    assert "weather in Test City" in response, "Weather function test failed"
    
    # Test time function
    response = await agent.chat("What time is it now?")
    assert "current time" in response.lower(), "Time function test failed"
    
    # Test calculation function
    response = await agent.chat("What is 2 + 2?")
    assert "4" in response, "Calculation function test failed"
    
    # Test database search function
    response = await agent.chat("Find products in the electronics category.")
    assert "results" in response.lower() or "products" in response.lower(), "Database search function test failed"
    
    print("All tests passed for Function-Calling Agent!")

if __name__ == "__main__":
    asyncio.run(test_function_calling_agent())
```

### Common Issues and Troubleshooting

1. **Function Schema Issues**: Ensure your function definitions have proper type hints and descriptions. The framework uses these to generate the schema for the AI model.

2. **Function Execution Errors**: If a function encounters an error during execution, the agent might not be able to recover. Implement proper error handling in your functions.

3. **Tool Selection**: Sometimes the AI model might not select the appropriate tool for a given query. Provide clear instructions and well-defined function descriptions to improve tool selection.

4. **Parameter Validation**: The AI model might pass invalid parameters to your functions. Validate parameters within your functions and handle errors gracefully.

### Variations and Extensions

1. **API Integration**: Replace the mock functions with real API calls:

```python
import requests

def get_real_weather(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
    """Get the weather for a given location using a real weather API."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Weather API key not configured."
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"The weather in {location} is {weather} with a temperature of {temp}°C."
        else:
            return f"Error retrieving weather data: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"
```

2. **Async Function Calls**: For functions that perform I/O operations, define them as async:

```python
async def get_weather_async(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
    """Get the weather for a given location asynchronously."""
    # Simulate an async operation
    await asyncio.sleep(1)
    
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."
```

3. **Dynamic Tool Loading**: Load tools dynamically based on configuration:

```python
def load_tools_from_config(config_file):
    """Load tools from a configuration file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    tools = []
    for tool_config in config.get("tools", []):
        if tool_config["name"] == "weather":
            tools.append(get_weather)
        elif tool_config["name"] == "time":
            tools.append(get_time)
        # Add more tools as needed
    
    return tools

# Usage
tools = load_tools_from_config("tools_config.json")
agent = FunctionCallingAgent(tools=tools)
```

With this Function-Calling Agent, you can now create AI assistants that can interact with external systems, retrieve real-time data, and perform actions beyond text generation. In the next section, we'll explore how to build an agent that can write and execute code.

<a name="agent3"></a>
## 5. Agent Type 3: Code Execution Agent

### Purpose and Use Cases

A Code Execution Agent is an AI agent that can write and execute code, typically Python, to solve problems, perform calculations, analyze data, or create visualizations. This capability is particularly useful for tasks that require computational power or data manipulation beyond what the AI model can do directly.

Common use cases include:
- Mathematical problem solving
- Data analysis and visualization
- Algorithm implementation and testing
- Scientific computing
- File processing and manipulation
- Web scraping and data extraction
- Automation of repetitive tasks

### Using the HostedCodeInterpreterTool

The Microsoft Agent Framework provides the `HostedCodeInterpreterTool` to enable code execution capabilities. This tool allows the agent to write and execute Python code in a secure sandboxed environment.

Let's create a Code Execution Agent:

```python
# agents/code_execution_agent.py
import asyncio
import json
import re
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIAssistantsClient
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

def extract_code_from_response(response):
    """
    Extract Python code from a response.
    
    Args:
        response (str): The response containing code.
        
    Returns:
        str: The extracted Python code.
    """
    # Look for code blocks marked with ```python
    python_code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
    
    if python_code_blocks:
        return python_code_blocks[0]
    
    # Look for code blocks marked with just ```
    code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
    
    if code_blocks:
        return code_blocks[0]
    
    # If no code blocks found, return the entire response
    return response

def get_code_interpreter_chunk(chunk):
    """
    Helper method to access code interpreter data from response chunks.
    
    Args:
        chunk: The response chunk from the agent.
        
    Returns:
        str or None: The code interpreter input if available, None otherwise.
    """
    if (
        hasattr(chunk, 'raw_representation') and
        hasattr(chunk.raw_representation, 'raw_representation') and
        isinstance(chunk.raw_representation.raw_representation, RunStepDeltaEvent) and
        hasattr(chunk.raw_representation.raw_representation, 'delta') and
        isinstance(chunk.raw_representation.raw_representation.delta, RunStepDelta) and
        hasattr(chunk.raw_representation.raw_representation.delta, 'step_details') and
        isinstance(chunk.raw_representation.raw_representation.delta.step_details, ToolCallDeltaObject) and
        hasattr(chunk.raw_representation.raw_representation.delta.step_details, 'tool_calls') and
        chunk.raw_representation.raw_representation.delta.step_details.tool_calls
    ):
        for tool_call in chunk.raw_representation.raw_representation.delta.step_details.tool_calls:
            if (
                isinstance(tool_call, CodeInterpreterToolCallDelta) and
                hasattr(tool_call, 'code_interpreter') and
                isinstance(tool_call.code_interpreter, CodeInterpreter) and
                tool_call.code_interpreter.input is not None
            ):
                return tool_call.code_interpreter.input
    return None

class CodeExecutionAgent:
    def __init__(self, name="CodeBot", instructions="You are a helpful assistant that can write and execute Python code to solve problems."):
        """
        Initialize the Code Execution Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedCodeInterpreterTool(),
        )
    
    async def execute_code(self, prompt, stream=True):
        """
        Send a prompt to the agent and execute the generated code.
        
        Args:
            prompt (str): The prompt describing what code to write and execute.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        if stream:
            response = ""
            generated_code = ""
            print(f"{self.name}: ", end="", flush=True)
            
            async for chunk in self.agent.run_stream(prompt):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
                
                # Extract code from the chunk
                code_chunk = get_code_interpreter_chunk(chunk)
                if code_chunk is not None:
                    generated_code += code_chunk
            
            print()  # New line after the complete response
            return response, generated_code
        else:
            response = await self.agent.run(prompt)
            print(f"{self.name}: {response}")
            
            # Extract code from the response
            generated_code = extract_code_from_response(response)
            return response, generated_code
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
                
            await self.execute_code(user_input)

async def main():
    # Create a code execution agent
    agent = CodeExecutionAgent(
        name="CodeBot",
        instructions="You are a helpful assistant that can write and execute Python code to solve problems. Always explain your approach before writing the code, and explain the results after execution."
    )
    
    # Example 1: Mathematical calculation
    print("=== Example 1: Mathematical Calculation ===")
    await agent.execute_code("Calculate the factorial of 10.")
    
    # Example 2: Data visualization
    print("\n=== Example 2: Data Visualization ===")
    await agent.execute_code("Create a bar chart showing the population of the 5 most populous countries.")
    
    # Example 3: File processing
    print("\n=== Example 3: File Processing ===")
    await agent.execute_code("Create a CSV file with sample data and then read it back to display the contents.")
    
    # Example 4: Web scraping
    print("\n=== Example 4: Web Scraping ===")
    await agent.execute_code("Fetch the titles of the top 3 stories from Hacker News.")
    
    # Example 5: Algorithm implementation
    print("\n=== Example 5: Algorithm Implementation ===")
    await agent.execute_code("Implement a quick sort algorithm and use it to sort a list of random numbers.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Accessing Code Interpreter Data from Response Chunks

When using streaming responses, the code being executed is sent in chunks. The `get_code_interpreter_chunk` helper function extracts these chunks and combines them to form the complete code. This allows you to see what code is being executed and potentially save it for later use.

Let's enhance our example to better handle and display the executed code:

```python
# agents/code_execution_agent_with_enhancements.py
import asyncio
import re
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIAssistantsClient
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

def get_code_interpreter_chunk(chunk):
    """
    Helper method to access code interpreter data from response chunks.
    
    Args:
        chunk: The response chunk from the agent.
        
    Returns:
        str or None: The code interpreter input if available, None otherwise.
    """
    if (
        hasattr(chunk, 'raw_representation') and
        hasattr(chunk.raw_representation, 'raw_representation') and
        isinstance(chunk.raw_representation.raw_representation, RunStepDeltaEvent) and
        hasattr(chunk.raw_representation.raw_representation, 'delta') and
        isinstance(chunk.raw_representation.raw_representation.delta, RunStepDelta) and
        hasattr(chunk.raw_representation.raw_representation.delta, 'step_details') and
        isinstance(chunk.raw_representation.raw_representation.delta.step_details, ToolCallDeltaObject) and
        hasattr(chunk.raw_representation.raw_representation.delta.step_details, 'tool_calls') and
        chunk.raw_representation.raw_representation.delta.step_details.tool_calls
    ):
        for tool_call in chunk.raw_representation.raw_representation.delta.step_details.tool_calls:
            if (
                isinstance(tool_call, CodeInterpreterToolCallDelta) and
                hasattr(tool_call, 'code_interpreter') and
                isinstance(tool_call.code_interpreter, CodeInterpreter) and
                tool_call.code_interpreter.input is not None
            ):
                return tool_call.code_interpreter.input
    return None

class EnhancedCodeExecutionAgent:
    def __init__(self, name="CodeBot", instructions="You are a helpful assistant that can write and execute Python code to solve problems."):
        """
        Initialize the Enhanced Code Execution Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedCodeInterpreterTool(),
        )
        self.executed_code = []
    
    async def execute_code(self, prompt, stream=True, save_code=False):
        """
        Send a prompt to the agent and execute the generated code.
        
        Args:
            prompt (str): The prompt describing what code to write and execute.
            stream (bool): Whether to stream the response or get it all at once.
            save_code (bool): Whether to save the executed code for later use.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        if stream:
            response = ""
            generated_code = ""
            print(f"{self.name}: ", end="", flush=True)
            
            async for chunk in self.agent.run_stream(prompt):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
                
                # Extract code from the chunk
                code_chunk = get_code_interpreter_chunk(chunk)
                if code_chunk is not None:
                    generated_code += code_chunk
            
            print()  # New line after the complete response
            
            if save_code and generated_code:
                self.executed_code.append(generated_code)
            
            # Display the executed code
            if generated_code:
                print("\n--- Executed Code ---")
                print(generated_code)
                print("---------------------\n")
            
            return response, generated_code
        else:
            response = await self.agent.run(prompt)
            print(f"{self.name}: {response}")
            
            # Extract code from the response
            generated_code = self.extract_code_from_response(response)
            
            if save_code and generated_code:
                self.executed_code.append(generated_code)
            
            # Display the executed code
            if generated_code:
                print("\n--- Executed Code ---")
                print(generated_code)
                print("---------------------\n")
            
            return response, generated_code
    
    def extract_code_from_response(self, response):
        """
        Extract Python code from a response.
        
        Args:
            response (str): The response containing code.
            
        Returns:
            str: The extracted Python code.
        """
        # Look for code blocks marked with ```python
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        
        if python_code_blocks:
            return python_code_blocks[0]
        
        # Look for code blocks marked with just ```
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0]
        
        # If no code blocks found, return the entire response
        return response
    
    def get_saved_code(self, index=None):
        """
        Get saved code.
        
        Args:
            index (int, optional): The index of the code to retrieve. If None, returns all saved code.
            
        Returns:
            str or list: The saved code or list of all saved code.
        """
        if index is not None:
            if 0 <= index < len(self.executed_code):
                return self.executed_code[index]
            else:
                return f"Index {index} out of range. Available indices: 0-{len(self.executed_code)-1}"
        else:
            return self.executed_code
    
    def save_code_to_file(self, filename, index=None):
        """
        Save saved code to a file.
        
        Args:
            filename (str): The filename to save the code to.
            index (int, optional): The index of the code to save. If None, saves all saved code.
        """
        code = self.get_saved_code(index)
        
        with open(filename, 'w') as f:
            if isinstance(code, list):
                for i, c in enumerate(code):
                    f.write(f"# Code block {i+1}\n")
                    f.write(c)
                    f.write("\n\n")
            else:
                f.write(code)
        
        print(f"Code saved to {filename}")
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands: 'save' to save code, 'list' to list saved code, 'get <index>' to get specific code, 'savefile <filename>' to save code to file")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower() == 'save':
                await self.execute_code(user_input, save_code=True)
            elif user_input.lower() == 'list':
                code_list = self.get_saved_code()
                if code_list:
                    print(f"Saved code blocks ({len(code_list)}):")
                    for i, code in enumerate(code_list):
                        print(f"{i}: {code[:50]}...")
                else:
                    print("No saved code blocks.")
            elif user_input.lower().startswith('get '):
                try:
                    index = int(user_input.split()[1])
                    code = self.get_saved_code(index)
                    print(f"Code block {index}:\n{code}")
                except (ValueError, IndexError):
                    print("Invalid command. Usage: 'get <index>'")
            elif user_input.lower().startswith('savefile '):
                try:
                    filename = user_input.split()[1]
                    self.save_code_to_file(filename)
                except IndexError:
                    print("Invalid command. Usage: 'savefile <filename>'")
            else:
                await self.execute_code(user_input)

async def main():
    # Create an enhanced code execution agent
    agent = EnhancedCodeExecutionAgent(
        name="CodeBot",
        instructions="You are a helpful assistant that can write and execute Python code to solve problems. Always explain your approach before writing the code, and explain the results after execution."
    )
    
    # Example 1: Mathematical calculation with code saving
    print("=== Example 1: Mathematical Calculation ===")
    await agent.execute_code("Calculate the factorial of 10.", save_code=True)
    
    # Example 2: Data visualization with code saving
    print("\n=== Example 2: Data Visualization ===")
    await agent.execute_code("Create a bar chart showing the population of the 5 most populous countries.", save_code=True)
    
    # Example 3: Retrieve saved code
    print("\n=== Example 3: Retrieve Saved Code ===")
    saved_code = agent.get_saved_code(0)
    print(f"First saved code block:\n{saved_code}")
    
    # Example 4: Save code to file
    print("\n=== Example 4: Save Code to File ===")
    agent.save_code_to_file("saved_code.py")
    
    # Example 5: Interactive conversation
    print("\n=== Example 5: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Security Considerations

When working with code execution agents, security is a critical concern. The `HostedCodeInterpreterTool` runs code in a sandboxed environment, but it's still important to be aware of potential security risks:

1. **Code Injection**: Malicious users might try to inject harmful code. Always validate and sanitize inputs when possible.

2. **Resource Consumption**: Executed code could consume excessive resources. Consider implementing timeouts or resource limits.

3. **File System Access**: The code interpreter has limited file system access, but be cautious about what files are created or accessed.

4. **Network Access**: The code interpreter might have network access, which could be used for malicious purposes.

Let's create a more secure version of our code execution agent:

```python
# agents/secure_code_execution_agent.py
import asyncio
import re
import time
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIAssistantsClient

class SecureCodeExecutionAgent:
    def __init__(self, name="SecureCodeBot", instructions="You are a helpful assistant that can write and execute Python code to solve problems."):
        """
        Initialize the Secure Code Execution Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedCodeInterpreterTool(),
        )
        self.max_execution_time = 30  # Maximum execution time in seconds
        self.forbidden_patterns = [
            r'import\s+os',  # Prevent direct os module usage
            r'import\s+subprocess',  # Prevent subprocess usage
            r'import\s+shutil',  # Prevent shutil usage
            r'eval\s*\(',  # Prevent eval usage
            r'exec\s*\(',  # Prevent exec usage
            r'open\s*\(',  # Prevent file operations
            r'__import__\s*\(',  # Prevent dynamic imports
        ]
    
    def is_code_safe(self, code):
        """
        Check if the code is safe to execute.
        
        Args:
            code (str): The code to check.
            
        Returns:
            bool: True if the code is safe, False otherwise.
        """
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code):
                return False
        return True
    
    async def execute_code(self, prompt, stream=True):
        """
        Send a prompt to the agent and execute the generated code with security checks.
        
        Args:
            prompt (str): The prompt describing what code to write and execute.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        if stream:
            response = ""
            generated_code = ""
            print(f"{self.name}: ", end="", flush=True)
            
            start_time = time.time()
            async for chunk in self.agent.run_stream(prompt):
                # Check execution time
                if time.time() - start_time > self.max_execution_time:
                    print("\n[Execution timed out]")
                    break
                
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
                
                # Extract code from the chunk
                code_chunk = get_code_interpreter_chunk(chunk)
                if code_chunk is not None:
                    generated_code += code_chunk
            
            print()  # New line after the complete response
            
            # Check if the generated code is safe
            if generated_code and not self.is_code_safe(generated_code):
                print("\n[Security Warning: Generated code contains potentially unsafe operations and was not executed.]")
                return response, None
            
            # Display the executed code
            if generated_code:
                print("\n--- Executed Code ---")
                print(generated_code)
                print("---------------------\n")
            
            return response, generated_code
        else:
            response = await self.agent.run(prompt)
            print(f"{self.name}: {response}")
            
            # Extract code from the response
            generated_code = self.extract_code_from_response(response)
            
            # Check if the generated code is safe
            if generated_code and not self.is_code_safe(generated_code):
                print("\n[Security Warning: Generated code contains potentially unsafe operations and was not executed.]")
                return response, None
            
            # Display the executed code
            if generated_code:
                print("\n--- Executed Code ---")
                print(generated_code)
                print("---------------------\n")
            
            return response, generated_code
    
    def extract_code_from_response(self, response):
        """
        Extract Python code from a response.
        
        Args:
            response (str): The response containing code.
            
        Returns:
            str: The extracted Python code.
        """
        # Look for code blocks marked with ```python
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        
        if python_code_blocks:
            return python_code_blocks[0]
        
        # Look for code blocks marked with just ```
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0]
        
        # If no code blocks found, return the entire response
        return response

async def main():
    # Create a secure code execution agent
    agent = SecureCodeExecutionAgent(
        name="SecureCodeBot",
        instructions="You are a helpful assistant that can write and execute Python code to solve problems. Always explain your approach before writing the code, and explain the results after execution."
    )
    
    # Example 1: Safe code execution
    print("=== Example 1: Safe Code Execution ===")
    await agent.execute_code("Calculate the factorial of 10.")
    
    # Example 2: Potentially unsafe code (will be blocked)
    print("\n=== Example 2: Potentially Unsafe Code ===")
    await agent.execute_code("List all files in the current directory.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing and Validation

Let's create a test for our Code Execution Agent:

```python
# tests/test_code_execution_agent.py
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.code_execution_agent import CodeExecutionAgent

async def test_code_execution_agent():
    """Test the Code Execution Agent."""
    agent = CodeExecutionAgent(
        name="TestBot",
        instructions="You are a test assistant. Always write and execute Python code to solve problems."
    )
    
    # Test mathematical calculation
    response, code = await agent.execute_code("Calculate 5 + 3.", stream=False)
    assert "8" in response, "Mathematical calculation test failed"
    assert code is not None, "Code generation test failed"
    
    # Test data visualization
    response, code = await agent.execute_code("Create a simple plot with matplotlib.", stream=False)
    assert "plot" in response.lower() or "chart" in response.lower(), "Data visualization test failed"
    assert code is not None, "Code generation test failed"
    
    print("All tests passed for Code Execution Agent!")

if __name__ == "__main__":
    asyncio.run(test_code_execution_agent())
```

### Common Issues and Troubleshooting

1. **Code Execution Failures**: Sometimes the generated code might have syntax errors or logical issues. The agent should be able to debug and fix the code, but in some cases, you might need to provide more specific instructions.

2. **Resource Limitations**: The code interpreter has resource limitations. If you're working with large datasets or complex computations, you might encounter resource constraints.

3. **Security Restrictions**: The code interpreter has security restrictions that prevent certain operations. If your code requires these operations, you might need to find alternative approaches.

4. **Dependency Issues**: The code interpreter has a predefined set of libraries. If you need to use a library that's not available, you won't be able to import it.

### Variations and Extensions

1. **Data Analysis Agent**: Create a specialized agent for data analysis:

```python
class DataAnalysisAgent(CodeExecutionAgent):
    def __init__(self, name="DataBot", instructions="You are a helpful assistant that can write and execute Python code to analyze data."):
        super().__init__(name, instructions)
    
    async def analyze_data(self, data_description, analysis_type):
        """
        Analyze data based on a description and analysis type.
        
        Args:
            data_description (str): Description of the data to analyze.
            analysis_type (str): Type of analysis to perform.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        prompt = f"I have {data_description}. Please perform {analysis_type} analysis on this data."
        return await self.execute_code(prompt)
```

2. **Visualization Agent**: Create a specialized agent for data visualization:

```python
class VisualizationAgent(CodeExecutionAgent):
    def __init__(self, name="VizBot", instructions="You are a helpful assistant that can write and execute Python code to create data visualizations."):
        super().__init__(name, instructions)
    
    async def create_visualization(self, data_description, chart_type):
        """
        Create a visualization based on data description and chart type.
        
        Args:
            data_description (str): Description of the data to visualize.
            chart_type (str): Type of chart to create.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        prompt = f"I have {data_description}. Please create a {chart_type} chart to visualize this data."
        return await self.execute_code(prompt)
```

3. **Algorithm Implementation Agent**: Create a specialized agent for implementing algorithms:

```python
class AlgorithmAgent(CodeExecutionAgent):
    def __init__(self, name="AlgoBot", instructions="You are a helpful assistant that can write and execute Python code to implement algorithms."):
        super().__init__(name, instructions)
    
    async def implement_algorithm(self, algorithm_description):
        """
        Implement an algorithm based on a description.
        
        Args:
            algorithm_description (str): Description of the algorithm to implement.
            
        Returns:
            tuple: A tuple containing the response and the executed code.
        """
        prompt = f"Please implement the following algorithm in Python: {algorithm_description}"
        return await self.execute_code(prompt)
```

With this Code Execution Agent, you can now create AI assistants that can write and execute code to solve complex problems, perform data analysis, create visualizations, and more. In the next section, we'll explore how to build an agent that can search through documents.

<a name="agent4"></a>
## 6. Agent Type 4: Knowledge Retrieval Agent

### Purpose and Use Cases

A Knowledge Retrieval Agent is an AI agent that can search through documents, extract relevant information, and answer questions based on the content of those documents. This capability is particularly useful for creating AI assistants that have access to specific knowledge bases or can reference external documents.

Common use cases include:
- Document-based Q&A systems
- Research assistants
- Customer support agents with product documentation
- Legal document analysis
- Medical information retrieval
- Educational content assistants
- Internal knowledge base access

### Setting Up File Search Capabilities

The Microsoft Agent Framework provides the `HostedFileSearchTool` to enable file search capabilities. This tool allows the agent to search through uploaded documents and retrieve relevant information.

Let's create a Knowledge Retrieval Agent:

```python
# agents/knowledge_retrieval_agent.py
import asyncio
import os
import json
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

class KnowledgeRetrievalAgent:
    def __init__(self, name="KnowledgeBot", instructions="You are a helpful assistant that can search through documents to answer questions."):
        """
        Initialize the Knowledge Retrieval Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedFileSearchTool(),
        )
        self.vector_stores = {}  # To store vector store information
    
    async def create_vector_store(self, name, files=None, file_paths=None):
        """
        Create a vector store with sample documents.
        
        Args:
            name (str): The name of the vector store.
            files (list, optional): List of file objects (content, filename).
            file_paths (list, optional): List of file paths to upload.
            
        Returns:
            str: The ID of the created vector store.
        """
        # Create a vector store
        vector_store = await self.client.client.vector_stores.create(
            name=name,
            expires_after={"anchor": "last_active_at", "days": 1},
        )
        
        # Upload files to the vector store
        if files:
            for file_content, filename in files:
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store.id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        if file_paths:
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                filename = os.path.basename(file_path)
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store.id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        # Store the vector store information
        self.vector_stores[name] = {
            "id": vector_store.id,
            "content": HostedVectorStoreContent(vector_store_id=vector_store.id)
        }
        
        return vector_store.id
    
    async def delete_vector_store(self, name):
        """
        Delete a vector store.
        
        Args:
            name (str): The name of the vector store to delete.
        """
        if name in self.vector_stores:
            vector_store_id = self.vector_stores[name]["id"]
            await self.client.client.vector_stores.delete(vector_store_id=vector_store_id)
            del self.vector_stores[name]
            print(f"Vector store '{name}' deleted.")
        else:
            print(f"Vector store '{name}' not found.")
    
    async def search_documents(self, query, vector_store_name=None, stream=True):
        """
        Search through documents to answer a query.
        
        Args:
            query (str): The query to search for.
            vector_store_name (str, optional): The name of the vector store to search. If None, searches all available vector stores.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Prepare tool resources
        tool_resources = {}
        
        if vector_store_name and vector_store_name in self.vector_stores:
            tool_resources["file_search"] = {
                "vector_store_ids": [self.vector_stores[vector_store_name]["content"].vector_store_id]
            }
        elif self.vector_stores:
            # Use all available vector stores
            vector_store_ids = [
                content["content"].vector_store_id 
                for content in self.vector_stores.values()
            ]
            tool_resources["file_search"] = {"vector_store_ids": vector_store_ids}
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(query, tool_resources=tool_resources):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(query, tool_resources=tool_resources)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands: 'list' to list vector stores, 'create <name>' to create a new vector store, 'delete <name>' to delete a vector store")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower() == 'list':
                if self.vector_stores:
                    print("Available vector stores:")
                    for name in self.vector_stores:
                        print(f"- {name}")
                else:
                    print("No vector stores available.")
            elif user_input.lower().startswith('create '):
                name = user_input[7:].strip()
                if name:
                    print(f"Creating vector store '{name}'...")
                    await self.create_vector_store(name)
                    print(f"Vector store '{name}' created.")
                else:
                    print("Please provide a name for the vector store.")
            elif user_input.lower().startswith('delete '):
                name = user_input[7:].strip()
                if name:
                    await self.delete_vector_store(name)
                else:
                    print("Please provide the name of the vector store to delete.")
            else:
                await self.search_documents(user_input)

async def main():
    # Create a knowledge retrieval agent
    agent = KnowledgeRetrievalAgent(
        name="KnowledgeBot",
        instructions="You are a helpful assistant that can search through documents to answer questions. Always cite your sources when providing information from the documents."
    )
    
    # Example 1: Create a vector store with sample documents
    print("=== Example 1: Creating a Vector Store ===")
    sample_files = [
        (b"The weather today is sunny with a high of 75F. Tomorrow will be partly cloudy with a high of 70F.", "weather.txt"),
        (b"The capital of France is Paris. Paris is known for the Eiffel Tower, the Louvre Museum, and French cuisine.", "france.txt"),
        (b"Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.", "python.txt"),
    ]
    
    vector_store_id = await agent.create_vector_store("sample_docs", files=sample_files)
    print(f"Vector store created with ID: {vector_store_id}")
    
    # Example 2: Search for weather information
    print("\n=== Example 2: Searching for Weather Information ===")
    await agent.search_documents("What's the weather like today?", vector_store_name="sample_docs")
    
    # Example 3: Search for information about France
    print("\n=== Example 3: Searching for Information about France ===")
    await agent.search_documents("What is the capital of France?", vector_store_name="sample_docs")
    
    # Example 4: Search for information about Python
    print("\n=== Example 4: Searching for Information about Python ===")
    await agent.search_documents("Who created Python and when was it first released?", vector_store_name="sample_docs")
    
    # Example 5: Interactive conversation
    print("\n=== Example 5: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()
    
    # Clean up
    await agent.delete_vector_store("sample_docs")

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating and Managing Vector Stores

Vector stores are used to index and search through documents efficiently. In our example, we've created methods to create, manage, and delete vector stores. Let's enhance our agent to better handle vector store operations:

```python
# agents/knowledge_retrieval_agent_with_enhancements.py
import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

class EnhancedKnowledgeRetrievalAgent:
    def __init__(self, name="KnowledgeBot", instructions="You are a helpful assistant that can search through documents to answer questions."):
        """
        Initialize the Enhanced Knowledge Retrieval Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedFileSearchTool(),
        )
        self.vector_stores = {}  # To store vector store information
        self.file_store = {}  # To store file information
    
    async def create_vector_store(self, name, files=None, file_paths=None, expires_after_days=1):
        """
        Create a vector store with sample documents.
        
        Args:
            name (str): The name of the vector store.
            files (list, optional): List of file objects (content, filename).
            file_paths (list, optional): List of file paths to upload.
            expires_after_days (int): Number of days after which the vector store expires.
            
        Returns:
            str: The ID of the created vector store.
        """
        # Create a vector store
        vector_store = await self.client.client.vector_stores.create(
            name=name,
            expires_after={"anchor": "last_active_at", "days": expires_after_days},
        )
        
        # Store file information
        file_ids = []
        
        # Upload files to the vector store
        if files:
            for file_content, filename in files:
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                file_ids.append(file.id)
                self.file_store[file.id] = {"filename": filename, "vector_store": name}
                
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store.id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        if file_paths:
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                filename = os.path.basename(file_path)
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                file_ids.append(file.id)
                self.file_store[file.id] = {"filename": filename, "vector_store": name, "path": file_path}
                
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store.id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        # Store the vector store information
        self.vector_stores[name] = {
            "id": vector_store.id,
            "content": HostedVectorStoreContent(vector_store_id=vector_store.id),
            "file_ids": file_ids,
            "expires_after_days": expires_after_days
        }
        
        return vector_store.id
    
    async def add_files_to_vector_store(self, vector_store_name, files=None, file_paths=None):
        """
        Add files to an existing vector store.
        
        Args:
            vector_store_name (str): The name of the vector store.
            files (list, optional): List of file objects (content, filename).
            file_paths (list, optional): List of file paths to upload.
            
        Returns:
            list: List of file IDs that were added.
        """
        if vector_store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{vector_store_name}' not found.")
        
        vector_store_id = self.vector_stores[vector_store_name]["id"]
        file_ids = []
        
        # Upload files to the vector store
        if files:
            for file_content, filename in files:
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                file_ids.append(file.id)
                self.file_store[file.id] = {"filename": filename, "vector_store": vector_store_name}
                
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store_id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        if file_paths:
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                filename = os.path.basename(file_path)
                file = await self.client.client.files.create(
                    file=(filename, file_content), purpose="user_data"
                )
                file_ids.append(file.id)
                self.file_store[file.id] = {"filename": filename, "vector_store": vector_store_name, "path": file_path}
                
                result = await self.client.client.vector_stores.files.create_and_poll(
                    vector_store_id=vector_store_id, file_id=file.id
                )
                if result.last_error is not None:
                    raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        # Update the vector store information
        self.vector_stores[vector_store_name]["file_ids"].extend(file_ids)
        
        return file_ids
    
    async def delete_vector_store(self, name):
        """
        Delete a vector store.
        
        Args:
            name (str): The name of the vector store to delete.
        """
        if name in self.vector_stores:
            vector_store_id = self.vector_stores[name]["id"]
            
            # Delete files associated with the vector store
            for file_id in self.vector_stores[name]["file_ids"]:
                await self.client.client.files.delete(file_id=file_id)
                if file_id in self.file_store:
                    del self.file_store[file_id]
            
            # Delete the vector store
            await self.client.client.vector_stores.delete(vector_store_id=vector_store_id)
            del self.vector_stores[name]
            print(f"Vector store '{name}' deleted.")
        else:
            print(f"Vector store '{name}' not found.")
    
    async def list_vector_stores(self):
        """
        List all vector stores with their details.
        
        Returns:
            dict: Dictionary of vector store information.
        """
        return {
            name: {
                "id": info["id"],
                "file_count": len(info["file_ids"]),
                "expires_after_days": info["expires_after_days"]
            }
            for name, info in self.vector_stores.items()
        }
    
    async def list_files_in_vector_store(self, vector_store_name):
        """
        List all files in a vector store.
        
        Args:
            vector_store_name (str): The name of the vector store.
            
        Returns:
            list: List of file information.
        """
        if vector_store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{vector_store_name}' not found.")
        
        file_info = []
        for file_id in self.vector_stores[vector_store_name]["file_ids"]:
            if file_id in self.file_store:
                file_info.append({
                    "id": file_id,
                    "filename": self.file_store[file_id]["filename"],
                    "path": self.file_store[file_id].get("path", "Uploaded file")
                })
        
        return file_info
    
    async def search_documents(self, query, vector_store_name=None, stream=True):
        """
        Search through documents to answer a query.
        
        Args:
            query (str): The query to search for.
            vector_store_name (str, optional): The name of the vector store to search. If None, searches all available vector stores.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Prepare tool resources
        tool_resources = {}
        
        if vector_store_name and vector_store_name in self.vector_stores:
            tool_resources["file_search"] = {
                "vector_store_ids": [self.vector_stores[vector_store_name]["content"].vector_store_id]
            }
        elif self.vector_stores:
            # Use all available vector stores
            vector_store_ids = [
                content["content"].vector_store_id 
                for content in self.vector_stores.values()
            ]
            tool_resources["file_search"] = {"vector_store_ids": vector_store_ids}
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(query, tool_resources=tool_resources):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(query, tool_resources=tool_resources)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'list' - List all vector stores")
        print("  'files <vector_store>' - List files in a vector store")
        print("  'create <name>' - Create a new vector store")
        print("  'add <vector_store> <file_path>' - Add a file to a vector store")
        print("  'delete <name>' - Delete a vector store")
        print("  'search <vector_store> <query>' - Search in a specific vector store")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower() == 'list':
                vector_stores = await self.list_vector_stores()
                if vector_stores:
                    print("Available vector stores:")
                    for name, info in vector_stores.items():
                        print(f"- {name} (ID: {info['id']}, Files: {info['file_count']}, Expires after: {info['expires_after_days']} days)")
                else:
                    print("No vector stores available.")
            elif user_input.lower().startswith('files '):
                vector_store_name = user_input[6:].strip()
                try:
                    files = await self.list_files_in_vector_store(vector_store_name)
                    if files:
                        print(f"Files in vector store '{vector_store_name}':")
                        for file_info in files:
                            print(f"- {file_info['filename']} (ID: {file_info['id']}, Path: {file_info['path']})")
                    else:
                        print(f"No files in vector store '{vector_store_name}'.")
                except ValueError as e:
                    print(str(e))
            elif user_input.lower().startswith('create '):
                name = user_input[7:].strip()
                if name:
                    print(f"Creating vector store '{name}'...")
                    await self.create_vector_store(name)
                    print(f"Vector store '{name}' created.")
                else:
                    print("Please provide a name for the vector store.")
            elif user_input.lower().startswith('add '):
                parts = user_input[4:].split(' ', 1)
                if len(parts) == 2:
                    vector_store_name, file_path = parts
                    try:
                        print(f"Adding file '{file_path}' to vector store '{vector_store_name}'...")
                        await self.add_files_to_vector_store(vector_store_name, file_paths=[file_path])
                        print(f"File added to vector store.")
                    except Exception as e:
                        print(f"Error adding file: {str(e)}")
                else:
                    print("Please provide both vector store name and file path. Usage: 'add <vector_store> <file_path>'")
            elif user_input.lower().startswith('delete '):
                name = user_input[7:].strip()
                if name:
                    await self.delete_vector_store(name)
                else:
                    print("Please provide the name of the vector store to delete.")
            elif user_input.lower().startswith('search '):
                parts = user_input[7:].split(' ', 1)
                if len(parts) == 2:
                    vector_store_name, query = parts
                    await self.search_documents(query, vector_store_name=vector_store_name)
                else:
                    await self.search_documents(user_input[7:])
            else:
                await self.search_documents(user_input)

async def main():
    # Create an enhanced knowledge retrieval agent
    agent = EnhancedKnowledgeRetrievalAgent(
        name="KnowledgeBot",
        instructions="You are a helpful assistant that can search through documents to answer questions. Always cite your sources when providing information from the documents."
    )
    
    # Example 1: Create a vector store with sample documents
    print("=== Example 1: Creating a Vector Store ===")
    sample_files = [
        (b"The weather today is sunny with a high of 75F. Tomorrow will be partly cloudy with a high of 70F.", "weather.txt"),
        (b"The capital of France is Paris. Paris is known for the Eiffel Tower, the Louvre Museum, and French cuisine.", "france.txt"),
        (b"Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.", "python.txt"),
    ]
    
    vector_store_id = await agent.create_vector_store("sample_docs", files=sample_files)
    print(f"Vector store created with ID: {vector_store_id}")
    
    # Example 2: List vector stores
    print("\n=== Example 2: Listing Vector Stores ===")
    vector_stores = await agent.list_vector_stores()
    print(json.dumps(vector_stores, indent=2))
    
    # Example 3: List files in a vector store
    print("\n=== Example 3: Listing Files in a Vector Store ===")
    files = await agent.list_files_in_vector_store("sample_docs")
    print(json.dumps(files, indent=2))
    
    # Example 4: Add more files to the vector store
    print("\n=== Example 4: Adding More Files to the Vector Store ===")
    more_files = [
        (b"The solar system consists of the Sun and the objects that orbit it, including eight planets.", "solar_system.txt"),
        (b"Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.", "ml.txt"),
    ]
    
    file_ids = await agent.add_files_to_vector_store("sample_docs", files=more_files)
    print(f"Added files with IDs: {file_ids}")
    
    # Example 5: Search for information
    print("\n=== Example 5: Searching for Information ===")
    await agent.search_documents("What is machine learning?", vector_store_name="sample_docs")
    
    # Example 6: Interactive conversation
    print("\n=== Example 6: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()
    
    # Clean up
    await agent.delete_vector_store("sample_docs")

if __name__ == "__main__":
    asyncio.run(main())
```

### Querying Documents for Information

Once you have documents uploaded to a vector store, you can query them for information. The agent will search through the documents and provide answers based on the content. Let's create a more specialized example that demonstrates document-based Q&A:

```python
# agents/document_qa_agent.py
import asyncio
import os
from typing import List, Dict, Any, Optional
from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.openai import OpenAIAssistantsClient

class DocumentQAAgent:
    def __init__(self, name="DocQABot", instructions="You are a helpful assistant that can answer questions based on the provided documents. Always cite your sources when providing information."):
        """
        Initialize the Document QA Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIAssistantsClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedFileSearchTool(),
        )
        self.vector_stores = {}  # To store vector store information
    
    async def create_document_collection(self, name, documents):
        """
        Create a document collection with the provided documents.
        
        Args:
            name (str): The name of the document collection.
            documents (list): List of document objects (title, content).
            
        Returns:
            str: The ID of the created vector store.
        """
        # Create a vector store
        vector_store = await self.client.client.vector_stores.create(
            name=name,
            expires_after={"anchor": "last_active_at", "days": 7},
        )
        
        # Upload documents to the vector store
        file_ids = []
        for title, content in documents:
            # Create a file with the document content
            file = await self.client.client.files.create(
                file=(f"{title}.txt", content.encode()), purpose="user_data"
            )
            file_ids.append(file.id)
            
            # Add the file to the vector store
            result = await self.client.client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store.id, file_id=file.id
            )
            if result.last_error is not None:
                raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")
        
        # Store the vector store information
        self.vector_stores[name] = {
            "id": vector_store.id,
            "content": HostedVectorStoreContent(vector_store_id=vector_store.id),
            "file_ids": file_ids
        }
        
        return vector_store.id
    
    async def ask_question(self, question, collection_name=None, stream=True):
        """
        Ask a question based on the documents.
        
        Args:
            question (str): The question to ask.
            collection_name (str, optional): The name of the document collection to search. If None, searches all available collections.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Prepare tool resources
        tool_resources = {}
        
        if collection_name and collection_name in self.vector_stores:
            tool_resources["file_search"] = {
                "vector_store_ids": [self.vector_stores[collection_name]["content"].vector_store_id]
            }
        elif self.vector_stores:
            # Use all available vector stores
            vector_store_ids = [
                content["content"].vector_store_id 
                for content in self.vector_stores.values()
            ]
            tool_resources["file_search"] = {"vector_store_ids": vector_store_ids}
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(question, tool_resources=tool_resources):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(question, tool_resources=tool_resources)
            print(f"{self.name}: {response}")
            return response
    
    async def summarize_document(self, collection_name, document_title=None):
        """
        Summarize a document or a collection of documents.
        
        Args:
            collection_name (str): The name of the document collection.
            document_title (str, optional): The title of a specific document to summarize. If None, summarizes the entire collection.
            
        Returns:
            str: The summary.
        """
        if document_title:
            question = f"Please summarize the document titled '{document_title}'."
        else:
            question = "Please provide a summary of the documents in this collection."
        
        return await self.ask_question(question, collection_name=collection_name)
    
    async def extract_key_points(self, collection_name, document_title=None):
        """
        Extract key points from a document or a collection of documents.
        
        Args:
            collection_name (str): The name of the document collection.
            document_title (str, optional): The title of a specific document to extract key points from. If None, extracts key points from the entire collection.
            
        Returns:
            str: The key points.
        """
        if document_title:
            question = f"Please extract the key points from the document titled '{document_title}'."
        else:
            question = "Please extract the key points from the documents in this collection."
        
        return await self.ask_question(question, collection_name=collection_name)
    
    async def compare_documents(self, collection_name, document_titles):
        """
        Compare multiple documents in a collection.
        
        Args:
            collection_name (str): The name of the document collection.
            document_titles (list): List of document titles to compare.
            
        Returns:
            str: The comparison.
        """
        if len(document_titles) < 2:
            return "Please provide at least two documents to compare."
        
        question = f"Please compare the following documents: {', '.join(document_titles)}. Highlight similarities and differences."
        return await self.ask_question(question, collection_name=collection_name)
    
    async def delete_document_collection(self, name):
        """
        Delete a document collection.
        
        Args:
            name (str): The name of the document collection to delete.
        """
        if name in self.vector_stores:
            vector_store_id = self.vector_stores[name]["id"]
            
            # Delete files associated with the vector store
            for file_id in self.vector_stores[name]["file_ids"]:
                await self.client.client.files.delete(file_id=file_id)
            
            # Delete the vector store
            await self.client.client.vector_stores.delete(vector_store_id=vector_store_id)
            del self.vector_stores[name]
            print(f"Document collection '{name}' deleted.")
        else:
            print(f"Document collection '{name}' not found.")

async def main():
    # Create a document QA agent
    agent = DocumentQAAgent(
        name="DocQABot",
        instructions="You are a helpful assistant that can answer questions based on the provided documents. Always cite your sources when providing information."
    )
    
    # Example 1: Create a document collection with sample documents
    print("=== Example 1: Creating a Document Collection ===")
    documents = [
        ("Climate Change", "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver of climate change since the 1800s, primarily due to burning fossil fuels, which produces heat-trapping gases."),
        ("Renewable Energy", "Renewable energy is energy derived from natural sources that are replenished on a human timescale. Examples include sunlight, wind, rain, tides, waves, and geothermal heat. Renewable energy is often used for electricity generation, heating and cooling, and transportation."),
        ("Electric Vehicles", "An electric vehicle (EV) is a vehicle that uses one or more electric motors for propulsion. EVs are seen as a key solution to reducing greenhouse gas emissions and combating climate change. The market for electric vehicles is growing rapidly as battery technology improves and prices decrease."),
    ]
    
    vector_store_id = await agent.create_document_collection("environment", documents)
    print(f"Document collection created with ID: {vector_store_id}")
    
    # Example 2: Ask a question about climate change
    print("\n=== Example 2: Asking a Question about Climate Change ===")
    await agent.ask_question("What is climate change and what causes it?", collection_name="environment")
    
    # Example 3: Ask a question about renewable energy
    print("\n=== Example 3: Asking a Question about Renewable Energy ===")
    await agent.ask_question("What are some examples of renewable energy?", collection_name="environment")
    
    # Example 4: Summarize a document
    print("\n=== Example 4: Summarizing a Document ===")
    await agent.summarize_document("environment", "Electric Vehicles")
    
    # Example 5: Extract key points from a document
    print("\n=== Example 5: Extracting Key Points from a Document ===")
    await agent.extract_key_points("environment", "Renewable Energy")
    
    # Example 6: Compare documents
    print("\n=== Example 6: Comparing Documents ===")
    await agent.compare_documents("environment", ["Climate Change", "Renewable Energy"])
    
    # Clean up
    await agent.delete_document_collection("environment")

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing and Validation

Let's create a test for our Knowledge Retrieval Agent:

```python
# tests/test_knowledge_retrieval_agent.py
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent

async def test_knowledge_retrieval_agent():
    """Test the Knowledge Retrieval Agent."""
    agent = KnowledgeRetrievalAgent(
        name="TestBot",
        instructions="You are a test assistant. Always search through documents to answer questions."
    )
    
    # Create a vector store with test documents
    test_files = [
        (b"The capital of France is Paris.", "france.txt"),
        (b"The capital of Japan is Tokyo.", "japan.txt"),
    ]
    
    vector_store_id = await agent.create_vector_store("test_docs", files=test_files)
    
    # Test document search
    response = await agent.search_documents("What is the capital of France?", vector_store_name="test_docs", stream=False)
    assert "Paris" in response, "Document search test failed"
    
    # Test document search with different query
    response = await agent.search_documents("What is the capital of Japan?", vector_store_name="test_docs", stream=False)
    assert "Tokyo" in response, "Document search test failed"
    
    # Clean up
    await agent.delete_vector_store("test_docs")
    
    print("All tests passed for Knowledge Retrieval Agent!")

if __name__ == "__main__":
    asyncio.run(test_knowledge_retrieval_agent())
```

### Common Issues and Troubleshooting

1. **File Processing Failures**: Sometimes files might fail to process during upload. Check the error messages and ensure the files are in a supported format.

2. **Vector Store Expiration**: Vector stores have an expiration time. If you're getting errors about expired vector stores, you might need to recreate them.

3. **Search Relevance**: The quality of search results depends on how well the documents are structured and indexed. For better results, ensure your documents are well-organized and contain clear, relevant information.

4. **Large Documents**: Very large documents might be truncated or not fully processed. Consider breaking down large documents into smaller, more focused sections.

### Variations and Extensions

1. **Multi-Collection Agent**: Create an agent that can manage and search across multiple document collections:

```python
class MultiCollectionKnowledgeAgent(KnowledgeRetrievalAgent):
    async def search_across_collections(self, query, collection_names=None):
        """
        Search across multiple document collections.
        
        Args:
            query (str): The query to search for.
            collection_names (list, optional): List of collection names to search. If None, searches all available collections.
            
        Returns:
            str: The agent's response.
        """
        if collection_names:
            # Use only the specified collections
            vector_store_ids = []
            for name in collection_names:
                if name in self.vector_stores:
                    vector_store_ids.append(self.vector_stores[name]["content"].vector_store_id)
            
            if not vector_store_ids:
                return "None of the specified collections were found."
            
            tool_resources = {"file_search": {"vector_store_ids": vector_store_ids}}
        else:
            # Use all available collections
            vector_store_ids = [
                content["content"].vector_store_id 
                for content in self.vector_stores.values()
            ]
            tool_resources = {"file_search": {"vector_store_ids": vector_store_ids}}
        
        response = await self.agent.run(query, tool_resources=tool_resources)
        return response
```

2. **Document Summarization Agent**: Create a specialized agent for summarizing documents:

```python
class DocumentSummarizationAgent(KnowledgeRetrievalAgent):
    async def summarize_collection(self, collection_name, summary_length="medium"):
        """
        Summarize a document collection.
        
        Args:
            collection_name (str): The name of the document collection.
            summary_length (str): The desired length of the summary (short, medium, long).
            
        Returns:
            str: The summary.
        """
        if summary_length == "short":
            prompt = "Please provide a brief summary of the documents in this collection, highlighting the main points in 2-3 sentences."
        elif summary_length == "medium":
            prompt = "Please provide a comprehensive summary of the documents in this collection, covering the main topics and key details in a paragraph."
        else:  # long
            prompt = "Please provide a detailed summary of the documents in this collection, covering all major topics, key details, and any important conclusions or recommendations."
        
        return await self.search_documents(prompt, vector_store_name=collection_name)
```

3. **Document Comparison Agent**: Create a specialized agent for comparing documents:

```python
class DocumentComparisonAgent(KnowledgeRetrievalAgent):
    async def compare_documents(self, collection_name, document_topics=None):
        """
        Compare documents in a collection based on specific topics.
        
        Args:
            collection_name (str): The name of the document collection.
            document_topics (list, optional): List of topics to compare. If None, compares all topics.
            
        Returns:
            str: The comparison.
        """
        if document_topics:
            prompt = f"Please compare the documents in this collection with respect to the following topics: {', '.join(document_topics)}. Highlight similarities and differences."
        else:
            prompt = "Please compare the documents in this collection, highlighting similarities and differences in content, perspective, and conclusions."
        
        return await self.search_documents(prompt, vector_store_name=collection_name)
```

With this Knowledge Retrieval Agent, you can now create AI assistants that can search through documents, extract relevant information, and answer questions based on the content of those documents. In the next section, we'll explore how to build an agent that can process multiple types of data, including text, images, and web data.

<a name="agent5"></a>
## 7. Agent Type 5: Multimodal Agent

### Purpose and Use Cases

A Multimodal Agent is an AI agent that can process and understand multiple types of data, including text, images, and web data. This capability allows the agent to provide more comprehensive and context-aware responses by leveraging information from various sources and formats.

Common use cases include:
- Image analysis and description
- Web research and information synthesis
- Content creation with text and images
- Data visualization and interpretation
- Multilingual content processing
- Accessibility tools for visually impaired users
- Educational content creation

### Processing Text and Images

The Microsoft Agent Framework supports vision capabilities through OpenAI's vision models, allowing agents to analyze and interpret images. Let's create a Multimodal Agent that can process both text and images:

```python
# agents/multimodal_agent.py
import asyncio
import base64
import os
from typing import List, Dict, Any, Optional, Union
from agent_framework import ChatAgent, HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

class MultimodalAgent:
    def __init__(self, name="MultiBot", instructions="You are a helpful assistant that can process text, images, and web data."):
        """
        Initialize the Multimodal Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIResponsesClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedWebSearchTool(),
        )
    
    def encode_image(self, image_path):
        """
        Encode an image file to base64.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The base64-encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def analyze_image(self, image_path, question=None, stream=True):
        """
        Analyze an image and optionally answer a question about it.
        
        Args:
            image_path (str): The path to the image file.
            question (str, optional): A specific question about the image.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Prepare the content
        content = [
            {"type": "text", "text": question or "Please describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(content):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(content)
            print(f"{self.name}: {response}")
            return response
    
    async def search_web(self, query, stream=True):
        """
        Search the web for information.
        
        Args:
            query (str): The search query.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(query, tools=[HostedWebSearchTool()]):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(query, tools=[HostedWebSearchTool()])
            print(f"{self.name}: {response}")
            return response
    
    async def analyze_image_with_web_context(self, image_path, question=None, stream=True):
        """
        Analyze an image with additional web context.
        
        Args:
            image_path (str): The path to the image file.
            question (str, optional): A specific question about the image.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # First, analyze the image
        image_description = await self.analyze_image(image_path, question or "Please describe this image in detail.", stream=False)
        
        # Then, search for related information on the web
        search_query = f"Information about {image_description[:100]}..."  # Use first 100 chars of description
        web_context = await self.search_web(search_query, stream=False)
        
        # Finally, combine the image analysis with web context
        combined_query = f"Based on the image description '{image_description}' and the web information '{web_context}', please provide a comprehensive analysis."
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(combined_query):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(combined_query)
            print(f"{self.name}: {response}")
            return response
    
    async def compare_images(self, image_paths, stream=True):
        """
        Compare multiple images.
        
        Args:
            image_paths (list): List of paths to the image files.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if len(image_paths) < 2:
            return "Please provide at least two images to compare."
        
        # Encode all images
        encoded_images = [self.encode_image(path) for path in image_paths]
        
        # Prepare the content
        content = [{"type": "text", "text": "Please compare these images and highlight similarities and differences."}]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "high"
                }
            })
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(content):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(content)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'image <path> [question]' - Analyze an image")
        print("  'web <query>' - Search the web")
        print("  'compare <path1> <path2> ...' - Compare multiple images")
        print("  'image_web <path> [question]' - Analyze an image with web context")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower().startswith('image '):
                parts = user_input[6:].split(' ', 1)
                image_path = parts[0]
                question = parts[1] if len(parts) > 1 else None
                await self.analyze_image(image_path, question)
            elif user_input.lower().startswith('web '):
                query = user_input[4:]
                await self.search_web(query)
            elif user_input.lower().startswith('compare '):
                paths = user_input[8:].split()
                await self.compare_images(paths)
            elif user_input.lower().startswith('image_web '):
                parts = user_input[10:].split(' ', 1)
                image_path = parts[0]
                question = parts[1] if len(parts) > 1 else None
                await self.analyze_image_with_web_context(image_path, question)
            else:
                # Default to text-based conversation
                if stream:
                    response = ""
                    print(f"{self.name}: ", end="", flush=True)
                    async for chunk in self.agent.run_stream(user_input):
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            response += chunk.text
                    print()  # New line after the complete response
                else:
                    response = await self.agent.run(user_input)
                    print(f"{self.name}: {response}")

async def main():
    # Create a multimodal agent
    agent = MultimodalAgent(
        name="MultiBot",
        instructions="You are a helpful assistant that can process text, images, and web data. Provide detailed and accurate information based on all available sources."
    )
    
    # Example 1: Text-based conversation
    print("=== Example 1: Text-Based Conversation ===")
    await agent.agent.run("Tell me about artificial intelligence.")
    
    # Example 2: Web search
    print("\n=== Example 2: Web Search ===")
    await agent.search_web("Latest developments in renewable energy")
    
    # Example 3: Image analysis (requires an image file)
    # Uncomment the following lines and provide a path to an image file
    # print("\n=== Example 3: Image Analysis ===")
    # await agent.analyze_image("path/to/your/image.jpg", "What do you see in this image?")
    
    # Example 4: Image comparison (requires two image files)
    # Uncomment the following lines and provide paths to two image files
    # print("\n=== Example 4: Image Comparison ===")
    # await agent.compare_images(["path/to/image1.jpg", "path/to/image2.jpg"])
    
    # Example 5: Interactive conversation
    print("\n=== Example 5: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integrating Web Search Capabilities

Web search capabilities allow the agent to access up-to-date information from the internet, greatly expanding its knowledge beyond its training data. Let's enhance our Multimodal Agent with more advanced web search capabilities:

```python
# agents/multimodal_agent_with_web_search.py
import asyncio
import base64
import os
from typing import List, Dict, Any, Optional, Union
from agent_framework import ChatAgent, HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

class EnhancedMultimodalAgent:
    def __init__(self, name="MultiBot", instructions="You are a helpful assistant that can process text, images, and web data."):
        """
        Initialize the Enhanced Multimodal Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIResponsesClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedWebSearchTool(),
        )
        self.web_search_tool = HostedWebSearchTool()
    
    def encode_image(self, image_path):
        """
        Encode an image file to base64.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The base64-encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def analyze_image(self, image_path, question=None, stream=True):
        """
        Analyze an image and optionally answer a question about it.
        
        Args:
            image_path (str): The path to the image file.
            question (str, optional): A specific question about the image.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Prepare the content
        content = [
            {"type": "text", "text": question or "Please describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(content):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(content)
            print(f"{self.name}: {response}")
            return response
    
    async def search_web(self, query, location=None, stream=True):
        """
        Search the web for information with optional location context.
        
        Args:
            query (str): The search query.
            location (dict, optional): Location context for the search.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Prepare additional properties for location context
        additional_properties = {}
        if location:
            additional_properties["user_location"] = location
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(
                query, 
                tools=[HostedWebSearchTool(additional_properties=additional_properties)]
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(
                query, 
                tools=[HostedWebSearchTool(additional_properties=additional_properties)]
            )
            print(f"{self.name}: {response}")
            return response
    
    async def research_topic(self, topic, depth="medium", stream=True):
        """
        Conduct comprehensive research on a topic.
        
        Args:
            topic (str): The topic to research.
            depth (str): The depth of research (shallow, medium, deep).
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The research results.
        """
        if depth == "shallow":
            query = f"Provide a brief overview of {topic}."
        elif depth == "medium":
            query = f"Provide a comprehensive overview of {topic}, including key concepts, recent developments, and future prospects."
        else:  # deep
            query = f"Conduct an in-depth analysis of {topic}, covering its history, current state, key challenges, recent research, and future directions."
        
        return await self.search_web(query, stream=stream)
    
    async def compare_topics(self, topics, stream=True):
        """
        Compare multiple topics based on web research.
        
        Args:
            topics (list): List of topics to compare.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The comparison results.
        """
        if len(topics) < 2:
            return "Please provide at least two topics to compare."
        
        query = f"Compare and contrast the following topics: {', '.join(topics)}. Highlight similarities, differences, advantages, and disadvantages."
        return await self.search_web(query, stream=stream)
    
    async def analyze_trends(self, topic, time_period="recent", stream=True):
        """
        Analyze trends related to a topic.
        
        Args:
            topic (str): The topic to analyze trends for.
            time_period (str): The time period for trend analysis (recent, past_year, past_5_years).
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The trend analysis results.
        """
        if time_period == "recent":
            query = f"What are the recent trends related to {topic}?"
        elif time_period == "past_year":
            query = f"What have been the major trends related to {topic} in the past year?"
        else:  # past_5_years
            query = f"What have been the major trends related to {topic} in the past 5 years?"
        
        return await self.search_web(query, stream=stream)
    
    async def find_experts(self, topic, stream=True):
        """
        Find experts or thought leaders in a specific field.
        
        Args:
            topic (str): The field or topic.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: List of experts or thought leaders.
        """
        query = f"Who are the leading experts or thought leaders in the field of {topic}?"
        return await self.search_web(query, stream=stream)
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'image <path> [question]' - Analyze an image")
        print("  'web <query>' - Search the web")
        print("  'web_loc <query> <country> <city>' - Search the web with location context")
        print("  'research <topic> [depth]' - Conduct research on a topic (depth: shallow, medium, deep)")
        print("  'compare <topic1> <topic2> ...' - Compare multiple topics")
        print("  'trends <topic> [period]' - Analyze trends (period: recent, past_year, past_5_years)")
        print("  'experts <topic>' - Find experts in a field")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower().startswith('image '):
                parts = user_input[6:].split(' ', 1)
                image_path = parts[0]
                question = parts[1] if len(parts) > 1 else None
                await self.analyze_image(image_path, question)
            elif user_input.lower().startswith('web '):
                query = user_input[4:]
                await self.search_web(query)
            elif user_input.lower().startswith('web_loc '):
                parts = user_input[9:].split(' ', 2)
                if len(parts) >= 3:
                    query, country, city = parts
                    location = {"country": country, "city": city}
                    await self.search_web(query, location)
                else:
                    print("Please provide query, country, and city. Usage: 'web_loc <query> <country> <city>'")
            elif user_input.lower().startswith('research '):
                parts = user_input[9:].split(' ', 1)
                topic = parts[0]
                depth = parts[1] if len(parts) > 1 else "medium"
                await self.research_topic(topic, depth)
            elif user_input.lower().startswith('compare '):
                topics = user_input[8:].split()
                await self.compare_topics(topics)
            elif user_input.lower().startswith('trends '):
                parts = user_input[7:].split(' ', 1)
                topic = parts[0]
                period = parts[1] if len(parts) > 1 else "recent"
                await self.analyze_trends(topic, period)
            elif user_input.lower().startswith('experts '):
                topic = user_input[8:]
                await self.find_experts(topic)
            else:
                # Default to text-based conversation
                response = ""
                print(f"{self.name}: ", end="", flush=True)
                async for chunk in self.agent.run_stream(user_input):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response += chunk.text
                print()  # New line after the complete response

async def main():
    # Create an enhanced multimodal agent
    agent = EnhancedMultimodalAgent(
        name="MultiBot",
        instructions="You are a helpful assistant that can process text, images, and web data. Provide detailed and accurate information based on all available sources."
    )
    
    # Example 1: Text-based conversation
    print("=== Example 1: Text-Based Conversation ===")
    await agent.agent.run("Tell me about artificial intelligence.")
    
    # Example 2: Web search
    print("\n=== Example 2: Web Search ===")
    await agent.search_web("Latest developments in renewable energy")
    
    # Example 3: Research a topic
    print("\n=== Example 3: Research a Topic ===")
    await agent.research_topic("quantum computing", depth="medium")
    
    # Example 4: Compare topics
    print("\n=== Example 4: Compare Topics ===")
    await agent.compare_topics(["machine learning", "deep learning"])
    
    # Example 5: Analyze trends
    print("\n=== Example 5: Analyze Trends ===")
    await agent.analyze_trends("electric vehicles", time_period="past_year")
    
    # Example 6: Find experts
    print("\n=== Example 6: Find Experts ===")
    await agent.find_experts("climate change")
    
    # Example 7: Location-based search
    print("\n=== Example 7: Location-Based Search ===")
    await agent.search_web("best restaurants", location={"country": "US", "city": "New York"})
    
    # Example 8: Interactive conversation
    print("\n=== Example 8: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Combining Multiple Modalities

The true power of a Multimodal Agent comes from its ability to combine information from multiple sources and formats. Let's create an example that demonstrates this capability:

```python
# agents/multimodal_synthesis_agent.py
import asyncio
import base64
import os
from typing import List, Dict, Any, Optional, Union
from agent_framework import ChatAgent, HostedWebSearchTool
from agent_framework.openai import OpenAIResponsesClient

class MultimodalSynthesisAgent:
    def __init__(self, name="SynthesisBot", instructions="You are a helpful assistant that can process text, images, and web data, and synthesize information from multiple sources."):
        """
        Initialize the Multimodal Synthesis Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIResponsesClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
            tools=HostedWebSearchTool(),
        )
    
    def encode_image(self, image_path):
        """
        Encode an image file to base64.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The base64-encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def analyze_image(self, image_path, question=None, stream=True):
        """
        Analyze an image and optionally answer a question about it.
        
        Args:
            image_path (str): The path to the image file.
            question (str, optional): A specific question about the image.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Prepare the content
        content = [
            {"type": "text", "text": question or "Please describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(content):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(content)
            print(f"{self.name}: {response}")
            return response
    
    async def search_web(self, query, stream=True):
        """
        Search the web for information.
        
        Args:
            query (str): The search query.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(query, tools=[HostedWebSearchTool()]):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(query, tools=[HostedWebSearchTool()])
            print(f"{self.name}: {response}")
            return response
    
    async def synthesize_image_and_web(self, image_path, web_query, synthesis_question=None, stream=True):
        """
        Synthesize information from an image and web search.
        
        Args:
            image_path (str): The path to the image file.
            web_query (str): The query for web search.
            synthesis_question (str, optional): A specific question for synthesis.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The synthesized response.
        """
        # Analyze the image
        print("Analyzing image...")
        image_analysis = await self.analyze_image(image_path, "Please describe this image in detail, focusing on key elements and context.", stream=False)
        
        # Search the web
        print("Searching the web...")
        web_info = await self.search_web(web_query, stream=False)
        
        # Synthesize the information
        if synthesis_question:
            synthesis_query = f"Based on the image analysis '{image_analysis}' and the web information '{web_info}', please answer the following question: {synthesis_question}"
        else:
            synthesis_query = f"Based on the image analysis '{image_analysis}' and the web information '{web_info}', please provide a comprehensive synthesis that connects the visual information with the web context."
        
        print("Synthesizing information...")
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(synthesis_query):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(synthesis_query)
            print(f"{self.name}: {response}")
            return response
    
    async def create_report_from_multiple_sources(self, topic, image_paths=None, web_queries=None, report_type="overview", stream=True):
        """
        Create a comprehensive report from multiple sources.
        
        Args:
            topic (str): The main topic of the report.
            image_paths (list, optional): List of paths to image files.
            web_queries (list, optional): List of web search queries.
            report_type (str): The type of report (overview, analysis, comparison).
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The generated report.
        """
        # Collect information from all sources
        all_info = {"topic": topic, "image_analyses": [], "web_info": []}
        
        # Analyze images
        if image_paths:
            print("Analyzing images...")
            for i, image_path in enumerate(image_paths):
                print(f"Analyzing image {i+1}/{len(image_paths)}...")
                image_analysis = await self.analyze_image(
                    image_path, 
                    f"Please describe this image in the context of {topic}.",
                    stream=False
                )
                all_info["image_analyses"].append(image_analysis)
        
        # Search the web
        if web_queries:
            print("Searching the web...")
            for i, query in enumerate(web_queries):
                print(f"Searching for {i+1}/{len(web_queries)}: {query}")
                web_info = await self.search_web(query, stream=False)
                all_info["web_info"].append(web_info)
        
        # Generate the report
        if report_type == "overview":
            report_query = f"Create a comprehensive overview report on {topic} based on the following information:"
        elif report_type == "analysis":
            report_query = f"Create an in-depth analysis report on {topic} based on the following information:"
        else:  # comparison
            report_query = f"Create a comparison report on {topic} based on the following information:"
        
        # Add all collected information to the query
        if all_info["image_analyses"]:
            report_query += f"\n\nImage Analyses:\n"
            for i, analysis in enumerate(all_info["image_analyses"]):
                report_query += f"Image {i+1}: {analysis}\n"
        
        if all_info["web_info"]:
            report_query += f"\n\nWeb Information:\n"
            for i, info in enumerate(all_info["web_info"]):
                report_query += f"Search {i+1}: {info}\n"
        
        report_query += "\n\nPlease create a well-structured report with sections, headings, and a conclusion."
        
        print("Generating report...")
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(report_query):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await self.agent.run(report_query)
            print(f"{self.name}: {response}")
            return response
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'image <path> [question]' - Analyze an image")
        print("  'web <query>' - Search the web")
        print("  'synthesize <image_path> <web_query> [question]' - Synthesize image and web info")
        print("  'report <topic> [type]' - Create a report from multiple sources")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower().startswith('image '):
                parts = user_input[6:].split(' ', 1)
                image_path = parts[0]
                question = parts[1] if len(parts) > 1 else None
                await self.analyze_image(image_path, question)
            elif user_input.lower().startswith('web '):
                query = user_input[4:]
                await self.search_web(query)
            elif user_input.lower().startswith('synthesize '):
                parts = user_input[10:].split(' ', 2)
                if len(parts) >= 2:
                    image_path, web_query = parts[:2]
                    synthesis_question = parts[2] if len(parts) > 2 else None
                    await self.synthesize_image_and_web(image_path, web_query, synthesis_question)
                else:
                    print("Please provide image path and web query. Usage: 'synthesize <image_path> <web_query> [question]'")
            elif user_input.lower().startswith('report '):
                parts = user_input[7:].split(' ', 1)
                topic = parts[0]
                report_type = parts[1] if len(parts) > 1 else "overview"
                
                # For simplicity, we'll use predefined image paths and web queries
                # In a real implementation, you would prompt the user for these
                print("Creating report with predefined sources...")
                await self.create_report_from_multiple_sources(
                    topic,
                    image_paths=None,  # Add actual image paths here
                    web_queries=[f"Latest information about {topic}", f"History of {topic}"],
                    report_type=report_type
                )
            else:
                # Default to text-based conversation
                response = ""
                print(f"{self.name}: ", end="", flush=True)
                async for chunk in self.agent.run_stream(user_input):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response += chunk.text
                print()  # New line after the complete response

async def main():
    # Create a multimodal synthesis agent
    agent = MultimodalSynthesisAgent(
        name="SynthesisBot",
        instructions="You are a helpful assistant that can process text, images, and web data, and synthesize information from multiple sources. Provide detailed and accurate information based on all available sources."
    )
    
    # Example 1: Text-based conversation
    print("=== Example 1: Text-Based Conversation ===")
    await agent.agent.run("Tell me about artificial intelligence.")
    
    # Example 2: Web search
    print("\n=== Example 2: Web Search ===")
    await agent.search_web("Latest developments in renewable energy")
    
    # Example 3: Synthesize image and web information
    # Uncomment the following lines and provide a path to an image file
    # print("\n=== Example 3: Synthesize Image and Web Information ===")
    # await agent.synthesize_image_and_web(
    #     "path/to/your/image.jpg",
    #     "Information about the subject in the image",
    #     "What can you tell me about this subject based on both the image and web information?"
    # )
    
    # Example 4: Create a report from multiple sources
    print("\n=== Example 4: Create a Report from Multiple Sources ===")
    await agent.create_report_from_multiple_sources(
        "quantum computing",
        image_paths=None,  # Add actual image paths here
        web_queries=["Latest developments in quantum computing", "Applications of quantum computing"],
        report_type="overview"
    )
    
    # Example 5: Interactive conversation
    print("\n=== Example 5: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing and Validation

Let's create a test for our Multimodal Agent:

```python
# tests/test_multimodal_agent.py
import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.multimodal_agent import MultimodalAgent

async def test_multimodal_agent():
    """Test the Multimodal Agent."""
    agent = MultimodalAgent(
        name="TestBot",
        instructions="You are a test assistant that can process text, images, and web data."
    )
    
    # Test text-based conversation
    response = await agent.agent.run("Tell me about artificial intelligence.")
    assert "intelligence" in response.lower(), "Text-based conversation test failed"
    
    # Test web search
    response = await agent.search_web("Python programming language", stream=False)
    assert "python" in response.lower(), "Web search test failed"
    
    print("All tests passed for Multimodal Agent!")

if __name__ == "__main__":
    asyncio.run(test_multimodal_agent())
```

### Common Issues and Troubleshooting

1. **Image Format Support**: Ensure your images are in a supported format (JPEG, PNG, etc.) and are not too large. Very large images might cause processing issues.

2. **Web Search Limitations**: Web search capabilities might be limited by the API's access to current information or by rate limits. If you're getting outdated information, it might be due to the search index not being fully up to date.

3. **Image Analysis Accuracy**: The accuracy of image analysis depends on the quality of the image and the complexity of the content. For best results, use clear, high-quality images with distinct subjects.

4. **Synthesis Quality**: The quality of synthesized information depends on the relevance and quality of the source information. Ensure your web queries are specific and relevant to the image content.

### Variations and Extensions

1. **Specialized Image Analysis Agent**: Create an agent specialized for analyzing specific types of images:

```python
class MedicalImageAnalysisAgent(MultimodalAgent):
    def __init__(self, name="MedImageBot", instructions="You are a helpful assistant specialized in analyzing medical images."):
        super().__init__(name, instructions)
    
    async def analyze_medical_image(self, image_path, image_type=None, question=None):
        """
        Analyze a medical image.
        
        Args:
            image_path (str): The path to the medical image file.
            image_type (str, optional): The type of medical image (X-ray, MRI, CT scan, etc.).
            question (str, optional): A specific question about the image.
            
        Returns:
            str: The analysis results.
        """
        if image_type:
            prompt = f"Please analyze this {image_type} medical image."
        else:
            prompt = "Please analyze this medical image."
        
        if question:
            prompt += f" Specifically, {question}"
        
        return await self.analyze_image(image_path, prompt)
```

2. **News Analysis Agent**: Create an agent specialized for analyzing news and current events:

```python
class NewsAnalysisAgent(MultimodalAgent):
    def __init__(self, name="NewsBot", instructions="You are a helpful assistant specialized in analyzing news and current events."):
        super().__init__(name, instructions)
    
    async def analyze_news_topic(self, topic, perspective="neutral"):
        """
        Analyze a news topic from different perspectives.
        
        Args:
            topic (str): The news topic to analyze.
            perspective (str): The perspective for analysis (neutral, positive, negative).
            
        Returns:
            str: The analysis results.
        """
        query = f"Analyze the news about {topic} from a {perspective} perspective."
        return await self.search_web(query)
    
    async def compare_news_sources(self, topic, sources):
        """
        Compare how different news sources cover a topic.
        
        Args:
            topic (str): The news topic.
            sources (list): List of news sources to compare.
            
        Returns:
            str: The comparison results.
        """
        query = f"Compare how {', '.join(sources)} cover the news about {topic}."
        return await self.search_web(query)
```

3. **Educational Content Agent**: Create an agent specialized for creating educational content:

```python
class EducationalContentAgent(MultimodalAgent):
    def __init__(self, name="EduBot", instructions="You are a helpful assistant specialized in creating educational content."):
        super().__init__(name, instructions)
    
    async def create_lesson_plan(self, topic, grade_level, duration):
        """
        Create a lesson plan for a specific topic.
        
        Args:
            topic (str): The topic for the lesson plan.
            grade_level (str): The grade level for the lesson.
            duration (str): The duration of the lesson.
            
        Returns:
            str: The lesson plan.
        """
        query = f"Create a {duration} lesson plan on {topic} for {grade_level} students."
        return await self.search_web(query)
    
    async def create_visual_explanation(self, topic, image_path):
        """
        Create a visual explanation of a topic using an image.
        
        Args:
            topic (str): The topic to explain.
            image_path (str): The path to an image related to the topic.
            
        Returns:
            str: The visual explanation.
        """
        image_analysis = await self.analyze_image(image_path, f"Analyze this image in the context of {topic}.")
        web_info = await self.search_web(f"Key concepts about {topic}")
        
        synthesis_query = f"Based on the image analysis '{image_analysis}' and the web information '{web_info}', create a visual explanation of {topic} suitable for educational purposes."
        
        return await self.agent.run(synthesis_query)
```

With this Multimodal Agent, you can now create AI assistants that can process and understand multiple types of data, including text, images, and web data, providing more comprehensive and context-aware responses. In the next section, we'll explore advanced topics and best practices for working with the Microsoft Agent Framework.

<a name="advanced"></a>
## 8. Advanced Topics and Best Practices

### Thread Management Across Sessions

Thread management is crucial for maintaining conversation context across multiple interactions. The Microsoft Agent Framework provides robust thread management capabilities that allow agents to remember previous conversations and maintain context.

Let's explore advanced thread management techniques:

```python
# agents/advanced_thread_management.py
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from agent_framework import ChatAgent, AgentThread, ChatMessageStore
from agent_framework.openai import OpenAIChatClient

class AdvancedThreadManager:
    def __init__(self, storage_path="thread_storage"):
        """
        Initialize the Advanced Thread Manager.
        
        Args:
            storage_path (str): The path to store thread data.
        """
        self.storage_path = storage_path
        self.threads = {}  # In-memory thread storage
        self.active_sessions = {}  # Active user sessions
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
    
    def create_thread(self, user_id, thread_name=None):
        """
        Create a new thread for a user.
        
        Args:
            user_id (str): The ID of the user.
            thread_name (str, optional): The name of the thread.
            
        Returns:
            str: The ID of the created thread.
        """
        thread_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        thread_name = thread_name or f"Thread {thread_id}"
        
        # Store thread information
        self.threads[thread_id] = {
            "user_id": user_id,
            "name": thread_name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": 0
        }
        
        # Save to disk
        self.save_thread_to_disk(thread_id)
        
        return thread_id
    
    def get_thread(self, thread_id):
        """
        Get thread information.
        
        Args:
            thread_id (str): The ID of the thread.
            
        Returns:
            dict: The thread information.
        """
        if thread_id in self.threads:
            return self.threads[thread_id]
        else:
            # Try to load from disk
            return self.load_thread_from_disk(thread_id)
    
    def update_thread(self, thread_id, message_count_increment=1):
        """
        Update thread information.
        
        Args:
            thread_id (str): The ID of the thread.
            message_count_increment (int): The number of messages to add to the count.
        """
        if thread_id in self.threads:
            self.threads[thread_id]["last_updated"] = datetime.now().isoformat()
            self.threads[thread_id]["message_count"] += message_count_increment
            
            # Save to disk
            self.save_thread_to_disk(thread_id)
    
    def list_user_threads(self, user_id):
        """
        List all threads for a user.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            list: List of thread information.
        """
        user_threads = []
        
        # Check in-memory threads
        for thread_id, thread_info in self.threads.items():
            if thread_info["user_id"] == user_id:
                user_threads.append({"id": thread_id, **thread_info})
        
        # Check disk threads
        for filename in os.listdir(self.storage_path):
            if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                thread_id = filename[:-5]  # Remove .json extension
                if thread_id not in self.threads:
                    thread_info = self.load_thread_from_disk(thread_id)
                    if thread_info:
                        user_threads.append({"id": thread_id, **thread_info})
        
        # Sort by last updated
        user_threads.sort(key=lambda x: x["last_updated"], reverse=True)
        
        return user_threads
    
    def save_thread_to_disk(self, thread_id):
        """
        Save thread information to disk.
        
        Args:
            thread_id (str): The ID of the thread.
        """
        if thread_id in self.threads:
            with open(os.path.join(self.storage_path, f"{thread_id}.json"), 'w') as f:
                json.dump(self.threads[thread_id], f)
    
    def load_thread_from_disk(self, thread_id):
        """
        Load thread information from disk.
        
        Args:
            thread_id (str): The ID of the thread.
            
        Returns:
            dict or None: The thread information if found, None otherwise.
        """
        try:
            with open(os.path.join(self.storage_path, f"{thread_id}.json"), 'r') as f:
                thread_info = json.load(f)
                self.threads[thread_id] = thread_info
                return thread_info
        except FileNotFoundError:
            return None
    
    def delete_thread(self, thread_id):
        """
        Delete a thread.
        
        Args:
            thread_id (str): The ID of the thread.
        """
        if thread_id in self.threads:
            del self.threads[thread_id]
        
        # Delete from disk
        try:
            os.remove(os.path.join(self.storage_path, f"{thread_id}.json"))
        except FileNotFoundError:
            pass
    
    def start_session(self, user_id, thread_id=None):
        """
        Start a session for a user.
        
        Args:
            user_id (str): The ID of the user.
            thread_id (str, optional): The ID of the thread to use. If None, creates a new thread.
            
        Returns:
            str: The ID of the thread for the session.
        """
        if thread_id is None:
            thread_id = self.create_thread(user_id)
        elif not self.get_thread(thread_id):
            thread_id = self.create_thread(user_id)
        
        self.active_sessions[user_id] = thread_id
        return thread_id
    
    def end_session(self, user_id):
        """
        End a session for a user.
        
        Args:
            user_id (str): The ID of the user.
        """
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
    
    def get_active_thread(self, user_id):
        """
        Get the active thread for a user.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            str or None: The ID of the active thread if found, None otherwise.
        """
        return self.active_sessions.get(user_id)

class PersistentConversationAgent:
    def __init__(self, name="PersistentBot", instructions="You are a helpful assistant with persistent memory."):
        """
        Initialize the Persistent Conversation Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.thread_manager = AdvancedThreadManager()
        self.agent_threads = {}  # Map thread_id to AgentThread
    
    def get_agent_thread(self, thread_id):
        """
        Get or create an AgentThread for a thread ID.
        
        Args:
            thread_id (str): The ID of the thread.
            
        Returns:
            AgentThread: The AgentThread instance.
        """
        if thread_id not in self.agent_threads:
            # Create a new agent thread
            agent = ChatAgent(
                chat_client=self.client,
                instructions=self.instructions,
            )
            self.agent_threads[thread_id] = agent.get_new_thread()
        
        return self.agent_threads[thread_id]
    
    async def start_conversation(self, user_id, thread_id=None):
        """
        Start a conversation with a user.
        
        Args:
            user_id (str): The ID of the user.
            thread_id (str, optional): The ID of the thread to use. If None, creates a new thread.
            
        Returns:
            str: The ID of the thread for the conversation.
        """
        thread_id = self.thread_manager.start_session(user_id, thread_id)
        return thread_id
    
    async def chat(self, user_id, message, thread_id=None):
        """
        Chat with a user.
        
        Args:
            user_id (str): The ID of the user.
            message (str): The message from the user.
            thread_id (str, optional): The ID of the thread to use. If None, uses the active thread.
            
        Returns:
            str: The agent's response.
        """
        # Determine which thread to use
        if thread_id is None:
            thread_id = self.thread_manager.get_active_thread(user_id)
            if thread_id is None:
                thread_id = await self.start_conversation(user_id)
        
        # Get the agent thread
        agent_thread = self.get_agent_thread(thread_id)
        
        # Get the agent
        agent = ChatAgent(
            chat_client=self.client,
            instructions=self.instructions,
        )
        
        # Send the message and get the response
        response = await agent.run(message, thread=agent_thread)
        
        # Update thread information
        self.thread_manager.update_thread(thread_id)
        
        return response
    
    async def list_threads(self, user_id):
        """
        List all threads for a user.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            list: List of thread information.
        """
        return self.thread_manager.list_user_threads(user_id)
    
    async def switch_thread(self, user_id, thread_id):
        """
        Switch to a different thread.
        
        Args:
            user_id (str): The ID of the user.
            thread_id (str): The ID of the thread to switch to.
            
        Returns:
            bool: True if the switch was successful, False otherwise.
        """
        if self.thread_manager.get_thread(thread_id):
            self.thread_manager.active_sessions[user_id] = thread_id
            return True
        return False
    
    async def delete_thread(self, user_id, thread_id):
        """
        Delete a thread.
        
        Args:
            user_id (str): The ID of the user.
            thread_id (str): The ID of the thread to delete.
        """
        self.thread_manager.delete_thread(thread_id)
        
        # Remove from agent threads if it exists
        if thread_id in self.agent_threads:
            del self.agent_threads[thread_id]
        
        # If this was the active thread, end the session
        if self.thread_manager.get_active_thread(user_id) == thread_id:
            self.thread_manager.end_session(user_id)

async def main():
    # Create a persistent conversation agent
    agent = PersistentConversationAgent(
        name="PersistentBot",
        instructions="You are a helpful assistant with persistent memory. Remember details from our conversations."
    )
    
    # Example 1: Start a conversation
    print("=== Example 1: Starting a Conversation ===")
    user_id = "user123"
    thread_id = await agent.start_conversation(user_id)
    print(f"Started conversation with thread ID: {thread_id}")
    
    # Example 2: Chat with the agent
    print("\n=== Example 2: Chatting with the Agent ===")
    response = await agent.chat(user_id, "My name is Alex and I love hiking.")
    print(f"Agent: {response}")
    
    response = await agent.chat(user_id, "What's my name?")
    print(f"Agent: {response}")
    
    # Example 3: List threads
    print("\n=== Example 3: Listing Threads ===")
    threads = await agent.list_threads(user_id)
    print(f"Threads for {user_id}:")
    for thread in threads:
        print(f"- {thread['name']} (ID: {thread['id']}, Messages: {thread['message_count']})")
    
    # Example 4: Create a new thread
    print("\n=== Example 4: Creating a New Thread ===")
    new_thread_id = await agent.start_conversation(user_id)
    print(f"Started new conversation with thread ID: {new_thread_id}")
    
    response = await agent.chat(user_id, "What's my name?")
    print(f"Agent: {response}")
    
    # Example 5: Switch back to the original thread
    print("\n=== Example 5: Switching Back to the Original Thread ===")
    success = await agent.switch_thread(user_id, thread_id)
    if success:
        print(f"Switched to thread {thread_id}")
        response = await agent.chat(user_id, "What's my name?")
        print(f"Agent: {response}")
    else:
        print(f"Failed to switch to thread {thread_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Structured Outputs

Structured outputs ensure that responses from the AI model follow predefined schemas, making it easier to parse and use the data. This is particularly useful when you need to integrate the agent's responses into other systems or when you need consistent data formats.

Let's explore how to use structured outputs with the Microsoft Agent Framework:

```python
# agents/structured_output_agent.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient

# Define structured output models
class WeatherInfo(BaseModel):
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    condition: str = Field(description="The weather condition (e.g., sunny, cloudy, rainy)")
    humidity: float = Field(description="The humidity percentage")
    wind_speed: float = Field(description="The wind speed in km/h")

class EventInfo(BaseModel):
    title: str = Field(description="The title of the event")
    date: str = Field(description="The date of the event in YYYY-MM-DD format")
    time: str = Field(description="The time of the event in HH:MM format")
    location: str = Field(description="The location of the event")
    description: str = Field(description="A description of the event")

class TaskInfo(BaseModel):
    title: str = Field(description="The title of the task")
    priority: str = Field(description="The priority of the task (low, medium, high)")
    due_date: Optional[str] = Field(description="The due date of the task in YYYY-MM-DD format", default=None)
    status: str = Field(description="The status of the task (todo, in_progress, completed)")
    description: str = Field(description="A description of the task")

class StructuredOutputAgent:
    def __init__(self, name="StructuredBot", instructions="You are a helpful assistant that provides structured responses."):
        """
        Initialize the Structured Output Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIResponsesClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
        )
    
    async def get_weather_info(self, location) -> WeatherInfo:
        """
        Get weather information in a structured format.
        
        Args:
            location (str): The location to get weather information for.
            
        Returns:
            WeatherInfo: The weather information.
        """
        # In a real implementation, you would call a weather API here
        # For this example, we'll use the agent to generate structured weather data
        
        prompt = f"Provide current weather information for {location} in a structured format."
        
        # Create a response schema based on the WeatherInfo model
        response_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "humidity": {"type": "number"},
                "wind_speed": {"type": "number"}
            },
            "required": ["location", "temperature", "condition", "humidity", "wind_speed"]
        }
        
        # Get the structured response
        response = await self.client.get_response(
            prompt,
            response_schema=response_schema
        )
        
        # Parse the response into a WeatherInfo object
        weather_data = json.loads(response)
        return WeatherInfo(**weather_data)
    
    async def create_event(self, title, date, time, location, description) -> EventInfo:
        """
        Create an event in a structured format.
        
        Args:
            title (str): The title of the event.
            date (str): The date of the event.
            time (str): The time of the event.
            location (str): The location of the event.
            description (str): The description of the event.
            
        Returns:
            EventInfo: The event information.
        """
        prompt = f"Create an event with the following details: Title: {title}, Date: {date}, Time: {time}, Location: {location}, Description: {description}. Return the information in a structured format."
        
        # Create a response schema based on the EventInfo model
        response_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
                "location": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["title", "date", "time", "location", "description"]
        }
        
        # Get the structured response
        response = await self.client.get_response(
            prompt,
            response_schema=response_schema
        )
        
        # Parse the response into an EventInfo object
        event_data = json.loads(response)
        return EventInfo(**event_data)
    
    async def create_task(self, title, priority, due_date=None, status="todo", description="") -> TaskInfo:
        """
        Create a task in a structured format.
        
        Args:
            title (str): The title of the task.
            priority (str): The priority of the task.
            due_date (str, optional): The due date of the task.
            status (str): The status of the task.
            description (str): The description of the task.
            
        Returns:
            TaskInfo: The task information.
        """
        prompt = f"Create a task with the following details: Title: {title}, Priority: {priority}, Due Date: {due_date or 'Not specified'}, Status: {status}, Description: {description}. Return the information in a structured format."
        
        # Create a response schema based on the TaskInfo model
        response_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string"},
                "due_date": {"type": "string"},
                "status": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["title", "priority", "status", "description"]
        }
        
        # Get the structured response
        response = await self.client.get_response(
            prompt,
            response_schema=response_schema
        )
        
        # Parse the response into a TaskInfo object
        task_data = json.loads(response)
        return TaskInfo(**task_data)
    
    async def extract_structured_info(self, text, output_model):
        """
        Extract structured information from text.
        
        Args:
            text (str): The text to extract information from.
            output_model: The Pydantic model for the output structure.
            
        Returns:
            An instance of the output_model with the extracted information.
        """
        # Get the model name and schema
        model_name = output_model.__name__
        model_schema = output_model.model_json_schema()
        
        prompt = f"Extract information from the following text and return it in a structured format based on the {model_name} model: {text}"
        
        # Get the structured response
        response = await self.client.get_response(
            prompt,
            response_schema=model_schema
        )
        
        # Parse the response into the specified model
        data = json.loads(response)
        return output_model(**data)
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'weather <location>' - Get weather information")
        print("  'event <title> <date> <time> <location> <description>' - Create an event")
        print("  'task <title> <priority> [due_date] [status] [description]' - Create a task")
        print("  'extract <text> <model>' - Extract structured information from text")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower().startswith('weather '):
                location = user_input[8:]
                weather_info = await self.get_weather_info(location)
                print(f"Weather Info: {weather_info.json()}")
            elif user_input.lower().startswith('event '):
                parts = user_input[6:].split(' ', 4)
                if len(parts) >= 4:
                    title, date, time, location = parts[:4]
                    description = parts[4] if len(parts) > 4 else ""
                    event_info = await self.create_event(title, date, time, location, description)
                    print(f"Event Info: {event_info.json()}")
                else:
                    print("Please provide title, date, time, and location. Usage: 'event <title> <date> <time> <location> [description]'")
            elif user_input.lower().startswith('task '):
                parts = user_input[5:].split(' ', 4)
                if len(parts) >= 2:
                    title, priority = parts[:2]
                    due_date = parts[2] if len(parts) > 2 and parts[2] else None
                    status = parts[3] if len(parts) > 3 and parts[3] else "todo"
                    description = parts[4] if len(parts) > 4 else ""
                    task_info = await self.create_task(title, priority, due_date, status, description)
                    print(f"Task Info: {task_info.json()}")
                else:
                    print("Please provide title and priority. Usage: 'task <title> <priority> [due_date] [status] [description]'")
            elif user_input.lower().startswith('extract '):
                parts = user_input[8:].split(' ', 1)
                if len(parts) >= 2:
                    text, model_name = parts
                    if model_name.lower() == "weatherinfo":
                        weather_info = await self.extract_structured_info(text, WeatherInfo)
                        print(f"Weather Info: {weather_info.json()}")
                    elif model_name.lower() == "eventinfo":
                        event_info = await self.extract_structured_info(text, EventInfo)
                        print(f"Event Info: {event_info.json()}")
                    elif model_name.lower() == "taskinfo":
                        task_info = await self.extract_structured_info(text, TaskInfo)
                        print(f"Task Info: {task_info.json()}")
                    else:
                        print(f"Unknown model: {model_name}. Available models: WeatherInfo, EventInfo, TaskInfo")
                else:
                    print("Please provide text and model name. Usage: 'extract <text> <model>'")
            else:
                print("Unknown command. Type 'quit' to exit.")

async def main():
    # Create a structured output agent
    agent = StructuredOutputAgent(
        name="StructuredBot",
        instructions="You are a helpful assistant that provides structured responses. Always ensure your responses follow the specified schema."
    )
    
    # Example 1: Get weather information
    print("=== Example 1: Getting Weather Information ===")
    weather_info = await agent.get_weather_info("New York")
    print(f"Weather Info: {weather_info.json()}")
    
    # Example 2: Create an event
    print("\n=== Example 2: Creating an Event ===")
    event_info = await agent.create_event(
        "Team Meeting",
        "2023-12-15",
        "14:00",
        "Conference Room A",
        "Weekly team sync to discuss project progress"
    )
    print(f"Event Info: {event_info.json()}")
    
    # Example 3: Create a task
    print("\n=== Example 3: Creating a Task ===")
    task_info = await agent.create_task(
        "Prepare presentation",
        "high",
        "2023-12-20",
        "todo",
        "Prepare slides for the quarterly review meeting"
    )
    print(f"Task Info: {task_info.json()}")
    
    # Example 4: Extract structured information from text
    print("\n=== Example 4: Extracting Structured Information ===")
    text = "The weather in London is partly cloudy with a temperature of 15°C, humidity at 65%, and wind speed of 10 km/h."
    extracted_weather = await agent.extract_structured_info(text, WeatherInfo)
    print(f"Extracted Weather Info: {extracted_weather.json()}")
    
    # Example 5: Interactive conversation
    print("\n=== Example 5: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Model Context Protocol (MCP) Integration

The Model Context Protocol (MCP) allows agents to integrate with external services and tools through a standardized protocol. This enables agents to access a wide range of capabilities beyond what's built into the framework.

Let's explore how to integrate MCP with the Microsoft Agent Framework:

```python
# agents/mcp_integration_agent.py
import asyncio
from typing import Dict, List, Any, Optional
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

class MCPIntegrationAgent:
    def __init__(self, name="MCPBot", instructions="You are a helpful assistant with access to external services through MCP."):
        """
        Initialize the MCP Integration Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.mcp_tools = {}
    
    def register_mcp_tool(self, name, url, description=None):
        """
        Register an MCP tool.
        
        Args:
            name (str): The name of the tool.
            url (str): The URL of the MCP service.
            description (str, optional): A description of the tool.
        """
        self.mcp_tools[name] = {
            "url": url,
            "description": description or f"MCP tool for {name}",
            "tool": None  # Will be created when needed
        }
    
    async def get_mcp_tool(self, name):
        """
        Get an MCP tool by name.
        
        Args:
            name (str): The name of the tool.
            
        Returns:
            MCPStreamableHTTPTool: The MCP tool.
        """
        if name not in self.mcp_tools:
            raise ValueError(f"MCP tool '{name}' not registered.")
        
        # Create the tool if it doesn't exist
        if self.mcp_tools[name]["tool"] is None:
            self.mcp_tools[name]["tool"] = MCPStreamableHTTPTool(
                name=name,
                url=self.mcp_tools[name]["url"],
            )
        
        return self.mcp_tools[name]["tool"]
    
    async def chat_with_mcp_tool(self, message, tool_name, stream=True):
        """
        Chat with the agent using a specific MCP tool.
        
        Args:
            message (str): The message to send to the agent.
            tool_name (str): The name of the MCP tool to use.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Get the MCP tool
        mcp_tool = await self.get_mcp_tool(tool_name)
        
        # Create an agent with the MCP tool
        agent = ChatAgent(
            chat_client=self.client,
            instructions=self.instructions,
            tools=mcp_tool,
        )
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in agent.run_stream(message):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await agent.run(message)
            print(f"{self.name}: {response}")
            return response
    
    async def chat_with_multiple_mcp_tools(self, message, tool_names, stream=True):
        """
        Chat with the agent using multiple MCP tools.
        
        Args:
            message (str): The message to send to the agent.
            tool_names (list): The names of the MCP tools to use.
            stream (bool): Whether to stream the response or get it all at once.
            
        Returns:
            str: The agent's response.
        """
        # Get the MCP tools
        mcp_tools = []
        for tool_name in tool_names:
            mcp_tool = await self.get_mcp_tool(tool_name)
            mcp_tools.append(mcp_tool)
        
        # Create an agent with the MCP tools
        agent = ChatAgent(
            chat_client=self.client,
            instructions=self.instructions,
            tools=mcp_tools,
        )
        
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in agent.run_stream(message):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
            return response
        else:
            response = await agent.run(message)
            print(f"{self.name}: {response}")
            return response
    
    async def list_mcp_tools(self):
        """
        List all registered MCP tools.
        
        Returns:
            dict: Dictionary of MCP tool information.
        """
        return {
            name: {
                "url": info["url"],
                "description": info["description"]
            }
            for name, info in self.mcp_tools.items()
        }
    
    async def start_conversation(self):
        """
        Start an interactive conversation with the agent.
        """
        print(f"Starting conversation with {self.name}. Type 'quit' to exit.")
        print("Commands:")
        print("  'list' - List all MCP tools")
        print("  'use <tool_name> <message>' - Use a specific MCP tool")
        print("  'use_multi <tool1,tool2,...> <message>' - Use multiple MCP tools")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"Goodbye!")
                break
            elif user_input.lower() == 'list':
                tools = await self.list_mcp_tools()
                if tools:
                    print("Available MCP tools:")
                    for name, info in tools.items():
                        print(f"- {name}: {info['description']} (URL: {info['url']})")
                else:
                    print("No MCP tools registered.")
            elif user_input.lower().startswith('use '):
                parts = user_input[4:].split(' ', 1)
                if len(parts) >= 2:
                    tool_spec, message = parts
                    if ',' in tool_spec:
                        # Multiple tools
                        tool_names = [name.strip() for name in tool_spec.split(',')]
                        await self.chat_with_multiple_mcp_tools(message, tool_names)
                    else:
                        # Single tool
                        await self.chat_with_mcp_tool(message, tool_spec)
                else:
                    print("Please provide tool name and message. Usage: 'use <tool_name> <message>'")
            else:
                print("Unknown command. Type 'quit' to exit.")

async def main():
    # Create an MCP integration agent
    agent = MCPIntegrationAgent(
        name="MCPBot",
        instructions="You are a helpful assistant with access to external services through MCP. Use the appropriate tools to answer questions."
    )
    
    # Register MCP tools
    agent.register_mcp_tool(
        "Microsoft Learn",
        "https://learn.microsoft.com/api/mcp",
        "Access to Microsoft Learn documentation"
    )
    
    agent.register_mcp_tool(
        "GitHub",
        "https://api.github.com/mcp",
        "Access to GitHub repositories and information"
    )
    
    # Example 1: List MCP tools
    print("=== Example 1: Listing MCP Tools ===")
    tools = await agent.list_mcp_tools()
    print(json.dumps(tools, indent=2))
    
    # Example 2: Use a single MCP tool
    print("\n=== Example 2: Using a Single MCP Tool ===")
    await agent.chat_with_mcp_tool(
        "How do I create an Azure storage account using Azure CLI?",
        "Microsoft Learn"
    )
    
    # Example 3: Use multiple MCP tools
    print("\n=== Example 3: Using Multiple MCP Tools ===")
    await agent.chat_with_multiple_mcp_tools(
        "Find information about deploying a Python web application to Azure.",
        ["Microsoft Learn", "GitHub"]
    )
    
    # Example 4: Interactive conversation
    print("\n=== Example 4: Interactive Conversation ===")
    # Uncomment the line below to start an interactive conversation
    # await agent.start_conversation()

if __name__ == "__main__":
    asyncio.run(main())
```

### Performance Optimization

Optimizing the performance of your AI agents is crucial for providing a good user experience, especially in production environments. Let's explore some techniques for optimizing agent performance:

```python
# agents/performance_optimized_agent.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

class PerformanceOptimizedAgent:
    def __init__(self, name="OptimizedBot", instructions="You are a helpful assistant."):
        """
        Initialize the Performance Optimized Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
        )
        
        # Performance optimization settings
        self.cache_enabled = True
        self.cache_size = 100
        self.streaming_enabled = True
        self.batch_processing_enabled = False
        self.max_concurrent_requests = 5
        
        # Performance metrics
        self.request_count = 0
        self.total_response_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    @lru_cache(maxsize=100)
    def _cached_response(self, query_hash):
        """
        Get a cached response for a query.
        
        Args:
            query_hash (str): The hash of the query.
            
        Returns:
            str or None: The cached response if available, None otherwise.
        """
        # This is a placeholder for a real caching implementation
        # In a real implementation, you would use a proper caching system like Redis
        return None
    
    def _cache_response(self, query_hash, response):
        """
        Cache a response for a query.
        
        Args:
            query_hash (str): The hash of the query.
            response (str): The response to cache.
        """
        # This is a placeholder for a real caching implementation
        # In a real implementation, you would use a proper caching system like Redis
        pass
    
    def _hash_query(self, query):
        """
        Generate a hash for a query.
        
        Args:
            query (str): The query to hash.
            
        Returns:
            str: The hash of the query.
        """
        # This is a simple hash function for demonstration
        # In a real implementation, you would use a proper hashing algorithm
        return str(hash(query))
    
    async def chat(self, message, stream=None, use_cache=None):
        """
        Chat with the agent with performance optimizations.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool, optional): Whether to stream the response. If None, uses the default setting.
            use_cache (bool, optional): Whether to use caching. If None, uses the default setting.
            
        Returns:
            str: The agent's response.
        """
        # Use default settings if not specified
        if stream is None:
            stream = self.streaming_enabled
        if use_cache is None:
            use_cache = self.cache_enabled
        
        # Check cache if enabled
        query_hash = None
        if use_cache:
            query_hash = self._hash_query(message)
            cached_response = self._cached_response(query_hash)
            if cached_response:
                self.cache_hits += 1
                print(f"{self.name}: {cached_response} (from cache)")
                return cached_response
            else:
                self.cache_misses += 1
        
        # Measure response time
        start_time = time.time()
        
        # Get the response
        if stream:
            response = ""
            print(f"{self.name}: ", end="", flush=True)
            async for chunk in self.agent.run_stream(message):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response += chunk.text
            print()  # New line after the complete response
        else:
            response = await self.agent.run(message)
            print(f"{self.name}: {response}")
        
        # Update performance metrics
        response_time = time.time() - start_time
        self.request_count += 1
        self.total_response_time += response_time
        
        # Cache the response if enabled
        if use_cache and query_hash:
            self._cache_response(query_hash, response)
        
        return response
    
    async def batch_chat(self, messages, use_cache=None):
        """
        Process multiple messages in batch.
        
        Args:
            messages (list): List of messages to process.
            use_cache (bool, optional): Whether to use caching. If None, uses the default setting.
            
        Returns:
            list: List of responses.
        """
        if not self.batch_processing_enabled:
            # Process messages sequentially
            responses = []
            for message in messages:
                response = await self.chat(message, stream=False, use_cache=use_cache)
                responses.append(response)
            return responses
        
        # Process messages concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_message(message):
            async with semaphore:
                return await self.chat(message, stream=False, use_cache=use_cache)
        
        tasks = [process_message(message) for message in messages]
        return await asyncio.gather(*tasks)
    
    def get_performance_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            dict: Performance metrics.
        """
        avg_response_time = self.total_response_time / self.request_count if self.request_count > 0 else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "request_count": self.request_count,
            "average_response_time": avg_response_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": self.cache_enabled,
            "streaming_enabled": self.streaming_enabled,
            "batch_processing_enabled": self.batch_processing_enabled,
            "max_concurrent_requests": self.max_concurrent_requests
        }
    
    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.request_count = 0
        self.total_response_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def configure_performance(self, cache_enabled=None, cache_size=None, streaming_enabled=None, 
                             batch_processing_enabled=None, max_concurrent_requests=None):
        """
        Configure performance settings.
        
        Args:
            cache_enabled (bool, optional): Whether to enable caching.
            cache_size (int, optional): The size of the cache.
            streaming_enabled (bool, optional): Whether to enable streaming by default.
            batch_processing_enabled (bool, optional): Whether to enable batch processing.
            max_concurrent_requests (int, optional): The maximum number of concurrent requests.
        """
        if cache_enabled is not None:
            self.cache_enabled = cache_enabled
        
        if cache_size is not None:
            self.cache_size = cache_size
            # Update the LRU cache size
            self._cached_response = lru_cache(maxsize=cache_size)(self._cached_response.__wrapped__)
        
        if streaming_enabled is not None:
            self.streaming_enabled = streaming_enabled
        
        if batch_processing_enabled is not None:
            self.batch_processing_enabled = batch_processing_enabled
        
        if max_concurrent_requests is not None:
            self.max_concurrent_requests = max_concurrent_requests

async def main():
    # Create a performance optimized agent
    agent = PerformanceOptimizedAgent(
        name="OptimizedBot",
        instructions="You are a helpful assistant optimized for performance."
    )
    
    # Configure performance settings
    agent.configure_performance(
        cache_enabled=True,
        cache_size=50,
        streaming_enabled=True,
        batch_processing_enabled=True,
        max_concurrent_requests=3
    )
    
    # Example 1: Single chat with caching
    print("=== Example 1: Single Chat with Caching ===")
    response1 = await agent.chat("What is the capital of France?")
    response2 = await agent.chat("What is the capital of France?")  # Should be from cache
    
    # Example 2: Batch processing
    print("\n=== Example 2: Batch Processing ===")
    messages = [
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
        "What is the capital of Greece?"
    ]
    responses = await agent.batch_chat(messages)
    
    # Example 3: Performance metrics
    print("\n=== Example 3: Performance Metrics ===")
    metrics = agent.get_performance_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Example 4: Performance comparison
    print("\n=== Example 4: Performance Comparison ===")
    
    # Test with caching enabled
    agent.reset_performance_metrics()
    start_time = time.time()
    await agent.chat("What is the capital of Japan?")
    await agent.chat("What is the capital of Japan?")  # Should be from cache
    cache_time = time.time() - start_time
    cache_metrics = agent.get_performance_metrics()
    
    # Test with caching disabled
    agent.configure_performance(cache_enabled=False)
    agent.reset_performance_metrics()
    start_time = time.time()
    await agent.chat("What is the capital of Japan?")
    await agent.chat("What is the capital of Japan?")
    no_cache_time = time.time() - start_time
    no_cache_metrics = agent.get_performance_metrics()
    
    print(f"With caching: {cache_time:.2f}s")
    print(f"Without caching: {no_cache_time:.2f}s")
    print(f"Performance improvement: {(no_cache_time - cache_time) / no_cache_time * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling and Troubleshooting

Robust error handling is essential for creating reliable AI agents. Let's explore best practices for error handling and troubleshooting:

```python
# agents/error_handling_agent.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandlingAgent:
    def __init__(self, name="ErrorBot", instructions="You are a helpful assistant."):
        """
        Initialize the Error Handling Agent.
        
        Args:
            name (str): The name of the agent.
            instructions (str): Instructions that guide the agent's behavior.
        """
        self.name = name
        self.instructions = instructions
        self.client = OpenAIChatClient()
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=instructions,
        )
        
        # Error handling settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.fallback_enabled = True
        self.error_logging_enabled = True
        
        # Error statistics
        self.error_count = 0
        self.retry_count = 0
        self.fallback_count = 0
    
    async def chat_with_retry(self, message, stream=True):
        """
        Chat with the agent with retry logic.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool): Whether to stream the response.
            
        Returns:
            str: The agent's response.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.retry_count += 1
                    logger.info(f"Retrying request (attempt {attempt + 1}/{self.max_retries + 1})")
                    # Exponential backoff
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                
                if stream:
                    response = ""
                    print(f"{self.name}: ", end="", flush=True)
                    async for chunk in self.agent.run_stream(message):
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            response += chunk.text
                    print()  # New line after the complete response
                    return response
                else:
                    response = await self.agent.run(message)
                    print(f"{self.name}: {response}")
                    return response
            
            except Exception as e:
                last_error = e
                self.error_count += 1
                
                if self.error_logging_enabled:
                    logger.error(f"Error in chat_with_retry (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries:
                    continue
                else:
                    # All retries failed, try fallback if enabled
                    if self.fallback_enabled:
                        return await self.fallback_response(message, str(e))
                    else:
                        raise e
    
    async def fallback_response(self, message, error):
        """
        Provide a fallback response when all retries fail.
        
        Args:
            message (str): The original message.
            error (str): The error that occurred.
            
        Returns:
            str: The fallback response.
        """
        self.fallback_count += 1
        logger.warning(f"Using fallback response for message: {message}")
        
        fallback_messages = [
            "I'm sorry, I'm having trouble processing your request right now. Please try again later.",
            "I apologize, but I encountered an error while trying to respond. Could you please rephrase your question?",
            "I'm experiencing some technical difficulties. Please try again in a few moments.",
            "Sorry, I couldn't process your request. Please try again or contact support if the issue persists."
        ]
        
        # Select a fallback message based on the error type
        if "rate limit" in error.lower():
            return "I'm receiving too many requests right now. Please wait a moment and try again."
        elif "timeout" in error.lower():
            return "I'm taking longer than expected to respond. Please try again with a shorter message."
        elif "connection" in error.lower():
            return "I'm having trouble connecting to my services. Please check your internet connection and try again."
        else:
            # Use a generic fallback message
            import random
            return random.choice(fallback_messages)
    
    async def safe_chat(self, message, stream=True, timeout=30):
        """
        Chat with the agent with additional safety measures.
        
        Args:
            message (str): The message to send to the agent.
            stream (bool): Whether to stream the response.
            timeout (int): The timeout in seconds.
            
        Returns:
            str: The agent's response.
        """
        try:
            # Add timeout to prevent hanging
            return await asyncio.wait_for(
                self.chat_with_retry(message, stream),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.error_count += 1
            if self.error_logging_enabled:
                logger.error(f"Timeout in safe_chat for message: {message}")
            
            if self.fallback_enabled:
                return await self.fallback_response(message, "Request timed out")
            else:
                raise Exception("Request timed out")
    
    def get_error_statistics(self):
        """
        Get error statistics.
        
        Returns:
            dict: Error statistics.
        """
        return {
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "fallback_enabled": self.fallback_enabled,
            "error_logging_enabled": self.error_logging_enabled
        }
    
    def reset_error_statistics(self):
        """Reset error statistics."""
        self.error_count = 0
        self.retry_count = 0
        self.fallback_count = 0
    
    def configure_error_handling(self, max_retries=None, retry_delay=None, 
                               fallback_enabled=None, error_logging_enabled=None):
        """
        Configure error handling settings.
        
        Args:
            max_retries (int, optional): The maximum number of retries.
            retry_delay (int, optional): The delay between retries in seconds.
            fallback_enabled (bool, optional): Whether to enable fallback responses.
            error_logging_enabled (bool, optional): Whether to enable error logging.
        """
        if max_retries is not None:
            self.max_retries = max_retries
        
        if retry_delay is not None:
            self.retry_delay = retry_delay
        
        if fallback_enabled is not None:
            self.fallback_enabled = fallback_enabled
        
        if error_logging_enabled is not None:
            self.error_logging_enabled = error_logging_enabled

async def main():
    # Create an error handling agent
    agent = ErrorHandlingAgent(
        name="ErrorBot",
        instructions="You are a helpful assistant with robust error handling."
    )
    
    # Configure error handling settings
    agent.configure_error_handling(
        max_retries=3,
        retry_delay=1,
        fallback_enabled=True,
        error_logging_enabled=True
    )
    
    # Example 1: Normal chat
    print("=== Example 1: Normal Chat ===")
    await agent.safe_chat("Tell me a joke.")
    
    # Example 2: Simulate an error (this is just for demonstration)
    print("\n=== Example 2: Simulated Error Handling ===")
    # In a real scenario, you might encounter errors due to network issues, API limits, etc.
    # For this example, we'll just show the error statistics
    error_stats = agent.get_error_statistics()
    print(json.dumps(error_stats, indent=2))
    
    # Example 3: Timeout handling
    print("\n=== Example 3: Timeout Handling ===")
    # Set a very short timeout to demonstrate timeout handling
    try:
        await agent.safe_chat("Write a detailed essay on the history of artificial intelligence.", timeout=0.001)
    except Exception as e:
        print(f"Caught exception: {str(e)}")
    
    # Example 4: Error statistics after various operations
    print("\n=== Example 4: Error Statistics ===")
    error_stats = agent.get_error_statistics()
    print(json.dumps(error_stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

<a name="conclusion"></a>
## 9. Conclusion and Next Steps

### Summary of Key Concepts

In this comprehensive guide, we've explored the Microsoft Agent Framework and learned how to build five different types of AI agents:

1. **Basic Conversational Agent**: A simple chatbot for basic interactions, demonstrating the fundamentals of the framework.

2. **Function-Calling Agent**: An agent that can call external functions and APIs, extending its capabilities beyond text generation.

3. **Code Execution Agent**: An agent that can write and execute code, useful for mathematical problem solving and data analysis.

4. **Knowledge Retrieval Agent**: An agent that can search through documents and answer questions based on their content.

5. **Multimodal Agent**: An agent that can process text, images, and web data, providing more comprehensive and context-aware responses.

We've also covered advanced topics like thread management, structured outputs, MCP integration, performance optimization, and error handling, providing you with the knowledge to build robust and efficient AI agents.

### Additional Resources

To continue your journey with the Microsoft Agent Framework, here are some additional resources:

1. **Official Documentation**: The official documentation for the Microsoft Agent Framework provides in-depth information about all features and capabilities.

2. **Community Forums**: Join the community forums to ask questions, share your projects, and learn from other developers.

3. **Sample Projects**: Explore the sample projects included with the framework to see how different features are implemented in real-world scenarios.

4. **GitHub Repository**: Check out the GitHub repository for the latest updates, bug fixes, and feature requests.

### Best Practices

When building AI agents with the Microsoft Agent Framework, keep these best practices in mind:

1. **Clear Instructions**: Provide clear and specific instructions to guide the agent's behavior.

2. **Error Handling**: Implement robust error handling to ensure your agents are reliable and user-friendly.

3. **Performance Optimization**: Use caching, streaming, and other optimization techniques to improve performance.

4. **Security**: Be mindful of security when dealing with user inputs, code execution, and external integrations.

5. **Testing**: Thoroughly test your agents to ensure they behave as expected in various scenarios.

6. **Monitoring**: Monitor your agents in production to identify and address issues promptly.

### Future Developments

The field of AI is rapidly evolving, and the Microsoft Agent Framework is continuously being improved. Keep an eye out for:

1. **New Model Capabilities**: As AI models become more powerful, the framework will likely support new capabilities.

2. **Enhanced Tool Integration**: Expect more tools and integrations to be added to the framework.

3. **Improved Performance**: Ongoing optimizations will make the framework faster and more efficient.

4. **Better Developer Experience**: The framework will likely continue to improve in terms of ease of use and developer productivity.

### Final Thoughts

The Microsoft Agent Framework provides a powerful and flexible platform for building AI agents. By understanding its components and capabilities, you can create agents that solve real-world problems and provide value to users.

Remember that building AI agents is an iterative process. Start with simple agents and gradually add more complexity as you become more familiar with the framework. Don't be afraid to experiment and try new approaches.

We hope this guide has been helpful in getting you started with the Microsoft Agent Framework. Happy coding, and we look forward to seeing the amazing agents you create!

---

*This guide was created to help developers understand and use the Microsoft Agent Framework. If you have any feedback or suggestions for improvement, please let us know.*

https://chat.z.ai/s/935ed3a1-2673-4d6e-b07a-47a22ca0f078