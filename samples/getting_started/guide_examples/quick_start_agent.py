import asyncio
from random import randint
from typing import Annotated

# Import paths adjusted for local syntax-only validation
# from agent_framework.openai import OpenAIChatClient
# Using a lightweight Field stub to keep annotations functional without pydantic
class Field:
    def __init__(self, description: str = ""):
        self.description = description

# Minimal stub for OpenAIChatClient.create_agent to allow syntax checks
class OpenAIChatClient:
    def create_agent(self, *args, **kwargs):
        return self
    async def run(self, query, **kwargs):
        return "stub-response"


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for")],
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
