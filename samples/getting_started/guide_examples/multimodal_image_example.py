import asyncio

# Minimal stubs to emulate ChatMessage/UriContent behavior used in examples
class ChatMessage:
    def __init__(self, role=None, contents=None):
        self.role = role
        self.contents = contents or []

class TextContent:
    def __init__(self, text):
        self.text = text

class UriContent:
    def __init__(self, uri, media_type=None):
        self.uri = uri
        self.media_type = media_type

class OpenAIResponsesClient:
    def create_agent(self, *args, **kwargs):
        return self
    async def run(self, user_message):
        class R:
            text = "stub-analysis"
        return R()

async def analyze_image_example():
    agent = OpenAIResponsesClient().create_agent(name="VisionAnalyst", instructions="You are a professional image analyst.")

    user_message = ChatMessage(
        role="user",
        contents=[TextContent(text="What do you see in this image?"), UriContent(uri="https://example.com/image.jpg", media_type="image/jpeg")],
    )

    result = await agent.run(user_message)
    print(result.text)

if __name__ == "__main__":
    asyncio.run(analyze_image_example())
