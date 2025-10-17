import asyncio


# Stubs for client classes to allow syntax checks
class OpenAIAssistantsClient:
    class client:
        class files:
            @staticmethod
            async def create(file, purpose=None):
                class F:
                    id = "file-id"

                return F()

        class vector_stores:
            @staticmethod
            async def create(name=None, expires_after=None):
                class V:
                    id = "vector-id"

                return V()

            class files:
                @staticmethod
                async def create_and_poll(vector_store_id=None, file_id=None):
                    class R:
                        status = "completed"
                        last_error = None

                    return R()


async def create_knowledge_base(client: OpenAIAssistantsClient):
    """Create a fake vector store to validate syntax."""
    documents = [("company_policy.txt", b"Policy...")]
    file_ids = []
    for filename, content in documents:
        file = await client.client.files.create(
            file=(filename, content),
            purpose="user_data",
        )
        file_ids.append(file.id)

    vector_store = await client.client.vector_stores.create(
        name="company_knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 7},
    )

    for file_id in file_ids:
        result = await client.client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file_id,
        )

        if result.status != "completed":
            raise Exception("Indexing failed")
    return file_ids, vector_store


async def main():
    client = OpenAIAssistantsClient()
    file_ids, vector_store = await create_knowledge_base(client)
    print(file_ids, getattr(vector_store, "id", None))


if __name__ == "__main__":
    asyncio.run(main())
