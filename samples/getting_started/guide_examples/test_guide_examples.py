import asyncio
import sys
import pathlib
import pytest

# Ensure the guide_examples directory is importable during pytest collection
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import code_interpreter_helper
import quick_start_agent
import rag_agent_setup
import multimodal_image_example

def test_code_interpreter_helper_syntax():
    # Ensure function exists and returns None for invalid input
    assert hasattr(code_interpreter_helper, 'get_code_interpreter_chunk')
    assert code_interpreter_helper.get_code_interpreter_chunk(object()) is None


@pytest.mark.asyncio
async def test_quick_start_agent_runs():
    # Run quick start main to ensure no syntax/runtime errors for the stub
    await quick_start_agent.main()


@pytest.mark.asyncio
async def test_rag_agent_setup():
    client = rag_agent_setup.OpenAIAssistantsClient()
    file_ids, vector_store = await rag_agent_setup.create_knowledge_base(client)
    assert isinstance(file_ids, list)


@pytest.mark.asyncio
async def test_multimodal_image_example():
    await multimodal_image_example.analyze_image_example()
