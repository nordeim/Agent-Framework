import sys
import pathlib
import pytest

# Ensure the guide_examples directory is importable during
# pytest collection and runtime
HERE = pathlib.Path(__file__).resolve().parent


def test_code_interpreter_helper_syntax():
    sys.path.insert(0, str(HERE))
    import importlib
    code_interpreter_helper = importlib.import_module(
        "code_interpreter_helper"
    )
    assert hasattr(code_interpreter_helper, "get_code_interpreter_chunk")
    assert code_interpreter_helper.get_code_interpreter_chunk(object()) is None


@pytest.mark.asyncio
async def test_quick_start_agent_runs():
    sys.path.insert(0, str(HERE))
    import importlib
    quick_start_agent = importlib.import_module("quick_start_agent")
    await quick_start_agent.main()


@pytest.mark.asyncio
async def test_rag_agent_setup():
    sys.path.insert(0, str(HERE))
    import importlib
    rag_agent_setup = importlib.import_module("rag_agent_setup")
    client = rag_agent_setup.OpenAIAssistantsClient()
    file_ids, vector_store = await rag_agent_setup.create_knowledge_base(
        client
    )
    assert isinstance(file_ids, list)


@pytest.mark.asyncio
async def test_multimodal_image_example():
    sys.path.insert(0, str(HERE))
    import importlib
    multimodal_image_example = importlib.import_module(
        "multimodal_image_example"
    )
    await multimodal_image_example.analyze_image_example()
