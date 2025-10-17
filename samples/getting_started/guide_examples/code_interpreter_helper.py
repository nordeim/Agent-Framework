from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import (
    CodeInterpreter,
)


class AgentRunResponseUpdate:
    """Stub for SDK response update used for local syntax checks."""
    pass


class ChatResponseUpdate:
    """Stub for SDK chat response used for local syntax checks."""
    pass


def get_code_interpreter_chunk(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter input from nested SDK objects.

    This function is defensive: it checks for the expected nested attributes
    and returns the `input` string when a CodeInterpreter tool call is found.
    If the structure is missing or any attribute is not present, it returns
    None. The implementation is intentionally lightweight so tests and
    static checks do not need the full OpenAI SDK available.
    """
    try:
        raw = getattr(chunk, "raw_representation", None)
        if (
            isinstance(raw, ChatResponseUpdate)
            and isinstance(raw.raw_representation, RunStepDeltaEvent)
            and isinstance(raw.raw_representation.delta, RunStepDelta)
            and isinstance(
                raw.raw_representation.delta.step_details, ToolCallDeltaObject
            )
            and raw.raw_representation.delta.step_details.tool_calls
        ):
            tool_calls = (
                raw.raw_representation.delta.step_details.tool_calls
            )
            for tool_call in tool_calls:
                if (
                    isinstance(tool_call, CodeInterpreterToolCallDelta)
                    and isinstance(tool_call.code_interpreter, CodeInterpreter)
                    and tool_call.code_interpreter.input is not None
                ):
                    return tool_call.code_interpreter.input
    except Exception:
    except Exception:
        # If structure differs or is missing, return None.
        return None
        return None
    return None
