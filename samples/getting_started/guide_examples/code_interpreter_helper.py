from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

# Minimal stub classes to allow local import and syntax checking without full SDK
class AgentRunResponseUpdate:
    pass

class ChatResponseUpdate:
    pass


def get_code_interpreter_chunk(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter data."""
    # The real implementation expects chunk to have nested attributes matching the SDK types.
    # Here we implement a defensive check pattern similar to the sample for syntax validation.
    try:
        if (
            isinstance(chunk.raw_representation, ChatResponseUpdate)
            and isinstance(chunk.raw_representation.raw_representation, RunStepDeltaEvent)
            and isinstance(chunk.raw_representation.raw_representation.delta, RunStepDelta)
            and isinstance(chunk.raw_representation.raw_representation.delta.step_details, ToolCallDeltaObject)
            and chunk.raw_representation.raw_representation.delta.step_details.tool_calls
        ):
            for tool_call in chunk.raw_representation.raw_representation.delta.step_details.tool_calls:
                if (
                    isinstance(tool_call, CodeInterpreterToolCallDelta)
                    and isinstance(tool_call.code_interpreter, CodeInterpreter)
                    and tool_call.code_interpreter.input is not None
                ):
                    return tool_call.code_interpreter.input
    except Exception:
        # Defensive: if structure is missing, return None
        return None
    return None
