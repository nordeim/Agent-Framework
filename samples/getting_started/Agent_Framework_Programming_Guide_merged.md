# Microsoft Agent Framework: Merged Programming Guide (Edition 1.0 + 2.0)

This merged guide starts from Edition 1 and incorporates Edition 2's practical
corrections, model-validation, and troubleshooting improvements. Model ID
placeholders in examples are illustrative only — validate at runtime using the
Model Validation pattern shown below.

Version: 1.0 (Merged)

---

## Table of Contents
1. Introduction & Getting Started
2. Core Concepts
3. Agent Type #1: Basic Conversational Agent
4. Agent Type #2: RAG Agent
5. Agent Type #3: Code Execution Agent
6. Agent Type #4: Multi-Modal Agent
7. Agent Type #5: MCP-Integrated Agent
8. Advanced Topics
9. Best Practices & Production Considerations
10. Troubleshooting Guide
11. Quick Reference & Next Steps

---

## Environment Setup (condensed)

### Step 1: Install the Framework
```bash
# Using pip
pip install agent-framework
# or, for environment-specific installs
python -m pip install agent-framework
```

> Note: Confirm the package name in your package index if you are using internal registries.

### Step 2: Set Environment Variables (Illustrative only)
The examples below contain model IDs that are **illustrative only**. Do **not** copy
them verbatim unless you have verified they exist in your account or deployment.
Use the Model Validation snippet below to select a safe default programmatically.

POSIX (macOS / Linux):
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_CHAT_MODEL_ID="GPT-5-Codex"   # illustrative only — validate at runtime
export OPENAI_RESPONSES_MODEL_ID="GPT-5-Codex"   # illustrative only — validate at runtime
```

PowerShell (Windows) - current session:
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
$env:OPENAI_CHAT_MODEL_ID = "GPT-5-Codex"   # illustrative only — validate at runtime
$env:OPENAI_RESPONSES_MODEL_ID = "GPT-5-Codex"   # illustrative only — validate at runtime
```

PowerShell (Windows) - persistent (use with caution):
```powershell
# Persisting environment variables with setx will not affect the current session.
# Use with care and prefer ephemeral session variables where possible.
setx OPENAI_API_KEY "your-api-key-here"
setx OPENAI_CHAT_MODEL_ID "gpt-4o"  # illustrative only — validate at runtime
setx OPENAI_RESPONSES_MODEL_ID "gpt-4o"  # illustrative only — validate at runtime
```

### Model Validation & Safe Fallback (merged from Edition 2)
Use this pattern to validate candidate model IDs and fall back to a safe default
if needed. This avoids runtime "Model Not Found" errors when example model names
are illustrative.

```python
# Example model validation snippet (minimal, illustrative)
VALID_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

def choose_model(candidate: str, available_models: list[str]) -> str:
    """Return candidate if available, else try to find a known-good fallback."""
    if candidate in available_models:
        return candidate
    for fallback in VALID_MODELS:
        if fallback in available_models:
            return fallback
    # Last resort: return the first available model
    return available_models[0] if available_models else "gpt-4o-mini"
```

> Recommendation: prefer `gpt-4o-mini` as a broadly-compatible safe default where
`gpt-4o` availability is uncertain.

---

## Troubleshooting (Merged & Expanded)

This section merges Edition 2's expanded troubleshooting guidance into Edition 1's layout.

### Common Errors & Fixes
- ModelNotFound / InvalidModel: Use the Model Validation snippet to select a supported model or list models for your account.
- RateLimitError: Implement exponential backoff (see samples).
- APIError / TimeoutError: Retry judiciously and increase timeouts where appropriate.
- ValidationError: Check function signatures and Pydantic models before sending to the agent.

### Model Not Found — Defensive Pattern (from Edition 2)
```python
def validate_and_select_model(client, env_model: str) -> str:
    # Fetch available models from provider API (pseudo-call)
    available_models = client.list_available_models()
    # Choose a validated model (see choose_model above)
    model_id = choose_model(env_model, available_models)
    if model_id != env_model:
        print(f"Warning: requested model '{env_model}' not available; using '{model_id}' instead.")
    return model_id
```

---

## Examples & Patterns (Preserved)
All agent-type examples (Conversational, RAG, Code Execution, Multi-Modal, MCP-Integrated)
are preserved from Edition 1 and Edition 2 with no breaking changes. Edition 2's clarifying
notes have been merged where they improve practicality (for example, clarifying expiration
options for vector stores and describing file size limits in RAG sections).

Note: For canonical helper implementations, the repository samples are the single source of
truth. The merged guide references and includes the canonical code-interpreter helper
implementation below (extracted from: `agents/openai/openai_assistants_with_code_interpreter.py`).

---

## 5. Agent Type #3: Code Execution Agent (canonical helper)

Include the canonical helper to extract code-interpreter input chunks from run-stream
responses. This matches the sample implementation and should be copy-paste safe.

```python
import asyncio

from agent_framework import AgentRunResponseUpdate, ChatResponseUpdate
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCallDelta,
    RunStepDelta,
    RunStepDeltaEvent,
    ToolCallDeltaObject,
)
from openai.types.beta.threads.runs.code_interpreter_tool_call_delta import CodeInterpreter

def get_code_interpreter_chunk(chunk: AgentRunResponseUpdate) -> str | None:
    """Helper method to access code interpreter data."""
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
    return None
```

---

## Quick Reference & Next Steps
- Use the merged model-validation snippet in your quick-start scripts.
- Mark model strings in examples as **illustrative only**.
- Run the included test harness from the repository to verify your environment:

```bash
python -m pytest -q samples/getting_started/guide_examples
```

---
## Changelog (what was merged from Edition 2)
- Added programmatic model validation and safe-fallback guidance.
- Clarified environment variable examples and explicitly labelled model IDs as illustrative.
- Expanded troubleshooting guidance and "Model Not Found" remediation flows.
- Minor editorial fixes and clarifications for RAG and vector store handling.

---

If you'd like, I can now:
- Apply these merged edits directly to `Agent_Framework_Programming_Guide_v3v.md` (in-place), or
- Keep this merged file (`Agent_Framework_Programming_Guide_merged.md`) and update the `guide_examples` tests to reference it for documentation.

Next step (suggested): run quick quality gates (pytest/flake8) against `guide_examples` to ensure no regressions.
