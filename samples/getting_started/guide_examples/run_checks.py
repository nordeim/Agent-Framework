import sys
import pathlib
import traceback
import asyncio

# Ensure imports work when running from this directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MODULES = [
    ('code_interpreter_helper', 'get_code_interpreter_chunk (syntax check)'),
    ('quick_start_agent', 'main (async run)'),
    ('rag_agent_setup', 'create_knowledge_base (async run)'),
    ('multimodal_image_example', 'analyze_image_example (async run)'),
]

results = {}

for mod_name, action in MODULES:
    try:
        mod = __import__(mod_name)
    except Exception:
        results[mod_name] = ('FAIL', 'ImportError', traceback.format_exc())
        continue

    try:
        if mod_name == 'code_interpreter_helper':
            # Simple call to ensure function exists and is callable
            fn = getattr(mod, 'get_code_interpreter_chunk', None)
            if fn is None:
                raise RuntimeError('get_code_interpreter_chunk not found')
            res = fn(object())
            results[mod_name] = ('PASS', str(res))
        else:
            # For async mains, run them via asyncio.run
            if hasattr(mod, 'main'):
                asyncio.run(getattr(mod, 'main')())
                results[mod_name] = ('PASS', 'main ran')
            else:
                # check other named async functions
                if mod_name == 'rag_agent_setup':
                    client = mod.OpenAIAssistantsClient()
                    asyncio.run(mod.create_knowledge_base(client))
                    results[mod_name] = ('PASS', 'create_knowledge_base ran')
                elif mod_name == 'multimodal_image_example':
                    asyncio.run(mod.analyze_image_example())
                    results[mod_name] = ('PASS', 'analyze_image_example ran')
                else:
                    results[mod_name] = ('FAIL', 'No runnable entrypoint')
    except Exception:
        results[mod_name] = ('FAIL', action, traceback.format_exc())

print('\nValidation results:')
for mod, (status, info) in results.items():
    print(f'- {mod}: {status} - {info}')

# Exit with non-zero if any FAIL
if any(status == 'FAIL' for status, _ in results.values()):
    sys.exit(2)
else:
    sys.exit(0)
