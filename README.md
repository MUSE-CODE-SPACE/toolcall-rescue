# toolcall-rescue

> Recover tool/function calls that **local & small LLMs emit as plain text** —
> when the structured `tool_calls` field comes back empty. Zero dependencies.

[![PyPI](https://img.shields.io/pypi/v/toolcall-rescue.svg)](https://pypi.org/project/toolcall-rescue/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)](tests/)

---

## The problem

Frontier models return tool calls in a structured field. Small and local models —
served through **Ollama, llama.cpp, vLLM, LM Studio** — often don't. They *narrate*
the call into the assistant `content` instead:

```
{"name": "get_weather", "arguments": {"city": "Seoul"}}
</tool_call>
```

When that happens, the SDK's `message.tool_calls` is empty, your agent loop sees
"no tool call," and the model ends up **making up an answer instead of running the
tool**. Anyone who has wired an agent to `qwen2.5`, `gemma`, `kimi`, `minimax`, or
a quantized local model on Ollama has hit this.

`toolcall-rescue` is a tiny salvage layer: hand it the content, get back the tool
calls it can safely recover.

## Install

```bash
pip install toolcall-rescue
```

Latest from GitHub, or just vendor the single `toolcall_rescue/` folder — it's pure stdlib, zero dependencies:

```bash
pip install "git+https://github.com/MUSE-CODE-SPACE/toolcall-rescue.git"
```

## Usage

```python
from toolcall_rescue import extract_tool_calls

calls, residual = extract_tool_calls(assistant_message.content)
for call in calls:
    print(call.name, call.arguments)   # ready to execute
```

Drop it into an agent loop as a fallback — only when the structured field is empty:

```python
resp = client.chat(model="qwen2.5", messages=msgs, tools=tools)
msg = resp.message

tool_calls = msg.tool_calls
if not tool_calls:
    # Structured field empty — the model may have emitted the call as text.
    recovered, _ = extract_tool_calls(msg.content, valid_names={t["name"] for t in tools})
    tool_calls = recovered

for call in tool_calls:
    run(call.name, call.arguments)
```

Pass **`valid_names`** (your tool registry) whenever you can — it's the safest mode,
and it unlocks recovery of calls embedded in prose without risking false positives.

## Supported formats

| Format | Looks like | Seen from |
|---|---|---|
| `tool_call_tag` | `<tool_call>{json}</tool_call>` — **opening tag optional** | qwen, many Ollama models |
| `bare_json` | whole content is `{"name":…, "arguments":…}` | small models, JSON-mode |
| `xml_function` | `<function name="X">{json}</function>` and `<function=X>{json}</function>` | gemma, llama.cpp templates |
| `invoke` | `<invoke name="X"><parameter …/></invoke>` | minimax, xml tool templates |
| `kimi` | `<\|tool_call_begin\|>…<\|tool_call_argument_begin\|>{json}<\|tool_call_end\|>` | Kimi K2 |
| `mistral` | `[TOOL_CALLS][{…}]` and `[TOOL_CALLS]name[ARGS]{json}` | Mistral / Devstral / Magistral |
| `llama_python_tag` | `<\|python_tag\|>{"name":…, "parameters":…}` | Llama 3.1 / 3.2 |
| `named_json` | `{"name": <known-tool>, …}` anywhere (only with `valid_names`) | catch-all, registry-gated |

It also tolerates Python-dict argument syntax (single quotes) and parallel calls.

## The hard part: *not* firing on prose

Promoting a call **executes** it, so a false positive is worse than a miss. A model
frequently *describes* a call ("then you'd emit `<tool_call>{…}</tool_call>`") in
the very same message it later uses to make one. `toolcall-rescue` anchors every
pattern to a structural boundary — start of content, start of a line, or the end of
another tag — so a brace sitting **mid-sentence is never matched**:

```python
extract_tool_calls('For example you would emit {"name": "drop_table"} </tool_call> here')
# -> ([], "...")   # nothing fires
```

The one exception is the `named_json` scanner, which *can* read mid-prose JSON — but
only when its `name` matches a tool you explicitly passed in `valid_names`. Safety by
construction.

## Why a separate library?

This bug shows up independently in lots of agent stacks (Ollama integrations,
open-source agent frameworks, local-first assistants). Rather than each project
re-deriving the same brittle regex, here's one small, **tested, zero-dependency**
module that handles the messy real-world formats and — crucially — the false-positive
gating. Drop it in, or copy the file.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest         # all tests run offline, no model needed
ruff check .
```

## Contributing

Found a model that emits tool calls in a shape this doesn't catch? Open an issue with
a sample of the raw `content` (and the model/backend), or send a PR adding a detector
plus a test. New formats are very welcome — that's the whole point.

## License

[MIT](LICENSE).
