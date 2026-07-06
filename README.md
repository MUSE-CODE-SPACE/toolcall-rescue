# toolcall-rescue

> Recover tool/function calls that **local & small LLMs emit as plain text** —
> when the structured `tool_calls` field comes back empty. Zero dependencies.

[![CI](https://github.com/MUSE-CODE-SPACE/toolcall-rescue/actions/workflows/ci.yml/badge.svg)](https://github.com/MUSE-CODE-SPACE/toolcall-rescue/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/toolcall-rescue.svg)](https://pypi.org/project/toolcall-rescue/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

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

The package is prepared for PyPI (release imminent):

```bash
pip install toolcall-rescue
```

If that isn't live yet, install straight from GitHub — or just vendor the single
`toolcall_rescue/` folder, it's pure stdlib with zero dependencies:

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

## Provider recipes

The pattern is always the same: *use the structured field if present, rescue from
`content` only when it's empty.* Below, `TOOL_NAMES = {t["function"]["name"] for t in tools}`.

<details>
<summary><strong>Ollama</strong> (native client)</summary>

```python
import ollama
from toolcall_rescue import extract_tool_calls

resp = ollama.chat(model="qwen2.5", messages=messages, tools=tools)
msg = resp["message"]

calls = msg.get("tool_calls") or []
if not calls:
    rescued, residual = extract_tool_calls(msg.get("content"), valid_names=TOOL_NAMES)
    calls = [
        {"function": {"name": c.name, "arguments": c.arguments}} for c in rescued
    ]
    if rescued:
        msg["content"] = residual  # don't leak raw tool markup into the chat
```
</details>

<details>
<summary><strong>llama.cpp server / LM Studio / vLLM / any OpenAI-compatible endpoint</strong></summary>

All three expose an OpenAI-compatible `/v1/chat/completions`, so one snippet covers
them — just change `base_url`:

```python
from openai import OpenAI
from toolcall_rescue import extract_tool_calls

client = OpenAI(
    base_url="http://localhost:8080/v1",   # llama.cpp server
    # base_url="http://localhost:1234/v1", # LM Studio
    # base_url="http://localhost:8000/v1", # vLLM
    api_key="not-needed",
)

resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
msg = resp.choices[0].message

if not msg.tool_calls:
    rescued, residual = extract_tool_calls(msg.content, valid_names=TOOL_NAMES)
    for c in rescued:
        execute_tool(c.name, c.arguments)
```

Notes:
- **vLLM** has built-in parsers (`--enable-auto-tool-choice --tool-call-parser hermes`
  etc.) — use them when one exists for your model's template. `toolcall-rescue` is
  for when the parser doesn't match what the model actually emitted (common with
  quantized or fine-tuned checkpoints), or when you can't control server flags.
- **llama.cpp** needs `--jinja` for native tool calling; without it (or with a
  template the model ignores) calls land in `content` — exactly the rescue case.
</details>

## Supported formats

| Format | The model actually emits | Seen from |
|---|---|---|
| `tool_call_tag` | `<tool_call>{"name": "f", "arguments": {…}}</tool_call>` — **opening tag optional**: quantized models often emit only `{…}</tool_call>` | qwen2.5, GLM, many Ollama models (Hermes-style template) |
| `bare_json` | the *whole* message is `{"name": "f", "arguments": {…}}` | small models, JSON-mode outputs |
| `xml_function` | `<function name="f">{"q": "x"}</function>` · `<function=f>{"q": "x"}</function>` | gemma; pythonic llama.cpp templates / Ollama-cloud proxies |
| `invoke` | `<invoke name="f"><parameter name="city">Seoul</parameter></invoke>` | MiniMax, Anthropic-style XML tool templates |
| `kimi` | `<\|tool_call_begin\|>functions.f:0<\|tool_call_argument_begin\|>{…}<\|tool_call_end\|>` | Kimi K2 |
| `mistral` | `[TOOL_CALLS][{"name": "f", "arguments": {…}}]` · `[TOOL_CALLS]f[ARGS]{…}` | Mistral / Devstral / Magistral |
| `llama_python_tag` | `<\|python_tag\|>{"name": "f", "parameters": {…}}` | Llama 3.1 / 3.2 |
| `named_json` | `{"name": <registered-tool>, …}` anywhere, even mid-prose — **only with `valid_names`** | catch-all, registry-gated |

It also tolerates Python-dict argument syntax (single quotes), `</think>`-prefixed
reasoning output, and parallel calls.

Found a model that emits a shape this doesn't catch? [Open an issue](https://github.com/MUSE-CODE-SPACE/toolcall-rescue/issues)
with the raw `content` — new detectors are the whole point.

## The hard part: *not* firing on prose

Promoting a call **executes** it, so a false positive is worse than a miss. A model
frequently *describes* a call ("then you'd emit `<tool_call>{…}</tool_call>`") in
the very same message it later uses to make one. `toolcall-rescue` anchors every
pattern to a structural boundary — start of content, start of a line, or the end of
another tag — so markup sitting **mid-sentence is never matched**:

```python
extract_tool_calls('you would emit <tool_call>{"name": "drop_table", "arguments": {}}</tool_call> here')
# -> ([], "...")   # nothing fires — even a complete, well-formed pair
```

The one exception is the `named_json` scanner, which *can* read mid-prose JSON — but
only when its `name` **exactly** matches a tool you explicitly passed in `valid_names`.
Safety by construction.

## Safety model

What this library will **never** do:

- **Never fuzzy-repair a name.** `{"name": "search"}` is *not* promoted to your
  `web_search` tool. A name lifted from free text is lower-trust than one from a
  structured field; "correcting" it risks executing the wrong tool. Fail closed.
- **Never match narrated markup.** Every framed format is anchored to a line start
  or tag boundary; `<function …>` forms are stricter still (line start only — not
  even a preceding inline tag, and no sentence-punctuation boundary).
- **Never promote arbitrary JSON.** Whole-content bare JSON must have exactly the
  call shape (`name` + `arguments`/`parameters`, nothing else) with arguments under
  16 KB — a JSON *answer* like `{"result": 42}` is left alone.
- **Never double-execute.** Overlapping detector matches on the same bytes are
  deduped; the most specific detector wins.
- **Never crash your loop.** Each detector is exception-isolated.

And what *you* control:

| Control | How | Use when |
|---|---|---|
| Global kill switch | `TOOLCALL_RESCUE_DISABLE=1` | debugging; disable recovery without a deploy |
| Bare-JSON off | `extract_tool_calls(…, include_bare_json=False)` or `TOOLCALL_RESCUE_NO_BARE_JSON=1` | your pipeline can feed **untrusted text** (web pages, files, user uploads) into model output — whole-message bare JSON is the one shape untrusted text could plausibly imitate; the tag/token-framed formats keep working |
| Exact-name gate | pass `valid_names=` | always, if you have a tool registry |

Threat model, plainly: if an attacker can make your model's output *be* a verbatim
`{"name": …}` JSON naming one of your real tools, bare-JSON recovery would promote
it. `valid_names` bounds the blast radius to tools you actually expose, and
`include_bare_json=False` closes the surface entirely while keeping the framed
formats (which untrusted prose can't hit — they require model-template markup at a
structural boundary).

## Why isn't this in my agent framework's core?

Fair question — and the frameworks are (mostly) right to say no. When a very
similar layer was proposed for NousResearch's hermes-agent
([PR #35129](https://github.com/NousResearch/hermes-agent/pull/35129), same author
as this library), the maintainers' verdict was that absorbing a provider's contract
violation *in core* is the wrong altitude: once core silently repairs malformed
output, every provider's malformation becomes core's compatibility promise forever,
and the bare-JSON form in particular is an injection-shaped surface a framework
can't responsibly default-on for everyone. That reasoning is sound — notably, the
review still endorsed the *mechanics* (line-start-only anchoring, the exact-name
gate with no fuzzy repair, kill switches, leaf-module isolation), which are exactly
the pieces this library is built around.

The right altitude for this fix is a **separate, opt-in, zero-dependency library**
at the integration seam *you* own: you decide it runs, you scope it with your tool
registry, you can turn the risky detector off, and no framework has to carry the
compatibility burden. Same cure, consenting patient.

So: LangChain, hermes-agent, pydantic-ai, or 40 lines of hand-rolled agent loop —
`toolcall-rescue` slots in at the one spot they all share: *right after the
response comes back, when `tool_calls` is empty.*

## API

```python
extract_tool_calls(content, valid_names=None, *, include_bare_json=True)
# -> (list[ToolCall], residual_text)

has_tool_call(content, valid_names=None, *, include_bare_json=True)    # -> bool
strip_tool_calls(content, valid_names=None, *, include_bare_json=True) # -> str

ToolCall  # frozen dataclass: .name, .arguments (dict), .raw (matched span), .format
```

The residual is the prose around the call, with recovered markup removed — safe to
display. Markup whose tool name fails the `valid_names` gate is *also* stripped
(not executed, not shown).

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
