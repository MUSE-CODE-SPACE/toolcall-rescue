"""toolcall-rescue — recover tool/function calls that local & small LLMs emit as
plain text in the ``content`` field, when the structured ``tool_calls`` field is empty.

Why this exists
---------------
Frontier models return tool calls in a structured field. But small / local models
served through Ollama, llama.cpp, vLLM, and friends frequently *narrate* the call
into the assistant ``content`` instead — as ``<tool_call>{...}</tool_call>``, a bare
JSON object, a ``<function=name>...`` block, Kimi/MiniMax special tokens, and so on.
When that happens the structured ``tool_calls`` is empty and the agent stalls: it
"makes up" an answer instead of executing the tool.

This library is a tiny, **zero-dependency** salvage layer. Give it the assistant
content; it returns the tool calls it can safely recover, plus the residual text.

Design goal: **never fire on prose.** Every pattern is anchored to a line start,
a tag boundary, or whole-content JSON — so a model *describing* a tool call
("you would emit ``<tool_call>{…}</tool_call>``") is left inert. Promotion executes
real calls, so a false positive is worse than a miss.

Quick start
-----------
    from toolcall_rescue import extract_tool_calls

    calls, residual = extract_tool_calls(assistant_message.content)
    for c in calls:
        print(c.name, c.arguments)   # ready to execute

Pass ``valid_names`` to only recover calls whose name is a real tool — the safest
mode, and recommended whenever you have the tool registry handy.
"""

from __future__ import annotations

from .core import (
    ToolCall,
    extract_tool_calls,
    has_tool_call,
    strip_tool_calls,
)

__version__ = "0.2.0"

__all__ = [
    "ToolCall",
    "extract_tool_calls",
    "has_tool_call",
    "strip_tool_calls",
]
