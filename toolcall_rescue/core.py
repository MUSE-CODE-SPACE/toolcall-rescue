"""Core extraction logic for toolcall-rescue.

Everything here is pure stdlib. The public entry point is ``extract_tool_calls``.

The hard part is not parsing — it is *not* parsing prose. A model often writes
about a tool call ("then I'd emit ``<tool_call>{...}</tool_call>``") in the same
content it later uses to actually make one. Promoting the narrated one would
execute a tool the user never asked for. So every detector below is anchored to a
structural boundary (start of content, start of a line, or the end of another
tag) and never matches a brace that sits mid-sentence.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# A brace/tag is only "real" when it begins at content start, after a newline, or
# right after another tag (e.g. a reasoning ``</think>`` block or a wrapper tag).
# Prose always precedes a real word/space, which this lookbehind rejects.
_ANCHOR = r"(?:(?<=^)|(?<=[\n\r])|(?<=>))[ \t]*"


@dataclass(frozen=True)
class ToolCall:
    """One recovered tool call.

    Attributes
    ----------
    name:
        The tool/function name the model asked to call.
    arguments:
        Parsed arguments as a dict. Empty dict if the model supplied none.
    raw:
        The exact substring that was matched (removed to form the residual text).
    format:
        Which detector recovered it (``tool_call_tag``, ``bare_json``,
        ``xml_function``, ``invoke``, ``kimi``, ``named_json``). Useful for logging.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    format: str = ""


# -- lenient parsing helpers -------------------------------------------------


def _loads(raw: str) -> Any | None:
    """Parse JSON, falling back to Python-literal syntax (single quotes etc.)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None


def _as_args(value: Any) -> dict[str, Any]:
    """Normalize an ``arguments`` payload into a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


# -- individual format detectors ---------------------------------------------
# Each detector yields (name, arguments_dict, matched_span).

_RawHit = tuple[str, dict[str, Any], str]


# 1) <tool_call>{json}</tool_call>. Two cases:
#    a) a COMPLETE pair — a strong signal; matched anywhere and consumes the
#       opening tag too (so it doesn't get left behind in the residual).
#    b) an ORPHAN close — some quantized local models (qwen2.5/gemma on Ollama,
#       observed in the wild) drop the opening tag and emit only </tool_call>.
#       The orphan form is line-start-anchored so a narrated "{…} </tool_call>"
#       sitting mid-sentence never fires.
_TOOL_CALL_PAIR_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_ORPHAN_RE = re.compile(
    r"(?:(?<=^)|(?<=[\n\r]))[ \t]*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _find_tool_call_tag(content: str) -> Iterable[_RawHit]:
    consumed: list[tuple[int, int]] = []
    for m in _TOOL_CALL_PAIR_RE.finditer(content):
        obj = _loads(m.group(1))
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            consumed.append((m.start(), m.end()))
            yield obj["name"], _as_args(obj.get("arguments", {})), m.group(0)
    for m in _TOOL_CALL_ORPHAN_RE.finditer(content):
        # Skip an orphan close that is just the tail of a pair we already matched.
        if any(s <= m.start() and m.end() <= e for s, e in consumed):
            continue
        obj = _loads(m.group(1))
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            yield obj["name"], _as_args(obj.get("arguments", {})), m.group(0)


# 2) Whole-content bare JSON: the entire content is just {"name":..., "arguments":...}.
def _find_bare_json(content: str) -> Iterable[_RawHit]:
    s = content.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return
    obj = _loads(s)
    if not isinstance(obj, dict):
        return
    # Only treat it as a call if it looks like one (has a string name and nothing
    # beyond the call shape) — avoids promoting arbitrary JSON answers.
    if isinstance(obj.get("name"), str) and obj.keys() <= {"name", "arguments", "parameters"}:
        args = obj.get("arguments", obj.get("parameters", {}))
        yield obj["name"], _as_args(args), content


# 3) <function name="X">{json}</function>  (gemma-style)  and
#    <function=X>{json}</function>          (pythonic / llama.cpp template)
_FUNCTION_RE = re.compile(
    _ANCHOR + r"<function(?:\s+name\s*=\s*\"(?P<q>[^\"]+)\"[^>]*|\s*=\s*(?P<b>[^\s>]+)\s*)>"
    r"(?P<body>(?:(?!</function>).)*)</function>",
    re.DOTALL | re.IGNORECASE,
)


def _find_xml_function(content: str) -> Iterable[_RawHit]:
    if "<function" not in content.lower():
        return
    for m in _FUNCTION_RE.finditer(content):
        name = (m.group("q") or m.group("b") or "").strip()
        if not name:
            continue
        yield name, _as_args(_loads(m.group("body"))), m.group(0)


# 4) <invoke name="X"><parameter name="p">v</parameter>...</invoke>  (minimax / xml tools)
_INVOKE_RE = re.compile(
    _ANCHOR + r"(?:<\w+:tool_call>\s*)?<invoke\b[^>]*\bname\s*=\s*\"([^\"]+)\"[^>]*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter\b[^>]*\bname\s*=\s*\"([^\"]+)\"[^>]*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _find_invoke(content: str) -> Iterable[_RawHit]:
    if "<invoke" not in content.lower():
        return
    for m in _INVOKE_RE.finditer(content):
        name = m.group(1).strip()
        args = {p.strip(): v.strip() for p, v in _PARAM_RE.findall(m.group(2))}
        if name:
            yield name, args, m.group(0)


# 5) Kimi K2 special tokens:
#    <|tool_calls_section_begin|> <|tool_call_begin|> functions.NAME:0
#    <|tool_call_argument_begin|> {json} <|tool_call_end|> <|tool_calls_section_end|>
_KIMI_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<id>.*?)\s*<\|tool_call_argument_begin\|>"
    r"(?P<args>.*?)<\|tool_call_end\|>",
    re.DOTALL,
)


def _find_kimi(content: str) -> Iterable[_RawHit]:
    if "<|tool_call" not in content:
        return
    for m in _KIMI_RE.finditer(content):
        name = m.group("id").strip().removeprefix("functions.").rsplit(":", 1)[0].strip()
        obj = _loads(m.group("args"))
        if name and isinstance(obj, dict):
            yield name, obj, m.group(0)


# 5b) Mistral / Devstral / Magistral special token `[TOOL_CALLS]`. Two shapes
#     emitted across model/tokenizer versions:
#       [TOOL_CALLS][{"name": "x", "arguments": {...}}, ...]   (JSON array)
#       [TOOL_CALLS]x[ARGS]{json}                              (name + [ARGS])
#     `[TOOL_CALLS]` is a control token, not prose, so presence-gating is safe.
_MISTRAL_TAG = "[TOOL_CALLS]"
_MISTRAL_NAME_RE = re.compile(r"([^\[\s]+)\s*\[ARGS\]\s*", re.IGNORECASE)


def _find_mistral(content: str) -> Iterable[_RawHit]:
    if _MISTRAL_TAG not in content:
        return
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        idx = content.find(_MISTRAL_TAG, cursor)
        if idx == -1:
            break
        j = idx + len(_MISTRAL_TAG)
        while j < len(content) and content[j] in " \t\r\n":
            j += 1
        cursor = j
        if j >= len(content):
            break
        nxt = content[j]
        if nxt == "[":  # JSON array of call objects
            try:
                arr, end = decoder.raw_decode(content, j)
            except json.JSONDecodeError:
                continue
            if isinstance(arr, list):
                for el in arr:
                    if isinstance(el, dict) and isinstance(el.get("name"), str):
                        yield el["name"], _as_args(el.get("arguments", {})), content[idx:end]
                cursor = end
        elif nxt == "{":  # single bare call object
            try:
                obj, end = decoder.raw_decode(content, j)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                yield obj["name"], _as_args(obj.get("arguments", {})), content[idx:end]
            cursor = end
        else:  # name [ARGS] {json}
            m = _MISTRAL_NAME_RE.match(content, j)
            if not m:
                continue
            try:
                obj, end = decoder.raw_decode(content, m.end())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield m.group(1), obj, content[idx:end]
            cursor = end


# 6) Name-gated JSON scanner (only runs when you pass valid_names). Finds
#    {"name": <known-tool>, "arguments": {...}} anywhere, even mid-prose — safe
#    *because* the name must match a real tool. This catches the messy cases the
#    structural detectors miss without risking prose false positives.
def _find_named_json(content: str, valid_names: frozenset[str]) -> Iterable[_RawHit]:
    decoder = json.JSONDecoder()
    idx = content.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            idx = content.find("{", idx + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in valid_names:
            args = obj.get("arguments", obj.get("parameters", {}))
            yield obj["name"], _as_args(args), content[idx:end]
        idx = content.find("{", max(end, idx + 1))


# Order matters: structural/tagged formats first (most specific), bare JSON last.
_DETECTORS = (
    ("tool_call_tag", _find_tool_call_tag),
    ("xml_function", _find_xml_function),
    ("invoke", _find_invoke),
    ("kimi", _find_kimi),
    ("mistral", _find_mistral),
    ("bare_json", _find_bare_json),
)


def extract_tool_calls(
    content: str | None,
    valid_names: Iterable[str] | None = None,
) -> tuple[list[ToolCall], str]:
    """Recover tool calls a model emitted inside ``content``.

    Parameters
    ----------
    content:
        The assistant message content. ``None``/empty returns no calls.
    valid_names:
        Optional set of known tool names. When provided, an extra name-gated JSON
        scanner runs (catching calls embedded in prose, safely, because the name
        must match), and every recovered call is filtered to this set. Highly
        recommended — pass your tool registry's names.

    Returns
    -------
    (calls, residual):
        ``calls`` is the list of recovered :class:`ToolCall` (in the order found);
        ``residual`` is ``content`` with the matched spans removed (the prose the
        model wrote around the call), trimmed.
    """
    if not content or not isinstance(content, str):
        return [], (content or "")

    names = frozenset(valid_names) if valid_names is not None else None

    # Collect (format_label, name, args, span) from every detector, in order.
    labelled: list[tuple[str, str, dict[str, Any], str]] = []
    for fmt_name, detector in _DETECTORS:
        for name, args, span in detector(content):
            labelled.append((fmt_name, name, args, span))

    if names is not None:
        # The name-gated scanner only makes sense with a registry; it also recovers
        # cases the structural detectors miss (call embedded in prose).
        for name, args, span in _find_named_json(content, names):
            labelled.append(("named_json", name, args, span))

    calls: list[ToolCall] = []
    residual = content
    seen_spans: set[str] = set()
    for fmt_name, name, args, span in labelled:
        if names is not None and name not in names:
            continue
        if span in seen_spans:
            continue
        seen_spans.add(span)
        calls.append(ToolCall(name=name, arguments=args, raw=span, format=fmt_name))
        if span in residual:
            residual = residual.replace(span, "", 1)

    return calls, residual.strip()


def has_tool_call(content: str | None, valid_names: Iterable[str] | None = None) -> bool:
    """True if at least one tool call can be recovered from ``content``."""
    calls, _ = extract_tool_calls(content, valid_names)
    return bool(calls)


def strip_tool_calls(content: str | None, valid_names: Iterable[str] | None = None) -> str:
    """Return ``content`` with any recoverable tool-call spans removed."""
    _, residual = extract_tool_calls(content, valid_names)
    return residual
