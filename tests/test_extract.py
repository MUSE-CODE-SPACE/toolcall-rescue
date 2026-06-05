"""Tests for toolcall-rescue. Pure stdlib, no model required."""

from __future__ import annotations

from toolcall_rescue import (
    ToolCall,
    extract_tool_calls,
    has_tool_call,
    strip_tool_calls,
)

# -- the formats we recover --------------------------------------------------


def test_wrapped_tool_call_tag():
    calls, residual = extract_tool_calls(
        '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
    )
    assert calls == [
        ToolCall(name="add", arguments={"a": 1, "b": 2}, raw=calls[0].raw, format="tool_call_tag")
    ]
    assert residual == ""


def test_orphan_closing_tag_with_leading_prose():
    # The signature real-world case: a quantized local model drops the opening
    # <tool_call> tag and emits only the closing one, with some leading text.
    content = 'leton\n{"name": "count_words", "arguments": {"text": "a b c"}}\n</tool_call>'
    calls, _ = extract_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "count_words"
    assert calls[0].arguments == {"text": "a b c"}


def test_reasoning_close_then_call():
    content = '</think>\n{"name": "now", "arguments": {}}\n</tool_call>'
    calls, _ = extract_tool_calls(content)
    assert [c.name for c in calls] == ["now"]


def test_bare_whole_content_json():
    calls, residual = extract_tool_calls('{"name": "get_time", "arguments": {}}')
    assert [c.name for c in calls] == ["get_time"]
    assert residual == ""


def test_gemma_xml_function_quoted():
    content = '<function name="search">{"query": "cats"}</function>'
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "cats"}


def test_pythonic_function_bare():
    content = '<function=multiply>{"a": 6, "b": 7}</function>'
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "multiply"
    assert calls[0].arguments == {"a": 6, "b": 7}


def test_minimax_invoke():
    content = (
        '<invoke name="weather">'
        '<parameter name="city">Seoul</parameter>'
        '<parameter name="unit">c</parameter>'
        "</invoke>"
    )
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "weather"
    assert calls[0].arguments == {"city": "Seoul", "unit": "c"}


def test_kimi_special_tokens():
    content = (
        "<|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|>"
        '{"id": 42}<|tool_call_end|>'
    )
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"id": 42}


def test_single_quoted_python_dict_args():
    # Some models emit Python-dict syntax instead of strict JSON.
    content = "<tool_call>{'name': 'add', 'arguments': {'a': 1}}</tool_call>"
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "add"
    assert calls[0].arguments == {"a": 1}


def test_multiple_parallel_calls():
    content = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    calls, _ = extract_tool_calls(content)
    assert [c.name for c in calls] == ["a", "b"]


# -- the part that matters most: NOT firing on prose -------------------------


def test_narrated_call_in_prose_is_ignored():
    # The model is *describing* a call mid-sentence, not making one.
    content = 'For example you would emit {"name": "drop_table"} </tool_call> to call it.'
    calls, _ = extract_tool_calls(content)
    assert calls == []


def test_plain_prose_yields_nothing():
    calls, residual = extract_tool_calls("Sure! The answer is 42. No tools needed.")
    assert calls == []
    assert residual == "Sure! The answer is 42. No tools needed."


def test_json_answer_that_is_not_a_call():
    # Whole-content JSON that is an *answer*, not a call, must not be promoted.
    calls, _ = extract_tool_calls('{"result": 42, "ok": true}')
    assert calls == []


def test_none_and_empty():
    assert extract_tool_calls(None) == ([], "")
    assert extract_tool_calls("") == ([], "")


# -- valid_names gating ------------------------------------------------------


def test_valid_names_filters_unknown():
    content = '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'
    calls, _ = extract_tool_calls(content, valid_names={"add", "search"})
    assert calls == []  # rm_rf is not a registered tool


def test_valid_names_recovers_embedded_call():
    # With a registry, a call embedded in prose is safely recoverable by name.
    content = 'Let me compute that. {"name": "add", "arguments": {"a": 2, "b": 3}} done.'
    calls, _ = extract_tool_calls(content, valid_names={"add"})
    assert len(calls) == 1
    assert calls[0].name == "add"
    assert calls[0].format == "named_json"


# -- residual + helpers ------------------------------------------------------


def test_residual_keeps_surrounding_prose():
    content = 'Okay! <tool_call>{"name": "ping", "arguments": {}}</tool_call> done.'
    calls, residual = extract_tool_calls(content)
    assert calls[0].name == "ping"
    assert "Okay!" in residual and "done." in residual
    assert "<tool_call>" not in residual


def test_has_and_strip_helpers():
    content = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    assert has_tool_call(content) is True
    assert has_tool_call("just text") is False
    assert strip_tool_calls(content) == ""
