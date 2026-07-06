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


def test_mistral_tool_calls_array():
    # Mistral/Devstral: [TOOL_CALLS] followed by a JSON array of call objects.
    content = '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}
    assert calls[0].format == "mistral"


def test_mistral_name_args_form():
    # Newer Mistral tokenizer: [TOOL_CALLS]name[ARGS]{json}.
    content = '[TOOL_CALLS]get_weather[ARGS]{"city": "Seoul"}'
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Seoul"}


def test_mistral_after_reasoning_text():
    # Reasoning model: prose/thinking, then the [TOOL_CALLS] block at the end.
    content = 'Let me check the weather for you.\n[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
    calls, residual = extract_tool_calls(content)
    assert calls[0].name == "get_weather"
    assert "[TOOL_CALLS]" not in residual
    assert "Let me check" in residual


def test_llama_python_tag():
    # Llama 3.1/3.2 emits <|python_tag|> then a JSON object (arguments or parameters).
    content = '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Busan"}}'
    calls, _ = extract_tool_calls(content)
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Busan"}
    assert calls[0].format == "llama_python_tag"


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


def test_narrated_complete_pair_in_prose_is_ignored():
    # A COMPLETE <tool_call> pair narrated mid-sentence must not fire either —
    # the pair regex is anchored to a line start or a preceding tag boundary.
    content = (
        "For example you would emit "
        '<tool_call>{"name": "drop_table", "arguments": {}}</tool_call> to call it.'
    )
    calls, residual = extract_tool_calls(content)
    assert calls == []
    assert residual == content


def test_pair_after_reasoning_close_tag_fires():
    # ...but a pair sitting right after another tag (e.g. </think>) is real.
    content = '</think><tool_call>{"name": "now", "arguments": {}}</tool_call>'
    calls, _ = extract_tool_calls(content)
    assert [c.name for c in calls] == ["now"]


def test_narrated_function_after_inline_tag_is_ignored():
    # <function> forms use the STRICTER line-start-only anchor: even a preceding
    # inline tag (here, prose HTML) must not arm them.
    content = 'as shown here <br><function name="drop_table">{"x": 1}</function> in docs'
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


def test_exact_name_gate_no_fuzzy_repair():
    # "search" != "web_search": a near-miss name from free text must fail closed,
    # never be "corrected" to a registered tool.
    content = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    calls, _ = extract_tool_calls(content, valid_names={"web_search"})
    assert calls == []


def test_rejected_span_still_stripped_from_residual():
    # A recognized-format span whose name fails the gate is not promoted, but
    # its markup is still removed so raw tags never leak to display.
    content = 'before\n<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>\nafter'
    calls, residual = extract_tool_calls(content, valid_names={"add"})
    assert calls == []
    assert "<tool_call>" not in residual
    assert "before" in residual and "after" in residual


def test_overlapping_detectors_yield_one_call():
    # tool_call_tag and the named_json scanner both see this call; byte-range
    # overlap dedupe must collapse them to ONE (structural detector wins).
    content = '<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>'
    calls, _ = extract_tool_calls(content, valid_names={"add"})
    assert len(calls) == 1
    assert calls[0].format == "tool_call_tag"


# -- kill switches & bare-JSON controls ---------------------------------------


def test_global_kill_switch_env(monkeypatch):
    monkeypatch.setenv("TOOLCALL_RESCUE_DISABLE", "1")
    content = '<tool_call>{"name": "add", "arguments": {}}</tool_call>'
    calls, residual = extract_tool_calls(content)
    assert calls == []
    assert residual == content  # content untouched when disabled


def test_bare_json_opt_out_flag():
    bare = '{"name": "get_time", "arguments": {}}'
    calls, _ = extract_tool_calls(bare, include_bare_json=False)
    assert calls == []
    # ...while tag-framed formats keep working (narrowness guard).
    tagged = '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
    calls, _ = extract_tool_calls(tagged, include_bare_json=False)
    assert [c.name for c in calls] == ["get_time"]


def test_bare_json_env_kill_switch(monkeypatch):
    monkeypatch.setenv("TOOLCALL_RESCUE_NO_BARE_JSON", "true")
    calls, _ = extract_tool_calls('{"name": "get_time", "arguments": {}}')
    assert calls == []


def test_bare_json_oversized_arguments_rejected():
    huge = '{"name": "write_file", "arguments": {"data": "' + "A" * 20_000 + '"}}'
    calls, _ = extract_tool_calls(huge)
    assert calls == []


def test_detector_exception_is_isolated(monkeypatch):
    # One throwing detector must not take down the rescue path.
    from toolcall_rescue import core

    def boom(content):
        raise RuntimeError("bad parser")
        yield  # pragma: no cover

    patched = (("boom", boom),) + core._DETECTORS
    monkeypatch.setattr(core, "_DETECTORS", patched)
    content = '<tool_call>{"name": "add", "arguments": {}}</tool_call>'
    calls, _ = extract_tool_calls(content)
    assert [c.name for c in calls] == ["add"]


# -- residual + helpers ------------------------------------------------------


def test_residual_keeps_surrounding_prose():
    content = 'Okay!\n<tool_call>{"name": "ping", "arguments": {}}</tool_call>\ndone.'
    calls, residual = extract_tool_calls(content)
    assert calls[0].name == "ping"
    assert "Okay!" in residual and "done." in residual
    assert "<tool_call>" not in residual


def test_same_line_call_after_prose_needs_registry():
    # A same-line-after-prose pair is indistinguishable from narration, so the
    # anchored detector stays silent — but with a registry the named_json
    # scanner recovers the real call safely (exact-name gated).
    content = 'Okay! <tool_call>{"name": "ping", "arguments": {}}</tool_call> done.'
    assert extract_tool_calls(content)[0] == []
    calls, _ = extract_tool_calls(content, valid_names={"ping"})
    assert [c.name for c in calls] == ["ping"]


def test_has_and_strip_helpers():
    content = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    assert has_tool_call(content) is True
    assert has_tool_call("just text") is False
    assert strip_tool_calls(content) == ""
