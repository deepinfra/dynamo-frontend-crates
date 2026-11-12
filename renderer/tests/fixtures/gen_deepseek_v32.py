# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes deepseek_v32.json: DeepSeek-V3.2 prompts rendered by vLLM.

    python3 gen_deepseek_v32.py <vllm>/vllm/tokenizers/deepseek_v32_encoding.py \
        [<hf DeepSeek-V3.2>/encoding/encoding_dsv32.py]

"encode" cases are message lists in the formatter's internal form (tools on the
system message, `reasoning_content`), rendered with vLLM's `encode_messages`
(drop_thinking as vLLM's tokenizer sets it). With the official HF encoder given,
every case must render the same there too.

"request" cases are OpenAI chat requests, rendered the way vLLM's DeepSeek-V3.2
path does for text-only messages: parse_chat_messages(content_format="string")
(text parts joined with "\\n", `reasoning_content` renamed to `reasoning`), tools
as `model_dump()` gives them, then the tokenizer's apply_chat_template (a new
leading system message carries the tools; thinking from `thinking` or
`enable_thinking`). A `known_diff` case is one where Dynamo renders differently
on purpose or for a reason not yet fixed; the test only checks it still renders.

vLLM's TOOLS_SYSTEM_TEMPLATE is a copy of the HF one without its blank lines;
Dynamo keeps the HF text, so it is swapped in before rendering.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

HF_TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}function_calls>
...
</{dsml_token}function_calls>

<function_results>
...
</function_results>

{thinking_start_token}...thinking about results{thinking_end_token}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "filters": {"type": "object"},
            },
            "required": ["query"],
        },
    },
}
NO_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "now",
        "parameters": {"type": "object", "properties": {}},
    },
}


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def agent_loop(reasoning, system=True, final=False, new_user=False):
    """user -> assistant(1 call) -> tool -> assistant(2 calls) -> 2 tools [...].

    reasoning: None (field absent, as agent clients send it), "" or text.
    """

    def assistant(i, **fields):
        msg = {"role": "assistant", **fields}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning if not reasoning else f"{reasoning} {i}"
        return msg

    messages = [{"role": "system", "content": "You are a coding agent."}] if system else []
    messages += [
        {"role": "user", "content": "What's the weather in Paris and Tokyo? Search news too."},
        assistant(1, content="", tool_calls=[call("c1", "get_weather", {"location": "Paris"})]),
        {"role": "tool", "tool_call_id": "c1", "content": "18C, cloudy"},
        assistant(
            2,
            content=None,
            tool_calls=[
                call("c2", "get_weather", {"location": "東京", "unit": "celsius"}),
                call("c3", "search", {"query": "Tokyo news", "limit": 3, "filters": {"lang": ["en", "ja"], "safe": True}}),
            ],
        ),
        {"role": "tool", "tool_call_id": "c2", "content": "22C, sunny"},
        {"role": "tool", "tool_call_id": "c3", "content": "[{\"title\": \"Tokyo\"}]"},
    ]
    if final:
        messages.append(assistant(3, content="Paris 18C cloudy, Tokyo 22C sunny."))
    if new_user:
        messages.append({"role": "user", "content": "Thanks. And London?"})
    return messages


def with_tools(messages, tools):
    """Formatter-internal form: tools on the first system message, or a new one."""
    messages = copy.deepcopy(messages)
    if messages and messages[0]["role"] == "system":
        messages[0]["tools"] = tools
    else:
        messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return messages


CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
MULTI_TURN = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi, what is 2+2?"},
    {"role": "assistant", "content": "4.", "reasoning_content": "Simple sum."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "6."},
    {"role": "user", "content": "Thanks!"},
]

ENCODE_CASES = {
    "plain_chat": CHAT,
    "multi_turn_chat": MULTI_TURN,
    "user_only": [{"role": "user", "content": "Hello!"}],
    "agent_loop_no_reasoning": with_tools(agent_loop(None), [WEATHER, SEARCH]),
    "agent_loop_empty_reasoning": with_tools(agent_loop(""), [WEATHER, SEARCH]),
    "agent_loop_with_reasoning": with_tools(agent_loop("Need data."), [WEATHER, SEARCH]),
    "agent_loop_final_no_reasoning": with_tools(agent_loop(None, final=True), [WEATHER, SEARCH]),
    "agent_loop_then_user_no_reasoning": with_tools(
        agent_loop(None, final=True, new_user=True), [WEATHER, SEARCH]
    ),
    "agent_loop_then_user_with_reasoning": with_tools(
        agent_loop("Need data.", final=True, new_user=True), [WEATHER, SEARCH]
    ),
    "tools_no_system_message": with_tools(
        [{"role": "user", "content": "Weather in Paris?"}], [WEATHER]
    ),
    "tools_no_system_message_agent_loop": with_tools(
        agent_loop(None, system=False), [WEATHER, SEARCH]
    ),
    "no_user_message": [
        {"role": "system", "content": "", "tools": [WEATHER]},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"location": "Paris"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
    ],
}

# known_diff: why Dynamo renders differently; None when it must match.
SYSTEM_PLACEMENT = (
    "vLLM puts the tools in a new leading system message, so the client's system "
    "text follows </functions> directly; Dynamo appends the tools to the client's "
    "system message after \\n\\n"
)
REQUEST_CASES = {
    "plain_chat": (dict(messages=CHAT), None),
    "multi_turn_reasoning_field": (
        dict(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello", "reasoning": "Greet back."},
                {"role": "user", "content": "Bye"},
            ]
        ),
        None,
    ),
    "tools_no_system_message": (
        dict(messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER]),
        None,
    ),
    "agent_loop_no_reasoning_no_system": (
        dict(messages=agent_loop(None, system=False), tools=[WEATHER, SEARCH]),
        None,
    ),
    "agent_loop_with_reasoning_no_system": (
        dict(messages=agent_loop("Need data.", system=False), tools=[WEATHER, SEARCH]),
        None,
    ),
    "agent_loop_no_reasoning_with_system": (
        dict(messages=agent_loop(None), tools=[WEATHER, SEARCH]),
        SYSTEM_PLACEMENT,
    ),
    "tool_without_description": (
        dict(messages=[{"role": "user", "content": "Time?"}], tools=[NO_DESCRIPTION]),
        "vLLM's model_dump() writes \"description\": null before parameters; Dynamo "
        "writes \"description\": \"\" after them",
    ),
    "user_content_parts": (
        dict(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part."},
                        {"type": "text", "text": "Second part."},
                    ],
                }
            ]
        ),
        "vLLM joins text parts with \\n (content_format=\"string\"); Dynamo concatenates",
    ),
}


def vllm_tool_dump(tool):
    """ChatCompletionToolsParam.model_dump(): FunctionDefinition fields in
    declaration order, strict/defer_loading dropped when unset."""
    fn = tool["function"]
    out = {
        "name": fn["name"],
        "description": fn.get("description"),
        "parameters": fn.get("parameters"),
    }
    if fn.get("strict") is not None:
        out["strict"] = fn["strict"]
    return {"type": "function", "function": out}


def vllm_parse_messages(messages):
    """parse_chat_messages(content_format="string") for text-only messages."""
    conversation = []
    for raw in messages:
        msg = dict(raw)
        reasoning = msg.pop("reasoning_content", None)
        if reasoning is not None and msg.get("reasoning") is None:
            msg["reasoning"] = reasoning
        content = msg.get("content")
        if content is None:
            texts = []
        elif isinstance(content, str):
            texts = [content]
        else:
            texts = [part["text"] for part in content if part.get("type") == "text"]
        out = {"role": msg["role"], "content": "\n".join(texts)}
        if msg["role"] == "assistant":
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            if msg.get("reasoning") is not None:
                out["reasoning"] = msg["reasoning"]
        if msg["role"] == "tool" and "tool_call_id" in msg:
            out["tool_call_id"] = msg["tool_call_id"]
        conversation.append(out)
    return conversation


def vllm_request_prompt(enc, request, thinking):
    """vllm/tokenizers/deepseek_v32.py apply_chat_template(tokenize=False)."""
    messages = vllm_parse_messages(request["messages"])
    tools = [vllm_tool_dump(t) for t in request.get("tools", [])]
    if tools:
        messages.insert(0, {"role": "system"})
        messages[0]["tools"] = tools
    return enc.encode_messages(
        messages,
        thinking_mode="thinking" if thinking else "chat",
        drop_thinking=messages[-1]["role"] == "user",
    )


def to_vllm_internal(messages):
    """Internal form uses reasoning_content (as HF does); vLLM reads reasoning."""
    out = []
    for msg in copy.deepcopy(messages):
        if "reasoning_content" in msg:
            msg["reasoning"] = msg.pop("reasoning_content")
        out.append(msg)
    return out


def main():
    enc = load(sys.argv[1], "vllm_deepseek_v32_encoding")
    enc.TOOLS_SYSTEM_TEMPLATE = HF_TOOLS_SYSTEM_TEMPLATE
    hf = load(sys.argv[2], "hf_encoding_dsv32") if len(sys.argv) > 2 else None

    encode = []
    for name, messages in ENCODE_CASES.items():
        for thinking in (False, True):
            mode = "thinking" if thinking else "chat"
            drop_thinking = messages[-1]["role"] == "user"
            case = {"name": f"{name}-{mode}", "thinking": thinking, "messages": messages}
            try:
                case["expected"] = enc.encode_messages(
                    to_vllm_internal(messages), thinking_mode=mode, drop_thinking=drop_thinking
                )
            except ValueError as e:
                # Dynamo renders these instead of failing the request.
                case["reference_error"] = str(e)
            if hf is not None:
                try:
                    hf_out = hf.encode_messages(
                        copy.deepcopy(messages), thinking_mode=mode, drop_thinking=drop_thinking
                    )
                except AssertionError:
                    hf_out = None
                assert hf_out == case.get("expected"), f"HF and vLLM disagree on {case['name']}"
            encode.append(case)

    request = []
    for name, (fields, known_diff) in REQUEST_CASES.items():
        for thinking in (False, True):
            mode = "thinking" if thinking else "chat"
            case = {
                "name": f"{name}-{mode}",
                "request": {"chat_template_kwargs": {"thinking": thinking}, **fields},
                "expected": vllm_request_prompt(enc, fields, thinking),
            }
            if known_diff:
                case["known_diff"] = known_diff
            request.append(case)

    out = Path(__file__).with_name("deepseek_v32.json")
    out.write_text(
        json.dumps({"encode": encode, "request": request}, ensure_ascii=False, indent=1) + "\n"
    )
    print(f"wrote {len(encode)} encode and {len(request)} request cases to {out}")


if __name__ == "__main__":
    main()
