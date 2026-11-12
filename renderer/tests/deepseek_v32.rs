// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V3.2 prompts against vLLM's renderer; fixtures/gen_deepseek_v32.py writes
//! deepseek_v32.json from vLLM's deepseek_v32_encoding.py (and checks it against the
//! official HF encoding_dsv32.py).

use dynamo_renderer::OAIPromptFormatter;
use dynamo_renderer::deepseek::v32::{DeepSeekV32Formatter, ThinkingMode, encode_messages};
use serde_json::Value;

struct Request {
    request: dynamo_protocols::types::CreateChatCompletionRequest,
    args: Option<std::collections::HashMap<String, Value>>,
}

impl dynamo_renderer::OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, Value>> {
        self.args.as_ref()
    }
}

fn fixture(section: &str) -> Vec<Value> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/deepseek_v32.json")).unwrap();
    fixture[section].as_array().unwrap().clone()
}

#[test]
fn encode_matches_vllm() {
    for case in fixture("encode") {
        let name = case["name"].as_str().unwrap();
        let mode = if case["thinking"].as_bool().unwrap() {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let output = encode_messages(case["messages"].as_array().unwrap(), mode, true)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        match case["expected"].as_str() {
            Some(expected) => assert_eq!(output, expected, "{name}"),
            // vLLM rejects a trailing assistant message with neither reasoning nor
            // tool calls in thinking mode; Dynamo closes an empty think block.
            None => assert!(
                output.contains("</think>Paris 18C cloudy, Tokyo 22C sunny.<｜end▁of▁sentence｜>"),
                "{name}: {output}"
            ),
        }
    }
}

#[test]
fn requests_match_vllm() {
    let mut known_diffs = 0;
    for case in fixture("request") {
        let name = case["name"].as_str().unwrap();
        let mut request = case["request"].clone();
        request["model"] = "deepseek-ai/DeepSeek-V3.2".into();
        let args = serde_json::from_value(request["chat_template_kwargs"].clone()).unwrap();
        let request = Request {
            request: serde_json::from_value(request).unwrap(),
            args: Some(args),
        };
        let output = DeepSeekV32Formatter::new_thinking()
            .render(&request)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected = case["expected"].as_str().unwrap();
        if case.get("known_diff").is_some() {
            // Still a divergence; drop known_diff from the generator once fixed.
            assert_ne!(output, expected, "{name} now matches vLLM");
            known_diffs += 1;
        } else {
            assert_eq!(output, expected, "{name}");
        }
    }
    assert!(known_diffs > 0);
}
