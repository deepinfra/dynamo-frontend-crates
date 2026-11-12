// SPDX-FileCopyrightText: Copyright (c) 2026 DeepInfra
// SPDX-License-Identifier: Apache-2.0

//! Lenient schema conformance for parsed GLM tool-call arguments.
//!
//! GLM's arguments are not grammar-constrained (only the tool name is), so the model
//! sometimes emits values the client's schema rejects, and a client that validates
//! arguments fails the call. After parsing, this pass repairs the shapes seen in
//! production traffic, each only where the value provably violates the schema, so a
//! valid call is never changed. A node whose schema this pass cannot fully read (an
//! unresolvable or scoped `$ref`, `$ref` with sibling constraints, `not`, `if`,
//! a multi-entry `allOf`) is left as the parser produced it:
//!
//! - type mismatches: numeric/boolean strings to numbers/booleans, scalars to strings,
//!   a scalar into a one-element array, JSON text into the declared array/object;
//! - required properties the model omitted: `null` when the schema provably accepts
//!   null (OpenAI-strict "optional" fields), else the schema's `default`, else an
//!   empty value for an unconstrained array, string or boolean (`[]`, `""`, `false`);
//! - undeclared keys under `additionalProperties: false` are dropped;
//! - enum/const: case and whitespace differences, a numeric/string mix-up, or the
//!   only allowed value;
//! - `maxLength` / `maxItems` overruns are cut;
//! - `anyOf` / `oneOf`: only when every branch provably rejects the value; then the
//!   object branch whose discriminators (const / one-value enum properties) and keys
//!   fit best, else the first branch the value can be converted to;
//! - arguments emitted flat when the schema wants them wrapped in its one missing
//!   required object (or array of objects), e.g. `read_files(path=...)` for
//!   `read_files(files=[{path: ...}])`.

use std::cell::Cell;

use serde_json::{Map, Number, Value};

use super::glm47_parser::{resolve_local_schema_ref, schema_permits_null};
use super::parsed_value::ParsedValue;

const MAX_CONFORM_DEPTH: usize = 8;
const MAX_REF_HOPS: usize = 16;

/// Conform a call's parsed top-level arguments (in emitted order) to the tool's
/// parameter schema `root`.
pub(super) fn conform_arguments(root: &Value, arguments: &mut Vec<(String, ParsedValue)>) {
    let Some(schema) = resolve(root, root) else {
        return;
    };
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        wrap_flattened_arguments(schema, props, root, arguments);
        for (key, value) in arguments.iter_mut() {
            if let (Some(prop), ParsedValue::Json(value)) = (props.get(key.as_str()), value) {
                conform_value(prop, root, value, 1);
            }
        }
        if rejects_undeclared(schema) {
            arguments.retain(|(key, _)| props.contains_key(key.as_str()));
        }
    }
    let fills = missing_required_fills(schema, root, |key| {
        arguments.iter().any(|(present, _)| present == key)
    });
    arguments.extend(fills.into_iter().map(|(key, value)| (key, value.into())));
}

fn conform_value(schema: &Value, root: &Value, value: &mut Value, depth: usize) {
    if depth > MAX_CONFORM_DEPTH {
        return;
    }
    let Some(schema) = resolve(schema, root) else {
        return;
    };
    if schema.get("type").is_none()
        && let Some(branches) = ["anyOf", "oneOf"]
            .iter()
            .find_map(|keyword| schema.get(*keyword).and_then(Value::as_array))
    {
        if let Some(branch) = pick_branch(branches, root, value) {
            conform_value(branch, root, value, depth + 1);
        }
        return;
    }
    let types = declared_types(schema);
    if !types.is_empty()
        && !types.iter().any(|ty| type_matches(value, ty))
        && let Some(coerced) = types.iter().find_map(|ty| coerce(value, ty))
    {
        *value = coerced;
    }
    conform_enum(schema, value);
    match value {
        Value::Object(map) => conform_object(schema, root, map, depth),
        Value::Array(items) => {
            if let Some(item_schema) = schema.get("items").filter(|items| items.is_object()) {
                for item in items.iter_mut() {
                    conform_value(item_schema, root, item, depth + 1);
                }
            }
            if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
                items.truncate(max as usize);
            }
        }
        Value::String(text) => {
            if let Some(max) = schema.get("maxLength").and_then(Value::as_u64)
                && let Some((cut, _)) = text.char_indices().nth(max as usize)
            {
                text.truncate(cut);
            }
        }
        _ => {}
    }
}

fn conform_object(schema: &Value, root: &Value, map: &mut Map<String, Value>, depth: usize) {
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for (key, item) in map.iter_mut() {
            if let Some(prop) = props.get(key) {
                conform_value(prop, root, item, depth + 1);
            }
        }
        if rejects_undeclared(schema) {
            map.retain(|key, _| props.contains_key(key));
        }
    }
    for (key, value) in missing_required_fills(schema, root, |key| map.contains_key(key)) {
        map.insert(key, value);
    }
}

/// Values for required, declared properties the model left out: null where the schema
/// accepts it, else the schema's default, else an empty array. Others stay missing.
fn missing_required_fills(
    schema: &Value,
    root: &Value,
    present: impl Fn(&str) -> bool,
) -> Vec<(String, Value)> {
    let (Some(required), Some(props)) = (
        schema.get("required").and_then(Value::as_array),
        schema.get("properties").and_then(Value::as_object),
    ) else {
        return Vec::new();
    };
    let mut fills: Vec<(String, Value)> = Vec::new();
    for key in required.iter().filter_map(Value::as_str) {
        if present(key) || fills.iter().any(|(seen, _)| seen == key) {
            continue;
        }
        let Some(prop) = props.get(key) else {
            continue;
        };
        let fill = if schema_permits_null(prop, root) {
            Some(Value::Null)
        } else {
            resolve(prop, root).and_then(|resolved| {
                resolved
                    .get("default")
                    .cloned()
                    .or_else(|| empty_value(resolved))
            })
        };
        if let Some(fill) = fill {
            fills.push((key.to_string(), fill));
        }
    }
    fills
}

/// Move flat arguments into the one missing required property that declares them.
fn wrap_flattened_arguments(
    schema: &Value,
    props: &Map<String, Value>,
    root: &Value,
    arguments: &mut Vec<(String, ParsedValue)>,
) {
    let undeclared: Vec<usize> = arguments
        .iter()
        .enumerate()
        .filter(|(_, (key, _))| !props.contains_key(key.as_str()))
        .map(|(index, _)| index)
        .collect();
    if undeclared.is_empty() {
        return;
    }
    let mut missing = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|key| {
            props.contains_key(*key)
                && !arguments
                    .iter()
                    .any(|(present, _)| present.as_str() == *key)
        });
    let (Some(target), None) = (missing.next(), missing.next()) else {
        return;
    };
    let Some(target_schema) = resolve(&props[target], root) else {
        return;
    };
    let (holder, as_array) = match declared_types(target_schema).as_slice() {
        ["array"] => match target_schema
            .get("items")
            .and_then(|items| resolve(items, root))
        {
            Some(items) => (items, true),
            None => return,
        },
        ["object"] => (target_schema, false),
        _ => return,
    };
    let Some(holder_props) = holder.get("properties").and_then(Value::as_object) else {
        return;
    };
    if !undeclared
        .iter()
        .all(|&index| holder_props.contains_key(arguments[index].0.as_str()))
    {
        return;
    }
    let position = undeclared[0];
    let mut moved = Map::new();
    for &index in undeclared.iter().rev() {
        let (key, value) = arguments.remove(index);
        moved.insert(key, value.into_value());
    }
    let mut wrapped = Value::Object(moved);
    if as_array {
        wrapped = Value::Array(vec![wrapped]);
    }
    arguments.insert(position, (target.to_string(), wrapped.into()));
}

/// The branch to conform `value` under, or `None` to leave it alone. A value that a
/// branch accepts, or may accept, is left as is. Otherwise an object goes to the object
/// branch that fits it best, and anything else to the first branch it converts to.
fn pick_branch<'a>(branches: &'a [Value], root: &'a Value, value: &mut Value) -> Option<&'a Value> {
    let resolved: Vec<&Value> = branches
        .iter()
        .map(|branch| resolve(branch, root))
        .collect::<Option<_>>()?;
    if resolved
        .iter()
        .any(|branch| validates(branch, root, value, 0) != Some(false))
    {
        return None;
    }
    if let Value::Object(map) = &*value {
        return resolved
            .iter()
            .copied()
            .filter(|branch| declared_types(branch).contains(&"object"))
            .filter_map(|branch| object_fit(branch, map).map(|score| (score, branch)))
            .max_by_key(|(score, _)| *score)
            .map(|(_, branch)| branch);
    }
    for branch in resolved {
        if let Some(coerced) = declared_types(branch)
            .iter()
            .find_map(|ty| coerce(value, ty))
        {
            *value = coerced;
            return Some(branch);
        }
    }
    None
}

/// How well `map` fits an object branch: `None` when a discriminator (a const or
/// one-value enum property) disagrees, else matched discriminators first, then
/// declared keys present, minus required keys missing.
fn object_fit(branch: &Value, map: &Map<String, Value>) -> Option<(usize, i64)> {
    let props = branch.get("properties").and_then(Value::as_object);
    let mut discriminators = 0;
    for (key, prop) in props.into_iter().flatten() {
        let fixed =
            prop.get("const")
                .or_else(|| match prop.get("enum").and_then(Value::as_array) {
                    Some(values) if values.len() == 1 => values.first(),
                    _ => None,
                });
        if let (Some(fixed), Some(actual)) = (fixed, map.get(key)) {
            if fixed != actual {
                return None;
            }
            discriminators += 1;
        }
    }
    let declared = map
        .keys()
        .filter(|key| props.is_some_and(|p| p.contains_key(*key)))
        .count();
    let missing = branch
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|key| !map.contains_key(*key))
        .count();
    Some((discriminators, declared as i64 - missing as i64))
}

/// Whether `value` satisfies `schema`: `Some(false)` only for a definite violation of a
/// keyword this pass understands, `None` when it cannot tell.
fn validates(schema: &Value, root: &Value, value: &Value, depth: usize) -> Option<bool> {
    if depth > MAX_CONFORM_DEPTH {
        return None;
    }
    let schema = resolve(schema, root)?;
    let unknown = Cell::new(false);
    let fold = |result: Option<bool>| match result {
        Some(false) => Some(false),
        None => {
            unknown.set(true);
            Some(true)
        }
        Some(true) => Some(true),
    };
    for keyword in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
            let results: Vec<Option<bool>> = branches
                .iter()
                .map(|branch| validates(branch, root, value, depth + 1))
                .collect();
            let any = if results.contains(&Some(true)) {
                Some(true)
            } else if results.contains(&None) {
                None
            } else {
                Some(false)
            };
            fold(any)?;
        }
    }
    if schema.get("pattern").is_some() || schema.get("format").is_some() {
        unknown.set(true);
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        return Some(false);
    }
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.contains(value)
    {
        return Some(false);
    }
    let types = declared_types(schema);
    if !types.is_empty() && !types.iter().any(|ty| type_matches(value, ty)) {
        return Some(false);
    }
    let bound = |keyword: &str| schema.get(keyword).and_then(Value::as_u64);
    match value {
        Value::Object(map) => {
            let props = schema.get("properties").and_then(Value::as_object);
            let required = schema.get("required").and_then(Value::as_array);
            if required
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .any(|key| !map.contains_key(key))
            {
                return Some(false);
            }
            for (key, item) in map {
                match props.and_then(|props| props.get(key)) {
                    Some(prop) => fold(validates(prop, root, item, depth + 1))?,
                    None if rejects_undeclared(schema) => return Some(false),
                    None => match schema.get("additionalProperties") {
                        Some(extra) if extra.is_object() => {
                            fold(validates(extra, root, item, depth + 1))?
                        }
                        _ => true,
                    },
                };
            }
        }
        Value::Array(items) => {
            if bound("maxItems").is_some_and(|max| items.len() as u64 > max)
                || bound("minItems").is_some_and(|min| (items.len() as u64) < min)
            {
                return Some(false);
            }
            if let Some(item_schema) = schema.get("items").filter(|items| items.is_object()) {
                for item in items {
                    fold(validates(item_schema, root, item, depth + 1))?;
                }
            }
        }
        Value::String(text) => {
            let chars = text.chars().count() as u64;
            if bound("maxLength").is_some_and(|max| chars > max)
                || bound("minLength").is_some_and(|min| chars < min)
            {
                return Some(false);
            }
        }
        _ => {}
    }
    if unknown.get() { None } else { Some(true) }
}

/// An empty value for a required property the model left out, when the property
/// constrains nothing that an empty value could break.
fn empty_value(schema: &Value) -> Option<Value> {
    let constrained = [
        "enum",
        "const",
        "pattern",
        "format",
        "minLength",
        "minItems",
        "anyOf",
        "oneOf",
    ]
    .iter()
    .any(|keyword| schema.get(*keyword).is_some());
    if constrained {
        return None;
    }
    match declared_types(schema).as_slice() {
        ["array"] => Some(Value::Array(Vec::new())),
        ["string"] => Some(Value::String(String::new())),
        ["boolean"] => Some(Value::Bool(false)),
        _ => None,
    }
}

fn conform_enum(schema: &Value, value: &mut Value) {
    if let Some(constant) = schema.get("const") {
        if value != constant {
            *value = constant.clone();
        }
        return;
    }
    let Some(allowed) = schema.get("enum").and_then(Value::as_array) else {
        return;
    };
    if allowed.is_empty() || allowed.contains(value) {
        return;
    }
    let replacement = match &*value {
        Value::String(text) => {
            let wanted = text.trim().to_lowercase();
            let mut folded = allowed.iter().filter(|candidate| {
                candidate
                    .as_str()
                    .is_some_and(|c| c.trim().to_lowercase() == wanted)
            });
            match (folded.next(), folded.next()) {
                (Some(only), None) => Some(only.clone()),
                _ => {
                    let number = serde_json::from_str::<Number>(text.trim())
                        .ok()
                        .map(Value::Number);
                    number.filter(|number| allowed.contains(number))
                }
            }
        }
        Value::Number(number) => {
            let text = number.to_string();
            allowed
                .iter()
                .find(|candidate| candidate.as_str() == Some(text.as_str()))
                .cloned()
        }
        _ => None,
    };
    if let Some(replacement) = replacement {
        *value = replacement;
    } else if let [only] = allowed.as_slice() {
        *value = only.clone();
    }
}

fn rejects_undeclared(schema: &Value) -> bool {
    schema.get("additionalProperties") == Some(&Value::Bool(false))
        && schema.get("patternProperties").is_none()
}

fn declared_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(ty)) => vec![ty.as_str()],
        Some(Value::Array(types)) => types.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn type_matches(value: &Value, ty: &str) -> bool {
    match ty {
        "string" => value.is_string(),
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn coerce(value: &Value, ty: &str) -> Option<Value> {
    match (ty, value) {
        ("integer", Value::String(text)) => text.trim().parse::<i64>().ok().map(Value::from),
        ("integer", Value::Number(number)) => number
            .as_f64()
            .filter(|n| n.fract() == 0.0 && n.abs() < 9.0e15)
            .map(|n| Value::from(n as i64)),
        ("number", Value::String(text)) => text
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number),
        ("boolean", Value::String(text)) => match text.trim().to_lowercase().as_str() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        ("string", Value::Number(number)) => Some(Value::String(number.to_string())),
        ("string", Value::Bool(flag)) => Some(Value::String(flag.to_string())),
        ("string", Value::Object(_) | Value::Array(_)) => {
            serde_json::to_string(value).ok().map(Value::String)
        }
        ("array", Value::String(text)) => match serde_json::from_str::<Value>(text.trim()) {
            Ok(parsed @ Value::Array(_)) => Some(parsed),
            _ => Some(Value::Array(vec![value.clone()])),
        },
        ("array", Value::Null) => Some(Value::Array(Vec::new())),
        ("array", _) => Some(Value::Array(vec![value.clone()])),
        ("object", Value::String(text)) => match serde_json::from_str::<Value>(text.trim()) {
            Ok(parsed @ Value::Object(_)) => Some(parsed),
            _ => None,
        },
        ("null", Value::String(text))
            if matches!(text.trim().to_lowercase().as_str(), "" | "null" | "none") =>
        {
            Some(Value::Null)
        }
        _ => None,
    }
}

const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
    "$schema",
];
const UNSUPPORTED: &[&str] = &[
    "$dynamicRef",
    "$recursiveRef",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// The schema node to conform against: follows local `$ref`s and single-entry `allOf`
/// wrappers. `None` when the node cannot be read with confidence.
fn resolve<'a>(schema: &'a Value, root: &'a Value) -> Option<&'a Value> {
    let mut node = schema;
    for _ in 0..MAX_REF_HOPS {
        if node.is_boolean() {
            return None;
        }
        if UNSUPPORTED
            .iter()
            .any(|keyword| node.get(*keyword).is_some())
            || (node.get("$id").is_some() && !std::ptr::eq(node, root))
        {
            return None;
        }
        let only_annotations_besides = |keyword: &str| {
            node.as_object().is_some_and(|map| {
                map.keys()
                    .all(|key| key == keyword || ANNOTATIONS.contains(&key.as_str()))
            })
        };
        if let Some(reference) = node.get("$ref") {
            if !only_annotations_besides("$ref") {
                return None;
            }
            node = resolve_local_schema_ref(reference.as_str()?, root)?;
            continue;
        }
        if let Some(all_of) = node.get("allOf").and_then(Value::as_array) {
            match all_of.as_slice() {
                [only] if only_annotations_besides("allOf") => {
                    node = only;
                    continue;
                }
                _ => return None,
            }
        }
        return Some(node);
    }
    None
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::tool_calling::ToolDefinition;
    use crate::tool_calling::config::Glm47ParserConfig;
    use crate::tool_calling::xml::try_tool_call_parse_glm47;

    fn parse(name: &str, parameters: Value, wire_args: &str) -> (Value, String) {
        let tools = vec![ToolDefinition {
            name: name.into(),
            parameters: Some(parameters),
            strict: None,
        }];
        let wire = format!("<tool_call>{name}{wire_args}</tool_call>");
        let (calls, _) =
            try_tool_call_parse_glm47(&wire, &Glm47ParserConfig::default(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        let raw = calls[0].function.arguments.clone();
        (serde_json::from_str(&raw).unwrap(), raw)
    }

    fn arg(key: &str, value: &str) -> String {
        format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>")
    }

    #[test]
    fn flat_arguments_are_wrapped_into_the_missing_required_array() {
        // Production call: read_files(path=..., start_line=..., end_line=...).
        let schema = json!({
            "type": "object",
            "properties": {"files": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": ["path"],
                "additionalProperties": false
            }}},
            "required": ["files"],
            "additionalProperties": false
        });
        let wire = arg("path", "src/a.py") + &arg("start_line", "157") + &arg("end_line", "291");
        let (arguments, _) = parse("read_files", schema, &wire);
        assert_eq!(
            arguments,
            json!({"files": [{"path": "src/a.py", "start_line": 157, "end_line": 291}]})
        );
    }

    #[test]
    fn values_are_converted_to_the_declared_types_at_any_depth() {
        let schema = json!({
            "type": "object",
            "properties": {
                "opts": {"type": "object", "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "dry": {"type": "boolean"},
                    "label": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "meta": {"type": "object"}
                }}
            }
        });
        let wire = arg(
            "opts",
            r#"{"count": "5", "ratio": "0.5", "dry": "TRUE", "label": 7, "tags": "x", "meta": "{\"a\": 1}"}"#,
        );
        let (arguments, _) = parse("probe", schema, &wire);
        assert_eq!(
            arguments,
            json!({"opts": {"count": 5, "ratio": 0.5, "dry": true, "label": "7", "tags": ["x"], "meta": {"a": 1}}})
        );
    }

    #[test]
    fn undeclared_keys_are_dropped_only_when_the_schema_forbids_them() {
        let closed = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "inner": {
                "type": "object", "properties": {"b": {"type": "string"}}, "additionalProperties": false
            }},
            "additionalProperties": false
        });
        let wire = arg("a", "x") + &arg("extra", "y") + &arg("inner", r#"{"b": "z", "c": 1}"#);
        let (arguments, _) = parse("probe", closed, &wire);
        assert_eq!(arguments, json!({"a": "x", "inner": {"b": "z"}}));

        let open = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let (arguments, _) = parse("probe", open, &(arg("a", "x") + &arg("extra", "y")));
        assert_eq!(arguments, json!({"a": "x", "extra": "y"}));
    }

    #[test]
    fn enums_lengths_and_required_fields_are_brought_into_range() {
        let schema = json!({
            "type": "object",
            "properties": {
                "i": {"type": "string", "description": "concise intent"},
                "command": {"type": "string"},
                "mode": {"type": "string", "enum": ["Fast", "slow"]},
                "kind": {"type": "string", "enum": ["only"]},
                "name": {"type": "string", "maxLength": 4},
                "ids": {"type": "array", "maxItems": 2},
                "async": {"type": "boolean"},
                "files": {"type": "array"},
                "path": {"type": "string", "minLength": 1}
            },
            "required": ["command", "i", "async", "files", "path"]
        });
        let wire = arg("command", "ls")
            + &arg("mode", " fast ")
            + &arg("kind", "other")
            + &arg("name", "abcdefg")
            + &arg("ids", "[1, 2, 3]");
        let (arguments, _) = parse("bash", schema, &wire);
        // `path` has minLength 1, so an empty fill would not help: it stays missing.
        assert_eq!(
            arguments,
            json!({
                "command": "ls", "mode": "Fast", "kind": "only", "name": "abcd", "ids": [1, 2],
                "i": "", "async": false, "files": []
            })
        );
    }

    #[test]
    fn union_values_are_left_alone_unless_every_branch_rejects_them() {
        let op = |action: &str, required: Value, extra: Value| {
            let mut properties = json!({
                "name": {"type": "string"},
                "action": {"type": "string", "enum": [action]}
            });
            properties
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json!({"type": "object", "additionalProperties": false,
                   "properties": properties, "required": required})
        };
        let schema = json!({
            "type": "object",
            "properties": {"operations": {"type": "array", "items": {"anyOf": [
                op("create", json!(["name", "action", "content"]), json!({"content": {"type": "string"}})),
                op("patch", json!(["name", "action", "old_string", "new_string"]),
                   json!({"old_string": {"type": "string"}, "new_string": {"type": "string"}}))
            ]}}},
            "required": ["operations"]
        });
        let valid = r#"[{"action": "patch", "name": "s", "old_string": "a", "new_string": "b"}]"#;
        let (arguments, _) = parse("skill_manage", schema.clone(), &arg("operations", valid));
        assert_eq!(
            arguments["operations"],
            serde_json::from_str::<Value>(valid).unwrap()
        );

        // Production call: a patch without `name` and `old_string`. It goes to the
        // `patch` branch (its discriminator), not the first object branch.
        let partial = r#"[{"action": "patch", "new_string": "b"}]"#;
        let (arguments, _) = parse("skill_manage", schema, &arg("operations", partial));
        assert_eq!(
            arguments["operations"],
            json!([{"action": "patch", "new_string": "b", "name": "", "old_string": ""}])
        );
    }

    #[test]
    fn json_members_trailing_an_array_argument_are_lifted_to_arguments() {
        // Production call: `checks` carried the rest of the call as JSON members.
        let schema = json!({
            "type": "object",
            "properties": {
                "checks": {"type": "array", "items": {"type": "object"}},
                "postId": {"type": "integer"},
                "sku": {"type": "string"}
            },
            "required": ["checks", "postId", "sku"]
        });
        let value = r#"[{"criterion": "a", "passed": true}],"postId": 246540,"sku": "116212"}"#;
        let (arguments, _) = parse("StructuredOutput", schema.clone(), &arg("checks", value));
        assert_eq!(
            arguments,
            json!({"checks": [{"criterion": "a", "passed": true}], "postId": 246540, "sku": "116212"})
        );
        // An argument the model also emitted on its own keeps that value.
        let (arguments, _) = parse(
            "StructuredOutput",
            schema,
            &(arg("checks", value) + &arg("sku", "x")),
        );
        assert_eq!(arguments["sku"], json!("x"));
    }

    #[test]
    fn valid_calls_are_unchanged() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 100},
                "limit": {"type": ["integer", "null"]},
                "filters": {"type": "array", "items": {"anyOf": [
                    {"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"]},
                    {"type": "string"}
                ]}},
                "mode": {"type": "string", "enum": ["a", "b"]}
            },
            "required": ["query", "limit"],
            "additionalProperties": false
        });
        let wire = arg("query", "hello")
            + &arg("limit", "null")
            + &arg("filters", r#"[{"field": "x", "op": "eq"}, "y"]"#)
            + &arg("mode", "b");
        let (_, raw) = parse("search", schema, &wire);
        assert_eq!(
            raw,
            r#"{"query":"hello","limit":null,"filters":[{"field":"x","op":"eq"},"y"],"mode":"b"}"#
        );
    }
}
