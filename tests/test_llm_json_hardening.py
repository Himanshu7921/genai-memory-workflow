from __future__ import annotations

from app.orchestrator.nodes.intent import IntentClassificationNode
from app.orchestrator.nodes.tool_plan import ToolPlanningNode


def test_tool_plan_parser_handles_markdown_and_single_quotes() -> None:
    node = ToolPlanningNode()
    raw = """```json
{'tool_calls': [{'tool_name': 'calculator', 'arguments': {'expression': '2+2'}}]}
```"""
    parsed = node._parse_llm_tool_plan(raw)
    assert parsed == [{"tool_name": "calculator", "arguments": {"expression": "2+2"}}]


def test_tool_plan_parser_supports_single_tool_schema_and_none() -> None:
    node = ToolPlanningNode()
    one_call = "{\"tool\": \"retrieval_tool\", \"arguments\": {\"query\": \"policy\"}}"
    parsed_one_call = node._parse_llm_tool_plan(one_call)
    assert parsed_one_call == [{"tool_name": "retrieval_tool", "arguments": {"query": "policy"}}]

    none_call = "{\"tool\": \"none\", \"arguments\": {}}"
    parsed_none = node._parse_llm_tool_plan(none_call)
    assert parsed_none == []


def test_classifier_parser_handles_markdown_and_single_quotes() -> None:
    node = IntentClassificationNode()
    raw = """```json
{'intent': 'general_qa', 'confidence': 0.88, 'labels': ['answer_only']}
```"""
    parsed = node._parse_classifier_json(raw)
    assert parsed is not None
    intent, confidence, labels = parsed
    assert intent == "general_qa"
    assert confidence == 0.88
    assert labels == ["answer_only"]
