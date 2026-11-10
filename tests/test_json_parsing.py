"""BaseAgent._parse_json: robust extraction of JSON from agent LLM responses."""
import pytest

from agents.base_agent import BaseAgent


class _Probe(BaseAgent):
    async def run(self, context):
        return {}


@pytest.fixture
def parse():
    return _Probe("probe", "probe")._parse_json


def test_raw_json(parse):
    assert parse('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}


def test_fenced_json_block(parse):
    assert parse('Here you go:\n```json\n{"verdict": "VETOED"}\n```\nThanks') == {"verdict": "VETOED"}


def test_fenced_block_without_language(parse):
    assert parse('```\n{"x": true}\n```') == {"x": True}


def test_json_embedded_in_prose(parse):
    assert parse('Analysis follows. {"risk_score": 42} End of analysis.') == {"risk_score": 42}


def test_nested_agent_response(parse):
    text = '{"reviewed_proposals": [{"symbol": "AAPL", "verdict": "APPROVED", "adjusted_size_pct": null}]}'
    assert parse(text)["reviewed_proposals"][0]["adjusted_size_pct"] is None


def test_trailing_garbage_after_json_is_recovered(parse):
    assert parse('{"a": {"b": 1}} trailing } junk') == {"a": {"b": 1}}


@pytest.mark.parametrize("bad", [
    "",
    "no json here at all",
    "{not: valid, json}",
    '{"a": 1,',                       # truncated
    "{'single': 'quotes'}",
    "[1, 2, 3]",                       # array, not an object
])
def test_malformed_returns_empty_dict(parse, bad):
    assert parse(bad) == {}
