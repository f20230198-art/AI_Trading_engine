"""
LLM fallback chain: Ollama -> Groq/Gemini rotation -> deterministic quant fallback.
Every provider is mocked; no network.
"""
import asyncio
import json

import pytest

from agents import base_agent
from agents.base_agent import BaseAgent


class _Probe(BaseAgent):
    async def run(self, context):
        return {}


def _agent():
    agent = _Probe("probe", "probe")
    agent._client = object()  # anything truthy; no real HTTP client
    return agent


async def _raise(*_a, **_k):
    raise RuntimeError("provider down")


def _provider(monkeypatch, name):
    async def _resolve():
        return name
    monkeypatch.setattr(base_agent, "_resolve_provider", _resolve)


def test_ollama_then_rotation_fail_falls_through_to_quant(monkeypatch):
    _provider(monkeypatch, "ollama")
    monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["fake-groq-key"])
    agent = _agent()
    calls = []

    async def ollama(*a):
        calls.append("ollama")
        raise RuntimeError("ollama down")

    async def rotation(*a):
        calls.append("rotation")
        raise RuntimeError("all keys exhausted")

    monkeypatch.setattr(agent, "_call_ollama", ollama)
    monkeypatch.setattr(agent, "_call_with_rotation", rotation)

    payload = json.loads(asyncio.run(agent._call_llm("sys", "user", context={})))
    assert calls == ["ollama", "rotation"]
    assert "reviewed_proposals" in payload
    assert "QUANT MODE" in payload["doom_speaks"]


@pytest.mark.parametrize("provider,method", [
    ("groq", "_call_with_rotation"),
    ("gemini", "_call_with_rotation"),
    ("anthropic", "_call_anthropic"),
])
def test_cloud_provider_failure_falls_through_to_quant(monkeypatch, provider, method):
    _provider(monkeypatch, provider)
    agent = _agent()
    monkeypatch.setattr(agent, method, _raise)
    sentinel = json.dumps({"quant": True})
    monkeypatch.setattr(agent, "_quant_fallback", lambda ctx: sentinel)
    assert asyncio.run(agent._call_llm("sys", "user", context={})) == sentinel


def test_quant_mode_never_calls_a_provider(monkeypatch):
    _provider(monkeypatch, "quant")
    agent = _agent()
    for m in ("_call_ollama", "_call_with_rotation", "_call_anthropic"):
        monkeypatch.setattr(agent, m, _raise)
    payload = json.loads(asyncio.run(agent._call_llm("sys", "user", context={})))
    assert payload["final_decisions"] == []


def test_successful_provider_short_circuits(monkeypatch):
    _provider(monkeypatch, "ollama")
    agent = _agent()

    async def ok(*a):
        return '{"ok": true}'

    monkeypatch.setattr(agent, "_call_ollama", ok)
    monkeypatch.setattr(agent, "_quant_fallback", lambda ctx: pytest.fail("fallback used"))
    assert asyncio.run(agent._call_llm("sys", "user")) == '{"ok": true}'


def test_rotation_tries_every_key_before_giving_up(monkeypatch):
    monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["groq-key-1111111", "groq-key-2222222"])
    monkeypatch.setattr(base_agent, "GEMINI_API_KEYS", ["gemini-key-3333333"])

    async def passthrough(coro):
        return await coro

    monkeypatch.setattr(base_agent, "_rate_limited_call", passthrough)
    agent = _agent()
    tried = []

    async def fail_key(s, u, m, key):
        tried.append(key)
        raise RuntimeError("down")

    monkeypatch.setattr(agent, "_call_groq_key", fail_key)
    monkeypatch.setattr(agent, "_call_gemini_key", fail_key)

    with pytest.raises(RuntimeError, match="exhausted"):
        asyncio.run(agent._call_with_rotation("s", "u", 10))
    assert tried == ["groq-key-1111111", "groq-key-2222222", "gemini-key-3333333"]
