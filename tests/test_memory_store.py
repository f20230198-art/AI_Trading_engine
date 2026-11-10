"""
Memory metadata sanitization.

ChromaDB accepts only scalars and non-empty homogeneous lists. Agents routinely
produce empty lists — an Orchestrator cycle approving no trades yields
approved_symbols=[] — which ChromaDB rejects outright. Before the fix this
raised on most cycles, so the Orchestrator silently lost its memory whenever it
approved nothing, which is the common case.
"""
from core.memory_store import AgentMemoryStore

sanitize = AgentMemoryStore._sanitize_metadata


def test_empty_list_becomes_placeholder():
    """The exact shape that used to crash the Orchestrator."""
    assert sanitize({"approved_symbols": []})["approved_symbols"] == "none"


def test_list_is_joined():
    assert sanitize({"syms": ["AAPL", "MSFT"]})["syms"] == "AAPL, MSFT"


def test_none_is_dropped():
    """ChromaDB rejects null metadata values, so the key must not survive."""
    assert "missing" not in sanitize({"missing": None, "kept": "yes"})


def test_scalars_are_preserved():
    out = sanitize({"regime": "CRISIS", "count": 3, "ratio": 1.5, "flag": True})
    assert out == {"regime": "CRISIS", "count": 3, "ratio": 1.5, "flag": True}
    assert isinstance(out["flag"], bool)


def test_unsupported_types_become_strings():
    assert sanitize({"obj": {"a": 1}})["obj"] == "{'a': 1}"


def test_empty_and_none_metadata_are_safe():
    assert sanitize({}) == {}
    assert sanitize(None) == {}


def test_every_value_is_chroma_compatible():
    """Whole-dict guard: nothing unsupported may survive sanitization."""
    messy = {
        "empty": [], "syms": ["A", "B"], "null": None, "nested": {"x": 1},
        "s": "str", "i": 1, "f": 2.0, "b": False, "tup": ("X",), "st": set(),
    }
    for key, value in sanitize(messy).items():
        assert isinstance(value, (str, int, float, bool)), f"{key} -> {type(value)}"
