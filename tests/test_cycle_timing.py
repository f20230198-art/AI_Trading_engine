"""
Cycle scheduling.

A manual strike used to BREAK the sleep loop without CONSUMING the flag, which
was only cleared at the top of the next iteration. The flag therefore survived
into every subsequent sleep, so a single War Room strike made the engine
free-run: cycles fired back to back instead of every cycle_interval_seconds,
burning LLM quota and rate limits. Observed live: cycles 2, 3 and 4 started
immediately after one another despite a 120s interval.
"""
import asyncio

from core.shared_state import SharedState


def test_strike_is_consumed_exactly_once():
    """Two reads of one strike: the second must be False."""
    state = SharedState()

    async def go():
        await state.request_manual_strike()
        return (
            await state.consume_manual_strike(),
            await state.consume_manual_strike(),
        )

    first, second = asyncio.run(go())
    assert first is True
    assert second is False, "strike flag survived consumption — engine would free-run"


def test_flag_is_cleared_after_consumption():
    state = SharedState()

    async def go():
        await state.request_manual_strike()
        assert state.manual_strike_requested is True
        await state.consume_manual_strike()
        return state.manual_strike_requested

    assert asyncio.run(go()) is False, (
        "manual_strike_requested still set — the sleep loop breaks on this flag, "
        "so leaving it set restarts a cycle on every subsequent sleep"
    )


def test_no_strike_consumes_false():
    state = SharedState()
    assert asyncio.run(state.consume_manual_strike()) is False


def test_repeated_strikes_each_run_once():
    state = SharedState()

    async def go():
        results = []
        for _ in range(3):
            await state.request_manual_strike()
            results.append(await state.consume_manual_strike())
            results.append(await state.consume_manual_strike())
        return results

    results = asyncio.run(go())
    assert results == [True, False, True, False, True, False]
