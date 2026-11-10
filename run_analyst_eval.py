"""
DOOM ENGINE — Evaluate the Analyst agent against the gold question set.

Usage:
    python run_analyst_eval.py                      # replay recorded LLM responses (offline, CI)
    python run_analyst_eval.py --mode record        # call the LLM for anything not yet recorded
    python run_analyst_eval.py --split test         # held-out questions only
    python run_analyst_eval.py --ids S1,G2 --mode record

Ground truth lives in evals/analyst_gold.json (computed by `python -m evals.analyst_gold`).
"""
import argparse
import asyncio
import json
import logging
import os
import sys

from evals.analyst_eval import DEFAULT_CACHE, render_markdown, run_eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Analyst agent")
    parser.add_argument("--mode", choices=["replay", "record", "off"], default="replay",
                        help="replay: cached responses only; record: call LLM on cache miss; off: always call")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ids = args.ids.split(",") if args.ids else None
    report = asyncio.run(run_eval(mode=args.mode, cache_path=args.cache, split=args.split, ids=ids))

    print()
    print(render_markdown(report))
    print("\nFailures:")
    for f in report["failures"]:
        print(f"  {f['id']} [{f['kind']}/{f['split']}] status={f['status']} {f['question']}")
        if f["kind"] == "answerable":
            print(f"      truth: {f['truth'][:4]}")
            print(f"      got:   {f.get('result_head')}")
        print(f"      sql:   {f['sql']}")
        if f.get("error"):
            print(f"      error: {f['error']}")
        if f.get("leaked_markers"):
            print(f"      leaked: {f['leaked_markers']}")
    if report["answer_failures"]:
        print("\nCorrect result, answer text missed a key value or was withheld:")
        for f in report["answer_failures"]:
            print(f"  {f['id']} grounded={f['grounded']} answer={f['answer'][:160]!r}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"\nWrote report to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
