"""Score and rank eval results across parameter sweeps.

Usage:
  ./.venv/bin/python tools/score_runs.py

Looks for any grid_summary.json under eval_results/** and prints a summary
per top_k inside each parameter directory.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Any, List


def score_run(run: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics for a single run (one k value)."""
    passed = 0
    total_chars = 0
    total_distinct = 0
    avg_scores: List[float] = []

    for q in run:
        if not q.get("ok"):
            continue
        comp = q["compliance"]
        name = q["name"]
        distinct_ok = comp["distinct_talks"] >= (3 if name == "multi_list_3" else 1)
        title_ok = comp["contains_title_in_response"]
        concise_ok = comp["is_concise"]
        if distinct_ok and title_ok and concise_ok:
            passed += 1
        total_chars += comp["prompt_chars"]
        total_distinct += comp["distinct_talks"]
        avg_scores.append(comp["avg_score"])

    return {
        "passed": passed,
        "total_chars": total_chars,
        "total_distinct": total_distinct,
        "mean_avg_score": sum(avg_scores) / len(avg_scores) if avg_scores else 0.0,
    }


def main() -> None:
    root = pathlib.Path("eval_results")
    summaries = list(root.rglob("grid_summary.json"))
    if not summaries:
        print("No grid_summary.json files found under eval_results/")
        return

    for summary_path in sorted(summaries):
        print(f"\n=== {summary_path.parent} ===")
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for run in data:
            if "error" in run:
                print(f"k={run.get('top_k')} ERROR: {run['error']}")
                continue
            metrics = score_run(run["results"])
            print(
                f"k={run['top_k']}: "
                f"passed={metrics['passed']} "
                f"chars={metrics['total_chars']} "
                f"distinct={metrics['total_distinct']} "
                f"mean_score={metrics['mean_avg_score']:.4f}"
            )


if __name__ == "__main__":
    main()
