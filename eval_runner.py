"""Evaluation runner for TED Talk RAG.

Sends the 4 assignment queries to a deployed API, collects metrics, and
prints a concise summary to help choose hyperparameters.

Usage:
  ./.venv/bin/python eval_runner.py --base-url https://your-app.vercel.app

Optional flags:
  --base-url    Base URL of the deployed app (default: http://localhost:8000)
  --namespace   Pinecone namespace hint (passed for display only)
  --timeout     Request timeout in seconds (default: 60)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List

import requests


EVAL_QUERIES: List[Dict[str, str]] = [
    {
        "name": "precise_fact",
        "question": "Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker.",
    },
    {
        "name": "multi_list_3",
        "question": "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles.",
    },
    {
        "name": "key_idea_summary",
        "question": "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary of the key idea.",
    },
    {
        "name": "recommendation",
        "question": "I'm looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend? Justify using the retrieved context.",
    },
]


def post_prompt(base_url: str, question: str, timeout: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/prompt"
    resp = requests.post(url, json={"question": question}, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"error": f"Non-JSON response: status={resp.status_code}", "text": resp.text}
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", **data}
    return data


def summarize_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"name": name, "ok": False, "error": result["error"], "raw": result}

    ctx = result.get("context", [])
    response = result.get("response", "")

    distinct_talks = len({c.get("talk_id") for c in ctx if c.get("talk_id")})
    scores = [float(c.get("score", 0.0)) for c in ctx if isinstance(c.get("score"), (int, float))]
    avg_score = round(statistics.mean(scores), 4) if scores else 0.0
    prompt_chars = sum(len(c.get("chunk", "")) for c in ctx)

    # Compliance heuristics (approximate, no labels required)
    contains_title = any((c.get("title", "").lower() in response.lower()) for c in ctx if c.get("title"))
    is_concise = len(response) <= 1200  # adjust if needed

    compliance = {
        "distinct_talks": distinct_talks,
        "avg_score": avg_score,
        "prompt_chars": prompt_chars,
        "contains_title_in_response": contains_title,
        "is_concise": is_concise,
    }

    return {
        "name": name,
        "ok": True,
        "response": response,
        "context_count": len(ctx),
        "compliance": compliance,
        "context_preview": [
            {
                "talk_id": c.get("talk_id"),
                "title": c.get("title"),
                "score": c.get("score"),
                "chunk_preview": (c.get("chunk", "")[:140] + "...") if c.get("chunk") else "",
            }
            for c in ctx[:3]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TED Talk RAG via HTTP")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=str, default=None, help="Save JSON summary to this file")
    args = parser.parse_args()

    print(f"Evaluating against {args.base_url} (namespace={args.namespace})\n")

    summaries: List[Dict[str, Any]] = []
    for q in EVAL_QUERIES:
        print(f"Query: {q['name']}")
        res = post_prompt(args.base_url, q["question"], timeout=args.timeout)
        summary = summarize_result(q["name"], res)
        summaries.append(summary)
        if not summary.get("ok"):
            print(json.dumps(summary, indent=2))
            print("\n" + "-" * 80 + "\n")
            continue
        # Print concise metrics
        comp = summary["compliance"]
        print(f"  distinct_talks: {comp['distinct_talks']}  avg_score: {comp['avg_score']}")
        print(f"  prompt_chars: {comp['prompt_chars']}  contains_title: {comp['contains_title_in_response']}  concise: {comp['is_concise']}")
        print("  context preview:")
        for c in summary["context_preview"]:
            print(f"    - [{c['talk_id']}] {c['title']} (score={c['score']})\n      {c['chunk_preview']}")
        print("\n" + "-" * 80 + "\n")

    print("\nFinal JSON summary:\n")
    print(json.dumps(summaries, indent=2))
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
