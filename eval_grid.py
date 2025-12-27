"""Grid search evaluation for different top_k values.

Runs eval_runner.py for each top_k value and saves results.
Note: This only affects local evaluation. For deployed API, update Vercel env vars.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Import config to get chunk_size and overlap_ratio
sys.path.insert(0, str(Path(__file__).parent))
from config import get_config


def run_eval(base_url: str, top_k: int, output_dir: Path, timeout: int = 60) -> dict:
    """Run evaluation with specified top_k value."""
    # Create subdirectory for this parameter combination
    param_dir = output_dir / f"k{top_k}"
    param_dir.mkdir(exist_ok=True, parents=True)
    output_file = param_dir / "eval_results.json"
    
    print(f"\n{'='*80}")
    print(f"Evaluating with TOP_K={top_k}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "eval_runner.py",
        "--base-url", base_url,
        "--timeout", str(timeout),
        "--top-k", str(top_k),
        "--output", str(output_file),
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Warning: eval_runner.py returned exit code {result.returncode}")
        return {"top_k": top_k, "error": "eval failed"}
    
    # Load and return the results
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"top_k": top_k, "results": data, "output_file": str(output_file)}
    else:
        return {"top_k": top_k, "error": "no output file"}


def summarize_comparison(all_results: list) -> None:
    """Print comparison table across different k values."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'k':<6} {'Query':<20} {'Distinct':<10} {'AvgScore':<10} {'Chars':<10} {'Title':<8} {'Concise':<8}")
    print("-" * 80)
    
    for run in all_results:
        if "error" in run:
            print(f"{run['top_k']:<6} ERROR: {run['error']}")
            continue
        
        k = run["top_k"]
        for query_result in run["results"]:
            if not query_result.get("ok"):
                continue
            name = query_result["name"][:18]
            comp = query_result["compliance"]
            print(f"{k:<6} {name:<20} {comp['distinct_talks']:<10} {comp['avg_score']:<10.4f} "
                  f"{comp['prompt_chars']:<10} {str(comp['contains_title_in_response']):<8} "
                  f"{str(comp['is_concise']):<8}")
    
    print("\n" + "="*80)
    print("\nDetailed results saved in eval_results/ directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search evaluation for top_k")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of deployed API")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 7, 10, 15], 
                        help="List of top_k values to test (default: 5 7 10 15)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"),
                        help="Directory to save results (default: eval_results)")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    
    # Load config to get current hyperparameters
    cfg = get_config()
    param_name = f"size={cfg.chunk_size}_overlap={cfg.overlap_ratio}"
    
    # Create output directory with parameter name
    param_dir = args.output_dir / param_name
    param_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Running grid search for k={args.k_values}")
    print(f"Hyperparameters: chunk_size={cfg.chunk_size}, overlap_ratio={cfg.overlap_ratio}")
    print(f"Target API: {args.base_url}")
    print(f"Results will be saved to: {param_dir}/\n")
    
    all_results = []
    for k in args.k_values:
        result = run_eval(args.base_url, k, param_dir, args.timeout)
        all_results.append(result)
    
    # Save combined summary
    summary_file = param_dir / "grid_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined summary to {summary_file}")
    
    # Print comparison table
    summarize_comparison(all_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
