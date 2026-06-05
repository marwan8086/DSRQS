# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Standalone script for analyzing results from DSRQS experiments
#
# Copyright (c) 2026
# =============================================================================
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsrqs.statistics import (
    compare_variants,
    summarize_results,
    calculate_confidence_interval,
)
from src.dsrqs.visualization import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(description="Analyze DSRQS experiment results")
    parser.add_argument("--results_dir", required=True, help="Directory containing results")
    parser.add_argument("--output_dir", help="Directory to save analysis (default: same as results)")
    parser.add_argument("--baseline", default="cosine", help="Baseline variant for comparison")
    parser.add_argument("--metric", default="PCS", help="Primary metric for comparison")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DSRQS Results Analysis")
    print("=" * 80)
    
    # Load all results
    all_results = {}
    for result_file in results_dir.rglob("results.json"):
        variant_name = result_file.parent.name
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results[variant_name] = data["results"]
        print(f"Loaded: {variant_name} ({len(data['results'])} runs)")
    
    if not all_results:
        print("ERROR: No results found!")
        return
    
    # Generate summaries
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for variant_name, results in all_results.items():
        summary = summarize_results(results)
        print(f"\nVariant: {variant_name}")
        
        if args.metric in summary["metrics"]:
            m = summary["metrics"][args.metric]
            ci = m["confidence_interval_95"]
            print(f"  {args.metric}: {m['mean']:.4f} ± {m['std']:.4f}")
            print(f"  95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
            print(f"  Median: {m['median']:.4f}")
    
    # Comparative analysis
    if len(all_results) > 1 and args.baseline in all_results:
        print("\n" + "=" * 80)
        print("STATISTICAL COMPARISON")
        print("=" * 80)
        
        comparison = compare_variants(all_results, args.baseline, args.metric)
        
        for variant_name, comp in comparison["variants"].items():
            test = comp["statistical_test"]
            print(f"\n{variant_name} vs {args.baseline}:")
            print(f"  p-value: {test['p_value']:.4f}")
            print(f"  Effect size (r): {test['effect_size_r']:.4f}")
            print(f"  Significant: {'YES' if test['significant'] else 'NO'}")
            print(f"  Improvement: {comp['improvement']:.4f} ({comp['improvement_percent']:.2f}%)")
        
        # Save comparison
        with open(output_dir / "comparative_analysis.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)
    
    # Generate plots
    if args.plots:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        
        figures_dir = output_dir / "figures"
        generate_all_visualizations(results_dir, figures_dir)
        print(f"Plots saved to: {figures_dir}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
