#!/usr/bin/env python3
# =============================================================================
# Generate paper-scale benchmarks: Orphanet-FQ274, DisGeNET-RD411, OMIM-Hop3
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dsrqs.benchmark import DATASET_META, generate_benchmark, generate_all_benchmarks


def main():
    p = argparse.ArgumentParser(description="Build DSRQS benchmark JSON files")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_META.keys()),
        choices=list(DATASET_META.keys()),
    )
    p.add_argument("--data_dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--orphanet", type=int, default=None, help="Override n queries")
    p.add_argument("--disgenet", type=int, default=None, help="Override n queries")
    p.add_argument("--omim", type=int, default=None, help="Override n queries")
    args = p.parse_args()

    overrides = {
        "orphanet_fq274": args.orphanet,
        "disgenet_rd411": args.disgenet,
        "omim_hop3": args.omim,
    }
    sizes = {k: v for k, v in overrides.items() if v is not None}

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for ds_key in args.datasets:
        n = sizes.get(ds_key, DATASET_META[ds_key]["n_queries"])
        print(f"\nBuilding {ds_key} ({n} queries)...")
        data = generate_benchmark(ds_key, n_queries=n, seed=args.seed)

        out_dir = data_dir / ds_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ds_key}_full.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        avg_rel = sum(len(q["relations"]) for q in data) / len(data)
        pce_pairs = sum(
            1 for q in data
            for r in {x["r"] for x in q["relations"]}
            if len({x["hop"] for x in q["relations"] if x["r"] == r}) > 1
            and len({x["label"] for x in q["relations"] if x["r"] == r}) > 1
        )
        print(f"  Saved {len(data)} queries → {out_path}")
        print(f"  Avg edges/query: {avg_rel:.1f}")
        print(f"  Queries with PCE-style label flip: {pce_pairs}")

    print("\nDone. Run: python main.py --mode prepare_data --dataset <name>")


if __name__ == "__main__":
    main()
