#!/usr/bin/env python3
# =============================================================================
# Optional: ingest user-provided Orphanet / DisGeNET / OMIM files into DSRQS JSON.
# Does NOT download restricted data — see DATA_SOURCES.md.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.dsrqs.benchmark import build_query_item, generate_benchmark


def load_disgenet_tsv(path: Path, limit: int = 500) -> List[Dict[str, str]]:
    """Parse DisGeNET gene-disease association TSV (geneID, diseaseName, ...)."""
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= limit:
                break
            disease = row.get("diseaseName") or row.get("disease_name") or ""
            gene = row.get("geneSymbol") or row.get("gene_symbol") or ""
            if disease and gene:
                rows.append({"disease": disease, "gene": gene})
    return rows


def build_disgenet_from_tsv(
    tsv_path: Path,
    n_queries: int = 411,
    seed: int = 42,
) -> List[Dict]:
    import random

    pairs = load_disgenet_tsv(tsv_path, limit=n_queries * 2)
    if not pairs:
        raise ValueError(f"No rows parsed from {tsv_path}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    intents = ["Etiology", "Treatment", "Phenotype", "Gene-Function"]
    data = []
    for qid, pair in enumerate(pairs[:n_queries]):
        data.append(
            build_query_item(
                qid,
                "disgenet_rd411",
                pair["disease"],
                pair["gene"],
                intents[qid % len(intents)],
                rng,
            )
        )
    return data


def main():
    p = argparse.ArgumentParser(
        description="Build DSRQS JSON from local KG files (user must obtain licenses)"
    )
    p.add_argument("--disgenet_tsv", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("data/disgenet_rd411/disgenet_rd411_full.json"))
    p.add_argument("--n", type=int, default=411)
    p.add_argument("--fallback_synthetic", action="store_true",
                   help="Use paper-faithful synthetic data if no TSV given")
    args = p.parse_args()

    if args.disgenet_tsv and args.disgenet_tsv.exists():
        print(f"Ingesting DisGeNET from {args.disgenet_tsv}...")
        data = build_disgenet_from_tsv(args.disgenet_tsv, n_queries=args.n)
    elif args.fallback_synthetic:
        print("No TSV — generating synthetic DisGeNET-RD411...")
        data = generate_benchmark("disgenet_rd411", n_queries=args.n)
    else:
        print("Provide --disgenet_tsv or --fallback_synthetic. See DATA_SOURCES.md.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} queries → {args.output}")


if __name__ == "__main__":
    main()
