# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   This file is part of the DSRQS framework for multi-hop reasoning over
#   biomedical knowledge graphs and retrieval-augmented generation (RAG).
#
# Copyright (c) 2026
# =============================================================================
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.model_selection import KFold

DATASETS = ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for ds_name in DATASETS:
        full_path = data_dir / ds_name / f"{ds_name}_full.json"
        if not full_path.exists():
            print(f"[SKIP] {full_path}")
            continue
        data    = json.load(open(full_path, encoding="utf-8"))
        out_dir = data_dir / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fold, (tr, te) in enumerate(kf.split(data)):
            json.dump([data[i] for i in tr],
                      open(out_dir / f"train_f{fold}.json", "w"))
            json.dump([data[i] for i in te],
                      open(out_dir / f"test_f{fold}.json",  "w"))
            print(f"[{ds_name}] fold {fold}: train={len(tr)} test={len(te)}")

    print("\nDone.")


if __name__ == "__main__":
    main()