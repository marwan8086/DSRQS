
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
# -*- coding: utf-8 -*-
from src.dsrqs.tracker import ExperimentTracker

from __future__ import annotations
from src.dsrqs.logger import get_logger


import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dsrqs.data    import KGRAGDataset, collate_fn
from src.dsrqs.losses  import DSRQSLoss
from src.dsrqs.metrics import compute_all_metrics
from src.dsrqs.model   import DSRQS
from src.dsrqs.utils   import load_config, set_seed

logger = get_logger()
logger.info("Starting ...")

DATASETS = ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]


def prepare_data_splits(cfg: Dict) -> None:
    data_dir = Path(cfg["paths"]["data_dir"])
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for ds_name in DATASETS:
        full_path = data_dir / ds_name / f"{ds_name}_full.json"
        if not full_path.exists():
            print(f"[SKIP] {full_path} not found.")
            continue
        data = json.load(open(full_path, encoding="utf-8"))
        out_dir = data_dir / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            train_split = [data[i] for i in train_idx]
            test_split  = [data[i] for i in test_idx]
            json.dump(train_split, open(out_dir / f"train_f{fold}.json", "w"))
            json.dump(test_split,  open(out_dir / f"test_f{fold}.json",  "w"))
            print(f"  [{ds_name}] fold {fold}: train={len(train_split)}, test={len(test_split)}")
    print("\nDone.")


def run_fold(cfg, train_path, test_path, fold, seed):
    set_seed(seed)
    device = cfg["device"]

    train_ds = KGRAGDataset(cfg, train_path)
    test_ds  = KGRAGDataset(cfg, test_path)

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          shuffle=True,  collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=cfg["train"]["batch_size"],
                          shuffle=False, collate_fn=collate_fn)

    model   = DSRQS(cfg).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = DSRQSLoss(cfg)
    theta   = cfg["eval"]["threshold"]

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / f"seed{seed}" / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_pcs   = -1.0
    best_state = None

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tqdm(train_dl, desc=f"S{seed} F{fold} E{epoch+1:02d}", leave=False):
            if batch is None:
                continue
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            scores = model(batch_gpu["q"], batch_gpu["r"], batch_gpu["hop"])
            loss   = loss_fn(scores, batch_gpu["label"], batch_gpu["qid"])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        all_preds, all_labels, all_gold_paths, all_pred_edges = [], [], [], []
        with torch.no_grad():
            for batch in test_dl:
                if batch is None:
                    continue
                batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                scores = model(batch_gpu["q"], batch_gpu["r"], batch_gpu["hop"])
                all_preds.extend(scores.cpu().tolist())
                all_labels.extend(batch["label"].tolist())
                for qid_tensor in batch["qid"]:
                    all_gold_paths.append(test_ds.qid_to_gold.get(qid_tensor.item(), []))
                batch_pred_edges = [batch["edges"][i]
                                    for i, s in enumerate(scores.cpu().tolist())
                                    if s >= theta]
                all_pred_edges.append(batch_pred_edges)

        metrics = compute_all_metrics(np.array(all_preds), np.array(all_labels),
                                      all_gold_paths, all_pred_edges, theta)
        if metrics["PCS"] > best_pcs:
            best_pcs   = metrics["PCS"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode",    default="full_eval",
                   choices=["prepare_data", "full_eval", "single_run"])
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--fold",    type=int, default=0)
    args = p.parse_args()
    cfg  = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  DSRQS | Dataset: {args.dataset} | Mode: {args.mode}")
    print(f"  Device: {cfg['device']}")
    print(f"{'='*60}\n")

    if args.mode == "prepare_data":
        prepare_data_splits(cfg)
        return

    data_dir = Path(cfg["paths"]["data_dir"]) / args.dataset

    if args.mode == "single_run":
        res = run_fold(cfg,
                       data_dir / f"train_f{args.fold}.json",
                       data_dir / f"test_f{args.fold}.json",
                       fold=args.fold, seed=args.seed)
        print(f"\nResult → PCS={res['PCS']:.3f}  Fe1={res['Fe1']:.3f}  H={res['H']:.1f}%")
        return

    results = []
    for seed in range(5):
        for fold in range(5):
            print(f">>> Seed {seed}  Fold {fold}")
            res = run_fold(cfg,
                           data_dir / f"train_f{fold}.json",
                           data_dir / f"test_f{fold}.json",
                           fold=fold, seed=seed)
            results.append(res)
            print(f"    PCS={res['PCS']:.3f}  Fe1={res['Fe1']:.3f}  H={res['H']:.1f}%\n")

    pcs_vals = [r["PCS"] for r in results]
    h_vals   = [r["H"]   for r in results]
    print(f"\n{'='*60}")
    print(f"  FINAL {args.dataset}  (5×5 CV)")
    print(f"  PCS : {np.mean(pcs_vals):.3f} ± {np.std(pcs_vals):.3f}")
    print(f"  H   : {np.mean(h_vals):.1f} %")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()