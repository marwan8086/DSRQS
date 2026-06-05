# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   DSRQS training & evaluation — 5-fold CV × 5 seeds (paper §5)
#   Enhanced for production use with better logging, result saving, and robustness
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dsrqs.data import KGRAGDataset, collate_fn, evaluate_query_retention
from src.dsrqs.losses import DSRQSLoss, SupConMarginLoss
from src.dsrqs.metrics import compute_all_metrics
from src.dsrqs.model import build_scorer
from src.dsrqs.logger import get_logger
from src.dsrqs.utils import load_config, set_seed

logger = get_logger()
logger.info("Starting DSRQS framework ...")

DATASETS = ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]
VARIANTS = {
    "dsrqs": "dsrqs",
    "b6": "dsrqs_no_dc",
    "b5": "bilinear_supcon",
    "b4": "bilinear",
    "b3": "cosine",
}


def prepare_data_splits(cfg: Dict) -> None:
    """Prepare 5-fold cross-validation splits for all datasets."""
    data_dir = Path(cfg["paths"]["data_dir"])
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for ds_name in DATASETS:
        full_path = data_dir / ds_name / f"{ds_name}_full.json"
        if not full_path.exists():
            print(f"[SKIP] {full_path} not found.")
            continue
        with open(full_path, encoding="utf-8") as fh:
            data = json.load(fh)
        out_dir = data_dir / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            with open(out_dir / f"train_f{fold}.json", "w", encoding="utf-8") as fh:
                json.dump([data[i] for i in train_idx], fh)
            with open(out_dir / f"test_f{fold}.json", "w", encoding="utf-8") as fh:
                json.dump([data[i] for i in test_idx], fh)
            print(f"  [{ds_name}] fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
    print("\nDone.")


def _evaluate_loader(model, test_dl, test_ds, loss_fn, device, cfg, dataset_name):
    """Run one validation pass; return metrics and mean latency per edge (ms)."""
    model.eval()
    theta = cfg["eval"]["threshold"]

    query_preds: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"scores": [], "labels": [], "hops": [], "r_names": []}
    )

    t0 = time.perf_counter()
    n_edges = 0
    with torch.no_grad():
        for batch in test_dl:
            if batch is None:
                continue
            batch_gpu = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            logits = model.forward_logits(
                batch_gpu["q"], batch_gpu["r"], batch_gpu["hop"]
            )
            scores = torch.sigmoid(logits).cpu().numpy()
            n_edges += len(scores)

            for i in range(len(scores)):
                qid = int(batch["qid"][i].item())
                query_preds[qid]["scores"].append(float(scores[i]))
                query_preds[qid]["labels"].append(int(batch["label"][i].item()))
                query_preds[qid]["hops"].append(int(batch["hop"][i].item()))
                query_preds[qid]["r_names"].append(batch["r_name"][i])

    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(n_edges, 1)

    all_preds, all_labels, all_hops = [], [], []
    all_gold_paths, all_retained = [], []

    for qid, buf in sorted(query_preds.items()):
        all_preds.extend(buf["scores"])
        all_labels.extend(buf["labels"])
        all_hops.extend(buf["hops"])
        all_gold_paths.append(test_ds.qid_to_gold.get(qid, []))
        all_retained.append(
            evaluate_query_retention(
                qid, buf["scores"], buf["r_names"], buf["hops"], theta
            )
        )

    metrics = compute_all_metrics(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_hops),
        all_gold_paths,
        all_retained,
        theta,
        dataset=dataset_name,
    )
    metrics["latency_ms"] = float(latency_ms)
    return metrics


def run_fold(cfg, train_path, test_path, fold, seed, dataset_name, variant, log_dir: Optional[Path] = None):
    """Run training and evaluation for a single fold with detailed logging."""
    set_seed(seed)
    device = cfg["device"]
    lam_dc = 0.0 if variant in ("b6", "dsrqs_no_dc", "b4", "bilinear", "b3", "cosine") else cfg["loss"]["lambda_dc"]
    cfg_run = {**cfg, "loss": {**cfg["loss"], "lambda_dc": lam_dc}, "variant": variant}

    train_ds = KGRAGDataset(cfg_run, train_path, train=True)
    test_ds  = KGRAGDataset(
        cfg_run, test_path, train=False, relation_vocab=train_ds.relation_vocab
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_scorer(cfg_run, variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = DSRQSLoss(cfg_run)
    supcon = SupConMarginLoss(cfg_run) if variant == "bilinear_supcon" else None

    ckpt_dir = (
        Path(cfg["paths"]["checkpoint_dir"])
        / dataset_name
        / variant
        / f"seed{seed}"
        / f"fold{fold}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    best_pcs = -1.0
    best_state = None
    best_metrics = None
    patience = cfg["train"].get("early_stop_patience", 5)
    stale_epochs = 0
    training_history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Training loop with loss tracking
        for batch in tqdm(
            train_dl,
            desc=f"{variant} S{seed} F{fold} E{epoch+1:02d}",
            leave=False,
        ):
            if batch is None:
                continue
            batch_gpu = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            logits = model.forward_logits(
                batch_gpu["q"], batch_gpu["r"], batch_gpu["hop"]
            )
            parts = loss_fn(
                logits,
                batch_gpu["label"],
                batch_gpu["hop"],
                batch_gpu["qid"],
                batch_gpu["rid"],
            )
            loss = parts["loss"]
            if supcon is not None:
                loss = loss + 0.1 * supcon(logits, batch_gpu["label"])

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        metrics = _evaluate_loader(
            model, test_dl, test_ds, loss_fn, device, cfg_run, dataset_name
        )

        # Log and save history
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_metrics": metrics,
        }
        training_history.append(epoch_stats)
        logger.info(
            f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f}, "
            f"PCS={metrics['PCS']:.4f}, F1={metrics['Fe1']:.4f}, H={metrics['H']:.1f}%"
        )

        # Check for best model
        if metrics["PCS"] > best_pcs:
            best_pcs = metrics["PCS"]
            best_metrics = metrics
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "best.pt")
            stale_epochs = 0
            logger.info(f"  → New best PCS: {best_pcs:.4f}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info(f"  → Early stopping after {epoch+1} epochs")
                break

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training history
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        history_file = log_dir / f"history_seed{seed}_fold{fold}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2, default=str)

    final_metrics = _evaluate_loader(
        model, test_dl, test_ds, loss_fn, device, cfg_run, dataset_name
    )
    return final_metrics, best_metrics


def save_results(results: List[Dict], output_path: Path, metadata: Dict):
    """Save comprehensive results to JSON file."""
    output = {
        "metadata": metadata,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "pcs_mean": float(np.mean([r["PCS"] for r in results])),
            "pcs_std": float(np.std([r["PCS"] for r in results])),
            "f1_mean": float(np.mean([r["Fe1"] for r in results])),
            "f1_std": float(np.std([r["Fe1"] for r in results])),
            "h_mean": float(np.mean([r["H"] for r in results])),
            "h_std": float(np.std([r["H"] for r in results])),
            "latency_mean": float(np.mean([r["latency_ms"] for r in results])),
            "latency_std": float(np.std([r["latency_ms"] for r in results])),
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def main():
    p = argparse.ArgumentParser(description="DSRQS — production-ready training & evaluation")
    p.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    p.add_argument("--dataset", required=True, choices=DATASETS, help="Dataset to use")
    p.add_argument(
        "--variant",
        default="dsrqs",
        choices=list(VARIANTS.keys()),
        help="Model variant: dsrqs | b6 | b5 | b4 | b3",
    )
    p.add_argument(
        "--mode",
        default="full_eval",
        choices=["prepare_data", "full_eval", "single_run", "all_baselines"],
        help="Operation mode",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (for single_run)")
    p.add_argument("--fold", type=int, default=0, help="Fold number (for single_run)")
    p.add_argument("--output_dir", default="results", help="Directory to save results")
    args = p.parse_args()
    cfg = load_config(args.config)

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.variant}_{timestamp}"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  DSRQS | Dataset: {args.dataset} | Variant: {args.variant} | Mode: {args.mode}")
    print(f"  Device: {cfg['device']} | Output: {output_dir}")
    print(f"{'='*80}\n")

    if args.mode == "prepare_data":
        prepare_data_splits(cfg)
        return

    data_dir = Path(cfg["paths"]["data_dir"]) / args.dataset
    variant = VARIANTS[args.variant]

    def run_one(seed, fold):
        return run_fold(
            cfg,
            data_dir / f"train_f{fold}.json",
            data_dir / f"test_f{fold}.json",
            fold,
            seed,
            args.dataset,
            variant,
            log_dir=log_dir,
        )

    if args.mode == "single_run":
        res, best_res = run_one(args.seed, args.fold)
        print(
            f"\nFinal Result → PCS={res['PCS']:.4f}  Fe1={res['Fe1']:.4f}  "
            f"H={res['H']:.1f}%  Δα={res['delta_alpha']:.4f}  "
            f"T={res['latency_ms']:.2f}ms/edge"
        )
        # Save single run results
        save_results([res], output_dir / "single_run_results.json", {
            "dataset": args.dataset,
            "variant": args.variant,
            "seed": args.seed,
            "fold": args.fold,
        })
        return

    variants_to_run = [variant]
    if args.mode == "all_baselines":
        variants_to_run = list(VARIANTS.values())

    for v in variants_to_run:
        print(f"\n--- Variant: {v} ---")
        results = []
        for seed in range(5):
            for fold in range(5):
                print(f">>> Seed {seed}  Fold {fold}")
                res, best_res = run_fold(
                    cfg,
                    data_dir / f"train_f{fold}.json",
                    data_dir / f"test_f{fold}.json",
                    fold,
                    seed,
                    args.dataset,
                    v,
                    log_dir=log_dir / v,
                )
                results.append(res)
                print(
                    f"    PCS={res['PCS']:.4f}  Fe1={res['Fe1']:.4f}  "
                    f"H={res['H']:.1f}%  Δα={res['delta_alpha']:.4f}\n"
                )

        # Save results for this variant
        variant_output_dir = output_dir / v
        save_results(
            results,
            variant_output_dir / "results.json",
            {
                "dataset": args.dataset,
                "variant": v,
                "mode": args.mode,
            }
        )

        # Print summary
        pcs_vals = [r["PCS"] for r in results]
        print(f"\n{'='*80}")
        print(f"  {args.dataset} / {v}  (5×5 CV)")
        print(f"  PCS : {np.mean(pcs_vals):.4f} ± {np.std(pcs_vals):.4f}")
        print(f"  F1  : {np.mean([r['Fe1'] for r in results]):.4f} ± {np.std([r['Fe1'] for r in results]):.4f}")
        print(f"  H   : {np.mean([r['H'] for r in results]):.1f} % ± {np.std([r['H'] for r in results]):.1f} %")
        print(f"  T   : {np.mean([r['latency_ms'] for r in results]):.2f} ms/edge ± {np.std([r['latency_ms'] for r in results]):.2f} ms/edge")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
