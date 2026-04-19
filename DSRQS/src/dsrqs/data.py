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
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class KGRAGDataset(Dataset):

    def __init__(self, cfg: Dict, json_path: Path) -> None:
        self.cfg    = cfg
        self.device = cfg["device"]

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Data file not found: {json_path}")

        self.data: List[Dict] = json.load(open(json_path, encoding="utf-8"))

        model_name     = cfg["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = (
            AutoModel.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

        self._cache: Dict[str, torch.Tensor] = {}

        self.qid_to_gold: Dict[int, List] = {
            item["qid"]: item.get("gold_paths", [])
            for item in self.data
        }

    def __len__(self) -> int:
        return len(self.data)

    @torch.no_grad()
    def _encode(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)
        emb = self.encoder(**inputs).pooler_output
        emb = F.normalize(emb, p=2, dim=1).squeeze(0).cpu()
        self._cache[text] = emb
        return emb

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        item  = self.data[idx]
        q_emb = self._encode(item["query"])
        samples = []
        for rel in item["relations"]:
            r_emb = self._encode(rel["r"])
            samples.append({
                "q":     q_emb,
                "r":     r_emb,
                "hop":   rel["hop"],
                "label": rel["label"],
                "qid":   item["qid"],
                "edge":  (item["qid"], rel["r"], rel["hop"]),
            })
        return samples


def collate_fn(batch: List[List[Dict]]) -> Dict[str, Any] | None:
    flat = [s for b in batch for s in b]
    if not flat:
        return None
    return {
        "q":     torch.stack([x["q"]   for x in flat]),
        "r":     torch.stack([x["r"]   for x in flat]),
        "hop":   torch.tensor([x["hop"]   for x in flat]),
        "label": torch.tensor([x["label"] for x in flat]),
        "qid":   torch.tensor([x["qid"]   for x in flat]),
        "edges": [x["edge"] for x in flat],
    }