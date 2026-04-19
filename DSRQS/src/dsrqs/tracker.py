# =============================================================================
# Author: Marwan Dhifallah*
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology 
#
# Description:
#   This file is part of the DSRQS framework for multi-hop reasoning over
#   biomedical knowledge graphs and retrieval-augmented generation (RAG).
#
# Copyright (c) 2026
# =============================================================================
import json
from datetime import datetime
from pathlib import Path


class ExperimentTracker:

    def __init__(self, cfg):
        self.run_dir = Path(cfg["paths"]["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.run_dir / f"run_{self.run_id}.json"

        self.data = {
            "run_id": self.run_id,
            "metrics": [],
            "config": cfg
        }

    def log_epoch(self, epoch, metrics):
        entry = {"epoch": epoch, **metrics}
        self.data["metrics"].append(entry)
        self._save()

    def log_final(self, results):
        self.data["final"] = results
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)