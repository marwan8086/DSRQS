# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Experiment tracking system for scientific experiments
#   Logs all runs, hyperparameters, and results
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict

import torch
import numpy as np


@dataclass
class RunMetadata:
    """Metadata for an experiment run."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    script_name: str = ""
    dataset: str = ""
    variant: str = ""
    seed: int = 0
    fold: int = 0
    git_commit: str = ""
    git_branch: str = ""
    hostname: str = field(default_factory=platform.node)
    python_version: str = field(default_factory=sys.version)
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_available: bool = field(default_factory=torch.cuda.is_available)
    cuda_version: str = ""
    gpu_name: str = ""
    
    def __post_init__(self):
        if torch.cuda.is_available():
            self.cuda_version = torch.version.cuda
            self.gpu_name = torch.cuda.get_device_name(0)


@dataclass
class RunResults:
    """Results from an experiment run."""
    duration_seconds: float = 0.0
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    training_history: Optional[List[Dict[str, Any]]] = None
    notes: str = ""


@dataclass
class Experiment:
    """Complete experiment data."""
    metadata: RunMetadata
    config: Dict[str, Any]
    results: Optional[RunResults] = None


class ExperimentTracker:
    """
    Track and log experiments for scientific reproducibility.
    """
    
    def __init__(
        self,
        experiment_dir: str = "runs",
        experiment_name: Optional[str] = None
    ):
        self.base_dir = Path(experiment_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self._start_time = None
        self._current_experiment = None
        
        # Write empty index if needed
        self._init_index()
    
    def _init_index(self):
        """Initialize or load experiment index."""
        index_path = self.base_dir / "index.json"
        if not index_path.exists():
            with open(index_path, "w") as f:
                json.dump({"experiments": []}, f)
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git commit and branch information."""
        try:
            import subprocess
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                universal_newlines=True
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                universal_newlines=True
            ).strip()
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": "", "branch": ""}
    
    def start_experiment(
        self,
        dataset: str,
        variant: str,
        seed: int,
        fold: int,
        config: Dict[str, Any]
    ) -> Experiment:
        """
        Start a new experiment run.
        """
        self._start_time = time.time()
        
        git_info = self._get_git_info()
        
        metadata = RunMetadata(
            script_name=Path(sys.argv[0]).name,
            dataset=dataset,
            variant=variant,
            seed=seed,
            fold=fold,
            git_commit=git_info["commit"],
            git_branch=git_info["branch"]
        )
        
        experiment = Experiment(metadata=metadata, config=config)
        self._current_experiment = experiment
        
        return experiment
    
    def end_experiment(
        self,
        final_metrics: Dict[str, float],
        best_metrics: Optional[Dict[str, float]] = None,
        training_history: Optional[List[Dict]] = None,
        notes: str = ""
    ) -> Experiment:
        """
        End the experiment and save results.
        """
        if self._current_experiment is None or self._start_time is None:
            raise RuntimeError("Experiment not started")
        
        duration = time.time() - self._start_time
        
        self._current_experiment.results = RunResults(
            duration_seconds=duration,
            final_metrics=final_metrics,
            best_metrics=best_metrics or final_metrics,
            training_history=training_history,
            notes=notes
        )
        
        # Save experiment
        self._save_experiment(self._current_experiment)
        
        return self._current_experiment
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to file."""
        # Create run directory
        run_name = (
            f"{experiment.metadata.dataset}_"
            f"{experiment.metadata.variant}_"
            f"seed{experiment.metadata.seed}_"
            f"fold{experiment.metadata.fold}"
        )
        run_dir = self.experiment_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full experiment
        exp_dict = {
            "metadata": asdict(experiment.metadata),
            "config": experiment.config,
            "results": asdict(experiment.results) if experiment.results else None
        }
        
        with open(run_dir / "experiment.json", "w") as f:
            json.dump(exp_dict, f, indent=2, default=str)
        
        # Update index
        self._update_index(experiment.metadata, run_dir)
    
    def _update_index(self, metadata: RunMetadata, run_dir: Path):
        """Update the experiment index."""
        index_path = self.base_dir / "index.json"
        
        with open(index_path, "r") as f:
            index = json.load(f)
        
        index["experiments"].append({
            "timestamp": metadata.timestamp,
            "dataset": metadata.dataset,
            "variant": metadata.variant,
            "seed": metadata.seed,
            "fold": metadata.fold,
            "path": str(run_dir.relative_to(self.base_dir)),
            "git_commit": metadata.git_commit
        })
        
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters (for compatibility with frameworks like W&B)."""
        pass  # Implement W&B/MLFlow integration if needed
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics during training."""
        pass  # Implement W&B/MLFlow integration if needed
    
    @staticmethod
    def load_experiment(experiment_path: str) -> Dict:
        """Load a saved experiment."""
        with open(Path(experiment_path) / "experiment.json", "r") as f:
            return json.load(f)
