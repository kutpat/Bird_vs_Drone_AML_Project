# src/aco.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math
import random
import json
import os

import torch
import torch.nn as nn

from .snn_model import SNNConfig, SpikingCNN
from .train import train_one_epoch, evaluate
from .utils import set_seed


@dataclass
class ACOConfig:
    n_iters: int = 10
    n_ants: int = 8
    rho: float = 0.2          # evaporation
    q: float = 1.0            # pheromone deposit scale
    top_k: int = 3            # reinforce top-k ants each iter
    alpha: float = 1.0        # pheromone exponent
    seed: int = 42

    # Speed controls for candidate evaluation:
    quick_epochs: int = 3
    max_train_batches: int = 30
    max_val_batches: int = 20

    # Fitness: val_f1 - lambda_spikes * spikes
    lambda_spikes: float = 0.0   # e.g. 1e-5 .. 1e-4 if you want spike efficiency


class DiscreteACO:
    """
    Discrete ACO over categorical choices per hyperparameter.
    Each hyperparam i has choices values[i] with pheromones tau[i][j].
    """
    def __init__(self, search_space: List[Tuple[str, List[Any]]], cfg: ACOConfig):
        self.search_space = search_space
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.tau: List[List[float]] = []
        for _, vals in self.search_space:
            self.tau.append([1.0 for _ in vals])  # init pheromones

        self.best = {"fitness": -1e9, "params": None, "metrics": None}

    def _sample_index(self, tau_row: List[float]) -> int:
        # probabilities proportional to tau^alpha
        alpha = self.cfg.alpha
        weights = [max(1e-12, t) ** alpha for t in tau_row]
        s = sum(weights)
        r = self.rng.random() * s
        c = 0.0
        for i, w in enumerate(weights):
            c += w
            if r <= c:
                return i
        return len(weights) - 1

    def sample_params(self) -> Dict[str, Any]:
        params = {}
        for (name, vals), tau_row in zip(self.search_space, self.tau):
            j = self._sample_index(tau_row)
            params[name] = vals[j]
        return params

    def _params_to_indices(self, params: Dict[str, Any]) -> List[int]:
        idxs = []
        for name, vals in self.search_space:
            idxs.append(vals.index(params[name]))
        return idxs

    def update_pheromones(self, ranked_solutions: List[Tuple[float, Dict[str, Any]]]):
        # evaporate
        for i in range(len(self.tau)):
            for j in range(len(self.tau[i])):
                self.tau[i][j] *= (1.0 - self.cfg.rho)

        # reinforce top-k (rank-based deposit)
        k = min(self.cfg.top_k, len(ranked_solutions))
        for rank in range(k):
            fitness, params = ranked_solutions[rank]
            deposit = self.cfg.q * (k - rank) / k  # simple rank-based
            idxs = self._params_to_indices(params)
            for i, j in enumerate(idxs):
                self.tau[i][j] += deposit

    def run(
        self,
        objective_fn,
        save_dir: str = "runs",
        save_name: str = "aco_search.json"
    ):
        os.makedirs(save_dir, exist_ok=True)
        history = []

        cache: Dict[Tuple[Tuple[str, Any], ...], Tuple[float, Dict[str, float]]] = {}

        for it in range(1, self.cfg.n_iters + 1):
            solutions = []

            for _ in range(self.cfg.n_ants):
                params = self.sample_params()
                key = tuple(sorted(params.items()))

                if key in cache:
                    fitness, metrics = cache[key]
                else:
                    fitness, metrics = objective_fn(params)
                    cache[key] = (fitness, metrics)

                solutions.append((fitness, params, metrics))

                if fitness > self.best["fitness"]:
                    self.best = {"fitness": fitness, "params": params, "metrics": metrics}

            # rank descending by fitness
            solutions.sort(key=lambda x: x[0], reverse=True)
            ranked = [(s[0], s[1]) for s in solutions]
            self.update_pheromones(ranked)

            best_it = solutions[0]
            history.append({
                "iter": it,
                "best_fitness": best_it[0],
                "best_params": best_it[1],
                "best_metrics": best_it[2],
                "global_best_fitness": self.best["fitness"],
                "global_best_params": self.best["params"],
            })

            print(
                f"ACO iter {it:02d} | best fitness {best_it[0]:.4f} "
                f"(val_f1={best_it[2]['val_f1']:.3f}, spikes={best_it[2]['val_spikes']:.1f}) | "
                f"global best {self.best['fitness']:.4f}"
            )

            with open(os.path.join(save_dir, save_name), "w", encoding="utf-8") as f:
                json.dump({"history": history, "best": self.best}, f, indent=2)

        return history, self.best


def make_objective(
    train_loader,
    val_loader,
    device: torch.device,
    base_snn_cfg: SNNConfig,
    aco_cfg: ACOConfig,
):
    """
    Returns objective_fn(params) -> (fitness, metrics_dict)
    params will override selected SNN / training hyperparams.
    """
    criterion = nn.CrossEntropyLoss()

    def objective_fn(params: Dict[str, Any]):
        # Build model config
        snn_cfg = SNNConfig(**base_snn_cfg.__dict__)

        # Override hyperparams we allow ACO to tune
        if "T" in params: snn_cfg.T = int(params["T"])
        if "beta" in params: snn_cfg.beta = float(params["beta"])
        if "threshold" in params: snn_cfg.threshold = float(params["threshold"])
        if "dropout" in params: snn_cfg.dropout = float(params["dropout"])

        lr = float(params.get("lr", 1e-3))

        # Fresh model each evaluation
        model = SpikingCNN(snn_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Quick train
        for _ in range(aco_cfg.quick_epochs):
            train_one_epoch(
                model, train_loader, optimizer, criterion,
                T=snn_cfg.T, device=device,
                lambda_spikes=0.0,  # don't regularize here; fitness handles it
                max_batches=aco_cfg.max_train_batches
            )

        # Quick val
        va = evaluate(
            model, val_loader, criterion,
            T=snn_cfg.T, device=device,
            max_batches=aco_cfg.max_val_batches
        )

        val_f1 = float(va["f1"])
        val_spikes = float(va["avg_spikes_per_sample"])

        fitness = val_f1 - aco_cfg.lambda_spikes * val_spikes

        metrics = {
            "val_f1": val_f1,
            "val_acc": float(va["accuracy"]),
            "val_spikes": val_spikes
        }
        return fitness, metrics

    return objective_fn
