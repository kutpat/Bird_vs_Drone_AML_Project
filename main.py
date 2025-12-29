# main.py

# - baseline: train/eval baseline SNN
# - aco: run Ant Colony Optimization to find good hyperparameters quickly
# - hybrid: retrain best ACO configuration fully + test
# - report: generate plots/tables for the written report
import argparse
import torch
import torch.nn as nn
import os

from src.data import DataConfig, get_dataloaders
from src.snn_model import SNNConfig, SpikingCNN
from src.train import TrainConfig, train_one_epoch, evaluate
from src.aco import ACOConfig, DiscreteACO, make_objective

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def run_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        data_dir="data",
        image_size=64,
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_cfg)

    snn_cfg = SNNConfig(
        num_classes=len(class_names),
        image_size=data_cfg.image_size,
        T=20,   # number of timesteps for rate coding
        beta=0.95,  # membrane decay in LIF
        threshold=1.0, # firing treshold
        dropout=0.2
    )
    model = SpikingCNN(snn_cfg).to(device)

    train_cfg = TrainConfig(epochs=10, lr=1e-3, device=str(device), lambda_spikes=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    os.makedirs("runs", exist_ok=True)

    for epoch in range(1, train_cfg.epochs + 1):
        # train_one_epoch returns loss/acc/spike statistics (spikes are a proxy for activity)
        tr = train_one_epoch(model, train_loader, optimizer, criterion, snn_cfg.T, device, train_cfg.lambda_spikes)
        va = evaluate(model, val_loader, criterion, snn_cfg.T, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} spikes {tr['avg_spikes_per_sample']:.1f} | "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} f1 {va['f1']:.3f} spikes {va['avg_spikes_per_sample']:.1f}"
        )

        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            torch.save(
                {"model_state": model.state_dict(), "snn_cfg": snn_cfg.__dict__, "classes": class_names},
                "runs/baseline_best.pt"
            )
    # Final test evaluation using best saved checkpoint
    ckpt = torch.load("runs/baseline_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te = evaluate(model, test_loader, criterion, snn_cfg.T, device)
    print(f"\nTEST | loss {te['loss']:.4f} acc {te['accuracy']:.3f} f1 {te['f1']:.3f} spikes {te['avg_spikes_per_sample']:.1f}")


def run_aco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        data_dir="data",
        image_size=64,
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_cfg)

    # Base config (will be copied and overridden by candidate hyperparams)
    base_snn_cfg = SNNConfig(
        num_classes=len(class_names),
        image_size=data_cfg.image_size,
        T=20,
        beta=0.95,
        threshold=1.0,
        dropout=0.2
    )

    # Keep search space small at first (fast + stable)
    search_space = [
        ("T", [10, 15, 20, 25]),
        ("lr", [1e-4, 3e-4, 1e-3]),
        ("beta", [0.90, 0.93, 0.95, 0.97]),
        ("threshold", [0.75, 1.0, 1.25]),
        ("dropout", [0.0, 0.2, 0.4]),
    ]

    aco_cfg = ACOConfig(
        n_iters=8,
        n_ants=6,
        rho=0.2,    # pheromone evaporation factor
        top_k=3,    # reinforce top candidates each iteration
        quick_epochs=3, # train only a few epochs per candidate
        max_train_batches=25,
        max_val_batches=15,
        lambda_spikes=0.0,   # if >0, fitness penalizes spikes (efficiency trade-off),
        seed=42
    )

    # objective_fn(params) -> (fitness, metrics)
    objective_fn = make_objective(train_loader, val_loader, device, base_snn_cfg, aco_cfg)

    aco = DiscreteACO(search_space, aco_cfg)
    history, best = aco.run(objective_fn, save_dir="runs", save_name="aco_search.json")

    print("\n=== ACO BEST ===")
    print("fitness:", best["fitness"])
    print("params:", best["params"])
    print("metrics:", best["metrics"])

def run_hybrid_from_aco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ACO best params
    with open("runs/aco_search.json", "r", encoding="utf-8") as f:
        aco_data = json.load(f)
    best_params = aco_data["best"]["params"]
    print("Loaded ACO best params:", best_params)

    # Data
    data_cfg = DataConfig(
        data_dir="data",
        image_size=64,
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_cfg)

    # Model config from ACO
    snn_cfg = SNNConfig(
        num_classes=len(class_names),
        image_size=data_cfg.image_size,
        T=int(best_params["T"]),
        beta=float(best_params["beta"]),
        threshold=float(best_params["threshold"]),
        dropout=float(best_params["dropout"]),
    )
    lr = float(best_params["lr"])

    model = SpikingCNN(snn_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Full training
    epochs = 12
    best_val_f1 = -1.0
    os.makedirs("runs", exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, snn_cfg.T, device, lambda_spikes=0.0)
        va = evaluate(model, val_loader, criterion, snn_cfg.T, device)

        print(
            f"[HYBRID] Epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} spikes {tr['avg_spikes_per_sample']:.1f} | "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} f1 {va['f1']:.3f} spikes {va['avg_spikes_per_sample']:.1f}"
        )

        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "snn_cfg": snn_cfg.__dict__,
                    "train_lr": lr,
                    "classes": class_names,
                    "aco_best_params": best_params,
                },
                "runs/hybrid_best.pt"
            )

    # Test best hybrid checkpoint
    ckpt = torch.load("runs/hybrid_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te = evaluate(model, test_loader, criterion, snn_cfg.T, device)
    print(f"\n[HYBRID TEST] loss {te['loss']:.4f} acc {te['accuracy']:.3f} f1 {te['f1']:.3f} spikes {te['avg_spikes_per_sample']:.1f}")


# --------- Report helpers (plots + CSV) ---------
def plot_confusion_matrix(cm, class_names, out_path: str, title: str):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    # Print raw counts inside the plot cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def plot_aco_convergence(in_path="runs/aco_search.json", out_path="results/aco_convergence.png"):
    """Plot best-per-iteration and global-best fitness over ACO iterations."""
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hist = data["history"]
    iters = [h["iter"] for h in hist]
    best_fit = [h["best_fitness"] for h in hist]
    global_best = [h["global_best_fitness"] for h in hist]

    plt.figure()
    plt.plot(iters, best_fit, label="Best in iteration")
    plt.plot(iters, global_best, label="Global best")
    plt.xlabel("ACO iteration")
    plt.ylabel("Fitness (val_f1 - λ*spikes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def load_model_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    snn_cfg_dict = ckpt["snn_cfg"]
    classes = ckpt["classes"]

    snn_cfg = SNNConfig(**snn_cfg_dict)
    model = SpikingCNN(snn_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, snn_cfg, classes


def run_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build same data split as training
    data_cfg = DataConfig(
        data_dir="data",
        image_size=64,
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_cfg)

    os.makedirs("results", exist_ok=True)

    # ACO convergence plot (only if ACO json exists)
    if os.path.exists("runs/aco_search.json"):
        plot_aco_convergence()

    # Load checkpoints
    baseline_model, baseline_cfg, baseline_classes = load_model_ckpt("runs/baseline_best.pt", device)
    hybrid_model, hybrid_cfg, hybrid_classes = load_model_ckpt("runs/hybrid_best.pt", device)

    # Predict labels on test set (also returns avg spikes/sample)
    from src.train import predict
    yb_true, yb_pred, yb_spikes = predict(baseline_model, test_loader, baseline_cfg.T, device)
    yh_true, yh_pred, yh_spikes = predict(hybrid_model, test_loader, hybrid_cfg.T, device)

    # Metrics
    from src.utils import compute_classification_metrics
    mb = compute_classification_metrics(yb_true, yb_pred)
    mh = compute_classification_metrics(yh_true, yh_pred)

    # Confusion matrices
    cm_b = confusion_matrix(yb_true, yb_pred)
    cm_h = confusion_matrix(yh_true, yh_pred)

    plot_confusion_matrix(cm_b, class_names, "results/cm_baseline.png", "Confusion Matrix - Baseline")
    plot_confusion_matrix(cm_h, class_names, "results/cm_hybrid.png", "Confusion Matrix - Hybrid (SNN+ACO)")

    # Comparison table (CSV)
    rows = [
        ["baseline", mb["accuracy"], mb["f1"], yb_spikes, baseline_cfg.T, baseline_cfg.beta, baseline_cfg.threshold, baseline_cfg.dropout],
        ["hybrid",   mh["accuracy"], mh["f1"], yh_spikes, hybrid_cfg.T, hybrid_cfg.beta, hybrid_cfg.threshold, hybrid_cfg.dropout],
    ]

    csv_path = "results/metrics_comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,accuracy,f1,avg_spikes_per_sample,T,beta,threshold,dropout\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("Saved:", csv_path)

    # Print summary + deltas
    acc_delta = mh["accuracy"] - mb["accuracy"]
    f1_delta = mh["f1"] - mb["f1"]
    spike_ratio = yh_spikes / max(1e-9, yb_spikes)

    print("\n=== REPORT SUMMARY (TEST) ===")
    print(f"Baseline: acc={mb['accuracy']:.3f} f1={mb['f1']:.3f} spikes={yb_spikes:.1f}")
    print(f"Hybrid:   acc={mh['accuracy']:.3f} f1={mh['f1']:.3f} spikes={yh_spikes:.1f}")
    print(f"Delta:    acc={acc_delta:+.3f} f1={f1_delta:+.3f} spike_ratio={spike_ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aco", "hybrid", "report"], default="report")
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline()
    elif args.mode == "aco":
        run_aco()
    elif args.mode == "hybrid":
        run_hybrid_from_aco()
    else:
        print("Running Report.")
        run_report()



if __name__ == "__main__":
    main()
