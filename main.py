# main.py
import argparse
import torch
import torch.nn as nn
import os

from src.data import DataConfig, get_dataloaders
from src.snn_model import SNNConfig, SpikingCNN
from src.train import TrainConfig, train_one_epoch, evaluate
from src.aco import ACOConfig, DiscreteACO, make_objective


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
        T=20,
        beta=0.95,
        threshold=1.0,
        dropout=0.2
    )
    model = SpikingCNN(snn_cfg).to(device)

    train_cfg = TrainConfig(epochs=10, lr=1e-3, device=str(device), lambda_spikes=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    os.makedirs("runs", exist_ok=True)

    for epoch in range(1, train_cfg.epochs + 1):
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
        rho=0.2,
        top_k=3,
        quick_epochs=3,
        max_train_batches=25,
        max_val_batches=15,
        lambda_spikes=0.0,   # start with 0; later set e.g. 1e-5 to favor efficiency
        seed=42
    )

    objective_fn = make_objective(train_loader, val_loader, device, base_snn_cfg, aco_cfg)

    aco = DiscreteACO(search_space, aco_cfg)
    history, best = aco.run(objective_fn, save_dir="runs", save_name="aco_search.json")

    print("\n=== ACO BEST ===")
    print("fitness:", best["fitness"])
    print("params:", best["params"])
    print("metrics:", best["metrics"])

import json
import matplotlib.pyplot as plt

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

    # Full training (you can increase to 20–30 later)
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aco", "hybrid"], default="hybrid")
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline()
    elif args.mode == "aco":
        run_aco()
    else:
        print("Running Hybrid.")
        run_hybrid_from_aco()

if __name__ == "__main__":
    main()
