# main.py
import os
import torch
import torch.nn as nn

from src.data import DataConfig, get_dataloaders
from src.snn_model import SNNConfig, SpikingCNN
from src.train import TrainConfig, train_one_epoch, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Model
    snn_cfg = SNNConfig(
        num_classes=len(class_names),
        image_size=data_cfg.image_size,
        T=20,
        beta=0.95,
        threshold=1.0,
        dropout=0.2
    )
    model = SpikingCNN(snn_cfg).to(device)

    # Train
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

    # Final test with best checkpoint
    ckpt = torch.load("runs/baseline_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te = evaluate(model, test_loader, criterion, snn_cfg.T, device)
    print(f"\nTEST | loss {te['loss']:.4f} acc {te['accuracy']:.3f} f1 {te['f1']:.3f} spikes {te['avg_spikes_per_sample']:.1f}")

if __name__ == "__main__":
    main()
