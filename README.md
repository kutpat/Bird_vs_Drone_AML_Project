# SNN + Ant Colony Optimization (ACO) — Bird vs Drone Classifier

GitHub Repository Link: https://github.com/kutpat/Bird_vs_Drone_AML_Project

This project implements a **Spiking Neural Network (SNN)** for **bird vs drone** image classification and a hybrid approach where **Ant Colony Optimization (ACO)** tunes key SNN hyperparameters.  
A bonus **Streamlit app** provides an interactive demo for uploading images and getting predictions.

---

## Project structure

```
project/
  data/                   # dataset root (folders: bird/, drone/)
  src/
    data.py               # dataloaders, transforms, stratified splits
    snn_model.py          # SNN architecture (Conv + LIF + spike-count readout)
    train.py              # training/evaluation + spike encoding
    aco.py                # ant colony optimizer for hyperparameter tuning
    utils.py              # seed + metrics helpers
  runs/                   # saved checkpoints (.pt) + ACO logs
  results/                # generated figures + tables
  main.py                 # entry point (baseline / aco / hybrid / report)
  streamlit_app.py        # bonus demo app
  requirements.txt
  README.md
```

---

## Setup

### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> **PyTorch note:** If you want GPU support, install PyTorch using the official instructions for your CUDA version.

---

## Dataset

Place your images like this:

```
data/
  bird/
    img1.jpg
    img2.jpg
    ...
  drone/
    img1.jpg
    img2.jpg
    ...
```

The code uses `torchvision.datasets.ImageFolder`, so folder names become class labels.

---

## Training & evaluation

All commands are run from the **project root**.

### 1) Train baseline model (approx. 1h runtime on my CPU)
```bash
python main.py --mode baseline
```

Saves:
- `runs/baseline_best.pt`

Prints:
- epoch-by-epoch train/val metrics
- final test metrics

### 2) Run ACO hyperparameter search (approx. 5h runtime on my CPU)
```bash
```bash 
python main.py --mode aco
```

Saves:
- `runs/aco_search.json` (ACO history + best config)

### 3) Retrain the best ACO configuration (hybrid) (approx. 1h runtime on my CPU)
```bash
python main.py --mode hybrid
```

Saves:
- `runs/hybrid_best.pt`

### 4) Generate report artifacts (plots + CSV)
```bash
python main.py --mode report
```

Creates:
- `results/aco_convergence.png`
- `results/cm_baseline.png`
- `results/cm_hybrid.png`
- `results/metrics_comparison.csv`

---

## Streamlit demo app (Bonus Task)

### Run the app
```bash
streamlit run streamlit_app.py
```

Features:
- Upload one or multiple images
- Dropdown to switch checkpoints (auto-detected from `runs/*.pt`)
- Prediction label + confidence
- Spike activity proxy (avg spikes/sample)

**Checkpoint requirement:**  
The app expects at least one `.pt` checkpoint in `runs/` (e.g., `runs/hybrid_best.pt` or `runs/baseline_best.pt`).

---

## Reproducibility

- Stratified splits: `val_split=0.15`, `test_split=0.15`
- Seed: `42`
- Input size: `64×64`
- Baseline config (example): `T=20`, `beta=0.95`, `threshold=1.0`, `dropout=0.2`, `lr=1e-3`
- ACO best config (example run): `T=25`, `beta=0.93`, `threshold=0.75`, `dropout=0.0`, `lr=1e-3`

---

## Troubleshooting

**"No checkpoints found in runs/" (Streamlit):**  
Check if you installed my given .pt or run baseline or hybrid first to generate `runs/*.pt`.

**Windows dataloader issues:**  
If you get dataloader worker errors, reduce `num_workers` in `DataConfig` (e.g., set to 0 or 1).

---

## License
For academic coursework / educational use.
