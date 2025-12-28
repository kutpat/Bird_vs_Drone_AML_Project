# streamlit_app.py
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms

from src.snn_model import SNNConfig, SpikingCNN
from src.train import encode_rate


st.set_page_config(page_title="Bird vs Drone (SNN)", layout="centered")

st.title("🕊️ vs 🚁 - Bird vs Drone Classifier")
st.caption("Upload an image and the SNN predicts whether it is a **bird** or a **drone**.")


RUNS_DIR = "runs"

st.markdown(
    """
    <style>
    /* Remove the internal scroll container so the whole page scrolls */
    section.main {
        overflow: visible !important;
    }
    .block-container {
        overflow: visible !important;
    }

    /* Ensure the app uses the browser page scroll */
    html, body, [data-testid="stAppViewContainer"] {
        height: auto !important;
        overflow: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def list_checkpoints(runs_dir: str = RUNS_DIR):
    """Return (display_names, path_by_display). Auto-detect *.pt in runs/."""
    paths = sorted(glob.glob(os.path.join(runs_dir, "*.pt")))
    path_by_display = {}

    def pretty_name(p: str) -> str:
        fn = os.path.basename(p)

        # Nice labels for common ones
        if fn == "baseline_best.pt":
            return "Baseline (best)"
        if fn == "hybrid_best.pt":
            return "Hybrid (best, ACO)"

        # Heuristic labeling
        low = fn.lower()
        if "baseline" in low:
            return f"Baseline: {fn}"
        if "hybrid" in low:
            return f"Hybrid: {fn}"
        if "aco" in low:
            return f"ACO: {fn}"
        return fn

    for p in paths:
        disp = pretty_name(p)
        # avoid collisions
        if disp in path_by_display:
            disp = disp + f" ({len(path_by_display)+1})"
        path_by_display[disp] = p

    return list(path_by_display.keys()), path_by_display


@st.cache_resource
def load_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    snn_cfg = SNNConfig(**ckpt["snn_cfg"])
    class_names = ckpt["classes"]

    model = SpikingCNN(snn_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, snn_cfg, class_names, device


def get_preprocess(image_size: int):
    # MUST match training normalization
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def predict_image(model, snn_cfg: SNNConfig, device, img: Image.Image):
    preprocess = get_preprocess(snn_cfg.image_size)

    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]

    x_spk = encode_rate(x, T=snn_cfg.T)          # [T,1,3,H,W]
    logits, stats = model(x_spk)                 # logits: [1, num_classes]

    probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]
    pred_idx = int(torch.argmax(probs).item())

    confidence = float(probs[pred_idx].item())
    avg_spikes = float(stats["avg_spikes_per_sample"].item())

    return pred_idx, probs.detach().cpu().numpy(), confidence, avg_spikes


# --- Sidebar: checkpoint selector ---
st.sidebar.header("Model")

displays, path_by_display = list_checkpoints(RUNS_DIR)
if not displays:
    st.error("No checkpoints found in `runs/`. Please train first to create `runs/baseline_best.pt` and/or `runs/hybrid_best.pt`.")
    st.stop()

# default selection preference: hybrid > baseline > first
default_disp = None
for candidate in ["Hybrid (best, ACO)", "Baseline (best)"]:
    if candidate in displays:
        default_disp = candidate
        break
if default_disp is None:
    default_disp = displays[0]

selected_disp = st.sidebar.selectbox(
    "Select checkpoint",
    options=displays,
    index=displays.index(default_disp),
)

ckpt_path = path_by_display[selected_disp]
model, snn_cfg, class_names, device = load_model(ckpt_path)

st.sidebar.write(f"Device: **{device}**")
st.sidebar.caption(f"Loaded: `{ckpt_path}`")
with st.sidebar.expander("Loaded SNN config"):
    st.json({
        "T": snn_cfg.T,
        "beta": snn_cfg.beta,
        "threshold": snn_cfg.threshold,
        "dropout": snn_cfg.dropout,
        "image_size": snn_cfg.image_size,
    })

st.divider()


# --- Main UI ---
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

def pretty_label(lbl: str) -> str:
    return "🕊️ bird" if lbl.lower() == "bird" else ("🚁 drone" if lbl.lower() == "drone" else lbl)

if uploaded_files:
    for uf in uploaded_files:
        img = Image.open(uf)

        left, right = st.columns([1.05, 1.0], vertical_alignment="top")
        with left:
            st.image(img, caption=uf.name, use_container_width=True)

        pred_idx, probs, conf, avg_spikes = predict_image(model, snn_cfg, device, img)
        pred_label = class_names[pred_idx]

        with right:
            # "Card" container
            with st.container():
                st.subheader("Result")

                # Big label + confidence
                st.markdown(f"### **{pretty_label(pred_label)}**")
                st.metric("Confidence", f"{conf:.3f}")

                # Confidence bar
                st.progress(min(max(conf, 0.0), 1.0))

                # Spike proxy metric
                st.metric("Avg spikes/sample (proxy)", f"{avg_spikes:,.1f}")

                # Probabilities breakdown
                st.write("**Probabilities**")
                for i, name in enumerate(class_names):
                    st.write(f"- {pretty_label(name)}: `{float(probs[i]):.3f}`")

        st.divider()
else:
    st.info("Upload an image to get a prediction.")
