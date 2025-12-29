# src/snn_model.py
# Defines the spiking CNN architecture used for baseline + ACO-tuned training.
# Key ideas:
# - Input is a spike tensor over time: [T, B, C, H, W]
# - We simulate an SNN by looping over timesteps and updating membrane states
# - Output is based on spike counts (rate over time)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


@dataclass
class SNNConfig:
    num_classes: int = 2
    image_size: int = 64
    # spike / time settings
    T: int = 20
    # neuron params
    beta: float = 0.95          # membrane decay (how fast it decays)
    threshold: float = 1.0      # firing threshold
    # training regularization
    dropout: float = 0.2


class SpikingCNN(nn.Module):
    """
    Simple spiking CNN:
      Conv -> LIF -> Pool -> Conv -> LIF -> Pool -> FC -> LIF -> Readout
    (Leaky integrate fire)
    Forward expects input spikes: [T, B, C, H, W]
    Outputs logits-like activity: [B, num_classes]  (based on spike counts)
    Also returns spike stats for efficiency metrics.
    """
    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        # Electrical Charge builds up (membrane)
        # -> If the bucket is too full it fire
        # firing means "spike"

        # Surrogate gradient for backprop through spikes (spike fn is non-differentiable)
        # as spikes is binary we need a smooth approx.
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Block 1: conv + spiking neuron + pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, spike_grad=spike_grad)
        self.pool1 = nn.AvgPool2d(2)

        # Block 2: conv + spiking neuron + pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, spike_grad=spike_grad)
        self.pool2 = nn.AvgPool2d(2)

        # compute flattened size after pooling twice
        feat_h = cfg.image_size // 4
        feat_w = cfg.image_size // 4
        self.flat_dim = 32 * feat_h * feat_w

        # Classifier head: dropout -> linear -> output LIF
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(self.flat_dim, cfg.num_classes)
        self.lif_out = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, spike_grad=spike_grad)

    def forward(self, x_spk: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x_spk: [T, B, 3, H, W]  (spikes over time)
        """
        T, B = x_spk.shape[0], x_spk.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # Sum of output spikes across time (used for classification)
        spk_out_sum = torch.zeros((B, self.cfg.num_classes), device=x_spk.device)

        # spike counters for efficiency (used as a simple activity/efficiency proxy)
        total_spikes = torch.tensor(0.0, device=x_spk.device)

        # Temporal simulation loop: process one timestep at a time
        for t in range(T):
            x = x_spk[t] # [B,3,H,W] spikes at time t

            x = self.conv1(x)
            spk1, mem1 = self.lif1(x, mem1) # LIF outputs spikes + updated membrane
            x = self.pool1(spk1) # pool spikes (spike tensor stays sparse)

            x = self.conv2(x)
            spk2, mem2 = self.lif2(x, mem2)
            x = self.pool2(spk2)

            x = x.view(B, -1)
            x = self.dropout(x)
            x = self.fc(x)

            spk_out, mem_out = self.lif_out(x, mem_out)
            spk_out_sum += spk_out

            # count spikes (proxy for energy)
            total_spikes += spk1.sum() + spk2.sum() + spk_out.sum()
        # Class with higher spikes wins

        # Use spike counts as "logits"
        # (higher spike count -> stronger class prediction)
        logits = spk_out_sum / T

        stats = {
            "total_spikes": total_spikes,
            "avg_spikes_per_sample": total_spikes / B,
        }
        return logits, stats
