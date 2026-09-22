"""
fingerspelling.model -- a small classifier over one static hand-landmark
feature vector (63 = 21 points x 3 coords), not a sequence model.

USL_alphabet_train is one photo per handshape (Phase: isolated dactyl
letters), unlike ml/models/lstm.py's temporal window over a sequence of
frames for isolated *words*. A 2-layer MLP is deliberately small: with
several classes down to 24-33 real samples total, a bigger model would
just memorize rather than learn -- see PROJECT_STATUS.md for the honest
per-class sample counts and what that does to reported accuracy for those
classes.
"""
from __future__ import annotations

import torch
from torch import nn

NUM_HAND_LANDMARKS = 21
INPUT_SIZE = NUM_HAND_LANDMARKS * 3


class FingerspellingMLP(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 64, input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
