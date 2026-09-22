"""
lstm — baseline temporal sign-classification model (section 6): a stacked
LSTM over the per-frame feature vector (ml/features/feature_vector.py),
followed by a linear classifier head over the final layer's hidden state.

This is deliberately the simplest architecture that can consume the
feature-vector sequences the CV pipeline (Phase 6-7) already produces --
configs/model.yaml leaves room for transformer/video_jepa alternatives
later (section 6), but only "lstm" is implemented as of Phase 9.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    input_size: int
    num_classes: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3


class LSTMSignClassifier(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """`sequence`: `(batch, seq_len, input_size)` -> `(batch, num_classes)` logits."""
        _, (final_hidden, _) = self.lstm(sequence)
        last_layer_hidden = final_hidden[-1]  # (batch, hidden_size)
        return self.classifier(last_layer_hidden)
