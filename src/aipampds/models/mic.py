"""Checkpoint-compatible ETFC classifiers for E. coli and S. aureus."""

import torch
from torch import nn


class AddNorm(nn.Module):
    def __init__(self, normalized, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, inputs, residual):
        return self.ln(self.dropout(residual) + inputs)


class PositionWiseFFN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.dense1 = nn.Linear(inputs, hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden, outputs)

    def forward(self, inputs):
        return self.dense2(self.relu(self.dense1(inputs)))


class AttentionEncode(nn.Module):
    def __init__(self):
        super().__init__()
        self.at1 = nn.MultiheadAttention(192, num_heads=8, dropout=0.6)
        self.addNorm1 = AddNorm([50, 192], 0.4)
        self.FFN = PositionWiseFFN(64, 192, 64)


class FANEncode(nn.Module):
    def __init__(self):
        super().__init__()
        self.addNorm = AddNorm([1, 2304], 0.4)
        self.FFN = PositionWiseFFN(2304, 4608, 2304)


class ETFC(nn.Module):
    """Retain all checkpoint parameters, including layers unused by inference."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 192)
        self.conv1 = nn.Conv1d(192, 64, kernel_size=2)
        self.conv2 = nn.Conv1d(192, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(192, 64, kernel_size=4)
        self.conv4 = nn.Conv1d(192, 64, kernel_size=5)
        self.MaxPool1d = nn.MaxPool1d(kernel_size=5)
        self.attention_encode = AttentionEncode()
        self.fan = FANEncode()
        self.full3 = nn.Linear(2304, 1000)
        self.full4 = nn.Linear(1152, 500)
        self.full5 = nn.Linear(500, 256)
        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)
        self.dropout_layer = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embedded = self.embed(inputs).permute(0, 2, 1)
        first = self.MaxPool1d(torch.relu(self.conv1(embedded)))
        third = self.MaxPool1d(torch.relu(self.conv3(embedded)))
        features = self.dropout_layer(torch.cat([first, third], dim=-1))
        features = features.reshape(inputs.shape[0], -1)
        hidden = torch.relu(self.full4(features))
        hidden = torch.relu(self.full5(hidden))
        hidden = torch.relu(self.Flatten(hidden))
        return self.sigmoid(self.out(hidden))


class MIC:
    """Predict both MIC classifier scores with consistent batch dimensions."""

    def __init__(self, ecoli_path, saureus_path, device):
        self.device = device
        self.models = []
        for path in (ecoli_path, saureus_path):
            model = ETFC()
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
            self.models.append(model.to(device).eval())

    def predict(self, sequences: list[str]) -> tuple[list[float], list[float]]:
        """Use the original X-prefixed amino acid vocabulary and right padding."""
        vocabulary = "XACDEFGHIKLMNPQRSTVWY"
        encoded = [
            [vocabulary.index(amino_acid) for amino_acid in sequence] + [0] * (50 - len(sequence))
            for sequence in sequences
        ]
        inputs = torch.tensor(encoded, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            scores = [model(inputs).reshape(-1).cpu().tolist() for model in self.models]
        return scores[0], scores[1]
