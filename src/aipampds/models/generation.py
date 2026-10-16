"""GPT peptide sampling matching the platform's three generator checkpoints."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(256, 256)
        self.query = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(256, 256)
        self.register_buffer("mask", torch.tril(torch.ones(190, 190)).view(1, 1, 190, 190))

    def forward(self, inputs):
        batch_size, length, embedding_size = inputs.shape
        keys = self.key(inputs).view(batch_size, length, 8, 32).transpose(1, 2)
        queries = self.query(inputs).view(batch_size, length, 8, 32).transpose(1, 2)
        values = self.value(inputs).view(batch_size, length, 8, 32).transpose(1, 2)
        attention = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(32))
        attention = attention.masked_fill(self.mask[:, :, :length, :length] == 0, float("-inf"))
        attention = self.attn_drop(functional.softmax(attention, dim=-1))
        output = (attention @ values).transpose(1, 2).contiguous()
        output = output.view(batch_size, length, embedding_size)
        return self.resid_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(
            nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256), nn.Dropout(0.1)
        )

    def forward(self, inputs):
        hidden = inputs + self.attn(self.ln1(inputs))
        return hidden + self.mlp(self.ln2(hidden))


class GPT(nn.Module):
    """Eight-layer, eight-head GPT with the original state dictionary layout."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(21, 256)
        self.type_emb = nn.Embedding(2, 256)
        self.pos_emb = nn.Parameter(torch.zeros(1, 190, 256))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block() for _ in range(8)])
        self.ln_f = nn.LayerNorm(256)
        self.head = nn.Linear(256, 21, bias=False)

    def forward(self, indices):
        batch_size, length = indices.shape
        types = torch.ones((batch_size, length), dtype=torch.long, device=indices.device)
        hidden = self.drop(
            self.tok_emb(indices) + self.pos_emb[:, :length, :] + self.type_emb(types)
        )
        return self.head(self.ln_f(self.blocks(hidden)))


START_FREQUENCIES = {
    "A": 0.06152815486054793,
    "C": 0.02912450195691266,
    "D": 0.022777758189062445,
    "E": 0.020415359119918198,
    "F": 0.07986319241211523,
    "G": 0.15094672261203765,
    "H": 0.011494658157328726,
    "I": 0.04717746200768661,
    "K": 0.11075067874898628,
    "L": 0.06498360424526639,
    "M": 0.10422763654314023,
    "N": 0.015866859419625543,
    "P": 0.019004971615951483,
    "Q": 0.019780684743133176,
    "R": 0.08991220337787807,
    "S": 0.03529494728676704,
    "T": 0.022248862875074927,
    "V": 0.043968830436162336,
    "W": 0.033249885406015305,
    "Y": 0.01738302598638976,
}


class Generator:
    """Sample with temperature 1, top-k 10 and the platform's start frequencies."""

    def __init__(self, path, device):
        self.device = device
        self.vocabulary = "<ACDEFGHIKLMNPQRSTVWY"
        self.model = GPT()
        self.model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        self.model.to(device).eval()

    def generate(self, max_length: int, num_sequences: int) -> list[str]:
        """Remove padding tokens after sampling; output length can be shorter."""
        iterations = math.ceil(num_sequences / 8)
        contexts = np.random.choice(
            list(START_FREQUENCIES), iterations, p=list(START_FREQUENCIES.values())
        )
        generated = []
        with torch.inference_mode():
            for context in contexts:
                indices = torch.full(
                    (8, 1), self.vocabulary.index(context), dtype=torch.long, device=self.device
                )
                for _ in range(max_length - 1):
                    logits = self.model(indices)[:, -1, :]
                    top_values, _ = torch.topk(logits, 10)
                    logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
                    probabilities = functional.softmax(logits, dim=-1)
                    next_indices = torch.multinomial(probabilities, num_samples=1)
                    indices = torch.cat((indices, next_indices), dim=1)
                for row in indices.cpu().tolist():
                    generated.append(
                        "".join(self.vocabulary[index] for index in row).replace("<", "")
                    )
        return generated[:num_sequences]
