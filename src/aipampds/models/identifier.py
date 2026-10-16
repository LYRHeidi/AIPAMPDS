"""Offline PeptideBERT-CNN-LSTM inference matching the platform."""

from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer


class PeptideBERT(nn.Module):
    """Frozen ProtBERT features followed by the original convolution and LSTM."""

    def __init__(self, config):
        super().__init__()
        self.protbert = BertModel(config)
        self.conv = nn.Sequential(
            nn.Conv1d(1024, 64, kernel_size=16),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=5, stride=1),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, inputs, attention_mask):
        hidden = self.protbert(inputs, attention_mask=attention_mask).last_hidden_state
        features = self.conv(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        recurrent, _ = self.lstm(features)
        return self.fc(recurrent)[:, -1, :]


class Identifier:
    """Load local assets and a fully matching, verified fine-tuned checkpoint."""

    def __init__(self, path: Path, device):
        assets = Path(__file__).parent.parent / "assets" / "prot_bert_bfd"
        config = BertConfig.from_json_file(str(assets / "config.json"))
        self.tokenizer = BertTokenizer.from_pretrained(
            str(assets), do_lower_case=False, local_files_only=True
        )
        self.device = device
        self.model = PeptideBERT(config)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        del checkpoint, state
        self.model.to(device).eval()

    def predict(self, sequences: list[str]) -> list[float]:
        """Predict a batch using the original fixed 300-token padding."""
        encoded = self.tokenizer(
            [" ".join(sequence) for sequence in sequences],
            truncation=True,
            padding="max_length",
            max_length=300,
            return_tensors="pt",
        )
        with torch.inference_mode():
            scores = self.model(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )
        return scores.reshape(-1).cpu().tolist()
