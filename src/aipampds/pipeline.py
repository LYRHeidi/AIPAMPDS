"""The shared local generation and screening API."""

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .io import AMINO_ACIDS, Record, read_records, validate_records
from .resources import GENERATOR_WEIGHTS, Resources
from .scoring import TARGETS, evaluate

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class AIPAMPDS:
    """Reuse lazily loaded models for local generation and screening.

    Args:
        weights_dir: Directory holding the seven official checkpoint files.
            Defaults to AIPAMPDS_WEIGHTS_DIR or the user's cache directory.
        device: 'auto', 'cpu', 'cuda', or 'cuda:N'. TensorFlow always uses CPU.
        batch_size: Maximum number of records per screening inference batch.
    """

    def __init__(
        self, weights_dir: str | Path | None = None, device: str = "auto", batch_size: int = 8
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(device, str) or (
            device not in ("auto", "cpu", "cuda")
            and not (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")
        self.resources = Resources(weights_dir)
        self.device = device
        self.batch_size = batch_size
        self._models = {}
        self._resolved_device = None

    def _resolve_device(self):
        if self._resolved_device is None:
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            resolved = torch.device(device)
            if resolved.type == "cuda" and (
                not torch.cuda.is_available()
                or (resolved.index is not None and resolved.index >= torch.cuda.device_count())
            ):
                raise ValueError(f"Requested CUDA device {device!r} is unavailable; use 'cpu'.")
            self._resolved_device = resolved
            logger.info("Using PyTorch device: %s; TensorFlow device: CPU", resolved)
        return self._resolved_device

    def _get_model(self, name):
        if name not in self._models:
            logger.info("Loading %s model", name)
            if name in TARGETS:
                path = self.resources.require(GENERATOR_WEIGHTS[name])
                from .models.generation import Generator

                model = Generator(path, self._resolve_device())
            elif name == "identifier":
                path = self.resources.require("amp_identifier.pt")
                from .models.identifier import Identifier

                model = Identifier(path, self._resolve_device())
            elif name == "mic":
                ecoli = self.resources.require("ec_model.h5")
                saureus = self.resources.require("sa_model.h5")
                from .models.mic import MIC

                model = MIC(ecoli, saureus, self._resolve_device())
            elif name == "hemolysis":
                path = self.resources.require("hemolysis.h5")
                from .models.hemolysis import Hemolysis

                model = Hemolysis(path)
            else:
                raise ValueError(f"Unknown model {name!r}.")
            self._models[name] = model
        return self._models[name]

    def screen(self, sequences: str | Iterable[str]) -> "pd.DataFrame":
        """Screen standard amino acid sequences of length 5-50; preserve duplicates.

        Results are stably sorted by descending Score. Sequence_ID identifies
        the original input position. All scores remain numeric, unrounded floats.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        records = []
        for index, sequence in enumerate(sequences, start=1):
            if index > 10_000:
                raise ValueError("Provide between 1 and 10000 sequence records.")
            records.append(Record(f"sequence_{index}", sequence))
        return self._screen(validate_records(records))

    def screen_file(self, path: str | Path) -> "pd.DataFrame":
        """Screen multiline FASTA or plain text, retaining complete FASTA headers."""
        return self._screen(read_records(path))

    def _predict(self, name, sequences):
        model = self._get_model(name)
        scores = [[], []] if name == "mic" else []
        total = len(sequences)
        for start in range(0, total, self.batch_size):
            batch = sequences[start : start + self.batch_size]
            logger.info("%s: records %s-%s of %s", name, start + 1, start + len(batch), total)
            prediction = model.predict(batch)
            groups = prediction if name == "mic" else [prediction]
            if len(groups) != (2 if name == "mic" else 1):
                raise RuntimeError(f"{name} returned an unexpected number of score groups.")
            for group in groups:
                if len(group) != len(batch) or any(
                    not math.isfinite(value) or not 0 <= value <= 1 for value in group
                ):
                    raise RuntimeError(f"{name} returned missing or invalid prediction scores.")
            if name == "mic":
                scores[0].extend(prediction[0])
                scores[1].extend(prediction[1])
            else:
                scores.extend(prediction)
        return scores

    def _screen(self, records, target="broad-spectrum", properties=None):
        import pandas as pd

        sequences = [record.sequence for record in records]
        amp = self._predict("identifier", sequences)
        ecoli, saureus = self._predict("mic", sequences)
        non_hemolytic = self._predict("hemolysis", sequences)
        evaluations = [
            evaluate(*scores, target) for scores in zip(amp, ecoli, saureus, non_hemolytic)
        ]
        frame = pd.DataFrame(
            {
                "Sequence_ID": [record.identifier for record in records],
                "Sequence": sequences,
            }
        )
        if properties is not None:
            frame = pd.concat([frame, properties.reset_index(drop=True)], axis=1)
        frame["AMP_Score"] = amp
        if target in ("broad-spectrum", "escherichia-coli"):
            frame["E.coli_Score"] = ecoli
        if target in ("broad-spectrum", "staphylococcus-aureus"):
            frame["S.aureus_Score"] = saureus
        frame["Hemolytic_Score"] = non_hemolytic
        frame["Score"] = [evaluation[0] for evaluation in evaluations]
        frame["Evaluate"] = [evaluation[1] for evaluation in evaluations]
        frame["Warning"] = [
            "Hemolysis model: length outside the recommended range 6-200."
            if len(sequence) < 6
            else ""
            for sequence in sequences
        ]
        return frame.sort_values("Score", ascending=False, kind="stable").reset_index(drop=True)

    def generate(
        self, target: str = "broad-spectrum", max_length: int = 50, num_sequences: int = 10
    ) -> "pd.DataFrame":
        """Generate, calculate descriptors and screen peptides for the chosen target.

        max_length is an upper bound, not an exact length. Up to ten sampling
        rounds replace sequences shorter than five residues. Duplicate generated
        sequences are retained, matching the platform.
        """
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; choose from {TARGETS}.")
        for name, value, minimum, maximum in (
            ("max_length", max_length, 10, 50),
            ("num_sequences", num_sequences, 1, 1000),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
        generator = self._get_model(target)
        sequences = []
        for attempt in range(10):
            remaining = num_sequences - len(sequences)
            logger.info("Generating %s sequences (sampling round %s/10)", remaining, attempt + 1)
            candidates = generator.generate(max_length, remaining)
            sequences.extend(
                sequence
                for sequence in candidates
                if 5 <= len(sequence) <= max_length and set(sequence) <= AMINO_ACIDS
            )
            if len(sequences) >= num_sequences:
                sequences = sequences[:num_sequences]
                break
        if len(sequences) != num_sequences:
            raise RuntimeError(
                f"Generated only {len(sequences)} supported sequences after 10 sampling rounds; "
                "try a larger max_length."
            )
        from .properties import calculate_properties

        properties = calculate_properties(sequences)
        records = [
            Record(f"generated_{index}", seq) for index, seq in enumerate(sequences, start=1)
        ]
        return self._screen(records, target, properties)
