from unittest.mock import Mock

import pandas as pd
import pytest

from aipampds import AIPAMPDS
from aipampds.scoring import (
    AMP_THRESHOLD,
    ECOLI_THRESHOLD,
    SAUREUS_THRESHOLD,
    TARGETS,
    evaluate,
)


def test_threshold_boundaries_and_targets():
    assert evaluate(AMP_THRESHOLD, ECOLI_THRESHOLD, SAUREUS_THRESHOLD, 0.5) == (1, "Bad")
    assert evaluate(1, 1, 1, 0.5) == (4, "Good")
    assert evaluate(1, 1, 0, 0.5, "escherichia-coli") == (3, "Good")
    assert evaluate(1, 0, 1, 0.5, "staphylococcus-aureus") == (3, "Good")
    assert evaluate(1, 1, 1, 0.49) == (3, "Mediocre")
    assert evaluate(1, 0, 0, 0.5, "escherichia-coli") == (2, "Mediocre")


def fake_pipeline(batch_size=2):
    pipeline = AIPAMPDS(batch_size=batch_size)
    values = {"GIGKFL": 0.99804, "RRRRR": 0.1}
    pipeline._models = {
        "identifier": Mock(predict=lambda sequences: [values[seq] for seq in sequences]),
        "mic": Mock(predict=lambda sequences: ([1.0] * len(sequences), [1.0] * len(sequences))),
        "hemolysis": Mock(predict=lambda sequences: [0.5] * len(sequences)),
    }
    return pipeline


def test_alignment_precision_stable_sort_and_warning(tmp_path):
    path = tmp_path / "input.fa"
    path.write_text(">lower\nRRRRR\n>first\nGIGKFL\n>duplicate\nGIGKFL\n", encoding="utf-8")
    result = fake_pipeline().screen_file(path)
    assert result.Sequence_ID.tolist() == ["first", "duplicate", "lower"]
    assert result.Score.tolist() == [4, 4, 3]
    assert result.AMP_Score.tolist() == [0.99804, 0.99804, 0.1]
    assert result.Warning.tolist()[:2] == ["", ""]
    assert "6-200" in result.Warning.iloc[2]


def test_single_string_has_one_row():
    result = fake_pipeline().screen("GIGKFL")
    assert result.Sequence.tolist() == ["GIGKFL"]


@pytest.mark.parametrize("bad_scores", [[], [float("nan")], [float("inf")], [-0.1], [1.1]])
def test_missing_and_invalid_predictions_fail(bad_scores):
    pipeline = fake_pipeline()
    pipeline._models["identifier"] = Mock(predict=lambda sequences: bad_scores)
    with pytest.raises(RuntimeError, match="missing or invalid"):
        pipeline.screen("GIGKFL")


@pytest.mark.parametrize("target", TARGETS)
def test_generation_resamples_and_uses_target_scoring(monkeypatch, target):
    pipeline = fake_pipeline()
    generator = Mock()
    generator.generate.side_effect = [["A", "GIGKFL"], ["GIGKFL"]]
    pipeline._models[target] = generator
    monkeypatch.setattr(
        "aipampds.properties.calculate_properties",
        lambda seqs: pd.DataFrame(
            {
                "Length": [len(seq) for seq in seqs],
                "MW": [100.0] * len(seqs),
            }
        ),
    )
    result = pipeline.generate(target, 10, 2)
    assert len(result) == 2
    assert generator.generate.call_args_list[1].args == (10, 1)
    assert result.Length.tolist() == [6, 6]
    assert result.Score.tolist() == [4 if target == "broad-spectrum" else 3] * 2
    assert ("E.coli_Score" in result) == (target != "staphylococcus-aureus")
    assert ("S.aureus_Score" in result) == (target != "escherichia-coli")


def test_generation_retry_limit():
    pipeline = AIPAMPDS()
    generator = Mock()
    generator.generate.return_value = ["A"]
    pipeline._models["broad-spectrum"] = generator
    with pytest.raises(RuntimeError, match="after 10 sampling rounds"):
        pipeline.generate(num_sequences=1)
    assert generator.generate.call_count == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "unknown"},
        {"max_length": 9},
        {"max_length": 51},
        {"max_length": True},
        {"num_sequences": 0},
        {"num_sequences": 1001},
    ],
)
def test_bad_generation_options(kwargs):
    with pytest.raises(ValueError):
        AIPAMPDS().generate(**kwargs)


@pytest.mark.parametrize("kwargs", [{"device": "mps"}, {"batch_size": 0}, {"batch_size": True}])
def test_bad_pipeline_options(kwargs):
    with pytest.raises(ValueError):
        AIPAMPDS(**kwargs)


def test_models_are_cached(monkeypatch, tmp_path):
    pipeline = AIPAMPDS(tmp_path)
    monkeypatch.setattr(pipeline.resources, "require", lambda filename: tmp_path / filename)
    monkeypatch.setattr(pipeline, "_resolve_device", lambda: "cpu")
    model = Mock()
    monkeypatch.setattr("aipampds.models.generation.Generator", lambda *args: model)
    assert pipeline._get_model("broad-spectrum") is model
    assert pipeline._get_model("broad-spectrum") is model


def test_unavailable_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        AIPAMPDS(device="cuda")._resolve_device()
