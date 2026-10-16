import importlib.util
import os
from pathlib import Path

import pytest

from aipampds import AIPAMPDS
from aipampds.io import AMINO_ACIDS
from aipampds.scoring import TARGETS

pytestmark = pytest.mark.integration
SEQUENCES = ["GIGKFLHSAKKFGKAFVGEIMNS", "KICKRRRSRRRRRRQRRKRCLCRRYQH"]


@pytest.fixture(scope="module")
def pipeline():
    directory = os.environ.get("AIPAMPDS_TEST_WEIGHTS")
    if not directory:
        pytest.skip("Set AIPAMPDS_TEST_WEIGHTS to run real model inference.")
    import torch

    torch.set_num_threads(2)
    return AIPAMPDS(directory, device=os.environ.get("AIPAMPDS_TEST_DEVICE", "cpu"), batch_size=2)


def test_real_single_batch_and_duplicate_alignment(pipeline):
    import numpy as np

    result = pipeline.screen(SEQUENCES + SEQUENCES[:1]).set_index("Sequence_ID")
    single = pipeline.screen(SEQUENCES[0]).iloc[0]
    for column in ("AMP_Score", "E.coli_Score", "S.aureus_Score", "Hemolytic_Score"):
        assert result[column].between(0, 1).all()
        np.testing.assert_allclose(result.loc["sequence_1", column], single[column], atol=1e-6)
        np.testing.assert_allclose(
            result.loc["sequence_1", column], result.loc["sequence_3", column], atol=1e-6
        )
    assert result.loc["sequence_2", "Sequence"] == SEQUENCES[1]


@pytest.mark.parametrize("target", TARGETS)
def test_real_generation_and_descriptors(pipeline, target):
    result = pipeline.generate(target, max_length=20, num_sequences=2)
    assert len(result) == 2
    assert result.Sequence.map(lambda seq: 5 <= len(seq) <= 20 and set(seq) <= AMINO_ACIDS).all()
    assert result.Length.tolist() == result.Sequence.str.len().tolist()
    assert result.MW.gt(0).all()
    assert result.Score.between(0, 4 if target == "broad-spectrum" else 3).all()
    assert {"Charge", "Instability Index", "Hydrophobicity", "Hydrophobic Moment"} <= set(result)


def load_reference(root, name):
    path = root / "models" / name / "inference.py"
    spec = importlib.util.spec_from_file_location(f"reference_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_platform_parity(pipeline, monkeypatch, tmp_path):
    root = os.environ.get("AIPAMPDS_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set AIPAMPDS_REFERENCE_ROOT to compare the original platform.")
    import gc

    import numpy as np
    import torch

    root = Path(root)
    directory = pipeline.resources.directory
    monkeypatch.setenv("AMP_IDENTIFIER_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_LOCAL_FILES_ONLY", "1")
    result = pipeline.screen(SEQUENCES).set_index("Sequence_ID").loc[["sequence_1", "sequence_2"]]
    reference = load_reference(root, "amp_identifier")
    predictor = reference.AMPPredictor(
        directory / "amp_identifier.pt", device=str(pipeline._resolve_device())
    )
    np.testing.assert_allclose(
        result.AMP_Score, predictor.predict(SEQUENCES, batch_size=2), rtol=1e-5, atol=1e-6
    )
    del predictor, reference
    gc.collect()
    reference = load_reference(root, "amp_mic")
    predictor = reference.MICPredictor(
        directory / "ec_model.h5", directory / "sa_model.h5", device=str(pipeline._resolve_device())
    )
    expected = predictor.predict(SEQUENCES, batch_size=2)
    for output, key in (("E.coli_Score", "E.coli_score"), ("S.aureus_Score", "S.aureus_score")):
        np.testing.assert_allclose(result[output], [row[key] for row in expected], atol=1e-6)
    del predictor, reference
    gc.collect()
    reference = load_reference(root, "amp_hemolysis")
    predictor = reference.HemolysisPredictor(directory / "hemolysis.h5")
    expected = predictor.predict(SEQUENCES)
    np.testing.assert_allclose(
        result.Hemolytic_Score, [row["hemolytic_score"] for row in expected], atol=1e-6
    )
    del predictor, reference
    reference = load_reference(root, "amp_generation")
    from aipampds.properties import calculate_properties

    actual_properties = calculate_properties(SEQUENCES)
    expected_properties = reference.calculate_physicochemical_properties(SEQUENCES)
    np.testing.assert_allclose(
        actual_properties, expected_properties[actual_properties.columns], atol=1e-8
    )
    from aipampds.resources import GENERATOR_WEIGHTS

    for target in TARGETS:
        generator = reference.Generate(
            directory / GENERATOR_WEIGHTS[target], device=str(pipeline._resolve_device())
        )
        np.random.seed(42)
        torch.manual_seed(42)
        expected = generator.generate(sequence_size=20, gen_size=2)
        np.random.seed(42)
        torch.manual_seed(42)
        actual = pipeline._get_model(target).generate(max_length=20, num_sequences=2)
        assert actual == expected
