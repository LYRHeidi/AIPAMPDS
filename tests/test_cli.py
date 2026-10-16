import subprocess
import sys
from unittest.mock import Mock

import pandas as pd
import pytest

from aipampds import cli


def test_import_and_help_do_not_import_ml_libraries():
    script = (
        "import sys; import aipampds; from aipampds.cli import build_parser; "
        "aipampds.AIPAMPDS(); build_parser().format_help(); "
        "assert not {'torch', 'tensorflow', 'transformers', 'pandas'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cli_export(monkeypatch, tmp_path):
    model = Mock()
    model.screen.return_value = pd.DataFrame({"Sequence": ["GIGKFL"], "AMP_Score": [0.99804]})
    monkeypatch.setattr(cli, "AIPAMPDS", lambda *args, **kwargs: model)
    output = tmp_path / "nested" / "results.csv"
    assert cli.main(["screen", "--sequence", "GIGKFL", "--output", str(output)]) == 0
    assert pd.read_csv(output).AMP_Score.tolist() == [0.99804]
    assert cli.main(["screen", "--sequence", "GIGKFL", "--output", str(output)]) == 2


def test_invalid_input_exit_code(tmp_path):
    assert cli.main(["screen", "--sequence", "XXXX", "--output", str(tmp_path / "out.csv")]) == 2


def test_missing_weight_exit_code(tmp_path):
    assert (
        cli.main(
            [
                "screen",
                "--weights-dir",
                str(tmp_path),
                "--sequence",
                "GIGKFL",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize("exception", [KeyboardInterrupt, EOFError])
def test_interactive_interruption(monkeypatch, exception):
    def stop(prompt):
        raise exception()

    monkeypatch.setattr("builtins.input", stop)
    assert cli.main(["interactive"]) == 130


def test_interactive_screen_reuses_pipeline_and_exits(monkeypatch, tmp_path):
    model = Mock()
    model.screen_file.return_value = pd.DataFrame({"Sequence": ["GIGKFL"]})
    factory = Mock(return_value=model)
    monkeypatch.setattr(cli, "AIPAMPDS", factory)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    answers = iter(
        [
            "",
            "cpu",
            "2",
            "1",
            "input.fa",
            str(first),
            "2",
            "1",
            "input.fa",
            str(second),
            "q",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["interactive"]) == 0
    assert first.is_file() and second.is_file()
    assert factory.call_count == 1
    assert model.screen_file.call_count == 2
