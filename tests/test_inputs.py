import pytest

from aipampds import AIPAMPDS
from aipampds.io import Record, parse_text, read_records, validate_records


def test_multiline_fasta_keeps_headers_and_duplicates():
    records = parse_text(">first description\ngigkfl\nhsakk\n>first description\nGIGKFLHSAKK\n")
    assert records == [Record("first description", "GIGKFLHSAKK")] * 2


def test_plain_text_normalizes_whitespace_and_case():
    assert parse_text(" gig kfl\n\n rrrkk ") == [
        Record("sequence_1", "GIGKFL"),
        Record("sequence_2", "RRRKK"),
    ]


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "between 1 and 10000"),
        (">empty\n>next\nGIGKFL", "empty"),
        (">\nGIGKFL", "identifier is empty"),
        ("GIGKFL\n>next\nGIGKFL", "before a header"),
        (">bad\nGIGXFL", "bad.*invalid amino acid"),
        (">short\nGIGK", "short.*length 4"),
        (">long\n" + "A" * 51, "long.*length 51"),
        ("GIG-KFL", "invalid amino acid"),
        ("GIG1KFL", "invalid amino acid"),
    ],
)
def test_invalid_records_report_cause(text, message):
    with pytest.raises(ValueError, match=message):
        parse_text(text)


def test_file_reads_bom(tmp_path):
    path = tmp_path / "peptides.fa"
    path.write_text("\ufeff>test\nGIGKFL", encoding="utf-8")
    assert read_records(path) == [Record("test", "GIGKFL")]


def test_all_inputs_validate_before_model_loading(monkeypatch):
    pipeline = AIPAMPDS()
    monkeypatch.setattr(pipeline, "_get_model", lambda name: pytest.fail("Loaded a model"))
    with pytest.raises(ValueError, match="sequence_2"):
        pipeline.screen(["GIGKFL", "X"])


def test_record_count_limit():
    with pytest.raises(ValueError, match="10000"):
        validate_records([Record("test", "GIGKFL")] * 10001)


@pytest.mark.parametrize("sequence", [None, 123, b"GIGKFL"])
def test_non_string_sequence(sequence):
    with pytest.raises(ValueError, match="expected a sequence string"):
        AIPAMPDS().screen([sequence])
