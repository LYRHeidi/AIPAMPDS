"""Sequence parsing and validation shared by all entry points."""

from dataclasses import dataclass
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Record:
    """An input record with its original identifier."""

    identifier: str
    sequence: str


def normalize_sequence(sequence: str, identifier: str) -> str:
    """Normalize whitespace and case, refusing unsupported sequences."""
    if not isinstance(sequence, str):
        raise ValueError(f"Record {identifier!r}: expected a sequence string.")
    normalized = "".join(sequence.split()).upper()
    invalid = sorted(set(normalized) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Record {identifier!r}: invalid amino acid characters: {invalid}.")
    if not 5 <= len(normalized) <= 50:
        raise ValueError(
            f"Record {identifier!r}: length {len(normalized)} is outside the supported range 5-50."
        )
    return normalized


def validate_records(records: list[Record]) -> list[Record]:
    """Validate the entire input before running any model."""
    if not 1 <= len(records) <= 10_000:
        raise ValueError("Provide between 1 and 10000 sequence records.")
    return [
        Record(record.identifier, normalize_sequence(record.sequence, record.identifier))
        for record in records
    ]


def parse_text(text: str) -> list[Record]:
    """Read multiline FASTA or one sequence per nonempty line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not any(line.startswith(">") for line in lines):
        return validate_records(
            [Record(f"sequence_{index}", line) for index, line in enumerate(lines, start=1)]
        )
    records = []
    identifier = None
    fragments = []
    for line_number, line in enumerate(lines, start=1):
        if line.startswith(">"):
            if identifier is not None:
                records.append(Record(identifier, "".join(fragments)))
            identifier = line[1:].strip()
            if not identifier:
                raise ValueError(f"FASTA line {line_number}: the record identifier is empty.")
            fragments = []
        else:
            if identifier is None:
                raise ValueError(f"FASTA line {line_number}: sequence appears before a header.")
            fragments.append(line)
    if identifier is not None:
        records.append(Record(identifier, "".join(fragments)))
    return validate_records(records)


def read_records(path: str | Path) -> list[Record]:
    """Read a UTF-8 FASTA or plain text file."""
    return parse_text(Path(path).read_text(encoding="utf-8-sig"))
