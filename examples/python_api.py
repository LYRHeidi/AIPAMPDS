"""Run with: uv run python examples/python_api.py --weights-dir /path/to/weights."""

import argparse
from pathlib import Path

from aipampds import AIPAMPDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    pipeline = AIPAMPDS(weights_dir=args.weights_dir, device=args.device)
    output = Path("results")
    output.mkdir(exist_ok=True)
    screened = pipeline.screen_file(Path(__file__).with_name("peptides.fasta"))
    screened.to_csv(output / "screened.csv", index=False)
    print(screened.to_string(index=False))
    generated = pipeline.generate(target="broad-spectrum", max_length=30, num_sequences=3)
    generated.to_csv(output / "generated.csv", index=False)
    print(generated.to_string(index=False))


if __name__ == "__main__":
    main()
