"""Command-line commands and an interactive terminal wizard."""

import argparse
import logging
import sys
from pathlib import Path

from .io import parse_text
from .pipeline import AIPAMPDS
from .resources import download_weights
from .scoring import TARGETS


def _runtime_options(parser):
    parser.add_argument(
        "--weights-dir", type=Path, help="Checkpoint directory (default: user cache)"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Screening batch size (default: 8)"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aipampds", description="Local antimicrobial peptide generation and screening."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate, characterize and screen peptides")
    _runtime_options(generate)
    generate.add_argument("--target", choices=TARGETS, default="broad-spectrum")
    generate.add_argument("--max-length", type=int, default=50, help="Maximum length, 10-50")
    generate.add_argument(
        "--num-sequences", type=int, default=10, help="Number of peptides, 1-1000"
    )
    generate.add_argument("--output", type=Path, required=True, help="Destination CSV file")
    screen = commands.add_parser("screen", help="Screen FASTA, plain text, or direct sequences")
    _runtime_options(screen)
    inputs = screen.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="FASTA or plain text file")
    inputs.add_argument("--sequence", action="append", help="Direct sequence; may be repeated")
    screen.add_argument("--output", type=Path, required=True, help="Destination CSV file")
    interactive_parser = commands.add_parser("interactive", help="Start the interactive wizard")
    _runtime_options(interactive_parser)
    download = commands.add_parser("download-weights", help="Download and verify all seven weights")
    download.add_argument("--weights-dir", type=Path, help="Destination (default: user cache)")
    return parser


def _save(frame, output):
    output = Path(output).expanduser()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as destination:
        frame.to_csv(destination, index=False)
    print(frame.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved {len(frame)} records to {output.resolve()}")


def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _ask_integer(prompt, default, minimum, maximum):
    while True:
        try:
            value = int(_ask(prompt, str(default)))
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter an integer between {minimum} and {maximum}.")


def _ask_choice(prompt, choices, default):
    while True:
        answer = _ask(prompt, default)
        if answer in choices:
            return answer
        print(f"Choose one of: {', '.join(choices)}.")


def _check_output(output):
    if Path(output).expanduser().exists():
        raise ValueError(f"Output already exists: {output}. Choose a new filename.")


def interactive(args):
    """Keep model instances alive across successive wizard operations."""
    print("AIPAMPDS local inference")
    weights_dir = _ask(
        "Weights directory (leave blank for user cache)", str(args.weights_dir or "")
    )
    device = _ask("Device: auto, cpu, cuda, or cuda:N", args.device)
    pipeline = AIPAMPDS(weights_dir=weights_dir or None, device=device, batch_size=args.batch_size)
    while True:
        operation = _ask_choice(
            "Choose 1: Generation, 2: Screening, 3: Download weights, q: Quit",
            ("1", "2", "3", "q"),
            "q",
        )
        if operation == "q":
            return
        try:
            if operation == "3":
                files = download_weights(pipeline.resources.directory)
                print(f"Verified {len(files)} checkpoint files in {pipeline.resources.directory}")
                continue
            if operation == "1":
                for index, target in enumerate(TARGETS, start=1):
                    print(f"  {index}: {target}")
                target_index = _ask_integer("Target", 1, 1, 3)
                max_length = _ask_integer("Maximum sequence length", 50, 10, 50)
                number = _ask_integer("Number of sequences", 10, 1, 1000)
                output = _ask("Output CSV", "results/generated.csv")
                _check_output(output)
                result = pipeline.generate(TARGETS[target_index - 1], max_length, number)
            else:
                source = _ask_choice("Input: 1: File, 2: Paste sequences or FASTA", ("1", "2"), "1")
                if source == "1":
                    path = _ask("FASTA or text file")
                    if not path:
                        raise ValueError("An input file is required.")
                else:
                    print("Paste FASTA or one sequence per line. Enter a blank line to finish.")
                    lines = []
                    while True:
                        line = input()
                        if not line.strip():
                            break
                        lines.append(line)
                    records = parse_text("\n".join(lines))
                output = _ask("Output CSV", "results/screened.csv")
                _check_output(output)
                result = pipeline.screen_file(path) if source == "1" else pipeline._screen(records)
            _save(result, output)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Error: {error}", file=sys.stderr)


def main(argv=None):
    """Run a command; return 0 on success, 1 on failure, 2 on invalid input, 130 on interrupt."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        if args.command == "download-weights":
            files = download_weights(args.weights_dir)
            print(f"Verified {len(files)} checkpoint files in {files[0].parent}")
        elif args.command == "interactive":
            interactive(args)
        else:
            _check_output(args.output)
            pipeline = AIPAMPDS(args.weights_dir, args.device, args.batch_size)
            if args.command == "generate":
                result = pipeline.generate(args.target, args.max_length, args.num_sequences)
            else:
                result = (
                    pipeline.screen_file(args.input)
                    if args.input
                    else pipeline.screen(args.sequence)
                )
            _save(result, args.output)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted.", file=sys.stderr)
        return 130
    except ValueError as error:
        print(f"Invalid input: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
