# AIPAMPDS

Local antimicrobial peptide generation and screening through a Python API, CLI, and interactive terminal wizard. Inference runs directly on your machine without a web server or task queue. Model architectures, preprocessing, sampling and evaluation follow the AIPAMPDS inference platform. This repository contains inference only.

## Requirements

- Linux x86-64 is the reference environment.
- Python 3.10, managed automatically by uv. The package supports Python 3.10 and 3.11.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.1 or newer.
- At least 8 GB available RAM; larger screening batches require more memory.
- Approximately 2 GB for checkpoints, plus space for the Python environment.
- Optional NVIDIA GPU with a driver supporting CUDA 12.8. PyTorch can use CUDA; TensorFlow hemolysis inference always uses CPU.

## Install with uv

From the repository root:

```bash
uv sync --locked --extra cpu
```

uv creates `.venv`, installs the project, and reproduces dependencies from `uv.lock`. Python is downloaded if needed. CPU is also the default when no extra is selected.

For CUDA 12.8:

```bash
uv sync --locked --extra cu128
uv run --extra cu128 aipampds --help
```

Keep `--extra cu128` on subsequent `uv run` commands to retain the GPU environment. The `cpu` and `cu128` extras are mutually exclusive; do not use `--all-extras`. Remaining examples use CPU.

Development dependencies are included by default. Use `uv sync --locked --extra cpu --no-dev` for an inference-only environment.

### modlamp dependency override

`modlamp==4.3.0` declares the unavailable legacy dependency `mysql-connector-python==8.0.17`. The uv configuration overrides it with `8.0.33`. Descriptor calculations do not use the database connector. Installation stays entirely under uv without a separate pip step. This override is a root-project uv setting; it is not encoded in wheel metadata.

## Prepare weights

```bash
uv run aipampds download-weights
```

This explicitly downloads and SHA256-verifies all seven official checkpoints from `https://share.pianlab.team/aipampds`. Valid existing files are reused. Temporary downloads replace destination files only after verification. Retry the same command after interruption or corruption.

Default location: `$XDG_CACHE_HOME/aipampds/weights`, or `~/.cache/aipampds/weights` when `XDG_CACHE_HOME` is unset. To choose another location:

```bash
uv run aipampds download-weights --weights-dir ./weights
uv run aipampds screen --weights-dir ./weights --input examples/peptides.fasta --output results/screened.csv
```

Set `AIPAMPDS_WEIGHTS_DIR` to reuse a directory. An explicit `weights_dir` argument or `--weights-dir` overrides it. Relative paths resolve when a pipeline is created; the default directory never depends on your current working directory.

```text
weights/
  amp_identifier.pt
  ec_model.h5
  sa_model.h5
  hemolysis.h5
  finetuning-with-Eco-Sta.pt
  finetuning-with-Eco.pt
  finetuning-with-Sta.pt
```

Models are verified before their first load in each pipeline. The AMP checkpoint uses legacy PyTorch serialization, so its official checksum is verified before deserialization. Arbitrary replacement checkpoints are not supported.

### Offline inference

The package includes the small ProtBERT configuration and tokenizer files. It does not download separate ProtBERT pretrained weights. After installation and weight preparation, inference makes no network requests:

```bash
uv run --offline aipampds screen --input examples/peptides.fasta --output results/offline.csv
```

Importing the package, creating a pipeline, and viewing help do not load models or download resources. Models load on first use and are reused within that pipeline.

## Python API

```python
from aipampds import AIPAMPDS

pipeline = AIPAMPDS(device="auto", batch_size=8)

# Screen one sequence or an iterable.
single = pipeline.screen("GIGKFLHSAKKFGKAFVGEIMNS")
screened = pipeline.screen([
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "KICKRRRSRRRRRRQRRKRCLCRRYQH",
])

# Preserve FASTA headers, multiline records and duplicate sequences.
from_file = pipeline.screen_file("examples/peptides.fasta")

# Generation includes descriptors and full screening.
generated = pipeline.generate(
    target="broad-spectrum",
    max_length=50,
    num_sequences=10,
)
generated.to_csv("generated.csv", index=False)
```

Methods return a `pandas.DataFrame`, stably sorted by descending `Score`. Scores remain numeric and unrounded before evaluation. `Sequence_ID` preserves original input identity after sorting; list inputs receive `sequence_1`, `sequence_2`, etc. Duplicate sequences and headers are retained. Generated records receive `generated_1`, `generated_2`, etc.

`device` accepts `auto`, `cpu`, `cuda`, or `cuda:N`. `auto` selects CUDA when available; explicitly requesting an unavailable CUDA device raises an error. `batch_size` controls screening batches; reduce it when memory is limited. Create one pipeline per process; concurrent calls on the same instance are not supported.

```bash
uv run python examples/python_api.py --device cpu
```

### Use from another uv project

Use a consuming project targeting Python 3.10 or 3.11. Sources and overrides are root-project uv settings. Add these to its `pyproject.toml`, merging any existing sections:

```toml
[tool.uv]
override-dependencies = ["mysql-connector-python==8.0.33"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then install:

```bash
uv add /absolute/path/to/aipampds-release
uv run python -c 'from aipampds import AIPAMPDS; print(AIPAMPDS())'
```

The same override is necessary when consuming a built wheel. For GPU use, point the consuming project's explicit PyTorch index at `https://download.pytorch.org/whl/cu128` instead.

## Command line

### Generation

```bash
uv run aipampds generate --target broad-spectrum --max-length 50 --num-sequences 10 --output results/generated.csv
uv run aipampds generate --target escherichia-coli --max-length 30 --num-sequences 5 --output results/ecoli.csv
uv run aipampds generate --target staphylococcus-aureus --max-length 30 --num-sequences 5 --output results/saureus.csv
```

`max_length`: 10-50; `num_sequences`: 1-1000. Sampling uses the original start-residue frequencies, batches of eight, temperature 1, and top-k 10. Padding tokens are removed, so length is an upper bound, not an exact length. Duplicate peptides are retained. Sequences shorter than five residues are replaced over at most ten sampling rounds; insufficient valid output raises an error.

Generation calculates descriptors and runs AMP, both MIC classifiers and hemolysis prediction. Evaluation uses only criteria relevant to the chosen target; species-specific CSVs contain only the relevant MIC score column, matching the platform.

### Screening

```bash
uv run aipampds screen --input examples/peptides.fasta --output results/screened.csv
uv run aipampds screen --input examples/peptides.txt --output results/text.csv
uv run aipampds screen --sequence GIGKFLHSAKKFGKAFVGEIMNS --sequence KICKRRRSRRRRRRQRRKRCLCRRYQH --output results/direct.csv
```

- Accept standard amino acids `ACDEFGHIKLMNPQRSTVWY`, length 5-50, up to 10,000 records.
- Lowercase becomes uppercase; whitespace is removed. Other characters, including `X`, gaps, digits and stop symbols, are rejected.
- FASTA sequences may span multiple lines. Plain text uses one sequence per nonempty line.
- All records are validated before inference. Errors identify the record and cause; records are never silently dropped or assigned default zero scores.
- Five-residue peptides carry a warning because the hemolysis model's recommended length starts at six.

Both commands support `--weights-dir`, `--device` and `--batch-size`. Progress goes to stderr; stdout shows up to ten rows and the output location. CSVs retain numeric precision. Existing output files are refused; choose a new filename. Parent directories are created automatically.

### Interactive wizard

```bash
uv run aipampds interactive
```

Select a weight directory and device, then choose Generation, Screening, Download weights, or Quit. Generation asks for target, maximum length, count and output path. Screening accepts a file or pasted FASTA/plain text; finish pasted input with a blank line. Enter accepts the displayed default. Models remain loaded between operations. Ctrl+C or end-of-input exits cleanly.

Help: `uv run aipampds --help`, `uv run aipampds screen --help`, or `uv run python -m aipampds --help`.

Exit codes: `0` success, `1` resource/inference failure, `2` invalid input/arguments, `130` interruption. Python methods raise exceptions instead of terminating the process.

## Results

| Column | Meaning |
| --- | --- |
| `Sequence_ID`, `Sequence` | Original record identifier and normalized peptide |
| `AMP_Score` | AMP classifier score |
| `E.coli_Score` | E. coli MIC classifier score, **not a concentration** |
| `S.aureus_Score` | S. aureus MIC classifier score, **not a concentration** |
| `Hemolytic_Score` | Platform-compatible name for **P(non-hemolytic)**; higher is better |
| `Score` | Number of passed criteria for the chosen target |
| `Evaluate` | `Good`, `Mediocre`, or `Bad` |
| `Warning` | Model applicability warning, otherwise empty |

Each passed criterion contributes one point:

| Criterion | Pass condition |
| --- | --- |
| AMP | `AMP_Score > 0.998` |
| E. coli | `E.coli_Score > 0.9980798959732056` |
| S. aureus | `S.aureus_Score > 0.9989857077598572` |
| Non-hemolytic | `Hemolytic_Score >= 0.5` |

Screening and broad-spectrum generation use four criteria: 4 is `Good`, 2-3 is `Mediocre`, 0-1 is `Bad`. Species-specific generation uses AMP, the selected species' MIC criterion and non-hemolysis: 3 is `Good`, 2 is `Mediocre`, 0-1 is `Bad`.

Generation also returns modlamp descriptors with the platform's amidation and pH conventions:

| Column | Definition |
| --- | --- |
| `Length` | Number of amino acids |
| `MW` | Amidated peptide molecular weight, Da |
| `Charge` | Net charge at pH 7.4 with C-terminal amidation |
| `Instability Index` | modlamp instability index |
| `Hydrophobicity` | Global Eisenberg hydrophobicity |
| `Hydrophobic Moment` | Eisenberg hydrophobic moment with modlamp defaults |

## Troubleshooting

- **Missing/corrupted checkpoint:** rerun `download-weights` with the same directory. Inference never downloads automatically.
- **CUDA unavailable:** use `--device cpu`, or synchronize the CUDA extra and check the driver. Selecting a CUDA device does not install GPU-enabled PyTorch.
- **Memory exhaustion:** lower `--batch-size` to `1`. The AMP identifier includes a large ProtBERT model.
- **Dependency resolution:** use the supplied lockfile and Python 3.10; check access to PyPI and PyTorch. Retain the modlamp override in consuming projects.
- **Output exists:** choose another filename; CLI commands preserve previous runs.
- **Invalid sequence:** fix the named record. Partial predictions are not exported.

## Development and verification

```bash
uv sync --locked --extra cpu
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Unit tests need no weights or network. Real-model tests are opt-in:

```bash
AIPAMPDS_TEST_WEIGHTS=/absolute/path/to/weights uv run pytest -m integration
```

They cover single/batched screening, duplicates, all three generation targets and descriptor output. To compare against the original platform where its source is available:

```bash
AIPAMPDS_TEST_WEIGHTS=/absolute/path/to/weights AIPAMPDS_REFERENCE_ROOT=/absolute/path/to/aipampds uv run pytest -m integration
```

For GPU checks, add `--extra cu128` to `uv run` and set `AIPAMPDS_TEST_DEVICE=cuda`. Real tests skip without `AIPAMPDS_TEST_WEIGHTS`; CPU is the default device.

```text
src/aipampds/   API, workflows, model implementations, CLI and tokenizer assets
examples/      Small inputs and a runnable Python example
tests/         Unit and opt-in real-model tests
pyproject.toml Dependencies, build metadata and CLI registration
uv.lock        Reproducible CPU/GPU dependency resolutions
```
