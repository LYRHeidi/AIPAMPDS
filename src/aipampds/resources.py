"""Explicit, checksum-verified model resource downloads."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

WEIGHTS = {
    "amp_identifier.pt": "ebb020ebc5cf36d751d78f81bb73aa4e4842d1231c1e617488116b680d1d2d57",
    "ec_model.h5": "c8e94e7e899a09107706fa43e63967feded1ce0652f68b0ba942aeff23522e1b",
    "sa_model.h5": "c7c49e869ecb2495643e7fcf60992bdd9e7759009fc740377692ce8f0fb122a5",
    "hemolysis.h5": "c46b413e10b19632c0b056102b908c907c788cfabd19ed06720d9a7c4a87af2a",
    "finetuning-with-Eco.pt": "a7088d2fe8c17103958c7c505f04c6d027aa7aa3942434beb6a7f28ee640b7b5",
    "finetuning-with-Eco-Sta.pt": "3c82251cbb784b009a94d4082f4f08858c924ebe6cff7ee3fb8d7002e1f537fb",
    "finetuning-with-Sta.pt": "dc8131449a972e06d7fb053f8fc3169c7ffc057ad909e808d50cad469afec10f",
}
GENERATOR_WEIGHTS = {
    "broad-spectrum": "finetuning-with-Eco-Sta.pt",
    "escherichia-coli": "finetuning-with-Eco.pt",
    "staphylococcus-aureus": "finetuning-with-Sta.pt",
}
DOWNLOAD_BASE = "https://share.pianlab.team/aipampds"


def default_weights_dir() -> Path:
    """Return a user-writable location independent of the working directory."""
    override = os.environ.get("AIPAMPDS_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "aipampds" / "weights"


def sha256(path: Path) -> str:
    """Compute a streaming SHA256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(weights_dir: str | Path | None = None) -> list[Path]:
    """Download all official checkpoints atomically, reusing verified files."""
    directory = Path(weights_dir).expanduser().resolve() if weights_dir else default_weights_dir()
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, expected in WEIGHTS.items():
        destination = directory / filename
        if destination.is_file() and sha256(destination) == expected:
            logger.info("Verified existing weight: %s", filename)
            downloaded.append(destination)
            continue
        logger.info("Downloading %s", filename)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(f"{DOWNLOAD_BASE}/{filename}", timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            if sha256(temporary) != expected:
                raise RuntimeError(f"SHA256 verification failed for {filename}.")
            temporary.replace(destination)
            downloaded.append(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return downloaded


class Resources:
    """Resolve and verify checkpoints before their first use."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = (
            Path(directory).expanduser().resolve() if directory else default_weights_dir()
        )
        self._verified = {}

    def require(self, filename: str) -> Path:
        """Refuse missing or corrupted official checkpoints before deserialization."""
        path = self.directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model weight: {path}. Run 'aipampds download-weights "
                f"--weights-dir {str(self.directory)!r}'."
            )
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(filename) != signature:
            if sha256(path) != WEIGHTS[filename]:
                raise RuntimeError(
                    f"SHA256 verification failed for {path}. "
                    "Run 'aipampds download-weights' with the same weights directory to repair it."
                )
            self._verified[filename] = signature
        return path
