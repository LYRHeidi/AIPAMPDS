import hashlib
from io import BytesIO

import pytest

from aipampds import resources


def configure_resource(monkeypatch):
    content = b"test checkpoint bytes"
    monkeypatch.setattr(resources, "WEIGHTS", {"test.pt": hashlib.sha256(content).hexdigest()})
    return content


def test_download_verifies_and_reuses(monkeypatch, tmp_path):
    content = configure_resource(monkeypatch)
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return BytesIO(content)

    monkeypatch.setattr(resources, "urlopen", open_url)
    assert resources.download_weights(tmp_path) == [tmp_path / "test.pt"]
    resources.download_weights(tmp_path)
    assert len(calls) == 1
    assert list(tmp_path.glob("*.part")) == []


def test_bad_download_preserves_previous_file(monkeypatch, tmp_path):
    configure_resource(monkeypatch)
    destination = tmp_path / "test.pt"
    destination.write_bytes(b"previous file")
    monkeypatch.setattr(resources, "urlopen", lambda *args, **kwargs: BytesIO(b"corrupted"))
    with pytest.raises(RuntimeError, match="SHA256"):
        resources.download_weights(tmp_path)
    assert destination.read_bytes() == b"previous file"
    assert list(tmp_path.glob("*.part")) == []


def test_download_network_failure_removes_partial(monkeypatch, tmp_path):
    configure_resource(monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(resources, "urlopen", fail)
    with pytest.raises(OSError):
        resources.download_weights(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_require_missing_corrupted_and_verified(monkeypatch, tmp_path):
    content = configure_resource(monkeypatch)
    manager = resources.Resources(tmp_path)
    with pytest.raises(FileNotFoundError, match="download-weights"):
        manager.require("test.pt")
    path = tmp_path / "test.pt"
    path.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA256"):
        manager.require("test.pt")
    path.write_bytes(content)
    assert manager.require("test.pt") == path


def test_cache_defaults_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("AIPAMPDS_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert resources.default_weights_dir() == tmp_path / "aipampds" / "weights"
    monkeypatch.setenv("AIPAMPDS_WEIGHTS_DIR", str(tmp_path / "custom"))
    assert resources.default_weights_dir() == tmp_path / "custom"
