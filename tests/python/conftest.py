from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def smoke_config_path(tmp_path: Path, project_root: Path) -> Path:
    source = project_root / "configs" / "local-smoke.toml"
    text = source.read_text()
    text = text.replace(
        'output_root = "../artifacts/local-smoke-diagnostics-v2"',
        f'output_root = "{tmp_path / "out"}"',
    )
    text = text.replace(
        'scratch_root = "../scratch/local-smoke-diagnostics-v2"',
        f'scratch_root = "{tmp_path / "scratch"}"',
    )
    text = text.replace('project_root = ".."', f'project_root = "{project_root}"')
    destination = tmp_path / "smoke.toml"
    destination.write_text(text)
    return destination


@pytest.fixture(autouse=True)
def writable_matplotlib_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "matplotlib"
    cache.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(cache))
