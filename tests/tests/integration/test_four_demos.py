"""Smoke tests for the four beginner-facing demo scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def find_repository_root(start: Path) -> Path:
    """Locate the repository root from any nested tests layout."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root containing pyproject.toml and examples/.")



def load_demo(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import demo: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "filename",
    ["demo_FID.py", "demo_GRE.py", "demo_EPI.py", "demo_TSE.py"],
)
def test_demo_runs_without_plotting_or_export(filename: str) -> None:
    repo_root = find_repository_root(Path(__file__).resolve())
    demo_path = repo_root / "examples" / filename
    if not demo_path.exists():
        pytest.skip(f"{demo_path} is not present in this checkout")

    module = load_demo(demo_path)
    module.main(plot=False, write_seq=False, write_json=False, debug=False)


@pytest.mark.parametrize(
    "filename",
    ["demo_FID.py", "demo_GRE.py", "demo_EPI.py", "demo_TSE.py"],
)
def test_demo_exports_files(filename: str, tmp_path: Path, monkeypatch) -> None:
    repo_root = find_repository_root(Path(__file__).resolve())
    demo_path = repo_root / "examples" / filename
    if not demo_path.exists():
        pytest.skip(f"{demo_path} is not present in this checkout")

    module = load_demo(demo_path)
    monkeypatch.chdir(tmp_path)

    module.main(plot=False, write_seq=True, write_json=True, debug=False)

    sequence_name = filename.removeprefix("demo_").removesuffix(".py").lower()
    output_dir = tmp_path / "out" / sequence_name
    assert list(output_dir.glob("*.seq"))
    assert list(output_dir.glob("*.seq.json"))
