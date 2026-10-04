from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from pypulseq_star.migration.validator import MigrationValidator

# ============================================================================
# Paths
# ============================================================================


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _reference_dir() -> Path:
    """Return the directory containing curated PyPulseq reference scripts."""
    return (
        _repo_root()
        / "src"
        / "pypulseq_star"
        / "migration"
        / "reference_scripts"
    )


def _migration_example_dir() -> Path:
    """Return the directory containing migrated PyPulseq-Star examples."""
    return _repo_root() / "examples" / "pp_to_ppstar"


# ============================================================================
# Fixture discovery
# ============================================================================


def _reference_scripts() -> list[Path]:
    """Discover curated PyPulseq migration reference scripts."""
    return sorted(
        path
        for path in _reference_dir().glob("*.py")
        if path.name != "__init__.py"
    )


def _migration_name(reference_script: Path) -> str:
    """Return the expected migrated example filename.

    Naming convention
    -----------------
    write_gre.py
        -> demo_GRE_pp_to_ppstar.py

    write_epi.py
        -> demo_EPI_pp_to_ppstar.py

    write_tse.py
        -> demo_TSE_pp_to_ppstar.py
    """
    stem = reference_script.stem

    if stem.startswith("write_"):
        sequence_name = stem.removeprefix("write_")
    else:
        sequence_name = stem

    return f"demo_{sequence_name.upper()}_pp_to_ppstar.py"


def _migration_example(reference_script: Path) -> Path:
    """Return the migrated example corresponding to a reference script."""
    return _migration_example_dir() / _migration_name(reference_script)


# ============================================================================
# Dynamic module loading
# ============================================================================


def _load_module(path: Path):
    """Load a Python module directly from a file path."""
    module_name = f"_migration_test_{path.stem}"

    spec = spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module: {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================================
# Test parameterization
# ============================================================================


REFERENCE_SCRIPTS = _reference_scripts()


@pytest.mark.parametrize(
    "reference_script",
    REFERENCE_SCRIPTS,
    ids=lambda path: path.stem,
)
def test_migration_validator(reference_script: Path):
    """Validate each curated PyPulseq-to-PyPulseq-Star migration.

    Each Python source fixture under ``migration/reference_scripts`` is paired
    with a migrated example under ``examples/pp_to_ppstar`` according to the
    repository naming convention.

    At this stage the test verifies that the migration validator can inspect
    both sides and produce structured results for all three validation domains.
    More stringent domain-specific registration criteria can be added as the
    validator matures.
    """

    migrated_path = _migration_example(reference_script)

    assert migrated_path.exists(), (
        f"No migrated PyPulseq-Star example found for "
        f"{reference_script.name}. Expected: {migrated_path}"
    )

    migrated_module = _load_module(migrated_path)

    assert hasattr(migrated_module, "main"), (
        f"Migrated example does not expose main(): {migrated_path}"
    )

    # ------------------------------------------------------------------------
    # Construct the migrated PyPulseq-Star sequence
    # ------------------------------------------------------------------------

    seq = migrated_module.main(
        plot=False,
        test_report=False,
        write_seq=False,
        write_json=False,
    )

    resolved = seq.resolve()

    # ------------------------------------------------------------------------
    # Run migration validation
    # ------------------------------------------------------------------------

    validator = MigrationValidator(
        source_path=reference_script,
        migrated_sequence=seq,
        migrated_realization=resolved,
    )

    report = validator.validate()

    print()
    print(f"Reference: {reference_script.name}")
    print(f"Migrated : {migrated_path.name}")
    print()
    print(report.to_text())

    # ------------------------------------------------------------------------
    # Baseline validator expectations
    # ------------------------------------------------------------------------

    assert report is not None

    assert report.acquisition, (
        f"No acquisition checks produced for {reference_script.name}"
    )

    assert report.structure, (
        f"No structure checks produced for {reference_script.name}"
    )

    assert report.timing, (
        f"No timing checks produced for {reference_script.name}"
    )


def test_reference_script_set_is_curated():
    """Ensure migration fixtures remain an explicitly curated test set."""

    scripts = [path.name for path in REFERENCE_SCRIPTS]

    assert scripts, (
        "No PyPulseq migration reference scripts were found."
    )

    assert "__init__.py" not in scripts