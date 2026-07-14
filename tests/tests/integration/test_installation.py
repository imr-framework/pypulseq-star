"""Installation-level smoke tests."""

from __future__ import annotations

import importlib.metadata

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def test_package_is_importable() -> None:
    import pypulseq_star

    assert pypulseq_star is not None


def test_installed_distribution_has_version() -> None:
    try:
        version = importlib.metadata.version("pypulseq-star")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Package is running from source rather than an installed distribution")

    assert version
