"""Utilities for validating PyPulseq-to-PyPulseq-Star migrations.

This package provides reusable infrastructure for inspecting source PyPulseq
scripts, validating migrated PyPulseq-Star representations, and comparing
their realized acquisitions.

Reference PyPulseq scripts used by the migration test suite are kept in the
``reference_scripts`` subpackage.
"""

from .validator import MigrationReport, MigrationValidator

__all__ = [
    "MigrationValidator",
    "MigrationReport",
]