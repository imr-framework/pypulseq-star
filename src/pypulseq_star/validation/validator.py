"""Composable validation primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationIssue:
    """A validation finding with a path and actionable message."""

    path: str
    message: str
    severity: str = "error"


class Validator:
    """Base validator interface."""

    def validate(self, obj: object) -> list[ValidationIssue]:
        """Return validation issues for an object."""

        return []
