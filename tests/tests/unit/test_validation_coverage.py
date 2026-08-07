from __future__ import annotations

import pytest

from pypulseq_star.validation.validator import ValidationIssue, Validator


@pytest.mark.unit
def test_validation_issue_defaults_and_base_validator() -> None:
    issue = ValidationIssue(path="sequence.block[0]", message="invalid")
    assert issue.severity == "error"
    assert issue.path == "sequence.block[0]"
    assert Validator().validate(object()) == []
