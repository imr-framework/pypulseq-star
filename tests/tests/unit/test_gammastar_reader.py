"""Regression and API tests for the standalone GammaStarReader.

The external fixture is an independently created gammaSTAR FLASH
``.seq.json`` stored under ``tests/sequences/Demo_FLASH_sequence.seq.json``.

The principal contract is lossless document handling:

    read -> inspect / modify -> write

must preserve complete JSON semantics, including fields that PyPulseq-Star
does not yet interpret.

These tests also exercise the document-level validation and mutation API so
the reader remains useful independently of future semantic-import support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pypulseq_star.readers import GammaStarDocument, GammaStarReader

SEQUENCE_DIR = Path(__file__).resolve().parents[2] / "sequences"
REFERENCE_JSON = SEQUENCE_DIR / "Demo_FLASH_sequence.seq.json"


# =============================================================================
# External gammaSTAR reference document
# =============================================================================


def test_gammastar_reader_loads_external_flash_document() -> None:
    """GammaStarReader should expose the complete external document."""

    assert REFERENCE_JSON.exists(), (
        f"Missing gammaSTAR reference sequence: {REFERENCE_JSON}"
    )

    document = GammaStarReader.read(
        REFERENCE_JSON
    )

    assert isinstance(
        document,
        GammaStarDocument,
    )

    assert document.source_path == REFERENCE_JSON

    # Known fields in the supplied external gammaSTAR FLASH document.
    assert document.name == "Demo FLASH sequence"

    assert document.get_parameter(
        "root.prot.TE"
    ) is not None

    assert document.get_parameter(
        "root.prot.TR"
    ) is not None

    assert document.get_sequence_element(
        "root"
    ) is not None

    # Structural validation should succeed without executing embedded Lua.
    assert document.validate(
        strict=True
    ) == []


# =============================================================================
# Lossless round trip
# =============================================================================


def test_gammastar_reader_lossless_json_roundtrip(
    tmp_path: Path,
) -> None:
    """Unmodified gammaSTAR JSON must survive read -> write without data loss."""

    original = json.loads(
        REFERENCE_JSON.read_text(
            encoding="utf-8"
        )
    )

    document = GammaStarReader.read(
        REFERENCE_JSON
    )

    roundtrip_path = document.write(
        tmp_path
        / "Demo_FLASH_sequence.roundtrip.seq.json"
    )

    roundtrip = json.loads(
        roundtrip_path.read_text(
            encoding="utf-8"
        )
    )

    # JSON whitespace and key order may change, but the parsed document must be
    # exactly equivalent, including unknown fields, scripts, tests, and UUIDs.
    assert roundtrip == original

    assert document.to_dict() == original


# =============================================================================
# Construction and copy semantics
# =============================================================================


def test_gammastar_document_deep_copies_input_mapping() -> None:
    """Mutating the caller's mapping must not mutate the stored document."""

    source = {
        "name": "example",
        "parameters": {
            "root.prot.TR": {
                "value": 1.0,
            }
        },
        "sequence_elements": {
            "root": {
                "type": "sequence",
            }
        },
    }

    document = GammaStarDocument(
        source
    )

    source["parameters"]["root.prot.TR"]["value"] = 2.0

    assert (
        document.parameters[
            "root.prot.TR"
        ]["value"]
        == 1.0
    )


def test_gammastar_document_to_dict_copy_and_live_view() -> None:
    """to_dict should support both safe-copy and explicit live mutation modes."""

    document = GammaStarReader.from_dict(
        {
            "name": "example",
            "parameters": {},
            "sequence_elements": {},
        }
    )

    copied = document.to_dict()

    copied["name"] = "changed-copy"

    assert document.name == "example"

    live = document.to_dict(
        copy_data=False
    )

    live["name"] = "changed-live"

    assert document.name == "changed-live"


# =============================================================================
# Convenience mutation API
# =============================================================================


def test_gammastar_document_mutation_helpers() -> None:
    """Named fields, parameters, and elements should be directly editable."""

    document = GammaStarReader.from_dict(
        {
            "name": "before",
            "parameters": {},
            "sequence_elements": {},
        }
    )

    document.name = "after"

    document.set_parameter(
        "root.prot.TE",
        {
            "value": 0.020,
        },
    )

    document.set_sequence_element(
        "root.kernel",
        {
            "type": "kernel",
        },
    )

    assert document.name == "after"

    assert document.get_parameter(
        "root.prot.TE"
    ) == {
        "value": 0.020,
    }

    assert document.get_sequence_element(
        "root.kernel"
    ) == {
        "type": "kernel",
    }

    # Missing values should honor caller-provided defaults.
    sentinel = object()

    assert (
        document.get_parameter(
            "missing.parameter",
            sentinel,
        )
        is sentinel
    )

    assert (
        document.get_sequence_element(
            "missing.element",
            sentinel,
        )
        is sentinel
    )


# =============================================================================
# Optional and unknown fields
# =============================================================================


def test_gammastar_document_context_and_extra_fields() -> None:
    """Optional writer context and unknown root fields must remain accessible."""

    document = GammaStarReader.from_dict(
        {
            "name": "example",
            "parameters": {},
            "sequence_elements": {},
            "seqstar_writer_context": {
                "backend": "gammaSTAR",
            },
            "custom_extension": {
                "keep": True,
            },
        }
    )

    assert document.seqstar_writer_context == {
        "backend": "gammaSTAR",
    }

    assert document.extra_fields == {
        "custom_extension": {
            "keep": True,
        }
    }


def test_gammastar_document_context_is_optional() -> None:
    """Absence of writer-specific metadata should return None."""

    document = GammaStarReader.from_dict(
        {
            "name": "example",
            "parameters": {},
            "sequence_elements": {},
        }
    )

    assert document.seqstar_writer_context is None


# =============================================================================
# Strict versus permissive structural validation
# =============================================================================


def test_gammastar_validation_strict_reports_missing_required_fields() -> None:
    """Strict validation should report missing document-level mappings."""

    document = GammaStarDocument(
        {
            "name": "incomplete",
        }
    )

    errors = document.validate(
        strict=True
    )

    assert (
        "Missing required top-level field 'parameters'."
        in errors
    )

    assert (
        "Missing required top-level field 'sequence_elements'."
        in errors
    )


def test_gammastar_validation_non_strict_allows_missing_fields() -> None:
    """Non-strict validation should permit omitted optional import structure."""

    document = GammaStarDocument(
        {
            "name": "partial",
        }
    )

    assert document.validate(
        strict=False
    ) == []


def test_gammastar_validation_reports_malformed_fields() -> None:
    """Malformed root fields should produce structural validation errors."""

    document = GammaStarDocument(
        {
            "name": 123,
            "parameters": [],
            "sequence_elements": "not-a-mapping",
            "seqstar_writer_context": [],
        }
    )

    errors = document.validate(
        strict=True
    )

    assert (
        "'name' must be a string when present."
        in errors
    )

    assert (
        "Top-level field 'parameters' must be a JSON object."
        in errors
    )

    assert (
        "Top-level field 'sequence_elements' must be a JSON object."
        in errors
    )

    assert (
        "'seqstar_writer_context' must be a JSON object when present."
        in errors
    )


# =============================================================================
# Reader validation failures
# =============================================================================


def test_gammastar_reader_from_dict_rejects_invalid_document() -> None:
    """from_dict should raise when strict structural validation fails."""

    with pytest.raises(
        ValueError,
        match="Invalid gammaSTAR document",
    ):
        GammaStarReader.from_dict(
            {
                "name": "invalid",
            },
            strict=True,
        )


def test_gammastar_reader_from_dict_can_be_non_strict() -> None:
    """from_dict should support partial documents when strict=False."""

    document = GammaStarReader.from_dict(
        {
            "name": "partial",
        },
        strict=False,
    )

    assert document.name == "partial"


def test_gammastar_reader_read_rejects_invalid_file(
    tmp_path: Path,
) -> None:
    """read should report malformed gammaSTAR structure from disk."""

    path = (
        tmp_path
        / "invalid.seq.json"
    )

    path.write_text(
        json.dumps(
            {
                "name": "invalid",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid gammaSTAR document",
    ) as exc_info:
        GammaStarReader.read(
            path,
            strict=True,
        )

    # Disk-based validation errors should identify their source.
    assert str(path) in str(
        exc_info.value
    )


# =============================================================================
# Type errors and malformed convenience fields
# =============================================================================


def test_gammastar_document_requires_mapping_input() -> None:
    """The document constructor should reject non-mapping JSON roots."""

    with pytest.raises(
        TypeError,
        match="must be a JSON object / mapping",
    ):
        GammaStarDocument(
            ["not", "a", "mapping"]  # type: ignore[arg-type]
        )


def test_gammastar_parameters_property_rejects_non_mapping() -> None:
    """Malformed parameters should fail when accessed through the convenience API."""

    document = GammaStarDocument(
        {
            "name": "invalid",
            "parameters": [],
            "sequence_elements": {},
        }
    )

    with pytest.raises(
        TypeError,
        match="'parameters' must be a JSON object",
    ):
        _ = document.parameters


def test_gammastar_sequence_elements_property_rejects_non_mapping() -> None:
    """Malformed sequence_elements should fail through the convenience API."""

    document = GammaStarDocument(
        {
            "name": "invalid",
            "parameters": {},
            "sequence_elements": [],
        }
    )

    with pytest.raises(
        TypeError,
        match="'sequence_elements' must be a JSON object",
    ):
        _ = document.sequence_elements


def test_gammastar_writer_context_property_rejects_non_mapping() -> None:
    """Malformed optional writer metadata should fail explicitly."""

    document = GammaStarDocument(
        {
            "name": "invalid",
            "parameters": {},
            "sequence_elements": {},
            "seqstar_writer_context": [],
        }
    )

    with pytest.raises(
        TypeError,
        match="'seqstar_writer_context' must be a JSON object",
    ):
        _ = document.seqstar_writer_context