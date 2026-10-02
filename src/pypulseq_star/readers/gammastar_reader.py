"""gammaSTAR ``.seq.json`` reader for pypulseq_star.

This module provides a lossless document-level import path for existing
GammaSTAR sequence JSON files.

Design goals
------------
* Preserve the complete source document, including fields that pypulseq_star
  does not yet interpret semantically.
* Validate the small top-level contract required for useful gammaSTAR sequence
  documents without attempting to execute embedded Lua scripts.
* Support a stable read -> inspect/modify -> write round trip.
* Keep document parsing separate from the STAR semantic sequence importer that
  can be added incrementally later.

The reader intentionally does *not* execute gammaSTAR parameter scripts and it
currently does not convert an arbitrary gammaSTAR document into a
``SeqStarSequence``.  Those are separate semantic-import concerns; preserving
unknown constructs losslessly is preferable to silently dropping them.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MISSING = object()


@dataclass(slots=True)
class GammaStarDocument:
    """Lossless in-memory representation of a gammaSTAR ``.seq.json`` document.

    Parameters
    ----------
    data
        Parsed JSON object.  A deep copy is stored so subsequent changes to the
        caller's dictionary do not mutate the document unexpectedly.
    source_path
        Optional path from which the document was loaded.

    Notes
    -----
    The complete JSON object is retained.  ``parameters`` and
    ``sequence_elements`` are exposed through convenience properties, but no
    unknown root-level or nested fields are discarded.
    """

    _data: dict[str, Any]
    source_path: Path | None = None

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError(
                "A gammaSTAR document must be a JSON object / mapping; "
                f"got {type(data).__name__}."
            )
        self._data = copy.deepcopy(dict(data))
        self.source_path = Path(source_path) if source_path is not None else None

    @property
    def name(self) -> str | None:
        """Return the document name when present."""

        value = self._data.get("name")
        return None if value is None else str(value)

    @name.setter
    def name(self, value: str) -> None:
        self._data["name"] = str(value)

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the mutable gammaSTAR parameter mapping."""

        value = self._data.setdefault("parameters", {})
        if not isinstance(value, dict):
            raise TypeError("gammaSTAR 'parameters' must be a JSON object.")
        return value

    @property
    def sequence_elements(self) -> dict[str, Any]:
        """Return the mutable gammaSTAR sequence-element mapping."""

        value = self._data.setdefault("sequence_elements", {})
        if not isinstance(value, dict):
            raise TypeError("gammaSTAR 'sequence_elements' must be a JSON object.")
        return value

    @property
    def seqstar_writer_context(self) -> dict[str, Any] | None:
        """Return optional pypulseq-star writer metadata, if present."""

        value = self._data.get("seqstar_writer_context")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(
                "gammaSTAR 'seqstar_writer_context' must be a JSON object when present."
            )
        return value

    @property
    def extra_fields(self) -> dict[str, Any]:
        """Return root-level fields outside the common gammaSTAR document keys."""

        common = {"name", "parameters", "sequence_elements", "seqstar_writer_context"}
        return {
            key: copy.deepcopy(value)
            for key, value in self._data.items()
            if key not in common
        }

    def to_dict(self, *, copy_data: bool = True) -> dict[str, Any]:
        """Return the complete gammaSTAR JSON object.

        ``copy_data=True`` is the safe default.  ``False`` exposes the live
        underlying dictionary for advanced callers that intentionally want
        direct mutation.
        """

        return copy.deepcopy(self._data) if copy_data else self._data

    def get_parameter(self, path: str, default: Any = None) -> Any:
        """Return a gammaSTAR parameter record by its full path."""

        return self.parameters.get(path, default)

    def set_parameter(self, path: str, value: Any) -> None:
        """Set or replace a gammaSTAR parameter record by its full path."""

        self.parameters[str(path)] = value

    def get_sequence_element(self, path: str, default: Any = None) -> Any:
        """Return a sequence-element blueprint/reference by full path."""

        return self.sequence_elements.get(path, default)

    def set_sequence_element(self, path: str, value: Any) -> None:
        """Set or replace a sequence-element blueprint/reference by full path."""

        self.sequence_elements[str(path)] = value

    def validate(self, *, strict: bool = True) -> list[str]:
        """Validate the document-level gammaSTAR structure.

        This is deliberately structural rather than executable validation: the
        reader does not run embedded Lua scripts or require knowledge of every
        gammaSTAR blueprint UUID.

        Parameters
        ----------
        strict
            If ``True``, missing ``parameters`` or ``sequence_elements`` are
            reported as validation errors.  If ``False``, only malformed fields
            that are present are reported.

        Returns
        -------
        list[str]
            Human-readable validation errors.  An empty list means the document
            passes this reader's structural checks.
        """

        errors: list[str] = []

        if "name" in self._data and not isinstance(self._data["name"], str):
            errors.append("'name' must be a string when present.")

        for key in ("parameters", "sequence_elements"):
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                if strict:
                    errors.append(f"Missing required top-level field {key!r}.")
                continue
            if not isinstance(value, Mapping):
                errors.append(f"Top-level field {key!r} must be a JSON object.")

        context = self._data.get("seqstar_writer_context", _MISSING)
        if context is not _MISSING and not isinstance(context, Mapping):
            errors.append(
                "'seqstar_writer_context' must be a JSON object when present."
            )

        return errors

    def write(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
    ) -> Path:
        """Write the complete document back to ``.seq.json``.

        JSON whitespace/key ordering may differ from the source file, but
        ``json.load(source) == json.load(roundtrip)`` is expected when the
        document has not been modified.
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self._data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
            ),
            encoding="utf-8",
        )
        return output


class GammaStarReader:
    """Read existing gammaSTAR ``.seq.json`` files losslessly.

    The primary API is the class method ``GammaStarReader.read(path)`` so users
    do not need to construct an empty sequence or writer merely to inspect an
    existing gammaSTAR artifact.
    """

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        strict: bool = True,
        encoding: str = "utf-8",
    ) -> GammaStarDocument:
        """Read a gammaSTAR JSON document from disk.

        Parameters
        ----------
        path
            Existing ``.seq.json`` (or compatible JSON) file.
        strict
            Apply the reader's document-level required-field checks.
        encoding
            Text encoding used to read the JSON file.

        Returns
        -------
        GammaStarDocument
            Lossless document representation suitable for inspection,
            modification, and re-export.
        """

        source = Path(path)
        data = json.loads(source.read_text(encoding=encoding))
        document = GammaStarDocument(data, source_path=source)
        cls._raise_for_validation_errors(document, strict=strict, source=source)
        return document

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> GammaStarDocument:
        """Construct and validate a document from an already-parsed mapping."""

        document = GammaStarDocument(data)
        cls._raise_for_validation_errors(document, strict=strict, source=None)
        return document

    @staticmethod
    def _raise_for_validation_errors(
        document: GammaStarDocument,
        *,
        strict: bool,
        source: Path | None,
    ) -> None:
        errors = document.validate(strict=strict)
        if not errors:
            return

        location = f" in {source}" if source is not None else ""
        formatted = "\n  - ".join(errors)
        raise ValueError(
            f"Invalid gammaSTAR document{location}:\n  - {formatted}"
        )
