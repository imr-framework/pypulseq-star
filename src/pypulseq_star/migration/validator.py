"""Validation utilities for PyPulseq-to-PyPulseq-Star migrations.

The migration validator is designed to answer three related questions:

1. Acquisition parameters
   Are acquisition parameters exposed by the source PyPulseq script represented
   appropriately in the migrated PyPulseq-Star protocol?

2. Sequence structure and events
   Are important source constructs such as loops, event mutations, and
   repetition-dependent behavior accounted for in the migrated sequence model?

3. Timing
   Are timing parameters and relationships retained, and does the resolved
   sequence remain timing-valid?

The validator intentionally distinguishes migration intent from concrete
realization equivalence. Matching Pulseq ``.seq`` files alone is not sufficient
to establish that a migration preserved the relationships expressed by the
source program.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# =============================================================================
# Validation result objects
# =============================================================================


@dataclass
class ValidationCheck:
    """Result of one deterministic migration-validation check."""

    domain: str
    name: str
    status: str

    source: Any | None = None
    migrated: Any | None = None

    reference_value: Any | None = None
    migrated_value: Any | None = None

    message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class MigrationReport:
    """Structured result returned by :class:`MigrationValidator`."""

    acquisition: list[ValidationCheck] = field(default_factory=list)
    structure: list[ValidationCheck] = field(default_factory=list)
    timing: list[ValidationCheck] = field(default_factory=list)

    unmapped_source_constructs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def checks(self) -> list[ValidationCheck]:
        """Return all validation checks in domain order."""
        return self.acquisition + self.structure + self.timing

    @property
    def passed(self) -> bool:
        """Return True when no required check has failed."""
        return all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        def serialize(check: ValidationCheck) -> dict[str, Any]:
            return {
                "domain": check.domain,
                "name": check.name,
                "status": check.status,
                "source": check.source,
                "migrated": check.migrated,
                "reference_value": check.reference_value,
                "migrated_value": check.migrated_value,
                "message": check.message,
            }

        return {
            "passed": self.passed,
            "acquisition": [serialize(c) for c in self.acquisition],
            "structure": [serialize(c) for c in self.structure],
            "timing": [serialize(c) for c in self.timing],
            "unmapped_source_constructs": list(
                self.unmapped_source_constructs
            ),
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        """Return a concise human-readable migration report."""

        lines = [
            "PyPulseq -> PyPulseq-Star migration validation",
            "=" * 48,
        ]

        sections = (
            ("Acquisition parameters", self.acquisition),
            ("Sequence structure / events", self.structure),
            ("Timing", self.timing),
        )

        for title, checks in sections:
            lines.extend(["", title, "-" * len(title)])

            if not checks:
                lines.append("  No checks performed.")
                continue

            for check in checks:
                marker = {
                    "pass": "PASS",
                    "fail": "FAIL",
                    "warn": "WARN",
                    "info": "INFO",
                }.get(check.status, check.status.upper())

                line = f"{marker:4}  {check.name}"

                if check.message:
                    line += f" -- {check.message}"

                lines.append(line)

        if self.unmapped_source_constructs:
            lines.extend(["", "Unmapped source constructs", "-" * 26])
            for item in self.unmapped_source_constructs:
                lines.append(f"WARN  {item}")

        if self.notes:
            lines.extend(["", "Notes", "-----"])
            for note in self.notes:
                lines.append(f"  {note}")

        lines.extend(
            [
                "",
                f"Overall: {'PASS' if self.passed else 'FAIL'}",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# Source-side representation
# =============================================================================


@dataclass
class SourceContract:
    """Deterministically extracted features of a PyPulseq source script."""

    function_parameters: dict[str, Any] = field(default_factory=dict)
    system_options: dict[str, Any] = field(default_factory=dict)
    definitions: dict[str, Any] = field(default_factory=dict)

    loops: list[dict[str, Any]] = field(default_factory=list)
    event_calls: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    mutations: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# AST helpers
# =============================================================================


def _name(node: ast.AST | None) -> str | None:
    """Return a dotted representation of a Python expression when possible."""

    if node is None:
        return None

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr

    return None


def _source_expression(node: ast.AST | None) -> str | None:
    """Return a compact source-like representation of an AST node."""

    if node is None:
        return None

    try:
        return ast.unparse(node)
    except Exception:
        return None


def _literal_or_expression(node: ast.AST | None) -> Any:
    """Return a literal value when possible, otherwise source expression."""

    if node is None:
        return None

    try:
        return ast.literal_eval(node)
    except Exception:
        return _source_expression(node)


class _PyPulseqSourceInspector(ast.NodeVisitor):
    """Extract a small deterministic contract from a PyPulseq source file."""

    EVENT_CONSTRUCTORS = {
        "make_adc",
        "make_arbitrary_grad",
        "make_block_pulse",
        "make_delay",
        "make_extended_trapezoid",
        "make_gauss_pulse",
        "make_sinc_pulse",
        "make_trapezoid",
    }

    def __init__(self) -> None:
        self.contract = SourceContract()
        self._loop_depth = 0

    # -------------------------------------------------------------------------
    # Function interface
    # -------------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Prefer the public sequence-construction function. For the first
        # migration fixture this is expected to be main().
        if node.name == "main":
            args = list(node.args.args)
            defaults = list(node.args.defaults)

            default_offset = len(args) - len(defaults)

            for index, arg in enumerate(args):
                default_index = index - default_offset

                if default_index >= 0:
                    default = _literal_or_expression(defaults[default_index])
                else:
                    default = None

                self.contract.function_parameters[arg.arg] = default

        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Calls
    # -------------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _name(node.func)

        if call_name:
            short_name = call_name.rsplit(".", 1)[-1]

            if short_name == "Opts":
                self.contract.system_options.update(
                    {
                        keyword.arg: _literal_or_expression(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg is not None
                    }
                )

            if short_name in self.EVENT_CONSTRUCTORS:
                self.contract.event_calls.append(
                    {
                        "constructor": short_name,
                        "call": _source_expression(node),
                        "inside_loop": self._loop_depth > 0,
                    }
                )

            if short_name == "set_definition":
                if len(node.args) >= 2:
                    key = _literal_or_expression(node.args[0])
                    value = _literal_or_expression(node.args[1])

                    if isinstance(key, str):
                        self.contract.definitions[key] = value

        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Loops
    # -------------------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        loop = {
            "target": _source_expression(node.target),
            "iterator": _source_expression(node.iter),
        }

        # Special-case the common PyPulseq pattern:
        #
        #     for i_phase in range(n_y):
        #
        if (
            isinstance(node.iter, ast.Call)
            and _name(node.iter.func) == "range"
        ):
            loop["range"] = [
                _source_expression(arg) for arg in node.iter.args
            ]

        self.contract.loops.append(loop)

        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    # -------------------------------------------------------------------------
    # Assignments and mutations
    # -------------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        value = _source_expression(node.value)

        for target in node.targets:
            target_name = _source_expression(target)

            entry = {
                "target": target_name,
                "value": value,
                "inside_loop": self._loop_depth > 0,
            }

            self.contract.assignments.append(entry)

            if isinstance(target, ast.Attribute):
                self.contract.mutations.append(entry)

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target = _source_expression(node.target)

        entry = {
            "target": target,
            "value": _source_expression(node.value),
            "operator": type(node.op).__name__,
            "inside_loop": self._loop_depth > 0,
        }

        self.contract.assignments.append(entry)

        if isinstance(node.target, ast.Attribute):
            self.contract.mutations.append(entry)

        self.generic_visit(node)


# =============================================================================
# Migration validator
# =============================================================================


class MigrationValidator:
    """Validate a PyPulseq-to-PyPulseq-Star migration.

    Parameters
    ----------
    source_path
        Path to the original PyPulseq source script.

    migrated_sequence
        Symbolic PyPulseq-Star sequence produced by the migrated script.

    source_sequence
        Optional concrete PyPulseq sequence produced by the source script.
        This will be used later for realization-level registration.

    migrated_realization
        Optional resolved PyPulseq-Star realization.

    Notes
    -----
    Validation is organized into three domains:

    1. acquisition parameters;
    2. sequence structure and events;
    3. timing.

    The source program is inspected statically using Python's AST. The source
    script is therefore not executed merely to determine its migration
    contract.
    """

    def __init__(
        self,
        source_path: str | Path,
        migrated_sequence: Any,
        *,
        source_sequence: Any | None = None,
        migrated_realization: Any | None = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.migrated_sequence = migrated_sequence
        self.source_sequence = source_sequence
        self.migrated_realization = migrated_realization

        self.source_contract: SourceContract | None = None

    # =========================================================================
    # Inspection
    # =========================================================================

    def inspect_source(self) -> SourceContract:
        """Inspect the source PyPulseq script without executing it."""

        if not self.source_path.exists():
            raise FileNotFoundError(
                f"PyPulseq source script not found: {self.source_path}"
            )

        source = self.source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.source_path))

        inspector = _PyPulseqSourceInspector()
        inspector.visit(tree)

        self.source_contract = inspector.contract
        return inspector.contract

    def inspect_migration(self) -> dict[str, Any]:
        """Return a conservative description of the STAR representation.

        This method intentionally uses public/high-level sequence attributes
        where possible. More detailed relationship inspection can be added as
        the migration API stabilizes.
        """

        sequence = self.migrated_sequence

        protocol = getattr(sequence, "protocol", None)
        system = getattr(sequence, "system", None)

        protocol_parameters = {}

        if protocol is not None:
            parameters = getattr(protocol, "parameters", None)

            if isinstance(parameters, Mapping):
                protocol_parameters = dict(parameters)

        return {
            "protocol_parameters": protocol_parameters,
            "protocol": protocol,
            "system": system,
            "sequence": sequence,
        }

    # =========================================================================
    # Acquisition parameters
    # =========================================================================

    def validate_acquisition_parameters(self) -> list[ValidationCheck]:
        """Validate migration of source acquisition parameters."""

        source = self._require_source_contract()
        migrated = self.inspect_migration()

        protocol_parameters = migrated["protocol_parameters"]

        checks: list[ValidationCheck] = []

        # Parameters that describe output behavior rather than acquisition
        # intent should not be required to appear in the STAR protocol.
        non_acquisition_parameters = {
            "plot",
            "test_report",
            "write_seq",
            "write_json",
            "seq_filename",
            "json_filename",
        }

        source_parameters = {
            name: value
            for name, value in source.function_parameters.items()
            if name not in non_acquisition_parameters
        }

        for name, source_value in source_parameters.items():
            present = name in protocol_parameters

            if present:
                checks.append(
                    ValidationCheck(
                        domain="acquisition",
                        name=name,
                        status="pass",
                        source=f"source parameter '{name}'",
                        migrated=f"protocol parameter '{name}'",
                        reference_value=source_value,
                        migrated_value=protocol_parameters[name],
                        message="source acquisition parameter is represented "
                        "in the STAR protocol",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        domain="acquisition",
                        name=name,
                        status="fail",
                        source=f"source parameter '{name}'",
                        migrated=None,
                        reference_value=source_value,
                        message="source acquisition parameter is not represented "
                        "in the STAR protocol",
                    )
                )

        # Report STAR protocol parameters with no source-function counterpart.
        for name, migrated_value in protocol_parameters.items():
            if name not in source_parameters:
                checks.append(
                    ValidationCheck(
                        domain="acquisition",
                        name=name,
                        status="info",
                        source=None,
                        migrated=f"protocol parameter '{name}'",
                        migrated_value=migrated_value,
                        message="STAR protocol parameter has no direct "
                        "source-function counterpart",
                    )
                )

        return checks

    # =========================================================================
    # Sequence structure / events
    # =========================================================================

    def validate_sequence_structure(self) -> list[ValidationCheck]:
        """Validate important sequence-structure migration constructs.

        The first implementation deliberately provides deterministic source
        observations rather than attempting general semantic equivalence.

        More specific source-to-STAR mappings, such as:

            source loop over n_y
                -> STAR repeated node with factor=p.n_y

            source rf.phase_offset mutation
                -> STAR phase_offset variation

        should be added here as reusable recognizers.
        """

        source = self._require_source_contract()

        checks: list[ValidationCheck] = []

        # ---------------------------------------------------------------------
        # Source repetition constructs
        # ---------------------------------------------------------------------

        for index, loop in enumerate(source.loops):
            iterator = loop.get("iterator")

            checks.append(
                ValidationCheck(
                    domain="structure",
                    name=f"source_loop_{index}",
                    status="info",
                    source=iterator,
                    message=(
                        "source loop detected; a corresponding STAR repetition "
                        "relationship should be established"
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Mutations inside loops
        # ---------------------------------------------------------------------

        for mutation in source.mutations:
            if not mutation.get("inside_loop"):
                continue

            target = mutation.get("target")

            checks.append(
                ValidationCheck(
                    domain="structure",
                    name=f"loop_mutation:{target}",
                    status="info",
                    source=mutation,
                    message=(
                        "repetition-dependent source mutation detected; "
                        "expected to map to a retained STAR variation"
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Event constructors
        # ---------------------------------------------------------------------

        constructors = sorted(
            {
                event["constructor"]
                for event in source.event_calls
                if event.get("constructor")
            }
        )

        for constructor in constructors:
            checks.append(
                ValidationCheck(
                    domain="structure",
                    name=f"event:{constructor}",
                    status="info",
                    source=constructor,
                    message="source event constructor detected",
                )
            )

        return checks

    # =========================================================================
    # Timing
    # =========================================================================

    def validate_timing(self) -> list[ValidationCheck]:
        """Validate timing-related aspects of the migration."""

        source = self._require_source_contract()

        checks: list[ValidationCheck] = []

        # ---------------------------------------------------------------------
        # Source timing parameters
        # ---------------------------------------------------------------------

        for parameter in ("te", "tr"):
            if parameter in source.function_parameters:
                checks.append(
                    ValidationCheck(
                        domain="timing",
                        name=f"{parameter}_source_parameter",
                        status="pass",
                        source=parameter,
                        reference_value=source.function_parameters[parameter],
                        message=(
                            f"{parameter.upper()} is explicitly exposed by "
                            "the source sequence"
                        ),
                    )
                )

        # ---------------------------------------------------------------------
        # STAR timing validity
        # ---------------------------------------------------------------------

        realization = self.migrated_realization

        if realization is None:
            checks.append(
                ValidationCheck(
                    domain="timing",
                    name="star_timing_check",
                    status="warn",
                    message=(
                        "no migrated realization supplied; resolved timing "
                        "validation was not performed"
                    ),
                )
            )
            return checks

        check_timing = getattr(realization, "check_timing", None)

        if not callable(check_timing):
            checks.append(
                ValidationCheck(
                    domain="timing",
                    name="star_timing_check",
                    status="warn",
                    message=(
                        "migrated realization does not expose check_timing()"
                    ),
                )
            )
            return checks

        try:
            ok, errors = check_timing()
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    domain="timing",
                    name="star_timing_check",
                    status="fail",
                    message=f"timing validation raised {type(exc).__name__}: {exc}",
                )
            )
            return checks

        if ok:
            checks.append(
                ValidationCheck(
                    domain="timing",
                    name="star_timing_check",
                    status="pass",
                    message="resolved PyPulseq-Star timing check passed",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    domain="timing",
                    name="star_timing_check",
                    status="fail",
                    migrated=errors,
                    message="resolved PyPulseq-Star timing check failed",
                )
            )

        return checks

    # =========================================================================
    # Complete validation
    # =========================================================================

    def validate(self) -> MigrationReport:
        """Run all currently implemented migration checks."""

        if self.source_contract is None:
            self.inspect_source()

        acquisition = self.validate_acquisition_parameters()
        structure = self.validate_sequence_structure()
        timing = self.validate_timing()

        unmapped = self._find_unmapped_source_constructs(structure)

        report = MigrationReport(
            acquisition=acquisition,
            structure=structure,
            timing=timing,
            unmapped_source_constructs=unmapped,
        )

        if self.source_sequence is None:
            report.notes.append(
                "No concrete PyPulseq reference sequence was supplied. "
                "Realization-level registration has not yet been performed."
            )

        return report

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _require_source_contract(self) -> SourceContract:
        if self.source_contract is None:
            return self.inspect_source()

        return self.source_contract

    @staticmethod
    def _find_unmapped_source_constructs(
        structure_checks: Iterable[ValidationCheck],
    ) -> list[str]:
        """Collect source constructs that have not yet been mapped.

        For the initial implementation, structure observations marked INFO are
        considered candidates awaiting an explicit source-to-STAR mapping.

        As recognizers are implemented, mapped constructs should instead emit
        PASS/FAIL and will naturally disappear from this list.
        """

        unmapped: list[str] = []

        for check in structure_checks:
            if check.status != "info":
                continue

            if check.name.startswith("source_loop_"):
                unmapped.append(
                    f"{check.name}: source repetition construct has not yet "
                    "been mapped to a STAR node"
                )

            elif check.name.startswith("loop_mutation:"):
                unmapped.append(
                    f"{check.name}: source repetition-dependent mutation has "
                    "not yet been mapped to a STAR variation"
                )

        return unmapped