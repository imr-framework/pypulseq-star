"""Validation and safe repair of unintended literals in gammaSTAR documents.

Policy
------
Literal defaults are allowed in protocol/system declarations, descriptive
metadata, and structural fields, including immutable event-local block offsets. Derived numeric values in the executable
sequence graph must be represented by relationships. ADC effective dwell is an
event-level value derived from protocol receiver bandwidth and system ADC raster.

The fixer is fail-closed:
- it applies relationships registered by the symbolic compiler;
- it applies a small set of universally valid identities;
- it verifies that every relationship input references an existing parameter;
- it evaluates literal policy independently from graph connectivity;
- it never infers a relationship from numerical coincidence.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LiteralIssue:
    path: str
    value: Any
    category: str
    message: str
    expected_source: str | None = None
    fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "value": self.value,
            "category": self.category,
            "message": self.message,
            "expected_source": self.expected_source,
            "fixed": self.fixed,
        }


@dataclass(slots=True)
class LiteralValidationReport:
    issues: list[LiteralIssue] = field(default_factory=list)
    fixed_paths: list[str] = field(default_factory=list)

    @property
    def unresolved(self) -> list[LiteralIssue]:
        return [issue for issue in self.issues if not issue.fixed]

    @property
    def ok(self) -> bool:
        return not self.unresolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fixed_paths": list(self.fixed_paths),
            "issues": [issue.to_dict() for issue in self.issues],
            "unresolved_count": len(self.unresolved),
        }


class GammaStarLiteralValidationError(ValueError):
    """Raised when unresolved derived literals remain."""

    def __init__(self, report: LiteralValidationReport) -> None:
        self.report = report
        lines = [
            "gammaSTAR literal validation failed.",
            (
                f"{len(report.unresolved)} derived literal(s) remain "
                "without an authoritative relationship."
            ),
        ]
        for issue in report.unresolved[:25]:
            lines.append(
                f"- {issue.path}: {issue.value!r} "
                f"[{issue.category}] — {issue.message}"
            )
        if len(report.unresolved) > 25:
            lines.append(
                f"... and {len(report.unresolved) - 25} additional issue(s)."
            )
        super().__init__("\n".join(lines))


class GammaStarLiteralValidator:
    """Detect and repair unintended literals in a gammaSTAR parameter graph."""

    _DECLARATION_PREFIXES = (
        "root.prot.",
        "root.sys.",
        "root.info.",
        "root.tests.",
        "root.expo.",
    )

    _STRUCTURAL_SUFFIXES = (
        ".counter",
        ".length",
        ".enabled",
        ".set_enabled",
        ".enabled_single",
        ".type",
        ".kind",
        ".role",
        ".mode",
        ".trajectory",
        ".channel",
        ".direction",
        ".polarity",
        ".node",
        ".parent",
        ".asymmetry",
        ".spoilphase",
        ".idx_slice",
        ".idx_kspace_encode_step_1",
        ".idx_kspace_encode_step_2",
        ".read_dir",
        ".phase_dir",
        ".slice_dir",
        ".position",
        ".offcenter",
        ".seqstar_local_tstart",
    )

    _DERIVED_SUFFIXES = (
        ".duration",
        ".tcenter",
        ".tend",
        ".flip_angle",
        ".frequency",
        ".phase",
        ".number_of_samples",
        ".num_samples",
        ".sample_time",
        ".dwell",
        ".area",
        ".amplitude",
        ".rise_time",
        ".flat_time",
        ".fall_time",
        ".window_data",
        ".samples",
        ".center_sample",
    )

    _ALLOWED_LOCAL_ZERO_TSTART = (
        re.compile(r"^root\.tstart$"),
        re.compile(r"\.kernel\.tstart$"),
        re.compile(r"\.rf\.tstart$"),
        re.compile(r"\.adc\.tstart$"),
        re.compile(r"\.grad\.tstart$"),
    )

    def __init__(
        self,
        *,
        fail_on_unresolved: bool = True,
        allow_structural_literals: bool = True,
    ) -> None:
        self.fail_on_unresolved = bool(fail_on_unresolved)
        self.allow_structural_literals = bool(allow_structural_literals)

    def validate_and_fix(
        self,
        parameters: dict[str, Any],
        *,
        relationship_replacements: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> LiteralValidationReport:
        """Apply safe fixes, then reject unresolved derived literals."""

        replacements = dict(relationship_replacements or {})
        report = LiteralValidationReport()

        for path, relationship in replacements.items():
            current = parameters.get(path)
            if current is None or not self._is_literal_parameter(current):
                continue
            old_value = self._literal_value(current)
            parameters[path] = dict(relationship)
            report.fixed_paths.append(path)
            report.issues.append(
                LiteralIssue(
                    path=path,
                    value=old_value,
                    category="registered_relationship",
                    message=(
                        "Replaced resolved literal using the relationship "
                        "registered by the symbolic compiler."
                    ),
                    expected_source="symbolic relationship registry",
                    fixed=True,
                )
            )

        self._apply_universal_repairs(parameters, report)
        self._validate_dependency_roots(parameters, report)
        self._validate_adc_bandwidth_control(parameters, report)
        self._validate_variation_backend_targets(parameters, report)
        self._validate_adc_header_offcenter(parameters, report)

        for path, parameter in list(parameters.items()):
            if not self._is_literal_parameter(parameter):
                continue
            value = self._literal_value(parameter)
            category = self._classify(path, value)
            if category is None:
                continue
            report.issues.append(
                LiteralIssue(
                    path=path,
                    value=value,
                    category=category,
                    message=self._message(category),
                    expected_source=self._expected_source(category),
                    fixed=False,
                )
            )

        unique: dict[tuple[str, str], LiteralIssue] = {}
        for issue in report.issues:
            unique[(issue.path, issue.category)] = issue
        report.issues = list(unique.values())

        if self.fail_on_unresolved and not report.ok:
            raise GammaStarLiteralValidationError(report)

        return report

    def _validate_dependency_roots(
        self,
        parameters: Mapping[str, Any],
        report: LiteralValidationReport,
    ) -> None:
        """Validate graph connectivity without imposing root-family policy.

        Relationship validity and literal policy are intentionally separate:

        - this pass verifies that every relationship input points to an
          existing gammaSTAR parameter and that the graph is traversable;
        - the later literal-classification pass decides whether a terminal
          executable literal is permitted.

        Event-to-event, block-to-block, and event-to-block dependencies are
        therefore valid graph edges. A prohibited derived literal at the end of
        such a chain is reported once at its own path instead of causing every
        downstream relationship to be reported as an invalid root.
        """

        cache: dict[str, bool] = {}
        visiting: set[str] = set()

        def valid(path: str) -> bool:
            if path in cache:
                return cache[path]

            if path in visiting:
                # Cycles are legal in the structural graph only when gammaSTAR
                # can evaluate them through its runtime ordering. Reference
                # existence is the responsibility of this pass; cycle policy
                # belongs to a dedicated graph validator.
                return True

            parameter = parameters.get(path)
            if not isinstance(parameter, Mapping):
                cache[path] = False
                return False

            inputs = parameter.get("inputs")
            if not isinstance(inputs, Mapping):
                cache[path] = False
                return False

            # A literal is a valid terminal node for graph-connectivity
            # purposes. Whether it is allowed at this path is checked later by
            # _classify().
            if not inputs:
                cache[path] = True
                return True

            visiting.add(path)
            ok = all(valid(str(source)) for source in inputs.values())
            visiting.remove(path)
            cache[path] = ok
            return ok

        for path, parameter in parameters.items():
            if not isinstance(parameter, Mapping):
                report.issues.append(
                    LiteralIssue(
                        path=path,
                        value=parameter,
                        category="invalid_dependency_root",
                        message=(
                            "Parameter is not represented as a gammaSTAR "
                            "relationship/literal mapping."
                        ),
                        expected_source="gammaSTAR parameter mapping",
                        fixed=False,
                    )
                )
                continue

            inputs = parameter.get("inputs")
            if not isinstance(inputs, Mapping):
                report.issues.append(
                    LiteralIssue(
                        path=path,
                        value=None,
                        category="invalid_dependency_root",
                        message=(
                            "Parameter does not provide a valid inputs mapping."
                        ),
                        expected_source="gammaSTAR inputs mapping",
                        fixed=False,
                    )
                )
                continue

            missing = [
                str(source)
                for source in inputs.values()
                if str(source) not in parameters
            ]
            if missing:
                report.issues.append(
                    LiteralIssue(
                        path=path,
                        value=None,
                        category="invalid_dependency_root",
                        message=(
                            "Relationship references missing gammaSTAR "
                            "parameter path(s): "
                            + ", ".join(sorted(set(missing)))
                        ),
                        expected_source="existing gammaSTAR parameter paths",
                        fixed=False,
                    )
                )
                continue

            if valid(path):
                continue

            report.issues.append(
                LiteralIssue(
                    path=path,
                    value=None,
                    category="invalid_dependency_root",
                    message=(
                        "Relationship graph contains an invalid or "
                        "untraversable dependency."
                    ),
                    expected_source="existing gammaSTAR parameter graph",
                    fixed=False,
                )
            )

    def _is_structural_literal_path(
        self,
        path: str,
        value: Any,
    ) -> bool:
        if path.endswith(self._STRUCTURAL_SUFFIXES):
            return True
        if isinstance(value, (str, bool)) or value is None:
            return True
        if isinstance(value, (list, tuple, dict)):
            return not path.endswith((".window_data", ".samples"))
        return self._allowed_zero_tstart(path, value)

    def _validate_adc_bandwidth_control(
        self,
        parameters: Mapping[str, Any],
        report: LiteralValidationReport,
    ) -> None:
        """Check that ADC sample time is bandwidth/raster controlled."""

        bandwidth_path = "root.prot.receiver_bandwidth"
        raster_path = "root.sys.raster_time_adc"

        if bandwidth_path not in parameters:
            return

        for path, parameter in parameters.items():
            if not path.endswith(".adc.sample_time"):
                continue
            if not isinstance(parameter, Mapping):
                continue

            inputs = parameter.get("inputs")
            if not isinstance(inputs, Mapping):
                continue

            sources = {str(source) for source in inputs.values()}
            missing = {
                bandwidth_path,
                raster_path,
            } - sources

            if not missing:
                continue

            report.issues.append(
                LiteralIssue(
                    path=path,
                    value=None,
                    category="adc_bandwidth_not_constrained",
                    message=(
                        "ADC sample time must be derived from receiver "
                        "bandwidth and the system ADC raster."
                    ),
                    expected_source=(
                        "root.prot.receiver_bandwidth and "
                        "root.sys.raster_time_adc"
                    ),
                    fixed=False,
                )
            )

    def _validate_adc_header_offcenter(
        self,
        parameters: dict[str, Any],
        report: LiteralValidationReport,
    ) -> None:
        """Ensure gammaSTAR ADC Header objects expose `offcenter`.

        gammaSTAR's ADC Header blueprint expects an `offcenter` parameter.
        SeqStar/gammaSTAR writer geometry already exports the same semantic
        quantity as `.position`. This validator makes the schema contract
        explicit and safely repairs older generated documents by aliasing
        `.offcenter` to `.position` when the latter exists.

        This is generic header-schema repair, not GRE-specific logic.
        """

        for position_path, position_param in list(parameters.items()):
            if not str(position_path).endswith(".header.position"):
                continue
            if not isinstance(position_param, Mapping):
                continue

            header_path = str(position_path)[: -len(".position")]
            offcenter_path = f"{header_path}.offcenter"
            if offcenter_path in parameters:
                continue

            parameters[offcenter_path] = {
                "inputs": {"position": str(position_path)},
                "script": "return position",
            }
            report.fixed_paths.append(offcenter_path)
            report.issues.append(
                LiteralIssue(
                    path=offcenter_path,
                    value=None,
                    category="adc_header_offcenter_alias",
                    message=(
                        "Added gammaSTAR ADC Header offcenter alias from the "
                        "existing header position relationship."
                    ),
                    expected_source=str(position_path),
                    fixed=True,
                )
            )

    def _validate_variation_backend_targets(
        self,
        parameters: dict[str, Any],
        report: LiteralValidationReport,
    ) -> None:
        """Validate/fix semantic ``node.vary`` aliases for gammaSTAR.

        The public API exposes PyPulseq-like semantic properties such as
        ``phase_offset`` and ``freq_offset``.  gammaSTAR RF/ADC executable
        representations consume ``.phase`` and ``.frequency``.  Therefore a
        relationship exported only to ``.phase_offset`` is not sufficient: the
        sibling consumed field must carry the same relationship.

        This pass is deliberately generic and alias based.  It does not know
        GRE, EPI, RF spoiling, or phase encoding.  It catches and safely fixes
        the central failure mode where a variation relationship exists on a
        metadata alias but not on the backend-consumed field.
        """

        alias_suffixes = {
            ".phase_offset": ".phase",
            ".rf_phase": ".phase",
            ".adc_phase": ".phase",
            ".rf_phase_offset": ".phase",
            ".adc_phase_offset": ".phase",
            ".freq_offset": ".frequency",
            ".frequency_offset": ".frequency",
            ".rf_frequency": ".frequency",
            ".adc_frequency": ".frequency",
            ".rf_frequency_offset": ".frequency",
            ".adc_frequency_offset": ".frequency",
        }

        for alias_path, alias_parameter in list(parameters.items()):
            if not isinstance(alias_parameter, Mapping):
                continue
            inputs = alias_parameter.get("inputs")
            if not isinstance(inputs, Mapping) or not inputs:
                continue

            target_path = None
            for alias_suffix, consumed_suffix in alias_suffixes.items():
                if alias_path.endswith(alias_suffix):
                    target_path = alias_path[: -len(alias_suffix)] + consumed_suffix
                    break
            if target_path is None or target_path not in parameters:
                continue

            target_parameter = parameters.get(target_path)
            if target_parameter == alias_parameter:
                continue

            if not isinstance(target_parameter, Mapping):
                continue

            target_inputs = target_parameter.get("inputs")
            # If the consumed target is already a live relationship, do not
            # overwrite it.  Record nothing; the backend target is covered.
            if isinstance(target_inputs, Mapping) and target_inputs:
                continue

            old_value = (
                self._literal_value(target_parameter)
                if self._is_literal_parameter(target_parameter)
                else None
            )
            parameters[target_path] = dict(alias_parameter)
            report.fixed_paths.append(target_path)
            report.issues.append(
                LiteralIssue(
                    path=target_path,
                    value=old_value,
                    category="variation_backend_alias",
                    message=(
                        "Copied a live variation relationship from a semantic "
                        "alias onto the gammaSTAR backend-consumed field."
                    ),
                    expected_source=alias_path,
                    fixed=True,
                )
            )

        warnings_param = parameters.get(
            "root.info.seqstar_variation_export_warning_count"
        )
        if self._is_literal_parameter(warnings_param):
            warning_count = self._literal_value(warnings_param)
            try:
                warning_count = int(warning_count)
            except Exception:
                warning_count = 0
            if warning_count:
                report.issues.append(
                    LiteralIssue(
                        path="root.info.seqstar_variation_export_warnings",
                        value=warning_count,
                        category="variation_export_contract_failed",
                        message=(
                            "The gammaSTAR writer reported unresolved "
                            "node.vary export warnings."
                        ),
                        expected_source="root.info.seqstar_variation_exports",
                        fixed=False,
                    )
                )

    def _apply_universal_repairs(
        self,
        parameters: dict[str, Any],
        report: LiteralValidationReport,
    ) -> None:
        """Repair identities that are true for every supported sequence."""

        for path, parameter in list(parameters.items()):
            if not self._is_literal_parameter(parameter):
                continue

            if path.endswith(".tend"):
                prefix = path[: -len(".tend")]
                tstart = f"{prefix}.tstart"
                duration = f"{prefix}.duration"
                if tstart in parameters and duration in parameters:
                    self._replace(
                        parameters,
                        report,
                        path,
                        parameter,
                        {
                            "inputs": {
                                "tstart": tstart,
                                "duration": duration,
                            },
                            "script": "return tstart + duration",
                        },
                        "timing_identity",
                        "tend = tstart + duration",
                    )
                    continue

            if path.endswith(".tcenter"):
                prefix = path[: -len(".tcenter")]
                tstart = f"{prefix}.tstart"
                duration = f"{prefix}.duration"
                if tstart in parameters and duration in parameters:
                    self._replace(
                        parameters,
                        report,
                        path,
                        parameter,
                        {
                            "inputs": {
                                "tstart": tstart,
                                "duration": duration,
                            },
                            "script": "return tstart + 0.5 * duration",
                        },
                        "timing_identity",
                        "tcenter = tstart + 0.5 * duration",
                    )
                    continue

            if path.endswith(".adc.duration"):
                prefix = path[: -len(".duration")]
                samples = f"{prefix}.number_of_samples"
                sample_time = f"{prefix}.sample_time"
                if samples in parameters and sample_time in parameters:
                    self._replace(
                        parameters,
                        report,
                        path,
                        parameter,
                        {
                            "inputs": {
                                "samples": samples,
                                "sample_time": sample_time,
                            },
                            "script": "return samples * sample_time",
                        },
                        "adc_identity",
                        "duration = number_of_samples * sample_time",
                    )

    @staticmethod
    def _replace(
        parameters: dict[str, Any],
        report: LiteralValidationReport,
        path: str,
        old_parameter: Mapping[str, Any],
        replacement: Mapping[str, Any],
        category: str,
        relationship: str,
    ) -> None:
        old_value = GammaStarLiteralValidator._literal_value(old_parameter)
        parameters[path] = dict(replacement)
        report.fixed_paths.append(path)
        report.issues.append(
            LiteralIssue(
                path=path,
                value=old_value,
                category=category,
                message=f"Replaced using universal relationship: {relationship}.",
                expected_source=relationship,
                fixed=True,
            )
        )

    def _classify(self, path: str, value: Any) -> str | None:
        if path.startswith(self._DECLARATION_PREFIXES):
            return None

        if self.allow_structural_literals:
            if path.endswith(self._STRUCTURAL_SUFFIXES):
                return None
            if isinstance(value, (str, bool)) or value is None:
                return None
            if isinstance(value, (list, tuple, dict)):
                if not path.endswith((".window_data", ".samples")):
                    return None
            if self._allowed_zero_tstart(path, value):
                return None

        if path.endswith(".seqstar_event_delay"):
            return "intrinsic_event_delay_literal"

        if path.endswith(".seqstar_kernel_tstart"):
            return "resolved_timing_snapshot"

        if ".seqstar_blocks." in path and path.endswith(".duration"):
            return "block_duration_literal"

        if path.endswith(".window_data"):
            return "materialized_event_table"

        if path.endswith(self._DERIVED_SUFFIXES):
            return "derived_event_property"

        if path.endswith(".tstart"):
            return "scheduled_timing_literal"

        if self._is_number(value) and self._is_executable_path(path):
            return "unclassified_executable_numeric_literal"

        return None

    def _allowed_zero_tstart(self, path: str, value: Any) -> bool:
        if not self._is_number(value) or float(value) != 0.0:
            return False

        if any(
            pattern.search(path)
            for pattern in self._ALLOWED_LOCAL_ZERO_TSTART
        ):
            return True

        # The first logical block defines the kernel-local time origin.
        # Downstream block starts must remain relationships.
        if (
            ".seqstar_blocks." in path
            and path.endswith(".tstart")
        ):
            block_token = path.rsplit(".seqstar_blocks.", 1)[1]
            block_token = block_token.rsplit(".tstart", 1)[0]
            return block_token.endswith(
                (
                    "excitation",
                    "prep",
                    "preparation",
                    "initialization",
                    "start",
                )
            )

        return False

    @staticmethod
    def _is_executable_path(path: str) -> bool:
        return (
            path.startswith("root.average.")
            or ".kernel." in path
            or ".seqstar_loop." in path
        )

    @staticmethod
    def _message(category: str) -> str:
        return {
            "intrinsic_event_delay_literal": (
                "Intrinsic RF/ADC delay must reference the corresponding "
                "System dead-time declaration."
            ),
            "resolved_timing_snapshot": (
                "Resolved Python timing must not appear in the parameterized graph."
            ),
            "block_duration_literal": (
                "Block duration must come from symbolic event/block relationships."
            ),
            "materialized_event_table": (
                "Materialized event table must be assembled from live nodes."
            ),
            "derived_event_property": (
                "Derived event property must reference protocol/system/event nodes."
            ),
            "scheduled_timing_literal": (
                "Scheduled timing must be represented by an explicit relationship."
            ),
            "unclassified_executable_numeric_literal": (
                "Numeric literal occurs in the executable graph and is not structural."
            ),
            "invalid_dependency_root": (
                "Parameter does not resolve to protocol/system or an allowed "
                "structural root."
            ),
            "adc_bandwidth_not_constrained": (
                "ADC sample time is not derived from receiver bandwidth and "
                "the system ADC raster."
            ),
            "variation_backend_alias": (
                "Semantic node.vary alias did not drive the backend-consumed field."
            ),
            "variation_export_contract_failed": (
                "One or more node.vary declarations were not exported to live backend fields."
            ),
        }[category]

    @staticmethod
    def _expected_source(category: str) -> str:
        return {
            "intrinsic_event_delay_literal": (
                "root.sys.rf_dead_time or root.sys.adc_dead_time"
            ),
            "resolved_timing_snapshot": "remove snapshot",
            "block_duration_literal": "symbolic block/event duration expression",
            "materialized_event_table": "live event-property projection",
            "derived_event_property": "protocol/system/event expression",
            "scheduled_timing_literal": "block-after, anchor, or loop relationship",
            "unclassified_executable_numeric_literal": "registered symbolic relationship",
            "invalid_dependency_root": (
                "root.prot.*, root.sys.*, or structural sequence nodes"
            ),
            "adc_bandwidth_not_constrained": (
                "root.prot.receiver_bandwidth and "
                "root.sys.raster_time_adc"
            ),
            "variation_backend_alias": "gammaSTAR backend-consumed phase/frequency field",
            "variation_export_contract_failed": "root.info.seqstar_variation_exports",
        }[category]

    @staticmethod
    def _is_literal_parameter(parameter: Any) -> bool:
        if not isinstance(parameter, Mapping):
            return False
        inputs = parameter.get("inputs")
        script = parameter.get("script")
        return (
            isinstance(inputs, Mapping)
            and not inputs
            and isinstance(script, str)
            and script.strip().startswith("return ")
        )

    @staticmethod
    def _literal_value(parameter: Mapping[str, Any]) -> Any:
        body = str(parameter.get("script", "")).strip()
        body = body[len("return "):].strip()
        if body == "nil":
            return None
        if body == "true":
            return True
        if body == "false":
            return False
        try:
            return ast.literal_eval(body)
        except Exception:
            try:
                return float(body)
            except Exception:
                return body

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
