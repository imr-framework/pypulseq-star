"""Generic gammaSTAR document builder for pypulseq_star.

This file deliberately does not replace the existing low-level converters.
It is a document-level path that walks an enriched SeqStarSequence and emits a
gammaSTAR-compatible JSON document.

Current support:
    - root / Protocol / System / Info / helper / expo / tests sections
    - one repeated average loop
    - one kernel per document
    - RF events inside the kernel
    - single-window ADC events inside the kernel
    - multi-window ADC trains represented as a gammaSTAR Loop over windows
    - delay-aware RF timing from enriched SeqStar metadata

Important design decision:
    For gammaSTAR export, ADC trains should remain semantically rich.

    However, a literal Python/Lua "windows" list under one ADC object is not
    enough for gammaSTAR's plot engine to show multiple ADC gates. Therefore,
    multi-window ADC trains are represented as:

        root.average.kernel.readout.window.adc

    where "window" is a Loop. The ADC object is evaluated once for each
    window.counter.

    This is different from Pulseq .seq writing, where ADC trains are lowered
    into individual standard ADC events/blocks at write time.

Second-pass TODO:
    When sequence-level composition matures, this writer should be expanded to
    support:
        - prep nodes before the kernel
        - calibration/reference branches
        - RF/ADC/gradient objects outside the kernel
        - simultaneous combined events, e.g., RF + slice-select gradient
        - multi-kernel documents
        - explicit readout/prep/calibration hierarchy inspired by gammaSTAR
          EPI, GRASE, PROPELLER-GRASE, radial, MP-RAGE, and SVS examples.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from time import perf_counter
from typing import Any

from pypulseq_star.calc_duration import calc_duration

# gammaSTAR generic document writer patch level: v37-adc-header-offcenter-alias


RF_RECT_BLUEPRINT = "184dfa36-f86e-425f-a653-40d47a4a99b2"
RF_PULSE_BLUEPRINT = "RFPulse"
KERNEL_BLUEPRINT = "a6bcd65a-b25b-4fe4-894f-c7fdf4fc8beb"
ROOT_BLUEPRINT = "004ebbab-558f-4a05-8cce-e907f52a4f37"
HELPER_BLUEPRINT = "3ec659dd-2b06-44a8-9af5-30a8496cfc57"
EXPO_BLUEPRINT = "c54aab35-2bd3-41e3-b892-35a072404b69"

# This UUID appears in the shared Demo_FID gammaSTAR JSON as the container
# holding an ADC leaf and Atomic node.
ADC_CONTAINER_BLUEPRINT = "221adc41-b6f7-4085-a0a1-2699db044fd1"
ADC_BLUEPRINT = "ADC"
ADC_HEADER_BLUEPRINT = "ADC header"
SINGLE_READOUT_BLUEPRINT = "ee153c9c-c247-48d1-ba59-d4e1bcef194f"

# Reference gammaSTAR gradient blueprints observed in EPI/GRASE examples.
# The child leaf that feeds the plot engine is always GradPulse.
TRAPEZOID_GRADIENT_BLUEPRINT = "fa6e1a33-0e34-4b83-a56f-95f5e878230b"
ARBITRARY_GRADIENT_BLUEPRINT = "57cf5ac4-75ff-4ec7-8c3c-65fcda338f6f"
GRAD_PULSE_BLUEPRINT = "GradPulse"


def _gammastar_debug_enabled() -> bool:
    return str(os.environ.get("PYPULSEQ_STAR_GAMMASTAR_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "debug",
    }


def _gammastar_debug(message: str) -> None:
    if _gammastar_debug_enabled():
        print(f"[gammastar.writer] {message}")


def _debug_value_summary(value: Any) -> str:
    try:
        canonical = value.to_canonical() if hasattr(value, "to_canonical") else None
    except Exception as exc:
        canonical = f"<canonical failed: {type(exc).__name__}>"
    try:
        deps = sorted(getattr(value, "dependencies", []) or [])
    except Exception:
        deps = []
    return (
        f"type={type(value).__name__} "
        f"contains_expr={_contains_protocol_expression(value)} "
        f"canonical={canonical!r} deps={deps!r} value={value!r}"
    )



class GenericGammaStarDocumentBuilder:
    """Build a gammaSTAR JSON document from a SeqStar sequence."""

    def __init__(
        self,
        sequence: Any,
        *,
        symbolic_sequence: Any | None = None,
        realization: Any | None = None,
        auto_resolve_relationships: bool = True,
        profile: bool = False,
    ) -> None:
        self.sequence = sequence
        # Preserve caller-supplied symbolic sequence even if the sequence
        # object is falsy through a node/children/container model.
        self.symbolic_sequence = sequence if symbolic_sequence is None else symbolic_sequence
        self.realization = realization
        self.system = sequence.system
        self.auto_resolve_relationships = auto_resolve_relationships
        self.profile = bool(profile)
        self._event_container_paths: dict[int, str] = {}
        self._event_container_events: dict[int, Any] = {}
        self._event_container_leaves: dict[int, str] = {}
        self._block_export_paths: dict[int, str] = {}
        self._block_export_records: list[dict[str, Any]] = []
        self._live_protocol_values: dict[str, Any] = {}
        self._active_loop_path = "root.average"
        self._variation_export_records: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        """Return a gammaSTAR JSON document.

        Export policy
        -------------
        A single SeqStar block is exported exactly as before. This preserves the
        FID/simple-event path that has already been working.

        Multiple sequential SeqStar/PyPulseq-style blocks are handled by a
        generic compacting pass before falling back to flat export:

        1. If the block list contains a repeated motif, export only one
           representative motif as the kernel and use the existing outer Loop
           length to repeat it.
        2. If no repeated motif is found, flatten the explicit sequential
           blocks into one kernel as a debug/fallback representation.

        The motif detector is intentionally sequence-agnostic. It does not know
        GRE, FLASH, EPI, FID, RARE, or spectroscopy. It only looks for repeated
        block role/signature patterns in the concrete SeqStar timeline.
        """

        total_start = perf_counter()
        resolve_start = perf_counter()
        if self.auto_resolve_relationships:
            _resolve_sequence_relationships_for_export(self.sequence)
        if self.profile:
            print(f"[profile] gammaSTAR relationship resolve  {perf_counter() - resolve_start:9.3f} s")

        blocks = list(_iter_blocks(self.sequence))

        if not blocks:
            raise ValueError("Cannot export an empty SeqStar sequence to gammaSTAR JSON.")

        self._live_protocol_values = _canonical_protocol_values(
            _merge_protocol_values_for_export(
                export_sequence=self.sequence,
                symbolic_sequence=self.symbolic_sequence,
            )
        )
        if _gammastar_debug_enabled():
            _gammastar_debug(
                "builder sequence ids: "
                f"export={id(self.sequence)} symbolic={id(self.symbolic_sequence)} "
                f"same={self.sequence is self.symbolic_sequence}"
            )
            for _key in ("phase_encode_start", "phase_encode_step", "n_y", "fov"):
                if _key in self._live_protocol_values:
                    _gammastar_debug(
                        f"live_protocol_values[{_key!r}] "
                        + _debug_value_summary(self._live_protocol_values[_key])
                    )
        compact_export = bool(
            self._live_protocol_values.get("compact_gamma_export", False)
            or self._live_protocol_values.get("seqstar_compact_export", False)
        )

        export_mode = "single_block"
        motif_info: dict[str, Any] | None = None
        export_blocks: list[Any] = []
        explicit_repeat_info: dict[str, Any] | None = None

        if len(blocks) == 1:
            block = blocks[0]
            export_blocks = [block]
            events = list(_iter_events(block))

            repetitions = _get_int(
                block,
                keys=("repetitions", "average", "averages", "n_avg", "num_averages"),
                default=_get_int(
                    self.sequence,
                    keys=("repetitions", "average", "averages", "n_avg", "num_averages"),
                    default=1,
                ),
            )

            tr = _get_repetition_time_for_block(
                block,
                sequence=self.sequence,
            )

            block_duration = max(calc_duration(block), _events_extent_duration(events))

            if tr is None:
                tr = block_duration
            elif tr < block_duration:
                tr = block_duration

        else:
            explicit_repeat_info = _explicit_repeated_node_info(
                self.sequence,
                blocks,
            )
            detected_motif_info = _detect_repeated_block_motif(
                self.sequence,
                blocks,
            )

            if detected_motif_info is not None:
                # The concrete timeline already contains repeated motifs.
                # Compact it to one representative motif. If the developer also
                # declared repetition through seq.set_node(...), reconcile the
                # two sources rather than multiplying them.
                motif_info = detected_motif_info
                motif_blocks = list(motif_info["blocks"])
                export_blocks = motif_blocks

                events, block_duration = _flatten_repeated_motif_to_kernel_events(
                    self.sequence,
                    blocks,
                    motif_length=int(motif_info["motif_length"]),
                    repetitions=int(motif_info["repetitions"]),
                )

                materialized_repetitions = int(motif_info["repetitions"])
                repetitions = materialized_repetitions

                if explicit_repeat_info is not None:
                    repeat_mode = explicit_repeat_info["repeat_mode"]
                    if repeat_mode == "loop":
                        raise ValueError(
                            "repeat_mode='loop' requires one representative timeline motif, "
                            "but repeated motifs are already present. Use repeat_mode='expanded'."
                        )
                    declared_repetitions = int(
                        explicit_repeat_info["repetitions"]
                    )

                    if declared_repetitions != materialized_repetitions:
                        raise ValueError(
                            "Logical-node repeat metadata disagrees with the "
                            "materialized repeated timeline. "
                            f"node={explicit_repeat_info['node']!r}, "
                            f"declared_repeat_count={declared_repetitions}, "
                            f"materialized_repetitions={materialized_repetitions}."
                        )

                    export_mode = "explicit_materialized_repeat"
                else:
                    export_mode = "compact_repeated_motif"

                tr = _get_float(
                    self.sequence,
                    keys=("repetition_time", "TR", "tr"),
                    default=None,
                )
                if tr is None:
                    tr = float(
                        motif_info.get(
                            "motif_duration",
                            block_duration,
                        )
                    )
                if tr < block_duration:
                    tr = block_duration

            elif explicit_repeat_info is not None:
                if explicit_repeat_info["repeat_mode"] != "loop":
                    # Expanded timelines need not have identical block signatures;
                    # preserve the concrete execution exactly once.
                    export_blocks = blocks
                    events, block_duration = _flatten_blocks_to_kernel_events(
                        self.sequence, blocks
                    )
                    repetitions = 1
                    export_mode = "explicit_expanded_timeline"
                    tr = block_duration
                else:
                    # The timeline contains one representative logical motif and
                    # seq.set_node(...) declares how many times it repeats.
                    export_blocks = list(explicit_repeat_info["blocks"])
                    events, block_duration = _flatten_blocks_to_kernel_events(
                        self.sequence,
                        export_blocks,
                    )
                    repetitions = int(explicit_repeat_info["repetitions"])
                    export_mode = "explicit_node_repeat"
    
                    tr = _get_float(
                        self.sequence,
                        keys=("repetition_time", "TR", "tr"),
                        default=None,
                    )
                    if tr is None:
                        repeat_period = explicit_repeat_info.get(
                            "repeat_period"
                        )
                        tr = (
                            float(repeat_period)
                            if repeat_period is not None
                            else block_duration
                        )
                    if tr < block_duration:
                        tr = block_duration
    
                    motif_info = {
                        "blocks": export_blocks,
                        "motif_length": len(export_blocks),
                        "repetitions": repetitions,
                        "motif_duration": block_duration,
                        "signature": [
                            _block_motif_signature(block)
                            for block in export_blocks
                        ],
                    }

            else:
                # No explicit or materialized repetition: preserve the concrete
                # sequential timeline as one kernel.
                export_blocks = blocks
                events, block_duration = _flatten_blocks_to_kernel_events(
                    self.sequence,
                    blocks,
                )
                repetitions = 1
                export_mode = "flat_multiblock_fallback"

                tr = _get_float(
                    self.sequence,
                    keys=("repetition_time", "TR", "tr"),
                    default=None,
                )
                if tr is None or tr < block_duration:
                    tr = block_duration

        rf_events = [event for event in events if _is_rf_event(event)]
        adc_events = [event for event in events if _is_adc_event(event)]
        gradient_events = [event for event in events if _is_gradient_event(event)]

        if not rf_events and not adc_events and not gradient_events:
            raise NotImplementedError(
                "The generic gammaSTAR writer currently supports RF, ADC, and/or "
                "gradient kernels."
            )

        self._live_protocol_values.setdefault("TR", tr)
        self._live_protocol_values.setdefault("repetition_time", tr)

        if export_mode == "compact_repeated_motif":
            # Heuristically compacted motifs use a neutral loop name because
            # their semantic counter is unknown.
            loop_path = "root.seqstar_loop"
        elif export_mode in {
            "explicit_node_repeat",
            "explicit_materialized_repeat",
        }:
            # The gammaSTAR loop path describes the repeated logical node, not
            # the protocol parameter that controls its length.  For example,
            # node="segment" with repeat_count="seqstar_segments" must export
            # root.segment, while root.segment.length remains driven by the
            # seqstar_segments protocol relationship.
            loop_token = str(
                explicit_repeat_info.get("node_loop_token", "seqstar_loop")
            )
            loop_path = f"root.{_safe_path_token(loop_token)}"
        else:
            loop_path = "root.average"
        self._active_loop_path = loop_path

        # Repeated-motif compaction creates timed event proxies before the
        # final gammaSTAR loop path is known. Retarget any stale placeholder
        # counter references to the actual active loop, e.g. root.n_y.counter.
        _retarget_export_event_loop_counter_paths(
            events,
            loop_counter_path=f"{self._active_loop_path}.counter",
        )

        kernel_path = f"{loop_path}.kernel"

        params: dict[str, Any] = {}

        sequence_elements: dict[str, str] = {
            "root": ROOT_BLUEPRINT,
            loop_path: "Loop",
            kernel_path: KERNEL_BLUEPRINT,
        }

        repeat_count_ref = None
        if explicit_repeat_info is not None:
            repeat_count_ref = explicit_repeat_info.get("record", {}).get(
                "repeat_count"
            )

        self._add_root_loop_and_kernel(
            params=params,
            loop_path=loop_path,
            kernel_path=kernel_path,
            repetitions=repetitions,
            repeat_count_ref=repeat_count_ref,
            tr=tr,
            block_duration=block_duration,
        )

        # Export diagnostics. These are harmless gammaSTAR parameters and make
        # it easy to verify that a repeated sequence was compacted instead of
        # fully unrolled.
        params["root.info.seqstar_export_mode"] = _literal(export_mode)
        params["root.info.seqstar_loop_path"] = _literal(loop_path)
        params["root.info.seqstar_source_blocks"] = _literal(len(blocks))
        params["root.info.seqstar_exported_kernel_blocks"] = _literal(
            len(motif_info["blocks"]) if motif_info is not None else len(blocks)
        )
        if motif_info is not None:
            params["root.info.seqstar_motif_length_blocks"] = _literal(
                int(motif_info["motif_length"])
            )
            params["root.info.seqstar_motif_repetitions"] = _expr(
                inputs={"count": f"{loop_path}.length"},
                script="return count",
            )
            params["root.info.seqstar_default_motif_repetitions"] = _literal(
                int(motif_info["repetitions"])
            )
            if not compact_export:
                params["root.info.seqstar_motif_signature"] = _literal(
                    list(motif_info.get("signature", []))
                )

        rf_leaf_paths: list[str] = []
        adc_leaf_paths: list[str] = []
        gradient_leaf_paths: list[str] = []
        self._event_container_paths = {}
        self._event_container_events = {}
        self._event_container_leaves = {}
        self._block_export_paths = {}
        self._block_export_records = []

        for index, rf_event in enumerate(rf_events):
            rf_name = "rf" if index == 0 else f"rf{index + 1}"
            rf_leaf_path = self._add_rf_event(
                params=params,
                sequence_elements=sequence_elements,
                loop_path=loop_path,
                kernel_path=kernel_path,
                rf_name=rf_name,
                rf_event=rf_event,
                event_index=index,
            )
            rf_leaf_paths.append(rf_leaf_path)
            self._event_container_paths[id(rf_event)] = rf_leaf_path.rsplit(".rf", 1)[0]
            self._event_container_leaves[id(rf_event)] = rf_leaf_path
            self._event_container_events[id(rf_event)] = rf_event

        for index, adc_event in enumerate(adc_events):
            adc_leaf_path = self._add_adc_event(
                params=params,
                sequence_elements=sequence_elements,
                loop_path=loop_path,
                kernel_path=kernel_path,
                adc_event=adc_event,
                event_index=index,
            )
            adc_leaf_paths.append(adc_leaf_path)
            self._event_container_paths[id(adc_event)] = adc_leaf_path.rsplit(".adc", 1)[0]
            self._event_container_leaves[id(adc_event)] = adc_leaf_path
            self._event_container_events[id(adc_event)] = adc_event

        for index, gradient_event in enumerate(gradient_events):
            gradient_leaf_path = self._add_gradient_event(
                params=params,
                sequence_elements=sequence_elements,
                loop_path=loop_path,
                kernel_path=kernel_path,
                gradient_event=gradient_event,
                event_index=index,
            )
            gradient_leaf_paths.append(gradient_leaf_path)
            self._event_container_paths[id(gradient_event)] = gradient_leaf_path.rsplit(".grad", 1)[0]
            self._event_container_leaves[id(gradient_event)] = gradient_leaf_path
            self._event_container_events[id(gradient_event)] = gradient_event

        self._add_seqstar_block_timing_dependencies(
            params=params,
            sequence_elements=sequence_elements,
            kernel_path=kernel_path,
            export_blocks=export_blocks,
            exported_events=events,
        )

        # Apply declarative loop-dependent event variations such as
        # kernel.vary(vary(gy_pre, "area", ...)). This is intentionally generic:
        # the variation metadata is attached by Sequence/NodeHandle and the
        # writer only scales the exported event property by the active loop
        # counter. It handles the common single-loop GRE phase-encode case.
        self._apply_single_loop_variations_and_bindings(
            params=params,
            loop_path=loop_path,
        )

        # Preserve protocol dependencies embedded directly in trapezoid
        # constructor inputs, e.g. spoiler area = 2*n_x/fov.  These gradients
        # are not loop variations, so without this pass gammaSTAR receives only
        # the resolved default waveform and interactive protocol edits cannot
        # update it.
        self._apply_symbolic_trapezoid_constructor_bindings(
            params=params,
        )
        self._add_variation_export_contract_tests(params=params)

        sequence_elements.update(
            {
                "root.expo": EXPO_BLUEPRINT,
                "root.helper": HELPER_BLUEPRINT,
                "root.info": "Info",
                "root.prot": "Protocol",
                "root.sys": "System",
                "root.tests": "Tests",
            }
        )

        if compact_export:
            # A compact developer/demo document keeps the editable sequence,
            # protocol, system, and plot representations but omits the large
            # generated validation-test table. Scanner-side validation remains
            # available through SeqStar and Pulseq timing checks.
            params["root.info.seqstar_compact_export"] = _literal(True)
            params["root.tests.all_tests"] = _expr(
                inputs={},
                script="return {}",
            )
        else:
            self._add_tests(
                params=params,
                rf_paths=rf_leaf_paths,
                adc_paths=adc_leaf_paths,
                gradient_paths=gradient_leaf_paths,
            )
        self._add_protocol(
            params=params,
            rf_events=rf_events,
            adc_events=adc_events,
            gradient_events=gradient_events,
            tr=tr,
            repetitions=(
                1
                if export_mode == "compact_repeated_motif"
                else repetitions
            ),
        )
        self._apply_computed_protocol_tables(params=params)

        # Protocol expressions are emitted by _add_protocol().  Rebind the
        # outer loop after that step so a derived repeat count is copied as a
        # direct expression rather than routed through a second derived leaf.
        # Some gammaSTAR editor versions do not reliably invalidate a
        # dependency chain of the form n_y -> segments -> loop.length after an
        # interactive protocol edit.  Copying the derived expression here
        # produces the equivalent direct dependency
        # n_y, ETL -> loop.length without introducing sequence semantics.
        self._rebind_outer_loop_length_to_protocol_expression(
            params=params,
            loop_path=loop_path,
            repeat_count_ref=repeat_count_ref,
            repetitions=repetitions,
        )
        self._flatten_protocol_relationship_target(
            params=params,
            target_path=f"{loop_path}.length",
        )
        if export_mode in {
            "compact_repeated_motif",
            "explicit_node_repeat",
            "explicit_materialized_repeat",
        }:
            loop_length_source = self._loop_length_protocol_source(
                loop_path=loop_path,
            )
            if loop_length_source is not None:
                params["root.prot.seqstar_loop_length"] = _expr(
                    inputs={"count": f"root.prot.{loop_length_source}"},
                    script="return count",
                )
            else:
                params["root.prot.seqstar_loop_length"] = _literal(
                    repetitions
                )
        self._add_system(params=params)
        self._add_info(
            params=params,
            rf_events=rf_events,
            adc_events=adc_events,
            gradient_events=gradient_events,
        )
        self._add_runtime_defaults(params=params)

        # _add_info writes generic info defaults. Re-apply export diagnostics so
        # they are always present in the final document.
        params["root.info.seqstar_export_mode"] = _literal(export_mode)
        params["root.info.seqstar_source_blocks"] = _literal(len(blocks))
        params["root.info.seqstar_exported_kernel_blocks"] = _literal(
            len(motif_info["blocks"]) if motif_info is not None else len(blocks)
        )
        if motif_info is not None:
            params["root.info.seqstar_motif_length_blocks"] = _literal(
                int(motif_info["motif_length"])
            )
            params["root.info.seqstar_motif_repetitions"] = _expr(
                inputs={"count": f"{loop_path}.length"},
                script="return count",
            )
            params["root.info.seqstar_default_motif_repetitions"] = _literal(
                int(motif_info["repetitions"])
            )
            if not compact_export:
                params["root.info.seqstar_motif_signature"] = _literal(
                    list(motif_info.get("signature", []))
                )

        # Reparent exported event branches beneath timing-neutral block groups
        # for the gammaSTAR workplace. This changes only document hierarchy;
        # event timing expressions remain in kernel coordinates.
        self._group_exported_events_by_block(
            params=params,
            sequence_elements=sequence_elements,
            kernel_path=kernel_path,
            exported_events=events,
        )

        if not compact_export:
            self._add_stale_literal_tstart_validator(
                params=params,
                kernel_path=kernel_path,
                exported_events=events,
            )

        self._apply_nested_loop_structure_and_bindings(
            params=params,
            sequence_elements=sequence_elements,
            kernel_path=kernel_path,
        )

        if not compact_export:
            self._rebuild_tests_after_final_hierarchy(params=params)

        if self.profile:
            print(f"[profile] gammaSTAR document build         {perf_counter() - total_start:9.3f} s")

        document = {
            "name": getattr(self.sequence, "name", "seqstar_sequence"),
            "parameters": params,
            "sequence_elements": _order_sequence_elements(sequence_elements),
        }

        document["seqstar_writer_context"] = {
            "used_realization": self.realization is not None,
            "symbolic_sequence_name": getattr(
                self.symbolic_sequence,
                "name",
                getattr(self.sequence, "name", "seqstar_sequence"),
            ),
            "numeric_sequence_name": getattr(
                self.sequence,
                "name",
                "seqstar_sequence",
            ),
        }

        diagnostics = getattr(self.realization, "diagnostics", None)
        if diagnostics:
            serialized = []
            for diagnostic in diagnostics:
                if hasattr(diagnostic, "to_dict"):
                    serialized.append(diagnostic.to_dict())
                elif isinstance(diagnostic, Mapping):
                    serialized.append(dict(diagnostic))
                else:
                    serialized.append({"message": str(diagnostic)})
            document["seqstar_writer_context"]["diagnostics"] = serialized

        return document

    def _apply_symbolic_trapezoid_constructor_bindings(
        self,
        *,
        params: dict[str, Any],
    ) -> None:
        """Export live protocol bindings from trapezoid constructors.

        A gradient can depend on protocol values without participating in a
        loop variation.  GRE spoilers are the canonical example::

            gx_spoil.area = 2 * n_x / fov
            gy_spoil.area = 2 * n_y / fov

        Resolution correctly rebuilds those gradients locally, but the generic
        gammaSTAR document previously serialized the rebuilt waveform as a
        literal.  Consequently, changing FOV on the gammaSTAR website updated
        the phase-encode prephaser (which is exported through ``node.vary``) but
        left the spoilers unchanged.

        This pass recognizes area-only symbolic trapezoid constructors and
        emits a protocol-driven shortest feasible trapezoid.  It also propagates
        the dynamic duration into the event container and source block so later
        blocks remain correctly positioned.  Explicitly varied gradients are
        skipped because ``_apply_single_loop_variations_and_bindings`` already
        owns their exported relationship.
        """

        affected_blocks: set[int] = set()

        for event_id, event in self._event_container_events.items():
            if not _is_gradient_event(event):
                continue

            leaf = self._event_container_leaves.get(event_id)
            container = self._event_container_paths.get(event_id)
            if not leaf or not container:
                continue

            metadata = getattr(event, "metadata", None)
            constructor = (
                metadata.get("symbolic_constructor")
                if isinstance(metadata, Mapping)
                else None
            )
            if not isinstance(constructor, Mapping):
                parameters = getattr(event, "parameters", None)
                constructor = (
                    parameters.get("_symbolic_constructor")
                    if isinstance(parameters, Mapping)
                    else None
                )
            if not isinstance(constructor, Mapping):
                continue
            if str(constructor.get("family", "")).lower() != "trapezoid":
                continue

            specs = constructor.get("specs")
            if not isinstance(specs, Mapping):
                continue

            area_spec = specs.get("area")
            # Only promote constructor expressions that are driven entirely by
            # protocol parameters. EventPropertyRef / anchor / block references
            # are valid local symbolic expressions, but they cannot be compiled
            # as root.prot relationships without first mapping them to exported
            # event paths. Those expressions must retain the already-resolved
            # literal waveform rather than aborting the whole gammaSTAR export.
            if not _contains_parameter_reference(area_spec):
                continue
            if _contains_non_protocol_reference(area_spec):
                continue

            # A node.vary(..., attribute="area") relationship has already
            # replaced this parameter.  Do not overwrite that stronger,
            # counter-dependent binding.
            if f"{leaf}.seqstar_variation_attribute" in params:
                continue

            # This implementation intentionally targets the common, generic
            # area-only shortest-trapezoid constructor.  Fixed-duration or
            # explicitly shaped trapezoids need a different feasibility model
            # and remain serialized exactly as resolved.
            if any(
                specs.get(name) is not None
                for name in (
                    "amplitude",
                    "duration",
                    "fall_time",
                    "flat_area",
                    "flat_time",
                    "rise_time",
                )
            ):
                continue

            area_inputs: dict[str, str] = {}
            source_to_name: dict[str, str] = {}
            try:
                # Inline derived protocol parameters into the backend-consumed
                # area target. gammaSTAR's interactive editor does not reliably
                # invalidate transitive chains such as
                # fov -> phase_encode_step -> spoiler.area -> spoiler.samples.
                # The same generic compiler is already used for node.vary()
                # targets and expands derived ParameterRef values to their
                # primitive editable protocol controls.
                area_body = self._variation_value_lua_expression(
                    area_spec,
                    inputs=area_inputs,
                    source_to_name=source_to_name,
                    target_path=f"{leaf}.area",
                )
            except TypeError as exc:
                # A generic writer enhancement must never make a previously
                # exportable sequence fail. Unsupported symbolic reference
                # forms are left as the resolved literal waveform and recorded
                # for diagnostics.
                warnings = getattr(
                    self,
                    "_symbolic_trapezoid_binding_warnings",
                    [],
                )
                warnings.append(
                    {
                        "event": str(getattr(event, "name", "<unnamed>")),
                        "leaf": leaf,
                        "reason": str(exc),
                    }
                )
                self._symbolic_trapezoid_binding_warnings = warnings
                continue
            params[f"{leaf}.area"] = _expr(
                inputs=area_inputs,
                script=f"return {area_body}",
            )

            # Reproduce calculate_shortest_params_for_area() in gammaSTAR
            # units.  Constructor area is in Hz/m*s (cycles/m); gammaSTAR
            # waveform amplitudes are T/m, hence the division by gamma.
            params[f"{leaf}.samples"] = _expr(
                inputs={
                    "area": f"{leaf}.area",
                    "gamma": "root.sys.gamma",
                    "grad_set": "root.gradient_settings",
                },
                script=(
                    "local raster = grad_set.raster_time or 1e-5\n"
                    "local max_grad = grad_set.max_grad_amp or 0.028\n"
                    "local max_slew = grad_set.max_grad_slew or 150.0\n"
                    "local area_si = (area or 0.0) / gamma\n"
                    "if math.abs(area_si) < 1e-20 then\n"
                    "  return {t={0.0, raster, 2*raster}, v={0.0, 0.0, 0.0}}\n"
                    "end\n"
                    "local function ceil_raster(value)\n"
                    "  return math.ceil(value / raster - 1e-12) * raster\n"
                    "end\n"
                    "local rise = ceil_raster(math.sqrt(math.abs(area_si) / max_slew))\n"
                    "if rise < raster then rise = raster end\n"
                    "local effective = rise\n"
                    "local amplitude = area_si / rise\n"
                    "if math.abs(amplitude) > max_grad + 1e-12 then\n"
                    "  effective = ceil_raster(math.abs(area_si) / max_grad)\n"
                    "  amplitude = area_si / effective\n"
                    "  rise = ceil_raster(math.abs(amplitude) / max_slew)\n"
                    "  if rise < raster then rise = raster end\n"
                    "end\n"
                    "local flat = effective - rise\n"
                    "if flat < 0 then flat = 0 end\n"
                    "local duration = 2*rise + flat\n"
                    "return {t={0.0, rise, rise+flat, duration}, \n"
                    "        v={0.0, amplitude, amplitude, 0.0}}"
                ),
            )
            params[f"{leaf}.duration"] = _expr(
                inputs={"samples": f"{leaf}.samples"},
                script=(
                    "if samples == nil or samples.t == nil or #samples.t == 0 then return 0.0 end\n"
                    "return samples.t[#samples.t]"
                ),
            )
            params[f"{leaf}.max_abs_amplitude"] = _expr(
                inputs={"samples": f"{leaf}.samples"},
                script=(
                    "local maxv = 0.0\n"
                    "if samples == nil or samples.v == nil then return maxv end\n"
                    "for i=1,#samples.v do\n"
                    "  local value = math.abs(samples.v[i])\n"
                    "  if value > maxv then maxv = value end\n"
                    "end\n"
                    "return maxv"
                ),
            )
            params[f"{container}.duration"] = _expr(
                inputs={"duration": f"{leaf}.duration"},
                script="return duration",
            )

            block_index = _event_source_block_index_for_export(event)
            if block_index is not None:
                affected_blocks.add(int(block_index))

        if not affected_blocks:
            return

        # Recompute affected source-block extents from the live event starts and
        # durations.  Grouping later rewrites the event paths while preserving
        # these dependencies.
        records_by_index = {
            int(record["index"]): record
            for record in self._block_export_records
        }
        for block_index in sorted(affected_blocks):
            record = records_by_index.get(block_index)
            if record is None:
                continue
            block_path = str(record["path"])
            inputs: dict[str, str] = {
                "block_tstart": f"{block_path}.tstart"
            }
            lines = ["local max_end = block_tstart"]
            event_number = 0
            for event_id, event in self._event_container_events.items():
                if _event_source_block_index_for_export(event) != block_index:
                    continue
                container = self._event_container_paths.get(event_id)
                if not container:
                    continue
                t_name = f"tstart_{event_number}"
                d_name = f"duration_{event_number}"
                inputs[t_name] = f"{container}.tstart"
                inputs[d_name] = f"{container}.duration"
                lines.append(
                    f"local event_end_{event_number} = {t_name} + {d_name}"
                )
                lines.append(
                    f"if event_end_{event_number} > max_end then max_end = event_end_{event_number} end"
                )
                event_number += 1
            lines.append("return max_end - block_tstart")
            params[f"{block_path}.duration"] = _expr(
                inputs=inputs,
                script="\n".join(lines),
            )

        # Preserve the requested TR when a protocol edit lengthens a late block
        # such as GRE spoiling.  The repetition-fill block starts after all
        # preceding blocks, so its duration is simply TR minus its live tstart.
        for record in self._block_export_records:
            role = str(
                _block_role(record["block"])
                or record.get("local_name")
                or ""
            ).lower()
            if role not in {
                "repetition_time_fill",
                "repetition_fill",
                "tr_fill",
                "repetition_delay",
            }:
                continue
            block_path = str(record["path"])
            params[f"{block_path}.duration"] = _expr(
                inputs={
                    "tr": "root.prot.TR",
                    "tstart": f"{block_path}.tstart",
                },
                script=(
                    "local remaining = tr - tstart\n"
                    "if remaining < 0 then return 0.0 end\n"
                    "return remaining"
                ),
            )

    def _add_variation_export_contract_tests(
        self,
        *,
        params: dict[str, Any],
    ) -> None:
        """Expose variation export coverage as gammaSTAR-visible tests.

        These tests are intentionally lightweight and sequence-agnostic.  They
        make failures in the central ``node.vary`` contract visible in the JSON
        without requiring a Python debugger:

        * every registered variation should map to at least one backend target;
        * RF/ADC phase-like variations should drive the backend-consumed
          ``.phase`` field, not only a metadata alias such as ``.phase_offset``;
        * each exported target must be present and relationship-backed.
        """

        records = list(getattr(self, "_variation_export_records", []))
        params["root.info.seqstar_variation_exports"] = _literal(records)
        params["root.info.seqstar_variation_export_count"] = _literal(
            len(records)
        )

        warnings: list[dict[str, Any]] = []
        for record in records:
            targets = [str(path) for path in record.get("targets", [])]
            attribute = str(record.get("attribute", "")).lower()
            event_type = str(record.get("event_type", "")).lower()

            if not targets:
                warnings.append(
                    {
                        "kind": "variation_has_no_export_target",
                        "event": record.get("event_name"),
                        "attribute": record.get("attribute"),
                        "leaf": record.get("leaf"),
                    }
                )
                continue

            if event_type in {"rf", "adc"} and attribute in {
                "phase",
                "phase_offset",
                "rf_phase",
                "adc_phase",
                "rf_phase_offset",
                "adc_phase_offset",
            }:
                phase_target = f"{record.get('leaf')}.phase"
                if phase_target not in targets:
                    warnings.append(
                        {
                            "kind": "phase_variation_not_backend_consumed",
                            "event": record.get("event_name"),
                            "attribute": record.get("attribute"),
                            "expected_target": phase_target,
                        }
                    )

            for target in targets:
                parameter = params.get(target)
                if not isinstance(parameter, Mapping):
                    warnings.append(
                        {
                            "kind": "variation_target_missing",
                            "event": record.get("event_name"),
                            "attribute": record.get("attribute"),
                            "target": target,
                        }
                    )
                    continue
                inputs = parameter.get("inputs")
                if not isinstance(inputs, Mapping) or not inputs:
                    warnings.append(
                        {
                            "kind": "variation_target_not_relationship",
                            "event": record.get("event_name"),
                            "attribute": record.get("attribute"),
                            "target": target,
                        }
                    )

        params["root.info.seqstar_variation_export_warnings"] = _literal(
            warnings
        )
        params["root.info.seqstar_variation_export_warning_count"] = _literal(
            len(warnings)
        )
        params["root.tests.seqstar_variations_exported"] = _expr(
            inputs={
                "warning_count": (
                    "root.info.seqstar_variation_export_warning_count"
                )
            },
            script="return warning_count == 0",
        )

    def _apply_computed_protocol_tables(
        self,
        *,
        params: dict[str, Any],
    ) -> None:
        """Replace declared protocol tables with live, dependency-driven forms.

        Demos may place specifications under
        ``protocol.metadata["computed_protocol_tables"]``. The writer only
        implements reusable table generators and does not inspect sequence
        names, event names, or TSE-specific node paths.
        """

        protocol = getattr(self.symbolic_sequence, "protocol", None)
        metadata = getattr(protocol, "metadata", None)
        if not isinstance(metadata, Mapping):
            return
        specifications = metadata.get("computed_protocol_tables")
        if not isinstance(specifications, Mapping):
            return

        for table_name, specification in specifications.items():
            if not isinstance(specification, Mapping):
                continue
            generator = str(specification.get("generator") or "").strip()
            inputs = specification.get("inputs")
            if not isinstance(inputs, Mapping):
                continue

            table_path = f"root.prot.{_protocol_key(str(table_name))}"
            if generator == "phase_encode_order_table":
                required = (
                    "n_y",
                    "echo_train_length",
                    "order",
                    "center_echo_index",
                )
                if any(key not in inputs for key in required):
                    continue
                params[table_path] = _expr(
                    inputs={
                        key: f"root.prot.{_protocol_key(str(inputs[key]))}"
                        for key in required
                    },
                    script=(
                        "local ny = math.floor((n_y or 1) + 0.5)\n"
                        "local etl = math.floor((echo_train_length or 1) + 0.5)\n"
                        "if ny < 1 or etl < 1 or ny % etl ~= 0 then return {} end\n"
                        "local nshots = ny / etl\n"
                        "local key = string.lower(tostring(order or 'linear'))\n"
                        "local acquisition = {}\n"
                        "if key == 'reverse' then\n"
                        "  for i=1,ny do acquisition[i] = ny/2 - i end\n"
                        "elseif key == 'centric' then\n"
                        "  acquisition[1] = 0\n"
                        "  local k = 2\n"
                        "  local radius = 1\n"
                        "  while k <= ny do\n"
                        "    acquisition[k] = -radius; k = k + 1\n"
                        "    if k <= ny then acquisition[k] = radius; k = k + 1 end\n"
                        "    radius = radius + 1\n"
                        "  end\n"
                        "else\n"
                        "  for i=1,ny do acquisition[i] = -ny/2 + (i-1) end\n"
                        "end\n"
                        "local zero_index = 1\n"
                        "for i=1,ny do if acquisition[i] == 0 then zero_index = i; break end end\n"
                        "local zero_col = math.floor((zero_index-1)/nshots)\n"
                        "local target_col = math.floor((center_echo_index or 0) + 0.5)\n"
                        "local shift = target_col - zero_col\n"
                        "local table_out = {}\n"
                        "for row=0,nshots-1 do\n"
                        "  local line = {}\n"
                        "  for col=0,etl-1 do\n"
                        "    local source_col = (col - shift) % etl\n"
                        "    local source_index = row + source_col*nshots + 1\n"
                        "    line[col+1] = acquisition[source_index]\n"
                        "  end\n"
                        "  table_out[row+1] = line\n"
                        "end\n"
                        "return table_out"
                    ),
                )

    def _apply_nested_loop_structure_and_bindings(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        kernel_path: str,
    ) -> None:
        """Preserve one nested child Loop and protocol-driven event bindings.

        This pass is generic: hierarchy comes from dotted ``set_node`` paths,
        and varying gradient values come from event binding metadata.
        """

        registry = _sequence_node_registry_for_export(self.sequence)
        repeated = [
            (name, record)
            for name, record in registry.items()
            if (
                isinstance(record, Mapping)
                and str(record.get("repeat_mode") or "").lower() == "loop"
            )
        ]
        repeated.sort(key=lambda item: _node_depth_for_export(item[0]))

        if len(repeated) < 2:
            return

        outer_name, outer_record = repeated[0]
        inner_candidates = [
            (name, record)
            for name, record in repeated[1:]
            if _node_is_within_for_export(name, outer_name)
        ]
        if not inner_candidates:
            return

        inner_name, inner_record = min(
            inner_candidates,
            key=lambda item: _node_depth_for_export(item[0]),
        )

        outer_loop = self._active_loop_path
        inner_token = _repeat_loop_token_for_export(
            inner_record.get("repeat_count"),
            inner_record.get("counter"),
        )
        inner_loop = f"{kernel_path}.{_safe_path_token(inner_token)}"

        sequence_elements[inner_loop] = "Loop"
        params[f"{inner_loop}.counter"] = _literal(0)

        inner_count_ref = inner_record.get("repeat_count")
        inner_count_source = _protocol_source_name_from_value(
            inner_count_ref,
            self._live_protocol_values,
        )
        if inner_count_source is None:
            symbolic_registry = _sequence_node_registry_for_export(
                self.symbolic_sequence
            )
            symbolic_inner = symbolic_registry.get(inner_name, {})
            if isinstance(symbolic_inner, Mapping):
                inner_count_source = _protocol_source_name_from_value(
                    symbolic_inner.get("repeat_count"),
                    self._live_protocol_values,
                )
        if inner_count_source is not None:
            params[f"{inner_loop}.length"] = _expr(
                inputs={
                    "count": f"root.prot.{_protocol_key(inner_count_source)}"
                },
                script="return count",
            )
        else:
            params[f"{inner_loop}.length"] = _literal(
                int(inner_count_ref or 1)
            )

        inner_period_ref = inner_record.get("repeat_every")
        inner_period_source = _protocol_source_name_from_value(
            inner_period_ref,
            self._live_protocol_values,
        )
        if inner_period_source is None:
            symbolic_registry = _sequence_node_registry_for_export(
                self.symbolic_sequence
            )
            symbolic_inner = symbolic_registry.get(inner_name, {})
            if isinstance(symbolic_inner, Mapping):
                inner_period_source = _protocol_source_name_from_value(
                    symbolic_inner.get("repeat_every"),
                    self._live_protocol_values,
                )
        if inner_period_source is not None:
            period_path = (
                f"root.prot.{_protocol_key(inner_period_source)}"
            )
            params[f"{inner_loop}.duration"] = _expr(
                inputs={"period": period_path},
                script="return period",
            )
            params[f"{inner_loop}.tstart"] = _expr(
                inputs={
                    "counter": f"{inner_loop}.counter",
                    "period": period_path,
                },
                script="return counter * period",
            )
        else:
            period = float(inner_period_ref or 0.0)
            params[f"{inner_loop}.duration"] = _literal(period)
            params[f"{inner_loop}.tstart"] = _expr(
                inputs={"counter": f"{inner_loop}.counter"},
                script=f"return counter * {period!r}",
            )

        # Move block groups belonging to the inner node beneath the inner Loop.
        group_names = {
            str(record.get("local_name") or "").strip()
            for record in self._block_export_records
            if _node_is_within_for_export(
                str(record.get("node") or ""),
                inner_name,
            )
        }
        group_names.discard("")

        prefix_map: dict[str, str] = {}
        for group_name in group_names:
            old = f"{kernel_path}.{_safe_path_token(group_name)}"
            if old in sequence_elements:
                prefix_map[old] = (
                    f"{inner_loop}.{_safe_path_token(group_name)}"
                )

        if prefix_map:
            ordered = sorted(prefix_map, key=len, reverse=True)

            def rewrite(path: str) -> str:
                for old in ordered:
                    if path == old or path.startswith(old + "."):
                        return prefix_map[old] + path[len(old):]
                return path

            rewritten_params: dict[str, Any] = {}
            for key, value in params.items():
                new_value = value
                if isinstance(value, Mapping):
                    new_value = dict(value)
                    inputs = value.get("inputs")
                    if isinstance(inputs, Mapping):
                        new_value["inputs"] = {
                            str(name): rewrite(str(path))
                            for name, path in inputs.items()
                        }
                rewritten_params[rewrite(str(key))] = new_value
            params.clear()
            params.update(rewritten_params)

            rewritten_elements = {
                rewrite(str(path)): blueprint
                for path, blueprint in sequence_elements.items()
            }
            sequence_elements.clear()
            sequence_elements.update(rewritten_elements)

        # Replace nested-loop-bound gradient samples with protocol-driven
        # expressions. The shape time axis and normalized waveform are retained
        # once; only amplitude is selected by shot/echo counters.
        for event_id, event in self._event_container_events.items():
            metadata = getattr(event, "metadata", None)
            expression_binding = (
                metadata.get("seqstar_nested_loop_expression_binding")
                if isinstance(metadata, Mapping)
                else None
            )
            binding = (
                metadata.get("seqstar_nested_loop_binding")
                if isinstance(metadata, Mapping)
                else None
            )
            if binding is None:
                binding = getattr(
                    event,
                    "_seqstar_nested_loop_binding",
                    None,
                )
            active_binding = (
                expression_binding
                if isinstance(expression_binding, Mapping)
                else binding
            )
            if not isinstance(active_binding, Mapping):
                continue

            leaf = self._event_container_leaves.get(event_id)
            if not leaf:
                continue

            # Account for any group-path rewrite above.
            if prefix_map:
                for old, new in prefix_map.items():
                    if leaf == old or leaf.startswith(old + "."):
                        leaf = new + leaf[len(old):]
                        break

            samples_key = f"{leaf}.samples"
            samples_param = params.get(samples_key)
            if not isinstance(samples_param, Mapping):
                continue

            grad_data = self._gradient_waveform_data(
                gradient_event=event
            )
            samples = grad_data.get("samples") or {}
            times = list(samples.get("t", []))
            values = list(samples.get("v", []))
            area = float(grad_data.get("area") or 0.0)
            if not values or abs(area) < 1e-20:
                continue

            normalized_values = [float(v) / area for v in values]
            params[f"{leaf}.normalized_samples"] = _literal(
                {"t": times, "v": normalized_values}
            )

            if isinstance(expression_binding, Mapping):
                protocol_inputs = expression_binding.get("protocol_inputs", {})
                loop_inputs = expression_binding.get("loop_inputs", {})
                script = str(expression_binding.get("script") or "return 0")
                if not isinstance(protocol_inputs, Mapping):
                    protocol_inputs = {}
                if not isinstance(loop_inputs, Mapping):
                    loop_inputs = {}

                expression_inputs = {
                    str(alias): f"root.prot.{_protocol_key(str(source))}"
                    for alias, source in protocol_inputs.items()
                }
                for alias, source in loop_inputs.items():
                    source_key = str(source).strip().lower()
                    if source_key == "outer":
                        expression_inputs[str(alias)] = f"{outer_loop}.counter"
                    elif source_key == "inner":
                        expression_inputs[str(alias)] = f"{inner_loop}.counter"
                    else:
                        expression_inputs[str(alias)] = str(source)

                params[f"{leaf}.area"] = _expr(
                    inputs=expression_inputs,
                    script=script,
                )
                sample_inputs = {"shape": f"{leaf}.normalized_samples"}
                sample_inputs.update(expression_inputs)
                params[samples_key] = _expr(
                    inputs=sample_inputs,
                    script=(
                        "local function seqstar_area()\n"
                        + script
                        + "\nend\n"
                        "local area = seqstar_area()\n"
                        "local out = {t=shape.t, v={}}\n"
                        "for i=1,#shape.v do out.v[i] = shape.v[i] * area end\n"
                        "return out"
                    ),
                )
                continue

            protocol_table = str(
                binding.get("protocol_table", "phase_encode_steps")
            )
            scale = float(binding.get("scale", 1.0))

            params[samples_key] = _expr(
                inputs={
                    "shape": f"{leaf}.normalized_samples",
                    "table": f"root.prot.{protocol_table}",
                    "shot": f"{outer_loop}.counter",
                    "echo": f"{inner_loop}.counter",
                },
                script=(
                    "local si = (shot or 0) + 1\n"
                    "local ei = (echo or 0) + 1\n"
                    f"local area = table[si][ei] * {scale!r}\n"
                    "local out = {t=shape.t, v={}}\n"
                    "for i=1,#shape.v do out.v[i] = shape.v[i] * area end\n"
                    "return out"
                ),
            )
            params[f"{leaf}.area"] = _expr(
                inputs={
                    "table": f"root.prot.{protocol_table}",
                    "shot": f"{outer_loop}.counter",
                    "echo": f"{inner_loop}.counter",
                },
                script=(
                    "local si = (shot or 0) + 1\n"
                    "local ei = (echo or 0) + 1\n"
                    f"return table[si][ei] * {scale!r}"
                ),
            )

    def _apply_single_loop_variations_and_bindings(
        self,
        *,
        params: dict[str, Any],
        loop_path: str,
    ) -> None:
        """Apply one-dimensional loop variation metadata to exported events.

        ``kernel.vary(vary(event, "area", strength=..., step=...))`` attaches a
        declarative binding to the event.  For a single active loop, the writer
        should not leave the event waveform literal; it should scale the shape
        by the loop counter.

        This pass is sequence-agnostic.  It does not know GRE/EPI.  It consumes
        event metadata written by ``SeqStarSequence._register_node_variation``
        and rewrites the exported event's selected property.
        """

        counter_path = f"{loop_path}.counter"
        if counter_path not in params:
            return

        for event_id, event in list(self._event_container_events.items()):
            binding = self._variation_binding_for_event(event)
            if not isinstance(binding, Mapping):
                continue

            attribute = str(binding.get("attribute") or "").strip().lower()
            leaf = self._event_container_leaves.get(event_id)
            if not leaf:
                continue

            export_targets = self._variation_export_targets(
                event=event,
                leaf=leaf,
                attribute=attribute,
                params=params,
            )
            if not export_targets:
                self._record_variation_export(
                    event=event,
                    leaf=leaf,
                    binding=binding,
                    attribute=attribute,
                    targets=[],
                    status="unsupported_attribute",
                )
                continue

            if _is_gradient_event(event) and attribute in {"area", "amplitude"}:
                self._apply_single_loop_gradient_variation(
                    params=params,
                    event=event,
                    leaf=leaf,
                    binding=binding,
                    counter_path=counter_path,
                    attribute=attribute,
                )
                self._record_variation_export(
                    event=event,
                    leaf=leaf,
                    binding=binding,
                    attribute=attribute,
                    targets=export_targets,
                    status="exported",
                )
                continue

            for target in export_targets:
                self._write_single_loop_linear_value(
                    params=params,
                    path=target,
                    binding=binding,
                    counter_path=counter_path,
                )

            self._record_variation_export(
                event=event,
                leaf=leaf,
                binding=binding,
                attribute=attribute,
                targets=export_targets,
                status="exported",
            )

    def _variation_export_targets(
        self,
        *,
        event: Any,
        leaf: str,
        attribute: str,
        params: Mapping[str, Any],
    ) -> list[str]:
        """Return gammaSTAR parameter paths affected by a public variation.

        ``node.vary()`` is a semantic API.  The developer varies event
        properties such as ``phase_offset`` or ``area``; each writer maps that
        semantic property onto the fields consumed by its backend.

        For gammaSTAR, RF/ADC basic representations consume ``.phase`` and
        ``.frequency``.  PyPulseq-style event objects expose
        ``phase_offset``/``freq_offset``.  Therefore a variation of
        ``phase_offset`` on RF/ADC must update the consumed ``.phase`` field;
        otherwise the relationship is exported but has no effect on the
        website plot or executable representation.

        The original semantic alias is also exported when possible for
        traceability/debugging.  This keeps the abstraction generic and avoids
        GRE-specific RF-spoiling logic.
        """

        attr = str(attribute or "").strip().lower()
        targets: list[str] = []

        def add(suffix: str) -> None:
            path = f"{leaf}.{suffix}"
            if path not in targets:
                targets.append(path)

        if _is_gradient_event(event):
            if attr in {"area", "amplitude"}:
                add(attr)
            return targets

        if _is_rf_event(event) or _is_adc_event(event):
            phase_aliases = {
                "phase",
                "phase_offset",
                "rf_phase",
                "adc_phase",
                "rf_phase_offset",
                "adc_phase_offset",
            }
            frequency_aliases = {
                "frequency",
                "freq_offset",
                "frequency_offset",
                "rf_frequency",
                "adc_frequency",
                "rf_frequency_offset",
                "adc_frequency_offset",
            }

            if attr in phase_aliases:
                # Backend-consumed field first.
                add("phase")
                # Preserve public/developer-facing spelling for inspection.
                if attr != "phase":
                    add(attr)
                if attr != "phase_offset":
                    add("phase_offset")
                return targets

            if attr in frequency_aliases:
                add("frequency")
                if attr != "frequency":
                    add(attr)
                if attr not in {"freq_offset", "frequency_offset"}:
                    add("frequency_offset")
                return targets

        # Last-resort generic path: if the backend already exposes a parameter
        # of the same name, vary it.  This lets custom event objects extend the
        # contract without modifying the writer.
        if f"{leaf}.{attr}" in params:
            add(attr)

        return targets

    def _record_variation_export(
        self,
        *,
        event: Any,
        leaf: str,
        binding: Mapping[str, Any],
        attribute: str,
        targets: list[str],
        status: str,
    ) -> None:
        """Store an auditable variation-export record for validators/tests."""

        records = getattr(self, "_variation_export_records", None)
        if records is None:
            records = []
            self._variation_export_records = records

        records.append(
            {
                "event_name": str(
                    getattr(event, "name", None)
                    or getattr(event, "path", None)
                    or event.__class__.__name__
                ),
                "event_type": (
                    "gradient"
                    if _is_gradient_event(event)
                    else "rf"
                    if _is_rf_event(event)
                    else "adc"
                    if _is_adc_event(event)
                    else event.__class__.__name__
                ),
                "leaf": str(leaf),
                "attribute": str(attribute),
                "targets": [str(target) for target in targets],
                "node": str(binding.get("node", "")),
                "counter": str(binding.get("counter", "")),
                "mode": str(binding.get("mode", "linear")),
                "status": str(status),
            }
        )

    def _apply_single_loop_gradient_variation(
        self,
        *,
        params: dict[str, Any],
        event: Any,
        leaf: str,
        binding: Mapping[str, Any],
        counter_path: str,
        attribute: str,
    ) -> None:
        """Scale a gradient waveform by a loop-varying area/amplitude."""

        samples_key = f"{leaf}.samples"
        if samples_key not in params:
            return

        grad_data = self._gradient_waveform_data(gradient_event=event)
        samples = grad_data.get("samples") or {}
        times = list(samples.get("t", []))
        values = list(samples.get("v", []))
        if not values:
            return

        # Prefer area normalization because GRE phase encoding varies area.  If
        # area is unavailable and the user varied amplitude, fall back to max
        # absolute amplitude normalization.
        area = _safe_float_or_none(grad_data.get("area"))
        max_abs = _safe_float_or_none(grad_data.get("max_abs_amplitude"))
        if area is not None and abs(area) > 1e-20:
            scale0 = area
        elif max_abs is not None and abs(max_abs) > 1e-20:
            scale0 = max_abs
        else:
            return

        normalized_values = [float(v) / scale0 for v in values]
        params[f"{leaf}.normalized_samples"] = _literal(
            {"t": times, "v": normalized_values}
        )

        varying_value_path = f"{leaf}.{attribute}"
        self._write_single_loop_linear_value(
            params=params,
            path=varying_value_path,
            binding=binding,
            counter_path=counter_path,
        )

        if attribute == "area":
            params[samples_key] = _expr(
                inputs={
                    "shape": f"{leaf}.normalized_samples",
                    "area": varying_value_path,
                },
                script=(
                    "local out = {t=shape.t, v={}}\n"
                    "for i=1,#shape.v do out.v[i] = shape.v[i] * area end\n"
                    "return out"
                ),
            )
            params[f"{leaf}.max_abs_amplitude"] = _expr(
                inputs={
                    "shape": f"{leaf}.normalized_samples",
                    "area": varying_value_path,
                },
                script=(
                    "local maxv = 0\n"
                    "for i=1,#shape.v do\n"
                    "  local v = math.abs(shape.v[i] * area)\n"
                    "  if v > maxv then maxv = v end\n"
                    "end\n"
                    "return maxv"
                ),
            )
        else:
            params[samples_key] = _expr(
                inputs={
                    "shape": f"{leaf}.normalized_samples",
                    "amplitude": varying_value_path,
                },
                script=(
                    "local out = {t=shape.t, v={}}\n"
                    "for i=1,#shape.v do out.v[i] = shape.v[i] * amplitude end\n"
                    "return out"
                ),
            )
            params[f"{leaf}.max_abs_amplitude"] = _expr(
                inputs={"amplitude": varying_value_path},
                script="return math.abs(amplitude)",
            )

        params[f"{leaf}.seqstar_variation_attribute"] = _literal(attribute)
        params[f"{leaf}.seqstar_variation_counter"] = _literal(counter_path)

    def _variation_value_lua_expression(
        self,
        value: Any,
        *,
        inputs: dict[str, str],
        source_to_name: dict[str, str],
        target_path: str,
        seen_protocol_keys: set[str] | None = None,
    ) -> str:
        """Compile a variation value while inlining derived protocol refs.

        gammaSTAR currently does not always invalidate transitive protocol
        dependencies in the live editor, e.g.

            root.prot.n_y -> root.prot.phase_encode_start -> gy.grad.area

        The JSON relationship is valid, but the plot may keep the old value of
        the intermediate derived parameter.  For loop-varying event properties,
        the plot target should therefore depend directly on the editable source
        protocol controls whenever the intermediate protocol parameter is itself
        a derived expression.

        This is generic expression inlining. It does not know GRE, EPI, ky,
        phase encoding, n_y, or fov.  It simply expands ParameterRef(X) when X
        is present in the merged live protocol table as another expression.
        """

        seen = set(seen_protocol_keys or ())

        if _is_seqstar_expression(value):
            if hasattr(value, "canonical_name"):
                canonical = str(value.canonical_name)
                gamma_key = _protocol_key(canonical)
                live_value = self._live_protocol_values.get(canonical)
                if live_value is None:
                    live_value = self._live_protocol_values.get(gamma_key)

                if (
                    live_value is not None
                    and _contains_protocol_expression(live_value)
                    and canonical not in seen
                    and gamma_key not in seen
                ):
                    return self._variation_value_lua_expression(
                        live_value,
                        inputs=inputs,
                        source_to_name=source_to_name,
                        target_path=target_path,
                        seen_protocol_keys=seen | {canonical, gamma_key},
                    )

                source_path = f"root.prot.{gamma_key}"
                if source_path == target_path:
                    raise TypeError(
                        f"Variation expression for {target_path!r} references itself."
                    )
                if source_path not in source_to_name:
                    name = _safe_input_name(source_path)
                    base = name
                    suffix = 2
                    used = set(inputs)
                    while name in used:
                        name = f"{base}_{suffix}"
                        suffix += 1
                    source_to_name[source_path] = name
                    inputs[name] = source_path
                return source_to_name[source_path]

            if hasattr(value, "value") and not hasattr(value, "operator_name"):
                return _to_lua(value.value)

            operator_name = getattr(value, "operator_name", None)

            if operator_name is not None and hasattr(value, "operand"):
                operand = self._variation_value_lua_expression(
                    value.operand,
                    inputs=inputs,
                    source_to_name=source_to_name,
                    target_path=target_path,
                    seen_protocol_keys=seen,
                )
                if operator_name == "neg":
                    return f"(-({operand}))"

            if (
                operator_name is not None
                and hasattr(value, "left")
                and hasattr(value, "right")
            ):
                left = self._variation_value_lua_expression(
                    value.left,
                    inputs=inputs,
                    source_to_name=source_to_name,
                    target_path=target_path,
                    seen_protocol_keys=seen,
                )
                right = self._variation_value_lua_expression(
                    value.right,
                    inputs=inputs,
                    source_to_name=source_to_name,
                    target_path=target_path,
                    seen_protocol_keys=seen,
                )
                operator_map = {
                    "add": "+",
                    "sub": "-",
                    "mul": "*",
                    "truediv": "/",
                    "pow": "^",
                }
                if operator_name in operator_map:
                    return f"(({left}) {operator_map[operator_name]} ({right}))"

            raise TypeError(
                "Cannot compile variation expression to Lua: "
                f"{type(value).__name__}."
            )

        return _to_lua(value)

    def _write_single_loop_linear_value(
        self,
        *,
        params: dict[str, Any],
        path: str,
        binding: Mapping[str, Any],
        counter_path: str,
    ) -> None:
        """Write value = strength + counter * step, with optional wrap.

        The helper parameters are still exported for inspectability, but the
        plotted target expression inlines derived protocol expressions so live
        protocol edits can invalidate the actual plotted waveform even if the
        UI does not propagate through intermediate derived protocol nodes.
        """

        strength_path = f"{path}_variation_strength"
        step_path = f"{path}_variation_step"
        wrap_path = f"{path}_variation_wrap"

        strength_value = binding.get("strength")
        step_value = binding.get("step")

        params[strength_path] = self._variation_scalar_parameter(
            strength_value,
            fallback=0.0,
        )
        params[step_path] = self._variation_scalar_parameter(
            step_value,
            fallback=0.0,
        )

        inputs: dict[str, str] = {"counter": counter_path}
        source_to_name: dict[str, str] = {}

        try:
            strength_lua = self._variation_value_lua_expression(
                strength_value,
                inputs=inputs,
                source_to_name=source_to_name,
                target_path=path,
            )
        except Exception:
            inputs["strength"] = strength_path
            strength_lua = "strength"

        try:
            step_lua = self._variation_value_lua_expression(
                step_value,
                inputs=inputs,
                source_to_name=source_to_name,
                target_path=path,
            )
        except Exception:
            inputs["step"] = step_path
            step_lua = "step"

        script = (
            f"local value = ({strength_lua}) + (counter or 0) * ({step_lua})\n"
        )

        wrap = binding.get("wrap")
        if wrap is not None:
            params[wrap_path] = self._variation_scalar_parameter(
                wrap,
                fallback=None,
            )
            try:
                wrap_lua = self._variation_value_lua_expression(
                    wrap,
                    inputs=inputs,
                    source_to_name=source_to_name,
                    target_path=path,
                )
            except Exception:
                inputs["wrap"] = wrap_path
                wrap_lua = "wrap"
            script += (
                f"local wrap_value = {wrap_lua}\n"
                "if wrap_value ~= nil and wrap_value ~= 0 then\n"
                "  value = value % wrap_value\n"
                "end\n"
            )

        script += "return value"
        params[path] = _expr(inputs=inputs, script=script)

        if _gammastar_debug_enabled():
            _gammastar_debug(
                "linear variation wrote/direct "
                f"path={path} "
                f"inputs={inputs!r} "
                f"script={script!r} "
                f"strength_helper={params.get(strength_path)!r} "
                f"step_helper={params.get(step_path)!r}"
            )

    def _variation_scalar_parameter(
        self,
        value: Any,
        *,
        fallback: float | None,
    ) -> dict[str, Any]:
        """Return a scalar variation value.

        Direct protocol references may remain live.  Compound symbolic
        expressions must *not* be collapsed to any token they merely contain.
        For example, ``-0.5 * n_y / fov`` contains ``fov``, but it is not equal
        to ``fov``.  The v23 writer used broad token-containment matching here,
        which incorrectly exported both GRE phase-encode strength and step as
        ``root.prot.fov``.  For v0.2, compound expressions are evaluated to the
        current protocol defaults; full symbolic parameter-range propagation
        remains the v0.3 feasibility-engine task.
        """

        # Only preserve genuinely direct protocol references as live links.
        # Do not use the broad token-containment fallback in
        # _protocol_source_name_from_value(), because variation expressions are
        # often compound expressions containing several protocol tokens.
        source = _direct_protocol_source_name_from_value(
            value,
            self._live_protocol_values,
        )
        if source is not None:
            return _expr(
                inputs={"value": f"root.prot.{_protocol_key(source)}"},
                script="return value",
            )

        numeric = _safe_float_or_none(value)
        if numeric is None and hasattr(value, "eval"):
            for context in (
                self.realization,
                self.sequence,
                self.symbolic_sequence,
                getattr(self.sequence, "protocol", None),
                getattr(self.symbolic_sequence, "protocol", None),
            ):
                try:
                    numeric = _safe_float_or_none(value.eval(context))
                except Exception:
                    numeric = None
                if numeric is not None:
                    break

        if numeric is None:
            numeric = fallback

        return _literal(numeric)

    def _variation_numeric_default(self, value: Any) -> float | None:
        """Evaluate a relationship value at the writer's current defaults.

        This helper is deliberately sequence-agnostic.  It knows nothing about
        EPI, FOV, matrix size, or gradient roles; it only evaluates the value
        declared by ``node.vary(...)`` using the available realization/protocol
        contexts.
        """

        numeric = _safe_float_or_none(value)
        if numeric is not None:
            return numeric

        if hasattr(value, "eval"):
            for context in (
                self.realization,
                self.sequence,
                self.symbolic_sequence,
                getattr(self.sequence, "protocol", None),
                getattr(self.symbolic_sequence, "protocol", None),
            ):
                if context is None:
                    continue
                try:
                    numeric = _safe_float_or_none(value.eval(context))
                except Exception:
                    numeric = None
                if numeric is not None:
                    return numeric

        return None

    def _variation_binding_for_event(self, event: Any) -> Mapping[str, Any] | None:
        """Return the first loop variation binding associated with an event."""

        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("seqstar_loop_binding")
            if isinstance(value, Mapping):
                return value
            variations = metadata.get("seqstar_variations")
            if isinstance(variations, Iterable):
                for item in variations:
                    if isinstance(item, Mapping):
                        return item

        parameters = getattr(event, "parameters", None)
        if isinstance(parameters, Mapping):
            variations = parameters.get("_seqstar_variations")
            if isinstance(variations, Iterable):
                for item in variations:
                    if isinstance(item, Mapping):
                        return item

        # A resolved/deepcopied event can lose Python object identity.  Fall
        # back to matching the variation record by event name from the sequence
        # node registry.
        event_name = str(
            getattr(event, "name", None)
            or getattr(event, "path", None)
            or event.__class__.__name__
        )
        registry = _sequence_node_registry_for_export(
            self.symbolic_sequence or self.sequence
        )
        for record in registry.values():
            if not isinstance(record, Mapping):
                continue
            variations = record.get("variations")
            if not isinstance(variations, Iterable):
                continue
            for item in variations:
                if not isinstance(item, Mapping):
                    continue
                names = item.get("event_names")
                if isinstance(names, Iterable) and event_name in {str(name) for name in names}:
                    return item
        return None

    def _add_seqstar_block_timing_dependencies(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        kernel_path: str,
        export_blocks: list[Any],
        exported_events: list[Any],
    ) -> None:
        """Export logical SeqStar block timing as dependency expressions.

        The concrete Pulseq timeline has already been resolved numerically. For
        gammaSTAR Protocol Control we also need the *dependency structure* that
        says a downstream block starts at the previous block end. This method
        emits lightweight logical block parameters such as::

            root.seqstar_loop.kernel.seqstar_blocks.kernel_readout.tstart
            root.seqstar_loop.kernel.seqstar_blocks.kernel_readout.tend

        and then rewrites event container ``.tstart`` parameters for events that
        are not explicitly anchored by a relationship so they reference their
        logical block start. If an event in a block is explicitly anchored, the
        block start is tied back to that exported event container. This makes
        downstream blocks, especially spoilers, follow protocol edits such as TE.
        """

        if not export_blocks:
            return

        block_records: list[dict[str, Any]] = []
        block_start = 0.0
        for block_index, block in enumerate(export_blocks):
            node = _block_node_for_export(block, default=f"block_{block_index:03d}")
            local_name = _block_local_name_for_export(block, node)
            token = _safe_path_token(node)
            block_path = f"{kernel_path}.seqstar_blocks.{token}"
            duration = max(
                _safe_block_duration(block),
                _events_extent_duration(list(_iter_events(block))),
            )
            parent = _block_parent_for_export(block, node)
            record = {
                "block": block,
                "index": block_index,
                "node": node,
                "parent": parent,
                "local_name": local_name,
                "path": block_path,
                "duration": float(duration),
                "resolved_start": float(block_start),
                "resolved_end": float(block_start + duration),
            }
            block_records.append(record)
            self._block_export_paths[block_index] = block_path
            sequence_elements.setdefault(block_path, "Info")
            block_start += float(duration)

        self._block_export_records = block_records

        # First emit a pure block-after chain. This is the fallback dependency
        # model and is also the source of ``tend`` for downstream blocks.
        previous_path: str | None = None
        for record in block_records:
            block_path = str(record["path"])
            params[f"{block_path}.node"] = _literal(record["node"])
            params[f"{block_path}.parent"] = _literal(record["parent"])
            params[f"{block_path}.role"] = _literal(_block_role(record["block"]) or record["local_name"])
            params[f"{block_path}.resolved_tstart"] = _literal(record["resolved_start"])
            params[f"{block_path}.duration"] = _literal(record["duration"])
            if previous_path is None:
                params[f"{block_path}.tstart"] = _literal(record["resolved_start"])
            else:
                params[f"{block_path}.tstart"] = _expr(
                    inputs={"previous_tend": f"{previous_path}.tend"},
                    script="return previous_tend",
                )
            params[f"{block_path}.tend"] = _expr(
                inputs={
                    "tstart": f"{block_path}.tstart",
                    "duration": f"{block_path}.duration",
                },
                script="return tstart + duration",
            )
            previous_path = block_path

        # If an explicitly anchored event lives in a block, the block start must
        # follow that event. This is what lets TE move the readout block, and via
        # block_after, the spoiler block.
        anchored_block_overrides = self._anchored_block_tstart_overrides(exported_events)
        for block_index, override in anchored_block_overrides.items():
            if block_index < 0 or block_index >= len(block_records):
                continue
            block_path = str(block_records[block_index]["path"])
            event_path = str(override["container_path"])
            local_tstart = float(override.get("local_tstart", 0.0))
            params[f"{block_path}.tstart"] = _expr(
                inputs={"event_tstart": f"{event_path}.tstart"},
                script=f"return event_tstart - {local_tstart!r}",
            )
            params[f"{block_path}.anchor_event"] = _literal(str(override.get("event_name", "event")))
            params[f"{block_path}.anchor_relationship"] = _literal(str(override.get("relationship_name", "relationship")))

        # Finally, make ordinary unanchored event containers follow their
        # source block tstart.
        #
        # Multi-window ADC trains are different: the enclosing readout Loop
        # selects each window position, while the representative
        # SingleReadout child must remain local at tstart=0. Rewriting that
        # child to ``block_tstart + first_delay`` duplicates timing already
        # represented by the Loop and collapses parent aggregation.
        for event in exported_events:
            if (
                _is_adc_event(event)
                and _has_multiple_adc_windows_for_export(event)
            ):
                continue

            container_path = self._event_container_paths.get(id(event))
            if not container_path:
                continue
            block_index = _event_source_block_index_for_export(event)
            if block_index is None or block_index not in self._block_export_paths:
                continue
            if self._event_has_export_timing_relationship(event):
                continue
            local_tstart = _event_local_tstart_within_source_block(event)
            block_path = self._block_export_paths[block_index]
            params[f"{container_path}.tstart"] = _expr(
                inputs={"block_tstart": f"{block_path}.tstart"},
                script=f"return block_tstart + {float(local_tstart)!r}",
            )

    def _anchored_block_tstart_overrides(self, exported_events: list[Any]) -> dict[int, dict[str, Any]]:
        """Return block-start overrides from explicit anchored event timing."""

        out: dict[int, dict[str, Any]] = {}
        for event in exported_events:
            relationship = _find_timing_relationship_for_target(
                self.sequence,
                target_event=event,
            )
            if relationship is None:
                continue
            # Prefer solved relationships over validation-only relationships for
            # defining block start. Validation-only relationships such as
            # gx.center == adc.center should not override the ADC-solved one if
            # both occur in the same block.
            if bool(relationship.get("validation_only", False)):
                continue
            container_path = self._event_container_paths.get(id(event))
            if not container_path:
                continue
            block_index = _event_source_block_index_for_export(event)
            if block_index is None:
                continue
            out.setdefault(
                block_index,
                {
                    "container_path": container_path,
                    "local_tstart": _event_local_tstart_within_source_block(event),
                    "event_name": getattr(event, "name", "event"),
                    "relationship_name": relationship.get("name", "relationship"),
                },
            )
        return out

    def _event_has_export_timing_relationship(self, event: Any) -> bool:
        """Return True when event tstart is already relationship-driven."""

        relationship = _find_timing_relationship_for_target(
            self.sequence,
            target_event=event,
        )
        return relationship is not None

    def _group_exported_events_by_block(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        kernel_path: str,
        exported_events: list[Any],
    ) -> None:
        """Group events by logical block and rebase child timing correctly.

        gammaSTAR applies parent timing to nested children. Therefore, once an
        event branch is moved beneath a block group, its public ``tstart`` must
        become block-local rather than remain kernel-relative.

        To preserve cross-block timing relationships, every event container also
        receives a private ``seqstar_kernel_tstart`` parameter containing its
        original kernel-relative expression. Relationship dependencies are
        redirected to that parameter, while plotting uses the rebased local
        ``tstart``.

        This is sequence-agnostic: groups come from block metadata and event
        membership comes from source block indices.
        """

        if not self._block_export_records or not exported_events:
            return

        group_paths: dict[int, str] = {}
        logical_paths: dict[int, str] = {}
        used_paths: set[str] = set(sequence_elements)

        for record in self._block_export_records:
            block_index = int(record["index"])
            local_name = (
                record.get("local_name")
                or str(record.get("node") or f"block_{block_index:03d}").rsplit(".", 1)[-1]
            )
            group_path = f"{kernel_path}.{_safe_path_token(local_name)}"
            if group_path in used_paths:
                group_path = f"{group_path}_{block_index:03d}"

            used_paths.add(group_path)
            group_paths[block_index] = group_path
            logical_path = str(record["path"])
            logical_paths[block_index] = logical_path

            # One-iteration Loop gives gammaSTAR a real aggregating scope.
            sequence_elements[group_path] = "Loop"
            params[f"{group_path}.counter"] = _literal(0)
            params[f"{group_path}.length"] = _literal(1)
            params[f"{group_path}.tstart"] = _expr(
                inputs={"block_tstart": f"{logical_path}.tstart"},
                script="return block_tstart",
            )
            params[f"{group_path}.duration"] = _expr(
                inputs={"block_duration": f"{logical_path}.duration"},
                script="return block_duration",
            )
            params[f"{group_path}.tend"] = _expr(
                inputs={
                    "tstart": f"{group_path}.tstart",
                    "duration": f"{group_path}.duration",
                },
                script="return tstart + duration",
            )
            params[f"{group_path}.node"] = _literal(record.get("node"))
            params[f"{group_path}.role"] = _literal(
                _block_role(record["block"]) or record.get("local_name")
            )

        prefix_map: dict[str, str] = {}
        old_group_by_container: dict[str, str] = {}
        event_by_old_container: dict[str, Any] = {}

        for event in exported_events:
            block_index = _event_source_block_index_for_export(event)
            if block_index is None or block_index not in group_paths:
                continue

            old_container = self._event_container_paths.get(id(event))
            if not old_container or not old_container.startswith(kernel_path + "."):
                continue

            # A multi-window ADC is represented as:
            #
            #   <readout_loop>.single_readout.atomic
            #   <readout_loop>.single_readout.adc
            #
            # The event-container registry points to ``single_readout``. When
            # grouping by source block, move the complete readout Loop ancestor,
            # not only the SingleReadout child. Otherwise ``windows``,
            # ``window_data``, and ``counter`` remain orphaned at the old path
            # and gammaSTAR can render only one/default ADC window.
            if _is_adc_event(event) and _has_multiple_adc_windows_for_export(event):
                if old_container.endswith(".single_readout"):
                    old_container = old_container.rsplit(".", 1)[0]

            leaf_name = old_container.rsplit(".", 1)[-1]
            candidate = f"{group_paths[block_index]}.{leaf_name}"
            suffix = 2
            while candidate in sequence_elements or candidate in prefix_map.values():
                candidate = f"{group_paths[block_index]}.{leaf_name}_{suffix}"
                suffix += 1

            prefix_map[old_container] = candidate
            old_group_by_container[old_container] = group_paths[block_index]
            event_by_old_container[old_container] = event

        if not prefix_map:
            return

        # Preserve original kernel-relative event starts before rebasing them.
        for old_container, group_path in old_group_by_container.items():
            tstart_key = f"{old_container}.tstart"
            original = params.get(tstart_key, _literal(0.0))
            params[f"{old_container}.seqstar_kernel_tstart"] = original
            params[tstart_key] = _expr(
                inputs={
                    "kernel_tstart": f"{old_container}.seqstar_kernel_tstart",
                    "group_tstart": f"{group_path}.tstart",
                },
                script="return kernel_tstart - group_tstart",
            )

        ordered_prefixes = sorted(prefix_map, key=len, reverse=True)

        def rewrite_path(value: str) -> str:
            for old_prefix in ordered_prefixes:
                if value == old_prefix or value.startswith(old_prefix + "."):
                    return prefix_map[old_prefix] + value[len(old_prefix):]
            return value

        # Cross-event relationships must continue to consume kernel-relative
        # timing, not the newly rebased public child tstart.
        absolute_tstart_map = {
            f"{old}.tstart": f"{new}.seqstar_kernel_tstart"
            for old, new in prefix_map.items()
        }

        def rewrite_input_path(value: str) -> str:
            if value in absolute_tstart_map:
                return absolute_tstart_map[value]
            return rewrite_path(value)

        def rewrite_parameter(value: Any) -> Any:
            if not isinstance(value, Mapping):
                return value
            out = dict(value)
            inputs = out.get("inputs")
            if isinstance(inputs, Mapping):
                out["inputs"] = {
                    str(name): rewrite_input_path(str(path))
                    for name, path in inputs.items()
                }
            return out

        rewritten_params: dict[str, Any] = {}
        for key, value in params.items():
            rewritten_params[rewrite_path(str(key))] = rewrite_parameter(value)
        params.clear()
        params.update(rewritten_params)

        rewritten_elements: dict[str, str] = {}
        for path, blueprint in sequence_elements.items():
            rewritten_elements[rewrite_path(str(path))] = blueprint
        sequence_elements.clear()
        sequence_elements.update(rewritten_elements)

        # Container-local derived timing should use the local public tstart.
        for old_container, new_container in prefix_map.items():
            for suffix in ("tcenter", "tend"):
                key = f"{new_container}.{suffix}"
                param = params.get(key)
                if isinstance(param, Mapping):
                    fixed = dict(param)
                    inputs = dict(fixed.get("inputs", {}))
                    if "tstart" in inputs:
                        inputs["tstart"] = f"{new_container}.tstart"
                    fixed["inputs"] = inputs
                    params[key] = fixed

            # Atomic absolute time must now include the block-group parent start.
            group_path = rewrite_path(old_group_by_container[old_container])
            atomic_prefix = f"{new_container}."
            for key in list(params):
                if (
                    key.startswith(atomic_prefix)
                    and key.endswith(".atomic.tstart_absolute")
                ):
                    param = params.get(key)
                    if not isinstance(param, Mapping):
                        continue
                    fixed = dict(param)
                    inputs = dict(fixed.get("inputs", {}))
                    # Add the block-group start without discarding timing
                    # levels already present in the Atomic expression.
                    #
                    # A simple event normally has:
                    #
                    #   t0 + t1 + t2 + t3
                    #
                    # A nested event such as a multi-window ADC may have:
                    #
                    #   t0 + t1 + t2 + t3 + t4
                    #
                    # where t3 is the train/loop iteration start and t4 is the
                    # SingleReadout-local start. The previous grouping rewrite
                    # replaced the expression with a fixed four-level formula
                    # and silently dropped t4. Preserve every existing numeric
                    # timing input and insert only the new group parent.
                    inputs["group_tstart"] = f"{group_path}.tstart"

                    # Atomic plotting timing must consume the grouped event's
                    # public block-local tstart. The private
                    # ``seqstar_kernel_tstart`` exists only for cross-event
                    # relationship calculations. Using that private value here
                    # and also adding ``group_tstart`` double-counts the block
                    # offset.
                    #
                    # Replace every private kernel-relative timing input within
                    # this moved event branch by its corresponding public local
                    # tstart. This also handles nested branches such as:
                    #
                    #   readout Loop -> SingleReadout -> Atomic
                    #
                    # without any sequence-specific path checks.
                    for input_name, input_path in list(inputs.items()):
                        suffix = ".seqstar_kernel_tstart"
                        if str(input_path).endswith(suffix):
                            inputs[input_name] = (
                                str(input_path)[: -len(suffix)] + ".tstart"
                            )

                    timing_terms = [
                        name
                        for name in inputs
                        if (
                            name.startswith("t")
                            and name[1:].isdigit()
                        )
                    ]
                    timing_terms.sort(
                        key=lambda name: int(name[1:])
                    )

                    sum_terms = timing_terms[:3]
                    sum_terms.append("group_tstart")
                    sum_terms.extend(timing_terms[3:])

                    fixed["inputs"] = inputs
                    fixed["script"] = (
                        "return " + " + ".join(sum_terms)
                    )
                    params[key] = fixed

        # Add one Atomic aggregator directly beneath each block group.
        #
        # Important gammaSTAR compatibility rule:
        # Build the block Atomic directly from each contained event's
        # ``basic_repr_*`` and ``basic_repr_*_tstart_relative`` parameters.
        # This mirrors native gammaSTAR sequence JSON such as FLASH.
        #
        # Do not dynamically merge child ``full_basic_repr_*`` tables with Lua
        # ``pairs`` loops. Although valid Lua, that pattern is not evaluated
        # reliably by all gammaSTAR workplace aggregation paths and can leave a
        # group visually empty even though its child event leaves exist.
        containers_by_group: dict[str, list[tuple[str, Any]]] = {}

        for old_container, new_container in prefix_map.items():
            group_path = rewrite_path(old_group_by_container[old_container])
            event = event_by_old_container.get(old_container)
            if event is None:
                continue
            containers_by_group.setdefault(group_path, []).append(
                (new_container, event)
            )

        for group_path, container_events in containers_by_group.items():
            # A block containing exactly one multi-window ADC train plus one or
            # more concurrent, regularly segmented gradients is lowered as one
            # repeated gammaSTAR readout cycle:
            #
            #   group
            #     readout Loop
            #       single_readout
            #         Atomic: gradient segment(s) + ADC window
            #
            # The source SeqStar timeline remains compact. No sequence name,
            # trajectory name, node name, or developer-facing export flag is
            # required.
            if self._lower_implicit_synchronized_train_group(
                params=params,
                sequence_elements=sequence_elements,
                group_path=group_path,
                container_events=container_events,
            ):
                continue

            group_atomic = f"{group_path}.atomic"
            sequence_elements[group_atomic] = "Atomic"

            params[f"{group_atomic}.tstart_absolute"] = _expr(
                inputs={
                    "t0": "root.tstart",
                    "t1": f"{self._active_loop_path}.tstart",
                    "t2": f"{kernel_path}.tstart",
                    "t3": f"{group_path}.tstart",
                },
                script="return t0 + t1 + t2 + t3",
            )

            category_entries: dict[str, list[dict[str, str]]] = {
                "GradPulse": [],
                "RFPulse": [],
                "ADC": [],
                "TriggerPulse": [],
            }
            adc_train_entries: list[dict[str, str]] = []

            for container, event in container_events:
                event_key = _safe_path_token(
                    container.rsplit(".", 1)[-1]
                )

                if _is_gradient_event(event):
                    category = "GradPulse"
                    basic_name = "grad"
                    repr_container = container
                elif _is_rf_event(event):
                    category = "RFPulse"
                    basic_name = "rf"
                    repr_container = container
                elif _is_adc_event(event):
                    if _has_multiple_adc_windows_for_export(event):
                        # Preserve the nested editable hierarchy:
                        #
                        #   readout Loop -> single_readout -> ADC
                        #
                        # and separately expose all windows to the parent block
                        # Atomic for plotting. The parent representation is
                        # generated from the generic window table and does not
                        # alter loop timing, dead-time policy, or child nodes.
                        adc_train_entries.append(
                            {
                                "event_key": event_key,
                                "windows_path": f"{container}.windows",
                                "repr_path": (
                                    f"{container}.single_readout.atomic."
                                    "basic_repr_adc"
                                ),
                                "num_windows": len(
                                    _get_adc_windows_for_export(event)
                                ),
                            }
                        )
                        continue

                    category = "ADC"
                    basic_name = "adc"
                    repr_container = container
                else:
                    # Unsupported event families are intentionally omitted from
                    # the Atomic representation rather than guessed.
                    continue

                category_entries[category].append(
                    {
                        "event_key": event_key,
                        "repr_path": (
                            f"{repr_container}.atomic.basic_repr_{basic_name}"
                        ),
                        "tstart_path": (
                            f"{repr_container}.atomic."
                            f"basic_repr_{basic_name}_tstart_relative"
                        ),
                    }
                )

            for category, field_name in (
                ("GradPulse", "full_basic_repr_GradPulse"),
                ("RFPulse", "full_basic_repr_RFPulse"),
                ("ADC", "full_basic_repr_ADC"),
                ("TriggerPulse", "full_basic_repr_TriggerPulse"),
            ):
                entries = category_entries[category]

                if category == "ADC" and adc_train_entries:
                    inputs: dict[str, str] = {}
                    script_lines: list[str] = [
                        "local result = {}",
                    ]

                    # Preserve ordinary single-window ADC entries, if any.
                    for index, entry in enumerate(entries):
                        repr_name = f"basic_repr_{index}"
                        tstart_name = f"basic_tstart_{index}"
                        inputs[repr_name] = entry["repr_path"]
                        inputs[tstart_name] = entry["tstart_path"]
                        script_lines.append(
                            "result['"
                            + entry["event_key"]
                            + "']={tstart_relative="
                            + tstart_name
                            + ", required_parameters="
                            + repr_name
                            + "}"
                        )

                    # Add every window from each compact ADC train to the
                    # parent Atomic using statically enumerated keys.
                    #
                    # gammaSTAR reliably evaluates explicit Atomic tables, while
                    # runtime-generated table keys from Lua loops are not
                    # consistently propagated into parent plots. The nested
                    # readout Loop remains authoritative for editing and timing;
                    # this static table is only a parent-plot projection.
                    for train_index, train in enumerate(adc_train_entries):
                        windows_name = f"windows_{train_index}"
                        repr_name = f"train_repr_{train_index}"
                        inputs[windows_name] = train["windows_path"]
                        inputs[repr_name] = train["repr_path"]

                        for window_index in range(int(train["num_windows"])):
                            lua_index = window_index + 1
                            key = (
                                f"{train['event_key']}_window_{window_index}"
                            )
                            script_lines.append(
                                "result['"
                                + key
                                + "']={"
                                + "tstart_relative=("
                                + windows_name
                                + f"[{lua_index}].tstart or "
                                + windows_name
                                + f"[{lua_index}].delay or 0), "
                                + f"required_parameters={repr_name}"
                                + "}"
                            )

                    script_lines.append("return result")

                    params[f"{group_atomic}.{field_name}"] = _expr(
                        inputs=inputs,
                        script="\n".join(script_lines),
                    )
                    continue

                if not entries:
                    params[f"{group_atomic}.{field_name}"] = _expr(
                        inputs={},
                        script="return {}",
                    )
                    continue

                inputs: dict[str, str] = {}
                table_lines: list[str] = ["return {"]

                for index, entry in enumerate(entries):
                    repr_name = f"basic_repr_{index}"
                    tstart_name = f"basic_tstart_{index}"

                    inputs[repr_name] = entry["repr_path"]
                    inputs[tstart_name] = entry["tstart_path"]

                    table_lines.append(
                        f"['{entry['event_key']}']={{"
                        f"tstart_relative={tstart_name}, "
                        f"required_parameters={repr_name}"
                        "},"
                    )

                table_lines.append("}")

                params[f"{group_atomic}.{field_name}"] = _expr(
                    inputs=inputs,
                    script="\n".join(table_lines),
                )

            params[f"{group_atomic}.full_basic_repr"] = _expr(
                inputs={
                    "full_basic_repr_TriggerPulse": (
                        f"{group_atomic}.full_basic_repr_TriggerPulse"
                    ),
                    "full_basic_repr_ADC": (
                        f"{group_atomic}.full_basic_repr_ADC"
                    ),
                    "full_basic_repr_RFPulse": (
                        f"{group_atomic}.full_basic_repr_RFPulse"
                    ),
                    "full_basic_repr_GradPulse": (
                        f"{group_atomic}.full_basic_repr_GradPulse"
                    ),
                },
                script=(
                    "return {\n"
                    "GradPulse=full_basic_repr_GradPulse,\n"
                    "RFPulse=full_basic_repr_RFPulse,\n"
                    "ADC=full_basic_repr_ADC,\n"
                    "TriggerPulse=full_basic_repr_TriggerPulse\n"
                    "}"
                ),
            )

        for event_id, old_container in list(self._event_container_paths.items()):
            self._event_container_paths[event_id] = rewrite_path(old_container)
        for event_id, old_leaf in list(self._event_container_leaves.items()):
            self._event_container_leaves[event_id] = rewrite_path(old_leaf)

        # A compact repeated kernel represents exactly one repetition
        # interval. Use the live protocol TR for the kernel display extent.
        # This is generic for any compacted repeated motif and keeps the kernel
        # workplace view synchronized with protocol edits.
        if self._active_loop_path.endswith("seqstar_loop"):
            params[f"{kernel_path}.duration"] = _expr(
                inputs={"TR": "root.prot.TR"},
                script="return TR",
            )

        params["root.info.seqstar_block_grouping_enabled"] = _literal(True)
        params["root.info.seqstar_block_group_count"] = _literal(len(group_paths))
        params["root.info.seqstar_block_grouping_mode"] = _literal(
            "block_local_rebased_loop_scope_tr_kernel_parent_atomic"
        )

    def _lower_implicit_synchronized_train_group(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        group_path: str,
        container_events: list[tuple[str, Any]],
    ) -> bool:
        """Lower one compact synchronized train into a repeated readout cycle.

        Detection is intentionally structural:

        * exactly one multi-window ADC event;
        * one or more gradient events in the same source block;
        * no RF event in that block;
        * every gradient waveform has the same number of regular segments as
          ADC windows.

        The source events remain compact in SeqStar. Only the gammaSTAR
        representation is lowered. Each readout-loop iteration contains one
        gradient segment from every concurrent gradient train and one ADC
        window. This applies to any regular multi-window acquisition, not only
        EPI.
        """

        adc_items = [
            (container, event)
            for container, event in container_events
            if (
                _is_adc_event(event)
                and _has_multiple_adc_windows_for_export(event)
            )
        ]
        gradient_items = [
            (container, event)
            for container, event in container_events
            if _is_gradient_event(event)
        ]

        if len(adc_items) != 1 or not gradient_items:
            return False

        if any(_is_rf_event(event) for _, event in container_events):
            return False

        adc_container, adc_event = adc_items[0]
        windows = _get_adc_windows_for_export(adc_event)

        if len(windows) < 2:
            return False

        readout_loop_path = adc_container
        single_readout_path = f"{readout_loop_path}.single_readout"
        single_atomic_path = f"{single_readout_path}.atomic"

        if (
            sequence_elements.get(readout_loop_path) != "Loop"
            or single_readout_path not in sequence_elements
            or single_atomic_path not in sequence_elements
        ):
            return False

        num_segments = len(windows)

        first_window_for_timing = windows[0]
        default_adc_duration = float(
            _adc_window_value(first_window_for_timing, "duration", 0.0)
        )
        first_window_metadata = _adc_window_value(
            first_window_for_timing,
            "metadata",
            {},
        )
        if not isinstance(first_window_metadata, Mapping):
            first_window_metadata = {}

        default_adc_delay = float(
            _adc_window_value(
                first_window_for_timing,
                "event_delay",
                _adc_window_value(
                    first_window_for_timing,
                    "readout_delay",
                    first_window_metadata.get("event_delay", 0.0),
                ),
            )
        )

        # Infer the regular segment period from consecutive ADC-window starts.
        window_starts = [
            float(
                _adc_window_value(
                    window,
                    "tstart",
                    _adc_window_value(
                        window,
                        "delay",
                        0.0,
                    ),
                )
            )
            for window in windows
        ]
        spacings = [
            window_starts[index + 1] - window_starts[index]
            for index in range(num_segments - 1)
        ]

        if not spacings or spacings[0] <= 0:
            return False

        segment_duration = float(spacings[0])
        tolerance = max(
            1e-12,
            abs(segment_duration) * 1e-9,
        )

        if any(
            abs(spacing - segment_duration) > tolerance
            for spacing in spacings[1:]
        ):
            return False

        segmented_gradients: list[dict[str, Any]] = []

        for gradient_index, (container, event) in enumerate(gradient_items):
            grad_data = self._gradient_waveform_data(
                gradient_event=event
            )
            samples = grad_data.get("samples")

            if not isinstance(samples, Mapping):
                return False

            times = list(samples.get("t", []))
            values = list(samples.get("v", []))

            if (
                not times
                or len(times) != len(values)
                or len(times) % num_segments != 0
            ):
                return False

            samples_per_segment = len(times) // num_segments
            active_duration = float(
                grad_data.get(
                    "active_duration",
                    grad_data.get("duration", 0.0),
                )
            )
            expected_duration = segment_duration * num_segments

            # Permit one gradient-raster-sized numerical discrepancy.
            grad_raster = float(
                getattr(
                    self.system,
                    "grad_raster_time",
                    10e-6,
                )
            )
            if abs(active_duration - expected_duration) > max(
                grad_raster,
                expected_duration * 1e-8,
            ):
                return False

            segment_table: list[dict[str, Any]] = []

            for segment_index in range(num_segments):
                start = segment_index * samples_per_segment
                stop = start + samples_per_segment
                time_offset = segment_index * segment_duration

                local_times = [
                    float(value) - time_offset
                    for value in times[start:stop]
                ]
                local_values = values[start:stop]

                if local_times and local_times[0] < -tolerance:
                    return False
                if local_times and local_times[-1] > segment_duration + tolerance:
                    return False

                segment_table.append(
                    {
                        "t": local_times,
                        "v": local_values,
                    }
                )

            event_name = _safe_path_token(
                str(
                    getattr(
                        event,
                        "name",
                        f"gradient_{gradient_index}",
                    )
                )
            )
            leaf_path = (
                f"{single_readout_path}."
                f"{event_name}_segment"
            )
            suffix = 2
            original_leaf_path = leaf_path
            while leaf_path in sequence_elements:
                leaf_path = f"{original_leaf_path}_{suffix}"
                suffix += 1

            segmented_gradients.append(
                {
                    "container": container,
                    "event": event,
                    "grad_data": grad_data,
                    "leaf_path": leaf_path,
                    "segment_table": segment_table,
                    "event_key": event_name,
                    "index": gradient_index,
                }
            )

        # This group is now a structural parent. Its duration is derived by
        # gammaSTAR from the nested repeated readout branch.
        params.pop(f"{group_path}.duration", None)
        params.pop(f"{group_path}.tend", None)

        grad_entries: list[dict[str, str]] = []

        for item in segmented_gradients:
            old_container = str(item["container"])
            event = item["event"]
            grad_data = item["grad_data"]
            leaf_path = str(item["leaf_path"])
            index = int(item["index"])
            event_key = str(item["event_key"])

            # Remove the full-train sibling branch. Its information is retained
            # as a segment table on the new leaf under SingleReadout.
            for key in list(params):
                if (
                    key == old_container
                    or key.startswith(old_container + ".")
                ):
                    params.pop(key, None)

            for path in list(sequence_elements):
                if (
                    path == old_container
                    or path.startswith(old_container + ".")
                ):
                    sequence_elements.pop(path, None)

            sequence_elements[leaf_path] = GRAD_PULSE_BLUEPRINT

            params[f"{leaf_path}.tstart"] = _literal(0.0)
            params[f"{leaf_path}.duration"] = _expr(
                inputs={
                    "duration": f"{readout_loop_path}.duration",
                },
                script="return duration",
            )
            params[f"{leaf_path}.logical_axis"] = _literal(
                _gradient_logical_axis_for_orientation(event)
            )
            params[f"{leaf_path}.direction_fallback"] = _literal(
                grad_data["direction"]
            )
            params[f"{leaf_path}.direction"] = self._gradient_direction_parameter(
                gradient_event=event,
                fallback_direction=grad_data["direction"],
            )
            params[f"{leaf_path}.channel"] = _literal(
                grad_data.get("channel")
            )
            params[f"{leaf_path}.kind"] = _literal(
                grad_data.get("kind")
            )
            params[f"{leaf_path}.role"] = _literal(
                grad_data.get("role")
            )
            params[f"{leaf_path}.enabled"] = _literal(
                bool(grad_data.get("enabled", True))
            )
            # Preserve a generic outer-loop gradient variation when the full
            # train is lowered into one segment per inner readout iteration.
            # The segment table stores normalized waveform pieces and the
            # selected piece is scaled by the live varied area/amplitude. This
            # is not EPI-specific; it applies to any synchronized arbitrary
            # gradient train carrying a SeqStar loop variation.
            binding = self._variation_binding_for_event(event)
            attribute = str(
                binding.get("attribute") if isinstance(binding, Mapping) else ""
            ).strip().lower()



            # Normalize against the *declared variation value at counter zero*,
            # not against an implementation-specific gradient-data area.  Some
            # arbitrary-gradient adapters report area in a representation whose
            # units differ from the waveform samples.  Dividing samples by that
            # value can collapse a valid waveform to nearly zero.  The declared
            # strength is the generic relationship contract: at the exported
            # defaults, multiplying the normalized waveform by strength must
            # reproduce the original waveform exactly.
            declared_scale = None
            if isinstance(binding, Mapping):
                declared_scale = self._variation_numeric_default(
                    binding.get("strength")
                )

            can_scale_area = (
                attribute == "area"
                and declared_scale is not None
                and abs(declared_scale) > 1e-20
            )
            can_scale_amplitude = (
                attribute == "amplitude"
                and declared_scale is not None
                and abs(declared_scale) > 1e-20
            )

            if isinstance(binding, Mapping) and (
                can_scale_area or can_scale_amplitude
            ):
                scale0 = declared_scale
                normalized_segments = []
                for segment in item["segment_table"]:
                    normalized_segments.append(
                        {
                            "t": list(segment.get("t", [])),
                            "v": [
                                float(value) / float(scale0)
                                for value in segment.get("v", [])
                            ],
                        }
                    )

                params[f"{leaf_path}.normalized_segment_samples"] = _literal(
                    normalized_segments
                )
                params[f"{leaf_path}.default_segment_duration"] = _literal(
                    segment_duration
                )
                params[f"{leaf_path}.default_adc_delay"] = _literal(
                    default_adc_delay
                )
                params[f"{leaf_path}.default_adc_duration"] = _literal(
                    default_adc_duration
                )
                varying_value_path = f"{leaf_path}.{attribute}"
                self._write_single_loop_linear_value(
                    params=params,
                    path=varying_value_path,
                    binding=binding,
                    counter_path=f"{self._active_loop_path}.counter",
                )
                params[f"{leaf_path}.samples"] = _expr(
                    inputs={
                        "segments": (
                            f"{leaf_path}.normalized_segment_samples"
                        ),
                        "counter": f"{readout_loop_path}.counter",
                        "scale": varying_value_path,
                        "window_data": f"{readout_loop_path}.window_data",
                        "segment_duration": f"{readout_loop_path}.duration",
                        "default_segment_duration": (
                            f"{leaf_path}.default_segment_duration"
                        ),
                        "default_adc_delay": f"{leaf_path}.default_adc_delay",
                        "default_adc_duration": (
                            f"{leaf_path}.default_adc_duration"
                        ),
                    },
                    script=(
                        "local n = #segments\n"
                        "local idx = ((counter or 0) % n) + 1\n"
                        "local src = segments[idx] or segments[1]\n"
                        "local md = window_data.metadata or {}\n"
                        "local new_pre = window_data.event_delay "
                        "or window_data.readout_delay or md.event_delay "
                        "or default_adc_delay or 0\n"
                        "local new_mid = window_data.duration "
                        "or default_adc_duration or 0\n"
                        "local new_total = segment_duration "
                        "or default_segment_duration or 0\n"
                        "local old_pre = default_adc_delay or 0\n"
                        "local old_mid = default_adc_duration or 0\n"
                        "local old_total = default_segment_duration or 0\n"
                        "local old_mid_end = old_pre + old_mid\n"
                        "local new_mid_end = new_pre + new_mid\n"
                        "local function warp(t)\n"
                        "  if t <= old_pre then\n"
                        "    if old_pre > 0 then return t*new_pre/old_pre end\n"
                        "    return t\n"
                        "  end\n"
                        "  if t <= old_mid_end then\n"
                        "    if old_mid > 0 then "
                        "return new_pre + (t-old_pre)*new_mid/old_mid end\n"
                        "    return new_pre\n"
                        "  end\n"
                        "  local old_post = old_total-old_mid_end\n"
                        "  local new_post = math.max(0, new_total-new_mid_end)\n"
                        "  if old_post > 0 then "
                        "return new_mid_end + (t-old_mid_end)*new_post/old_post end\n"
                        "  return new_mid_end\n"
                        "end\n"
                        "local out = {t={}, v={}}\n"
                        "for i=1,#src.v do "
                        "out.t[i] = warp(src.t[i]); "
                        "out.v[i] = src.v[i] * scale end\n"
                        "return out"
                    ),
                )
                params[f"{leaf_path}.seqstar_variation_attribute"] = (
                    _literal(attribute)
                )
                params[f"{leaf_path}.seqstar_variation_counter"] = _literal(
                    f"{self._active_loop_path}.counter"
                )
            else:
                params[f"{leaf_path}.segment_samples"] = _literal(
                    item["segment_table"]
                )
                params[f"{leaf_path}.default_segment_duration"] = _literal(
                    segment_duration
                )
                params[f"{leaf_path}.default_adc_delay"] = _literal(
                    default_adc_delay
                )
                params[f"{leaf_path}.default_adc_duration"] = _literal(
                    default_adc_duration
                )
                params[f"{leaf_path}.samples"] = _expr(
                    inputs={
                        "segments": f"{leaf_path}.segment_samples",
                        "counter": f"{readout_loop_path}.counter",
                        "window_data": f"{readout_loop_path}.window_data",
                        "segment_duration": f"{readout_loop_path}.duration",
                        "default_segment_duration": (
                            f"{leaf_path}.default_segment_duration"
                        ),
                        "default_adc_delay": f"{leaf_path}.default_adc_delay",
                        "default_adc_duration": (
                            f"{leaf_path}.default_adc_duration"
                        ),
                    },
                    script=(
                        "local n = #segments\n"
                        "local idx = ((counter or 0) % n) + 1\n"
                        "local src = segments[idx] or segments[1]\n"
                        "local md = window_data.metadata or {}\n"
                        "local new_pre = window_data.event_delay "
                        "or window_data.readout_delay or md.event_delay "
                        "or default_adc_delay or 0\n"
                        "local new_mid = window_data.duration "
                        "or default_adc_duration or 0\n"
                        "local new_total = segment_duration "
                        "or default_segment_duration or 0\n"
                        "local old_pre = default_adc_delay or 0\n"
                        "local old_mid = default_adc_duration or 0\n"
                        "local old_total = default_segment_duration or 0\n"
                        "local old_mid_end = old_pre + old_mid\n"
                        "local new_mid_end = new_pre + new_mid\n"
                        "local function warp(t)\n"
                        "  if t <= old_pre then\n"
                        "    if old_pre > 0 then return t*new_pre/old_pre end\n"
                        "    return t\n"
                        "  end\n"
                        "  if t <= old_mid_end then\n"
                        "    if old_mid > 0 then "
                        "return new_pre + (t-old_pre)*new_mid/old_mid end\n"
                        "    return new_pre\n"
                        "  end\n"
                        "  local old_post = old_total-old_mid_end\n"
                        "  local new_post = math.max(0, new_total-new_mid_end)\n"
                        "  if old_post > 0 then "
                        "return new_mid_end + (t-old_mid_end)*new_post/old_post end\n"
                        "  return new_mid_end\n"
                        "end\n"
                        "local out = {t={}, v=src.v}\n"
                        "for i=1,#src.t do out.t[i] = warp(src.t[i]) end\n"
                        "return out"
                    ),
                )

            basic_repr_name = f"basic_repr_grad_{index}"
            basic_tstart_name = (
                f"basic_repr_grad_{index}_tstart_relative"
            )

            params[
                f"{single_atomic_path}.{basic_tstart_name}"
            ] = _expr(
                inputs={"t0": f"{leaf_path}.tstart"},
                script="return t0",
            )
            params[
                f"{single_atomic_path}.{basic_repr_name}"
            ] = _expr(
                inputs={
                    "enabled": f"{leaf_path}.enabled",
                    "samples": f"{leaf_path}.samples",
                    "direction": f"{leaf_path}.direction",
                },
                script=(
                    "return {\n"
                    "direction=direction,\n"
                    "samples=samples,\n"
                    "enabled=enabled\n"
                    "}"
                ),
            )

            grad_entries.append(
                {
                    "event_key": event_key,
                    "repr_path": (
                        f"{single_atomic_path}.{basic_repr_name}"
                    ),
                    "tstart_path": (
                        f"{single_atomic_path}.{basic_tstart_name}"
                    ),
                }
            )

            self._event_container_paths[id(event)] = (
                single_readout_path
            )
            self._event_container_leaves[id(event)] = leaf_path

        grad_inputs: dict[str, str] = {}
        grad_lines: list[str] = ["return {"]

        for index, entry in enumerate(grad_entries):
            repr_name = f"basic_repr_{index}"
            tstart_name = f"basic_tstart_{index}"
            grad_inputs[repr_name] = entry["repr_path"]
            grad_inputs[tstart_name] = entry["tstart_path"]
            grad_lines.append(
                f"['{entry['event_key']}']={{"
                f"tstart_relative={tstart_name}, "
                f"required_parameters={repr_name}"
                "},"
            )

        grad_lines.append("}")

        params[
            f"{single_atomic_path}.full_basic_repr_GradPulse"
        ] = _expr(
            inputs=grad_inputs,
            script="\n".join(grad_lines),
        )

        # The existing SingleReadout Atomic already contains the indexed ADC
        # representation. Its full_basic_repr expression references this
        # GradPulse field, so both event families now travel through the same
        # repeated cycle.
        params[
            "root.info.seqstar_synchronized_train_lowering"
        ] = _literal(True)
        params[
            "root.info.seqstar_synchronized_train_segments"
        ] = _literal(num_segments)

        return True

    def _add_stale_literal_tstart_validator(
        self,
        *,
        params: dict[str, Any],
        kernel_path: str,
        exported_events: list[Any],
    ) -> None:
        """Add debug warnings for downstream literal tstarts.

        This does not fail export. It records a machine-readable warning list in
        ``root.info`` so tests and the relationship dashboard can flag cases in
        which an object is downstream of a live/protocol-controlled timing path
        but its exported tstart is still a literal.
        """

        warnings: list[dict[str, Any]] = []
        live_seen = False
        for record in self._block_export_records:
            path = str(record["path"])
            tstart_key = f"{path}.tstart"
            tstart_param = params.get(tstart_key)
            is_literal = _is_literal_parameter(tstart_param)
            if live_seen and is_literal:
                warnings.append(
                    {
                        "kind": "stale_literal_block_tstart",
                        "path": path,
                        "node": record.get("node"),
                        "message": "Block is downstream of a live timing dependency but exports a literal tstart.",
                    }
                )
            if not is_literal:
                live_seen = True

        live_blocks = {
            idx
            for idx, path in self._block_export_paths.items()
            if not _is_literal_parameter(params.get(f"{path}.tstart"))
        }
        for event in exported_events:
            container_path = self._event_container_paths.get(id(event))
            if not container_path:
                continue
            block_index = _event_source_block_index_for_export(event)
            if block_index not in live_blocks:
                continue
            tstart_param = params.get(f"{container_path}.tstart")
            if _is_literal_parameter(tstart_param):
                warnings.append(
                    {
                        "kind": "stale_literal_event_tstart",
                        "path": container_path,
                        "event": str(getattr(event, "name", "event")),
                        "source_block_index": block_index,
                        "message": "Event container is in a live-timed block but exports a literal tstart.",
                    }
                )

        params["root.info.seqstar_stale_literal_tstart_warnings"] = _literal(warnings)
        params["root.info.seqstar_stale_literal_tstart_warning_count"] = _literal(len(warnings))
        params["root.tests.seqstar_no_stale_literal_tstarts"] = _expr(
            inputs={"warning_count": "root.info.seqstar_stale_literal_tstart_warning_count"},
            script="return warning_count == 0",
        )

    def _flatten_protocol_relationship_target(
        self,
        *,
        params: dict[str, Any],
        target_path: str,
        max_depth: int = 8,
    ) -> None:
        """Inline a derived protocol relationship into an exported target.

        A target such as ``root.<loop>.length`` may initially be a transparent
        alias of ``root.prot.some_derived_value``.  Some interactive runtimes do
        not invalidate that two-hop chain reliably.  This routine follows only
        declared parameter references and copies the referenced expression.  It
        does not inspect parameter names or infer sequence physics.
        """

        for _ in range(max_depth):
            current = params.get(target_path)
            if not isinstance(current, Mapping):
                return
            inputs = current.get("inputs")
            script = str(current.get("script") or "").strip()
            if not isinstance(inputs, Mapping) or len(inputs) != 1:
                return
            source_path = next(iter(inputs.values()))
            if not isinstance(source_path, str) or not source_path.startswith(
                "root.prot."
            ):
                return
            # Only flatten a transparent alias.
            if script not in {"return count", "return value", "return length", "return duration"}:
                return
            source = params.get(source_path)
            if not isinstance(source, Mapping):
                return
            source_inputs = source.get("inputs")
            if not isinstance(source_inputs, Mapping) or not source_inputs:
                return
            params[target_path] = {
                "inputs": dict(source_inputs),
                "script": str(source.get("script") or "return 1"),
            }

    def _rebind_outer_loop_length_to_protocol_expression(
        self,
        *,
        params: dict[str, Any],
        loop_path: str,
        repeat_count_ref: Any,
        repetitions: int,
    ) -> None:
        """Bind a loop directly to its declared protocol relationship.

        ``repeat_count`` may be supplied as a string, a ``ParameterRef`` or a
        compound expression.  The resolver needs the expression object, while
        gammaSTAR benefits from a direct dependency on the primitive protocol
        leaves.  This method therefore identifies the matching protocol source
        and copies its exported relationship onto ``<loop>.length``.

        The implementation is sequence-agnostic: it does not inspect EPI,
        ``n_y`` or ETL.  It only follows the relationship declared by the node.
        """

        length_path = f"{loop_path}.length"
        source_name = _protocol_source_name_from_value(
            repeat_count_ref,
            self._live_protocol_values,
        )

        if source_name is None:
            params.setdefault(length_path, _literal(int(repetitions)))
            return

        protocol_path = f"root.prot.{_protocol_key(source_name)}"
        protocol_parameter = params.get(protocol_path)

        if not isinstance(protocol_parameter, Mapping):
            params[length_path] = _expr(
                inputs={"count": protocol_path},
                script="return count",
            )
            return

        inputs = protocol_parameter.get("inputs")
        script = protocol_parameter.get("script")

        if isinstance(inputs, Mapping) and inputs:
            params[length_path] = {
                "inputs": dict(inputs),
                "script": str(script or "return 1"),
            }
        else:
            params[length_path] = _expr(
                inputs={"count": protocol_path},
                script="return count",
            )

        params["root.info.seqstar_motif_repetitions"] = _expr(
            inputs={"count": length_path},
            script="return count",
        )

    def _add_root_loop_and_kernel(
        self,
        *,
        params: dict[str, Any],
        loop_path: str,
        kernel_path: str,
        repetitions: int,
        repeat_count_ref: Any = None,
        tr: float,
        block_duration: float,
    ) -> None:
        """Add root-loop/kernel timing while preserving symbolic loop length."""

        params["root.tstart"] = _literal(0.0)
        params[f"{loop_path}.counter"] = _literal(0)

        # Prefer the exact repeat_count supplied to seq.set_node(...). This is
        # the authoritative relationship and avoids reconstructing semantics
        # from a loop token such as ``segment`` or ``ky``.
        loop_length_source = _protocol_source_name_from_value(
            repeat_count_ref,
            self._live_protocol_values,
        )
        if loop_length_source is None:
            loop_length_source = self._loop_length_protocol_source(
                loop_path=loop_path,
            )

        if loop_length_source is not None:
            params[f"{loop_path}.length"] = _expr(
                inputs={
                    "count": f"root.prot.{_protocol_key(loop_length_source)}"
                },
                script="return count",
            )
        else:
            params[f"{loop_path}.length"] = _literal(int(repetitions))

        params[f"{loop_path}.tstart"] = _expr(
            inputs={
                "counter": f"{loop_path}.counter",
                "period": "root.prot.TR",
            },
            script="return counter * period",
        )

        params[f"{kernel_path}.tstart"] = _literal(0.0)
        params[f"{kernel_path}.duration"] = _literal(block_duration)

    def _add_rf_event(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        loop_path: str,
        kernel_path: str,
        rf_name: str,
        rf_event: Any,
        event_index: int,
    ) -> str:
        """Add one RFRect/RFPulse branch and return the RF leaf path."""

        rf_rect_path = f"{kernel_path}.{rf_name}"
        atomic_path = f"{rf_rect_path}.atomic"
        rf_path = f"{rf_rect_path}.rf"

        sequence_elements[rf_rect_path] = RF_RECT_BLUEPRINT
        sequence_elements[atomic_path] = "Atomic"
        sequence_elements[rf_path] = RF_PULSE_BLUEPRINT

        active_duration = _event_active_duration(rf_event)
        delay = _event_delay(rf_event)
        flip_angle_rad = float(getattr(rf_event, "flip_angle"))
        flip_angle_deg = math.degrees(flip_angle_rad)

        phase = _get_float(
            rf_event,
            keys=("phase", "phase_offset", "rf_phase", "excitation_phase"),
            default=0.0,
        )
        frequency = _get_float(
            rf_event,
            keys=("frequency", "freq_offset", "frequency_offset", "rf_frequency_offset"),
            default=0.0,
        )
        enabled = bool(_get_any(rf_event, keys=("enabled",), default=True))
        asymmetry = float(_get_any(rf_event, keys=("asymmetry",), default=0.5))
        rf_type = _rf_type_string(rf_event)

        params[f"{rf_rect_path}.tstart"] = _literal(delay)
        params[f"{rf_rect_path}.duration"] = _literal(active_duration)
        params[f"{rf_rect_path}.flip_angle"] = _literal(flip_angle_deg)
        params[f"{rf_rect_path}.spoilphase"] = _literal(0.0)

        params[f"{rf_rect_path}.tcenter"] = _expr(
            inputs={
                "tstart": f"{rf_rect_path}.tstart",
                "duration": f"{rf_rect_path}.duration",
                "asymmetry": f"{rf_path}.asymmetry",
            },
            script="return tstart + asymmetry*duration",
        )

        params[f"{rf_rect_path}.tend"] = _expr(
            inputs={
                "tstart": f"{rf_rect_path}.tstart",
                "duration": f"{rf_rect_path}.duration",
            },
            script="return tstart + duration",
        )

        params[f"{atomic_path}.tstart_absolute"] = _expr(
            inputs={
                "t0": "root.tstart",
                "t1": f"{loop_path}.tstart",
                "t2": f"{kernel_path}.tstart",
                "t3": f"{rf_rect_path}.tstart",
            },
            script="return t0 + t1 + t2 + t3",
        )

        self._add_atomic_full_basic_repr(
            params=params,
            atomic_path=atomic_path,
        )

        params[f"{atomic_path}.full_basic_repr_RFPulse"] = _expr(
            inputs={
                "basic_repr_rf_tstart_relative": f"{atomic_path}.basic_repr_rf_tstart_relative",
                "basic_repr_rf": f"{atomic_path}.basic_repr_rf",
            },
            script=(
                "return {\n"
                f"['{rf_name}']="
                "{tstart_relative=basic_repr_rf_tstart_relative, "
                "required_parameters=basic_repr_rf}\n"
                "\n"
                "}"
            ),
        )

        params[f"{atomic_path}.basic_repr_rf_tstart_relative"] = _expr(
            inputs={"t0": f"{rf_path}.tstart"},
            script="return t0",
        )

        params[f"{atomic_path}.basic_repr_rf"] = _expr(
            inputs={
                "enabled": f"{rf_path}.enabled",
                "type": f"{rf_path}.type",
                "asymmetry": f"{rf_path}.asymmetry",
                "phase": f"{rf_path}.phase",
                "frequency": f"{rf_path}.frequency",
                "samples": f"{rf_path}.samples",
                "duration": f"{rf_path}.duration",
            },
            script=(
                "return {\n"
                "duration=duration,\n"
                "samples=samples,\n"
                "frequency=frequency,\n"
                "phase=phase,\n"
                "asymmetry=asymmetry,\n"
                "type=type,\n"
                "enabled=enabled\n"
                "}"
            ),
        )

        params[f"{rf_path}.duration"] = _expr(
            inputs={"duration": f"{rf_rect_path}.duration"},
            script="return duration",
        )
        params[f"{rf_path}.enabled"] = _literal(enabled)
        params[f"{rf_path}.tstart"] = _literal(0.0)
        params[f"{rf_path}.asymmetry"] = _literal(asymmetry)
        params[f"{rf_path}.frequency"] = _literal(frequency)
        params[f"{rf_path}.phase"] = _literal(phase)
        params[f"{rf_path}.type"] = _literal(rf_type)

        params[f"{rf_path}.samples"] = self._rf_samples_parameter(
            rf_event=rf_event,
            rf_rect_path=rf_rect_path,
            rf_path=rf_path,
        )

        params[f"{rf_path}.is_timing_increasing_and_rastered_and_same_am_length"] = _expr(
            inputs={
                "samples": f"{rf_path}.samples",
                "rf_set": "root.rf_settings",
            },
            script=(
                "if #samples.t > 0 then\n"
                "  if not ge(samples.t[1], 0) or "
                "modulo(samples.t[1], 0.5*rf_set.raster_samples) ~= 0 then\n"
                "    return false\n"
                "  end\n"
                "  for cha = 1, #samples.v do\n"
                "    if samples.v[cha].am[1] == nil then\n"
                "      return false\n"
                "    end\n"
                "  end\n"
                "  for i = 2, #samples.t do\n"
                "    if not (samples.t[i] > samples.t[i-1]) or "
                "modulo(samples.t[i], 0.5*rf_set.raster_samples) ~= 0 then\n"
                "      return false\n"
                "    end\n"
                "    for cha = 1, #samples.v do\n"
                "      if samples.v[cha].am[i] == nil then\n"
                "        return false\n"
                "      end\n"
                "    end\n"
                "  end\n"
                "end\n"
                "return true"
            ),
        )

        return rf_path

    def _add_gradient_event(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        loop_path: str,
        kernel_path: str,
        gradient_event: Any,
        event_index: int,
    ) -> str:
        """Add one standalone gradient event and return the GradPulse leaf path.

        This intentionally does not express RF/ADC/gradient relationships yet.
        Each gradient event is written as its own small container with an Atomic
        child and a GradPulse leaf. The plot payload follows the reference
        gammaSTAR contract:

            atomic.full_basic_repr_GradPulse
                ['grad'] = {tstart_relative=..., required_parameters={
                    samples={t={...}, v={...}},
                    direction={...},
                    enabled=true,
                }}
        """

        grad_data = self._gradient_waveform_data(gradient_event=gradient_event)
        grad_name = _gradient_container_name(gradient_event, event_index)
        grad_container_path = f"{kernel_path}.{grad_name}"
        atomic_path = f"{grad_container_path}.atomic"
        grad_path = f"{grad_container_path}.grad"

        blueprint = (
            TRAPEZOID_GRADIENT_BLUEPRINT
            if str(grad_data.get("kind", "")).lower() == "trap"
            else ARBITRARY_GRADIENT_BLUEPRINT
        )

        sequence_elements[grad_container_path] = blueprint
        sequence_elements[atomic_path] = "Atomic"
        sequence_elements[grad_path] = GRAD_PULSE_BLUEPRINT

        tstart = float(grad_data.get("tstart", 0.0))
        duration = float(grad_data.get("duration", 0.0))
        active_duration = float(grad_data.get("active_duration", duration))
        if duration <= 0:
            duration = tstart + active_duration

        tstart_parameter = self._event_tstart_parameter(
            target_event=gradient_event,
            target_container_path=grad_container_path,
            target_duration_path=f"{grad_container_path}.duration",
            fallback_tstart=tstart,
            local_tstart=0.0,
        )

        params[f"{grad_container_path}.tstart"] = tstart_parameter
        params[f"{grad_container_path}.duration"] = _literal(active_duration)
        params[f"{grad_container_path}.tcenter"] = _expr(
            inputs={
                "tstart": f"{grad_container_path}.tstart",
                "duration": f"{grad_container_path}.duration",
            },
            script="return tstart + 0.5*duration",
        )
        params[f"{grad_container_path}.tend"] = _expr(
            inputs={
                "tstart": f"{grad_container_path}.tstart",
                "duration": f"{grad_container_path}.duration",
            },
            script="return tstart + duration",
        )

        params[f"{atomic_path}.tstart_absolute"] = _expr(
            inputs={
                "t0": "root.tstart",
                "t1": f"{loop_path}.tstart",
                "t2": f"{kernel_path}.tstart",
                "t3": f"{grad_container_path}.tstart",
            },
            script="return t0 + t1 + t2 + t3",
        )

        self._add_atomic_full_basic_repr(params=params, atomic_path=atomic_path)

        params[f"{atomic_path}.full_basic_repr_GradPulse"] = _expr(
            inputs={
                "basic_repr_grad_tstart_relative": f"{atomic_path}.basic_repr_grad_tstart_relative",
                "basic_repr_grad": f"{atomic_path}.basic_repr_grad",
            },
            script=(
                "return {\n"
                "['grad']={tstart_relative=basic_repr_grad_tstart_relative, "
                "required_parameters=basic_repr_grad}\n"
                "\n"
                "}"
            ),
        )
        params[f"{atomic_path}.basic_repr_grad_tstart_relative"] = _expr(
            inputs={"t0": f"{grad_path}.tstart"},
            script="return t0",
        )
        params[f"{atomic_path}.basic_repr_grad"] = _expr(
            inputs={
                "enabled": f"{grad_path}.enabled",
                "samples": f"{grad_path}.samples",
                "direction": f"{grad_path}.direction",
            },
            script=(
                "return {\n"
                "direction=direction,\n"
                "samples=samples,\n"
                "enabled=enabled\n"
                "}"
            ),
        )

        variant_table = _gradient_variant_table(gradient_event)

        params[f"{grad_path}.tstart"] = _literal(0.0)
        params[f"{grad_path}.logical_axis"] = _literal(
            _gradient_logical_axis_for_orientation(gradient_event)
        )
        params[f"{grad_path}.direction_fallback"] = _literal(grad_data["direction"])
        params[f"{grad_path}.direction"] = self._gradient_direction_parameter(
            gradient_event=gradient_event,
            fallback_direction=grad_data["direction"],
        )
        params[f"{grad_path}.channel"] = _literal(grad_data.get("channel"))
        params[f"{grad_path}.kind"] = _literal(grad_data.get("kind"))
        params[f"{grad_path}.role"] = _literal(grad_data.get("role"))
        params[f"{grad_path}.duration"] = _literal(active_duration)

        if variant_table is None:
            params[f"{grad_path}.enabled"] = _literal(bool(grad_data.get("enabled", True)))
            params[f"{grad_path}.samples"] = _literal(grad_data["samples"])
            params[f"{grad_path}.area"] = _literal(grad_data.get("area"))
            params[f"{grad_path}.max_abs_amplitude"] = _literal(grad_data.get("max_abs_amplitude", 0.0))
        else:
            params[f"{grad_path}.variant_samples"] = _literal(variant_table["samples"])
            params[f"{grad_path}.variant_area"] = _literal(variant_table["area"])
            params[f"{grad_path}.variant_max_abs_amplitude"] = _literal(variant_table["max_abs_amplitude"])
            params[f"{grad_path}.variant_enabled"] = _literal(variant_table["enabled"])

            counter_path = str(
                variant_table.get("counter_path")
                or self._active_loop_path + ".counter"
            )
            if counter_path == "root.seqstar_loop.counter":
                counter_path = self._active_loop_path + ".counter"

            params[f"{grad_path}.samples"] = _expr(
                inputs={
                    "table": f"{grad_path}.variant_samples",
                    "counter": counter_path,
                },
                script="local idx = (counter or 0) + 1\nreturn table[idx] or table[1]",
            )
            params[f"{grad_path}.area"] = _expr(
                inputs={"table": f"{grad_path}.variant_area", "counter": counter_path},
                script="local idx = (counter or 0) + 1\nreturn table[idx] or table[1]",
            )
            params[f"{grad_path}.max_abs_amplitude"] = _expr(
                inputs={"table": f"{grad_path}.variant_max_abs_amplitude", "counter": counter_path},
                script="local idx = (counter or 0) + 1\nreturn table[idx] or table[1]",
            )
            params[f"{grad_path}.enabled"] = _expr(
                inputs={"table": f"{grad_path}.variant_enabled", "counter": counter_path},
                script="local idx = (counter or 0) + 1\nlocal v = table[idx]\nif v == nil then return true end\nreturn v",
            )

        params[f"{grad_path}.is_timing_increasing_and_rastered"] = _expr(
            inputs={
                "samples": f"{grad_path}.samples",
                "grad_set": "root.gradient_settings",
            },
            script=(
                "if samples == nil or samples.t == nil or samples.v == nil then\n"
                "  return false\n"
                "end\n"
                "if #samples.t ~= #samples.v then\n"
                "  return false\n"
                "end\n"
                "for i = 1, #samples.t do\n"
                "  if samples.t[i] < 0 then return false end\n"
                "  if modulo(samples.t[i], 0.5*grad_set.raster_samples) ~= 0 then\n"
                "    return false\n"
                "  end\n"
                "  if i > 1 and not (samples.t[i] > samples.t[i-1]) then\n"
                "    return false\n"
                "  end\n"
                "end\n"
                "return true"
            ),
        )

        return grad_path

    def _gradient_waveform_data(self, *, gradient_event: Any) -> dict[str, Any]:
        """Return normalized gammaSTAR gradient waveform data."""

        if hasattr(gradient_event, "to_gammastar_waveform"):
            data = gradient_event.to_gammastar_waveform()
        else:
            data = _fallback_gradient_waveform_data(gradient_event)
        return _normalize_gradient_data(data, gradient_event=gradient_event)

    def _add_adc_event(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        loop_path: str,
        kernel_path: str,
        adc_event: Any,
        event_index: int,
    ) -> str:
        """Add one ADC branch and return the ADC leaf path.

        Single-window ADCs use a simple generic ADC branch.

        Multi-window ADCs use a real gammaSTAR Loop over windows so the website
        plot engine sees multiple ADC gates.
        """

        adc_data = self._adc_windows_data(adc_event=adc_event)
        windows = adc_data["windows"]

        if len(windows) <= 1:
            adc_container_name = _adc_container_name(adc_event, event_index)
            return self._add_single_window_adc_event(
                params=params,
                sequence_elements=sequence_elements,
                loop_path=loop_path,
                kernel_path=kernel_path,
                adc_container_name=adc_container_name,
                adc_event=adc_event,
                adc_data=adc_data,
                event_index=event_index,
            )

        readout_name = "readout" if event_index == 0 else f"readout{event_index + 1}"
        return self._add_multi_window_adc_event(
            params=params,
            sequence_elements=sequence_elements,
            loop_path=loop_path,
            kernel_path=kernel_path,
            readout_name=readout_name,
            adc_event=adc_event,
            adc_data=adc_data,
            event_index=event_index,
        )

    def _add_single_window_adc_event(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        loop_path: str,
        kernel_path: str,
        adc_container_name: str,
        adc_event: Any,
        adc_data: dict[str, Any],
        event_index: int,
    ) -> str:
        """Add one simple single-window ADC event."""

        adc_container_path = f"{kernel_path}.{adc_container_name}"
        atomic_path = f"{adc_container_path}.atomic"
        adc_path = f"{adc_container_path}.adc"
        header_path = f"{adc_path}.header"

        sequence_elements[adc_container_path] = ADC_CONTAINER_BLUEPRINT
        sequence_elements[atomic_path] = "Atomic"
        sequence_elements[adc_path] = ADC_BLUEPRINT
        sequence_elements[header_path] = ADC_HEADER_BLUEPRINT

        windows = adc_data["windows"]
        first_window = windows[0]

        # The ADC container is positioned at the resolved absolute window start.
        # The ADC leaf itself stays local to the container with tstart=0.
        # This keeps gammaSTAR plotting from starting ADC at kernel t=0 and
        # preserves generic relationship-derived timing such as TE.
        first_tstart = float(first_window.get("tstart", first_window.get("delay", 0.0)))
        duration = float(first_window.get("duration", 0.0))
        if duration <= 0:
            duration = float(adc_data.get("duration", 0.0))
        if duration <= 0:
            duration = max(
                float(window.get("duration", 0.0))
                for window in windows
            )

        params[f"{adc_container_path}.duration"] = _literal(duration)

        tstart_parameter = self._event_tstart_parameter(
            target_event=adc_event,
            target_container_path=adc_container_path,
            target_duration_path=f"{adc_container_path}.duration",
            fallback_tstart=first_tstart,
            # Relationship-derived timing already solves the exported container
            # start. Do not add an ADC-window absolute/local tstart here; that
            # would double-count timing for any sequence whose ADC windows have
            # already been materialized in kernel/block coordinates.
            local_tstart=0.0,
        )

        # Last-mile safety for gammaSTAR Protocol Control:
        # if the relationship object could not be mapped and the result is still
        # literal, preserve live protocol timing for ADC/readout events through
        # a generic timing-parameter expression such as root.prot.TE + delta.
        if _is_literal_parameter(tstart_parameter):
            forced_tstart = self._force_adc_protocol_relative_tstart_parameter(
                target_event=adc_event,
                fallback_tstart=first_tstart,
                local_tstart_expr=None,
            )
            if forced_tstart is not None:
                tstart_parameter = forced_tstart

        params[f"{adc_container_path}.tstart"] = tstart_parameter
        params[f"{adc_container_path}.tcenter"] = _expr(
            inputs={
                "tstart": f"{adc_container_path}.tstart",
                "duration": f"{adc_container_path}.duration",
            },
            script="return tstart + 0.5*duration",
        )
        params[f"{adc_container_path}.tend"] = _expr(
            inputs={
                "tstart": f"{adc_container_path}.tstart",
                "duration": f"{adc_container_path}.duration",
            },
            script="return tstart + duration",
        )

        params[f"{atomic_path}.tstart_absolute"] = _expr(
            inputs={
                "t0": "root.tstart",
                "t1": f"{loop_path}.tstart",
                "t2": f"{kernel_path}.tstart",
                "t3": f"{adc_container_path}.tstart",
            },
            script="return t0 + t1 + t2 + t3",
        )

        self._add_atomic_full_basic_repr(params=params, atomic_path=atomic_path)

        self._add_adc_leaf_parameters(
            params=params,
            atomic_path=atomic_path,
            adc_path=adc_path,
            header_path=header_path,
            adc_event=adc_event,
            adc_data=adc_data,
            window_expr=None,
            container_tstart_path=f"{adc_container_path}.tstart",
            adc_repr_key="adc",
        )

        return adc_path

    def _add_multi_window_adc_event(
        self,
        *,
        params: dict[str, Any],
        sequence_elements: dict[str, str],
        loop_path: str,
        kernel_path: str,
        readout_name: str,
        adc_event: Any,
        adc_data: dict[str, Any],
        event_index: int,
    ) -> str:
        """Add a multi-window ADC train as a gammaSTAR Loop over readout leaves.

        The reference gammaSTAR EPI sequence does *not* place a Loop inside a
        SingleReadout element. Instead, the Loop is the parent and each loop
        iteration contains one SingleReadout leaf with an Atomic child and ADC
        child. This matters for plotting: if the SingleReadout blueprint is used
        as the full train-level parent, gammaSTAR shades that parent duration as
        one continuous ADC/readout block.

        Therefore the generic writer emits:

            kernel.readout                 Loop over ADC windows
            kernel.readout.single_readout  SingleReadout blueprint
            kernel.readout.single_readout.atomic
            kernel.readout.single_readout.adc

        The full train is still preserved as a literal windows table on the Loop
        node, but the plotted object is one SingleReadout/ADC per iteration.
        """

        readout_loop_path = f"{kernel_path}.{readout_name}"
        single_readout_path = f"{readout_loop_path}.single_readout"
        atomic_path = f"{single_readout_path}.atomic"
        adc_path = f"{single_readout_path}.adc"
        header_path = f"{adc_path}.header"

        sequence_elements[readout_loop_path] = "Loop"
        sequence_elements[single_readout_path] = SINGLE_READOUT_BLUEPRINT
        sequence_elements[atomic_path] = "Atomic"
        sequence_elements[adc_path] = ADC_BLUEPRINT
        sequence_elements[header_path] = ADC_HEADER_BLUEPRINT

        windows = adc_data["windows"]
        duration = float(adc_data.get("duration", 0.0))
        if duration <= 0:
            duration = max(
                float(window.get("tstart", window.get("delay", 0.0)))
                + float(window.get("duration", 0.0))
                for window in windows
            )

        # Store full-train metadata on the loop node. Do not make a
        # SingleReadout blueprint span this duration, because gammaSTAR plots
        # SingleReadout as an active readout/ADC object.
        params[f"{readout_loop_path}.windows"] = _literal(windows)

        # Prefer live protocol controls for a regular ADC train. This keeps the
        # inner readout loop responsive to protocol edits such as ETL and
        # readout duration. The fallback remains the concrete exported table.
        live_etl_path = None
        for key in ("echo_train_length", "etl", "turbo_factor"):
            candidate = f"root.prot.{_protocol_key(key)}"
            if candidate in params or key in self._live_protocol_values:
                live_etl_path = candidate
                break

        live_readout_duration_path = None
        for key in ("readout_duration", "adc_duration"):
            candidate = f"root.prot.{_protocol_key(key)}"
            if candidate in params or key in self._live_protocol_values:
                live_readout_duration_path = candidate
                break

        live_echo_spacing_path = None
        for key in ("echo_spacing", "readout_spacing"):
            candidate = f"root.prot.{_protocol_key(key)}"
            if candidate in params or key in self._live_protocol_values:
                live_echo_spacing_path = candidate
                break

        live_samples_path = None
        for key in ("adc_samples", "n_x", "num_samples"):
            candidate = f"root.prot.{_protocol_key(key)}"
            if candidate in params or key in self._live_protocol_values:
                live_samples_path = candidate
                break

        params[f"{readout_loop_path}.num_windows"] = (
            _expr(inputs={"count": live_etl_path}, script="return count")
            if live_etl_path is not None
            else _literal(len(windows))
        )
        params[f"{readout_loop_path}.train_duration"] = _literal(duration)
        params[f"{readout_loop_path}.mode"] = _literal(str(adc_data.get("mode", "windows")))
        params[f"{readout_loop_path}.trajectory"] = _literal(adc_data.get("trajectory"))
        params[f"{readout_loop_path}.dead_time_policy"] = _literal(
            _adc_train_metadata_value(
                adc_event,
                "dead_time_policy",
                default="train_level_once",
            )
        )
        params[f"{readout_loop_path}.logical_node"] = _literal(
            _adc_train_metadata_value(
                adc_event,
                "logical_readout_node",
                default=None,
            )
        )
        params[f"{single_readout_path}.logical_node"] = _literal(
            _adc_train_metadata_value(
                adc_event,
                "logical_single_readout_node",
                default=None,
            )
        )

        params[f"{readout_loop_path}.counter"] = _literal(0)
        params[f"{readout_loop_path}.length"] = _expr(
            inputs={"num_windows": f"{readout_loop_path}.num_windows"},
            script="return num_windows",
        )

        first_window = windows[0]
        first_metadata = _adc_window_value(first_window, "metadata", {})
        if not isinstance(first_metadata, Mapping):
            first_metadata = {}
        first_tstart = float(
            _adc_window_value(
                first_window,
                "tstart",
                _adc_window_value(first_window, "delay", 0.0),
            )
        )
        first_event_delay = float(
            _adc_window_value(
                first_window,
                "event_delay",
                _adc_window_value(
                    first_window,
                    "readout_delay",
                    first_metadata.get("event_delay", 0.0),
                ),
            )
        )
        first_duration = float(
            _adc_window_value(first_window, "duration", 0.0)
        )



        if (
            live_readout_duration_path is not None
            and live_echo_spacing_path is not None
            and live_samples_path is not None
        ):
            params[f"{readout_loop_path}.window_data"] = _expr(
                inputs={
                    "counter": f"{readout_loop_path}.counter",
                    "readout_duration": live_readout_duration_path,
                    "echo_spacing": f"{readout_loop_path}.duration",
                    "num_samples": live_samples_path,
                    "first_tstart": f"{readout_loop_path}.first_tstart",
                    "event_delay": f"{readout_loop_path}.event_delay",
                },
                script=(
                    "local duration = readout_duration or 0\n"
                    "local samples = num_samples or 1\n"
                    "return {tstart=first_tstart + counter*echo_spacing, "
                    "delay=first_tstart + counter*echo_spacing, "
                    "duration=duration, num_samples=samples, "
                    "number_of_samples=samples, sample_time=duration/samples, "
                    "dwell=duration/samples, event_delay=event_delay, "
                    "metadata={event_delay=event_delay}}"
                ),
            )
            params[f"{readout_loop_path}.first_tstart"] = _literal(first_tstart)
            params[f"{readout_loop_path}.event_delay"] = _literal(first_event_delay)
        else:
            params[f"{readout_loop_path}.window_data"] = _expr(
                inputs={
                    "windows": f"{readout_loop_path}.windows",
                    "counter": f"{readout_loop_path}.counter",
                },
                script=(
                    "local idx = counter + 1\n"
                    "return windows[idx] or windows[#windows]\n"
                ),
            )

        # The repeated synchronized interval spans one complete gradient
        # segment, not only the ADC gate. Prefer a declared live segment period
        # (for example a protocol echo/readout spacing); otherwise retain the
        # concrete spacing inferred from the exported window table.
        params[f"{readout_loop_path}.default_iteration_duration"] = _literal(
            (
                float(
                    _adc_window_value(
                        windows[1],
                        "tstart",
                        _adc_window_value(windows[1], "delay", 0.0),
                    )
                )
                - float(
                    _adc_window_value(
                        windows[0],
                        "tstart",
                        _adc_window_value(windows[0], "delay", 0.0),
                    )
                )
            )
            if len(windows) > 1
            else first_event_delay + first_duration
        )
        if live_echo_spacing_path is not None:
            params[f"{readout_loop_path}.duration"] = _expr(
                inputs={"duration": live_echo_spacing_path},
                script="return duration",
            )
            # Inline a declared derived spacing relationship into the actual
            # repeated interval.  This is generic relationship lowering: the
            # writer does not inspect parameter names or sequence type.
            self._flatten_protocol_relationship_target(
                params=params,
                target_path=f"{readout_loop_path}.duration",
            )
        else:
            params[f"{readout_loop_path}.duration"] = _expr(
                inputs={
                    "duration": (
                        f"{readout_loop_path}.default_iteration_duration"
                    )
                },
                script="return duration",
            )

        # Keep the full train duration live when either the repeated interval
        # or loop length changes.
        params[f"{readout_loop_path}.train_duration"] = _expr(
            inputs={
                "count": f"{readout_loop_path}.length",
                "iteration": f"{readout_loop_path}.duration",
            },
            script="return count * iteration",
        )
        # For a multi-window ADC train, ``window_data.tstart`` is the ADC gate
        # start, not the start of the synchronized readout gradient segment.
        # The SingleReadout loop must begin at the gradient-segment start, and
        # the ADC child must carry the intra-segment delay. Otherwise gammaSTAR
        # plots the ADC gate from the beginning of the ramp and EPI-03 fails.
        params[f"{readout_loop_path}.tstart"] = _expr(
            inputs={"window_data": f"{readout_loop_path}.window_data"},
            script=(
                "local wstart = window_data.tstart or window_data.delay or 0\n"
                "local md = window_data.metadata or {}\n"
                "local event_delay = window_data.event_delay "
                "or window_data.readout_delay or md.event_delay or 0\n"
                "return wstart - event_delay\n"
            ),
        )

        # The SingleReadout container spans the complete synchronized
        # gradient/ADC interval. The ADC child retains its own intra-segment
        # delay and gate duration.
        params[f"{single_readout_path}.tstart"] = _literal(0.0)
        params[f"{single_readout_path}.duration"] = _expr(
            inputs={"duration": f"{readout_loop_path}.duration"},
            script="return duration",
        )
        # Generic synchronized-window timing diagnostics. These expose the
        # remaining non-ADC interval and make invalid protocol edits visible
        # without assuming a particular sequence type.
        params[f"{readout_loop_path}.timing_margin"] = _expr(
            inputs={
                "iteration": f"{readout_loop_path}.duration",
                "window_data": f"{readout_loop_path}.window_data",
            },
            script=(
                "local md = window_data.metadata or {}\n"
                "local delay = window_data.event_delay "
                "or window_data.readout_delay or md.event_delay or 0\n"
                "local gate = window_data.duration or 0\n"
                "return iteration - delay - gate"
            ),
        )
        params[f"{readout_loop_path}.timing_valid"] = _expr(
            inputs={"margin": f"{readout_loop_path}.timing_margin"},
            script="return margin >= -1e-12",
        )
        params[f"{readout_loop_path}.adc_flat_start"] = _expr(
            inputs={"window_data": f"{readout_loop_path}.window_data"},
            script=(
                "local md = window_data.metadata or {}\n"
                "return window_data.event_delay "
                "or window_data.readout_delay or md.event_delay or 0"
            ),
        )
        params[f"{readout_loop_path}.adc_flat_end"] = _expr(
            inputs={
                "start": f"{readout_loop_path}.adc_flat_start",
                "window_data": f"{readout_loop_path}.window_data",
            },
            script="return start + (window_data.duration or 0)",
        )
        params[f"{single_readout_path}.tcenter"] = _expr(
            inputs={
                "tstart": f"{single_readout_path}.tstart",
                "duration": f"{single_readout_path}.duration",
            },
            script="return tstart + 0.5*duration",
        )
        params[f"{single_readout_path}.tend"] = _expr(
            inputs={
                "tstart": f"{single_readout_path}.tstart",
                "duration": f"{single_readout_path}.duration",
            },
            script="return tstart + duration",
        )

        params[f"{atomic_path}.tstart_absolute"] = _expr(
            inputs={
                "t0": "root.tstart",
                "t1": f"{loop_path}.tstart",
                "t2": f"{kernel_path}.tstart",
                "t3": f"{readout_loop_path}.tstart",
                "t4": f"{single_readout_path}.tstart",
            },
            script="return t0 + t1 + t2 + t3 + t4",
        )

        self._add_atomic_full_basic_repr(params=params, atomic_path=atomic_path)

        self._add_adc_leaf_parameters(
            params=params,
            atomic_path=atomic_path,
            adc_path=adc_path,
            header_path=header_path,
            adc_event=adc_event,
            adc_data=adc_data,
            window_expr=f"{readout_loop_path}.window_data",
            container_tstart_path=f"{single_readout_path}.tstart",
            adc_repr_key="adc",
        )

        return adc_path

    def _add_adc_leaf_parameters(
        self,
        *,
        params: dict[str, Any],
        atomic_path: str,
        adc_path: str,
        header_path: str,
        adc_event: Any,
        adc_data: dict[str, Any],
        window_expr: str | None,
        container_tstart_path: str,
        adc_repr_key: str,
    ) -> None:
        """Add one ADC leaf.

        If window_expr is None, values are literal from the first window.
        If window_expr is provided, values are expressions indexed by a loop.
        """

        windows = adc_data["windows"]
        first_window = windows[0]
        first_duration = float(first_window.get("duration", 0.0))
        first_samples = int(
            first_window.get("num_samples", first_window.get("number_of_samples", 1))
        )
        first_sample_time = float(
            first_window.get(
                "sample_time",
                first_window.get("dwell", first_duration / first_samples),
            )
        )

        phase = float(_get_any(adc_event, keys=("phase", "phase_offset"), default=0.0))
        frequency = float(
            _get_any(
                adc_event,
                keys=("frequency", "freq_offset", "frequency_offset"),
                default=0.0,
            )
        )
        enabled = bool(_get_any(adc_event, keys=("enabled",), default=True))

        params[f"{atomic_path}.full_basic_repr_ADC"] = _expr(
            inputs={
                "basic_repr_adc_tstart_relative": f"{atomic_path}.basic_repr_adc_tstart_relative",
                "basic_repr_adc": f"{atomic_path}.basic_repr_adc",
            },
            script=(
                "return {\n"
                f"['{adc_repr_key}']="
                "{tstart_relative=basic_repr_adc_tstart_relative, "
                "required_parameters=basic_repr_adc}\n"
                "\n"
                "}"
            ),
        )

        params[f"{atomic_path}.basic_repr_adc_tstart_relative"] = _expr(
            inputs={"t0": f"{adc_path}.tstart"},
            script="return t0",
        )

        params[f"{atomic_path}.basic_repr_adc"] = _expr(
            inputs={
                "enabled": f"{adc_path}.enabled",
                "phase": f"{adc_path}.phase",
                "frequency": f"{adc_path}.frequency",
                "header_data": f"{adc_path}.header_data",
                "mode": f"{adc_path}.mode",
                "trajectory": f"{adc_path}.trajectory",
            },
            script=(
                "return {\n"
                "header_data=header_data,\n"
                "frequency=frequency,\n"
                "phase=phase,\n"
                "enabled=enabled,\n"
                "mode=mode,\n"
                "trajectory=trajectory\n"
                "}"
            ),
        )

        params[f"{adc_path}.set_enabled"] = _literal(enabled)
        params[f"{adc_path}.enabled_single"] = _expr(
            inputs={"set_enabled": f"{adc_path}.set_enabled"},
            script="return set_enabled",
        )
        params[f"{adc_path}.enabled"] = _expr(
            inputs={
                f"{_safe_input_name(adc_path)}_enabled_single": f"{adc_path}.enabled_single"
            },
            script=f"return {_safe_input_name(adc_path)}_enabled_single",
        )

        if window_expr is None:
            params[f"{adc_path}.tstart"] = _literal(0.0)
            params[f"{adc_path}.duration"] = _literal(first_duration)
            params[f"{adc_path}.number_of_samples"] = _literal(first_samples)
            params[f"{adc_path}.sample_time"] = _literal(first_sample_time)
            params[f"{adc_path}.window_data"] = _literal(first_window)
        else:
            params[f"{adc_path}.tstart"] = _expr(
                inputs={"window_data": window_expr},
                script=(
                    "local md = window_data.metadata or {}\n"
                    "return window_data.event_delay "
                    "or window_data.readout_delay or md.event_delay or 0\n"
                ),
            )
            params[f"{adc_path}.duration"] = _expr(
                inputs={"window_data": window_expr},
                script="return window_data.duration or 0",
            )
            params[f"{adc_path}.number_of_samples"] = _expr(
                inputs={"window_data": window_expr},
                script="return window_data.num_samples or window_data.number_of_samples or 0",
            )
            params[f"{adc_path}.sample_time"] = _expr(
                inputs={"window_data": window_expr},
                script="return window_data.sample_time or window_data.dwell or 0",
            )
            params[f"{adc_path}.window_data"] = _expr(
                inputs={"window_data": window_expr},
                script="return window_data",
            )

        params[f"{adc_path}.tcenter"] = _expr(
            inputs={
                "tstart": f"{adc_path}.tstart",
                "duration": f"{adc_path}.duration",
            },
            script="return tstart + 0.5 * duration",
        )
        params[f"{adc_path}.tend"] = _expr(
            inputs={
                "tstart": f"{adc_path}.tstart",
                "duration": f"{adc_path}.duration",
            },
            script="return tstart + duration",
        )

        params[f"{adc_path}.phase"] = _literal(phase)
        params[f"{adc_path}.frequency"] = _literal(frequency)
        params[f"{adc_path}.mode"] = _literal(str(adc_data.get("mode", "single")))
        # gammaSTAR's serializer expects a concrete value. An empty table is
        # the appropriate representation for acquisitions with no k-space
        # trajectory, rather than a Lua nil that disappears during table export.
        params[f"{adc_path}.trajectory"] = _literal(
            adc_data.get("trajectory") if adc_data.get("trajectory") is not None else {}
        )
        params[f"{adc_path}.sample_time_us"] = _expr(
            inputs={"sample_time": f"{adc_path}.sample_time"},
            script="return sample_time * 1e6",
        )

        params[f"{adc_path}.header_data"] = _expr(
            inputs={
                "number_of_samples": f"{header_path}.number_of_samples",
                "sample_time_us": f"{header_path}.sample_time_us",
                "center_sample": f"{header_path}.center_sample",
                "idx_kspace_encode_step_1": f"{header_path}.idx_kspace_encode_step_1",
                "idx_kspace_encode_step_2": f"{header_path}.idx_kspace_encode_step_2",
                "idx_slice": f"{header_path}.idx_slice",
                "read_dir": f"{header_path}.read_dir",
                "phase_dir": f"{header_path}.phase_dir",
                "slice_dir": f"{header_path}.slice_dir",
                "position": f"{header_path}.position",
                "offcenter": f"{header_path}.offcenter",
                "matrix_size": f"{header_path}.matrix_size",
                "field_of_view": f"{header_path}.field_of_view",
            },
            script=(
                "return {\n"
                "number_of_samples=number_of_samples,\n"
                "sample_time_us=sample_time_us,\n"
                "center_sample=center_sample,\n"
                "idx_kspace_encode_step_1=idx_kspace_encode_step_1,\n"
                "idx_kspace_encode_step_2=idx_kspace_encode_step_2,\n"
                "idx_slice=idx_slice,\n"
                "read_dir=read_dir,\n"
                "phase_dir=phase_dir,\n"
                "slice_dir=slice_dir,\n"
                "position=position,\n"
                "offcenter=offcenter,\n"
                "matrix_size=matrix_size,\n"
                "field_of_view=field_of_view\n"
                "}"
            ),
        )

        params[f"{header_path}.number_of_samples"] = _expr(
            inputs={"number_of_samples": f"{adc_path}.number_of_samples"},
            script="return number_of_samples",
        )
        params[f"{header_path}.sample_time_us"] = _expr(
            inputs={"sample_time_us": f"{adc_path}.sample_time_us"},
            script="return sample_time_us",
        )

        # Header values are intentionally live relationships. They are the
        # bridge from editable protocol controls and loop counters to scanner
        # acquisition metadata. This keeps GRE simple while giving EPI a stable
        # place to add echo, segment, and polarity metadata later.
        params[f"{header_path}.center_sample"] = _expr(
            inputs={"number_of_samples": f"{adc_path}.number_of_samples"},
            script="return math.floor(0.5 * (number_of_samples - 1))",
        )

        phase_counter_path = self._phase_encode_counter_path(
            params=params,
            adc_event=adc_event,
        )

        if window_expr is None:
            window_line_index = int(first_window.get("line_index") or 0)
            if phase_counter_path is not None:
                params[f"{header_path}.idx_kspace_encode_step_1"] = _expr(
                    inputs={"counter": phase_counter_path},
                    script="return counter or 0",
                )
            else:
                params[f"{header_path}.idx_kspace_encode_step_1"] = _literal(
                    window_line_index
                )
            params[f"{header_path}.idx_kspace_encode_step_2"] = _literal(
                int(first_window.get("partition_index") or 0)
            )
            params[f"{header_path}.idx_slice"] = self._slice_index_parameter(
                default=int(first_window.get("slice_index") or 0)
            )
        else:
            if phase_counter_path is not None:
                params[f"{header_path}.idx_kspace_encode_step_1"] = _expr(
                    inputs={
                        "window_data": window_expr,
                        "counter": phase_counter_path,
                    },
                    script=(
                        "if window_data.line_index ~= nil then\n"
                        "  return window_data.line_index\n"
                        "end\n"
                        "if window_data.phase_encode_index ~= nil then\n"
                        "  return window_data.phase_encode_index\n"
                        "end\n"
                        "return counter or 0"
                    ),
                )
            else:
                params[f"{header_path}.idx_kspace_encode_step_1"] = _expr(
                    inputs={"window_data": window_expr},
                    script=(
                        "return window_data.line_index "
                        "or window_data.phase_encode_index or 0"
                    ),
                )
            params[f"{header_path}.idx_kspace_encode_step_2"] = _expr(
                inputs={"window_data": window_expr},
                script="return window_data.partition_index or 0",
            )
            params[f"{header_path}.idx_slice"] = _expr(
                inputs={
                    "window_data": window_expr,
                    "slice_default": self._slice_index_default_path(params),
                },
                script=(
                    "if window_data.slice_index ~= nil then\n"
                    "  return window_data.slice_index\n"
                    "end\n"
                    "return slice_default or 0"
                ),
            )

        params[f"{header_path}.read_dir"] = _expr(
            inputs={"read_dir": "root.info.encoding_read_dir"},
            script="return read_dir",
        )
        params[f"{header_path}.phase_dir"] = _expr(
            inputs={"phase_dir": "root.info.encoding_phase_dir"},
            script="return phase_dir",
        )
        params[f"{header_path}.slice_dir"] = _expr(
            inputs={"slice_dir": "root.info.encoding_slice_dir"},
            script="return slice_dir",
        )
        params[f"{header_path}.position"] = _expr(
            inputs={"position": "root.info.encoding_position"},
            script="return position",
        )
        # gammaSTAR's ADC Header blueprint expects an `offcenter` parameter.
        # SeqStar stores the same semantic quantity as the acquisition
        # position/encoding position. Export both names so the backend header
        # is complete while keeping the public geometry model unchanged.
        params[f"{header_path}.offcenter"] = _expr(
            inputs={"position": f"{header_path}.position"},
            script="return position",
        )
        # Reuse the same global geometry objects consumed by gammaSTAR's
        # exporter. This prevents an ADC header from drifting away from
        # root.mat_size/root.acq_size or from referencing absent protocol leaves.
        params[f"{header_path}.matrix_size"] = _expr(
            inputs={"mat_size": "root.mat_size"},
            script="return {mat_size[1], mat_size[2], mat_size[3]}",
        )
        params[f"{header_path}.field_of_view"] = _expr(
            inputs={"fov": "root.fov"},
            script="return {fov[1], fov[2], fov[3]}",
        )

        params[f"{adc_path}.is_duration_nonnegative_and_rastered_and_samples_pos"] = _expr(
            inputs={
                "samples": f"{adc_path}.number_of_samples",
                "duration": f"{adc_path}.duration",
                "adc_set": "root.adc_settings",
            },
            script=(
                "return ge(duration, 0) and "
                "modulo(duration, adc_set.raster_samples) == 0 and samples > 0"
            ),
        )

        params[f"{adc_path}.is_adc_windows_timing_increasing_and_valid"] = _expr(
            inputs={
                "duration": f"{adc_path}.duration",
                "samples": f"{adc_path}.number_of_samples",
                "sample_time": f"{adc_path}.sample_time",
                "adc_set": "root.adc_settings",
            },
            script=(
                "if duration <= 0 or samples <= 0 or sample_time <= 0 then\n"
                "  return false\n"
                "end\n"
                "if modulo(duration, adc_set.raster_samples) ~= 0 then\n"
                "  return false\n"
                "end\n"
                "return true"
            ),
        )


    def _loop_length_protocol_source(
        self,
        *,
        loop_path: str,
    ) -> str | None:
        """Return the protocol key that should drive a gammaSTAR Loop length.

        The visible loop token can be semantic, e.g. ``root.ky`` from a
        ``ky_index`` counter, while the matrix-size protocol key is ``n_y``.
        This helper keeps the loop length live by recovering that source from
        the symbolic node registry when possible, and by applying conservative
        k-space token aliases as a fallback.
        """

        loop_token = _safe_path_token(loop_path.rsplit(".", 1)[-1])

        # Direct protocol match, e.g. root.average.length <- root.prot.average.
        direct = _protocol_key(loop_token)
        if direct in self._live_protocol_values:
            return direct

        # Matrix/encoding loop aliases. These are semantic protocol dimensions,
        # not sequence names.
        alias_candidates = {
            "ky": ("n_y", "ny", "phase_encode_steps"),
            "phase": ("n_y", "ny", "phase_encode_steps"),
            "pe": ("n_y", "ny", "phase_encode_steps"),
            "kx": ("n_x", "nx", "readout_points"),
            "read": ("n_x", "nx", "readout_points"),
            "kz": ("n_z", "nz", "partitions"),
            "partition": ("n_z", "nz", "partitions"),
            "slice": ("n_slices", "num_slices", "slices"),
        }.get(loop_token, ())

        for candidate in alias_candidates:
            key = _protocol_key(candidate)
            if key in self._live_protocol_values:
                return key

        # Recover source from the symbolic sequence, since the realization view
        # may already have resolved repeat_count/factor to a number.
        source = self._loop_length_source_from_symbolic_nodes(
            loop_token=loop_token,
        )
        if source is not None:
            return source

        return None

    def _loop_length_source_from_symbolic_nodes(
        self,
        *,
        loop_token: str,
    ) -> str | None:
        """Inspect symbolic set_node metadata for repeat_count/factor source."""

        registry = _sequence_node_registry_for_export(self.symbolic_sequence)

        for _node_name, record in registry.items():
            if not isinstance(record, Mapping):
                continue

            record_token = _repeat_loop_token_for_export(
                record.get("repeat_count"),
                record.get("counter"),
            )
            record_name_token = _safe_path_token(
                str(
                    record.get("name")
                    or record.get("node")
                    or _node_name
                ).rsplit(".", 1)[-1]
            )
            # The emitted gammaSTAR loop path is derived from the logical node
            # name (for example, root.kernel), whereas the semantic loop token
            # may come from its counter (for example, ky_index -> ky). Match
            # either identity before recovering the symbolic repeat-count
            # source. Otherwise factor=p.n_y may be exported as a fixed literal.
            if (
                _safe_path_token(record_token) != loop_token
                and record_name_token != loop_token
            ):
                continue

            source = _protocol_source_name_from_value(
                record.get("repeat_count"),
                self._live_protocol_values,
            )
            if source is not None:
                return source

            # Some timeline implementations keep the original expression under
            # a metadata/source key after resolving repeat_count numerically.
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                for key in (
                    "repeat_count_source",
                    "factor_source",
                    "loop_length_source",
                    "protocol_source",
                ):
                    source = _protocol_source_name_from_value(
                        metadata.get(key),
                        self._live_protocol_values,
                    )
                    if source is not None:
                        return source

            for key in (
                "factor",
                "factor_source",
                "repeat_count_source",
                "loop_length_source",
            ):
                source = _protocol_source_name_from_value(
                    record.get(key),
                    self._live_protocol_values,
                )
                if source is not None:
                    return source

        return None

    def _gradient_direction_parameter(
        self,
        *,
        gradient_event: Any,
        fallback_direction: Any,
    ) -> dict[str, Any]:
        """Return live gammaSTAR direction expression for a logical gradient."""

        logical_axis = _gradient_logical_axis_for_orientation(gradient_event)
        if logical_axis == "read":
            return _expr(
                inputs={"direction": "root.info.encoding_read_dir"},
                script="return direction",
            )
        if logical_axis == "phase":
            return _expr(
                inputs={"direction": "root.info.encoding_phase_dir"},
                script="return direction",
            )
        if logical_axis == "slice":
            return _expr(
                inputs={"direction": "root.info.encoding_slice_dir"},
                script="return direction",
            )
        return _literal(fallback_direction)

    def _phase_encode_counter_path(
        self,
        *,
        params: Mapping[str, Any],
        adc_event: Any,
    ) -> str | None:
        """Return the loop counter that should drive k-space line index.

        The outer active loop is the semantic phase-encode loop for compact GRE
        style exports (root.ky.counter).  We prefer explicit event metadata when
        it exists, then fall back to the active loop unless it is the generic
        root.average loop.
        """

        metadata = getattr(adc_event, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in (
                "phase_encode_counter_path",
                "line_counter_path",
                "kspace_encode_step_1_counter_path",
            ):
                value = metadata.get(key)
                if isinstance(value, str) and value in params:
                    return value

        counter_path = _event_loop_counter_path(
            adc_event,
            default=f"{self._active_loop_path}.counter",
        )
        if isinstance(counter_path, str) and counter_path in params:
            # Do not use a generic averaging loop as a phase-encode line index.
            if counter_path.endswith(".counter") and ".average." not in counter_path:
                return counter_path
            if counter_path != "root.average.counter":
                return counter_path

        active_counter = f"{self._active_loop_path}.counter"
        if (
            self._active_loop_path != "root.average"
            and active_counter in params
        ):
            return active_counter

        return None

    def _slice_index_default_path(self, params: dict[str, Any]) -> str:
        """Ensure a scalar default slice-index parameter exists."""

        path = "root.info.default_slice_index"
        params.setdefault(path, self._slice_index_parameter(default=0))
        return path

    def _slice_index_parameter(self, *, default: int = 0) -> dict[str, Any]:
        """Return a live slice-index parameter when the protocol declares one."""

        for key in ("slice_index", "idx_slice", "slice"):
            if key in self._live_protocol_values:
                return _expr(
                    inputs={"slice_index": f"root.prot.{_protocol_key(key)}"},
                    script="return slice_index or 0",
                )
        return _literal(int(default))

    def _event_container_path_for_endpoint(self, endpoint: Any) -> str | None:
        """Return the exported gammaSTAR container path for an event endpoint.

        Relationship objects may point to the original prototype event, while
        the compacted/exported motif may contain a copied/frozen occurrence
        such as ``adc_readout_occ001``. Match by identity first, then by
        source/prototype aliases, and finally by normalized event name. This is
        generic relationship endpoint resolution; it is not sequence-specific.
        """

        if endpoint is None:
            return None

        direct = self._event_container_paths.get(id(endpoint))
        if direct is not None:
            return direct

        for event_id, event in self._event_container_events.items():
            if _relationship_event_matches(endpoint, event):
                return self._event_container_paths.get(event_id)

        return None

    def _event_tstart_parameter(
        self,
        *,
        target_event: Any,
        target_container_path: str,
        target_duration_path: str,
        fallback_tstart: float | None,
        local_tstart: float = 0.0,
        local_tstart_expr: str | None = None,
    ) -> dict[str, Any]:
        """Return a gammaSTAR parameter for an event/container start time.

        This first tries to preserve an explicit PyPulseq-Star timing
        relationship as a live gammaSTAR expression. If the relationship object
        is not discoverable in the current sequence container, it falls back to a
        generic protocol-relative expression for ADC/readout timing:

            live_tstart = root.prot.<offset> + (resolved_tstart - resolved_offset)

        That fallback is intentionally sequence-general. It does not know FID,
        FLASH, GRE, EPI, ASL, or pCASL. It only uses common protocol timing
        parameters such as TE that already exist in root.prot and the resolved
        event timing already materialized by the relationship resolver.
        """

        relationship = _find_timing_relationship_for_target(
            self.sequence,
            target_event=target_event,
        )

        if relationship is None:
            fallback = self._protocol_relative_tstart_parameter(
                target_event=target_event,
                fallback_tstart=fallback_tstart,
                local_tstart_expr=local_tstart_expr,
            )
            if fallback is not None:
                return fallback

            if local_tstart_expr is not None:
                return _expr(
                    inputs={"window_data": local_tstart_expr},
                    script="return window_data.tstart or window_data.delay or 0",
                )
            return _literal(float(fallback_tstart or 0.0))

        reference_event = relationship.get("reference")
        reference_path = self._event_container_path_for_endpoint(reference_event)

        # If the relationship endpoint cannot be mapped back to a gammaSTAR
        # container path, still preserve live protocol behavior using the
        # resolved timing offset. This handles relationship implementations that
        # store endpoint names/ids rather than object references.
        if reference_event is None or reference_path is None:
            fallback = self._protocol_relative_tstart_parameter(
                target_event=target_event,
                fallback_tstart=fallback_tstart,
                local_tstart_expr=local_tstart_expr,
                offset_override=relationship.get("offset"),
            )
            if fallback is not None:
                return fallback

            if local_tstart_expr is not None:
                return _expr(
                    inputs={"window_data": local_tstart_expr},
                    script="return window_data.tstart or window_data.delay or 0",
                )
            return _literal(float(fallback_tstart or 0.0))

        offset = relationship.get("offset", 0.0)
        target_anchor = _anchor_fraction(relationship.get("target_anchor", "start"))
        reference_anchor = _anchor_fraction(relationship.get("reference_anchor", "start"))

        inputs = {
            "reference_tstart": f"{reference_path}.tstart",
            "reference_duration": f"{reference_path}.duration",
            "target_duration": target_duration_path,
        }

        if isinstance(offset, str):
            offset_key = _protocol_key(offset)
            inputs["offset"] = f"root.prot.{offset_key}"
            offset_script = "offset"
        else:
            try:
                offset_value = float(offset)
            except (TypeError, ValueError):
                offset_value = 0.0
            offset_script = repr(offset_value)

        if local_tstart_expr is not None:
            inputs["window_data"] = local_tstart_expr
            # The relationship expression positions the exported target event or
            # train container.  Per-window timing must therefore be relative to
            # the relationship-anchored base window, not an absolute tstart that
            # may already include block/kernel time.  The ADC normalizer adds
            # relative_tstart/relative_delay generically for this purpose.
            local_script = "(window_data.relative_tstart or window_data.relative_delay or 0)"
        else:
            # local_tstart is intentionally a small offset inside the exported
            # target container.  Call sites must not pass a resolved absolute
            # event/window tstart here.
            local_script = repr(float(local_tstart))

        return _expr(
            inputs=inputs,
            script=(
                f"return reference_tstart + ({reference_anchor!r})*reference_duration "
                f"+ {offset_script} - ({target_anchor!r})*target_duration "
                f"+ {local_script}"
            ),
        )

    def _protocol_relative_tstart_parameter(
        self,
        *,
        target_event: Any,
        fallback_tstart: float | None,
        local_tstart_expr: str | None,
        offset_override: Any = None,
    ) -> dict[str, Any] | None:
        """Return a live protocol-relative tstart expression when possible.

        This is a generic bridge for gammaSTAR Protocol Control. If the Python
        relationship resolver has already materialized an event start time, but
        the relationship object is not available in a form the writer can
        inspect, this preserves editability by linking the emitted tstart back
        to the protocol offset parameter.

        Example:
            resolved ADC tstart = 0.02055
            resolved root.prot.TE = 0.02000

            emitted:
                return root.prot.TE + 0.00055

        Editing TE in gammaSTAR then moves the readout while preserving the
        original resolved timing offset.
        """

        # Keep this conservative. For now, only ADC/readout-like events get this
        # fallback, because TE is conventionally a readout timing parameter. This
        # is not sequence-specific; it is event-role/parameter semantics.
        if not _is_adc_event(target_event):
            return None

        offset_key = None
        if isinstance(offset_override, str) and offset_override:
            candidate = _protocol_key(offset_override)
            if candidate in self._live_protocol_values:
                offset_key = candidate

        if offset_key is None:
            for candidate in ("TE", "echo_time", "EchoTime"):
                candidate_key = _protocol_key(candidate)
                if candidate_key in self._live_protocol_values:
                    offset_key = candidate_key
                    break

        if offset_key is None:
            return None

        try:
            resolved_offset = float(self._live_protocol_values[offset_key])
        except (TypeError, ValueError):
            return None

        if local_tstart_expr is not None:
            inputs = {
                "offset": f"root.prot.{offset_key}",
                "window_data": local_tstart_expr,
            }
            return _expr(
                inputs=inputs,
                script=(
                    f"local resolved_offset = {resolved_offset!r}\n"
                    "local resolved_tstart = window_data.tstart or window_data.delay or 0\n"
                    "return offset + (resolved_tstart - resolved_offset)"
                ),
            )

        if fallback_tstart is None:
            return None

        delta = float(fallback_tstart) - resolved_offset
        return _expr(
            inputs={"offset": f"root.prot.{offset_key}"},
            script=f"return offset + {delta!r}",
        )

    def _force_adc_protocol_relative_tstart_parameter(
        self,
        *,
        target_event: Any,
        fallback_tstart: float | None,
        local_tstart_expr: str | None,
    ) -> dict[str, Any] | None:
        """Force a live protocol-relative ADC/readout start expression.

        This is deliberately sequence-general. It does not inspect sequence
        names or sequence types. It only uses the generic fact that ADC/readout
        timing is commonly controlled by protocol timing parameters such as TE.

        If a resolved ADC/readout start exists at export time and the matching
        protocol timing value exists, emit:

            tstart = root.prot.<timing_key> + (resolved_tstart - resolved_timing)

        This preserves the resolved timing for the initially exported file and
        makes Protocol Control edits live on the gammaSTAR website.
        """

        if not _is_adc_event(target_event):
            return None

        timing_key, resolved_timing = self._best_adc_protocol_timing_reference()
        if timing_key is None or resolved_timing is None:
            return None

        protocol_path = f"root.prot.{timing_key}"

        if local_tstart_expr is not None:
            return _expr(
                inputs={
                    "offset": protocol_path,
                    "window_data": local_tstart_expr,
                },
                script=(
                    f"local resolved_offset = {float(resolved_timing)!r}\n"
                    "local resolved_tstart = window_data.tstart or window_data.delay or 0\n"
                    "return offset + (resolved_tstart - resolved_offset)"
                ),
            )

        if fallback_tstart is None:
            return None

        delta = float(fallback_tstart) - float(resolved_timing)
        return _expr(
            inputs={"offset": protocol_path},
            script=f"return offset + {delta!r}",
        )

    def _best_adc_protocol_timing_reference(self) -> tuple[str | None, float | None]:
        """Return the best live protocol timing key for ADC/readout placement."""

        # Prefer canonical TE if available. This is a protocol-timing concept,
        # not a sequence-specific branch.
        candidate_keys = (
            "TE",
            "echo_time",
            "EchoTime",
            "echoTime",
        )

        values = _canonical_protocol_values(_collect_protocol_values(self.sequence))
        values.update(self._live_protocol_values)

        for key in candidate_keys:
            canonical = _protocol_key(key)
            if canonical in values and values[canonical] is not None:
                try:
                    return canonical, float(values[canonical])
                except (TypeError, ValueError):
                    pass
            if key in values and values[key] is not None:
                try:
                    return canonical, float(values[key])
                except (TypeError, ValueError):
                    pass

        return None, None

    def _add_atomic_full_basic_repr(
        self,
        *,
        params: dict[str, Any],
        atomic_path: str,
    ) -> None:
        """Add Atomic full_basic_repr scaffolding."""

        params[f"{atomic_path}.full_basic_repr_GradPulse"] = _literal({})
        params[f"{atomic_path}.full_basic_repr_RFPulse"] = _literal({})
        params[f"{atomic_path}.full_basic_repr_ADC"] = _literal({})
        params[f"{atomic_path}.full_basic_repr_TriggerPulse"] = _literal({})

        params[f"{atomic_path}.full_basic_repr"] = _expr(
            inputs={
                "full_basic_repr_TriggerPulse": f"{atomic_path}.full_basic_repr_TriggerPulse",
                "full_basic_repr_ADC": f"{atomic_path}.full_basic_repr_ADC",
                "full_basic_repr_RFPulse": f"{atomic_path}.full_basic_repr_RFPulse",
                "full_basic_repr_GradPulse": f"{atomic_path}.full_basic_repr_GradPulse",
            },
            script=(
                "return {\n"
                "GradPulse=full_basic_repr_GradPulse,\n"
                "RFPulse=full_basic_repr_RFPulse,\n"
                "ADC=full_basic_repr_ADC,\n"
                "TriggerPulse=full_basic_repr_TriggerPulse\n"
                "}"
            ),
        )

    def _rf_samples_parameter(
        self,
        *,
        rf_event: Any,
        rf_rect_path: str,
        rf_path: str,
    ) -> dict[str, Any]:
        """Return the gammaSTAR RF samples parameter."""

        shape = getattr(rf_event, "shape", None)
        flip_angle = getattr(rf_event, "flip_angle", None)

        class_name = shape.__class__.__name__.lower() if shape is not None else ""
        kind = str(getattr(shape, "kind", "")).lower() if shape is not None else ""

        if shape is None or "block" in class_name or "block" in kind:
            return _expr(
                inputs={
                    "gamma": "root.sys.gamma",
                    "flip_angle": f"{rf_rect_path}.flip_angle",
                    "duration": f"{rf_path}.duration",
                },
                script=(
                    "local rf_amp = "
                    "(flip_angle/180*math.pi)/(2*math.pi*gamma*duration)\n"
                    "local samples_t = {duration/4, 3*duration/4}\n"
                    "local samples_am = {rf_amp, rf_amp}\n"
                    "local samples_fm = {0, 0}\n"
                    "return {t=samples_t, v={{am=samples_am, fm=samples_fm}}}"
                ),
            )

        if hasattr(shape, "to_gammastar_samples") and flip_angle is not None:
            samples = shape.to_gammastar_samples(
                flip_angle=float(flip_angle),
                gamma_hz_per_t=float(self.system.gamma),
                max_rf=None,
            )
            return _literal(samples)

        raise ValueError(f"Cannot export RF samples for event {rf_event!r}")

    def _adc_windows_data(self, *, adc_event: Any) -> dict[str, Any]:
        """Return normalized gammaSTAR ADC-window data."""

        if hasattr(adc_event, "to_gammastar_windows"):
            data = adc_event.to_gammastar_windows(include_sample_times=False)
            return _normalize_adc_data(data, adc_event=adc_event)

        if hasattr(adc_event, "to_gammastar_samples"):
            data = adc_event.to_gammastar_samples(include_sample_times=False)
            return _normalize_adc_data(data, adc_event=adc_event)

        shape = getattr(adc_event, "shape", None)

        if shape is not None and hasattr(shape, "to_gammastar_windows"):
            data = shape.to_gammastar_windows(include_sample_times=False)
            return _normalize_adc_data(data, adc_event=adc_event)

        if shape is not None and hasattr(shape, "to_gammastar_samples"):
            data = shape.to_gammastar_samples(include_sample_times=False)
            return _normalize_adc_data(data, adc_event=adc_event)

        return _normalize_adc_data(_fallback_single_adc_data(adc_event), adc_event=adc_event)

    def _rebuild_tests_after_final_hierarchy(
        self,
        *,
        params: dict[str, Any],
    ) -> None:
        """Rebuild root.tests.all_tests after all hierarchy rewrites.

        The writer creates RF/ADC/gradient test parameters before block
        grouping and nested-loop rewrites. Those rewrites correctly retarget
        parameter paths and inputs, but the Lua table inside
        ``root.tests.all_tests`` can still contain stale display keys and nil
        descriptions. This final pass scans the finished parameter graph and
        rebuilds the aggregate test table from the actual final test paths.
        """

        tests: list[tuple[str, str, str]] = []

        for path in sorted(params, key=_hierarchy_sort_key):
            if path == "root.tests.all_tests":
                continue
            if path.endswith(
                ".is_timing_increasing_and_rastered_and_same_am_length"
            ):
                tests.append(
                    (
                        path,
                        _test_display_key(path),
                        "RF sample times are strictly increasing, raster-aligned, and match RF amplitude-vector length.",
                    )
                )
            elif path.endswith(
                ".is_adc_windows_timing_increasing_and_valid"
            ):
                tests.append(
                    (
                        path,
                        _test_display_key(path),
                        "ADC acquisition window has positive duration, positive sample count, positive dwell time, and raster-aligned duration.",
                    )
                )
            elif path.endswith(".is_timing_increasing_and_rastered"):
                tests.append(
                    (
                        path,
                        _test_display_key(path),
                        "Gradient sample times are non-negative, strictly increasing, and gradient-raster aligned.",
                    )
                )

        stale_test = "root.tests.seqstar_no_stale_literal_tstarts"
        if stale_test in params:
            tests.append(
                (
                    stale_test,
                    _test_display_key(stale_test),
                    "No downstream block or event tstart remains a stale literal after live timing dependencies are introduced.",
                )
            )

        if not tests:
            params["root.tests.all_tests"] = _expr(
                inputs={},
                script="return {}",
            )
            params["root.info.seqstar_test_count"] = _literal(0)
            return

        inputs: dict[str, str] = {}
        result_lines: list[str] = ["return {"]
        used_names: set[str] = set()

        for index, (path, display_key, description) in enumerate(tests):
            base_name = _safe_input_name(path)
            safe_name = base_name
            if safe_name in used_names:
                safe_name = f"{base_name}_{index}"
            used_names.add(safe_name)
            inputs[safe_name] = path
            result_lines.append(
                f"[{_lua_string(display_key)}] = "
                f"{{ ok = {safe_name} ~= nil and {safe_name}, "
                f"desc = {_lua_string(description)} }},"
            )

        result_lines.append("}")
        params["root.tests.all_tests"] = _expr(
            inputs=inputs,
            script="\n".join(result_lines),
        )
        params["root.info.seqstar_test_count"] = _literal(len(tests))
        params["root.info.seqstar_tests_rebuilt_after_hierarchy"] = _literal(True)

    def _add_tests(
        self,
        *,
        params: dict[str, Any],
        rf_paths: list[str],
        adc_paths: list[str],
        gradient_paths: list[str] | None = None,
    ) -> None:
        """Add gammaSTAR tests."""

        gradient_paths = list(gradient_paths or [])

        if not rf_paths and not adc_paths and not gradient_paths:
            params["root.tests.all_tests"] = _literal({})
            return

        inputs: dict[str, str] = {}
        result_lines: list[str] = ["return {"]

        for rf_path in rf_paths:
            safe_name = _safe_input_name(rf_path)
            test_name = f"{rf_path}.is_timing_increasing_and_rastered_and_same_am_length"
            inputs[safe_name] = test_name
            result_lines.append(
                f'["{rf_path.removeprefix("root.")}.'
                'is_timing_increasing_and_rastered_and_same_am_length"] = '
                f"{{ ok = {safe_name} ~= nil and {safe_name}, desc = 'RF sample timing/raster validation.' }},"
            )

        for adc_path in adc_paths:
            safe_name = _safe_input_name(adc_path)
            test_name = f"{adc_path}.is_adc_windows_timing_increasing_and_valid"
            inputs[safe_name] = test_name
            result_lines.append(
                f'["{adc_path.removeprefix("root.")}.'
                'is_adc_windows_timing_increasing_and_valid"] = '
                f"{{ ok = {safe_name} ~= nil and {safe_name}, desc = 'ADC window timing and dwell validation.' }},"
            )

        for grad_path in gradient_paths:
            safe_name = _safe_input_name(grad_path)
            test_name = f"{grad_path}.is_timing_increasing_and_rastered"
            inputs[safe_name] = test_name
            result_lines.append(
                f'["{grad_path.removeprefix("root.")}.is_timing_increasing_and_rastered"] = '
                f"{{ ok = {safe_name} ~= nil and {safe_name}, desc = 'Gradient sample timing/raster validation.' }},"
            )

        result_lines.append("}")

        params["root.tests.all_tests"] = _expr(
            inputs=inputs,
            script="\n".join(result_lines),
        )

    def _add_protocol(
        self,
        *,
        params: dict[str, Any],
        rf_events: list[Any],
        adc_events: list[Any],
        gradient_events: list[Any] | None = None,
        tr: float,
        repetitions: int,
    ) -> None:
        """Add protocol parameters from the enriched sequence."""

        # Use the symbolic/numeric merged protocol values prepared in to_dict().
        # This is the key writer-boundary source of truth: numeric event geometry
        # comes from self.sequence, while root.prot.* should preserve symbolic
        # protocol relationships from self.symbolic_sequence whenever available.
        protocol_values = dict(getattr(self, "_live_protocol_values", {}) or {})
        if not protocol_values:
            protocol_values = _canonical_protocol_values(
                _merge_protocol_values_for_export(
                    export_sequence=self.sequence,
                    symbolic_sequence=self.symbolic_sequence,
                )
            )


        protocol_values.setdefault("TR", tr)
        protocol_values.setdefault("average", repetitions)
        protocol_values.setdefault("averages", repetitions)
        protocol_values.setdefault("repetitions", repetitions)
        protocol_values.setdefault("seq_dim", "2D")
        protocol_values.setdefault("minimal_TE", False)

        if rf_events:
            first_rf = rf_events[0]
            protocol_values.setdefault(
                "flip_angle",
                math.degrees(float(getattr(first_rf, "flip_angle"))),
            )

        gradient_events = list(gradient_events or [])

        if adc_events:
            first_adc = adc_events[0]
            adc_data = self._adc_windows_data(adc_event=first_adc)
            first_window = adc_data["windows"][0]

            protocol_values.setdefault(
                "adc_samples",
                int(first_window.get("num_samples", first_window.get("number_of_samples", 1))),
            )
            protocol_values.setdefault(
                "adc_delay",
                float(first_window.get("tstart", first_window.get("delay", 0.0))),
            )
            protocol_values.setdefault(
                "readout_duration",
                float(first_window.get("duration", 0.0)),
            )
            protocol_values.setdefault(
                "echo_train_length",
                int(adc_data.get("num_windows", len(adc_data["windows"]))),
            )
            protocol_values.setdefault(
                "trajectory",
                adc_data.get("trajectory")
                if adc_data.get("trajectory") is not None
                else {},
            )

            # Generic acquisition-geometry fallbacks. They apply to any
            # sequence with ADC samples and incomplete spatial metadata; no
            # sequence-family names or gradient assumptions are involved.
            read_samples = int(
                first_window.get(
                    "num_samples",
                    first_window.get("number_of_samples", 1),
                )
            )
            protocol_values.setdefault("n_x", max(1, read_samples))
            protocol_values.setdefault("n_y", 1)
            protocol_values.setdefault("n_z", 1)
            protocol_values.setdefault("slice_thickness", 1.0)
            if "fov" not in protocol_values:
                fov_read = protocol_values.get("fov_read")
                fov_phase = protocol_values.get("fov_phase")
                if fov_read is not None or fov_phase is not None:
                    read_fov = fov_read if fov_read is not None else fov_phase
                    phase_fov = fov_phase if fov_phase is not None else fov_read
                    protocol_values["fov"] = [
                        read_fov,
                        phase_fov,
                        protocol_values["slice_thickness"],
                    ]
                else:
                    protocol_values["fov"] = 1.0

            if len(adc_data["windows"]) > 1:
                t0 = float(
                    adc_data["windows"][0].get(
                        "tstart",
                        adc_data["windows"][0].get("delay", 0.0),
                    )
                )
                t1 = float(
                    adc_data["windows"][1].get(
                        "tstart",
                        adc_data["windows"][1].get("delay", 0.0),
                    )
                )
                protocol_values.setdefault("echo_spacing", t1 - t0)

        if gradient_events:
            first_grad = gradient_events[0]
            grad_data = self._gradient_waveform_data(gradient_event=first_grad)
            protocol_values.setdefault("gradient_channel", grad_data.get("channel"))
            protocol_values.setdefault("gradient_kind", grad_data.get("kind"))
            protocol_values.setdefault("gradient_duration", grad_data.get("active_duration"))
            protocol_values.setdefault("gradient_area", grad_data.get("area"))
            protocol_values.setdefault("gradient_max_abs_amplitude", grad_data.get("max_abs_amplitude"))
            protocol_values.setdefault("num_gradient_events", len(gradient_events))

        defaults = {
            "orientation": "axial",
            "TE": 0.0,
            "echo_spacing": 0.0,
            "echo_train_length": 1,
            "adc_samples": 0,
            "adc_delay": 0.0,
            "readout_duration": 0.0,
            "trajectory": {},
            "gradient_channel": None,
            "gradient_kind": None,
            "gradient_duration": 0.0,
            "gradient_area": 0.0,
            "gradient_max_abs_amplitude": 0.0,
            "num_gradient_events": 0,
            "n_x": 1,
            "n_y": 1,
            "n_z": 1,
            "fov": 1.0,
            "slice_thickness": 1.0,
            "PAT_factor_phase": 1,
            "PAT_factor_slice": 1,
            "PAT_mode": "None",
            "PAT_ref_lines_phase": 32,
            "PAT_ref_lines_slice": 16,
            "multiband_factor": 1,
            "phase_oversampling": 0.0,
            "phase_partial_fourier": 1,
            "read_oversampling": 1.0,
            "read_partial_fourier": 1,
            "slice_oversampling": 0.0,
            "slice_partial_fourier": 1,
            "slice_distance_factor": 0.0,
            "slice_reorder_scheme": "Interleaved",
            "turbo_factor": 1,
            "offcenter_exc_1": [0.0, 0.0, 0.0],
            "orientation_exc_1": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "rot_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "partitions": None,
            "point_of_interest": None,
            "slices": None,
            "thickness_exc_1": -1,
        }

        for key, value in defaults.items():
            protocol_values.setdefault(key, value)

        params["root.prot.ui_specification"] = _literal(
            json.dumps(_build_protocol_ui_specification(protocol_values), ensure_ascii=False)
        )

        for key, value in protocol_values.items():
            gamma_key = _protocol_key(key)
            target_path = f"root.prot.{gamma_key}"
            new_param = _protocol_parameter(
                value,
                target_path=target_path,
            )
            old_param = params.get(target_path)
            if _gammastar_debug_enabled() and gamma_key in {
                "phase_encode_start",
                "phase_encode_step",
                "n_y",
                "fov",
                "TR",
                "orientation",
            }:
                _gammastar_debug(
                    f"_add_protocol write {target_path}: "
                    f"old={old_param!r} new={new_param!r} "
                    f"value={_debug_value_summary(value)}"
                )

            # Protocol controls are authoritative for root.prot.*. Earlier
            # passes may create placeholder/default literals so dependent graph
            # nodes can refer to the path, but the final protocol export must
            # replace those placeholders with the merged live protocol source.
            params[target_path] = new_param

        # Avoid two independent orientation state variables in the website.
        # Some gammaSTAR templates expose "slice_orientation"; keep it as a
        # compatibility alias of the PyPulseq-Star orientation parameter.
        params["root.prot.slice_orientation"] = _expr(
            inputs={"orientation": "root.prot.orientation"},
            script="return orientation",
        )

    def _add_system(self, *, params: dict[str, Any]) -> None:
        """Add system parameters from SeqStar Opts."""

        system = self.system

        params["root.sys.gamma"] = _literal(float(system.gamma))
        params["root.sys.max_rf_amp"] = _literal(float(system.max_rf))
        params["root.sys.raster_samples_rf"] = _literal(float(system.rf_raster_time))
        params["root.sys.raster_time_rf"] = _literal(float(system.rf_raster_time))

        params["root.sys.max_grad_amp"] = _literal(_max_grad_t_per_m(system))
        params["root.sys.max_grad_slew"] = _literal(_max_slew_t_per_m_s(system))

        params["root.sys.raster_samples_grad"] = _literal(float(system.grad_raster_time))
        params["root.sys.raster_time_grad"] = _literal(float(system.grad_raster_time))
        params["root.sys.raster_samples_atomic"] = _literal(float(system.block_duration_raster))
        params["root.sys.raster_time_atomic"] = _literal(float(system.block_duration_raster))
        params["root.sys.raster_samples_trig"] = _literal(float(system.block_duration_raster))
        params["root.sys.raster_time_trig"] = _literal(float(system.block_duration_raster))

        adc_raster = float(getattr(system, "adc_raster_time", 1e-6))
        adc_sample_raster = float(getattr(system, "adc_sample_raster_time", adc_raster))
        params["root.sys.raster_samples_adc"] = _literal(adc_sample_raster)
        params["root.sys.raster_time_adc"] = _literal(adc_raster)

        params["root.sys.frequency"] = _literal([123200000.0, 0.0])
        params["root.sys.scanner_type"] = _literal(str(getattr(system, "scanner_type", "Generic")))
        params["root.sys.system_specs"] = _literal({})
        params["root.sys.coil_values_for_pns"] = _literal({})
        params["root.sys.fat_shift"] = _literal(0.0)
        params["root.sys.acoustic_resonance_frequencies"] = _literal(
            [[585.0, 100.0], [1120.0, 220.0]]
        )
        params["root.sys.min_distance_between_adc_and_grad"] = _literal(0.0)
        params["root.sys.min_distance_between_adc_and_rf"] = _literal(0.0)
        params["root.sys.min_distance_between_grad_and_rf"] = _literal(0.0)

        params["root.rf_settings"] = _expr(
            inputs={
                "raster_time": "root.sys.raster_time_rf",
                "raster_samples": "root.sys.raster_samples_rf",
                "max_rf_amp": "root.sys.max_rf_amp",
            },
            script=(
                "return {max_rf_amp=max_rf_amp, "
                "raster_time=raster_time, raster_samples=raster_samples}"
            ),
        )

        params["root.gradient_settings"] = _expr(
            inputs={
                "raster_time": "root.sys.raster_time_grad",
                "raster_samples": "root.sys.raster_samples_grad",
                "max_grad_amp": "root.sys.max_grad_amp",
                "max_grad_slew": "root.sys.max_grad_slew",
            },
            script=(
                "return {max_grad_amp=max_grad_amp, max_grad_slew=max_grad_slew, "
                "raster_time=raster_time, raster_samples=raster_samples}"
            ),
        )

        params["root.gradient_settings_reduced_performance"] = _expr(
            inputs={"gradient_settings": "root.gradient_settings"},
            script="return gradient_settings",
        )

        params["root.adc_settings"] = _expr(
            inputs={
                "raster_time": "root.sys.raster_time_adc",
                "raster_samples": "root.sys.raster_samples_adc",
            },
            script="return {raster_time=raster_time, raster_samples=raster_samples}",
        )

        params["root.atomic_settings"] = _expr(
            inputs={
                "raster_time": "root.sys.raster_time_atomic",
                "raster_samples": "root.sys.raster_samples_atomic",
            },
            script="return {raster_time=raster_time, raster_samples=raster_samples}",
        )

        params["root.trigger_settings"] = _expr(
            inputs={
                "raster_time": "root.sys.raster_time_trig",
                "raster_samples": "root.sys.raster_samples_trig",
            },
            script="return {raster_time=raster_time, raster_samples=raster_samples}",
        )

    def _add_info(
        self,
        *,
        params: dict[str, Any],
        rf_events: list[Any],
        adc_events: list[Any],
        gradient_events: list[Any] | None = None,
    ) -> None:
        """Add info section."""

        description = _get_any(
            self.sequence,
            keys=("description",),
            default="Sequence generated by PyPulseq-Star.",
        )

        gradient_events = list(gradient_events or [])

        has_adc_train = any(_adc_event_num_windows(event) > 1 for event in adc_events)
        is_epi = any(
            str(_get_any(event, keys=("trajectory",), default="")).lower().find("epi") >= 0
            for event in adc_events
        )

        encoding = _encoding_frame_info(self.sequence)

        params["root.info.description"] = _literal(description)
        # Sequence dimensionality is protocol metadata, not a property that can
        # be inferred from the presence of spatial gradients. Gradient-free
        # acquisitions such as spectroscopy/FID still need a valid scanner
        # dimensionality for gammaSTAR import/export. Accept the common numeric
        # and string forms and use 2D as the conservative compatibility default.
        params["root.info.seq_dim"] = _expr(
            inputs={"seq_dim": "root.prot.seq_dim"},
            script=(
                "if type(seq_dim) == 'number' then\n"
                "  if seq_dim >= 3 then return 3 end\n"
                "  return 2\n"
                "end\n"
                "local key = string.upper(tostring(seq_dim or '2D'))\n"
                "if key == '3' or key == '3D' then return 3 end\n"
                "return 2"
            ),
        )
        params["root.info.is_epi"] = _literal(bool(is_epi))
        params["root.info.has_rf"] = _literal(bool(rf_events))
        params["root.info.has_adc"] = _literal(bool(adc_events))
        params["root.info.has_adc_train"] = _literal(bool(has_adc_train))
        params["root.info.has_gradient"] = _literal(bool(gradient_events))
        params["root.info.num_gradient_events"] = _literal(len(gradient_events))
        params["root.info.encoding_frame_name"] = _expr(
            inputs={"orientation": "root.prot.orientation"},
            script=(
                "local key = string.lower(tostring(orientation or 'axial'))\n"
                "if key == 'tra' or key == 'transverse' or key == 'ax' then key = 'axial' end\n"
                "if key == 'cor' then key = 'coronal' end\n"
                "if key == 'sag' then key = 'sagittal' end\n"
                "return key"
            ),
        )
        params["root.info.encoding_rotation"] = _expr(
            inputs={"orientation": "root.prot.orientation"},
            script=_orientation_lua_script("rotation"),
        )
        params["root.info.image_transform"] = _expr(
            inputs={
                "rotation": "root.info.encoding_rotation",
                "position": "root.info.encoding_position",
            },
            script=(
                "return {\n"
                "{rotation[1][1], rotation[1][2], rotation[1][3], position[1]},\n"
                "{rotation[2][1], rotation[2][2], rotation[2][3], position[2]},\n"
                "{rotation[3][1], rotation[3][2], rotation[3][3], position[3]},\n"
                "{0,0,0,1}\n"
                "}"
            ),
        )
        params["root.info.encoding_convention"] = _literal(
            encoding["convention"]
        )
        params["root.info.encoding_read_dir"] = _expr(
            inputs={"rotation": "root.info.encoding_rotation"},
            script="return {rotation[1][1], rotation[2][1], rotation[3][1]}",
        )
        params["root.info.encoding_phase_dir"] = _expr(
            inputs={"rotation": "root.info.encoding_rotation"},
            script="return {rotation[1][2], rotation[2][2], rotation[3][2]}",
        )
        params["root.info.encoding_slice_dir"] = _expr(
            inputs={"rotation": "root.info.encoding_rotation"},
            script="return {rotation[1][3], rotation[2][3], rotation[3][3]}",
        )
        params["root.info.encoding_position"] = _literal(
            encoding["position"]
        )
        params.setdefault("root.info.default_slice_index", _literal(0))

    def _add_runtime_defaults(self, *, params: dict[str, Any]) -> None:
        """Add generic runtime helpers used by gammaSTAR plots/import."""

        defaults: dict[str, Any] = {
            "root.LoopInfo": _literal({}),
            "root.PNSPaths": _literal({}),
            "root.kernel_info": _literal({}),
            # Keep the global geometry and each ADC header on one shared set of
            # live protocol relationships. This is important whenever spatial
            # metadata is incomplete (for example, an RF+ADC acquisition with
            # no gradients): the ADC sample count still defines a meaningful
            # acquisition matrix, while FOV remains a neutral protocol value.
            "root.fov": _expr(
                inputs={
                    "fov": "root.prot.fov",
                    "slice_thickness": "root.prot.slice_thickness",
                },
                script=(
                    "local z = tonumber(slice_thickness) or 1.0\n"
                    "if z <= 0 then z = 1.0 end\n"
                    "if type(fov) == 'table' then\n"
                    "  local x = tonumber(fov[1]) or 1.0\n"
                    "  local y = tonumber(fov[2]) or x\n"
                    "  local fz = tonumber(fov[3]) or z\n"
                    "  return {x, y, fz}\n"
                    "end\n"
                    "local x = tonumber(fov) or 1.0\n"
                    "if x <= 0 then x = 1.0 end\n"
                    "return {x, x, z}"
                ),
            ),
            "root.mat_size": _expr(
                inputs={
                    "n_x": "root.prot.n_x",
                    "n_y": "root.prot.n_y",
                    "n_z": "root.prot.n_z",
                },
                script=(
                    "local nx = math.max(1, math.floor((tonumber(n_x) or 1) + 0.5))\n"
                    "local ny = math.max(1, math.floor((tonumber(n_y) or 1) + 0.5))\n"
                    "local nz = math.max(1, math.floor((tonumber(n_z) or 1) + 0.5))\n"
                    "return {nx, ny, nz}"
                ),
            ),
            "root.acq_size": _expr(
                inputs={"mat_size": "root.mat_size"},
                script="return {mat_size[1], mat_size[2], mat_size[3]}",
            ),
            "root.RegridTable": _expr(
                inputs={"kernel_info": "root.kernel_info"},
                script="return kernel_info.RegridTable or {}",
            ),
            "root.expo.ADCdelay": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.ADCdelay",
            ),
            "root.expo.ADCduration": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.ADCduration",
            ),
            "root.expo.ADCsamples": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.ADCsamples",
            ),
            "root.expo.GradientFT": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.GradientFT",
            ),
            "root.expo.GradientRDT": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.GradientRDT",
            ),
            "root.expo.GradientRUT": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.GradientRUT",
            ),
            "root.expo.Mode": _expr(
                inputs={"RegridTable": "root.RegridTable"},
                script="return RegridTable.Mode",
            ),
            "root.expo.apply_freq_corr": _literal(False),
            "root.expo.echo_spacing": _expr(
                inputs={"kernel_info": "root.kernel_info"},
                script="return kernel_info.echo_spacing",
            ),
            "root.expo.feedback_paths": _literal({}),
            "root.expo.image_scale_factor": _literal(1.0),
            "root.expo.is_feedback": _expr(
                inputs={"fb_paths": "root.expo.feedback_paths"},
                script="return #fb_paths > 0",
            ),
            "root.expo.is_POI_stop": _literal(False),
            "root.expo.LoopLengthAverage": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_average or 1",
            ),
            "root.expo.LoopLengthContrast": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_contrast or 1",
            ),
            "root.expo.LoopLengthLine": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_line or 1",
            ),
            "root.expo.LoopLengthPartition": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_partition or 1",
            ),
            "root.expo.LoopLengthRepetition": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_repetition or 1",
            ),
            "root.expo.LoopLengthSegment": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_segment or 1",
            ),
            "root.expo.LoopLengthSlice": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_slice or 1",
            ),
            "root.expo.LoopLengthSlicegroup": _expr(
                inputs={"LoopInfo": "root.LoopInfo"},
                script="return LoopInfo.max_slicegroup or 1",
            ),
            "root.expo.paths_for_PNS_estimation": _expr(
                inputs={"PNSPaths": "root.PNSPaths"},
                script="return PNSPaths",
            ),
            "root.helper.constants": _expr(
                inputs={},
                script=(
                    "return {\n"
                    "golden_ratio_1d=0.618033988749,\n"
                    "golden_ratio_2d_1=0.465571231876,\n"
                    "golden_ratio_2d_2=0.682327803828,\n"
                    "spoilphase_inc_inc=117\n"
                    "}"
                ),
            ),
            "root.helper.functions": _expr(
                inputs={},
                script=(
                    "function modulo(a, b)\n"
                    "  if b == nil or b == 0 then return 0 end\n"
                    "  return math.abs(a - math.floor(a / b + 0.5) * b)\n"
                    "end\n"
                    "function ge(a, b)\n"
                    "  return a >= b or math.abs(a-b) < 1e-12\n"
                    "end\n"
                    "return {modulo=modulo, ge=ge}"
                ),
            ),
            "root.helper.complex": _literal({}),
        }

        for key, value in defaults.items():
            params.setdefault(key, value)



def _orientation_lua_script(field: str) -> str:
    """Return Lua code for live orientation presets."""

    prefix = (
        "local key = string.lower(tostring(orientation or 'axial'))\n"
        "if key == 'tra' or key == 'transverse' or key == 'ax' then key = 'axial' end\n"
        "if key == 'cor' then key = 'coronal' end\n"
        "if key == 'sag' then key = 'sagittal' end\n"
    )

    if field == "rotation":
        return (
            prefix
            + "if key == 'coronal' then\n"
            + "  return {{1,0,0},{0,0,-1},{0,1,0}}\n"
            + "end\n"
            + "if key == 'sagittal' then\n"
            + "  return {{0,0,1},{1,0,0},{0,1,0}}\n"
            + "end\n"
            + "return {{1,0,0},{0,1,0},{0,0,1}}"
        )
    raise ValueError(f"Unsupported orientation field {field!r}")


def _gradient_logical_axis_for_orientation(event: Any) -> str | None:
    """Return read/phase/slice for a gradient event, with channel fallback."""

    for attr in ("logical_axis", "axis_role", "encoding_role"):
        try:
            value = getattr(event, attr)
        except Exception:
            value = None
        value = str(value or "").strip().lower()
        if value in {"read", "phase", "slice"}:
            return value

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("logical_axis", "axis_role", "encoding_role"):
            value = str(metadata.get(key) or "").strip().lower()
            if value in {"read", "phase", "slice"}:
                return value

    channel = str(getattr(event, "channel", "") or "").lower()
    return {"x": "read", "y": "phase", "z": "slice"}.get(channel)



def _direct_protocol_source_name_from_value(
    value: Any,
    live_protocol_values: Mapping[str, Any],
) -> str | None:
    """Return a protocol key only for a genuine direct protocol reference.

    This intentionally avoids the broad token-containment logic used by
    _protocol_source_name_from_value().  For variation strength/step values,
    compound expressions such as ``-0.5 * n_y / fov`` must not be collapsed to
    ``fov`` merely because the token appears in the expression.
    """

    if value is None:
        return None

    def _candidate_matches(candidate: Any) -> str | None:
        text = str(candidate or "").strip()
        if not text:
            return None

        # Direct string keys such as "fov" or "root.prot.fov".
        key = _protocol_key(text)
        if key in live_protocol_values:
            return key

        # ParameterRef-like canonical strings sometimes include one of these
        # forms.  Accept only forms that represent one whole protocol symbol,
        # not expressions containing operators.
        for prefix in ("root.prot.", "prot.", "protocol."):
            if text.startswith(prefix):
                tail = _protocol_key(text[len(prefix):])
                if tail in live_protocol_values:
                    return tail

        return None

    if isinstance(value, str):
        return _candidate_matches(value)

    for attr in ("name", "key", "identity", "symbol", "parameter"):
        item = getattr(value, attr, None)
        if item is not None:
            match = _candidate_matches(item)
            if match is not None:
                return match

    # Canonical methods are accepted only when the canonical form is a bare
    # parameter reference.  Expressions containing operators are rejected here
    # and evaluated to numeric defaults by the caller.
    for attr in ("to_canonical", "canonical"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                text = str(method()).strip()
            except Exception:
                continue
            if re.search(r"[+\-*/()]", text):
                continue
            match = _candidate_matches(text)
            if match is not None:
                return match

    # Last resort: string form, but only if it is a bare identifier or a simple
    # root.prot.<identifier> path.  Reject anything expression-like.
    text = str(value).strip()
    if re.search(r"[+\-*/()]", text):
        return None
    return _candidate_matches(text)


def _protocol_source_name_from_value(
    value: Any,
    live_protocol_values: Mapping[str, Any],
) -> str | None:
    """Return a protocol key referenced by a symbolic value, if identifiable."""

    if value is None:
        return None

    if isinstance(value, str):
        key = _protocol_key(value)
        if key in live_protocol_values:
            return key

    # Expression/ParameterRef objects in this package generally expose either
    # to_canonical(), name, key, identity, or symbol-like string forms.
    candidates: list[str] = []

    for attr in ("to_canonical", "canonical"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                candidates.append(str(method()))
            except Exception:
                pass

    for attr in ("name", "key", "identity", "symbol", "parameter"):
        item = getattr(value, attr, None)
        if item is not None:
            candidates.append(str(item))

    candidates.append(str(value))

    for candidate in candidates:
        direct = _protocol_key(candidate)
        if direct in live_protocol_values:
            return direct

    # Fall back to token containment. Prefer longer keys so n_y wins before y.
    keys = sorted(
        (str(key) for key in live_protocol_values),
        key=len,
        reverse=True,
    )
    for candidate in candidates:
        for key in keys:
            if not key:
                continue
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", candidate):
                canonical = _protocol_key(key)
                if canonical in live_protocol_values:
                    return canonical
                if key in live_protocol_values:
                    return key

    return None


def _encoding_frame_info(sequence: Any) -> dict[str, Any]:
    """Return frozen Python-side encoding-frame metadata for header export.

    The current sprint keeps orientation selectable in Python and exports the
    selected frame consistently. Live gammaSTAR orientation switching is a later
    writer task.
    """

    metadata = getattr(sequence, "metadata", None)
    frame = None
    if isinstance(metadata, Mapping):
        frame = metadata.get("encoding_frame")

    if frame is None:
        encoding_frame = getattr(sequence, "encoding_frame", None)
        if encoding_frame is not None:
            try:
                frame = encoding_frame.to_dict()
            except Exception:
                frame = None

    if not isinstance(frame, Mapping):
        frame = {
            "name": "axial",
            "rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "position": [0.0, 0.0, 0.0],
            "transform4x4": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "convention": "columns_are_read_phase_slice_in_physical_xyz",
        }

    rotation = _coerce_matrix3(
        frame.get(
            "rotation",
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
    )
    position = _coerce_vector3(frame.get("position", (0.0, 0.0, 0.0)))

    transform = frame.get("transform4x4")
    if not (
        isinstance(transform, (list, tuple))
        and len(transform) == 4
    ):
        transform = [
            [rotation[0][0], rotation[0][1], rotation[0][2], position[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], position[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        transform = [
            [float(value) for value in row]
            for row in transform
        ]

    # Columns are read, phase, slice directions in physical xyz.
    read_dir = [rotation[0][0], rotation[1][0], rotation[2][0]]
    phase_dir = [rotation[0][1], rotation[1][1], rotation[2][1]]
    slice_dir = [rotation[0][2], rotation[1][2], rotation[2][2]]

    return {
        "name": str(frame.get("name", "encoding")),
        "rotation": rotation,
        "position": position,
        "transform4x4": transform,
        "convention": str(
            frame.get(
                "convention",
                "columns_are_read_phase_slice_in_physical_xyz",
            )
        ),
        "read_dir": read_dir,
        "phase_dir": phase_dir,
        "slice_dir": slice_dir,
    }


def _coerce_matrix3(value: Any) -> list[list[float]]:
    rows = list(value)
    if len(rows) != 3:
        raise ValueError("Encoding rotation must have three rows.")
    out: list[list[float]] = []
    for row in rows:
        row_values = list(row)
        if len(row_values) != 3:
            raise ValueError("Encoding rotation rows must have three values.")
        out.append([float(item) for item in row_values])
    return out


def _coerce_vector3(value: Any) -> list[float]:
    values = list(value)
    if len(values) != 3:
        raise ValueError("Encoding position must have three values.")
    return [float(item) for item in values]


def _find_timing_relationship_for_target(
    sequence: Any,
    *,
    target_event: Any,
) -> dict[str, Any] | None:
    """Return a timing/anchor relationship that targets ``target_event``.

    This is used to preserve live protocol dependencies in gammaSTAR JSON. It
    supports both solved relationships such as ``set_anchor_after`` and
    validation-only alignment relationships such as ``require_same_anchor``.

    The endpoint lookup deliberately prefers metadata object references
    (``target_object`` / ``reference_object``) before the public ``target`` and
    ``reference`` dictionaries, because SeqStarRelationship.target is commonly a
    JSON-safe mapping like ``{"event": "adc_readout", "anchor": "center"}``.
    """

    timing_kinds = {
        "set_center_after",
        "center_after",
        "timing.set_center_after",
        "set_anchor_after",
        "timing.set_anchor_after",
        "require_same_anchor",
        "timing.require_same_anchor",
        "use_same_anchor",
        "timing.use_same_anchor",
    }

    for relationship in _iter_sequence_relationships(sequence):
        kind = _relationship_kind_for_export(relationship)
        name = str(_relationship_get(relationship, "name", "") or "")
        set_property = str(_relationship_get(relationship, "set_property", "") or "")
        validation_only = bool(_relationship_get(relationship, "validation_only", False))

        is_timing_relation = (
            kind in timing_kinds
            or "center_after" in kind
            or "anchor_after" in kind
            or "require_same_anchor" in kind
            or "use_same_anchor" in kind
            or "center_after" in name
            or "requires_adc" in name
            or set_property in {"delay", "tstart", "start"}
            or validation_only
        )

        if not is_timing_relation:
            continue

        target = _relationship_endpoint_for_export(relationship, "target")
        if not _relationship_event_matches(target, target_event):
            continue

        reference = _relationship_endpoint_for_export(relationship, "reference")
        resolved = _relationship_resolved_mapping_for_export(relationship)

        return {
            "relationship": relationship,
            "kind": kind,
            "name": name,
            "target": target_event,
            "reference": reference,
            "target_anchor": _relationship_get(
                relationship,
                "target_anchor",
                resolved.get("target_anchor", "center"),
            ),
            "reference_anchor": _relationship_get(
                relationship,
                "reference_anchor",
                resolved.get("reference_anchor", "center"),
            ),
            "offset": _relationship_get(
                relationship,
                "offset",
                resolved.get("offset", resolved.get("offset_parameter", 0.0)),
            ),
            "set_property": set_property or resolved.get("set_property"),
            "validation_only": validation_only,
        }

    return None


def _relationship_endpoint_for_export(relationship: Any, role: str) -> Any:
    """Return a target/reference endpoint object or event-name string."""

    object_key = f"{role}_object"
    endpoint = _relationship_get(relationship, object_key, None)
    if endpoint is not None:
        return endpoint

    # Object-id lookup is not available inside the writer, but object ids remain
    # useful debug metadata. Fall through to event names when only ids exist.
    endpoint = _relationship_get(relationship, role, None)

    if isinstance(endpoint, Mapping):
        for key in ("event", "name", "id"):
            value = endpoint.get(key)
            if value is not None:
                return value
        return endpoint

    if endpoint is not None:
        return endpoint

    # Some older relationship representations store event-like names directly
    # in metadata.
    for key in (f"{role}_event", f"{role}_name", f"{role}_id"):
        value = _relationship_get(relationship, key, None)
        if value is not None:
            return value

    return None

def _relationship_get(relationship: Any, key: str, default: Any = None) -> Any:
    """Get a field from a relationship object, dict, or nested metadata."""

    if isinstance(relationship, Mapping):
        if key in relationship:
            return relationship[key]
        metadata = relationship.get("metadata")
        if isinstance(metadata, Mapping) and key in metadata:
            return metadata[key]
        resolved = relationship.get("resolved")
        if isinstance(resolved, Mapping) and key in resolved:
            return resolved[key]
        return default

    value = getattr(relationship, key, default)
    if value is not default:
        return value

    metadata = getattr(relationship, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]

    resolved = getattr(relationship, "resolved", None)
    if isinstance(resolved, Mapping) and key in resolved:
        return resolved[key]

    return default


def _relationship_event_matches(candidate: Any, event: Any) -> bool:
    """Return True if a relationship endpoint appears to refer to an event.

    The writer often receives relationship endpoints that refer to prototype
    objects, while compact motif export may operate on copied/frozen occurrences
    such as ``adc_readout_occ001``. Therefore matching uses identity, explicit
    ids, source/prototype aliases, and normalized event names.
    """

    if candidate is event:
        return True

    if candidate is None or event is None:
        return False

    # Some relationships store a weakref/callable wrapper.
    if callable(candidate):
        try:
            return candidate() is event
        except Exception:
            pass

    candidate_object_id = id(candidate) if not isinstance(candidate, str) else None
    event_alias_ids = set(_event_alias_ids_for_export(event))
    if candidate_object_id is not None and candidate_object_id in event_alias_ids:
        return True

    candidate_alias_ids = set(_event_alias_ids_for_export(candidate))
    if candidate_alias_ids and (
        id(event) in candidate_alias_ids
        or candidate_alias_ids.intersection(event_alias_ids)
    ):
        return True

    candidate_id = getattr(candidate, "id", None)
    event_id = getattr(event, "id", None)
    if candidate_id is not None and event_id is not None and candidate_id == event_id:
        return True

    candidate_name = candidate if isinstance(candidate, str) else getattr(candidate, "name", None)
    event_name_value = getattr(event, "name", None)
    if candidate_name and event_name_value:
        if str(candidate_name) == str(event_name_value):
            return True
        if _normalized_event_name_for_export(candidate_name) == _normalized_event_name_for_export(event_name_value):
            return True

    if isinstance(candidate, str):
        if event_id and candidate == str(event_id):
            return True

    return False


def _event_alias_ids_for_export(event: Any) -> list[int]:
    """Return source/prototype object ids associated with an exported event."""

    out: list[int] = []
    for container in (getattr(event, "parameters", None), getattr(event, "metadata", None)):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "source_event_object_id",
            "source_object_id",
            "original_event_object_id",
            "original_object_id",
            "seqstar_source_object_id",
        ):
            value = container.get(key)
            if isinstance(value, int):
                out.append(value)

    seen: set[int] = {id(event)}
    unique: list[int] = []
    for value in out:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _normalized_event_name_for_export(value: Any) -> str:
    """Normalize copied occurrence suffixes for endpoint matching."""

    text = str(value or "")
    # Match names such as adc_readout_occ001 back to adc_readout.
    text = re.sub(r"_occ\d+$", "", text)
    return text


def _anchor_fraction(anchor: Any) -> float:
    """Convert start/center/end-style anchor names to a duration fraction."""

    text = str(anchor or "start").lower()
    if text in {"center", "centre", "middle", "mid"}:
        return 0.5
    if text in {"end", "finish", "tend", "stop"}:
        return 1.0
    return 0.0


def _build_protocol_ui_specification(protocol_values: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a generic gammaSTAR Protocol Manager specification.

    gammaSTAR expects ``root.prot.ui_specification`` to be a JSON string.  The
    website Protocol Manager reads that JSON and creates editable controls from
    entries that point to ``root.prot.*`` parameters.

    This implementation is deliberately sequence-general.  It does not assume
    FID, FLASH, GRE, EPI, ASL, pCASL, spectroscopy, or any other sequence.  It
    only recognizes common protocol *parameter names* such as TE, TR, flip angle,
    averages, ADC samples, and readout duration.  Unknown scalar parameters are
    still exposed under a generic PyPulseq-Star group.
    """

    specs: list[dict[str, Any]] = []

    preferred_order = [
        "Name",
        "sequence_type",
        "seq_dim",
        "orientation",
        "slice_orientation",
        "TE",
        "TR",
        "minimal_TE",
        "average",
        "averages",
        "repetitions",
        "flip_angle",
        "rf_duration",
        "adc_delay",
        "adc_samples",
        "adc_dwell",
        "readout_duration",
        "echo_spacing",
        "echo_train_length",
        "trajectory",
        "gradient_channel",
        "gradient_kind",
        "gradient_duration",
        "gradient_area",
        "gradient_max_abs_amplitude",
        "num_gradient_events",
        "PAT_mode",
        "PAT_factor_phase",
        "PAT_factor_slice",
        "PAT_ref_lines_phase",
        "PAT_ref_lines_slice",
        "phase_partial_fourier",
        "read_partial_fourier",
        "slice_partial_fourier",
        "offcenter_exc_1",
        "orientation_exc_1",
        "thickness_exc_1",
    ]

    keys = [key for key in preferred_order if key in protocol_values]
    keys.extend(key for key in protocol_values if key not in keys)

    seen_paths: set[str] = set()

    for key in keys:
        value = protocol_values.get(key)
        spec = _protocol_ui_entry(key, value)

        if spec is None:
            continue

        path = str(spec.get("path", ""))
        if path in seen_paths:
            continue

        specs.append(spec)
        seen_paths.add(path)

    return specs


def _protocol_ui_entry(key: str, value: Any) -> dict[str, Any] | None:
    """Return one gammaSTAR UI entry for a protocol value.

    The fields mirror the FLASH reference protocol schema: description, groups,
    visibility/read-only functions, sources, min/max/step, unit/unit_scaling,
    display name, value type, options, and the target parameter path.
    """

    gamma_key = _protocol_key(key)
    path = f"root.prot.{gamma_key}"
    lowered = key.lower()

    # ``seqstar_*`` values are derived/export bookkeeping. They remain in the
    # protocol graph so relationships can use them, but they must not become
    # independent editable controls that can disagree with their source values.
    if lowered.startswith("seqstar_"):
        return None

    # Symbolic/derived protocol values are displayed through their source
    # controls rather than exposed as competing editable fields.
    if _contains_protocol_expression(value):
        return None

    # Vectors/matrices are valid protocol parameters. Only expose the most
    # common geometry vectors in the UI, because arbitrary lists are hard to edit
    # safely in the generic website Protocol Manager.
    if isinstance(value, (list, tuple)):
        if lowered in {"offcenter_exc_1", "orientation_exc_1", "rot_matrix"}:
            return _protocol_ui_dict(
                key=key,
                path=path,
                description=f"Generic PyPulseq-Star geometry parameter: {key}",
                groups=[["Geometry", "Positioning"]],
                value_type="float",
                value_dimensions="[3,3]" if lowered in {"orientation_exc_1", "rot_matrix"} else "[3]",
                minimum=-10000 if lowered == "offcenter_exc_1" else -1,
                maximum=10000 if lowered == "offcenter_exc_1" else 1,
                step=None,
                unit="mm" if lowered == "offcenter_exc_1" else None,
                unit_scaling=0.001 if lowered == "offcenter_exc_1" else None,
            )
        return None

    value_type = _protocol_value_type(value)
    if value_type is None:
        return None

    # Standard-plane orientation. Use a drop-down instead of a generic string:
    # the gammaSTAR Protocol Manager reliably propagates drop-down changes into
    # dependent relationships, whereas free-text string controls can appear to
    # edit visually without invalidating all downstream plot fields.
    if lowered == "orientation":
        return _protocol_ui_dict(
            key=key,
            path="root.prot.orientation",
            name="Orientation",
            description=(
                "Logical read/phase/slice orientation preset used by "
                "PyPulseq-Star and the gammaSTAR plot relationships."
            ),
            groups=[["PyPulseq-Star", "Protocol"]],
            value_type="drop_down",
            options=[
                {"label": "Axial", "value": "axial"},
                {"label": "Coronal", "value": "coronal"},
                {"label": "Sagittal", "value": "sagittal"},
            ],
        )

    if lowered in {"phase_encode_order", "phase_order"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            name="Phase-encode order",
            description="Ordering used to assign phase-encode lines to echoes.",
            groups=[["PyPulseq-Star", "Protocol"]],
            value_type="drop_down",
            options=[
                {"label": "Linear", "value": "linear"},
                {"label": "Centric", "value": "centric"},
                {"label": "Reverse", "value": "reverse"},
            ],
        )

    # Historical/gammaSTAR geometry name. Hide it from the generic
    # PyPulseq-Star panel to avoid two editable orientation controls. The
    # writer below keeps root.prot.slice_orientation aliased to
    # root.prot.orientation for compatibility with existing documents.
    if lowered == "slice_orientation":
        return None

    # Common timing parameters.  Match the reference style: user-facing ms/us
    # with unit_scaling back to seconds in root.prot.*.
    if lowered in {"te"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=(
                "The echo time (TE) is the time between the center of the RF "
                "excitation and the acquisition of the k-space center."
            ),
            groups=[["Contrast", "Timing"]],
            value_type="float",
            minimum=0,
            maximum=3000,
            step=0.1,
            unit="ms",
            unit_scaling=0.001,
        )

    if lowered in {"tr", "repetition_time", "repeat_period", "period"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            name="Repetition Time" if lowered != "tr" else "Repetition Time TR",
            description=(
                "The repetition time (TR) is the time between consecutive "
                "repetitions of the sequence kernel."
            ),
            groups=[["Contrast", "Timing"]],
            value_type="float",
            minimum=0,
            maximum=10000,
            step=0.1,
            unit="ms",
            unit_scaling=0.001,
        )

    if lowered in {"rf_duration", "adc_delay", "readout_duration", "echo_spacing", "gradient_duration"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star timing parameter: {key}",
            groups=[["Contrast", "Timing"]],
            value_type="float",
            minimum=0,
            maximum=1000000,
            step=1,
            unit="us",
            unit_scaling=1e-6,
        )

    if "dwell" in lowered or "sample_time" in lowered:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star sample timing parameter: {key}",
            groups=[["Contrast", "Timing"]],
            value_type="float",
            minimum=0,
            maximum=100000,
            step=0.1,
            unit="us",
            unit_scaling=1e-6,
        )

    # Common RF/acquisition controls.
    if "flip_angle" in lowered:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="RF excitation flip angle.",
            groups=[["Contrast", "Excitation"]],
            value_type="int" if isinstance(value, int) else "float",
            minimum=0,
            maximum=360,
            step=1,
            unit="°",
            unit_scaling=None,
        )

    if lowered in {"average", "averages", "repetitions", "n_avg", "num_averages"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="Number of averages or repetitions.",
            groups=[["Special"]],
            value_type="int",
            minimum=1,
            maximum=128,
            step=1,
        )

    if lowered in {"adc_samples", "num_samples", "number_of_samples"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="Number of ADC samples.",
            groups=[["Contrast", "Readout"]],
            value_type="int",
            minimum=1,
            maximum=10000000,
            step=1,
        )

    if lowered in {"echo_train_length", "turbo_factor"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="Number of echoes/readout windows in the train.",
            groups=[["Contrast", "Readout"]],
            value_type="int",
            minimum=1,
            maximum=100000,
            step=1,
        )

    if lowered == "pat_mode":
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="Parallel acquisition mode.",
            groups=[["PAT", "Acceleration"]],
            value_type="drop_down",
            options=[
                {"label": "None", "value": "None"},
                {"label": "GRAPPA Prescan", "value": "grappa_prescan"},
                {"label": "GRAPPA Integrated", "value": "grappa_integrated"},
            ],
        )

    if lowered.startswith("pat_"):
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Parallel acquisition parameter: {key}",
            groups=[["PAT", "Acceleration"]],
            value_type="int" if isinstance(value, int) else value_type,
            minimum=1 if isinstance(value, int) else None,
            maximum=128 if isinstance(value, int) else None,
            step=1 if isinstance(value, int) else None,
        )

    if lowered in {"sequence_type", "trajectory", "gradient_channel", "gradient_kind", "name"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star metadata parameter: {key}",
            groups=[["PyPulseq-Star", "Metadata"]],
            value_type="string",
            readonly=True,
        )

    if lowered == "seq_dim":
        return _protocol_ui_dict(
            key=key,
            path=path,
            description="Sequence dimensionality.",
            groups=[["Special"]],
            value_type="drop_down",
            options=[
                {"label": "2D", "value": "2D"},
                {"label": "3D", "value": "3D"},
            ],
        )

    if value_type == "bool":
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star boolean parameter: {key}",
            groups=[["PyPulseq-Star", "Protocol"]],
            value_type="bool",
        )

    if value_type in {"int", "float"}:
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star protocol parameter: {key}",
            groups=[["PyPulseq-Star", "Protocol"]],
            value_type=value_type,
            minimum=0 if value_type in {"int", "float"} else None,
            maximum=1000000 if value_type in {"int", "float"} else None,
            step=1 if value_type == "int" else 1e-6,
            unit=_protocol_unit(key),
            unit_scaling=_protocol_unit_scaling(key),
        )

    if value_type == "string":
        return _protocol_ui_dict(
            key=key,
            path=path,
            description=f"Generic PyPulseq-Star protocol parameter: {key}",
            groups=[["PyPulseq-Star", "Protocol"]],
            value_type="string",
        )

    return None


def _protocol_ui_dict(
    *,
    key: str,
    path: str,
    description: str,
    groups: list[list[str]],
    value_type: str,
    name: str | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    unit: str | None = None,
    unit_scaling: int | float | None = None,
    value_dimensions: str | None = None,
    options: list[dict[str, Any]] | None = None,
    readonly: bool = False,
    visible_function: str | None = None,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return one Protocol Manager UI specification dictionary."""

    return {
        "description": description,
        "groups": groups,
        "is_visible_function": visible_function,
        "is_readonly_function": "return true" if readonly else None,
        "check_and_fix_function": None,
        "check_strategy": None,
        "sources": sources or {},
        "max": maximum,
        "min": minimum,
        "step": step,
        "unit": unit,
        "unit_scaling": unit_scaling,
        "name": name or _pretty_protocol_name(key),
        "value_dimensions": value_dimensions,
        "value_type": value_type,
        "options": options,
        "path": path,
    }


def _protocol_value_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return None


def _protocol_unit(key: str) -> str | None:
    lowered = key.lower()
    if lowered in {"te", "tr", "repetition_time", "period", "repeat_period"}:
        return "ms"
    if lowered in {"rf_duration", "adc_delay", "readout_duration", "echo_spacing", "gradient_duration"}:
        return "us"
    if "dwell" in lowered or "sample_time" in lowered:
        return "us"
    if "flip_angle" in lowered:
        return "°"
    return None


def _protocol_unit_scaling(key: str) -> float | None:
    lowered = key.lower()
    if lowered in {"te", "tr", "repetition_time", "period", "repeat_period"}:
        return 0.001
    if lowered in {"rf_duration", "adc_delay", "readout_duration", "echo_spacing", "gradient_duration"}:
        return 1e-6
    if "dwell" in lowered or "sample_time" in lowered:
        return 1e-6
    return None


def _pretty_protocol_name(key: str) -> str:
    special = {
        "TE": "Echo Time",
        "TR": "Repetition Time",
        "Name": "Name",
        "sequence_type": "Sequence Type",
        "seq_dim": "Sequence Dimension",
        "minimal_TE": "Minimal TE",
        "average": "Averages",
        "averages": "Averages",
        "repetitions": "Repetitions",
        "flip_angle": "Flip Angle",
        "rf_duration": "RF Duration",
        "adc_delay": "ADC Delay",
        "adc_samples": "ADC Samples",
        "adc_dwell": "ADC Dwell",
        "readout_duration": "Readout Duration",
        "echo_spacing": "Echo Spacing",
        "echo_train_length": "Echo Train Length",
    }
    if key in special:
        return special[key]
    return str(key).replace("_", " ").strip().title()


def _adc_container_name(adc_event: Any, event_index: int) -> str:
    """Return a stable, gammaSTAR-safe ADC container name.

    This deliberately avoids sequence-specific naming. If the event has a
    usable name, preserve it. Otherwise use adc, adc2, adc3, ...
    """

    raw = str(getattr(adc_event, "name", "") or "")

    if raw:
        cleaned = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in raw.lower()
        )
        cleaned = cleaned.strip("_")

        if cleaned and cleaned[0].isalpha():
            return cleaned

    return "adc" if event_index == 0 else f"adc{event_index + 1}"


def _gradient_container_name(gradient_event: Any, event_index: int) -> str:
    """Return a stable, gammaSTAR-safe gradient container name."""

    raw = str(getattr(gradient_event, "name", "") or "")
    if raw:
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw.lower())
        cleaned = cleaned.strip("_")
        if cleaned and cleaned[0].isalpha():
            return cleaned
    return "grad" if event_index == 0 else f"grad{event_index + 1}"


def _gradient_direction(channel: str | None) -> list[float]:
    """Return gammaSTAR direction vector for x/y/z channel."""

    channel = str(channel or "x").lower()
    if channel == "x":
        return [1.0, 0.0, 0.0]
    if channel == "y":
        return [0.0, 1.0, 0.0]
    if channel == "z":
        return [0.0, 0.0, 1.0]
    return [1.0, 0.0, 0.0]


def _gradient_gamma_hz_per_t(gradient_event: Any, data: Mapping[str, Any] | None = None) -> float:
    """Return the gradient gyromagnetic ratio in Hz/T for unit conversion."""

    data = data or {}

    for key in ("gamma_hz_per_t", "gamma", "gamma_Hz_per_T"):
        value = data.get(key)
        if value is not None:
            try:
                gamma = float(value)
                if gamma > 1e5:
                    return gamma
            except Exception:
                pass

    owners = [
        gradient_event,
        getattr(gradient_event, "system", None),
        getattr(gradient_event, "opts", None),
        getattr(gradient_event, "shape", None),
    ]

    for owner in owners:
        if owner is None:
            continue
        for key in ("gamma", "gamma_hz_per_t", "gamma_Hz_per_T"):
            value = getattr(owner, key, None)
            if value is not None:
                try:
                    gamma = float(value)
                    if gamma > 1e5:
                        return gamma
                except Exception:
                    pass

    # Pulseq/clinical proton default in Hz/T. This fallback should be used only
    # when no system object is reachable from the event.
    return 42_575_575.0


def _normalize_gradient_data(data: Mapping[str, Any], *, gradient_event: Any) -> dict[str, Any]:
    """Normalize gradient serialization into gammaSTAR GradPulse payload data.

    SeqStar/PyPulseq gradient waveforms are internally represented in
    frequency-normalized units (Hz/m), following the Pulseq convention. Native
    gammaSTAR GradPulse samples are interpreted as physical gradient amplitudes
    in T/m; the website then labels the plot in mT/m. Therefore the export
    boundary must divide the waveform by gamma exactly once.

    The internal ``area`` is intentionally left in k-space units (1/m), because
    loop variation tables and phase-encoding formulas are expressed in those
    same Pulseq-style units.
    """

    data_dict = dict(data)
    if str(data_dict.get("kind", "")).lower() == "split":
        raise ValueError(
            "Split-gradient containers should be expanded into their ramp/flat/ramp "
            "events before gammaSTAR writing. Pass the tuple returned by split_gradient "
            "to seq.add_block(*parts) or add each part to the sequence."
        )

    shape = getattr(gradient_event, "shape", None)
    channel = str(data_dict.get("channel", getattr(gradient_event, "channel", "x"))).lower()
    kind = str(data_dict.get("kind", getattr(gradient_event, "type", "grad"))).lower()
    role = data_dict.get("role", getattr(gradient_event, "role", None))

    tstart = float(data_dict.get("tstart", getattr(gradient_event, "delay", 0.0)))
    active_duration = float(
        data_dict.get(
            "active_duration",
            getattr(gradient_event, "active_duration", data_dict.get("duration", 0.0)),
        )
    )
    total_duration = tstart + active_duration

    if kind == "trap" and all(
        hasattr(shape, attr) for attr in ("rise_time", "flat_time", "fall_time", "amplitude")
    ):
        rut = float(getattr(shape, "rise_time"))
        ft = float(getattr(shape, "flat_time"))
        rdt = float(getattr(shape, "fall_time"))
        amp = float(getattr(shape, "amplitude"))
        raw_samples = {"t": [0.0, rut, rut + ft, rut + ft + rdt], "v": [0.0, amp, amp, 0.0]}
    else:
        tt = data_dict.get("tt")
        wf = data_dict.get("waveform")
        if tt is None and shape is not None and hasattr(shape, "time_axis"):
            tt = shape.time_axis().tolist()
        if wf is None and shape is not None and hasattr(shape, "waveform"):
            wf = shape.waveform().tolist()
        if tt is None or wf is None:
            raise ValueError(f"Cannot export gradient waveform for event {gradient_event!r}.")
        raw_samples = {"t": [float(x) for x in tt], "v": [float(x) for x in wf]}

    raw_values = [float(v) for v in raw_samples.get("v", [])]
    gamma_hz_per_t = _gradient_gamma_hz_per_t(gradient_event, data_dict)

    unit = str(
        data_dict.get("waveform_unit")
        or data_dict.get("gradient_unit")
        or data_dict.get("unit")
        or "Hz/m"
    ).strip().lower()

    already_physical = (
        "t/m" in unit
        or "tesla" in unit
        or unit in {"t", "mt/m", "mtesla/m"}
    )

    if already_physical:
        physical_values = raw_values
    else:
        physical_values = [value / gamma_hz_per_t for value in raw_values]

    samples = {
        "t": list(raw_samples.get("t", [])),
        "v": physical_values,
    }

    area = data_dict.get("area", getattr(gradient_event, "area", None))
    if area is not None:
        area = float(area)

    direction = data_dict.get(
        "physical_direction",
        getattr(gradient_event, "physical_direction", None),
    )
    if direction is None:
        direction = _gradient_direction(channel)
    else:
        direction = [float(value) for value in list(direction)]

    return {
        "kind": kind,
        "channel": channel,
        "role": role,
        "tstart": tstart,
        "active_duration": active_duration,
        "duration": total_duration,
        "area": area,
        "samples": samples,
        "samples_internal_hz_per_m": raw_samples,
        "samples_unit": "T/m",
        "samples_source_unit": unit,
        "direction": direction,
        "enabled": bool(data_dict.get("enabled", getattr(gradient_event, "enabled", True))),
        "max_abs_amplitude": max((abs(v) for v in physical_values), default=0.0),
        "max_abs_amplitude_unit": "T/m",
    }


def _event_repetition_variants(event: Any) -> list[Any]:
    """Return representative-event variants attached by compact export."""

    variants = getattr(event, "_seqstar_repetition_variants", None)
    if isinstance(variants, list) and variants:
        return variants
    return [event]


def _event_loop_counter_path(event: Any, *, default: str = "root.average.counter") -> str:
    """Return the gammaSTAR loop counter path for compact variants."""

    value = getattr(event, "_seqstar_loop_counter_path", None)
    if value:
        return str(value)
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("seqstar_loop_counter_path")
        if value:
            return str(value)
    return default


def _event_variant_policy(event: Any) -> str:
    """Return the compact-loop variant policy for an event.

    This is a writer-side guard for gammaSTAR compact exports. It is deliberately
    generic and can be set on any event through an attribute, parameters, or
    metadata. Supported values:

        auto      Default. Vary only when variants differ and no guard applies.
        vary      Force loop-indexed variant tables when variants differ.
        variant   Synonym for vary.
        table     Synonym for vary.
        constant  Export the representative event literally.
        literal   Synonym for constant.
        none/off  Synonym for constant.

    The helper also follows common wrapper/proxy links used by the writer.
    """

    keys = (
        "gammastar_variant_policy",
        "variant_policy",
        "loop_variant_policy",
        "export_variant_policy",
        "seqstar_variant_policy",
    )

    for key in keys:
        value = getattr(event, key, None)
        if value is not None:
            return str(value).strip().lower()

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in keys:
            value = parameters.get(key)
            if value is not None:
                return str(value).strip().lower()

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return str(value).strip().lower()

    # Follow the writer proxy/base links, if present.
    for attr in ("_base", "_event", "_source_event", "source_event"):
        source = getattr(event, attr, None)
        if source is not None and source is not event:
            policy = _event_variant_policy(source)
            if policy != "auto":
                return policy

    return "auto"


def _event_text_tags(event: Any) -> set[str]:
    """Return normalized textual tags from event/block metadata.

    Tags are used only for generic writer guards. They intentionally do not
    inspect sequence names.
    """

    tags: set[str] = set()
    keys = (
        "name",
        "role",
        "axis_role",
        "encoding_role",
        "use",
        "label",
        "source_event_name",
        "source_block_role",
        "block_role",
    )

    for key in keys:
        value = getattr(event, key, None)
        if value is not None:
            tags.add(_normalize_signature_token(str(value)))

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in keys:
            value = parameters.get(key)
            if value is not None:
                tags.add(_normalize_signature_token(str(value)))

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                tags.add(_normalize_signature_token(str(value)))

    for attr in ("_base", "_event", "_source_event", "source_event"):
        source = getattr(event, attr, None)
        if source is not None and source is not event:
            tags.update(_event_text_tags(source))

    return {tag for tag in tags if tag}


def _is_spoiler_like_event(event: Any) -> bool:
    """Return True for spoiler/spoiling-like events or blocks.

    In compact repeated exports, a spoiler is normally a constant dephasing
    moment. If a user really wants a spoiler to vary, they can set
    ``gammastar_variant_policy='vary'``. This is an MRI-event semantic guard,
    not a GRE-specific branch.
    """

    tags = _event_text_tags(event)
    return any(
        tag == "spoil"
        or tag == "spoiler"
        or tag == "spoiling"
        or tag.endswith("_spoil")
        or tag.endswith("_spoiler")
        or tag.endswith("_spoiling")
        or "spoiler" in tag
        or "spoiling" in tag
        for tag in tags
    )


def _retarget_export_event_loop_counter_paths(
    events: Iterable[Any],
    *,
    loop_counter_path: str,
) -> None:
    """Retarget compacted event variants to the active gammaSTAR loop.

    Motif compaction happens before the final loop path is known. Earlier
    generic code used ``root.seqstar_loop.counter`` as a placeholder. Once the
    active loop has been chosen, for example ``root.n_y`` or ``root.average``,
    every variant-producing event must point to that real counter path.

    This is sequence-family agnostic: it updates only stale placeholder
    counter references and leaves explicit user/event counter paths untouched.
    """

    for event in events:
        current = getattr(event, "_seqstar_loop_counter_path", None)
        if current in (None, "root.seqstar_loop.counter"):
            try:
                setattr(
                    event,
                    "_seqstar_loop_counter_path",
                    loop_counter_path,
                )
            except Exception:
                pass

        for container_name in ("metadata", "parameters"):
            container = getattr(event, container_name, None)
            if not isinstance(container, dict):
                continue

            current = container.get("seqstar_loop_counter_path")
            if current in (None, "root.seqstar_loop.counter"):
                container["seqstar_loop_counter_path"] = loop_counter_path



def _adc_window_value(
    window: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read one ADC-window field from either an object or mapping."""

    if isinstance(window, Mapping):
        return window.get(key, default)

    return getattr(window, key, default)



def _get_adc_windows_for_export(event: Any) -> list[Any]:
    """Return the concrete windows from a multi-window ADC event."""

    for owner in (event, getattr(event, "shape", None)):
        if owner is None:
            continue

        windows = getattr(owner, "windows", None)
        if windows is not None:
            try:
                return list(windows)
            except TypeError:
                pass

        method = getattr(owner, "to_pulseq_windows", None)
        if callable(method):
            try:
                return list(method())
            except Exception:
                pass

    return []



def _adc_train_metadata_value(
    event: Any,
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Return one generic ADC-train metadata value.

    Values may live directly in ``parameters``/``metadata`` or in a nested
    ``adc_train`` mapping. No sequence names or sequence-family assumptions are
    used.
    """

    for container_name in ("parameters", "metadata"):
        container = getattr(event, container_name, None)
        if not isinstance(container, Mapping):
            continue

        if key in container:
            return container[key]

        nested = container.get("adc_train")
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]

    return default



def _has_multiple_adc_windows_for_export(event: Any) -> bool:
    """Return True when an ADC event contains more than one acquisition window."""

    candidates = [
        getattr(event, "windows", None),
        getattr(getattr(event, "shape", None), "windows", None),
    ]

    for windows in candidates:
        if windows is None:
            continue
        try:
            return len(windows) > 1
        except TypeError:
            try:
                return len(list(windows)) > 1
            except TypeError:
                continue

    for owner in (event, getattr(event, "shape", None)):
        if owner is None:
            continue
        method = getattr(owner, "to_pulseq_windows", None)
        if callable(method):
            try:
                return len(method()) > 1
            except Exception:
                continue

    return False



def _gradient_variant_table(gradient_event: Any) -> dict[str, Any] | None:
    """Return per-loop gradient variant tables for compact repeated export.

    Writer-side guards
    ------------------
    1. Explicit event policy wins. ``constant/literal/none/off`` disables
       loop-indexed tables; ``vary/variant/table`` forces them if variants
       differ.
    2. In the default ``auto`` mode, spoiler-like gradients are exported as
       constants. This prevents a phase-encode rewinder or mutable event reuse
       from accidentally turning a nominal spoiler into a loop-varying object.
    3. Fully identical gradients are always exported literally.
    """

    policy = _event_variant_policy(gradient_event)
    if policy in {"constant", "literal", "none", "off", "fixed", "static"}:
        return None

    if policy == "auto" and _is_spoiler_like_event(gradient_event):
        return None

    variants = _event_repetition_variants(gradient_event)
    if len(variants) <= 1:
        return None

    samples_table: list[Any] = []
    area_table: list[Any] = []
    max_amp_table: list[Any] = []
    enabled_table: list[Any] = []

    for variant in variants:
        if hasattr(variant, "to_gammastar_waveform"):
            raw = variant.to_gammastar_waveform()
        else:
            raw = _fallback_gradient_waveform_data(variant)
        data = _normalize_gradient_data(raw, gradient_event=variant)
        samples_table.append(data.get("samples"))
        area_table.append(data.get("area"))
        max_amp_table.append(data.get("max_abs_amplitude", 0.0))
        enabled_table.append(bool(data.get("enabled", True)))

    # Avoid creating tables for fully identical gradients. This keeps compact
    # exports small and leaves non-varying RF/Gx/Gz/spoilers literal.
    if all(item == samples_table[0] for item in samples_table) and all(item == area_table[0] for item in area_table):
        return None

    # In strict constant mode the guard returned above. In auto mode, only
    # non-spoiler varying gradients become loop-indexed. Explicit vary/variant
    # reaches this point and is therefore emitted as a table.
    return {
        "samples": samples_table,
        "area": area_table,
        "max_abs_amplitude": max_amp_table,
        "enabled": enabled_table,
        "counter_path": _event_loop_counter_path(gradient_event, default="root.seqstar_loop.counter"),
    }


def _fallback_gradient_waveform_data(gradient_event: Any) -> dict[str, Any]:
    """Build gradient waveform data from a generic gradient-like event."""

    shape = getattr(gradient_event, "shape", None)
    source = shape if shape is not None else gradient_event
    if hasattr(source, "waveform"):
        waveform = source.waveform()
    elif hasattr(source, "waveform_samples"):
        waveform = getattr(source, "waveform_samples")
    else:
        raise ValueError(f"Gradient event {gradient_event!r} does not expose a waveform.")

    if hasattr(source, "time_axis"):
        tt = source.time_axis()
    else:
        raster = float(getattr(source, "grad_raster_time", 10e-6))
        tt = [(i + 0.5) * raster for i in range(len(waveform))]

    return {
        "kind": getattr(gradient_event, "type", getattr(source, "kind", "grad")),
        "channel": getattr(gradient_event, "channel", getattr(source, "channel", "x")),
        "tstart": getattr(gradient_event, "delay", getattr(source, "delay", 0.0)),
        "duration": getattr(gradient_event, "duration", getattr(source, "duration", 0.0)),
        "active_duration": getattr(gradient_event, "active_duration", getattr(source, "active_duration", 0.0)),
        "area": getattr(gradient_event, "area", getattr(source, "area", None)),
        "waveform": list(waveform),
        "tt": list(tt),
        "role": getattr(gradient_event, "role", getattr(source, "role", None)),
    }


def _normalize_adc_data(data: Mapping[str, Any], *, adc_event: Any) -> dict[str, Any]:
    """Normalize ADC event/shape serialization into one dictionary.

    Timing convention
    -----------------
    SeqStar ADC objects may expose window timing in two common ways:

    1. event.delay/tstart carries the event-level start time, while each window
       tstart/delay is local to that event. This is the default for enriched ADC
       shapes/trains and is the most useful generic representation.
    2. window tstart/delay values are already absolute within the kernel.

    Unless the data explicitly marks windows as absolute, this normalizer treats
    window timing as local and adds the resolved event delay. This is what makes
    relationship-derived timing such as TE visible to gammaSTAR without adding
    any FID/GRE/EPI-specific code.
    """

    data_dict = dict(data)
    windows_raw = data_dict.get("windows")

    if windows_raw is None:
        windows_raw = []

    parent_delay = _event_delay(adc_event)
    if parent_delay < -1e-12:
        raise ValueError(
            f"ADC event delay must be non-negative; got {parent_delay}."
        )
    parent_delay = max(parent_delay, 0.0)

    windows_are_absolute = bool(
        data_dict.get("windows_are_absolute", False)
        or data_dict.get("absolute_timing", False)
        or data_dict.get("is_absolute_timing", False)
        or str(data_dict.get("timing_reference", "")).lower() == "absolute"
    )

    windows: list[dict[str, Any]] = []

    for index, window in enumerate(windows_raw):
        if hasattr(window, "to_dict"):
            window_dict = dict(window.to_dict())
        else:
            window_dict = dict(window)

        local_tstart = float(
            window_dict.get("tstart", window_dict.get("delay", 0.0)) or 0.0
        )
        if local_tstart < -1e-12:
            raise ValueError(
                f"ADC window {index} delay must be non-negative; "
                f"got {local_tstart}."
            )
        local_tstart = max(local_tstart, 0.0)
        tstart = local_tstart if windows_are_absolute else parent_delay + local_tstart

        num_samples = int(
            window_dict.get("num_samples", window_dict.get("number_of_samples", 0))
        )

        sample_time = window_dict.get("sample_time", window_dict.get("dwell"))
        duration = window_dict.get("duration")

        if sample_time is None and duration is None:
            raise ValueError(f"ADC window {index} must define sample_time/dwell or duration.")

        if sample_time is None:
            sample_time = float(duration) / num_samples

        if duration is None:
            duration = num_samples * float(sample_time)

        window_dict["index"] = int(window_dict.get("index", index))
        window_dict["local_tstart"] = local_tstart
        window_dict["local_delay"] = local_tstart
        window_dict["parent_delay"] = parent_delay
        window_dict["tstart"] = tstart
        window_dict["delay"] = tstart
        window_dict["num_samples"] = num_samples
        window_dict["number_of_samples"] = num_samples
        window_dict["sample_time"] = float(sample_time)
        window_dict["dwell"] = float(sample_time)
        window_dict["duration"] = float(duration)
        window_dict["tend"] = tstart + float(duration)
        window_dict.setdefault("role", "imaging")
        window_dict.setdefault("polarity", 1)
        window_dict.setdefault("trajectory", data_dict.get("trajectory"))

        windows.append(window_dict)

    if not windows:
        fallback = _fallback_single_adc_data(adc_event)
        return _normalize_adc_data(fallback, adc_event=adc_event)

    # Add timing coordinates relative to the first window. This lets relationship
    # expressions position an ADC/readout train once, while loop iterations add
    # only a per-window offset. The calculation is generic and does not depend
    # on GRE/EPI/FID-specific semantics.
    base_local = float(windows[0].get("local_tstart", windows[0].get("tstart", 0.0)))
    base_abs = float(windows[0].get("tstart", windows[0].get("delay", 0.0)))
    for item in windows:
        item_local = float(item.get("local_tstart", item.get("tstart", 0.0)))
        item_abs = float(item.get("tstart", item.get("delay", 0.0)))
        item["relative_tstart"] = item_local - base_local
        item["relative_delay"] = item_local - base_local
        item["relative_absolute_tstart"] = item_abs - base_abs

    duration = data_dict.get("duration")
    if duration is None or float(duration) <= 0:
        duration = max(window["tend"] for window in windows)
    elif not windows_are_absolute and parent_delay > 0:
        # Keep train duration as kernel extent when the source reported only a
        # local train duration.
        duration = parent_delay + float(duration)

    data_dict["windows"] = windows
    data_dict["duration"] = float(duration)
    data_dict["num_windows"] = int(data_dict.get("num_windows", len(windows)))
    data_dict["total_num_samples"] = int(
        data_dict.get(
            "total_num_samples",
            sum(window["num_samples"] for window in windows),
        )
    )
    data_dict.setdefault("mode", getattr(adc_event, "mode", "single"))
    data_dict.setdefault("trajectory", getattr(adc_event, "trajectory", None))
    data_dict.setdefault("kind", getattr(adc_event, "kind", "adc_train"))
    data_dict.setdefault("type", "adc")
    data_dict["parent_delay"] = parent_delay
    data_dict["timing_contract"] = "event_delay_plus_window_local_delay"
    data_dict["windows_are_absolute"] = True

    return data_dict


def _fallback_single_adc_data(adc_event: Any) -> dict[str, Any]:
    """Build ADC data from a simple single-window ADC-like event."""

    num_samples = int(
        _get_any(adc_event, keys=("num_samples", "number_of_samples", "adc_samples"), default=0)
    )
    if num_samples <= 0:
        raise ValueError(f"ADC event {adc_event!r} is missing num_samples.")

    dwell = _get_any(adc_event, keys=("dwell", "sample_time", "adc_dwell"), default=None)
    duration = _get_any(adc_event, keys=("duration", "adc_duration", "readout_duration"), default=None)

    if dwell is None and duration is None:
        raise ValueError(f"ADC event {adc_event!r} must define dwell/sample_time or duration.")

    if dwell is None:
        dwell = float(duration) / num_samples
    if duration is None:
        duration = num_samples * float(dwell)

    delay = float(_get_any(adc_event, keys=("delay", "tstart", "adc_delay"), default=0.0))

    return {
        "type": "adc",
        "kind": "adc_train",
        "mode": "single",
        "windows_are_absolute": True,
        "trajectory": _get_any(adc_event, keys=("trajectory",), default=None),
        "duration": delay + float(duration),
        "num_windows": 1,
        "total_num_samples": num_samples,
        "windows": [
            {
                "index": 0,
                "label": getattr(adc_event, "name", "adc"),
                "role": getattr(adc_event, "role", "imaging"),
                "tstart": delay,
                "delay": delay,
                "duration": float(duration),
                "tend": delay + float(duration),
                "num_samples": num_samples,
                "number_of_samples": num_samples,
                "sample_time": float(dwell),
                "dwell": float(dwell),
                "phase": float(_get_any(adc_event, keys=("phase", "phase_offset"), default=0.0)),
                "frequency": float(
                    _get_any(
                        adc_event,
                        keys=("frequency", "freq_offset", "frequency_offset"),
                        default=0.0,
                    )
                ),
                "center_sample": math.floor(0.5 * (num_samples - 1)),
                "trajectory": _get_any(adc_event, keys=("trajectory",), default=None),
            }
        ],
    }


def _adc_event_num_windows(adc_event: Any) -> int:
    """Return number of ADC windows for an event."""

    try:
        if hasattr(adc_event, "to_gammastar_windows"):
            data = adc_event.to_gammastar_windows(include_sample_times=False)
            return int(data.get("num_windows", len(data.get("windows", []))))
    except Exception:
        pass

    shape = getattr(adc_event, "shape", None)
    if shape is not None and hasattr(shape, "windows"):
        return len(shape.windows)

    windows = getattr(adc_event, "windows", None)
    if windows is not None:
        return len(windows)

    return 1




def _copy_block_export_metadata_to_mapping(block: Any, mapping: dict[str, Any]) -> None:
    """Copy SeqStar block hierarchy metadata into an event proxy mapping."""

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in (
            "seqstar_node",
            "seqstar_parent",
            "seqstar_local_name",
            "seqstar_role",
            "seqstar_varies",
            "seqstar_counter",
            "seqstar_repeat_parent",
            "seqstar_previous_block_node",
        ):
            if key in metadata:
                mapping.setdefault(f"source_block_{key}", metadata[key])

    node = _block_node_for_export(block, default=None)
    if node is not None:
        mapping.setdefault("source_block_node", node)
    parent = _block_parent_for_export(block, node)
    if parent is not None:
        mapping.setdefault("source_block_parent", parent)
    local = _block_local_name_for_export(block, node)
    if local is not None:
        mapping.setdefault("source_block_local_name", local)


def _block_node_for_export(block: Any, default: str | None = None) -> str | None:
    """Return a SeqStar block node path for writer/export use."""

    for attr in ("node", "path", "seqstar_node"):
        value = getattr(block, attr, None)
        if value:
            return str(value)
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("seqstar_node", "node", "path", "seqstar_path"):
            value = metadata.get(key)
            if value:
                return str(value)
    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("seqstar_node", "node", "path", "seqstar_path"):
            value = parameters.get(key)
            if value:
                return str(value)
    return default


def _block_parent_for_export(block: Any, node: str | None) -> str | None:
    """Return/infer block parent node."""

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("seqstar_parent") or metadata.get("parent")
        if value:
            return str(value)
    parent = getattr(block, "parent", None)
    if isinstance(parent, str) and parent:
        return parent
    if node and "." in str(node):
        return str(node).rsplit(".", 1)[0]
    return "sequence"


def _block_local_name_for_export(block: Any, node: str | None) -> str | None:
    """Return/infer block local name."""

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("seqstar_local_name") or metadata.get("local_name")
        if value:
            return str(value)
    if node:
        return str(node).rsplit(".", 1)[-1]
    value = getattr(block, "name", None)
    return str(value) if value else None


def _safe_path_token(value: Any) -> str:
    """Return a gammaSTAR-path-safe token for logical metadata nodes."""

    text = str(value or "node")
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "node"


def _event_source_block_index_for_export(event: Any) -> int | None:
    """Return source block index attached by _TimedEventProxy."""

    for attr in ("_block_index", "source_block_index"):
        value = getattr(event, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for container in (getattr(event, "parameters", None), getattr(event, "metadata", None)):
        if isinstance(container, Mapping):
            value = container.get("source_block_index")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _event_local_tstart_within_source_block(event: Any) -> float:
    """Return local event/container tstart inside its source block."""

    value = getattr(event, "_base_delay", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    # ADC proxies often expose local window starts in the generated windows. Use
    # base delay first, then fall back to public event delay.
    base = getattr(event, "_base", None)
    if base is not None:
        try:
            return float(_event_delay(base))
        except Exception:
            pass
    try:
        return float(_event_delay(event))
    except Exception:
        return 0.0


def _is_literal_parameter(parameter: Any) -> bool:
    """Return True when a gammaSTAR parameter has no dependencies."""

    if not isinstance(parameter, Mapping):
        return True
    inputs = parameter.get("inputs")
    return not isinstance(inputs, Mapping) or len(inputs) == 0

def _events_extent_duration(events: Iterable[Any]) -> float:
    """Return the maximum event extent needed by the kernel.

    calc_duration(block) can underestimate enriched ADC trains because a train
    may expose a representative one-window duration for PyPulseq compatibility.
    gammaSTAR kernel duration should cover the full semantic train.
    """

    extent = 0.0
    for event in events:
        try:
            if _is_adc_event(event):
                data = _normalize_adc_data_from_event(event)
                windows = data.get("windows", [])
                if windows:
                    extent = max(
                        extent,
                        max(
                            float(window.get("tstart", window.get("delay", 0.0)))
                            + float(window.get("duration", 0.0))
                            for window in windows
                        ),
                    )
                    continue
            if _is_gradient_event(event):
                grad_data = _normalize_gradient_data(
                    event.to_gammastar_waveform() if hasattr(event, "to_gammastar_waveform") else _fallback_gradient_waveform_data(event),
                    gradient_event=event,
                )
                extent = max(extent, float(grad_data.get("duration", 0.0)))
                continue
            extent = max(extent, _event_delay(event) + _event_active_duration(event))
        except Exception:
            try:
                extent = max(extent, calc_duration(event))
            except Exception:
                pass
    return extent


def _normalize_adc_data_from_event(adc_event: Any) -> dict[str, Any]:
    """Small standalone ADC normalizer used before the builder instance exists."""

    if hasattr(adc_event, "to_gammastar_windows"):
        return _normalize_adc_data(
            adc_event.to_gammastar_windows(include_sample_times=False),
            adc_event=adc_event,
        )

    if hasattr(adc_event, "to_gammastar_samples"):
        return _normalize_adc_data(
            adc_event.to_gammastar_samples(include_sample_times=False),
            adc_event=adc_event,
        )

    shape = getattr(adc_event, "shape", None)
    if shape is not None and hasattr(shape, "to_gammastar_windows"):
        return _normalize_adc_data(
            shape.to_gammastar_windows(include_sample_times=False),
            adc_event=adc_event,
        )

    if shape is not None and hasattr(shape, "to_gammastar_samples"):
        return _normalize_adc_data(
            shape.to_gammastar_samples(include_sample_times=False),
            adc_event=adc_event,
        )

    return _normalize_adc_data(_fallback_single_adc_data(adc_event), adc_event=adc_event)




def _explicit_repeated_node_info(
    sequence: Any,
    blocks: list[Any],
) -> dict[str, Any] | None:
    """Return one explicitly repeated outer logical-node motif.

    The source of truth is node metadata created by ``seq.set_node``. This
    function does not infer sequence type or assign meaning to node names.

    Only outermost repeated nodes are considered in this document-level pass.
    Nested repeated nodes remain available for a later recursive lowering pass.
    """

    registry = _sequence_node_registry_for_export(sequence)

    repeated_nodes: list[tuple[str, Mapping[str, Any]]] = []

    for name, record in registry.items():
        if not isinstance(record, Mapping):
            continue

        if (
            record.get("repeat_count") is None
            and record.get("repeat_every") is None
        ):
            continue

        repeated_nodes.append((str(name), record))

    if not repeated_nodes:
        return None

    repeated_nodes.sort(
        key=lambda item: (_node_depth_for_export(item[0]), item[0])
    )

    selected: list[tuple[str, Mapping[str, Any]]] = []

    for name, record in repeated_nodes:
        if any(
            _node_is_within_for_export(name, parent_name)
            for parent_name, _ in selected
        ):
            continue
        selected.append((name, record))

    for node_name, record in selected:
        owned_blocks = [
            block
            for block in blocks
            if _block_belongs_to_repeat_node_for_export(
                block,
                node_name,
            )
        ]

        if not owned_blocks:
            continue

        # The document builder currently emits one outer loop and one kernel.
        # Accept the repeated node only when it owns one contiguous timeline
        # motif.
        owned_indices = [
            index
            for index, block in enumerate(blocks)
            if block in owned_blocks
        ]

        if owned_indices != list(
            range(owned_indices[0], owned_indices[-1] + 1)
        ):
            raise ValueError(
                f"Repeated logical node {node_name!r} does not occupy one "
                "contiguous timeline range."
            )

        repeat_count_ref = record.get("repeat_count")
        repetitions = _resolve_repeat_value_for_export(
            sequence,
            repeat_count_ref,
            default=1,
        )
        repetitions = int(round(float(repetitions)))

        if repetitions < 1:
            raise ValueError(
                f"Repeated logical node {node_name!r} resolved to invalid "
                f"repeat_count={repetitions}."
            )

        repeat_every_ref = record.get("repeat_every")
        repeat_period = None

        if repeat_every_ref is not None:
            repeat_period = float(
                _resolve_repeat_value_for_export(
                    sequence,
                    repeat_every_ref,
                    default=0.0,
                )
            )

        loop_token = _repeat_loop_token_for_export(
            repeat_count_ref,
            record.get("counter"),
        )
        node_loop_token = _repeat_node_loop_token_for_export(
            node_name,
            record.get("counter"),
        )

        repeat_mode = str(record.get("repeat_mode") or "").strip().lower()
        if repeat_mode not in {"loop", "expanded"}:
            raise ValueError(
                f"Repeated node {node_name!r} requires repeat_mode='loop' or 'expanded'."
            )

        return {
            "node": node_name,
            "repeat_mode": repeat_mode,
            "record": dict(record),
            "blocks": owned_blocks,
            "repetitions": repetitions,
            "repeat_period": repeat_period,
            "loop_token": loop_token,
            "node_loop_token": node_loop_token,
        }

    return None


def _sequence_node_registry_for_export(
    sequence: Any,
) -> dict[str, Any]:
    """Return logical-node metadata from common sequence locations."""

    candidates = (
        getattr(sequence, "nodes", None),
        getattr(getattr(sequence, "timeline", None), "nodes", None),
        getattr(
            getattr(sequence, "timeline", None),
            "node_registry",
            None,
        ),
    )

    metadata = getattr(sequence, "metadata", None)

    if isinstance(metadata, Mapping):
        candidates = candidates + (
            metadata.get("seqstar_nodes"),
            metadata.get("nodes"),
            metadata.get("node_registry"),
        )

    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)

    return {}


def _block_belongs_to_repeat_node_for_export(
    block: Any,
    node_name: str,
) -> bool:
    """Return True when a concrete block belongs to a repeated node."""

    metadata = getattr(block, "metadata", None)
    block_node = None
    repeat_parent = None

    if isinstance(metadata, Mapping):
        block_node = (
            metadata.get("seqstar_node")
            or metadata.get("node")
            or metadata.get("path")
        )
        repeat_parent = (
            metadata.get("seqstar_repeat_parent")
            or metadata.get("repeat_parent")
        )

    if block_node is None:
        block_node = (
            getattr(block, "node", None)
            or getattr(block, "path", None)
            or getattr(block, "name", None)
        )

    if repeat_parent is None:
        repeat_parent = getattr(block, "repeat_parent", None)

    return (
        _node_is_within_for_export(str(block_node or ""), node_name)
        or str(repeat_parent or "") == node_name
        or _node_is_within_for_export(
            str(repeat_parent or ""),
            node_name,
        )
    )


def _resolve_repeat_value_for_export(
    sequence: Any,
    value: Any,
    *,
    default: Any,
) -> Any:
    """Resolve a literal or exact sequence-parameter reference."""

    if value is None:
        return default

    if not isinstance(value, str):
        return value

    parameters = getattr(sequence, "parameters", None)

    if isinstance(parameters, Mapping) and value in parameters:
        return parameters[value]

    metadata = getattr(sequence, "metadata", None)

    if isinstance(metadata, Mapping):
        for key in (
            "parameters",
            "protocol_parameters",
            "protocol",
        ):
            container = metadata.get(key)
            if isinstance(container, Mapping) and value in container:
                return container[value]

    protocol = getattr(sequence, "protocol", None)

    if protocol is not None:
        protocol_parameters = getattr(protocol, "parameters", None)

        if (
            isinstance(protocol_parameters, Mapping)
            and value in protocol_parameters
        ):
            return protocol_parameters[value]

        get_parameter = getattr(protocol, "get_parameter", None)

        if callable(get_parameter):
            try:
                resolved = get_parameter(value)
            except Exception:
                resolved = None

            if resolved is not None:
                return resolved

    try:
        return float(value)
    except ValueError:
        return default


def _repeat_node_loop_token_for_export(
    node_name: Any,
    counter_ref: Any,
) -> str:
    """Return the leaf token of the repeated logical node.

    Loop identity and loop length are separate concerns: the node supplies the
    gammaSTAR path, while ``repeat_count`` supplies the protocol dependency.
    """

    node = str(node_name or "").strip(".")
    if node:
        token = node.rsplit(".", 1)[-1]
        if token:
            return token

    if isinstance(counter_ref, str) and counter_ref:
        token = counter_ref
        for suffix in ("_index", "_counter"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
        if token:
            return token

    return "seqstar_loop"


def _repeat_loop_token_for_export(
    repeat_count_ref: Any,
    counter_ref: Any,
) -> str:
    """Return a semantic loop token without sequence-specific assumptions."""

    if isinstance(repeat_count_ref, str) and repeat_count_ref:
        return _protocol_key(repeat_count_ref)

    if isinstance(counter_ref, str) and counter_ref:
        token = counter_ref
        for suffix in ("_index", "_counter"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
        if token:
            return token

    return "seqstar_loop"


def _node_is_within_for_export(
    node: str,
    ancestor: str,
) -> bool:
    node = str(node or "").strip(".")
    ancestor = str(ancestor or "").strip(".")

    return bool(
        node
        and ancestor
        and (
            node == ancestor
            or node.startswith(f"{ancestor}.")
        )
    )


def _node_depth_for_export(node: str) -> int:
    node = str(node or "").strip(".")
    return len(node.split(".")) if node else 0


def _detect_repeated_block_motif(
    sequence: Any,
    blocks: list[Any],
) -> dict[str, Any] | None:
    """Detect a repeated block motif in a sequence-agnostic way.

    The detector looks for the smallest repeated pattern of block signatures.
    A signature is based on block role when available, otherwise on a compact
    event-family/channel/role summary. It deliberately ignores numerical event
    names such as ``gy_pre_000`` versus ``gy_pre_063`` so phase-encode style
    variations can still be recognized as one motif.

    Returns None when there is no clean repeated motif.
    """

    n_blocks = len(blocks)
    if n_blocks < 2:
        return None

    signatures = [_block_motif_signature(block) for block in blocks]

    # Keep the search bounded and conservative. Most PyPulseq-style kernels are
    # short block motifs; this avoids accidentally compressing arbitrary long
    # preparation sections.
    max_motif = min(32, n_blocks // 2)

    for motif_length in range(1, max_motif + 1):
        if n_blocks % motif_length != 0:
            continue

        repetitions = n_blocks // motif_length
        if repetitions < 2:
            continue

        motif = signatures[:motif_length]
        ok = True
        for start in range(motif_length, n_blocks, motif_length):
            if signatures[start : start + motif_length] != motif:
                ok = False
                break

        if not ok:
            continue

        motif_blocks = blocks[:motif_length]
        _, motif_duration = _flatten_blocks_to_kernel_events(sequence, motif_blocks)

        return {
            "blocks": motif_blocks,
            "motif_length": motif_length,
            "repetitions": repetitions,
            "motif_duration": motif_duration,
            "signature": motif,
        }

    return None


def _block_motif_signature(block: Any) -> str:
    """Return a compact, repeat-stable block signature."""

    role = _block_role(block)
    events = list(_iter_events(block))
    event_sig = "+".join(_event_motif_signature(event) for event in events)

    if role:
        return f"role:{role}|events:{event_sig}"

    return f"events:{event_sig}"


def _block_role(block: Any) -> str | None:
    """Return a normalized block role, if present."""

    for attr in ("role", "block_role", "label", "name"):
        value = getattr(block, attr, None)
        if value:
            return _normalize_signature_token(str(value))

    parameters = getattr(block, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("role", "block_role", "label", "name"):
            value = parameters.get(key)
            if value:
                return _normalize_signature_token(str(value))

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("role", "block_role", "label", "name"):
            value = metadata.get(key)
            if value:
                return _normalize_signature_token(str(value))

    return None


def _event_motif_signature(event: Any) -> str:
    """Return a compact event signature for motif detection."""

    if _is_rf_event(event):
        family = "rf"
    elif _is_adc_event(event):
        family = "adc"
    elif _is_gradient_event(event):
        family = "grad"
    else:
        family = "event"

    channel = _get_any(event, keys=("channel", "axis"), default=None)
    role = _get_any(event, keys=("role", "axis_role", "encoding_role", "use"), default=None)
    kind = _get_any(event, keys=("event_type", "kind", "type"), default=None)

    parts = [family]
    if channel is not None:
        parts.append(f"ch={_normalize_signature_token(str(channel))}")
    if role is not None:
        parts.append(f"role={_normalize_signature_token(str(role))}")
    if kind is not None:
        parts.append(f"kind={_normalize_signature_token(str(kind))}")

    return ":".join(parts)


def _normalize_signature_token(value: str) -> str:
    """Normalize names/roles so indexed repetitions compare equal."""

    text = str(value).strip().lower()

    # Remove common trailing phase/line indices: gy_pre_000, line-12, rf2.
    while text and text[-1].isdigit():
        text = text[:-1]
    text = text.rstrip("_- .")

    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {"_", "-", ".", " ", "/"}:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unnamed"


class _TimedEventProxy:
    """Event proxy with a kernel-local absolute tstart for gammaSTAR export.

    The proxy keeps the original event semantics but reports delay/tstart as a
    kernel-local absolute start. This is used only by the generic multi-block
    flattening fallback so existing RF/ADC/gradient leaf writers can remain
    event-local and sequence-agnostic.
    """

    def __init__(
        self,
        base: Any,
        *,
        block_start: float,
        event_index: int,
        block_index: int,
        repetition_variants: list[Any] | None = None,
        loop_counter_path: str | None = None,
        source_block: Any | None = None,
    ) -> None:
        self._base = base
        self._block_start = float(block_start)
        self._base_delay = _event_delay(base)
        self._global_tstart = self._block_start + self._base_delay
        self._event_index = int(event_index)
        self._block_index = int(block_index)
        self._seqstar_repetition_variants = list(repetition_variants or [base])
        self._seqstar_loop_counter_path = loop_counter_path
        self._source_block = source_block

        raw_name = str(getattr(base, "name", f"event_{event_index}") or f"event_{event_index}")
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name)
        cleaned = cleaned.strip("_") or f"event_{event_index}"
        self.name = f"{cleaned}_b{block_index:03d}_e{event_index:03d}"

        self.delay = self._global_tstart
        self.tstart = self._global_tstart
        self.start = self._global_tstart
        self.start_s = self._global_tstart

        # Copy parameters/metadata shallowly so gammaSTAR export can inspect
        # sequence context while seeing the flattened timing.
        base_parameters = getattr(base, "parameters", None)
        self.parameters = dict(base_parameters) if isinstance(base_parameters, Mapping) else {}
        self.parameters["delay"] = self._global_tstart
        self.parameters["tstart"] = self._global_tstart
        self.parameters["source_event_name"] = raw_name
        self.parameters["source_block_index"] = block_index
        self.parameters["source_event_index"] = event_index
        if source_block is not None:
            _copy_block_export_metadata_to_mapping(source_block, self.parameters)

        base_metadata = getattr(base, "metadata", None)
        self.metadata = dict(base_metadata) if isinstance(base_metadata, Mapping) else {}
        self.metadata["source_event_name"] = raw_name
        self.metadata["source_block_index"] = block_index
        self.metadata["source_event_index"] = event_index
        if source_block is not None:
            _copy_block_export_metadata_to_mapping(source_block, self.metadata)
        self.metadata["flattened_kernel_tstart"] = self._global_tstart
        self.metadata["seqstar_num_repetition_variants"] = len(self._seqstar_repetition_variants)
        if loop_counter_path:
            self.metadata["seqstar_loop_counter_path"] = loop_counter_path

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def to_gammastar_waveform(self) -> Any:
        """Return base gradient waveform with flattened tstart."""

        if hasattr(self._base, "to_gammastar_waveform"):
            data = self._base.to_gammastar_waveform()
            if isinstance(data, Mapping):
                data = dict(data)
                data["tstart"] = self._global_tstart
                data["delay"] = self._global_tstart
                data["timing_reference"] = "absolute"
            return data

        raise AttributeError("Base event does not expose to_gammastar_waveform")

    def to_gammastar_windows(self, *args: Any, **kwargs: Any) -> Any:
        """Return ADC windows with flattened absolute timing."""

        data = None
        if hasattr(self._base, "to_gammastar_windows"):
            data = self._base.to_gammastar_windows(*args, **kwargs)
        elif hasattr(self._base, "to_gammastar_samples"):
            data = self._base.to_gammastar_samples(*args, **kwargs)

        if not isinstance(data, Mapping):
            raise AttributeError("Base event does not expose gammaSTAR ADC windows")

        out = dict(data)
        windows = []
        for window in out.get("windows", []) or []:
            window_dict = dict(window.to_dict()) if hasattr(window, "to_dict") else dict(window)
            local = float(window_dict.get("local_tstart", window_dict.get("tstart", window_dict.get("delay", 0.0))))
            # If the base representation was already event-relative and also
            # included event.delay, subtract the original event delay once.
            # This keeps simple make_adc(delay=gx.rise_time) windows from
            # becoming block_start + delay + delay.
            if not bool(out.get("windows_are_absolute", False)):
                local = local
            absolute = self._block_start + local
            window_dict["local_tstart"] = local
            window_dict["tstart"] = absolute
            window_dict["delay"] = absolute
            window_dict["tend"] = absolute + float(window_dict.get("duration", 0.0))
            windows.append(window_dict)

        out["windows"] = windows
        out["windows_are_absolute"] = True
        out["absolute_timing"] = True
        out["timing_reference"] = "absolute"
        return out

    def to_gammastar_samples(self, *args: Any, **kwargs: Any) -> Any:
        return self.to_gammastar_windows(*args, **kwargs)



def _flatten_repeated_motif_to_kernel_events(
    sequence: Any,
    blocks: list[Any],
    *,
    motif_length: int,
    repetitions: int,
) -> tuple[list[Any], float]:
    """Flatten one representative motif and attach per-repetition variants.

    This is the compact export path. It emits only the first motif in time, but
    each representative event proxy carries the corresponding event objects
    from all repeated motifs. Event writers can then index line-/repetition-
    dependent quantities by the gammaSTAR loop counter without duplicating RF,
    ADC, and gradient containers.
    """

    motif_length = int(motif_length)
    repetitions = int(repetitions)
    if motif_length <= 0 or repetitions <= 0:
        return _flatten_blocks_to_kernel_events(sequence, blocks)

    motif_blocks = list(blocks[:motif_length])
    groups = [list(blocks[i * motif_length : (i + 1) * motif_length]) for i in range(repetitions)]

    events: list[Any] = []
    current_time = 0.0
    event_index = 0

    for block_index, block in enumerate(motif_blocks):
        block_start = current_time
        block_events = list(_iter_events(block))
        block_duration = max(
            _safe_block_duration(block),
            _events_extent_duration(block_events),
        )

        for event_pos, event in enumerate(block_events):
            variants: list[Any] = []
            for group in groups:
                if block_index >= len(group):
                    continue
                group_events = list(_iter_events(group[block_index]))
                if event_pos < len(group_events):
                    variants.append(group_events[event_pos])

            if not variants:
                variants = [event]

            proxy = _TimedEventProxy(
                event,
                block_start=block_start,
                event_index=event_index,
                block_index=block_index,
                repetition_variants=variants,
                loop_counter_path="root.seqstar_loop.counter",
                source_block=block,
            )
            block_role = _block_role(block)
            if block_role:
                proxy.parameters["source_block_role"] = block_role
                proxy.metadata["source_block_role"] = block_role
            events.append(proxy)
            event_index += 1

        current_time = block_start + block_duration

    return events, current_time

def _flatten_blocks_to_kernel_events(
    sequence: Any,
    blocks: list[Any],
) -> tuple[list[Any], float]:
    """Flatten sequential blocks into kernel-local timed event proxies."""

    events: list[Any] = []
    current_time = 0.0
    event_index = 0

    for block_index, block in enumerate(blocks):
        block_start = current_time
        block_events = list(_iter_events(block))
        block_duration = max(
            _safe_block_duration(block),
            _events_extent_duration(block_events),
        )

        for event in block_events:
            proxy = _TimedEventProxy(
                event,
                block_start=block_start,
                event_index=event_index,
                block_index=block_index,
                source_block=block,
            )
            block_role = _block_role(block)
            if block_role:
                proxy.parameters["source_block_role"] = block_role
                proxy.metadata["source_block_role"] = block_role
            events.append(proxy)
            event_index += 1

        current_time = block_start + block_duration

    return events, current_time


def _safe_block_duration(block: Any) -> float:
    """Best-effort block duration for sequential gammaSTAR flattening."""

    try:
        return float(calc_duration(block))
    except Exception:
        pass

    events = list(_iter_events(block))
    return _events_extent_duration(events)


def _resolve_sequence_relationships_for_export(sequence: Any) -> None:
    """Resolve relationship-derived timing before gammaSTAR export.

    This is sequence-general. The gammaSTAR writer does not know whether the
    sequence is FID, GRE, EPI, ASL, pCASL, spectroscopy, or anything else. It
    only ensures that relationship-derived timing has been materialized before
    serialization.
    """

    try:
        from pypulseq_star import relationships as _relationships
    except Exception:
        _relationships = None

    if _relationships is not None and hasattr(_relationships, "resolve"):
        _relationships.resolve(sequence)
        return

    for method_name in (
        "resolve_relationships",
        "resolve_timing_relationships",
        "resolve",
    ):
        method = getattr(sequence, method_name, None)
        if callable(method):
            method()
            return


def _get_repetition_time_for_block(
    block: Any,
    *,
    sequence: Any,
) -> float | None:
    """Return repetition time using generic protocol aliases.

    Search order:
        1. block parameters
        2. sequence parameters
        3. resolved repeat_every relationship metadata
    """

    value = _get_float(
        block,
        keys=(
            "repetition_time",
            "TR",
            "tr",
            "period",
            "repeat_period",
        ),
        default=None,
    )

    if value is not None:
        return value

    value = _get_float(
        sequence,
        keys=(
            "repetition_time",
            "TR",
            "tr",
            "period",
            "repeat_period",
        ),
        default=None,
    )

    if value is not None:
        return value

    return _get_resolved_relationship_float(
        sequence,
        keys=(
            "period",
            "TR",
            "tr",
            "target_value",
        ),
        relationship_kinds=(
            "repeat_every",
            "repeats_every",
            "set_repetition_time",
            "timing.repeat_every",
        ),
    )


def _get_resolved_relationship_float(
    sequence: Any,
    *,
    keys: tuple[str, ...],
    relationship_kinds: tuple[str, ...],
) -> float | None:
    """Read a resolved scalar from relationship metadata."""

    for relationship in _iter_sequence_relationships(sequence):
        kind = _relationship_kind_for_export(relationship)

        if kind not in relationship_kinds:
            continue

        resolved = _relationship_resolved_mapping_for_export(relationship)

        for key in keys:
            if key in resolved and resolved[key] is not None:
                try:
                    return float(resolved[key])
                except (TypeError, ValueError):
                    continue

    return None


def _iter_sequence_relationships(sequence: Any) -> Iterable[Any]:
    """Yield relationship-like objects attached to a sequence."""

    candidate_attrs = (
        "relationship_definitions",
        "relationships",
        "_relationships",
        "_seqstar_relationships",
        "_pypulseq_star_relationships",
    )

    for attr in candidate_attrs:
        value = getattr(sequence, attr, None)

        if value is None:
            continue

        if isinstance(value, Mapping):
            yield from value.values()
            continue

        if isinstance(value, (list, tuple, set)):
            yield from value
            continue

        try:
            yield from list(value)
        except TypeError:
            yield value


def _relationship_kind_for_export(relationship: Any) -> str:
    """Return a normalized relationship kind/type string."""

    if isinstance(relationship, Mapping):
        for key in ("kind", "type", "relationship_type"):
            value = relationship.get(key)
            if value is not None:
                return str(value)

        metadata = relationship.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("kind", "type", "relationship_type"):
                value = metadata.get(key)
                if value is not None:
                    return str(value)

        return "relationship"

    for attr in ("kind", "type", "relationship_type"):
        value = getattr(relationship, attr, None)
        if value is not None:
            return str(value)

    metadata = getattr(relationship, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("kind", "type", "relationship_type"):
            value = metadata.get(key)
            if value is not None:
                return str(value)

    return "relationship"


def _relationship_resolved_mapping_for_export(relationship: Any) -> dict[str, Any]:
    """Return resolved relationship values as a plain dictionary."""

    if isinstance(relationship, Mapping):
        resolved = relationship.get("resolved")
        if isinstance(resolved, Mapping):
            return dict(resolved)

        metadata = relationship.get("metadata")
        if isinstance(metadata, Mapping):
            resolved = metadata.get("resolved")
            if isinstance(resolved, Mapping):
                return dict(resolved)

        return {}

    resolved = getattr(relationship, "resolved", None)
    if isinstance(resolved, Mapping):
        return dict(resolved)

    metadata = getattr(relationship, "metadata", None)
    if isinstance(metadata, Mapping):
        resolved = metadata.get("resolved")
        if isinstance(resolved, Mapping):
            return dict(resolved)

    return {}


def _iter_blocks(sequence: Any) -> Iterable[Any]:
    """Iterate over blocks from SeqStarSequence."""

    if hasattr(sequence, "timeline") and hasattr(sequence.timeline, "blocks"):
        blocks = sequence.timeline.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    if hasattr(sequence, "blocks"):
        blocks = sequence.blocks
        if isinstance(blocks, Mapping):
            yield from blocks.values()
        else:
            yield from blocks
        return

    raise TypeError("Sequence does not expose timeline.blocks or blocks.")


def _iter_events(block: Any) -> Iterable[Any]:
    """Iterate over events in a SeqStar block."""

    visited: set[int] = set()
    yield from _iter_node_events(block, visited=visited, is_root=True)


def _iter_node_events(
    node: Any,
    *,
    visited: set[int],
    is_root: bool = False,
) -> Iterable[Any]:
    """Recursively yield events from a hierarchy node."""

    if node is None or isinstance(node, (float, int, str, bool)):
        return

    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if not is_root and _looks_event(node):
        yield node
        return

    if isinstance(node, Mapping):
        for value in node.values():
            yield from _iter_node_events(value, visited=visited)
        return

    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from _iter_node_events(value, visited=visited)
        return

    if hasattr(node, "events"):
        yield from _iter_node_events(getattr(node, "events"), visited=visited)

    if hasattr(node, "children"):
        yield from _iter_node_events(getattr(node, "children"), visited=visited)

    if is_dataclass(node):
        for field_info in fields(node):
            if field_info.name in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                continue

            try:
                value = getattr(node, field_info.name)
            except AttributeError:
                continue

            yield from _iter_node_events(value, visited=visited)

    elif hasattr(node, "__dict__"):
        for key, value in vars(node).items():
            if key in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                continue

            yield from _iter_node_events(value, visited=visited)


def _looks_event(obj: Any) -> bool:
    """Return True if object looks like an event."""

    if obj is None or isinstance(obj, (float, int, str, bool)):
        return False

    class_name = obj.__class__.__name__.lower()

    if "shape" in class_name and not any(
        hasattr(obj, marker)
        for marker in (
            "flip_angle",
            "num_samples",
            "dwell",
            "windows",
            "channel",
            "axis",
            "rise_time",
            "flat_time",
            "fall_time",
        )
    ):
        return False

    return _is_rf_event(obj) or _is_adc_event(obj) or _is_gradient_event(obj)


def _is_rf_event(event: Any) -> bool:
    event_type = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    role = str(getattr(event, "role", "")).lower()
    use = str(getattr(event, "use", "")).lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "")).lower()

    return (
        event_type == "rf"
        or kind.startswith("rf")
        or role in {"rf", "excitation", "refocusing", "inversion"}
        or use in {"rf", "excitation", "refocusing", "inversion"}
        or "rf" in class_name
        or "rf" in name
        or hasattr(event, "flip_angle")
    )


def _is_adc_event(event: Any) -> bool:
    event_type = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "")).lower()

    return (
        event_type == "adc"
        or kind.startswith("adc")
        or "adc" in class_name
        or "adc" in name
        or hasattr(event, "num_samples")
        or hasattr(event, "windows")
    )


def _is_gradient_event(event: Any) -> bool:
    event_type = str(getattr(event, "type", "")).lower()
    kind = str(getattr(event, "kind", "")).lower()
    class_name = event.__class__.__name__.lower()
    name = str(getattr(event, "name", "")).lower()

    return (
        event_type in {"grad", "gradient", "trap"}
        or "grad" in kind
        or "gradient" in class_name
        or "trapezoid" in class_name
        or "grad" in name
        or hasattr(event, "channel")
        or hasattr(event, "axis")
        or all(hasattr(event, attr) for attr in ("rise_time", "flat_time", "fall_time"))
    )


def _event_delay(event: Any) -> float:
    return float(
        _get_any(
            event,
            keys=("delay", "tstart", "start", "start_s"),
            default=0.0,
        )
    )


def _event_active_duration(event: Any) -> float:
    """Return active event duration, excluding delay and post-time."""

    if _is_adc_event(event):
        duration = _get_any(
            event,
            keys=("duration", "shape_duration", "adc_duration", "readout_duration"),
            default=None,
        )
        if duration is not None:
            return float(duration)

    duration = _get_any(
        event,
        keys=("duration", "shape_duration"),
        default=None,
    )

    if duration is not None:
        return float(duration)

    shape = getattr(event, "shape", None)
    if shape is not None and hasattr(shape, "duration"):
        return float(shape.duration)

    raise ValueError(f"Could not determine active duration for event {event!r}")


def _rf_type_string(event: Any) -> str:
    value = _get_any(event, keys=("type", "use", "role"), default="Excitation")
    value = str(value)

    normalized = value.lower()
    if normalized in {"excitation", "excite", "rf"}:
        return "Excitation"
    if normalized in {"refocusing", "refocus"}:
        return "Refocusing"
    if normalized in {"inversion", "invert"}:
        return "Inversion"

    return value



def _merge_protocol_values_for_export(
    *,
    export_sequence: Any,
    symbolic_sequence: Any | None,
) -> dict[str, Any]:
    """Merge numeric export defaults with symbolic protocol relationships.

    GammaStarWriter(seq).write(defaults=resolved) exports the resolved/default
    sequence for stable numeric event geometry, but protocol-control fields must
    still come from the symbolic sequence when they are derived relationships.

    Example:
        phase_encode_start = -0.5 * p.n_y / p.fov

    In the resolved sequence this is already -125.0 for the default 64-line
    protocol.  If we export that literal, changing root.prot.n_y on gammaSTAR
    changes the loop length but not the k-space start.  Therefore symbolic
    protocol expressions from symbolic_sequence must override resolved numeric
    defaults for root.prot.* export.
    """

    values = dict(_collect_protocol_values(export_sequence))

    if symbolic_sequence is None or symbolic_sequence is export_sequence:
        return values

    symbolic_values = _collect_protocol_values(symbolic_sequence)
    for key, value in symbolic_values.items():
        key_text = str(key)
        canonical_key = _protocol_key(key_text)

        # Symbolic relationships override resolved numeric defaults.
        if _contains_protocol_expression(value):
            values[key_text] = value
            values[canonical_key] = value
            continue

        # Preserve symbolic-only values not present in the resolved export.
        if key_text not in values and canonical_key not in values:
            values[key_text] = value

    return values



def _canonical_protocol_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return protocol values with both original and gammaSTAR-canonical keys."""

    canonical: dict[str, Any] = {}

    for key, value in values.items():
        text_key = str(key)
        canonical[text_key] = value
        canonical[_protocol_key(text_key)] = value

    return canonical


def _is_literal_parameter(parameter: Mapping[str, Any]) -> bool:
    """Return True when a gammaSTAR parameter has no live inputs."""

    if not isinstance(parameter, Mapping):
        return False

    inputs = parameter.get("inputs")
    return not bool(inputs)


def _is_literal_or_window_absolute_parameter(parameter: Mapping[str, Any]) -> bool:
    """Return True for literal/window-only tstart parameters with no protocol input."""

    if not isinstance(parameter, Mapping):
        return False

    inputs = parameter.get("inputs") or {}
    if not inputs:
        return True

    return all(str(path).endswith(".window_data") for path in inputs.values())


def _collect_protocol_values(sequence: Any) -> dict[str, Any]:
    """Collect protocol values from sequence.parameters and protocol-like APIs."""

    values: dict[str, Any] = {}

    parameters = getattr(sequence, "parameters", None)
    if isinstance(parameters, Mapping):
        values.update(parameters)

    protocol = getattr(sequence, "protocol", None)
    if protocol is not None:
        protocol_parameters = getattr(protocol, "parameters", None)
        if isinstance(protocol_parameters, Mapping):
            values.update(protocol_parameters)

        if hasattr(protocol, "to_template_context"):
            try:
                context = protocol.to_template_context()
                if isinstance(context, Mapping):
                    # Do not let a resolved/template context overwrite explicit
                    # protocol.parameters entries that are still symbolic. This
                    # is essential for v0.2 protocol-control relationships such
                    # as:
                    #
                    #   phase_encode_start = -0.5 * p.n_y / p.fov
                    #   phase_encode_step  =  1.0 / p.fov
                    #
                    # Protocol.to_template_context() is allowed to contain
                    # resolved numeric defaults for display/export, but those
                    # defaults must not replace symbolic protocol relationships
                    # before _protocol_parameter() has a chance to compile them
                    # into gammaSTAR inputs/scripts.
                    for key, value in context.items():
                        existing = values.get(key)
                        canonical_existing = values.get(_protocol_key(str(key)))
                        if (
                            _contains_protocol_expression(existing)
                            or _contains_protocol_expression(canonical_existing)
                        ):
                            continue
                        values[key] = value
            except Exception:
                pass

    return values


def _protocol_key(key: str) -> str:
    """Map SeqStar protocol names to gammaSTAR protocol names."""

    mapping = {
        "repetition_time": "TR",
        "tr": "TR",
        "TR": "TR",
        "echo_time": "TE",
        "te": "TE",
        "TE": "TE",
        "flip_angle_excitation": "flip_angle",
        "flip_angle": "flip_angle",
        "fa": "flip_angle",
        "FA": "flip_angle",
        "averages": "average",
        "n_avg": "average",
        "num_averages": "average",
        "repetitions": "average",
        "num_samples": "adc_samples",
        "number_of_samples": "adc_samples",
        "adc_num_samples": "adc_samples",
        "first_delay": "adc_delay",
        "dwell": "adc_dwell",
        "sample_time": "adc_dwell",
        "num_echoes": "echo_train_length",
        "etl": "echo_train_length",
        "ETL": "echo_train_length",
    }

    return mapping.get(key, key)


def _get_int(obj: Any, *, keys: tuple[str, ...], default: int) -> int:
    return int(_get_any(obj, keys=keys, default=default))


def _get_float(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: float | None,
) -> float | None:
    value = _get_any(obj, keys=keys, default=default)

    if value is None:
        return None

    return float(value)


def _get_any(obj: Any, *, keys: tuple[str, ...], default: Any = None) -> Any:
    """Read value from attribute, parameters, metadata, or metadata.timing."""

    for key in keys:
        if hasattr(obj, key) and getattr(obj, key) is not None:
            return getattr(obj, key)

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in keys:
            if key in parameters and parameters[key] is not None:
                return parameters[key]

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            for key in keys:
                if key in timing and timing[key] is not None:
                    return timing[key]

        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]

    return default


def _max_grad_t_per_m(system: Any) -> float:
    """Return gradient limit in T/m for gammaSTAR system metadata."""

    if hasattr(system, "input_max_grad") and hasattr(system, "input_grad_unit"):
        return _grad_to_t_per_m(float(system.input_max_grad), str(system.input_grad_unit))

    unit = str(getattr(system, "grad_unit", "Hz/m")).lower()
    value = float(system.max_grad)

    if unit == "hz/m":
        return value / float(system.gamma)

    return _grad_to_t_per_m(value, unit)


def _max_slew_t_per_m_s(system: Any) -> float:
    """Return slew limit in T/m/s for gammaSTAR system metadata."""

    if hasattr(system, "input_max_slew") and hasattr(system, "input_slew_unit"):
        return _slew_to_t_per_m_s(float(system.input_max_slew), str(system.input_slew_unit))

    unit = str(getattr(system, "slew_unit", "Hz/m/s")).lower()
    value = float(system.max_slew)

    if unit == "hz/m/s":
        return value / float(system.gamma)

    return _slew_to_t_per_m_s(value, unit)


def _grad_to_t_per_m(value: float, unit: str) -> float:
    unit = unit.lower()

    if unit in {"t/m", "tesla/m"}:
        return value
    if unit in {"mt/m", "millitesla/m"}:
        return value * 1e-3
    if unit in {"hz/m"}:
        raise ValueError("Hz/m gradient conversion requires system.gamma.")

    return value


def _slew_to_t_per_m_s(value: float, unit: str) -> float:
    unit = unit.lower()

    if unit in {"t/m/s", "tesla/m/s"}:
        return value
    if unit in {"mt/m/ms", "millitesla/m/ms"}:
        return value
    if unit in {"mt/m/s", "millitesla/m/s"}:
        return value * 1e-3
    if unit in {"hz/m/s"}:
        raise ValueError("Hz/m/s slew conversion requires system.gamma.")

    return value


def _test_display_key(path: str) -> str:
    """Return a clean final-hierarchy display key for a test parameter."""

    key = path.removeprefix("root.")
    replacements = {
        ".is_timing_increasing_and_rastered_and_same_am_length": ": RF timing/raster/amplitude-length",
        ".is_adc_windows_timing_increasing_and_valid": ": ADC timing/dwell",
        ".is_timing_increasing_and_rastered": ": gradient timing/raster",
    }
    for suffix, label in replacements.items():
        if key.endswith(suffix):
            return key[: -len(suffix)] + label
    return key


def _lua_string(value: Any) -> str:
    """Return a Lua-safe quoted string literal."""

    return json.dumps(str(value))


def _order_sequence_elements(sequence_elements: dict[str, str]) -> dict[str, str]:
    """Return sequence_elements in gammaSTAR-friendly graph order."""

    ordered: dict[str, str] = {}
    remaining = dict(sequence_elements)

    if "root" in remaining:
        ordered["root"] = remaining.pop("root")

    # Preserve the actual exported loop path, e.g. root.ky, rather than
    # assuming root.average. Service scopes remain after the executable graph.
    service_scopes = {
        "root.expo",
        "root.helper",
        "root.info",
        "root.prot",
        "root.sys",
        "root.tests",
    }
    executable_roots = sorted(
        (
            path
            for path in remaining
            if path.startswith("root.")
            and path.count(".") == 1
            and path not in service_scopes
        ),
        key=_hierarchy_sort_key,
    )

    for executable_root in executable_roots:
        if executable_root in remaining:
            ordered[executable_root] = remaining.pop(executable_root)
        prefix = executable_root + "."
        for path in sorted(
            [item for item in list(remaining) if item.startswith(prefix)],
            key=_hierarchy_sort_key,
        ):
            ordered[path] = remaining.pop(path)

    service_order = [
        "root.expo",
        "root.helper",
        "root.info",
        "root.prot",
        "root.sys",
        "root.tests",
    ]

    for path in service_order:
        if path in remaining:
            ordered[path] = remaining.pop(path)

    for path in sorted(remaining, key=_hierarchy_sort_key):
        ordered[path] = remaining[path]

    return ordered


def _hierarchy_sort_key(path: str) -> tuple[int, list[int], str]:
    """Sort paths parent-before-child with useful child priorities."""

    priority_by_name = {
        "average": 0,
        "ky": 0,
        "kx": 0,
        "kz": 0,
        "kernel": 0,
        "readout": 0,
        "readout2": 0,
        "readout3": 0,
        "adc": 0,
        "adc2": 0,
        "adc3": 0,
        "window": 1,
        "rf": 2,
        "rf2": 2,
        "rf3": 2,
        "header": 4,
        "grad": 5,
        "gx": 5,
        "gy": 6,
        "gz": 7,
        "atomic": 10,
        "expo": 90,
        "helper": 91,
        "info": 92,
        "prot": 93,
        "sys": 94,
        "tests": 95,
    }

    parts = path.split(".")
    priorities = [priority_by_name.get(part, 50) for part in parts]
    return (len(parts), priorities, path)


def _safe_input_name(path: str) -> str:
    """Convert a dotted path to a Lua-safe input variable name."""

    return path.replace(".", "_").replace("-", "_")


def _protocol_parameter(
    value: Any,
    *,
    target_path: str,
) -> dict[str, Any]:
    """Return a literal or live protocol parameter.

    Sequence definitions such as FOV may contain Protocol symbols, e.g.
    ``[p.fov, p.fov, p.slice_thickness]``.  Those symbols are expression
    objects, not JSON/Lua literals, so they must be compiled into gammaSTAR
    relationships rather than sent through ``_literal()``.
    """

    if not _contains_protocol_expression(value):
        return _literal(value)

    inputs: dict[str, str] = {}
    source_to_name: dict[str, str] = {}

    body = _protocol_value_lua_expression(
        value,
        inputs=inputs,
        source_to_name=source_to_name,
        target_path=target_path,
    )

    return _expr(
        inputs=inputs,
        script=f"return {body}",
    )



def _contains_parameter_reference(value: Any) -> bool:
    """Return True when a nested expression contains a protocol ParameterRef."""

    if _is_seqstar_expression(value):
        if hasattr(value, "canonical_name"):
            return True
        if hasattr(value, "operand"):
            return _contains_parameter_reference(value.operand)
        if hasattr(value, "left") and hasattr(value, "right"):
            return (
                _contains_parameter_reference(value.left)
                or _contains_parameter_reference(value.right)
            )
        return False

    if isinstance(value, Mapping):
        return any(
            _contains_parameter_reference(key)
            or _contains_parameter_reference(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return any(_contains_parameter_reference(item) for item in value)

    return False


def _contains_non_protocol_reference(value: Any) -> bool:
    """Detect event, anchor, block, or other non-protocol references."""

    if _is_seqstar_expression(value):
        if hasattr(value, "canonical_name"):
            return False

        kind = getattr(value, "kind", None)
        if kind is not None:
            return True

        if hasattr(value, "operand"):
            return _contains_non_protocol_reference(value.operand)
        if hasattr(value, "left") and hasattr(value, "right"):
            return (
                _contains_non_protocol_reference(value.left)
                or _contains_non_protocol_reference(value.right)
            )
        return False

    if isinstance(value, Mapping):
        return any(
            _contains_non_protocol_reference(key)
            or _contains_non_protocol_reference(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return any(_contains_non_protocol_reference(item) for item in value)

    return False

def _contains_protocol_expression(value: Any) -> bool:
    """Return True when ``value`` recursively contains a SeqStar expression."""

    if _is_seqstar_expression(value):
        return True

    if isinstance(value, Mapping):
        return any(
            _contains_protocol_expression(key)
            or _contains_protocol_expression(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return any(_contains_protocol_expression(item) for item in value)

    return False


def _is_seqstar_expression(value: Any) -> bool:
    """Duck-type the expression layer without importing a specific version."""

    if value is None or not hasattr(value, "to_canonical"):
        return False

    # Current Expression subclasses expose dependencies, but keep this robust
    # across minor implementation changes and slotted dataclasses.
    if hasattr(value, "dependencies"):
        return True

    # ParameterRef / UnaryExpression / BinaryExpression shape fallback.
    if hasattr(value, "canonical_name"):
        return True
    if hasattr(value, "operator_name") and (
        hasattr(value, "operand")
        or (hasattr(value, "left") and hasattr(value, "right"))
    ):
        return True

    return False


def _protocol_value_lua_expression(
    value: Any,
    *,
    inputs: dict[str, str],
    source_to_name: dict[str, str],
    target_path: str,
) -> str:
    """Compile a nested Python/protocol value into a Lua expression body."""

    if _is_seqstar_expression(value):
        return _seqstar_expression_to_lua(
            value,
            inputs=inputs,
            source_to_name=source_to_name,
            target_path=target_path,
        )

    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            item_lua = _protocol_value_lua_expression(
                item,
                inputs=inputs,
                source_to_name=source_to_name,
                target_path=target_path,
            )
            if isinstance(key, str) and key.isidentifier():
                parts.append(f"{key}={item_lua}")
            else:
                key_lua = _protocol_value_lua_expression(
                    key,
                    inputs=inputs,
                    source_to_name=source_to_name,
                    target_path=target_path,
                )
                parts.append(f"[{key_lua}]={item_lua}")
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(
            _protocol_value_lua_expression(
                item,
                inputs=inputs,
                source_to_name=source_to_name,
                target_path=target_path,
            )
            for item in value
        ) + "}"

    return _to_lua(value)


def _seqstar_expression_to_lua(
    expression: Any,
    *,
    inputs: dict[str, str],
    source_to_name: dict[str, str],
    target_path: str,
) -> str:
    """Compile simple protocol expressions into Lua.

    This intentionally supports the expression shapes that can appear in
    protocol/definition values: ParameterRef, LiteralExpression, UnaryExpression,
    and BinaryExpression.  Event/block references are executable-graph concepts
    and should not be embedded directly inside root.prot values.
    """

    if hasattr(expression, "canonical_name"):
        source_path = f"root.prot.{_protocol_key(str(expression.canonical_name))}"
        if source_path == target_path:
            # Avoid accidental self-references.  This path should normally not
            # occur because actual protocol parameter values are literals, while
            # aliases/definitions reference other protocol parameters.
            raise TypeError(
                f"Protocol expression for {target_path!r} references itself."
            )
        if source_path not in source_to_name:
            name = _safe_input_name(source_path)
            base = name
            suffix = 2
            used = set(inputs)
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            source_to_name[source_path] = name
            inputs[name] = source_path
        return source_to_name[source_path]

    if hasattr(expression, "value") and not hasattr(expression, "operator_name"):
        return _to_lua(expression.value)

    operator_name = getattr(expression, "operator_name", None)

    if operator_name is not None and hasattr(expression, "operand"):
        operand = _seqstar_expression_to_lua(
            expression.operand,
            inputs=inputs,
            source_to_name=source_to_name,
            target_path=target_path,
        )
        if operator_name == "neg":
            return f"(-({operand}))"

    if (
        operator_name is not None
        and hasattr(expression, "left")
        and hasattr(expression, "right")
    ):
        left = _seqstar_expression_to_lua(
            expression.left,
            inputs=inputs,
            source_to_name=source_to_name,
            target_path=target_path,
        )
        right = _seqstar_expression_to_lua(
            expression.right,
            inputs=inputs,
            source_to_name=source_to_name,
            target_path=target_path,
        )
        operator_map = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "truediv": "/",
            "pow": "^",
        }
        if operator_name in operator_map:
            return f"(({left}) {operator_map[operator_name]} ({right}))"

    raise TypeError(
        "Cannot compile protocol expression to Lua literal/relationship: "
        f"{type(expression).__name__}."
    )


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _literal(value: Any) -> dict[str, Any]:
    """Return a gammaSTAR literal parameter."""

    return {
        "inputs": {},
        "script": f"return {_to_lua(value)}",
    }


def _expr(*, inputs: dict[str, str], script: str) -> dict[str, Any]:
    """Return a gammaSTAR expression parameter."""

    return {
        "inputs": inputs,
        "script": script,
    }


def _to_lua(value: Any) -> str:
    """Serialize a Python literal into a small Lua literal."""

    if value is None:
        return "nil"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return repr(value)

    if isinstance(value, str):
        return repr(value)

    if isinstance(value, Mapping):
        if not value:
            return "{}"

        parts = []
        for key, item in value.items():
            if item is None:
                continue

            if isinstance(key, str) and key.isidentifier():
                parts.append(f"{key}={_to_lua(item)}")
            else:
                parts.append(f"[{_to_lua(key)}]={_to_lua(item)}")

        if not parts:
            return "{}"

        return "{" + ", ".join(parts) + "}"

    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_to_lua(item) for item in value) + "}"

    raise TypeError(f"Cannot serialize {type(value).__name__} to Lua literal.")
