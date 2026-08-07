"""RF event conversion into gammaSTAR sequence JSON syntax."""

from __future__ import annotations

import math
from typing import Any

from pypulseq_star.core import SeqStarExpression
from pypulseq_star.events import SeqStarRFEvent
from pypulseq_star.sequence import SeqStarSequence

RF_RECT_BLUEPRINT = "184dfa36-f86e-425f-a653-40d47a4a99b2"
RF_PULSE_BLUEPRINT = "RFPulse"
KERNEL_BLUEPRINT = "a6bcd65a-b25b-4fe4-894f-c7fdf4fc8beb"
ROOT_BLUEPRINT = "004ebbab-558f-4a05-8cce-e907f52a4f37"
HELPER_BLUEPRINT = "3ec659dd-2b06-44a8-9af5-30a8496cfc57"
EXPO_BLUEPRINT = "c54aab35-2bd3-41e3-b892-35a072404b69"


class GammaStarRFConverter:
    """Translate RF SeqStar events into gammaSTAR RFRect/RFPulse nodes."""

    def __init__(self, sequence: SeqStarSequence) -> None:
        self.sequence = sequence

    def can_convert(self) -> bool:
        """Return true when this document is an RF-only milestone."""

        return len(_rf_events(self.sequence)) == 1 and "TR" in self.sequence.parameters

    def to_document(self) -> dict[str, Any]:
        """Build a minimal RF-only gammaSTAR document.

        The sequence program defines the event, shape, repetition count, and
        TR. This converter owns only gammaSTAR-specific syntax: RFRect/RFPulse
        node names, parameter paths, and script serialization.
        """

        rf = _single_rf_event(self.sequence)
        repetitions = int(self.sequence.parameters.get("repetitions", 1))
        tr = float(self.sequence.parameters.get("TR", rf.parameters["duration"]))
        duration = float(rf.parameters["duration"])
        flip_angle_rad = float(rf.parameters["flip_angle"])
        flip_angle_deg = math.degrees(flip_angle_rad)
        gamma = float(self.sequence.system.gamma)

        loop_path = "root.average"
        kernel_path = f"{loop_path}.kernel"
        # gammaSTAR represents the RFRect shape at kernel.rf and stores the
        # renderable RFPulse samples on the kernel.rf.rf leaf.
        rf_rect_path = f"{kernel_path}.rf"
        atomic_path = f"{rf_rect_path}.atomic"
        rf_path = f"{rf_rect_path}.rf"

        params: dict[str, Any] = {
            "root.prot.ui_specification": _literal("[]"),
            "root.tests.all_tests": {
                "inputs": {
                    "average_kernel_rf_rf_is_timing_increasing_and_rastered_and_same_am_length": (
                        f"{rf_path}.is_timing_increasing_and_rastered_and_same_am_length"
                    ),
                },
                "script": (
                    "return {\n"
                    "[\"average.kernel.rf.rf."
                    "is_timing_increasing_and_rastered_and_same_am_length\"] = "
                    "{ ok = average_kernel_rf_rf_is_timing_increasing_and_rastered_and_same_am_length "
                    "~= nil and "
                    "average_kernel_rf_rf_is_timing_increasing_and_rastered_and_same_am_length, "
                    "desc = nil }\n"
                    "}"
                ),
            },
            "root.tstart": _literal(0.0),
            f"{loop_path}.counter": _literal(0),
            f"{loop_path}.length": _literal(repetitions),
            f"{loop_path}.tstart": SeqStarExpression.relation(
                "average_counter * TR",
                average_counter=f"{loop_path}.counter",
                TR="root.prot.TR",
            ).to_gammastar_parameter(),
            f"{kernel_path}.tstart": _literal(0.0),
            f"{kernel_path}.duration": _literal(duration),
            f"{rf_rect_path}.tstart": _literal(0.0),
            f"{rf_rect_path}.duration": _literal(duration),
            f"{rf_rect_path}.flip_angle": _literal(flip_angle_deg),
            f"{rf_rect_path}.spoilphase": _literal(0.0),
            f"{rf_rect_path}.tcenter": {
                "inputs": {
                    "tstart": f"{rf_rect_path}.tstart",
                    "duration": f"{rf_rect_path}.duration",
                    "asymmetry": f"{rf_path}.asymmetry",
                },
                "script": "return tstart + asymmetry*duration",
            },
            f"{rf_rect_path}.tend": {
                "inputs": {
                    "tstart": f"{rf_rect_path}.tstart",
                    "duration": f"{rf_rect_path}.duration",
                },
                "script": "return tstart + duration",
            },
            f"{atomic_path}.tstart_absolute": {
                "inputs": {
                    "t0": "root.tstart",
                    "t1": f"{loop_path}.tstart",
                    "t2": f"{kernel_path}.tstart",
                    "t3": f"{rf_rect_path}.tstart",
                },
                "script": "return t0 + t1 + t2 + t3",
            },
            f"{atomic_path}.full_basic_repr": {
                "inputs": {
                    "full_basic_repr_TriggerPulse": (
                        f"{atomic_path}.full_basic_repr_TriggerPulse"
                    ),
                    "full_basic_repr_ADC": f"{atomic_path}.full_basic_repr_ADC",
                    "full_basic_repr_RFPulse": f"{atomic_path}.full_basic_repr_RFPulse",
                    "full_basic_repr_GradPulse": f"{atomic_path}.full_basic_repr_GradPulse",
                },
                "script": (
                    "return {\n"
                    "GradPulse=full_basic_repr_GradPulse,\n"
                    "RFPulse=full_basic_repr_RFPulse,\n"
                    "ADC=full_basic_repr_ADC,\n"
                    "TriggerPulse=full_basic_repr_TriggerPulse\n"
                    "}"
                ),
            },
            f"{atomic_path}.full_basic_repr_GradPulse": _literal({}),
            f"{atomic_path}.full_basic_repr_ADC": _literal({}),
            f"{atomic_path}.full_basic_repr_TriggerPulse": _literal({}),
            f"{atomic_path}.full_basic_repr_RFPulse": {
                "inputs": {
                    "basic_repr_rf_tstart_relative": (
                        f"{atomic_path}.basic_repr_rf_tstart_relative"
                    ),
                    "basic_repr_rf": f"{atomic_path}.basic_repr_rf",
                },
                "script": (
                    "return {\n"
                    "['rf']={tstart_relative=basic_repr_rf_tstart_relative, "
                    "required_parameters=basic_repr_rf}\n"
                    "\n"
                    "}"
                ),
            },
            f"{atomic_path}.basic_repr_rf_tstart_relative": {
                "inputs": {
                    "t0": f"{rf_path}.tstart",
                },
                "script": "return t0",
            },
            f"{atomic_path}.basic_repr_rf": {
                "inputs": {
                    "enabled": f"{rf_path}.enabled",
                    "type": f"{rf_path}.type",
                    "asymmetry": f"{rf_path}.asymmetry",
                    "phase": f"{rf_path}.phase",
                    "frequency": f"{rf_path}.frequency",
                    "samples": f"{rf_path}.samples",
                    "duration": f"{rf_path}.duration",
                },
                "script": (
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
            },
            f"{rf_path}.duration": {
                "inputs": {
                    "duration": f"{rf_rect_path}.duration",
                },
                "script": "return duration",
            },
            f"{rf_path}.enabled": _literal(bool(rf.parameters["enabled"])),
            f"{rf_path}.tstart": _literal(0.0),
            f"{rf_path}.asymmetry": _literal(float(rf.parameters["asymmetry"])),
            f"{rf_path}.frequency": _literal(float(rf.parameters["frequency"])),
            f"{rf_path}.phase": _literal(float(rf.parameters["phase"])),
            f"{rf_path}.type": _literal(str(rf.parameters["type"])),
            f"{rf_path}.samples": {
                "inputs": {
                    "gamma": "root.sys.gamma",
                    "flip_angle": f"{rf_rect_path}.flip_angle",
                    "duration": f"{rf_path}.duration",
                },
                "script": (
                    "local rf_amp = "
                    "(flip_angle/180*math.pi)/(2*math.pi*gamma*duration)\n"
                    "local samples_t = {duration/4, 3*duration/4}\n"
                    "local samples_am = {rf_amp, rf_amp}\n"
                    "local samples_fm = {0, 0}\n"
                    "return {t=samples_t, v={{am=samples_am, fm=samples_fm}}}"
                ),
            },
            f"{rf_path}.is_timing_increasing_and_rastered_and_same_am_length": {
                "inputs": {
                    "samples": f"{rf_path}.samples",
                    "rf_set": "root.rf_settings",
                },
                "script": (
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
            },
            "root.prot.TR": _literal(tr),
            "root.prot.flip_angle": _literal(flip_angle_deg),
            "root.info.description": _literal(
                "Repeated RF block-pulse train generated by PyPulseq-Star."
            ),
            "root.info.seq_dim": _literal(0),
            "root.info.is_epi": _literal(False),
            "root.sys.gamma": _literal(gamma),
            "root.sys.max_rf_amp": _literal(self.sequence.system.max_rf),
            "root.sys.raster_samples_rf": _literal(self.sequence.system.rf_raster_time),
            "root.sys.raster_time_rf": _literal(self.sequence.system.rf_raster_time),
            "root.rf_settings": {
                "inputs": {
                    "raster_time": "root.sys.raster_time_rf",
                    "raster_samples": "root.sys.raster_samples_rf",
                    "max_rf_amp": "root.sys.max_rf_amp",
                },
                "script": (
                    "return {max_rf_amp=max_rf_amp, "
                    "raster_time=raster_time, raster_samples=raster_samples}"
                ),
            },
        }
        _add_runtime_defaults(
            params=params,
            sequence=self.sequence,
            repetitions=repetitions,
            tr=tr,
            flip_angle_deg=flip_angle_deg,
        )

        return {
            "name": self.sequence.name,
            "parameters": params,
            "sequence_elements": {
                "root": ROOT_BLUEPRINT,
                loop_path: "Loop",
                kernel_path: KERNEL_BLUEPRINT,
                rf_rect_path: RF_RECT_BLUEPRINT,
                atomic_path: "Atomic",
                rf_path: RF_PULSE_BLUEPRINT,
                "root.expo": EXPO_BLUEPRINT,
                "root.helper": HELPER_BLUEPRINT,
                "root.info": "Info",
                "root.prot": "Protocol",
                "root.sys": "System",
                "root.tests": "Tests",
            },
        }


def _rf_events(sequence: SeqStarSequence) -> list[SeqStarRFEvent]:
    events = [event for block in sequence.timeline.blocks for event in block.events]
    return [event for event in events if isinstance(event, SeqStarRFEvent)]


def _single_rf_event(sequence: SeqStarSequence) -> SeqStarRFEvent:
    rf_events = _rf_events(sequence)
    if len(rf_events) != 1:
        raise ValueError("Expected exactly one RF event for RF-only gammaSTAR export")
    return rf_events[0]


def _literal(value: Any) -> dict[str, Any]:
    return SeqStarExpression.literal(value).to_gammastar_parameter()


def _add_runtime_defaults(
    *,
    params: dict[str, Any],
    sequence: SeqStarSequence,
    repetitions: int,
    tr: float,
    flip_angle_deg: float,
) -> None:
    """Add generic gammaSTAR runtime/protocol helpers used by plot import.

    These are intentionally not sequence-specific. They mirror the support
    envelope that the gammaSTAR examples carry around every sequence graph:
    protocol defaults, system limits, and helper settings. The RF event and
    hierarchy remain defined by the Python sequence program.
    """

    system = sequence.system
    max_grad_t_per_m = _grad_limit_t_per_m(system.max_grad, system.grad_unit)
    max_slew_t_per_m_s = _slew_limit_t_per_m_s(system.max_slew, system.slew_unit)

    defaults: dict[str, Any] = {
        "root.LoopInfo": _literal({}),
        "root.PNSPaths": _literal({}),
        "root.kernel_info": _literal({}),
        "root.fov": _literal([1.0, 1.0, 1.0]),
        "root.mat_size": _literal([1, 1, 1]),
        "root.acq_size": {
            "inputs": {"mat_size": "root.mat_size"},
            "script": "return mat_size",
        },
        "root.gradient_settings": {
            "inputs": {
                "raster_time": "root.sys.raster_time_grad",
                "raster_samples": "root.sys.raster_samples_grad",
                "max_grad_amp": "root.sys.max_grad_amp",
                "max_grad_slew": "root.sys.max_grad_slew",
            },
            "script": (
                "return {max_grad_amp=max_grad_amp, max_grad_slew=max_grad_slew, "
                "raster_time=raster_time, raster_samples=raster_samples}"
            ),
        },
        "root.gradient_settings_reduced_performance": {
            "inputs": {"gradient_settings": "root.gradient_settings"},
            "script": "return gradient_settings",
        },
        "root.adc_settings": {
            "inputs": {
                "raster_time": "root.sys.raster_time_adc",
                "raster_samples": "root.sys.raster_samples_adc",
            },
            "script": "return {raster_time=raster_time, raster_samples=raster_samples}",
        },
        "root.atomic_settings": {
            "inputs": {
                "raster_time": "root.sys.raster_time_atomic",
                "raster_samples": "root.sys.raster_samples_atomic",
            },
            "script": "return {raster_time=raster_time, raster_samples=raster_samples}",
        },
        "root.trigger_settings": {
            "inputs": {
                "raster_time": "root.sys.raster_time_trig",
                "raster_samples": "root.sys.raster_samples_trig",
            },
            "script": "return {raster_time=raster_time, raster_samples=raster_samples}",
        },
        "root.RegridTable": {
            "inputs": {"kernel_info": "root.kernel_info"},
            "script": "return kernel_info.RegridTable or {}",
        },
        "root.expo.ADCdelay": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.ADCdelay",
        },
        "root.expo.ADCduration": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.ADCduration",
        },
        "root.expo.ADCsamples": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.ADCsamples",
        },
        "root.expo.GradientFT": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.GradientFT",
        },
        "root.expo.GradientRDT": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.GradientRDT",
        },
        "root.expo.GradientRUT": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.GradientRUT",
        },
        "root.expo.Mode": {
            "inputs": {"RegridTable": "root.RegridTable"},
            "script": "return RegridTable.Mode",
        },
        "root.expo.apply_freq_corr": _literal(False),
        "root.expo.echo_spacing": {
            "inputs": {"kernel_info": "root.kernel_info"},
            "script": "return kernel_info.echo_spacing",
        },
        "root.expo.feedback_paths": _literal({}),
        "root.expo.image_scale_factor": _literal(1.0),
        "root.expo.is_feedback": {
            "inputs": {"fb_paths": "root.expo.feedback_paths"},
            "script": "return #fb_paths > 0",
        },
        "root.expo.is_POI_stop": _literal(False),
        "root.expo.LoopLengthAverage": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_average or 1",
        },
        "root.expo.LoopLengthContrast": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_contrast or 1",
        },
        "root.expo.LoopLengthLine": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_line or 1",
        },
        "root.expo.LoopLengthPartition": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_partition or 1",
        },
        "root.expo.LoopLengthRepetition": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_repetition or 1",
        },
        "root.expo.LoopLengthSegment": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_segment or 1",
        },
        "root.expo.LoopLengthSlice": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_slice or 1",
        },
        "root.expo.LoopLengthSlicegroup": {
            "inputs": {"LoopInfo": "root.LoopInfo"},
            "script": "return LoopInfo.max_slicegroup or 1",
        },
        "root.expo.paths_for_PNS_estimation": {
            "inputs": {"PNSPaths": "root.PNSPaths"},
            "script": "return PNSPaths",
        },
        "root.helper.constants": {
            "inputs": {},
            "script": (
                "return {\n"
                "golden_ratio_1d=0.618033988749,\n"
                "golden_ratio_2d_1=0.465571231876,\n"
                "golden_ratio_2d_2=0.682327803828,\n"
                "spoilphase_inc_inc=117\n"
                "}"
            ),
        },
        "root.helper.functions": {
            "inputs": {},
            "script": (
                "function modulo(a, b)\n"
                "  if b == nil or b == 0 then return 0 end\n"
                "  return math.abs(a - math.floor(a / b + 0.5) * b)\n"
                "end\n"
                "function ge(a, b)\n"
                "  return a >= b or math.abs(a-b) < 1e-12\n"
                "end\n"
                "return {modulo=modulo, ge=ge}"
            ),
        },
        "root.helper.complex": _literal({}),
        "root.info.is_epi": _literal(False),
        "root.info.seq_dim": _literal(0),
        "root.prot.PAT_factor_phase": _literal(1),
        "root.prot.PAT_factor_slice": _literal(1),
        "root.prot.PAT_mode": _literal("None"),
        "root.prot.PAT_ref_lines_phase": _literal(32),
        "root.prot.PAT_ref_lines_slice": _literal(16),
        "root.prot.TE": _literal(0.0),
        "root.prot.TR": _literal(tr),
        "root.prot.echo_train_length": _literal(1),
        "root.prot.flip_angle": _literal(flip_angle_deg),
        "root.prot.multiband_factor": _literal(1),
        "root.prot.offcenter_exc_1": _literal([0.0, 0.0, 0.0]),
        "root.prot.orientation_exc_1": _literal(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "root.prot.partitions": _literal(None),
        "root.prot.phase_oversampling": _literal(0.0),
        "root.prot.phase_partial_fourier": _literal(1),
        "root.prot.point_of_interest": _literal(None),
        "root.prot.read_oversampling": _literal(1.0),
        "root.prot.read_partial_fourier": _literal(1),
        "root.prot.rot_matrix": _literal(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "root.prot.slice_distance_factor": _literal(0.0),
        "root.prot.slice_oversampling": _literal(0.0),
        "root.prot.slice_partial_fourier": _literal(1),
        "root.prot.slice_reorder_scheme": _literal("Interleaved"),
        "root.prot.slices": _literal(None),
        "root.prot.thickness_exc_1": _literal(-1),
        "root.prot.turbo_factor": _literal(1),
        "root.sys.acoustic_resonance_frequencies": _literal(
            [[585.0, 100.0], [1120.0, 220.0]]
        ),
        "root.sys.coil_values_for_pns": _literal({}),
        "root.sys.fat_shift": _literal(0.0),
        "root.sys.frequency": _literal([123200000.0, 0.0]),
        "root.sys.gamma": _literal(system.gamma),
        "root.sys.max_grad_amp": _literal(max_grad_t_per_m),
        "root.sys.max_grad_slew": _literal(max_slew_t_per_m_s),
        "root.sys.max_rf_amp": _literal(system.max_rf),
        "root.sys.min_distance_between_adc_and_grad": _literal(0.0),
        "root.sys.min_distance_between_adc_and_rf": _literal(0.0),
        "root.sys.min_distance_between_grad_and_rf": _literal(0.0),
        "root.sys.raster_samples_adc": _literal(0.1e-6),
        "root.sys.raster_samples_atomic": _literal(system.block_duration_raster),
        "root.sys.raster_samples_grad": _literal(system.grad_raster_time),
        "root.sys.raster_samples_rf": _literal(system.rf_raster_time),
        "root.sys.raster_samples_trig": _literal(system.block_duration_raster),
        "root.sys.raster_time_adc": _literal(1e-6),
        "root.sys.raster_time_atomic": _literal(system.block_duration_raster),
        "root.sys.raster_time_grad": _literal(system.grad_raster_time),
        "root.sys.raster_time_rf": _literal(system.rf_raster_time),
        "root.sys.raster_time_trig": _literal(system.block_duration_raster),
        "root.sys.scanner_type": _literal("Generic"),
        "root.sys.system_specs": _literal({}),
    }

    for key, value in defaults.items():
        params.setdefault(key, value)


def _grad_limit_t_per_m(value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"t/m", "tesla/m"}:
        return value
    if normalized in {"mt/m", "millitesla/m"}:
        return value * 1e-3
    return value


def _slew_limit_t_per_m_s(value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"t/m/s", "tesla/m/s"}:
        return value
    if normalized in {"mt/m/ms", "millitesla/m/ms"}:
        return value
    if normalized in {"mt/m/s", "millitesla/m/s"}:
        return value * 1e-3
    return value
