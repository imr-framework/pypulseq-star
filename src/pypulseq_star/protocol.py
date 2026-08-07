# src/pypulseq_star/protocol.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class ProtocolParameterSpec:
    """
    Definition of a possible protocol parameter.

    This is a UI/export specification, not an enforced sequence requirement.
    If a developer does not provide this parameter, it is not emitted by default.
    """

    key: str
    label: str
    group: str
    value_type: str = "number"
    unit: str | None = None
    default: Any = None
    choices: tuple[Any, ...] = ()
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class Protocol:
    """
    Protocol-control layer for pypulseq_star.

    This class captures high-level protocol parameters that are usually present
    in gammaSTAR-style seq.json/template structures, for example grouped
    protocol controls under Contrast, Geometry, MT Preparation, PAT, Recon,
    and Special.

    The class is intentionally permissive:
        - Developers may provide only the parameters they need.
        - Missing parameters are not emitted unless include_unset=True.
        - Unknown parameters are preserved under the 'special' group by default.

    Example
    -------
    protocol = ppstar.Protocol(
        name="Block RF train",
        description="A simple repeated RF block-pulse train",
        parameters={
            "average": 20,
            "TR": 2.0,
        },
    )
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    strict: bool = False

    GROUP_ORDER: ClassVar[tuple[str, ...]] = (
        "contrast",
        "geometry",
        "mt_preparation",
        "pat",
        "recon",
        "special",
    )

    GROUP_LABELS: ClassVar[dict[str, str]] = {
        "contrast": "Contrast",
        "geometry": "Geometry",
        "mt_preparation": "MT Preparation",
        "pat": "PAT",
        "recon": "Recon",
        "special": "Special",
    }

    PARAMETER_SPECS: ClassVar[dict[str, ProtocolParameterSpec]] = {
        # ------------------------------------------------------------------
        # Contrast / timing / excitation
        # ------------------------------------------------------------------
        "echo_time": ProtocolParameterSpec(
            key="echo_time",
            label="Echo Time",
            group="contrast",
            value_type="number",
            unit="s",
            aliases=("TE", "te"),
            description="Echo time.",
        ),
        "echo_times": ProtocolParameterSpec(
            key="echo_times",
            label="Echo Time",
            group="contrast",
            value_type="array",
            unit="s",
            aliases=("TEs", "tes"),
            description="Echo times for multi-echo acquisitions.",
        ),
        "repetition_time": ProtocolParameterSpec(
            key="repetition_time",
            label="Repetition Time",
            group="contrast",
            value_type="number",
            unit="s",
            aliases=("TR", "tr"),
            description="Repetition time.",
        ),
        "minimal_te": ProtocolParameterSpec(
            key="minimal_te",
            label="Minimal TE(s)",
            group="contrast",
            value_type="boolean",
            aliases=("min_te", "minimum_te"),
            description="Whether the minimal TE should be used.",
        ),
        "readout_duration": ProtocolParameterSpec(
            key="readout_duration",
            label="Readout Duration",
            group="contrast",
            value_type="number",
            unit="s",
            aliases=("adc_duration", "ro_duration"),
            description="ADC/readout duration.",
        ),
        "flip_angle_excitation": ProtocolParameterSpec(
            key="flip_angle_excitation",
            label="Flip Angle Excitation",
            group="contrast",
            value_type="number",
            unit="deg",
            aliases=("flip_angle", "fa", "FA"),
            description="Excitation flip angle.",
        ),
        "excitation_phase": ProtocolParameterSpec(
            key="excitation_phase",
            label="Excitation Phase",
            group="contrast",
            value_type="number",
            unit="rad",
            aliases=("phase", "rf_phase"),
            description="RF excitation phase.",
        ),
        "rf_frequency_offset": ProtocolParameterSpec(
            key="rf_frequency_offset",
            label="RF Frequency Offset",
            group="contrast",
            value_type="number",
            unit="Hz",
            aliases=("freq_offset", "frequency_offset"),
            description="RF frequency offset.",
        ),
        "inversion_recovery": ProtocolParameterSpec(
            key="inversion_recovery",
            label="Inversion Recovery",
            group="contrast",
            value_type="choice",
            choices=("None", "IR", "DIR", "FLAIR", "STIR"),
            aliases=("ir", "IR"),
            description="Inversion recovery mode.",
        ),
        "inversion_time": ProtocolParameterSpec(
            key="inversion_time",
            label="Inversion Time",
            group="contrast",
            value_type="number",
            unit="s",
            aliases=("TI", "ti"),
            description="Inversion time.",
        ),
        "phase_pf": ProtocolParameterSpec(
            key="phase_pf",
            label="Phase PF",
            group="contrast",
            value_type="number",
            aliases=("partial_fourier_phase", "pf_phase"),
            description="Partial Fourier factor in phase direction.",
        ),
        "readout_pf": ProtocolParameterSpec(
            key="readout_pf",
            label="Readout PF",
            group="contrast",
            value_type="number",
            aliases=("partial_fourier_readout", "pf_readout"),
            description="Partial Fourier factor in readout direction.",
        ),
        "average": ProtocolParameterSpec(
            key="average",
            label="Averages",
            group="contrast",
            value_type="integer",
            aliases=("averages", "n_avg", "num_averages", "NSA", "NEX"),
            description="Number of averages/repetitions.",
        ),
        "echo_train_length": ProtocolParameterSpec(
            key="echo_train_length",
            label="Echo Train Length",
            group="contrast",
            value_type="integer",
            aliases=("etl", "ETL", "turbo_factor"),
            description="Echo train length or turbo factor.",
        ),
        "spoiling": ProtocolParameterSpec(
            key="spoiling",
            label="Spoiling",
            group="contrast",
            value_type="choice",
            choices=("None", "RF", "Gradient", "RF+Gradient"),
            description="Spoiling strategy.",
        ),
        "spoiling_phase": ProtocolParameterSpec(
            key="spoiling_phase",
            label="Spoiling Phase",
            group="contrast",
            value_type="number",
            unit="rad",
            aliases=("spoilphase", "rf_spoiling_phase"),
            description="RF spoiling phase increment or spoiling phase.",
        ),
        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        "fov": ProtocolParameterSpec(
            key="fov",
            label="Field of View",
            group="geometry",
            value_type="array",
            unit="m",
            aliases=("FOV", "field_of_view"),
            description="Field of view. Can be scalar or vector.",
        ),
        "fov_read": ProtocolParameterSpec(
            key="fov_read",
            label="Readout FOV",
            group="geometry",
            value_type="number",
            unit="m",
            aliases=("fov_ro",),
            description="Readout-direction field of view.",
        ),
        "fov_phase": ProtocolParameterSpec(
            key="fov_phase",
            label="Phase FOV",
            group="geometry",
            value_type="number",
            unit="m",
            aliases=("fov_pe",),
            description="Phase-direction field of view.",
        ),
        "matrix": ProtocolParameterSpec(
            key="matrix",
            label="Matrix",
            group="geometry",
            value_type="array",
            aliases=("matrix_size", "encoding_matrix"),
            description="Acquisition matrix.",
        ),
        "matrix_read": ProtocolParameterSpec(
            key="matrix_read",
            label="Readout Matrix",
            group="geometry",
            value_type="integer",
            aliases=("nx", "n_readout"),
            description="Readout matrix size.",
        ),
        "matrix_phase": ProtocolParameterSpec(
            key="matrix_phase",
            label="Phase Matrix",
            group="geometry",
            value_type="integer",
            aliases=("ny", "n_phase"),
            description="Phase-encode matrix size.",
        ),
        "num_slices": ProtocolParameterSpec(
            key="num_slices",
            label="Slices",
            group="geometry",
            value_type="integer",
            aliases=("slices", "n_slices"),
            description="Number of slices.",
        ),
        "slice_thickness": ProtocolParameterSpec(
            key="slice_thickness",
            label="Slice Thickness",
            group="geometry",
            value_type="number",
            unit="m",
            aliases=("thickness",),
            description="Slice thickness.",
        ),
        "slice_gap": ProtocolParameterSpec(
            key="slice_gap",
            label="Slice Gap",
            group="geometry",
            value_type="number",
            unit="m",
            description="Slice gap.",
        ),
        "slice_orientation": ProtocolParameterSpec(
            key="slice_orientation",
            label="Slice Orientation",
            group="geometry",
            value_type="choice",
            choices=("axial", "coronal", "sagittal", "oblique"),
            aliases=("orientation",),
            description="Slice orientation.",
        ),
        "readout_axis": ProtocolParameterSpec(
            key="readout_axis",
            label="Readout Axis",
            group="geometry",
            value_type="choice",
            choices=("x", "y", "z"),
            description="Readout gradient axis.",
        ),
        "phase_axis": ProtocolParameterSpec(
            key="phase_axis",
            label="Phase Axis",
            group="geometry",
            value_type="choice",
            choices=("x", "y", "z"),
            description="Phase-encoding gradient axis.",
        ),
        "slice_axis": ProtocolParameterSpec(
            key="slice_axis",
            label="Slice Axis",
            group="geometry",
            value_type="choice",
            choices=("x", "y", "z"),
            description="Slice-select gradient axis.",
        ),
        # ------------------------------------------------------------------
        # MT preparation
        # ------------------------------------------------------------------
        "mt_enabled": ProtocolParameterSpec(
            key="mt_enabled",
            label="MT Enabled",
            group="mt_preparation",
            value_type="boolean",
            aliases=("MT", "magnetization_transfer"),
            description="Enable magnetization transfer preparation.",
        ),
        "mt_flip_angle": ProtocolParameterSpec(
            key="mt_flip_angle",
            label="MT Flip Angle",
            group="mt_preparation",
            value_type="number",
            unit="deg",
            description="MT pulse flip angle.",
        ),
        "mt_offset": ProtocolParameterSpec(
            key="mt_offset",
            label="MT Offset",
            group="mt_preparation",
            value_type="number",
            unit="Hz",
            description="MT frequency offset.",
        ),
        "mt_duration": ProtocolParameterSpec(
            key="mt_duration",
            label="MT Duration",
            group="mt_preparation",
            value_type="number",
            unit="s",
            description="MT pulse duration.",
        ),
        # ------------------------------------------------------------------
        # PAT / parallel imaging / acceleration
        # ------------------------------------------------------------------
        "acceleration_factor": ProtocolParameterSpec(
            key="acceleration_factor",
            label="Acceleration Factor",
            group="pat",
            value_type="integer",
            aliases=("R", "ipat", "parallel_imaging_factor"),
            description="Parallel imaging acceleration factor.",
        ),
        "phase_acceleration": ProtocolParameterSpec(
            key="phase_acceleration",
            label="Phase Acceleration",
            group="pat",
            value_type="integer",
            aliases=("r_phase",),
            description="Phase-direction acceleration factor.",
        ),
        "slice_acceleration": ProtocolParameterSpec(
            key="slice_acceleration",
            label="Slice Acceleration",
            group="pat",
            value_type="integer",
            aliases=("sms_factor", "multiband_factor"),
            description="Slice/SMS acceleration factor.",
        ),
        "reference_lines": ProtocolParameterSpec(
            key="reference_lines",
            label="Reference Lines",
            group="pat",
            value_type="integer",
            aliases=("acs_lines",),
            description="Reference/calibration lines.",
        ),
        # ------------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------------
        "recon_mode": ProtocolParameterSpec(
            key="recon_mode",
            label="Recon Mode",
            group="recon",
            value_type="choice",
            choices=("none", "online", "offline", "external"),
            aliases=("reconstruction",),
            description="Reconstruction mode.",
        ),
        "coil_combination": ProtocolParameterSpec(
            key="coil_combination",
            label="Coil Combination",
            group="recon",
            value_type="choice",
            choices=("none", "sum_of_squares", "adaptive", "external"),
            description="Coil-combination method.",
        ),
        "denoise": ProtocolParameterSpec(
            key="denoise",
            label="Denoise",
            group="recon",
            value_type="boolean",
            description="Enable denoising.",
        ),
        "super_resolution": ProtocolParameterSpec(
            key="super_resolution",
            label="Super Resolution",
            group="recon",
            value_type="boolean",
            aliases=("sr", "SR"),
            description="Enable super-resolution reconstruction.",
        ),
        "output_format": ProtocolParameterSpec(
            key="output_format",
            label="Output Format",
            group="recon",
            value_type="choice",
            choices=("raw", "dicom", "nifti", "ismrmrd", "hdf5"),
            description="Desired reconstruction/output format.",
        ),
        # ------------------------------------------------------------------
        # Special / sequence-control / custom
        # ------------------------------------------------------------------
        "acquisition_time": ProtocolParameterSpec(
            key="acquisition_time",
            label="Acquisition Time",
            group="special",
            value_type="number",
            unit="s",
            aliases=("scan_time", "TA"),
            description="Total acquisition time.",
        ),
        "sequence_type": ProtocolParameterSpec(
            key="sequence_type",
            label="Sequence Type",
            group="special",
            value_type="string",
            aliases=("sequence",),
            description="Sequence family or type.",
        ),
        "trajectory": ProtocolParameterSpec(
            key="trajectory",
            label="Trajectory",
            group="special",
            value_type="choice",
            choices=("cartesian", "radial", "spiral", "epi", "projection", "custom"),
            description="K-space trajectory.",
        ),
        "dummy_scans": ProtocolParameterSpec(
            key="dummy_scans",
            label="Dummy Scans",
            group="special",
            value_type="integer",
            aliases=("dummy_shots", "dummy_repetitions"),
            description="Number of dummy scans.",
        ),
        "triggering": ProtocolParameterSpec(
            key="triggering",
            label="Triggering",
            group="special",
            value_type="choice",
            choices=("none", "external", "cardiac", "respiratory"),
            description="Triggering mode.",
        ),
        "comments": ProtocolParameterSpec(
            key="comments",
            label="Comments",
            group="special",
            value_type="string",
            aliases=("comment", "notes"),
            description="Free-text comments.",
        ),
    }

    def __post_init__(self) -> None:
        self.aliases = dict(self.aliases)
        self.parameters = self._normalize_parameters(dict(self.parameters))
        self._validate_aliases()

    @classmethod
    def built_in_aliases(cls) -> dict[str, str]:
        """
        Return alias-to-canonical-key mapping.
        """
        alias_map: dict[str, str] = {}
        for key, spec in cls.PARAMETER_SPECS.items():
            alias_map[key] = key
            for alias in spec.aliases:
                alias_map[alias] = key
        return alias_map

    def alias_map(self) -> dict[str, str]:
        """Return built-in aliases merged with protocol-specific overrides."""
        merged = self.built_in_aliases()
        merged.update(self.aliases)

        for canonical_name in self.parameters:
            merged.setdefault(canonical_name, canonical_name)

        return merged

    def canonical_name(self, key: str) -> str:
        """Return the canonical parameter name for a key or alias."""
        return self.alias_map().get(key, key)

    @property
    def symbols(self):
        """Return symbolic references to protocol parameters."""
        from pypulseq_star.expressions import ParameterNamespace

        namespace = getattr(self, "_symbols_namespace", None)
        if namespace is None:
            namespace = ParameterNamespace(self, aliases=self.alias_map())
            self._symbols_namespace = namespace
        return namespace

    @classmethod
    def parameter_spec(cls, key: str) -> ProtocolParameterSpec | None:
        """Compatibility alias used by the symbolic parameter namespace."""
        return cls.get_spec(key)

    @classmethod
    def get_spec(cls, key: str) -> ProtocolParameterSpec | None:
        """
        Return the parameter specification for a canonical key or alias.
        """
        canonical_key = cls.built_in_aliases().get(key, key)
        return cls.PARAMETER_SPECS.get(canonical_key)

    def _validate_aliases(self) -> None:
        """Validate protocol-specific alias targets when strict mode is enabled."""
        if not self.strict:
            return

        for alias, canonical_name in self.aliases.items():
            if canonical_name not in self.PARAMETER_SPECS:
                raise KeyError(
                    f"Alias {alias!r} targets unknown protocol parameter "
                    f"{canonical_name!r} while strict=True."
                )

    def _normalize_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Convert aliases such as TR, TE, FA, and TI to canonical keys.
        """
        alias_map = self.alias_map()
        normalized: dict[str, Any] = {}

        for key, value in parameters.items():
            canonical_key = alias_map.get(key, key)

            if self.strict and canonical_key not in self.PARAMETER_SPECS:
                raise KeyError(
                    f"Unknown protocol parameter '{key}'. "
                    "Use strict=False to allow custom parameters."
                )

            normalized[canonical_key] = value

        return normalized

    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set or update a protocol parameter.

        Aliases are accepted. Unknown parameters are accepted unless strict=True.
        """
        canonical_key = self.alias_map().get(key, key)

        if self.strict and canonical_key not in self.PARAMETER_SPECS:
            raise KeyError(
                f"Unknown protocol parameter '{key}'. "
                "Use strict=False to allow custom parameters."
            )

        self.parameters[canonical_key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a protocol parameter by canonical key or alias.
        """
        canonical_key = self.alias_map().get(key, key)
        return self.parameters.get(canonical_key, default)

    def remove_parameter(self, key: str) -> None:
        """
        Remove a protocol parameter by canonical key or alias.
        """
        canonical_key = self.alias_map().get(key, key)
        self.parameters.pop(canonical_key, None)

    def group_for_parameter(self, key: str) -> str:
        """
        Return the protocol group for a parameter.

        Unknown parameters are routed to the 'special' group.
        """
        spec = self.PARAMETER_SPECS.get(key)
        if spec is None:
            return "special"
        return spec.group

    def grouped_parameters(self, include_unset: bool = False) -> dict[str, dict[str, Any]]:
        """
        Return protocol parameters grouped by protocol-control section.

        By default, only explicitly provided parameters are emitted.
        If include_unset=True, known but unset parameters are emitted with
        their spec default values.
        """
        grouped: dict[str, dict[str, Any]] = {
            group: {} for group in self.GROUP_ORDER
        }

        if include_unset:
            for key, spec in self.PARAMETER_SPECS.items():
                grouped[spec.group][key] = spec.default

        for key, value in self.parameters.items():
            group = self.group_for_parameter(key)
            grouped.setdefault(group, {})
            grouped[group][key] = value

        return {
            group: values
            for group, values in grouped.items()
            if values or include_unset
        }

    def grouped_parameter_entries(
        self,
        include_unset: bool = False,
        include_specs: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Return grouped parameters as UI/export-friendly entries.

        This is closer to a gammaSTAR-style protocol-control template than a
        flat Python dictionary.
        """
        grouped_values = self.grouped_parameters(include_unset=include_unset)
        grouped_entries: dict[str, list[dict[str, Any]]] = {}

        for group, values in grouped_values.items():
            entries: list[dict[str, Any]] = []

            for key, value in values.items():
                spec = self.PARAMETER_SPECS.get(key)

                if spec is None:
                    entry = {
                        "key": key,
                        "label": key,
                        "value": value,
                        "value_type": self._infer_value_type(value),
                        "unit": None,
                    }
                else:
                    entry = {
                        "key": key,
                        "label": spec.label,
                        "value": value,
                        "value_type": spec.value_type,
                        "unit": spec.unit,
                    }

                    if include_specs:
                        entry["choices"] = list(spec.choices)
                        entry["description"] = spec.description
                        entry["aliases"] = list(spec.aliases)

                entries.append(entry)

            grouped_entries[group] = entries

        return grouped_entries

    def to_dict(
        self,
        include_unset: bool = False,
        include_specs: bool = True,
    ) -> dict[str, Any]:
        """
        Serialize the protocol to a general pypulseq_star dictionary.

        This is useful for debugging, generic JSON export, and template filling.
        """
        return {
            "type": "protocol",
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "parameters": self.parameters,
            "groups": [
                {
                    "key": group,
                    "label": self.GROUP_LABELS.get(group, group),
                    "parameters": entries,
                }
                for group, entries in self.grouped_parameter_entries(
                    include_unset=include_unset,
                    include_specs=include_specs,
                ).items()
            ],
        }

    def to_template_context(
        self,
        include_unset: bool = False,
        include_specs: bool = True,
    ) -> dict[str, Any]:
        """
        Return a context dictionary intended for a seq.json/template writer.

        The writer can insert this under the template's Protocol/Control section.
        """
        return {
            "protocol": self.to_dict(
                include_unset=include_unset,
                include_specs=include_specs,
            )
        }

    def to_gammastar_like_dict(
        self,
        include_unset: bool = False,
        include_specs: bool = True,
    ) -> dict[str, Any]:
        """
        Return a gammaSTAR-style protocol-control object.

        This does not claim to be a full gammaSTAR schema implementation.
        It is a stable pypulseq_star representation designed to be mapped into
        the final seq.json template.
        """
        grouped_entries = self.grouped_parameter_entries(
            include_unset=include_unset,
            include_specs=include_specs,
        )

        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "groups": {
                group: {
                    "label": self.GROUP_LABELS.get(group, group),
                    "parameters": entries,
                }
                for group, entries in grouped_entries.items()
            },
            "metadata": self.metadata,
        }

    @staticmethod
    def _infer_value_type(value: Any) -> str:
        """
        Infer a simple UI/export value type for custom parameters.
        """
        if isinstance(value, bool):
            return "boolean"

        if isinstance(value, int) and not isinstance(value, bool):
            return "integer"

        if isinstance(value, float):
            return "number"

        if isinstance(value, str):
            return "string"

        if isinstance(value, (list, tuple)):
            return "array"

        if isinstance(value, dict):
            return "object"

        return "unknown"

    def __repr__(self) -> str:
        return (
            "Protocol("
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"parameters={self.parameters!r}, "
            f"aliases={self.aliases!r}"
            ")"
        )