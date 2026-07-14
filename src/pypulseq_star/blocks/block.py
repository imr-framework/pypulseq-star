"""Block container for events that share a local time origin."""

from __future__ import annotations

from dataclasses import dataclass, field

from pypulseq_star.core import SeqStarNode
from pypulseq_star.events import SeqStarEvent


@dataclass(slots=True)
class SeqStarBlock(SeqStarNode):
    """A PyPulseq-like block with explicit semantic role metadata."""

    events: list[SeqStarEvent] = field(default_factory=list)

    def add_event(self, event: SeqStarEvent) -> SeqStarEvent:
        """Add an event and mirror the relationship in the node graph."""

        self.events.append(event)
        self.add_child(event, relationship="contains_event")
        return event

    def get_event(self, name: str) -> SeqStarEvent:
        """Return the unique event matching a public or source-event name.

        Concrete timeline occurrences may be copied and renamed to keep their
        debug paths unique. The stable constructor name is preserved in event
        provenance as ``source_event_name``. This lookup therefore accepts:

        - the concrete occurrence ``event.name``;
        - ``metadata["source_event_name"]``;
        - ``parameters["source_event_name"]``.

        Raises
        ------
        KeyError
            If no event matches the requested name.
        ValueError
            If more than one event matches the requested name.
        """

        requested = str(name)
        matches: list[SeqStarEvent] = []

        for event in self.events:
            candidate_names: set[str] = set()

            event_name = getattr(event, "name", None)
            if event_name is not None:
                candidate_names.add(str(event_name))

            for container_name in ("metadata", "parameters"):
                container = getattr(event, container_name, None)
                if not isinstance(container, dict):
                    continue

                for key in (
                    "source_event_name",
                    "sequence_event_name",
                    "original_event_name",
                ):
                    value = container.get(key)
                    if value is not None:
                        candidate_names.add(str(value))

            if requested in candidate_names:
                matches.append(event)

        if not matches:
            available: list[str] = []

            for event in self.events:
                event_name = getattr(event, "name", None)
                metadata = getattr(event, "metadata", None)
                parameters = getattr(event, "parameters", None)

                source_name = None
                if isinstance(metadata, dict):
                    source_name = metadata.get("source_event_name")
                if source_name is None and isinstance(parameters, dict):
                    source_name = parameters.get("source_event_name")

                label = str(event_name or "<unnamed>")
                if source_name and str(source_name) != label:
                    label += f" [source={source_name}]"
                available.append(label)

            raise KeyError(
                f"Block {getattr(self, 'name', '<unnamed>')!r} has no event "
                f"matching {requested!r}. Available events: {available!r}."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Block {getattr(self, 'name', '<unnamed>')!r} contains "
                f"{len(matches)} events matching {requested!r}. Use unique "
                "event/source names within a block."
            )

        return matches[0]
