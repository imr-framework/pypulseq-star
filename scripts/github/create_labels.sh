#!/usr/bin/env bash

set -euo pipefail

# PyPulseq-Star repository label setup
#
# Usage:
#   ./scripts/github/create_labels.sh
#
# Optional explicit repository:
#   REPO=imr-framework/pypulseq-star ./scripts/github/create_labels.sh
#
# Existing labels are updated because gh label create is called with --force.

REPO="${REPO:-imr-framework/pypulseq-star}"

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI 'gh' is not installed." >&2
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI is not authenticated. Run: gh auth login" >&2
    exit 1
fi

echo "Configuring labels for: ${REPO}"
echo

create_label() {
    local name="$1"
    local description="$2"
    local color="$3"

    printf "  %-24s" "${name}"

    gh label create "${name}" \
        --repo "${REPO}" \
        --description "${description}" \
        --color "${color}" \
        --force >/dev/null

    echo "created/updated"
}

# ---------------------------------------------------------------------------
# Release scope and priority
# ---------------------------------------------------------------------------

create_label \
    "release-blocker" \
    "Must be resolved before the current public release." \
    "B60205"

create_label \
    "priority:high" \
    "Important work requiring prompt attention but not necessarily release-blocking." \
    "D93F0B"

create_label \
    "priority:medium" \
    "Important planned work without immediate release urgency." \
    "FBCA04"

create_label \
    "priority:low" \
    "Useful work that can be scheduled after higher-priority items." \
    "C2E0C6"

create_label \
    "target:v0.2.0" \
    "Included in the PyPulseq-Star v0.2.0 release scope." \
    "0052CC"

create_label \
    "deferred:v0.3.0" \
    "Valid work intentionally deferred beyond the v0.2.0 release." \
    "6E7781"

create_label \
    "known-limitation" \
    "Behavior accepted for the current release and documented publicly." \
    "FBCA04"

create_label \
    "needs-triage" \
    "Issue has not yet been classified, prioritized, or assigned to a release." \
    "D4C5F9"

# ---------------------------------------------------------------------------
# Work classification
# ---------------------------------------------------------------------------

create_label \
    "bug" \
    "Incorrect or unexpected behavior." \
    "D73A4A"

create_label \
    "regression" \
    "Behavior that previously worked but is now broken or inconsistent." \
    "E99695"

create_label \
    "enhancement" \
    "New capability or improvement to existing behavior." \
    "A2EEEF"

create_label \
    "testing" \
    "Tests, fixtures, coverage, validation, or continuous integration." \
    "1D76DB"

create_label \
    "documentation" \
    "README, documentation, examples, tutorials, or release notes." \
    "0075CA"

create_label \
    "maintenance" \
    "Refactoring, dependency updates, cleanup, or repository maintenance." \
    "C5DEF5"

create_label \
    "performance" \
    "Runtime, memory, output-size, or computational-efficiency work." \
    "F9D0C4"

create_label \
    "softwarex" \
    "Work required for the SoftwareX release, manuscript, figures, or submission." \
    "5319E7"

create_label \
    "mrm-validation" \
    "Future experimental, multi-site, or multi-vendor MR validation work." \
    "BFDADC"

# ---------------------------------------------------------------------------
# Sequence-specific labels
# ---------------------------------------------------------------------------

create_label \
    "sequence:FID" \
    "Work specific to the FID reference sequence or its tests." \
    "0E8A16"

create_label \
    "sequence:GRE" \
    "Work specific to the GRE reference sequence or its tests." \
    "159818"

create_label \
    "sequence:EPI" \
    "Work specific to the EPI reference sequence or its tests." \
    "2EA44F"

create_label \
    "sequence:TSE" \
    "Work specific to the TSE reference sequence or its tests." \
    "34D058"

create_label \
    "sequence:general" \
    "Sequence-development behavior affecting multiple sequence families." \
    "7BC96F"

create_label \
    "sequence:new" \
    "Proposal or implementation of a new community sequence example." \
    "B7E4C7"

# ---------------------------------------------------------------------------
# Core library classes and concepts
# ---------------------------------------------------------------------------

create_label \
    "class:Sequence" \
    "Sequence container, block insertion, timing, or public Sequence behavior." \
    "7057FF"

create_label \
    "class:Protocol" \
    "Protocol parameters, symbols, aliases, validation, or derived values." \
    "8250DF"

create_label \
    "class:Node" \
    "Node hierarchy, roles, repetition metadata, or compact sequence structure." \
    "9C63D7"

create_label \
    "class:Relationship" \
    "Timing, anchoring, dependency, or relationship declarations." \
    "B392F0"

create_label \
    "class:Expression" \
    "Symbolic expressions, evaluation, dependency inputs, or serialization." \
    "C061CB"

create_label \
    "class:ADCTrain" \
    "ADC-train construction, windows, dead time, or lowering behavior." \
    "D876E3"

create_label \
    "class:EncodingFrame" \
    "Logical read, phase, and slice directions or physical-axis mapping." \
    "A663CC"

# ---------------------------------------------------------------------------
# Library subsystems
# ---------------------------------------------------------------------------

create_label \
    "subsystem:protocol" \
    "Protocol-layer behavior spanning parameters, symbols, and derived values." \
    "7057FF"

create_label \
    "subsystem:relationships" \
    "Relationship declaration, resolution, timing, and dependency behavior." \
    "8A63D2"

create_label \
    "subsystem:resolver" \
    "Expression evaluation, sequence resolution, or materialization." \
    "6F42C1"

create_label \
    "subsystem:loops" \
    "Repetitions, nested loops, counters, variants, or loop lowering." \
    "9C27B0"

create_label \
    "subsystem:timing" \
    "TE, TR, echo spacing, anchors, fills, or raster alignment." \
    "BFD4F2"

create_label \
    "subsystem:adc-train" \
    "ADC windows, trains, sampling alignment, or dead-time policy." \
    "B14FC5"

create_label \
    "subsystem:orientation" \
    "Logical-to-physical read, phase, and slice axis mapping." \
    "C061CB"

create_label \
    "subsystem:plotter" \
    "Local plotting, timeline rendering, or relationship visualization." \
    "D876E3"

create_label \
    "subsystem:writer" \
    "Generic export, backend lowering, or serialization behavior." \
    "5B3CC4"

create_label \
    "backend:pulseq" \
    "Pulseq .seq export, validation, or compatibility." \
    "0366D6"

create_label \
    "backend:gammastar" \
    "gammaSTAR JSON export, live protocol editing, or visualization." \
    "1F6FEB"

create_label \
    "subsystem:packaging" \
    "Package metadata, installation, distribution, or release artifacts." \
    "006B75"

create_label \
    "subsystem:ci" \
    "Continuous integration workflows and automated repository checks." \
    "0052CC"

create_label \
    "subsystem:api" \
    "Supported public API, compatibility, naming, or migration concerns." \
    "D4C5F9"

# ---------------------------------------------------------------------------
# Community and contribution labels
# ---------------------------------------------------------------------------

create_label \
    "good first issue" \
    "A clearly scoped issue suitable for a new contributor." \
    "7057FF"

create_label \
    "help wanted" \
    "Community assistance or external expertise would be valuable." \
    "008672"

create_label \
    "question" \
    "Further information, clarification, or user guidance is needed." \
    "D876E3"

create_label \
    "discussion-needed" \
    "Requires design or community discussion before implementation." \
    "CC317C"

create_label \
    "community-contribution" \
    "Proposed or implemented by an external community contributor." \
    "0E8A16"

create_label \
    "needs-reproduction" \
    "Reported behavior must be reproduced before diagnosis or implementation." \
    "F9D0C4"

create_label \
    "needs-design" \
    "Requires an agreed design or API contract before coding begins." \
    "D4C5F9"

create_label \
    "needs-tests" \
    "Implementation exists but requires additional automated tests." \
    "1D76DB"

create_label \
    "needs-documentation" \
    "Implementation exists but user-facing documentation is incomplete." \
    "0075CA"

echo
echo "Label configuration complete."
echo
echo "Repository labels:"
gh label list \
    --repo "${REPO}" \
    --limit 200 \
    --json name,description,color \
    --template '{{range .}}{{printf "%-26s #%s  %s\n" .name .color .description}}{{end}}'