#!/usr/bin/env bash

set -euo pipefail

# PyPulseq-Star milestone setup
#
# Usage:
#   ./scripts/github/create_milestones.sh
#
# Optional repository override:
#   REPO=owner/repository ./scripts/github/create_milestones.sh

REPO="${REPO:-imr-framework/pypulseq-star}"

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI 'gh' is not installed." >&2
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI is not authenticated. Run: gh auth login" >&2
    exit 1
fi

echo "Configuring milestones for: ${REPO}"
echo

find_milestone_number() {
    local title="$1"

    gh api \
        --paginate \
        "repos/${REPO}/milestones?state=all&per_page=100" \
        --jq ".[] | select(.title == \"${title}\") | .number" \
        | head -n 1
}

create_or_update_milestone() {
    local title="$1"
    local description="$2"
    local due_on="${3:-}"

    local number
    number="$(find_milestone_number "${title}")"

    if [[ -n "${number}" ]]; then
        echo "Updating milestone: ${title}"

        if [[ -n "${due_on}" ]]; then
            gh api \
                --method PATCH \
                "repos/${REPO}/milestones/${number}" \
                -f title="${title}" \
                -f description="${description}" \
                -f state="open" \
                -f due_on="${due_on}" \
                >/dev/null
        else
            gh api \
                --method PATCH \
                "repos/${REPO}/milestones/${number}" \
                -f title="${title}" \
                -f description="${description}" \
                -f state="open" \
                >/dev/null
        fi
    else
        echo "Creating milestone: ${title}"

        if [[ -n "${due_on}" ]]; then
            gh api \
                --method POST \
                "repos/${REPO}/milestones" \
                -f title="${title}" \
                -f description="${description}" \
                -f due_on="${due_on}" \
                >/dev/null
        else
            gh api \
                --method POST \
                "repos/${REPO}/milestones" \
                -f title="${title}" \
                -f description="${description}" \
                >/dev/null
        fi
    fi
}

create_or_update_milestone \
    "v0.2.0 / SoftwareX" \
    "Release and publication milestone for PyPulseq-Star v0.2.0. Includes the four foundational reference sequences, release-critical testing, code coverage, documentation, private-to-public repository preparation, release tagging, archival metadata, SoftwareX figures, manuscript, and submission. Excludes work explicitly labeled deferred:v0.3.0." \
    "2026-08-06T23:59:59Z"

create_or_update_milestone \
    "v0.3.0" \
    "Post-SoftwareX development milestone for deferred v0.2.0 items, broader sequence development, stricter live-editing contracts, expanded test coverage, API refinement, and community-contributed sequence implementations."

echo
echo "Milestone configuration complete."
echo
echo "Current milestones:"

gh api \
    --paginate \
    "repos/${REPO}/milestones?state=all&per_page=100" \
    --jq '.[] | "\(.number)\t\(.state)\t\(.title)\topen=\(.open_issues)\tclosed=\(.closed_issues)\tdue=\(.due_on // "none")"'