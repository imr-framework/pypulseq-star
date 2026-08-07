# Continuous integration setup

Copy the hidden `.github` directory from this archive into the repository root.

The active workflow is:

```text
.github/workflows/ci.yml
```

It builds and installs the wheel, verifies a clean installed import, runs the
installation tests, and constructs all four beginner demos without export.

The future full-suite workflow is stored as:

```text
.github/workflows/full-tests.yml.disabled
```

Rename it to `full-tests.yml` after the writer warnings, ADC raster issue, and
sequence repr issue have been fixed.

The `github_workflows_visible/` directory contains duplicate copies solely so
archive viewers that hide dot-directories can show the files. Do not copy that
visible directory into the repository.
