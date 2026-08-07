# Security Policy

## Supported versions

PyPulseq-Star is currently under active development. Security fixes are applied
to the latest released version and the current `main` branch.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` branch | Yes |
| Older releases | Best effort |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a suspected security
vulnerability.

Use one of the following private reporting routes:

1. **Preferred:** GitHub private vulnerability reporting through the
   repository's **Security** tab.
2. **Fallback:** Email `imr.framework2018@gmail.com`.

Include, when possible:

- a clear description of the issue;
- the affected version or commit;
- steps to reproduce;
- the expected and observed behavior;
- potential impact;
- relevant logs, traces, or proof-of-concept code;
- whether the issue has already been disclosed elsewhere.

Do not include patient data, protected health information, credentials,
proprietary scanner information, or other sensitive data.

## Response process

The maintainers will aim to:

- acknowledge receipt within 5 business days;
- assess severity and reproducibility;
- request additional information when needed;
- coordinate a fix and release;
- credit the reporter unless anonymity is requested.

These are response targets rather than guarantees.

## Scope

Security reports may include:

- arbitrary code execution;
- dependency or packaging vulnerabilities;
- unsafe file parsing;
- path traversal;
- untrusted template or JSON execution;
- disclosure of credentials or sensitive information;
- unsafe handling of externally supplied sequence definitions;
- vulnerabilities in documentation or examples that could lead users to run
  unsafe commands.

General bugs, sequence-timing questions, installation problems, and feature
requests should be reported through the normal support route in `SUPPORT.md`.

## MRI safety disclaimer

PyPulseq-Star is research software. Generated sequences must be independently
reviewed, validated, and tested under appropriate institutional and scanner
safety procedures before use on any MRI system.

A software security report is not a substitute for reporting an MRI safety
incident. Scanner safety incidents should also be reported through the relevant
institutional, manufacturer, and regulatory channels.
