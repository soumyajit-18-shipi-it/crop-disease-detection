# Security Policy

## Supported Versions

- This repository currently maintains the `main` branch.

## Reporting a Vulnerability

Please report security issues responsibly.

- Preferred: open a GitLab/GitHub Security Issue (or email the maintainers listed in the repository).
- Do **not** open a public issue with exploit details.

When reporting, include:
- A clear description of the vulnerability.
- Steps to reproduce (or a minimal proof-of-concept).
- Impact analysis.
- Any relevant logs, stack traces, or affected endpoints.

## Response Process

- Acknowledge receipt within 3 business days.
- Triage severity (critical/high/medium/low).
- Provide an ETA for a fix or mitigation.

## Security Hardening

This project uses automated security checks in CI:
- SAST (Semgrep)
- Dependency scanning
- Secret scanning (Gitleaks)
- SCA hardening guidance (Bandit for common Python issues)

## Handling Secrets

- Never commit secrets to the repository.
- Use `.env` for local development and `.env.example` for documentation.
