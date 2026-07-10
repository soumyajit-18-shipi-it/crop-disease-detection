# Spec: Project Compliance Hardening (v0.1.0)

## Summary
Implement repository health, licensing (AGPLv3), Python quality gates, security scanning, CI/CD integration, and Spec-Kit compliance artifacts.

## Motivation
Improve GitLab Compliance Checker score by ensuring automated verification and clear governance artifacts.

## Goals
- Ensure AGPLv3 compliance artifacts are present and referenced.
- Establish lint/type/security/test tooling with consistent configuration.
- Add GitLab CI pipeline and pre-commit hooks that enforce quality gates.
- Add Docker build support.
- Add Spec-Kit templates and feature specs.

## Non-Goals
- Changing application business logic.
- Re-architecting repository structure.

## Functional Requirements
- FR-1: Add required governance/security/legal files.
- FR-2: Add AGPLv3 LICENSE and update README references.
- FR-3: Configure ruff, mypy, bandit, pyupgrade.
- FR-4: Configure pytest + coverage threshold >= 80%.
- FR-5: Configure pre-commit and GitLab CI.
- FR-6: Add secret scanning via gitleaks.
- FR-7: Add Spec-Kit folders + example feature specification.

## Security Requirements
- SR-1: CI must fail if SAST/secret scanning detect issues.
- SR-2: No secrets committed; `.env.example` documents safe defaults.

## Acceptance Criteria
- AC-1: GitLab CI runs and fails on tool violations.
- AC-2: `ruff`, `mypy`, `bandit`, `pytest --cov-fail-under=80` succeed.
- AC-3: gitleaks scan completes (exit non-zero on findings).
- AC-4: Docker build completes.

## References
- `SECURITY.md`, `LICENSE`, `.gitlab-ci.yml`, `.pre-commit-config.yaml`
