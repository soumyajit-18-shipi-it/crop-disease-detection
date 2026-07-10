# Plan: Project Compliance Hardening

## Approach
- Add governance and legal documents at repo root.
- Add unified Python tooling config in `pyproject.toml`.
- Add `tests/` to cover FastAPI app health checks and classification logic.
- Add `.pre-commit-config.yaml` and `.gitlab-ci.yml`.
- Add Dockerfile and `.dockerignore`.
- Add Spec-Kit directories and templates.

## Milestones
- Milestone 1: Licensing + documentation updates.
- Milestone 2: Tooling config + unit tests.
- Milestone 3: Security scanning + CI integration.
- Milestone 4: Docker and Spec-Kit templates/specs.

## Risks & Mitigations
- Risk: Windows local dependency lock errors.
  - Mitigation: Use clean local test environment.
