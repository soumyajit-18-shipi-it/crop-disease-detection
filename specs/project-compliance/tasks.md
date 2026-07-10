# Tasks: Project Compliance Hardening

## Required Tasks
- [ ] Create root compliance/legal files (.editorconfig, CHANGELOG, SECURITY, CODE_OF_CONDUCT, .env.example)
- [ ] Create AGPLv3 LICENSE and update README references
- [ ] Add Python tooling configuration (ruff, mypy, bandit, pyupgrade)
- [ ] Add pytest + coverage threshold with unit tests
- [ ] Add `.pre-commit-config.yaml` with required hooks
- [ ] Add `.gitlab-ci.yml` with lint/security/test/build stages
- [ ] Add Dockerfile and `.dockerignore`
- [ ] Add Spec-Kit constitution, templates, and example spec under `specs/`

## Definition of Done
- All new configs validate syntactically.
- CI pipeline commands are deterministic and fail on violations.
