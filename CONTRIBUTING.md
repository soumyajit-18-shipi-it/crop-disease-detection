# CONTRIBUTIONS.md

# Crop Disease Detection – Contribution Guidelines

Thank you for your interest in contributing to **Crop Disease Detection**!
This document outlines how developers, designers, and contributors can safely improve the repository.

---

## Table of Contents

1. How to Contribute
2. Branching & Pull Requests
3. Code Standards
4. Feature Guidelines
5. Testing
6. Reporting Issues
7. Best Practices

---

## 1. How to Contribute

### Steps

1. **Fork the repository** and clone locally.
2. Create a **new branch** for your feature/fix.
3. Make your changes.
4. Test locally.
5. Submit a **Pull Request** to the main repository.

### Communication

* Use clear commit messages describing changes.
* Link issues or tasks when applicable.
* Add reviewers if unsure.

---

## 2. Branching & Pull Requests

### Branch Naming

* Features: `feature/<short-description>`
* Bugfixes: `bugfix/<short-description>`
* Experiments: `experiment/<short-description>`

### Pull Request Guidelines

* PR title should be concise and descriptive.
* Reference relevant issues (if any).
* Ensure all tests pass before merging.

---

## 3. Code Standards

### Python (Backend/AI/ML)

* Language: Python 3.x
* Use **Ruff** for linting and formatting.
* Use **MyPy** for strict typing checks.
* Use **Bandit** and **Vulture** for security/dead code checking.

---

## 4. Feature Guidelines

### High-Impact Contributions

* Improving computer vision classification models.
* Adding support for new crop diseases.
* Optimizing inference pipeline performance.
* Improving documentation and local dev manuals.

---

## 5. Testing

* Run unit tests via `pytest`.
* Ensure minimum code coverage matches project configurations (>= 80%).

---

## 6. Reporting Issues

* Use GitLab/GitHub Issues to report bugs or propose enhancements.
* Provide:
  * Steps to reproduce
  * Expected vs. actual behavior
  * Sample images or datasets if applicable

---

## 7. Best Practices

* **Keep Pull Requests small**: One logical change per PR.
* **Document changes**: Update README, USER_MANUAL.md, or specs if needed.
* **Collaborate actively**: Communicate with other contributors to avoid duplicate work.

---

> Thank you for helping Crop Disease Detection improve agricultural tools. Every contribution makes the platform more effective and user-friendly!
