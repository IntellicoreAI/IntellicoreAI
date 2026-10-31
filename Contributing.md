# Contributing to IntelliCore AI

Thank you for contributing to **IntelliCore AI**.

IntelliCore AI is developed collaboratively, with multiple team members working on different areas of the system. This guide defines the development workflow, Git conventions, Pull Request process, and security practices that should be followed by every contributor.

The goal is to keep the codebase **stable, organized, maintainable, and easy to collaborate on**.

---

## 1. Core Development Rule

### Never develop directly on `main`

The `main` branch represents the stable version of IntelliCore AI.

All development should happen on a separate branch.

```text
main
 |
 +---- feature/...
 |
 +---- fix/...
 |
 +---- docs/...
 |
 +---- experiment/...
```

Changes should be merged into `main` through a Pull Request.

---

## 2. Branch Naming Convention

Branches should use a consistent naming convention.

### Features

```text
feature/<component>/<description>
```

Examples:

```text
feature/ml/model-monitoring
feature/llm/evaluation-pipeline
feature/agentic/tool-monitoring
```

### Bug Fixes

```text
fix/<component>/<description>
```

Examples:

```text
fix/backend/api-validation
fix/ml/drift-calculation
fix/frontend/dashboard-layout
```

### Documentation

```text
docs/<description>
```

Examples:

```text
docs/system-architecture
docs/api-documentation
```

### Experiments

```text
experiment/<component>/<description>
```

Examples:

```text
experiment/ml/drift-detection
experiment/llm/evaluation-model
```

Keep branch names short, descriptive, and related to the actual work being performed.

---

## 3. Creating a New Branch

Always start from the latest `main`.

```bash
git checkout main
git pull origin main
git checkout -b feature/<component>/<description>
```

Example:

```bash
git checkout main
git pull origin main
git checkout -b feature/ml/model-monitoring
```

This reduces the possibility of starting development from an outdated version of the project.

---

## 4. Commit Guidelines

Commits should represent **one logical change**.

Use clear and descriptive commit messages.

Recommended format:

```text
<type>: <short description>
```

Common types include:

```text
feat
fix
docs
test
refactor
chore
```

Examples:

```text
feat: add model monitoring service
fix: handle missing model metadata
docs: update architecture documentation
test: add evaluation pipeline tests
refactor: simplify monitoring service
chore: update project dependencies
```

Avoid vague commit messages such as:

```text
changes
updated
final
final2
working
stuff
```

A good commit message should allow another developer to understand what changed without opening the entire commit.

---

## 5. Pull Request Workflow

When development on a feature or fix is complete:

```text
Feature Branch
      |
      v
Development
      |
      v
Commit Changes
      |
      v
Push Branch
      |
      v
Create Pull Request
      |
      v
GitHub Actions
      |
      v
Code Review
      |
      v
Merge into main
```

Every Pull Request should clearly describe:

* What was changed
* Why the change was required
* Which component was affected
* How the change was tested
* Any known limitations
* Any future work required

Keep Pull Requests focused. Avoid combining unrelated features or fixes into one PR.

---

## 6. GitHub Actions and CI

Every Pull Request should allow the GitHub Actions workflow to complete.

The IntelliCore CI pipeline currently contains checks for:

* Python
* Frontend
* Tauri / Rust
* Docker
* Security scanning

The final CI gate is:

```text
IntelliCore CI Status
```

A Pull Request should not be merged while the CI status is failing.

If CI fails:

1. Identify the failing job.
2. Understand the cause.
3. Fix the issue.
4. Push the fix to the same branch.
5. Allow CI to run again.
6. Verify that `IntelliCore CI Status` passes.

Do not ignore or bypass a failing check without understanding why it failed.

---

## 7. Code Review

Significant changes should be reviewed by at least one other team member before being merged.

Reviewers should consider:

* Correctness
* Readability
* Maintainability
* Security
* Performance
* Error handling
* Test coverage
* Compatibility with existing components
* Unnecessary complexity

Code review is a collaborative process. Review comments should focus on improving the project rather than criticizing the developer.

---

## 8. Keep `main` Stable

The `main` branch should contain the most stable version of the project.

Do not merge:

* Untested code
* Debug code
* Hard-coded credentials
* Temporary files
* Broken builds
* Unrelated changes
* Experimental functionality that is not ready

For experimental work, use an `experiment/` branch.

---

## 9. Security Rules

Security is especially important because IntelliCore AI will eventually contain AI models, APIs, services, and infrastructure configurations.

### Never commit secrets

Do not commit:

```text
.env
API keys
access tokens
passwords
private keys
cloud credentials
database credentials
service-account files
```

Use environment variables or appropriate secret-management mechanisms instead.

### If a secret is accidentally committed

Immediately:

1. Stop using the exposed credential.
2. Revoke or rotate the credential.
3. Inform the team.
4. Remove the secret from the repository/history where necessary.
5. Verify that the repository no longer contains the secret.
6. Check whether the secret was exposed anywhere else.

Deleting the file alone is not sufficient if the credential has already been committed to Git history.

---

## 10. Working on IntelliCore Components

IntelliCore AI consists of multiple major areas.

The initial development areas include:

```text
Machine Learning
        |
        +-- Model monitoring
        +-- Model evaluation
        +-- Drift detection


LLM
        |
        +-- LLM evaluation
        +-- Prompt monitoring
        +-- Generation metrics


Agentic AI
        |
        +-- Agent monitoring
        +-- Tool usage
        +-- Agent evaluation
```

Team members can work independently on their assigned components.

However, changes involving shared interfaces or architecture should be discussed with the relevant team members before implementation.

---

## 11. Shared Code and Architecture

Before introducing a major dependency, service, database, communication protocol, or architectural change, discuss it with the team.

Examples include:

* Introducing a new database
* Changing the backend architecture
* Adding a new infrastructure service
* Changing communication between services
* Changing the AI evaluation architecture
* Moving functionality between major modules
* Introducing a new external API
* Changing the monitoring architecture

The objective is to avoid situations where different components are independently developed using incompatible assumptions.

---

## 12. Before Opening a Pull Request

Before creating a Pull Request, verify:

* [ ] Code is formatted
* [ ] Tests pass locally where applicable
* [ ] No secrets are present
* [ ] No unnecessary files are committed
* [ ] Commit messages are meaningful
* [ ] Branch is based on the latest `main`
* [ ] Documentation is updated if necessary
* [ ] GitHub Actions has been triggered
* [ ] CI passes
* [ ] Pull Request description explains the change

---

## 13. Before Merging

Before merging a Pull Request, confirm:

* [ ] CI is passing
* [ ] Code has been reviewed
* [ ] Review comments have been addressed
* [ ] No unresolved issues remain
* [ ] The change belongs in `main`
* [ ] The change does not contain secrets
* [ ] The project remains buildable/testable

---

## 14. Keeping Your Branch Updated

If `main` has changed while you are working, update your branch before merging.

A simple approach is:

```bash
git checkout main
git pull origin main

git checkout <your-branch>
git merge main
```

Resolve any conflicts carefully and test the project again before opening or updating the Pull Request.

---

## 15. General Development Principle

Prefer small, understandable changes:

```text
Small Change
     |
     v
Test
     |
     v
Commit
     |
     v
Push
     |
     v
Pull Request
     |
     v
CI
     |
     v
Review
     |
     v
Merge
```

Avoid making large unrelated changes in a single branch.

The objective is to keep IntelliCore AI stable while allowing all team members to work independently.

---

## 16. Team Communication

If your work affects another contributor's component, communicate with them before making changes that could affect their implementation.

For example:

```text
ML
 |
 +------> Shared Monitoring Interface
                  ^
                  |
LLM --------------+
                  |
Agentic ----------+
```

Shared interfaces should be agreed upon before multiple components begin depending on them.

---

## 17. Final Rule

When in doubt:

> **Keep `main` stable, keep changes focused, keep secrets out of Git, and communicate before making architectural changes.**

The quality of IntelliCore AI depends not only on the code we write, but also on how consistently we work together.
