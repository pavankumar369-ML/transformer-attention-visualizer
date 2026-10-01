# Development workflow

Solo project, but run like a team one — the git history is part of what a
reviewer reads.

## Branches

`main` is always runnable and demo-ready. Each lens gets a branch:

```
feature/attention-lens
feature/shap-lens
feature/probing-lens
feature/bias-lens
feature/polish-deploy
```

Build on the branch, open a Pull Request into `main`, write a short
description of what changed and what you found, then merge. PRs with
descriptions read as engineering work; 40 commits named "update" on `main`
read as a submission.

## Commit messages

```
attention: add rollout across all layers
shap: cache explainer between runs
fix: arc view crashed on single-token input
docs: add demo GIF
```

## Ground rules

1. Load models only through `shared/model_loader.py`.
2. Colours only from `shared/config.py`.
3. Every lens must run on every probe in `shared/sentences.py`.
4. Each lens ships with one written finding in `docs/findings.md`.
