# How we work on this repo

Four people, four lenses, one app. These rules exist so the merge at the end
is boring instead of painful.

## Branches

Nobody pushes to `main`. Ever.

```
main                      always runnable, always demo-ready
feature/attention-lens    Pavan
feature/shap-lens         Person B
feature/probing-lens      Person C
feature/bias-lens         Person D
```

Work on your branch, push often, open a Pull Request into `main` when your
lens runs end to end. At least one teammate reviews before merge.

Why bother: a repo with real branches, real PRs and real review comments
reads as engineering work. A repo with 40 commits called "update" on `main`
reads as a course submission. Recruiters can tell the difference in about
ten seconds.

## Commit messages

```
attention: add rollout across all layers
shap: cache explainer between runs
docs: add demo script
fix: arc view crashed on single-token input
```

Prefix with your lens. Present tense. One change per commit where you can.

## Ground rules

1. **Never load a model yourself.** Import from `shared/model_loader.py`.
   Four different tokenisers means four views that quietly disagree.
2. **Never hardcode a colour.** Import from `shared/config.py`.
3. **Test on `shared/sentences.py`.** If your lens only works on sentences
   you picked yourself, it does not work.
4. **Don't edit another person's lens folder.** Open an issue instead.
5. **`shared/` changes need a heads-up** in the group chat — everyone
   depends on it.

## Definition of done, per lens

- [ ] Runs on every probe in `shared/sentences.py` without crashing
- [ ] Wired into its tab in `app/main.py`
- [ ] Has at least one visualization a non-expert understands in 5 seconds
- [ ] Folder README updated with what you actually built
- [ ] One finding written down in `docs/` — something the lens *showed* you

That last one matters most. Four working plots is a tool. Four plots plus
four findings is a project.

## Weekly rhythm

- **Mon** — 15 min sync: what landed, what's blocked
- **Thu** — push whatever you have, even if broken, so nobody is invisible
- **Sun** — merge window; PRs reviewed and merged into `main`
