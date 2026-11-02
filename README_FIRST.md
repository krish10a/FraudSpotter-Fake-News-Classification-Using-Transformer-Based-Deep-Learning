# FraudSpotter documentation pack — start here

1. Extract this folder into your existing local project directory (`C:\Users\ASUS\Desktop\nlp`, based on the Antigravity setup report).
2. Merge `docs/` with the existing `docs/` folder. Make sure `docs/MASTER_SPEC.md` exists afterward.
3. Review the supplied `AGENTS.md`. Compare it with the one Antigravity created; use this copy as the canonical project rules if it is more complete.
4. In Antigravity, ask it to read `AGENTS.md` and all files under `docs/`, then perform a gap analysis against the already-created scaffold. It should edit only genuine mismatches, not regenerate the whole project.
5. Ask Antigravity to validate the local scaffold and show actual output for `pytest tests -v` and `ruff check .`.
6. Initialize Git only after reviewing `.gitignore` and `git status`; confirm that no data, model weights, `.env` or secrets are staged.
7. Create the GitHub repository and connect it. Then copy its URL into the Colab notebook's `REPO_URL` cell.
8. In Colab, reuse the already-mounted Drive and verified T4. Do not forcibly clear `/content/drive`. Clone the GitHub repository and proceed to source integration.

The current setup report says Antigravity created a scaffold, and Colab verified a T4 and mounted Drive. Those are reported states, not independently re-verified by this pack. Run the checks above before training.
