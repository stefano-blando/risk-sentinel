# RiskSentinel Submission Day Checklist

Target delivery: **Sunday, March 8, 2026**.

## 1. Technical sanity (5 min)

```bash
./venv/bin/python -m pytest -q
make demo-check
make submission-bundle
make submission-audit
# opzionale (solo se credenziali Azure disponibili)
make azure-live-check
```

Expected:
- tests passing
- demo-check `5/5 passed`
- new `artifacts/submission_bundle_*.zip`
- `submission-audit` with no blockers
- `azure-live-check` passa oppure viene saltato se non attivato

## 2. Public links (blocking)

- Set final **Live Demo URL** in `site/index.html` (replace current placeholder card link).
- Set final **Video URL** in `site/index.html` (replace current placeholder card link).

After editing links:

```bash
git add site/index.html
git commit -m "chore: set final demo and video links"
git push origin main
make submission-audit
```

## 3. Submission package

Prepare these final assets:
- GitHub repository must be **public** at submission/review time
- GitHub repository URL
- GitHub Pages project site URL
- Live demo URL
- Public video URL (<= 2 min)
- Latest bundle zip from `artifacts/`

## 4. Final form QA

- Pitch and demo script align with current behavior.
- No contradictory test counts or stale claims.
- Links open without auth barriers.
- Demo path runs cleanly (Build -> Query -> Explainability -> Export).
