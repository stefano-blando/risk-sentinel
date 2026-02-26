# Streamlit Cloud Setup (Judge-Friendly)

This setup keeps the app public while protecting Azure usage.

## 1. Deploy app

1. Push repo to GitHub (public).
2. In Streamlit Community Cloud: `New app`.
3. Select:
   - Repository: your repo
   - Branch: main
   - Main file path: `src/app.py`

## 2. Configure secrets

1. Open app `Settings` -> `Secrets`.
2. Paste content from `.streamlit/secrets.toml.example`.
3. Replace placeholders with your real Azure values.

## 3. Recommended values for hackathon demo

- `AZURE_OPENAI_DEPLOYMENT = "gpt-4o"` (quality mode)
- `AZURE_OPENAI_FALLBACK_DEPLOYMENT = "gpt-4o-mini"` (fast fallback)
- `JUDGE_ACCESS_CODE = "<your-code>"` to enable judge-only GPT
- `GPT_MAX_CALLS_PER_MINUTE_SESSION = 8`
- `GPT_MAX_CALLS_PER_MINUTE_GLOBAL = 20`
- `GPT_MAX_CALLS_PER_SESSION = 120`

## 4. How this works in app

- If `JUDGE_ACCESS_CODE` is set, GPT starts locked.
- Judge enters code in sidebar -> GPT unlocks for that session.
- If soft rate limits are hit, app auto-falls back to local simulation mode.
- No Azure keys are exposed to browser clients.

## 5. Submit to hackathon

Use in Innovation Studio:

1. App URL (public Streamlit link)
2. GitHub repo URL
3. Testing instructions:
   - "Open app URL"
   - "Use Judge access code: <code>"
   - "Run one of the demo queries from sidebar Demo Mode"
