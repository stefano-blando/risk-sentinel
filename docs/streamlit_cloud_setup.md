# Streamlit Cloud Setup

This setup keeps the app public while protecting Azure usage with soft request limits.

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
4. Use your Azure OpenAI resource endpoint in this form:
   - `https://<resource-name>.openai.azure.com/`
   - not `https://<resource-name>.api.cognitive.microsoft.com/`

## 3. Data mode on Cloud

- Default behavior: if PhD `data/processed` files are missing, RiskSentinel runs with deterministic synthetic demo data (no extra upload needed).
- Optional real-data mode: set `RISKSENTINEL_DATA_ROOT` to a folder containing processed files (including `sector_mapping.parquet` and `networks/node_centralities.pkl`).
- To disable synthetic fallback explicitly, set `RISKSENTINEL_ALLOW_SYNTHETIC_DATA = "0"`.

## 4. Recommended values for hackathon demo

- `AZURE_OPENAI_DEPLOYMENT = "gpt-4o"` (quality mode)
- `AZURE_OPENAI_FALLBACK_DEPLOYMENT = "gpt-4o-mini"` (fast fallback)
- `AZURE_OPENAI_API_VERSION = "2025-03-01-preview"` for direct OpenAI client calls
- `AZURE_OPENAI_AGENT_API_VERSION = "preview"` for Microsoft Agent Framework Responses client
- `GPT_MAX_CALLS_PER_MINUTE_SESSION = 8`
- `GPT_MAX_CALLS_PER_MINUTE_GLOBAL = 20`
- `GPT_MAX_CALLS_PER_SESSION = 120`
- `GPT_MAX_CALLS_PER_DAY_GLOBAL = 600`

## 5. How this works in app

- GPT is open by default (no unlock code required).
- If soft rate limits are hit, app auto-falls back to local simulation mode.
- No Azure keys are exposed to browser clients.

## 6. Submit to hackathon

Use in Innovation Studio:

1. App URL (public Streamlit link)
2. GitHub repo URL
3. Testing instructions:
   - "Open app URL"
   - "Run one of the demo queries from sidebar Demo Mode"
