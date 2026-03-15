# RiskSentinel

**Agentic Systemic Risk Simulator for Financial Contagion**

> *"What happens if JPMorgan crashes 40%?"* — Ask a question, watch the contagion spread across 210 S&P 500 stocks in real time.

RiskSentinel is a multi-agent AI system that combines **Network Science** and **Generative AI** to simulate systemic financial risk. A squad of specialized agents builds correlation networks, propagates shock cascades, and delivers actionable risk mitigation advice, all visualized as an interactive animated graph.

Built for the **Microsoft AI Dev Days Hackathon 2026**.

---

## Demo

![RiskSentinel Screenshot](risksentinel.jpg)

Public entry points:
- Live demo app: `https://risk-sentinel-hxq8pzyujwbmbokegefcaq.streamlit.app/`
- Video demo: `https://youtu.be/fMlGtGxb8vY`
- Project site: `https://stefano-blando.github.io/risk-sentinel/`

**Key features:**
- Natural language queries: *"What if Tesla crashes 60%?"*
- Control-plane orchestration (`Planner -> Architect+Quant -> Advisor -> Critic`) with hard guardrails
- Evidence-RAG context (historical crises + prior runs) injected with citations (`R1..Rn`)
- Reliability dashboard KPIs (critic pass-rate, factual consistency, p95 latency, fallback rate)
- One-click `Run Full Agentic Demo` flow (Build + Commander + Autonomous + Co-Pilot)
- Agentic Ops pack: Scenario Commander, Autonomous Stress Test, Portfolio Co-Pilot
- Auto-generated portfolio from network topology (PageRank + sector diversification)
- Formal business KPI formula (expected stress, coverage, efficiency, avoided loss)
- Animated cascade propagation with Play/Pause and wave slider
- 3 contagion models: DebtRank, Linear Threshold, Cascade Removal
- Side-by-side model comparison
- 5 historical crisis presets (COVID-19, SVB, Japan Carry Trade, Volmageddon, Russia-Ukraine)
- Blast radius view (affected subgraph only)
- Network health timeline (density + VIX over 12 years)
- Downloadable risk reports

---

## Architecture

```
User query
  │
  ▼
🧭 Control Plane (deterministic policy engine)
  ├─ state machine: received -> local_facts -> analysis -> critic -> finalize
  ├─ evidence ledger: E1, E2, ...
  ├─ tool gateway: timeout/retry/schema/error taxonomy
  └─ model router: lite(planner/critic) + strong(advisor)
  │
  ├─ Planner (short plan)
  ├─ Architect + Quant (parallel)
  ├─ Advisor (strict JSON synthesis)
  └─ Critic (hard validation gate, max 1 revision)
  │
  ▼
Streamlit/Chainlit output + explainability trace + judge KPIs
```

### Agent Squad

| Agent | Role | Tools |
|-------|------|-------|
| **The Planner** | Short bounded orchestration plan | No tools |
| **The Architect** | Network topology & market regime analysis | `build_network_for_date`, `get_top_systemic_nodes`, `get_node_connections`, `get_market_regime` |
| **The Quant** | Shock propagation simulation | `run_shock_simulation`, `compare_shock_models`, `get_cascade_waves` |
| **The Advisor** | Risk assessment & mitigation advice | `get_risk_summary`, `run_shock_simulation`, `get_node_connections`, `get_market_regime` |
| **The Critic** | Validation gate on deterministic evidence | No tools |

### Contagion Models

- **DebtRank** (Battiston et al. 2012) — Iterative distress propagation with saturation. Most realistic.
- **Linear Threshold** — Cascading activation when cumulative neighbor stress exceeds threshold. Shows worst-case.
- **Cascade Removal** — Structural node removal model. Most conservative.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | **Microsoft Agent Framework** (`agent-framework`) |
| Azure Integration | **Azure OpenAI** (GPT-4o / GPT-4o-mini) |
| LLM | **Azure OpenAI GPT-4o** |
| Tool Contract | **MCP-compatible JSON envelope** for tool outputs |
| Network Engine | NetworkX |
| Visualization | Streamlit + Plotly (native animation frames) |
| Data | S&P 500, 210 stocks, 3,081 daily snapshots (2013–2025) |

### Hero Technologies

1. **Microsoft Agent Framework** — Multi-agent orchestration with agent-as-tool pattern
2. **Azure OpenAI** — cloud LLM inference with deployment routing and fallback
3. **MCP-compatible Tool Contract** — structured tool result format for robust orchestration

---

## Dataset

Pre-computed from academic research (PhD project, Scuola Superiore Sant'Anna):

- **210 S&P 500 stocks** across 11 GICS sectors
- **3,081 daily network snapshots** (Sept 2013 – Dec 2025)
- **60-day rolling Pearson correlation** windows
- Node centralities (degree, betweenness, eigenvector, PageRank)
- Market regime classification (VIX-based)
- Crisis event annotations

---

## Quick Start

### Prerequisites
- Python 3.11+
- Azure OpenAI access (for agent mode)

### Install

```bash
git clone https://github.com/stefano-blando/risk-sentinel.git
cd risk-sentinel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure Azure (optional — needed for agent mode)

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Run

```bash
python -m streamlit run src/app.py
```

The app opens at `http://localhost:8501`. The simulation engine works fully offline — Azure is only needed for the LLM-powered agent analysis.

### Run (Chainlit chat)

```bash
python -m chainlit run apps/chainlit/app.py -w
```

Optional GPT control-plane mode:

```bash
CHAINLIT_USE_GPT=1 python -m chainlit run apps/chainlit/app.py -w
```

### Deploy on Streamlit Cloud

- Full guide: `docs/streamlit_cloud_setup.md`
- Secrets template: `.streamlit/secrets.toml.example`
- If PhD `data/processed` files are not present, app auto-falls back to a deterministic synthetic demo dataset.
- Optional: set `RISKSENTINEL_DATA_ROOT` in Streamlit secrets/env to point to real processed files.

### Public Links

- Repository: `https://github.com/stefano-blando/risk-sentinel`
- Project site: `https://stefano-blando.github.io/risk-sentinel/`
- Live demo app: `https://risk-sentinel-hxq8pzyujwbmbokegefcaq.streamlit.app/`
- Video demo: `https://youtu.be/fMlGtGxb8vY`

### Hackathon Docs

- Pitch: `docs/pitch.md`
- Demo script: `docs/demo_script.md`
- Teleprompter script: `docs/demo_script_teleprompter.md`
- Architecture diagram: `docs/architecture_diagram.md`

### Project Website (Judge Landing)

```bash
python -m http.server 8080 --directory site
```

Open `http://localhost:8080`.

GitHub Pages (auto-deploy from `main`):

- Workflow: `.github/workflows/pages.yml`
- Expected URL: `https://stefano-blando.github.io/risk-sentinel/`
- First-time setup on GitHub:
1. `Settings -> Pages`
2. `Build and deployment -> Source: GitHub Actions`
3. Save and rerun the latest `Deploy Site to GitHub Pages` workflow if needed

---

## Project Structure

```
├── src/
│   ├── agents/              # Agent definitions (Microsoft Agent Framework)
│   │   ├── architect.py     # The Architect — Network Agent
│   │   ├── simulator.py     # The Quant — Simulator Agent
│   │   ├── advisor.py       # The Advisor — Strategy Agent
│   │   ├── critic.py        # The Critic — Validation Agent
│   │   ├── orchestrator.py  # Orchestrator + control-plane routing
│   │   ├── control_plane.py # Policy/state-machine/evidence orchestration
│   │   ├── tool_gateway.py  # Unified deterministic tool gateway
│   │   ├── evaluation.py    # Judge/evaluation KPI helpers
│   │   ├── evidence_rag.py  # Evidence retrieval and prompt block formatting
│   │   └── tools.py         # MCP-ready JSON tool wrappers
│   ├── core/                # Simulation engine (decoupled from agents)
│   │   ├── data_loader.py   # Data ingestion from pre-computed datasets
│   │   ├── network.py       # NetworkX graph construction & metrics
│   │   └── contagion.py     # Shock propagation algorithms
│   ├── utils/
│   │   └── azure_config.py  # Azure/Foundry configuration
│   ├── agentic_ops.py       # Deterministic agentic ops (commander/autonomous/portfolio)
│   ├── reporting.py         # Action pack and JSON-safe reporting payloads
│   ├── ui_panels.py         # Explainability badges + KPI display helpers
│   └── app.py               # Streamlit main app
├── apps/
│   └── chainlit/
│       └── app.py           # Chainlit chat app
├── site/                    # One-page project landing (GitHub Pages-ready)
├── scripts/                 # CLI helpers (demo-check, submission bundle)
├── tests/                   # 65 tests (+ 1 optional Azure smoke test skipped by default)
├── docs/                    # Hackathon submission materials
├── Makefile                 # One-command QA/bundle shortcuts
├── requirements.txt
└── CLAUDE.md                # Project architecture document
```

---

## How It Works

1. **Build Network** — Load a correlation matrix for a date, threshold filter edges, create a weighted graph of 210 stocks
2. **Analyze Topology** — Compute centrality metrics, detect systemic nodes, classify market regime
3. **Simulate Shock** — Apply initial stress to a target node, propagate through weighted edges wave by wave
4. **Assess Risk** — Classify severity tiers, break down by sector, generate risk rating and hedging advice
5. **Visualize** — Animated Plotly graph with smooth transitions, sector breakdown, downloadable report

---

## Testing

```bash
pytest tests/ -v
```

Unit tests cover data loading, network construction, contagion models, control plane, gateway, evaluation, Evidence-RAG, agentic ops, and reporting serialization.

### Optional Azure Live Smoke Test

```bash
RUN_AZURE_INTEGRATION_TESTS=1 pytest -q tests/test_azure_live_integration.py
```

This test is skipped by default and runs only when Azure credentials are configured.

## Demo Reliability & Submission

```bash
make demo-check
make submission-bundle
make submission-audit
```

- `make demo-check` runs deterministic smoke checks for the 5 showcase crisis scenarios and writes `artifacts/demo_check_latest.json`.
- `make submission-bundle` creates a timestamped zip in `artifacts/` with docs, screenshot, and manifest metadata.
- `make submission-audit` verifies blockers (missing public links, missing artifacts, dirty tree) and writes `artifacts/submission_audit_latest.json`.

## Limitations

- The primary network is based on rolling Pearson correlations: it captures co-movement, not direct causality.
- Lead-lag and nonlinear dependencies are only partially represented in this version.
- Production real-time ingestion is not included; current runs rely on pre-computed snapshots (plus synthetic cloud fallback).

---

## Author

**Stefano Blando** — PhD Candidate, Scuola Superiore Sant'Anna (Pisa)
Research: Network Science, Agent-Based Models, Financial Risk

---

## License

MIT

---

*Built with Microsoft Agent Framework, Azure AI Foundry, and NetworkX for the Microsoft AI Dev Days Hackathon 2026.*
