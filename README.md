# RiskSentinel

**Agentic Systemic Risk Simulator for Financial Contagion**

> *"What happens if JPMorgan crashes 40%?"* — Ask a question, watch the contagion spread across 210 S&P 500 stocks in real time.

RiskSentinel is a multi-agent AI system that combines **Network Science** and **Generative AI** to simulate systemic financial risk. A squad of specialized agents builds correlation networks, propagates shock cascades, and delivers actionable risk mitigation advice — all visualized as an interactive animated graph.

Built for the **Microsoft AI Dev Days Hackathon 2026**.

---

## Demo

![RiskSentinel Screenshot](docs/screenshot.png)

**Key features:**
- Natural language queries: *"What if Tesla crashes 60%?"*
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
User: "What happens if JPMorgan crashes 40%?"
  │
  ▼
🛡️ Orchestrator (Agent Framework)
  │
  ├─→ 🔧 The Architect — builds S&P 500 correlation network (210 nodes)
  │     → identifies JPM's connections, centrality, market regime
  │
  ├─→ 📊 The Quant — runs DebtRank shock propagation
  │     → 3 cascade waves, severity tiers, sector breakdown
  │
  └─→ 📋 The Advisor — interprets results via GPT-4o
        → risk rating, hedging strategies, monitoring triggers
  │
  ▼
Interactive animated graph + agent analysis + risk report
```

### Agent Squad

| Agent | Role | Tools |
|-------|------|-------|
| **The Architect** | Network topology & market regime analysis | `build_network`, `get_top_systemic_nodes`, `get_node_connections`, `get_market_regime` |
| **The Quant** | Shock propagation simulation | `run_shock_simulation`, `compare_shock_models`, `get_cascade_waves` |
| **The Advisor** | Risk assessment & mitigation advice | `get_risk_summary`, `run_shock_simulation`, `get_node_connections` |
| **Orchestrator** | Routes queries, coordinates pipeline | Agent-as-tool pattern |

### Contagion Models

- **DebtRank** (Battiston et al. 2012) — Iterative distress propagation with saturation. Most realistic.
- **Linear Threshold** — Cascading activation when cumulative neighbor stress exceeds threshold. Shows worst-case.
- **Cascade Removal** — Structural node removal model. Most conservative.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | **Microsoft Agent Framework** (`agent-framework`) |
| Agent Hosting | **Azure AI Foundry** Agent Service |
| LLM | **Azure OpenAI GPT-4o** |
| Tool Integration | **Azure MCP** (Model Context Protocol) |
| Network Engine | NetworkX |
| Visualization | Streamlit + Plotly (native animation frames) |
| Data | S&P 500, 210 stocks, 3,081 daily snapshots (2013–2025) |

### Hero Technologies

1. **Microsoft Agent Framework** — Multi-agent orchestration with agent-as-tool pattern
2. **Microsoft Foundry** — Azure AI Foundry for model hosting and agent service
3. **Azure MCP** — Tool integration layer

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
git clone https://github.com/YOUR_USERNAME/risksentinel.git
cd risksentinel
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
streamlit run src/app.py
```

The app opens at `http://localhost:8501`. The simulation engine works fully offline — Azure is only needed for the LLM-powered agent analysis.

---

## Project Structure

```
├── src/
│   ├── agents/              # Agent definitions (Microsoft Agent Framework)
│   │   ├── architect.py     # The Architect — Network Agent
│   │   ├── simulator.py     # The Quant — Simulator Agent
│   │   ├── advisor.py       # The Advisor — Strategy Agent
│   │   ├── orchestrator.py  # Multi-agent orchestration
│   │   └── tools.py         # 8 tool functions for agents
│   ├── core/                # Simulation engine (decoupled from agents)
│   │   ├── data_loader.py   # Data ingestion from pre-computed datasets
│   │   ├── network.py       # NetworkX graph construction & metrics
│   │   └── contagion.py     # Shock propagation algorithms
│   ├── utils/
│   │   └── azure_config.py  # Azure/Foundry configuration
│   └── app.py               # Streamlit main app
├── tests/                   # 41 unit tests
├── docs/                    # Hackathon submission materials
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

41 tests covering data loading, network construction, and all 3 contagion models.

---

## Author

**Stefano Blando** — PhD Candidate, Scuola Superiore Sant'Anna (Pisa)
Research: Network Science, Agent-Based Models, Financial Risk

---

## License

MIT

---

*Built with Microsoft Agent Framework, Azure AI Foundry, and NetworkX for the Microsoft AI Dev Days Hackathon 2026.*
