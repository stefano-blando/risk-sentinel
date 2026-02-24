# RiskSentinel: Agentic Systemic Risk Simulator

## Project Overview
**RiskSentinel** is a multi-agent system that combines Network Science and Generative AI to simulate and prevent systemic financial risk. Users ask natural-language "what-if" questions (e.g., "What happens if JPMorgan crashes 40%?"), and a squad of specialized AI agents builds the network, simulates the contagion cascade, and delivers actionable risk mitigation advice — all visualized as an interactive animated graph.

**Hackathon**: Microsoft AI Dev Days 2026 (deadline: March 15, 2026)
**Developer**: Stefano Blando (solo)
**Target Categories**: Grand Prize "AI Apps & Agents" + "Best Multi-Agent System"

## Architecture

### Agent Squad (Microsoft Agent Framework)
1. **The Architect (Network Agent)** — Builds and manages the S&P 500 correlation network topology from market data. Computes centrality metrics (degree, betweenness, eigenvector), community detection, and structural vulnerability scores. Tools: `build_network`, `compute_metrics`, `detect_communities`.
2. **The Quant (Simulator Agent)** — Executes contagion simulations: stresses nodes, propagates shocks through weighted edges, calculates cascade depth/breadth. Runs Python code in Azure Dynamic Sessions sandbox. Tools: `run_shock`, `cascade_analysis`, `code_interpreter`.
3. **The Advisor (Strategy Agent)** — Interprets simulation results, identifies critical nodes and systemic bottlenecks, generates natural-language risk reports with hedging recommendations. Tools: `generate_report`, `suggest_mitigation`, RAG on financial risk docs.
4. **Orchestrator Workflow** — Graph-based workflow (Agent Framework Workflows) managing agent coordination: User query → Architect → Quant → Advisor → Response with visualization.

### The "Wow" Flow (MVP User Story)
```
User: "What happens if JPMorgan crashes 40%?"
  │
  ▼
🔧 Architect: builds correlation network, highlights JPM's connections
  │            [interactive graph appears, 200 nodes]
  ▼
📊 Quant: propagates shock through weighted edges
  │        [animation: nodes turn red in cascade, 3 waves]
  │        → 23 firms impacted, 3 sectors, contagion depth: 4 levels
  ▼
📋 Advisor: "Goldman Sachs and BofA at highest risk due to direct
  │          correlation (ρ=0.82). Suggest hedging financials via
  │          XLF puts. Diversify into low-centrality tech names."
  ▼
User sees: animated graph + stats panel + risk report
```

### Tech Stack
- **Agent Orchestration**: Microsoft Agent Framework (`pip install agent-framework --pre`) — graph-based Workflows
- **Agent Hosting**: Azure AI Foundry Agent Service — managed runtime for agents, threads, tools
- **LLM**: Azure OpenAI GPT-4o (via Foundry endpoint)
- **Tool Integration**: Azure MCP (Model Context Protocol) for agent-to-tool communication
- **Network Engine**: NetworkX (graph construction & metrics)
- **Compute Sandbox**: Azure Container Apps Dynamic Sessions (safe Python execution by Quant)
- **Data**: S&P 500 (200 stocks, 2015-2024, 60-day rolling correlation windows)
- **Frontend**: Streamlit with pyvis/plotly for interactive graph visualization
- **Storage**: Azure Blob Storage for scenario snapshots

### Hackathon Hero Technologies Used
1. **Microsoft Agent Framework** — core multi-agent orchestration (primary hero tech)
2. **Microsoft Foundry** — Azure AI Foundry for model hosting and agent service
3. **Azure MCP** — tool integration layer for agents

### Data Pipeline
1. Load pre-computed correlation matrices (from PhD project)
2. Build weighted graph (threshold filtering on correlation strength)
3. Compute network metrics per node and global
4. Store as snapshots for time-series navigation

## Project Structure
```
├── CLAUDE.md              # This file - project instructions
├── STATUS.md              # Current progress and next steps
├── src/
│   ├── agents/            # Agent definitions and tools
│   │   ├── architect.py   # The Architect — Network Agent
│   │   ├── simulator.py   # The Quant — Simulator Agent
│   │   ├── advisor.py     # The Advisor — Strategy Agent
│   │   └── orchestrator.py # Workflow-based multi-agent orchestration
│   ├── core/              # Core simulation engine (decoupled from agents)
│   │   ├── network.py     # NetworkX graph construction & metrics
│   │   ├── contagion.py   # Shock propagation algorithms
│   │   └── data_loader.py # Data ingestion from PhD datasets
│   ├── utils/             # Helpers
│   │   └── azure_config.py # Azure/Foundry service configuration
│   └── app.py             # Streamlit main app
├── frontend/              # Additional frontend assets
├── data/                  # Processed data files (gitignored raw data)
├── tests/                 # Unit and integration tests
├── docs/                  # Hackathon submission materials
│   ├── pitch.md           # Project pitch description
│   └── demo_script.md     # 2-min video demo script
├── .codex/                # Codex CLI prompt templates for delegated tasks
├── requirements.txt
└── README.md              # Public GitHub repo README
```

## Development Conventions
- Python 3.11+, type hints on public functions
- Use `uv` for dependency management if available, else `pip`
- NetworkX for graph ops, pyvis for visualization
- All Azure credentials via environment variables (never hardcoded), loaded with `python-dotenv`
- Agents defined as Agent Framework agents with explicit tool functions
- Multi-agent coordination via Agent Framework Workflows (graph-based)
- Keep simulation logic (core/) decoupled from agent logic (agents/)
- Agent tools in agents/ call into core/ functions — agents are thin wrappers
- **MVP first**: one complete user story end-to-end before adding features
- **Demo > rigor**: impressive UX beats scientific perfection for hackathon scoring

## Dual-Tool Workflow
- **Claude Code**: architecture, agent logic, complex integration, debugging
- **Codex CLI**: boilerplate generation, data wrangling scripts, plotting code, test stubs, repetitive refactoring
- Codex prompts stored in `.codex/` folder for reproducibility

## Key PhD Assets to Reuse
- Correlation network construction: `~/Scrivania/PHD/research/active/topological-stock-prediction/`
- Network metrics computation: same project
- Statistical framework: `~/Scrivania/PHD/research/active/slld-project/src/`
- Multiple equilibria detection: `~/Scrivania/PHD/research/active/multiple-equilibria/`

## Evaluation Criteria (equal weight 20% each)
1. **Tech Implementation** — Clean code, effective use of Azure AI + Agent Framework
2. **Agentic Design & Innovation** — Creative multi-agent patterns, orchestration sophistication
3. **Real-World Impact** — Problem significance, production-readiness
4. **UX & Presentation** — Intuitive design, clear 2-min demo
5. **Category Adherence** — Alignment with selected challenge track (hero tech usage)

## Key References
- [Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Agent Framework GitHub](https://github.com/microsoft/agent-framework)
- [Azure AI Foundry Agents](https://learn.microsoft.com/en-us/agent-framework/agents/providers/azure-ai-foundry)
- [AI Dev Days Hackathon repo](https://github.com/Azure/AI-Dev-Days-Hackathon)
