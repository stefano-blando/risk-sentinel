# RiskSentinel — Project Status

## Current Phase: Azure Ready — Wiring Agents
**Last Updated**: 2026-02-25

## Timeline
| Milestone | Target | Status |
|---|---|---|
| Project setup & architecture | Feb 22 | ✅ Done |
| Data pipeline (load PhD data → NetworkX) | Feb 22 | ✅ Done |
| Core simulation engine (contagion.py) | Feb 22 | ✅ Done |
| Agent definitions (Agent Framework) | Feb 22 | ✅ Done (offline, no LLM) |
| Streamlit frontend & graph viz | Feb 22-23 | ✅ Done |
| README.md | Feb 23 | ✅ Done |
| Git first commit & push | Feb 24 | ✅ Done |
| Azure setup (account, OpenAI resource, deployments) | Feb 25 | ✅ Done |
| Wire agents to GPT-4o (LLM-powered) | Feb 25+ | 🟡 In Progress |
| Orchestrator → Streamlit wiring | Next | ⬜ Pending |
| Demo video & pitch document | Mar 13-14 | ⬜ Pending |
| docs/pitch.md (Innovation Studio) | Next | ⬜ Pending |
| docs/demo_script.md (2-min video) | Next | ⬜ Pending |
| Final submission | Mar 15 | ⬜ Pending |

## Azure Configuration (Feb 25)
- **Subscription**: Azure subscription 1 (Pay-As-You-Go with 167€ credits, expires Mar 27)
- **Resource Group**: `rg-risksentinel-dev`
- **OpenAI Resource**: `risksentinel-swedencentral` (Sweden Central, S0)
- **Deployments**:
  - `gpt-4o` — GPT-4o (2024-11-20), Standard SKU, 1K TPM
  - `gpt-4o-mini` — GPT-4o-mini (2024-07-18), Standard SKU, 1K TPM
- **API tested**: working (confirmed Feb 25)
- **Quota note**: GlobalStandard SKU has 0 quota (blocks GPT-5.x, GPT-4.1). Standard SKU works for GPT-4o/mini. Can request quota increase via Azure portal if needed.

## Completed (Feb 22-25)

### Core Engine
- [x] `src/core/data_loader.py` — loads PhD data (210 stocks, 3081 snapshots, correlation matrices, centralities, regime data, network metrics)
- [x] `src/core/network.py` — NetworkX graph construction, 5 centrality metrics, global metrics, sector subgraphs
- [x] `src/core/contagion.py` — 3 contagion models (DebtRank, Linear Threshold, Cascade Removal), ShockResult dataclass, compare_models()
- [x] Fixed Cascade Removal formula (was too conservative — stress now propagates via edge weight * failed node stress, threshold 0.4)

### Agents (offline — templates, not LLM-powered yet)
- [x] `src/agents/tools.py` — 8 tool functions (JSON-returning wrappers for agents)
- [x] `src/agents/architect.py` — The Architect agent definition + instructions
- [x] `src/agents/simulator.py` — The Quant agent definition + instructions
- [x] `src/agents/advisor.py` — The Advisor agent definition + instructions
- [x] `src/agents/orchestrator.py` — Orchestrator with agent-as-tool pattern + simple single-agent mode
- [x] `src/utils/azure_config.py` — lazy Settings class, get_openai_client(), get_azure_credential()

### Streamlit App (`src/app.py`, ~880 lines)
- [x] Dark theme with styled agent message cards
- [x] Natural language chat input (local regex parser, company name mapping)
- [x] 5 crisis presets (COVID-19, SVB, Japan Carry Trade, Volmageddon, Russia-Ukraine)
- [x] Plotly native animated graph (Play/Pause + wave slider, smooth transitions)
- [x] Blast radius toggle (affected subgraph only)
- [x] Model comparison view (table + grouped bar chart for all 3 models)
- [x] Sector impact table with progress bars (avg stress, not binary %)
- [x] Top 10 most vulnerable nodes
- [x] Network health timeline (density + VIX over 12 years with crisis markers)
- [x] Downloadable text report
- [x] Metrics bar (nodes, edges, density, regime, VIX, cascade waves, avg stress)

### Testing
- [x] 41 tests passing (data_loader: 17, network: 8, contagion: 16)

### Infrastructure
- [x] Git repo with initial commit pushed
- [x] Python venv with all deps installed
- [x] agent-framework 1.0.0rc1 verified working
- [x] Azure OpenAI resource deployed and API tested

## Next Steps (priority order)
1. **Wire agents to GPT-4o** — replace template f-strings with real LLM calls via Agent Framework
2. **Wire orchestrator to Streamlit** — chat input → orchestrator → GPT-4o agents → real analysis
3. **docs/pitch.md** — Innovation Studio project description
4. **docs/demo_script.md** — 2-min video script
5. **Screenshot** for README.md
6. **Demo video recording** (2 min)

## Decisions Log
| Date | Decision | Rationale |
|---|---|---|
| Feb 22 | Hybrid NetShock + RiskSentinel concept | Combines multi-agent architecture with real S&P 500 data |
| Feb 22 | Unified name: **RiskSentinel** | Stronger brand |
| Feb 22 | Microsoft Agent Framework | Hero tech, successor to AutoGen+SK |
| Feb 22 | Azure AI Foundry for model hosting | GPT-4o via Foundry endpoint |
| Feb 22 | No LangGraph needed | Agent Framework Workflows cover orchestration |
| Feb 22 | MVP-first approach | One complete user story end-to-end |
| Feb 22 | Correlation threshold 0.5 default | 0.3 too dense, everything infects everything |
| Feb 23 | Plotly native frames for animation | Replaced time.sleep() blocking with smooth client-side animation |
| Feb 23 | Crisis preset thresholds tuned per date | COVID 0.75, SVB 0.6, etc. — avoids "all 100%" |
| Feb 23 | Cascade Removal formula fix | w * stress (not w/degree) + threshold 0.4 (not 0.8) |
| Feb 25 | Sweden Central region | Best EU region for Azure OpenAI Standard SKU quota availability |
| Feb 25 | GPT-4o (not GPT-5.x) | GPT-5.x requires GlobalStandard (quota 0 on free-tier account). GPT-4o works with Standard SKU |

## Notes
- Solo developer (Stefano Blando)
- Deadline: March 15, 2026
- Target: Grand Prize "AI Apps & Agents" + "Best Multi-Agent System"
- Key insight: demo > rigor — impressive UX beats scientific perfection
- 41 tests all passing
