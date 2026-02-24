# RiskSentinel — Project Status

## Current Phase: 🚀 Core Complete — Awaiting Azure
**Last Updated**: 2026-02-23

## Timeline
| Milestone | Target | Status |
|---|---|---|
| Project setup & architecture | Feb 22 | ✅ Done |
| Data pipeline (load PhD data → NetworkX) | Feb 22 | ✅ Done |
| Core simulation engine (contagion.py) | Feb 22 | ✅ Done |
| Agent definitions (Agent Framework) | Feb 22 | ✅ Done (offline, no LLM) |
| Streamlit frontend & graph viz | Feb 22-23 | ✅ Done |
| README.md | Feb 23 | ✅ Done |
| Azure integration & real agents | TBD | 🔴 Blocked (no credits yet) |
| Orchestrator → Streamlit wiring | After Azure | ⬜ Pending |
| Demo video & pitch document | Mar 13-14 | ⬜ Pending |
| docs/pitch.md (Innovation Studio) | Next | ⬜ Pending |
| docs/demo_script.md (2-min video) | Next | ⬜ Pending |
| Git first commit & push | Next | ⬜ Pending |
| Final submission | Mar 15 | ⬜ Pending |

## Completed (Feb 22-23)

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
- [x] 5 crisis presets (COVID-19, SVB, Japan Carry Trade, Volmageddon, Russia-Ukraine) with tuned thresholds
- [x] Plotly native animated graph (Play/Pause + wave slider, smooth transitions, no time.sleep blocking)
- [x] Blast radius toggle (affected subgraph only)
- [x] Model comparison view (table + grouped bar chart for all 3 models)
- [x] Sector impact table with progress bars (avg stress, not binary %)
- [x] Top 10 most vulnerable nodes
- [x] Network health timeline (density + VIX over 12 years with crisis markers)
- [x] Downloadable text report
- [x] Metrics bar (nodes, edges, density, regime, VIX, cascade waves, avg stress)
- [x] scrollZoom enabled

### Testing
- [x] `tests/test_data_loader.py` — 17 tests passed
- [x] `tests/test_network.py` — 8 tests passed
- [x] `tests/test_contagion.py` — 16 tests passed (41 total)

### Documentation
- [x] `CLAUDE.md` — full architecture document
- [x] `README.md` — public GitHub README with architecture diagram, tech stack, quick start
- [x] `.env.example` — Azure credential template
- [x] `.codex/` — 4 Codex CLI prompt templates

### Infrastructure
- [x] Git repo initialized (no commits yet)
- [x] Python venv with all deps installed
- [x] agent-framework 1.0.0rc1 verified working

## Blockers
- **Azure credits**: hackathon provides $1000 but activation link not found. Emailed support. University accounts (UniPi, Sant'Anna) have no subscriptions. Innovation Studio project page created.
- **Without Azure**: agents are template-based (f-string), not LLM-powered. Chat input uses regex, not GPT-4o. This means "Agentic Design" (20%) and "Category Adherence" (20%) scores would be near zero.

## Next Steps (priority order)
1. **Resolve Azure access** — this is THE blocker for 40% of the hackathon score
2. **Wire orchestrator to Streamlit** — chat input → orchestrator → GPT-4o agents → real analysis
3. **Git first commit + push to GitHub**
4. **docs/pitch.md** — Innovation Studio project description
5. **docs/demo_script.md** — 2-min video script
6. **Screenshot** for README.md
7. **Demo video recording** (2 min)

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

## Architecture Summary
```
User query
  → Local NL parser (regex) OR chat_input
  → do_build_network() → data_loader + network.py → NetworkX graph
  → do_run_shock() → contagion.py → ShockResult
  → Template-based agent messages (Architect/Quant/Advisor)
  → Plotly animated graph + tables + report

[PENDING: Azure integration]
User query
  → Orchestrator agent (Agent Framework)
  → Architect agent (GPT-4o + tools) → network analysis
  → Quant agent (GPT-4o + tools) → shock simulation
  → Advisor agent (GPT-4o + tools) → risk assessment
  → Streamlit renders LLM-generated insights
```

## Notes
- Solo developer (Stefano Blando)
- Deadline: March 15, 2026
- Target: Grand Prize "AI Apps & Agents" + "Best Multi-Agent System"
- Key insight: demo > rigor — impressive UX beats scientific perfection
- Codex CLI available for mechanical tasks (separate from this project)
- 41 tests all passing
