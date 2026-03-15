# RiskSentinel — Hackathon Pitch

## One-liner
RiskSentinel is an agentic systemic-risk simulator that turns "what-if" shock questions into deterministic contagion analytics, GPT-backed strategy, and audit-ready explainability.

## Problem
Portfolio teams can ask strategic risk questions quickly, but answers are often:
- not grounded in deterministic evidence,
- hard to validate,
- weak in explainability/auditability.

## Solution
RiskSentinel combines network science and multi-agent orchestration:
- deterministic shock engine (NetworkX + contagion models),
- control-plane workflow (`Planner -> Architect+Quant -> Advisor -> Critic`),
- one-click full demo flow (Build + Commander + Autonomous + Co-Pilot),
- evidence-first outputs with citations and critic gate,
- reliability metrics for critic pass-rate, factual consistency, and latency.

## Why It Is Novel
- Hard separation of control plane and LLM reasoning.
- Immutable evidence ledger (`E1..`) + critic hard gate (max 1 revision).
- Evidence-RAG from historical crises + prior runs.
- Built-in benchmark and scenario-pack evaluation for demo reliability.

## Microsoft Stack
- Microsoft Agent Framework (`agent-framework`)
- Azure OpenAI (GPT-4o / GPT-4o-mini)
- Azure-ready deployment configuration with request caps for budget control

## Live Demo Flow (2 min)
1. Ask: "What if JPM crashes 40% on 2025-12-01?"
2. Show deterministic cascade (nodes affected, waves, stress).
3. Switch to complex compare query (JPM vs GS) and show GPT control-plane output.
4. Open Explainability panel: evidence injected, critic status, route trace.
5. Show Planner/Executor/Critic badges + Judge Dashboard KPIs.
6. Export submission bundle (`report + trace + KPI + scenario eval`).

## Impact
- Faster risk triage and clearer mitigation actions.
- Stronger trust via explicit evidence and validation gates.
- Demo-safe reliability with graceful local fallback.

## Limitations (Transparent Scope)
- Correlation networks capture co-movement, not direct causality.
- This version uses pre-computed snapshots; live market ingestion is outside current scope.
- Nonlinear and lead-lag dependencies are partially represented and planned for future extensions.

## Current Status
- Core app working on Streamlit.
- Chainlit chat app available (`apps/chainlit/app.py`).
- Test suite passing (`65 passed, 1 skipped`) and deterministic demo checks passing.

## Submission Links
- Repository: `https://github.com/stefano-blando/risk-sentinel`
- Project site: `https://stefano-blando.github.io/risk-sentinel/`
- Live demo app: `https://risk-sentinel-hxq8pzyujwbmbokegefcaq.streamlit.app/`
- Video demo: pending final public URL
