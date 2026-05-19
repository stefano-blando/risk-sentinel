# RiskSentinel Architecture Diagram

This diagram is aligned with the current control-plane implementation and judge-facing explainability flow.

## System View

```mermaid
flowchart LR
    U[User Query] --> UI[Streamlit / Chainlit UI]
    UI --> CP[Deterministic Control Plane]
    CP --> P[Planner]
    CP --> A[Architect]
    CP --> Q[Quant]
    A --> TG[Tool Gateway]
    Q --> TG
    TG --> CORE[Core Engine: Data + Network + Contagion]
    CP --> RAG[Evidence-RAG]
    A --> ADV[Advisor]
    Q --> ADV
    RAG --> ADV
    ADV --> C[Critic Gate]
    C -->|approved| OUT[Final Answer + Mitigation Actions]
    C -->|needs repair, max 1| ADV
    OUT --> KPIS[Judge KPI Dashboard]
    OUT --> EXP[Explainability Trace + Bundle Export]
```

## Control-Plane State Machine

```mermaid
stateDiagram-v2
    [*] --> received
    received --> local_facts
    local_facts --> analysis
    analysis --> critic
    critic --> finalize: approved
    critic --> analysis: repair_once
    finalize --> [*]
```

## Guardrails

- Hard route policy (`local_fast_mode`, `gpt`, fallback paths)
- Tool gateway with timeout/retry/schema checks
- Immutable evidence ledger for deterministic references
- Critic hard gate with bounded repair loop
