"""RiskSentinel Control Plane.

Deterministic orchestration layer separated from agent prompts. Implements:
- explicit state machine / DAG transitions
- planner -> workers -> critic bounded loop
- immutable evidence ledger (E1, E2, ...)
- hard guardrails (time/steps/tool calls/revisions)
- lightweight episodic + semantic memory with TTL
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .advisor import create_advisor_agent
from .architect import create_architect_agent
from .critic import create_critic_agent
from .simulator import create_quant_agent
from .evidence_validation import validate_payload_evidence
from .tool_gateway import GatewayResult, ToolGateway
from src.core import data_loader
from .tools import (
    build_network_for_date,
    get_market_regime,
    get_node_connections,
    run_shock_simulation,
)


WORKFLOW_TRANSITIONS: dict[str, set[str]] = {
    "received": {"local_facts"},
    "local_facts": {"analysis", "finalize"},
    "analysis": {"critic", "finalize"},
    "critic": {"analysis", "finalize"},
    "finalize": set(),
}


@dataclass
class ExecutionBudget:
    max_total_sec: int = 45
    max_steps: int = 12
    max_tool_calls: int = 8
    max_revisions: int = 1


@dataclass(frozen=True)
class RoleExecutionPolicy:
    timeout_sec: int
    max_retries: int


@dataclass(frozen=True)
class RoleModelRouter:
    planner: str | None = None
    worker: str | None = None
    advisor: str | None = None
    critic: str | None = None


class PolicyEngine:
    """Deterministic execution policy for each control-plane role."""

    def __init__(self, total_timeout_sec: int) -> None:
        self.total_timeout_sec = max(20, int(total_timeout_sec))

    def role_policy(self, role: str) -> RoleExecutionPolicy:
        if role == "planner":
            return RoleExecutionPolicy(timeout_sec=min(14, self.total_timeout_sec), max_retries=1)
        if role in {"architect", "quant"}:
            return RoleExecutionPolicy(timeout_sec=min(18, self.total_timeout_sec), max_retries=1)
        if role == "advisor":
            return RoleExecutionPolicy(timeout_sec=min(20, self.total_timeout_sec), max_retries=1)
        if role == "critic":
            return RoleExecutionPolicy(timeout_sec=min(14, self.total_timeout_sec), max_retries=1)
        return RoleExecutionPolicy(timeout_sec=min(12, self.total_timeout_sec), max_retries=0)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: str
    kind: str
    content: str
    metadata: dict[str, Any]
    created_at_utc: float


@dataclass(frozen=True)
class ControlEvent:
    ts_utc: float
    step: str
    status: str  # started|completed|failed
    detail: str


class WorkflowStateMachine:
    def __init__(self) -> None:
        self.state = "received"

    def transition(self, next_state: str) -> None:
        allowed = WORKFLOW_TRANSITIONS.get(self.state, set())
        if next_state not in allowed:
            raise RuntimeError(f"Invalid transition: {self.state} -> {next_state}")
        self.state = next_state


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: list[EvidenceItem] = []

    def add(self, *, source: str, kind: str, content: str, metadata: dict[str, Any] | None = None) -> EvidenceItem:
        eid = f"E{len(self._items) + 1}"
        item = EvidenceItem(
            evidence_id=eid,
            source=source,
            kind=kind,
            content=content,
            metadata=dict(metadata or {}),
            created_at_utc=time.time(),
        )
        self._items.append(item)
        return item

    def list(self) -> list[EvidenceItem]:
        return list(self._items)

    def to_prompt(self, max_chars: int = 8000) -> str:
        chunks: list[str] = []
        used = 0
        for item in self._items:
            block = (
                f"[{item.evidence_id}] source={item.source} kind={item.kind}\n"
                f"{item.content.strip()}\n"
            )
            if used + len(block) > max_chars:
                break
            chunks.append(block)
            used += len(block)
        return "\n".join(chunks)

    def ids(self) -> list[str]:
        return [it.evidence_id for it in self._items]


class ExecutionTracker:
    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self.started = time.perf_counter()
        self.steps = 0
        self.tool_calls = 0
        self.revisions = 0

    def checkpoint(self, *, step_increment: int = 1, tool_calls_increment: int = 0) -> None:
        self.steps += step_increment
        self.tool_calls += tool_calls_increment
        elapsed = time.perf_counter() - self.started
        if elapsed > self.budget.max_total_sec:
            raise TimeoutError(f"Budget exceeded: elapsed {elapsed:.1f}s > {self.budget.max_total_sec}s")
        if self.steps > self.budget.max_steps:
            raise RuntimeError(f"Budget exceeded: steps {self.steps} > {self.budget.max_steps}")
        if self.tool_calls > self.budget.max_tool_calls:
            raise RuntimeError(f"Budget exceeded: tool calls {self.tool_calls} > {self.budget.max_tool_calls}")


class EventBus:
    def __init__(self) -> None:
        self._events: list[ControlEvent] = []

    def emit(self, *, step: str, status: str, detail: str = "") -> None:
        self._events.append(
            ControlEvent(ts_utc=time.time(), step=step, status=status, detail=detail)
        )

    def list(self) -> list[ControlEvent]:
        return list(self._events)


@dataclass
class MemoryRecord:
    ts_utc: float
    key: str
    value: str
    regime: str
    data_tag: str


class MemoryStore:
    def __init__(self, ttl_sec: int = 6 * 60 * 60, max_items: int = 200):
        self.ttl_sec = ttl_sec
        self.max_items = max_items
        self.episodic: list[MemoryRecord] = []
        self.semantic: dict[str, MemoryRecord] = {}

    def _prune(self) -> None:
        cutoff = time.time() - self.ttl_sec
        self.episodic = [r for r in self.episodic if r.ts_utc >= cutoff]
        self.semantic = {k: v for k, v in self.semantic.items() if v.ts_utc >= cutoff}
        if len(self.episodic) > self.max_items:
            self.episodic = self.episodic[-self.max_items :]

    def add_episode(self, query: str, summary: str, regime: str, data_tag: str) -> None:
        self._prune()
        self.episodic.append(
            MemoryRecord(
                ts_utc=time.time(),
                key=query,
                value=summary,
                regime=regime,
                data_tag=data_tag,
            )
        )

    def get_episode_hints(self, query: str, top_k: int = 2) -> str:
        self._prune()
        q_tokens = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        scored: list[tuple[int, MemoryRecord]] = []
        for row in self.episodic:
            row_tokens = set(re.findall(r"[a-zA-Z]{3,}", row.key.lower()))
            score = len(q_tokens & row_tokens)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        hints = [f"- prior='{r.key[:70]}' -> {r.value[:120]}" for _, r in scored[:top_k]]
        return "\n".join(hints)

    def put_semantic(self, key: str, value: str, regime: str, data_tag: str) -> None:
        self._prune()
        self.semantic[key] = MemoryRecord(
            ts_utc=time.time(),
            key=key,
            value=value,
            regime=regime,
            data_tag=data_tag,
        )

    def get_semantic(self, key: str, regime: str, data_tag: str) -> str:
        self._prune()
        row = self.semantic.get(key)
        if not row:
            return ""
        if row.regime != regime:
            return ""
        if row.data_tag != data_tag:
            return ""
        return row.value


GLOBAL_MEMORY = MemoryStore()


def _is_retryable_error(exc: Exception) -> bool:
    txt = f"{type(exc).__name__}: {exc}".lower()
    return any(sig in txt for sig in ["timeout", "timed out", "429", "rate limit", "temporarily unavailable", "service unavailable", "connection"])


def _extract_json_dict(text: str) -> dict[str, Any]:
    raw = text.strip()
    candidates = [raw]
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    obj = re.search(r"(\{[\s\S]*\})", raw)
    if obj:
        candidates.append(obj.group(1).strip())
    for item in candidates:
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _clip(text: str, limit: int = 3500) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


def _parse_scenario(query: str) -> dict[str, Any]:
    known = set(data_loader.get_sector_dict().keys())
    ticker_candidates = re.findall(r"\b([A-Z]{1,5})\b", query.upper())
    ticker = ""
    for cand in ticker_candidates:
        if cand in known:
            ticker = cand
            break
    shock_match = re.search(r"(\d{1,3})\s*%", query)
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", query)
    return {
        "ticker": ticker,
        "shock_pct": int(shock_match.group(1)) if shock_match else 40,
        "date": date_match.group(1) if date_match else "2025-12-01",
        "model": "debtrank",
    }


def _build_semantic_key(scenario: dict[str, Any]) -> str:
    material = f"{scenario.get('ticker','')}-{scenario.get('shock_pct',0)}-{scenario.get('date','')}-{scenario.get('model','debtrank')}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _gateway_result_to_text(res: GatewayResult) -> str:
    payload = {
        "tool": res.tool,
        "status": res.status,
        "error_code": res.error_code,
        "retryable": res.retryable,
        "message": res.message,
        "attempts": res.attempts,
        "data": res.data,
    }
    return json.dumps(payload, ensure_ascii=True)


async def _run_query_with_retry(
    agent,
    prompt: str,
    *,
    timeout_sec: int,
    max_retries: int,
) -> str:
    attempt = 0
    while True:
        try:
            out = await asyncio.wait_for(agent.run(prompt), timeout=timeout_sec)
            return out.text
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            attempt += 1
            await asyncio.sleep(min(1.5 * attempt, 3.0))


async def run_control_plane_workflow(
    client,
    query: str,
    timeout_sec: int = 45,
    client_factory: Callable[[str], Any] | None = None,
    model_router: RoleModelRouter | None = None,
) -> str:
    """Run planner -> workers(parallel) -> advisor -> critic with bounded revision."""
    machine = WorkflowStateMachine()
    ledger = EvidenceLedger()
    bus = EventBus()
    policy_engine = PolicyEngine(total_timeout_sec=timeout_sec)
    tracker = ExecutionTracker(
        ExecutionBudget(
            max_total_sec=max(20, int(timeout_sec)),
            max_steps=14,
            max_tool_calls=8,
            max_revisions=1,
        )
    )
    role_clients = _build_role_clients(
        default_client=client,
        client_factory=client_factory,
        router=model_router,
    )

    planner = role_clients["planner"].as_agent(
        name="ThePlanner",
        description="Deterministic planning agent for orchestration only.",
        instructions=(
            "You produce a short plan for financial contagion analysis. "
            "Return only JSON with keys: plan_steps (list max 4), objective, constraints."
        ),
        tools=[],
    )
    architect = create_architect_agent(role_clients["worker"])
    quant = create_quant_agent(role_clients["worker"])
    advisor = create_advisor_agent(role_clients["advisor"])
    critic = create_critic_agent(role_clients["critic"])

    gateway = ToolGateway(
        {
            "get_market_regime": get_market_regime,
            "build_network_for_date": build_network_for_date,
            "run_shock_simulation": run_shock_simulation,
            "get_node_connections": get_node_connections,
        },
        timeout_sec=8.0,
        max_retries=1,
    )

    scenario = _parse_scenario(query)
    memory_hint = GLOBAL_MEMORY.get_episode_hints(query)
    sem_key = _build_semantic_key(scenario)
    semantic_hint = ""

    try:
        bus.emit(step="received", status="started", detail="query accepted")
        tracker.checkpoint()
        machine.transition("local_facts")
        bus.emit(step="local_facts", status="started", detail="collect deterministic evidence")

        local_facts: dict[str, Any] = {
            "query": query,
            "scenario": scenario,
            "memory_hint": memory_hint,
            "semantic_hint": semantic_hint,
        }
        ledger.add(
            source="control_plane",
            kind="local_facts",
            content=json.dumps(local_facts, ensure_ascii=True),
            metadata={"stage": "local_facts"},
        )

        regime_res = await asyncio.to_thread(gateway.invoke, "get_market_regime", date=scenario["date"])
        tracker.checkpoint(tool_calls_increment=1)
        ledger.add(
            source="tool_gateway",
            kind="get_market_regime",
            content=_gateway_result_to_text(regime_res),
            metadata={"tool": "get_market_regime"},
        )

        net_res = await asyncio.to_thread(
            gateway.invoke,
            "build_network_for_date",
            date=scenario["date"],
            threshold=0.5,
        )
        tracker.checkpoint(tool_calls_increment=1)
        ledger.add(
            source="tool_gateway",
            kind="build_network_for_date",
            content=_gateway_result_to_text(net_res),
            metadata={"tool": "build_network_for_date"},
        )

        if scenario["ticker"]:
            conn_res = await asyncio.to_thread(
                gateway.invoke,
                "get_node_connections",
                ticker=scenario["ticker"],
                date=scenario["date"],
                top_n=12,
                threshold=0.5,
            )
            tracker.checkpoint(tool_calls_increment=1)
            ledger.add(
                source="tool_gateway",
                kind="get_node_connections",
                content=_gateway_result_to_text(conn_res),
                metadata={"tool": "get_node_connections"},
            )

            sim_res = await asyncio.to_thread(
                gateway.invoke,
                "run_shock_simulation",
                ticker=scenario["ticker"],
                shock_magnitude=max(0.0, min(1.0, scenario["shock_pct"] / 100.0)),
                model=scenario["model"],
                date=scenario["date"],
                threshold=0.5,
            )
            tracker.checkpoint(tool_calls_increment=1)
            ledger.add(
                source="tool_gateway",
                kind="run_shock_simulation",
                content=_gateway_result_to_text(sim_res),
                metadata={"tool": "run_shock_simulation"},
            )

        bus.emit(step="local_facts", status="completed", detail=f"evidence={len(ledger.list())}")

        planner_prompt = (
            "Build a short orchestration plan using deterministic evidence. Return strict JSON only.\n\n"
            f"User request:\n{query}\n\n"
            f"Evidence:\n{ledger.to_prompt(3200)}\n\n"
            "Rules: max 4 plan steps, no invented data."
        )
        planner_out = await _run_query_with_retry(
            planner,
            planner_prompt,
            timeout_sec=policy_engine.role_policy("planner").timeout_sec,
            max_retries=policy_engine.role_policy("planner").max_retries,
        )
        tracker.checkpoint()
        ledger.add(
            source="planner",
            kind="plan",
            content=_clip(planner_out, 1500),
            metadata={"role": "planner"},
        )

        machine.transition("analysis")
        bus.emit(step="analysis", status="started", detail="run architect+quant in parallel")
        regime = "unknown"
        if isinstance(regime_res.data, dict):
            regime = str(regime_res.data.get("regime", "unknown"))
        semantic_hint = GLOBAL_MEMORY.get_semantic(
            sem_key,
            regime=regime,
            data_tag="v1",
        )
        if semantic_hint:
            ledger.add(
                source="semantic_memory",
                kind="prior_scenario",
                content=semantic_hint,
                metadata={"key": sem_key, "regime": regime},
            )

        architect_prompt = (
            "Use deterministic evidence as source of truth. Analyze topology/regime/links."
            " Keep output concise and numeric.\n\n"
            f"User request:\n{query}\n\n"
            f"Evidence:\n{ledger.to_prompt(5000)}"
        )
        quant_prompt = (
            "Use deterministic evidence as source of truth. Analyze cascade impact and stress tiers."
            " Keep output concise and numeric.\n\n"
            f"User request:\n{query}\n\n"
            f"Evidence:\n{ledger.to_prompt(5000)}"
        )

        arch_task = asyncio.create_task(
            _run_query_with_retry(
                architect,
                architect_prompt,
                timeout_sec=policy_engine.role_policy("architect").timeout_sec,
                max_retries=policy_engine.role_policy("architect").max_retries,
            )
        )
        quant_task = asyncio.create_task(
            _run_query_with_retry(
                quant,
                quant_prompt,
                timeout_sec=policy_engine.role_policy("quant").timeout_sec,
                max_retries=policy_engine.role_policy("quant").max_retries,
            )
        )
        architect_out, quant_out = await asyncio.gather(arch_task, quant_task)
        tracker.checkpoint(step_increment=2)

        ledger.add(
            source="TheArchitect",
            kind="analysis",
            content=_clip(architect_out, 2200),
            metadata={"role": "architect"},
        )
        ledger.add(
            source="TheQuant",
            kind="analysis",
            content=_clip(quant_out, 2200),
            metadata={"role": "quant"},
        )
        bus.emit(step="analysis", status="completed", detail="workers completed")

        machine.transition("critic")
        bus.emit(step="critic", status="started", detail="synthesis + validation")

        advisor_prompt = (
            "Synthesize final answer using ONLY evidence references (E ids).\n"
            "Return strict JSON schema v1 only with keys: "
            "schema_version,situation,quant_results,risk_rating,actions,monitoring_triggers,evidence_used,notes,"
            "insufficient_data,uncertainty_score,confidence_reason.\n"
            "Do not invent numeric values.\n\n"
            f"User request:\n{query}\n\n"
            f"Evidence ledger:\n{ledger.to_prompt(7000)}"
        )
        advisor_out = await _run_query_with_retry(
            advisor,
            advisor_prompt,
            timeout_sec=policy_engine.role_policy("advisor").timeout_sec,
            max_retries=policy_engine.role_policy("advisor").max_retries,
        )
        tracker.checkpoint()
        candidate = _extract_json_dict(advisor_out)
        if not candidate:
            raise RuntimeError("Advisor did not return valid JSON payload")

        critic_prompt = (
            "Validate candidate JSON against evidence. Return strict JSON:\n"
            "{approved:boolean, issues:list, required_fixes:list, uncertainty_score:number, confidence_reason:string}.\n\n"
            f"User request:\n{query}\n\n"
            f"Evidence:\n{ledger.to_prompt(7000)}\n\n"
            f"Candidate JSON:\n{json.dumps(candidate, ensure_ascii=True)}"
        )
        critic_out = await _run_query_with_retry(
            critic,
            critic_prompt,
            timeout_sec=policy_engine.role_policy("critic").timeout_sec,
            max_retries=policy_engine.role_policy("critic").max_retries,
        )
        tracker.checkpoint()
        critic_json = _extract_json_dict(critic_out)

        approved = bool(critic_json.get("approved", False))
        if (not approved) and tracker.revisions < tracker.budget.max_revisions:
            tracker.revisions += 1
            revision_prompt = (
                "Revise the JSON candidate using critic feedback. Return strict JSON schema v1 only.\n\n"
                f"Critic issues: {critic_json.get('issues', [])}\n"
                f"Required fixes: {critic_json.get('required_fixes', [])}\n"
                f"Evidence ledger:\n{ledger.to_prompt(6500)}\n\n"
                f"Previous candidate:\n{json.dumps(candidate, ensure_ascii=True)}"
            )
            revised_out = await _run_query_with_retry(
                advisor,
                revision_prompt,
                timeout_sec=max(10, policy_engine.role_policy("advisor").timeout_sec - 3),
                max_retries=0,
            )
            tracker.checkpoint()
            revised_candidate = _extract_json_dict(revised_out)
            if revised_candidate:
                candidate = revised_candidate
                recheck_prompt = (
                    "Validate candidate JSON against evidence. Return strict JSON only.\n\n"
                    f"Evidence:\n{ledger.to_prompt(6500)}\n\n"
                    f"Candidate JSON:\n{json.dumps(candidate, ensure_ascii=True)}"
                )
                recheck_out = await _run_query_with_retry(
                    critic,
                    recheck_prompt,
                    timeout_sec=max(8, policy_engine.role_policy("critic").timeout_sec - 2),
                    max_retries=0,
                )
                tracker.checkpoint()
                critic_json = _extract_json_dict(recheck_out) or critic_json
                approved = bool(critic_json.get("approved", False))

        candidate.setdefault("schema_version", "v1")
        candidate.setdefault("evidence_used", ledger.ids())
        local_evidence_gate = validate_payload_evidence(
            candidate,
            allowed_e_refs=set(ledger.ids()),
            allowed_r_refs=set(),
            require_reference_for_numeric_claims=True,
            facts_available=True,
        )
        if not local_evidence_gate["approved"]:
            approved = False
            issues = list(critic_json.get("issues", []) if isinstance(critic_json, dict) else [])
            fixes = list(critic_json.get("required_fixes", []) if isinstance(critic_json, dict) else [])
            issues.extend(local_evidence_gate["issues"])
            fixes.extend(local_evidence_gate["required_fixes"])
            critic_json = {
                **(critic_json if isinstance(critic_json, dict) else {}),
                "approved": False,
                "issues": sorted(set(str(x) for x in issues if str(x).strip())),
                "required_fixes": sorted(set(str(x) for x in fixes if str(x).strip())),
            }
        candidate["validation"] = {
            "critic_approved": approved,
            "critic_issues": critic_json.get("issues", []) if critic_json else [],
            "required_fixes": critic_json.get("required_fixes", []) if critic_json else [],
            "local_evidence_gate": local_evidence_gate,
            "control_plane_state": machine.state,
            "events": [
                {
                    "step": e.step,
                    "status": e.status,
                    "detail": e.detail,
                    "ts_utc": e.ts_utc,
                }
                for e in bus.list()[-12:]
            ],
        }

        if not approved:
            candidate["insufficient_data"] = True
            candidate["uncertainty_score"] = max(float(candidate.get("uncertainty_score", 0.5)), 0.75)
            candidate["confidence_reason"] = "Critic did not approve final candidate."

        bus.emit(step="critic", status="completed", detail=f"approved={approved}")
        machine.transition("finalize")
        bus.emit(step="finalize", status="completed", detail="workflow complete")

        summary = (
            f"risk_rating={candidate.get('risk_rating', 'UNKNOWN')} "
            f"approved={approved} uncertainty={candidate.get('uncertainty_score', 0.5)}"
        )
        GLOBAL_MEMORY.add_episode(query=query, summary=summary, regime=regime, data_tag="v1")
        GLOBAL_MEMORY.put_semantic(sem_key, summary, regime=regime, data_tag="v1")

        return json.dumps(candidate)

    except Exception as exc:
        bus.emit(step=machine.state, status="failed", detail=f"{type(exc).__name__}: {exc}")
        try:
            machine.transition("finalize")
        except Exception:
            pass
        fallback = {
            "schema_version": "v1",
            "situation": ["Control plane workflow failed before approved synthesis."],
            "quant_results": [
                "Returned deterministic fallback payload.",
                f"error={type(exc).__name__}",
            ],
            "risk_rating": "ELEVATED",
            "actions": ["Retry with narrower scope and explicit ticker/date/shock."],
            "monitoring_triggers": ["Control plane completed with critic_approved=true."],
            "evidence_used": ledger.ids(),
            "notes": f"control_plane_err={type(exc).__name__}: {str(exc)[:220]}",
            "insufficient_data": True,
            "uncertainty_score": 0.8,
            "confidence_reason": "Execution guardrail or model step failed.",
            "validation": {
                "critic_approved": False,
                "control_plane_state": machine.state,
                "events": [
                    {
                        "step": e.step,
                        "status": e.status,
                        "detail": e.detail,
                        "ts_utc": e.ts_utc,
                    }
                    for e in bus.list()[-12:]
                ],
            },
        }
        return json.dumps(fallback)


def _build_role_clients(
    *,
    default_client,
    client_factory: Callable[[str], Any] | None,
    router: RoleModelRouter | None,
) -> dict[str, Any]:
    role_clients = {
        "planner": default_client,
        "worker": default_client,
        "advisor": default_client,
        "critic": default_client,
    }
    if not client_factory or router is None:
        return role_clients

    role_to_deployment = {
        "planner": router.planner,
        "worker": router.worker,
        "advisor": router.advisor,
        "critic": router.critic,
    }
    for role, deployment in role_to_deployment.items():
        if not deployment:
            continue
        try:
            role_clients[role] = client_factory(deployment)
        except Exception:
            role_clients[role] = default_client
    return role_clients
