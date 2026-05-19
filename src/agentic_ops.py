"""Agentic operations, routing policy, and portfolio helpers for RiskSentinel."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

SECTOR_HEDGE_MAP = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def critic_round_limit(auto_repair: bool) -> int:
    """Return critic gate rounds based on auto-repair policy."""
    return 2 if auto_repair else 1


def choose_execution_policy(
    *,
    parsed: dict | None,
    complex_query: bool,
    in_scope: bool,
    agent_mode: bool,
    gpt_for_parseable_queries: bool,
    access_allowed: bool,
    selected_strategy: str,
) -> dict:
    """Deterministic routing policy for local/GPT paths."""
    route = "local_only"
    reason = []

    if not in_scope:
        route = "guardrail_block"
        reason.append("out_of_scope")
    elif parsed and (not agent_mode or not access_allowed):
        route = "local_only"
        reason.append("agent_disabled_or_locked")
    elif parsed and not complex_query and not gpt_for_parseable_queries:
        route = "local_fast_mode"
        reason.append("parseable_local_fast")
    elif agent_mode and access_allowed:
        route = "gpt"
        reason.append("gpt_enabled")
    elif not parsed:
        route = "parse_failed"
        reason.append("unparseable_without_gpt")

    effective_strategy = selected_strategy
    compare_intent = bool(parsed and len(parsed.get("tickers", [])) >= 2 and complex_query)
    if route == "gpt":
        if compare_intent:
            effective_strategy = "commentary_direct"
        elif complex_query and selected_strategy in {"orchestrator", "workflow_parallel"}:
            effective_strategy = "workflow_parallel"
        elif (not complex_query) and selected_strategy == "workflow_parallel":
            effective_strategy = "simple"
        elif selected_strategy not in {"simple", "orchestrator", "workflow_parallel"}:
            effective_strategy = "simple"

    if effective_strategy == "workflow_parallel":
        timeout_sec = 28
        max_retries = 0
    elif effective_strategy == "orchestrator":
        timeout_sec = 20
        max_retries = 0
    else:
        timeout_sec = 18
        max_retries = 1

    return {
        "route": route,
        "run_local_first": bool(parsed),
        "should_run_gpt": route == "gpt",
        "effective_strategy": effective_strategy,
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
        "reason": ", ".join(reason) if reason else "default",
    }


def build_policy_plan(
    *,
    query: str,
    parsed: dict | None,
    compare_query: bool,
    in_scope: bool,
    execution_policy: dict,
    selected_date: str,
    threshold: float,
    model_for_query: str,
    max_compare_tickers: int = 12,
) -> list[str]:
    """Deterministic planner output (Policy role) shown in explainability."""
    if not in_scope:
        return [
            "Scope guardrail check: block out-of-domain request.",
            "Return concise redirection to network/crisis/contagion query.",
        ]

    steps: list[str] = [
        f"Parse intent and parameters from query: {query[:100]}",
    ]
    if parsed:
        date = parsed.get("date") or selected_date
        shock = int(parsed.get("shock", 50))
        if compare_query:
            tickers = ", ".join(parsed.get("tickers", [])[:max_compare_tickers])
            steps.append(
                f"Executor: run local deterministic compare on [{tickers}] at {shock}% on {date} "
                f"(threshold {threshold:.2f}, model {model_for_query})."
            )
        else:
            ticker = parsed.get("ticker", "n/a")
            steps.append(
                f"Executor: build network + run local shock for {ticker} at {shock}% on {date} "
                f"(threshold {threshold:.2f}, model {model_for_query})."
            )
    else:
        steps.append("Executor: use current context facts (latest built graph/regime) if available.")

    if execution_policy.get("should_run_gpt"):
        steps.append(
            "Executor: run GPT synthesis with critic gate and auto-repair loop until approved or bounded failure."
        )
    else:
        steps.append("Executor: skip GPT and return deterministic local output (fast path).")
    return steps


def summarize_executor_log(events: list[dict], limit: int = 18) -> list[dict]:
    """Compact executor timeline from trace events."""
    rows: list[dict] = []
    for idx, evt in enumerate(events[-limit:], start=1):
        rows.append(
            {
                "step": idx,
                "event": str(evt.get("label", "")),
                "detail": str(evt.get("detail", "")),
                "t_sec": evt.get("t_sec"),
            }
        )
    return rows


def score_shock_summary(summary: dict, total_nodes: int) -> float:
    """Unified 0-100 stress score for commander/autonomous ranking."""
    n_nodes = max(1, int(total_nodes))
    avg_pct = float(summary.get("avg_stress", 0.0)) * 100.0
    aff_pct = (float(summary.get("n_affected", 0)) / n_nodes) * 100.0
    depth = min(100.0, float(summary.get("cascade_depth", 0)) * 12.5)
    default_pct = (float(summary.get("n_defaulted", 0)) / n_nodes) * 100.0
    score = 0.40 * avg_pct + 0.30 * aff_pct + 0.20 * depth + 0.10 * default_pct
    return round(max(0.0, min(100.0, score)), 2)


def _build_graph_for_analysis(
    *,
    date_str: str,
    threshold: float,
    sector_dict: dict[str, str],
    data_loader_mod,
    network_mod,
) -> tuple[nx.Graph, str, dict, str, float]:
    corr, actual_date = data_loader_mod.get_correlation_matrix(date_str)
    G = network_mod.build_network(corr, threshold=threshold, sector_dict=sector_dict)
    metrics = network_mod.compute_global_metrics(G)
    regimes = data_loader_mod.load_regime_data()
    ts = data_loader_mod.find_nearest_date(date_str, regimes.index.tolist())
    regime_row = regimes.loc[ts]
    return G, str(actual_date.date()), metrics, str(regime_row["Regime"]), float(regime_row["VIX"])


def run_scenario_commander(
    *,
    date_str: str,
    threshold: float,
    shock_pct: int,
    model: str,
    top_n: int,
    sector_dict: dict[str, str],
    data_loader_mod,
    network_mod,
    contagion_mod,
) -> dict:
    """Agentic multi-scenario commander: picks top systemic nodes and ranks outcomes."""
    G, actual_date, metrics, regime, vix = _build_graph_for_analysis(
        date_str=date_str,
        threshold=threshold,
        sector_dict=sector_dict,
        data_loader_mod=data_loader_mod,
        network_mod=network_mod,
    )
    centralities = network_mod.compute_node_centralities(G)
    seeds = [t for t, _ in network_mod.get_top_nodes(centralities, metric="pagerank", top_n=top_n)]
    rows: list[dict] = []
    for ticker in seeds:
        res = contagion_mod.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
        summ = res.summary()
        rows.append(
            {
                "ticker": ticker,
                "sector": sector_dict.get(ticker, "Unknown"),
                "cascade_depth": int(summ["cascade_depth"]),
                "n_affected": int(summ["n_affected"]),
                "n_defaulted": int(summ["n_defaulted"]),
                "avg_stress_pct": round(float(summ["avg_stress"]) * 100.0, 2),
                "total_stress": float(summ["total_stress"]),
                "risk_score": score_shock_summary(summ, G.number_of_nodes()),
            }
        )
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    plan = [
        f"Planner: select top-{top_n} seeds by PageRank on {actual_date} (threshold {threshold:.2f}).",
        f"Executor: run {len(seeds)} deterministic {model} shocks at {shock_pct}%.",
        "Executor: rank scenarios by unified risk_score and surface top vulnerabilities.",
    ]
    return {
        "mode": "scenario_commander",
        "date": actual_date,
        "regime": regime,
        "vix": round(vix, 2),
        "threshold": round(float(threshold), 3),
        "model": model,
        "shock_pct": int(shock_pct),
        "network": metrics,
        "plan_steps": plan,
        "rows": rows,
        "top_pick": rows[0] if rows else None,
    }


def run_autonomous_stress_test(
    *,
    date_str: str,
    threshold: float,
    model: str,
    shock_grid: list[int],
    max_seeds: int,
    sector_dict: dict[str, str],
    data_loader_mod,
    network_mod,
    contagion_mod,
) -> dict:
    """Autonomous mode: explore hidden fragilities without user choosing target ticker."""
    G, actual_date, metrics, regime, vix = _build_graph_for_analysis(
        date_str=date_str,
        threshold=threshold,
        sector_dict=sector_dict,
        data_loader_mod=data_loader_mod,
        network_mod=network_mod,
    )
    centralities = network_mod.compute_node_centralities(G)
    ranked = network_mod.get_top_nodes(centralities, metric="pagerank", top_n=max(30, max_seeds * 2))

    sector_seen: set[str] = set()
    seeds: list[str] = []
    for ticker, _ in ranked:
        sector = sector_dict.get(ticker, "Unknown")
        if sector not in sector_seen:
            seeds.append(ticker)
            sector_seen.add(sector)
        if len(seeds) >= max_seeds:
            break
    if len(seeds) < max_seeds:
        for ticker, _ in ranked:
            if ticker not in seeds:
                seeds.append(ticker)
            if len(seeds) >= max_seeds:
                break

    rows: list[dict] = []
    for ticker in seeds:
        for shock_pct in shock_grid:
            res = contagion_mod.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
            summ = res.summary()
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector_dict.get(ticker, "Unknown"),
                    "shock_pct": int(shock_pct),
                    "cascade_depth": int(summ["cascade_depth"]),
                    "n_affected": int(summ["n_affected"]),
                    "avg_stress_pct": round(float(summ["avg_stress"]) * 100.0, 2),
                    "risk_score": score_shock_summary(summ, G.number_of_nodes()),
                }
            )
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    top_rows = rows[: min(20, len(rows))]

    hidden = [r for r in top_rows if r["shock_pct"] <= 50 and r["risk_score"] >= 45]
    return {
        "mode": "autonomous_stress_test",
        "date": actual_date,
        "regime": regime,
        "vix": round(vix, 2),
        "threshold": round(float(threshold), 3),
        "model": model,
        "shock_grid": shock_grid,
        "network": metrics,
        "seed_tickers": seeds,
        "rows": top_rows,
        "hidden_fragilities": hidden[:8],
    }


def parse_portfolio_positions(text: str, allowed_tickers: set[str]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errs: list[str] = []
    if not text.strip():
        return rows, ["Portfolio input is empty."]

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"[,\s;]+", line) if p.strip()]
        if len(parts) < 2:
            errs.append(f"Line {line_no}: expected `TICKER,weight`.")
            continue
        ticker = parts[0].upper()
        try:
            w_raw = parts[1].replace("%", "")
            weight = float(w_raw)
        except Exception:
            errs.append(f"Line {line_no}: invalid weight `{parts[1]}`.")
            continue
        if "%" in parts[1] or weight > 1.0:
            weight = weight / 100.0
        if ticker not in allowed_tickers:
            errs.append(f"Line {line_no}: unknown ticker `{ticker}`.")
            continue
        rows.append({"ticker": ticker, "weight": float(weight)})

    total_abs = sum(abs(r["weight"]) for r in rows)
    if total_abs <= 0:
        return [], errs + ["No valid non-zero portfolio weights found."]
    for r in rows:
        r["weight_norm"] = r["weight"] / total_abs
    rows.sort(key=lambda x: abs(x["weight_norm"]), reverse=True)
    return rows, errs


def compute_business_kpi(
    *,
    expected_stress_pct: float,
    top_sector_count: int,
    risk_profile: str,
) -> dict:
    """Formalized KPI assumptions for expected loss avoided."""
    coverage_pct = min(90.0, 35.0 + 12.0 * max(1, top_sector_count))
    efficiency_map = {
        "conservative": 72.0,
        "balanced": 62.0,
        "aggressive": 52.0,
    }
    efficiency_pct = efficiency_map.get(risk_profile, 62.0)
    avoided_pct = expected_stress_pct * (coverage_pct / 100.0) * (efficiency_pct / 100.0)
    after_pct = max(0.0, expected_stress_pct - avoided_pct)
    return {
        "formula": "avoided = expected_stress * coverage * efficiency",
        "expected_stress_pct": round(float(expected_stress_pct), 2),
        "hedge_coverage_pct": round(float(coverage_pct), 2),
        "hedge_efficiency_pct": round(float(efficiency_pct), 2),
        "estimated_loss_avoided_pct": round(float(avoided_pct), 2),
        "expected_stress_pct_after_hedge": round(float(after_pct), 2),
        "assumptions": [
            "Coverage scales with number of concentrated sectors (capped).",
            f"Efficiency tied to risk profile = {risk_profile}.",
            "KPI is comparative demo metric, not PnL forecast.",
        ],
    }


def run_portfolio_copilot(
    *,
    portfolio_text: str,
    date_str: str,
    threshold: float,
    model: str,
    stress_shock_pct: int,
    risk_profile: str,
    tickers: list[str],
    sector_dict: dict[str, str],
    data_loader_mod,
    network_mod,
    contagion_mod,
) -> dict:
    """Portfolio Co-Pilot: position-aware stress diagnostics and hedge runbook."""
    allowed = set(tickers)
    positions, errors = parse_portfolio_positions(portfolio_text, allowed)
    if not positions:
        return {"ok": False, "errors": errors, "positions": [], "actions": []}

    G, actual_date, metrics, regime, vix = _build_graph_for_analysis(
        date_str=date_str,
        threshold=threshold,
        sector_dict=sector_dict,
        data_loader_mod=data_loader_mod,
        network_mod=network_mod,
    )
    rows: list[dict] = []
    for pos in positions:
        ticker = pos["ticker"]
        sim = contagion_mod.run_shock_scenario(G, ticker, stress_shock_pct / 100.0, model)
        summ = sim.summary()
        risk_score = score_shock_summary(summ, G.number_of_nodes())
        weighted_risk = abs(pos["weight_norm"]) * risk_score
        rows.append(
            {
                "ticker": ticker,
                "sector": sector_dict.get(ticker, "Unknown"),
                "weight_norm_pct": round(pos["weight_norm"] * 100.0, 2),
                "avg_stress_pct": round(float(summ["avg_stress"]) * 100.0, 2),
                "n_affected": int(summ["n_affected"]),
                "cascade_depth": int(summ["cascade_depth"]),
                "risk_score": risk_score,
                "weighted_risk": round(weighted_risk, 3),
            }
        )
    rows.sort(key=lambda r: r["weighted_risk"], reverse=True)

    sector_weights: dict[str, float] = {}
    for pos in positions:
        sector = sector_dict.get(pos["ticker"], "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(pos["weight_norm"])
    sector_rows = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [s for s, _ in sector_rows[:3]]

    actions: list[str] = []
    for sector in top_sectors:
        hedge = SECTOR_HEDGE_MAP.get(sector)
        if hedge:
            actions.append(f"Hedge {sector} concentration via {hedge} puts/collars.")
    if rows:
        actions.append(f"Reduce gross exposure in top risk name: {rows[0]['ticker']}.")
    actions.append("Set intraday trigger if VIX rises > +4 from current regime baseline.")

    expected_portfolio_stress = sum(abs(r["weight_norm_pct"]) * r["avg_stress_pct"] for r in rows) / 100.0
    kpi = compute_business_kpi(
        expected_stress_pct=expected_portfolio_stress,
        top_sector_count=len(top_sectors),
        risk_profile=risk_profile,
    )

    return {
        "ok": True,
        "errors": errors,
        "date": actual_date,
        "regime": regime,
        "vix": round(vix, 2),
        "threshold": round(float(threshold), 3),
        "model": model,
        "stress_shock_pct": int(stress_shock_pct),
        "network": metrics,
        "positions": rows,
        "sector_weights": [{"sector": s, "weight_pct": round(w * 100.0, 2)} for s, w in sector_rows],
        "actions": actions,
        "expected_stress_pct": kpi["expected_stress_pct"],
        "expected_stress_pct_after_hedge": kpi["expected_stress_pct_after_hedge"],
        "estimated_loss_avoided_pct": kpi["estimated_loss_avoided_pct"],
        "kpi": kpi,
    }


def build_auto_portfolio_from_network(
    *,
    date_str: str,
    threshold: float,
    n_positions: int,
    sector_dict: dict[str, str],
    data_loader_mod,
    network_mod,
) -> dict:
    """Generate portfolio candidates from network (pagerank + sector diversification)."""
    n_target = max(3, min(12, int(n_positions)))
    G, actual_date, metrics, regime, vix = _build_graph_for_analysis(
        date_str=date_str,
        threshold=threshold,
        sector_dict=sector_dict,
        data_loader_mod=data_loader_mod,
        network_mod=network_mod,
    )
    if G.number_of_nodes() == 0:
        return {"ok": False, "error": "Empty graph.", "portfolio_text": ""}

    pr = nx.pagerank(G, weight="abs_weight")
    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    selected: list[tuple[str, float]] = []
    used_sectors: set[str] = set()

    for ticker, score in ranked:
        sector = sector_dict.get(ticker, "Unknown")
        if sector in used_sectors:
            continue
        selected.append((ticker, float(score)))
        used_sectors.add(sector)
        if len(selected) >= n_target:
            break

    if len(selected) < n_target:
        already = {t for t, _ in selected}
        for ticker, score in ranked:
            if ticker in already:
                continue
            selected.append((ticker, float(score)))
            if len(selected) >= n_target:
                break

    if not selected:
        return {"ok": False, "error": "No ticker candidates found.", "portfolio_text": ""}

    weights = np.array([max(1e-9, s) for _, s in selected], dtype=float)
    weights = weights / weights.sum()
    rounded = np.round(weights, 3)
    drift = float(1.0 - rounded.sum())
    rounded[-1] = round(float(rounded[-1] + drift), 3)

    rows = []
    lines = []
    for (ticker, score), w in zip(selected, rounded, strict=False):
        sector = sector_dict.get(ticker, "Unknown")
        wf = float(max(0.0, w))
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "weight": wf,
                "weight_pct": round(wf * 100.0, 2),
                "pagerank": round(float(score), 6),
            }
        )
        lines.append(f"{ticker},{wf:.3f}")

    return {
        "ok": True,
        "date": actual_date,
        "regime": regime,
        "vix": round(vix, 2),
        "threshold": round(float(threshold), 3),
        "network": metrics,
        "rows": rows,
        "portfolio_text": "\n".join(lines),
        "method": "pagerank+sector_diversification",
    }


def build_full_demo_steps(now_utc: str | None = None) -> dict:
    return {
        "started_at_utc": now_utc or datetime.utcnow().isoformat(),
        "steps": [],
        "status": "running",
    }


def append_demo_step(run: dict, step: str, status: str, detail: str = "") -> None:
    run.setdefault("steps", []).append({"step": step, "status": status, "detail": detail})

