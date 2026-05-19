"""Deterministic simulation services shared across the Streamlit app."""

from __future__ import annotations

import html


def build_simulation_facts_html(graph_data: dict | None, shock_result) -> str:
    if not graph_data or not shock_result:
        return ""

    summary = shock_result.summary()
    threshold = graph_data.get("threshold")
    threshold_txt = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "n/a"
    facts = [
        "<b>Simulation Facts (Deterministic)</b>",
        (
            f"• Date: {html.escape(str(graph_data.get('date', 'n/a')))} | "
            f"Threshold: {threshold_txt} | Regime: {html.escape(str(graph_data.get('regime', 'n/a')))} "
            f"(VIX {graph_data.get('vix', 0.0):.1f})"
        ),
        (
            f"• Scenario: {html.escape(summary['shocked_node'])} shock "
            f"{summary['shock_magnitude'] * 100:.0f}% with {html.escape(summary['model'])}"
        ),
        (
            f"• Cascade: {summary['cascade_depth']} waves | Affected: {summary['n_affected']} | "
            f"Defaulted: {summary['n_defaulted']}"
        ),
        f"• Total stress: {summary['total_stress']:.2f} | Avg stress: {summary['avg_stress'] * 100:.2f}%",
    ]
    return "<br>".join(facts)


def compute_compare_rows(
    G,
    tickers: list[str],
    shock_pct: int,
    model: str,
    *,
    sector_dict: dict[str, str] | None,
    contagion_module,
    max_compare_tickers: int,
) -> tuple[list[dict], dict]:
    sectors = sector_dict or {}
    rows: list[dict] = []
    result_by_ticker: dict[str, object] = {}
    for ticker in tickers[:max_compare_tickers]:
        if ticker not in G:
            continue
        result = contagion_module.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
        result_by_ticker[ticker] = result
        summary = result.summary()

        sector_stress: dict[str, float] = {}
        for node, stress in result.node_stress.items():
            if stress <= 0:
                continue
            sector = sectors.get(node, "Unknown")
            sector_stress[sector] = sector_stress.get(sector, 0.0) + stress
        top_sectors = sorted(sector_stress.items(), key=lambda item: item[1], reverse=True)[:2]
        top_sector_text = ", ".join(f"{sector} {value:.2f}" for sector, value in top_sectors) or "n/a"

        rows.append(
            {
                "ticker": ticker,
                "cascade_depth": summary["cascade_depth"],
                "n_affected": summary["n_affected"],
                "n_defaulted": summary["n_defaulted"],
                "total_stress": summary["total_stress"],
                "avg_stress_pct": summary["avg_stress"] * 100,
                "top_sectors": top_sector_text,
            }
        )
    rows.sort(key=lambda row: (-row["total_stress"], -row["cascade_depth"], row["ticker"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows, result_by_ticker


def build_compare_facts_html(
    rows: list[dict],
    date: str,
    threshold: float,
    regime: str,
    vix: float,
    shock_pct: int,
    model: str,
) -> str:
    if not rows:
        return ""
    lines = [
        "<b>Simulation Facts (Deterministic, Compare)</b>",
        (
            f"• Date: {html.escape(date)} | Threshold: {threshold:.2f} | "
            f"Regime: {html.escape(regime)} (VIX {vix:.1f}) | "
            f"Shock: {shock_pct}% | Model: {html.escape(model)}"
        ),
    ]
    for row in rows:
        lines.append(
            "• "
            f"{html.escape(row['ticker'])}: waves {row['cascade_depth']} | "
            f"affected {row['n_affected']} | defaulted {row['n_defaulted']} | "
            f"total {row['total_stress']:.2f} | avg {row['avg_stress_pct']:.2f}% | "
            f"top sectors {html.escape(row['top_sectors'])}"
        )
    return "<br>".join(lines)


def execute_build_network(
    *,
    data_loader_obj,
    network_module,
    compute_layout_fn,
    sector_dict: dict[str, str] | None,
    date_str: str,
    threshold: float,
) -> dict[str, object]:
    corr, actual_date = data_loader_obj.get_correlation_matrix(date_str)
    G = network_module.build_network(corr, threshold=threshold, sector_dict=sector_dict)
    metrics = network_module.compute_global_metrics(G)

    regimes = data_loader_obj.load_regime_data()
    ts = data_loader_obj.find_nearest_date(date_str, regimes.index.tolist())
    regime_row = regimes.loc[ts]
    pos = compute_layout_fn(G)

    graph_data = {
        "G": G,
        "date": str(actual_date.date()),
        "metrics": metrics,
        "regime": str(regime_row["Regime"]),
        "vix": float(regime_row["VIX"]),
        "threshold": threshold,
    }
    architect_message = (
        f"Network built for <b>{actual_date.date()}</b>: "
        f"{metrics['n_nodes']} nodes, {metrics['n_edges']:,} edges, "
        f"density {metrics['density']:.3f}. "
        f"Regime: <b>{regime_row['Regime']}</b> (VIX: {regime_row['VIX']:.1f})."
    )
    return {
        "G": G,
        "pos": pos,
        "graph_data": graph_data,
        "architect_message": architect_message,
    }


def execute_shock_scenario(
    *,
    G,
    ticker: str,
    shock_pct: int,
    model: str,
    sector_dict: dict[str, str] | None,
    risk_profile: str,
    network_module,
    contagion_module,
) -> dict[str, object]:
    sectors = sector_dict or {}
    if ticker not in G:
        return {
            "ok": False,
            "messages": [
                ("Sentinel", "🛡️", "agent-sentinel", f"⚠️ <b>{ticker}</b> not in network at this threshold. Try lower threshold.")
            ],
        }

    result = contagion_module.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
    summary = result.summary()

    neighbors = network_module.get_node_neighbors(G, ticker)[:5]
    neighbors_text = ", ".join(f"{name} (ρ={corr:+.2f})" for name, corr in neighbors)
    architect_message = (
        f"<b>{ticker}</b> ({sectors.get(ticker, '?')}) — "
        f"{len(list(network_module.get_node_neighbors(G, ticker)))} connections. "
        f"Strongest: {neighbors_text}."
    )

    tiers = {"Critical >80%": 0, "High 50-80%": 0, "Moderate 20-50%": 0, "Low <20%": 0}
    for node, stress in result.node_stress.items():
        if node == ticker:
            continue
        if stress >= 0.8:
            tiers["Critical >80%"] += 1
        elif stress >= 0.5:
            tiers["High 50-80%"] += 1
        elif stress >= 0.2:
            tiers["Moderate 20-50%"] += 1
        elif stress > 0.01:
            tiers["Low <20%"] += 1
    tier_text = " · ".join(f"{name}: <b>{count}</b>" for name, count in tiers.items() if count > 0)
    quant_message = (
        f"<b>{model.replace('_', ' ').title()}</b> — "
        f"{ticker} at {shock_pct}% shock.<br>"
        f"→ Cascade: <b>{summary['cascade_depth']}</b> waves<br>"
        f"→ {tier_text}<br>"
        f"→ Total systemic stress: {summary['total_stress']:.1f} "
        f"(avg {summary['avg_stress']*100:.1f}%)"
    )

    avg_stress = summary["avg_stress"] * 100
    profile_hint = {
        "conservative": "Keep higher hedge ratio and reduce beta quickly.",
        "balanced": "Balance hedging with portfolio carry.",
        "aggressive": "Use tactical hedges and preserve selective upside.",
    }.get(risk_profile, "Balance hedging with portfolio carry.")
    if avg_stress > 30:
        risk_level, risk_class = "CRITICAL", "risk-critical"
        advice = (
            f"Systemic event. Avg stress {avg_stress:.1f}%. "
            f"<b>Act now:</b> (1) Broad hedges (SPY puts), "
            f"(2) Liquidate high-centrality names, (3) Cash up. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    elif avg_stress > 15:
        risk_level, risk_class = "HIGH", "risk-high"
        advice = (
            f"Severe contagion. Avg stress {avg_stress:.1f}%. "
            f"<b>Actions:</b> (1) Sector hedges, "
            f"(2) Review {ticker} counterparty exposure, (3) Tighten stops. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    elif avg_stress > 5:
        risk_level, risk_class = "ELEVATED", "risk-elevated"
        advice = (
            f"Moderate contagion. Avg stress {avg_stress:.1f}%. "
            f"<b>Monitor:</b> (1) VIX trajectory, "
            f"(2) Direct {ticker} exposure, (3) No broad hedging yet. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    else:
        risk_level, risk_class = "LOW", "risk-low"
        advice = (
            f"Contained. Avg stress {avg_stress:.1f}%. Minimal systemic impact. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )

    advisor_message = f'Risk: <span class="{risk_class}"><b>{risk_level}</b></span><br>{advice}'
    return {
        "ok": True,
        "result": result,
        "current_wave": result.cascade_depth,
        "messages": [
            ("Architect", "🔧", "agent-architect", architect_message),
            ("Quant", "📊", "agent-quant", quant_message),
            ("Advisor", "📋", "agent-advisor", advisor_message),
        ],
    }
