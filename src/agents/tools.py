"""
RiskSentinel — Agent Tools
Function tools that agents can call. These are thin wrappers around
core/ functions, with type annotations and descriptions for the LLM.
"""

import json
from typing import Annotated, Any

from pydantic import Field

from src.core import data_loader, network, contagion


# ---------------------------------------------------------------------------
# ARCHITECT TOOLS
# ---------------------------------------------------------------------------
def build_network_for_date(
    date: Annotated[str, Field(description="Date in YYYY-MM-DD format to build the correlation network for.")],
    threshold: Annotated[float, Field(description="Minimum |correlation| to create an edge. Default 0.5.")] = 0.5,
) -> str:
    """Build the S&P 500 correlation network for a specific date. Returns network summary with node/edge counts and global metrics."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    metrics = network.compute_global_metrics(G)
    return json.dumps({
        "date_requested": date,
        "date_used": str(actual_date.date()),
        "threshold": threshold,
        **metrics,
    }, indent=2)


def get_top_systemic_nodes(
    date: Annotated[str, Field(description="Date in YYYY-MM-DD format.")],
    metric: Annotated[str, Field(description="Centrality metric: degree, betweenness, closeness, eigenvector, or pagerank.")] = "pagerank",
    top_n: Annotated[int, Field(description="Number of top nodes to return.")] = 10,
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Get the most systemically important nodes by centrality metric. Returns ranked list with ticker, sector, and centrality value."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    centralities = network.compute_node_centralities(G)
    sector_dict = data_loader.get_sector_dict()
    top = network.get_top_nodes(centralities, metric=metric, top_n=top_n)
    result = [
        {"rank": i + 1, "ticker": t, "sector": sector_dict.get(t, "Unknown"), metric: round(v, 6)}
        for i, (t, v) in enumerate(top)
    ]
    return json.dumps(
        {"date": str(actual_date.date()), "metric": metric, "threshold": threshold, "top_nodes": result},
        indent=2,
    )


def get_node_connections(
    ticker: Annotated[str, Field(description="Stock ticker symbol (e.g., JPM, AAPL, NVDA).")],
    date: Annotated[str, Field(description="Date in YYYY-MM-DD format.")] = "2025-12-01",
    top_n: Annotated[int, Field(description="Number of top connections to return.")] = 15,
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Get the strongest connections (neighbors) of a stock in the correlation network. Returns list of connected stocks with correlation values."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    sector_dict = data_loader.get_sector_dict()
    neighbors = network.get_node_neighbors(G, ticker)[:top_n]
    result = [
        {"ticker": t, "sector": sector_dict.get(t, "Unknown"), "correlation": round(c, 4)}
        for t, c in neighbors
    ]
    node_sector = sector_dict.get(ticker, "Unknown")
    return json.dumps({
        "target": ticker,
        "sector": node_sector,
        "date": str(actual_date.date()),
        "threshold": threshold,
        "n_total_connections": len(list(network.get_node_neighbors(G, ticker))),
        "top_connections": result,
    }, indent=2)


def get_market_regime(
    date: Annotated[str, Field(description="Date in YYYY-MM-DD format.")] = "2025-12-01",
) -> str:
    """Get the current market regime (Calm/Normal/Elevated/High/Crisis) and VIX level for a date."""
    regimes = data_loader.load_regime_data()
    ts = data_loader.find_nearest_date(date, regimes.index.tolist())
    row = regimes.loc[ts]
    return json.dumps({
        "date": str(ts.date()),
        "regime": str(row["Regime"]),
        "vix": round(float(row["VIX"]), 2),
        "sp500_return": round(float(row["SP500_Return"]), 4),
    }, indent=2)


# ---------------------------------------------------------------------------
# QUANT (SIMULATOR) TOOLS
# ---------------------------------------------------------------------------
def run_shock_simulation(
    ticker: Annotated[str, Field(description="Stock ticker to shock (e.g., JPM, AAPL).")],
    shock_magnitude: Annotated[float, Field(description="Shock level from 0.0 to 1.0. 0.5 = 50% stress, 1.0 = full default.")] = 0.5,
    model: Annotated[str, Field(description="Contagion model: debtrank (recommended), linear_threshold, or cascade_removal.")] = "debtrank",
    date: Annotated[str, Field(description="Date for the network snapshot.")] = "2025-12-01",
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Run a contagion shock simulation. Shocks a node and propagates stress through the correlation network. Returns affected nodes, cascade depth, and systemic damage."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    result = contagion.run_shock_scenario(G, ticker, shock_magnitude, model)
    summary = result.summary()
    sector_dict = data_loader.get_sector_dict()

    # Enrich top affected with sector info
    for item in summary["top_10_affected"]:
        item["sector"] = sector_dict.get(item["ticker"], "Unknown")

    summary["date"] = str(actual_date.date())
    summary["threshold"] = threshold
    summary["total_nodes"] = G.number_of_nodes()
    summary["pct_affected"] = round(summary["n_affected"] / G.number_of_nodes() * 100, 1)

    # Sector breakdown of affected nodes
    sector_stress = {}
    for node, stress in result.node_stress.items():
        if stress > 0:
            sector = sector_dict.get(node, "Unknown")
            if sector not in sector_stress:
                sector_stress[sector] = {"count": 0, "total_stress": 0}
            sector_stress[sector]["count"] += 1
            sector_stress[sector]["total_stress"] = round(
                sector_stress[sector]["total_stress"] + stress, 4
            )
    summary["sector_breakdown"] = dict(sorted(
        sector_stress.items(), key=lambda x: x[1]["total_stress"], reverse=True
    ))

    return json.dumps(summary, indent=2)


def compare_shock_models(
    ticker: Annotated[str, Field(description="Stock ticker to shock.")],
    shock_magnitude: Annotated[float, Field(description="Shock level (0-1).")] = 0.5,
    date: Annotated[str, Field(description="Date for the network.")] = "2025-12-01",
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Compare all three contagion models on the same shock scenario. Returns side-by-side comparison of results."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    results = contagion.compare_models(G, ticker, shock_magnitude)
    comparison = {}
    for model_name, result in results.items():
        s = result.summary()
        comparison[model_name] = {
            "n_affected": s["n_affected"],
            "n_defaulted": s["n_defaulted"],
            "cascade_depth": s["cascade_depth"],
            "total_stress": s["total_stress"],
            "pct_affected": round(s["n_affected"] / G.number_of_nodes() * 100, 1),
        }
    return json.dumps({
        "ticker": ticker,
        "shock_magnitude": shock_magnitude,
        "date": str(actual_date.date()),
        "threshold": threshold,
        "models": comparison,
    }, indent=2)


def get_cascade_waves(
    ticker: Annotated[str, Field(description="Stock ticker to shock.")],
    shock_magnitude: Annotated[float, Field(description="Shock level (0-1).")] = 0.5,
    date: Annotated[str, Field(description="Date for the network.")] = "2025-12-01",
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Get detailed wave-by-wave cascade propagation. Shows which nodes get hit in each propagation wave."""
    G, actual_date = network.build_network_for_date(date, threshold=threshold)
    result = contagion.debtrank(G, ticker, shock_magnitude)
    sector_dict = data_loader.get_sector_dict()
    waves = []
    for wave_num, nodes in result.cascade_waves:
        wave_detail = {
            "wave": wave_num,
            "n_nodes_hit": len(nodes),
            "nodes": [
                {"ticker": n, "sector": sector_dict.get(n, "Unknown"),
                 "stress": round(result.node_stress[n], 4)}
                for n in nodes[:20]  # Cap at 20 per wave for readability
            ],
        }
        waves.append(wave_detail)
    return json.dumps({
        "ticker": ticker,
        "shock_magnitude": shock_magnitude,
        "date": str(actual_date.date()),
        "threshold": threshold,
        "total_waves": len(waves),
        "waves": waves,
    }, indent=2)


# ---------------------------------------------------------------------------
# ADVISOR TOOLS
# ---------------------------------------------------------------------------
def get_risk_summary(
    date: Annotated[str, Field(description="Date in YYYY-MM-DD format.")] = "2025-12-01",
    threshold: Annotated[float, Field(description="Minimum |correlation| edge threshold. Default 0.5.")] = 0.5,
) -> str:
    """Get a comprehensive risk summary: market regime, network density, top systemic nodes, and historical crisis comparison."""
    regimes = data_loader.load_regime_data()
    sector_dict = data_loader.get_sector_dict()

    ts = data_loader.find_nearest_date(date, regimes.index.tolist())
    regime_row = regimes.loc[ts]

    # Get top systemic nodes
    G, _ = network.build_network_for_date(date, threshold=threshold)
    centralities = network.compute_node_centralities(G)
    g_metrics = network.compute_global_metrics(G)
    top = network.get_top_nodes(centralities, "pagerank", 5)

    return json.dumps({
        "date": str(ts.date()),
        "regime": str(regime_row["Regime"]),
        "vix": round(float(regime_row["VIX"]), 2),
        "network": {
            "threshold": threshold,
            "density": round(float(g_metrics.get("density", 0.0)), 4),
            "avg_degree": round(float(g_metrics.get("avg_degree", 0.0)), 2),
            "avg_clustering": round(float(g_metrics.get("avg_clustering", 0.0)), 4),
            "avg_weight": round(float(g_metrics.get("avg_weight", 0.0)), 4),
        },
        "top_systemic_nodes": [
            {"ticker": t, "sector": sector_dict.get(t, "Unknown"), "pagerank": round(v, 6)}
            for t, v in top
        ],
        "crisis_events": data_loader.CRISIS_EVENTS,
    }, indent=2)


# ---------------------------------------------------------------------------
# TOOL COLLECTIONS (for easy agent assignment)
# ---------------------------------------------------------------------------
ARCHITECT_TOOLS = [
    build_network_for_date,
    get_top_systemic_nodes,
    get_node_connections,
    get_market_regime,
]

QUANT_TOOLS = [
    run_shock_simulation,
    compare_shock_models,
    get_cascade_waves,
]

ADVISOR_TOOLS = [
    get_risk_summary,
    run_shock_simulation,  # Advisor can also run simulations for context
    get_node_connections,
    get_market_regime,
]

ALL_TOOLS = list(set(ARCHITECT_TOOLS + QUANT_TOOLS + ADVISOR_TOOLS))
