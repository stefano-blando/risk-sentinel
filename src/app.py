"""
RiskSentinel — Streamlit App
Agentic Systemic Risk Simulator for Financial Contagion
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx

from src.core import data_loader, network, contagion

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .agent-msg {
        background: #1a1f2e;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }
    .agent-architect { border-color: #4fc3f7; }
    .agent-quant { border-color: #ff8a65; }
    .agent-advisor { border-color: #81c784; }
    .agent-sentinel { border-color: #ce93d8; }
    .risk-low { color: #00d26a; }
    .risk-moderate { color: #ffc107; }
    .risk-elevated { color: #ff8c00; }
    .risk-high { color: #ff4444; }
    .risk-critical { color: #ff0000; font-weight: bold; }
    .main-header {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .sector-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }
    .crisis-btn button {
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CRISIS PRESETS
# ---------------------------------------------------------------------------
CRISIS_PRESETS = {
    "COVID-19 Crash": {"date": "2020-03-16", "ticker": "BAC", "shock": 60, "threshold": 0.75},
    "SVB Crisis": {"date": "2023-03-13", "ticker": "SCHW", "shock": 50, "threshold": 0.6},
    "Japan Carry Trade": {"date": "2024-08-05", "ticker": "GS", "shock": 40, "threshold": 0.6},
    "Volmageddon 2018": {"date": "2018-02-08", "ticker": "JPM", "shock": 40, "threshold": 0.65},
    "Russia-Ukraine": {"date": "2022-03-01", "ticker": "XOM", "shock": 50, "threshold": 0.65},
}


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
defaults = {
    "graph_data": None, "shock_result": None, "agent_messages": [],
    "pos": None, "current_wave": -1,
    "sector_dict": None, "tickers": None,
    "chat_history": [], "comparison": None,
    "sel_date": None, "sel_ticker": "JPM", "sel_shock": 50,
    "sel_model": "debtrank", "sel_threshold": 0.5,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.sector_dict is None:
    st.session_state.sector_dict = data_loader.get_sector_dict()
if st.session_state.tickers is None:
    st.session_state.tickers = data_loader.get_ticker_list()


# ---------------------------------------------------------------------------
# NATURAL LANGUAGE PARSER (local, no LLM needed)
# ---------------------------------------------------------------------------
def parse_chat_query(query: str) -> dict | None:
    """Parse natural language shock queries into parameters.
    Returns dict with ticker, shock, date or None if not parseable.
    """
    query_upper = query.upper().strip()
    tickers = st.session_state.tickers

    # Find ticker
    found_ticker = None
    # Try company name mapping first
    name_map = {
        "JPMORGAN": "JPM", "JP MORGAN": "JPM", "GOLDMAN": "GS", "GOLDMAN SACHS": "GS",
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
        "AMAZON": "AMZN", "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META",
        "FACEBOOK": "META", "NETFLIX": "NFLX", "BANK OF AMERICA": "BAC",
        "CITIGROUP": "C", "CITI": "C", "MORGAN STANLEY": "MS",
        "EXXON": "XOM", "CHEVRON": "CVX", "PFIZER": "PFE",
        "JOHNSON": "JNJ", "VISA": "V", "MASTERCARD": "MA",
        "DISNEY": "DIS", "BOEING": "BA", "INTEL": "INTC", "AMD": "AMD",
        "WELLS FARGO": "WFC", "BLACKROCK": "BLK",
    }
    for name, ticker in name_map.items():
        if name in query_upper:
            found_ticker = ticker
            break

    if not found_ticker:
        for t in tickers:
            if re.search(rf'\b{t}\b', query_upper):
                found_ticker = t
                break

    if not found_ticker:
        return None

    # Find shock percentage
    shock = 50  # default
    pct_match = re.search(r'(\d+)\s*%', query)
    if pct_match:
        shock = min(100, max(10, int(pct_match.group(1))))

    # Find date
    date = None
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
    if date_match:
        date = date_match.group(1)

    return {"ticker": found_ticker, "shock": shock, "date": date}


# ---------------------------------------------------------------------------
# CORE: BUILD NETWORK
# ---------------------------------------------------------------------------
def do_build_network(date_str: str, threshold: float):
    corr, actual_date = data_loader.get_correlation_matrix(date_str)
    G = network.build_network(corr, threshold=threshold, sector_dict=st.session_state.sector_dict)
    metrics = network.compute_global_metrics(G)

    regimes = data_loader.load_regime_data()
    ts = data_loader.find_nearest_date(date_str, regimes.index.tolist())
    regime_row = regimes.loc[ts]

    pos = _compute_layout(G)
    st.session_state.pos = pos
    st.session_state.graph_data = {
        "G": G, "date": str(actual_date.date()), "metrics": metrics,
        "regime": str(regime_row["Regime"]), "vix": float(regime_row["VIX"]),
    }
    st.session_state.shock_result = None
    st.session_state.current_wave = -1

    st.session_state.agent_messages.append(
        ("Architect", "🔧", "agent-architect",
         f"Network built for <b>{actual_date.date()}</b>: "
         f"{metrics['n_nodes']} nodes, {metrics['n_edges']:,} edges, "
         f"density {metrics['density']:.3f}. "
         f"Regime: <b>{regime_row['Regime']}</b> (VIX: {regime_row['VIX']:.1f}).")
    )
    return G


# ---------------------------------------------------------------------------
# CORE: RUN SHOCK
# ---------------------------------------------------------------------------
def do_run_shock(G: nx.Graph, ticker: str, shock_pct: int, model: str):
    if ticker not in G:
        st.session_state.agent_messages.append(
            ("Sentinel", "🛡️", "agent-sentinel",
             f"⚠️ <b>{ticker}</b> not in network at this threshold. Try lower threshold.")
        )
        return

    result = contagion.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
    st.session_state.shock_result = result
    st.session_state.current_wave = result.cascade_depth

    sector_dict = st.session_state.sector_dict
    summary = result.summary()

    # Architect
    neighbors = network.get_node_neighbors(G, ticker)[:5]
    n_text = ", ".join(f"{t} (ρ={c:+.2f})" for t, c in neighbors)
    st.session_state.agent_messages.append(
        ("Architect", "🔧", "agent-architect",
         f"<b>{ticker}</b> ({sector_dict.get(ticker, '?')}) — "
         f"{len(list(network.get_node_neighbors(G, ticker)))} connections. "
         f"Strongest: {n_text}.")
    )

    # Quant — severity tiers
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
    tier_text = " · ".join(f"{k}: <b>{v}</b>" for k, v in tiers.items() if v > 0)

    st.session_state.agent_messages.append(
        ("Quant", "📊", "agent-quant",
         f"<b>{model.replace('_', ' ').title()}</b> — "
         f"{ticker} at {shock_pct}% shock.<br>"
         f"→ Cascade: <b>{summary['cascade_depth']}</b> waves<br>"
         f"→ {tier_text}<br>"
         f"→ Total systemic stress: {summary['total_stress']:.1f} "
         f"(avg {summary['avg_stress']*100:.1f}%)")
    )

    # Advisor
    avg_stress = summary['avg_stress'] * 100
    if avg_stress > 30:
        risk_level, risk_class = "CRITICAL", "risk-critical"
        advice = (f"Systemic event. Avg stress {avg_stress:.1f}%. "
                  f"<b>Act now:</b> (1) Broad hedges (SPY puts), "
                  f"(2) Liquidate high-centrality names, (3) Cash up.")
    elif avg_stress > 15:
        risk_level, risk_class = "HIGH", "risk-high"
        advice = (f"Severe contagion. Avg stress {avg_stress:.1f}%. "
                  f"<b>Actions:</b> (1) Sector hedges, "
                  f"(2) Review {ticker} counterparty exposure, (3) Tighten stops.")
    elif avg_stress > 5:
        risk_level, risk_class = "ELEVATED", "risk-elevated"
        advice = (f"Moderate contagion. Avg stress {avg_stress:.1f}%. "
                  f"<b>Monitor:</b> (1) VIX trajectory, "
                  f"(2) Direct {ticker} exposure, (3) No broad hedging yet.")
    else:
        risk_level, risk_class = "LOW", "risk-low"
        advice = f"Contained. Avg stress {avg_stress:.1f}%. Minimal systemic impact."

    st.session_state.agent_messages.append(
        ("Advisor", "📋", "agent-advisor",
         f'Risk: <span class="{risk_class}"><b>{risk_level}</b></span><br>{advice}')
    )


# ---------------------------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def _compute_layout(G: nx.Graph) -> dict:
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        return nx.spring_layout(G, k=0.4, iterations=80, seed=42)


def _stress_color(stress: float) -> str:
    if stress >= 0.8:
        return "#ff1744"
    elif stress >= 0.5:
        return "#ff6d00"
    elif stress >= 0.2:
        return "#ffc107"
    elif stress > 0.01:
        return "#4fc3f7"
    return "#2a3040"


def _build_base_layout(height: int = 580, title_text: str = "") -> go.Layout:
    """Shared Plotly layout for network graphs."""
    return go.Layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=40, l=10, r=10, t=50),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        title=dict(text=title_text, font=dict(color="#ff8a65", size=16),
                   x=0.5, xanchor="center") if title_text else None,
    )


def _bg_edge_trace(G: nx.Graph, pos: dict) -> go.Scatter:
    """Single trace with all edges as background (muted)."""
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.3, color="rgba(42, 48, 64, 0.25)"),
        hoverinfo="none", showlegend=False,
    )


def _stressed_edge_trace(G: nx.Graph, pos: dict, stress: dict) -> go.Scatter:
    """Single trace for edges where at least one endpoint is stressed."""
    ex, ey = [], []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        ms = max(stress.get(u, 0), stress.get(v, 0))
        if ms > 0.01:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=1.8, color="rgba(255, 100, 0, 0.45)"),
        hoverinfo="none", showlegend=False,
    )


def _node_trace(G: nx.Graph, pos: dict, stress: dict | None,
                shocked_node: str | None) -> go.Scatter:
    """Build the node scatter trace with stress coloring."""
    sector_dict = st.session_state.sector_dict
    valid = [n for n in G.nodes() if n in pos]
    nx_, ny_ = [pos[n][0] for n in valid], [pos[n][1] for n in valid]

    colors, sizes, texts, labels, outlines = [], [], [], [], []
    for node in valid:
        sector = sector_dict.get(node, "Unknown")
        s = stress.get(node, 0) if stress else 0
        is_s = node == shocked_node

        if stress:
            colors.append("#ffffff" if is_s else _stress_color(s))
            sizes.append(28 if is_s else max(5, int(5 + s * 20)))
        else:
            colors.append(data_loader.SECTOR_COLORS.get(sector, "#cccccc"))
            sizes.append(7)

        s_txt = f"<br>Stress: {s:.1%}" if stress else ""
        texts.append(f"<b>{node}</b><br>Sector: {sector}<br>Connections: {G.degree(node)}{s_txt}")
        labels.append(node if (is_s or (stress and s >= 0.5)) else "")
        outlines.append("#ff0000" if is_s else "#0e1117")

    return go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        hoverinfo="text", hovertext=texts,
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color=outlines)),
        text=labels, textposition="top center",
        textfont=dict(size=9, color="white"),
        showlegend=False,
    )


def build_graph_figure(G: nx.Graph, pos: dict) -> go.Figure:
    """Static network figure (no shock — sector-colored)."""
    return go.Figure(
        data=[_bg_edge_trace(G, pos), _node_trace(G, pos, None, None)],
        layout=_build_base_layout(),
    )


def build_animated_figure(
    G: nx.Graph, pos: dict, result, blast_radius_only: bool = False,
) -> go.Figure:
    """Animated figure with Plotly native frames — one frame per cascade wave.

    Uses 3 traces per frame:
      0 — background edges (static)
      1 — stressed edges (changes per wave)
      2 — nodes (colors/sizes change per wave)
    Play/Pause/Slider built into the Plotly chart.
    """
    sector_dict = st.session_state.sector_dict
    shocked_node = result.shocked_node
    n_waves = result.cascade_depth

    # Optional blast-radius subgraph
    if blast_radius_only:
        visible = {n for n, s in result.node_stress.items() if s > 0.01}
        visible.add(shocked_node)
        for n in list(visible):
            for nbr in G.neighbors(n):
                visible.add(nbr)
        G = G.subgraph(visible).copy()

    bg = _bg_edge_trace(G, pos)
    valid = [n for n in G.nodes() if n in pos]

    # Pre-compute stress dict per wave
    def _wave_stress(w: int) -> dict:
        s = {n: 0.0 for n in G.nodes()}
        s[shocked_node] = result.shock_magnitude
        for wn, nodes in result.cascade_waves:
            if wn <= w:
                for node in nodes:
                    s[node] = result.node_stress[node]
        return s

    # Build frames
    frames = []
    for w in range(n_waves + 1):
        ws = _wave_stress(w)
        label = f"Wave {w}/{n_waves}" if w > 0 else "Initial Shock"
        frames.append(go.Frame(
            data=[
                bg,
                _stressed_edge_trace(G, pos, ws),
                _node_trace(G, pos, ws, shocked_node),
            ],
            name=str(w),
            layout=go.Layout(title=dict(text=label, font=dict(color="#ff8a65", size=16),
                                        x=0.5, xanchor="center")),
        ))

    # Initial state = wave 0
    fig = go.Figure(data=frames[0].data, frames=frames, layout=_build_base_layout(
        height=580, title_text="Initial Shock"))

    # Slider steps
    steps = []
    for w in range(n_waves + 1):
        steps.append(dict(
            args=[[str(w)], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                 transition=dict(duration=300))],
            label=f"W{w}" if w > 0 else "💥",
            method="animate",
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.08, y=1.12, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(
                         frame=dict(duration=1200, redraw=True),
                         transition=dict(duration=400, easing="cubic-in-out"),
                         fromcurrent=True, mode="immediate",
                     )]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                     )]),
            ],
        )],
        sliders=[dict(
            active=0, steps=steps,
            x=0.1, len=0.8, y=-0.02,
            currentvalue=dict(prefix="Cascade: ", visible=True,
                              font=dict(color="#ff8a65", size=13)),
            tickcolor="#8b95a5", font=dict(color="#8b95a5"),
            bgcolor="#1a1f2e", bordercolor="#1a1f2e",
            activebgcolor="#ff8a65",
        )],
    )

    return fig


def build_severity_df(result: contagion.ShockResult) -> pd.DataFrame:
    sector_dict = st.session_state.sector_dict
    sector_data = {}
    for node, stress in result.node_stress.items():
        sector = sector_dict.get(node, "Unknown")
        if sector not in sector_data:
            sector_data[sector] = {"stresses": [], "defaulted": 0}
        sector_data[sector]["stresses"].append(stress)
        if stress >= 1.0:
            sector_data[sector]["defaulted"] += 1
    rows = []
    for sector, d in sector_data.items():
        avg = np.mean(d["stresses"])
        hit = sum(1 for s in d["stresses"] if s > 0.01)
        rows.append({
            "Sector": sector, "Avg Stress %": round(avg * 100, 1),
            "Nodes Hit": hit, "Total": len(d["stresses"]), "Defaulted": d["defaulted"],
        })
    return pd.DataFrame(rows).sort_values("Avg Stress %", ascending=False)


def build_timeline_figure() -> go.Figure | None:
    """Mini timeline chart: network density + VIX over time."""
    try:
        nm = data_loader.load_network_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["density"], name="Density",
            line=dict(color="#4fc3f7", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["vix"] / 100, name="VIX / 100",
            line=dict(color="#ff8a65", width=1.5),
            yaxis="y2",
        ))

        # Mark crises
        for name, (start, end) in data_loader.CRISIS_EVENTS.items():
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255, 0, 0, 0.1)", line_width=0,
                annotation_text=name.split(" ")[0],
                annotation_position="top left",
                annotation_font_size=8,
                annotation_font_color="#ff8a65",
            )

        # Mark selected date
        if st.session_state.graph_data:
            fig.add_vline(
                x=st.session_state.graph_data["date"],
                line_dash="dash", line_color="white", line_width=1,
            )

        fig.update_layout(
            height=200, margin=dict(l=40, r=40, t=20, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(size=10, color="#8b95a5")),
            xaxis=dict(showgrid=False, color="#8b95a5"),
            yaxis=dict(title="Density", showgrid=True, gridcolor="#1a1f2e", color="#4fc3f7", range=[0, 0.8]),
            yaxis2=dict(title="VIX/100", overlaying="y", side="right", showgrid=False,
                        color="#ff8a65", range=[0, 0.8]),
        )
        return fig
    except Exception:
        return None


def agent_message(name: str, icon: str, css_class: str, text: str):
    """Render a styled agent message."""
    st.markdown(
        f'<div class="agent-msg {css_class}">'
        f'<b>{icon} {name}</b><br>{text}</div>',
        unsafe_allow_html=True,
    )


def generate_report_text() -> str:
    """Generate a text report for download."""
    gd = st.session_state.graph_data
    sr = st.session_state.shock_result
    if not gd or not sr:
        return "No simulation results to report."

    summary = sr.summary()
    lines = [
        "=" * 60,
        "RISKSENTINEL — SYSTEMIC RISK ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Date: {gd['date']}",
        f"Market Regime: {gd['regime']} (VIX: {gd['vix']:.1f})",
        f"Network: {gd['metrics']['n_nodes']} nodes, {gd['metrics']['n_edges']} edges, "
        f"density {gd['metrics']['density']:.3f}",
        "",
        "--- SHOCK SCENARIO ---",
        f"Target: {summary['shocked_node']}",
        f"Magnitude: {summary['shock_magnitude']*100:.0f}%",
        f"Model: {summary['model']}",
        "",
        "--- RESULTS ---",
        f"Nodes Affected: {summary['n_affected']}",
        f"Nodes Defaulted: {summary['n_defaulted']}",
        f"Cascade Depth: {summary['cascade_depth']} waves",
        f"Total Systemic Stress: {summary['total_stress']:.2f}",
        f"Average Stress: {summary['avg_stress']*100:.2f}%",
        "",
        "--- TOP 10 AFFECTED ---",
    ]
    sector_dict = st.session_state.sector_dict
    for item in summary['top_10_affected']:
        t = item['ticker']
        lines.append(f"  {t:6s} ({sector_dict.get(t, '?'):30s}) stress={item['stress']*100:.1f}%")

    lines.extend(["", "--- SECTOR BREAKDOWN ---"])
    sev_df = build_severity_df(sr)
    for _, row in sev_df.iterrows():
        lines.append(f"  {row['Sector']:30s} avg={row['Avg Stress %']:5.1f}%  hit={row['Nodes Hit']}/{row['Total']}")

    lines.extend([
        "",
        "--- AGENT MESSAGES ---",
    ])
    for name, icon, css, text in st.session_state.agent_messages:
        clean = re.sub(r'<[^>]+>', '', text)
        lines.append(f"[{name}] {clean}")

    lines.extend(["", "=" * 60,
                   "Generated by RiskSentinel — Microsoft AI Dev Days Hackathon 2026",
                   "=" * 60])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ RiskSentinel")
    st.caption("Agentic Systemic Risk Simulator")
    st.divider()

    # === CRISIS PRESETS ===
    st.markdown("### ⚡ Crisis Presets")
    preset_cols = st.columns(2)
    for i, (name, params) in enumerate(CRISIS_PRESETS.items()):
        col = preset_cols[i % 2]
        if col.button(name, key=f"preset_{name}", use_container_width=True):
            st.session_state.sel_date = params["date"]
            st.session_state.sel_ticker = params["ticker"]
            st.session_state.sel_shock = params["shock"]
            st.session_state.sel_threshold = params["threshold"]
            st.session_state.agent_messages = []
            # Build + shock
            G = do_build_network(params["date"], params["threshold"])
            do_run_shock(G, params["ticker"], params["shock"], "debtrank")
            st.rerun()

    st.divider()

    # === MANUAL CONTROLS ===
    st.markdown("### 📅 Network Date")
    available_dates = data_loader.get_available_dates()
    date_strings = [str(d.date()) for d in available_dates]
    init_date = st.session_state.sel_date or date_strings[-1]
    if init_date not in date_strings:
        init_date = min(date_strings, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(init_date)))
    selected_date = st.select_slider("Date", options=date_strings, value=init_date, label_visibility="collapsed")

    st.divider()
    st.markdown("### 💥 Shock Scenario")

    shocked_ticker = st.selectbox(
        "Target stock", options=st.session_state.tickers,
        index=st.session_state.tickers.index(st.session_state.sel_ticker)
        if st.session_state.sel_ticker in st.session_state.tickers else 0,
    )
    shock_pct = st.slider("Shock %", 10, 100, st.session_state.sel_shock, 10)
    shock_model = st.selectbox("Model", ["debtrank", "linear_threshold", "cascade_removal"], index=0)
    threshold = st.slider("Corr. threshold", 0.2, 0.8, st.session_state.sel_threshold, 0.05,
                          help="Higher = sparser = more realistic contagion")

    st.divider()
    col1, col2 = st.columns(2)
    build_btn = col1.button("🔨 Build", use_container_width=True)
    shock_btn = col2.button("💥 Shock", use_container_width=True, type="primary")
    compare_btn = st.button("⚖️ Compare All 3 Models", use_container_width=True)

    st.divider()
    st.markdown("### 🏷️ Sectors")
    for sector, color in data_loader.SECTOR_COLORS.items():
        short = sector.replace("Information ", "Info ").replace("Consumer ", "Cons. ").replace("Communication ", "Comm. ")
        st.markdown(f'<span class="sector-dot" style="background:{color}"></span><small>{short}</small>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BUILD / SHOCK ACTIONS
# ---------------------------------------------------------------------------
if build_btn or st.session_state.graph_data is None:
    st.session_state.agent_messages = []
    do_build_network(selected_date, threshold)

if shock_btn and st.session_state.graph_data:
    st.session_state.comparison = None
    do_run_shock(st.session_state.graph_data["G"], shocked_ticker, shock_pct, shock_model)

if compare_btn and st.session_state.graph_data:
    G = st.session_state.graph_data["G"]
    if shocked_ticker in G:
        st.session_state.comparison = contagion.compare_models(G, shocked_ticker, shock_pct / 100.0)
        # Also set shock_result to debtrank for the main view
        st.session_state.shock_result = st.session_state.comparison["debtrank"]
        st.session_state.current_wave = st.session_state.shock_result.cascade_depth
        st.session_state.agent_messages = [
            ("Quant", "📊", "agent-quant",
             f"<b>Model Comparison</b> — {shocked_ticker} at {shock_pct}% shock. "
             f"See comparison table below the network graph.")
        ]


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    '<h1 style="color:white; margin:0;">🛡️ RiskSentinel</h1>'
    '<p style="color:#90caf9; margin:0;">Agentic Systemic Risk Simulator — S&P 500 Financial Contagion</p>'
    '</div>',
    unsafe_allow_html=True,
)

# === CHAT INPUT ===
chat_query = st.chat_input("Ask RiskSentinel... (e.g. 'What if Tesla crashes 60%?')")
if chat_query:
    st.session_state.chat_history.append(("user", chat_query))
    parsed = parse_chat_query(chat_query)
    if parsed:
        date = parsed.get("date") or selected_date
        st.session_state.agent_messages = []
        st.session_state.agent_messages.append(
            ("Sentinel", "🛡️", "agent-sentinel",
             f'Understanding query: "<i>{chat_query}</i>"<br>'
             f'→ Target: <b>{parsed["ticker"]}</b>, Shock: <b>{parsed["shock"]}%</b>')
        )
        G = do_build_network(date, threshold)
        do_run_shock(G, parsed["ticker"], parsed["shock"], "debtrank")
        st.rerun()
    else:
        st.session_state.agent_messages.append(
            ("Sentinel", "🛡️", "agent-sentinel",
             f'Could not parse query: "<i>{chat_query}</i>". '
             f'Try: "What if JPM crashes 40%?" or "Simulate AAPL 60% shock"')
        )
        st.rerun()


# Metrics bar
if st.session_state.graph_data:
    gd = st.session_state.graph_data
    m = gd["metrics"]
    sr = st.session_state.shock_result
    cols = st.columns(7 if sr else 5)
    cols[0].metric("Nodes", m["n_nodes"])
    cols[1].metric("Edges", f"{m['n_edges']:,}")
    cols[2].metric("Density", f"{m['density']:.3f}")
    cols[3].metric("Regime", gd["regime"])
    cols[4].metric("VIX", f"{gd['vix']:.1f}")
    if sr:
        summary = sr.summary()
        cols[5].metric("Cascade", f"{summary['cascade_depth']} waves")
        cols[6].metric("Avg Stress", f"{summary['avg_stress']*100:.1f}%")

# --- Main area ---
graph_col, info_col = st.columns([3, 2])

with graph_col:
    st.markdown("### 🌐 Correlation Network")

    if st.session_state.graph_data:
        G = st.session_state.graph_data["G"]
        pos = st.session_state.pos
        sr = st.session_state.shock_result

        if sr:
            blast_view = st.toggle("🎯 Blast radius only", value=False,
                                   help="Show only affected subgraph")
            fig = build_animated_figure(G, pos, sr, blast_radius_only=blast_view)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": True})
            st.markdown(
                "⚪ Shocked &nbsp; 🔴 Critical &nbsp; 🟠 High &nbsp; "
                "🟡 Moderate &nbsp; 🔵 Low &nbsp; ⚫ Unaffected &emsp; | &emsp; "
                "Use **▶ Play** or drag the **slider** below the graph")
        else:
            fig = build_graph_figure(G, pos)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": True})

    # Model comparison
    if st.session_state.comparison:
        st.markdown("### ⚖️ Model Comparison")
        comp = st.session_state.comparison
        comp_rows = []
        for model_name, res in comp.items():
            s = res.summary()
            comp_rows.append({
                "Model": model_name.replace("_", " ").title(),
                "Affected": s["n_affected"],
                "Defaulted": s["n_defaulted"],
                "Waves": s["cascade_depth"],
                "Avg Stress %": round(s["avg_stress"] * 100, 1),
                "Total Stress": round(s["total_stress"], 1),
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Avg Stress %": st.column_config.ProgressColumn(
                              min_value=0, max_value=100, format="%.1f%%"),
                      })

        # Bar chart comparison
        fig_comp = go.Figure()
        model_colors = {"Linear Threshold": "#4fc3f7", "Debtrank": "#ff8a65", "Cascade Removal": "#ce93d8"}
        for _, row in comp_df.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row["Model"],
                x=["Affected", "Defaulted", "Waves"],
                y=[row["Affected"], row["Defaulted"], row["Waves"]],
                marker_color=model_colors.get(row["Model"], "#888"),
                text=[row["Affected"], row["Defaulted"], row["Waves"]],
                textposition="auto",
            ))
        fig_comp.update_layout(
            barmode="group", height=250,
            margin=dict(l=40, r=20, t=20, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#8b95a5", size=11)),
            xaxis=dict(color="#8b95a5"), yaxis=dict(color="#8b95a5", showgrid=True, gridcolor="#1a1f2e"),
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    # Network timeline
    st.markdown("### 📈 Network Health Timeline")
    timeline_fig = build_timeline_figure()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})


with info_col:
    st.markdown("### 🤖 Agent Analysis")

    if st.session_state.agent_messages:
        for name, icon, css, text in st.session_state.agent_messages:
            agent_message(name, icon, css, text)
    else:
        st.info("Type a question below, use a **Crisis Preset**, or click **Build** → **Shock**.")

    if st.session_state.shock_result:
        sr = st.session_state.shock_result

        # Sector impact
        st.markdown("### 📊 Sector Impact")
        sev_df = build_severity_df(sr)
        st.dataframe(sev_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Avg Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                      })

        # Top nodes
        st.markdown("### 🎯 Most Vulnerable")
        affected = sr.affected_nodes[:10]
        if affected:
            df = pd.DataFrame(affected, columns=["Ticker", "Stress"])
            df["Sector"] = df["Ticker"].map(st.session_state.sector_dict)
            df["Stress %"] = (df["Stress"] * 100).round(1)
            st.dataframe(df[["Ticker", "Sector", "Stress %"]], use_container_width=True, hide_index=True,
                          column_config={
                              "Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                          })

        # Download report
        st.divider()
        report = generate_report_text()
        st.download_button(
            "📥 Download Report", report,
            file_name=f"risksentinel_report_{sr.shocked_node}_{st.session_state.graph_data['date']}.txt",
            mime="text/plain", use_container_width=True,
        )


# Footer
st.divider()
st.caption(
    "RiskSentinel — Microsoft AI Dev Days Hackathon 2026 | "
    "Built with Microsoft Agent Framework, Azure AI Foundry, NetworkX, Streamlit"
)
