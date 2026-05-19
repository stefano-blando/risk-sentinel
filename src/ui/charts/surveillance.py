"""Plotly figures for the Surveillance tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.ui.services.surveillance import build_severity_df, compute_systemic_risk_index


def build_sector_impact_bar_figure(
    result,
    *,
    sector_dict: dict[str, str] | None,
    palette: dict[str, str],
    stress_colorscale,
    top_n: int = 10,
) -> go.Figure:
    df = build_severity_df(result, sector_dict).copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values(["Nodes Hit", "Avg Stress %"], ascending=[False, False]).head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Sector"],
            y=df["Nodes Hit"],
            marker=dict(
                color=df["Avg Stress %"],
                colorscale=stress_colorscale,
                colorbar=dict(title="Avg Stress %"),
            ),
            customdata=np.stack([df["Avg Stress %"], df["Defaulted"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Nodes Hit: %{y}<br>"
                "Avg Stress: %{customdata[0]:.1f}%<br>"
                "Defaulted: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Affected Nodes by Sector",
        height=320,
        margin=dict(l=30, r=20, t=40, b=30),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        xaxis=dict(color=palette["text_muted"], tickangle=-20),
        yaxis=dict(color=palette["text_muted"], showgrid=True, gridcolor=palette["surface_1"]),
    )
    return fig


def build_stress_tier_donut_figure(
    result,
    *,
    palette: dict[str, str],
    risk_colors: dict[str, str],
) -> go.Figure:
    tiers = {"Critical >80%": 0, "High 50-80%": 0, "Moderate 20-50%": 0, "Low 1-20%": 0}
    for node, stress in result.node_stress.items():
        if node == result.shocked_node or stress <= 0.01:
            continue
        if stress >= 0.8:
            tiers["Critical >80%"] += 1
        elif stress >= 0.5:
            tiers["High 50-80%"] += 1
        elif stress >= 0.2:
            tiers["Moderate 20-50%"] += 1
        else:
            tiers["Low 1-20%"] += 1

    labels = [key for key, value in tiers.items() if value > 0]
    values = [value for _, value in tiers.items() if value > 0]
    if not labels:
        labels = ["No stressed nodes"]
        values = [1]

    colors = {
        "Critical >80%": risk_colors["critical"],
        "High 50-80%": risk_colors["high"],
        "Moderate 20-50%": risk_colors["moderate"],
        "Low 1-20%": risk_colors["low"],
        "No stressed nodes": risk_colors["none"],
    }
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=[colors.get(label, risk_colors["none"]) for label in labels]),
                sort=False,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Stress Tier Distribution",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        legend=dict(font=dict(color=palette["text_muted"], size=10)),
    )
    return fig


def build_systemic_risk_gauge_figure(
    result,
    total_nodes: int,
    *,
    palette: dict[str, str],
) -> tuple[go.Figure, float, str]:
    score, label = compute_systemic_risk_index(result, total_nodes)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"color": palette["text_primary"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": palette["text_muted"]},
                "bar": {"color": palette["accent_warm"]},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 50], "color": "#713f12"},
                    {"range": [50, 70], "color": "#7c2d12"},
                    {"range": [70, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(
        title=f"Systemic Risk Index ({label})",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
    )
    return fig, score, label


def build_wave_trend_figure(
    result,
    *,
    palette: dict[str, str],
) -> go.Figure:
    rows = [{"Wave": 0, "Nodes Hit": 1, "Wave Stress %": round(result.shock_magnitude * 100, 2)}]
    for wave, nodes in result.cascade_waves:
        wave_stress = sum(result.node_stress.get(node, 0.0) for node in nodes)
        rows.append(
            {
                "Wave": int(wave),
                "Nodes Hit": int(len(nodes)),
                "Wave Stress %": round(float(wave_stress) * 100, 2),
            }
        )
    waves = [row["Wave"] for row in rows]
    nodes_hit = [row["Nodes Hit"] for row in rows]
    wave_stress_pct = [row["Wave Stress %"] for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=waves,
            y=nodes_hit,
            name="Nodes Hit",
            marker_color=palette["accent_cool"],
            opacity=0.85,
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=waves,
            y=wave_stress_pct,
            name="Wave Stress %",
            mode="lines+markers",
            line=dict(color=palette["accent_warm"], width=2),
            marker=dict(size=6),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Cascade Wave Dynamics",
        height=320,
        margin=dict(l=35, r=35, t=40, b=30),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        xaxis=dict(title="Wave", color=palette["text_muted"], dtick=1),
        yaxis=dict(title="Nodes Hit", color=palette["accent_cool"], showgrid=True, gridcolor=palette["surface_1"]),
        yaxis2=dict(
            title="Wave Stress %",
            overlaying="y",
            side="right",
            color=palette["accent_warm"],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=palette["text_muted"], size=10)),
    )
    return fig


def build_timeline_figure(
    network_metrics,
    *,
    palette: dict[str, str],
    crisis_events: dict[str, tuple[str, str]],
    selected_date: str | None = None,
    event_fill: str = "rgba(239, 68, 68, 0.12)",
) -> go.Figure | None:
    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=network_metrics.index,
                y=network_metrics["density"],
                name="Density",
                line=dict(color=palette["accent_cool"], width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=network_metrics.index,
                y=network_metrics["vix"] / 100,
                name="VIX / 100",
                line=dict(color=palette["accent_warm"], width=1.5),
                yaxis="y2",
            )
        )

        for name, (start, end) in crisis_events.items():
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=event_fill,
                line_width=0,
                annotation_text=name.split(" ")[0],
                annotation_position="top left",
                annotation_font_size=8,
                annotation_font_color=palette["accent_warm"],
            )

        if selected_date:
            fig.add_vline(
                x=selected_date,
                line_dash="dash",
                line_color=palette["text_primary"],
                line_width=1,
            )

        fig.update_layout(
            height=200,
            margin=dict(l=40, r=40, t=20, b=30),
            plot_bgcolor=palette["bg_main"],
            paper_bgcolor=palette["bg_main"],
            legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(size=10, color=palette["text_muted"])),
            xaxis=dict(showgrid=False, color=palette["text_muted"]),
            yaxis=dict(
                title="Density",
                showgrid=True,
                gridcolor=palette["surface_1"],
                color=palette["accent_cool"],
                range=[0, 0.8],
            ),
            yaxis2=dict(
                title="VIX/100",
                overlaying="y",
                side="right",
                showgrid=False,
                color=palette["accent_warm"],
                range=[0, 0.8],
            ),
        )
        return fig
    except Exception:
        return None
