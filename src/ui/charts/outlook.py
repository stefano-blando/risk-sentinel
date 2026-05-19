"""Plotly figures for the Outlook tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


REGIME_LABELS = {
    0: "Calm",
    1: "Normal",
    2: "Elevated",
    3: "High",
    4: "Crisis",
}

REGIME_FILL = {
    "Calm": "rgba(34, 197, 94, 0.08)",
    "Normal": "rgba(56, 189, 248, 0.08)",
    "Elevated": "rgba(251, 191, 36, 0.08)",
    "High": "rgba(251, 146, 60, 0.09)",
    "Crisis": "rgba(239, 68, 68, 0.10)",
}


def _regime_label(value: float | int | None) -> str:
    if value is None:
        return "Unknown"
    try:
        return REGIME_LABELS.get(int(round(float(value))), "Unknown")
    except Exception:
        return "Unknown"


def format_outlook_metric_label(metric: str) -> str:
    labels = {
        "density": "Network Density",
        "avg_abs_weight": "Average Absolute Correlation",
        "avg_clustering": "Average Clustering",
        "risk_pressure": "Risk Pressure",
        "regime_numeric": "Regime Score",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def _add_regime_bands(fig: go.Figure, joined: pd.DataFrame) -> None:
    if joined is None or joined.empty or "regime_numeric" not in joined.columns:
        return
    regime_series = joined["regime_numeric"].dropna()
    if regime_series.empty:
        return
    values = regime_series.round().astype(int)
    segment_start = regime_series.index[0]
    segment_value = int(values.iloc[0])
    prev_ts = regime_series.index[0]
    for ts, value in zip(regime_series.index[1:], values.iloc[1:]):
        current_value = int(value)
        if current_value != segment_value:
            fig.add_vrect(
                x0=segment_start,
                x1=ts,
                fillcolor=REGIME_FILL.get(_regime_label(segment_value), "rgba(56, 189, 248, 0.06)"),
                opacity=1.0,
                line_width=0,
                layer="below",
            )
            segment_start = ts
            segment_value = current_value
        prev_ts = ts
    fig.add_vrect(
        x0=segment_start,
        x1=prev_ts,
        fillcolor=REGIME_FILL.get(_regime_label(segment_value), "rgba(56, 189, 248, 0.06)"),
        opacity=1.0,
        line_width=0,
        layer="below",
    )


def build_outlook_timeseries_figure(
    joined: pd.DataFrame,
    metric: str,
    palette: dict[str, str],
    focus_date: str | pd.Timestamp | None = None,
) -> go.Figure | None:
    if joined is None or joined.empty:
        return None
    pred_col = f"pred_{metric}"
    if metric not in joined.columns or pred_col not in joined.columns:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined[metric],
            mode="lines",
            name="Actual",
            line=dict(color=palette["accent_cool"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined[pred_col],
            mode="lines",
            name="Forecast",
            line=dict(color=palette["accent_warm"], width=2.5, dash="dash"),
        )
    )
    if focus_date is not None:
        focus_ts = pd.Timestamp(focus_date)
        if focus_ts in joined.index:
            fig.add_vline(x=focus_ts, line_width=1.2, line_dash="dot", line_color=palette["text_muted"], opacity=0.7)
            fig.add_trace(
                go.Scatter(
                    x=[focus_ts],
                    y=[float(joined.loc[focus_ts, metric])],
                    mode="markers",
                    name="Actual Focus",
                    marker=dict(color=palette["accent_cool"], size=9, line=dict(color=palette["bg_main"], width=1)),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[focus_ts],
                    y=[float(joined.loc[focus_ts, pred_col])],
                    mode="markers",
                    name="Forecast Focus",
                    marker=dict(
                        color=palette["accent_warm"],
                        size=9,
                        symbol="diamond",
                        line=dict(color=palette["bg_main"], width=1),
                    ),
                    showlegend=False,
                )
            )
    _add_regime_bands(fig, joined)
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=20, t=30, b=30),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=palette["text_muted"], size=11)),
        xaxis=dict(color=palette["text_muted"], showgrid=False),
        yaxis=dict(color=palette["text_muted"], showgrid=True, gridcolor=palette["surface_1"]),
        title=f"Actual vs Forecast - {metric}",
    )
    return fig


def build_outlook_compact_figure(
    joined: pd.DataFrame,
    metric: str,
    palette: dict[str, str],
    focus_date: str | pd.Timestamp | None = None,
) -> go.Figure | None:
    fig = build_outlook_timeseries_figure(joined, metric, palette, focus_date=focus_date)
    if fig is None:
        return None
    fig.update_layout(height=260, margin=dict(l=24, r=16, t=12, b=24), title="")
    fig.update_xaxes(showgrid=False)
    return fig


def build_outlook_spread_figure(
    joined: pd.DataFrame,
    metric: str,
    palette: dict[str, str],
    focus_date: str | pd.Timestamp | None = None,
) -> go.Figure | None:
    if joined is None or joined.empty:
        return None
    pred_col = f"pred_{metric}"
    if metric not in joined.columns or pred_col not in joined.columns:
        return None
    spread = (joined[pred_col] - joined[metric]).astype(float)
    fig = go.Figure()
    colors = [palette["accent_warm"] if val >= 0 else palette["accent_cool"] for val in spread]
    fig.add_trace(
        go.Bar(
            x=joined.index,
            y=spread,
            marker_color=colors,
            opacity=0.82,
            name="Forecast - Actual",
        )
    )
    fig.add_hline(y=0.0, line_width=1.1, line_dash="dot", line_color=palette["text_muted"], opacity=0.7)
    if focus_date is not None:
        focus_ts = pd.Timestamp(focus_date)
        if focus_ts in joined.index:
            fig.add_vline(x=focus_ts, line_width=1.2, line_dash="dot", line_color=palette["text_muted"], opacity=0.7)
    _add_regime_bands(fig, joined)
    fig.update_layout(
        height=260,
        margin=dict(l=24, r=16, t=12, b=24),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=palette["text_muted"], size=11)),
        xaxis=dict(color=palette["text_muted"], showgrid=False),
        yaxis=dict(color=palette["text_muted"], showgrid=True, gridcolor=palette["surface_1"], zeroline=False),
        title="",
    )
    return fig


def build_outlook_checkpoint_rows(joined: pd.DataFrame, metric: str) -> list[dict[str, object]]:
    if joined is None or joined.empty:
        return []
    pred_col = f"pred_{metric}"
    if metric not in joined.columns or pred_col not in joined.columns:
        return []
    n = len(joined.index)
    checkpoint_specs = [
        ("T+1", 0),
        ("T+5", 4),
        ("T+10", 9),
        ("T+20", 19),
        ("End", n - 1),
    ]
    rows: list[dict[str, object]] = []
    seen_idx: set[int] = set()
    for label, idx in checkpoint_specs:
        idx = min(max(idx, 0), n - 1)
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        row = joined.iloc[idx]
        actual_val = float(row[metric])
        forecast_val = float(row[pred_col])
        spread = forecast_val - actual_val
        rows.append(
            {
                "Checkpoint": label,
                "Date": str(joined.index[idx].date()),
                "Actual": round(actual_val, 4),
                "Forecast": round(forecast_val, 4),
                "Spread": round(spread, 4),
                "Signal": "Over" if spread > 1e-6 else "Under" if spread < -1e-6 else "Flat",
            }
        )
    return rows


def build_outlook_animation_figure(
    joined: pd.DataFrame,
    metric: str,
    palette: dict[str, str],
    focus_date: str | pd.Timestamp | None = None,
) -> go.Figure | None:
    if joined is None or joined.empty:
        return None
    pred_col = f"pred_{metric}"
    if metric not in joined.columns or pred_col not in joined.columns:
        return None

    frame_df = joined[[metric, pred_col]].dropna(how="all").copy()
    if frame_df.empty:
        return None

    dates = list(frame_df.index)
    base_actual = frame_df[metric].tolist()
    base_forecast = frame_df[pred_col].tolist()
    initial_idx = 0
    if focus_date is not None:
        focus_ts = pd.Timestamp(focus_date)
        for idx, dt in enumerate(dates):
            if pd.Timestamp(dt) == focus_ts:
                initial_idx = idx
                break
    frames: list[go.Frame] = []
    for idx, current_date in enumerate(dates):
        frames.append(
            go.Frame(
                name=str(current_date.date()),
                data=[
                    go.Scatter(
                        x=dates,
                        y=base_actual,
                        mode="lines",
                        name="Actual Baseline",
                        line=dict(color=palette["accent_cool"], width=1.2),
                        opacity=0.18,
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=dates,
                        y=base_forecast,
                        mode="lines",
                        name="Forecast Baseline",
                        line=dict(color=palette["accent_warm"], width=1.2, dash="dot"),
                        opacity=0.14,
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=dates[: idx + 1],
                        y=base_actual[: idx + 1],
                        mode="lines",
                        name="Actual",
                        line=dict(color=palette["accent_cool"], width=3),
                    ),
                    go.Scatter(
                        x=dates[: idx + 1],
                        y=base_forecast[: idx + 1],
                        mode="lines",
                        name="Forecast",
                        line=dict(color=palette["accent_warm"], width=3, dash="dash"),
                    ),
                    go.Scatter(
                        x=[dates[idx]],
                        y=[base_actual[idx]],
                        mode="markers",
                        name="Actual Point",
                        marker=dict(color=palette["accent_cool"], size=9, line=dict(color=palette["bg_main"], width=1)),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=[dates[idx]],
                        y=[base_forecast[idx]],
                        mode="markers",
                        name="Forecast Point",
                        marker=dict(color=palette["accent_warm"], size=9, symbol="diamond", line=dict(color=palette["bg_main"], width=1)),
                        showlegend=False,
                    ),
                ],
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=dates,
                y=base_actual,
                mode="lines",
                name="Actual Baseline",
                line=dict(color=palette["accent_cool"], width=1.2),
                opacity=0.18,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=dates,
                y=base_forecast,
                mode="lines",
                name="Forecast Baseline",
                line=dict(color=palette["accent_warm"], width=1.2, dash="dot"),
                opacity=0.14,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[dates[initial_idx]],
                y=[base_actual[initial_idx]],
                mode="lines",
                name="Actual",
                line=dict(color=palette["accent_cool"], width=3),
            ),
            go.Scatter(
                x=[dates[initial_idx]],
                y=[base_forecast[initial_idx]],
                mode="lines",
                name="Forecast",
                line=dict(color=palette["accent_warm"], width=3, dash="dash"),
            ),
            go.Scatter(
                x=[dates[initial_idx]],
                y=[base_actual[initial_idx]],
                mode="markers",
                name="Actual Point",
                marker=dict(color=palette["accent_cool"], size=9, line=dict(color=palette["bg_main"], width=1)),
                showlegend=False,
            ),
            go.Scatter(
                x=[dates[initial_idx]],
                y=[base_forecast[initial_idx]],
                mode="markers",
                name="Forecast Point",
                marker=dict(color=palette["accent_warm"], size=9, symbol="diamond", line=dict(color=palette["bg_main"], width=1)),
                showlegend=False,
            ),
        ],
        frames=frames,
    )
    _add_regime_bands(fig, joined)
    fig.update_layout(
        height=360,
        margin=dict(l=30, r=20, t=30, b=30),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        font=dict(color=palette["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            font=dict(color=palette["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            color=palette["text_muted"],
            showgrid=False,
            range=[dates[0], dates[-1]],
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            color=palette["text_muted"],
            showgrid=True,
            gridcolor=palette["surface_1"],
            zeroline=False,
        ),
        title="",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=1.0,
                y=1.18,
                xanchor="right",
                yanchor="top",
                bgcolor=palette["surface_1"],
                bordercolor=palette["border"],
                font=dict(color=palette["text_primary"], size=11),
                pad=dict(r=6, t=0),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            {
                "active": initial_idx,
                "y": -0.08,
                "x": 0.08,
                "len": 0.92,
                "currentvalue": {
                    "prefix": "Scrub Date: ",
                    "font": {"color": palette["text_muted"], "size": 12},
                    "visible": True,
                },
                "steps": [
                    {
                        "label": str(pd.Timestamp(dt).date()),
                        "method": "animate",
                        "args": [
                            [str(pd.Timestamp(dt).date())],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
                        ],
                    }
                    for dt in dates
                ],
            }
        ],
    )
    return fig
