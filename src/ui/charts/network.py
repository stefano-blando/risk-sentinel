"""Network graph figures shared by Stress Lab and Outlook."""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go


def compute_layout(G: nx.Graph) -> dict:
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        return nx.spring_layout(G, k=0.4, iterations=80, seed=42)


def _stress_color(stress: float, risk_colors: dict[str, str]) -> str:
    if stress >= 0.8:
        return risk_colors["critical"]
    if stress >= 0.5:
        return risk_colors["high"]
    if stress >= 0.2:
        return risk_colors["moderate"]
    if stress > 0.01:
        return risk_colors["low"]
    return "#334155"


def _build_base_layout(palette: dict[str, str], height: int = 580, title_text: str = "") -> go.Layout:
    return go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=10, r=10, t=50),
        plot_bgcolor=palette["bg_main"],
        paper_bgcolor=palette["bg_main"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        title=dict(
            text=title_text,
            font=dict(color=palette["accent_warm"], size=16),
            x=0.5,
            xanchor="center",
        ) if title_text else None,
    )


def _bg_edge_trace(G: nx.Graph, pos: dict, edge_bg_color: str) -> go.Scatter:
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex,
        y=ey,
        mode="lines",
        line=dict(width=0.3, color=edge_bg_color),
        hoverinfo="none",
        showlegend=False,
    )


def _stressed_edge_trace(G: nx.Graph, pos: dict, stress: dict, edge_stress_color: str) -> go.Scatter:
    ex, ey = [], []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        max_stress = max(stress.get(u, 0), stress.get(v, 0))
        if max_stress > 0.01:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex,
        y=ey,
        mode="lines",
        line=dict(width=1.8, color=edge_stress_color),
        hoverinfo="none",
        showlegend=False,
    )


def _node_trace(
    G: nx.Graph,
    pos: dict,
    *,
    stress: dict | None,
    shocked_node: str | None,
    sector_dict: dict[str, str] | None,
    sector_colors: dict[str, str],
    risk_colors: dict[str, str],
    palette: dict[str, str],
) -> go.Scatter:
    sectors = sector_dict or {}
    valid = [node for node in G.nodes() if node in pos]
    nx_vals = [pos[node][0] for node in valid]
    ny_vals = [pos[node][1] for node in valid]

    colors, sizes, texts, labels, outlines = [], [], [], [], []
    for node in valid:
        sector = sectors.get(node, "Unknown")
        node_stress = stress.get(node, 0) if stress else 0
        is_shocked = node == shocked_node

        if stress:
            colors.append("#f8fafc" if is_shocked else _stress_color(node_stress, risk_colors))
            sizes.append(28 if is_shocked else max(5, int(5 + node_stress * 20)))
        else:
            colors.append(sector_colors.get(sector, risk_colors["none"]))
            sizes.append(7)

        stress_text = f"<br>Stress: {node_stress:.1%}" if stress else ""
        texts.append(f"<b>{node}</b><br>Sector: {sector}<br>Connections: {G.degree(node)}{stress_text}")
        labels.append(node if (is_shocked or (stress and node_stress >= 0.5)) else "")
        outlines.append(risk_colors["critical"] if is_shocked else palette["bg_main"])

    return go.Scatter(
        x=nx_vals,
        y=ny_vals,
        mode="markers+text",
        hoverinfo="text",
        hovertext=texts,
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color=outlines)),
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color=palette["text_primary"]),
        showlegend=False,
    )


def build_graph_figure(
    G: nx.Graph,
    pos: dict,
    *,
    sector_dict: dict[str, str] | None,
    sector_colors: dict[str, str],
    risk_colors: dict[str, str],
    palette: dict[str, str],
    edge_bg_color: str,
) -> go.Figure:
    return go.Figure(
        data=[
            _bg_edge_trace(G, pos, edge_bg_color),
            _node_trace(
                G,
                pos,
                stress=None,
                shocked_node=None,
                sector_dict=sector_dict,
                sector_colors=sector_colors,
                risk_colors=risk_colors,
                palette=palette,
            ),
        ],
        layout=_build_base_layout(palette),
    )


def build_animated_figure(
    G: nx.Graph,
    pos: dict,
    result,
    *,
    sector_dict: dict[str, str] | None,
    sector_colors: dict[str, str],
    risk_colors: dict[str, str],
    palette: dict[str, str],
    edge_bg_color: str,
    edge_stress_color: str,
    blast_radius_only: bool = False,
) -> go.Figure:
    shocked_node = result.shocked_node
    n_waves = result.cascade_depth

    if blast_radius_only:
        visible = {node for node, node_stress in result.node_stress.items() if node_stress > 0.01}
        visible.add(shocked_node)
        for node in list(visible):
            for neighbor in G.neighbors(node):
                visible.add(neighbor)
        G = G.subgraph(visible).copy()

    bg = _bg_edge_trace(G, pos, edge_bg_color)

    def wave_stress(wave_idx: int) -> dict:
        stress_map = {node: 0.0 for node in G.nodes()}
        stress_map[shocked_node] = result.shock_magnitude
        for wave_no, nodes in result.cascade_waves:
            if wave_no <= wave_idx:
                for node in nodes:
                    stress_map[node] = result.node_stress[node]
        # Cascade-removal can assign stress to neighbors that never cross the
        # failure threshold; they won't appear in cascade_waves. Show the full
        # final stress map on the last frame to avoid a misleading "single-node"
        # visualization when impact is contained but non-zero.
        if getattr(result, "model", "") == "cascade_removal" and wave_idx == n_waves:
            for node, node_stress in result.node_stress.items():
                if node in stress_map:
                    stress_map[node] = max(stress_map[node], float(node_stress))
        return stress_map

    frames = []
    for wave_idx in range(n_waves + 1):
        stress_map = wave_stress(wave_idx)
        label = f"Wave {wave_idx}/{n_waves}" if wave_idx > 0 else "Initial Shock"
        frames.append(
            go.Frame(
                data=[
                    bg,
                    _stressed_edge_trace(G, pos, stress_map, edge_stress_color),
                    _node_trace(
                        G,
                        pos,
                        stress=stress_map,
                        shocked_node=shocked_node,
                        sector_dict=sector_dict,
                        sector_colors=sector_colors,
                        risk_colors=risk_colors,
                        palette=palette,
                    ),
                ],
                name=str(wave_idx),
                layout=go.Layout(
                    title=dict(
                        text=label,
                        font=dict(color=palette["accent_warm"], size=16),
                        x=0.5,
                        xanchor="center",
                    )
                ),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=_build_base_layout(palette, height=580, title_text="Initial Shock"),
    )

    steps = []
    for wave_idx in range(n_waves + 1):
        steps.append(
            dict(
                args=[
                    [str(wave_idx)],
                    dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=300)),
                ],
                label=f"W{wave_idx}" if wave_idx > 0 else "💥",
                method="animate",
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.08,
                y=1.12,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1200, redraw=True),
                                transition=dict(duration=400, easing="cubic-in-out"),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=steps,
                x=0.1,
                len=0.8,
                y=-0.02,
                currentvalue=dict(
                    prefix="Cascade: ",
                    visible=True,
                    font=dict(color=palette["accent_warm"], size=13),
                ),
                tickcolor=palette["text_muted"],
                font=dict(color=palette["text_muted"]),
                bgcolor=palette["surface_1"],
                bordercolor=palette["surface_1"],
                activebgcolor=palette["accent_warm"],
            )
        ],
    )
    return fig
