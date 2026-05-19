"""
RiskSentinel — Contagion Engine
Shock propagation algorithms on financial correlation networks.

Three propagation models (increasing complexity):
1. Linear Threshold — simple: shock spreads if cumulative neighbor stress > threshold
2. DebtRank — Battiston et al. (2012): iterative distress propagation with saturation
3. Cascade Removal — node failure triggers neighbor stress, failures cascade

All models return a ShockResult with per-node impact and global stats.
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class ShockResult:
    """Result of a contagion simulation."""
    # Input
    shocked_node: str
    shock_magnitude: float
    model: str

    # Per-node state after propagation
    node_stress: dict[str, float] = field(default_factory=dict)
    # stress ∈ [0, 1]: 0=healthy, 1=defaulted

    # Propagation trace: list of (wave_number, nodes_affected_in_wave)
    cascade_waves: list[tuple[int, list[str]]] = field(default_factory=list)

    # Summary stats
    @property
    def n_affected(self) -> int:
        """Number of nodes with stress > 0 (excluding initial shock)."""
        return sum(1 for n, s in self.node_stress.items()
                   if s > 0 and n != self.shocked_node)

    @property
    def n_defaulted(self) -> int:
        """Number of nodes with stress == 1."""
        return sum(1 for s in self.node_stress.values() if s >= 1.0)

    @property
    def cascade_depth(self) -> int:
        """Number of propagation waves."""
        return len(self.cascade_waves)

    @property
    def total_stress(self) -> float:
        """Sum of all node stress levels (systemic damage measure)."""
        return sum(self.node_stress.values())

    @property
    def avg_stress(self) -> float:
        """Average stress across all nodes."""
        n = len(self.node_stress)
        return self.total_stress / n if n > 0 else 0.0

    @property
    def affected_nodes(self) -> list[tuple[str, float]]:
        """Nodes with stress > 0, sorted by stress descending."""
        return sorted(
            [(n, s) for n, s in self.node_stress.items() if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )

    def summary(self) -> dict:
        """Summary dict for agent consumption."""
        return {
            "shocked_node": self.shocked_node,
            "shock_magnitude": self.shock_magnitude,
            "model": self.model,
            "n_affected": self.n_affected,
            "n_defaulted": self.n_defaulted,
            "cascade_depth": self.cascade_depth,
            "total_stress": round(self.total_stress, 4),
            "avg_stress": round(self.avg_stress, 6),
            "top_10_affected": [
                {"ticker": n, "stress": round(s, 4)}
                for n, s in self.affected_nodes[:10]
            ],
        }


# ---------------------------------------------------------------------------
# MODEL 1: LINEAR THRESHOLD
# ---------------------------------------------------------------------------
def linear_threshold(
    G: nx.Graph,
    shocked_node: str,
    shock_magnitude: float = 1.0,
    activation_threshold: float = 0.5,
    max_waves: int = 20,
) -> ShockResult:
    """Simple threshold-based contagion.

    A node becomes "activated" (stress=1) if the sum of edge weights
    from its already-activated neighbors exceeds activation_threshold.
    Uses abs_weight for propagation strength.

    Args:
        G: NetworkX graph with abs_weight edge attribute.
        shocked_node: Ticker to shock initially.
        shock_magnitude: Initial shock level (0-1). 1.0 = full default.
        activation_threshold: Cumulative weight threshold to activate a node.
        max_waves: Maximum propagation waves.
    """
    if shocked_node not in G:
        raise ValueError(f"Node '{shocked_node}' not in graph")

    stress = {n: 0.0 for n in G.nodes()}
    stress[shocked_node] = shock_magnitude
    activated = {shocked_node}
    waves = []

    for wave in range(max_waves):
        new_activated = set()
        for node in G.nodes():
            if node in activated:
                continue
            # Sum weights from activated neighbors
            pressure = sum(
                G[node][nbr].get("abs_weight", 0)
                for nbr in G.neighbors(node)
                if nbr in activated
            )
            if pressure >= activation_threshold:
                new_activated.add(node)
                stress[node] = 1.0

        if not new_activated:
            break

        waves.append((wave + 1, sorted(new_activated)))
        activated.update(new_activated)

    return ShockResult(
        shocked_node=shocked_node,
        shock_magnitude=shock_magnitude,
        model="linear_threshold",
        node_stress=stress,
        cascade_waves=waves,
    )


# ---------------------------------------------------------------------------
# MODEL 2: DEBTRANK (Battiston et al., 2012)
# ---------------------------------------------------------------------------
def debtrank(
    G: nx.Graph,
    shocked_node: str,
    shock_magnitude: float = 0.5,
    max_waves: int = 20,
) -> ShockResult:
    """DebtRank contagion model.

    Iterative distress propagation where each node transmits a fraction
    of its stress increase to neighbors, weighted by edge strength.
    Nodes can only transmit stress ONCE (inactive after first transmission)
    to prevent infinite loops.

    Stress update for node j at wave t:
        Δh_j = min(1 - h_j, Σ_i w_ij * Δh_i)
    where w_ij = abs_weight(i,j) / max_weight (normalized).

    Args:
        G: NetworkX graph with abs_weight edge attribute.
        shocked_node: Ticker to shock initially.
        shock_magnitude: Initial stress level (0-1).
        max_waves: Maximum propagation rounds.
    """
    if shocked_node not in G:
        raise ValueError(f"Node '{shocked_node}' not in graph")

    # Normalize weights to [0, 1]
    max_w = max(
        (d.get("abs_weight", 0) for _, _, d in G.edges(data=True)),
        default=1.0,
    )
    if max_w == 0:
        max_w = 1.0

    # State: h = stress level, status: undistressed / distressed / inactive
    h = {n: 0.0 for n in G.nodes()}
    h[shocked_node] = shock_magnitude

    # Track which nodes have already propagated (become inactive)
    has_propagated = set()

    # Nodes that got stressed in current wave (and will propagate next wave)
    newly_stressed = {shocked_node}
    waves = []

    for wave in range(max_waves):
        # These nodes now propagate their stress to neighbors
        propagators = newly_stressed - has_propagated
        if not propagators:
            break

        delta_h = {n: 0.0 for n in G.nodes()}

        for node in propagators:
            for nbr in G.neighbors(node):
                if nbr in has_propagated or nbr == shocked_node:
                    continue
                w = G[node][nbr].get("abs_weight", 0) / max_w
                delta_h[nbr] += w * h[node]

        has_propagated.update(propagators)

        # Apply stress updates (capped at 1.0)
        wave_affected = []
        for node in G.nodes():
            if delta_h[node] > 0 and node not in has_propagated:
                old_h = h[node]
                h[node] = min(1.0, h[node] + delta_h[node])
                if h[node] > old_h:
                    wave_affected.append(node)

        newly_stressed = set(wave_affected)

        if wave_affected:
            waves.append((wave + 1, sorted(wave_affected)))

    return ShockResult(
        shocked_node=shocked_node,
        shock_magnitude=shock_magnitude,
        model="debtrank",
        node_stress=h,
        cascade_waves=waves,
    )


# ---------------------------------------------------------------------------
# MODEL 3: CASCADE REMOVAL
# ---------------------------------------------------------------------------
def cascade_removal(
    G: nx.Graph,
    shocked_node: str,
    shock_magnitude: float = 1.0,
    failure_threshold: float = 0.4,
    max_waves: int = 20,
) -> ShockResult:
    """Cascade failure model.

    When a node fails (stress >= failure_threshold), it is removed from
    the network. Its neighbors receive stress proportional to how much
    connectivity they lose. If their stress exceeds the threshold, they
    also fail — creating a cascade.

    Stress for node j when neighbor i fails:
        Δh_j = w_ij / degree_j  (fraction of connectivity lost)

    Args:
        G: NetworkX graph with abs_weight edge attribute.
        shocked_node: Ticker to remove initially.
        shock_magnitude: Initial shock (typically 1.0 = full failure).
        failure_threshold: Stress level at which a node fails.
        max_waves: Maximum cascade rounds.
    """
    if shocked_node not in G:
        raise ValueError(f"Node '{shocked_node}' not in graph")

    G_live = G.copy()
    stress = {n: 0.0 for n in G.nodes()}
    stress[shocked_node] = shock_magnitude
    failed = {shocked_node}
    waves = []

    nodes_to_remove = {shocked_node}

    for wave in range(max_waves):
        # Remove failed nodes and stress their neighbors
        new_failures = set()

        for node in nodes_to_remove:
            if node not in G_live:
                continue
            node_stress_level = stress[node]
            neighbors = list(G_live.neighbors(node))
            for nbr in neighbors:
                if nbr in failed:
                    continue
                # Stress transmitted = edge weight * failed node's stress
                w = G_live[node][nbr].get("abs_weight", 0.5)
                stress[nbr] = min(1.0, stress[nbr] + w * node_stress_level)
                if stress[nbr] >= failure_threshold:
                    new_failures.add(nbr)
            G_live.remove_node(node)

        if not new_failures:
            break

        waves.append((wave + 1, sorted(new_failures)))
        failed.update(new_failures)
        nodes_to_remove = new_failures

    return ShockResult(
        shocked_node=shocked_node,
        shock_magnitude=shock_magnitude,
        model="cascade_removal",
        node_stress=stress,
        cascade_waves=waves,
    )


# ---------------------------------------------------------------------------
# CONVENIENCE: RUN ALL MODELS
# ---------------------------------------------------------------------------
def run_shock_scenario(
    G: nx.Graph,
    shocked_node: str,
    shock_magnitude: float = 0.5,
    model: str = "debtrank",
) -> ShockResult:
    """Run a shock scenario with the specified model.

    Args:
        G: NetworkX graph.
        shocked_node: Ticker to shock.
        shock_magnitude: Shock level (0-1).
        model: One of "linear_threshold", "debtrank", "cascade_removal".
    """
    models = {
        "linear_threshold": lambda: linear_threshold(
            G, shocked_node, shock_magnitude,
        ),
        "debtrank": lambda: debtrank(
            G, shocked_node, shock_magnitude,
        ),
        "cascade_removal": lambda: cascade_removal(
            G, shocked_node, shock_magnitude,
        ),
    }
    if model not in models:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(models.keys())}")
    return models[model]()


def compare_models(
    G: nx.Graph,
    shocked_node: str,
    shock_magnitude: float = 0.5,
) -> dict[str, ShockResult]:
    """Run all three models and return results for comparison."""
    return {
        model: run_shock_scenario(G, shocked_node, shock_magnitude, model)
        for model in ["linear_threshold", "debtrank", "cascade_removal"]
    }
