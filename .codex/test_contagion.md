Write unit tests for the file `src/core/contagion.py`.

The module implements three shock propagation models on NetworkX graphs.
Test file should be created at `tests/test_contagion.py`.

Requirements:
- Use pytest
- Test the ShockResult dataclass properties: n_affected, n_defaulted, cascade_depth, total_stress, avg_stress, affected_nodes, summary
- Test each model function: linear_threshold, debtrank, cascade_removal
- Test convenience functions: run_shock_scenario, compare_models
- Create a small synthetic graph fixture (10 nodes, known edge weights) for fast unit tests
- For each model:
  - Verify returns ShockResult
  - Verify shocked_node has stress > 0
  - Verify n_affected >= 0
  - Verify cascade_depth >= 0
  - Verify all stress values are in [0, 1]
  - Verify summary() returns dict with expected keys
- Test edge cases:
  - Shock a node with no neighbors (isolated node) — should affect only itself
  - Invalid node name — should raise ValueError
  - shock_magnitude=0 — should produce minimal propagation
  - shock_magnitude=1 — should produce maximum propagation
- Add one integration test with real data: load graph from build_network_for_date('2025-12-01'), run debtrank on 'JPM' with 0.5, verify n_affected > 50 (known from testing)
- Mark integration tests with @pytest.mark.slow
- Add `import sys; sys.path.insert(0, '.')` at top

The synthetic graph fixture should be:
```python
@pytest.fixture
def small_graph():
    G = nx.Graph()
    for i in range(10):
        G.add_node(f"N{i}", ticker=f"N{i}", sector="TestSector")
    # Chain: N0-N1-N2-...-N9 with decreasing weights
    for i in range(9):
        w = 0.9 - i * 0.05
        G.add_edge(f"N{i}", f"N{i+1}", weight=w, abs_weight=w)
    # Add a hub: N0 connected to N5, N7
    G.add_edge("N0", "N5", weight=0.6, abs_weight=0.6)
    G.add_edge("N0", "N7", weight=0.5, abs_weight=0.5)
    return G
```
