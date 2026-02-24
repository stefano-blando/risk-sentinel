Write unit tests for the file `src/core/network.py`.

The module builds NetworkX graphs from correlation matrices and computes metrics.
Test file should be created at `tests/test_network.py`.

Requirements:
- Use pytest
- Test each public function: build_network, build_network_for_date, compute_global_metrics, compute_node_centralities, get_top_nodes, get_node_neighbors, get_sector_subgraph, compare_networks
- For build_network: create a small synthetic 5x5 correlation DataFrame, build graph with threshold=0.3, verify correct number of edges are created, verify edge weights are set
- For build_network_for_date: test with date '2025-12-01', verify returns (nx.Graph, pd.Timestamp), graph has 210 nodes
- For compute_global_metrics: verify all expected keys exist (n_nodes, n_edges, density, avg_degree, n_components, largest_cc_pct, avg_clustering, avg_weight, max_weight)
- For compute_node_centralities: verify returns dict with all nodes, each having keys (degree, betweenness, closeness, eigenvector, pagerank)
- For get_top_nodes: verify returns list of tuples (str, float), length <= top_n, sorted descending
- For get_node_neighbors: test with 'JPM', verify returns list of (str, float) sorted by |corr|, first neighbor should be 'GS' (highest correlation)
- For get_sector_subgraph: test with 'Financials', verify all nodes have sector='Financials'
- For compare_networks: create two graphs, verify result has _before, _after, _delta keys
- Use a pytest fixture to load the graph once: build_network_for_date('2025-12-01')
- Add `import sys; sys.path.insert(0, '.')` at top
- Mark slow tests (those loading real data) with @pytest.mark.slow
