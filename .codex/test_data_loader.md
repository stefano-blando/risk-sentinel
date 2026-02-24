Write unit tests for the file `src/core/data_loader.py`.

The module loads pre-computed financial data from parquet and pickle files.
Test file should be created at `tests/test_data_loader.py`.

Requirements:
- Use pytest
- Test each public function: load_sector_mapping, get_sector_dict, get_ticker_list, load_close_prices, load_returns, load_market_data, load_regime_data, load_network_metrics, load_network_features, load_sector_centralities, load_node_centralities, get_available_dates, find_nearest_date, get_correlation_matrix, get_node_centralities_for_date, centralities_to_dataframe, load_mvp_data
- For each loader test: return type is correct (pd.DataFrame or dict), shape is non-empty, expected columns exist
- For sector_mapping: verify 210 rows, columns ['Ticker', 'Sector'], 11 unique sectors
- For get_ticker_list: verify returns list of 210 strings
- For regime_data: verify Regime column has values in {'Calm', 'Normal', 'Elevated', 'High', 'Crisis'}
- For find_nearest_date: test with exact date and approximate date
- For centralities_to_dataframe: verify columns are ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
- For load_mvp_data: verify dict has keys 'sector_mapping', 'sector_dict', 'tickers', 'market', 'regimes', 'network_metrics', 'network_features', 'node_centralities'
- Add `import sys; sys.path.insert(0, '.')` at top for path resolution
- Keep tests fast: no need to load the 1 GB correlation matrices pickle, skip that test with @pytest.mark.slow
