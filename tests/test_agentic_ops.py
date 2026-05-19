import networkx as nx
import pandas as pd

from src import agentic_ops


class _FakeDataLoader:
    def get_correlation_matrix(self, date_str):
        idx = ["AAA", "BBB", "CCC", "DDD"]
        corr = pd.DataFrame(
            [
                [1.0, 0.7, 0.2, 0.1],
                [0.7, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.6],
                [0.1, 0.2, 0.6, 1.0],
            ],
            index=idx,
            columns=idx,
        )
        return corr, pd.Timestamp("2025-12-01")

    def load_regime_data(self):
        return pd.DataFrame(
            {"Regime": ["Calm"], "VIX": [14.2]},
            index=[pd.Timestamp("2025-12-01")],
        )

    def find_nearest_date(self, date_str, dates):
        return pd.Timestamp("2025-12-01")


class _FakeNetwork:
    def build_network(self, corr, threshold, sector_dict):
        graph = nx.Graph()
        for ticker in corr.index:
            graph.add_node(ticker, sector=sector_dict.get(ticker, "Unknown"))
        for i, t1 in enumerate(corr.index):
            for j, t2 in enumerate(corr.columns):
                if j <= i:
                    continue
                weight = float(corr.loc[t1, t2])
                if abs(weight) > threshold:
                    graph.add_edge(t1, t2, weight=weight, abs_weight=abs(weight))
        return graph

    def compute_global_metrics(self, graph):
        return {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "density": float(nx.density(graph)),
        }


class _FakeShockResult:
    def __init__(self, ticker):
        self.ticker = ticker

    def summary(self):
        base = {
            "AAA": {"avg_stress": 0.20, "n_affected": 3, "cascade_depth": 2, "n_defaulted": 0, "total_stress": 1.3},
            "BBB": {"avg_stress": 0.12, "n_affected": 2, "cascade_depth": 1, "n_defaulted": 0, "total_stress": 0.8},
            "CCC": {"avg_stress": 0.10, "n_affected": 2, "cascade_depth": 1, "n_defaulted": 0, "total_stress": 0.7},
        }
        return base.get(
            self.ticker,
            {"avg_stress": 0.05, "n_affected": 1, "cascade_depth": 1, "n_defaulted": 0, "total_stress": 0.3},
        )


class _FakeContagion:
    def run_shock_scenario(self, graph, ticker, shock, model):
        return _FakeShockResult(ticker)


def test_parse_portfolio_positions_normalizes_and_reports_errors():
    rows, errs = agentic_ops.parse_portfolio_positions(
        "AAA,60%\nBBB,0.4\nZZZ,0.2\n",
        {"AAA", "BBB", "CCC"},
    )
    assert len(rows) == 2
    assert any("unknown ticker `ZZZ`" in err for err in errs)
    assert abs(sum(abs(r["weight_norm"]) for r in rows) - 1.0) < 1e-9


def test_routing_policy_and_critic_round_limit():
    policy = agentic_ops.choose_execution_policy(
        parsed={"tickers": ["AAA", "BBB"]},
        complex_query=True,
        in_scope=True,
        agent_mode=True,
        gpt_for_parseable_queries=True,
        access_allowed=True,
        selected_strategy="orchestrator",
    )
    assert policy["route"] == "gpt"
    assert policy["effective_strategy"] == "commentary_direct"
    assert agentic_ops.critic_round_limit(True) == 2
    assert agentic_ops.critic_round_limit(False) == 1


def test_build_auto_portfolio_from_network():
    out = agentic_ops.build_auto_portfolio_from_network(
        date_str="2025-12-01",
        threshold=0.15,
        n_positions=3,
        sector_dict={
            "AAA": "Tech",
            "BBB": "Financials",
            "CCC": "Energy",
            "DDD": "Industrials",
        },
        data_loader_mod=_FakeDataLoader(),
        network_mod=_FakeNetwork(),
    )
    assert out["ok"] is True
    assert len(out["rows"]) == 3
    assert abs(sum(float(r["weight"]) for r in out["rows"]) - 1.0) <= 1e-6
    assert len(out["portfolio_text"].splitlines()) == 3


def test_run_portfolio_copilot_returns_kpi():
    out = agentic_ops.run_portfolio_copilot(
        portfolio_text="AAA,0.6\nBBB,0.4",
        date_str="2025-12-01",
        threshold=0.15,
        model="debtrank",
        stress_shock_pct=50,
        risk_profile="balanced",
        tickers=["AAA", "BBB", "CCC"],
        sector_dict={
            "AAA": "Information Technology",
            "BBB": "Financials",
            "CCC": "Energy",
            "DDD": "Industrials",
        },
        data_loader_mod=_FakeDataLoader(),
        network_mod=_FakeNetwork(),
        contagion_mod=_FakeContagion(),
    )
    assert out["ok"] is True
    assert out["kpi"]["formula"] == "avoided = expected_stress * coverage * efficiency"
    assert out["expected_stress_pct_after_hedge"] <= out["expected_stress_pct"]
    assert len(out["actions"]) >= 1
