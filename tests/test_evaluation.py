from src.agents.evaluation import EvalSample, evaluate_samples


def test_evaluate_samples_empty():
    out = evaluate_samples([])
    assert out["n"] == 0
    assert out["critic_pass_rate_pct"] == 0.0


def test_evaluate_samples_metrics():
    samples = [
        EvalSample(True, True, 1.2, False),
        EvalSample(True, False, 2.5, False),
        EvalSample(False, False, 3.0, True),
        EvalSample(True, True, 4.0, False),
    ]
    out = evaluate_samples(samples)
    assert out["n"] == 4
    assert out["critic_pass_rate_pct"] == 75.0
    assert out["factual_consistency_pct"] == 50.0
    assert out["fallback_rate_pct"] == 25.0
    assert out["latency_p95_sec"] >= 3.0
