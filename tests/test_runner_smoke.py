import pytest

from tnsbench.runner.run import run_benchmark


def test_mock_safe_runs_on_five_tasks(tmp_path):
    out = tmp_path / "r.jsonl"
    results = run_benchmark(
        "mock_safe", "scripted", "adversarial", trials=1, limit=5, out_path=out
    )
    assert len(results) == 5
    assert all(isinstance(r.safety_pass, bool) for r in results)


def test_mock_unsafe_triggers_violation(tmp_path):
    out = tmp_path / "r.jsonl"
    results = run_benchmark(
        "mock_unsafe", "scripted", "adversarial", trials=1, limit=12, out_path=out
    )
    assert any(r.unsafe_action or r.unsafe_content for r in results)


def test_benign_spec_rejected(tmp_path):
    """TnSBench-Hard is adversarial-only; benign task specs must raise."""
    out = tmp_path / "r.jsonl"
    for spec in ("benign_control", "overrefusal_calibration", "calibration", "benign"):
        with pytest.raises(ValueError):
            run_benchmark(
                "mock_safe", "scripted", spec, trials=1, limit=5, out_path=out
            )
