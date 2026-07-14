"""P8 Task 35 — 통계 집계 리포터 테스트 (08-p8-scale-ablation-design.md §1-1/§1-2/§Task35).

M1~M6 지표 산출 + 클래스별 재현율(≥70% 판정선) + gain 분포/null 비율/retire율.
다문제 result fixture(gain/null/retire 혼재)로 정확도 검증. torch 불요.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from report_p8_stats import (  # noqa: E402
    compute_metrics_for_problem,
    aggregate_stats,
    ProblemStats,
)


def _track(label, curve, retire_count=0, stop_reason="converged", signals=None,
          gate_fail=False):
    n = len(curve)
    sigs = signals if signals is not None else [{} if gate_fail else {"load_eff": 0.9}] * n
    return {
        "label": label, "metric_curve": curve, "fired_rules": ["fp32_no_tensorcore"] * n,
        "wasted_rounds": 0, "retire_count": retire_count, "stop_reason": stop_reason,
        "rounds": n, "signals": sigs,
    }


def _gain_result(problem="matmul", on_curve=None, off_curve=None, retire_count=0):
    # metric_curve = [(round_idx, energy_per_iter_j 대용 metric)] — 실제론 -energy 부호
    # 관례를 따르되, 테스트는 값 자체를 energy 대용으로 두고 최솟값=best로 취급.
    on = _track("evolve_ON", on_curve or [(0, 10.0), (1, 2.0)], retire_count=retire_count)
    off = _track("evolve_OFF", off_curve or [(0, 10.0), (1, 10.0)])
    return {problem: {"on": on, "off": off}}


def _null_result(problem="kb_softmax"):
    on = _track("evolve_ON", [(0, 10.0)], stop_reason="below_weight_gate")
    off = _track("evolve_OFF", [(0, 10.0)], stop_reason="below_weight_gate")
    return {problem: {"on": on, "off": off}}


def _gate_fail_result(problem="bad_problem"):
    on = _track("evolve_ON", [(0, -1.0)], stop_reason="gate_fail", gate_fail=True)
    off = _track("evolve_OFF", [(0, -1.0)], stop_reason="gate_fail", gate_fail=True)
    return {problem: {"on": on, "off": off}}


def test_compute_metrics_m1_track_internal_gain_ratio():
    res = _gain_result(on_curve=[(0, 10.0), (1, 2.0)])
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    assert isinstance(stats, ProblemStats)
    # M1 = seed(round0) / best(min) = 10.0 / 2.0 = 5.0
    assert abs(stats.m1_track_gain - 5.0) < 1e-9


def test_compute_metrics_m2_cross_track_gap():
    res = _gain_result(on_curve=[(0, 10.0), (1, 2.0)], off_curve=[(0, 10.0), (1, 8.0)])
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    # M2 = ON best / OFF best = 2.0 / 8.0 (gain 방향에 따라 역수 표기 가능 — 값 유한/양수만 확인)
    assert stats.m2_cross_track_gap > 0


def test_compute_metrics_m3_null_detection_true_for_flat_curve():
    res = _null_result()
    stats = compute_metrics_for_problem("kb_softmax", res["kb_softmax"],
                                        bucket="memory-reduce-elementwise")
    assert stats.m3_null is True


def test_compute_metrics_m3_null_false_for_improving_curve():
    res = _gain_result()
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    assert stats.m3_null is False


def test_compute_metrics_m4_retire_count_from_on_track():
    res = _gain_result(retire_count=1)
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    assert stats.m4_retire_count == 1


def test_compute_metrics_m5_gate_fail_true_when_stop_reason_gate_fail():
    res = _gate_fail_result()
    stats = compute_metrics_for_problem("bad_problem", res["bad_problem"], bucket="compute-matmul")
    assert stats.m5_gate_fail is True


def test_compute_metrics_m5_gate_fail_false_for_normal_result():
    res = _gain_result()
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    assert stats.m5_gate_fail is False


def test_compute_metrics_m6_wasted_rounds_present():
    res = _gain_result()
    stats = compute_metrics_for_problem("matmul", res["matmul"], bucket="compute-matmul")
    assert stats.m6_wasted_rounds == 0


def test_aggregate_stats_reproduction_rate_compute_class():
    # 3 compute-matmul 문제: 2개 M1>1.3, 1개 미달 → 재현율 2/3 ≈ 66.7% (<70% 판정)
    problems = {
        "a": _gain_result(problem="a", on_curve=[(0, 10.0), (1, 2.0)])["a"],
        "b": _gain_result(problem="b", on_curve=[(0, 10.0), (1, 5.0)])["b"],  # gain 2.0>1.3
        "c": _gain_result(problem="c", on_curve=[(0, 10.0), (1, 9.0)])["c"],  # gain 1.11<1.3
    }
    buckets = {"a": "compute-matmul", "b": "compute-matmul", "c": "compute-matmul"}
    agg = aggregate_stats(problems, buckets)
    assert agg.compute_reproduction_rate == 2 / 3
    assert agg.compute_class_pass is False  # <70%


def test_aggregate_stats_memory_class_null_reproduction():
    problems = {
        "x": _null_result(problem="x")["x"],
        "y": _null_result(problem="y")["y"],
    }
    buckets = {"x": "memory-norm", "y": "memory-reduce-elementwise"}
    agg = aggregate_stats(problems, buckets)
    assert agg.memory_null_rate == 1.0
    assert agg.memory_class_pass is True  # ≥70%


def test_aggregate_stats_excludes_gate_fail_from_gain_distribution():
    problems = {
        "good": _gain_result(problem="good")["good"],
        "bad": _gate_fail_result(problem="bad")["bad"],
    }
    buckets = {"good": "compute-matmul", "bad": "compute-matmul"}
    agg = aggregate_stats(problems, buckets)
    assert "bad" not in [p for p in agg.gain_distribution]
    assert agg.gate_fail_count == 1


def test_aggregate_stats_retire_rate_present():
    problems = {"a": _gain_result(problem="a", retire_count=1)["a"]}
    buckets = {"a": "compute-matmul"}
    agg = aggregate_stats(problems, buckets)
    assert agg.retire_rate == 1.0
