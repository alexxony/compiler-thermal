"""P8 Task 35 — 통계 집계 리포터 (08-p8-scale-ablation-design.md §1-1/§1-2/Task 35).

배치별 결과 JSON(run_ablation_remote.py __ABL_RESULT__ 포맷, merge_batch_results로
병합 가능)을 입력받아 §1-1 M1~M6 지표를 문제별로 산출하고, §1-2 클래스별 재현율
(≥70% 판정선)을 집계한다. report_p4_deltat.py와 별개 — ΔT 축이 아니라 energy 축
통계 전용. 실측(run) 없음, 순수 후처리. torch 불요.

M1(트랙 내부 개선 배율): evolve_ON seed(round 0) energy ÷ best(최솟값) round energy.
  best는 라벨이 아니라 값 기준(06-p6-signal-provenance-design §7-2 라벨 오프셋 주의).
M2(ON/OFF 교차 격차): evolve_ON best ÷ evolve_OFF best.
M3(null 여부): seed=best(개선 라운드 없음, 곡선 단조 불변 또는 즉시 STOP)면 True.
M4(retire 발생): evolve_ON retire_count.
M5(gate_fail 여부): stop_reason=="gate_fail"이거나 신호가 전부 빈 dict(gate_fail 라운드).
M6(헛라운드 분포): wasted_rounds.
"""
from __future__ import annotations
from dataclasses import dataclass, field

WEIGHT_GATE_M1_THRESHOLD = 1.3   # §1-2 compute-bound 판정 배율(사전 등록)
CLASS_PASS_THRESHOLD = 0.70      # §1-2 판정선 — 클래스 내 ≥70% 재현

_COMPUTE_BUCKETS = ("compute-matmul", "compute-conv-fusion")
_MEMORY_BUCKETS = ("memory-norm", "memory-reduce-elementwise")


@dataclass
class ProblemStats:
    problem: str
    bucket: str
    m1_track_gain: float          # evolve_ON 트랙 내부 배율 (seed/best)
    m2_cross_track_gap: float     # ON best / OFF best
    m3_null: bool
    m4_retire_count: int
    m5_gate_fail: bool
    m6_wasted_rounds: int


def _seed_and_best(curve: list[tuple[int, float]]) -> tuple[float, float]:
    """curve = [(round_idx, metric)]. metric은 energy_per_iter_j 규약을 따른다고
    가정(값이 작을수록 좋음 — energy 축 관례). seed=round0, best=min(metric).
    """
    if not curve:
        raise ValueError("빈 metric_curve — 문제 결과 이상")
    seed_val = curve[0][1]
    best_val = min(v for _, v in curve)
    return seed_val, best_val


def compute_metrics_for_problem(problem: str, result: dict, bucket: str) -> ProblemStats:
    """result = {"on": track_dict, "off": track_dict} (run_ablation_remote._to_track
    이전 raw dict 형태 — 이 함수는 raw dict를 직접 소비, TrackResult 의존 없음).
    """
    on = result["on"]
    off = result["off"]

    on_seed, on_best = _seed_and_best(on["metric_curve"])
    off_seed, off_best = _seed_and_best(off["metric_curve"])

    # M5: gate_fail 판정 — stop_reason만이 권위 있는 신호(harness.py:80, correctness
    # 게이트 실패 시에만 RoundRecord.signal={}와 함께 "gate_fail"로 종료). 빈 signal
    # dict 자체는 gate_fail 전용 마커가 아님(below_weight_gate 등 정상 STOP도 신호
    # 추출을 안 할 수 있음) — signal 공백 휴리스틱은 오탐 유발이라 배제.
    gate_fail = (on.get("stop_reason") == "gate_fail"
                or off.get("stop_reason") == "gate_fail")

    m1 = (on_seed / on_best) if on_best not in (0, 0.0) else float("inf")
    m2 = (on_best / off_best) if off_best not in (0, 0.0) else float("inf")

    # M3 null: ON 트랙이 seed=best(개선 없음, 즉시 STOP류) — below_weight_gate 등
    # STOP 라벨 stop_reason이거나 곡선이 사실상 변화 없음(round<=1이고 seed==best).
    m3_null = (on_seed == on_best) or on.get("stop_reason") in (
        "below_weight_gate", "memory_saturated", "tensorcore_saturated")

    return ProblemStats(
        problem=problem, bucket=bucket,
        m1_track_gain=m1, m2_cross_track_gap=m2, m3_null=m3_null,
        m4_retire_count=int(on.get("retire_count", 0)),
        m5_gate_fail=gate_fail,
        m6_wasted_rounds=int(on.get("wasted_rounds", 0)),
    )


@dataclass
class AggregateStats:
    per_problem: dict[str, ProblemStats]
    compute_reproduction_rate: float
    compute_class_pass: bool
    memory_null_rate: float
    memory_class_pass: bool
    retire_rate: float
    gate_fail_count: int
    gain_distribution: dict[str, float] = field(default_factory=dict)  # problem -> M1 (gate_fail 제외)
    wasted_rounds_distribution: dict[str, int] = field(default_factory=dict)


def aggregate_stats(results: dict[str, dict], buckets: dict[str, str]) -> AggregateStats:
    """results: {problem: {"on":..., "off":...}}, buckets: {problem: bucket}.

    §1-2 판정: compute-bound(§1-2 표) 클래스는 M1 > 1.3인 비율 ≥70% → pass.
    memory-bound 클래스는 M3 null=True 비율 ≥70% → pass.
    gate_fail 문제는 gain_distribution/재현율 산출에서 제외(무효 표본 격리, §1-2 M5).
    """
    per_problem: dict[str, ProblemStats] = {}
    for name, res in results.items():
        bucket = buckets.get(name, "")
        per_problem[name] = compute_metrics_for_problem(name, res, bucket)

    valid = {n: s for n, s in per_problem.items() if not s.m5_gate_fail}
    gate_fail_count = sum(1 for s in per_problem.values() if s.m5_gate_fail)

    compute_probs = {n: s for n, s in valid.items() if s.bucket in _COMPUTE_BUCKETS}
    memory_probs = {n: s for n, s in valid.items() if s.bucket in _MEMORY_BUCKETS}

    if compute_probs:
        hits = sum(1 for s in compute_probs.values()
                   if s.m1_track_gain > WEIGHT_GATE_M1_THRESHOLD)
        compute_rate = hits / len(compute_probs)
    else:
        compute_rate = 0.0

    if memory_probs:
        null_hits = sum(1 for s in memory_probs.values() if s.m3_null)
        memory_null_rate = null_hits / len(memory_probs)
    else:
        memory_null_rate = 0.0

    retire_hits = sum(1 for s in valid.values() if s.m4_retire_count > 0)
    retire_rate = (retire_hits / len(valid)) if valid else 0.0

    gain_distribution = {n: s.m1_track_gain for n, s in valid.items()}
    wasted_distribution = {n: s.m6_wasted_rounds for n, s in valid.items()}

    return AggregateStats(
        per_problem=per_problem,
        compute_reproduction_rate=compute_rate,
        compute_class_pass=compute_rate >= CLASS_PASS_THRESHOLD,
        memory_null_rate=memory_null_rate,
        memory_class_pass=memory_null_rate >= CLASS_PASS_THRESHOLD,
        retire_rate=retire_rate,
        gate_fail_count=gate_fail_count,
        gain_distribution=gain_distribution,
        wasted_rounds_distribution=wasted_distribution,
    )


def format_report(agg: AggregateStats) -> str:
    """사람이 읽는 요약 리포트 텍스트(§1-2 표 형태) — CLI 출력용."""
    lines = []
    lines.append("=== P8 §1-2 사전 등록 기준 대조 ===")
    lines.append(f"compute-bound 재현율: {agg.compute_reproduction_rate:.1%} "
                f"({'PASS' if agg.compute_class_pass else 'FAIL'} — 판정선 {CLASS_PASS_THRESHOLD:.0%})")
    lines.append(f"memory-bound null율: {agg.memory_null_rate:.1%} "
                f"({'PASS' if agg.memory_class_pass else 'FAIL'} — 판정선 {CLASS_PASS_THRESHOLD:.0%})")
    lines.append(f"retire 발생률: {agg.retire_rate:.1%}")
    lines.append(f"gate_fail 무효 표본: {agg.gate_fail_count}건")
    lines.append("")
    lines.append("=== 문제별 M1~M6 ===")
    for name, s in sorted(agg.per_problem.items()):
        flag = "[GATE_FAIL]" if s.m5_gate_fail else ""
        lines.append(f"  {name:30s} bucket={s.bucket:26s} M1={s.m1_track_gain:6.2f}x "
                    f"M2={s.m2_cross_track_gap:6.2f}x null={s.m3_null!s:5s} "
                    f"retire={s.m4_retire_count} wasted={s.m6_wasted_rounds} {flag}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import json
    import sys
    from pathlib import Path

    if "--selfcheck" in argv:
        fake_results = {
            "matmul": {"on": {"label": "evolve_ON", "metric_curve": [(0, 10.0), (1, 2.0)],
                              "stop_reason": "converged", "retire_count": 0,
                              "wasted_rounds": 0, "signals": [{"x": 1}, {"x": 2}]},
                      "off": {"label": "evolve_OFF", "metric_curve": [(0, 10.0), (1, 9.0)],
                             "stop_reason": "converged", "retire_count": 0,
                             "wasted_rounds": 1, "signals": [{"x": 1}, {"x": 1}]}},
            "kb_softmax": {"on": {"label": "evolve_ON", "metric_curve": [(0, 5.0)],
                                  "stop_reason": "below_weight_gate", "retire_count": 0,
                                  "wasted_rounds": 0, "signals": [{}]},
                          "off": {"label": "evolve_OFF", "metric_curve": [(0, 5.0)],
                                 "stop_reason": "below_weight_gate", "retire_count": 0,
                                 "wasted_rounds": 0, "signals": [{}]}},
        }
        buckets = {"matmul": "compute-matmul", "kb_softmax": "memory-reduce-elementwise"}
        agg = aggregate_stats(fake_results, buckets)
        assert agg.compute_reproduction_rate == 1.0
        assert agg.memory_null_rate == 1.0
        report = format_report(agg)
        assert "PASS" in report
        print(report)
        print("report_p8_stats.py self-check PASS")
        return 0

    if len(argv) < 1:
        print("usage: report_p8_stats.py <result1.json>[,<result2.json>...] "
              "[--buckets=<buckets.json>] [--selfcheck]", file=sys.stderr)
        return 2

    result_paths = [Path(p) for p in argv[0].split(",")]
    merged: dict = {"chip": "", "results": {}}
    for p in result_paths:
        data = json.loads(p.read_text())
        merged["results"].update(data.get("results") or {})

    buckets_arg = next((a for a in argv if a.startswith("--buckets=")), None)
    if buckets_arg:
        buckets = json.loads(Path(buckets_arg.split("=", 1)[1]).read_text())
    else:
        buckets = {name: "" for name in merged["results"]}

    agg = aggregate_stats(merged["results"], buckets)
    print(format_report(agg))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
