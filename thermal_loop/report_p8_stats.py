"""P8 Task 35 — 통계 집계 리포터 (08-p8-scale-ablation-design.md §1-1/§1-2/Task 35).

배치별 결과 JSON(run_ablation_remote.py __ABL_RESULT__ 포맷, merge_batch_results로
병합 가능)을 입력받아 §1-1 M1~M6 지표를 문제별로 산출하고, §1-2 클래스별 재현율
(≥70% 판정선)을 집계한다. report_p4_deltat.py와 별개 — ΔT 축이 아니라 energy 축
통계 전용. 실측(run) 없음, 순수 후처리. torch 불요.

부호 규약(2026-07-19 수정, s2-799 재발 방지): metric_curve는 harness._metric의
저장값 그대로 = **-energy_per_iter_j**(항상 음수, 클수록/0에 가까울수록 좋음 —
"메트릭 우상향해야" harness.py 테스트 주석 참조). 이 파일은 소비 시점에
energy = -metric으로 환원해 "값이 작을수록 좋음" 축으로 다룬다. 원래
`_seed_and_best`가 raw metric에 직접 min()을 적용해 seed(round0, 가장 나쁜
raw 값)를 best로 오판 — M1이 모든 문제에서 1.00 아티팩트로 고정되는 버그였다
(P9-A ncu 스팟체크 실측으로 확정, JOURNAL 2026-07-19T13:04:29+09:00).

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
    """curve = [(round_idx, metric)]. metric은 harness 저장 규약 그대로
    **-energy_per_iter_j**(항상 음수, 값이 클수록/0에 가까울수록 좋음)를 따른다.

    이 함수는 energy = -metric으로 환원한 뒤 "값이 작을수록 좋음" 축에서
    seed=round0 energy, best=min(energy)를 반환한다(반환값은 항상 양수 energy —
    호출측이 seed/best 나눗셈으로 배율을 계산하는 관례와 부합). 곡선에 양수
    metric(v > 0)이 하나라도 섞여 있으면 부호 규약 위반(energy/latency 모드
    모두 metric은 음수여야 함, harness.py::_metric 참조) — ValueError.
    """
    if not curve:
        raise ValueError("빈 metric_curve — 문제 결과 이상")
    if any(v > 0 for _, v in curve):
        raise ValueError(
            f"metric_curve에 양수 값 포함 — 부호 규약 위반(metric은 -energy, "
            f"항상 음수여야 함): {curve}")
    energies = [-v for _, v in curve]
    seed_energy = energies[0]
    best_energy = min(energies)
    return seed_energy, best_energy


def compute_metrics_for_problem(problem: str, result: dict, bucket: str) -> ProblemStats:
    """result = {"on": track_dict, "off": track_dict} (run_ablation_remote._to_track
    이전 raw dict 형태 — 이 함수는 raw dict를 직접 소비, TrackResult 의존 없음).

    _seed_and_best가 이미 -metric 환원을 마친 양수 energy를 반환하므로 아래
    m1/m2/m3 계산은 "energy 축(작을수록 좋음)" 그대로 성립 — 부호 처리는
    _seed_and_best 한 곳에만 있다(이중 반전 방지).
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
        # 부호 규약(harness._metric): metric_curve는 -energy_per_iter_j — 항상 음수,
        # 값이 클수록(0에 가까울수록) 좋음. 아래 fake curve 전부 음수로 교체
        # (s2-799 재발 방지 — 예전 selfcheck는 양수 곡선만 써서 부호 버그를 못 잡았음).
        fake_results = {
            # matmul: seed round0 energy=10.0(=-(-10.0)) -> best round1 energy=2.0
            # (=-(-2.0)) -> M1 = 10.0/2.0 = 5.0 (개선 실재, seed != best).
            "matmul": {"on": {"label": "evolve_ON", "metric_curve": [(0, -10.0), (1, -2.0)],
                              "stop_reason": "converged", "retire_count": 0,
                              "wasted_rounds": 0, "signals": [{"x": 1}, {"x": 2}]},
                      "off": {"label": "evolve_OFF", "metric_curve": [(0, -10.0), (1, -9.0)],
                             "stop_reason": "converged", "retire_count": 0,
                             "wasted_rounds": 1, "signals": [{"x": 1}, {"x": 1}]}},
            "kb_softmax": {"on": {"label": "evolve_ON", "metric_curve": [(0, -5.0)],
                                  "stop_reason": "below_weight_gate", "retire_count": 0,
                                  "wasted_rounds": 0, "signals": [{}]},
                          "off": {"label": "evolve_OFF", "metric_curve": [(0, -5.0)],
                                 "stop_reason": "below_weight_gate", "retire_count": 0,
                                 "wasted_rounds": 0, "signals": [{}]}},
        }
        buckets = {"matmul": "compute-matmul", "kb_softmax": "memory-reduce-elementwise"}
        agg = aggregate_stats(fake_results, buckets)
        assert agg.compute_reproduction_rate == 1.0
        assert agg.memory_null_rate == 1.0
        assert abs(agg.per_problem["matmul"].m1_track_gain - 5.0) < 1e-9, (
            f"M1 부호 회귀 — {agg.per_problem['matmul'].m1_track_gain} (기대 5.0)")
        report = format_report(agg)
        assert "PASS" in report
        print(report)

        # (a) seed≠best 개선 곡선, 비단조 포함 — [(0,-5.0),(1,-3.0),(2,-1.0),(3,-4.0)]
        # 처럼 중간에 악화(-4.0 < -1.0, energy 기준 4.0 > 1.0)가 섞여도 best는
        # round2(energy=1.0) 유지 -> M1 = seed(5.0)/best(1.0) = 5.0.
        nonmonotone_curve = [(0, -5.0), (1, -3.0), (2, -1.0), (3, -4.0)]
        seed_e, best_e = _seed_and_best(nonmonotone_curve)
        assert abs(seed_e - 5.0) < 1e-9 and abs(best_e - 1.0) < 1e-9, (
            f"비단조 곡선 seed/best 오산: seed={seed_e} best={best_e}")
        assert abs((seed_e / best_e) - 5.0) < 1e-9, "비단조 곡선 M1 오산"
        print("  비단조 개선 곡선 체크: PASS (seed=5.0, best=1.0(round2), M1=5.0)")

        # (b) 양수 곡선 입력 -> ValueError (부호 규약 위반 감지).
        try:
            _seed_and_best([(0, 5.0), (1, 2.0)])
            raise AssertionError("양수 metric_curve가 ValueError 없이 통과함 — 회귀")
        except ValueError:
            print("  양수 곡선 방어 체크: PASS (ValueError 발생 확인)")

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
