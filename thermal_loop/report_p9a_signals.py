"""P9-A Task 41 — R0 signal_dict 판별 리포터 (10-p9a-rule-coverage-design.md §2-1/
§2-3/§4 Task 41, T-conv + T-fire).

기존 P8 배치 result JSON(run_ablation_remote.py 포맷)에서 conv-fusion R0 STOP 7건 +
(d) 미발화 2건 + matmul 대조군의 evolve_ON 트랙 R0 signal_dict(signals[0])를 추출해
결정론으로 판별한다. A100 재실측 불요 — signals가 P8 배치 시점에 이미 라운드별로
보존됨([[06-p6-signal-provenance-design]]).

판별 로직은 rules.py의 실제 Rule.cond를 **그대로 재사용**(하드코딩 복제 금지,
design §4 Task 41 명시) — seed_rules()에서 label로 찾은 cond 함수를 signal에
직접 호출한다. 이래야 rules.py 임계값이 바뀌면 이 리포터도 자동으로 따라간다.

STOP 7건 판별:
- (i) legit: R0 신호가 tensorcore_saturated 또는 memory_saturated cond를 실제로
  충족 → STOP이 정당(진화 시작 자체가 무의미).
- (ii) misfire: 둘 다 미충족인데 STOP으로 기록 → 신호-판정 불일치(오발화 의심,
  Task 42 조정 후보).

(d) 2건 판별:
- (i) legit: fp32_no_tensorcore cond의 3개 조건(weight_pct>=GATE, compute_tput>0,
  not tensorcore_active) 중 하나 이상이 실제로 미충족 → 미발화가 정당.
- (ii) tuning_room: cond가 사실 다 충족되는데 미발화로 기록됨 → 문턱 조정 여지
  (또는 상위 priority 룰 선점 등 다른 요인 — 재실측/추가 조사 대상).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from rules import seed_rules
from signals import Signal, from_dict


def _cond_for(label: str):
    """rules.seed_rules()에서 label에 해당하는 Rule의 cond 함수를 그대로 가져온다.
    하드코딩 복제 금지 — rules.py 임계값 변경 시 이 리포터도 자동 추종.
    """
    for r in seed_rules():
        if r.label == label:
            return r.cond
    raise KeyError(f"rules.py에 label={label!r} 룰이 없음")


@dataclass
class Verdict:
    verdict: str            # "legit" | "misfire" | "tuning_room" | "control"
    matched_cond: str = ""  # STOP 판별: 어느 cond가 충족됐는지
    reason: str = ""        # 비발화 판별: 어느 하위조건이 미충족인지


def classify_stop_signal(sig: Signal) -> Verdict:
    """conv-fusion STOP 7건 — tensorcore_saturated/memory_saturated cond 실충족 여부.

    두 cond 모두 rules.py에서 그대로 가져와 호출(하드코딩 금지).
    """
    tc_sat_cond = _cond_for("tensorcore_saturated")
    mem_sat_cond = _cond_for("memory_saturated")

    if tc_sat_cond(sig):
        return Verdict(verdict="legit", matched_cond="tensorcore_saturated")
    if mem_sat_cond(sig):
        return Verdict(verdict="legit", matched_cond="memory_saturated")
    return Verdict(verdict="misfire",
                   reason="tensorcore_saturated/memory_saturated 둘 다 미충족인데 STOP 기록")


def classify_nonfire_signal(sig: Signal) -> Verdict:
    """(d) 33_Gemm/86_Matmul — fp32_no_tensorcore 미발화 원인 판별.

    cond 자체(weight_pct>=WEIGHT_GATE and compute_tput>0 and not tensorcore_active)를
    rules.py에서 가져와 재사용. 하위조건을 분해해 어느 조건이 미충족인지 근거로 남긴다
    (design §2-3: "미발화 원인 필드 명시").
    """
    from rules import WEIGHT_GATE

    fp32_cond = _cond_for("fp32_no_tensorcore")

    reasons = []
    if not (sig.weight_pct >= WEIGHT_GATE):
        reasons.append(f"weight_pct={sig.weight_pct:.4f} < WEIGHT_GATE={WEIGHT_GATE}")
    if not (sig.compute_tput > 0):
        reasons.append(f"compute_tput={sig.compute_tput:.4f} <= 0")
    if sig.tensorcore_active:
        reasons.append(f"tensorcore_active={sig.tensorcore_active} (not 미충족)")

    if fp32_cond(sig):
        # cond가 실제로 다 충족되는데 미발화 기록 — 모순, 문턱조정/우선순위 요인 의심.
        return Verdict(verdict="tuning_room",
                       reason="fp32_no_tensorcore cond 실충족인데 미발화 기록 — "
                              "상위 priority 선점 또는 문턱조정 여지 의심")
    return Verdict(verdict="legit", reason="; ".join(reasons) or "cond 미충족(원인 불명)")


def extract_r0_signal(track: dict) -> Signal | None:
    """track(예: result["problem"]["on"])의 signals[0](R0)을 Signal로 추출.

    signals 키 자체가 없거나 빈 리스트면 None(구포맷/데이터 없음). signals[0]이
    빈 dict({})인 경우는 gate_fail 라운드일 수 있음(P6 관례) — 크래시 없이 기본값
    Signal 반환, 호출측이 별도로 gate_fail 여부 판단.
    """
    sigs = track.get("signals")
    if not sigs:
        return None
    return from_dict(sigs[0])


@dataclass
class SignalRow:
    problem: str
    verdict: str
    signal: Signal
    detail: str = ""


def build_signal_table(results: dict[str, dict], stop_problems: set[str],
                       nonfire_problems: set[str],
                       control_problems: set[str] | None = None) -> dict[str, SignalRow]:
    """result dict({problem: {"on":..., "off":...}}) → 문제별 판별 표.

    stop_problems: conv STOP 7건 등 STOP 판별 대상.
    nonfire_problems: (d) 미발화 2건 등 미발화 판별 대상.
    control_problems: matmul 대조군 등 — 신호값만 표에 넣고 verdict="control".
    """
    control_problems = control_problems or set()
    table: dict[str, SignalRow] = {}
    for name, res in results.items():
        on = res.get("on") or {}
        sig = extract_r0_signal(on)
        if sig is None:
            continue
        if name in stop_problems:
            v = classify_stop_signal(sig)
            table[name] = SignalRow(problem=name, verdict=v.verdict, signal=sig,
                                    detail=v.matched_cond or v.reason)
        elif name in nonfire_problems:
            v = classify_nonfire_signal(sig)
            table[name] = SignalRow(problem=name, verdict=v.verdict, signal=sig,
                                    detail=v.reason)
        elif name in control_problems:
            table[name] = SignalRow(problem=name, verdict="control", signal=sig)
    return table


def format_signal_report(table: dict[str, SignalRow]) -> str:
    lines = []
    lines.append("=== P9-A Task 41 — R0 signal_dict 판별 (T-conv + T-fire) ===")
    for name, row in sorted(table.items()):
        s = row.signal
        lines.append(
            f"  {name:55s} verdict={row.verdict:12s} "
            f"tc_active={s.tensorcore_active!s:5s} compute_tput={s.compute_tput:6.4f} "
            f"bw_pct={s.bw_pct:6.4f} weight_pct={s.weight_pct:6.4f} "
            f"latency_us={s.latency_us:10.2f}")
        if row.detail:
            lines.append(f"      근거: {row.detail}")
    stop_rows = [r for r in table.values() if r.verdict in ("legit", "misfire")]
    n_misfire = sum(1 for r in stop_rows if r.verdict == "misfire")
    lines.append("")
    lines.append(f"요약: STOP 판별 대상 {len(stop_rows)}건 중 오발화(misfire) {n_misfire}건")
    return "\n".join(lines)


# ── 실 P8 배치 result JSON 대상 문제 목록 (design §4 Task 41 명시) ──
# p8_buckets.json compute-conv-fusion 10문제 중 R0 STOP 7건(v4 §9-2 근거).
CONV_STOP_PROBLEMS = {
    "21_Conv2d_Add_Scale_Sigmoid_GroupNorm",
    "27_Conv3d_HardSwish_GroupNorm_Mean",
    "62_conv_standard_2D__square_input__asymmetric_kernel",
    "66_conv_standard_3D__asymmetric_input__asymmetric_kernel",
    "71_Conv2d_Divide_LeakyReLU",
    "75_Gemm_GroupNorm_Min_BiasAdd",
    "90_Conv3d_LeakyReLU_Sum_Clamp_GELU",
}
# (d) 미발화 2건.
NONFIRE_PROBLEMS = {"33_Gemm_Scale_BatchNorm", "86_Matmul_Divide_GELU"}
# matmul 대조군(신호 분포 대비용).
CONTROL_PROBLEMS = {"13_Matmul_for_symmetric_matrices", "3_Batched_matrix_multiplication"}


def main(argv: list[str]) -> int:
    if "--selfcheck" in argv:
        # (a) STOP legit — tensorcore_saturated 실충족
        v = classify_stop_signal(from_dict({
            "tensorcore_active": True, "compute_tput": 0.9,
            "latency_us": 200.0, "weight_pct": 1.0}))
        assert v.verdict == "legit" and v.matched_cond == "tensorcore_saturated", v

        # (b) STOP legit — memory_saturated 실충족
        v = classify_stop_signal(from_dict({"bw_pct": 0.9, "tensorcore_active": False}))
        assert v.verdict == "legit" and v.matched_cond == "memory_saturated", v

        # (c) STOP misfire — 둘 다 미충족
        v = classify_stop_signal(from_dict({
            "tensorcore_active": False, "compute_tput": 0.1,
            "bw_pct": 0.2, "latency_us": 10.0, "weight_pct": 0.3}))
        assert v.verdict == "misfire", v

        # (d) 미발화 legit — weight_pct 미달
        v = classify_nonfire_signal(from_dict({
            "weight_pct": 0.01, "compute_tput": 0.5, "tensorcore_active": False}))
        assert v.verdict == "legit", v

        # (e) 미발화 legit — tensorcore 이미 활성
        v = classify_nonfire_signal(from_dict({
            "weight_pct": 0.5, "compute_tput": 0.5, "tensorcore_active": True}))
        assert v.verdict == "legit", v

        # (f) 미발화 tuning_room — cond 실충족인데 미발화 기록(모순 케이스)
        v = classify_nonfire_signal(from_dict({
            "weight_pct": 0.5, "compute_tput": 0.5, "tensorcore_active": False}))
        assert v.verdict == "tuning_room", v

        # (g) extract_r0_signal 방어 — 빈 signals/gate_fail dict
        assert extract_r0_signal({}) is None
        assert extract_r0_signal({"signals": []}) is None
        assert extract_r0_signal({"signals": [{}]}) is not None

        print("report_p9a_signals.py self-check PASS")
        return 0

    import json
    import sys
    from pathlib import Path

    if len(argv) < 1:
        print("usage: report_p9a_signals.py <result1.json>[,<result2.json>...] "
              "[--selfcheck]", file=sys.stderr)
        return 2

    merged: dict = {}
    for p in argv[0].split(","):
        data = json.loads(Path(p).read_text())
        merged.update(data.get("results") or {})

    table = build_signal_table(merged, CONV_STOP_PROBLEMS, NONFIRE_PROBLEMS,
                               CONTROL_PROBLEMS)
    print(format_signal_report(table))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
