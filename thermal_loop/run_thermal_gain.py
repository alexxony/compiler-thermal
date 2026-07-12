"""열-gain 드라이버 (Compiler_Thermal P3) — run_gain_hypcond.py 열축 fork.

목적: matmul TF32가 latency뿐 아니라 energy_j(→T_hbm)도 줄이는가.
설계: docs/03-p3-thermal-gain-design.md. GPU-Solver run_gain_hypcond의 가설-조건부
콜백 패턴(발화 룰 라벨→코드 선택)을 그대로 재사용 — rules/harness/evolver/ledger
전부 무변경. 바뀐 건 metric_mode="thermal"(harness._metric이 -energy_j 반환)과
submit_kind="thermal"(executor가 ncu+power 분리 런으로 분기)뿐.

ON/OFF 두 트랙 공정 비교(run_gain_hypcond와 동일 원칙):
  ON : fp32_no_tensorcore 발화 → TF32 variant → 재측정(energy_j 개선?).
  OFF: evolve_enabled=False — 판정 없이 같은 콜백 로직, retire 불가.
  두 트랙 모두 동일 콜백·동일 variant. 차이는 evolver(retire)에서만.

실행: python run_thermal_gain.py matmul [max_rounds] --colab-cli --session=<name>
  전제: Colab GPU 세션 + thermal/*.py 업로드(setup_remote 확장 필요, §5).
"""
from __future__ import annotations
import sys
from pathlib import Path

from runner import run_problem
from ledger import Ledger
from rules import seed_rules
from generator import CallbackGenerator
from run_e2e import git_sync, MAILBOX, PROBLEMS, make_colab_profiler


def _make_hypcond_callback(seed_code: str, variant_map: dict[str, str]):
    """발화 룰 라벨 → 코드 매핑 콜백. run_gain_hypcond와 동일 로직(재사용)."""
    def cb(problem, hyp, prev_code):
        label = hyp if isinstance(hyp, str) else None
        code = variant_map.get(label, seed_code) if label else seed_code
        which = label if (label in variant_map) else "seed(base)"
        print(f"  [generate] 직전가설={hyp!r} → 코드={which}")
        return code
    return cb


def _run_track(label, problem, seed_code, variant_map, max_rounds,
               evolve_enabled, ledger_path, profiler=None):
    if Path(ledger_path).exists():
        Path(ledger_path).unlink()
    rules = seed_rules()
    gen = CallbackGenerator(_make_hypcond_callback(seed_code, variant_map))
    print(f"\n=== 트랙 {label} (evolve={'ON' if evolve_enabled else 'OFF'}, metric=thermal) ===")
    res = run_problem(problem, seed_code, MAILBOX, ledger_path,
                      sync_fn=git_sync, max_rounds=max_rounds, poll_s=5.0,
                      timeout_s=900.0, rules=rules, generator=gen,
                      evolve_enabled=evolve_enabled, metric_mode="thermal",
                      profiler=profiler, submit_kind="thermal")
    led = Ledger(str(ledger_path))
    recs = [r for r in led.records if r.problem == problem]
    curve = led.metric_curve(problem)   # [(round, -energy_j)]
    fired = [r.hypothesis_label for r in recs]
    wasted = sum(1 for r in recs if r.passed and not r.improved)
    retires = sum(1 for e in res.events if e.kind == "retire")
    # 열 축 원신호 보존(리포트용) — energy_j/p_hbm_w/p_die_w는 metric 곡선(음수화)엔 없음.
    thermal_series = [(r.round_idx, r.signal.get("energy_j", 0.0),
                       r.signal.get("p_hbm_w", 0.0), r.signal.get("p_die_w", 0.0))
                      for r in recs if r.passed]
    return {
        "label": label, "curve": curve, "fired_rules": fired,
        "wasted_rounds": wasted, "retire_count": retires,
        "stop_reason": res.stopped_reason, "rounds": res.rounds,
        "thermal_series": thermal_series,
    }


def _report(on: dict, off: dict) -> None:
    print("\n" + "=" * 56)
    print("열-gain 비교 — 진화 ON vs OFF (metric=thermal, energy_j)")
    print("=" * 56)

    def _best_energy(t):
        vals = [m for _, m in t["curve"] if m != 0.0]
        return (-max(vals)) if vals else None   # metric=-energy_j → best(max)=최소 에너지

    for t in (off, on):
        print(f"\n[{t['label']}] rounds={t['rounds']} stop={t['stop_reason']}")
        print(f"  energy 곡선(J): {[round(e, 4) for _, e, _, _ in t['thermal_series']]}")
        print(f"  p_hbm 곡선(W) : {[round(p, 2) for _, _, p, _ in t['thermal_series']]}")
        print(f"  p_die 곡선(W) : {[round(p, 2) for _, _, _, p in t['thermal_series']]}")
        be = _best_energy(t)
        print(f"  best(최소) energy: {round(be, 4) if be is not None else 'N/A'} J")
        print(f"  발화 룰     : {t['fired_rules']}")
        print(f"  헛라운드    : {t['wasted_rounds']}")
        print(f"  retire 수   : {t['retire_count']}")

    print("\n--- 판정 ---")
    bon, boff = _best_energy(on), _best_energy(off)
    if bon is not None and boff is not None:
        if bon < boff:
            print(f"  ★열-gain★ ON best={bon:.4f}J < OFF best={boff:.4f}J (진화→에너지 덜 쓰는 커널)")
        else:
            print(f"  ⚠️ 열-gain 없음 — ON best={bon:.4f}J, OFF best={boff:.4f}J")
    else:
        print("  ⚠️ 에너지 측정 부재 (칩/전력 미탐지 — merge_signals 미실행 라운드만 있었음)")

    print("\n⚠️ 정직 캐비앗: TF32는 DRAM 트래픽을 거의 안 바꾼다(A@B 크기 불변) →")
    print("   p_hbm_w는 ON/OFF 간 거의 불변이 예상 방향. energy 개선이 있다면 그건")
    print("   주로 p_die_w(연산시간 단축)에서 온다 — speed-gain과 heat-gain이 같은 방향인지")
    print("   원인(트래픽 vs 연산시간)까지 이 리포트의 p_hbm/p_die 곡선으로 분리해서 봐야 한다.")


def main() -> int:
    profiler, use_colab_cli, argv = make_colab_profiler(sys.argv[1:])
    problem = argv[0] if len(argv) > 0 else "matmul"
    max_rounds = int(argv[1]) if len(argv) > 1 else 8

    if problem != "matmul":
        print(f"ERR: P3는 matmul만 지원 (시드룰 단일 문제 범위, 03-p3 계획 §6)", file=sys.stderr)
        return 2

    seed_path = PROBLEMS / problem / "solve.py"
    vdir = PROBLEMS / problem / "variants"
    if not use_colab_cli and not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음 {MAILBOX}", file=sys.stderr); return 2

    tf32on = vdir / "R_tf32on.py"
    if not tf32on.exists():
        print(f"ERR: 없음 {tf32on}", file=sys.stderr); return 2
    seed_code = seed_path.read_text()
    variant_map = {"fp32_no_tensorcore": tf32on.read_text()}

    print(f"열-gain (가설-조건부) — {problem}, max_rounds={max_rounds}")
    print(f"  variant_map: {list(variant_map)} (seed=base)")
    print(f"  ON=fp32 발화→TF32→energy_j 재측정 / OFF=fp32 영원 발화, retire 불가\n")

    base = MAILBOX.parent / f"thermal-gain-{problem}"
    off = _run_track("evolve_OFF", problem, seed_code, variant_map,
                     max_rounds, False, f"{base}-off.jsonl", profiler=profiler)
    on = _run_track("evolve_ON", problem, seed_code, variant_map,
                    max_rounds, True, f"{base}-on.jsonl", profiler=profiler)
    _report(on, off)
    return 0


if __name__ == "__main__":
    sys.exit(main())
