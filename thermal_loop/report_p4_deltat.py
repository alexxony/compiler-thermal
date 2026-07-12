"""P4 Task 16 — RcBackend ΔT 리포트. P3 artifacts/ ledger를 읽어 시나리오 (a)/(b) 결과표 출력.

design: docs/04-p4-rc-deltat-design.md §5 Task 16, §6 실행 절차.
Colab 실행 아님 — 로컬 배치, GPU/Colab 세션 불요. run_thermal_gain.py의 리포트
포맷 패턴(구간 print) 재사용, 실측 없이 이미 있는 P3 raw 신호만 후처리.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thermal.duty_power import to_power_series, duty_avg_power
from thermal.twin_eval import RcBackend

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# RcBackend 기존 검증 파라미터(tests/test_twin_eval.py make_backend()와 동일) —
# 계획 §2 "기본안: 기존 검증 파라미터 그대로 사용" 결정 그대로.
RC_KW = dict(r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
             c_die=50.0, c_hbm=10.0, t_ambient=45.0)

# 계획 §3 idle 결정(팀리드 승인): 문헌치 A100 idle ≈ 55W. HBM 쪽은 die의 1/10 근사
# (정밀치 미확보 — sensitivity 테스트(Task 15)가 이 가정의 민감도를 별도 검증).
IDLE_DIE_W = 55.0
IDLE_HBM_W = 5.5

WINDOW_S = 60.0  # test_twin_eval.py 60s 창(시정수~6.7s의 9배) 그대로 상속


def _load_round(path: Path, round_idx: int) -> dict:
    with open(path) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    rec = next(r for r in lines if r["round_idx"] == round_idx)
    sig = rec["signal"]
    return {"p_die_w": sig["p_die_w"], "p_hbm_w": sig["p_hbm_w"],
            "kernel_time_s": sig["kernel_time_s"],
            "energy_per_iter_j": sig.get("energy_per_iter_j", 0.0),
            "hypothesis_label": rec["hypothesis_label"]}


def _steady_t_hbm(power_df) -> float:
    out = RcBackend(**RC_KW).evaluate(power_df)
    return out["t_hbm_c"].iloc[-1]


def _scenario_a(sig: dict) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="saturated", window_s=WINDOW_S,
                              idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W)
    return _steady_t_hbm(series)


def _scenario_b(sig: dict, repeat_period_s: float) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="iso_work", window_s=WINDOW_S,
                              idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W,
                              repeat_period_s=repeat_period_s)
    return _steady_t_hbm(series)


def main() -> int:
    off_path = ARTIFACTS / "thermal-gain-matmul-off.jsonl"
    if not off_path.exists():
        print(f"ERR: 없음 {off_path} — P3 실측 결과 필요", file=sys.stderr)
        return 2

    fp32 = _load_round(off_path, 0)   # round 0 = seed(fp32)
    tf32 = _load_round(off_path, 1)   # round 1 = TF32 variant

    print("=" * 60)
    print("P4 RcBackend ΔT 리포트 — 시나리오 (a) 포화 vs (b) iso-work")
    print("=" * 60)
    print(f"\n입력(P3 실측, thermal-gain-matmul-off.jsonl):")
    print(f"  fp32(seed) : p_die_w={fp32['p_die_w']:.2f}W  p_hbm_w={fp32['p_hbm_w']:.2f}W  "
          f"kernel_time_s={fp32['kernel_time_s']*1000:.3f}ms  energy/iter={fp32['energy_per_iter_j']:.4f}J")
    print(f"  TF32       : p_die_w={tf32['p_die_w']:.2f}W  p_hbm_w={tf32['p_hbm_w']:.2f}W  "
          f"kernel_time_s={tf32['kernel_time_s']*1000:.3f}ms  energy/iter={tf32['energy_per_iter_j']:.4f}J")
    energy_ratio = fp32["energy_per_iter_j"] / tf32["energy_per_iter_j"]
    print(f"  energy_per_iter_j 개선 배율: {energy_ratio:.2f}x (P3 결론 참고)")

    a_fp32 = _scenario_a(fp32)
    a_tf32 = _scenario_a(tf32)
    print(f"\n--- 시나리오 (a) 포화 실행(연속 100% duty) ---")
    print(f"  fp32 t_hbm_c(정상상태) = {a_fp32:.4f} C  (ΔT={a_fp32 - RC_KW['t_ambient']:.4f}K)")
    print(f"  TF32 t_hbm_c(정상상태) = {a_tf32:.4f} C  (ΔT={a_tf32 - RC_KW['t_ambient']:.4f}K)")
    a_verdict = "TF32 >= fp32 (예상대로 열에 불리)" if a_tf32 >= a_fp32 else "⚠️ 예상과 다름(TF32 < fp32)"
    print(f"  판정: {a_verdict}")

    repeat_period_s = fp32["kernel_time_s"]
    b_fp32 = _scenario_b(fp32, repeat_period_s)
    b_tf32 = _scenario_b(tf32, repeat_period_s)
    print(f"\n--- 시나리오 (b) 등작업량(iso-work) duty cycle ---")
    duty_tf32 = tf32["kernel_time_s"] / repeat_period_s
    print(f"  duty_tf32 = {tf32['kernel_time_s']*1000:.3f}/{repeat_period_s*1000:.3f} = {duty_tf32*100:.2f}%")
    print(f"  idle_die_w={IDLE_DIE_W}W idle_hbm_w={IDLE_HBM_W}W (문헌치, 팀리드 승인)")
    print(f"  fp32 t_hbm_c(정상상태) = {b_fp32:.4f} C  (ΔT={b_fp32 - RC_KW['t_ambient']:.4f}K)")
    print(f"  TF32 t_hbm_c(정상상태) = {b_tf32:.4f} C  (ΔT={b_tf32 - RC_KW['t_ambient']:.4f}K)")
    b_verdict = "★ TF32 < fp32 (energy_per_iter_j 5.3× 결론과 정합)" if b_tf32 < b_fp32 else "⚠️ 예상과 다름(TF32 >= fp32)"
    print(f"  판정: {b_verdict}")

    delta_temp_gap = (b_fp32 - RC_KW["t_ambient"]) - (b_tf32 - RC_KW["t_ambient"])
    print(f"\n--- (a) vs (b) 대조 ---")
    print(f"  (a) TF32 ΔT {'≥' if a_tf32 >= a_fp32 else '<'} fp32 ΔT,  "
          f"(b) TF32 ΔT {'<' if b_tf32 < b_fp32 else '≥'} fp32 ΔT")
    print(f"  → duty cycle 정의가 결론을 뒤집음: {'실측 확인됨' if (a_tf32 >= a_fp32 and b_tf32 < b_fp32) else '재검토 필요'}")
    print(f"  (b) 온도축 개선폭: ΔT_fp32 - ΔT_TF32 = {delta_temp_gap:.4f} K "
          f"(energy_per_iter_j 배율 {energy_ratio:.2f}x와 자릿수 비교는 사람이 해석)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
