"""P4 Task 11~15 — 전력→시계열 변환 + duty-cycle 평균전력 + 시나리오 (a)/(b) + sensitivity.

design: docs/04-p4-rc-deltat-design.md §3(시나리오 정의)/§4(검증 기준)/§5(태스크 분해).
전부 로컬(GPU/Colab 불요) — P3 실측값(loop/artifacts/thermal-gain-matmul-*.jsonl)을
그대로 상수로 박아 쓴다(재측정 없음, 계획 문서 §2/§6과 일치).
"""
import pandas as pd
import pytest

from thermal.duty_power import to_power_series, duty_avg_power
from thermal.twin_eval import RcBackend

# P3 실측값(2026-07-12, thermal-p3b 세션, loop/artifacts/thermal-gain-matmul-{off,on}.jsonl
# round_idx=0(seed fp32)/1(TF32) 그대로 — 재측정 없음, 계획 §2/§6).
FP32 = {"p_die_w": 320.13431282651925, "p_hbm_w": 4.7542538401473795,
        "kernel_time_s": 0.00946672}
TF32 = {"p_die_w": 378.96262771639044, "p_hbm_w": 10.907638950276242,
        "kernel_time_s": 0.001482752}

# RcBackend 기존 검증 파라미터 그대로(tests/test_twin_eval.py make_backend()와 동일) —
# 계획 §2 "기본안: 기존 검증 파라미터 그대로 사용" 결정.
RC_KW = dict(r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
             c_die=50.0, c_hbm=10.0, t_ambient=45.0)

# 계획 §3 idle 결정: 문헌치 A100 idle ≈ 55W(팀리드 승인). HBM 쪽 idle은 die보다
# 훨씬 작게(refresh 전력 수준) — 여기선 die의 1/10 수준으로 잡음(정밀치 미확보,
# 계획 §3 "정확한 분해값은 미확보" 그대로 반영, sensitivity 테스트가 이 가정의
# 민감도를 별도로 검증).
IDLE_DIE_W = 55.0
IDLE_HBM_W = 5.5


# ── Task 11: to_power_series ──

def test_to_power_series_saturated_is_constant():
    df = to_power_series(p_die_w=FP32["p_die_w"], p_hbm_w=FP32["p_hbm_w"],
                          kernel_time_s=FP32["kernel_time_s"],
                          duty_scenario="saturated", window_s=1.0,
                          idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W, dt_s=0.1)
    assert list(df.columns) == ["Time", "p_die_w", "p_hbm_w"]
    assert (df["p_die_w"] == FP32["p_die_w"]).all()
    assert (df["p_hbm_w"] == FP32["p_hbm_w"]).all()
    assert df["Time"].iloc[0] == 0.0
    assert df["Time"].is_monotonic_increasing


def test_to_power_series_iso_work_duty_ratio():
    """iso_work: busy 구간 비율이 duty = kernel_time_s/repeat_period_s와 근사해야."""
    repeat_period_s = FP32["kernel_time_s"]  # 계획 §3 (b): fp32 kernel_time_s가 반복주기
    dt_s = 0.0002  # 반복주기(9.467ms)보다 훨씬 촘촘해야 duty 비율이 근사됨
    df = to_power_series(p_die_w=TF32["p_die_w"], p_hbm_w=TF32["p_hbm_w"],
                          kernel_time_s=TF32["kernel_time_s"],
                          duty_scenario="iso_work", window_s=repeat_period_s * 20,
                          idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W,
                          repeat_period_s=repeat_period_s, dt_s=dt_s)
    duty_expected = TF32["kernel_time_s"] / repeat_period_s
    busy_frac = (df["p_die_w"] == TF32["p_die_w"]).mean()
    assert busy_frac == pytest.approx(duty_expected, abs=0.03)
    # idle 구간은 idle_die_w/idle_hbm_w여야
    idle_rows = df[df["p_die_w"] != TF32["p_die_w"]]
    assert (idle_rows["p_die_w"] == IDLE_DIE_W).all()
    assert (idle_rows["p_hbm_w"] == IDLE_HBM_W).all()


def test_to_power_series_iso_work_requires_repeat_period():
    with pytest.raises((TypeError, ValueError)):
        to_power_series(p_die_w=1.0, p_hbm_w=1.0, kernel_time_s=0.001,
                        duty_scenario="iso_work", window_s=1.0,
                        idle_die_w=1.0, idle_hbm_w=1.0)


# ── Task 12: duty_avg_power ──

def test_duty_avg_power_formula():
    duty = 0.25
    busy, idle = 100.0, 20.0
    got = duty_avg_power(duty=duty, busy_power_w=busy, idle_power_w=idle)
    assert got == pytest.approx(0.25 * 100.0 + 0.75 * 20.0)


def test_duty_avg_power_duty_one_equals_busy():
    """duty=1.0(fp32, 이번 P3 raw 데이터 조건)이면 idle 영향 없이 busy_power 그대로."""
    got = duty_avg_power(duty=1.0, busy_power_w=320.13, idle_power_w=999.0)
    assert got == pytest.approx(320.13)


def test_duty_avg_power_from_kernel_times():
    """duty = kernel_time_s / repeat_period_s 공식(§3) 경유 계산도 동일해야."""
    duty = TF32["kernel_time_s"] / FP32["kernel_time_s"]
    assert duty == pytest.approx(0.1567, abs=0.001)
    got = duty_avg_power(duty=duty, busy_power_w=TF32["p_die_w"], idle_power_w=IDLE_DIE_W)
    # 순간전력(378.96W)보다는 훨씬 낮아야(대부분 시간을 idle로 보내므로)
    assert got < TF32["p_die_w"]
    assert got > IDLE_DIE_W  # idle보다는 높아야(일부 시간은 busy)


# ── Task 13: 시나리오 (a) 포화 실행 — TF32 ΔT >= fp32 ΔT ──

def _steady_t_hbm(power_df: pd.DataFrame) -> float:
    out = RcBackend(**RC_KW).evaluate(power_df)
    return out["t_hbm_c"].iloc[-1]


def test_scenario_a_saturated_tf32_hotter_or_equal():
    window_s = 60.0  # test_twin_eval.py 60s 창(시정수~6.7s의 9배) 그대로 상속
    fp32_series = to_power_series(p_die_w=FP32["p_die_w"], p_hbm_w=FP32["p_hbm_w"],
                                   kernel_time_s=FP32["kernel_time_s"],
                                   duty_scenario="saturated", window_s=window_s,
                                   idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W)
    tf32_series = to_power_series(p_die_w=TF32["p_die_w"], p_hbm_w=TF32["p_hbm_w"],
                                   kernel_time_s=TF32["kernel_time_s"],
                                   duty_scenario="saturated", window_s=window_s,
                                   idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W)
    fp32_t = _steady_t_hbm(fp32_series)
    tf32_t = _steady_t_hbm(tf32_series)
    assert tf32_t >= fp32_t, (
        f"시나리오(a) 기준 위반: TF32 t_hbm={tf32_t:.4f} < fp32 t_hbm={fp32_t:.4f} "
        "(설계 §4: 포화 실행에서는 TF32가 순간전력이 더 높아 ΔT도 높거나 같아야)"
    )


# ── Task 14: 시나리오 (b) iso-work — TF32 ΔT < fp32 ΔT ──

def _iso_work_series(sig: dict, repeat_period_s: float, window_s: float) -> pd.DataFrame:
    return to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                           kernel_time_s=sig["kernel_time_s"],
                           duty_scenario="iso_work", window_s=window_s,
                           idle_die_w=IDLE_DIE_W, idle_hbm_w=IDLE_HBM_W,
                           repeat_period_s=repeat_period_s)


def test_scenario_b_iso_work_tf32_cooler():
    repeat_period_s = FP32["kernel_time_s"]
    window_s = 60.0
    fp32_series = _iso_work_series(FP32, repeat_period_s, window_s)
    tf32_series = _iso_work_series(TF32, repeat_period_s, window_s)
    fp32_t = _steady_t_hbm(fp32_series)
    tf32_t = _steady_t_hbm(tf32_series)
    assert tf32_t < fp32_t, (
        f"시나리오(b) 기준 위반: TF32 t_hbm={tf32_t:.4f} >= fp32 t_hbm={fp32_t:.4f} "
        "(설계 §4: iso-work에서는 energy_per_iter_j 5.3× 개선과 같은 방향으로 "
        "TF32가 더 낮은 ΔT를 보여야)"
    )
    # 배율 기록용(assert 아님, §4 "배율 비례는 참고 정보") — 콘솔에 로그만.
    delta_fp32 = fp32_t - RC_KW["t_ambient"]
    delta_tf32 = tf32_t - RC_KW["t_ambient"]
    print(f"\n  [P4 (b) 기록] ΔT_fp32={delta_fp32:.4f}K ΔT_TF32={delta_tf32:.4f}K "
          f"(energy_per_iter_j 배율 참고: 5.32x~5.75x, P3 결과)")


# ── Task 15: idle_die_w sensitivity ──

@pytest.mark.parametrize("idle_mult", [0.7, 1.0, 1.3])
def test_scenario_b_sensitivity_to_idle_power(idle_mult):
    """idle_die_w를 ±30% 흔들어도 (b)의 방향성(TF32 ΔT < fp32 ΔT)이 안 뒤집혀야."""
    idle_die = IDLE_DIE_W * idle_mult
    idle_hbm = IDLE_HBM_W * idle_mult
    repeat_period_s = FP32["kernel_time_s"]
    window_s = 60.0

    def series(sig):
        return to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                               kernel_time_s=sig["kernel_time_s"],
                               duty_scenario="iso_work", window_s=window_s,
                               idle_die_w=idle_die, idle_hbm_w=idle_hbm,
                               repeat_period_s=repeat_period_s)

    fp32_t = _steady_t_hbm(series(FP32))
    tf32_t = _steady_t_hbm(series(TF32))
    assert tf32_t < fp32_t, (
        f"sensitivity 실패(idle_mult={idle_mult}): 방향성 반전 — idle 값 정밀 측정 "
        "필요(계획 §3/§7, Colab 측정 승격 검토 대상)"
    )
