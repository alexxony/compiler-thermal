"""P3 순간전력 → RcBackend 시계열 입력 변환 (P4 Task 11~12).

design: docs/04-p4-rc-deltat-design.md §2(입력 정의)/§3(시나리오 정의).
P3 ledger의 라운드 1건 = 순간전력 스칼라(p_die_w, p_hbm_w, kernel_time_s) —
RcBackend.evaluate()가 요구하는 Time 시계열로 바꾸는 어댑터. 물리 모델(RC) 자체는
무변경(thermal/twin_eval.py), 여기선 "무엇을 입력으로 넣을지"만 결정한다.
"""
from __future__ import annotations
import pandas as pd


def to_power_series(p_die_w: float, p_hbm_w: float, kernel_time_s: float,
                     duty_scenario: str, window_s: float,
                     idle_die_w: float, idle_hbm_w: float,
                     repeat_period_s: float | None = None,
                     dt_s: float = 0.05) -> pd.DataFrame:
    """P3 ledger 순간전력 스칼라 → RcBackend 입력 시계열.

    duty_scenario="saturated": window_s 내내 p_die_w/p_hbm_w 고정(설계 §3-(a)).
      "이 커널만 쉬지 않고 반복 실행"이라는 비현실적이지만 정직한 대조군.
    duty_scenario="iso_work": 반복주기(repeat_period_s) 중 kernel_time_s만큼만
      busy(p_die_w/p_hbm_w), 나머지는 idle_die_w/idle_hbm_w(설계 §3-(b)).
      repeat_period_s 필수(§3: "같은 처리량 기한을 공유한다고 가정" — 보통
      비교대상 fp32의 kernel_time_s를 반복주기로 씀, 호출자가 명시).
    """
    if duty_scenario not in ("saturated", "iso_work"):
        raise ValueError(f"알 수 없는 duty_scenario: {duty_scenario!r}")
    if duty_scenario == "iso_work" and repeat_period_s is None:
        raise ValueError("duty_scenario='iso_work'는 repeat_period_s가 필수(설계 §3-(b))")

    n = int(window_s / dt_s) + 1
    times = [i * dt_s for i in range(n)]

    if duty_scenario == "saturated":
        die = [p_die_w] * n
        hbm = [p_hbm_w] * n
    else:  # iso_work
        die, hbm = [], []
        for t in times:
            phase = t % repeat_period_s
            busy = phase < kernel_time_s
            die.append(p_die_w if busy else idle_die_w)
            hbm.append(p_hbm_w if busy else idle_hbm_w)

    return pd.DataFrame({"Time": times, "p_die_w": die, "p_hbm_w": hbm})


def duty_avg_power(duty: float, busy_power_w: float, idle_power_w: float) -> float:
    """duty-cycle 시간평균 전력 (설계 §3 공식).

    duty = 반복주기 중 busy 비율(0~1). duty=1.0이면 idle 영향 없이 busy_power_w
    그대로(fp32가 이번 P3 raw 데이터에서 이미 이 조건 — §3 "이번 P3 raw 데이터
    자체는 이미 duty=1.0 조건"과 일치하는 회귀 성질).
    """
    return duty * busy_power_w + (1.0 - duty) * idle_power_w
