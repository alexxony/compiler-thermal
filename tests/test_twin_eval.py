import pandas as pd
import pytest
from thermal.twin_eval import RcBackend

# 60 s: die 시정수 ~6.7 s의 9배 — 포화 구간 확보
STEP = pd.DataFrame({
    "Time": [i * 0.1 for i in range(600)],
    "p_die_w": [0.0] * 20 + [150.0] * 580,
    "p_hbm_w": [0.0] * 20 + [30.0] * 580,
})


def make_backend(**kw):
    return RcBackend(
        r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
        c_die=50.0, c_hbm=10.0, t_ambient=45.0, **kw,
    )


def test_deterministic():
    a = make_backend().evaluate(STEP)
    b = make_backend().evaluate(STEP)
    pd.testing.assert_frame_equal(a, b)


def test_step_response_rises_and_saturates():
    out = make_backend().evaluate(STEP)
    assert {"Time", "t_die_c", "t_hbm_c"} <= set(out.columns)
    assert out["t_hbm_c"].iloc[0] == pytest.approx(45.0, abs=0.5)   # 초기 = ambient
    assert out["t_hbm_c"].iloc[-1] > out["t_hbm_c"].iloc[30]        # 상승
    tail = out["t_hbm_c"].iloc[-20:]
    assert tail.max() - tail.min() < 0.5                            # 포화
    assert out["t_hbm_c"].iloc[-1] > 45.0                           # 발열로 ambient 초과


def test_less_hbm_power_means_lower_hbm_temp():
    cool = STEP.copy()
    cool["p_hbm_w"] = cool["p_hbm_w"] * 0.5
    hot_t = make_backend().evaluate(STEP)["t_hbm_c"].iloc[-1]
    cool_t = make_backend().evaluate(cool)["t_hbm_c"].iloc[-1]
    assert cool_t < hot_t                                           # 연구 판정 방향성


def test_matches_twinbuilder_reference():
    """Ansys Twin Builder TR1 실측 CSV 대비 회귀 (G1 판정 문서 02 참조).

    조건: P_die=P_hbm=1 W 상수, Θ=45 ℃, 초기 293.0 K = 19.85 ℃ (역추론 검증 완료).
    """
    import pathlib
    ref = pd.read_csv(pathlib.Path(__file__).parent / "data" / "twinbuilder_tr1_ref.csv")
    power = pd.DataFrame({
        "Time": ref["Time [s]"],
        "p_die_w": 1.0,
        "p_hbm_w": 1.0,
    })
    out = make_backend(t_initial=293.0 - 273.15).evaluate(power)
    t_hbm_k = out["t_hbm_c"] + 273.15
    err = (t_hbm_k - ref["C_hbm.T [kel]"]).abs().max()
    assert err < 0.01, f"max |T_hbm err| = {err:.4f} K"
