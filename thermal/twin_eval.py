"""전력 시계열 → 온도 시계열.

주 백엔드 = RcBackend (자체 2노드 RC). G1 판정(02-g1-twin-export):
Student 라이선스에 .twin/FMU export 부재 → pytwin/fmpy 백엔드 불성립.
RcBackend는 Twin Builder TR solver 대비 max 오차 0.0006 K로 교차 검증됨
(tests/test_twin_eval.py::test_matches_twinbuilder_reference).

입력 df: Time, p_die_w, p_hbm_w / 출력 df: Time, t_die_c, t_hbm_c
"""
import pandas as pd


class RcBackend:
    def __init__(self, r_die_hbm, r_die_sink, r_hbm_sink, c_die, c_hbm,
                 t_ambient, t_initial=None):
        self.r_dh, self.r_ds, self.r_hs = r_die_hbm, r_die_sink, r_hbm_sink
        self.c_d, self.c_h, self.t_amb = c_die, c_hbm, t_ambient
        self.t_init = t_ambient if t_initial is None else t_initial

    def evaluate(self, power_df: pd.DataFrame) -> pd.DataFrame:
        t_d = t_h = self.t_init
        rows = []
        prev_time = float(power_df["Time"].iloc[0])
        for _, r in power_df.iterrows():
            dt = float(r["Time"]) - prev_time
            prev_time = float(r["Time"])
            if dt > 0:
                q_dh = (t_d - t_h) / self.r_dh
                dtd = (r["p_die_w"] - q_dh - (t_d - self.t_amb) / self.r_ds) / self.c_d
                dth = (r["p_hbm_w"] + q_dh - (t_h - self.t_amb) / self.r_hs) / self.c_h
                t_d += dtd * dt
                t_h += dth * dt
            rows.append({"Time": r["Time"], "t_die_c": t_d, "t_hbm_c": t_h})
        return pd.DataFrame(rows)
