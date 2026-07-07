def split_power(total_power_w: float, dram_bytes: float, kernel_time_s: float,
                pj_per_bit: float) -> dict:
    """보드 총 전력을 HBM 몫과 die 몫으로 분해.

    P_hbm = traffic(bit) × E_bit / t. 추정치가 총 전력을 넘으면 클램프 —
    계수 불확실성에 대한 안전장치이며, 클램프 발생은 계수 재검토 신호.
    """
    p_hbm = dram_bytes * 8.0 * pj_per_bit * 1e-12 / kernel_time_s
    p_hbm = min(p_hbm, total_power_w)
    return {"p_hbm_w": p_hbm, "p_die_w": total_power_w - p_hbm}
