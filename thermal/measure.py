"""변형 1개 측정 신호 병합: ncu 런 + 전력 런(분리 실행) → thermal signals dict."""
from thermal.chip_caps import CHIP_CAPS
from thermal.hbm_split import split_power


def merge_signals(chip: str, ncu: dict, power: dict) -> dict:
    caps = CHIP_CAPS[chip]
    dram_total = ncu["dram_bytes_read"] + ncu["dram_bytes_write"]
    split = split_power(
        total_power_w=power["avg_power_w"],
        dram_bytes=dram_total,
        kernel_time_s=ncu["kernel_time_s"],
        pj_per_bit=caps["pj_per_bit"],
    )
    # energy_j(전력 런 전체 창 적분)는 창 길이가 고정이라 반복 수를 안 나눠
    # 빠른 커널을 불리하게 판정한다(A100 TF32 실측서 발견) — 목적함수는 대신
    # 일(work) 단위로 정규화한 energy_per_iter_j를 쓴다: 총 전력 × 커널 1회
    # 실행시간(ncu 측정, 전력 런과 무관한 별도 실행). energy_j는 원시 데이터로 유지.
    energy_per_iter = power["avg_power_w"] * ncu["kernel_time_s"]
    return {
        "chip": chip,
        "dram_bytes_total": dram_total,
        "kernel_time_s": ncu["kernel_time_s"],
        "power_avg_w": power["avg_power_w"],
        "energy_j": power["energy_j"],
        "energy_per_iter_j": energy_per_iter,
        "p_hbm_w": split["p_hbm_w"],
        "p_die_w": split["p_die_w"],
    }
