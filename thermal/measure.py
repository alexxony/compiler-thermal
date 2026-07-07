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
    return {
        "chip": chip,
        "dram_bytes_total": dram_total,
        "kernel_time_s": ncu["kernel_time_s"],
        "power_avg_w": power["avg_power_w"],
        "energy_j": power["energy_j"],
        "p_hbm_w": split["p_hbm_w"],
        "p_die_w": split["p_die_w"],
    }
