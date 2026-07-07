from thermal.measure import merge_signals


def test_merge_signals():
    sig = merge_signals(
        chip="NVIDIA A100-SXM4-40GB",
        ncu={"dram_bytes_read": 6e8, "dram_bytes_write": 4e8, "kernel_time_s": 1e-3},
        power={"avg_power_w": 200.0, "energy_j": 0.2},
    )
    assert sig["dram_bytes_total"] == 1e9
    assert sig["p_hbm_w"] > 0
    assert sig["p_die_w"] == sig["power_avg_w"] - sig["p_hbm_w"]
    assert sig["energy_j"] == 0.2
