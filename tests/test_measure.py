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


def test_merge_signals_energy_per_iter():
    """energy_per_iter_j = power_avg_w × kernel_time_s(ncu 단일 실행) —
    energy_j(전력 런 고정 창 전체 적분)와 독립. 커널이 빨라지면 kernel_time_s가
    줄어 energy_per_iter_j도 같이 줄어야(정규화 성립) — energy_j는 창이
    고정이라 이 관계를 못 보임(A100 TF32 실측서 발견 버그의 회귀 테스트).
    """
    slow = merge_signals(
        chip="NVIDIA A100-SXM4-40GB",
        ncu={"dram_bytes_read": 6e8, "dram_bytes_write": 4e8, "kernel_time_s": 9.46e-3},
        power={"avg_power_w": 309.0, "energy_j": 927.0},  # 3.0s 고정창 적분(원시)
    )
    fast = merge_signals(
        chip="NVIDIA A100-SXM4-40GB",
        ncu={"dram_bytes_read": 6e8, "dram_bytes_write": 4e8, "kernel_time_s": 1.48e-3},
        power={"avg_power_w": 355.0, "energy_j": 1065.0},  # 같은 고정창, 순간전력만 더 높음
    )
    assert slow["energy_per_iter_j"] == 309.0 * 9.46e-3
    assert fast["energy_per_iter_j"] == 355.0 * 1.48e-3
    # 핵심 회귀: 순간전력이 더 높아도(355>309) 커널이 6배 이상 빨라지면
    # energy_per_iter_j는 줄어야 한다(일 단위 정규화가 실제로 작동함을 검증).
    assert fast["energy_per_iter_j"] < slow["energy_per_iter_j"]
    # energy_j(원시, 고정창)는 반대로 fast가 더 큼(=버그였던 그 방향) — 원시 필드는
    # 여전히 유지되지만 목적함수(energy_per_iter_j)는 그 아티팩트에서 자유로움.
    assert fast["energy_j"] > slow["energy_j"]
