import pytest
from thermal.hbm_split import split_power
from thermal.chip_caps import CHIP_CAPS


def test_split_basic():
    # 1e9 bytes * 8 bit * 4e-12 J/bit = 0.032 J → /1e-3 s = 32 W
    r = split_power(total_power_w=200.0, dram_bytes=1e9, kernel_time_s=1e-3, pj_per_bit=4.0)
    assert r["p_hbm_w"] == pytest.approx(32.0)
    assert r["p_die_w"] == pytest.approx(168.0)


def test_split_clamps_to_total():
    # 추정 HBM 전력이 총 전력 초과 시 총 전력으로 클램프 (분해 모델 한계 안전장치)
    r = split_power(total_power_w=50.0, dram_bytes=1e12, kernel_time_s=1e-3, pj_per_bit=4.0)
    assert r["p_hbm_w"] == 50.0
    assert r["p_die_w"] == 0.0


def test_chip_caps_has_required_keys():
    for chip, caps in CHIP_CAPS.items():
        assert {"tdp_w", "pj_per_bit", "mem_type"} <= set(caps), chip
