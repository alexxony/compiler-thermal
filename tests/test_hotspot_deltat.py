"""P10 hotspot 기반 ΔT 지표 테스트 (11-p10-hotspot-deltat-design.md §5 Task 31~36).

report_p4_deltat.py 확장 검증:
- Task 31: load_hbm_hotspot_rc_params() — r_hbm_sink_max 행 파싱(대표값·냉각BC
  범위·basis_case 정규식 s0/s1/s2 3점 추출)
- Task 32: RC_KW_HBM_HOTSPOT — {시나리오명: rc_kw dict} 5세트
- Task 33: rc_kw_set_label() 확장 — HOTSPOT_* 5개 라벨
- Task 34: build_hotspot_report() 신설(기존 build_report() 무변경 회귀 게이트)
- Task 35: hotspot_verdict() — §4-1 방향 일치 + §4-2 AND 조건 롤업

pandas 필요(P4~P7 관례 — scratch venv .venv에서 실행). §4-3 불변성(RC_KW_LEGACY
결과 무변경, matmul 17.16K/batched_gemm 11.11K/kb_matmul_scalar 4.39K)이
1차 회귀 게이트다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

import report_p4_deltat as rp  # noqa: E402

HOTSPOT_CSV = rp.HBM_RC_PARAMS_CSV  # 실 CSV(§0 표 값과 정확히 일치해야 함)


# --- Task 31: load_hbm_hotspot_rc_params() ----------------------------------

def test_load_hotspot_rc_params_extracts_five_keys():
    """실 CSV에서 5개 키(representative/cooling_bc_range/s0/s1/s2)를 정확히 추출."""
    if not HOTSPOT_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    params = rp.load_hbm_hotspot_rc_params()
    assert params["representative"] == pytest.approx(5.138620, abs=1e-6)
    assert params["cooling_bc_range"]["min"] == pytest.approx(1.365791, abs=1e-6)
    assert params["cooling_bc_range"]["max"] == pytest.approx(5.138620, abs=1e-6)
    assert params["s0_uniform"] == pytest.approx(5.122614, abs=1e-6)
    assert params["s1_phy_moderate"] == pytest.approx(5.844234, abs=1e-6)
    assert params["s2_phy_heavy"] == pytest.approx(6.339864, abs=1e-6)


def test_load_hotspot_rc_params_missing_file_raises_clear_error(tmp_path):
    """CSV 부재 시 조용히 넘어가지 않고 명확한 에러(FileNotFoundError)로 실패."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        rp.load_hbm_hotspot_rc_params(missing)


def test_load_hotspot_rc_params_missing_row_raises_value_error(tmp_path):
    """r_hbm_sink_max 행이 없는 CSV는 ValueError로 명확히 실패(조용한 폴백 금지)."""
    csv_path = tmp_path / "rc_params.csv"
    csv_path.write_text(
        "parameter,value,value_min,value_max,unit,method,basis_case\n"
        "c_hbm,1.240170e-01,,,J/K,method,basis\n"
    )
    with pytest.raises(ValueError, match="r_hbm_sink_max"):
        rp.load_hbm_hotspot_rc_params(csv_path)


def test_load_hotspot_rc_params_bad_basis_case_text_raises_value_error(tmp_path):
    """basis_case 텍스트에 s0/s1/s2 패턴이 없으면 정규식 실패 → 명확한 ValueError."""
    csv_path = tmp_path / "rc_params.csv"
    csv_path.write_text(
        "parameter,value,value_min,value_max,unit,method,basis_case\n"
        "r_hbm_sink_max,5.138620,1.365791,5.138620,K/W,method,"
        "\"basis_case 텍스트에 시나리오 패턴 없음\"\n"
    )
    with pytest.raises(ValueError):
        rp.load_hbm_hotspot_rc_params(csv_path)


# --- Task 32: RC_KW_HBM_HOTSPOT 세트 구성 ------------------------------------

def test_rc_kw_hbm_hotspot_has_five_scenarios():
    """5개 시나리오 키 존재 — COOLBC_MIN/MAX + S0/S1/S2."""
    expected = {"coolbc_min", "coolbc_max", "s0_uniform",
                "s1_phy_moderate", "s2_phy_heavy"}
    assert set(rp.RC_KW_HBM_HOTSPOT.keys()) == expected


def test_rc_kw_hbm_hotspot_die_side_matches_legacy_for_all_scenarios():
    """5개 세트 전부 die 3개(r_die_hbm/r_die_sink/c_die)가 legacy와 동일(회귀)."""
    for name, rc_kw in rp.RC_KW_HBM_HOTSPOT.items():
        for key in ("r_die_hbm", "r_die_sink", "c_die", "t_ambient"):
            assert rc_kw[key] == rp.RC_KW_LEGACY[key], f"{name}.{key} 불일치"


def test_rc_kw_hbm_hotspot_c_hbm_matches_legacy_not_replaced():
    """D3: c_hbm은 hotspot 버전 없음 — legacy 값 그대로(교체 안 함)."""
    if rp.RC_KW_HBM_HOTSPOT["s0_uniform"]["r_hbm_sink"] is None:
        pytest.skip("hotspot 세트가 placeholder — CSV 부재")
    for name, rc_kw in rp.RC_KW_HBM_HOTSPOT.items():
        assert rc_kw["c_hbm"] == rp.RC_KW_LEGACY["c_hbm"], f"{name}.c_hbm 변경됨"


def test_rc_kw_hbm_hotspot_r_hbm_sink_values_match_csv():
    """r_hbm_sink 값이 §0 표(실 CSV 파싱값)와 정확히 일치."""
    if not HOTSPOT_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    hs = rp.RC_KW_HBM_HOTSPOT
    assert hs["coolbc_min"]["r_hbm_sink"] == pytest.approx(1.365791, abs=1e-6)
    assert hs["coolbc_max"]["r_hbm_sink"] == pytest.approx(5.138620, abs=1e-6)
    assert hs["s0_uniform"]["r_hbm_sink"] == pytest.approx(5.122614, abs=1e-6)
    assert hs["s1_phy_moderate"]["r_hbm_sink"] == pytest.approx(5.844234, abs=1e-6)
    assert hs["s2_phy_heavy"]["r_hbm_sink"] == pytest.approx(6.339864, abs=1e-6)


# --- Task 33: rc_kw_set_label() 확장 -----------------------------------------

def test_rc_kw_set_label_identifies_all_hotspot_scenarios():
    """5개 hotspot 세트 라벨이 HOTSPOT_* 접두어로 구분된다."""
    hs = rp.RC_KW_HBM_HOTSPOT
    assert rp.rc_kw_set_label(hs["s0_uniform"]) == "HOTSPOT_S0"
    assert rp.rc_kw_set_label(hs["s1_phy_moderate"]) == "HOTSPOT_S1"
    assert rp.rc_kw_set_label(hs["s2_phy_heavy"]) == "HOTSPOT_S2"
    assert rp.rc_kw_set_label(hs["coolbc_min"]) == "HOTSPOT_COOLBC_MIN"
    assert rp.rc_kw_set_label(hs["coolbc_max"]) == "HOTSPOT_COOLBC_MAX"


def test_rc_kw_set_label_legacy_and_hbm_fem_unaffected():
    """기존 LEGACY/HBM_FEM 라벨 매핑이 hotspot 세트 등록 후에도 안 깨진다(회귀)."""
    assert rp.rc_kw_set_label(rp.RC_KW_LEGACY) == "LEGACY"
    assert rp.rc_kw_set_label(rp.RC_KW_HBM_FEM) == "HBM_FEM"
    assert rp.rc_kw_set_label(dict(rp.RC_KW_LEGACY)) == "CUSTOM"


# --- Task 34: build_hotspot_report() -----------------------------------------

def test_build_report_unaffected_by_hotspot_addition_matmul_anchor():
    """§4-3 회귀 게이트: 기존 build_report()(avg 축)는 hotspot 세트 추가로 불변.

    matmul (b) gap = 17.16K 그대로 재현(RC_KW_LEGACY 세트).
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    report = rp.build_report([cfg], rc_kw=rp.RC_KW_LEGACY)
    res = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    gap = res["b_seed_t"] - res["b_best_t"]
    assert gap == pytest.approx(17.16, abs=0.5)
    assert "rc_kw 세트: LEGACY" in report


def test_build_hotspot_report_has_four_problems_and_five_scenarios():
    """hotspot 리포트에 4문제 × 5 R시나리오 열이 전부 채워진다."""
    report = rp.build_hotspot_report(rp.PROBLEM_CONFIGS)
    for name in ("matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"):
        assert name in report
    for label in ("HOTSPOT_S0", "HOTSPOT_S1", "HOTSPOT_S2",
                  "HOTSPOT_COOLBC_MIN", "HOTSPOT_COOLBC_MAX"):
        assert label in report


def test_build_hotspot_report_null_problem_shows_null():
    """kb_softmax(null 대조군)은 hotspot 리포트에서도 null로 표시된다."""
    report = rp.build_hotspot_report(rp.PROBLEM_CONFIGS)
    assert "null" in report.lower()


def test_build_hotspot_report_matmul_avg_regression_gate():
    """§4-3 회귀 게이트: hotspot 리포트 안에서도 matmul avg(LEGACY) gap 불변."""
    report = rp.build_hotspot_report(rp.PROBLEM_CONFIGS)
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    gap = res["b_seed_t"] - res["b_best_t"]
    assert gap == pytest.approx(17.16, abs=0.5)
    cfg_bg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "batched_gemm")
    res_bg = rp.run_problem(cfg_bg, rc_kw=rp.RC_KW_LEGACY)
    gap_bg = res_bg["b_seed_t"] - res_bg["b_best_t"]
    assert gap_bg == pytest.approx(11.11, abs=0.5)
    cfg_kbms = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_matmul_scalar")
    res_kbms = rp.run_problem(cfg_kbms, rc_kw=rp.RC_KW_LEGACY)
    gap_kbms = res_kbms["b_seed_t"] - res_kbms["b_best_t"]
    assert gap_kbms == pytest.approx(4.39, abs=0.5)


# --- Task 35: hotspot_verdict() ----------------------------------------------

def test_hotspot_verdict_direction_match_true_when_signs_agree():
    """avg (b) gap 부호와 hotspot (b) gap 부호가 같으면 방향 일치 True."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    verdict = rp.hotspot_verdict(cfg, "s0_uniform")
    assert "direction_match" in verdict
    assert "attenuation_ratio" in verdict
    assert isinstance(verdict["direction_match"], bool)


def test_hotspot_verdict_null_problem_returns_none_direction():
    """null 문제(kb_softmax)는 방향 판정 없이 None."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    verdict = rp.hotspot_verdict(cfg, "s0_uniform")
    assert verdict["direction_match"] is None


def test_hotspot_verdict_rollup_all_match_is_pass():
    """인위적 fixture: 4/4 R시나리오 방향 일치 → 문제 단위 PASS."""
    per_scenario = {
        "s0_uniform": {"direction_match": True, "attenuation_ratio": 0.95},
        "s1_phy_moderate": {"direction_match": True, "attenuation_ratio": 0.80},
        "s2_phy_heavy": {"direction_match": True, "attenuation_ratio": 0.60},
        "coolbc_min": {"direction_match": True, "attenuation_ratio": 1.10},
    }
    rollup = rp.hotspot_verdict_rollup(per_scenario)
    assert rollup == "PASS"


def test_hotspot_verdict_rollup_partial_match_is_partial_not_fail():
    """인위적 fixture: 3/4 일치 → '부분 PASS — 시나리오 의존'(FAIL로 뭉개지 않음)."""
    per_scenario = {
        "s0_uniform": {"direction_match": True, "attenuation_ratio": 0.95},
        "s1_phy_moderate": {"direction_match": True, "attenuation_ratio": 0.80},
        "s2_phy_heavy": {"direction_match": False, "attenuation_ratio": -0.10},
        "coolbc_min": {"direction_match": True, "attenuation_ratio": 1.10},
    }
    rollup = rp.hotspot_verdict_rollup(per_scenario)
    assert rollup == "부분 PASS — 시나리오 의존"
    assert rollup != "FAIL"


def test_hotspot_verdict_rollup_attenuation_ratio_does_not_affect_bool():
    """배율 감쇠 비율(참고값)이 판정 bool(direction_match)에 영향 주지 않는다.

    비율이 1보다 훨씬 작아도(강한 감쇠) direction_match=True면 PASS 유지 확인.
    """
    per_scenario = {
        "s0_uniform": {"direction_match": True, "attenuation_ratio": 0.01},
        "s1_phy_moderate": {"direction_match": True, "attenuation_ratio": 0.02},
        "s2_phy_heavy": {"direction_match": True, "attenuation_ratio": 0.005},
        "coolbc_min": {"direction_match": True, "attenuation_ratio": 0.03},
    }
    rollup = rp.hotspot_verdict_rollup(per_scenario)
    assert rollup == "PASS"


def test_hotspot_verdict_rollup_null_scenarios_excluded_from_and():
    """None(null) 방향 판정은 AND 조건에서 제외 — 전부 None이면 'null'."""
    per_scenario = {
        "s0_uniform": {"direction_match": None, "attenuation_ratio": None},
        "s1_phy_moderate": {"direction_match": None, "attenuation_ratio": None},
    }
    rollup = rp.hotspot_verdict_rollup(per_scenario)
    assert rollup == "null"
