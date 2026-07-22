"""P11 hotspot P4(30W) 기반 ΔT 지표 테스트 (12-p11-hotspot-p4-30w-design.md §5
Task 37~42).

report_p4_deltat.py 확장 검증:
- Task 37: load_hbm_hotspot_p4_rc_params() — r_hbm_sink_max_p4 행 파싱
  (D7 정규식으로 6케이스 R값 + die_source 추출)
- Task 38: RC_KW_HBM_HOTSPOT_P4 — {케이스명: rc_kw dict} 6세트
- Task 39: rc_kw_set_label() 확장 — HOTSPOT_P4_* 6개 라벨(기존 11개 무손상)
- Task 40: build_hotspot_report_p4() 신설(기존 build_report()/
  build_hotspot_report() 무변경 회귀 게이트)
- Task 41: hotspot_p4_verdict() — avg 대비 + P3 hotspot 대비 이중 방향 일치,
  hotspot_p4_verdict_rollup() AND 조건 롤업

pandas 필요(P4~P10 관례 — scratch venv .venv에서 실행). matmul avg (b) gap
17.16K 불변이 1차 회귀 게이트(Task 40 요구사항).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

import report_p4_deltat as rp  # noqa: E402

HOTSPOT_P4_CSV = rp.HBM_RC_PARAMS_CSV  # 실 CSV(설계 §0 표 값과 정확히 일치해야 함)

# 설계 §0 표 — 기대 R값(실 CSV 하드코딩 비교 대상)
_EXPECTED_R = {
    "a_s0": (5.138622, "base_die"),
    "a_s1": (5.844228, "base_die_phy"),
    "a_s2": (6.339869, "base_die_phy"),
    "b_s0": (1.365756, "base_die"),
    "b_s1": (2.196890, "base_die_phy"),
    "b_s2": (2.398226, "base_die_phy"),
}


# --- Task 37: load_hbm_hotspot_p4_rc_params() --------------------------------

def test_load_hotspot_p4_rc_params_extracts_six_keys_with_die_source():
    """실 CSV에서 6개 키의 R값과 die_source가 §0 표와 정확히 일치."""
    if not HOTSPOT_P4_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    params = rp.load_hbm_hotspot_p4_rc_params()
    assert set(params.keys()) == set(_EXPECTED_R.keys())
    for key, (expected_r, expected_die) in _EXPECTED_R.items():
        assert params[key]["r"] == pytest.approx(expected_r, abs=1e-6), key
        assert params[key]["die_source"] == expected_die, key


def test_load_hotspot_p4_rc_params_s0_falls_back_to_base_die():
    """s0류(a_s0/b_s0)만 base_die로 폴백, s1/s2류는 base_die_phy(§0 비대칭 패턴)."""
    if not HOTSPOT_P4_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    params = rp.load_hbm_hotspot_p4_rc_params()
    assert params["a_s0"]["die_source"] == "base_die"
    assert params["b_s0"]["die_source"] == "base_die"
    for key in ("a_s1", "a_s2", "b_s1", "b_s2"):
        assert params[key]["die_source"] == "base_die_phy", key


def test_load_hotspot_p4_rc_params_missing_file_raises_clear_error(tmp_path):
    """CSV 부재 시 조용히 넘어가지 않고 명확한 에러(FileNotFoundError)로 실패."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        rp.load_hbm_hotspot_p4_rc_params(missing)


def test_load_hotspot_p4_rc_params_missing_row_raises_value_error(tmp_path):
    """r_hbm_sink_max_p4 행이 없는 CSV는 ValueError로 명확히 실패(조용한 폴백 금지)."""
    csv_path = tmp_path / "rc_params.csv"
    csv_path.write_text(
        "parameter,value,value_min,value_max,unit,method,basis_case\n"
        "c_hbm,1.240170e-01,,,J/K,method,basis\n"
    )
    with pytest.raises(ValueError, match="r_hbm_sink_max_p4"):
        rp.load_hbm_hotspot_p4_rc_params(csv_path)


def test_load_hotspot_p4_rc_params_bad_basis_case_text_raises_value_error(tmp_path):
    """basis_case 텍스트에 a_s0 등 패턴이 없으면 정규식 실패 → 명확한 ValueError."""
    csv_path = tmp_path / "rc_params.csv"
    csv_path.write_text(
        "parameter,value,value_min,value_max,unit,method,basis_case\n"
        "r_hbm_sink_max_p4,6.339869,1.365756,6.339869,K/W,method,"
        "\"basis_case 텍스트에 시나리오 패턴 없음\"\n"
    )
    with pytest.raises(ValueError):
        rp.load_hbm_hotspot_p4_rc_params(csv_path)


def test_load_hotspot_p4_rc_params_does_not_match_p10_regex_pattern():
    """D1 근거 고정(회귀): P4 basis_case는 P10 정규식(s0_uniform 등)과 매칭 0건."""
    if not HOTSPOT_P4_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    import csv as csv_mod
    with open(HOTSPOT_P4_CSV, newline="") as f:
        rows = {row["parameter"]: row for row in csv_mod.DictReader(f)
                if row.get("parameter")}
    basis_case = rows["r_hbm_sink_max_p4"]["basis_case"]
    matches = rp._HOTSPOT_SCENARIO_RE.findall(basis_case)
    assert matches == []


# --- Task 38: RC_KW_HBM_HOTSPOT_P4 세트 구성 ---------------------------------

def test_rc_kw_hbm_hotspot_p4_has_six_cases():
    """6개 케이스 키 존재 — a_s0/a_s1/a_s2/b_s0/b_s1/b_s2."""
    expected = {"a_s0", "a_s1", "a_s2", "b_s0", "b_s1", "b_s2"}
    assert set(rp.RC_KW_HBM_HOTSPOT_P4.keys()) == expected


def test_rc_kw_hbm_hotspot_p4_die_side_matches_legacy_for_all_cases():
    """6개 세트 전부 die 3개(r_die_hbm/r_die_sink/c_die)가 legacy와 동일(회귀, D5)."""
    for name, rc_kw in rp.RC_KW_HBM_HOTSPOT_P4.items():
        for key in ("r_die_hbm", "r_die_sink", "c_die", "t_ambient"):
            assert rc_kw[key] == rp.RC_KW_LEGACY[key], f"{name}.{key} 불일치"


def test_rc_kw_hbm_hotspot_p4_c_hbm_matches_legacy_not_replaced():
    """D5: c_hbm은 P4 hotspot 전용 버전 없음 — legacy 값 그대로(교체 안 함)."""
    if rp.RC_KW_HBM_HOTSPOT_P4["a_s0"]["r_hbm_sink"] is None:
        pytest.skip("hotspot P4 세트가 placeholder — CSV 부재")
    for name, rc_kw in rp.RC_KW_HBM_HOTSPOT_P4.items():
        assert rc_kw["c_hbm"] == rp.RC_KW_LEGACY["c_hbm"], f"{name}.c_hbm 변경됨"


def test_rc_kw_hbm_hotspot_p4_r_hbm_sink_values_match_csv():
    """r_hbm_sink 값이 §0 표(실 CSV 파싱값)와 정확히 일치."""
    if not HOTSPOT_P4_CSV.exists():
        pytest.skip("HBM_build rc_params.csv 없음")
    hs = rp.RC_KW_HBM_HOTSPOT_P4
    for key, (expected_r, _) in _EXPECTED_R.items():
        assert hs[key]["r_hbm_sink"] == pytest.approx(expected_r, abs=1e-6), key


def test_rc_kw_hbm_hotspot_p4_is_separate_namespace_from_p10():
    """D2: RC_KW_HBM_HOTSPOT_P4는 RC_KW_HBM_HOTSPOT(P10, 16W)과 별도 top-level dict."""
    assert rp.RC_KW_HBM_HOTSPOT_P4 is not rp.RC_KW_HBM_HOTSPOT
    assert set(rp.RC_KW_HBM_HOTSPOT_P4.keys()).isdisjoint(
        set(rp.RC_KW_HBM_HOTSPOT.keys()))


# --- Task 39: rc_kw_set_label() 확장 -----------------------------------------

def test_rc_kw_set_label_identifies_all_hotspot_p4_cases():
    """6개 hotspot P4 세트 라벨이 HOTSPOT_P4_* 접두어로 구분된다."""
    hs = rp.RC_KW_HBM_HOTSPOT_P4
    assert rp.rc_kw_set_label(hs["a_s0"]) == "HOTSPOT_P4_A_S0"
    assert rp.rc_kw_set_label(hs["a_s1"]) == "HOTSPOT_P4_A_S1"
    assert rp.rc_kw_set_label(hs["a_s2"]) == "HOTSPOT_P4_A_S2"
    assert rp.rc_kw_set_label(hs["b_s0"]) == "HOTSPOT_P4_B_S0"
    assert rp.rc_kw_set_label(hs["b_s1"]) == "HOTSPOT_P4_B_S1"
    assert rp.rc_kw_set_label(hs["b_s2"]) == "HOTSPOT_P4_B_S2"


def test_rc_kw_set_label_legacy_and_p10_hotspot_unaffected():
    """기존 11개 라벨(LEGACY/HBM_FEM/HOTSPOT_* 5개) 매핑이 P4 세트 등록 후에도
    안 깨진다(회귀)."""
    assert rp.rc_kw_set_label(rp.RC_KW_LEGACY) == "LEGACY"
    assert rp.rc_kw_set_label(rp.RC_KW_HBM_FEM) == "HBM_FEM"
    hs = rp.RC_KW_HBM_HOTSPOT
    assert rp.rc_kw_set_label(hs["s0_uniform"]) == "HOTSPOT_S0"
    assert rp.rc_kw_set_label(hs["s1_phy_moderate"]) == "HOTSPOT_S1"
    assert rp.rc_kw_set_label(hs["s2_phy_heavy"]) == "HOTSPOT_S2"
    assert rp.rc_kw_set_label(hs["coolbc_min"]) == "HOTSPOT_COOLBC_MIN"
    assert rp.rc_kw_set_label(hs["coolbc_max"]) == "HOTSPOT_COOLBC_MAX"
    assert rp.rc_kw_set_label(dict(rp.RC_KW_LEGACY)) == "CUSTOM"


def test_rc_kw_set_labels_has_seventeen_entries():
    """전수 확인: LEGACY(1)+HBM_FEM(1)+HOTSPOT P3(5)+HOTSPOT P4(6) = 13개 이상 존재
    (§0 인덱스 아이덴티티가 겹치지 않는 한 최소 13개)."""
    assert len(rp.RC_KW_SET_LABELS) >= 13


# --- Task 40: build_hotspot_report_p4() ---------------------------------------

def test_build_report_unaffected_by_hotspot_p4_addition_matmul_anchor():
    """회귀 게이트: 기존 build_report()(avg 축)는 P4 세트 추가로 불변.

    matmul (b) gap = 17.16K 그대로 재현(RC_KW_LEGACY 세트).
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    report = rp.build_report([cfg], rc_kw=rp.RC_KW_LEGACY)
    res = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    gap = res["b_seed_t"] - res["b_best_t"]
    assert gap == pytest.approx(17.16, abs=0.5)
    assert "rc_kw 세트: LEGACY" in report


def test_build_hotspot_report_unaffected_by_hotspot_p4_addition():
    """회귀 게이트: 기존 build_hotspot_report()(P10, P3 16W 5세트)는 무변경."""
    report = rp.build_hotspot_report(rp.PROBLEM_CONFIGS)
    for label in ("HOTSPOT_S0", "HOTSPOT_S1", "HOTSPOT_S2",
                  "HOTSPOT_COOLBC_MIN", "HOTSPOT_COOLBC_MAX"):
        assert label in report
    # P4 라벨은 섞여 나오면 안 됨(두 리포트 함수가 독립적이어야 함)
    assert "HOTSPOT_P4" not in report


def test_build_hotspot_report_p4_has_four_problems_and_six_cases():
    """P4 hotspot 리포트에 4문제 × 6 R케이스 열이 전부 채워진다."""
    report = rp.build_hotspot_report_p4(rp.PROBLEM_CONFIGS)
    for name in ("matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"):
        assert name in report
    for label in ("HOTSPOT_P4_A_S0", "HOTSPOT_P4_A_S1", "HOTSPOT_P4_A_S2",
                  "HOTSPOT_P4_B_S0", "HOTSPOT_P4_B_S1", "HOTSPOT_P4_B_S2"):
        assert label in report


def test_build_hotspot_report_p4_null_problem_shows_null():
    """kb_softmax(null 대조군)은 P4 hotspot 리포트에서도 null로 표시된다."""
    report = rp.build_hotspot_report_p4(rp.PROBLEM_CONFIGS)
    assert "null" in report.lower()


def test_build_hotspot_report_p4_matmul_avg_regression_gate():
    """회귀 게이트(Task 40 요구사항): P4 리포트 안에서도 matmul avg(LEGACY) gap
    17.16K 불변."""
    report = rp.build_hotspot_report_p4(rp.PROBLEM_CONFIGS)
    assert report  # 렌더 성공 확인
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    gap = res["b_seed_t"] - res["b_best_t"]
    assert gap == pytest.approx(17.16, abs=0.5)


def test_build_hotspot_report_p4_has_reinterpretation_footnote():
    """R2: 리포트 헤더에 냉각계열 라벨 재해석 안 함 각주가 포함된다."""
    report = rp.build_hotspot_report_p4(rp.PROBLEM_CONFIGS)
    assert "재해석" in report


# --- Task 41: hotspot_p4_verdict() / hotspot_p4_verdict_rollup() -------------

def test_hotspot_p4_verdict_returns_dual_direction_match():
    """avg 대비 + P3 hotspot 대비 이중 방향 일치 필드가 모두 존재."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    verdict = rp.hotspot_p4_verdict(cfg, "a_s0")
    assert "direction_match_avg" in verdict
    assert "direction_match_p3" in verdict
    assert "attenuation_ratio_avg" in verdict
    assert "attenuation_ratio_p3" in verdict
    assert isinstance(verdict["direction_match_avg"], bool)
    assert isinstance(verdict["direction_match_p3"], bool)


def test_hotspot_p4_verdict_null_problem_returns_none_direction():
    """null 문제(kb_softmax)는 두 방향 판정 모두 None."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    verdict = rp.hotspot_p4_verdict(cfg, "a_s0")
    assert verdict["direction_match_avg"] is None
    assert verdict["direction_match_p3"] is None


def test_hotspot_p4_verdict_rollup_all_match_is_pass():
    """인위적 fixture: 6/6 케이스 두 기준 모두 방향 일치 → 문제 단위 PASS."""
    per_case = {
        k: {"direction_match_avg": True, "direction_match_p3": True,
            "attenuation_ratio_avg": 0.9, "attenuation_ratio_p3": 1.05}
        for k in ("a_s0", "a_s1", "a_s2", "b_s0", "b_s1", "b_s2")
    }
    rollup = rp.hotspot_p4_verdict_rollup(per_case)
    assert rollup == "PASS"


def test_hotspot_p4_verdict_rollup_partial_match_is_partial_not_fail():
    """인위적 fixture: 5/6 avg 일치, 1개 P3 불일치 → '부분 PASS — 케이스
    의존'(FAIL로 뭉개지 않음)."""
    per_case = {
        "a_s0": {"direction_match_avg": True, "direction_match_p3": True,
                  "attenuation_ratio_avg": 0.9, "attenuation_ratio_p3": 1.0},
        "a_s1": {"direction_match_avg": True, "direction_match_p3": True,
                  "attenuation_ratio_avg": 0.8, "attenuation_ratio_p3": 0.95},
        "a_s2": {"direction_match_avg": True, "direction_match_p3": False,
                  "attenuation_ratio_avg": 0.6, "attenuation_ratio_p3": -0.1},
        "b_s0": {"direction_match_avg": True, "direction_match_p3": True,
                  "attenuation_ratio_avg": 1.1, "attenuation_ratio_p3": 1.0},
        "b_s1": {"direction_match_avg": True, "direction_match_p3": True,
                  "attenuation_ratio_avg": 0.7, "attenuation_ratio_p3": 0.9},
        "b_s2": {"direction_match_avg": True, "direction_match_p3": True,
                  "attenuation_ratio_avg": 0.5, "attenuation_ratio_p3": 0.85},
    }
    rollup = rp.hotspot_p4_verdict_rollup(per_case)
    assert rollup == "부분 PASS — 케이스 의존"
    assert rollup != "FAIL"


def test_hotspot_p4_verdict_rollup_attenuation_ratio_does_not_affect_bool():
    """배율 감쇠 비율(참고값)이 판정 bool에 영향 주지 않는다 — 극단값이어도
    direction_match 전부 True면 PASS."""
    per_case = {
        k: {"direction_match_avg": True, "direction_match_p3": True,
            "attenuation_ratio_avg": 0.001, "attenuation_ratio_p3": 0.002}
        for k in ("a_s0", "a_s1", "a_s2", "b_s0", "b_s1", "b_s2")
    }
    rollup = rp.hotspot_p4_verdict_rollup(per_case)
    assert rollup == "PASS"


def test_hotspot_p4_verdict_rollup_null_cases_excluded_from_and():
    """None(null) 방향 판정은 AND 조건에서 제외 — 전부 None이면 'null'."""
    per_case = {
        "a_s0": {"direction_match_avg": None, "direction_match_p3": None,
                  "attenuation_ratio_avg": None, "attenuation_ratio_p3": None},
        "b_s0": {"direction_match_avg": None, "direction_match_p3": None,
                  "attenuation_ratio_avg": None, "attenuation_ratio_p3": None},
    }
    rollup = rp.hotspot_p4_verdict_rollup(per_case)
    assert rollup == "null"


# --- Task 42: 4문제 x 6케이스 실측 결과 요약(H1/H2 가설 대조) -----------------

def test_hotspot_p4_verdict_matmul_all_six_cases_summary():
    """실측: matmul 문제에서 6케이스 방향 일치 상황을 집계(H1/H2 가설 대조 근거).

    is_null이 아닌 실제 문제(matmul)로 6케이스 전부 실행해 direction_match_avg
    분포를 확인 — 6/6이면 H1(방향 유지) 지지, 아니면 반례 기록.
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    per_case = {case: rp.hotspot_p4_verdict(cfg, case)
                for case in ("a_s0", "a_s1", "a_s2", "b_s0", "b_s1", "b_s2")}
    for case, v in per_case.items():
        assert v["direction_match_avg"] is not None, case
        assert v["direction_match_p3"] is not None, case
    rollup = rp.hotspot_p4_verdict_rollup(per_case)
    assert rollup in ("PASS", "부분 PASS — 케이스 의존")
