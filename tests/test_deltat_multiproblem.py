"""P7 ΔT 축 다문제 연동 테스트 (07-p7-deltat-multiproblem-design §4 Task 25~29).

report_p4_deltat.py의 다문제 일반화 검증:
- Task 25: P6 `__ABL_RESULT__` 포맷 로더(`_load_round_from_ablresult`)
- Task 26: 문제 리스트 루프(config 테이블 구동)
- Task 27: kb_matmul_scalar ΔT (a)/(b) 방향 판정
- Task 28: null 대조군(seed=best) 처리
- Task 29: 문제별 idle sensitivity 3점

pandas 필요(P4/P5 관례 — scratch venv에서 실행). matmul 앵커 회귀(P4 §8)도 포함.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

import report_p4_deltat as rp  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
P6_JSON = ARTIFACTS / "p6_kbms_retry_20260713_result.json"
MATMUL_OFF = ARTIFACTS / "thermal-gain-matmul-off.jsonl"


# --- Task 25: P6 __ABL_RESULT__ 포맷 로더 ---

def test_load_ablresult_seed_extracts_first_round():
    """seed selector는 signals[0](최초 라운드)의 순간전력·kernel_time을 낸다."""
    sig = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "on", "seed")
    assert sig["p_die_w"] == pytest.approx(141.53, abs=0.01)
    assert sig["p_hbm_w"] == pytest.approx(0.2265, abs=0.001)
    assert sig["kernel_time_s"] == pytest.approx(319.3e-6, rel=1e-3)


def test_load_ablresult_best_is_min_energy_not_label():
    """best selector는 라벨이 아니라 energy_per_iter_j 최소 라운드(P6 §7-2)."""
    sig = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "on", "best")
    # ON best = r9: p_die 134.85W, kt 54.98µs, tensorcore_active
    assert sig["p_die_w"] == pytest.approx(134.85, abs=0.01)
    assert sig["p_hbm_w"] == pytest.approx(1.3235, abs=0.001)
    assert sig["kernel_time_s"] == pytest.approx(54.98e-6, rel=1e-2)


def test_load_ablresult_best_beats_seed_energy():
    """best의 energy_per_iter_j가 seed보다 작아야(선택 규약 정합)."""
    seed = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "on", "seed")
    best = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "on", "best")
    assert best["energy_per_iter_j"] < seed["energy_per_iter_j"]
    ratio = seed["energy_per_iter_j"] / best["energy_per_iter_j"]
    # 설계 §1-2/§2-1은 "5.86×"로 적었으나 그 인용값(0.04526/0.007486)조차 6.046×를
    # 낸다 — 5.86은 P6 라벨 이월 오기. 실 데이터 seed/best min-energy = 6.05×가 정답.
    assert ratio == pytest.approx(6.05, abs=0.05)


def test_load_ablresult_off_track_seed_equals_best():
    """off 트랙은 개선 라운드 없음 → seed=best(energy_per_iter_j 동률)."""
    seed = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "off", "seed")
    best = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "off", "best")
    # off r0가 min e/it(0.04390) — seed=best
    assert best["energy_per_iter_j"] == pytest.approx(seed["energy_per_iter_j"], rel=1e-9)


def test_load_ablresult_unknown_selector_raises():
    with pytest.raises(ValueError):
        rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "on", "bogus")


# --- Task 26: 문제 리스트 루프(config 구동) ---

def test_problem_config_table_has_matmul_and_kbms():
    """config 테이블에 matmul(P3 JSONL)·kb_matmul_scalar(P6 JSON) 둘 다 있다."""
    names = {c["problem"] for c in rp.PROBLEM_CONFIGS}
    assert "matmul" in names
    assert "kb_matmul_scalar" in names


def test_run_problem_returns_scenario_dict():
    """문제 1건 실행이 (a)/(b) t_hbm·ΔT·판정을 담은 dict를 낸다."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_matmul_scalar")
    res = rp.run_problem(cfg)
    assert res["problem"] == "kb_matmul_scalar"
    for key in ("a_seed_t", "a_best_t", "b_seed_t", "b_best_t",
                "b_verdict_pass", "is_null"):
        assert key in res


def test_report_string_has_per_problem_sections():
    """리포트 문자열에 문제명별 섹션이 나온다."""
    report = rp.build_report(rp.PROBLEM_CONFIGS)
    assert "matmul" in report
    assert "kb_matmul_scalar" in report


# --- Task 27: kb_matmul_scalar ΔT (a)/(b) 방향 판정 ---

def test_kbms_scenario_b_tf32_cooler():
    """(b) iso-work에서 best(TF32) t_hbm < seed(fp32) t_hbm (§5 방향 일치)."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_matmul_scalar")
    res = rp.run_problem(cfg)
    assert not res["is_null"]
    assert res["b_best_t"] < res["b_seed_t"], "energy 5.86×와 같은 방향이어야"
    assert res["b_verdict_pass"] is True


def test_kbms_scenario_a_direction():
    """(a) 포화에서 방향(순간전력 기반) — best p_die가 seed보다 낮아 (a) TF32 ≤ fp32.

    kb_matmul_scalar는 best p_die(134.85W)가 seed(141.53W)보다 낮으므로 (a)에서도
    TF32가 낮게 나올 수 있다. matmul(p_die 상승)과 달라 방향이 문제 의존 — 값만 기록.
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_matmul_scalar")
    res = rp.run_problem(cfg)
    # 단언은 (b)만(§5 1차 기준); (a)는 값 존재만 확인
    assert res["a_seed_t"] > rp.RC_KW["t_ambient"]
    assert res["a_best_t"] > rp.RC_KW["t_ambient"]


# --- matmul 앵커 회귀 (P4 §8) ---

def test_matmul_anchor_regression():
    """일반화 후에도 matmul (b) gap ≈ 17.16K, duty ≈ 15.66% 재현(P4 §8)."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res = rp.run_problem(cfg)
    gap = res["b_seed_t"] - res["b_best_t"]  # ΔT_fp32 - ΔT_TF32
    assert gap == pytest.approx(17.16, abs=0.5)
    assert res["b_best_t"] < res["b_seed_t"]  # (b) TF32 < fp32
    assert res["a_best_t"] >= res["a_seed_t"]  # (a) TF32 ≥ fp32


# --- Task 28: null 대조군 처리 ---

def test_null_control_seed_equals_best_labeled_null():
    """seed=best(개선 없음) → is_null=True, 방향 판정 skip."""
    seed = {"p_die_w": 140.0, "p_hbm_w": 0.5, "kernel_time_s": 300e-6,
            "energy_per_iter_j": 0.044, "hypothesis_label": "seed"}
    best = dict(seed)  # 동일 입력
    res = rp.run_problem_from_sigs("fake_null", seed, best)
    assert res["is_null"] is True
    assert res["b_verdict_pass"] is None  # 판정 skip


def test_null_control_report_prints_null_label():
    """null 문제는 리포트에 'null' 문구가 나온다."""
    seed = {"p_die_w": 140.0, "p_hbm_w": 0.5, "kernel_time_s": 300e-6,
            "energy_per_iter_j": 0.044, "hypothesis_label": "seed"}
    res = rp.run_problem_from_sigs("fake_null", seed, dict(seed))
    text = rp.format_problem_result(res)
    assert "null" in text.lower()


def test_kbms_off_track_is_null():
    """실 데이터: kb_matmul_scalar off 트랙(seed=best)은 null로 처리된다."""
    seed = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "off", "seed")
    best = rp._load_round_from_ablresult(P6_JSON, "kb_matmul_scalar", "off", "best")
    res = rp.run_problem_from_sigs("kb_matmul_scalar_off", seed, best)
    assert res["is_null"] is True


# --- Task 29: 문제별 idle sensitivity 3점 ---

def test_idle_sensitivity_direction_invariant_kbms():
    """kb_matmul_scalar: idle_die ×0.7/×1.0/×1.3 전부 (b) 방향 불변(best < seed)."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_matmul_scalar")
    points = rp.idle_sensitivity(cfg)
    assert len(points) == 3
    for pt in points:
        assert pt["b_best_t"] < pt["b_seed_t"], f"idle×{pt['idle_factor']}서 방향 뒤집힘"
    assert all(not pt["flipped"] for pt in points)


def test_idle_sensitivity_direction_invariant_matmul():
    """matmul(앵커): idle 3점 전부 (b) 방향 불변."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    points = rp.idle_sensitivity(cfg)
    assert len(points) == 3
    assert all(pt["b_best_t"] < pt["b_seed_t"] for pt in points)
    assert all(not pt["flipped"] for pt in points)


def test_idle_sensitivity_null_problem_skips():
    """null 문제는 sensitivity가 방향 판정 대신 null 표식만 낸다."""
    cfg = {"problem": "kb_matmul_scalar", "fmt": "ablresult", "path": str(P6_JSON),
           "track": "off"}  # off 트랙 = null
    points = rp.idle_sensitivity(cfg)
    assert all(pt.get("is_null") for pt in points)


# --- Task 30: batched_gemm / kb_softmax 재실측 통합 (§2 대안 A) ---

BGEMM_SOFTMAX_JSON = ARTIFACTS / "p7_bgemm_softmax_20260714_result.json"


def test_config_table_has_four_problems():
    """4문제 완전판 config — matmul/kb_matmul_scalar/batched_gemm/kb_softmax."""
    names = {c["problem"] for c in rp.PROBLEM_CONFIGS}
    assert names == {"matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"}


def test_batched_gemm_scenario_b_tf32_cooler():
    """batched_gemm (b): best(TF32) < seed(fp32) — matmul과 같은 방향(§1-2 예상).

    gap은 matmul(17.16K)보다 작아도 방향만 맞으면 성공(§5). energy ≈3.3×.
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "batched_gemm")
    res = rp.run_problem(cfg)
    assert not res["is_null"]
    assert res["b_best_t"] < res["b_seed_t"]
    assert res["b_verdict_pass"] is True
    # p_die가 seed<best로 상승(matmul형) → (a)에서 TF32 ≥ fp32
    assert res["a_tf32_ge"] is True


def test_kb_softmax_is_null_control():
    """kb_softmax: 개선 라운드 없음(seed=best) → Task 28 null 경로."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    res = rp.run_problem(cfg)
    assert res["is_null"] is True
    assert res["b_verdict_pass"] is None


def test_kb_softmax_hbm_power_dominant():
    """kb_softmax는 memory-bound — p_hbm_w가 처음으로 유의미(§1-2 예고).

    compute-bound(matmul 4.75W, kbms 0.23W)와 달리 kb_softmax p_hbm≈22.5W.
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    res = rp.run_problem(cfg)
    assert res["seed"]["p_hbm_w"] > 10.0


def test_batched_gemm_idle_sensitivity_invariant():
    """batched_gemm: idle 3점 전부 (b) 방향 불변(Task 29 편입)."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "batched_gemm")
    points = rp.idle_sensitivity(cfg)
    assert len(points) == 3
    assert all(pt["b_best_t"] < pt["b_seed_t"] for pt in points)
    assert all(not pt["flipped"] for pt in points)


def test_four_problem_report_has_all_sections():
    """4문제 완전판 리포트에 문제명 4개 전부 + kb_softmax null 문구."""
    report = rp.build_report(rp.PROBLEM_CONFIGS)
    for name in ("matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"):
        assert name in report
    assert "null" in report.lower()


# --- HBM_build P2 Task 4: rc_kw A/B 인터페이스 ------------------------------
# design: /mnt/c/ObsidianVault/HBM_build/docs/06-p2-rc-backport-design.md Task 4.
# RcBackend 클래스(thermal/twin_eval.py)는 무변경 — A/B는 이 모듈의 rc_kw 딕셔너리
# 선택으로만 구현. LEGACY는 P4 기준선(ΔT 17.16K)과 반드시 일치해야 한다(회귀 고정).


def test_rc_kw_legacy_is_default_and_matches_make_backend():
    """RC_KW_LEGACY가 test_twin_eval.make_backend()와 동일한 5개 값이어야 한다."""
    assert rp.RC_KW_LEGACY == dict(
        r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
        c_die=50.0, c_hbm=10.0, t_ambient=45.0,
    )
    # 하위호환 별칭 RC_KW도 동일 객체(기존 코드/테스트 무손상).
    assert rp.RC_KW is rp.RC_KW_LEGACY


def test_matmul_anchor_regression_explicit_legacy_set():
    """LEGACY 세트를 명시 전달해도 P4 기준선(ΔT 17.16K)이 그대로 재현된다.

    회귀 고정: run_problem(cfg)의 기본 rc_kw와 rc_kw=RC_KW_LEGACY 명시 호출이
    완전히 동일한 결과를 내야 한다(A/B 인터페이스 도입이 기본 경로를 바꾸지 않음).
    """
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res_default = rp.run_problem(cfg)
    res_explicit = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    assert res_default["b_seed_t"] == pytest.approx(res_explicit["b_seed_t"])
    assert res_default["b_best_t"] == pytest.approx(res_explicit["b_best_t"])
    gap = res_explicit["b_seed_t"] - res_explicit["b_best_t"]
    assert gap == pytest.approx(17.16, abs=0.5)
    assert res_explicit["rc_kw_label"] == "LEGACY"


def test_rc_kw_hbm_fem_die_side_unchanged_from_legacy():
    """RC_KW_HBM_FEM: die 쪽 3개(r_die_hbm/r_die_sink/c_die)는 legacy 값 유지.

    설계 §2 스코프 결정 — HBM_build FEM은 HBM 스택만 모델하므로 교체 가능한
    파라미터는 r_hbm_sink·c_hbm 2개뿐.
    """
    for key in ("r_die_hbm", "r_die_sink", "c_die", "t_ambient"):
        assert rp.RC_KW_HBM_FEM[key] == rp.RC_KW_LEGACY[key]


def test_rc_kw_hbm_fem_hbm_side_differs_from_legacy_or_is_placeholder():
    """RC_KW_HBM_FEM의 r_hbm_sink/c_hbm은 legacy와 다른 값(FEM 로드 성공)이거나,
    T2 미완료 시 placeholder(None)여야 한다 — 조용히 legacy로 값이 새면 안 된다.
    """
    r_hbm_sink = rp.RC_KW_HBM_FEM["r_hbm_sink"]
    c_hbm = rp.RC_KW_HBM_FEM["c_hbm"]
    if r_hbm_sink is None:
        assert c_hbm is None  # T2 대기 placeholder
        assert "PLACEHOLDER" in rp.RC_KW_HBM_FEM["_fem_source"]
    else:
        assert r_hbm_sink != rp.RC_KW_LEGACY["r_hbm_sink"]
        assert c_hbm != rp.RC_KW_LEGACY["c_hbm"]


def test_rc_kw_set_label_identifies_legacy_and_hbm_fem():
    """rc_kw_set_label이 두 표준 세트를 정확히 구분한다."""
    assert rp.rc_kw_set_label(rp.RC_KW_LEGACY) == "LEGACY"
    assert rp.rc_kw_set_label(rp.RC_KW_HBM_FEM) == "HBM_FEM"
    assert rp.rc_kw_set_label(dict(rp.RC_KW_LEGACY)) == "CUSTOM"  # 별개 dict


def test_load_hbm_fem_rc_params_missing_file_raises_clear_error(tmp_path):
    """CSV 부재 시 조용히 넘어가지 않고 FileNotFoundError로 명확히 실패한다."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="HBM_build P2 T2"):
        rp.load_hbm_fem_rc_params(missing)


def test_load_hbm_fem_rc_params_reads_real_csv_when_present():
    """T2 산출물 rc_params.csv가 있으면 r_hbm_sink·c_hbm을 실값으로 읽는다."""
    if not rp.HBM_RC_PARAMS_CSV.exists():
        pytest.skip("HBM_build T2 rc_params.csv 아직 없음 — placeholder 경로만 검증됨")
    params = rp.load_hbm_fem_rc_params()
    assert set(params) == {"r_hbm_sink", "c_hbm"}
    assert params["r_hbm_sink"] > 0
    assert params["c_hbm"] > 0


def test_run_problem_hbm_fem_set_produces_different_gap_from_legacy():
    """HBM_FEM 세트가 실값 로드에 성공한 경우 LEGACY와 다른 ΔT gap을 낸다
    (파라미터 교체가 실제로 계산에 반영됨을 확인 — A/B 대비의 최소 요건).
    """
    if rp.RC_KW_HBM_FEM["r_hbm_sink"] is None:
        pytest.skip("HBM_FEM 세트가 placeholder(T2 대기) — 대비 계산 skip")
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res_legacy = rp.run_problem(cfg, rc_kw=rp.RC_KW_LEGACY)
    res_fem = rp.run_problem(cfg, rc_kw=rp.RC_KW_HBM_FEM)
    gap_legacy = res_legacy["b_seed_t"] - res_legacy["b_best_t"]
    gap_fem = res_fem["b_seed_t"] - res_fem["b_best_t"]
    assert gap_fem != pytest.approx(gap_legacy, abs=1e-9)
    assert res_fem["rc_kw_label"] == "HBM_FEM"


def test_build_report_labels_rc_kw_set_in_header_and_sections():
    """리포트 헤더 + 문제별 섹션에 rc_kw 세트 라벨이 명시된다(Task 4 요구사항)."""
    cfg = next(c for c in rp.PROBLEM_CONFIGS if c["problem"] == "matmul")
    report_legacy = rp.build_report([cfg], rc_kw=rp.RC_KW_LEGACY)
    assert "rc_kw 세트: LEGACY" in report_legacy
    assert "[rc_kw=LEGACY]" in report_legacy

    report_fem = rp.build_report([cfg], rc_kw=rp.RC_KW_HBM_FEM)
    assert "rc_kw 세트: HBM_FEM" in report_fem
    assert "[rc_kw=HBM_FEM]" in report_fem
