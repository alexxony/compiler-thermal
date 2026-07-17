"""HBM_build P2 Task 5 테스트 — P4·P7 ΔT A/B 재계산 + r_hbm_sink 범위 민감도.

design: /mnt/c/ObsidianVault/HBM_build/docs/06-p2-rc-backport-design.md Task 5.
전제: HBM_build P2 T2 rc_params.csv 존재(T5 스코프 — 부재 시 관련 테스트 skip).
"""
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

import report_p2_t5_ab as t5  # noqa: E402
import report_p4_deltat as p4  # noqa: E402

HAS_RC_CSV = p4.HBM_RC_PARAMS_CSV.exists()
skip_if_no_csv = pytest.mark.skipif(
    not HAS_RC_CSV, reason="HBM_build T2 rc_params.csv 없음 — T5는 T2 완료 전제"
)


# --- r_hbm_sink 범위 로딩 ---

@skip_if_no_csv
def test_load_r_hbm_sink_range_matches_csv():
    r_min, r_max, r_rep = t5.load_r_hbm_sink_range()
    assert r_min == pytest.approx(0.929032, abs=1e-5)
    assert r_max == pytest.approx(4.670561, abs=1e-5)
    assert r_rep == pytest.approx(4.670561, abs=1e-5)  # 대표값=baseline_8hi=max


@skip_if_no_csv
def test_sweep_points_are_min_mid_representative():
    points = t5.r_hbm_sink_sweep_points()
    assert len(points) == 3
    labels = [p[0] for p in points]
    assert any("min" in l for l in labels)
    assert any("mid" in l for l in labels)
    assert any("대표" in l for l in labels)
    values = [p[1] for p in points]
    assert values == sorted(values)  # min < mid < 대표(max) 오름차순


def test_load_r_hbm_sink_range_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="HBM_build P2 T2"):
        t5.load_r_hbm_sink_range(missing)


# --- A/B 재계산 (요구사항 1) ---

@skip_if_no_csv
def test_ab_recompute_matmul_direction_unchanged():
    """matmul: LEGACY/HBM_FEM 둘 다 (b) TF32<fp32 — 세트 교체로 방향 안 뒤집힘."""
    cfg = next(c for c in p4.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res = t5.ab_recompute(cfg)
    assert not res["data_gap"]
    assert res["legacy"]["b_verdict_pass"] is True
    assert res["fem"]["b_verdict_pass"] is True
    # HBM_FEM 세트는 legacy와 다른(더 큰) gap을 낸다 — 실제로 파라미터가 반영됨.
    assert res["fem"]["b_gap_k"] != pytest.approx(res["legacy"]["b_gap_k"], abs=1e-9)


@skip_if_no_csv
def test_ab_recompute_all_four_problems_no_missing_config():
    """4문제(matmul/kb_matmul_scalar/batched_gemm/kb_softmax) 전부 config 존재
    — ab_recompute가 예외 없이(데이터 갭이든 정상이든) dict를 반환한다."""
    names = set()
    for cfg in p4.PROBLEM_CONFIGS:
        res = t5.ab_recompute(cfg)
        names.add(res["problem"])
        assert "data_gap" in res
    assert names == {"matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"}


@skip_if_no_csv
def test_ab_recompute_kb_softmax_is_null_both_sets():
    """kb_softmax는 null 대조군 — LEGACY/HBM_FEM 둘 다 is_null=True."""
    cfg = next(c for c in p4.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    res = t5.ab_recompute(cfg)
    assert not res["data_gap"]
    assert res["legacy"]["is_null"] is True
    assert res["fem"]["is_null"] is True


def test_ab_recompute_missing_data_reports_gap():
    """존재하지 않는 signals 경로 → 예외 없이 data_gap=True로 표식."""
    bogus_cfg = {"problem": "ghost_problem", "fmt": "p3_jsonl",
                 "path": "/nonexistent/path/does_not_exist.jsonl", "track": "off"}
    res = t5.ab_recompute(bogus_cfg)
    assert res["data_gap"] is True
    assert res["problem"] == "ghost_problem"


# --- r_hbm_sink 민감도 스윕 (요구사항 2) ---

@skip_if_no_csv
def test_sensitivity_sweep_matmul_three_points_no_flip():
    cfg = next(c for c in p4.PROBLEM_CONFIGS if c["problem"] == "matmul")
    res = t5.sensitivity_sweep(cfg)
    assert not res["data_gap"]
    assert not res["is_null"]
    assert len(res["points"]) == 3
    for pt in res["points"]:
        assert pt["direction_pass"] is True
    assert res["flipped"] is False


@skip_if_no_csv
def test_sensitivity_sweep_kb_softmax_null_skips_points():
    cfg = next(c for c in p4.PROBLEM_CONFIGS if c["problem"] == "kb_softmax")
    res = t5.sensitivity_sweep(cfg)
    assert not res["data_gap"]
    assert res["is_null"] is True
    assert res["points"] == []


@skip_if_no_csv
def test_sensitivity_sweep_gap_monotonic_with_r_hbm_sink():
    """r_hbm_sink가 클수록(단열 강할수록) gap도 커지는 물리적 단조성 확인
    (min < mid < 대표 순서로 gap도 증가해야 함 — HBM 저항 상승이 절대 ΔT를 늘림)."""
    cfg = next(c for c in p4.PROBLEM_CONFIGS if c["problem"] == "batched_gemm")
    res = t5.sensitivity_sweep(cfg)
    gaps = [pt["gap_k"] for pt in res["points"]]
    assert gaps == sorted(gaps)


# --- rank flip 판정 (요구사항 3) ---

@skip_if_no_csv
def test_rank_flip_check_no_flip_currently():
    ab_results = [t5.ab_recompute(cfg) for cfg in p4.PROBLEM_CONFIGS]
    rank = t5.rank_flip_check(ab_results)
    assert rank["rank_flipped"] is False
    assert rank["legacy_order"] == rank["fem_order"]
    assert set(rank["legacy_order"]) == {"matmul", "kb_matmul_scalar", "batched_gemm"}


# --- 리포트 생성 ---

@skip_if_no_csv
def test_build_markdown_report_has_all_sections():
    report = t5.build_markdown_report()
    assert "## 1. 문제별 LEGACY vs HBM_FEM" in report
    assert "## 2. r_hbm_sink 범위 민감도" in report
    assert "## 3. 문제 간 순위 뒤집힘" in report
    assert "## 4. 종합 해석" in report
    for name in ("matmul", "kb_matmul_scalar", "batched_gemm", "kb_softmax"):
        assert name in report


@skip_if_no_csv
def test_build_markdown_report_states_direction_unchanged_conclusion():
    """현재 데이터로는 flip이 없으므로 리포트가 '방향 불변' 결론을 명시해야 한다."""
    report = t5.build_markdown_report()
    assert "방향 불변" in report
