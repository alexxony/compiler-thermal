"""P9-A Task 40 — kb_convert 산출물 전수 docstring-코드 TF32 감사 테스트
(10-p9a-rule-coverage-design.md §2-4/§4 Task 40).

목적: solve.py(seed)와 variants/*.py가 실제로 "TF32 OFF/ON" 서술과 일치하는
allow_tf32 코드값을 갖는지 정적(ast) 대조. P5 TF32 버그(docstring "OFF" 서술인데
코드가 ON이던 사고) 동계열 재발을 39문제 전수로 재검사한다. torch 불요.

규약(kb_convert.py 확정 근거):
- solve.py(seed): matmul 사용 문제는 top-level "# seed = TF32 OFF (...)" 주석 +
  torch.backends.cuda.matmul.allow_tf32 = False 가 함께 나옴(같은 코드블록에서
  같이 방출됨 — 이 자체는 항상 일치, kb_convert.py:327-329). 감사 대상은
  "주석이 OFF라는데 실제 코드가 True"인 불일치.
- variants/R_tf32on.py, variants/R_tf32.py: seed의 False를 True로 뒤집는 파일
  (convert_to_tf32_variant_source). 파일명이 R_tf32*로 시작하면 반드시
  allow_tf32=True여야 함(seed와 달라야 "TF32 처방" 인과가 성립) — 불일치 시
  seed와 동일한 값(치환 실패/자기 치환)으로 감지.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from audit_kb_convert import (  # noqa: E402
    audit_seed_docstring,
    audit_variant_file,
    audit_problem,
    audit_all_problems,
    format_audit_report,
)


# ── audit_seed_docstring: solve.py 자체의 "주석 OFF" ↔ "코드 allow_tf32=False" 대조 ──

def test_audit_seed_docstring_consistent_matmul_problem():
    src = (
        "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.backends.cudnn.allow_tf32 = False\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    result = audit_seed_docstring(src)
    assert result.consistent is True
    assert result.reason == ""


def test_audit_seed_docstring_detects_mismatch_comment_off_code_true():
    # P5 동계열 사고 재현 fixture: 주석은 OFF라는데 코드는 True.
    src = (
        "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "torch.backends.cudnn.allow_tf32 = True\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    result = audit_seed_docstring(src)
    assert result.consistent is False
    assert "allow_tf32" in result.reason


def test_audit_seed_docstring_non_matmul_problem_no_tf32_lines_ok():
    # kb_softmax류 — TF32 주석/코드 자체가 없는 게 정상(matmul 미사용).
    src = (
        '"""KernelBench L1-23 Softmax."""\n'
        "def reference(case, device):\n    return case['X'].softmax(dim=-1)\n"
    )
    result = audit_seed_docstring(src)
    assert result.consistent is True


def test_audit_seed_docstring_comment_present_but_code_missing_flagged():
    # 주석만 있고 실제 allow_tf32 대입이 아예 없는 경우도 불일치.
    src = (
        "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    result = audit_seed_docstring(src)
    assert result.consistent is False


# ── audit_variant_file: 파일명(R_tf32*) 규약 ↔ 코드 실값 대조 ──

def test_audit_variant_file_tf32on_consistent_when_true():
    src = "torch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32 = True\n"
    result = audit_variant_file("R_tf32on.py", src)
    assert result.consistent is True


def test_audit_variant_file_tf32on_detects_mismatch_when_false():
    # 파일명은 ON을 약속하는데 실제로는 seed와 동일한 False — 치환 실패/자기 치환 감지.
    src = "torch.backends.cuda.matmul.allow_tf32 = False\ntorch.backends.cudnn.allow_tf32 = False\n"
    result = audit_variant_file("R_tf32on.py", src)
    assert result.consistent is False
    assert "allow_tf32" in result.reason


def test_audit_variant_file_tf32_scalar_name_also_requires_true():
    src = "torch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32 = True\n"
    result = audit_variant_file("R_tf32.py", src)
    assert result.consistent is True


def test_audit_variant_file_non_tf32_variant_skipped():
    # R_coalesced.py/R_fused.py 등은 TF32 규약 대상이 아님 — 항상 consistent(스킵).
    result = audit_variant_file("R_coalesced.py", "# no tf32 here\n")
    assert result.consistent is True


# ── audit_problem: 문제 1개(solve.py + variants/*) 전체 감사 ──

def test_audit_problem_all_consistent(tmp_path):
    prob = tmp_path / "fake_matmul"
    (prob / "variants").mkdir(parents=True)
    (prob / "solve.py").write_text(
        "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.backends.cudnn.allow_tf32 = False\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    (prob / "variants" / "R_tf32on.py").write_text(
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "torch.backends.cudnn.allow_tf32 = True\n"
    )
    result = audit_problem(prob)
    assert result.all_consistent is True
    assert result.mismatches == []


def test_audit_problem_detects_variant_mismatch(tmp_path):
    prob = tmp_path / "fake_matmul_bad"
    (prob / "variants").mkdir(parents=True)
    (prob / "solve.py").write_text(
        "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.backends.cudnn.allow_tf32 = False\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    (prob / "variants" / "R_tf32on.py").write_text(
        # 자기 치환 버그 재현: variant가 seed와 동일(False) — 1_Square 조사 오류가
        # 실제로 이랬다면 잡혔어야 할 케이스(§2-4 diff로 실제론 정상 확인됨).
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.backends.cudnn.allow_tf32 = False\n"
    )
    result = audit_problem(prob)
    assert result.all_consistent is False
    assert len(result.mismatches) == 1
    assert "R_tf32on.py" in result.mismatches[0]


def test_audit_problem_missing_solve_py_reports_gracefully(tmp_path):
    prob = tmp_path / "empty_dir"
    prob.mkdir()
    result = audit_problem(prob)
    assert result.all_consistent is False
    assert any("solve.py" in m for m in result.mismatches)


# ── audit_all_problems: 실 problems/ 39문제 전수 스캔 ──

def test_audit_all_problems_real_problems_dir_zero_mismatches():
    # 실측 회귀: 1_Square 오판(§2-4)이 있었으므로 전수 재검사가 필수.
    # 불일치 0이 기대값(팀리드 diff+git log 검증과 일치해야 함).
    report = audit_all_problems()
    assert report.total_problems >= 35, (
        f"문제 수가 예상보다 적음(35 P8 + 4 레거시 기대): {report.total_problems}")
    assert report.mismatches == [], f"TF32 docstring-코드 불일치 발견: {report.mismatches}"


def test_audit_all_problems_includes_1_square_and_is_consistent():
    # §2-4 재확인: 1_Square가 조사 오류(§4 p9a_cause_analysis)로 "동일 파일"
    # 오판됐던 문제 — 감사기가 이를 consistent로 정확히 판별해야 함(실제로는
    # variant가 True로 정상 반전되어 있음).
    report = audit_all_problems()
    square = [p for p in report.per_problem if "Square" in p]
    assert square, "1_Square 계열 문제가 감사 대상에 없음"
    for name in square:
        assert report.per_problem[name].all_consistent is True, (
            f"{name} 불일치 — 1_Square 미해명 해소(§2-4) 회귀 의심")


# ── format_audit_report: 사람이 읽는 요약 ──

def test_format_audit_report_shows_pass_when_zero_mismatches():
    report = audit_all_problems()
    text = format_audit_report(report)
    assert "불일치" in text
    if not report.mismatches:
        assert "0건" in text or "PASS" in text


def test_selfcheck_entrypoint_runs_clean(capsys):
    import audit_kb_convert
    rc = audit_kb_convert.main(["--selfcheck"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
