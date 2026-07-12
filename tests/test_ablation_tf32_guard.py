"""P5 kb_matmul_scalar gate_fail(reference TF32 오염, max_err=3.00e-02, 2026-07-12) 재발
방지용 정적 체크 회귀 테스트. torch 불필요 — ast 소스 스캔만.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from run_ablation_remote import (  # noqa: E402
    _reference_uses_matmul,
    check_tf32_guard,
    check_all_tf32_guards,
)


def test_reference_uses_matmul_detects_at_operator():
    src = "def reference(case, device):\n    return case['A'] @ case['B']\n"
    assert _reference_uses_matmul(ast.parse(src)) is True


def test_reference_uses_matmul_detects_torch_matmul_call():
    src = "def reference(case, device):\n    return torch.matmul(case['A'], case['B'])\n"
    assert _reference_uses_matmul(ast.parse(src)) is True


def test_reference_uses_matmul_detects_bmm_and_mm():
    for call in ("torch.bmm(a, b)", "torch.mm(a, b)"):
        src = f"def reference(case, device):\n    return {call}\n"
        assert _reference_uses_matmul(ast.parse(src)) is True


def test_reference_uses_matmul_ignores_non_matmul():
    src = "def reference(case, device):\n    return case['X'].softmax(dim=-1)\n"
    assert _reference_uses_matmul(ast.parse(src)) is False


def test_check_tf32_guard_flags_missing_guard(tmp_path, monkeypatch):
    prob_dir = tmp_path / "fake_matmul"
    prob_dir.mkdir()
    (prob_dir / "solve.py").write_text(
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    monkeypatch.setattr("run_ablation_remote.PROBLEMS", tmp_path)
    warning = check_tf32_guard("fake_matmul")
    assert warning is not None
    assert "allow_tf32" in warning


def test_check_tf32_guard_passes_with_guard(tmp_path, monkeypatch):
    prob_dir = tmp_path / "fake_matmul_guarded"
    prob_dir.mkdir()
    (prob_dir / "solve.py").write_text(
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "def reference(case, device):\n    return case['A'] @ case['B']\n"
    )
    monkeypatch.setattr("run_ablation_remote.PROBLEMS", tmp_path)
    assert check_tf32_guard("fake_matmul_guarded") is None


def test_check_tf32_guard_skips_non_matmul_problem(tmp_path, monkeypatch):
    prob_dir = tmp_path / "fake_softmax"
    prob_dir.mkdir()
    (prob_dir / "solve.py").write_text(
        "def reference(case, device):\n    return case['X'].softmax(dim=-1)\n"
    )
    monkeypatch.setattr("run_ablation_remote.PROBLEMS", tmp_path)
    assert check_tf32_guard("fake_softmax") is None


def test_check_all_tf32_guards_on_real_problems_dir():
    # 실제 problems/ 전체가 통과해야 함 — kb_matmul_scalar 수정 반영 확인 포함.
    warnings = check_all_tf32_guards()
    assert warnings == [], f"TF32 가드 누락: {warnings}"
