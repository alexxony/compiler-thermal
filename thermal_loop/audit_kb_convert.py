"""P9-A Task 40 — kb_convert 산출물 전수 docstring-코드 TF32 감사
(10-p9a-rule-coverage-design.md §2-4/§4 Task 40, T-seed).

목적: 39문제(35 P8 + 4 레거시) 전체 solve.py(seed) + variants/*.py의 TF32
docstring/파일명 서술이 실제 코드 `allow_tf32` 값과 일치하는지 정적(ast) 대조.
P5 kb_matmul_scalar 사고(docstring/의도가 OFF라는데 실행 시 TF32에 오염된 사고,
`check_tf32_guard`가 잡는 것과는 다른 각도 — 이건 "서술 vs 코드" 일치 전수 검사)와
[[p9a_cause_analysis]] §4의 1_Square "동일 파일" 오판(§2-4에서 diff로 기각 확정)
동계열 재발을 막는다. torch 불요, 순수 소스 스캔.

규약 (kb_convert.py 확정 근거 — 10-p9a-rule-coverage-design.md §2-4):
- solve.py(seed): matmul 사용 문제는 top-level 주석
  "# seed = TF32 OFF (...)" + `torch.backends.cuda.matmul.allow_tf32 = False`가
  같은 코드블록에서 함께 방출된다(kb_convert.py:327-329). 감사 대상은 "주석은
  OFF라는데 실제 대입값이 True(또는 주석만 있고 대입 자체가 없음)"인 불일치.
  주석도 대입도 없는 비-matmul 문제(kb_softmax 등)는 정상(해당 없음).
- variants/R_tf32on.py, variants/R_tf32.py: seed의 False를 True로 뒤집는
  파일(convert_to_tf32_variant_source) — 파일명이 R_tf32로 시작하면 반드시
  allow_tf32=True여야 함. False로 남아있으면 치환 실패/자기 치환(1_Square가
  조사 오류로 이렇게 오판됐던 케이스, §2-4)을 코드로 검출한다.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path

from run_e2e import PROBLEMS


@dataclass
class TF32AuditResult:
    consistent: bool
    reason: str = ""


def _find_allow_tf32_assignments(tree: ast.AST) -> list[bool]:
    """모듈 top-level(들여쓰기 없는) `torch.backends.{cuda.matmul,cudnn}.allow_tf32 = <bool>`
    대입값 목록. reference() 내부의 지역 방어(prev 저장/finally 원복 패턴)는
    함수 본문 안이라 ast.Module.body 순회에 안 잡힘 — 의도적으로 top-level만 본다
    (kb_convert.py convert_to_tf32_variant_source가 앵커로 삼는 것과 동일 범위).
    """
    vals: list[bool] = []
    for node in tree.body:  # top-level 문만 (함수 내부 제외)
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0] if node.targets else None
        if not isinstance(target, ast.Attribute) or target.attr != "allow_tf32":
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            vals.append(node.value.value)
    return vals


def _has_seed_off_comment(src: str) -> bool:
    return any("seed = TF32 OFF" in line for line in src.splitlines()
               if line.strip().startswith("#"))


def audit_seed_docstring(src: str) -> TF32AuditResult:
    """solve.py(seed) 소스 1개 — "# seed = TF32 OFF" 주석 ↔ allow_tf32 코드값 대조.

    주석 없음 + 대입 없음 = 비-matmul 문제, 정상(consistent). 주석 있는데
    대입이 없거나 대입값이 True면 불일치(P5 동계열 사고).
    """
    tree = ast.parse(src)
    has_comment = _has_seed_off_comment(src)
    vals = _find_allow_tf32_assignments(tree)

    if not has_comment:
        return TF32AuditResult(consistent=True)  # 비-matmul 문제, 서술 자체가 없음

    if not vals:
        return TF32AuditResult(
            consistent=False,
            reason="주석 'seed = TF32 OFF'는 있으나 allow_tf32 대입이 코드에 없음")

    if any(v is not False for v in vals):
        return TF32AuditResult(
            consistent=False,
            reason=f"주석은 'TF32 OFF'인데 allow_tf32 대입값에 True 포함: {vals}")

    return TF32AuditResult(consistent=True)


# TF32 variant로 취급할 파일명 규약 (design §2-0 name_map과 동일 범위,
# _variant_map_for가 인식하는 R_tf32*만 이 규약 대상 — 다른 variant는 스킵).
_TF32_VARIANT_PREFIXES = ("R_tf32on.py", "R_tf32.py")


def audit_variant_file(filename: str, src: str, seed_uses_top_level_tf32: bool = True
                       ) -> TF32AuditResult:
    """variants/<filename> — 파일명이 TF32 계열이면 allow_tf32=True(top-level)를 요구.

    R_coalesced.py/R_fused.py 등 TF32 무관 variant는 이 규약 대상이 아니므로
    항상 consistent(스킵). seed_uses_top_level_tf32=False(예: kb_matmul_scalar —
    module top-level에 allow_tf32 대입이 아예 없고 run_solve() 내부에서 지역
    prev/finally 패턴으로만 토글하는 레거시 hand-written 문제)면 top-level 검사
    자체가 부적용이라 스킵 — kb_convert 자동변환 규약(top-level 대입 반전) 대상이
    아닌 문제에 그 규약을 강제하면 오탐(kb_matmul_scalar/R_tf32.py가 대표 사례,
    §4 Task 40 실측에서 발견).
    """
    if filename not in _TF32_VARIANT_PREFIXES:
        return TF32AuditResult(consistent=True)
    if not seed_uses_top_level_tf32:
        return TF32AuditResult(consistent=True)  # top-level 규약 비대상 문제, 스킵

    tree = ast.parse(src)
    vals = _find_allow_tf32_assignments(tree)
    if not vals:
        return TF32AuditResult(
            consistent=False,
            reason=f"{filename}: TF32 variant인데 top-level allow_tf32 대입이 코드에 없음")
    if any(v is not True for v in vals):
        return TF32AuditResult(
            consistent=False,
            reason=f"{filename}: TF32 variant(파일명상 ON 약속)인데 allow_tf32 대입값에 "
                   f"False 포함(치환 실패/자기 치환 의심): {vals}")
    return TF32AuditResult(consistent=True)


@dataclass
class ProblemAuditResult:
    problem: str
    all_consistent: bool
    mismatches: list[str] = field(default_factory=list)


def audit_problem(prob_dir: Path) -> ProblemAuditResult:
    """문제 1개 디렉토리 — solve.py + variants/*.py 전체 감사."""
    name = prob_dir.name
    mismatches: list[str] = []

    solve_path = prob_dir / "solve.py"
    if not solve_path.exists():
        return ProblemAuditResult(
            problem=name, all_consistent=False,
            mismatches=[f"{name}: solve.py 없음"])

    seed_src = solve_path.read_text()
    seed_result = audit_seed_docstring(seed_src)
    if not seed_result.consistent:
        mismatches.append(f"{name}/solve.py: {seed_result.reason}")

    # seed 자체가 module top-level allow_tf32를 쓰는 문제만 kb_convert의 top-level
    # 반전 규약(convert_to_tf32_variant_source) 대상 — kb_matmul_scalar류(레거시,
    # run_solve() 내부 지역 prev/finally 토글만 사용)는 top-level 대입이 없으므로
    # 이 규약 비대상(§4 Task 40 실측에서 발견, audit_variant_file 스킵 조건 참조).
    seed_uses_top_level_tf32 = bool(_find_allow_tf32_assignments(ast.parse(seed_src)))

    vdir = prob_dir / "variants"
    if vdir.exists():
        for vf in sorted(vdir.iterdir()):
            if not vf.is_file() or vf.suffix != ".py":
                continue
            vresult = audit_variant_file(vf.name, vf.read_text(), seed_uses_top_level_tf32)
            if not vresult.consistent:
                mismatches.append(f"{name}/variants/{vf.name}: {vresult.reason}")

    return ProblemAuditResult(
        problem=name, all_consistent=(len(mismatches) == 0), mismatches=mismatches)


@dataclass
class AuditReport:
    total_problems: int
    per_problem: dict[str, ProblemAuditResult]
    mismatches: list[str] = field(default_factory=list)


def audit_all_problems() -> AuditReport:
    """problems/ 전체(39문제) 전수 스캔."""
    per_problem: dict[str, ProblemAuditResult] = {}
    all_mismatches: list[str] = []
    for d in sorted(p for p in PROBLEMS.iterdir() if p.is_dir()):
        r = audit_problem(d)
        per_problem[d.name] = r
        all_mismatches.extend(r.mismatches)
    return AuditReport(
        total_problems=len(per_problem), per_problem=per_problem,
        mismatches=all_mismatches)


def format_audit_report(report: AuditReport) -> str:
    lines = []
    lines.append("=== P9-A Task 40 — kb_convert TF32 docstring-코드 감사 ===")
    lines.append(f"전수 스캔: {report.total_problems}문제")
    if not report.mismatches:
        lines.append("불일치: 0건 (PASS)")
    else:
        lines.append(f"불일치: {len(report.mismatches)}건")
        for m in report.mismatches:
            lines.append(f"  - {m}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if "--selfcheck" in argv:
        # (a) 일치 fixture
        ok = audit_seed_docstring(
            "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
            "torch.backends.cuda.matmul.allow_tf32 = False\n")
        assert ok.consistent is True, ok

        # (b) 불일치 fixture — P5 동계열 사고 재현(주석 OFF, 코드 True)
        bad = audit_seed_docstring(
            "# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).\n"
            "torch.backends.cuda.matmul.allow_tf32 = True\n")
        assert bad.consistent is False, bad

        # (c) variant 규약 — R_tf32on.py는 True 필수
        vok = audit_variant_file(
            "R_tf32on.py", "torch.backends.cuda.matmul.allow_tf32 = True\n")
        assert vok.consistent is True, vok
        vbad = audit_variant_file(
            "R_tf32on.py", "torch.backends.cuda.matmul.allow_tf32 = False\n")
        assert vbad.consistent is False, vbad

        # (d) 실 problems/ 전수 스캔 — 은폐 금지, 불일치 있으면 그대로 노출
        report = audit_all_problems()
        print(format_audit_report(report))
        assert report.total_problems >= 35, report.total_problems

        print("audit_kb_convert.py self-check PASS")
        return 0

    report = audit_all_problems()
    print(format_audit_report(report))
    return 0 if not report.mismatches else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
