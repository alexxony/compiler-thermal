"""P8 Task 36 잔여 — T1 로컬 정적 케이스 크기 감사 (A100 소비 0).

problems/<이름>/solve.py를 AST로 정적 파싱해 make_case/_make_inputs가 만드는
텐서들의 GPU 상주 바이트 총량을 추정한다. torch import 없음 — 순수 ast 기반.

판정 기준(앵커 실측 기반 재보정):
  33_BatchNorm(4.0GB 단일 텐서)는 실측 통과, 34_InstanceNorm/35_GroupNorm_
  (동일 shape 7.0GB 단일 텐서)은 CUDA OOM 확정("Tried to allocate 7.00 GiB",
  당시 세션 기점유 35.48GiB — A100 40GB 세션 누적 상태에서의 실패). 즉 7GB
  자체가 클린 GPU에서 불가능한 크기가 아니라, "이 케이스 크기가 배치 후반부
  세션 누적 상태에서 살아남을 수 있는가"가 실제 판별 기준이다. 30GB 같은
  절대 상한은 이 두 앵커(4GB vs 7GB)를 전혀 구분하지 못해 감사 무의미 —
  대신 두 앵커 사이의 실측 경계값을 직접 임계로 채택한다.

  단일 최대 텐서 바이트 기준 임계: OOM_THRESHOLD_GB = 5.0GB
  (33=4.0GB 미만 → PROCEED, 34/35/36/37=7.0GB 초과 → ISOLATE, 정확히 분리)

  이 임계는 "절대 GPU 용량"이 아니라 "배치 후반 세션에서 안전한 단일 케이스
  크기"의 경험적 근사 — 세션 누적 위험 프록시로 이해할 것.

검증 앵커(known-good/known-bad):
  33_BatchNorm       (64, 64, 512, 512)  = 4.0GB  — 실측 통과(정답)
  34_InstanceNorm    (112, 64, 512, 512) = 7.0GB  — CUDA OOM 확정("Tried to
                                                      allocate 7.00 GiB")
  35_GroupNorm_      (112, 64, 512, 512) = 7.0GB  — 34와 동일 death 시그니처
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PROBLEMS_DIR = REPO_ROOT / "problems"

FP32_BYTES = 4
GB = 1024 ** 3

# reference() 순전파 도중 입력 텐서 대비 추가로 상주하는 중간/출력 텐서 배수.
# 참고용으로만 계산·출력한다(판정에는 사용하지 않음 — 판정은 단일 최대 텐서
# 원본 바이트 기준, 앵커 경계값 재보정 근거는 모듈 docstring 참조).
REFERENCE_OVERHEAD_FACTOR = 2.0

# 33(4.0GB, PASS)과 34/35(7.0GB, OOM 확정) 사이의 실측 경계값 — 두 앵커를
# 정확히 분리하는 값으로 채택(절대 GPU 용량 상한이 아님, docstring 참조).
OOM_THRESHOLD_GB = 5.0

TENSOR_ALLOC_FUNCS = {"rand", "randn", "zeros", "ones", "empty"}

TARGET_PROBLEMS = [
    # 배치 2b
    "60_ConvTranspose3d_Swish_GroupNorm_HardSwish",
    "19_ConvTranspose2d_GELU_GroupNorm",
    "31_ELU",
    "75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__",
    "100_ConvTranspose3d_Clamp_Min_Divide",
    # 배치 3
    "15_Matmul_for_lower_triangular_matrices",
    "25_Swish",
    "79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__",
    "90_cumprod",
    "78_ConvTranspose3d_Max_Max_Sum",
    # 처분 대기
    "36_RMSNorm_",
    "37_FrobeniusNorm_",
]

ANCHOR_PROBLEMS = {
    "33_BatchNorm": "PASS(실측 통과)",
    "34_InstanceNorm": "OOM(확정, 7.00GiB 요청 실패)",
    "35_GroupNorm_": "OOM(확정, 34와 동일 시그니처)",
}


class ConstCollector(ast.NodeVisitor):
    """모듈 top-level 정수/실수 상수 할당을 수집."""

    def __init__(self):
        self.consts: dict[str, object] = {}
        self.tuple_consts: dict[str, tuple] = {}

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        val = self._const_value(stmt.value)
                        if val is not None:
                            self.consts[tgt.id] = val
                        elif isinstance(stmt.value, (ast.Tuple, ast.List)):
                            # 튜플/리스트 전체를 한 이름에 바인딩: input_shape = (32768,)
                            elems = []
                            ok = True
                            for elt in stmt.value.elts:
                                v = self._const_value(elt)
                                if v is None:
                                    ok = False
                                    break
                                elems.append(int(v))
                            if ok:
                                self.tuple_consts[tgt.id] = tuple(elems)
                    elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(
                        stmt.value, (ast.Tuple, ast.List)
                    ):
                        # 튜플/리스트 언패킹 할당: depth, height, width = 16, 32, 32
                        for name_node, val_node in zip(tgt.elts, stmt.value.elts):
                            if isinstance(name_node, ast.Name):
                                val = self._const_value(val_node)
                                if val is not None:
                                    self.consts[name_node.id] = val

    @staticmethod
    def _const_value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        return None


def _resolve_dim(node: ast.AST, consts: dict, local_vars: dict) -> "int | None":
    """dim 인자 AST 노드를 정수로 환원. 이름은 local_vars 우선, 없으면 모듈 상수."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in local_vars:
            return _resolve_dim(local_vars[node.id], consts, local_vars)
        if node.id in consts and isinstance(consts[node.id], int):
            return consts[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left = _resolve_dim(node.left, consts, local_vars)
        right = _resolve_dim(node.right, consts, local_vars)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
    return None


def _collect_local_assigns(func_node: ast.FunctionDef) -> dict:
    """함수 본문 내 단순 이름=값 할당(리스트/튜플 포함)을 수집 — dim 힌트 확장용."""
    local_vars: dict = {}
    for stmt in ast.walk(func_node):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    local_vars[tgt.id] = stmt.value
    return local_vars


def _tensor_bytes_from_call(call: ast.Call, consts: dict, local_vars: dict) -> "tuple[int, list] | None":
    """torch.rand/randn/zeros/ones/empty(...) 호출에서 shape을 뽑아 바이트 계산.

    반환: (총 바이트, 해석된 dim 리스트) 또는 해석 불가 시 None.
    """
    func = call.func
    fname = None
    if isinstance(func, ast.Attribute):
        fname = func.attr
    if fname not in TENSOR_ALLOC_FUNCS:
        return None

    # 위치 인자 중 정수 리터럴/이름만 shape dim으로 취급. device=/dtype=/generator= 등
    # 키워드 인자는 제외. *args 스프레드(예: torch.rand(batch_size, *input_shape))는
    # local_vars에서 튜플 리터럴을 찾아 展開.
    dims: list[int] = []
    unresolved = False
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            inner = arg.value
            if isinstance(inner, ast.Name) and inner.id in local_vars:
                tup_node = local_vars[inner.id]
                if isinstance(tup_node, (ast.Tuple, ast.List)):
                    for elt in tup_node.elts:
                        d = _resolve_dim(elt, consts, local_vars)
                        if d is None:
                            unresolved = True
                        else:
                            dims.append(d)
                    continue
            if isinstance(inner, ast.Name) and inner.id in consts.get("__tuple_consts__", {}):
                for d in consts["__tuple_consts__"][inner.id]:
                    dims.append(d)
                continue
            unresolved = True
            continue
        d = _resolve_dim(arg, consts, local_vars)
        if d is None:
            unresolved = True
        else:
            dims.append(d)

    if unresolved or not dims:
        return None

    total_elems = 1
    for d in dims:
        total_elems *= d
    return total_elems * FP32_BYTES, dims


def audit_problem(problem_dir: Path) -> dict:
    solve_path = problem_dir / "solve.py"
    result = {
        "problem": problem_dir.name,
        "solve_py_exists": solve_path.exists(),
        "tensors": [],
        "total_bytes": 0,
        "total_gb": 0.0,
        "estimated_with_overhead_gb": 0.0,
        "verdict": "UNKNOWN",
        "reason": "",
    }
    if not solve_path.exists():
        result["reason"] = "solve.py 없음"
        result["verdict"] = "SKIP"
        return result

    src = solve_path.read_text()
    try:
        tree = ast.parse(src, filename=str(solve_path))
    except SyntaxError as e:
        result["reason"] = f"AST 파싱 실패: {e}"
        result["verdict"] = "PARSE_ERROR"
        return result

    cc = ConstCollector()
    cc.visit(tree)
    consts = dict(cc.consts)
    consts["__tuple_consts__"] = cc.tuple_consts

    total_bytes = 0
    found_any = False
    unresolved_calls = 0

    # _make_inputs, _init_args가 참조하는 nn.Parameter(torch.randn(...)) 등도
    # 클래스 __init__ 내부에 있을 수 있어 함수 전체를 대상으로 훑는다.
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        if func.name not in ("_make_inputs", "__init__"):
            continue
        local_vars = _collect_local_assigns(func)
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                out = _tensor_bytes_from_call(node, consts, local_vars)
                if out is None:
                    fname = getattr(node.func, "attr", None)
                    if fname in TENSOR_ALLOC_FUNCS:
                        unresolved_calls += 1
                    continue
                nbytes, dims = out
                found_any = True
                total_bytes += nbytes
                result["tensors"].append({
                    "in_func": func.name,
                    "dims": dims,
                    "bytes": nbytes,
                    "gb": round(nbytes / GB, 4),
                })

    if not found_any:
        result["reason"] = (
            "텐서 할당 호출 해석 실패"
            + (f" (미해석 {unresolved_calls}건)" if unresolved_calls else " (없음)")
        )
        result["verdict"] = "PARSE_ERROR"
        return result

    result["total_bytes"] = total_bytes
    result["total_gb"] = round(total_bytes / GB, 4)
    # 참고용 수치(판정에는 미사용) — 순전파 중간/출력 텐서 포함 시 대략 규모.
    est_with_overhead = (total_bytes * REFERENCE_OVERHEAD_FACTOR) / GB
    result["estimated_with_overhead_gb"] = round(est_with_overhead, 4)

    # 판정은 단일 케이스 텐서 총합(overhead 미적용) 대 임계값 — 앵커 경계값
    # 재보정 근거(4.0GB PASS vs 7.0GB OOM)를 그대로 반영.
    if result["total_gb"] > OOM_THRESHOLD_GB:
        result["verdict"] = "ISOLATE"
        result["reason"] = (
            f"추정 상주 {result['total_gb']}GB > {OOM_THRESHOLD_GB}GB 임계"
            f"(참고: overhead 포함 시 ~{result['estimated_with_overhead_gb']}GB)"
        )
    else:
        result["verdict"] = "PROCEED"
        result["reason"] = (
            f"추정 상주 {result['total_gb']}GB <= {OOM_THRESHOLD_GB}GB 임계"
            f"(참고: overhead 포함 시 ~{result['estimated_with_overhead_gb']}GB)"
        )

    if unresolved_calls:
        result["reason"] += f" (주의: 미해석 할당 {unresolved_calls}건 존재, 과소추정 가능)"

    return result


def main():
    all_names = list(ANCHOR_PROBLEMS.keys()) + TARGET_PROBLEMS
    results = []
    for name in all_names:
        pdir = PROBLEMS_DIR / name
        r = audit_problem(pdir)
        r["is_anchor"] = name in ANCHOR_PROBLEMS
        if name in ANCHOR_PROBLEMS:
            r["anchor_expected"] = ANCHOR_PROBLEMS[name]
        results.append(r)

    print(f"{'문제':<70} {'판정':<10} {'추정GB':>8} {'+overheadGB':>12}  사유")
    print("-" * 140)
    for r in results:
        tag = " [앵커]" if r.get("is_anchor") else ""
        print(f"{r['problem']+tag:<70} {r['verdict']:<10} {r['total_gb']:>8} "
              f"{r['estimated_with_overhead_gb']:>12}  {r['reason']}")

    # 앵커 정합성 체크
    print()
    print("=== 앵커 정합성 검증 ===")
    anchor_ok = True
    for r in results:
        if not r.get("is_anchor"):
            continue
        expected = r["anchor_expected"]
        if "PASS" in expected and r["verdict"] != "PROCEED":
            print(f"불일치: {r['problem']} 기대={expected} 감사판정={r['verdict']}")
            anchor_ok = False
        elif "OOM" in expected and r["verdict"] != "ISOLATE":
            print(f"불일치: {r['problem']} 기대={expected} 감사판정={r['verdict']}")
            anchor_ok = False
        else:
            print(f"정합: {r['problem']} 기대={expected} 감사판정={r['verdict']}")

    out_path = REPO_ROOT / "artifacts" / "p8_case_size_audit.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "reference_overhead_factor": REFERENCE_OVERHEAD_FACTOR,
            "oom_threshold_gb": OOM_THRESHOLD_GB,
            "anchor_consistency_ok": anchor_ok,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    isolate = [r["problem"] for r in results if r["verdict"] == "ISOLATE" and not r.get("is_anchor")]
    proceed = [r["problem"] for r in results if r["verdict"] == "PROCEED" and not r.get("is_anchor")]
    parse_err = [r["problem"] for r in results if r["verdict"] == "PARSE_ERROR" and not r.get("is_anchor")]
    print(f"\n격리 대상({len(isolate)}): {isolate}")
    print(f"실측 진행({len(proceed)}): {proceed}")
    if parse_err:
        print(f"파싱 실패(수동 확인 필요, {len(parse_err)}): {parse_err}")

    return 0 if anchor_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
