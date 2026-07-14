"""P8 Task 31 — KernelBench `Model` 포맷 → 자체 `problems/<name>/solve.py` 변환기.

정책 경계(08-p8-scale-ablation-design.md §2 전체 관통): AutoKernel/KernelBench는
**문제(reference 사양)만 흡수**. bench_kb.py/scorer.py/orchestrate.py/verify.py
(판정·최적화 엔진) 어느 것도 import·이식하지 않는다. 이 파일은 KernelBench 원본
`.py`(`Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`)를 읽어 자체
executor 계약(make_case/run_solve/reference/GATE_SIZES/PROFILE_SIZE)으로 **텍스트
변환**할 뿐 — KernelBench의 판정 로직은 참조하지 않는다.

golden 기준(§2-3): 기존 kb_matmul_scalar/kb_softmax가 이 변환의 작동 전례.
변환된 solve.py는 그 두 파일과 동일한 함수 시그니처 계약을 만족해야 한다.

torch 불요 — 이 모듈 자체는 ast/텍스트 처리만 한다(로컬에 torch 없음, P8 실측
전 로컬 selfcheck 전제). 변환된 solve.py는 torch를 import하지만 그건 원격(Colab)
실행 시점 얘기다.
"""
from __future__ import annotations
import ast
import re
from dataclasses import dataclass, field


@dataclass
class KBProblem:
    """KernelBench 원본 .py에서 뽑은 구조화 정보."""
    name: str
    forward_src: str            # Model.forward 본문 소스(텍스트, 들여쓰기 제거)
    forward_args: list[str]     # forward(self, A, B) → ["A", "B"]
    init_args: list[str]        # __init__(self, in_features, ...) → 파라미터명
    init_body_src: str          # Model.__init__ 본문 소스(nn.Linear 등 계층 선언 — 분류용)
    init_inputs_src: str        # get_init_inputs() 리턴 표현식 소스 텍스트
    get_inputs_src: str         # get_inputs() 리턴 표현식 소스 텍스트
    module_body_src: str        # Model/get_inputs/get_init_inputs 제외한 모듈 top-level 소스(상수 등)
    stateful: bool              # get_init_inputs()가 빈 리스트가 아님 = 학습 파라미터 있는 nn.Module
    docstring: str = ""


def _get_func(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _get_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _src_of(src: str, node: ast.AST) -> str:
    return ast.get_source_segment(src, node) or ""


def parse_kb_source(src: str, name: str) -> KBProblem:
    """KernelBench 원본 .py 텍스트를 KBProblem으로 파싱.

    KernelBench 계약(bridge.py 로더가 요구하는 것과 동일, §2-3 표 — 로더 로직은
    참고만, 엔진 미이식): class Model(nn.Module) + forward() + get_inputs() +
    get_init_inputs().
    """
    tree = ast.parse(src, filename=f"<kb:{name}>")
    model_cls = _get_class(tree, "Model")
    if model_cls is None:
        raise ValueError(f"{name}: class Model 없음 — KernelBench 계약 위반")

    forward_fn = None
    init_fn = None
    for item in model_cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            forward_fn = item
        elif isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_fn = item
    if forward_fn is None:
        raise ValueError(f"{name}: Model.forward 없음")

    forward_args = [a.arg for a in forward_fn.args.args if a.arg != "self"]
    init_args = [a.arg for a in init_fn.args.args if a.arg != "self"] if init_fn else []
    init_body_src = "\n".join(_src_of(src, n) for n in init_fn.body) if init_fn else ""

    # forward 본문(반환 표현식 등) — docstring 제외한 statement 텍스트.
    body_stmts = [n for n in forward_fn.body
                  if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str))]
    forward_src = "\n".join(_src_of(src, n) for n in body_stmts)

    get_inputs_fn = _get_func(tree, "get_inputs")
    get_init_fn = _get_func(tree, "get_init_inputs")
    get_inputs_src = _return_expr_src(src, get_inputs_fn) if get_inputs_fn else "[]"
    init_inputs_src = _return_expr_src(src, get_init_fn) if get_init_fn else "[]"

    docstring = ast.get_docstring(model_cls) or ""

    # 모듈 top-level 상수(N, batch_size 등) — Model/get_inputs/get_init_inputs
    # 제외한 나머지 Assign 문. 변환 solve.py의 make_case에서 재사용.
    excluded = {id(model_cls)}
    if get_inputs_fn:
        excluded.add(id(get_inputs_fn))
    if get_init_fn:
        excluded.add(id(get_init_fn))
    const_stmts = [n for n in tree.body
                   if id(n) not in excluded
                   and not isinstance(n, (ast.Import, ast.ImportFrom))]
    module_body_src = "\n".join(_src_of(src, n) for n in const_stmts)

    stateful = init_inputs_src.strip() not in ("[]", "()")

    return KBProblem(
        name=name, forward_src=forward_src, forward_args=forward_args,
        init_args=init_args, init_body_src=init_body_src,
        init_inputs_src=init_inputs_src,
        get_inputs_src=get_inputs_src, module_body_src=module_body_src,
        stateful=stateful, docstring=docstring.strip(),
    )


def _return_expr_src(src: str, fn: ast.FunctionDef) -> str:
    for n in fn.body:
        if isinstance(n, ast.Return) and n.value is not None:
            return _src_of(src, n.value)
    return "[]"


# ── §3-2 4버킷 분류 — bridge.py analyze의 op 패턴 매칭 방식 참고, 판정은 자체. ──
_MATMUL_PAT = re.compile(r"\b(matmul|torch\.mm\(|torch\.bmm\(|nn\.Linear\b|@\s*\w)")
_CONV_PAT = re.compile(r"\bnn\.Conv[123]d\b|\bconv[123]d\(")
_NORM_PAT = re.compile(r"\b(norm|LayerNorm|BatchNorm|InstanceNorm|GroupNorm|RMSNorm)\b", re.I)
_REDUCE_ELEM_PAT = re.compile(
    r"\b(sum|mean|max|min|softmax|sigmoid|tanh|gelu|selu|hardtanh|"
    r"cumsum|reduce)\b|relu|\belu\b", re.I)


def op_pattern_bucket(kb: KBProblem) -> str:
    """§3-2 4버킷: compute-matmul / compute-conv-fusion / memory-norm /
    memory-reduce-elementwise. 소스 텍스트 op 패턴 매칭(자체 판정 — bridge.py의
    analyze 방식은 참고했으나 로직 비이식, 독립 재구현).
    """
    text = (kb.forward_src + " " + kb.init_body_src + " "
            + kb.module_body_src + " " + kb.docstring)
    has_matmul = bool(_MATMUL_PAT.search(text))
    has_conv = bool(_CONV_PAT.search(text))
    has_norm = bool(_NORM_PAT.search(text))
    has_reduce_elem = bool(_REDUCE_ELEM_PAT.search(text))

    if has_matmul and not has_conv and not (has_norm or has_reduce_elem):
        return "compute-matmul"
    if has_matmul or has_conv:
        # matmul/conv가 다른 연산과 융합되어 있으면 fusion 버킷 (Level 2 다수 케이스)
        return "compute-conv-fusion"
    if has_norm:
        return "memory-norm"
    return "memory-reduce-elementwise"


def classify_workload(kb: KBProblem) -> str:
    """워크로드 클래스 판정 — op_pattern_bucket의 공개 별칭(§1-2 compute/memory 클래스 매핑)."""
    return op_pattern_bucket(kb)


_COMPUTE_BUCKETS = {"compute-matmul", "compute-conv-fusion"}


def is_compute_bound(bucket: str) -> bool:
    return bucket in _COMPUTE_BUCKETS


def variant_map_for_bucket(bucket: str) -> set[str]:
    """run_ablation_remote._variant_map_for 명명 규약과 정합(§4-2 "이미 대규모에
    버티는 부분" — 그 함수 자체는 무변경, 여기선 변환 시점에 어떤 variant 파일을
    생성해야 하는지만 결정).

    matmul형(순수 matmul, fusion 아님) = R_tf32on.py만(TF32가 유일 처방).
    그 외(fusion/memory) = R_tf32.py + R_coalesced.py(kb_matmul_scalar golden과 동일 패턴).
    """
    if bucket == "compute-matmul":
        return {"R_tf32on.py"}
    return {"R_tf32.py", "R_coalesced.py"}


def convert_to_solve_source(kb: KBProblem, problem_name: str) -> str:
    """KBProblem → 자체 solve.py 텍스트.

    executor 계약(thermal_loop/executor.py): make_case(size, device) -> case,
    run_solve(case, device) -> Tensor, reference(case, device) -> Tensor,
    GATE_SIZES, PROFILE_SIZE(옵션 GATE_ATOL/RTOL).

    matmul 계열은 reference()에 allow_tf32 지역 방어를 자동 주입(P5 kb_matmul_scalar
    gate_fail 재발 방지, run_ablation_remote.check_tf32_guard가 이걸 정적으로 검증).
    stateful(nn.Linear 등 학습 파라미터 보유) 모델은 make_case에서 torch.manual_seed로
    고정한 뒤 Model을 인스턴스화 — 매 호출 동일 가중치 재현(결정론, correctness gate 전제).
    """
    bucket = op_pattern_bucket(kb)
    uses_matmul = bucket in ("compute-matmul", "compute-conv-fusion")

    forward_body = kb.forward_src or "return x"
    fwd_args = kb.forward_args or ["x"]
    call_args = ", ".join(f"case[{a!r}]" for a in fwd_args)

    lines: list[str] = []
    lines.append(f'"""KernelBench {kb.name} — P8 Task 31 자동 변환 ({bucket}).')
    if kb.docstring:
        lines.append(kb.docstring)
    lines.append('KernelBench Model→solve.py 변환(kb_convert.py). 엔진 미접촉 — 문제 사양만 흡수.')
    lines.append('executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.')
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("import argparse")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("")
    lines.append("GATE_ATOL = 6e-2" if uses_matmul else "GATE_ATOL = 1e-4")
    lines.append("GATE_RTOL = 6e-2" if uses_matmul else "GATE_RTOL = 1e-4")
    lines.append("GATE_SIZES = (16, 64, 256)")
    lines.append("PROFILE_SIZE = 512")
    lines.append("")
    if uses_matmul:
        lines.append("# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).")
        lines.append("torch.backends.cuda.matmul.allow_tf32 = False")
        lines.append("torch.backends.cudnn.allow_tf32 = False")
        lines.append("")

    # 원본 KernelBench Model 클래스 그대로 보존 — forward()가 이걸 감싸는 형태.
    lines.append("class _KBModel(nn.Module):")
    if kb.init_args:
        init_sig = ", ".join(kb.init_args)
        lines.append(f"    def __init__(self, {init_sig}):")
    else:
        lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    if kb.stateful:
        # 원본 __init__ 본문을 그대로 재현할 정보가 없으므로(생성 시점엔 없음),
        # 변환 시 실제로는 원본 __init__ 소스를 그대로 이식해야 함 — 이 함수는
        # 골격만 만들고 상세 __init__ 이식은 호출측(convert_problem_file)이 채운다.
        lines.append("        pass  # __init__ body injected by convert_problem_file")
    else:
        lines.append("        pass")
    fwd_sig = ", ".join(["self"] + fwd_args)
    lines.append(f"    def forward({fwd_sig}):")
    for stmt in forward_body.splitlines() or ["return x"]:
        lines.append(f"        {stmt}")
    lines.append("")

    if kb.stateful:
        lines.append("_SEED = 0")
        lines.append("")
        lines.append("def _init_args():")
        lines.append(f"    return {kb.init_inputs_src}")
        lines.append("")
        lines.append("def _make_model(device):")
        lines.append("    torch.manual_seed(_SEED)  # 결정론 가중치 재현")
        lines.append("    m = _KBModel(*_init_args()).to(device)")
        lines.append("    m.eval()")
        lines.append("    return m")
        lines.append("")

    lines.append("def _make_inputs(device):")
    lines.append("    torch.manual_seed(_SEED)" if kb.stateful else "    torch.manual_seed(0)")
    lines.append(f"    vals = {kb.get_inputs_src}")
    lines.append("    return [v.to(device) if hasattr(v, 'to') else v for v in vals]")
    lines.append("")

    lines.append("def make_case(size, device):")
    lines.append("    # KernelBench get_inputs()는 고정 크기 — size는 게이트/프로파일 스윕용 스케일 힌트.")
    lines.append("    xs = _make_inputs(device)")
    case_dict = ", ".join(f"{a!r}: xs[{i}]" for i, a in enumerate(fwd_args))
    lines.append(f"    case = {{{case_dict}}}")
    if kb.stateful:
        lines.append("    case['_model'] = _make_model(device)")
    lines.append("    return case")
    lines.append("")

    lines.append("def run_solve(case, device):")
    if kb.stateful:
        lines.append("    model = case['_model']")
        lines.append(f"    return model({call_args})")
    else:
        lines.append("    model = _KBModel()")
        lines.append(f"    return model({call_args})")
    lines.append("")

    lines.append("def reference(case, device):")
    if uses_matmul:
        lines.append("    prev = torch.backends.cuda.matmul.allow_tf32")
        lines.append("    torch.backends.cuda.matmul.allow_tf32 = False")
        lines.append("    try:")
        lines.append("        return run_solve(case, device)")
        lines.append("    finally:")
        lines.append("        torch.backends.cuda.matmul.allow_tf32 = prev")
    else:
        lines.append("    return run_solve(case, device)")
    lines.append("")

    lines.append('if __name__ == "__main__":')
    lines.append("    ap = argparse.ArgumentParser()")
    lines.append('    ap.add_argument("--check", action="store_true")')
    lines.append('    ap.add_argument("--profile", action="store_true")')
    lines.append("    args = ap.parse_args()")
    lines.append('    dev = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("    case = make_case(PROFILE_SIZE, dev)")
    lines.append("    if args.profile:")
    lines.append("        run_solve(case, dev)")
    lines.append('        if dev == "cuda":')
    lines.append("            torch.cuda.synchronize()")
    lines.append("    else:")
    lines.append("        out = run_solve(case, dev)")
    lines.append("        ref = reference(case, dev)")
    lines.append("        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)")
    lines.append('        print("PASS" if ok else "FAIL")')
    lines.append("")

    return "\n".join(lines)
