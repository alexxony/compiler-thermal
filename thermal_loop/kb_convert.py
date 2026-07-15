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
from pathlib import Path


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
    # 검증 루프 버그 패턴 A 수정(2026-07-15): get_inputs()/get_init_inputs()가
    # `A = torch.rand(...); return [A, B]`처럼 return 앞에 로컬 할당문을 두는 경우
    # (예: 1_Square_matrix_multiplication_), 기존엔 return 표현식(get_inputs_src/
    # init_inputs_src)만 뽑고 이 할당문들을 버려서 생성 _make_inputs()가 `vals = [A, B]`를
    # 방출하는데 A/B가 미정의 → NameError. 아래 두 필드가 return 앞 statement들의
    # 소스 텍스트를 보존 — convert_to_solve_source가 vals 대입 직전에 그대로 방출한다.
    get_inputs_body_src: str = ""   # get_inputs() 본문 중 최종 return 이전 statement들
    init_inputs_body_src: str = ""  # get_init_inputs() 본문 중 최종 return 이전 statement들
    # 검증 루프 버그 패턴 B 수정(2026-07-15): __init__ 파라미터 중 디폴트값이 있는
    # 것(예: num_groups=4, bias=True)은 get_init_inputs()가 값을 안 주는 경우가 흔함
    # (Python 쪽 디폴트에 위임). 기존엔 init_args가 파라미터 "이름"만 보존하고 디폴트를
    # 버려서 생성 _KBModel.__init__ 시그니처가 전부 필수 인자가 되어 _init_args() 반환
    # 튜플이 짧으면 TypeError(missing N required positional arguments). 이 필드가
    # (파라미터명 → 디폴트 소스 텍스트) 매핑을 보존해 시그니처에 디폴트를 재주입한다.
    init_arg_defaults: dict = field(default_factory=dict)
    # 검증 루프 버그 패턴(2026-07-15, 4차): 원본 KernelBench 파일이 `import torch`/
    # `import torch.nn as nn` 외 추가 import(가장 흔한 예: `import torch.nn.functional
    # as F`, 27_Conv3d_HardSwish_GroupNorm_Mean이 forward()에서 F.hardswish 사용)를
    # 쓰는 경우, 기존엔 module_body_src 추출 시 ast.Import/ImportFrom을 전부 걸러내
    # 생성 파일에 전혀 방출되지 않아 reference()/forward()가 F를 참조하면 NameError.
    # 이 필드가 torch/torch.nn 표준 2줄 이외의 원본 import 문 소스 텍스트를 보존한다.
    extra_import_lines: list = field(default_factory=list)


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


class _SuperCallNormalizer(ast.NodeTransformer):
    """`super(Model, self).__init__()`(구식 명시 super) → `super().__init__()`(암묵 super).

    생성 파일의 클래스명은 `_KBModel`(원본 `Model`이 아님) — 원본 텍스트를 그대로
    이식하면 `super(Model, self)`가 정의되지 않은 `Model` 이름을 참조해 즉시
    `NameError`/`TypeError`(3차 검증 버그 2, 2026-07-14, 200/200 KernelBench 파일이
    이 패턴 사용 — 실사 확인). 문자열 치환 대신 AST 기반 변환(팀리드 명시 요구) —
    `super(<임의 2-인자>)` 호출만 표적으로 삼아 무관한 텍스트의 우발적 치환을 차단.
    """
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "super"
                and len(node.args) == 2):
            return ast.Call(func=ast.Name(id="super", ctx=ast.Load()),
                            args=[], keywords=[])
        return node


def _normalize_super_body_stmts(init_fn: ast.FunctionDef) -> list[ast.stmt]:
    """__init__ 본문 statement 리스트를 deepcopy해 구식 super() 호출만 정규화.

    원본 AST(전체 소스 파싱 결과)는 변경하지 않음 — deepcopy 위에서만 변환해
    다른 필드(forward_src 등)가 참조하는 원본 트리에 부작용 없음.
    """
    import copy
    normalized = [_SuperCallNormalizer().visit(copy.deepcopy(stmt))
                  for stmt in init_fn.body]
    for stmt in normalized:
        ast.fix_missing_locations(stmt)
    return normalized


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
    # 패턴 B: __init__(self, in_channels, ..., num_groups=4, bias=True)의 디폴트값들을
    # 파라미터명에 매핑. ast.arguments.defaults는 뒤쪽 N개 인자에만 대응(파이썬 규칙 —
    # 디폴트 없는 인자가 항상 앞에 옴)하므로 뒤에서부터 zip.
    init_arg_defaults: dict[str, str] = {}
    if init_fn is not None and init_fn.args.defaults:
        non_self_args = [a.arg for a in init_fn.args.args if a.arg != "self"]
        defaulted_names = non_self_args[len(non_self_args) - len(init_fn.args.defaults):]
        for pname, dnode in zip(defaulted_names, init_fn.args.defaults):
            init_arg_defaults[pname] = _src_of(src, dnode)
    # 3차 검증 버그 2 수정: super(Model, self).__init__() 구식 호출을 AST 기반으로
    # super().__init__()로 정규화(원본 텍스트 그대로 이식 시 존재하지 않는 클래스명
    # "Model" 참조로 즉시 NameError — 생성 클래스명은 "_KBModel"). 정규화 후 각
    # statement를 ast.unparse로 재직렬화(정규화 안 걸린 statement는 원본 소스 그대로
    # 두는 게 이상적이나, unparse가 전체적으로 일관된 포맷을 내므로 전 statement에
    # 동일 처리 — 서식만 달라질 뿐 의미는 원본과 동일).
    init_body_src = ("\n".join(ast.unparse(n) for n in _normalize_super_body_stmts(init_fn))
                     if init_fn else "")

    # forward 본문(반환 표현식 등) — docstring 제외한 statement 텍스트.
    body_stmts = [n for n in forward_fn.body
                  if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str))]
    forward_src = "\n".join(_src_of(src, n) for n in body_stmts)

    get_inputs_fn = _get_func(tree, "get_inputs")
    get_init_fn = _get_func(tree, "get_init_inputs")
    get_inputs_src = _return_expr_src(src, get_inputs_fn) if get_inputs_fn else "[]"
    init_inputs_src = _return_expr_src(src, get_init_fn) if get_init_fn else "[]"
    # 패턴 A: return 앞에 오는 로컬 할당문(예: `A = torch.rand(N, N)`) 보존.
    get_inputs_body_src = _body_before_return_src(src, get_inputs_fn) if get_inputs_fn else ""
    init_inputs_body_src = _body_before_return_src(src, get_init_fn) if get_init_fn else ""

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

    # 4차 검증 버그(2026-07-15): 원본이 `import torch.nn.functional as F` 등 표준
    # 2줄(torch, torch.nn) 이외의 import를 쓰면(forward()/reference()가 F.hardswish
    # 등을 참조) 방출 누락 시 NameError. torch/torch.nn 표준 2줄과 동일한 것은
    # 중복 방지를 위해 제외하고 나머지만 보존.
    _STANDARD_IMPORTS = {"import torch", "import torch.nn as nn"}
    extra_import_lines = [
        _src_of(src, n) for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and _src_of(src, n) not in _STANDARD_IMPORTS
    ]

    stateful = init_inputs_src.strip() not in ("[]", "()")

    return KBProblem(
        name=name, forward_src=forward_src, forward_args=forward_args,
        init_args=init_args, init_body_src=init_body_src,
        init_inputs_src=init_inputs_src,
        get_inputs_src=get_inputs_src, module_body_src=module_body_src,
        stateful=stateful, docstring=docstring.strip(),
        get_inputs_body_src=get_inputs_body_src,
        init_inputs_body_src=init_inputs_body_src,
        init_arg_defaults=init_arg_defaults,
        extra_import_lines=extra_import_lines,
    )


def _return_expr_src(src: str, fn: ast.FunctionDef) -> str:
    for n in fn.body:
        if isinstance(n, ast.Return) and n.value is not None:
            return _src_of(src, n.value)
    return "[]"


def _body_before_return_src(src: str, fn: ast.FunctionDef) -> str:
    """fn 본문 중 최종 return 문 이전 statement들의 소스 텍스트(줄바꿈 결합).

    패턴 A 수정: get_inputs()/get_init_inputs()가 `A = torch.rand(...)` 같은 로컬
    할당문을 return 앞에 두는 경우, 이 텍스트를 생성 solve.py에서 return 표현식
    대입 직전에 그대로 재생해 미정의 이름(NameError)을 막는다. docstring(문자열
    상수 Expr)은 forward_src 처리와 동일하게 제외.
    """
    stmts = [n for n in fn.body
             if not isinstance(n, ast.Return)
             and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                      and isinstance(n.value.value, str))]
    return "\n".join(_src_of(src, n) for n in stmts)


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
    # 4차 검증 버그 수정: 원본이 쓰는 추가 import(예: torch.nn.functional as F)를
    # 표준 2줄 뒤에 그대로 재생 — forward/reference가 F.hardswish 등을 참조할 때
    # NameError 방지(27_Conv3d_HardSwish_GroupNorm_Mean에서 실사 확인).
    for imp_line in kb.extra_import_lines:
        lines.append(imp_line)
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

    # 3차 검증 버그 1 수정: 원본 KernelBench 모듈 top-level 상수(batch_size,
    # in_features 등, kb.module_body_src — parse_kb_source가 이미 추출)를 생성
    # 파일에 실제로 방출. 이전엔 파싱만 하고 방출을 누락해 _init_args()/
    # _make_inputs()가 미정의 이름을 참조 → NameError(검증자 AST 전수 스캔으로
    # 101건 발견, stateful 21/35 문제 전체 영향). 클래스·함수 정의보다 먼저 둬야
    # 그 아래 함수 본문이 모듈 스코프에서 상수를 참조할 때 이미 바인딩되어 있음.
    if kb.module_body_src.strip():
        lines.append("# KernelBench 원본 모듈 top-level 상수 (make_case/get_init_inputs 스케일 파라미터)")
        for stmt in kb.module_body_src.splitlines():
            lines.append(stmt)
        lines.append("")

    # 원본 KernelBench Model 클래스 그대로 보존 — forward()가 이걸 감싸는 형태.
    lines.append("class _KBModel(nn.Module):")
    if kb.init_args:
        # 패턴 B 수정: 디폴트값 있는 파라미터(num_groups=4 등)는 시그니처에 디폴트를
        # 되살린다 — get_init_inputs()가 그 값들을 안 주는 걸 전제로 해도 호출이
        # TypeError 없이 성립하도록(디폴트로 채워짐).
        init_sig = ", ".join(
            f"{a}={kb.init_arg_defaults[a]}" if a in kb.init_arg_defaults else a
            for a in kb.init_args
        )
        lines.append(f"    def __init__(self, {init_sig}):")
    else:
        lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    if kb.stateful and kb.init_body_src.strip():
        # 원본 __init__ 본문(nn.Linear 등 계층 선언)을 그대로 이식 — 팀리드 지적
        # 갭(2026-07-14) 수정: 이전엔 "pass" 스텁만 남겨 학습 파라미터가 실제로
        # 생성되지 않는 correctness 결함이 있었음. kb.init_body_src는 parse_kb_source가
        # 이미 뽑아둔 원본 소스 텍스트라 여기서 그대로 재사용(재구성 불요).
        for stmt in kb.init_body_src.splitlines():
            lines.append(f"        {stmt}")
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
        # 패턴 A 수정(get_init_inputs 쪽): return 앞 로컬 할당문 보존.
        if kb.init_inputs_body_src.strip():
            for stmt in kb.init_inputs_body_src.splitlines():
                lines.append(f"    {stmt}")
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
    # 패턴 A 수정(get_inputs 쪽): return 앞 로컬 할당문(예: `A = torch.rand(N, N)`)을
    # vals 대입 이전에 그대로 재생 — get_inputs_src(반환 표현식)가 그 이름들을 참조.
    if kb.get_inputs_body_src.strip():
        for stmt in kb.get_inputs_body_src.splitlines():
            lines.append(f"    {stmt}")
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

    # reference()는 run_solve()로 위임하지 않고 forward 본문을 **직접 인라인**한다.
    # 갭 수정(팀리드 지적, 2026-07-14 2차): run_ablation_remote._reference_uses_matmul는
    # reference() 함수 몸체를 ast.walk해 matmul/bmm/mm 어트리뷰트 호출 또는 @ 연산자를
    # 직접 찾는다 — reference()가 run_solve()/model(...)을 통해 간접 호출하면(제너릭
    # Call, Attribute 아님) 이 정적 검사가 실패를 놓친다(golden matmul/batched_gemm의
    # reference()도 이 패턴 — case["A"] @ case["B"] 직접 계산, 위임 없음). 검증:
    # 실제 kb_convert 산출물에 _reference_uses_matmul을 직접 실행해 True 확인함
    # (fix 전 False로 확인된 회귀 — 정적 가드가 조용히 무력화되는 결함).
    lines.append("def reference(case, device):")
    # forward(self, A, B, ...) 매개변수를 case dict에서 지역 변수로 바인딩 —
    # forward_body가 그 이름(A, B 등)을 그대로 참조하므로 인라인 전에 필요.
    arg_bindings = [f"{a} = case[{a!r}]" for a in fwd_args]
    body_stmts = forward_body.splitlines() or ["return x"]
    if kb.stateful:
        ref_body_stmts = ["model = case['_model']"] + arg_bindings + [
            s.replace("self.", "model.") for s in body_stmts]
    else:
        ref_body_stmts = arg_bindings + body_stmts
    if uses_matmul:
        lines.append("    prev = torch.backends.cuda.matmul.allow_tf32")
        lines.append("    torch.backends.cuda.matmul.allow_tf32 = False")
        lines.append("    try:")
        for stmt in ref_body_stmts:
            lines.append(f"        {stmt}")
        lines.append("    finally:")
        lines.append("        torch.backends.cuda.matmul.allow_tf32 = prev")
    else:
        for stmt in ref_body_stmts:
            lines.append(f"    {stmt}")
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


def convert_to_tf32_variant_source(seed_solve_src: str) -> str:
    """seed solve.py 텍스트 → TF32-ON variant 텍스트.

    변환: 모듈 top-level `torch.backends.cuda.matmul.allow_tf32 = False`(seed 기본값)를
    `True`로 뒤집는다. `reference()` 내부의 지역 방어(`prev = ...; allow_tf32 = False;
    try/finally`)는 **건드리지 않음** — correctness gate가 TF32로 오염되면 안 되므로
    (P5 kb_matmul_scalar 버그 재현 방지). 이 변환은 seed와 정확히 한 줄만 달라
    "TF32 처방"의 인과를 깨끗하게 격리한다(matmul/batched_gemm 골든과 동일 패턴).

    비-matmul 순수 버킷(reference가 matmul을 쓰지 않는 문제)에 이 함수를 호출하면
    안 됨 — 호출측(convert_problem_file)이 uses_matmul 여부로 게이팅.
    """
    # 모듈 top-level 대입만 치환(들여쓰기 없는 라인) — reference() 내부의 들여쓰기된
    # "        torch.backends.cuda.matmul.allow_tf32 = False"는 매치 안 되게 앵커.
    out = seed_solve_src.replace(
        "torch.backends.cuda.matmul.allow_tf32 = False\n",
        "torch.backends.cuda.matmul.allow_tf32 = True\n", 1)
    out = out.replace(
        "torch.backends.cudnn.allow_tf32 = False\n",
        "torch.backends.cudnn.allow_tf32 = True\n", 1)
    return out


def convert_problem_file(kb: KBProblem, problems_root: str | Path,
                         problem_name: str) -> Path:
    """P8 Task 31 갭 보완 — KBProblem을 실제 `problems/<problem_name>/` 디렉토리로 파일 쓰기.

    이전엔 convert_to_solve_source가 텍스트만 반환하고 디스크에 아무것도 쓰지 않아
    p8-measure가 실행할 산출물이 없었음(팀리드 지적, 2026-07-14). 이 함수가 그 갭을
    메운다 — golden 4문제(matmul/batched_gemm/kb_matmul_scalar/kb_softmax)와 동일한
    디렉토리 레이아웃(`solve.py` + 선택적 `variants/R_*.py`)으로 생성.

    variant 생성 정책(정직성 우선): matmul을 실제로 쓰는 버킷에만 TF32 variant를
    생성한다 — compute-matmul은 `R_tf32on.py`(run_ablation_remote._variant_map_for
    규약), 그 외 matmul 관여 버킷은 `R_tf32.py`. **R_coalesced.py는 생성하지 않는다**
    — 그건 kb_matmul_scalar golden처럼 실제 커널 재작성(블록 타일링 등)이 필요한
    수작업 최적화라 KernelBench Model에서 자동 도출 불가. 파일이 없으면
    `_variant_map_for`가 자연스럽게 스킵(콜백이 seed로 폴백) — 회귀 아님, 이미
    문서화된 동작(run_ablation_remote.py:172-176 "variant 없는 룰이 발화하면
    콜백이 seed로 폴백").

    반환: 생성된 문제 디렉토리 경로. 컴파일 실패(문법 오류) 시 예외 전파 —
    조용히 깨진 solve.py를 쓰지 않는다(호출측 convert_problem_batch가 개별 격리).
    """
    problems_root = Path(problems_root)
    solve_src = convert_to_solve_source(kb, problem_name=problem_name)
    compile(solve_src, f"<{problem_name}/solve.py>", "exec")  # 문법 검증 — 깨진 파일 쓰기 전 차단

    prob_dir = problems_root / problem_name
    prob_dir.mkdir(parents=True, exist_ok=True)
    (prob_dir / "solve.py").write_text(solve_src)

    bucket = op_pattern_bucket(kb)
    uses_matmul = bucket in ("compute-matmul", "compute-conv-fusion")
    if uses_matmul:
        variant_name = "R_tf32on.py" if bucket == "compute-matmul" else "R_tf32.py"
        variant_src = convert_to_tf32_variant_source(solve_src)
        compile(variant_src, f"<{problem_name}/variants/{variant_name}>", "exec")
        vdir = prob_dir / "variants"
        vdir.mkdir(exist_ok=True)
        (vdir / variant_name).write_text(variant_src)

    return prob_dir


def convert_problem_batch(kb_and_names: list[tuple[KBProblem, str]],
                          problems_root: str | Path) -> list[tuple[str, bool, str]]:
    """여러 KBProblem을 순차 변환·파일쓰기. 개별 실패가 배치 전체를 죽이지 않음
    (35문제 중 1개 파싱/컴파일 실패해도 나머지는 계속 — 정직한 부분 성공 보고).

    반환: [(problem_name, ok, message)] — ok=False면 message에 에러 텍스트.
    """
    results: list[tuple[str, bool, str]] = []
    for kb, name in kb_and_names:
        try:
            convert_problem_file(kb, problems_root=problems_root, problem_name=name)
            results.append((name, True, ""))
        except (SyntaxError, ValueError) as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
    return results


def _find_kb_file(kb_root: Path, name: str) -> Path | None:
    """이름(예: "1_Square_matrix_multiplication_")으로 KernelBench 원본 .py를
    level1/level2에서 탐색. 층화 샘플 JSON의 name 필드가 KernelBench stem 그대로라
    이 매핑이 유일한 조회 경로.
    """
    for level_dir in ("level1", "level2"):
        p = kb_root / level_dir / f"{name}.py"
        if p.exists():
            return p
    return None


def main(argv: list[str]) -> int:
    """P8 Task 31 갭 보완 CLI — 층화 35문제 샘플 JSON을 실제 `problems/` 산출물로 변환.

    입력: KernelBench clone 루트(level1/level2 보유) + kb_stratify.run_extraction
    산출물(35문제 sample 목록, name/level/bucket/role 필드) + problems 출력 루트.
    실행: python3 kb_convert.py <kernelbench_root> <sample.json> [--problems-root=<path>]
          [--selfcheck]

    p8-measure가 그대로 실행 가능한 `problems/<name>/solve.py`(+variants/)가
    "완료 정의"(팀리드, 2026-07-14) — 이 CLI가 그 산출물을 만드는 유일한 경로.
    """
    import json
    import sys

    if "--selfcheck" in argv:
        # GPU·실 clone 불요 — 합성 KernelBench 문제로 배치 변환 배선만 확인.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            kb_root = Path(d) / "kb"
            (kb_root / "level1").mkdir(parents=True)
            (kb_root / "level1" / "1_Fake_matmul.py").write_text(
                "import torch\nimport torch.nn as nn\n"
                "class Model(nn.Module):\n"
                "    def __init__(self):\n        super().__init__()\n"
                "    def forward(self, A, B):\n        return torch.matmul(A, B)\n"
                "def get_inputs():\n    return [torch.rand(4, 4), torch.rand(4, 4)]\n"
                "def get_init_inputs():\n    return []\n")
            problems_root = Path(d) / "problems"
            kbs = []
            src = (kb_root / "level1" / "1_Fake_matmul.py").read_text()
            kb = parse_kb_source(src, name="1_Fake_matmul")
            written = convert_problem_batch([(kb, "1_Fake_matmul")], problems_root=problems_root)
            assert written == [("1_Fake_matmul", True, "")]
            assert (problems_root / "1_Fake_matmul" / "solve.py").exists()
            assert (problems_root / "1_Fake_matmul" / "variants" / "R_tf32on.py").exists()

            # 교차모듈 가드 재확인(합성 경로에서도).
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import run_ablation_remote as rar
            orig = rar.PROBLEMS
            try:
                rar.PROBLEMS = problems_root
                w = rar.check_tf32_guard("1_Fake_matmul")
            finally:
                rar.PROBLEMS = orig
            assert w is None, f"selfcheck 가드 실패: {w}"
        print("kb_convert.py main --selfcheck PASS "
              "(합성 문제 배치 변환 + TF32 가드 교차확인 포함)")
        return 0

    if len(argv) < 2:
        print("usage: kb_convert.py <kernelbench_root> <sample.json> "
              "[--problems-root=<path>] [--selfcheck]", file=sys.stderr)
        return 2

    kb_root = Path(argv[0])
    sample_path = Path(argv[1])
    problems_root = Path(__file__).resolve().parents[1] / "problems"
    for a in argv[2:]:
        if a.startswith("--problems-root="):
            problems_root = Path(a.split("=", 1)[1])

    sample = json.loads(sample_path.read_text())
    entries = sample.get("sample", sample if isinstance(sample, list) else [])

    kb_and_names: list[tuple[KBProblem, str]] = []
    missing: list[str] = []
    for entry in entries:
        name = entry["name"]
        kb_file = _find_kb_file(kb_root, name)
        if kb_file is None:
            missing.append(name)
            continue
        kb = parse_kb_source(kb_file.read_text(), name=name)
        kb_and_names.append((kb, name))

    if missing:
        print(f"[WARN] KernelBench 원본 파일 미발견 {len(missing)}건: {missing}",
              file=sys.stderr)

    written = convert_problem_batch(kb_and_names, problems_root=problems_root)
    ok = [n for n, s, _ in written if s]
    failed = [(n, m) for n, s, m in written if not s]

    print(f"[kb_convert] 변환 시도 {len(written)}건 — 성공 {len(ok)}, 실패 {len(failed)}")
    for n, m in failed:
        print(f"  [FAIL] {n}: {m}", file=sys.stderr)
    for n in ok:
        print(f"  [OK] {n} -> {problems_root / n}")

    return 0 if not failed and not missing else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
