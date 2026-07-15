"""P8 Task 31 — KernelBench Model → 자체 solve.py 변환기 테스트.

torch 불요 — 변환기는 ast/텍스트 처리만 (torch는 변환된 solve.py가 import하지만
변환 로직 자체는 순수 파이썬). golden 기준: 기존 kb_matmul_scalar/kb_softmax의
executor 계약(make_case/run_solve/reference/GATE_SIZES/PROFILE_SIZE)과 구조 일치.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from kb_convert import (  # noqa: E402
    KBProblem,
    parse_kb_source,
    classify_workload,
    convert_to_solve_source,
    op_pattern_bucket,
    variant_map_for_bucket,
    convert_problem_file,
    convert_problem_batch,
)

_KB_MATMUL_SRC = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

N = 4096

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
'''

_KB_SOFTMAX_SRC = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 1024

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
'''

_KB_LINEAR_SRC = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
'''


def test_parse_kb_source_extracts_model_and_get_inputs():
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    assert isinstance(kb, KBProblem)
    assert kb.name == "1_Square_matrix_multiplication_"
    assert "torch.matmul" in kb.forward_src or "matmul" in kb.forward_src
    assert kb.init_inputs_src.strip() == "[]"
    assert kb.stateful is False  # get_init_inputs() 빈 리스트 → 학습 파라미터 없음


def test_parse_kb_source_detects_stateful_model():
    kb = parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU")
    assert kb.stateful is True  # nn.Linear + get_init_inputs 비어있지 않음


def test_classify_workload_matmul_is_compute_matmul_bucket():
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    bucket = classify_workload(kb)
    assert bucket == "compute-matmul"


def test_classify_workload_softmax_is_memory_bucket():
    kb = parse_kb_source(_KB_SOFTMAX_SRC, name="23_Softmax")
    bucket = classify_workload(kb)
    assert bucket in ("memory-norm", "memory-reduce-elementwise")


def test_classify_workload_fusion_linear_is_compute_conv_fusion_bucket():
    kb = parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU")
    bucket = classify_workload(kb)
    assert bucket == "compute-conv-fusion"


def test_op_pattern_bucket_four_buckets_only():
    # §3-2 4버킷: compute-matmul / compute-conv-fusion / memory-norm / memory-reduce-elementwise
    valid = {"compute-matmul", "compute-conv-fusion",
             "memory-norm", "memory-reduce-elementwise"}
    for src, name in ((_KB_MATMUL_SRC, "matmul"), (_KB_SOFTMAX_SRC, "softmax"),
                      (_KB_LINEAR_SRC, "linear")):
        kb = parse_kb_source(src, name=name)
        assert op_pattern_bucket(kb) in valid


def test_convert_to_solve_source_produces_required_contract_symbols():
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    src = convert_to_solve_source(kb, problem_name="kbnew_matmul")
    tree = ast.parse(src)
    top_level_funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"make_case", "run_solve", "reference"} <= top_level_funcs
    assigned_names = {
        t.id for n in tree.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }
    assert "GATE_SIZES" in assigned_names
    assert "PROFILE_SIZE" in assigned_names


def test_convert_to_solve_source_matmul_injects_tf32_guard():
    # 정책: matmul 계열 reference()엔 allow_tf32 방어 자동 주입 (run_ablation_remote 가드 통과 전제)
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    src = convert_to_solve_source(kb, problem_name="kbnew_matmul")
    assert "allow_tf32" in src


def test_convert_to_solve_source_non_matmul_no_guard_needed():
    kb = parse_kb_source(_KB_SOFTMAX_SRC, name="23_Softmax")
    src = convert_to_solve_source(kb, problem_name="kbnew_softmax")
    # 가드가 없어도 무방 (run_ablation_remote.check_tf32_guard가 non-matmul은 스킵)
    tree = ast.parse(src)  # 최소 컴파일 가능해야 함
    assert tree is not None


def test_convert_to_solve_source_stateful_model_uses_seeded_init():
    kb = parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU")
    src = convert_to_solve_source(kb, problem_name="kbnew_linear")
    assert "manual_seed" in src  # 결정론 재현 위해 시드 고정 필요
    tree = ast.parse(src)
    assert tree is not None


def test_convert_to_solve_source_compiles_for_all_three_goldens():
    for src_text, name in ((_KB_MATMUL_SRC, "matmul"), (_KB_SOFTMAX_SRC, "softmax"),
                           (_KB_LINEAR_SRC, "linear")):
        kb = parse_kb_source(src_text, name=name)
        out = convert_to_solve_source(kb, problem_name=f"kbnew_{name}")
        compile(out, f"<{name}>", "exec")  # 문법 검증(로컬 torch 없음 — import는 스킵)


def test_variant_map_for_bucket_matmul_is_tf32on_only():
    # run_ablation_remote._variant_map_for 규약과 정합: matmul류=R_tf32on
    vm = variant_map_for_bucket("compute-matmul")
    assert vm == {"R_tf32on.py"}


def test_variant_map_for_bucket_non_matmul_is_tf32_plus_coalesced():
    vm = variant_map_for_bucket("compute-conv-fusion")
    assert vm == {"R_tf32.py", "R_coalesced.py"}
    vm2 = variant_map_for_bucket("memory-norm")
    assert vm2 == {"R_tf32.py", "R_coalesced.py"}


# ── 갭 보완(팀리드 지적, 2026-07-14) — convert_problem_file: 실제 problems/ 파일 생성 ──
# 기존 convert_to_solve_source는 KBProblem→텍스트 순수함수뿐, 디스크 쓰기 경로가 없어
# p8-measure가 그대로 쓸 수 있는 산출물이 없었음(kb_convert.py:234 주석이 참조하는
# convert_problem_file이 실제로는 미구현). 아래는 이 갭을 메우는 파일-쓰기 드라이버.

def test_convert_to_solve_source_stateful_injects_real_init_body_not_pass_stub():
    # 회귀 확인: 최초 구현은 stateful 모델의 __init__ 본문을 "pass" 스텁으로 남겨뒀음
    # (골격만 생성, 실제 nn.Linear 등 계층 선언이 유실 — correctness 결함).
    kb = parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU")
    src = convert_to_solve_source(kb, problem_name="kbnew_linear")
    assert "nn.Linear" in src  # 원본 __init__ 본문(self.gemm = nn.Linear(...))이 실제로 이식됨
    assert "pass  # __init__ body injected" not in src  # 스텁 잔재 없음


def test_convert_problem_file_writes_solve_py_under_problems_root(tmp_path):
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    out_dir = convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_matmul")
    solve_path = tmp_path / "kbnew_matmul" / "solve.py"
    assert solve_path.exists()
    assert out_dir == tmp_path / "kbnew_matmul"
    tree = ast.parse(solve_path.read_text())
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"make_case", "run_solve", "reference"} <= funcs


def test_convert_problem_file_matmul_bucket_writes_r_tf32on_variant(tmp_path):
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_matmul")
    variant_path = tmp_path / "kbnew_matmul" / "variants" / "R_tf32on.py"
    assert variant_path.exists()
    vsrc = variant_path.read_text()
    assert "allow_tf32 = True" in vsrc
    ast.parse(vsrc)  # 문법 유효


def test_convert_problem_file_non_matmul_bucket_writes_r_tf32_variant(tmp_path):
    kb = parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU")
    convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_linear")
    variant_path = tmp_path / "kbnew_linear" / "variants" / "R_tf32.py"
    assert variant_path.exists()
    ast.parse(variant_path.read_text())


def test_convert_problem_file_no_matmul_no_variants_dir_needed(tmp_path):
    kb = parse_kb_source(_KB_SOFTMAX_SRC, name="23_Softmax")
    convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_softmax")
    solve_path = tmp_path / "kbnew_softmax" / "solve.py"
    assert solve_path.exists()
    # non-matmul 문제는 TF32 variant가 무의미 — variants/ 디렉토리 자체가 없어도 됨
    # (run_ablation_remote._variant_map_for가 없는 파일은 자연스럽게 스킵).


def test_convert_problem_file_written_solve_py_passes_tf32_guard_check(tmp_path):
    # run_ablation_remote.check_tf32_guard와 동일한 검사를 직접 재현 — 파일로 써도
    # allow_tf32 방어가 살아있는지 확인(가드 로직 자체는 run_ablation_remote 소관,
    # 여기선 산출물이 그 검사를 통과할 형태인지만 검증).
    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    out_dir = convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_matmul")
    solve_src = (out_dir / "solve.py").read_text()
    tree = ast.parse(solve_src)
    uses_matmul = any(
        isinstance(n, ast.FunctionDef) and n.name == "reference"
        and any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult)
                or (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in ("matmul", "bmm", "mm"))
                for sub in ast.walk(n))
        for n in tree.body
    )
    assert uses_matmul
    assert "allow_tf32" in solve_src


def test_convert_problem_batch_writes_multiple_problems(tmp_path):
    kbs = [
        parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_"),
        parse_kb_source(_KB_SOFTMAX_SRC, name="23_Softmax"),
        parse_kb_source(_KB_LINEAR_SRC, name="12_Gemm_Multiply_LeakyReLU"),
    ]
    names = ["kbnew_matmul", "kbnew_softmax", "kbnew_linear"]
    written = convert_problem_batch(list(zip(kbs, names)), problems_root=tmp_path)
    assert len(written) == 3
    for n in names:
        assert (tmp_path / n / "solve.py").exists()


def test_convert_problem_batch_reports_errors_without_aborting(tmp_path):
    # 문제 하나가 파싱 불가(예: Model 클래스 없음)라도 나머지는 계속 처리 —
    # 35문제 배치 중 1개 실패로 전체가 죽지 않게(정직한 부분 성공 보고).
    good_kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    bad_kb = KBProblem(name="broken", forward_src="return x + y", forward_args=["x", "y"],
                       init_args=[], init_body_src="", init_inputs_src="[]",
                       get_inputs_src="not valid python (((", module_body_src="",
                       stateful=False)
    written = convert_problem_batch(
        [(good_kb, "kbnew_good"), (bad_kb, "kbnew_bad")], problems_root=tmp_path)
    assert (tmp_path / "kbnew_good" / "solve.py").exists()
    # bad_kb는 get_inputs_src가 문법 오류라 컴파일이 실패해야 함 — written 목록에서 제외되거나
    # 에러로 기록되어야 정직(조용히 깨진 파일을 쓰면 안 됨).
    good_names = {name for name, ok, _ in written if ok}
    bad_names = {name for name, ok, _ in written if not ok}
    assert "kbnew_good" in good_names
    assert "kbnew_bad" in bad_names


def test_generated_solve_py_passes_real_run_ablation_remote_guard_check(tmp_path):
    """교차모듈 회귀 잠금 — run_ablation_remote.check_tf32_guard(실제 정적 가드)를
    kb_convert 산출물에 직접 실행. 이전 버그(2026-07-14 2차 갭): reference()가
    run_solve()로 위임하면 _reference_uses_matmul이 matmul 호출을 못 찾아 가드가
    조용히 무력화됨(경고 0인데 실제론 미방어) — 이 테스트가 그 회귀를 원천 차단.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))
    import run_ablation_remote as rar  # noqa: E402

    kb = parse_kb_source(_KB_MATMUL_SRC, name="1_Square_matrix_multiplication_")
    out_dir = convert_problem_file(kb, problems_root=tmp_path, problem_name="kbnew_matmul")

    original_problems = rar.PROBLEMS
    try:
        rar.PROBLEMS = tmp_path
        warning = rar.check_tf32_guard("kbnew_matmul")
    finally:
        rar.PROBLEMS = original_problems
    assert warning is None, f"TF32 가드가 산출물에서 경고를 냄(방어 실패): {warning}"

    # 역방향 확인: reference()에서 allow_tf32 문자열을 강제로 지우면 가드가 반드시
    # 경고를 내야 함(가드 자체가 무력화된 게 아니라 정말 통과한 건지 대조).
    solve_path = out_dir / "solve.py"
    stripped = solve_path.read_text().replace("allow_tf32", "ALLOW_TF32_REMOVED")
    solve_path.write_text(stripped)
    try:
        rar.PROBLEMS = tmp_path
        warning2 = rar.check_tf32_guard("kbnew_matmul")
    finally:
        rar.PROBLEMS = original_problems
    assert warning2 is not None, "allow_tf32 제거했는데도 가드가 경고를 안 냄 — 가드 자체가 무력"
