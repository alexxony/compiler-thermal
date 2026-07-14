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
