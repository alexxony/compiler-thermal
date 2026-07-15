"""KernelBench 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish — P8 Task 31 자동 변환 (compute-matmul).
Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
applies LogSumExp, and finally applies the Mish activation function.
KernelBench Model→solve.py 변환(kb_convert.py). 엔진 미접촉 — 문제 사양만 흡수.
executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

GATE_ATOL = 6e-2
GATE_RTOL = 6e-2
GATE_SIZES = (16, 64, 256)
PROFILE_SIZE = 512

# seed = TF32 OFF (P5/P7 matmul/batched_gemm 관례와 동일 base).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# KernelBench 원본 모듈 top-level 상수 (make_case/get_init_inputs 스케일 파라미터)
batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

class _KBModel(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    def forward(self, x):
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)
        return x

_SEED = 0

def _init_args():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]

def _make_model(device):
    torch.manual_seed(_SEED)  # 결정론 가중치 재현
    m = _KBModel(*_init_args()).to(device)
    m.eval()
    return m

def _make_inputs(device):
    torch.manual_seed(_SEED)
    vals = [torch.rand(batch_size, input_size)]
    return [v.to(device) if hasattr(v, 'to') else v for v in vals]

def make_case(size, device):
    # KernelBench get_inputs()는 고정 크기 — size는 게이트/프로파일 스윕용 스케일 힌트.
    xs = _make_inputs(device)
    case = {'x': xs[0]}
    case['_model'] = _make_model(device)
    return case

def run_solve(case, device):
    model = case['_model']
    return model(case['x'])

def reference(case, device):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = case['_model']
        x = case['x']
        x = model.matmul(x)
        x = x * model.scale_factor
        x = x + x
        x = torch.clamp(x, model.clamp_min, model.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)
        return x
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    case = make_case(PROFILE_SIZE, dev)
    if args.profile:
        run_solve(case, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
    else:
        out = run_solve(case, dev)
        ref = reference(case, dev)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        print("PASS" if ok else "FAIL")
