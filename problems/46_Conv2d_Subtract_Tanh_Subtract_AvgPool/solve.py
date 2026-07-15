"""KernelBench 46_Conv2d_Subtract_Tanh_Subtract_AvgPool — P8 Task 31 자동 변환 (compute-conv-fusion).
Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
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
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# KernelBench 원본 모듈 top-level 상수 (make_case/get_init_inputs 스케일 파라미터)
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

class _KBModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

_SEED = 0

def _init_args():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]

def _make_model(device):
    torch.manual_seed(_SEED)  # 결정론 가중치 재현
    m = _KBModel(*_init_args()).to(device)
    m.eval()
    return m

def _make_inputs(device):
    torch.manual_seed(_SEED)
    vals = [torch.rand(batch_size, in_channels, height, width)]
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
        x = model.conv(x)
        x = x - model.subtract1_value
        x = torch.tanh(x)
        x = x - model.subtract2_value
        x = model.avgpool(x)
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
