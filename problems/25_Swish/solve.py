"""KernelBench 25_Swish — P8 Task 31 자동 변환 (memory-reduce-elementwise).
Simple model that performs a Swish activation.
KernelBench Model→solve.py 변환(kb_convert.py). 엔진 미접촉 — 문제 사양만 흡수.
executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

GATE_ATOL = 1e-4
GATE_RTOL = 1e-4
GATE_SIZES = (16, 64, 256)
PROFILE_SIZE = 512

# KernelBench 원본 모듈 top-level 상수 (make_case/get_init_inputs 스케일 파라미터)
batch_size = 4096
dim = 393216

class _KBModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        return x * torch.sigmoid(x)

def _make_inputs(device):
    torch.manual_seed(0)
    x = torch.rand(batch_size, dim)
    vals = [x]
    return [v.to(device) if hasattr(v, 'to') else v for v in vals]

def make_case(size, device):
    # KernelBench get_inputs()는 고정 크기 — size는 게이트/프로파일 스윕용 스케일 힌트.
    xs = _make_inputs(device)
    case = {'x': xs[0]}
    return case

def run_solve(case, device):
    model = _KBModel()
    return model(case['x'])

def reference(case, device):
    x = case['x']
    return x * torch.sigmoid(x)

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
