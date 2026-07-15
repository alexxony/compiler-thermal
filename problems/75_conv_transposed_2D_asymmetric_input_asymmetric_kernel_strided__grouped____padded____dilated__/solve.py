"""KernelBench 75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__ — P8 Task 31 자동 변환 (memory-reduce-elementwise).
Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
grouped, padded, and dilated.

Args:
    in_channels (int): Number of channels in the input tensor.
    out_channels (int): Number of channels produced by the convolution.
    kernel_size (tuple): Size of the convolution kernel (height, width).
    stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
    padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
    dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
    groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
    bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

class _KBModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super().__init__()
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        return self.conv_transpose2d(x)

_SEED = 0

def _init_args():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]

def _make_model(device):
    torch.manual_seed(_SEED)  # 결정론 가중치 재현
    m = _KBModel(*_init_args()).to(device)
    m.eval()
    return m

def _make_inputs(device):
    torch.manual_seed(_SEED)
    x = torch.rand(batch_size, in_channels, height, width)
    vals = [x]
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
    model = case['_model']
    x = case['x']
    return model.conv_transpose2d(x)

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
