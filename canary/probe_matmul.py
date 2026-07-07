# canary/probe_matmul.py
"""측정 대상 커널: 4096^2 matmul. 인자 --loop-seconds N 주면 N초 반복(전력 런),
없으면 1회 실행(ncu 트래픽 런). 스펙 3절: 두 런은 반드시 분리 실행."""
import sys
import time

import torch


def main():
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    if "--loop-seconds" in sys.argv:
        secs = float(sys.argv[sys.argv.index("--loop-seconds") + 1])
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            c = a @ b
            torch.cuda.synchronize()
    else:
        c = a @ b
        torch.cuda.synchronize()
    print("PROBE_DONE", c.shape)


if __name__ == "__main__":
    main()
