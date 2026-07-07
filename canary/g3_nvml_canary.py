# canary/g3_nvml_canary.py
"""G3: nvml 전력 샘플링이 Colab GPU에서 유효한지 판정.
부하 중 평균 전력이 idle 대비 1.3배 이상 상승하면 PASS."""
import json
import threading
import time


def _sampler(handle, out, stop):
    import pynvml
    while not stop.is_set():
        out.append((time.monotonic(), pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0))
        time.sleep(0.1)


def main() -> int:
    import pynvml
    import torch
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    samples, stop = [], threading.Event()
    thread = threading.Thread(target=_sampler, args=(handle, samples, stop))
    thread.start()

    time.sleep(3.0)                       # idle 구간
    idle_end = time.monotonic()
    a = torch.randn(8192, 8192, device="cuda")
    t_load = time.monotonic()
    while time.monotonic() - t_load < 5.0:  # 부하 구간 ~5s
        b = a @ a
        torch.cuda.synchronize()
    stop.set()
    thread.join()

    idle = [w for t, w in samples if t < idle_end]
    load = [w for t, w in samples if t >= idle_end + 0.5]
    idle_avg = sum(idle) / len(idle)
    load_avg = sum(load) / len(load)
    verdict = {
        "gate": "G3", "chip": str(name), "n_samples": len(samples),
        "idle_avg_w": round(idle_avg, 1), "load_avg_w": round(load_avg, 1),
        "ratio": round(load_avg / idle_avg, 2), "passed": load_avg > idle_avg * 1.3,
    }
    print("G3_VERDICT " + json.dumps(verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
