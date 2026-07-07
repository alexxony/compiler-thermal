"""백그라운드 전력 샘플러. read_power_w 콜러블 주입 — 실환경은 pynvml, 테스트는 fake.

주의(스펙 3절): ncu replay 중 전력은 왜곡됨. 전력 런과 ncu 트래픽 런은
반드시 동일 커널을 2회 따로 실행해서 얻는다.
"""
import threading
import time
from dataclasses import dataclass, field


@dataclass
class PowerResult:
    samples: list = field(default_factory=list)  # [(t_monotonic, watts)]

    @property
    def avg_power_w(self) -> float:
        return sum(w for _, w in self.samples) / len(self.samples)

    @property
    def energy_j(self) -> float:
        e = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            e += (w0 + w1) / 2.0 * (t1 - t0)
        return e


def nvml_reader(device_index: int = 0):
    """실환경용 read_power_w 팩토리 (Colab에서만 호출)."""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


class PowerSampler:
    def __init__(self, read_power_w, interval_s: float = 0.1):
        self._read = read_power_w
        self._interval = interval_s
        self._result = PowerResult()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self._result.samples.append((time.monotonic(), self._read()))
            time.sleep(self._interval)

    def stop(self) -> PowerResult:
        self._stop.set()
        self._thread.join()
        return self._result
