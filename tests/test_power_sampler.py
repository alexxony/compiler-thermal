import time
from thermal.power_sampler import PowerSampler


class FakeNvml:
    """read_power_w 콜러블 규약: () -> float(W)"""
    def __init__(self):
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 100.0 + self.calls  # 단조 증가 → 시계열 검증 가능


def test_sampler_collects_series_and_energy():
    fake = FakeNvml()
    s = PowerSampler(read_power_w=fake, interval_s=0.01)
    s.start()
    time.sleep(0.15)
    result = s.stop()
    assert len(result.samples) >= 5           # (t, W) 튜플 시계열
    assert result.energy_j > 0                # 사다리꼴 적분
    assert result.avg_power_w > 100.0
    ts = [t for t, _ in result.samples]
    assert ts == sorted(ts)
