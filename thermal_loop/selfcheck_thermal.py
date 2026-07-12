"""열-gain 드라이버 전용 fake-profiler self-check (Compiler_Thermal P3).

지난 실측(2026-07-12, Colab A100)에서 로컬 self-check가 못 잡은 버그 2건
(executor.py `import time` 누락, chip 키 명명 충돌)이 실제 GPU 라운드에서야
발견됐다 — 원인은 기존 self-check가 fake profiler에 완성된 signal_dict를
그대로 주입해서 merge_signals/energy_per_iter_j 계산 경로 자체를 안 태웠기
때문. 이 파일은 그 구멍을 메운다: fake profiler가 ncu/power 원시값(kernel_time_s,
avg_power_w, dram_bytes 등 A100 실측 범위)만 주고, 실제 thermal.measure.merge_signals를
호출해 energy_per_iter_j까지 계산시킨 뒤 harness._metric(mode="thermal")이 그
필드를 옳게 읽는지 검증한다. GPU·colab-cli·ncu 프로세스는 전혀 안 씀.

run_thermal_gain.py를 직접 실행하지 않는 이유: 그 드라이버는 colab_profiler를
통해서만 실제 배관을 태우므로, 여기선 runner.run_problem에 fake profiler를
주입하는 동일 패턴(mailbox.py/runner.py self-check와 동일 스타일)으로
run_thermal_gain._run_track과 같은 경로(submit_kind="thermal")를 재현한다.
"""
from __future__ import annotations
import sys
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from runner import run_problem
from glue import ProfileResult, GateResult
from mailbox import MailboxResult
from thermal.measure import merge_signals


@dataclass
class _FakeThermalProfiler:
    """submit(code, problem, kind=...) — merge_signals를 실제로 호출.

    code 문자열로 "seed"/"tf32" 두 트랙을 구분(실 variant_map 라벨과 무관,
    self-check 전용 더미 판별). A100 실측(2026-07-12) 범위값 사용:
      seed(fp32): kernel_time_s≈9.46ms, avg_power_w≈309W → energy_per_iter_j≈2.92J
      tf32      : kernel_time_s≈1.48ms, avg_power_w≈355W → energy_per_iter_j≈0.53J
    즉 TF32가 순간전력은 높아도(355>309) 일 단위 에너지는 5.5배 가까이 줄어야
    한다 — 이게 이번에 고친 목적함수(energy_per_iter_j)가 잡아야 하는 방향.
    """
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def submit(self, code: str, problem: str,
               profile_opts: dict | None = None, kind: str | None = None) -> MailboxResult:
        self.calls.append(code)
        is_tf32 = "tf32" in code.lower()
        ncu = {
            "dram_bytes_read": 4.1e8 if is_tf32 else 1.24e9,
            "dram_bytes_write": 5.9e7 if is_tf32 else 6.6e7,
            "kernel_time_s": 1.48e-3 if is_tf32 else 9.46e-3,
            "latency_us": 1480.0 if is_tf32 else 9460.0,
        }
        power = {
            "avg_power_w": 355.0 if is_tf32 else 309.0,
            "energy_j": 355.0 * 3.0 if is_tf32 else 309.0 * 3.0,  # 고정 3s 창(원시, 참고용)
        }
        thermal_sig = merge_signals(chip="NVIDIA A100-SXM4-40GB", ncu=ncu, power=power)
        signal_dict = {
            **thermal_sig,
            "latency_us": ncu["latency_us"],
            "weight_pct": 1.0,
            "tensorcore_active": is_tf32,
            "compute_tput": 0.88 if is_tf32 else 0.98,
            "bw_pct": 0.23 if is_tf32 else 0.09,
            "load_eff": 0.0 if is_tf32 else 1.0,
        }
        return MailboxResult(
            profile=ProfileResult(signal_dict, ncu["latency_us"]),
            gate=GateResult(True, 1e-5),
            error=None,
        )

    def profile(self, code: str, problem: str) -> ProfileResult:
        return self.submit(code, problem).profile


def main() -> int:
    seed_code = "# seed fp32 matmul (fake)"
    tf32_code = "# tf32 matmul variant (fake)"

    fd, ledger_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(ledger_path)
    try:
        prof = _FakeThermalProfiler()

        class _AlwaysTf32Generator:
            """1라운드=seed, 2라운드부터=tf32 — 실제 hypcond 콜백 패턴 축소판."""
            def __init__(self):
                self.n = 0

            def generate(self, problem, hypothesis_prompt, prev_code):
                from glue import GenResult, code_hash
                self.n += 1
                code = seed_code if self.n == 1 else tf32_code
                return GenResult(code, code_hash(code))

        res = run_problem(
            "matmul", seed_code, mailbox_dir="/nonexistent", ledger_path=ledger_path,
            sync_fn=lambda _d: None, max_rounds=4, poll_s=0.0,
            generator=_AlwaysTf32Generator(), metric_mode="thermal",
            profiler=prof, submit_kind="thermal",
        )

        from ledger import Ledger
        led = Ledger(ledger_path)
        recs = [r for r in led.records if r.problem == "matmul"]
        assert len(recs) >= 2, f"seed+tf32 최소 2라운드 기록돼야, got {len(recs)}"

        seed_rec, tf32_rec = recs[0], recs[1]
        seed_epi = seed_rec.signal["energy_per_iter_j"]
        tf32_epi = tf32_rec.signal["energy_per_iter_j"]
        print(f"  seed energy_per_iter_j = {seed_epi:.4f} J")
        print(f"  tf32  energy_per_iter_j = {tf32_epi:.4f} J")
        assert seed_epi > 0 and tf32_epi > 0, "energy_per_iter_j가 실제로 채워져야(0이면 배관 끊김)"
        assert tf32_epi < seed_epi, (
            "TF32가 순간전력은 높아도(355>309W) energy_per_iter_j는 줄어야 "
            "(목적함수가 일 단위 정규화를 실제로 반영하는지 검증)"
        )
        ratio = seed_epi / tf32_epi
        print(f"  ratio = {ratio:.2f}x (팀리드 예측 ~5.3x 근방이어야)")
        assert 3.0 < ratio < 8.0, f"비율이 예상 범위(3~8x) 밖: {ratio:.2f}x"

        # metric 곡선도 -energy_per_iter_j 부호로 실려야 (harness._metric 배관 확인)
        curve = led.metric_curve("matmul")
        assert curve[0][1] == -seed_epi, "metric 곡선 1라운드가 -energy_per_iter_j여야"
        assert curve[1][1] == -tf32_epi, "metric 곡선 2라운드가 -energy_per_iter_j여야"
        assert curve[1][1] > curve[0][1], "TF32가 seed보다 metric(더 큰 값=더 나음)이어야"

        # ON 트랙이면(evolve_enabled=True 기본) TF32가 개선으로 판정→retire 후보권 진입해야.
        assert tf32_rec.improved, "TF32 라운드가 seed 대비 improved=True여야 (에너지 절감 반영)"

        print(f"  발화 라운드: {res.rounds}, stop={res.stopped_reason}")
        print("selfcheck_thermal.py PASS — energy_per_iter_j 배관 end-to-end 검증됨")
        return 0
    finally:
        Path(ledger_path).exists() and Path(ledger_path).unlink()


if __name__ == "__main__":
    sys.exit(main())
