# canary/p1_measure_canary.py
"""P1 통합 카나리 (Colab GPU에서 실행): ① ncu 트래픽 런 ② PowerSampler 전력 런
분리 실행 후 merge_signals 병합. stdout 마지막 줄 'P1_VERDICT {json}'.
판정: 모든 키 유효값 + p_hbm_w < power_avg_w.

ncu 명령 형식·CSV 파싱은 GPU-Solver executor.py 검증분 재사용.
"""
import csv
import io
import json
import subprocess
import sys
import time
import traceback

NCU_METRICS = "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"

_BYTE_MULT = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9}
_SEC_MULT = {"nsecond": 1e-9, "ns": 1e-9, "usecond": 1e-6, "us": 1e-6,
             "msecond": 1e-3, "ms": 1e-3, "second": 1.0, "s": 1.0}


def _parse_ncu_csv(stdout: str) -> list:
    lines = stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Metric Name" in ln), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [row for row in reader if row.get("Metric Name")]


def _to_float(v: str) -> float:
    return float(str(v).replace(",", ""))


def ncu_run(probe: str) -> dict:
    """트래픽 런: dram bytes + 커널 시간 합산 (단위행 정규화)."""
    cmd = ["ncu", "--metrics", NCU_METRICS, "--csv",
           "--target-processes", "all",
           "--", sys.executable, probe]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ncu rc={r.returncode}: {r.stderr[-300:]}")
    rows = _parse_ncu_csv(r.stdout)
    if not rows:
        raise RuntimeError("ncu csv empty: " + r.stdout[-300:])
    acc = {"dram_bytes_read": 0.0, "dram_bytes_write": 0.0, "kernel_time_s": 0.0}
    for row in rows:
        name = row["Metric Name"]
        unit = str(row.get("Metric Unit", "")).lower()
        val = _to_float(row.get("Metric Value", "0"))
        if "dram__bytes_read" in name:
            acc["dram_bytes_read"] += val * _BYTE_MULT.get(unit, 1)
        elif "dram__bytes_write" in name:
            acc["dram_bytes_write"] += val * _BYTE_MULT.get(unit, 1)
        elif "duration" in name.lower():
            acc["kernel_time_s"] += val * _SEC_MULT.get(unit, 1e-9)
    return acc


def power_run(probe: str, secs: float = 5.0) -> dict:
    """전력 런: 백그라운드 샘플링 중 probe 반복 실행 (ncu 없이 — replay 왜곡 회피)."""
    from thermal.power_sampler import PowerSampler, nvml_reader
    sampler = PowerSampler(read_power_w=nvml_reader(), interval_s=0.1)
    sampler.start()
    r = subprocess.run([sys.executable, probe, "--loop-seconds", str(secs)],
                       capture_output=True, text=True, timeout=300)
    result = sampler.stop()
    if r.returncode != 0:
        raise RuntimeError(f"probe rc={r.returncode}: {r.stderr[-300:]}")
    return {"avg_power_w": result.avg_power_w, "energy_j": result.energy_j,
            "n_samples": len(result.samples)}


def main(probe: str = "probe_matmul.py") -> int:
    verdict = {"gate": "P1-integration", "passed": False, "stage": "chip"}
    try:
        import pynvml
        pynvml.nvmlInit()
        chip = str(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)))
        verdict["chip"] = chip
        verdict["stage"] = "ncu_run"
        ncu = ncu_run(probe)
        verdict["ncu"] = ncu
        verdict["stage"] = "power_run"
        power = power_run(probe)
        verdict["power"] = {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in power.items()}
        verdict["stage"] = "merge"
        from thermal.measure import merge_signals
        sig = merge_signals(chip=chip, ncu=ncu, power=power)
        verdict["signals"] = {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in sig.items()}
        verdict["passed"] = (
            all(v is not None for v in sig.values())
            and sig["dram_bytes_total"] > 0
            and sig["kernel_time_s"] > 0
            and 0 < sig["p_hbm_w"] < sig["power_avg_w"]
        )
    except Exception as exc:
        verdict["error_type"] = type(exc).__name__
        verdict["error"] = str(exc)[:500]
        verdict["trace_tail"] = traceback.format_exc()[-500:]
    print("P1_VERDICT " + json.dumps(verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "probe_matmul.py"))
