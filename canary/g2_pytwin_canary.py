# canary/g2_pytwin_canary.py
"""G2: pytwin이 이 환경에서 .twin을 evaluate할 수 있는지 기계 판정.
stdout 마지막 줄 'G2_VERDICT {json}' 이 판정값. exit 0=PASS, 1=FAIL."""
import json
import sys
import traceback


def main(twin_path: str) -> int:
    verdict = {"gate": "G2", "twin_path": twin_path, "passed": False, "stage": "import"}
    try:
        from pytwin import TwinModel
        verdict["stage"] = "load"
        model = TwinModel(twin_path)
        verdict["stage"] = "initialize"
        model.initialize_evaluation()
        verdict["stage"] = "step"
        model.evaluate_step_by_step(step_size=0.1)
        out = model.outputs
        verdict["outputs"] = {k: float(v) for k, v in out.items()} if isinstance(out, dict) else str(out)
        verdict["passed"] = True
    except Exception as exc:
        verdict["error_type"] = type(exc).__name__
        verdict["error"] = str(exc)[:500]
        verdict["trace_tail"] = traceback.format_exc()[-500:]
    print("G2_VERDICT " + json.dumps(verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/rc_thermal.twin"))
