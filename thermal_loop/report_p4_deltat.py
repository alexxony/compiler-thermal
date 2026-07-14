"""P4 Task 16 → P7 Task 25~29 — RcBackend ΔT 리포트, 다문제 일반화.

design: docs/04-p4-rc-deltat-design.md §5 Task 16 (원본, matmul 단일);
        docs/07-p7-deltat-multiproblem-design.md §3~5 (다문제 일반화).
Colab 실행 아님 — 로컬 배치, GPU/Colab 세션 불요. 실측 없이 이미 있는 P3/P6 raw
신호만 후처리. 물리·측정 로직(duty_power/twin_eval)은 무변경, 여기선 로더 어댑터 +
문제 리스트 루프만 추가한다(설계 §3-2).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thermal.duty_power import to_power_series, duty_avg_power  # noqa: F401
from thermal.twin_eval import RcBackend

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# RcBackend 기존 검증 파라미터(tests/test_twin_eval.py make_backend()와 동일) —
# 계획 §2 "기본안: 기존 검증 파라미터 그대로 사용" 결정 그대로.
RC_KW = dict(r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
             c_die=50.0, c_hbm=10.0, t_ambient=45.0)

# 계획 §3 idle 결정(팀리드 승인): 문헌치 A100 idle ≈ 55W. HBM 쪽은 die의 1/10 근사
# (정밀치 미확보 — sensitivity 테스트(Task 15)가 이 가정의 민감도를 별도 검증).
IDLE_DIE_W = 55.0
IDLE_HBM_W = 5.5

WINDOW_S = 60.0  # test_twin_eval.py 60s 창(시정수~6.7s의 9배) 그대로 상속

# P7 §4 Task 26: 문제별 (포맷, 경로, 트랙) config 테이블. matmul=P3 JSONL,
# kb_matmul_scalar=P6 __ABL_RESULT__ JSON. batched_gemm/kb_softmax는 §2 대안 A
# 재실측 결과 JSON 도착 시 여기 항목 추가만으로 확장(fmt="ablresult", 경로·트랙 지정).
_BGEMM_SOFTMAX = str(ARTIFACTS / "p7_bgemm_softmax_20260714_result.json")
PROBLEM_CONFIGS = [
    {"problem": "matmul", "fmt": "p3_jsonl",
     "path": str(ARTIFACTS / "thermal-gain-matmul-off.jsonl"), "track": "off"},
    {"problem": "kb_matmul_scalar", "fmt": "ablresult",
     "path": str(ARTIFACTS / "p6_kbms_retry_20260713_result.json"), "track": "on"},
    # P7 Task 30: §2 대안 A 재실측 회수분(P6 포맷). batched_gemm=개선 2R,
    # kb_softmax=개선 없음 1R(seed=best → Task 28 null 경로).
    {"problem": "batched_gemm", "fmt": "ablresult",
     "path": _BGEMM_SOFTMAX, "track": "on"},
    {"problem": "kb_softmax", "fmt": "ablresult",
     "path": _BGEMM_SOFTMAX, "track": "on"},
]


def _load_round(path: Path, round_idx: int) -> dict:
    """P3 라운드별 JSONL 포맷 로더(무변경 — P4 앵커 경로)."""
    with open(path) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    rec = next(r for r in lines if r["round_idx"] == round_idx)
    sig = rec["signal"]
    return {"p_die_w": sig["p_die_w"], "p_hbm_w": sig["p_hbm_w"],
            "kernel_time_s": sig["kernel_time_s"],
            "energy_per_iter_j": sig.get("energy_per_iter_j", 0.0),
            "hypothesis_label": rec["hypothesis_label"]}


def _load_round_from_ablresult(json_path, problem: str, track: str,
                               selector: str) -> dict:
    """P7 Task 25 — P6 `__ABL_RESULT__` 파싱본 로더.

    구조: results[problem][track].signals[] (라운드별 Signal 전체 dict, P6 §2-3).
    selector="seed": signals[0](최초 라운드).
    selector="best": energy_per_iter_j **최소** 라운드 — 라벨이 아니라 값 기준
      (P6 §7-2 라벨 오프셋 주의: 라벨은 다음 라운드 가설이라 오독 위험).
    """
    if selector not in ("seed", "best"):
        raise ValueError(f"알 수 없는 selector: {selector!r} (seed|best)")
    with open(json_path) as f:
        doc = json.load(f)
    signals = doc["results"][problem][track]["signals"]
    if not signals:
        raise ValueError(f"signals 비어있음: {problem}/{track}")
    if selector == "seed":
        sig = signals[0]
    else:  # best = energy_per_iter_j 최소 라운드
        sig = min(signals, key=lambda s: s["energy_per_iter_j"])
    return {"p_die_w": sig["p_die_w"], "p_hbm_w": sig["p_hbm_w"],
            "kernel_time_s": sig["kernel_time_s"],
            "energy_per_iter_j": sig.get("energy_per_iter_j", 0.0),
            "hypothesis_label": sig.get("op_name", selector)}


def _load_pair(cfg: dict) -> tuple[dict, dict]:
    """config 1건 → (seed, best) 순간전력 dict 페어(포맷 어댑터 분기)."""
    fmt = cfg["fmt"]
    if fmt == "p3_jsonl":
        path = Path(cfg["path"])
        # P4 규약: round 0 = fp32 seed, round 1 = TF32 best (matmul 앵커).
        return _load_round(path, 0), _load_round(path, 1)
    if fmt == "ablresult":
        seed = _load_round_from_ablresult(cfg["path"], cfg["problem"],
                                          cfg["track"], "seed")
        best = _load_round_from_ablresult(cfg["path"], cfg["problem"],
                                          cfg["track"], "best")
        return seed, best
    raise ValueError(f"알 수 없는 fmt: {fmt!r}")


def _steady_t_hbm(power_df) -> float:
    out = RcBackend(**RC_KW).evaluate(power_df)
    return out["t_hbm_c"].iloc[-1]


def _scenario_a(sig: dict, idle_die_w: float = IDLE_DIE_W,
                idle_hbm_w: float = IDLE_HBM_W) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="saturated", window_s=WINDOW_S,
                              idle_die_w=idle_die_w, idle_hbm_w=idle_hbm_w)
    return _steady_t_hbm(series)


def _scenario_b(sig: dict, repeat_period_s: float,
                idle_die_w: float = IDLE_DIE_W,
                idle_hbm_w: float = IDLE_HBM_W) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="iso_work", window_s=WINDOW_S,
                              idle_die_w=idle_die_w, idle_hbm_w=idle_hbm_w,
                              repeat_period_s=repeat_period_s)
    return _steady_t_hbm(series)


def _is_null(seed: dict, best: dict) -> bool:
    """개선 라운드 없음(seed=best) → null 대조군(설계 §3-2/§5 Task 28).

    energy_per_iter_j가 동률이면 (a)/(b)가 같은 입력이 되어 ΔT gap≈0 — 방향 판정
    무의미. 미세 부동소수 차이는 상대오차로 흡수.
    """
    se, be = seed["energy_per_iter_j"], best["energy_per_iter_j"]
    if se == 0.0 and be == 0.0:
        return True
    denom = max(abs(se), abs(be), 1e-12)
    return abs(se - be) / denom < 1e-9


def run_problem_from_sigs(problem: str, seed: dict, best: dict,
                          idle_die_w: float = IDLE_DIE_W,
                          idle_hbm_w: float = IDLE_HBM_W) -> dict:
    """seed/best 순간전력 페어 → (a)/(b) t_hbm·ΔT·판정 dict (Task 27/28 코어).

    seed=fp32(비교 기준), best=TF32(개선안). repeat_period_s는 seed(fp32)
    kernel_time_s(설계 §3-2 규약 — "같은 처리량 기한 공유"). null이면 판정 skip.
    """
    null = _is_null(seed, best)
    a_seed = _scenario_a(seed, idle_die_w, idle_hbm_w)
    a_best = _scenario_a(best, idle_die_w, idle_hbm_w)
    repeat_period_s = seed["kernel_time_s"]
    b_seed = _scenario_b(seed, repeat_period_s, idle_die_w, idle_hbm_w)
    b_best = _scenario_b(best, repeat_period_s, idle_die_w, idle_hbm_w)
    duty_best = best["kernel_time_s"] / repeat_period_s
    energy_ratio = (seed["energy_per_iter_j"] / best["energy_per_iter_j"]
                    if best["energy_per_iter_j"] else float("nan"))

    ta = RC_KW["t_ambient"]
    res = {
        "problem": problem,
        "is_null": null,
        "seed": seed, "best": best,
        "a_seed_t": a_seed, "a_best_t": a_best,
        "b_seed_t": b_seed, "b_best_t": b_best,
        "a_seed_dt": a_seed - ta, "a_best_dt": a_best - ta,
        "b_seed_dt": b_seed - ta, "b_best_dt": b_best - ta,
        "b_gap_k": b_seed - b_best,          # ΔT_fp32 - ΔT_TF32 (양수=개선)
        "duty_best": duty_best,
        "energy_ratio": energy_ratio,
        "idle_die_w": idle_die_w, "idle_hbm_w": idle_hbm_w,
    }
    # §5 1차 기준: (b)에서 best(TF32) < seed(fp32)면 방향 일치(성공). null은 skip.
    res["b_verdict_pass"] = None if null else bool(b_best < b_seed)
    res["a_tf32_ge"] = None if null else bool(a_best >= a_seed)
    return res


def run_problem(cfg: dict) -> dict:
    """config 1건 로드 → run_problem_from_sigs (Task 26 루프 단위)."""
    seed, best = _load_pair(cfg)
    return run_problem_from_sigs(cfg["problem"], seed, best)


def idle_sensitivity(cfg: dict, factors=(0.7, 1.0, 1.3)) -> list[dict]:
    """P7 Task 29 — idle_die_w ×factors 3점에서 (b) 방향 불변 확인.

    각 점에서 run_problem_from_sigs를 idle 스케일해 재실행. null 문제는 방향 판정
    대신 is_null 표식만. 방향이 뒤집히면 flipped=True(문제 승격 조건, 설계 §3-3).
    """
    seed, best = _load_pair(cfg)
    points = []
    for f in factors:
        res = run_problem_from_sigs(cfg["problem"], seed, best,
                                    idle_die_w=IDLE_DIE_W * f,
                                    idle_hbm_w=IDLE_HBM_W * f)
        flipped = (not res["is_null"]) and (res["b_best_t"] >= res["b_seed_t"])
        points.append({
            "idle_factor": f,
            "is_null": res["is_null"],
            "b_seed_t": res["b_seed_t"], "b_best_t": res["b_best_t"],
            "flipped": flipped,
        })
    return points


def format_problem_result(res: dict) -> str:
    """문제 1건 결과 → 리포트 텍스트 섹션(Task 26/27/28 출력)."""
    p = res["problem"]
    lines = [f"\n{'='*60}", f"문제: {p}", "=" * 60]
    seed, best = res["seed"], res["best"]
    lines.append(
        f"  seed(fp32): p_die={seed['p_die_w']:.2f}W p_hbm={seed['p_hbm_w']:.3f}W "
        f"kt={seed['kernel_time_s']*1e6:.1f}us e/it={seed['energy_per_iter_j']:.5f}J")
    lines.append(
        f"  best(TF32): p_die={best['p_die_w']:.2f}W p_hbm={best['p_hbm_w']:.3f}W "
        f"kt={best['kernel_time_s']*1e6:.1f}us e/it={best['energy_per_iter_j']:.5f}J")

    if res["is_null"]:
        lines.append("  → null 대조군(seed=best, 개선 라운드 없음) — "
                     "(a)/(b) 동일 입력, ΔT 대조 무의미. 방향 판정 skip.")
        return "\n".join(lines)

    lines.append(f"  energy_per_iter_j 개선 배율: {res['energy_ratio']:.2f}x")
    lines.append(
        f"  (a) 포화: fp32 ΔT={res['a_seed_dt']:.4f}K  TF32 ΔT={res['a_best_dt']:.4f}K  "
        f"→ {'TF32≥fp32(열 불리, 예상대로)' if res['a_tf32_ge'] else 'TF32<fp32(문제 의존)'}")
    lines.append(f"  (b) iso-work(duty={res['duty_best']*100:.2f}%): "
                 f"fp32 ΔT={res['b_seed_dt']:.4f}K  TF32 ΔT={res['b_best_dt']:.4f}K")
    verdict = ("★ TF32 < fp32 (energy 개선과 방향 일치)" if res["b_verdict_pass"]
               else "⚠️ TF32 ≥ fp32 (energy와 어긋남 — 조사)")
    lines.append(f"  (b) 판정: {verdict}   gap(ΔT_fp32-ΔT_TF32)={res['b_gap_k']:.4f}K")
    return "\n".join(lines)


def build_report(configs) -> str:
    """전체 config 루프 → 리포트 문자열(Task 26)."""
    parts = ["=" * 60,
             "P7 RcBackend ΔT 다문제 리포트 — (a)포화 vs (b)iso-work",
             "=" * 60,
             f"idle_die_w={IDLE_DIE_W}W idle_hbm_w={IDLE_HBM_W}W (문헌치, 팀리드 승인)"]
    for cfg in configs:
        res = run_problem(cfg)
        parts.append(format_problem_result(res))
        # Task 29 idle sensitivity 3점 요약
        pts = idle_sensitivity(cfg)
        if not res["is_null"]:
            inv = all(not q["flipped"] for q in pts)
            summ = "  idle sensitivity(×0.7/×1.0/×1.3): (b) 방향 " + (
                "불변 ✓" if inv else "⚠️ 뒤집힘 — idle 정밀측정 승격 조건 발동")
            parts.append(summ)
    return "\n".join(parts)


def main() -> int:
    for cfg in PROBLEM_CONFIGS:
        if not Path(cfg["path"]).exists():
            print(f"ERR: 없음 {cfg['path']} — 실측 결과 필요", file=sys.stderr)
            return 2
    print(build_report(PROBLEM_CONFIGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
