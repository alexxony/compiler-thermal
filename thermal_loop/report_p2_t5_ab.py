"""HBM_build P2 Task 5 — P4·P7 ΔT A/B 재계산 + r_hbm_sink 범위 민감도.

design: /mnt/c/ObsidianVault/HBM_build/docs/06-p2-rc-backport-design.md Task 5
        (§2 "냉각 BC 불일치 처리" — r_hbm_sink 범위 3점 민감도로 방향 판정).
prereq: report_p4_deltat.py Task 4(rc_kw A/B 인터페이스, RC_KW_LEGACY/RC_KW_HBM_FEM)
        + HBM_build P2 Task 2 산출물(~/workspace/hbm_build/results/rc_params.csv).

RcBackend 클래스(thermal/twin_eval.py)는 무변경 — 이 모듈도 report_p4_deltat.py와
동일하게 호출측 rc_kw 파라미터화만 사용한다.

요구사항 3건(T5):
  1. A/B 재계산 — 4문제(matmul, kb_matmul_scalar, batched_gemm, kb_softmax) ×
     LEGACY/HBM_FEM 두 세트로 (b) ΔT gap 비교.
  2. r_hbm_sink 범위 민감도 — rc_params.csv의 실측 범위 [0.929032, 4.670561]에서
     3점(min/mid/대표=max) 스윕, (b) 방향(TF32<fp32) 뒤집힘(flipped) 여부 확인.
     대표값(4.670561)이 범위의 max와 같다(설계 §2: baseline_8hi 케이스, "보수적
     상한"으로 채택) — min≠대표이므로 3점은 {min, mid, 대표} 형태가 된다.
  3. flipped 판정 — 문제 간 (b) gap 순위, ON/OFF(TF32/fp32) 우열이 세트별로
     뒤집히는지 종합 판정.
"""
from __future__ import annotations

import csv
from pathlib import Path

import report_p4_deltat as p4

HBM_RC_PARAMS_CSV = p4.HBM_RC_PARAMS_CSV
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_r_hbm_sink_range(csv_path: Path = HBM_RC_PARAMS_CSV) -> tuple[float, float, float]:
    """rc_params.csv에서 r_hbm_sink의 (min, max, 대표=value)를 읽는다.

    T2 CSV 스키마: value=대표값(baseline_8hi), value_min/value_max=냉각 BC 두
    케이스의 범위. 현재 데이터는 대표값이 곧 value_max(설계 §2 "보수적 상한"으로
    baseline_8hi 채택) — 대표값이 항상 max와 같다고 가정하지 않고 CSV에서 그대로
    읽는다(향후 T2 재산출로 대표값 선택이 바뀌어도 이 함수는 안전).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HBM FEM RC 파라미터 CSV 없음: {csv_path} "
            "(HBM_build P2 T2 미완료 — rc_extract.py 실행 후 재시도)"
        )
    with open(csv_path, newline="") as f:
        rows = {row["parameter"]: row for row in csv.DictReader(f) if row.get("parameter")}
    row = rows["r_hbm_sink"]
    return float(row["value_min"]), float(row["value_max"]), float(row["value"])


def r_hbm_sink_sweep_points(csv_path: Path = HBM_RC_PARAMS_CSV) -> list[tuple[str, float]]:
    """3점 스윕 좌표 — (라벨, r_hbm_sink) 리스트.

    idle sensitivity(설계 §"idle sensitivity 동형 방식")와 동형: min / mid(산술
    평균) / 대표(=max, baseline_8hi) 3점. mid는 min·max 산술 평균 — 대표값이
    이미 max이므로 별도 "대표" 지점은 mid로 보강한다(설계 §3.5 표기 {0.929, 2.8,
    4.671}과 일치, mid=(0.929032+4.670561)/2=2.7998).
    """
    r_min, r_max, r_rep = load_r_hbm_sink_range(csv_path)
    r_mid = (r_min + r_max) / 2.0
    return [("min(cooling_top_bottom)", r_min),
            ("mid", r_mid),
            ("대표/max(baseline_8hi)", r_rep)]


def _rc_kw_with_r_hbm_sink(r_hbm_sink: float) -> dict:
    """HBM_FEM 세트에서 r_hbm_sink만 교체한 커스텀 rc_kw(c_hbm은 T2 실값 유지)."""
    base = dict(p4.RC_KW_HBM_FEM)
    base["r_hbm_sink"] = r_hbm_sink
    base["_fem_source"] = f"{base.get('_fem_source', '')} (r_hbm_sink override={r_hbm_sink})"
    return base


def ab_recompute(cfg: dict) -> dict:
    """config 1건 → LEGACY/HBM_FEM 두 세트의 (b) 결과 dict.

    데이터 갭(signals 부재)이 있으면 예외를 잡아 gap 표식으로 반환한다(요구사항
    "부재 문제는 데이터 갭 명시" — T5 스코프에서 새 실측은 하지 않는다).
    """
    problem = cfg["problem"]
    try:
        legacy = p4.run_problem(cfg, rc_kw=p4.RC_KW_LEGACY)
    except (FileNotFoundError, KeyError, StopIteration) as e:
        return {"problem": problem, "data_gap": True, "reason": str(e)}

    if p4.RC_KW_HBM_FEM["r_hbm_sink"] is None:
        return {"problem": problem, "data_gap": True,
                "reason": "RC_KW_HBM_FEM placeholder(T2 rc_params.csv 부재)"}

    fem = p4.run_problem(cfg, rc_kw=p4.RC_KW_HBM_FEM)
    return {"problem": problem, "data_gap": False, "legacy": legacy, "fem": fem}


def sensitivity_sweep(cfg: dict) -> dict:
    """r_hbm_sink 3점 스윕 — 각 점에서 (b) gap·방향(flipped) 계산.

    flipped: HBM_FEM 대표값 대비 방향(TF32<fp32)이 스윕 점에서 뒤집히는가.
    null 문제는 방향 판정 자체가 무의미하므로 스킵 표식만 남긴다.
    """
    problem = cfg["problem"]
    try:
        seed_probe = p4.run_problem(cfg, rc_kw=p4.RC_KW_LEGACY)
    except (FileNotFoundError, KeyError, StopIteration) as e:
        return {"problem": problem, "data_gap": True, "reason": str(e)}

    if seed_probe["is_null"]:
        return {"problem": problem, "data_gap": False, "is_null": True, "points": []}

    points = []
    for label, r_val in r_hbm_sink_sweep_points():
        rc_kw = _rc_kw_with_r_hbm_sink(r_val)
        res = p4.run_problem(cfg, rc_kw=rc_kw)
        gap = res["b_seed_t"] - res["b_best_t"]
        points.append({
            "label": label, "r_hbm_sink": r_val,
            "b_seed_t": res["b_seed_t"], "b_best_t": res["b_best_t"],
            "gap_k": gap, "direction_pass": res["b_verdict_pass"],
        })
    flipped = not all(p["direction_pass"] for p in points)
    return {"problem": problem, "data_gap": False, "is_null": False,
            "points": points, "flipped": flipped}


def rank_flip_check(ab_results: list[dict]) -> dict:
    """문제 간 (b) gap 순위가 LEGACY↔HBM_FEM 세트 교체로 뒤집히는지 확인.

    순위는 gap_k(ΔT_fp32-ΔT_TF32) 내림차순. null/데이터 갭 문제는 순위에서
    제외한다(순위 비교가 의미 있는 실개선 문제만 대상).
    """
    usable = [r for r in ab_results if not r["data_gap"] and not r["legacy"]["is_null"]]
    legacy_rank = sorted(usable, key=lambda r: r["legacy"]["b_gap_k"], reverse=True)
    fem_rank = sorted(usable, key=lambda r: r["fem"]["b_gap_k"], reverse=True)
    legacy_order = [r["problem"] for r in legacy_rank]
    fem_order = [r["problem"] for r in fem_rank]
    return {
        "legacy_order": legacy_order,
        "fem_order": fem_order,
        "rank_flipped": legacy_order != fem_order,
    }


def build_markdown_report(configs=None) -> str:
    """T5 산출물 — 4문제 LEGACY/HBM_FEM/민감도 3점 ΔT 표 + flipped 판정."""
    if configs is None:
        configs = p4.PROBLEM_CONFIGS

    lines = [
        "# P2 T5 — P4·P7 ΔT A/B 재계산 + r_hbm_sink 범위 민감도",
        "",
        "design: [06-p2-rc-backport-design.md](../../../ObsidianVault/HBM_build/"
        "docs/06-p2-rc-backport-design.md) Task 5",
        "",
        "RcBackend 클래스 무변경 — `report_p4_deltat.py` rc_kw 호출측 파라미터화만"
        " 사용(P2 Task 4 인터페이스). HBM_FEM 세트: r_hbm_sink=4.670561 K/W,"
        " c_hbm=0.124017 J/K (HBM_build T2 `rc_params.csv` 실값, die 3개는 legacy"
        " 유지).",
        "",
        "## 1. 문제별 LEGACY vs HBM_FEM (b) ΔT gap",
        "",
        "| 문제 | LEGACY gap(K) | HBM_FEM gap(K) | 방향(둘다 TF32<fp32?) | 비고 |",
        "|---|---|---|---|---|",
    ]

    ab_results = []
    for cfg in configs:
        res = ab_recompute(cfg)
        ab_results.append(res)
        if res["data_gap"]:
            lines.append(f"| {res['problem']} | — | — | — | **데이터 갭**: {res['reason']} |")
            continue
        legacy, fem = res["legacy"], res["fem"]
        if legacy["is_null"]:
            lines.append(f"| {res['problem']} | null | null | — | 개선 라운드 없음(seed=best) |")
            continue
        both_pass = bool(legacy["b_verdict_pass"] and fem["b_verdict_pass"])
        note = "일치" if legacy["b_verdict_pass"] == fem["b_verdict_pass"] else "**뒤집힘**"
        lines.append(
            f"| {res['problem']} | {legacy['b_gap_k']:.4f} | {fem['b_gap_k']:.4f} | "
            f"{'✓' if both_pass else '✗'} | 세트간 방향 {note} |"
        )

    lines += [
        "",
        "## 2. r_hbm_sink 범위 민감도 (3점: min/mid/대표)",
        "",
        f"범위 출처: HBM_build T2 `rc_params.csv` — "
        f"[{load_r_hbm_sink_range()[0]:.6f}, {load_r_hbm_sink_range()[1]:.6f}] K/W "
        f"(cooling_top_bottom / baseline_8hi 두 냉각 BC 케이스, 대표값=baseline_8hi=max).",
        "",
        "| 문제 | min gap(K) | mid gap(K) | 대표 gap(K) | flipped |",
        "|---|---|---|---|---|",
    ]

    sens_results = []
    for cfg in configs:
        sens = sensitivity_sweep(cfg)
        sens_results.append(sens)
        if sens["data_gap"]:
            lines.append(f"| {sens['problem']} | — | — | — | 데이터 갭 |")
            continue
        if sens["is_null"]:
            lines.append(f"| {sens['problem']} | null | null | null | — (개선 없음) |")
            continue
        gaps = {p["label"]: p["gap_k"] for p in sens["points"]}
        min_g = gaps.get("min(cooling_top_bottom)", float("nan"))
        mid_g = gaps.get("mid", float("nan"))
        rep_g = gaps.get("대표/max(baseline_8hi)", float("nan"))
        flipped_mark = "**예**" if sens["flipped"] else "아니오"
        lines.append(f"| {sens['problem']} | {min_g:.4f} | {mid_g:.4f} | {rep_g:.4f} | {flipped_mark} |")

    rank = rank_flip_check(ab_results)
    lines += [
        "",
        "## 3. 문제 간 순위 뒤집힘 (rank flip) 판정",
        "",
        f"- LEGACY 세트 gap 내림차순: {rank['legacy_order']}",
        f"- HBM_FEM 세트 gap 내림차순: {rank['fem_order']}",
        f"- 순위 뒤집힘: {'**예**' if rank['rank_flipped'] else '아니오'}",
        "",
        "## 4. 종합 해석",
        "",
    ]

    any_dir_flip = any(
        (not r["data_gap"]) and (not r["legacy"]["is_null"])
        and (r["legacy"]["b_verdict_pass"] != r["fem"]["b_verdict_pass"])
        for r in ab_results
    )
    any_sens_flip = any(
        (not s["data_gap"]) and (not s.get("is_null", True)) and s["flipped"]
        for s in sens_results
    )
    gaps_present = [r for r in ab_results if r["data_gap"]]

    if any_dir_flip or any_sens_flip or rank["rank_flipped"]:
        lines.append(
            "**결론: 방향 불변 아님** — 아래 근거 중 하나 이상에서 flipped 발생. "
            "\"Ansys FEM 캘리브레이션으로도 결론 불변\" 주장은 보류하고 원인 문제를 "
            "개별 재검토해야 한다."
        )
    else:
        lines.append(
            "**결론: 방향 불변** — LEGACY↔HBM_FEM 세트 교체, r_hbm_sink 범위 "
            "[min, mid, 대표] 3점 스윕, 문제 간 순위 어느 쪽에서도 (b) TF32<fp32 "
            "판정이 뒤집히지 않았다. \"Ansys FEM 캘리브레이션으로도 결론 불변\" "
            "주장을 추가할 수 있다(설계 문서 §목표)."
        )
    if gaps_present:
        lines.append(
            f"\n데이터 갭: {[r['problem'] for r in gaps_present]} — "
            "위 표에 개별 사유 명시, 신규 GPU 실측은 T5 스코프 밖."
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_markdown_report()
    print(report)
    out_path = RESULTS_DIR / "p2_t5_ab_report.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\n[report written to {out_path}]")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
