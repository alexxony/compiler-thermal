"""P9-A ncu 스팟체크 — seed vs variant 디스패치 커널명 비교 (원인 진단 도구).

배경: P8에서 compute-bound 문제들이 variant(TF32 플래그 반전) 적용에도 M1=1.00x로
gain이 0이었던 원인 진단. seed와 variant를 각각 ncu로 실행해 **디스패치 커널명**을
비교한다 — 커널명이 같으면 TF32 플래그가 하드웨어 수준에서 무효(환경이 플래그를
차단), 다르면(예: tf32/xmma 계열 커널이 등장) 플래그는 작동하나 gain이 없다는 뜻.

executor.py::_profile_ncu / _ncu_breakdown과 동일한 ncu 호출 관례를 따른다
(`--` 구분자 필수 — 없으면 --metrics를 실행파일로 오인, s2-105). CSV 파싱은
executor._parse_ncu_csv를 재사용(헤더 앞 배너 줄 스킵 관례 동일).

실행: python3 spot_ncu_kernels.py --problems <p1>[,<p2>...] [--root <repo_root>]
      [--out <result.json>] [--selfcheck]
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

# executor._parse_ncu_csv 재사용 — ncu CSV 파싱 방식(헤더 앞 배너 줄 스킵)을
# 이 파일에서 재구현하지 않는다(원본 파싱 로직과 어긋날 위험 방지).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor import _parse_ncu_csv  # noqa: E402

DEFAULT_OUT = "artifacts/p9a_spot_result.json"


def _find_root(explicit: str | None) -> Path:
    """repo 루트 결정. --root 명시 없으면 __file__ 기준 상위(= loop 루트).

    Colab 등 /content/loop 배치에서도 __file__ 기준 상위가 loop 루트이므로
    그대로 동작 — 별도 분기 불필요.
    """
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parent.parent


def _find_variant_file(problem_dir: Path) -> Path:
    """problems/<p>/variants/ 안의 유일한 .py 파일을 자동 탐지.

    2개 이상이면 어느 것을 비교해야 할지 모호하므로 에러(운영자가 명시적으로
    정리하도록 강제 — 조용히 첫 파일을 집는 폴백은 오진 위험).
    """
    vdir = problem_dir / "variants"
    if not vdir.is_dir():
        raise FileNotFoundError(f"variants 디렉토리 없음: {vdir}")
    py_files = sorted(vdir.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(f"variants/*.py 없음: {vdir}")
    if len(py_files) > 1:
        raise ValueError(
            f"variants/에 .py 파일이 {len(py_files)}개 — 유일해야 함: "
            f"{[p.name for p in py_files]}"
        )
    return py_files[0]


def _run_ncu(code_path: Path) -> dict:
    """ncu --csv로 code_path --profile 실행 → {'ok','kernels','durations_ns','stderr_tail'}.

    executor._profile_ncu/_ncu_breakdown과 동일한 커맨드 패턴(`--` 구분자 필수,
    timeout 600s). --print-kernel-base demangled로 사람이 읽을 수 있는 커널명 확보
    (_ncu_breakdown 관례와 동일).
    """
    cmd = [
        "ncu", "--csv", "--target-processes", "all",
        "--metrics", "gpu__time_duration.sum",
        "--print-kernel-base", "demangled",
        "--", sys.executable, str(code_path), "--profile",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "kernels": [], "durations_ns": {},
                "stderr_tail": f"timeout: {e}"}
    except FileNotFoundError as e:
        return {"ok": False, "kernels": [], "durations_ns": {},
                "stderr_tail": f"ncu 없음: {e}"}
    if r.returncode != 0:
        return {"ok": False, "kernels": [], "durations_ns": {},
                "stderr_tail": (r.stderr or "")[-800:]}
    rows = _parse_ncu_csv(r.stdout)
    if not rows:
        return {"ok": False, "kernels": [], "durations_ns": {},
                "stderr_tail": "ncu CSV 파싱 결과 0행 (헤더 미검출 또는 커널 미실행)"}
    kernels: list[str] = []          # 등장 순서 보존, 중복 카운트
    durations: dict[str, float] = {}  # 커널별 duration 합산 (ns)
    for row in rows:
        if "duration" not in row.get("Metric Name", "").lower():
            continue
        kname = row.get("Kernel Name") or row.get("ID") or "?"
        kernels.append(kname)
        try:
            dur = float(str(row.get("Metric Value", "0")).replace(",", ""))
        except ValueError:
            dur = 0.0
        durations[kname] = durations.get(kname, 0.0) + dur
    if not kernels:
        return {"ok": False, "kernels": [], "durations_ns": {},
                "stderr_tail": "duration 메트릭 행 없음"}
    return {"ok": True, "kernels": kernels, "durations_ns": durations, "stderr_tail": ""}


def _kernel_counts(kernels: list[str]) -> dict[str, int]:
    """등장 순서 보존한 유니크 이름 -> 호출 횟수."""
    counts: dict[str, int] = {}
    for k in kernels:
        counts[k] = counts.get(k, 0) + 1
    return counts


def _has_tf32_evidence(kernel_names: list[str]) -> bool:
    return any("tf32" in k.lower() for k in kernel_names)


def compare_dispatch(seed_run: dict, variant_run: dict, variant_file: str) -> dict:
    """seed/variant ncu 실행 결과를 비교·판정.

    verdict 규약:
      - ncu_error: 어느 한쪽이라도 ok=False
      - flag_ineffective_same_kernels: 유니크 커널명 집합 동일 (플래그가 하드웨어
        수준에서 무효 — 환경이 TF32 디스패치를 차단)
      - flag_effective_tf32_dispatch: 집합 다르고 variant 쪽에 tf32 마커 존재
        (플래그는 작동하나 gain이 없는 경우 — 다른 원인 조사 필요)
      - different_kernels_no_tf32_marker: 집합 다르지만 tf32 증거 없음 (플래그
        무관한 다른 변화 — 커널명만으로는 원인 불명)
    """
    if not seed_run.get("ok") or not variant_run.get("ok"):
        tail = seed_run.get("stderr_tail") or variant_run.get("stderr_tail") or ""
        return {
            "seed_kernels": {}, "variant_kernels": {}, "variant_file": variant_file,
            "same_dispatch": None, "tf32_evidence": None,
            "verdict": "ncu_error", "stderr_tail": tail,
        }
    seed_kernels = _kernel_counts(seed_run["kernels"])
    variant_kernels = _kernel_counts(variant_run["kernels"])
    same_dispatch = set(seed_kernels) == set(variant_kernels)
    tf32_evidence = _has_tf32_evidence(list(variant_kernels))
    if same_dispatch:
        verdict = "flag_ineffective_same_kernels"
    elif tf32_evidence:
        verdict = "flag_effective_tf32_dispatch"
    else:
        verdict = "different_kernels_no_tf32_marker"
    return {
        "seed_kernels": seed_kernels, "variant_kernels": variant_kernels,
        "variant_file": variant_file, "same_dispatch": same_dispatch,
        "tf32_evidence": tf32_evidence, "verdict": verdict, "stderr_tail": "",
    }


def spot_check_problem(root: Path, problem: str) -> dict:
    """문제 1개에 대한 seed/variant ncu 스팟체크 실행."""
    problem_dir = root / "problems" / problem
    seed_path = problem_dir / "solve.py"
    if not seed_path.exists():
        return {"problem": problem, "verdict": "ncu_error",
                "stderr_tail": f"seed 없음: {seed_path}"}
    try:
        variant_path = _find_variant_file(problem_dir)
    except (FileNotFoundError, ValueError) as e:
        return {"problem": problem, "verdict": "ncu_error", "stderr_tail": str(e)}

    seed_run = _run_ncu(seed_path)
    variant_run = _run_ncu(variant_path)
    result = compare_dispatch(seed_run, variant_run, variant_path.name)
    result["problem"] = problem
    result["seed_duration_ns_total"] = sum(seed_run.get("durations_ns", {}).values())
    result["variant_duration_ns_total"] = sum(variant_run.get("durations_ns", {}).values())
    return result


def run_spot_check(root: Path, problems: list[str]) -> dict:
    return {p: spot_check_problem(root, p) for p in problems}


# ── selfcheck: fake ncu CSV 2케이스(동일/상이) — 실제 ncu 호출 없음. ──
_FAKE_CSV_SEED = '''"ID","Kernel Name","Metric Name","Metric Value"
"0","ampere_sgemm_128x64_nn","gpu__time_duration.sum","120000"
"1","ampere_sgemm_128x64_nn","gpu__time_duration.sum","121000"
'''

_FAKE_CSV_VARIANT_TF32 = '''"ID","Kernel Name","Metric Name","Metric Value"
"0","ampere_h1688gemm_128x64_tf32_nn","gpu__time_duration.sum","40000"
"1","ampere_h1688gemm_128x64_tf32_nn","gpu__time_duration.sum","41000"
'''

_FAKE_CSV_VARIANT_SAME = '''"ID","Kernel Name","Metric Name","Metric Value"
"0","ampere_sgemm_128x64_nn","gpu__time_duration.sum","119000"
"1","ampere_sgemm_128x64_nn","gpu__time_duration.sum","118000"
'''


def _parse_fake(csv_text: str) -> dict:
    """selfcheck 전용 — 실제 ncu 실행 없이 CSV 문자열을 _run_ncu와 같은 형태로 파싱."""
    rows = _parse_ncu_csv(csv_text)
    kernels: list[str] = []
    durations: dict[str, float] = {}
    for row in rows:
        if "duration" not in row.get("Metric Name", "").lower():
            continue
        kname = row.get("Kernel Name") or "?"
        kernels.append(kname)
        durations[kname] = durations.get(kname, 0.0) + float(row.get("Metric Value", "0"))
    return {"ok": True, "kernels": kernels, "durations_ns": durations, "stderr_tail": ""}


def _selfcheck() -> None:
    # 케이스 1: 커널명 상이 + tf32 마커 존재 → flag_effective_tf32_dispatch
    seed = _parse_fake(_FAKE_CSV_SEED)
    variant_tf32 = _parse_fake(_FAKE_CSV_VARIANT_TF32)
    r1 = compare_dispatch(seed, variant_tf32, "R_tf32on.py")
    assert r1["same_dispatch"] is False, r1
    assert r1["tf32_evidence"] is True, r1
    assert r1["verdict"] == "flag_effective_tf32_dispatch", r1
    assert r1["seed_kernels"] == {"ampere_sgemm_128x64_nn": 2}, r1
    assert r1["variant_kernels"] == {"ampere_h1688gemm_128x64_tf32_nn": 2}, r1

    # 케이스 2: 커널명 동일 → flag_ineffective_same_kernels
    variant_same = _parse_fake(_FAKE_CSV_VARIANT_SAME)
    r2 = compare_dispatch(seed, variant_same, "R_tf32on.py")
    assert r2["same_dispatch"] is True, r2
    assert r2["tf32_evidence"] is False, r2
    assert r2["verdict"] == "flag_ineffective_same_kernels", r2

    # 케이스 3: ncu 실패(ok=False) → ncu_error, stderr_tail 보존
    err_run = {"ok": False, "kernels": [], "durations_ns": {}, "stderr_tail": "boom"}
    r3 = compare_dispatch(err_run, variant_tf32, "R_tf32on.py")
    assert r3["verdict"] == "ncu_error", r3
    assert r3["stderr_tail"] == "boom", r3

    # 케이스 4: 커널명 상이 + tf32 마커 없음 → different_kernels_no_tf32_marker
    other = {"ok": True, "kernels": ["some_other_kernel"],
              "durations_ns": {"some_other_kernel": 1000.0}, "stderr_tail": ""}
    r4 = compare_dispatch(seed, other, "R_x.py")
    assert r4["same_dispatch"] is False, r4
    assert r4["tf32_evidence"] is False, r4
    assert r4["verdict"] == "different_kernels_no_tf32_marker", r4

    print("selfcheck PASS: 4케이스(tf32 유효/무효/ncu_error/tf32무관 상이) 판정 로직 확인")


def main() -> int:
    ap = argparse.ArgumentParser(description="P9-A ncu 스팟체크 — seed vs variant 디스패치 커널명 비교")
    ap.add_argument("--problems", type=str, default="",
                     help="콤마 구분 문제 디렉토리명")
    ap.add_argument("--root", type=str, default=None,
                     help="repo 루트 (기본: __file__ 기준 상위 = loop 루트)")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                     help="결과 JSON 저장 경로")
    ap.add_argument("--selfcheck", action="store_true",
                     help="GPU/ncu/torch 없이 파싱·비교 로직만 검증")
    args = ap.parse_args()

    if args.selfcheck:
        try:
            _selfcheck()
        except AssertionError as e:
            print(f"selfcheck FAIL: {e}", file=sys.stderr)
            return 1
        return 0

    if not args.problems:
        print("ERR: --problems 필요 (콤마 구분 문제 디렉토리명)", file=sys.stderr)
        return 2

    root = _find_root(args.root)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    result = run_spot_check(root, problems)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[out] {out_path}")

    print("__SPOT_RESULT__" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
