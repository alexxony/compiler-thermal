"""ablation 원격 배치 드라이버 — ON/OFF 트랙 전체를 colab exec 1방으로 (A안).

문제(오늘 실측): run_gain_hypcond는 라운드마다 colab exec 왕복 → 12+7라운드에
벽시계 ~30분, 전송 스톨 노출 19회, GPU 유휴 99%. 룰 엔진은 결정론(LLM 0)이고
hypcond 콜백도 순수 dict 매핑이라 원격 실행에 제약 없음.

해결: loop 모듈 10종(전부 순수 파이썬)을 /content/loop에 배포하고, 문제 N개 ×
ON/OFF 트랙 전체를 **단일 colab exec**로 원격 실행. 라운드별 진행은 stdout,
최종 결과는 `__ABL_RESULT__<json>` + /content/abl_result.json(전송 유실 대비 이중화).
로컬은 결과 JSON을 TrackResult로 복원해 기존 _report로 판정 출력.

효과: 왕복 19회→1회, 세션 체류 ~30분→측정시간만, 전송 스톨 노출 1회.
LLM 대행 generate가 필요한 run_gain_round는 범위 밖(왕복 필수) — 이건 ablation 전용.

실행: python3 run_ablation_remote.py <problem>[,<problem>...] [max_rounds] --session=<s>
      [--skip-upload]  (모듈 이미 배포된 세션 재사용 시)
      [--resume=<prev_result.json>]  (P8 Task 34 — 완결 문제 skip, 잔여만 재실행)
      [--selfcheck]    (GPU·colab 0 — 원격 드라이버 생성·컴파일 검증만)
variant 규약은 run_gain_hypcond와 동일: matmul=R_tf32on, 그 외=R_tf32+R_coalesced.

P8 Task 33(§4-1) — 결과 회수는 `/content/abl_result.json` 파일 다운로드가 주 경로
(대규모 35~200문제서 signals 포함 시 단일 stdout 라인이 truncate될 위험 — §4-1 표
"결과 전송" 지점). stdout `__ABL_RESULT__` 파싱은 파일 회수 실패 시 폴백으로만.
P8 Task 34 — `--resume=` 지정 시 이전 배치 결과 JSON에서 완결(off+on 모두 존재)
문제를 호출 전 필터링(`filter_uncompleted`) — 배치 세션사 후 재실행 시 완료분
재계산 방지. `merge_batch_results`로 여러 배치 결과 JSON을 문제 단위 병합 가능
(report_p8_stats.py 등 리포터 입력 준비용).
"""
from __future__ import annotations
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from run_e2e import PROBLEMS
from run_gain_compare import TrackResult, _report

# P8 Task 33/34 (08-p8-scale-ablation-design.md §4-1) — 대규모 배치 내구성.
# 신규 인프라 없음: 기존 _fetch_result_file/ledger 배선 위에 얇은 래퍼만 추가.

LOOP_DIR = Path(__file__).resolve().parent
THERMAL_DIR = LOOP_DIR.parent / "thermal"
REMOTE_DIR = "/content/loop"
# runner가 import하는 순수 모듈 전부 (executor 포함 — LocalProfiler가 직결).
MODULES = ("signals.py", "rules.py", "evolver.py", "ledger.py", "glue.py",
           "generator.py", "mailbox.py", "harness.py", "runner.py", "executor.py")
# thermal 축(metric_mode="thermal")일 때만 추가 배포 — executor._profile_thermal이
# import하는 6파일. P3 setup_remote(thermal=True)가 한 번 빠뜨렸던 실수(커밋
# d529093)를 여기서도 반복하지 않도록 P5 Task 20에서 명시적으로 추가(설계 §1-2).
THERMAL_MODULES = ("__init__.py", "chip_caps.py", "hbm_split.py",
                   "power_sampler.py", "measure.py", "twin_eval.py")

# ── 원격 드라이버 템플릿: 센티널 치환(.replace) — 코드 내 중괄호와 무충돌. ──
_REMOTE_TEMPLATE = r'''
import sys, json, importlib
sys.path.insert(0, __REMOTE_DIR__)
importlib.invalidate_caches()

CONFIG = json.loads(__CONFIG_JSON__)

import executor
from glue import ProfileResult, GateResult
from mailbox import MailboxResult
from runner import run_problem
from rules import seed_rules
from ledger import Ledger
from generator import CallbackGenerator
from signals import Context


class LocalProfiler:
    """submit → executor.execute_request 직결 (같은 프로세스, 왕복 0).

    kind=None(기본, 회귀 0) = latency 축 그대로. kind="thermal"이면 cmd에 실려
    executor._profile_thermal로 분기(ncu 트래픽 런 + power 런 분리 실행,
    P3에서 검증된 배선 — thermal_loop/colab_profiler.ColabExecProfiler.submit과
    동일 규약, P5 Task 19).
    """
    def submit(self, code, problem, profile_opts=None, kind=None):
        cmd = {"id": "abl", "problem": problem, "code": code,
               "profile_opts": profile_opts or {"ncu": True}}
        if kind:
            cmd["kind"] = kind
        res = executor.execute_request(cmd)
        sig = dict(res.get("signal_dict") or {})
        lat = res.get("latency_us", sig.get("latency_us", 0.0))
        return MailboxResult(
            profile=ProfileResult(sig, lat),
            gate=GateResult(bool(res.get("passed", False)),
                            float(res.get("max_abs_err", 0.0))),
            error=res.get("error"))

    def profile(self, code, problem):
        return self.submit(code, problem).profile


def _make_cb(seed_code, variant_map):
    def cb(problem, hyp, prev_code):
        label = hyp if isinstance(hyp, str) else None
        code = variant_map.get(label, seed_code) if label else seed_code
        which = label if (label in variant_map) else "seed(base)"
        print(f"  [generate] hyp={hyp!r} -> {which}", flush=True)
        return code
    return cb


def _run_track(problem, seed_code, variant_map, max_rounds, evolve_enabled,
               ledger_path, ctx, metric_mode="latency"):
    import os
    if os.path.exists(ledger_path):    # 이전 런 잔재가 곡선에 섞이는 것 방지
        os.unlink(ledger_path)
    rules = seed_rules()
    gen = CallbackGenerator(_make_cb(seed_code, variant_map))
    label = "evolve_ON" if evolve_enabled else "evolve_OFF"
    print(f"== {problem} / {label} (metric={metric_mode}) ==", flush=True)
    submit_kind = "thermal" if metric_mode == "thermal" else None
    res = run_problem(problem, seed_code, "/content/mb-unused", ledger_path,
                      sync_fn=lambda _p: None, max_rounds=max_rounds,
                      rules=rules, generator=gen, evolve_enabled=evolve_enabled,
                      metric_mode=metric_mode, ctx=ctx, profiler=LocalProfiler(),
                      submit_kind=submit_kind)
    led = Ledger(ledger_path)
    recs = [r for r in led.records if r.problem == problem]
    return {
        "label": label,
        "metric_curve": led.metric_curve(problem),
        "fired_rules": [r.hypothesis_label for r in recs],
        "wasted_rounds": sum(1 for r in recs if r.passed and not r.improved),
        "retire_count": sum(1 for e in res.events if e.kind == "retire"),
        "stop_reason": res.stopped_reason,
        "rounds": res.rounds,
        # P6: signal_dict 라운드별 보존 (추출 방식, 06-p6-signal-provenance-design §2-3).
        # gate_fail 라운드는 r.signal={}(harness.py의 RoundRecord(..., {}, "gate_fail", ...))
        # 그대로 실림 — 소비측(역직렬화/판정)이 빈 dict를 방어적으로 다뤄야 함.
        "signals": [r.signal for r in recs],
    }


def main():
    try:
        chip = executor._detect_chip_now()
    except Exception:
        chip = ""
    ctx = Context(chip=chip) if chip else None
    metric_mode = CONFIG.get("metric_mode", "latency")
    print(f"[remote] chip={chip!r} problems={list(CONFIG['problems'])} "
          f"max_rounds={CONFIG['max_rounds']} metric_mode={metric_mode!r}", flush=True)

    out = {"chip": chip, "metric_mode": metric_mode, "results": {}}
    for name, spec in CONFIG["problems"].items():
        vmap = spec["variant_map"]
        out["results"][name] = {
            "off": _run_track(name, spec["seed"], vmap, CONFIG["max_rounds"],
                              False, f"/content/abl-{name}-off.jsonl", ctx,
                              metric_mode=metric_mode),
            "on": _run_track(name, spec["seed"], vmap, CONFIG["max_rounds"],
                             True, f"/content/abl-{name}-on.jsonl", ctx,
                             metric_mode=metric_mode),
        }
        with open("/content/abl_result.json", "w") as f:   # 중간에도 덮어씀
            json.dump(out, f)
    print("__ABL_RESULT__" + json.dumps(out), flush=True)


main()
'''


def _variant_map_for(problem: str) -> dict[str, str]:
    """발화 룰 라벨 → 충실 variant 코드. **존재하는 파일만** 매핑(자연 다문제 전제).

    variant 없는 룰이 발화하면 콜백이 seed로 폴백 — STOP 룰(memory_saturated 등)로
    정지하는 문제는 variant 0개여도 배치에 태울 수 있음. 파일명 규약:
    R_tf32(on).py=fp32_no_tensorcore, R_coalesced.py=uncoalesced, R_fused.py=memory_bound_fusable.
    """
    vdir = PROBLEMS / problem / "variants"
    name_map = {
        "R_tf32on.py": "fp32_no_tensorcore",   # matmul형 (TF32 실효)
        "R_tf32.py": "fp32_no_tensorcore",     # scalar형 (null)
        "R_coalesced.py": "uncoalesced",
        "R_fused.py": "memory_bound_fusable",
    }
    vmap: dict[str, str] = {}
    for fname, label in name_map.items():
        p = vdir / fname
        if p.exists() and label not in vmap:
            vmap[label] = p.read_text()
    return vmap


_MATMUL_CALL_NAMES = {"matmul", "bmm", "mm"}  # torch.matmul / torch.bmm / torch.mm


def _reference_uses_matmul(tree: ast.AST) -> bool:
    """reference() 함수 본문에 행렬곱(torch.matmul/bmm/mm 또는 @ 연산자)이 있는지."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "reference":
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult):
                    return True
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in _MATMUL_CALL_NAMES):
                    return True
    return False


def check_tf32_guard(problem: str) -> str | None:
    """P5 kb_matmul_scalar gate_fail 재발 방지 정적 체크 (torch 불필요, 소스 스캔만).

    reference()가 행렬곱을 쓰는데 파일 어디에도 allow_tf32 언급이 없으면 경고 문자열
    반환(A100 Colab의 allow_tf32 기본값 True가 reference를 오염시켜 correctness gate가
    커널이 아니라 reference의 TF32 오차를 잡는 사고 — kb_matmul_scalar 실측 2026-07-12
    참조). 문제없으면 None.
    """
    solve_path = PROBLEMS / problem / "solve.py"
    if not solve_path.exists():
        return None
    src = solve_path.read_text()
    tree = ast.parse(src, filename=str(solve_path))
    if not _reference_uses_matmul(tree):
        return None  # 행렬곱 없는 문제(예: kb_softmax)는 해당 없음
    if "allow_tf32" in src:
        return None  # 모듈 top-level이든 reference() 내부든 방어 존재
    return (f"{problem}/solve.py: reference()가 행렬곱을 쓰는데 allow_tf32 방어가 "
            f"없음 — A100에서 reference가 TF32로 오염될 수 있음 (kb_matmul_scalar "
            f"max_err=3.00e-02 사고 재현 위험)")


def check_all_tf32_guards() -> list[str]:
    """problems/ 전체를 스캔해 TF32 방어 누락 경고 목록 반환."""
    warnings = []
    for p in sorted(d.name for d in PROBLEMS.iterdir() if d.is_dir()):
        w = check_tf32_guard(p)
        if w:
            warnings.append(w)
    return warnings


def _build_remote_driver(problems: list[str], max_rounds: int,
                         metric_mode: str = "latency") -> str:
    config = {"max_rounds": max_rounds, "metric_mode": metric_mode, "problems": {}}
    for p in problems:
        seed = (PROBLEMS / p / "solve.py").read_text()
        config["problems"][p] = {"seed": seed, "variant_map": _variant_map_for(p)}
    return (_REMOTE_TEMPLATE
            .replace("__REMOTE_DIR__", repr(REMOTE_DIR))
            .replace("__CONFIG_JSON__",
                     repr(json.dumps(config, ensure_ascii=False))))


def _colab(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _retry(what: str, fn, attempts: int = 3):
    """세션 웜업 직후 HTTP ReadTimeout 등 전송 플레이크 흡수."""
    last = ""
    for i in range(1, attempts + 1):
        try:
            p = fn()
        except subprocess.TimeoutExpired:
            last = f"{what}: subprocess timeout (attempt {i}/{attempts})"
            print(f"  [deploy] {last} — 재시도", file=sys.stderr)
            continue
        if p.returncode == 0:
            return p
        last = f"{what}: rc={p.returncode} {p.stderr[-200:]} (attempt {i}/{attempts})"
        print(f"  [deploy] {last} — 재시도", file=sys.stderr)
    raise RuntimeError(f"재시도 소진 — {last}")


def _upload_modules(session: str, thermal: bool = False) -> None:
    mk = (f"import os; os.makedirs({REMOTE_DIR!r}, exist_ok=True); "
          f"os.makedirs({REMOTE_DIR!r} + '/thermal', exist_ok=True); print('MKDIR OK')"
          if thermal else
          f"import os; os.makedirs({REMOTE_DIR!r}, exist_ok=True); print('MKDIR OK')")
    _retry("mkdir", lambda: subprocess.run(
        ["colab", "exec", "-s", session, "--timeout", "120"], input=mk,
        capture_output=True, text=True, timeout=180))
    for name in MODULES:
        _retry(f"upload {name}", lambda n=name: _colab(
            ["colab", "upload", "-s", session,
             str(LOOP_DIR / n), f"{REMOTE_DIR}/{n}"], 120))
    if thermal:
        for name in THERMAL_MODULES:
            _retry(f"upload thermal/{name}", lambda n=name: _colab(
                ["colab", "upload", "-s", session,
                 str(THERMAL_DIR / n), f"{REMOTE_DIR}/thermal/{n}"], 120))
        print(f"[deploy] 모듈 {len(MODULES)}종 + thermal 패키지 {len(THERMAL_MODULES)}종 업로드 완료")
    else:
        print(f"[deploy] 모듈 {len(MODULES)}종 업로드 완료")


def _fetch_result_file(session: str) -> dict | None:
    """전송 유실 대비 — 원격 /content/abl_result.json 직접 회수."""
    code = "print(open('/content/abl_result.json').read())"
    p = subprocess.run(["colab", "exec", "-s", session], input=code,
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        return None
    for ln in reversed(p.stdout.splitlines()):
        if ln.strip().startswith("{"):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue
    return None


def completed_problems_from_result_dict(data: dict) -> set[str]:
    """배치 결과 dict에서 **완결**(off+on 둘 다 존재) 문제만 추출.

    P8 Task 36 이상 징후 4 — main()이 부분 결과(요청 문제 중 일부만 기록)를
    완료로 오판해 exit 0을 반환한 사고 재발 방지용 완결성 검증에도 재사용.
    """
    done = set()
    for name, tracks in (data.get("results") or {}).items():
        if "off" in tracks and "on" in tracks:
            done.add(name)
    return done


def completed_problems_from_result(result_path: Path) -> set[str]:
    """P8 Task 34 재개 로직 — 배치 결과 JSON 파일에서 완결 문제만 추출.

    §4-1 재개 규칙: off만 있고 on이 없는 등 부분 완료는 skip 대상 아님(전량 재실행 —
    부분 트랙만 이어 돌리는 반쪽 재개는 신뢰도 위험, P5 §4 안전망 계승 정직한 보수적 선택).
    파일 없으면 빈 집합(신규 배치, 회귀 0).
    """
    result_path = Path(result_path)
    if not result_path.exists():
        return set()
    try:
        data = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return completed_problems_from_result_dict(data)


def filter_uncompleted(problems: list[str], done: set[str]) -> list[str]:
    """P8 Task 34 — done 집합에 있는 문제를 배치 대상에서 제외(순서 보존)."""
    return [p for p in problems if p not in done]


def check_metric_mode_compat(result_dict: dict, expected: str) -> str | None:
    """P8 Task 36 재발 방지 — 회수/재개 결과의 metric_mode가 요청과 일치하는지.

    2026-07-17 사고: `--metric=thermal` 플래그 누락 세션이 기본값 latency로
    A100 실측을 돌렸는데, 결과 JSON에 metric_mode 스탬프가 없어 사후 검증도
    불가능했음. 이제 결과 dict에 metric_mode가 실리므로(main() 참조) 요청
    모드와 대조 가능 — 불일치면 경고 문자열 반환, 일치/키 부재(구포맷)면 None.

    반환값 규약: 불일치("...") → 호출측이 폐기/에러 처리, None → 통과(키 부재
    포함 — 구포맷 파일은 별도 경고만 내고 차단하지 않음, 호출측 책임).
    """
    got = result_dict.get("metric_mode")
    if got is None:
        return None
    if got != expected:
        return (f"metric_mode 불일치 (기대 {expected!r}, 실제 {got!r})")
    return None


def merge_batch_results(batch_result_paths: list[Path]) -> dict:
    """P8 Task 33/34 — 여러 배치 결과 JSON을 문제 단위로 병합.

    같은 문제가 여러 배치 파일에 있으면(재개로 재실행된 경우 등) **나중 경로가 우선**
    (리스트 순서 = 최신순 가정, 호출측이 mtime 등으로 정렬해 전달). chip은 마지막 값 채택.
    """
    merged: dict = {"chip": "", "results": {}}
    for p in batch_result_paths:
        p = Path(p)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        if data.get("chip"):
            merged["chip"] = data["chip"]
        merged["results"].update(data.get("results") or {})
    return merged


def _to_track(d: dict) -> TrackResult:
    # P6: "signals" 키는 06-p6-signal-provenance-design 이후 __ABL_RESULT__에
    # 새로 실림 — 그 이전 로그(P5 이전) 역직렬화 시 하위호환으로 빈 리스트.
    return TrackResult(d["label"], [tuple(x) for x in d["metric_curve"]],
                       d["fired_rules"], d["wasted_rounds"], d["retire_count"],
                       d["stop_reason"], d["rounds"], d.get("signals", []))


def main() -> int:
    argv = sys.argv[1:]
    if "--selfcheck" in argv:
        # 기존 회귀(latency 축) — 원격 드라이버 생성·컴파일 OK.
        src = _build_remote_driver(["kb_matmul_scalar"], 2)
        compile(src, "<remote_driver>", "exec")
        assert "__ABL_RESULT__" in src and "LocalProfiler" in src

        # P5 Task 19 — metric_mode="thermal" 조합도 컴파일 확인.
        src_thermal = _build_remote_driver(["matmul"], 2, metric_mode="thermal")
        compile(src_thermal, "<remote_driver_thermal>", "exec")
        assert '"metric_mode": "thermal"' in src_thermal

        # 2026-07-17 사고 재발 방지 — 원격 드라이버가 결과 dict에도 metric_mode를
        # 스탬프하는 코드를 담고 있는지(회수측 검증의 전제 조건).
        assert '"metric_mode": metric_mode' in src_thermal, (
            "원격 드라이버 결과 dict에 metric_mode 스탬프 코드 누락")

        # P5 §1-2-1 — 3문제(batched_gemm/kb_matmul_scalar/kb_softmax) variant map을
        # 실제 원격 config에 넣어 JSON 직렬화까지 확인(원격에서야 터지는 부류 방지).
        expected = {
            "batched_gemm": {"fp32_no_tensorcore"},
            "kb_matmul_scalar": {"fp32_no_tensorcore", "uncoalesced"},
            "kb_softmax": set(),
        }
        multi_src = _build_remote_driver(list(expected), 12, metric_mode="thermal")
        compile(multi_src, "<remote_driver_multi>", "exec")
        for name, expect_keys in expected.items():
            vmap = _variant_map_for(name)
            assert set(vmap.keys()) == expect_keys, (
                f"{name} variant map 불일치: got={set(vmap.keys())} expect={expect_keys}"
            )
        print(f"  variant map 확인: { {k: sorted(v) for k, v in expected.items()} }")

        # P5 kb_matmul_scalar gate_fail 재발 방지 — 정적 TF32 방어 체크(torch 불필요).
        tf32_warnings = check_all_tf32_guards()
        if tf32_warnings:
            for w in tf32_warnings:
                print(f"  [WARN] {w}", file=sys.stderr)
            raise AssertionError(
                f"TF32 가드 누락 {len(tf32_warnings)}건 — reference()가 A100에서 "
                f"오염될 위험. 위 경고 참조.")
        print("  TF32 가드 체크: 전 문제 통과 (행렬곱 reference 전부 allow_tf32 방어 확인)")

        # P6(06-p6-signal-provenance-design Task 24) — 가짜 신호로 signals 추출
        # 왕복 확인. torch·colab 불요, ledger.py는 순수 파이썬이라 실제 _run_track
        # 내부 로직(led.records에서 signal 뽑아 반환 dict에 담는 부분)을 직접
        # 재현해 검증한다 — "배선이 살아있다"를 A100 스펜드 전에 로컬로 증명.
        from ledger import Ledger, RoundRecord
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "abl-selfcheck.jsonl"
            led = Ledger(p)
            # 비-기본값 신호(load_eff=0.5, 팀리드 지적대로 0.0 기본값과 구분되는
            # 값 사용) + gate_fail 빈 dict 혼재 — P5 1차 로그(gate_fail 연속) 재현.
            led.append(RoundRecord("selfcheck_p", 0, "h0", {}, "gate_fail",
                                   "correctness 실패", -1, -1.0, False, False))
            led.append(RoundRecord("selfcheck_p", 1, "h1",
                                   {"load_eff": 0.5, "occupancy": 0.4},
                                   "uncoalesced", "메모리 비합착", 4, 0.03,
                                   True, True))
            recs = [r for r in led.records if r.problem == "selfcheck_p"]
            signals = [r.signal for r in recs]
            assert signals == [{}, {"load_eff": 0.5, "occupancy": 0.4}], (
                f"signals 추출 불일치: {signals}")
            # __ABL_RESULT__ 이 결과 dict를 JSON 직렬화하는 경로까지 확인.
            fake_result = {"label": "evolve_ON", "metric_curve": [(0, -1.0), (1, 0.03)],
                           "fired_rules": [r.hypothesis_label for r in recs],
                           "wasted_rounds": 0, "retire_count": 0,
                           "stop_reason": "stop_label", "rounds": 2,
                           "signals": signals}
            round_tripped = json.loads(json.dumps(fake_result, ensure_ascii=False))
            track = _to_track(round_tripped)
            assert track.signals[1]["load_eff"] == 0.5, (
                "TrackResult.signals 왕복 후 load_eff 값 유실")
            assert track.signals[0] == {}, "gate_fail 빈 dict가 왕복 중 변형됨"
        print("  signal_dict 왕복 체크: PASS (gate_fail 빈 dict + 비기본값 신호 "
              "혼재 케이스 포함)")

        print("run_ablation_remote.py self-check PASS (원격 드라이버 생성·컴파일 OK, "
              "thermal 축+3문제 variant map 포함, TF32 가드 체크 포함, "
              "signal_dict 왕복 체크 포함)")
        return 0

    session = None
    gpu = None
    metric_mode = "latency"
    skip_upload = "--skip-upload" in argv
    resume_path = None
    rest = []
    for a in argv:
        if a.startswith("--session="):
            session = a.split("=", 1)[1]
        elif a.startswith("--gpu="):
            gpu = a.split("=", 1)[1]
        elif a.startswith("--metric="):
            metric_mode = a.split("=", 1)[1]
        elif a.startswith("--resume="):
            # P8 Task 34 — 이 경로의 배치 결과 JSON에서 완결(off+on) 문제를 skip.
            resume_path = a.split("=", 1)[1]
        elif a not in ("--skip-upload",):
            rest.append(a)
    if not session or not rest:
        print("usage: run_ablation_remote.py <p1>[,<p2>...] [max_rounds] "
              "--session=<s> [--gpu=A100] [--metric=thermal] [--skip-upload] "
              "[--resume=<prev_result.json>] [--selfcheck]", file=sys.stderr)
        return 2
    problems = rest[0].split(",")
    max_rounds = int(rest[1]) if len(rest) > 1 else 8

    if resume_path:
        # 2026-07-17 사고 재발 방지 — resume 파일의 metric_mode가 현재 요청과
        # 다르면 캠페인 혼입(예: thermal 캠페인 재개인데 latency 기본값으로
        # 실행) 위험. 키 부재(구포맷, 기존 batch0/1 파일은 thermal로 검증된
        # 데이터라 차단 금지)는 경고만.
        try:
            resume_data = json.loads(Path(resume_path).read_text())
        except (OSError, json.JSONDecodeError):
            resume_data = {}
        resume_mismatch = check_metric_mode_compat(resume_data, metric_mode)
        if resume_mismatch:
            print(f"ERR: resume 파일 {resume_mismatch} — 캠페인 혼입 금지",
                  file=sys.stderr)
            return 1
        if resume_data and resume_data.get("metric_mode") is None:
            print("[warn] resume 파일에 metric_mode 스탬프 없음(구포맷)")
        done = completed_problems_from_result(Path(resume_path))
        remaining = filter_uncompleted(problems, done)
        if done:
            print(f"[resume] {resume_path}: 완결 {sorted(done)} skip, "
                  f"잔여 {remaining}")
        if not remaining:
            print("[resume] 잔여 문제 없음 — 배치 전량 완료 상태, exec 생략")
            return 0
        problems = remaining

    # 2026-07-17 사고 재발 방지 — exec 직전 로컬 모드를 눈에 띄게 배너로 출력.
    # metric=latency인데 --metric= 플래그가 argv에 없으면(기본값 사용) thermal
    # 캠페인 실행 실수를 놓치기 쉬우므로 명시 경고.
    print(f"[mode] metric_mode={metric_mode!r}")
    if metric_mode == "latency" and not any(a.startswith("--metric=") for a in argv):
        print("[warn] metric_mode=latency는 기본값 — thermal 캠페인이면 "
              "--metric=thermal 명시", file=sys.stderr)

    driver_src = _build_remote_driver(problems, max_rounds, metric_mode=metric_mode)
    total_rounds = len(problems) * 2 * max_rounds
    timeout = total_rounds * 150 + 300          # 라운드당 여유 150s + 부팅 여유
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(driver_src)
        path = f.name

    def _alive() -> bool:
        p = subprocess.run(["colab", "exec", "-s", session, "--timeout", "60"],
                           input="print('ALIVE')", capture_output=True,
                           text=True, timeout=120)
        return p.returncode == 0 and "ALIVE" in p.stdout

    is_thermal = metric_mode == "thermal"
    result = None
    for attempt in (1, 2):                       # 세션 자기치유: 죽었으면 재생성 1회
        if not _alive():
            if not gpu:
                print(f"ERR: 세션 {session} 죽음 (--gpu 미지정이라 재생성 불가)",
                      file=sys.stderr)
                return 1
            print(f"[heal] 세션 {session} 죽음 → colab new --gpu {gpu} (attempt {attempt})")
            _retry("colab new", lambda: _colab(
                ["colab", "new", "--gpu", gpu, f"--session={session}"], 420))
            _upload_modules(session, thermal=is_thermal)
        elif not skip_upload and attempt == 1:
            _upload_modules(session, thermal=is_thermal)

        print(f"[launch] 문제 {problems} × ON/OFF × {max_rounds}R — "
              f"colab exec 1방 (timeout {timeout:.0f}s, attempt {attempt})")
        exec_rc = None
        try:
            p = _colab(["colab", "exec", "-s", session, "-f", path,
                        "--timeout", str(timeout)], timeout + 120)
            sys.stdout.write(p.stdout)
            exec_rc = p.returncode
            if p.returncode != 0:
                print(f"[warn] exec rc={p.returncode}: {p.stderr[-400:]}",
                      file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("[warn] 전송 타임아웃 — 원격 결과 파일 회수 시도", file=sys.stderr)

        # P8 Task 33(§4-1) — 파일 다운로드를 주 경로로 승격. 대규모 문제(35~200)면
        # signals 포함 결과가 단일 stdout 라인 캡을 넘어 __ABL_RESULT__ 파싱이
        # truncate될 수 있음(P5/P7 소규모에선 무해했으나 규모 확장의 파괴 지점,
        # 설계 §4-1 표). exec 성공 여부와 무관하게 파일부터 회수 시도.
        result = _fetch_result_file(session)
        if result is None and exec_rc == 0:
            # 파일 회수 실패했는데 exec 자체는 성공했으면 stdout 라인 파싱 폴백
            # (구세션·소규모 배치 하위호환 — 파일 미기록 구버전 드라이버 대비).
            for ln in reversed(p.stdout.splitlines()):
                if ln.startswith("__ABL_RESULT__"):
                    try:
                        result = json.loads(ln[len("__ABL_RESULT__"):])
                    except json.JSONDecodeError:
                        result = None
                    break
        # 2026-07-17 사고 재발 방지 — 회수 결과의 metric_mode가 요청과 다르면
        # 원격에 남은 이전 런의 stale abl_result.json을 집었거나 모드 혼선이므로
        # 폐기하고 attempt 재시도 경로로 자연 합류(구포맷=키 부재는 경고만).
        if result is not None:
            mismatch = check_metric_mode_compat(result, metric_mode)
            if mismatch:
                print(f"[warn] 회수 결과 {mismatch} — 폐기", file=sys.stderr)
                result = None
            elif result.get("metric_mode") is None:
                print("[warn] 결과에 metric_mode 스탬프 없음(구포맷 드라이버)",
                      file=sys.stderr)
        # P8 Task 36 이상 징후 4 — 부분 결과(요청 문제 중 일부만 기록)를 완료로
        # 오판하지 않도록 완결성 검증. off+on 둘 다 있어야 그 문제는 "완료"로 인정
        # (completed_problems_from_result과 동일 기준). 미완이면 attempt 2로 재시도.
        missing = []
        if result is not None:
            missing = [p for p in problems
                       if p not in completed_problems_from_result_dict(result)]
        if result is not None and not missing:
            break
        if missing:
            print(f"[warn] 부분 결과 — 누락 {missing} (attempt {attempt}/2)",
                  file=sys.stderr)
    if result is None:
        print("ERR: 결과 회수 실패 (stdout·파일·재생성 재시도 전부)", file=sys.stderr)
        return 1
    missing = [p for p in problems
               if p not in completed_problems_from_result_dict(result)]
    if missing:
        print(f"ERR: 결과 불완전 — 요청 {len(problems)}문제 중 누락 {missing} "
              f"(재생성 재시도 후에도 미해소)", file=sys.stderr)
        return 1
    final_mismatch = check_metric_mode_compat(result, metric_mode)
    if final_mismatch:
        print(f"ERR: 최종 결과 {final_mismatch}", file=sys.stderr)
        return 1

    print(f"\n[chip={result.get('chip')!r}]")
    for name, r in result["results"].items():
        print(f"\n################ {name} ################")
        _report(_to_track(r["on"]), _to_track(r["off"]), metric_mode=metric_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
