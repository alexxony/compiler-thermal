"""P4 Task 16 → P7 Task 25~29 → P2 Task 4 — RcBackend ΔT 리포트, 다문제 일반화,
A/B 파라미터 세트 인터페이스.

design: docs/04-p4-rc-deltat-design.md §5 Task 16 (원본, matmul 단일);
        docs/07-p7-deltat-multiproblem-design.md §3~5 (다문제 일반화);
        HBM_build docs/06-p2-rc-backport-design.md Task 4 (A/B 인터페이스,
        vault /mnt/c/ObsidianVault/HBM_build/).
Colab 실행 아님 — 로컬 배치, GPU/Colab 세션 불요. 실측 없이 이미 있는 P3/P6 raw
신호만 후처리. 물리·측정 로직(duty_power/twin_eval)은 무변경, 여기선 로더 어댑터 +
문제 리스트 루프 + rc_kw 세트 선택만 추가한다(P7 설계 §3-2, P2 설계 Task 4).

RcBackend 클래스(thermal/twin_eval.py)는 P2 설계 §"RcBackend 절대 무변경" 제약에
따라 무변경 — A/B는 이 모듈의 호출측 rc_kw 딕셔너리 선택으로만 구현한다.
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thermal.duty_power import to_power_series, duty_avg_power  # noqa: F401
from thermal.twin_eval import RcBackend

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# --- P2 Task 4: rc_kw A/B 세트 -----------------------------------------------
#
# LEGACY = RcBackend 기존 검증 파라미터(tests/test_twin_eval.py make_backend()와
# 동일) — P4/P7 계획 §2 "기본안: 기존 검증 파라미터 그대로 사용" 결정 그대로.
# 전 함수의 기본 rc_kw이며, 이 세트로 계산한 결과는 P4 기준선(ΔT 17.16K)과
# 반드시 일치해야 한다(회귀 고정, tests/test_deltat_multiproblem.py).
RC_KW_LEGACY = dict(r_die_hbm=0.5, r_die_sink=0.15, r_hbm_sink=0.8,
                     c_die=50.0, c_hbm=10.0, t_ambient=45.0)

# 하위호환 별칭 — 기존 코드/테스트가 RC_KW를 직접 참조하는 지점 무손상.
RC_KW = RC_KW_LEGACY

# HBM_build P2 T2 산출물 경로(설계 문서 §T2). T2가 병렬 진행 중이라 부재할 수
# 있음 — 부재 시 RC_KW_HBM_FEM은 r_hbm_sink/c_hbm이 None인 placeholder로 구성.
HBM_RC_PARAMS_CSV = (
    Path.home() / "workspace" / "hbm_build" / "results" / "rc_params.csv"
)


def load_hbm_fem_rc_params(csv_path: Path = HBM_RC_PARAMS_CSV) -> dict:
    """T2 `rc_params.csv`에서 r_hbm_sink(K/W)·c_hbm(J/K)을 읽는다.

    P2 설계 §2 스코프 결정: HBM_build FEM은 HBM 스택만 모델하므로 교체 가능한
    파라미터는 이 2개뿐(die 쪽 r_die_hbm/r_die_sink/c_die는 legacy 유지).

    CSV 실제 스키마(T2 산출물, wide 아닌 long format) —
    컬럼: parameter,value,value_min,value_max,unit,method,basis_case.
    행 = parameter별 1건(`c_hbm`, `r_hbm_sink`). 대표값은 `value` 컬럼.
    r_hbm_sink는 냉각 BC 상이한 두 케이스로 범위(value_min~value_max) 제시됨
    (설계 §2 "냉각 BC 불일치 처리") — 대표값(`value`=baseline_8hi 케이스)을
    채택하고, 범위 민감도 판정은 T5 스코프로 남긴다.

    파일이 없으면 명확한 에러(FileNotFoundError, T2 상태 언급)로 실패한다 —
    조용히 legacy 값으로 되돌아가지 않는다(A/B 세트 오염 방지).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HBM FEM RC 파라미터 CSV 없음: {csv_path} "
            "(HBM_build P2 T2 미완료 — rc_extract.py 실행 후 재시도)"
        )
    with open(csv_path, newline="") as f:
        rows = {row["parameter"]: row for row in csv.DictReader(f) if row.get("parameter")}
    missing = {"r_hbm_sink", "c_hbm"} - rows.keys()
    if missing:
        raise ValueError(
            f"HBM FEM RC 파라미터 CSV에 필요 파라미터 없음: {sorted(missing)} "
            f"({csv_path})"
        )
    return {
        "r_hbm_sink": float(rows["r_hbm_sink"]["value"]),
        "c_hbm": float(rows["c_hbm"]["value"]),
    }


def _build_rc_kw_hbm_fem() -> dict:
    """RC_KW_HBM_FEM 세트 구성 — die 3개는 legacy 유지, HBM 2개만 FEM 값.

    T2 CSV 부재 시(병렬 진행 중 예상 상태) r_hbm_sink/c_hbm을 None으로 채운
    placeholder를 반환 — 이 세트로 evaluate()를 호출하면 TypeError로 즉시
    실패해 "placeholder인 채로 계산됨" 오염을 막는다. T5에서 실값 주입 예정
    (설계 문서 T4 요구사항).
    """
    base = dict(RC_KW_LEGACY)
    try:
        fem = load_hbm_fem_rc_params()
        base.update(fem)
        base["_fem_source"] = str(HBM_RC_PARAMS_CSV)
    except (FileNotFoundError, ValueError) as e:
        base["r_hbm_sink"] = None
        base["c_hbm"] = None
        base["_fem_source"] = f"PLACEHOLDER(T2 대기): {e}"
    return base


RC_KW_HBM_FEM = _build_rc_kw_hbm_fem()

# --- P10 Task 31: hotspot r_hbm_sink_max 로더 --------------------------------
#
# design: docs/11-p10-hotspot-deltat-design.md §2-2·§5 Task 31. avg 세트
# (r_hbm_sink)와 달리 hotspot 세트(r_hbm_sink_max)는 냉각BC 범위(value_min/
# value_max, 기존 파서와 동일 컬럼 접근) 외에 P3 3-시나리오(s0/s1/s2) 개별값이
# basis_case 컬럼의 자유 텍스트 안에 있어 정규식 파싱이 필요하다(§2-2).
_HOTSPOT_SCENARIO_RE = re.compile(
    r"(s0_uniform|s1_phy_moderate|s2_phy_heavy)\s*:\s*"
    r"dT=[-\d.]+K/P=[-\d.]+W->R=([-\d.]+)K/W"
)


def load_hbm_hotspot_rc_params(csv_path: Path = HBM_RC_PARAMS_CSV) -> dict:
    """`rc_params.csv`의 `r_hbm_sink_max` 행에서 hotspot R 5점을 읽는다.

    반환: {"representative": float, "cooling_bc_range": {"min": float,
    "max": float}, "s0_uniform": float, "s1_phy_moderate": float,
    "s2_phy_heavy": float} (설계 §5 Task 31 반환 스키마).

    대표값·냉각BC 범위는 기존 파서(`load_hbm_fem_rc_params`)와 동일하게
    value/value_min/value_max 컬럼에서 바로 읽는다. s0/s1/s2 개별값은
    basis_case 컬럼 자유 텍스트를 정규식으로 파싱한다(§2-2).

    파일 부재·행 부재·basis_case 파싱 실패는 전부 명확한 에러로 실패한다
    (조용한 폴백 금지 — `load_hbm_fem_rc_params`와 동일 원칙 계승).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HBM hotspot RC 파라미터 CSV 없음: {csv_path} "
            "(HBM_build r_hbm_sink_max 행 도착 전 — 재시도)"
        )
    with open(csv_path, newline="") as f:
        rows = {row["parameter"]: row for row in csv.DictReader(f) if row.get("parameter")}
    if "r_hbm_sink_max" not in rows:
        raise ValueError(
            f"HBM hotspot RC 파라미터 CSV에 r_hbm_sink_max 행 없음 ({csv_path})"
        )
    row = rows["r_hbm_sink_max"]
    basis_case = row.get("basis_case", "")
    matches = dict(_HOTSPOT_SCENARIO_RE.findall(basis_case))
    missing = {"s0_uniform", "s1_phy_moderate", "s2_phy_heavy"} - matches.keys()
    if missing:
        raise ValueError(
            f"r_hbm_sink_max basis_case에서 시나리오 패턴 추출 실패: "
            f"누락={sorted(missing)} (basis_case={basis_case!r})"
        )
    return {
        "representative": float(row["value"]),
        "cooling_bc_range": {
            "min": float(row["value_min"]),
            "max": float(row["value_max"]),
        },
        "s0_uniform": float(matches["s0_uniform"]),
        "s1_phy_moderate": float(matches["s1_phy_moderate"]),
        "s2_phy_heavy": float(matches["s2_phy_heavy"]),
    }


# --- P10 Task 32: RC_KW_HBM_HOTSPOT 세트 구성 --------------------------------
#
# design §3-2·§5 Task 32/§7 D1~D3. hotspot R은 시나리오 의존(단일 dict 아님)이라
# {시나리오명: rc_kw dict} 구조. die 3개는 legacy 유지, c_hbm도 legacy 유지(D3 —
# hotspot 대응 c_hbm 버전 없음), r_hbm_sink만 시나리오별로 교체한다.
_HOTSPOT_SCENARIO_KEYS = (
    "coolbc_min", "coolbc_max", "s0_uniform", "s1_phy_moderate", "s2_phy_heavy",
)


def _build_rc_kw_hbm_hotspot() -> dict:
    """RC_KW_HBM_HOTSPOT 5세트 구성 — CSV 부재 시 5개 전부 placeholder(None).

    `_build_rc_kw_hbm_fem`의 placeholder 패턴 재사용(설계 §5 Task 32 요구사항).
    """
    try:
        hotspot = load_hbm_hotspot_rc_params()
        source = str(HBM_RC_PARAMS_CSV)
        r_values = {
            "coolbc_min": hotspot["cooling_bc_range"]["min"],
            "coolbc_max": hotspot["cooling_bc_range"]["max"],
            "s0_uniform": hotspot["s0_uniform"],
            "s1_phy_moderate": hotspot["s1_phy_moderate"],
            "s2_phy_heavy": hotspot["s2_phy_heavy"],
        }
    except (FileNotFoundError, ValueError) as e:
        source = f"PLACEHOLDER(hotspot 자산 대기): {e}"
        r_values = {k: None for k in _HOTSPOT_SCENARIO_KEYS}

    sets = {}
    for name in _HOTSPOT_SCENARIO_KEYS:
        base = dict(RC_KW_LEGACY)  # die 3개 + c_hbm(D3) legacy 유지
        base["r_hbm_sink"] = r_values[name]
        base["_hotspot_source"] = source
        sets[name] = base
    return sets


RC_KW_HBM_HOTSPOT = _build_rc_kw_hbm_hotspot()

# 리포트 출력용 라벨 — 세트 dict identity로 이름을 역조회한다(build_report에서 사용).
RC_KW_SET_LABELS = {
    id(RC_KW_LEGACY): "LEGACY",
    id(RC_KW_HBM_FEM): "HBM_FEM",
    id(RC_KW_HBM_HOTSPOT["coolbc_min"]): "HOTSPOT_COOLBC_MIN",
    id(RC_KW_HBM_HOTSPOT["coolbc_max"]): "HOTSPOT_COOLBC_MAX",
    id(RC_KW_HBM_HOTSPOT["s0_uniform"]): "HOTSPOT_S0",
    id(RC_KW_HBM_HOTSPOT["s1_phy_moderate"]): "HOTSPOT_S1",
    id(RC_KW_HBM_HOTSPOT["s2_phy_heavy"]): "HOTSPOT_S2",
}


def rc_kw_set_label(rc_kw: dict) -> str:
    """rc_kw 딕셔너리 → 세트 라벨 문자열(리포트 표기용, Task 4/33 요구사항)."""
    label = RC_KW_SET_LABELS.get(id(rc_kw))
    if label is not None:
        return label
    return "CUSTOM"

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


def _steady_t_hbm(power_df, rc_kw: dict = RC_KW_LEGACY) -> float:
    # RcBackend 생성자 인자만 추출 — rc_kw에 섞인 메타데이터 키(예: _fem_source,
    # RC_KW_HBM_FEM 출처 표기)는 evaluate 호출에 흘러들어가면 안 된다.
    ctor_kw = {k: v for k, v in rc_kw.items() if not k.startswith("_")}
    out = RcBackend(**ctor_kw).evaluate(power_df)
    return out["t_hbm_c"].iloc[-1]


def _scenario_a(sig: dict, idle_die_w: float = IDLE_DIE_W,
                idle_hbm_w: float = IDLE_HBM_W,
                rc_kw: dict = RC_KW_LEGACY) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="saturated", window_s=WINDOW_S,
                              idle_die_w=idle_die_w, idle_hbm_w=idle_hbm_w)
    return _steady_t_hbm(series, rc_kw)


def _scenario_b(sig: dict, repeat_period_s: float,
                idle_die_w: float = IDLE_DIE_W,
                idle_hbm_w: float = IDLE_HBM_W,
                rc_kw: dict = RC_KW_LEGACY) -> float:
    series = to_power_series(p_die_w=sig["p_die_w"], p_hbm_w=sig["p_hbm_w"],
                              kernel_time_s=sig["kernel_time_s"],
                              duty_scenario="iso_work", window_s=WINDOW_S,
                              idle_die_w=idle_die_w, idle_hbm_w=idle_hbm_w,
                              repeat_period_s=repeat_period_s)
    return _steady_t_hbm(series, rc_kw)


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
                          idle_hbm_w: float = IDLE_HBM_W,
                          rc_kw: dict = RC_KW_LEGACY) -> dict:
    """seed/best 순간전력 페어 → (a)/(b) t_hbm·ΔT·판정 dict (Task 27/28 코어).

    seed=fp32(비교 기준), best=TF32(개선안). repeat_period_s는 seed(fp32)
    kernel_time_s(설계 §3-2 규약 — "같은 처리량 기한 공유"). null이면 판정 skip.
    rc_kw: P2 Task 4 A/B 세트(기본값 RC_KW_LEGACY) — RC_KW_HBM_FEM 등 교체 가능.
    """
    null = _is_null(seed, best)
    a_seed = _scenario_a(seed, idle_die_w, idle_hbm_w, rc_kw)
    a_best = _scenario_a(best, idle_die_w, idle_hbm_w, rc_kw)
    repeat_period_s = seed["kernel_time_s"]
    b_seed = _scenario_b(seed, repeat_period_s, idle_die_w, idle_hbm_w, rc_kw)
    b_best = _scenario_b(best, repeat_period_s, idle_die_w, idle_hbm_w, rc_kw)
    duty_best = best["kernel_time_s"] / repeat_period_s
    energy_ratio = (seed["energy_per_iter_j"] / best["energy_per_iter_j"]
                    if best["energy_per_iter_j"] else float("nan"))

    ta = rc_kw["t_ambient"]
    res = {
        "problem": problem,
        "is_null": null,
        "rc_kw_label": rc_kw_set_label(rc_kw),
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


def run_problem(cfg: dict, rc_kw: dict = RC_KW_LEGACY) -> dict:
    """config 1건 로드 → run_problem_from_sigs (Task 26 루프 단위).

    rc_kw: P2 Task 4 A/B 세트 파라미터화(기본값 RC_KW_LEGACY = 기존 결과 불변).
    """
    seed, best = _load_pair(cfg)
    return run_problem_from_sigs(cfg["problem"], seed, best, rc_kw=rc_kw)


def idle_sensitivity(cfg: dict, factors=(0.7, 1.0, 1.3),
                     rc_kw: dict = RC_KW_LEGACY) -> list[dict]:
    """P7 Task 29 — idle_die_w ×factors 3점에서 (b) 방향 불변 확인.

    각 점에서 run_problem_from_sigs를 idle 스케일해 재실행. null 문제는 방향 판정
    대신 is_null 표식만. 방향이 뒤집히면 flipped=True(문제 승격 조건, 설계 §3-3).
    rc_kw: P2 Task 4 A/B 세트(기본값 RC_KW_LEGACY).
    """
    seed, best = _load_pair(cfg)
    points = []
    for f in factors:
        res = run_problem_from_sigs(cfg["problem"], seed, best,
                                    idle_die_w=IDLE_DIE_W * f,
                                    idle_hbm_w=IDLE_HBM_W * f,
                                    rc_kw=rc_kw)
        flipped = (not res["is_null"]) and (res["b_best_t"] >= res["b_seed_t"])
        points.append({
            "idle_factor": f,
            "is_null": res["is_null"],
            "b_seed_t": res["b_seed_t"], "b_best_t": res["b_best_t"],
            "flipped": flipped,
        })
    return points


def format_problem_result(res: dict) -> str:
    """문제 1건 결과 → 리포트 텍스트 섹션(Task 26/27/28 출력).

    P2 Task 4: 어느 rc_kw 세트로 계산했는지 라벨 표기(res["rc_kw_label"]).
    """
    p = res["problem"]
    label = res.get("rc_kw_label", "LEGACY")
    lines = [f"\n{'='*60}", f"문제: {p}  [rc_kw={label}]", "=" * 60]
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


def build_report(configs, rc_kw: dict = RC_KW_LEGACY) -> str:
    """전체 config 루프 → 리포트 문자열(Task 26).

    rc_kw: P2 Task 4 A/B 세트(기본값 RC_KW_LEGACY) — 헤더에 세트 라벨 명시.
    """
    parts = ["=" * 60,
             "P7 RcBackend ΔT 다문제 리포트 — (a)포화 vs (b)iso-work",
             "=" * 60,
             f"rc_kw 세트: {rc_kw_set_label(rc_kw)}",
             f"idle_die_w={IDLE_DIE_W}W idle_hbm_w={IDLE_HBM_W}W (문헌치, 팀리드 승인)"]
    for cfg in configs:
        res = run_problem(cfg, rc_kw=rc_kw)
        parts.append(format_problem_result(res))
        # Task 29 idle sensitivity 3점 요약
        pts = idle_sensitivity(cfg, rc_kw=rc_kw)
        if not res["is_null"]:
            inv = all(not q["flipped"] for q in pts)
            summ = "  idle sensitivity(×0.7/×1.0/×1.3): (b) 방향 " + (
                "불변 ✓" if inv else "⚠️ 뒤집힘 — idle 정밀측정 승격 조건 발동")
            parts.append(summ)
    return "\n".join(parts)


# --- P10 Task 34: build_hotspot_report() -------------------------------------
#
# design §3-1·§5 Task 34. avg 리포트(build_report, 문제×2시나리오(a/b))와 열
# 구조가 달라(문제×5 R시나리오) 별도 함수로 신설(D5). build_report()는 무변경.
_HOTSPOT_SCENARIO_LABELS = (
    ("s0_uniform", "HOTSPOT_S0"),
    ("s1_phy_moderate", "HOTSPOT_S1"),
    ("s2_phy_heavy", "HOTSPOT_S2"),
    ("coolbc_min", "HOTSPOT_COOLBC_MIN"),
    ("coolbc_max", "HOTSPOT_COOLBC_MAX"),
)


def build_hotspot_report(configs) -> str:
    """4문제 × 5 R시나리오(hotspot) 표 렌더 — avg 축(LEGACY)도 참고로 병기.

    §3-1 표 구조: 문제별로 avg(LEGACY) (b) gap과 5개 hotspot R시나리오의
    (b) gap을 함께 렌더. null 문제(kb_softmax)는 "null"로 표시(§4-1 4행).
    build_report()는 이 함수와 완전히 독립 — 무변경(§4-3 회귀 게이트).
    """
    parts = ["=" * 60,
             "P10 RcBackend ΔT hotspot 리포트 — avg vs hotspot(5 R시나리오)",
             "=" * 60]
    for cfg in configs:
        problem = cfg["problem"]
        res_avg = run_problem(cfg, rc_kw=RC_KW_LEGACY)
        parts.append(f"\n{'-'*60}")
        parts.append(f"문제: {problem}")
        parts.append("-" * 60)
        if res_avg["is_null"]:
            parts.append("  → null 대조군(seed=best) — R세트 무관, 방향 판정 skip")
            parts.append(f"  avg(LEGACY) (b) gap: null")
            for scenario_key, label in _HOTSPOT_SCENARIO_LABELS:
                parts.append(f"  {label} (b) gap: null")
            continue
        gap_avg = res_avg["b_seed_t"] - res_avg["b_best_t"]
        parts.append(f"  avg(LEGACY) (b) gap: {gap_avg:.4f}K")
        for scenario_key, label in _HOTSPOT_SCENARIO_LABELS:
            rc_kw = RC_KW_HBM_HOTSPOT[scenario_key]
            if rc_kw.get("r_hbm_sink") is None:
                parts.append(f"  {label} (b) gap: null (hotspot 자산 placeholder)")
                continue
            res_hs = run_problem(cfg, rc_kw=rc_kw)
            gap_hs = res_hs["b_seed_t"] - res_hs["b_best_t"]
            parts.append(f"  {label} (b) gap: {gap_hs:.4f}K")
    return "\n".join(parts)


# --- P10 Task 35: hotspot_verdict() / hotspot_verdict_rollup() ---------------
#
# design §4-1·§4-2. 방향 일치 bool + 배율 감쇠 비율(참고값, 판정에 미사용).
# 문제 단위 롤업은 AND 조건 — 1개라도 반전이면 "부분 PASS — 시나리오 의존"
# (FAIL로 뭉개지 않음, §4-2).
_HOTSPOT_SCENARIO_KEY_MAP = dict(_HOTSPOT_SCENARIO_LABELS)


def hotspot_verdict(cfg: dict, scenario: str) -> dict:
    """문제 cfg, hotspot R시나리오 1건 → 방향 일치 + 배율 감쇠 비율.

    반환: {"direction_match": bool|None, "attenuation_ratio": float|None}.
    direction_match: avg(LEGACY) (b) gap 부호와 hotspot (b) gap 부호가 같은지
    (§4-1 1행). null 문제는 None(판정 skip, §4-1 4행).
    attenuation_ratio: hotspot gap / avg gap — 참고 기록용(§4-1 3행, 판정 무관).
    """
    res_avg = run_problem(cfg, rc_kw=RC_KW_LEGACY)
    if res_avg["is_null"]:
        return {"direction_match": None, "attenuation_ratio": None}
    gap_avg = res_avg["b_seed_t"] - res_avg["b_best_t"]
    rc_kw = RC_KW_HBM_HOTSPOT[scenario]
    res_hs = run_problem(cfg, rc_kw=rc_kw)
    gap_hs = res_hs["b_seed_t"] - res_hs["b_best_t"]
    direction_match = (gap_avg > 0) == (gap_hs > 0)
    attenuation_ratio = gap_hs / gap_avg if gap_avg != 0 else float("nan")
    return {"direction_match": bool(direction_match),
            "attenuation_ratio": attenuation_ratio}


def hotspot_verdict_rollup(per_scenario: dict) -> str:
    """§4-2 AND 조건 문제 단위 롤업.

    per_scenario: {시나리오명: {"direction_match": bool|None, ...}}.
    전부 True → "PASS". 일부 False(True/False 혼재) → "부분 PASS — 시나리오
    의존"(FAIL로 뭉개지 않음). 전부 None(null 문제) → "null".
    attenuation_ratio는 이 함수의 판정에 영향 주지 않는다(참고값 전용).
    """
    matches = [v["direction_match"] for v in per_scenario.values()]
    non_null = [m for m in matches if m is not None]
    if not non_null:
        return "null"
    if all(non_null):
        return "PASS"
    return "부분 PASS — 시나리오 의존"


# --- P11 Task 37: hotspot r_hbm_sink_max_p4 로더 -----------------------------
#
# design: docs/12-p11-hotspot-p4-30w-design.md §3 D1·D7·§5 Task 37. P10 로더
# (load_hbm_hotspot_rc_params, 5키·flat 반환)와 반환 스키마가 근본적으로 달라
# (6키·1단계 중첩·die_source 포함) 신설한다 — 파라미터화하면 P10 정규식이
# 이 basis_case 텍스트를 조용히 0건 매칭해 버리는 침묵 실패를 우회하기 위함
# (D1 근거). P10 로더는 이 함수와 완전히 무관 — 무변경.
_HOTSPOT_P4_SCENARIO_RE = re.compile(
    r"(a_s0|a_s1|a_s2|b_s0|b_s1|b_s2)\[(base_die(?:_phy)?)\]\s*:\s*"
    r"dT=[-\d.]+K/P=[-\d.]+W->R=([-\d.]+)K/W"
)

_HOTSPOT_P4_SCENARIO_KEYS = ("a_s0", "a_s1", "a_s2", "b_s0", "b_s1", "b_s2")


def load_hbm_hotspot_p4_rc_params(csv_path: Path = HBM_RC_PARAMS_CSV) -> dict:
    """`rc_params.csv`의 `r_hbm_sink_max_p4` 행에서 hotspot R 6점(30W)을 읽는다.

    반환 스키마 — P10 로더(load_hbm_hotspot_rc_params, flat 5키:
    representative/cooling_bc_range/s0_uniform/s1_phy_moderate/s2_phy_heavy)와
    **다르다**: 이 함수는 1단계 중첩 dict-of-dict를 반환한다 —
    {"a_s0": {"r": 5.138622, "die_source": "base_die"},
     "a_s1": {"r": 5.844228, "die_source": "base_die_phy"}, ...} 6키
    (a_s0/a_s1/a_s2/b_s0/b_s1/b_s2). die_source는 §0 표의 대괄호 die 폴백
    표기(`a_s0[base_die]` 등)를 그대로 보존한 값 — s0류만 base_die로 폴백되고
    s1/s2류는 base_die_phy라는 비대칭 패턴을 리포트에서 추적 가능하게 한다
    (설계 §3 D7, R4 — 인터페이스 차이는 이 docstring으로 문서화).

    30W 고정 조건(P3 hotspot 세트의 16W와 다른 별도 FEM 캠페인) —
    representative/cooling_bc_range 같은 필드는 이 행에 대응 개념이 없어
    반환하지 않는다(D1).

    파일 부재·행 부재·basis_case 파싱 실패는 P10 로더와 동일하게 명확한 에러
    (FileNotFoundError/ValueError)로 실패한다 — 조용한 폴백 금지 원칙 계승.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HBM hotspot P4 RC 파라미터 CSV 없음: {csv_path} "
            "(HBM_build r_hbm_sink_max_p4 행 도착 전 — 재시도)"
        )
    with open(csv_path, newline="") as f:
        rows = {row["parameter"]: row for row in csv.DictReader(f) if row.get("parameter")}
    if "r_hbm_sink_max_p4" not in rows:
        raise ValueError(
            f"HBM hotspot P4 RC 파라미터 CSV에 r_hbm_sink_max_p4 행 없음 ({csv_path})"
        )
    row = rows["r_hbm_sink_max_p4"]
    basis_case = row.get("basis_case", "")
    matches = _HOTSPOT_P4_SCENARIO_RE.findall(basis_case)
    result = {key: {"r": float(r), "die_source": die}
              for key, die, r in matches}
    missing = set(_HOTSPOT_P4_SCENARIO_KEYS) - result.keys()
    if missing:
        raise ValueError(
            f"r_hbm_sink_max_p4 basis_case에서 시나리오 패턴 추출 실패: "
            f"누락={sorted(missing)} (basis_case={basis_case!r})"
        )
    return result


# --- P11 Task 38: RC_KW_HBM_HOTSPOT_P4 세트 구성 -----------------------------
#
# design §3 D2·D5·§5 Task 38. P10 RC_KW_HBM_HOTSPOT(5키, 16W)과 별도 top-level
# 딕셔너리로 분리(통합 안 함, D2) — 두 세트는 물리적으로 다른 FEM 조건(16W/30W)
# 이라 같은 네임스페이스에 두면 "같은 종류의 선택지"로 오인될 착시가 생긴다.
# die 3개(r_die_hbm/r_die_sink/c_die)+c_hbm은 legacy 유지(D5, P10 D3 계승),
# r_hbm_sink만 케이스별로 교체한다. _build_rc_kw_hbm_hotspot 패턴 재사용.

def _build_rc_kw_hbm_hotspot_p4() -> dict:
    """RC_KW_HBM_HOTSPOT_P4 6세트 구성 — CSV 부재 시 6개 전부 placeholder(None).

    `_build_rc_kw_hbm_hotspot`(P10)과 동일 placeholder 패턴(설계 §5 Task 38).
    """
    try:
        hotspot = load_hbm_hotspot_p4_rc_params()
        source = str(HBM_RC_PARAMS_CSV)
        r_values = {k: v["r"] for k, v in hotspot.items()}
        die_sources = {k: v["die_source"] for k, v in hotspot.items()}
    except (FileNotFoundError, ValueError) as e:
        source = f"PLACEHOLDER(hotspot P4 자산 대기): {e}"
        r_values = {k: None for k in _HOTSPOT_P4_SCENARIO_KEYS}
        die_sources = {k: None for k in _HOTSPOT_P4_SCENARIO_KEYS}

    sets = {}
    for name in _HOTSPOT_P4_SCENARIO_KEYS:
        base = dict(RC_KW_LEGACY)  # die 3개 + c_hbm(D5) legacy 유지
        base["r_hbm_sink"] = r_values[name]
        base["_hotspot_p4_source"] = source
        base["_hotspot_p4_die_source"] = die_sources[name]
        sets[name] = base
    return sets


RC_KW_HBM_HOTSPOT_P4 = _build_rc_kw_hbm_hotspot_p4()

# P11 Task 39: RC_KW_SET_LABELS 6개 엔트리 추가(D3) — 기존 11개(LEGACY/HBM_FEM/
# HOTSPOT_* 5개) 매핑은 그대로, rc_kw_set_label() 함수 자체는 무변경.
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["a_s0"])] = "HOTSPOT_P4_A_S0"
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["a_s1"])] = "HOTSPOT_P4_A_S1"
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["a_s2"])] = "HOTSPOT_P4_A_S2"
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["b_s0"])] = "HOTSPOT_P4_B_S0"
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["b_s1"])] = "HOTSPOT_P4_B_S1"
RC_KW_SET_LABELS[id(RC_KW_HBM_HOTSPOT_P4["b_s2"])] = "HOTSPOT_P4_B_S2"


# --- P11 Task 40: build_hotspot_report_p4() -----------------------------------
#
# design §3 D4·§5 Task 40. build_hotspot_report()(P10, 5시나리오)와 별개 함수로
# 신설 — 열 개수부터 다름(6케이스). 독립 완결 문자열을 반환(호출부에서 P10
# 리포트 뒤에 이어붙이거나 단독 출력 가능, 결합도 최소화). build_report()/
# build_hotspot_report()는 이 함수와 무관 — 무변경(회귀 게이트).
_HOTSPOT_P4_SCENARIO_LABELS = (
    ("a_s0", "HOTSPOT_P4_A_S0"),
    ("a_s1", "HOTSPOT_P4_A_S1"),
    ("a_s2", "HOTSPOT_P4_A_S2"),
    ("b_s0", "HOTSPOT_P4_B_S0"),
    ("b_s1", "HOTSPOT_P4_B_S1"),
    ("b_s2", "HOTSPOT_P4_B_S2"),
)


def build_hotspot_report_p4(configs) -> str:
    """4문제 × 6 R케이스(hotspot P4, 30W) 표 렌더 — avg 축(LEGACY)도 참고 병기.

    §3 D4: 냉각계열 라벨(a_/b_)은 basis_case 원문 그대로 사용하며 물리적 BC로
    재해석하지 않는다(R2 — 리포트 헤더 각주로 명시). 독립적으로 완결된 문자열을
    반환 — build_hotspot_report()(P10, P3 16W 5세트)와 결합하지 않는다.
    """
    parts = ["=" * 60,
             "P11 RcBackend ΔT hotspot P4 리포트 — avg vs hotspot(6 R케이스, 30W)",
             "=" * 60,
             "※ 냉각계열 라벨(a/b)은 HBM_build basis_case 원문을 그대로 사용 — "
             "물리적 BC(top+bottom/top-only 등)로 재해석하지 않음(§8 범위 제외)."]
    for cfg in configs:
        problem = cfg["problem"]
        res_avg = run_problem(cfg, rc_kw=RC_KW_LEGACY)
        parts.append(f"\n{'-'*60}")
        parts.append(f"문제: {problem}")
        parts.append("-" * 60)
        if res_avg["is_null"]:
            parts.append("  → null 대조군(seed=best) — R세트 무관, 방향 판정 skip")
            parts.append(f"  avg(LEGACY) (b) gap: null")
            for scenario_key, label in _HOTSPOT_P4_SCENARIO_LABELS:
                parts.append(f"  {label} (b) gap: null")
            continue
        gap_avg = res_avg["b_seed_t"] - res_avg["b_best_t"]
        parts.append(f"  avg(LEGACY) (b) gap: {gap_avg:.4f}K")
        for scenario_key, label in _HOTSPOT_P4_SCENARIO_LABELS:
            rc_kw = RC_KW_HBM_HOTSPOT_P4[scenario_key]
            if rc_kw.get("r_hbm_sink") is None:
                parts.append(f"  {label} (b) gap: null (hotspot P4 자산 placeholder)")
                continue
            res_hs = run_problem(cfg, rc_kw=rc_kw)
            gap_hs = res_hs["b_seed_t"] - res_hs["b_best_t"]
            die_src = rc_kw.get("_hotspot_p4_die_source", "?")
            parts.append(f"  {label} (b) gap: {gap_hs:.4f}K  [die={die_src}]")
    return "\n".join(parts)


# --- P11 Task 41: hotspot_p4_verdict() / hotspot_p4_verdict_rollup() ---------
#
# design §3 D6·§5 Task 41. P10 §4-1 판정 표를 계승하되 비교 기준을 3원화 —
# avg 대비뿐 아니라 P3 hotspot(16W) 대비 방향 일치도 함께 기록한다(이중 방향
# 일치). 대응 시나리오 매핑: a_s1/a_s2 -> s1_phy_moderate/s2_phy_heavy,
# b_s1/b_s2 -> 동일(냉각계열은 P3 쪽엔 축 자체가 없어 s0/s1/s2만 대응).
# a_s0/b_s0는 P3 쪽 s0_uniform과 대응.
_HOTSPOT_P4_TO_P3_SCENARIO = {
    "a_s0": "s0_uniform", "a_s1": "s1_phy_moderate", "a_s2": "s2_phy_heavy",
    "b_s0": "s0_uniform", "b_s1": "s1_phy_moderate", "b_s2": "s2_phy_heavy",
}


def hotspot_p4_verdict(cfg: dict, case: str) -> dict:
    """문제 cfg, hotspot P4 R케이스 1건 → avg 대비 + P3 hotspot 대비 이중 방향 일치.

    반환: {"direction_match_avg": bool|None, "direction_match_p3": bool|None,
    "attenuation_ratio_avg": float|None, "attenuation_ratio_p3": float|None}.
    direction_match_avg: avg(LEGACY) (b) gap 부호와 hotspot P4 (b) gap 부호가
    같은지(§3 D6 표 1행). direction_match_p3: 대응하는 P3 hotspot(16W) 시나리오
    (b) gap 부호와 같은지(D6 표 2행, 신규). null 문제는 둘 다 None.
    attenuation_ratio_*: 참고 기록용(판정에 미사용, P10과 동일 원칙).
    """
    res_avg = run_problem(cfg, rc_kw=RC_KW_LEGACY)
    if res_avg["is_null"]:
        return {"direction_match_avg": None, "direction_match_p3": None,
                "attenuation_ratio_avg": None, "attenuation_ratio_p3": None}
    gap_avg = res_avg["b_seed_t"] - res_avg["b_best_t"]
    rc_kw_p4 = RC_KW_HBM_HOTSPOT_P4[case]
    res_p4 = run_problem(cfg, rc_kw=rc_kw_p4)
    gap_p4 = res_p4["b_seed_t"] - res_p4["b_best_t"]

    p3_scenario = _HOTSPOT_P4_TO_P3_SCENARIO[case]
    rc_kw_p3 = RC_KW_HBM_HOTSPOT[p3_scenario]
    res_p3 = run_problem(cfg, rc_kw=rc_kw_p3)
    gap_p3 = res_p3["b_seed_t"] - res_p3["b_best_t"]

    return {
        "direction_match_avg": bool((gap_avg > 0) == (gap_p4 > 0)),
        "direction_match_p3": bool((gap_p3 > 0) == (gap_p4 > 0)),
        "attenuation_ratio_avg": gap_p4 / gap_avg if gap_avg != 0 else float("nan"),
        "attenuation_ratio_p3": gap_p4 / gap_p3 if gap_p3 != 0 else float("nan"),
    }


def hotspot_p4_verdict_rollup(per_case: dict) -> str:
    """§3 D6 AND 조건 문제 단위 롤업 — avg 대비·P3 대비 두 기준 모두 반영.

    per_case: {케이스명: {"direction_match_avg": bool|None,
    "direction_match_p3": bool|None, ...}}. 두 기준 모두 전 케이스 True →
    "PASS". 어느 한 기준이라도 일부 불일치(True/False 혼재) → "부분 PASS —
    케이스 의존"(FAIL로 뭉개지 않음, P10 §4-2 계승). 전부 None(null 문제) →
    "null".
    """
    avg_matches = [v["direction_match_avg"] for v in per_case.values()]
    p3_matches = [v["direction_match_p3"] for v in per_case.values()]
    avg_non_null = [m for m in avg_matches if m is not None]
    p3_non_null = [m for m in p3_matches if m is not None]
    if not avg_non_null and not p3_non_null:
        return "null"
    if all(avg_non_null) and all(p3_non_null):
        return "PASS"
    return "부분 PASS — 케이스 의존"


def main() -> int:
    for cfg in PROBLEM_CONFIGS:
        if not Path(cfg["path"]).exists():
            print(f"ERR: 없음 {cfg['path']} — 실측 결과 필요", file=sys.stderr)
            return 2
    print(build_report(PROBLEM_CONFIGS, rc_kw=RC_KW_LEGACY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
