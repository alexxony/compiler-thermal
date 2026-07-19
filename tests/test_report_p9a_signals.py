"""P9-A Task 41 — R0 signal_dict 판별 리포터 테스트
(10-p9a-rule-coverage-design.md §2-1/§2-3/§4 Task 41, T-conv + T-fire).

목적: conv-fusion R0 STOP 7건이 (i)정당(신호가 STOP cond를 실제로 충족)인지
(ii)오발화(신호-판정 불일치)인지, (d) 미발화 2건이 (i)정당 미달인지 (ii)문턱
미세조정 여지인지를 rules.py의 cond 상수를 그대로 재사용해 결정론으로 판별한다.
하드코딩 복제 금지(design §4 Task 41 명시) — tensorcore_saturated/memory_saturated/
fp32_no_tensorcore의 판정은 반드시 rules.seed_rules()에서 가져온 실제 Rule 객체의
cond를 호출해 검증한다(팀리드 A/B 판정과 값이 벌어지면 이 재사용 원칙 위반 의심).
torch 불요, 순수 signal dict 판정.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from report_p9a_signals import (  # noqa: E402
    classify_stop_signal,
    classify_nonfire_signal,
    extract_r0_signal,
    build_signal_table,
    format_signal_report,
)
from rules import seed_rules  # noqa: E402
from signals import from_dict  # noqa: E402


# ── classify_stop_signal: conv STOP 7건 (i)정당 / (ii)오발화 판별 ──

def test_classify_stop_signal_legit_tensorcore_saturated():
    # tensorcore_saturated cond 실제 충족(tc_active ∧ compute_tput>0.5 ∧ latency>SLOW).
    sig = from_dict({"tensorcore_active": True, "compute_tput": 0.52,
                     "latency_us": 15387.0, "weight_pct": 1.0})
    result = classify_stop_signal(sig)
    assert result.verdict == "legit", result
    assert "tensorcore_saturated" in result.matched_cond


def test_classify_stop_signal_legit_memory_saturated():
    sig = from_dict({"bw_pct": 0.85, "tensorcore_active": False, "weight_pct": 1.0})
    result = classify_stop_signal(sig)
    assert result.verdict == "legit", result
    assert "memory_saturated" in result.matched_cond


def test_classify_stop_signal_misfire_neither_cond_holds():
    # 어느 STOP cond도 실제로 안 맞는데 STOP으로 기록됐다면 오발화(신호-판정 불일치).
    sig = from_dict({"tensorcore_active": False, "compute_tput": 0.1,
                     "bw_pct": 0.3, "latency_us": 30.0, "weight_pct": 0.5})
    result = classify_stop_signal(sig)
    assert result.verdict == "misfire", result


def test_classify_stop_signal_misfire_tc_active_but_not_saturated():
    # tc_active=True인데 compute_tput<=0.5 (memory-bound TC 미세활성, rules.py 게이트A
    # 실측 회귀 방지 가드 케이스와 동일 신호) — tensorcore_saturated 미충족.
    sig = from_dict({"tensorcore_active": True, "compute_tput": 0.27,
                     "bw_pct": 0.5, "latency_us": 178.0, "weight_pct": 1.0})
    result = classify_stop_signal(sig)
    # bw_pct=0.5는 memory_saturated(>0.8) 미충족이므로 legit 아닌 misfire.
    assert result.verdict == "misfire", result


def test_classify_stop_signal_uses_actual_rule_cond_not_hardcoded():
    # 재사용 원칙 검증: rules.py의 실제 cond 객체가 이 판별에 쓰이는지 간접 확인.
    # SLOW_LATENCY_US 경계값 바로 아래/위로 legit 여부가 rules.py 상수와 일치해야 함.
    from rules import SLOW_LATENCY_US
    sig_below = from_dict({"tensorcore_active": True, "compute_tput": 0.9,
                           "latency_us": SLOW_LATENCY_US - 1.0, "weight_pct": 1.0,
                           "bw_pct": 0.1})
    result = classify_stop_signal(sig_below)
    assert result.verdict == "misfire", (
        f"latency<SLOW_LATENCY_US인데 legit 판정 — rules.py 상수 미재사용 의심: {result}")


# ── classify_nonfire_signal: (d) 미발화 2건 (i)정당 / (ii)문턱조정여지 판별 ──

def test_classify_nonfire_signal_legit_weight_pct_below_gate():
    # fp32_no_tensorcore 미충족 원인 = weight_pct<0.05(게이트 미달) — 정당 미발화.
    sig = from_dict({"weight_pct": 0.02, "compute_tput": 0.5, "tensorcore_active": False})
    result = classify_nonfire_signal(sig)
    assert result.verdict == "legit", result
    assert "weight_pct" in result.reason


def test_classify_nonfire_signal_legit_tensorcore_already_active():
    # tensorcore_active=True라 not tensorcore_active 미충족 — 정당 미발화.
    sig = from_dict({"weight_pct": 0.5, "compute_tput": 0.5, "tensorcore_active": True})
    result = classify_nonfire_signal(sig)
    assert result.verdict == "legit", result
    assert "tensorcore_active" in result.reason


def test_classify_nonfire_signal_tuning_room_when_all_conds_actually_hold():
    # fp32_no_tensorcore cond가 사실 다 충족되는데(weight>=0.05, compute>0, not tc)
    # 미발화로 기록됐다면 문턱조정 여지(모순 케이스 — 실제로는 리포터가 R0 signal만
    # 보고 판단하므로, cond 충족인데도 미발화라면 상위 룰 우선순위 등 다른 요인 의심).
    sig = from_dict({"weight_pct": 0.5, "compute_tput": 0.5, "tensorcore_active": False})
    result = classify_nonfire_signal(sig)
    assert result.verdict == "tuning_room", result


# ── extract_r0_signal: result JSON에서 signals[0] 안전 추출 ──

def test_extract_r0_signal_normal_case():
    track = {"signals": [{"weight_pct": 0.5, "tensorcore_active": True}, {"weight_pct": 0.1}]}
    sig = extract_r0_signal(track)
    assert sig is not None
    assert sig.weight_pct == 0.5
    assert sig.tensorcore_active is True


def test_extract_r0_signal_defends_empty_gate_fail_dict():
    # P6 관례: gate_fail 라운드는 signal_dict가 빈 {}일 수 있음 — 방어적으로 None 아닌
    # 기본 Signal 반환(크래시 금지), 호출측이 gate_fail 여부로 별도 판단.
    track = {"signals": [{}]}
    sig = extract_r0_signal(track)
    assert sig is not None
    assert sig.weight_pct == 0.0


def test_extract_r0_signal_missing_signals_key_returns_none():
    track = {}
    assert extract_r0_signal(track) is None


def test_extract_r0_signal_empty_signals_list_returns_none():
    track = {"signals": []}
    assert extract_r0_signal(track) is None


# ── build_signal_table / format_signal_report: 실 result JSON 통합 ──

def test_build_signal_table_classifies_stop_and_nonfire_problems():
    results = {
        "conv_a": {"on": {"stop_reason": "stop_label",
                          "signals": [{"tensorcore_active": True, "compute_tput": 0.9,
                                      "latency_us": 200.0, "weight_pct": 1.0}]}},
        "gemm_b": {"on": {"stop_reason": "converged",
                          "signals": [{"weight_pct": 0.5, "compute_tput": 0.5,
                                      "tensorcore_active": True}]}},
    }
    stop_problems = {"conv_a"}
    nonfire_problems = {"gemm_b"}
    table = build_signal_table(results, stop_problems, nonfire_problems)
    assert "conv_a" in table
    assert table["conv_a"].verdict == "legit"
    assert "gemm_b" in table
    assert table["gemm_b"].verdict == "legit"  # tc_active=True → 정당 미발화


def test_build_signal_table_control_matmul_problems_included_unclassified():
    # 대조군(13_Matmul 등)은 STOP도 미발화 판별 대상도 아님 — 신호값만 표에 포함되고
    # verdict는 "control"로 표시(design §4: "신호 분포 대비용").
    results = {
        "matmul_ctrl": {"on": {"stop_reason": "converged",
                               "signals": [{"weight_pct": 1.0, "compute_tput": 0.98,
                                           "tensorcore_active": False}]}},
    }
    table = build_signal_table(results, stop_problems=set(), nonfire_problems=set(),
                               control_problems={"matmul_ctrl"})
    assert table["matmul_ctrl"].verdict == "control"


def test_format_signal_report_shows_field_values_and_verdict():
    results = {
        "conv_a": {"on": {"stop_reason": "stop_label",
                          "signals": [{"tensorcore_active": True, "compute_tput": 0.9,
                                      "latency_us": 200.0, "weight_pct": 1.0,
                                      "bw_pct": 0.1}]}},
    }
    table = build_signal_table(results, stop_problems={"conv_a"}, nonfire_problems=set())
    text = format_signal_report(table)
    assert "conv_a" in text
    assert "legit" in text
    assert "tc_active" in text or "tensorcore_active" in text


def test_selfcheck_entrypoint_runs_clean(capsys):
    import report_p9a_signals
    rc = report_p9a_signals.main(["--selfcheck"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
