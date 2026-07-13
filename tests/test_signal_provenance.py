"""P6 signal_dict 라운드별 보존 회귀 테스트 (06-p6-signal-provenance-design).

torch 불필요 — ledger.py/run_gain_compare.py는 순수 파이썬. _run_track 자체는
executor 경유(torch 의존)라 직접 안 부르고, 그 안의 추출 로직(led.records에서
signal 뽑아 반환 dict에 담는 부분)을 Ledger/RoundRecord로 직접 재현해 검증한다.
"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from ledger import Ledger, RoundRecord  # noqa: E402
from run_gain_compare import TrackResult  # noqa: E402
from run_ablation_remote import _to_track  # noqa: E402


def _extract_signals(led: Ledger, problem: str) -> list[dict]:
    """_run_track의 "signals": [r.signal for r in recs] 로직 재현(격리 테스트용)."""
    recs = [r for r in led.records if r.problem == problem]
    return [r.signal for r in recs]


def test_extraction_carries_signal_per_round(tmp_path):
    """Task 22: 정상 라운드 — signal_dict가 라운드별로 그대로 실린다."""
    p = tmp_path / "abl-test.jsonl"
    led = Ledger(p)
    led.append(RoundRecord("kb_matmul_scalar", 0, "hash0",
                           {"load_eff": 0.9, "occupancy": 0.5}, "fp32_no_tensorcore",
                           "rationale", 0, 0.01, False, True))
    led.append(RoundRecord("kb_matmul_scalar", 1, "hash1",
                           {"load_eff": 0.3, "occupancy": 0.6}, "uncoalesced",
                           "rationale", 4, 0.02, True, True))
    signals = _extract_signals(led, "kb_matmul_scalar")
    assert signals == [{"load_eff": 0.9, "occupancy": 0.5},
                        {"load_eff": 0.3, "occupancy": 0.6}]


def test_extraction_preserves_gate_fail_empty_dict(tmp_path):
    """Task 22: gate_fail 라운드 — harness.py의 RoundRecord(..., {}, "gate_fail", ...)
    패턴대로 signal={}가 그대로 실려야 함(P5 1차 로그처럼 gate_fail 연속인
    케이스를 흉내). 크래시하거나 라운드가 누락되면 안 된다."""
    p = tmp_path / "abl-test2.jsonl"
    led = Ledger(p)
    for i in range(6):  # P5 1차 kb_matmul_scalar gate_fail×6 재현
        led.append(RoundRecord("kb_matmul_scalar", i, f"hash{i}", {}, "gate_fail",
                               "correctness 실패", -1, -1.0, False, False))
    signals = _extract_signals(led, "kb_matmul_scalar")
    assert signals == [{}] * 6
    assert len(signals) == 6


def test_extraction_mixed_gate_fail_and_real_signal(tmp_path):
    """gate_fail({} 라운드)과 정상 라운드가 섞여도 순서·개수 보존."""
    p = tmp_path / "abl-test3.jsonl"
    led = Ledger(p)
    led.append(RoundRecord("p", 0, "h0", {}, "gate_fail", "x", -1, -1.0, False, False))
    led.append(RoundRecord("p", 1, "h1", {"load_eff": 0.5}, "uncoalesced",
                           "y", 4, 0.03, True, True))
    led.append(RoundRecord("p", 2, "h2", {}, "gate_fail", "x", -1, -1.0, False, False))
    signals = _extract_signals(led, "p")
    assert signals == [{}, {"load_eff": 0.5}, {}]


def test_track_result_has_signals_field_with_default():
    """Task 23: TrackResult가 signals 필드를 갖고, 미지정 시 기본값 빈 리스트."""
    t = TrackResult("evolve_ON", [(0, 0.1)], ["uncoalesced"], 0, 1, "stop_label", 1)
    assert t.signals == []


def test_to_track_reads_signals_key():
    """Task 23: _to_track이 결과 dict의 signals 키를 TrackResult로 옮긴다."""
    d = {"label": "evolve_ON", "metric_curve": [[0, 0.1], [1, 0.02]],
         "fired_rules": ["fp32_no_tensorcore", "uncoalesced"],
         "wasted_rounds": 1, "retire_count": 1, "stop_reason": "stop_label",
         "rounds": 2, "signals": [{"load_eff": 0.9}, {"load_eff": 0.3}]}
    t = _to_track(d)
    assert t.signals == [{"load_eff": 0.9}, {"load_eff": 0.3}]


def test_to_track_backward_compatible_missing_signals_key():
    """Task 23: 구버전 __ABL_RESULT__ JSON(P5 이전, signals 키 없음) 역직렬화 시
    크래시 없이 기본값 빈 리스트로 처리돼야 함(하위호환)."""
    d = {"label": "evolve_OFF", "metric_curve": [[0, -1.0]],
         "fired_rules": ["gate_fail"], "wasted_rounds": 0, "retire_count": 0,
         "stop_reason": "gate_fail", "rounds": 1}   # "signals" 키 자체가 없음
    t = _to_track(d)
    assert t.signals == []


def test_to_track_backward_compatible_gate_fail_only_result():
    """Task 23: P5 1차 로그(kb_matmul_scalar gate_fail×6, signals 키 없음) 같은
    실제 __ABL_RESULT__ 형태를 파싱해도 크래시/오판정 없어야 함."""
    d = {"label": "evolve_OFF",
         "metric_curve": [[i, -1.0] for i in range(6)],
         "fired_rules": ["gate_fail"] * 6,
         "wasted_rounds": 0, "retire_count": 0, "stop_reason": "gate_fail",
         "rounds": 6}
    t = _to_track(d)
    assert t.signals == []
    assert t.rounds == 6
    assert t.fired_rules == ["gate_fail"] * 6


def test_signal_asdict_roundtrip_matches_extraction():
    """RoundRecord.signal이 asdict() 결과와 동일 구조임을 재확인 —
    _run_track이 임의 변형 없이 있는 그대로 넘긴다는 전제 검증."""
    rec = RoundRecord("p", 0, "h", {"load_eff": 0.42, "bw_pct": 0.1},
                      "uncoalesced", "r", 4, 0.5, True, True)
    d = asdict(rec)
    assert d["signal"] == {"load_eff": 0.42, "bw_pct": 0.1}
