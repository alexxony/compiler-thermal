"""P8 Task 33/34 — 결과 회수 파일 다운로드 승격 + 재개(skip-completed) 로직 테스트.

torch/colab 불요 — 순수 파이썬 로직만(파일 I/O는 tmp_path 픽스처).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from run_ablation_remote import (  # noqa: E402
    completed_problems_from_result,
    completed_problems_from_result_dict,
    filter_uncompleted,
    merge_batch_results,
    _to_track,
)


def _fake_track_dict(label):
    return {"label": label, "metric_curve": [(0, -1.0), (1, -0.5)],
            "fired_rules": ["fp32_no_tensorcore"], "wasted_rounds": 0,
            "retire_count": 0, "stop_reason": "converged", "rounds": 2,
            "signals": [{}, {"load_eff": 0.9}]}


def test_completed_problems_from_result_empty_when_no_file(tmp_path):
    missing = tmp_path / "abl_result.json"
    assert completed_problems_from_result(missing) == set()


def test_completed_problems_from_result_reads_both_tracks_present(tmp_path):
    result_path = tmp_path / "abl_result.json"
    payload = {"chip": "a100", "results": {
        "matmul": {"off": _fake_track_dict("evolve_OFF"),
                   "on": _fake_track_dict("evolve_ON")},
    }}
    result_path.write_text(json.dumps(payload))
    assert completed_problems_from_result(result_path) == {"matmul"}


def test_completed_problems_from_result_excludes_partial_problem(tmp_path):
    # off만 있고 on이 없으면 미완결 — skip 대상 아님(전량 재실행 필요)
    result_path = tmp_path / "abl_result.json"
    payload = {"chip": "a100", "results": {
        "matmul": {"off": _fake_track_dict("evolve_OFF")},
    }}
    result_path.write_text(json.dumps(payload))
    assert completed_problems_from_result(result_path) == set()


def test_filter_uncompleted_skips_already_done_problems():
    problems = ["matmul", "kb_softmax", "batched_gemm"]
    done = {"matmul"}
    remain = filter_uncompleted(problems, done)
    assert remain == ["kb_softmax", "batched_gemm"]


def test_filter_uncompleted_no_skip_when_empty_done_set():
    problems = ["matmul", "kb_softmax"]
    assert filter_uncompleted(problems, set()) == problems


def test_merge_batch_results_combines_multiple_batch_jsons(tmp_path):
    b1 = {"chip": "a100", "results": {"matmul": {"off": _fake_track_dict("evolve_OFF"),
                                                  "on": _fake_track_dict("evolve_ON")}}}
    b2 = {"chip": "a100", "results": {"kb_softmax": {"off": _fake_track_dict("evolve_OFF"),
                                                      "on": _fake_track_dict("evolve_ON")}}}
    p1 = tmp_path / "batch0_result.json"
    p2 = tmp_path / "batch1_result.json"
    p1.write_text(json.dumps(b1))
    p2.write_text(json.dumps(b2))
    merged = merge_batch_results([p1, p2])
    assert set(merged["results"].keys()) == {"matmul", "kb_softmax"}
    assert merged["chip"] == "a100"


def test_merge_batch_results_later_batch_overwrites_same_problem(tmp_path):
    # 재개 시 동일 문제가 두 배치 결과 파일에 있으면 나중(최신) 것을 채택
    b1 = {"chip": "a100", "results": {"matmul": {"off": _fake_track_dict("evolve_OFF"),
                                                  "on": _fake_track_dict("evolve_ON")}}}
    b2_track = _fake_track_dict("evolve_ON")
    b2_track["rounds"] = 99  # 구분용 마커
    b2 = {"chip": "a100", "results": {"matmul": {"off": _fake_track_dict("evolve_OFF"),
                                                  "on": b2_track}}}
    p1 = tmp_path / "batch0_result.json"
    p2 = tmp_path / "batch1_result.json"
    p1.write_text(json.dumps(b1))
    p2.write_text(json.dumps(b2))
    merged = merge_batch_results([p1, p2])
    assert merged["results"]["matmul"]["on"]["rounds"] == 99


def test_completed_problems_from_result_dict_matches_file_variant():
    # P8 Task 36 이상 징후 4 — main()의 완결성 검증(요청 문제 vs 회수 결과)이 쓰는
    # dict 직접 버전. 파일 버전과 동일 판정 기준(off+on 둘 다) 공유 확인.
    payload = {"chip": "a100", "results": {
        "matmul": {"off": _fake_track_dict("evolve_OFF"),
                   "on": _fake_track_dict("evolve_ON")},
        "kb_softmax": {"off": _fake_track_dict("evolve_OFF")},  # on 없음 — 미완결
    }}
    assert completed_problems_from_result_dict(payload) == {"matmul"}


def test_completed_problems_from_result_dict_empty_for_no_results_key():
    assert completed_problems_from_result_dict({"chip": "a100"}) == set()


def test_partial_batch_result_flagged_incomplete_against_requested_problems():
    # P8 Task 36 이상 징후 4 재현 — 요청 5문제 중 2문제만 원격이 기록한 부분 결과를
    # main()이 완료로 오판했던 사고. 완결성 검증 로직이 요청 집합과 대조해
    # 누락을 정확히 잡아내는지 확인(회귀 방지).
    requested = ["66_conv_standard_3D", "33_BatchNorm", "35_GroupNorm_",
                 "36_RMSNorm_", "37_FrobeniusNorm_"]
    partial_result = {"chip": "a100", "results": {
        "66_conv_standard_3D": {"off": _fake_track_dict("evolve_OFF"),
                                 "on": _fake_track_dict("evolve_ON")},
        "33_BatchNorm": {"off": _fake_track_dict("evolve_OFF"),
                          "on": _fake_track_dict("evolve_ON")},
    }}
    done = completed_problems_from_result_dict(partial_result)
    missing = [p for p in requested if p not in done]
    assert missing == ["35_GroupNorm_", "36_RMSNorm_", "37_FrobeniusNorm_"]


def test_merge_batch_results_roundtrips_through_to_track(tmp_path):
    b1 = {"chip": "a100", "results": {"matmul": {"off": _fake_track_dict("evolve_OFF"),
                                                  "on": _fake_track_dict("evolve_ON")}}}
    p1 = tmp_path / "batch0_result.json"
    p1.write_text(json.dumps(b1))
    merged = merge_batch_results([p1])
    track = _to_track(merged["results"]["matmul"]["on"])
    assert track.rounds == 2
    assert track.signals[1]["load_eff"] == 0.9
