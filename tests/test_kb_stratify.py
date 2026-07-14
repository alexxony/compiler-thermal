"""P8 Task 32 — 층화 추출 + 배치 분할 테스트 (08-p8-scale-ablation-design.md §3-2/§5).

200문제(fixture)를 4버킷 분류 → compute 15 + memory 15 + retire 후보 5 = 35 추출.
결정론 시드 재현성, 클래스 균형, retire 캡(배치당 ≤2) 검증. torch 불요 — 순수 로직.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thermal_loop"))

from kb_stratify import (  # noqa: E402
    ProblemMeta,
    stratified_sample,
    split_into_batches,
    build_pool_from_kernelbench,
    run_extraction,
    RETIRE_CANDIDATE_CAP_PER_BATCH,
)


def _fixture_pool(n_compute_matmul=40, n_compute_fusion=40,
                   n_memory_norm=60, n_memory_reduce=60):
    pool = []
    for i in range(n_compute_matmul):
        pool.append(ProblemMeta(name=f"cm_{i}", level=1, bucket="compute-matmul"))
    for i in range(n_compute_fusion):
        pool.append(ProblemMeta(name=f"cf_{i}", level=2, bucket="compute-conv-fusion"))
    for i in range(n_memory_norm):
        pool.append(ProblemMeta(name=f"mn_{i}", level=1, bucket="memory-norm"))
    for i in range(n_memory_reduce):
        pool.append(ProblemMeta(name=f"mr_{i}", level=1, bucket="memory-reduce-elementwise"))
    return pool


def test_stratified_sample_returns_35_total():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    assert len(sample) == 35


def test_stratified_sample_class_balance_compute_15_memory_15_retire_5():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    compute_n = sum(1 for p in sample if p.role == "compute")
    memory_n = sum(1 for p in sample if p.role == "memory")
    retire_n = sum(1 for p in sample if p.role == "retire_candidate")
    assert compute_n == 15
    assert memory_n == 15
    assert retire_n == 5


def test_stratified_sample_is_deterministic_given_seed():
    pool = _fixture_pool()
    s1 = stratified_sample(pool, seed=7)
    s2 = stratified_sample(pool, seed=7)
    assert [p.name for p in s1] == [p.name for p in s2]


def test_stratified_sample_different_seed_can_differ():
    pool = _fixture_pool()
    s1 = stratified_sample(pool, seed=1)
    s2 = stratified_sample(pool, seed=2)
    assert [p.name for p in s1] != [p.name for p in s2]


def test_stratified_sample_no_duplicates():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    names = [p.name for p in sample]
    assert len(names) == len(set(names))


def test_stratified_sample_compute_drawn_from_both_compute_buckets():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    compute_names = {p.name for p in sample if p.role == "compute"}
    assert any(n.startswith("cm_") for n in compute_names)
    assert any(n.startswith("cf_") for n in compute_names)


def test_stratified_sample_raises_when_pool_insufficient():
    small_pool = [ProblemMeta(name="x", level=1, bucket="compute-matmul")]
    try:
        stratified_sample(small_pool, seed=1)
        assert False, "should have raised"
    except ValueError:
        pass


def test_split_into_batches_respects_size_cap():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    batches = split_into_batches(sample, batch_size=10)
    assert all(len(b) <= 10 for b in batches)
    assert sum(len(b) for b in batches) == 35


def test_split_into_batches_retire_candidates_capped_per_batch():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    batches = split_into_batches(sample, batch_size=10)
    for b in batches:
        retire_in_batch = sum(1 for p in b if p.role == "retire_candidate")
        assert retire_in_batch <= RETIRE_CANDIDATE_CAP_PER_BATCH


def test_split_into_batches_four_batches_for_35_at_size_10():
    pool = _fixture_pool()
    sample = stratified_sample(pool, seed=42)
    batches = split_into_batches(sample, batch_size=10)
    assert len(batches) == 4  # 10+10+10+5 (§3-3 8~10문제/배치 → 층화 35 = 4배치)


# ── 재현성 보완(팀리드 갭 지적) — end-to-end CLI: KernelBench 경로 → pool → 추출 ──

def _write_kb_fixture(root):
    """합성 KernelBench 디렉토리(level1/level2, Model 포맷 .py) 생성 — 실 clone 불요.

    kb_convert.parse_kb_source가 요구하는 최소 계약(Model/forward/get_inputs/
    get_init_inputs)을 만족하는 합성 파일들로 build_pool_from_kernelbench의
    디렉토리 워크 + 분류 파이프라인을 검증(실 200파일과 별개, 로직 검증용).
    """
    import textwrap
    level1 = root / "level1"
    level2 = root / "level2"
    level1.mkdir(parents=True)
    level2.mkdir(parents=True)

    def _matmul_src(n):
        return textwrap.dedent(f'''\
            import torch
            import torch.nn as nn
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, A, B):
                    return torch.matmul(A, B)
            N = {n}
            def get_inputs():
                return [torch.rand(N, N), torch.rand(N, N)]
            def get_init_inputs():
                return []
            ''')

    def _softmax_src():
        return textwrap.dedent('''\
            import torch
            import torch.nn as nn
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x):
                    return torch.softmax(x, dim=1)
            def get_inputs():
                return [torch.rand(4, 8)]
            def get_init_inputs():
                return []
            ''')

    # compute-matmul 다수(15+ 채우기 위해 20개) + memory(softmax류, norm 텍스트 포함) 20개
    for i in range(1, 21):
        (level1 / f"{i}_Matmul_case_{i}.py").write_text(_matmul_src(i))
    for i in range(21, 41):
        (level1 / f"{i}_Softmax_norm_case_{i}.py").write_text(_softmax_src())
    # retire 후보 힌트(transposed/diagonal 등) 포함 matmul 변형 5개
    hints = ("transposed", "diagonal", "triangular", "symmetric", "transposed_both")
    for i, h in enumerate(hints, start=41):
        (level1 / f"{i}_Matmul_{h}_case.py").write_text(_matmul_src(i))
    # level2에도 소량(버킷 다양성)
    for i in range(1, 6):
        (level2 / f"{i}_Gemm_fusion_case_{i}.py").write_text(_matmul_src(100 + i))


def test_build_pool_from_kernelbench_walks_level1_and_level2(tmp_path):
    _write_kb_fixture(tmp_path)
    pool = build_pool_from_kernelbench(tmp_path)
    names = {p.name for p in pool}
    assert len(pool) == 20 + 20 + 5 + 5  # level1 40 + retire5 + level2 5 = 50
    assert any(n.startswith("1_Matmul_case_1") for n in names)
    assert all(p.level in (1, 2) for p in pool)


def test_build_pool_from_kernelbench_classifies_buckets(tmp_path):
    _write_kb_fixture(tmp_path)
    pool = build_pool_from_kernelbench(tmp_path)
    buckets = {p.bucket for p in pool}
    assert "compute-matmul" in buckets
    assert any(b.startswith("memory-") for b in buckets)


def test_run_extraction_end_to_end_produces_35_and_batches(tmp_path):
    _write_kb_fixture(tmp_path)
    result = run_extraction(str(tmp_path), seed=20260714)
    assert result["seed"] == 20260714
    assert len(result["sample"]) == 35
    assert sum(len(b) for b in result["batches"]) == 35


def test_run_extraction_is_reproducible_same_seed_same_output(tmp_path):
    _write_kb_fixture(tmp_path)
    r1 = run_extraction(str(tmp_path), seed=20260714)
    r2 = run_extraction(str(tmp_path), seed=20260714)
    assert r1 == r2
