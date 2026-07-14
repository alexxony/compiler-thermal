"""P8 Task 32 — 층화 추출(안 B) + 배치 분할 (08-p8-scale-ablation-design.md §3-2/§3-3).

정책: 실측 없음(이 모듈은 목록 생성만). 전 로직 순수 파이썬, torch 불요.

층화 규약(§3-2): 200문제 모집단 → compute-matmul/compute-conv-fusion/memory-norm/
memory-reduce-elementwise 4버킷(kb_convert.classify_workload) → compute 15
(matmul+fusion 버킷에서 균등 추출) + memory 15(norm+reduce 버킷에서 균등 추출) +
retire 후보 5 = 35. 결정론 시드로 재현성(§3-2 "추출 시드·목록을 결과 §8에 기록").

retire 후보 정의(휴리스틱, 명시적 판단): compute-matmul 버킷 중 표준 dense GEMM이
아닌 특이 형태(전치/대각/삼각/대칭 등, KernelBench 명명 패턴 "transposed"/"diagonal"/
"triangular"/"symmetric") — P5 kb_matmul_scalar(비표준 scalar 커널이 TF32 오탐→retire)
전례와 구조적으로 유사한 후보군. 실제 retire 발생은 실측(Task 36)에서만 확인 가능 —
이 휴리스틱은 "후보 지정"이지 "retire 보장"이 아님. 후보 부족 시 compute-matmul
나머지에서 보충.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path

RETIRE_CANDIDATE_CAP_PER_BATCH = 2  # §3-3: retire 후보는 배치당 ≤2로 캡

_COMPUTE_BUCKETS = ("compute-matmul", "compute-conv-fusion")
_MEMORY_BUCKETS = ("memory-norm", "memory-reduce-elementwise")

_RETIRE_NAME_HINTS = ("transposed", "diagonal", "triangular", "symmetric")

N_COMPUTE = 15
N_MEMORY = 15
N_RETIRE = 5
N_TOTAL = N_COMPUTE + N_MEMORY + N_RETIRE  # 35


@dataclass
class ProblemMeta:
    """추출 대상 메타 — kb_convert 결과 + 위치 정보만. 실제 소스코드는 안 들고 다님."""
    name: str
    level: int                # 1 또는 2
    bucket: str                # kb_convert.classify_workload 결과
    role: str = ""             # 추출 후 채워짐: "compute" | "memory" | "retire_candidate"


def _is_retire_hint(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in _RETIRE_NAME_HINTS)


def stratified_sample(pool: list[ProblemMeta], seed: int) -> list[ProblemMeta]:
    """§3-2 층화 추출: compute 15 + memory 15 + retire 후보 5 = 35.

    결정론(같은 pool+seed → 같은 결과, 순서 무관하게 pool 정렬 후 시드 적용).
    ValueError: 모집단이 부족해 요구 표본을 못 채우면(각 롤별) 조기 실패 —
    추측 추출 방지(빈 배치를 결과로 조용히 반환하지 않음).
    """
    # 결정론 보장을 위해 이름순 정렬 후 독립 random.Random(seed) 사용 (pool 순서 무관).
    sorted_pool = sorted(pool, key=lambda p: p.name)
    rng = random.Random(seed)

    compute_pool = [p for p in sorted_pool if p.bucket in _COMPUTE_BUCKETS]
    memory_pool = [p for p in sorted_pool if p.bucket in _MEMORY_BUCKETS]
    retire_hint_pool = [p for p in compute_pool if _is_retire_hint(p.name)]

    if len(retire_hint_pool) < N_RETIRE:
        # 후보 힌트 부족 시 나머지 compute-matmul에서 보충 (버킷 텍스트 명명이 다양해
        # transposed/diagonal 등 키워드가 늘 존재한다는 보장 없음 — 정직한 폴백).
        fallback = [p for p in compute_pool if p not in retire_hint_pool]
        rng.shuffle(fallback)
        retire_hint_pool = retire_hint_pool + fallback

    if len(retire_hint_pool) < N_RETIRE:
        raise ValueError(
            f"retire 후보 부족: {len(retire_hint_pool)} < {N_RETIRE} (compute 모집단 전체 소진)")

    rng_retire = random.Random(seed * 1000 + 1)
    retire_sample = rng_retire.sample(retire_hint_pool, N_RETIRE)
    retire_names = {p.name for p in retire_sample}

    compute_remain = [p for p in compute_pool if p.name not in retire_names]
    if len(compute_remain) < N_COMPUTE:
        raise ValueError(
            f"compute 모집단 부족: {len(compute_remain)} < {N_COMPUTE} (retire 후보 차감 후)")
    # 두 compute 버킷에서 균등 추출 — 한쪽으로 쏠리지 않게 절반씩 우선 배분.
    by_bucket = {b: [p for p in compute_remain if p.bucket == b] for b in _COMPUTE_BUCKETS}
    compute_sample = _balanced_sample(by_bucket, N_COMPUTE, random.Random(seed * 1000 + 2))

    if len(memory_pool) < N_MEMORY:
        raise ValueError(f"memory 모집단 부족: {len(memory_pool)} < {N_MEMORY}")
    by_bucket_m = {b: [p for p in memory_pool if p.bucket == b] for b in _MEMORY_BUCKETS}
    memory_sample = _balanced_sample(by_bucket_m, N_MEMORY, random.Random(seed * 1000 + 3))

    for p in compute_sample:
        p.role = "compute"
    for p in memory_sample:
        p.role = "memory"
    for p in retire_sample:
        p.role = "retire_candidate"

    result = compute_sample + memory_sample + retire_sample
    assert len(result) == N_TOTAL, f"추출 총합 불일치: {len(result)} != {N_TOTAL}"
    return result


def _balanced_sample(by_bucket: dict[str, list[ProblemMeta]], n: int,
                     rng: random.Random) -> list[ProblemMeta]:
    """버킷별 균등 목표 배분 후 셔플 추출. 한 버킷이 모자라면 다른 버킷에서 보충."""
    buckets = list(by_bucket.keys())
    per_bucket = n // len(buckets)
    remainder = n - per_bucket * len(buckets)
    result: list[ProblemMeta] = []
    shortfall = 0
    for i, b in enumerate(buckets):
        target = per_bucket + (1 if i < remainder else 0)
        pool = list(by_bucket[b])
        rng.shuffle(pool)
        take = pool[:target]
        result.extend(take)
        shortfall += max(0, target - len(take))
    if shortfall > 0:
        # 부족분을 다른 버킷의 잔여에서 보충
        used_names = {p.name for p in result}
        leftover: list[ProblemMeta] = []
        for b in buckets:
            leftover.extend(p for p in by_bucket[b] if p.name not in used_names)
        rng.shuffle(leftover)
        result.extend(leftover[:shortfall])
    if len(result) < n:
        raise ValueError(f"균등 추출 실패: {len(result)} < {n} (버킷 모집단 부족)")
    return result[:n]


def split_into_batches(sample: list[ProblemMeta], batch_size: int = 10,
                       ) -> list[list[ProblemMeta]]:
    """§3-3 배치 분할: 배치 크기 상한 + retire 후보 배치당 ≤2 캡 + 클래스 혼합.

    단순 순차 분할이 아니라, retire 후보를 배치 전체에 고르게 분산(캡 준수)한 뒤
    나머지(compute/memory)로 각 배치를 batch_size까지 채운다.
    """
    retire = [p for p in sample if p.role == "retire_candidate"]
    others = [p for p in sample if p.role != "retire_candidate"]

    n_batches = max(1, -(-len(sample) // batch_size))  # ceil division
    batches: list[list[ProblemMeta]] = [[] for _ in range(n_batches)]

    # retire 후보를 라운드로빈으로 배치에 분산(캡 자연 준수 — 5개를 n_batches에 분산).
    for i, p in enumerate(retire):
        batches[i % n_batches].append(p)

    # 나머지를 순차로 채우되 batch_size 넘지 않게.
    oi = 0
    for b in batches:
        while len(b) < batch_size and oi < len(others):
            b.append(others[oi])
            oi += 1
    while oi < len(others):
        # 모든 배치가 batch_size에 도달했는데 남았으면 새 배치 추가(안전망).
        batches.append([])
        while len(batches[-1]) < batch_size and oi < len(others):
            batches[-1].append(others[oi])
            oi += 1

    batches = [b for b in batches if b]  # 빈 배치 제거
    return batches


# ── 재현성 보완(팀리드 검증 갭 지적, 2026-07-14) — end-to-end CLI ──
# ad-hoc으로 1회 실행했던 pool 생성 스크립트를 정식 함수로 재구성. stratified_sample이
# 내부에서 pool을 이름순 재정렬(위 §결정론 보장 주석) 하므로, 여기서의 파일 워크
# 순서 자체는 결과에 영향 없음 — 재현성은 "같은 파일 집합 + 같은 seed"로 충분.

def build_pool_from_kernelbench(kb_root: str | Path) -> list[ProblemMeta]:
    """KernelBench 로컬 clone 루트(레벨별 level1/level2 하위디렉토리 보유)에서
    Level 1+2 문제 전체를 읽어 ProblemMeta 리스트로 변환(§2-2 모집단 = Level1+2).

    kb_convert.parse_kb_source + classify_workload 재사용(변환기와 동일 판정
    로직 — 분류 이중 구현 금지). AutoKernel/KernelBench 엔진 비접촉(문제 텍스트만 읽음).
    """
    import kb_convert  # 지연 import — kb_stratify 단독 임포트 시 순환·불필요 의존 방지 안전망

    kb_root = Path(kb_root)
    pool: list[ProblemMeta] = []
    for level_n, level_dir in ((1, "level1"), (2, "level2")):
        d = kb_root / level_dir
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            src = f.read_text()
            kb = kb_convert.parse_kb_source(src, name=f.stem)
            bucket = kb_convert.classify_workload(kb)
            pool.append(ProblemMeta(name=f.stem, level=level_n, bucket=bucket))
    return pool


def run_extraction(kb_root: str | Path, seed: int, batch_size: int = 10) -> dict:
    """§Task32 end-to-end: KernelBench clone 경로 → pool → 층화 추출 → 배치 분할 →
    artifacts/p8_task32_stratified_sample_*.json과 동일 구조의 dict.

    이 함수가 재현성의 단일 소스 — 같은 kb_root(같은 파일 집합) + 같은 seed는
    항상 같은 결과(byte 동일한 JSON 직렬화)를 낸다. 팀리드 검증 갭(추출 재현
    경로 부재) 대응.
    """
    pool = build_pool_from_kernelbench(kb_root)
    sample = stratified_sample(pool, seed=seed)
    batches = split_into_batches(sample, batch_size=batch_size)
    return {
        "seed": seed,
        "sample": [{"name": p.name, "level": p.level, "bucket": p.bucket,
                    "role": p.role} for p in sample],
        "batches": [[p.name for p in b] for b in batches],
    }


def main(argv: list[str]) -> int:
    import json
    import sys

    if "--selfcheck" in argv:
        # GPU·KernelBench clone 불요 — 합성 pool로 CLI 배선 확인만.
        synth_pool = (
            [ProblemMeta(name=f"cm_{i}", level=1, bucket="compute-matmul") for i in range(20)]
            + [ProblemMeta(name=f"mn_{i}", level=1, bucket="memory-norm") for i in range(20)]
        )
        sample = stratified_sample(synth_pool, seed=1)
        assert len(sample) == 35
        print("kb_stratify.py self-check PASS (합성 pool 35문제 추출 확인)")
        return 0

    if len(argv) < 1:
        print("usage: kb_stratify.py <kernelbench_root> [--seed=N] [--batch-size=N] "
              "[--out=<path.json>] [--selfcheck]", file=sys.stderr)
        return 2

    kb_root = argv[0]
    seed = 20260714
    batch_size = 10
    out_path = None
    for a in argv[1:]:
        if a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
        elif a.startswith("--batch-size="):
            batch_size = int(a.split("=", 1)[1])
        elif a.startswith("--out="):
            out_path = a.split("=", 1)[1]

    result = run_extraction(kb_root, seed=seed, batch_size=batch_size)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if out_path:
        Path(out_path).write_text(text)
        print(f"[kb_stratify] seed={seed} sample={len(result['sample'])} "
              f"batches={len(result['batches'])} -> {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
