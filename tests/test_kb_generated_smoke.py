"""P8 Task 31 — 생성 problems/ 전수 CPU 스모크 테스트 (3차 검증 재작업, 2026-07-14).

목적: compile()/AST 스캔은 구조 버그(미정의 이름, 문법 오류)만 잡고 **런타임 버그**
(모듈 상수 미방출, super() 클래스명 불일치 등)는 놓친다 — 3차 검증에서 stateful
21/35 문제가 compile은 통과하고 실제 import/실행에서 죽는 것으로 확인됨(팀리드
지적, 2026-07-14). 이 테스트가 "컴파일은 되나 런타임에 죽는" 버그 클래스를 구조적으로
차단 — `problems/`의 모든 solve.py를 실제 import → make_case() → reference() 1회
실행까지 CPU(GPU 불요)에서 예외 없이 도는지 전수 검증한다.

GPU 불요 — CPU torch로 충분(`pip install --index-url https://download.pytorch.org/whl/cpu
torch`로 로컬 venv에 설치, 이 프로젝트 관례상 로컬엔 torch 없었으나 이 검증엔 필수라
설치). run_solve()는 커널 자체(Triton 등 GPU 전용 코드 포함 가능)라 스모크 대상에서
제외 — make_case/reference(순수 PyTorch 조합)만 검증(§executor 계약 중 GPU-무관 부분).

**격리 실행 이유**: 초기 구현은 단일 pytest 프로세스 안에서 in-process import로
35~40문제를 순차 실행했는데, 큰 KernelBench 원본 치수(예: ConvTranspose3d
batch=16·channel 64→128·spatial 24×48×48)를 쓰는 문제가 프로세스당 RSS 3GB+를
잡아먹어 WSL 7.5GB 메모리 한도에서 OOM kill 발생(dmesg 확인:
`Killed process ... anon-rss:3680268kB`). 단일 문제만 별도 프로세스로 돌리면
3GB peak·5.5초에 정상 완주하는 것도 확인됨 — 문제는 누적이 아니라 in-process
공유 상태(torch 스레드풀·이전 텐서 참조)였다. 고쳐서: 문제당 **완전히 독립된
subprocess**로 스모크(각 프로세스가 끝나면 OS가 메모리를 전부 회수 — GC 신뢰
불요). 배치 크기가 크지 않으니(35~40) 여전히 marker 분리 없이 전수 실행(팀리드
명시 지시 "이번 검증에선 전수 실행"), 단 각 항목이 subprocess라 서로 격리됨.

**SIGKILL(rc=-9) skip 처리 근거(P8 Task 31 4차 검증, 2026-07-15)**: kb_convert.py
버그(패턴 A/B/import 누락) 수정 완료 후 재실행에서도 일부 문제가 rc=-9로 계속
실패. 원인 조사 결과 이 로컬 WSL 세션의 스왑(2.0Gi)이 100% 소진된 전역 메모리
압박 상태였고, **텐서 크기와 무관하게** SIGKILL이 발생함을 실측으로 확인:
(1) `90_cumprod`의 실제 텐서는 `input_shape=(32768,)` 1차원 벡터(~128KB, 전혀
메모리 무거운 문제 아님)인데도 반복 실행 모두 SIGKILL — 크기 기반 판단이 거짓
양성을 낸다는 반증 사례. (2) `78_ConvTranspose3d_Max_Max_Sum`은 동일 코드로
1회차 PASS·2회차 FAIL — flaky, 즉 문제 자체의 결정적 결함이 아니라 그 순간의
시스템 여유 메모리에 의존. 따라서 SIGKILL은 "이 solve.py가 깨졌다"는 코드
결함 증거가 아니라 "이 로컬 환경이 지금 메모리 여유가 없다"는 **환경 신호**로
취급한다 — 문제 이름 기반 하드코딩 skip 리스트는 만들지 않는다(그 자체가
90_cumprod 같은 오분류를 낳음). rc == -9인 경우에만 pytest.skip 처리하고, 그
외 모든 비정상 종료(예외 트레이스백, 다른 시그널, 논제로 rc)는 그대로 실패
처리 — 최종 정오답 판정은 원격 A100 VM 스모크(메모리 여유·GPU 확보)가 맡는다.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

PROBLEMS_ROOT = Path(__file__).resolve().parents[1] / "problems"
_PYTHON = sys.executable

_SMOKE_SNIPPET = """
import sys
sys.path.insert(0, {prob_dir!r})
import torch
torch.set_num_threads(1)  # 스레드풀 버퍼로 인한 불필요 메모리 압박 축소
import solve
assert hasattr(solve, "make_case"), "make_case 없음"
assert hasattr(solve, "reference"), "reference 없음"
assert hasattr(solve, "GATE_SIZES"), "GATE_SIZES 없음"
size = solve.GATE_SIZES[0]
case = solve.make_case(size, "cpu")
assert case is not None, "make_case가 None 반환"
ref = solve.reference(case, "cpu")
assert ref is not None, "reference가 None 반환"
assert isinstance(ref, torch.Tensor), f"reference 반환값이 Tensor 아님 ({{type(ref)}})"
print("SMOKE_OK")
"""


def _discover_problem_dirs() -> list[Path]:
    if not PROBLEMS_ROOT.is_dir():
        return []
    return sorted(p for p in PROBLEMS_ROOT.iterdir()
                  if p.is_dir() and (p / "solve.py").exists())


_PROBLEM_DIRS = _discover_problem_dirs()
_PROBLEM_IDS = [p.name for p in _PROBLEM_DIRS]


@pytest.mark.parametrize("prob_dir", _PROBLEM_DIRS, ids=_PROBLEM_IDS)
def test_generated_problem_cpu_smoke(prob_dir: Path):
    """독립 subprocess에서 import → make_case(GATE_SIZES[0], 'cpu') →
    reference(case, 'cpu') 예외 없이 완주(각 프로세스 종료 시 OS가 메모리 전량 회수
    — 이전 문제의 텐서/스레드풀 잔재가 다음 문제로 누적되지 않음).

    실패 시 pytest가 문제 이름(ids=_PROBLEM_IDS)을 테스트명에 노출해 35개 중
    정확히 몇 번 문제가 죽었는지 즉시 식별 가능(정직한 실패 보고, 뭉뚱그리지 않음).
    타임아웃(300초/문제)은 무한 대기 방지용 안전망 — 정상 케이스는 수 초~수십 초.

    rc==-9(SIGKILL) skip 처리(팀리드 결정, 2026-07-15): 모듈 docstring의 근거대로
    이 로컬 환경의 시스템 전역 메모리 압박에 의한 신호일 뿐 코드 결함이 아니므로
    fail이 아닌 skip — 원격 A100 스모크가 최종 판정. **문제 이름 기반 분기는
    절대 금지** — rc 값 하나만 보고 판단(하드코딩 리스트는 90_cumprod 같은
    오분류를 낳음, 모듈 docstring 참조). 그 외 모든 비정상 종료(트레이스백 있는
    rc=1, 다른 시그널, 타임아웃 등)는 그대로 fail 유지.

    triton 미설치 skip 처리(2026-07-22): rc==1이면서 stderr에
    `ModuleNotFoundError.*triton` 패턴이 잡히는 경우도 동일 원칙(문제 이름이
    아니라 stderr 신호로 판단)으로 skip — 이 로컬 venv에 triton이 없어 생기는
    환경 문제이지 solve.py의 코드 결함이 아니다.
    """
    snippet = _SMOKE_SNIPPET.format(prob_dir=str(prob_dir))
    result = subprocess.run([_PYTHON, "-c", snippet], capture_output=True,
                            text=True, timeout=300)
    if result.returncode == -9:
        pytest.skip("SIGKILL(OOM 추정) — 로컬 메모리 제약, A100 원격 스모크에서 검증")
    if result.returncode == 1 and re.search(r"ModuleNotFoundError.*triton", result.stderr):
        pytest.skip("triton 미설치 — 로컬 환경에 triton 미설치, 해당 KB 변환 문제는 CPU 스모크 스킵")
    assert result.returncode == 0 and "SMOKE_OK" in result.stdout, (
        f"{prob_dir.name} 스모크 실패 (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr[-3000:]}")


def test_at_least_one_problem_discovered():
    # discover 함수 자체가 조용히 빈 리스트를 내면 위 parametrize가 0개 테스트로
    # 사라져 "전수 통과"처럼 보이는 거짓 양성 위험 — 최소 1개 발견을 별도 강제.
    assert len(_PROBLEM_DIRS) > 0, "problems/ 하위에서 solve.py 보유 디렉토리 0개 발견"
