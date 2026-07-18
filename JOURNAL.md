# JOURNAL.md — Compiler_Thermal 원장 (append-only)

관례: 이벤트 발생 즉시 1~2줄 append + 즉시 커밋한다. 정제된 단계별 요약은
vault `/mnt/c/ObsidianVault/Compiler_Thermal/PROGRESS.md`에 기록한다(2층 구조,
2026-07-18 승인). 타임스탬프는 ISO 8601 + 명시적 시간대(KST, `+09:00`)를 쓴다.

---

## 2026-07-18

- `2026-07-18T09:41:54+09:00` — P8 metric_mode 가드 도입: 결과 스탬프·회수/resume 모드 검증·기본값 경고 배너 (`cb999e2`). latency 모드 오염 재발 방지.
- `2026-07-18T09:49:51+09:00` — 배치 2a 사고 후속 2건 수정: `_alive` TimeoutExpired 처리 + 트랙 종료마다 GPU 캐시 반납 (`ca13c68`).
- `2026-07-18T09:xx` (GroupNorm 진단, 커밋 없음) — 34_InstanceNorm/35_GroupNorm_ 등 6문제에서 `torch.OutOfMemoryError` 재현: `_run_gate`가 케이스 텐서+후보 출력+reference 중간값을 GPU에 동시 보유 → A100 40GB 초과(35.48GiB 점유 중 7GiB 추가요청). 6문제 격리 확정, 진단 로그 `artifacts/p8_35_groupnorm_diag.log` 보존.
- `2026-07-18T (시각 미상, 정적 감사)` — `audit_case_sizes.py`로 35문제 케이스 크기 정적 감사 + 격리 6문제 확정 실측(29/35), 통계 1차 산출(compute 0% FAIL, memory null 100% PASS) (`8ad395f`). 산출물: `artifacts/p8_case_size_audit.json`, `artifacts/p8_stats_final_20260718.txt`, `artifacts/p8_buckets.json`.
- `2026-07-18T (T1 착수)` — gate 메모리 정책 수정: `_run_gate`가 후보 출력을 얻은 직후 CPU로 옮기고 GPU 참조 해제+`empty_cache()`, reference 실행도 OOM 시 케이스 전체(nn.Module 포함, `_case_to` 신설)를 CPU로 내려 재시도하도록 변경. 비교는 CPU 텐서끼리, atol/rtol 값 무변경 (`1b8367a`). 측정 경로(`_profile_ncu`/`_profile_event`/`_profile_power`)는 별도 함수라 비접촉 확인.
- `2026-07-18T18:09 (KST, 추정)` — 구제 런1(31_ELU,25_Swish,36_RMSNorm_) 완료 확인되었으나 로컬 미저장 + 세션 사멸로 결과 유실. 원인: `run_ablation_remote.py`의 fetch-first 미이행 — fetch 성공 후 로컬 저장이 호출자 관례에 위임되어 있어, 세션이 idle 종료되면 원격 유일 사본이 함께 소멸하는 유실 창(window)이 존재. 이후 3.4시간 무진행 상태로 방치됨(백그라운드 완료 알림이 하네스 단에서 유실되어 재확인 트리거가 없었음).
- `2026-07-18T22:00:02+09:00` — 러너 autosave 수정: `main()`이 fetch 성공 즉시 결과를 무조건 `artifacts/abl_autosave_<ISO8601+tz>[_partial].json`으로 로컬 보존하도록 변경(부분 결과 포함) — 유실 창 구조 제거 (`4435c6b`).
- `2026-07-18T (오케스트레이터 직접 회수)` — 팀리드가 T2(격리 6문제 구제 실측)를 직접 회수: 31_ELU,25_Swish,36_RMSNorm_,37_FrobeniusNorm_,34_InstanceNorm,35_GroupNorm_ 6문제 전체를 단일 배치로 재발진(`p8_rescue_all6_20260718.log`) → 6/6 완결(off+on 전부 존재, `metric_mode=thermal` 스탬프 확인). gate 수정(T1, `1b8367a`) 적용 이후 실행이라 CPU 폴백 0회 — 첫 시도(GPU)에서 전부 통과, `artifacts/p8_rescue_all6_20260718_result.json` 저장.
- `2026-07-18T22:39:32+09:00` — T3: `report_p8_stats.py`로 기존 29표본(batch0_full+1a+1b_partial+2a_thermal_partial+2b_thermal+3_thermal, 문제 집합 검증 완료 — 원본 stats 파일과 정확히 일치) + 구제 6문제를 병합해 35표본 전체 재판정 → `artifacts/p8_stats_final_v2_20260718.txt`. compute-bound 재현율 0.0%(FAIL, 판정선 70% 유지), memory-bound null율 100.0%(PASS), retire 발생률 27.6%→28.6%, gate_fail 무효 표본 0건 — 구제 6문제 편입 후에도 §1-2 판정 결론 불변.
- `2026-07-18T (T4)` — JOURNAL.md 도입: append-only 원장 관례 명문화, 오늘자 이력 소급 기록.

- `2026-07-18T23:15:00+09:00` — P9 후보 A 착수: P8 compute-bound 0% 원인 조사(로컬 정적 분석 전용). `_variant_map_for`/`_make_cb` 폴백 구조 확인 + 35표본 전체 fired_rules/variant 파일 실재 여부 대조.
- `2026-07-18T23:35:00+09:00` — P9 후보 A 완료: 원가설(variant 파일 부재→seed 폴백) 기각 — compute 20문제 전부 variant 파일 보유(초기 ls 오탐 정정, os.listdir 재확인). 실제 분류: (a)0 (b)variant적용+gain없음 9 (c)STOP즉시발화 9 (d)미발화 2. retire 10건은 메커니즘 정상이나 도착지도 무효. 레거시(matmul 등) 대비 코드 구조 동일·결과만 다름 — A100 ncu_breakdown 재실측 권고(claude-smart s1-308 프로토콜). 산출물 `artifacts/p9a_cause_analysis.md`(미커밋, 관례).
