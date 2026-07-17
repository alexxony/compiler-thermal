# P2 T5 — P4·P7 ΔT A/B 재계산 + r_hbm_sink 범위 민감도

design: [06-p2-rc-backport-design.md](../../../ObsidianVault/HBM_build/docs/06-p2-rc-backport-design.md) Task 5

RcBackend 클래스 무변경 — `report_p4_deltat.py` rc_kw 호출측 파라미터화만 사용(P2 Task 4 인터페이스). HBM_FEM 세트: r_hbm_sink=4.670561 K/W, c_hbm=0.124017 J/K (HBM_build T2 `rc_params.csv` 실값, die 3개는 legacy 유지).

## 1. 문제별 LEGACY vs HBM_FEM (b) ΔT gap

| 문제 | LEGACY gap(K) | HBM_FEM gap(K) | 방향(둘다 TF32<fp32?) | 비고 |
|---|---|---|---|---|
| matmul | 17.1603 | 27.6192 | ✓ | 세트간 방향 일치 |
| kb_matmul_scalar | 4.3938 | 6.6176 | ✓ | 세트간 방향 일치 |
| batched_gemm | 11.1140 | 18.1690 | ✓ | 세트간 방향 일치 |
| kb_softmax | null | null | — | 개선 라운드 없음(seed=best) |

## 2. r_hbm_sink 범위 민감도 (3점: min/mid/대표)

범위 출처: HBM_build T2 `rc_params.csv` — [0.929032, 4.670561] K/W (cooling_top_bottom / baseline_8hi 두 냉각 BC 케이스, 대표값=baseline_8hi=max).

| 문제 | min gap(K) | mid gap(K) | 대표 gap(K) | flipped |
|---|---|---|---|---|
| matmul | 18.5703 | 25.5455 | 27.6192 | 아니오 |
| kb_matmul_scalar | 4.4256 | 6.1133 | 6.6176 | 아니오 |
| batched_gemm | 12.2124 | 16.8081 | 18.1690 | 아니오 |
| kb_softmax | null | null | null | — (개선 없음) |

## 3. 문제 간 순위 뒤집힘 (rank flip) 판정

- LEGACY 세트 gap 내림차순: ['matmul', 'batched_gemm', 'kb_matmul_scalar']
- HBM_FEM 세트 gap 내림차순: ['matmul', 'batched_gemm', 'kb_matmul_scalar']
- 순위 뒤집힘: 아니오

## 4. 종합 해석

**결론: 방향 불변** — LEGACY↔HBM_FEM 세트 교체, r_hbm_sink 범위 [min, mid, 대표] 3점 스윕, 문제 간 순위 어느 쪽에서도 (b) TF32<fp32 판정이 뒤집히지 않았다. "Ansys FEM 캘리브레이션으로도 결론 불변" 주장을 추가할 수 있다(설계 문서 §목표).
