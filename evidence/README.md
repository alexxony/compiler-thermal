# Evidence

This folder holds a curated subset of raw run artifacts that back the headline
numbers in the top-level [README.md](../README.md). It is **not** the full
data set — `artifacts/` and `results/` remain gitignored/local per this
project's convention (see README "Repository layout" note); these specific
files are pulled out and committed so the headline claims can be checked by
someone who doesn't have local access to the rest of the run history.

Each entry below names the README claim it backs, which file(s) to open, which
field to look at, and a one-line command to re-derive the number.

## P3 — TF32 vs fp32 energy gain, 5.32×–5.75×

Files: `thermal-gain-matmul-on.jsonl`, `thermal-gain-matmul-off.jsonl`

Each file is 2 JSONL rows: round 0 (`fp32_no_tensorcore` hypothesis) and
round 1 (`tensorcore_saturated` hypothesis, TF32). The gain is the round-0 to
round-1 improvement in `signal.energy_per_iter_j`, computed independently in
each file (ON and OFF are two separate replicate runs of the same two-round
sequence, not a round-for-round comparison across files).

```
python3 -c "
import json
for name in ('thermal-gain-matmul-off.jsonl', 'thermal-gain-matmul-on.jsonl'):
    rows = [json.loads(l) for l in open(name)]
    r0 = rows[0]['signal']['energy_per_iter_j']
    r1 = rows[1]['signal']['energy_per_iter_j']
    print(name, r0 / r1)
"
```
Expect: OFF ≈ 5.320, ON ≈ 5.753 — matches the README's "5.32×–5.75×".

## P4 / P7 — RC ΔT axis, 17.16 K anchor, duty-cycle direction flip, kb_softmax null control

Files: `p6_kbms_retry_20260713_result.json`, `p7_bgemm_softmax_20260714_result.json`

These are the raw power/energy signal ledgers the ΔT figures are computed
*from* — the 17.16 K number itself is not stored in either file. It is a
derived quantity: raw `p_die_w`/`p_hbm_w` signals here feed
`thermal/twin_eval.py`'s `RcBackend` (parameters `RC_KW_LEGACY`, defined in
`tests/test_hotspot_deltat.py`, cross-validated against Twin Builder — see
next section) via `thermal_loop/report_p4_deltat.py`. The regression gate
`matmul avg(LEGACY) (b) gap == pytest.approx(17.16, abs=0.5)` is asserted
directly in `tests/test_hotspot_deltat.py` and `tests/test_hotspot_p4_deltat.py`
in the repo, and independently re-confirmed in `p11_verify_status.md` §5
below (same gate, PASS in both standalone and full-suite runs).

`p7_bgemm_softmax_20260714_result.json` → `results.kb_softmax.off` has a
single round with `stop_reason: "stop_label"` and no second round — this is
the "null control": the loop fires once (`memory_saturated`) and stops
immediately, so there is no ΔT gain claim for this problem, matching the
README's "kb_softmax null 대조군" line.

Re-derivation requires running `thermal_loop/report_p4_deltat.py` against
these signal files with `RC_KW_LEGACY`; the ledger files here let you confirm
the *inputs* (power/energy signals) are what the report code consumed, and
the regression-gate tests in the repo (not duplicated into `evidence/`,
since they are code, not data) prove the derivation.

## P8 — KernelBench 35-problem ablation, v3 40% FAIL / v4 87.5% PASS, 2.61×–6.00×

File: `p8_stats_final_v3_20260719.txt`

Per-problem M1 (energy gain ratio) and `null` columns are listed directly.
`compute-matmul`-bucket rows with `null=False` give the gain range: `2.61x`
(`7_Matmul_with_small_K_dimension_`) through `6.00x`
(`18_Matmul_with_transposed_both`). The file's own header line states
"compute-bound 재현율: 40.0% (FAIL — 판정선 70%)" and
"memory-bound null율: 100.0% (PASS)" — this is the v3, pre-registered-criterion
figure quoted in the README. The v4 87.5%/7-of-8 figure is a reclassification
of the same underlying per-problem results by bucket (not re-measured); this
file is the v3 source both are built from.

```
grep -E "^===|재현율|null율" p8_stats_final_v3_20260719.txt
grep "bucket=compute-matmul" p8_stats_final_v3_20260719.txt
```

## P10 / P11 — hotspot ΔT amplification, 1.23×–1.64×, P11 second condition (30 W), 6/6 direction match

File: `p11_verify_status.md`

This is the independent verifier's report for P11, already written as a
disk-forensic re-derivation (not a self-report by the implementer). §5
re-confirms the 17.16 K legacy anchor gate; §6 confirms the 6 new
`HOTSPOT_P4_*` RC parameter sets (30 W condition, A/B cooling series × S0–S2)
were added without touching the pre-existing P10 (16 W) labels; §4 spot-checks
all 6 `r_hbm_sink_max_p4` values against the raw HBM_build CSV to 6 decimal
places. The 1.23×–1.64× amplification ratio and the 6/6 direction-match count
are asserted in the repo's `tests/test_hotspot_p4_deltat.py` (regression-gated,
27 tests, all passing per §2–3 here) — this status file is the independent
re-run confirming those gates actually pass, not a restatement of the
implementer's claim.

The underlying `rc_params.csv` (HBM_build project, external repo) is not
duplicated here; §4 of this file lists the 6 raw values read directly from
it, so the number can be checked without needing that repo.

## RC backend validation — max error 0.0006 K / 10 s vs Twin Builder

Files: `twinbuilder_tr1_ref.csv`

Reference step-response trace from Ansys Twin Builder (TR1 solver). Compared
against this project's own `RcBackend` (forward-Euler 2-node RC model,
`thermal/twin_eval.py`) in `tests/test_twin_eval.py`, which reads this exact
CSV (`tests/data/twinbuilder_tr1_ref.csv` in the repo — same file, copied
here for evidence). Re-derive by running:

```
.venv/bin/python -m pytest tests/test_twin_eval.py -v
```

The max-error assertion against this reference trace is what backs the
"0.0006 K / 10 s" figure.

## Notes

- Values were cross-checked against the files above before this folder was
  committed; where a headline number could not be directly reproduced from a
  single file (P4/P7 ΔT axis, P10/P11 amplification ratios), that is stated
  explicitly rather than silently treating the file as if it contained the
  final number — those are derived quantities computed by
  `thermal_loop/report_p4_deltat.py` from the RC model constants plus the raw
  signal ledgers, not measured directly.
- `artifacts/` and `results/` at the repo root remain the full local run
  history and stay outside version control; this folder is a fixed,
  point-in-time export, not a live mirror.
