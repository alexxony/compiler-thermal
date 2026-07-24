"""Regenerate charts/p8_bucket_gain.svg from evidence/p8_stats_final_v3_20260719.txt.

Backs README claim: KernelBench 35-problem ablation, v3 40% FAIL / v4 87.5% PASS,
2.61x-6.00x gain range on the compute-matmul bucket (section 4, P8). Parses the
per-problem M1 (energy gain ratio) lines directly from the evidence text file --
no numbers are hand-transcribed.

Run: python3 charts/make_p8_bucket_gain.py
"""
import pathlib
import re

import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence" / "p8_stats_final_v3_20260719.txt"
OUT = ROOT / "charts" / "p8_bucket_gain.svg"

ROW_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+bucket=(?P<bucket>\S+)\s+"
    r"M1=\s*(?P<m1>[\d.]+)x\s+M2=\s*(?P<m2>[\d.]+)x\s+"
    r"null=(?P<null>True|False)\s+retire=(?P<retire>\d+)\s+wasted=(?P<wasted>\d+)"
)


def parse_rows():
    rows = []
    for line in EVIDENCE.read_text().splitlines():
        m = ROW_RE.match(line)
        if m:
            rows.append(m.groupdict())
    return rows


def main():
    rows = parse_rows()
    if not rows:
        raise RuntimeError(f"no per-problem rows parsed from {EVIDENCE}")

    matmul = [r for r in rows if r["bucket"] == "compute-matmul" and r["null"] == "False"]
    matmul.sort(key=lambda r: float(r["m1"]))
    if not matmul:
        raise RuntimeError("no non-null compute-matmul rows found")

    names = [r["name"] for r in matmul]
    gains = [float(r["m1"]) for r in matmul]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gains,
        y=names,
        orientation="h",
        marker_color="#2f7a3d",
        text=[f"{g:.2f}x" for g in gains],
        textposition="outside",
    ))

    lo, hi = min(gains), max(gains)
    fig.update_layout(
        title=(f"P8 ablation — compute-matmul bucket, non-null gain range "
               f"{lo:.2f}x–{hi:.2f}x (n={len(gains)}/8 reproduced at v4 criterion)"),
        xaxis_title="energy gain (M1, TF32 vs fp32)",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13),
        margin=dict(t=70, b=40, l=260, r=60),
        width=760,
        height=360,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT))
    print(f"compute-matmul non-null rows: {len(gains)}, range {lo:.2f}x-{hi:.2f}x")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
