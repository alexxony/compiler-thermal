"""Regenerate charts/matmul_energy.svg from evidence/thermal-gain-matmul-{on,off}.jsonl.

Backs README claim: TF32 vs fp32 energy gain, 5.32x-5.75x (section 4, P3).
Each evidence file holds 2 JSONL rows (round 0 = fp32_no_tensorcore,
round 1 = tensorcore_saturated / TF32). Gain is computed independently within
each file (ON and OFF are separate replicate runs, not a round-for-round
cross-file comparison) -- see evidence/README.md P3 section for the exact
derivation this script mirrors.

Run: python3 charts/make_matmul_energy.py
"""
import json
import pathlib

import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "charts" / "matmul_energy.svg"


def load(name):
    rows = [json.loads(line) for line in (EVIDENCE / name).read_text().splitlines()]
    r0 = rows[0]["signal"]["energy_per_iter_j"]
    r1 = rows[1]["signal"]["energy_per_iter_j"]
    return r0, r1, r0 / r1


def main():
    off_r0, off_r1, off_gain = load("thermal-gain-matmul-off.jsonl")
    on_r0, on_r1, on_gain = load("thermal-gain-matmul-on.jsonl")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="fp32 (round 0, fp32_no_tensorcore)",
        x=["OFF replicate", "ON replicate"],
        y=[off_r0, on_r0],
        marker_color="#8a1f1f",
        text=[f"{off_r0:.2f} J", f"{on_r0:.2f} J"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="TF32 (round 1, tensorcore_saturated)",
        x=["OFF replicate", "ON replicate"],
        y=[off_r1, on_r1],
        marker_color="#2f7a3d",
        text=[f"{off_r1:.2f} J", f"{on_r1:.2f} J"],
        textposition="outside",
    ))

    fig.add_annotation(x="OFF replicate", y=off_r0, yshift=40,
                        text=f"{off_gain:.2f}x", showarrow=False,
                        font=dict(size=14, color="#5c5c5c"))
    fig.add_annotation(x="ON replicate", y=on_r0, yshift=40,
                        text=f"{on_gain:.2f}x", showarrow=False,
                        font=dict(size=14, color="#5c5c5c"))

    fig.update_layout(
        barmode="group",
        title="matmul energy_per_iter_j: fp32 to TF32, two independent replicates",
        yaxis_title="energy per iteration (J)",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        margin=dict(t=90, b=40, l=60, r=20),
        width=720,
        height=440,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT))
    print(f"OFF gain: {off_gain:.4f}x  ON gain: {on_gain:.4f}x")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
