"""칩 1급 상수. pj_per_bit는 문헌값 — 01-prior-art-thermal 조사 노트 인용 (HBM2E ≈ 4.3 pJ/bit)."""
CHIP_CAPS = {
    "NVIDIA A100-SXM4-40GB": {"tdp_w": 400, "mem_type": "HBM2e", "pj_per_bit": 4.3},
    "NVIDIA L4":             {"tdp_w": 72,  "mem_type": "GDDR6",  "pj_per_bit": 7.0},
    "Tesla T4":              {"tdp_w": 70,  "mem_type": "GDDR6",  "pj_per_bit": 7.0},
}
