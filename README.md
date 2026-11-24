# gdn-triton-ascend

Triton-Ascend implementation and performance optimization of **Gated Delta Rule (GDN)** training kernels on **Ascend 910B (910B2C)**.

The repository contains:

- a self-contained Triton-Ascend implementation of `GatedDeltaRuleTraining` (forward + backward, varlen, bf16/fp32);
- optimization notes for the WY-adjoint and dqkwg rounds;
- frozen cases and raw benchmark results for a four-way comparison.

## Results

### 3 representative cases (official verifier, warmup 5 / repeats 50)

| Metric | Value |
|---|---:|
| Precision | **3/3 PASS** |
| Speedup vs flash ACLNN | **1.1379x** |
| Speedup vs pure Triton | **2.3151x** |
| Implementation latency | **21.04 ms** |
| Flash ACLNN reference | 23.95 ms |
| Pure Triton reference | 48.74 ms |
| Sanitized-source recheck | 1.1214x, impl 21.23 ms, 3/3 |

### Full 576 cases (Event, warmup 5 / repeats 10)

| Implementation | Median latency | vs flash ACLNN | vs pure Triton |
|---|---:|---:|---:|
| flash ACLNN | 27.28 ms | 1.000x | 2.030x |
| pure Triton | 55.39 ms | 0.495x | 1.000x |
| g087 版 | 25.11 ms | 1.088x | 2.206x |
| **dqkwg 版 (this repo)** | **24.29 ms** | **1.133x** | **2.281x** |

- The dqkwg implementation is >1.05x vs flash ACLNN on **576/576** cases, >1.10x on **564/576**, and >1.15x on **52/576**; minimum 1.0842x.
- It is **3.4%** faster than the g087 version (25.11 → 24.29 ms).

## Repository layout

```
.
├── README.md
├── LICENSE                       # placeholder; see THIRD_PARTY_NOTICES.md
├── THIRD_PARTY_NOTICES.md
├── SHA256SUMS
├── src/
│   ├── GatedDeltaRuleTraining_triton_optimized.py   # optimized implementation (self-contained)
│   ├── GatedDeltaRuleTraining_torch.py              # reference ACLNN wrapper
│   └── GatedDeltaRuleTraining.json                  # 3 representative cases
├── docs/
│   ├── optimization_report.md                       # dqkwg optimization notes
│   ├── optimization_attempts.md                     # attempts / rejected directions
│   └── performance_comparison.md                    # 576-case comparison report
├── results/
│   ├── verify_result_aclnn.json                     # 3-case precision (3/3)
│   ├── perf_vs_aclnn.json                           # 3-case vs ACLNN (1.1379x)
│   ├── perf_vs_puretriton.json                      # 3-case vs pure Triton (2.3151x)
│   ├── recheck_verify.json                          # sanitized-source recheck (3/3)
│   ├── recheck_perf_vs_aclnn.json                   # sanitized-source recheck (1.1214x)
│   ├── full_flash.json                              # 576-case flash ACLNN
│   ├── full_pure.json                               # 576-case pure Triton
│   ├── full_g087.json                               # 576-case g087 version
│   └── full_local_dqkwg.json                        # 576-case dqkwg version
└── cases/
    └── full/GatedDeltaRuleTraining.json             # 576-case corpus
```

## Quick start

Tested environment: Ascend 910B2C, CANN 9.0.0, torch 2.10.0+cpu, torch-npu 2.10.0.post2,
triton 3.2.0, triton-ascend 3.2.1, `fla_npu` with the matching OPP.

The optimized implementation is self-contained:

```python
import importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    "gdr_opt", "src/GatedDeltaRuleTraining_triton_optimized.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.ModelNew().eval()

# q/k/v/grad_o: bf16 [1, T, H, K/V]; g: fp32 [1, T, H]; beta: bf16; cu_seqlens: int32
out, dq, dk, dv, dg, dbeta = model(q, k, v, g, beta, cu_seqlens, cu_seqlens_list, grad_o)
```

Reference ACLNN wrapper (`src/GatedDeltaRuleTraining_torch.py`):

```bash
export GDR_VEOMNI_ROOT=/path/to/VeOmni   # must contain veomni/ops/kernels/gated_delta_rule/npu_varlen/
export GDR_SKIP_SOURCE_HASH=1            # optional: skip the exact-snapshot SHA check
```

Cases:

- `src/GatedDeltaRuleTraining.json` — 3 representative cases (1/196/540);
- `cases/full/GatedDeltaRuleTraining.json` — 576 cases (192 shapes × 3).

## Reports

- [docs/optimization_report.md](docs/optimization_report.md) — dqkwg optimization points and end-to-end results.
- [docs/optimization_attempts.md](docs/optimization_attempts.md) — adopted/reverted attempts and rejected directions.
- [docs/performance_comparison.md](docs/performance_comparison.md) — full 576-case comparison and per-case distribution.

## Precision

- Strict precision uses the eco open-source standard, golden = flash ACLNN. The optimized version passes 3/3 representative cases.
- The 576-case run checks `isfinite` for every case; it is a performance sweep, not a full 576-case tolerance check.
- Pure Triton differs from flash ACLNN in fp32 `dg` reference behavior (strict matched_ratio ≈ 0.19, max_abs ≈ 4.5e-3 with 0 elements outside the hard bound), so pure Triton is used only as a performance baseline.

## Source fingerprints

| File | Size | SHA256 (first 16) |
|---|---:|---|
| `src/GatedDeltaRuleTraining_triton_optimized.py` | 211,388 B | `5936a228988c39a9` |
| `src/GatedDeltaRuleTraining_torch.py` | 5,167 B | `947e0f504796b624` |
| `src/GatedDeltaRuleTraining.json` | 6,499 B | `8308f7c9f6bd49ca` |
| `cases/full/GatedDeltaRuleTraining.json` | 1,257,720 B | `2acfc24ac6d99b04` |

Optimized-source md5: `f77755ba8194116b206518946bd7cab6`.

## License and attribution

`LICENSE` is a **placeholder**. This code is derived from / compatible with upstream
projects including flash-linear-attention (FLA), VeOmni, MindSpeed-MM / `fla_npu`, and
Triton-Ascend. Retain upstream copyright headers and confirm the license terms before
redistribution. See `THIRD_PARTY_NOTICES.md`.

## Notes

- Absolute latency depends on session, device load, and CANN/driver versions; compare ratios, not absolute milliseconds across sessions.
- The 576-case results were produced with a profiler-free `torch.npu.Event` full-call harness; the 3-case official results use `triton-op-verifier` `benchmark.py`.
