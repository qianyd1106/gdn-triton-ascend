# gdn-triton-ascend

Triton-Ascend implementation and performance optimization of **Gated Delta Rule (GDN)** training kernels on **Ascend 910B (910B2C)**.

The repository contains:

- a self-contained Triton-Ascend implementation of `GatedDeltaRuleTraining` (forward + backward, varlen, bf16/fp32);
- optimization notes for the WY-adjoint, dqkwg, and fwd_o rounds, plus the full optimization history;
- frozen cases and raw benchmark results for a four-way comparison.

## Results

### 3 representative cases (official verifier, warmup 5 / repeats 50)

| Metric | Value |
|---|---:|
| Precision | **3/3 PASS** |
| Speedup vs flash ACLNN | **1.1541x** |
| Speedup vs pure Triton | **2.3396x** |
| Implementation latency | **20.72 ms** |
| Flash ACLNN reference | 23.91 ms |
| Pure Triton reference | 48.72 ms |
| Sanitized-source recheck | 1.1489x, impl 20.77 ms, 3/3 |
| Full 576-case precision | **576/576 PASS** (golden = flash ACLNN, 6 outputs per case) |

### Full 576 cases (Event, warmup 5 / repeats 10)

| Implementation | Median latency | vs flash ACLNN | vs pure Triton |
|---|---:|---:|---:|
| flash ACLNN | 27.28 ms | 1.000x | 2.030x |
| pure Triton | 55.39 ms | 0.495x | 1.000x |
| g087 版 | 25.11 ms | 1.088x | 2.206x |
| **fwdoport 版 (this repo)** | **23.63 ms** | **1.159x** | **2.344x** |

- The fwdoport implementation is >1.05x vs flash ACLNN on **576/576** cases, >1.10x on **575/576**, and >1.15x on **334/576**; minimum 1.0986x.
- It is **6.3%** faster than the g087 version (25.11 → 23.63 ms).

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
│   ├── optimization_report.md                       # fwdoport optimization notes
│   ├── optimization_attempts.md                     # attempts / rejected directions
│   ├── optimization_history.md                      # full optimization history
│   ├── fwdoport_round_report.md                     # detailed fwd_o round report
│   └── performance_comparison.md                    # 576-case comparison report
├── results/
│   ├── verify_result_aclnn.json                     # 3-case precision (3/3)
│   ├── perf_vs_aclnn.json                           # 3-case vs ACLNN (1.1541x)
│   ├── perf_vs_puretriton.json                      # 3-case vs pure Triton (2.3396x)
│   ├── recheck_verify.json                          # sanitized-source recheck (3/3)
│   ├── recheck_perf_vs_aclnn.json                   # sanitized-source recheck (1.1489x)
│   ├── verify_result_full_576.json                  # full 576-case precision (576/576)
│   ├── full_flash.json                              # 576-case flash ACLNN
│   ├── full_pure.json                               # 576-case pure Triton
│   ├── full_g087.json                               # 576-case g087 version
│   └── full_local_fwdoport.json                     # 576-case fwdoport version
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

## Optimization points (fwd_o two-kernel split)

1. New `chunk_fwd_kernel_o_gated_score_producer`: `q@kᵀ` → fp32 gate `exp(g_i−g_j)` → fp32 causal mask → whole 64×64 unmasked cast to bf16 `score` workspace.
2. New `chunk_fwd_kernel_o_score_consumer`: `q@h` → `*exp(g)` → read `score` → `b_o*scale + dot(A,b_v)*scale` → store `o`; `RUN_CHUNKS=2` amortizes context reload over two adjacent chunks.
3. New metadata-only route `_route_chunk_fwd_o_gated_score`; falls back to the original `chunk_fwd_kernel_o_tile_owned` on score allocation failure (original kernel kept byte-identical).
4. Program slots for the two new kernels 24→48 (`num_vectorcore`).

Single-kernel fwd_o: **1756.4 → 1356.4 us (−22.8%)**.

## Reports

- [docs/optimization_report.md](docs/optimization_report.md) — fwdoport optimization points and end-to-end results.
- [docs/optimization_attempts.md](docs/optimization_attempts.md) — adopted/reverted attempts.
- [docs/optimization_history.md](docs/optimization_history.md) — full optimization history and failure mechanisms.
- [docs/fwdoport_round_report.md](docs/fwdoport_round_report.md) — detailed fwd_o round evidence.
- [docs/performance_comparison.md](docs/performance_comparison.md) — full 576-case comparison and per-case distribution.

## Precision

- Strict precision uses the eco open-source standard, golden = flash ACLNN. The optimized version passes 3/3 representative cases.
- **Full 576-case precision: 576/576 PASS, 0 failures** (golden = flash ACLNN, 6 outputs per case; ~94 min). The 576-case performance sweep also checks `isfinite`.
- Pure Triton differs from flash ACLNN in fp32 `dg` reference behavior (strict matched_ratio ≈ 0.19, max_abs ≈ 4.5e-3 with 0 elements outside the hard bound), so pure Triton is used only as a performance baseline.

## Source fingerprints

| File | Size | SHA256 (first 16) |
|---|---:|---|
| `src/GatedDeltaRuleTraining_triton_optimized.py` | 220,254 B | `0c2e45bae06c1e08` |
| `src/GatedDeltaRuleTraining_torch.py` | 5,167 B | `947e0f504796b624` |
| `src/GatedDeltaRuleTraining.json` | 6,499 B | `8308f7c9f6bd49ca` |
| `cases/full/GatedDeltaRuleTraining.json` | 1,257,720 B | `2acfc24ac6d99b04` |

Optimized-source md5: `9f5754c2d637743682def2e1b44ed948`.

## License and attribution

`LICENSE` is a **placeholder**. This code is derived from / compatible with upstream
projects including flash-linear-attention (FLA), VeOmni, MindSpeed-MM / `fla_npu`, and
Triton-Ascend. Retain upstream copyright headers and confirm the license terms before
redistribution. See `THIRD_PARTY_NOTICES.md`.

## Notes

- Absolute latency depends on session, device load, and CANN/driver versions; compare ratios, not absolute milliseconds across sessions.
- The 576-case results were produced with a profiler-free `torch.npu.Event` full-call harness; the 3-case official results use `triton-op-verifier` `benchmark.py`.
