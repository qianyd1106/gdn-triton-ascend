# fwdoport 优化尝试记录

指标为 `benchmark.py --warmup 5 --repeats 50` 的 3-case 几何平均加速比，基线为 ACLNN framework；单核列为 fwd_o（producer + consumer 合计）的 3-case 均值。

## 1. 采纳 / 回退

| # | 阶段 | 改动 | verify | speedup | fwd_o us | 决策 |
|---|---|---|---|---|---|---|
| 0 | Phase 3 | fwd_o 两核拆分：producer 算 `q@kᵀ + 门控 + causal mask → bf16 score`；consumer 算 `q@h + score → o`；`RUN_CHUNKS=2`；metadata-only 路由 + 原 fused 核 fallback | 3/3 | **1.1508**（重测 1.1688） | **1356.4** | 采纳 |
| 1 | Phase 4 opt_iter_1 | 两个新核 program 槽位 `min(tiles, num_aicore)=24` → `min(2*grid_x, NT*H)=48`（num_vectorcore） | 3/3 | 1.1508 | 1356.4 | 采纳 |
| 2 | Phase 4 opt_iter_2 | `RUN_CHUNKS 2→4` | — | — | — | 回退：最小 shape 可复现 core dump（exit 139） |

## 2. 单核证据

| 口径 | 拆分前 | 拆分后 | Δ |
|---|---:|---:|---:|
| fwd_o 合计（3-case 均值） | 1756.4 us | **1356.4 us** | **−400.0 us（−22.8%）** |
| `chunk_fwd_kernel_o_gated_score_producer` | — | 459.0 us | — |
| `chunk_fwd_kernel_o_score_consumer` | — | 897.4 us | — |

逐 case fwd_o（producer + consumer）：1498.4 / 1069.3 / 1501.5 us；对比拆分前 fused 单核 1947.5 / 1363.4 / 1958.4 us，分别 −23.1% / −21.6% / −23.3%。

## 3. 结论

- fwd_o 从 1756.4 us 降到 1356.4 us（−22.8%），整机 3-case 官方 1.1541x，全量 576-case 1.159x vs flash。
- 流量虽然 +33%，但每个新核的 UB 活跃集小到可双缓冲，有效带宽从 424 提升到 ~740 GB/s，净收益 −400 us。
- `RUN_CHUNKS` 继续增大（2→4）会 core dump，不再尝试。
