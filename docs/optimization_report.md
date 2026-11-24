# GDN Triton 优化报告（fwdoport 版）

## 1. 版本与结果

| 项 | 值 |
|---|---|
| 优化版本 | fwdoport（本轮 best） |
| 核心源码 | `src/GatedDeltaRuleTraining_triton_optimized.py` |
| 源码 md5 | `9f5754c2d637743682def2e1b44ed948` |
| 源码 sha256 | `0c2e45bae06c1e084546bf242e3107243aee15477d280b515981dd063660f339` |
| 3-case 官方 vs ACLNN | **1.1541x**（framework 23.9131 ms / impl 20.7200 ms） |
| 3-case 官方 vs 纯 Triton | **2.3396x**（framework 48.7212 ms / impl 20.8249 ms） |
| 3-case 逐 case vs ACLNN | 1.1795 / 1.1319 / 1.1514 |
| 3-case 精度 | **3/3 PASS**（eco_open_source） |
| 3-case sanitized 源码复测 | **1.1489x**（impl 20.7693 ms / framework 23.8617 ms），3/3 |
| 576-case 全量 vs flash ACLNN | **1.1595x**（23.63 ms vs 27.40 ms 中位延迟） |
| 576-case 全量 vs 纯 Triton | **2.3438x** |

## 2. 优化目标

本版集中优化前向输出核 `chunk_fwd_kernel_o_tile_owned`。该核在此前版本中 3-case 聚合约 **1755.5 us**，是当时最大的单核缺口之一。目标是关闭该核的重复读取和同步开销，同时保持公共 ABI、六个输出、bf16/fp32 舍入边界和 576-case 语义不变。

## 3. 优化点

| # | 优化点 | 具体做法 | 机制 |
|---|---|---|---|
| 1 | 新增 `chunk_fwd_kernel_o_gated_score_producer` | 只算 `b_A = q@kᵀ`（NK=1）；在 fp32 中做 `exp(g_i−g_j)` 门控和 causal mask；整块 64×64 无掩码一次性 cast 成 bf16 `score` workspace | 把门控/掩码与输出计算解耦，producer 的 UB 活跃集更小 |
| 2 | 新增 `chunk_fwd_kernel_o_score_consumer` | `b_o = q@h`（NK=1）→ `*exp(g)` → 读 `score` → `b_o*scale + dot(A,b_v)*scale` → 存 `o`；`RUN_CHUNKS=2` 把上下文重载摊到相邻两个 chunk | 两个核都可双缓冲，有效带宽从 424 提升到 ~740 GB/s |
| 3 | 新增 metadata-only 路由 `_route_chunk_fwd_o_gated_score` | 按 workspace 布局、dtype/device/contiguity/shape、`NT*H ≥ 2*grid_x` 选择新路径；`score` 分配失败时回落原 `chunk_fwd_kernel_o_tile_owned`（原核保留、字节未动） | 纯 metadata 谓词，无 case id / 无精确 T / 无 seed / 无张量值 |
| 4 | 两个新核 program 槽位 24→48 | `min(tiles, num_aicore)=24` → `min(2*grid_x, NT*H)=48`（num_vectorcore=48） | 匹配 MIX 核的 24 AIC / 48 AIV 结构 |

**数值边界**：`b_A` 仍是 fp32 累加 → fp32 门控 → fp32 mask → 一次性 cast bf16；`b_o`、`exp(g)`、两次 scale 乘法的顺序与次数不变。拆分后的 `o` 与基线在同一条 bf16 舍入边界上可比。

## 4. 效果

### 4.1 单核

| 口径 | 拆分前 | 拆分后 | Δ |
|---|---:|---:|---:|
| fwd_o 合计（3-case 均值） | 1756.4 us | **1356.4 us** | **−400.0 us（−22.8%）** |
| `chunk_fwd_kernel_o_gated_score_producer` | — | 459.0 us | — |
| `chunk_fwd_kernel_o_score_consumer` | — | 897.4 us | — |

逐 case fwd_o（producer + consumer）：1498.4 / 1069.3 / 1501.5 us；对比拆分前 fused 单核 1947.5 / 1363.4 / 1958.4 us，分别 **−23.1% / −21.6% / −23.3%**。

### 4.2 整机

| 口径 | 结果 |
|---|---|
| round-local（warmup 5 / repeats 50，3 case） | **1.1508x**，impl 20.6801 ms |
| 同一代码重测 | **1.1688x**，impl 20.6454 ms |
| 官方 verifier recheck | **1.1541x**，impl 20.7200 ms，3/3 |
| sanitized 源码复测 | 1.1489x，impl 20.7693 ms，3/3 |
| 全量 576 case（Event，warmup 5 / repeats 10） | **23.63 ms 中位**，相对 flash **1.1595x**，相对纯 Triton **2.3438x** |

全量 576 case 中，fwdoport 版 **576/576** 个 case 超过 flash 1.05x，**575/576** 超过 1.10x，**334/576** 超过 1.15x；逐 case 加速比 min / 中位 / max = 1.0986x / 1.1528x / 1.2000x。

## 5. Phase 4 尝试

| 尝试 | 结果 | 处理 |
|---|---|---|
| 两个新核 program 槽位 24→48 | 1.1508；fwd_o 1366.4→1356.4 us；三 case 均无退化 | **采纳** |
| `RUN_CHUNKS 2→4` | core dump（exit 139），最小 shape 可复现 | **回退** |

## 6. 结论

- fwd_o 的重复读取/同步缺口已关闭：单核 1756.4 → 1356.4 us（−22.8%），整机 3-case 官方 1.1541x，全量 576-case 1.1595x vs flash。
- 流量虽然 +33%，但 UB 活跃集缩小到可双缓冲，带宽提升抵消了额外流量；继续增大 `RUN_CHUNKS` 会 core dump，不再尝试。
- 相对 flash ACLNN 的剩余差距主要集中在 dhu（+1.35 ms）、dqkwg（+0.4~0.8 ms）、wy（+0.5 ms）和 fwd_o（+0.4 ms），属于 vendor 融合核/调度器级结构差；完整历史与失败机制见 `optimization_history.md`。
