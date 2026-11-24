# GDN Triton 优化报告（dqkwg 版）

## 1. 版本与结果

| 项 | 值 |
|---|---|
| 优化版本 | dqkwg（本轮 best） |
| 核心源码 | `src/GatedDeltaRuleTraining_triton_optimized.py` |
| 源码 md5 | `f77755ba8194116b206518946bd7cab6` |
| 源码 sha256 | `5936a228988c39a949f62e9d30ebe64c6a3cad11a8ab733461d96603e5d71e39` |
| 3-case 官方 vs ACLNN | **1.1379x**（framework 23.9456 ms / impl 21.0431 ms） |
| 3-case 官方 vs 纯 Triton | **2.3151x**（framework 48.7380 ms / impl 21.0521 ms） |
| 3-case 逐 case vs ACLNN | 1.1600 / 1.1133 / 1.1409 |
| 3-case 精度 | **3/3 PASS**（eco_open_source） |
| 3-case 复测 vs ACLNN | 1.1214x（impl 21.2275 ms / framework 23.8036 ms）；同代码仅注释整理，差异在运行噪声内 |
| 576-case 全量 vs flash ACLNN | **1.1334x**（24.29 ms vs 27.53 ms 中位延迟） |
| 576-case 全量 vs 纯 Triton | **2.2809x** |

## 2. 优化目标

本版集中优化 `chunk_bwd_kernel_dqkwg_postgate_ds_reuse`。该核在此前版本中 3-case 聚合约 **5096.3 us**，是单核耗时最大的热点之一。目标是减少该核内重复 load、提前发射可预取的输入，并去掉不必要的显式 fence，同时保持算术、舍入边界、输出和 ABI 完全不变。

## 3. 优化点（5 项）

| # | 优化点 | 具体做法 | 机制 |
|---|---|---|---|
| 1 | K1 复用 K0 的 `do` tile | K1 直接使用 K0 已载入的 `b_do_k0`，不再重载 | 真删除 load，约 −16 KB/head |
| 2 | DG-MM5 复用已载入的 `q`/`k` | DG-MM5 第一项使用当前已载入的 `b_q`/`b_k`，不再重载 panel 0 | 真删除 load，约 −16 KB/head |
| 3 | K0 的 `k`/`g` load 上提 | 把 K0 的 `k` panel 和一次 `g` load 提到产生 V 循环内，与 `v`/`do`/`h`/`dh`/`dv` 同批发射 | 改善 MTE2 与 Cube 重叠，不增加字节 |
| 4 | K1 saved-H 预取 | 在 K0 尾部预取 K1 的 saved-H panel（`b_h_k1_prefetched`），K1 不再重建 `p_h` / 重载 | 把 K1 的 load 压到 K0 dot 链下，不增加字节 |
| 5 | `dv` 前移 + DW 相位 fence 消除 | K0/K1 的 `dv` load 移到状态 dot 之前；两处 DW 相位 `tl.debug_barrier()` 用 dtype 谓词 constexpr 消除 | 少 2 个显式 fence，改善发射顺序 |

**保持不变**：`i32` mask 原样（未改 fp32）、BK=64 / BV=128 / BT=64、kernel 未分裂、未引入 gdiff workspace、autotune 配置未动、`dg` 的 NK=2 panel 存储 + `_sum_first_axis` + transpose 归约路径未动。无算术、无舍入边界、无输出、无 ABI 改动。

## 4. 效果

### 4.1 单核

| 口径 | 优化前 | 优化后 | Δ |
|---|---:|---:|---:|
| `chunk_bwd_kernel_dqkwg_postgate_ds_reuse` 3-case 聚合 | 5096.3 us | **4514.0 us** | **−582.3 us（−11.4%）** |
| case1 单核 | 5614.8 us | 4980.3 us | −634.5 us |

### 4.2 整机

| 口径 | 结果 |
|---|---|
| round-local（warmup 5 / repeats 50，3 case） | **1.1290x**，impl 21.0676 ms |
| 官方 verifier recheck | **1.1379x**，impl 21.0431 ms，3/3 |
| 复测（同一份代码） | 1.1238x（impl 21.2179 ms，核 4564.7 us），在 ±0.9% 运行带内 |
| 本轮 sanitized 源码复测 | 1.1214x（impl 21.2275 ms，3/3） |
| 全量 576 case（Event，warmup 5 / repeats 10） | **24.29 ms 中位**，相对 flash **1.1334x**，相对纯 Triton **2.2809x** |

全量 576 case 中，dqkwg 版 **576/576** 个 case 超过 flash 1.05x，**564/576** 超过 1.10x，52/576 超过 1.15x；逐 case 加速比 min / 中位 / max = 1.0842x / 1.1281x / 1.1760x。

## 5. Phase 4 尝试

| 尝试 | 结果 | 处理 |
|---|---|---|
| K1 的 `q`/`k` panel 只载一次（循环不变量外提） | 1.1171x；dqkwg 4514.0 → 4754.2 us（+5.3%） | 回退：live range 拉长，负收益 |
| K1 复用 K0 的 `v`/`dv` tile | 未执行 | 按同一 live-range 机制否决，记录为后续 Pareto 候选 |

## 6. 结论

- dqkwg 核的重复 load / fence 缺口已关闭：单核 5096.3 → 4514.0 us（−11.4%），整机 3-case 官方 1.1379x，全量 576-case 1.1334x vs flash。
- 该核继续通过“再删 load 但延长 live range”的方式收益为负；剩余的 4514 us 主要受 Cube↔Vector 握手和 UB 常驻结构限制，Triton 层不再有明显空间。
- 后续轮次可转向其余大单项：wy pair 约 4712 us、dhu 约 3812 us、fwd_h 约 2033 us；`fwd_o` 的 score consumer/producer 仍有约 402 us 的结构差。
