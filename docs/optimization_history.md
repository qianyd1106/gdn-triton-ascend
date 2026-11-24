# GatedDeltaRuleTraining Triton 优化全历程与结论（2026-11-24）

## 0. 口径与最终结果

- 参考实现：veomni npu_varlen ACLNN 混合版 `flash_chunk_gated_delta_rule.py`（SHA `481838f8…`）；
  纯 Triton 参考：`chunk_gated_delta_rule_mm.py`；对照版本：g087 版（纯 Triton）。
- 两个性能轴：
  - **官方 verifier vs ACLNN**（fresh-cache，framework = vendor 实现本身）；
  - **官方 verifier vs 纯 Triton**（归一化到 `chunk_gated_delta_rule_mm` 基线）。
  - 轮内决策另用 `benchmark.py --warmup 5 --repeats 50` 的 3-case 几何平均（同 ACLNN 轴），跨轮噪声约 ±0.9%。
- 最终 best：**fwdoport**，md5 `b6aef21d6fe626941df06b714b0bd3e9`，3/3 精度通过。

| 指标 | 最终结果 | 上一版 dqkwgport | 更早 wyadjoint | postgate（首个官方口径） |
|---|---:|---:|---:|---:|
| 官方 vs ACLNN | **1.1541x**（impl 20.72 ms / fw 23.91 ms） | 1.1379x | 1.1006x | 1.0652x（postgate） |
| 官方 vs 纯 Triton | **2.3396x**（fw 48.72 ms） | 2.3151x | 2.2522x | 2.1789x |
| 逐 case vs ACLNN | 1.1795 / 1.1319 / 1.1514 | — | — | — |
| g087 版对照 | 纯 Triton 轴快 ~5.5%（2.3396 vs 2.2166） | ~4.4% | ~1.6% | — |

所有数字为 near-exact proxy（CANN 9.0.0 vs trace 9.0.1，共享 host）。

## 1. 优化轨迹总览

| 阶段 | 轮次/工作目录 | 方向 | 结果（speedup） | 判定 |
|---|---|---|---:|---|
| 探索 | naive-pilot01 | 初始 LLM 探索 | 未产出有效 round | 放弃 |
| 探索 | naive-pilot02 r1 | 初始 LLM 探索 | 0.8216 | 被 MCTS 取代 |
| MCTS | mcts r1 | chunked WY 状态扫描（3 阶段 WY 算法） | 0.8306 | 采纳为基线 |
| MCTS | **mcts r2** | solve_tril Neumann 截断 + fwd_h 单状态 tile v128 | **0.9037** | **采纳（当前分支）** |
| MCTS | mcts r3 | dw 拆分 + dqkwg BK=128 分支 | 0.8792 | 编译期 UB 溢出，淘汰 |
| MCTS | mcts r4 | dqkwg 三分裂 + dg 行项外移 | 0.8792 | 实测 +3.0~3.6%，回退 |
| 插件（独立路线） | triton-optimizer r1 | solve_tril 行提取去 GM 重载 | 内核 −2.5%，整机 +0.2% | 该路线内采纳，**未并入最终代码** |
| 插件（独立路线） | triton-optimizer r2 | fwd_h 去边界分支（整块循环+尾块） | fwd_h −1.1~−5.7%，整机 +0.7% | 该路线内采纳，**未并入最终代码** |
| 插件（独立路线） | triton-optimizer r3 | dhu full-K（9→5 MMA/chunk） | dhu +6.4~11.1% | 淘汰（sync_block 8-9→15-30） |
| 插件（独立路线） | triton-optimizer r4 | dqkwg DG mm5 折叠为 full-K dot | 整机 +0.2% | 该路线内采纳（边际），**未并入最终代码** |
| 插件（独立路线） | triton-optimizer r5 | wy stage1+2 融合 | 相对父轮 +1.1~1.2% | 淘汰（dA 重物化） |
| AscendC | ascendc_evo | chunk_fwd_o AscendC 进化（2 变体） | 1.0031x（目标 1.2x） | 未达标 |
| naive | naive-fwdO | fwd_o 单核 lean 改写 | 0.9024 < 0.9037 | 回退（带宽线性，无空间） |
| naive | naive-fuseO | fwd_o 融入 fwd_h 扫描 | 0.8927 < 0.9037 | 方向证伪，回退 |
| naive | naive-maskgdiff | i32 mask→fp32；gdiff workspace+BK128 | 0.9094 vs 0.9104（噪声内）/ 编译失败 | 均未采纳 |
| naive | **naive-postgate** | 序列排序 + postgate-DS 复用 + 门控端点向量窗口 + 行向量归约 | **1.0755**（+19.0%） | **采纳** |
| naive | naive-wyscalar | wy/fwd_o/recompute 标量下降 | 1.0665（唯一采纳项 +0.03%，噪声内） | 未晋级（< postgate 1.0755） |
| naive | **naive-wyadjoint** | 两阶段 WY adjoint | **1.1000**（官方 1.1006 / 2.2522） | **采纳** |
| naive | **naive-dqkwgport** | dqkwg 5 项微优化 | **1.1290**（官方 1.1379 / 2.3151） | **采纳** |
| naive | **naive-fwdoport** | fwd_o 两核拆分 | **1.1508**（官方 1.1541 / 2.3396） | **采纳（最终 best）** |
| naive | naive-dhusplit | dhu 走非融合 dv_local + 轻扫描 | 1.0963（拆分失败后回退） | **证伪**，代码与父轮逐字节相同 |
| naive | naive-slottune | 持久核槽位 24→48/96 | 全部手臂回退 | **证伪**（MIX 核 24 槽已占满 48 AIV） |

## 2. 采纳的优化点（最终代码中包含）

### 2.1 算法/结构级

| # | 优化点 | 引入 | 实测效果 | 机制 |
|---|---|---|---|---|
| 1 | `solve_tril` 62 步串行前代 → 3-dot Neumann 截断 `(I+L)(I+L²)+L⁴` | mcts r2 | 4.374 → 1.588 ms | L 幂零；必须 fp32 `input_precision="ieee"`，bf16 操作数使 `dg` matched_ratio 掉到 0.979 |
| 2 | `fwd_h` 两个 64 行 K 半区 → 单个 `[K,V]` 状态 tile（v128 特化） | mcts r2 | 4.711 → 4.206 ms | 每 chunk dot 4→2、store 2→1、load 4→2 |
| 3 | fwd_h + dhu 按 chunk 数降序的序列排序（metadata-only） | postgate iter_0 | 整机 +1.9%（0.9037→0.9207） | 长序列先启动，缓解尾部不均衡 |
| 4 | **两阶段 WY adjoint**（pair-interleaved stage1 + batch2） | wyadjoint | 1.0755 → 1.1000 | 每 program 两条独立 2×2 二阶 adjoint dot 链交错，掩盖 Cube WAIT_FLAG_DEVI |
| 5 | **fwd_o 两核拆分**（producer: q@kᵀ+门控+mask→bf16 score；consumer: q@h+score→o） | fwdoport | fwd_o 1756.4 → 1356.4 us（−22.8%） | 流量 +33%，但 UB 活跃集小到可双缓冲，有效带宽 424→~740 GB/s |

### 2.2 单核微结构（postgate 轮）

| # | 优化点 | 实测效果 |
|---|---|---|
| 6 | dqkwg postgate-DS 复用（K0/K1） | +0.4%（0.9207→0.9239） |
| 7 | fwd_h gate 端点 32 宽向量窗口（替代逐 chunk 0-D load+exp） | +4.9%（0.9239→0.9695） |
| 8 | dhu gate 端点 32 宽向量窗口 | +1.2%（0.9695→0.9812） |
| 9 | dqkwg 两处 2D→0D 归约改行向量归约 | +9.6%（0.9812→1.0755） |

### 2.3 单核微细节优化（dqkwg / fwd_o）

| # | 优化点 | 实测效果 |
|---|---|---|
| 10 | dqkwg：K1 复用 K0 do tile、DG-MM5 复用 q/k、K0 load 重排、K1 saved-H 预取、DW 相位 barrier 消除 | dqkwg 5096 → 4514 us（−11.4%），追平 g087 4508 us |
| 11 | fwd_o 两新核 program 槽位 24→48（num_vectorcore） | fwd_o 1366.4 → 1356.4 us；整机 +0.74% |

> 独立 triton-optimizer 路线在其自身基线上采纳了 3 项（solve_tril 行提取、fwd_h 去边界分支、dqkwg mm5 折叠），
> 但该路线未并入最终 cannbot 代码；其失败轮（full-K dhu、wy stage1+2 融合）的机制结论仍被后续轮沿用。



## 3. 失败/放弃清单与原因

### A. UB 活跃区间/容量（最常见失败机制）

| 尝试 | 轮次 | 实测 | 结论 |
|---|---|---|---|
| wy_bwd `k@kᵀ` 前移（16→14 dot/head） | mcts r2 | 5.971 → 6.104 ms | 淘汰：常驻 [BT,BT] fp32 累加器抬高 UB 活跃区间 |
| dhu `[K,V]` 状态合并（9→5 dot） | mcts r2 | 5.121 → 5.403 ms | 淘汰：64 KB 状态 + 每步 bf16 往返副本 |
| dhu full-K（9→5 MMA/chunk） | triton-opt r3 | dhu +6.4~11.1% | 淘汰：sync_block 对 8-9→15-30 |
| dqkwg BK=128（14→7 dot） | mcts r2/r3、maskgdiff | UB 224~254 KB > 192 KB | 不可行 |
| gdiff workspace + BK=128 | maskgdiff | UB 240.5 KB > 192 KB，6 个 autotune 配置全失败 | 不可行 |
| dqkwg K1 q/k panel 外提 | dqkwgport opt_iter_1 | dqkwg +5.3% | 回退（删除 16 KB/head 流量抵不过活跃区间变宽） |
| dqkwg K1 复用 v/dv tile | dqkwgport opt_iter_2 | 未执行 | 同机制且压力更大，按机制否决 |
| i32 mask→fp32 比较（12 处） | maskgdiff A | 整机 −0.11%（噪声）；dqkwg +0.8% | 未采纳：dqkwg 的 [64,64] fp32 mask 常驻抬 UB |
| dhu 拆分 BV=64 档 | dhusplit | dhu 10829 us（+2.85x） | 灾难性回退 |

### B. 序列化 / 同步代价

| 尝试 | 轮次 | 实测 | 结论 |
|---|---|---|---|
| wy stage1+2 融合 | triton-opt r5 | 相对父轮 +1.1~1.2% | 淘汰：dA 被编译器重物化到 GM 并重载两次，串行在 `sync_block_wait[CUBE,MTE3,PIPE_S]` 后 |
| fwd_o 融入 fwd_h 扫描 | fuseO | 融合核 6.67 ms；整机 0.8927 < 0.9037 | 方向证伪：搬到扫描里没有更快 |
| dqkwg 三分裂（dw/dg 独立 kernel） | mcts r4 | +3.0~3.6% | 回退：dg 行项搬出后失去 Cube/Vector 重叠（+1.07 ms）+ workspace 流量（~0.4 ms） |

### C. 带宽线性 / 无同步浪费

| 尝试 | 轮次 | 实测 | 结论 |
|---|---|---|---|
| fwd_o 单核 lean 改写 + 全部编译旋钮 | fwdO | 候选 0.9024 < 0.9037；配置旋钮 ±0.3% | 回退：`no_h` 消融证明带宽线性，与 ACLNN 的 2x 差距来自 scheduler/AIC-AIV 分离/每核 workspace |

### D. 实测证伪（含机制）

| 尝试 | 轮次 | 实测 | 结论 |
|---|---|---|---|
| dhu 拆分为并行 dv_local + 轻扫描 | dhusplit | 拆分 dhu 3802.2 vs 融合 3804.9 us；dv_local 另付 1057.4 us；整机 1.0963 | 证伪：串行 scan 中 A/A@do dot 被 DMA 完全掩盖（移除后 −0.07%），dhu 瓶颈是 DMA+串行状态链（~241 GB/s）；拆分还使 UB 209.6 KB > 192 KB 失去双缓冲 |
| 持久核槽位 24→48/96 | slottune | wy −0.05%/+0.31%；recompute+fwd_o 整机 1.1532→1.1391（回退） | 证伪：MIX 核 `Block Num=24 / Mix Block Num=48`，24 AIC 已派生 48 AIV（Cube:Vector=1:2），加槽只多 wave |
| wy/fwd_o/recompute 标量下降 | wyscalar | 3 次迭代：−0.29% / −0.06% / +0.03%；simulator 证明 stage1 Cube FLOWCTRL 37.1% | 未晋级（< postgate 1.0755）：标量不是瓶颈 |
| fwd_o RUN_CHUNKS 2→4 | fwdoport opt_iter_2 | core dump（exit 139），最小 shape 可复现 | 回退 |

### E. 工具/路线级

- **AscendC 进化路线**（chunk_fwd_o，2 变体）：最好 998.04 us vs 基线 1001.09 us = **1.0031x**，未达 1.2x 目标；AIV 侧零开销上限 ~1.02x（AIC/workspace 受限）；激进 epilogue 事件合并会导致 pong 槽 q@H 恒 0（确定性错误），已回退。
- **pilot01**：未产出有效 round；**pilot02**：0.8216，被 MCTS 分支取代。

## 4. 可复用机制结论

1. **串行扫描核（dhu）里，dot 是"免费"的**：移除每 chunk 的 A/A@do 三个 dot 后内核时间不变（−0.07%）；瓶颈是 DMA + 串行状态链（实测 ~241 GB/s）。扫描类核不应先做 dot 预算。
2. **只有缩小 UB 活跃集的拆分才是正收益**：fwd_o 拆分流量 +33% 但换来双缓冲（424→~740 GB/s）；dhu 拆分把 16 KB b_A 换成 32 KB dv 双缓冲副本，UB 209.6 KB > 192 KB，适得其反。
3. **在该 Triton-Ascend lowering 上，per-body 跨引擎握手数（`sync_block_wait/set`）比 MMA/load 数更决定性能**：full-K dhu（mma 9→5，sync 8-9→15-30）与 wy 融合（所有 IR 门禁通过但 dA 重物化）都因此回退；dqkwg mm5 折叠只换来 0.2%。
4. **MIX 核 24 槽 = 48 AIV 核**（Cube:Vector=1:2），槽位不是可调杠杆；fwd_o 的 24→48 属个例。
5. **带宽线性核没有可回收的同步/latency 浪费**：fwd_o 单核与 ACLNN 的 2x 差距是 vendor 调度器/引擎分离/每核 workspace，不是 Triton 层可补。
6. **精度边界**：solve_tril 必须 fp32 ieee dot；bf16 操作数让 `dg` matched_ratio 0.979 < 0.99。其余改动保持操作数集合与舍入边界不变。
7. **跨轮噪声 ±0.9%**：低于该带的收益不晋级（wyscalar +0.03%、maskgdiff −0.11%）。

## 5. 最终状态与剩余空间

最终 best（fwdoport）逐核 vs ACLNN（官方 recheck，3-case 聚合）：

| 链路 | 我们 | ACLNN 对应 | 差 | 判断 |
|---|---:|---:|---:|---|
| dhu（scan+dv local） | 3796.7 us | 1822.5 + 629.1 | +1345 | 已证伪（拆分无效），vendor 结构差 |
| dqkwg | 4521.5 us | 3728.6（+Add 362.5?） | +430~793 | 与 g087 已持平；需新分解思路，历史尝试均负 |
| wy stage1+2+recompute | 5603.2 us | 5086.9 | +516 | adjoint 已做；融合已证伪 |
| fwd_o | 1355.3 us | 941.0 | +414 | 已与 g087 持平，需赢过 g087 结构 |
| fwd_h | 2041.9 us | 2271.3 | −229 | 我们更快 |
| solve_tril | 1651.6 us | 3806.9 | −2155 | 我们更快（Neumann） |
| transpose/zeros/cast 等 | ~213 us | ~3496 | −3283 | 我们更快 |

结论：相对同族 Triton（g087）已全面持平并整体快 ~5.5%，相对 ACLNN 的剩余差距属于 vendor 融合核/调度器级结构差，Triton 层可用的结构、数据流、存储、配置手段已穷尽；这套架构本地已到实际上限。

## 6. 提交材料

- 最终包：`results/GDR_fwdoport_best_package/`
  （源码 md5 `b6aef21d…`、`OPTIMIZATION_REPORT.md`、`report.md`、官方 verify/perf JSON、SHA256SUMS）
- 压缩包：`results/GDR_fwdoport_best_20261124.tar.gz`
  （sha256 `d4330429e083d3ddf91e56e2dd94228d81e5757f904b58a2b38fd6bc004f2dac`）
- 稳定副本：`results/final_candidate_fwdoport/`
- 历史包：`results/GDR_dqkwgport_best_20261124.tar.gz`、`results/GDR_wyadjoint_best_20261123.tar.gz`
- 各轮完整报告：`work/cannbot_project/triton_ascend_output/GatedDeltaRuleTraining-naive-*/opt-round-1/report.md`
- 官方验证：`GDR_fwdoport_best_package/results/verify_result_aclnn.json`（3/3）、
  `GDR_fwdoport_best_package/results/perf_vs_aclnn.json`、`GDR_fwdoport_best_package/results/perf_vs_puretriton.json`
