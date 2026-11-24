# GatedDeltaRuleTraining — Round 1 Report (naive / fwdoport)

## 基本信息

| 项 | 值 |
|---|---|
| arch | ascend910b（910B2C，num_aicore=24 / num_vectorcore=48，CANN 9.0.0，Triton-Ascend 3.2.0） |
| 工作目录 | `triton_ascend_output/GatedDeltaRuleTraining-naive-fwdoport/opt-round-1` |
| round_index | 1（algorithm = naive, run_tag = fwdoport） |
| baseline_dir | `triton_ascend_output/GatedDeltaRuleTraining-naive-dqkwgport/opt-round-1`（md5 `8b9df67f`） |
| global_baseline_dir | `triton_ascend_output/GatedDeltaRuleTraining-naive-fwdoport/global_baseline`（空目录，见下） |
| direction | `pattern: custom` |
| round_strategy / analysis_policy | structural_change / bottleneck_analysis |
| hypothesis | 从 dqkwgport 最优（1.1379x vs ACLNN / 2.3151x vs pure Triton，3/3，md5 `8b9df67f`）出发，关闭下一个单核缺口：本仓 `chunk_fwd_kernel_o_tile_owned` 1755.5 us vs 两核 fwd_o 参考（`score_consumer` 894.2 + `gated_score_producer` 459.0 = 1353.2 us），+402 us。引入两核 fwd_o 分解及其 launcher 调用路径，保持公共 ABI、六个输出、bf16/fp32 舍入边界与 576-case 语义 |
| mode | A（torch 标杆 + 3 个 varlen case；另有 GPU kernel 参考文件，但**无 `gpu_perf.csv`**） |

> `global_baseline/` 为空目录（Orchestrator 未落实现），因此加速比统一使用
> `benchmark.py` 的 `speedup_vs_torch`（framework = vendor ACLNN/veomni 参考实现本身）作为口径，
> 与父轮及全部历史轮次一致。

## 生成结果

| 项 | 值 |
|---|---|
| Phase 3 迭代 | 1 次（iter_0 首轮 verify 3/3，无需修复） |
| Phase 4 迭代 | 2 次（opt_iter_1 采纳；opt_iter_2 崩溃回退） |
| 最终版本来源 | `output/opt_iter_1/optimized_code.py`（md5 `b6aef21d`） |
| 最终代码 | `GatedDeltaRuleTraining_generated.py`（= 上述文件，35 个 @triton.jit kernel） |

## 目标加速比与实际

| 项 | 值 |
|---|---|
| target_speedup | 5.0 |
| target_reached | **false**（harness 的 1.0 口径是 vendor 框架实现本身，5x 结构性不可达；父轮与全部历史轮次同样如此） |
| 父 round best（同 harness） | 1.1290（impl 21.0676 ms） |
| **本轮 best_speedup（几何平均，主测）** | **1.1508**（impl **20.6801 ms**，framework 23.7980 ms） |
| 同一代码重测（warmup=5/repeats=50） | **1.1688**（impl 20.6454 ms） |
| 相对父 round 增益 | **+1.93%（主测）/ +3.52%（重测）** |
| 逐 case（主测 / 重测） | 1.1780 / 1.2336，1.1307 / 1.1303，1.1441 / 1.1450 —— **三个 case 全部 ≥ 1.13，无退化** |

### 单核证据（本轮真正的结论）

| 口径 | 父 round | 本轮 | Δ |
|---|---|---|---|
| `chunk_fwd_kernel_o_tile_owned`（fused，3-case 均值） | **1756.4 us** | — （已由 split 取代） | — |
| `chunk_fwd_kernel_o_score_consumer` | — | **897.4 us** | — |
| `chunk_fwd_kernel_o_gated_score_producer` | — | **459.0 us** | — |
| **fwd_o 合计** | **1756.4 us** | **1356.4 us** | **−400.0 us（−22.8%）** |

⇒ 假设被逐核证实：`+402 us` 的缺口被完整关闭，与两核参考合计 1353.2 us
相差 **+0.97%**（在 run-to-run 噪声内）；两次独立 benchmark 的 fwd_o 逐核合计都是 1356.4 us，
说明该收益不是运行波动。

**逐 case 单核（us）**：

| case | producer | consumer | 合计 |
|---|---|---|---|
| 1（T=16358） | 505.1 | 993.3 | 1498.4 |
| 2（T=11166） | 371.2 | 698.1 | 1069.3 |
| 3（T=16334） | 500.8 | 1000.7 | 1501.5 |

对比父轮同 case 的 fused 单核：1947.5 / 1363.4 / 1958.4 us ⇒ 三个 case 分别
**−23.1% / −21.6% / −23.3%**。

**关于 case1 整机口径的说明（重要，勿误读）**：Phase 3 代码的 case1 整机
23.6195 → 23.6706 ms（+51.1 us），但同 case1 的 fwd_o 单核是 −405.6 us。差额完全来自
**其他 kernel 的抖动**（recompute_w_u +134.3、wy_stage2 +102.8、dhu +35.5、fwd_h +32.3、
solve_tril +31.9 us；case2/case3 无同向抖动）。最终代码的两次测量 case1 分别为
1.1780 / 1.2336，均高于父轮 1.1579。

## Shape 通过率（以 verify 为准）

- 最终代码 `output/opt_iter_1/verify/verify_result_optimized.json`：
  passed_cases = **3** / total_cases = **3**，failures = **[]**
- Phase 3 基线 `output/iter_0/verify/verify_result.json`：passed_cases = **3** / total_cases = **3**
- 精度标准：`eco_open_source`（第一标准），bf16 输出 rtol=atol=1.562e-2 + `max(1.0, 32ULP)`；
  fp32 `dg` atol=1.526e-5 / rtol=9.766e-4 + `max(1e-2, 32ULP)`；finite 全覆盖
  （case1 67002368/67002368、case2 45735936/45735936、case3 66904064/66904064）
- 性能侧：nan/inf/zero/negative/none 索引全为空；`failure_class = None`；重试 3 次内全部通过

## GPU 参考性能

未找到 `gpu_perf.csv`（manifest `input_files.gpu_perf_csv = null`，且描述文件同级向上 3 级
无该文件），故**不输出 Ascend/GPU 延迟对比**。`gpu_kernel_ref.py` 仅作算法结构参考。

## 本轮改动（全部集中在 fwd_o 调用路径）

改动在 `chunk_fwd_o` 的 `workspace_native_layout` 分支内新增一条路由，其余 **33 个 kernel
与全部 host 代码字节不变**（确定性 patch：反向替换可逐字节还原基线，已断言）：

1. 新增 `chunk_fwd_kernel_o_gated_score_producer`：只算 `b_A = q@kᵀ`（NK=1），在 fp32 里做
   `exp(g_i−g_j)` 门控与 causal mask，然后**整块 64×64 无掩码**存成 bf16 `score` workspace。
2. 新增 `chunk_fwd_kernel_o_score_consumer`：`b_o = q@h`（NK=1）→ `*exp(g)` → 读 `score`
   → `b_o*scale + dot(A,b_v)*scale` → 存 `o`。`RUN_CHUNKS=2` 把上下文重载（bos/eos/
   offsets/int64 基址）摊到 2 个相邻 chunk 上。
3. 新增 `_route_chunk_fwd_o_gated_score`：**纯 metadata 谓词**（workspace_native_layout、
   `g_gamma is None`、BT/BK/BV/K/V、`NT*H ≥ 2*grid_x`、dtype/device/contiguity/shape），
   无 case id / 无精确 T / 无 seed / 无张量值；`score` 分配 OOM 时返回 False 落到
   `chunk_fwd_kernel_o_tile_owned`（原核保留为 fallback，字节未动）。
4. Phase 4 采纳项：这两个新核的 program 槽位由 `min(tiles, num_aicore)=24` 提到
   `min(2*grid_x, NT*H)=48`（num_vectorcore=48），其余不变。

**数值等价性**：`b_A` 仍是 fp32 累加 → fp32 门控 → fp32 mask → **一次性** cast 到 bf16
（原先是在寄存器里 `.to(bf16)` 后立刻喂 `dot`，现在改为存盘再读）；`b_o`、`exp(g)`、两次
`scale` 乘法的顺序与次数全部保持原样。因此 `o` 与基线在同一条 bf16 舍入边界上可比。

## 性能数据

| 项 | 值 |
|---|---|
| speedup_vs_torch（几何平均，主测） | **1.1508** |
| 总延时（impl avg / framework avg） | 20.6801 ms / 23.7980 ms |
| 逐 case 延时（impl ms） | 23.1850 / 16.3409 / 23.3441 |
| speedup_vs_baseline（相对本轮 Phase 3 代码） | 1.1508 / 1.1423 = **+0.74%** |
| 重测 | 1.1688（impl 20.6454 ms） |

## 性能明细（status 以 verify 为准；延时取 perf_result）

| case | status | shape | framework ms | impl ms | speedup | fwd_o 合计 us |
|---|---|---|---|---|---|---|
| 1 | pass | B=1 T=16358 H=32 K=V=128 N=12 | 27.3127 | 23.1850 | 1.1780 | 1498.4 |
| 2 | pass | B=1 T=11166 H=32 K=V=128 N=18 | 18.4759 | 16.3409 | 1.1307 | 1069.3 |
| 3 | pass | B=1 T=16334 H=32 K=V=128 N=16 | 26.7087 | 23.3441 | 1.1441 | 1501.5 |

## Phase 4 记录

| iter | 动作 | 结果 | 决策 |
|---|---|---|---|
| opt_iter_1 | 优化点 3（分核）：两个新核 program 槽位 24→48 | 1.1508 > 1.1423；单核合计 1366.4→1356.4 us（case1 −2.8%、case2 +1.1%、case3 持平）；三个 case 整机均无退化 | **采纳** |
| opt_iter_2 | 优化点 31（IR）：`RUN_CHUNKS 2→4` | **core dump（exit 139）**，最小 shape（T=128/NT=2/H=32）可复现 | **回退** |

## 代码路径

- 最终：`{work_dir}/GatedDeltaRuleTraining_generated.py`（md5 `b6aef21d6fe626941df06b714b0bd3e9`）
- 逐轮产物：`output/iter_0/generated_code.py`、`output/opt_iter_1/optimized_code.py`、
  `output/opt_iter_2/optimized_code.py`（已否决）
- 验证/性能：`output/iter_0/verify/`、`output/opt_iter_1/verify/`、
  `output/iter_0/perf_result.json`、`output/opt_iter_1/optimized_perf_result.json`、
  `output/opt_iter_1/optimized_perf_result_recheck.json`
- 诊断：`output/ir_probe/ir_output/`（6 个 .mlir）、`output/sim/coverage_table.md`
