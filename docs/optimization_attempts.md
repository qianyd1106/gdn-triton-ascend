# dqkwg 优化尝试记录

指标为 `benchmark.py --warmup 5 --repeats 50` 的 3-case 几何平均加速比，基线为 ACLNN framework；单核列为 `chunk_bwd_kernel_dqkwg_postgate_ds_reuse` 的 3-case 聚合耗时。

## 1. 采纳 / 回退

| # | 阶段 | 改动 | verify | speedup | dqkwg us | 决策 |
|---|---|---|---|---|---|---|
| 0 | Phase 3 | 5 项 dqkwg 微优化：K1 复用 K0 `do` tile、DG-MM5 复用 `q`/`k`、K0 `k`/`g` load 上提、K1 saved-H 尾部预取、`dv` 前移 + DW fence 消除 | 3/3 | **1.1290** | **4514.0** | 采纳 |
| 1 | Phase 4 opt_iter_1 | K1 的 `q`/`k` panel 只载一次（循环不变量外提） | 3/3 | 1.1171 | 4754.2 | 回退：live range 拉长 |
| 2 | Phase 4 opt_iter_2 | K1 复用 K0 的 `v`/`dv` tile | — | — | — | 未执行：同 live-range 机制否决 |

## 2. 不重试的方向

| 方向 | 原因 |
|---|---|
| dqkwg 三分裂（dqkw + dg + dw，BK=128） | 更早轮次实测 −3.0% ~ −3.6% |
| BK=128 单 panel / gdiff workspace | UB 溢出（254 KB / 240.5 KB） |
| `i32` mask → `fp32`、mask 外提、手工 CSE | 实测零收益或负收益 |
| multibuffer / CV balance 编译旋钮 | 该 workload 上为噪声级 |
| `al.scope` 显式 CV 融合 / `al.multibuffer` | 910B2 上编译器崩溃 / 输出 NaN |

## 3. 结论

- 可删的重复 load 已删完；继续删 load 会拉长 UB live range，实测负收益。
- 单核从 5096.3 us 降到 4514.0 us（−11.4%），整机 3-case 1.1290（round-local）/ 1.1379（官方 recheck）。
- 后续应转向 wy pair、dhu、fwd_h 以及 fwd_o 的 score consumer/producer。
