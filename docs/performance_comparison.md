# GDN 全量 Case 性能对比报告

## 1. 评测设置

- 用例：**576 个 case**（192 个 shape × 3 个 case，每个 case `profile_calls=30`）
- 计时：`torch.npu.Event.elapsed_time`，完整 forward+backward Model call，profiler 关闭
- 每 case：warmup 5 + repeats 10，取逐 case 中位数；聚合值为 576 个中位数的中位数
- 基线：flash ACLNN，同一进程内同 case 对拍
- 设备：Ascend 910B2C，CANN 9.0.0，torch-npu 2.10.0.post2，triton-ascend 3.2.1

## 2. 结果总表

| 实现 | 中位延迟 (ms) | 相对 flash ACLNN | 相对纯 Triton | 逐 case 中位加速比 |
|---|---:|---:|---:|---:|
| flash ACLNN | 27.28 | 1.000x | 2.030x | 1.000x |
| 纯 Triton | 55.39 | 0.495x | 1.000x | 0.493x |
| g087 版 | 25.11 | 1.088x | 2.206x | 1.083x |
| **dqkwg 版（最新）** | **24.29** | **1.133x** | **2.281x** | **1.128x** |

## 3. 逐 case 分布（576 case）

| 实现 | min | p10 | 中位 | p90 | max | >1.00x | >1.05x | >1.10x | >1.15x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g087 版 | 1.0108x | 1.0671x | 1.0829x | 1.1025x | 1.1224x | 576/576 | 566/576 | 81/576 | 0/576 |
| **dqkwg 版** | **1.0842x** | **1.1111x** | **1.1281x** | **1.1488x** | **1.1760x** | **576/576** | **576/576** | **564/576** | **52/576** |

## 4. 结论

- **dqkwg 版最快**：中位 24.29 ms，相对 flash **1.133x**，相对纯 Triton **2.281x**。
- dqkwg 版在 **576/576** 个 case 上都超过 flash 1.05x，其中 564/576 超过 1.10x、52/576 超过 1.15x；最低 1.0842x，分布集中。
- 相比 g087 版（25.11 ms），dqkwg 版快 **3.4%**。
- flash ACLNN 27.28 ms；纯 Triton 55.39 ms，flash 比纯 Triton 快 **2.03x**。
- 绝对延迟随 session/设备负载变化：trace 记录的 flash/g087 为 24.37/22.56 ms（1.080x），本报告同 session 为 27.28/25.11 ms（1.088x），比值一致；dqkwg 同 session 为 24.29 ms（1.133x）。

## 5. 精度

- flash ACLNN、g087 版、dqkwg 版均在 3 个代表 case 上 **3/3 PASS**（golden = flash ACLNN）。
- 纯 Triton 与 flash 在 fp32 `dg` 上存在参考实现差异（严格 matched_ratio≈0.19，max_abs≈4.5e-3，硬上限内 0 越限），因此纯 Triton 仅作为性能基线。
- 全量 576 case 本轮检查每个 case 输出 `isfinite`；严格精度验收以 flash 为 golden。

## 6. 原始数据

- `results/full_flash.json`
- `results/full_pure.json`
- `results/full_g087.json`
- `results/full_local_dqkwg.json`
