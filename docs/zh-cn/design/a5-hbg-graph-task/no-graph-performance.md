# 无 Graph 完整性能对照：T/H8

本页汇总 PR #2864 的完整 19 项无 Graph 性能复测：固定 benchmark 7 项、筛选 ST 11 项、CSA 1 项。
被测提交为 `dbf39563cb966fd7600dc1d7b93c6d4bd7c48e5c`，默认 **T/H8**。
本表是该提交的历史实测；后续重放主线 `313499d61` 并增加 PARAM 视图范围检查，
新版本的性能验证须单独记录，不能把本表改标为新提交的测量。

**性能门禁尚未通过**：7 项 Device 中位数比历史主线高出 5% 以上。
历史主线为 `8f42cef9b3f0ebbda800e239aab0949e95552f25`，未重采最新主线；
设备与 PTO-ISA 版本差异保留，表内差值不能直接视作代码修改的因果结论。
Qwen 另见[模型性能与队列比较](qwen-performance.md)。

## 采样与来源

- 无 Graph 19 项：每项 golden + 三个独立进程，每进程100稳态轮；普通用例预热3轮，CSA预热5轮。表内为三批统计量的中位数。
- Qwen：40层、B16、seq_len=3500、seed=1234；3轮完整 out/K/V 校验后，独立进程5轮预热+100稳态。
- 本次补齐13项；此前刚完成的6项和Qwen经HEAD、runtime哈希、设备及参数核对后保留，不重复占用设备。两次任务和每项来源保存在comparison.json中。
- 所有20项golden通过，19项共76个功能/计时进程核验resident模式；Qwen108次运行核验fresh native模式。
- 历史主线固定为8f42cef9，未重采最新主线、未做交叉测试。主线无Graph在设备7、Qwen在设备3；当前均为设备6，PTO-ISA也随主线升级，小幅差异不直接作为代码因果结论。

原始逐轮数据、运行库哈希和进程模式记录保存在本地
`outputs/a5-hbg-graph-task/pr2864-full-perf-20261104/`，不随仓库分发。
历史入口特化候选的结果保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`；
本页表格仅展示当前 H8 测量，不拼接不同候选作为当前结果。

## Device（无 Graph，µs）

| 用例 | 历史主线中位数 | 当前中位数 | 变化 | 当前p95 | p95变化 |
| ---- | -------------- | ---------- | ---- | ------- | ------- |
| alternating_matmul_add/Case1 | 717.27 | 717.59 | +0.04% | 726.52 | +0.09% |
| paged_attention_unroll/Case1 | 895.36 | 947.17 | +5.79% | 980.08 | +8.42% |
| paged_attention_unroll/Case2 | 627.27 | 624.26 | -0.48% | 645.56 | -0.28% |
| batch_paged_attention/Case1 | 3101.50 | 3433.48 | +10.70% | 4255.92 | +12.98% |
| benchmark_bgemm/Case0 | 743.04 | 764.65 | +2.91% | 777.08 | +3.61% |
| paged_attention_unroll_manual_scope/Case1 | 1106.40 | 1159.91 | +4.84% | 1190.04 | +6.43% |
| paged_attention_unroll_manual_scope/Case2 | 694.13 | 729.60 | +5.11% | 766.84 | +8.00% |
| fanout_imbalance/fanout_1024 | 1861.85 | 1899.34 | +2.01% | 1925.30 | +2.54% |
| multi_core_dag/mixed_chain_1024 | 6205.00 | 7553.07 | +21.73% | 7594.36 | +21.23% |
| multi_core_dag/mixed_fanin32_1024 | 6988.77 | 5497.44 | -21.34% | 6952.34 | -1.74% |
| multi_core_dag/mixed_random_1024 | 538.80 | 535.76 | -0.56% | 559.41 | -0.57% |
| multi_core_dag/mixed_multi_root_4096 | 497.38 | 895.26 | +79.99% | 953.83 | +89.77% |
| single_block_mix/chain16_mask7 | 1071.64 | 1055.54 | -1.50% | 1086.35 | -1.23% |
| single_block_mix/ordinary_mix_interleaved | 349.50 | 309.23 | -11.52% | 324.79 | -11.38% |
| single_core_dag/aic_chain_64 | 365.93 | 461.08 | +26.00% | 473.47 | +25.66% |
| single_core_dag/aiv_chain_64 | 387.21 | 656.69 | +69.60% | 667.06 | +68.10% |
| spmd_mixed_waves/waves_40x24_stress | 23285.20 | 21068.62 | -9.52% | 21366.79 | -9.94% |
| spmd_multiblock_aic/aiv_block_num_128_waves | 270.28 | 260.36 | -3.67% | 268.81 | -4.21% |
| deepseek_v4_pro_csa/B4S2Seed0 | 3131.39 | 2879.61 | -8.04% | 2932.70 | -7.68% |

## Host 与 bind（µs）

| 用例 | Host历史 | Host当前 | 变化 | bind历史 | bind当前 | 变化 |
| ---- | -------- | -------- | ---- | -------- | -------- | ---- |
| alternating_matmul_add/Case1 | 24295.46 | 24549.86 | +1.05% | 14589.37 | 14750.99 | +1.11% |
| paged_attention_unroll/Case1 | 30971.17 | 30425.12 | -1.76% | 27490.18 | 26920.81 | -2.07% |
| paged_attention_unroll/Case2 | 11110.66 | 11626.76 | +4.65% | 8430.77 | 8657.49 | +2.69% |
| batch_paged_attention/Case1 | 35456.06 | 35309.01 | -0.41% | 28892.40 | 28380.17 | -1.77% |
| benchmark_bgemm/Case0 | 13638.14 | 14460.17 | +6.03% | 9384.60 | 9844.26 | +4.90% |
| paged_attention_unroll_manual_scope/Case1 | 30995.53 | 29708.84 | -4.15% | 27444.89 | 26308.57 | -4.14% |
| paged_attention_unroll_manual_scope/Case2 | 11309.57 | 11723.73 | +3.66% | 8558.83 | 8604.98 | +0.54% |
| fanout_imbalance/fanout_1024 | 3457.48 | 3759.99 | +8.75% | 413.07 | 482.61 | +16.83% |
| multi_core_dag/mixed_chain_1024 | 7661.64 | 9500.83 | +24.01% | 405.89 | 508.58 | +25.30% |
| multi_core_dag/mixed_fanin32_1024 | 9735.74 | 7984.59 | -17.99% | 1090.44 | 1001.56 | -8.15% |
| multi_core_dag/mixed_random_1024 | 1652.60 | 2029.72 | +22.82% | 347.47 | 479.75 | +38.07% |
| multi_core_dag/mixed_multi_root_4096 | 4112.36 | 4788.90 | +16.45% | 1446.60 | 1447.19 | +0.04% |
| single_block_mix/chain16_mask7 | 5921.59 | 4785.24 | -19.19% | 2670.07 | 2121.39 | -20.55% |
| single_block_mix/ordinary_mix_interleaved | 5533.57 | 4785.60 | -13.52% | 3110.36 | 2582.99 | -16.96% |
| single_core_dag/aic_chain_64 | 796.94 | 1889.69 | +137.12% | 64.36 | 238.50 | +270.60% |
| single_core_dag/aiv_chain_64 | 1229.85 | 1044.97 | -15.03% | 154.43 | 64.14 | -58.47% |
| spmd_mixed_waves/waves_40x24_stress | 30946.07 | 26752.59 | -13.55% | 1619.77 | 1136.42 | -29.84% |
| spmd_multiblock_aic/aiv_block_num_128_waves | 1482.51 | 1639.47 | +10.59% | 202.19 | 229.39 | +13.46% |
| deepseek_v4_pro_csa/B4S2Seed0 | 26395.69 | 32342.71 | +22.53% | 19643.73 | 25356.39 | +29.08% |

## AIV chain 64 的归因边界

此前将退化归因于全局队列的说法不成立：本页测量使用 T/H8，所有核可访问队列并不等于使用 T/G 单队列。
单轮 L3 与历史主线 L3 均显示由一个核执行 64 个任务，并发生 63 次 direct refill。
refill 本身累计耗时未增加，主要可见增量位于 refill 结束至下一 kernel entry：单次中位数由 2.660 µs 增至 5.743 µs。
具体检查、缓存维护或执行入口开销尚未分离；带插桩的单次诊断不能直接用于分摊表内稳态差值。
