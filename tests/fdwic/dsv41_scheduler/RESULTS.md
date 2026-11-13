# DSV4.1 三组配置 L0 平均耗时

9 个 workload，三组配置共 **27 项**，全部通过数值校验。单位 **ms**；每项 1 次预热、10 次正式测量，表中仅列算术平均值。

| Case | PyPTO-lib 原版 | DSL 原版 | DSL ≤64 blocks 加 sync_start |
| ---- | -------------- | -------- | ---------------------------- |
| swa | 0.876225 | 1.288730 | 1.502727 |
| c2a_full | 1.445026 | 1.749357 | 1.957607 |
| c2a_reuse | 1.246232 | 1.708909 | 1.904286 |
| c1a_full | 1.447094 | 1.748604 | 1.980592 |
| c1a_reindex | 1.382551 | 1.760035 | 1.916404 |
| c1a_reuse | 1.227372 | 1.669830 | 1.884636 |
| moe | 3.575450 | 1.644509 | 2.214012 |
| engram | 1.244601 | 1.246669 | 1.489751 |
| layers_19_21 | 11.469886 | 9.633620 | 11.937680 |

## 测量口径

- 使用干净源码 `nalinaly/main@2f61d0c6c`（已同步 `hw-native-sys/main@5ece1ec72`）及匹配的 HBG runtime、runner、replay，保留 benchmark 启动和预热边界修复。
- A5 Ascend950DT_9581、CANN 9.3.0、驱动 25.6.rc2.b023；默认 `-O3 -g`，L0、resident scheduler mode 1，设备由 `task-submit` 独占。
- 指标为 `chip.run.runner_run.device_wall`，不含 Host 准备、前缀准备和预热，包含 case 内通信。MoE 和 layers_19_21 为 EP2，每轮取两个 rank 的较大耗时，再计算 10 轮平均值。
- PyPTO-lib 列实测于 **2026-11-12**：单卡设备 7，EP2 设备 3、4。两个 DSL 列实测于 **2026-11-13**：原版与变体按 case 依次同卡运行，单卡设备 4，EP2 设备 0、4。跨实现的数据来自不同批次；同步开销比较使用后两列的同卡配对数据。
- 6 组 attention 的 ≤64-block 变体与同日全同步变体逐字节一致，沿用已完成的配对测量。MoE、Engram、layers_19_21 排除超限任务后，原版与变体均重新测量。
- 复用原业务 kernel 和固定零权重输入：每 rank 72 个有效 token、补齐至 80 行、64 个本地 expert。结论只覆盖这套输入、源码和运行顺序。

## DSL 同步变体

仅对选定 case 中 `1 < block_num <= 64` 的计算 task 新增 `require_sync_start=true`，保留已有同步标记。block 数、kernel、输入、任务 ID、依赖及切片外前缀不变；单 block 和控制 task 不改。

超过 64 blocks 的任务不新增同步：

- MoE：`dispatch_push`，源任务 57，128 个 AIV blocks。
- Engram：`engram_widen`，源任务 88，500 个 AIV blocks。
- layers_19_21：`dispatch_push`，源任务 1362、1442、1506，各 128 个 AIV blocks。

9 组按单 rank 分别计数后相加，280 个多 block task 中 275 个带同步、5 个不带，相对原配置新增 211 个。MIX 算一个逻辑 task，不按 lane 重复，也不跨用例去重。AIC/MIX 任务均不超过 32 blocks，符合本机对应核型容量。

用例生成命令见 [README.md](README.md#dsl-bounded-sync-start-variant)，入口为 [production/force_sync_start.py](production/force_sync_start.py)。已验证全部 9 组、两个 rank plan 的二进制差异只涉及指定的同步标记；超限任务不变，重复转换不改变结果。
