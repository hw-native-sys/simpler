# DSV4.1 L0 平均耗时

本地实测日期：**2026-11-12**。**18 组全部通过**；每组 1 次预热、10 次正式测量，逐轮数值校验。
工作负载见 [README.md](README.md)。

## 测量口径

- 源码：`nalinaly/main@2f61d0c6c`，已同步 `hw-native-sys/main@5ece1ec72`；本次使用干净源码及匹配的 runtime、runner、replay，保留 benchmark 启动、预热边界修复。
- Ascend950DT_9581（A5），CANN 9.3.0，驱动 25.6.rc2.b023；默认 `-O3 -g`，L0，resident scheduler mode 1；通过 `task-submit` 独占设备。
- 复用固定输入和原业务 kernel：每 rank 72 个有效 token、补齐至 80 行、64 个本地 expert；零权重输入，数值验证范围限于这套输入。
- 指标为 `chip.run.runner_run.device_wall`，单位 **ms**，不含 Host 准备、前缀准备和预热，包含 case 内真实通信。
- EP2 在两 rank 完成 `prepare_native_run` 后对齐启动；每轮先取两 rank 耗时的较大值，再计算 10 轮的算术平均值。

| Case | 实现 | 平均值 |
| ---- | ---- | ------ |
| swa | PyPTO-lib | 0.876225 |
| swa | DSL | 1.318693 |
| c2a_full | PyPTO-lib | 1.445026 |
| c2a_full | DSL | 1.731658 |
| c2a_reuse | PyPTO-lib | 1.246232 |
| c2a_reuse | DSL | 1.672896 |
| c1a_full | PyPTO-lib | 1.447094 |
| c1a_full | DSL | 1.759561 |
| c1a_reindex | PyPTO-lib | 1.382551 |
| c1a_reindex | DSL | 1.729716 |
| c1a_reuse | PyPTO-lib | 1.227372 |
| c1a_reuse | DSL | 1.660273 |
| moe | PyPTO-lib | 3.575450 |
| moe | DSL | 1.635543 |
| engram | PyPTO-lib | 1.244601 |
| engram | DSL | 1.248369 |
| layers_19_21 | PyPTO-lib | 11.469886 |
| layers_19_21 | DSL | 9.642324 |
