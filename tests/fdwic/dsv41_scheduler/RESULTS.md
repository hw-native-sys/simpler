# DSV4.1 L0 平均耗时

本地实测日期：**2026-11-12**。**18 组全部通过**；每组 1 次预热、10 次正式测量，逐轮数值校验。
工作负载见 [README.md](README.md)。

## 测量口径

- Runtime、runner 和 replay 均从 `1b3ee8f91` 的独立源码快照重新编译，包含 #3009 及启动、预热边界修复。
- Ascend950DT_9581（A5），CANN 9.3.0，驱动 25.6.rc2.b023；默认 `-O3 -g`，L0，resident scheduler mode 1；通过 `task-submit` 独占设备。
- 复用固定输入和原业务 kernel：每 rank 72 个有效 token、补齐至 80 行、64 个本地 expert；零权重输入，数值验证范围限于这套输入。
- 指标为 `chip.run.runner_run.device_wall`，单位 **ms**，不含 Host 准备、前缀准备和预热，包含 case 内真实通信。
- EP2 在两 rank 完成 `prepare_native_run` 后对齐启动；每轮先取两 rank 耗时的较大值，再计算 10 轮的算术平均值。

| Case | 实现 | 平均值 |
| ---- | ---- | ------ |
| swa | PyPTO-lib | 0.878180 |
| swa | DSL | 1.317157 |
| c2a_full | PyPTO-lib | 1.518331 |
| c2a_full | DSL | 1.679095 |
| c2a_reuse | PyPTO-lib | 1.222154 |
| c2a_reuse | DSL | 1.621038 |
| c1a_full | PyPTO-lib | 1.512397 |
| c1a_full | DSL | 1.723948 |
| c1a_reindex | PyPTO-lib | 1.447787 |
| c1a_reindex | DSL | 1.745245 |
| c1a_reuse | PyPTO-lib | 1.216660 |
| c1a_reuse | DSL | 1.604008 |
| moe | PyPTO-lib | 3.552518 |
| moe | DSL | 1.783283 |
| engram | PyPTO-lib | 1.435163 |
| engram | DSL | 1.452578 |
| layers_19_21 | PyPTO-lib | 11.022069 |
| layers_19_21 | DSL | 9.901623 |
