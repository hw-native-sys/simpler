# A5 两个 AIC 之间的 DMB 写入与 CntNotify 可见顺序

同一纯 AIC kernel 启动两个 block。接收核上报当次物理 core ID、原始 DMB 低字及不同于原值的通知序号。发送核通过驱动映射表定位该核的 `DATA_MAIN_BASE`，只写低 32 位。AICPU 不转发通知。判定写入成功的证据是接收核用本核 `MOV DATA_MAIN_BASE` 观察到新序号。[收发核实现](probe.cpp) · [Host 映射与启动器](launch.cpp)

接收核先把 CntNotify 设为 1 并读回，表示已有一个前驱完成。发送核发出最后一次 `INC(1)`，执行选定的屏障，再写 DMB。接收核看见目标序号后的第一次 `ld_dev` 直接读 CntNotify，立即保留该值。后续有界回查只用于诊断最终计数。整个试验只有一个真实的 DMB 写者，另一前驱用已确认的初值表示。每次执行先跑两个 GM 通知对照。提前通知对照在读者记录初值后才允许发送方递增，因此能检验记录逻辑确实捕获了第一次读数。

## 2026-11-25 实测

队列任务 `task_20261125_195243_255696917470`，实际设备 1，Ascend950DT_9581，工具链 CANN 9.3.0。三个独立 launch 各采一次，均落在物理 AIC 0 → AIC 54，按 DAV_3510 映射属于两个 die。真实试写只有最后一行。该行接收核 DMB 低字由 0 变为 `0x1357`。stream 和结果回传均成功，结束时设备健康状态为 OK。[核 ID 布局](../../../src/a5/platform/onboard/host/host_regs.cpp)

| 通知路径 | 发送顺序 | 接收核首次计数 | 最终计数 |
| -------- | -------- | -------------: | -------: |
| GM 提前通知对照 | 写 GM 通知，收到首次取样应答，再 INC | 1 | 2 |
| GM 顺序对照 | INC、DSB_DDR、写 GM 通知 | 2 | 2 |
| 对端 DMB | INC、DSB_DDR、写目标 DMB | 2 | 2 |

该样本验证了指定方向的可达性，并给出一次屏障后的观测。无屏障与 DSB_ALL 的对端结果尚未采集。Ascend C 文档将 DSB_DDR 定义为等待 GM 访问，DSB_ALL 为 GM 和 UB。该定义没有给出 STARS 计数与对端 DMB 之间的可见顺序保证。当前设计以发送端 `INC(1) → dsb(DSB_DDR) → 写对端 DMB` 保序为工作假设。本顺序探针在观察到本轮新 DMB 值时取样完成计数。表中的“最终计数”来自试验取样后的诊断回查。[双前驱协议](../../../docs/zh-cn/design/percounter-peer-dmb.md) · [Ascend C 数据同步屏障](https://asc.gitcode.com/api/SIMD-API/tensor_api/sync/data_barrier.html)

## 运行

在 Simpler 根目录设置 `PROBE_BUILD` 为工作区内的绝对输出路径。每次 `peer-*` 调用执行两组 GM 对照，然后由 AIC 对另一核 DMB 做一次试写。可选参数为 `control`（仅 GM 对照）、`peer-none`、`peer-ddr` 和 `peer-all`。首次计数偏旧时程序返回非零并打印原始读数。接收超时另有独立状态。

```bash
export PROBE_BUILD="$PWD/build/peer-dmb-ordering"
export TASKQUEUE_DEVICE_CONF="$(realpath ../../performance-task-submit.conf)"
tests/fdwic/peer_dmb_ordering/build.sh "$PROBE_BUILD"
task-submit --device auto --device-num 1 --env PROBE_BUILD --run \
  'source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && .claude/skills/onboard-arch-precheck/check.sh a5 && "$PROBE_BUILD/launch" "$PROBE_BUILD/probe.bin" peer-ddr'
```
