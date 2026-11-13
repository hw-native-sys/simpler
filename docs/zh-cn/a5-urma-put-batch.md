# A5 URMA 批量 PUT

A5 TMR Program 可在同一 Device 通信域、同一 AIV 和同一 peer 内批量提交 PUT。L3 调用在本次
`CallConfig` 上设置 `enable_urma_batch = True`。域用 `workspace_engines=("urma",)` 获取 Device URMA
上下文，双 jetty 沿用 `UrmaConfig(jetties_per_aiv=2)`。

```cpp
const UrmaRequestDescriptor<GlobalData, GlobalData> requests[] = {
    UrmaTput(dst0, src0, workspace, peer),
    UrmaTput(dst1, src1, workspace, peer),
};
if (!send_urma_put_batch(args, comm, requests, 2, 2)) return;
```

普通 PUT 使用 `send_request_entry` 逐条提交。使用批量入口时，源和目的须分别位于域的本地注册窗口和对端 MR，
调用方显式传入请求列表的实际条数和正数 `batch_size`。
PTO-ISA 按非空请求数执行阈值提交，simpler 在最后一个非空请求上显式结束批次。
例如阈值为 4 时，3 个请求会在第 3 个提交，5 个请求会在第 4、5 个分别提交。
空批次直接成功，分片由 PTO-ISA 执行。立即和批量路径共用只读 SQ/CQ 与完成条目容量预检，
AICPU 负责消费 CQE。

成功返回表示全部已发布。最后一个已确认的 flush 事件含所有此前提交的目标，调用结束时复用原
完成协议登记实际参与的各 jetty。AICPU 等齐后才释放下游依赖。换 peer 时另起一次调用。
与 TGET、立即 PUT 或跨队列消费者衔接时，用任务依赖建立顺序。

固定 ISA 版本在返回失败事件前不会发布本次 flush。AICPU 排空此前已确认的 DMA 后，
Host 通过本次运行的终态记录返回原始错误并释放资源。设备中断或已发布事件无法登记时，
资源继续由原所有者保留。peer 0 的全部 BB 目标恰好回绕到 0 时，固定 ISA 也会拒绝发布。

仅显式启用批量的 Program 需要在计算流结束后读取 DMA 完成记录。批量使用框架流，
不支持 joined launch、caller stream 或 capture。同卡普通运行沿用原路径。

## 选择批大小

同一 AIV、同一 peer、单 jetty、每轮 33 个同尺寸 PUT 的实测中，4–32 KiB/请求可先尝试 Batch 16，
64–128 KiB/请求可先尝试 Batch 4。更大请求默认可选立即提交。128 KiB 是这组负载的性能建议，
批量接口按 ISA 能力和队列容量接收请求，允许显式测试更大的 payload。

两卡 A5DT、CANN 9.3.0、PTO-ISA `90a3796597` 上，128 KiB 的 Batch 4 相对立即提交，
单向 PUT 吞吐提升 50.3%，反序复测提升 55.4%，双向完整链路提升 11.8%。
单向吞吐收益在 192–224 KiB 之间反转。完整对照与计时口径见 [#2855 实测记录](https://github.com/hw-native-sys/simpler/issues/2855#issuecomment-6770413129)。

`TestUrmaPutBenchmark` 在相同请求数、有效字节数和数据校验下比较立即提交与 Batch 1、4、8、16、32。
使用 `task-submit` 锁定同一卡对，按相同规模先跑立即提交，再跑各批量档。
例如下面的命令运行 32 KiB 的 Batch 16。

```bash
.venv/bin/python -m pytest \
  tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion/test_urma_deferred_completion.py::TestUrmaPutBenchmark::test_run \
  --resource-child --level 3 --runtime tensormap_and_ringbuffer \
  --case put_32k_batch16 --manual only --platform a5 --device "$TASK_DEVICE" \
  --require-pto-isa --rounds 80 --log-level timing -s -v
```

逐轮校验数据、保护区和 SQ 推进量。独立计数实验测得，单 jetty 每轮 33 个 PUT 的
SQ doorbell 次数在立即、Batch 1、4、8、16、32 下分别为 33、33、9、5、3、2。
回归还覆盖实际 SQ/CQ 槽位回绕、双 jetty、首批提交后注入失败，以及 256 MiB + 4 KiB 的真机分片。
