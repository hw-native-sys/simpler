# torch_npu 多流 AICPU 错误传播探针

该探针建立与 Simpler kernel 模式相同的流间依赖：

```text
caller record start
→ producer wait start
→ producer launch AICPU task
→ producer record done
→ caller wait done
```

它不增加结果判断 task。AICPU 入口只返回 CANN 约定的 native `0/2`，由
RTS 决定错误在 caller stream、device synchronize 和 ACLGraph replay
中的实际可见行为。Python 侧使用 `torch_npu.npu.Stream/Event/NPUGraph`；
Host bridge 只负责加载探针 SO 并在 Torch stream 上调用
`rtsLaunchCpuKernel`。

每个错误观察场景必须在独立进程执行。设备进入错误状态后，进程直接
退出，不 reset、不卸载 SO、不复用 context。

构建 `device/` 和 `host/` 后运行：

```bash
python torch_probe.py DEVICE DISPATCHER_SO PROBE_SO HOST_BRIDGE \
  eager caller success --expect-error 0
python torch_probe.py DEVICE DISPATCHER_SO PROBE_SO HOST_BRIDGE \
  eager caller error --expect-error 1
python torch_probe.py DEVICE DISPATCHER_SO PROBE_SO HOST_BRIDGE \
  eager device error --expect-error 1
python torch_probe.py DEVICE DISPATCHER_SO PROBE_SO HOST_BRIDGE \
  replay device error --expect-error 1
```

`--aicpu-num` 可将同一入口按多个 AICPU block 启动，用于对照 Simpler 的
正式 launch。输出中的 `observed_error` 是机制事实；只有传入
`--expect-error` 时才断言观察结果。预期值应在目标 CANN/torch_npu
版本上实测后固定到集成测试，而不是由探针预先猜测具体异常码。
