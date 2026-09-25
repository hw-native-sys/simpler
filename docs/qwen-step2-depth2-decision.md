# Qwen 步骤二 bounded depth=2 决策记录

本记录属于步骤二收口 PR，范围是当前 bounded P4 program 路径。它不覆盖 distinct request/KV through HBG 的后处理问题。

`reference_worker_submit.py` 接受 `--launch-depth 1|2`。由于真实 decode 的下一步输入来自上一 run 的 sampled token 和 KV 状态，driver 默认对 depth=2 请求执行安全串行回退：`launch_depth_requested=2`、`launch_depth_effective=1`。报告会记录回退原因和后续 depth=1 是否继续通过。

`--depth2-policy attempt` 只用于能力探针。它保留 Worker 的 `launch_depth=2` 配置，但 driver 仍在每个 step 读取 sampled output 后再准备下一步，因此该模式不能证明连续 early enqueue；只有具有真实前驱数据的独立 probe 记录了 joined native launch，才能把 bounded depth=2 标记为支持。

本次真实闭环 probe：`task_20260925_043921_34195018347`。请求 depth=2，effective depth=1，127 个 dispatch 全部通过；step 0、117、118、126 的 40 层 K/V readback 全部通过。该任务给出步骤二需要的安全回退结论，不把当前 Qwen host feedback 路径标记为 joined depth=2 支持。

验收必须绑定同一 workload manifest、模型、KV bundle、artifact、代码 head 和 runtime。报告至少包含：

- 请求的 launch depth、实际 effective depth 和回退原因；
- run/dispatch identity、token、logits、KV readback 和错误归属；
- depth=1 继续执行的硬件 task ID；
- 当前能力范围以及转入后续 P5/DEVICE capability work 的最小缺口。

host sampled-token feedback 需要显式等待和回拷；本记录不通过隐式 D2H、stream sync 或 host 读取 device 值来制造 depth=2 依赖。
