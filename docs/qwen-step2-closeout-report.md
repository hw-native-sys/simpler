# Qwen 步骤二收口报告（closeout draft）

## 范围

本报告覆盖 PR #2447 之后的 Qwen 步骤二收口工作：同一真实 Qwen KV workload 上的 depth=2 请求、串行反馈边界、KV readback 和 bounded P4 能力结论。16 路 distinct KV through HBG 的后处理问题由独立工作线处理，不纳入本 PR。

## depth=2 结果

硬件任务：`task_20260925_043921_34195018347`

结果目录：`/data/sunkaixuan/lcw_subdir/qwen-serving-compatible/qualify-step2-depth2-fallback`

- 请求的 `launch_depth=2`，driver 根据 host sampled-token feedback 合同安全回退到 effective depth=1。
- 127 个 decode dispatch 全部完成，run ID `1..127` 连续。
- 2032 个 sampled token 精确匹配。
- 全部 logits 有限；最大 relative L2 `0.0348324`，最小 cosine `0.9995546`。
- step 0、117、118、126 均完成 40 层 K/V readback；每次 80 个 K/V 检查全部通过。
- step 118 跨页写入检查通过；原始前缀保持逐位不变，未使用尾部保持为零。
- 报告字段记录：`launch_depth_requested=2`、`launch_depth=1`、`depth2_conclusion=safe_serial_fallback_host_sampled_token_feedback`。

这个结果证明 depth=2 请求在当前真实 Qwen driver 上能够安全回退并继续完成完整 depth=1 链；它不宣称当前 driver 已实现真实 joined early enqueue。下一步若要宣称 depth=2 支持，需要满足真实前驱依赖下的 joined native launch 和连续补队证据，属于后续 DEVICE/P5 capability work。

## 当前步骤二出口

步骤二核心路径现在具备：

- 固定 workload manifest、模型和 KV/artifact identity；
- 真实 Qwen `Worker.submit` depth=1 正确性；
- 127-step autoregressive sampled-token feedback；
- token、logits、KV、跨页和资源归属证据；
- depth=2 请求的安全串行回退结论；
- 依赖矩阵和适用范围说明。

步骤二仍受 bounded P4 能力边界约束，不扩展到完整 vLLM serving、A5/TMR、capture/replay、隐式 host/device 同步或 distinct HBG 后处理。
