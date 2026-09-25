# Qwen 步骤二依赖矩阵

| 对象 | 生产者 | 消费者 | 可提前准备 | 可提前 native 提交 | 必须等待/回退 |
| --- | --- | --- | --- | --- | --- |
| 只读权重、RoPE | fixture/model setup | 每个 decode run | 是，驻留后复用 | 是 | 版本和地址就绪 |
| 初始 KV page | `ReferenceFixture.layer()` + upload | 当前 decode attention | 是，上传完成后 | 是 | upload 完成、页表有效 |
| 新写入 KV | 当前 decode kernels | 下一 decode step | 否 | 否 | 当前 run 完成且调用方完成必要同步 |
| sampled token | 当前 run device output | 下一步 `sampled_ids_in` | 否 | 否 | `RunHandle.result()` 和 host copyback |
| seq_lens / slot_mapping / block_table | `ReferenceFixture.step()` | 当前 decode run | 是，host 值已知 | 是 | 与当前 KV/page 状态一致 |
| logits / sampled output | current run | comparison / adapter state | 否 | 否 | run completion、copyback |
| output buffer | Worker-owned device buffer | current run / host readback | 地址可提前准备 | 依赖输出生命周期 | 最后消费者读取后复用 |
| workspace / staging | Worker/runtime | current run | 按容量准备 | 由 runtime 管理 | 最后使用者退休 |
| depth=2 successor | 当前 driver 的 host feedback 路径 | 下一 run | 只能准备固定元数据 | 当前 driver 安全回退 depth=1 | sampled token/KV 前驱完成 |

当前矩阵的关键结论是：host sampled-token feedback 是真实 decode 的串行边界。depth=2 请求已完成安全回退验证；它不被解释为 joined native early enqueue 支持。

该矩阵适用于步骤二当前 bounded P4 范围，不引入隐式 D2H、隐式 stream sync 或跨 run 自动依赖跟踪。
