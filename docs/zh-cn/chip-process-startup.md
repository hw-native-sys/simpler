# 芯片子进程启动与共享输入

onboard 平台的分层 Worker 使用新解释器启动芯片子进程。父进程关闭直接 L2 Worker 后，L3 芯片进程不再继承父进程中的 SDK、context 和驱动缓存。直接 L2 的同进程调用方式保持不变；模拟平台和 Python SUB 仍使用原有 fork 路径。

芯片任务的主机输入应使用显式共享 Buffer。`FORK_SHM` 和 `FORK_COW` 只在确实继承了原地址空间的端点有效，不能传给新的芯片解释器。即使 Tensor 在 `init()` 前已分配，初始化后才登记的旧地址也不会自动变成可共享的 backing。

```python
from simpler.task_interface import DataType, TaskArgs, TensorArgType

shared = worker.create_buffer(16)
# 将输入写入 shared.shm.buf，例如通过 struct.pack_into 或 torch.frombuffer。
args = TaskArgs()
args.add_tensor(shared.tensor((4,), DataType.FLOAT32), TensorArgType.INPUT)
```

`Orchestrator.alloc()` 和 `Worker.alloc_shared_tensor()` 的中间结果由 Worker 的共享 HeapRing 管理。描述符记录 backing fd 和分配偏移；芯片或 SUB 可在不同虚拟地址映射同一 backing。调用者不应保存中间结果并跨越所属 run/scope 使用。

创建者拥有 backing，导入者只拥有本地映射。多个分配共用同一物理映射；所有子进程退出并完成清理后，Worker 才释放自身资源。初始化失败沿用现有回滚与 `waitpid` 回收协议；关闭失败仍按原约定保留尚未清理的资源。
