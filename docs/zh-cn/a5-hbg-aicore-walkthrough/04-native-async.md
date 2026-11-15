# 4. Native Graph 与异步完成

[总索引](README.md) · [上一段：协作与完成](03-cooperative-completion.md)

## Graph 如何进入同一条执行链

Host 对每个 shell 展开 ENTER、children、EXIT，并保存原始 TaskId 与 expanded ref 的映射。
原始 Graph 只由 Host 持有；设备每个唯一 Graph 保存一份紧凑拓扑与 child plan，各 instance 独立绑定参数 tail。
Host 预构造 child 的全部 block/lane Call，设备没有 GraphExecution materializer 或 child 参数补建路径。

例如 A → G(c0 → c1) → B 展开为 A0 → ENTER1 → c0(2) → c1(3) → EXIT4 → B5，
completion_refs=[0,4,5]。AICPU 启动后只监督；AIC/AIV 从目录取得 c0/c1 的具体执行，
通过 ENTER 与 child gate 的 fanin proof 获得依赖许可。
EXIT 通过全部 sink gate 后完成，外部 B 才能运行。

ENTER、EXIT 和 DUMMY 是无业务 slot 的控制节点。即使两个 kernel slot 都满，仍执行有限控制 scan。
Graph 内有多个不连通分量时，EXIT 也必须等待每个 sink，不能把最后记录的 child 当作整个实例完成。

80 B 顶层镜像的 Host 镜像提供本次 shell boundary；Host 完成重定位，设备通过 native header/template/instance 访问 child 拓扑与已绑定参数。
Graph view v6 的可信 extent 先验证 header 范围，再读取 native image v5；
offset/count、对齐、非重叠、节点数量和 epoch 损坏都会阻止合法执行。

## 参数区与稳定 Binding

每 child 的实际参数 tail 为 align64(tensor_count×128)+align64(scalar_count×8)，
有 predicate 再加 64 B。实例 args_base=heap_base+required_heap；无参数 child 可以使用零 tail。
公共 GraphExecution 的 execution_storage_bytes 不参与默认 HBG 的参数预算。
Host 缓存每 Graph 的参数预算，提交与 bind 使用同一检查算法。

PARAM 来自本次 boundary，SUB_TASK 来自本实例业务 heap；scalar 按值绑定，
predicate operand 地址按来源重定位，值在执行时读。
不同 instance 不能共享可变参数/控制或借用另一个 run 的 publication source。
Call args[48]/args[49] 指向 execution 的 LocalContext/GlobalContext，核内 slot 复用不改变这些地址。

## Kernel 返回与 async DONE

GlobalContext 已包含本轮 async image base、extent、epoch、group offset/index。
实际执行入口只校验并接收这个 prebound group，用户 kernel 按需取得 AsyncCtx 并注册条件。
它不再按 task/block/lane 在设备重新映射或构造一份 async Context。

每次执行 seal 自己的 registration；非空条件保留一份额外 debt。
CHECK 有界检查 COUNTER/SDMA 等条件，FINISH 用唯一 lease 结清其注册责任。
只有 compute 和 async debt 都归零，producer gate 才向消费者证明 DONE。
consumer 在依赖满足后执行实际业务输入的观察，不能从已接收 immutable Call 推导数据可读。

本核预绑定 Context 可供 kernel 使用，返回结算仍使用 executor 保留的可信 group identity，
不能让 kernel 对公开 Context 字段的修改冒充另一个 group 的完成。
空注册不制造新的完成贡献，未完成的 reservation、非法条件或 stale epoch 按首错处理。

Graph EXIT 等待 sink 的逻辑 DONE，因此自然包含其 async 完成；不引入独立的 Graph 全局 child 计数器。
整个 run 的 backing 仍需经过资源/trace 和 EXIT/ACK 清退证明后才能回收。

更完整的 async 条件、容量、lease 和错误合同见[异步完成设计](../design/a5-hbg-async-completion.md)。
