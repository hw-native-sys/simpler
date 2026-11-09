# A5 HBG 消费者数据可见性

[返回总览](../a5-hbg-async-completion.md)。逻辑依赖解除与消费者读取到新数据分别验证，不以新增公共资源协议作为前提。

## 1. 先验证既有读取路径

同步消费者也可能读取 DMA/peer 产生的数据。条件 READY 证明后端对应操作完成，但观察核的缓存维护不自动作用于其他消费核。必须检查实际生成的读指令、后端写入/发布顺序和已有 kernel 获取操作。

若现有 backend 和读取路径已经保证获取，以指令核对及真实旧-cache 用例证明后复用，不增加空转准备状态机。若存在缺口，再在实际消费者核补充有界获取；不得在新数据到达后写回旧 dirty 数据覆盖它。

## 2. 从现有图解析范围

普通任务已有参数方向、buffer、offset、shape、stride 和依赖边。Graph 的 PARAM 可解析为当前调用参数，SUB_TASK 可解析为本实例 heap；不同 invocation 分开物化。需要缓存范围时，从这些表示与实际后端操作生成 run 内旁表，不扩展 TensorData、Graph Definition、Callable 或跨进程参数协议。

signal 地址不能独自描述 payload，控制依赖也不能凭空给出数据范围。优先使用已有 kernel 参数、SDMA descriptor、编译器通信操作信息和显式数据依赖。纯控制通知不需要虚构 payload。无法证明范围和读取路径的组合保持未支持，不能静默跳过获取。

处理 view、别名及非连续区域时计算实际覆盖范围，避免把 allocation 全部当成待失效数据。cache line 扩展不能破坏其他任务的 dirty 数据。相同 allocation 不等于相同发布版本；同 run 覆写须受正确的数据依赖约束。

## 3. 必要获取的执行位置

run 入口保留既有全 DCache 获取。首次非零 group 发布前置位单调活动标记；依赖该等待的消费者取得执行许可时，必须能通过 bypass load 观察到该标记。活动开始后，基础实现沿用既有全 DCache 观察原语，在依赖满足、最终执行许可确定后，由实际执行核获取输入，再写入本次执行的新异步上下文并调用 kernel。整个 run 从未登记条件时跳过这次额外获取。标记只能在 run 开始前清零，且发布者使用 bypass store 加设备屏障；不得把原子发布、可能被 bypass 读到旧值的字直接用作此门控。这样同步消费者也能读取外部更新的 payload，不依赖完成条件由哪个核观察。该位置之前的调度记录写入必须已发布；不能用获取操作代替发布旧 dirty payload 的所有权处理。具体读取路径和维护成本仍按硬件用例及性能矩阵验证。

只对确实缺少获取保证的读取路径增加准备：依赖满足、参数地址物化后，在读取 predicate 或进入业务 kernel 前，在实际核完成缓存操作。predicate=false 仍需先保证 predicate operand 可见，但不执行业务 kernel。

长范围按实际缓存维护工作量分段。进度绑定当前核、slot generation 和数据发布，不能把别核的准备结果当成可移交 lease。准备期间不等待外部条件、不占 async CHECK lease，并继续遵守调度预算。

MIX/sync-start 若需要额外本核准备，必须覆盖全部实际参与成员；mask 6 中 inactive AIC coordinator 仍履行现有协调职责。最终 start 许可之后直接执行业务，不再插入会改变准备状态的工作。

## 4. 验收

- AIC/AIV 提交者、观察者和消费者不相同，消费者预先持有旧 cache。
- 同步消费者、多 fanin、同 allocation 不同 view、Graph ENTER/EXIT 后的数据读取。
- predicate true/skip、SPMD、MIX 3/5/6/7、sync-start 和 slot 复用。
- 多轮 run 及多卡通知：数据与 signal 顺序、peer 最后访问和覆写约束分别验证。

获取成功不释放资源，也不代表远端已经停止访问。生命周期仍由实际 owner 和 run/domain 退休协议负责。
