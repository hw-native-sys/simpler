# 协作任务的 Host 准备与 Device 校验

普通图和 native Graph 展开后的每个任务实例都有独立的 `task_ref`。Host 在发布设备镜像前固定依赖、执行形状和调度顺序。Device 按实际拓扑、领取结果和本轮执行进度确定谁能进入 kernel。

## Host 发布什么

- Host 对实例 DAG 排序，为每个实例分配唯一的 `rank`。数据依赖与 Graph 控制边都必须从较低 rank 指向较高 rank。AIC、AIV 目录分别保留这个顺序在本引擎上的投影，MIX 进入 AIV 目录。
- `SchedulerTaskCatalogOrder` 保存 rank、协作记录索引、每条物理 lane 的 `required_floor` 以及下一个 MIX 实例。对有冲突的自取任务和 MIX，所需边界取自较早的另一类任务；纯 AIV 可以占用两条物理 AIV，MIX 发布还要计入 AIV0 的职责。
- 形状检查在启动前排除超过不同执行位置容量的多块 `sync_start`。纯 AIC 按 AIC 数计，纯 AIV 按 AIV 数计，MIX 按簇数计。合法形状的实际 block 归属随后由 Device 领取确定。

## Device 如何使用

启动时，`scheduler_initialize_order` 核对冻结目录与实际拓扑。每簇 AIV0 在 SSBUF 发布本簇 MIX 描述符和单调的 `mix_floor`。任务拿到具体 block 后，owner 可以提前接收 Call 并做固定准备。第一次承诺有冲突的资源前，须结合本 lane 的阈值、本地尚未退休的责任及另一来源的实际进度作判断。

MIX 各计算成员确认依赖条件后发布本代 ARRIVED，观察到活动成员全部到齐才发布最终 payload。组完成由活动的记账成员收齐 FIN 后贡献。纯任务等待前驱逻辑完成后发布自己的最终 payload。供给关闭只允许目录继续分发，异步任务仍须结清真实条件信用才会释放数据后继。

具体分发、入口、退出及性能测量见 [HBG 调度设计](build-execute-separation.md)。早期实验中的操作和测量条件见[历史调查](../../investigations/2026-11-a5-hbg-admission-shortcuts.md)。
