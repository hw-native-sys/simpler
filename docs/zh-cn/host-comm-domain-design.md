# Host 接入通信域：双机 ACL 设计

状态：接口、ACL 后端、控制流程与[单／双机验收脚本](host-domain-mpi-validation.md)已实现；
单机两卡本体验收已通过，用户于 2026-10-23 反馈真机 utpy、单机双卡和双机四卡测试全部 PASS，
结果来源及证据范围见验收文档。基于 [#2668](https://github.com/hw-native-sys/simpler/pull/2668)
的 `1251133c07adfd25610614821a8bf80d45dc5235`，复用其 Buffer/Tensor、grant 和跨 run 使用追踪。
本文描述实现契约；既有接口见[通信域文档](../comm-domain.md)。

## 1. 目标与边界

首个实现交付 A3 HCCS 双机闭环。沿用 Direct MPI：一个 MPI 作业内 rank 0 为 L4，
另两个 rank 各为一台机器的 L3 executor，每个 L3 管理两个 NPU child。
沿用现有 fork-before-MPI-init 启动次序和 supervisor，不增加启动器。
单机对照采用同一流程，仅减少为一个 L3 和两个 NPU。

| 节点 | 计算参与者 | 域内物理窗口 |
| ---- | ---------- | ------------ |
| A | Host CPU、NPU0、NPU1 | DRAM_A、HBM_A0、HBM_A1 |
| B | Host CPU、NPU0、NPU1 | DRAM_B、HBM_B0、HBM_B1 |

每节点只有一块域内 Host DRAM，两个本地及两个远端 NPU 均访问这同一 allocation。
四块 HBM 保持现有设备域窗口；域 payload 共六个物理 allocation。
MPI 负责进程启动和控制消息，ACL 负责内存和数据传输；控制消息只交换描述符、范围与结果。
含 Host 的域由 L4 复用逐节点 MPI 控制命令收集描述符并执行事务，暂不使用 NPU 域的
MPI group/allgather 快路径；无 Host 的域保留原流程。

| 发起者与目标 | 实际访问方式 | 完成条件 |
| ------------ | ------------ | -------- |
| CPU → 本节点 DRAM | owner CPU VA 上的受保护字节访问 | CPU 操作完成，满足 CPU/设备交接契约 |
| CPU → 其他 DRAM 或任意 HBM | 选本节点参与 NPU，ACL SDMA 读写 | 选定的同步 ACL copy 成功完成 |
| NPU kernel → 任意 DRAM | 本 chip 导入的 Device VA，通过 Tensor 参数消费 | kernel 所属 run 完成 |
| NPU kernel → HBM | 本地 Tensor，或现有 CommContext 的 peer window | kernel 所属 run 完成 |

“Host 进入通信域”包含 Host 内存供 NPU 访问，以及 Host 发起显式传输。
CPU 原生指针只用于本节点 owner DRAM；本提案不要求跨机 Host-target import 或远端 CPU load/store。
不引入 HIXL/ADXL 头文件、链接或引擎初始化，不交付 A2/A5 新能力、原子操作或自动故障恢复。

## 2. 最小公开接口

只扩展现有 L4 建域，首版不增加 L3 独立创建 Host 域的路径。

```python
domain = orch.get_or_allocate_domain(
    name="mixed",
    members=[(node_a, 0), (node_a, 1), (node_b, 0), (node_b, 1)],
    window_size=N,
    buffers=[CommBufferSpec("payload", "uint8", N, N)],
    host_windows={
        node_a: HostWindowSpec(numa_id=0, nbytes=N),
        node_b: HostWindowSpec(numa_id=0, nbytes=N),
    },
    retain_after_run=True,
)
host_a = domain.host(node_a)                  # 逻辑 Buffer
host_tensor = host_a.tensor([N], DataType.UINT8)
hbm = domain.member(0).buffers["payload"]    # 保留现有接口
```

`HostWindowSpec` 只有合法 NUMA id 和正整数字节数；一个节点一个 byte pool，不增加 Host 命名切片体系。
`host_windows` 为空时保持既有行为，非空时必须恰好覆盖参与域的全部 L3 节点。
按规范化节点顺序保存配置，纳入现有不可变快照；dict 插入顺序不影响复用，配置冲突在副作用前拒绝。
Host allocation 的逻辑大小与对齐后的映射大小分别记录，所有公开字节访问受逻辑大小约束。

下级通过现有 `orch.get_domain(ref)` 借用 handle。所属 L3 可获取 `domain.local_host()`，
复用 `PayloadPart.write(offset, buffer, nbytes=None)` 和 `read(offset, writable_buffer, nbytes=None)`。
薄 access backend 每次操作检查 owner、commit、generation、未 release 及范围，并持有现有 operation lease 和控制预约。
L4、非 owner L3 和失效对象拒绝访问；缓存访问对象不保活，不公开 CPU 裸指针或 memoryview。

在 L3 增加一条同步操作：

```python
orch.copy_domain(
    dst=domain.host(node_b),
    src=domain.host(node_a),
    worker=0,
    nbytes=N,
    dst_offset=0,
    src_offset=0,
)
```

`src/dst` 为同一域的逻辑窗口 Buffer；`worker` 必须是本节点参与该域的 chip。
两端完整 descriptor 必须匹配当前 grant，分别具备读/写权限；字节数为正，范围限于各自逻辑 Buffer。
既有 L3→chip 控制通道只发送池定位和范围，由 chip 解析自己的 Device VA。
CPU 读远端时传输到本机 Host pool 后 `local_host().read(...)`，写远端时反向操作。
HBM 使用已有 member Buffer，Host 使用 `host(node)`；不增加公开 stream、异步 ticket 或通用 Remote COPY 框架。
同物理池重叠范围拒绝，按 pool 与绝对偏移判断，不能只比较命名 Buffer identity。

## 3. 内存与现有实现的接入点

### ACL 路径

L3 在既有 child fork 完成后，建立本节点参与设备的 ACL context，按需使用 native `AclRuntimeApi` 管理 Host owner：
`aclrtMemGetAllocationGranularity` 查询 `HOST_NUMA/P2P_HUGE` 粒度，
`aclrtMallocPhysical` 分配，`aclrtReserveMemAddress` 与 `aclrtMapMem` 建立 CPU owner VA，
`aclrtMemExportToShareableHandleV2(..., FABRIC, ...)` 导出同一物理 allocation。
NUMA id 由每节点配置给出，不按设备编号猜测。

每个参与 chip 使用 `aclrtMemImportFromShareableHandleV2` 导入两个 Host token，
预约 Device VA，`aclrtMapMem` 并按 CANN 约束设置本设备访问权限。
本节点 Host DRAM 同样建立 Device view，CPU owner VA 不传给 chip，也不跨进程解引用。
四块 HBM 的 allocation/export 与 peer 映射复用现有 A3 后端。
设备编号转换复用现有 logical/driver-visible 规则。

用于 SDMA 的 pool VA 必须满足 CANN NoUC 约束，包括作为端点的本地 Host 别名及 HBM。
`aclrtReserveMemAddressNoUCMemory` 要求显式地址及 1 GiB 预约粒度。
地址在 CANN 文档给定的 40–224 TiB 区间内，从现有 native 域 view 推导活跃预约区间并选择空隙；
确认释放后立即复用，不维护第二份地址注册表。外部库占用选定 VA 时明确失败，不覆盖映射、不盲目扫描重试。
这只增加 VA 预约量，不增加物理分配量。指定地址能力在 CANN 文档中仍为实验接口，需按目标 CANN/A3 环境验证。
同一组 NoUC Device view 同时服务 SDMA 与 kernel；单卡预验证已覆盖父进程持有 DRAM、子进程导入及完整窗口校验。

选定 chip 解析两端 Device VA 后固定执行 `aclrtMemcpyAsync + stream synchronize`。
CANN 对 VMM Device 间拷贝列出的接口及硬件预验证均采用此路径。
沿既有 stream 创建通路为每 chip 持有一个私有 copy stream，跨域复用；
每次 copy 必须完成同步才能返回成功。同步失败保留未知完成状态，阻止 view/owner 回收及设备 finalize。
不在 copy 出错后自动改试另一种实现；第一次调用可能已写入部分数据。
既有 stream 销毁是 best-effort，不将其返回值作为物理销毁证据；域回收依赖每次 copy 的完成同步。
新增 ACL 符号只在 Host-enabled A3 路径按需解析，保留其他后端的原初始化行为。

### 描述符与授权

Host 配置只保存 owner node、NUMA、逻辑大小和 canonical identity；HBM、Host 共用一张有序 export descriptor 表，
分别占 `[0,R)`、`[R,R+H)`，mapping size 与 Fabric token 只保存一次。`SDB3` 标记配置布局，旧布局明确拒绝。
Host 不占 NPU rank，不改固定 `CommContext` ABI。
每个 Host pool 只有一个 canonical identity；各 chip VA 不同，但 backing 身份相同。

`DOMAIN_WINDOW` body 保持 28 字节，原 `rank` 字段明确改为内存窗口槽位 `window_slot`：

```text
domain_id:u64 | generation:u64 | window_slot:u32 | slice_offset:u64
```

域有 R 个设备窗口、H 个 Host 窗口时，`[0,R)` 仍对应 device rank，
`[R,R+H)` 按规范化节点顺序对应 Host pool；NPU 成员表和 CommContext 仍只有 R 项。
不增加 kind 字段、BackendKind 或 Buffer 总布局。槽位含义由同一份不可变域元数据确定。
通用 codec 校验格式与最大槽位数；当前 H≤R≤64，最多 128 个槽位，端点再按实际 R+H 校验。
字节数不变不代表语义兼容；通信版本、绑定一致性检查和旧端拒绝仍须更新。
Host Buffer 的 `address_space=DEVICE` 描述 Tensor 的 Device 访问视图，物理归属由槽位元数据标明。

每 chip 复用 `_L2GlobalDomain.buffers`，在 import/commit 时登记全部 HBM slice 和 Host Buffer，保留完整 descriptor。
显式 copy 允许已授权的 peer HBM；Tensor 解析仍检查设备 slot 的目标 chip，peer kernel 继续用 CommContext。
复用 ImportRegistry，包括缓存命中时的 grant 重验、完整 descriptor 比对、generation 和撤销检查。
Host Tensor 沿用普通 TaskArgs 与跨 run 登记。
任何仅通过 Ref、闭包或 scalar 发起的后代访问，包括 CPU 字节操作、同步 copy 和 CommContext 使用，
根 L4 callback 必须在提交后代前 `get_domain(ref)` 登记本 run；L3 借用 handle 本身不为 L4 owner 保活。

配置阶段不分配 identity；创建事务分配一次 Host identity，各端按命令重建 slot 描述符，复用时不重新分配。
`_GlobalNodeDomainState` 及现有清理 journal 持有 L3 native Host owner，后者保存 handle、CPU VA、实际大小和释放进度。
chip 的 `DomainAllocation` 保存 imported views，`window_at(slot)` 从现有 view 取 VA，Python grant 与 native view 共用 slot 顺序。
显式 copy 的本地控制请求只携带域身份、两端 Buffer identity、偏移和字节数；
L2 查同一份已登记 Buffer 得出 slot、切片范围与权限，native 再按 slot 和映射范围检查，禁止请求直接提供 VA。
不另建 owner/address registry、session、launcher、adapter 或 copy 专用命令快照；可从既有状态推导的字段不重复缓存。
不复用会搬运 payload 的 `COPY_TO_DOMAIN/COPY_FROM_DOMAIN` staging 命令冒充 Fabric copy。

```text
建域：L4 allocate → L3 prepare owner/export → chip import → 全员 commit
CPU：L3 local_host → 现有门禁/lease → owner read/write
搬运：L3 copy_domain → 已有 mailbox → chip 校验/解析 → 同步 ACL copy
计算：TaskArgs Host Tensor → 既有 grant resolver → 本 chip Device VA → kernel
回收：既有 run fence → 所有 VIEWS 确认 → 所有 OWNERS 确认 → freed
```

## 4. 建域、排序与回收

### 建域与复用

沿用 `PREPARE_EXPORT → IMPORT → COMMIT`。每节点只创建一次 Host owner，准备现有 HBM 和 Host exports；
控制面核对节点/池完整性、大小与域身份，向所有 chip 发送完整描述符。
全员 import 成功后才发布 Buffer grant 与 CPU 接口，精确复用不重新分配、导入或清零。
发送 PREPARE 前登记 attempted 节点；响应丢失也必须纳入 ABORT，不能视作未创建。

### 完成与可见性

CPU 字节访问与同步 copy 沿用现有 direct-control 门禁：callback 内只能在本 run 首个 submit 前调用；
callback 外要求 outstanding runs 已完成，并持有现有控制预约。`scope_end` 不代替完成等待。
操作全程持有 mapping/operation lease，release 不能撤走正在使用的映射。

这些操作不创建 DAG 依赖，不自动等待已排队任务。选定实现的同步 ACL API 返回成功，
或其异步 API 后的 stream 同步成功，只证明本次传输完成；
跨节点同范围交接由已完成的全域 L4 run 或明确任务依赖排序。
CPU↔NPU 可见性采用 CANN 支持的内存属性与同步契约，须用真实同址交接验证；
普通 CPU fence 不能替代缓存一致性证明，不加入 EL0 cache 指令或 DMA 回读来掩盖 CPU 校验失败。
首版不承诺无序并发写的一致性。

### 正常释放与 ABORT

在既有 RELEASE/ABORT 命令增加内部释放阶段，Host-enabled 域执行相同顺序：

1. 撤销新使用，等待已接纳的 run、kernel 和 copy 完成。
2. `VIEWS`：所有相关节点/chip 释放 imported Device views，收齐每个必要节点的成功确认。
3. `OWNERS`：确认齐备后，各节点释放本地 Host owner 和 HBM owner；全员成功才标记域 `freed`。

Host-enabled 域拒绝绕过阶段的完整单节点 free；无 Host 的域保留现有释放入口。
每条命令带域身份、generation 和阶段，重复调用幂等；阶段进度记在既有域状态。
ABORT 覆盖全部可能创建/导入的节点，准备失败的局部清理不能提前释放已发布 token 的 owner。
不能先对 A 完整 free，再询问 B 是否仍持有 A 的 view。

view 释放失败、确认丢失或设备完成未知时，域失效并停止新访问；owner 进程存活时不主动 free，
保留清理责任并报告失败/UNKNOWN。copy 同步超时不证明 DMA 已静止，不能因此 unmap。
底层操作只有成功才撤销对应资源状态，禁止清空句柄后将失败当作可重试成功。
`Worker.close()` 复用 #2668 journal，但 Host-enabled 域仍须经过同一阶段门禁；
不能由通用 Buffer 清扫、RAII 析构或整树 teardown 提前释放尚未确认安全的 owner。
VIEWS 或 DMA 完成未确认时，正常 close 保留相关 native context，不能继续会销毁它的 device finalize/reset。
若转为作业终止，则按故障退出记录 UNKNOWN，不能报告有序释放成功。
chip finalize 回执不证明 L3 持有的 Host allocation 已释放。
borrowed handle 也不能在仅完成 VIEWS 时标记 `freed`。
MPI Abort/进程退出可能回收内存，本提案不保证失联 owner 留存或恢复；未知回收不报告成功。

## 5. 验收与实现约束

一份参数化流程覆盖单机正对照与必测双机，不为每种方向另写脚本。
双机由现有 Direct MPI supervisor 启动完整作业，域内实际 owner 为两块 DRAM、四块 HBM。
六个发起者对六个池的 READ/WRITE 共 72 项，分别标注 CPU 本地、ACL SDMA 与 kernel；本地结果不代替跨机证明。
源、目标、结果区互不重叠，每项比较完整逻辑窗口和两侧 guard，不访问对齐 padding。
同一 DRAM 区间另做 `CPU → NPU0 → NPU1 → CPU` 及远端 NPU 交接，以完成的 L4 run 分轮等待且不重置 allocation。
详细矩阵、范围布局、A/B 命令及成功条件以[验收流程](host-domain-mpi-validation.md)为准；
六个 allocation 只统计域 payload，结果 Buffer/workspace 单独计数。释放和 executor 清理必须成功，缺硬件、跳过或 sim 不记作双机 PASS。

最小回归集合：

- 参数化 UT：配置规范化与精确复用、单 owner、pool/identity/generation/范围/grant 拒绝、释放后缓存对象失效、
  纯 Ref 后代 CPU/copy 使用的根 run 登记。
- 失败注入 UT：第二 chip import 失败、PREPARE 响应丢失、VIEWS 确认丢失及 view/owner free 失败；
  校验未收齐确认不 free owner、不 finalize/reset context、不误报 `freed`、清理责任不丢失。
- 同一 E2E 流程的单机与 Direct MPI 双机矩阵、同址交接、释放顺序和域 allocation 计数。
- 原有 HBM 域、Buffer codec 和跨 run 使用回归，以及构建/链接检查无新增 HIXL/ADXL 依赖。

旧原型和 #2668 的 MPI group 测试只作参考，不替代这个组合验收。
若需要别名，逐个登记其 lease 与清理责任，不增加物理 payload 副本。
每处新状态须有唯一写入者、使用者和撤销点；测试复用既有 fixture 与参数化流程。
