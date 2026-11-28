# A3 窗口、建链与生命周期

[返回设计入口](../rdma-a3-simpler.md)

## 1. alloc_domain 在哪里调用，是否需要修改

以本设计的 L3 SDK 路径为例，`Worker.init()` 准备执行环境和绑定到 Worker 的 Host Orchestrator。随后应用调用 `Worker.run(rdma_host_orchestration,args,config)`；`run()` 经 `submit()`、`_submit_locked()` 到 `_submit_l3_locked()`，后者在当前 run scope 内执行 `callable(self._orch,args,cfg)`。这个 callable 就是传入的 Python Host 编排函数 `rdma_host_orchestration()`。域创建/复用必须由该函数显式调用，不是 register/init 自动调用，也不在 AICPU 的薄层 `aicpu_orchestration_entry()` 或业务编排入口中执行。

应用 Host 回调调用 `allocate_domain()` / `get_or_allocate_domain()` 后进入：

```text
Orchestrator._domain
  → Worker._comm_domain
  → Worker._allocate_domain
  → Worker._ensure_comm_base（首次懒初始化，成功后缓存）
  → Worker._dispatch_control_domain(op="alloc")
  → chip 子进程 _handle_ctrl_alloc_domain
  → ChipWorker::comm_alloc_domain_windows
  → A3 platform::comm_alloc_domain_windows
  → 文件内 domain_alloc_via_rdma（拟新增薄层 helper）
  → PTO GetCommWorkspaceSize / BuildComm
      → SetupRdmaTransport → InitRdmaWorkspace → manager Init（PTO 已有）
```

上图是未命中缓存的创建分支。命中已有且配置一致的域时，`_comm_domain()` 登记本轮使用并直接返回 handle，不调用 `_allocate_domain()` 或重复建链。现有首次创建分支通过 `_ensure_comm_base()` 向各 chip 发送 `CTRL_COMM_INIT`，建立基础 HCCL communicator；这与随后 PTO BuildComm 的 RDMA MR/QP 准备是不同步骤。

当前 A3 的 native 函数拒绝非零 `workspace_mask`。本设计保持上层函数签名，修改此处增加 `mask == comm_workspace_bit(RDMA)` 的专用分支；零 mask 仍走原来的窗口路径，混合 RDMA+SDMA/URMA/CCU 请求明确拒绝。

RDMA 分支只分配本 rank 的 HBM 业务窗口，不导入 peer Fabric/IPC 映射。`CommContext.windowsIn/Out[peer]` 保存 peer 注册 MR 的 VA，PUT/GET helper 通过 PTO 使用这些地址。它们不是该 rank 的普通远端映射，不能用于同步 TLOAD/TSTORE/TNOTIFY。

域内 `rankId` 按 members 顺序密集编号。physical device id、Worker chip id 和 domain rank 是三种身份，Bootstrap 必须明确转换，不能取 `rank % device_count` 替代 Worker 真实设备选择。

## 2. 初始化及全 rank 发布

### 现有 DomainAllocation 的 RDMA 初始化分支

1. L3 parent 校验 members、DEVICE backing、TMR、RDMA-only、窗口 carve 和一致配置。在派发任何业务任务前完成同步域准备。
2. coordinator 按有序 members 分发相同 QP 数、端口请求和各自 local IP。自动请求以 0 编码；native 使用相同 run_token/allocation_id 提议候选并全 rank 投票。记录 owner 身份和绝对截止时间，保存端口独占锁，并在任何 BuildComm 开始前完成预留确认。
3. native 复用已有 `DomainAllocation`，追加 PTO Host context、外部 GM 分配、端口占用记录和状态字段，尽早放入已有 pending owner map。初始化使用文件内 `domain_alloc_via_rdma()`；不增加另一套 owner 类和 Prepare/Close API。析构不得替代有返回状态的清理，异常不能让已注册资源失去 owner。
4. `aclrtMalloc` 创建本地窗口，记录实际 owner/extent。注册和 `CommContext.winSize` 对外统一采用已协商业务 extent，不能各 rank 按不同物理 padding 生成不一致 MR。
5. 配置 PTO `CommConfig`：RDMA-only、`rdmaOptional=false`、显式 deviceId/dense rank/rankNum、symBytes、qpNum、basePort、localIp；设置 `HostExchange`。使用 RDMA-only 的 callback Bootstrap，无需为每个域再创建 HCCL communicator。
6. `GetCommWorkspaceSize()` 查询并分配三块独立外部存储：业务窗口 `rdmaSymmetric`、RDMA workspace、PTO 临时设备 context。业务窗口已经存在时将其绑定，不能重复申请一份未被应用使用的窗口。
7. `BuildComm()` 负责能力/地址协商、HCCP 服务/rdev lease、业务 MR 和 operand MR、socket 连接、AI QP、MR/QP 绑定、QP ready、GM metadata 及 peer 地址回填。
8. 复用 PTO 的配置/建链检查，在分配分支校验 `rdmaReport=Ready`、IN_DIE/v3 和 simpler 窗口/工作区 extent 等接入条件；不增加独立 validate API。读取 PTO 的 Host context，逐字段转换为 simpler `CommContext`，复用 `comm_domain_onboard::upload_context()` 上传镜像。
9. PTO 完成全 rank ready agreement 后，复用 Worker 既有分配请求 join/abort：所有 native context 准备成功后才发布 Python handle。任何 rank 失败都不发布半成品域，不增加另一套公开 prepare/commit 协议。

PTO `BuildComm()` 既有的可选后端降级不在本设计使用：RDMA-only 要么完整 ready，要么失败。已有 base HCCL 环境和 PTO HCCP 服务共享必须尊重其 owned/borrowed 标记；域关闭不得关闭由 HCCL 持有的服务。

### HostExchange 的协议

使用已有本机 allocation 控制会合机制实现 callback，首版不新增 MPI 依赖。会合键至少包含 `run_token/allocation_id/generation/stage/round`；文件采用写临时文件后原子发布，不允许把上次失败遗留文件当新回复。

保留一个私有 `domain_host_all_gather()` 适配函数，复用 marker 路径、原子发布和有界等待，增加记录汇集。PTO 的 HCCL all-gather 只有在 communicator 与 domain 的成员和顺序完全匹配时可用；本设计支持任意 members 子集，不借用可能更大的 base communicator，也不新建每域 HCCL communicator。普通 file barrier 不能传递初始化记录。

callback 接收 PTO 的固定长 Host 记录；外层保存版本、长度、rank 数、stage 和校验信息，按 dense rank 顺序返回。所有参与者必须调用相同序列，失败也要结束已派发的请求后再进入 abort。外层协议逐字段编码，不把 C++ padding 或 Host 指针当稳定网络格式。

PTO callback 的 payload 按不透明字节承载，字段解释由固定版本的 PTO 负责；该路径只用于同一 Host、同一构建 ABI 的参与者，不承诺将这些原始记录作为跨架构的稳定 wire ABI。外层 header 使用文本字段，payload 使用定长十六进制编码，避免二进制 NUL 截断。

PR 2155 的 `Connection::Exchange()` 另外负责通过 HCCP socket 交换业务 MR 的 base/size/rkey、rank 和 QP 数，并检查预先协商的 peer base。simpler 不重复发送 lkey，也不发明另一套 QP 私有数据格式。

### 端口租约

同一物理设备的多个 RDMA 域不能无条件同时监听 60032。parent 派发的 native 分配事务在建链前为所有参与设备取得同一个空闲 base port，租约归 run/allocation owner 所有。

自动模式先提议端口，再全 rank 预留/确认；任一失败则撤销本轮预留，并在**尚未开始 QP 建链**时选择新端口。显式端口冲突直接报错。当前实现使用每 UID/物理设备/端口的文件独占 flock，持有 fd 即代表进程内 owner；进程结束时内核释放锁。打开时检查 inode、文件所有者和链接数；不 unlink 活跃的锁文件。该约束覆盖同 UID 多 Worker/进程；不同用户及外部 HCCP 应用仍由 provider 检测实际冲突，不能只凭“Host TCP bind 成功”断言 HCCP 端口可用。

外部 HCCP 应用未参与租约时仍可能冲突，provider 失败必须按初始化失败清理，不自动在部分建链后盲目重试。租约在所有相关 QP/socket 成功销毁后释放；失败 owner 保留租约防止误复用。

PTO 的 `TryDestroyComm()` 保留可继续清理的 provider 状态；外部 GM 的 `aclrtFree()` 出错则不同：调用可能已产生释放副作用，simpler 将 owner 标记为 terminal cleanup failure，后续 close 只报告 retained，不自动再次 free 同一地址。诊断 owner 仍在原 map 中，析构不替代有返回状态的清理。

端口协调逻辑接入现有 `_allocate_domain()` 和 native RDMA 分支，占用 fd/实际端口保存在已有 owner，释放在既有清理路径完成。不新增 RdmaPortLease 类接口；当前 SDMA/PTO 没有提供可直接套用的多域端口分配器，仍需补这部分私有实现。现有 GenerationLease 使用共享锁保护 generation 文件回收，不能用来排斥同设备同端口的另一个 owner。

## 3. workspace 和窗口分别保存什么

| 资源 | 创建者 / 阶段 | 内容 / 用途 | 释放者 |
| ---- | ------------- | ----------- | ------ |
| 既有 Host `DomainAllocation` 的 RDMA 字段 | native allocator / RDMA 分支 | allocation 身份、配置、PTO owner、内存、端口、错误和保留状态 | 既有 release/abort 路径，失败保留 map entry |
| 本地业务窗口 | simpler A3 platform / RDMA 分支 | 应用的输入、输出及 CommBufferSpec 切片；被 HCCP 注册 | provider 脱离 MR 后 simpler 才 free |
| PTO RDMA workspace | simpler 分配，PTO 初始化 | 64 B RdmaInfo、每 peer/QP 的 Queue、MR 表、operand 页 | provider 销毁后 simpler free |
| NIC SQ/CQ 和 doorbell 信息 | HCCP / CreateQp | 真实 WQE/CQE ring、队列深度、地址和队列号 | HCCP/PTO 负责 QP 生命周期 |
| PTO 设备 context | simpler 分配，BuildComm 回填 | PTO 的窗口寻址格式，作为转换依据 | PTO 不再使用后 simpler free |
| simpler GM `CommContext` | simpler / 全 rank ready 前 | 当前 rank、窗口 extent、peer MR VA、workspaces[RDMA] | 所有引用结束后 simpler free |
| AIV UB scratch | built-in AIV / 每次执行 | WQE、MTE 和 session 暂存，公共预算 256 B | 本内核退出后可复用 |
| DeferredCompletionSlab | 现有 TMR / AIV dispatch | 本次请求的完成条件和错误；不是业务窗口 | 按现有 FIN/条件移交协议回收 |
| AICPU observer 状态 | runtime / 每个完成条件登记时 | 该条件独占的 cursor、已证明成功前缀、失败位置、owner epoch | 当前条件 retire 时清空；workspace 的寿命仍由 run lease 保护 |

每个 rank 有自己的一份 GM context/workspace，不是全体 rank 共用一份。远端 VA 可以不同；“symmetric”在这里要求逻辑布局一致，并不要求所有进程 VA 数值相同。

PTO 的 workspace v3 仍使用 `RdmaInfo` 头；IN_DIE 下 `sqPtr/scqPtr` 指向同一 Queue 数组，每项同时包含 SQ/CQ，不是 HNS1825 的两组独立结构。通知 operand 页是该 provider 的内部资源，本设计即使只暴露 PUT/GET，也按其 sizing/初始化合同分配，不能擅自缩小。

## 4. 多次使用与直接 simpler SDK

以下是调用顺序示意，`TaskArgs` 的 scalar 增加方法按当前 SDK 实际类型构建；这不是可复制即运行的示例程序：

```text
创建 L3 Worker + A3 chip children
通用编译/装配：KernelCompiler → CoreCallable.build → ChipCallable.build
Worker.register(callable)；Worker.init()

每次 Worker.run(rdma_host_orchestration, args, config):
  domain = orch.get_or_allocate_domain(
      相同 name/members/buffers/window_size,
      retain_after_run=True,
      workspace_engines={"rdma": RdmaConfig(...)})
  ctx = domain.device_context(rank, workspace_engines=("rdma",))
  args = 六个本轮 uint64（ctx, peer, local_offset, remote_offset, nbytes, qp）
  orch.submit_next_level(put_or_get_handle, args, deps配置, worker=rank)

所有 rank 业务阶段结束 → domain.release() → Worker.close()
```

窗口内容的保留还取决于现有 persistent/reset 策略；需要复用内容时明确关闭 persistent window reset，不能依赖 `None` 默认值。`get_or_allocate_domain` 命中不重新建链、清零或重新运行 initializer。不能对 retained handle 随意使用会在退出时 release 的 context manager。

每次 run 内调用 `device_context()`，使已有 `_use_domain()` 持有本轮 owner。仅缓存上次 run 的裸 GM 指针会绕过此引用追踪，不属于安全使用方式。SDK 是受信任的低层接口，检查非空不能使任意 GM 地址自动变成有效 context。

observer 的 `owner_epoch` 由当前 TaskId 派生，是私有 cursor 的任务归属标识，不是独立的跨 run 世代。每次登记和退休都重置 cursor；domain/workspace 的跨 run 存续由已有 run lease 保护。相同 owner 的队列计数跨 run 保留，但旧任务 token 不可跨 run 重放。域在 run lease 结束前不能释放，因此同一 run 内不得出现相同 VA 已被重分配为另一个域的 ABA。此机制不向 PTO header 的 reserved 字段塞 generation，也不扩展六字参数 ABI。

## 5. 结束、失败与保留

正常释放顺序：停止接收新请求 → 本地所有 AIV FIN 和 RDMA 条件完成 → 业务确认所有远端访问静默 → 全 rank release agreement → `TryDestroyComm()` → 释放 PTO/业务/context 外部 GM → 释放端口和 owner。已有 C ABI 的 collective release 可以承载会合，但不能用一次 barrier 证明调用方没有尚未提交的业务访问。

实现入口复用 `comm_release_domain_windows()` / `comm_abort_domain_windows()` → `release_domain_allocation()`；RDMA provider 与内存的清理加入既有 `release_domain_windows()` 分支。正常 release 的静默协议和 initialization abort 的 parent join 前置条件保留区别，底层清理机制共用，不新增 Close/AbortInitialization 方法。

当前 A3 清理路径存在“barrier 超时也释放本地状态”和 map erase 路径。新增 RDMA 分支必须改为 fail closed：不能在远端还可能访问已注册窗口时继续 free，也不能因清理函数返回错误就丢掉 owner。

| 状态 | 可否提交 | 允许的动作 |
| ---- | -------- | ---------- |
| PREPARING | 否 | 继续阶段协商，或所有准备请求终止后调用既有 comm_abort_domain_windows |
| READY | 是 | 受依赖和范围约束提交，或开始静默关闭 |
| QUIESCING | 否 | 等待已知任务和应用远端阶段，不重置队列 |
| INIT_RETAINED | 否 | 返回 INITIALIZATION_ABORT_REQUIRED；join 全部准备请求后按 owner 记录清理 |
| RUN_RETAINED / CLEANUP_UNCONFIRMED | 否 | 返回原错误并保留环境、窗口、QP 和端口；需明确恢复证明，普通 close 不盲目重试 |
| CLOSED | 否 | 资源已销毁；后续旧 handle 使用报错 |

PTO `Finalize()` 可在 Init 失败后仍持有资源；调用方只能依据其明确成功结果释放 caller-owned GM。正常运行未知完成与纯初始化残留不能混用同一种 abort。普通接口不宣称具备设备错误后的自动恢复能力；无法证明静默时保留 owner 并报告未确认回收，不执行设备 reset 或隐式重发 PUT/GET。
