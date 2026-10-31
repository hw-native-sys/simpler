# A3 单机 Host 通信域

L2 与 L3 共用 Buffer / Tensor / TaskArgs 前端和内部 HostDomainResource；不增加 HostTable、HostTarget 或第二套地址解析模型。Host 保持 HOST 身份，不占 NPU rank。A5、多机和自动选路不属于本分支。

公开签名、支持范围和兼容性见 [Python API 参考](../user/reference/python-api.md#memory)。不带 `host_window` 的 `create_buffer` 保留原有 POSIX SHM 语义。

Host Fabric 用例仅验收 A3；混合 `a2a3` runner 若明确报告为 A2（`Ascend910B*`）则跳过这些用例，A3 与模拟器仍执行完整断言。查询失败、未知 SoC 和 A3 上的分配错误仍失败，不视为不支持而静默跳过。

## L3 多卡使用

```python
domain = orch.get_or_allocate_domain(
    name="host-table", members=[0, 1], window_size=65536,
    buffers=[CommBufferSpec("output", "float32", 16384, 65536)],
    host_window=HostWindowSpec(nbytes=65536, numa_id=0),
    retain_after_run=True,
)
orch.copy_to_host_domain(domain, data)
native = domain.host().tensor((16384,), DataType.FLOAT32)
mapped = domain.map_view(native, domain_rank=0, access=AccessMode.READ)
args = TaskArgs()
args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
# 沿用 orch.submit_next_level(..., worker=0)。
```

`members` 使用实际 chip worker 编号；`domain_rank` 是该列表中的位置。`copy_to_host_domain` / `copy_from_host_domain` 访问本地 Host backing，`copy_to_domain` / `copy_from_domain` 的 rank 只指 NPU。

## 直接 L2 使用

```python
buffer = worker.create_buffer(nbytes, host_window=HostWindowSpec(nbytes))
native = buffer.tensor(shape, dtype)
mapped = worker.map_view(native, target=worker.device_location, access=AccessMode.READ)
# 原有 TaskArgs / kernel 提交和完成路径。
worker.unmap_view(mapped)  # 在所有已接纳使用完成之后；保留 backing，可重新 map_view。
worker.release_buffer(buffer)  # 在所有使用完成之后。
```

须先初始化拥有设备 context 的 Worker 并加载支持通信的 runtime；借用 context 的 Host backing 创建会在分配前拒绝。内部复用单 rank communicator 和 64 B 逻辑 HBM 控制窗口；驱动按分配粒度对齐，它不是整表 staging。用户不创建 domain，也没有额外 Direct 公共包装。

映射 view 必须显式使用 `TensorTransfer.NONE`。直接 L2 支持同一调用中的 native/H2D 与 mapped/NONE 共存；L3 native Host 参数不能作为设备消费入口。

`Worker.unmap_view(mapped)` 和 `CommDomainHandle.unmap_view(mapped)` 撤销一个映射记录；重新对 native Tensor 调用 `map_view` 会签发新 ref，旧 Tensor 不会恢复有效。Tensor 的 144 B 布局及 byte 142 的 `mapping_ref` 参见 [Buffer ABI](../buffer-abi.md)。

## 映射、复制与资源职责

- Buffer 保存同一 backing 的 identity、位置、范围和权限；Tensor 选择几何与显式访问 view；TaskArgs 保留每次消费的方向和 transfer。
- 分配只确认物理 view，不预登记设备消费 grant。`map_view` 按请求返回新 Tensor，不修改 native Tensor 或 backing identity；实际目标端的 ImportRegistry 登记受限 grant，并通过 qualified adapter、execution environment、范围和存活检查完成解析。
- A3 backend 分配 Host VMM 物理 backing，owner 保留 CPU view；各 chip 经 Fabric PID 授权、导入和 NoUC 映射。`numa_id` 选择 Host 分配位置，Buffer 暴露逻辑范围。MTE 与 PTO SDMA 复用既有提交及完成机制，各自验收。
- 映射没有 H2D 复制语义。CPU 原生访问仅支持创建 endpoint，不能把 Host 引用转发给 Python 子任务；不提供异进程裸地址消费。

## 控制协议边界

单机 Host export 复用 `GlobalDomainDescriptor` 的 v2 native 布局和 profile 校验，Host 不进入 rank descriptor 表，也不单独升版该 native 布局。Python 与 native 必须来自同一构建；未来修改布局需成对迁移，此复用不代表已实现完整公共 Host attachment 协议。

Host Buffer 使用现有 `REMOTE_SIDECAR` 表达地址无关描述，CPU 原生绑定及各 NPU 映射均须显式登记并检查 identity、目标、范围、权限和存活状态；不能凭这个 backend tag 自动解析或获得访问权。不新增 BackendKind。

L2 CPU 读写使用创建端点的 `Buffer.base`，由调用者保证没有在途使用；L3 的 `copy_to_host_domain` 提供受控的写入入口。二者不是等价的同步保障，不为形式对称新增 API。初始化 PID 通过固定头中的显式回复字段返回，不依赖 rootinfo 路径长度或共享内存实际页大小。

`map_view(access=...)` 为本次请求签发受限记录。每个 backing 的 `mapping_ref` 从 1 到 255 单调分配，不编码 rank 或 access，不复用；耗尽时显式报错，需创建新 backing。ref 是不透明的端点本地记录索引，不是可转交给陌生端点的授权凭证。实际读写按子端已注册 callable signature 检查；TaskArgs tag 用于依赖图，不作为另一套 Host 读写权限权威。

mapped Tensor 不能通过 `_with_mapping_ref` 换成其他 ref；READ 记录不因此获得 WRITE 权限。native ref 0 保持原生视图，直接 L2 仍可显式使用 H2D，并与 mapped/NONE 共存。撤销后旧 ref 保持失效，重新映射取得新 ref。

`DEVICE_FABRIC_V2_PEER_IMPORT` 记录后端已确认的导入来源。地址无关 Host sidecar 必须先经授权和平台导入，再由实际目标端登记受限 grant；qualified adapter 只能解析已确认且属于当前 execution environment 的记录，不能从空 body 隐式导入或扩大权限。

`HOST_VMM_MAP` 仅标记 L2 创建进程内已确认的 CPU owner 绑定，不是 region planner 候选，不进入 GlobalDomainCommand，也没有新增 wire profile id。`host_domain` 模块及其映射控制是包内实现，不属于用户 API。

### 实现边界与复用

```text
HostWindowSpec → HostDomainResource → A3HostBacking → A3 native owner
                                      │ 分配 / export / PID 授权
                                      ↓
                 comm_attach_domain_host（A3 platform 导入 / NoUC 映射）
                                      ↓
               map_view → 目标 ImportRegistry 登记受限 grant
                                      ↓
             Buffer → Tensor 显式 view → TaskArgs → MTE / PTO SDMA
```

owner 的 NUMA/P2P 分配、Fabric/PID 和逐步清理策略位于 A3 platform，只编入 A3 `host_runtime.so`，直接使用该平台已链接的 CANN ACL API；不搬迁既有 HBM helper，仅使其接受重复 ACL 初始化，以支持两种 owner 共存。Buffer/Tensor、grant、HostDomainResource 和 owner 桥接保持平台无关，供 A5 复用；`_task_interface` 不编译 A3 owner。owner 不负责全局 finalize 或 device reset。

桥接复用 RuntimeModule，通过私有、同构建的 Host backing C 入口调用平台 owner；不增加独立 DSO、版本协商或公共模型。入口将异常转换为状态和错误信息；部分失败保留 opaque owner，确认释放后才销毁。owner 未释放时保留模块至进程退出，避免卸载仍被使用的实现。L3 fork 前仅保存 runtime 路径，owner 在创建进程首次使用时加载；CPU view 仍属于创建进程。A5 可在自己的 runtime 实现相同入口，当前 A5/sim 不编译或启用 A3 实现。

独立清理复用同构建的可选 C 入口 `comm_detach_domain_host(handle, allocation_id)`，与现有 attach 使用同一 runtime 边界。它只清理 Host view，保留 HBM/domain 和 CPU owner；不增加 DSO 或版本框架。映射记录及准入仍位于 common 层，A3 platform 只负责物理映射与分阶段清理。

现有 HBM VMM helper 使用 DEVICE/HBM 属性、不同的保留标志及导出方式，不能直接代替 Host NUMA/Fabric/PID 契约。Host 部分释放失败还必须保留原状态；不能套用失败后清空字段的 best-effort window 清理。传输、消费者完成、参数解析和使用保留则复用既有链路，不再增加一套 copy/completion 通道。

分配、staging、映射控制、提交和释放复用 Worker operation lease、设备控制入口和端点锁。L2 `map_view` / `unmap_view` 与 release 共享端点锁，release 在锁内重查 owner，成功后才移除 ledger/journal。L3 建立 view 不等于保留一次使用，提交时仍校验记录存活并登记使用；collective 控制与 L2 单端控制保留各自编排。各 chip view 先释放、owner 后释放，失败债务遵守下节规则。

## 顺序、释放与失败保留

Host staging 复用 operation lease、控制预留及编排顺序检查：回调内先 copy、后提交任务；回调外等待所有在飞 Run 完成。违反顺序直接拒绝，不隐式等待，也不先写入数据。

CPU 指针、NumPy 或 `ctypes` 裸写绕过这些检查。调用者必须保证所有 reader / writer 已完成，写入期间不并发提交新任务。生命周期保留不等于写入互斥。

`map_view` / `unmap_view` 拒绝在途使用，不隐式等待。L3 回调中的映射控制必须位于首次提交之前；回调外须没有在途 Run。`unmap_view` 先撤销目标记录，只有该目标最后一个 active record 退出后才调用 native detach。其他记录仍可继续使用同一物理 view；CPU owner 和 HBM/domain 均保留。

独立 detach 失败进入 CLOSING，旧记录保持撤销，新的 attach 被拒绝；重试完成剩余清理阶段，成功后才允许以新 ref 映射。整域释放一旦开始就永久关闭该 allocation 的 attach 准入，即使后续单独清理 Host view 成功也不恢复。

L3 domain release 在已接纳使用完成后释放各 chip view，再关闭 Host owner；L2 `release_buffer` 使用相同的 views-before-owner 原则。`Buffer.close()` 撤销新使用，不能替代 Worker 的物理清理。释放与提交使用既有锁和使用保留；未知完成、未确认 view 释放及部分失败都不能当成回收成功。

A3 native 清理失败保留 allocation；重复清理不重播 collective barrier，barrier 超时或 VMM 结果未知保持失败。Python L3 的整域失败结果仍是保守回放，不把本地清理重试当成全域已确认释放。

某 rank 的 Host 清理成功仅说明其导入 view / handle 已退出，不代表创建端的物理 backing 已释放。若 barrier 或其他窗口清理结果未确认，allocation 的失败记录及父侧 owner 可继续保留，不能由一个 view 的成功推断整域回收成功。

owner 清理可在 view 释放已确认后重试；未知 view 结果保守保留到进程退出。Worker.close 继续处理子进程 teardown 并报告未清理债务。不能在无释放证明时通过 owner 析构自动释放 backing。

Fabric PID 白名单来源为各 chip 进程实际身份；控制面按可信进程模型运行。已验证授权调用与正常访问，未授权 PID 的真机负向验收仍待补，不能宣称已验证恶意进程隔离。

## 验收范围

已有 A3 Fabric 真机证据包括直接 L2、双卡 MTE 双向、BF16 设备选行、在途释放、PTO SDMA 完成及消费者可见性，以及持久 8 MiB 表、64 KiB 输出的稀疏读取。首尾行、重复 ID、跨 Run 复用和 CPU 更新可见性由对应测试覆盖；历史通过结果不能替代后续重构的增量验收。

HBG 与 TMR 已通过相同的 A3 真机 MTE/映射矩阵：native/mapped 同调用、独立 unmap、旧引用拒绝、重新映射数值复验、双卡双向访问、READ 拒写、已接纳任务释放，以及跨 Run BF16 稀疏取行。HBG 使用自身编译入口及实际 host orchestration，并非只复用测试标记。PTO SDMA 双向与消费者完成依赖另由 TMR 验收。

持续回归保留 mapping 权限/context、unmap/关闭保留、MTE 与 SDMA 消费者数值闭环，以及 native detach/release 的代表性故障注入。扩展 owner 故障矩阵、私有 mailbox 重建和专用稀疏 SDMA 组合仅作独立验收，不属于本 PR 的持续回归；未授权 PID 负向与延迟隐藏性能仍待验收。

[#2657](https://github.com/hw-native-sys/simpler/issues/2657) 要求设备发起异步稀疏 fetch、独立计算继续推进，并在实际完成后放行消费者。当前提供 A3 资源、映射及消费基础；未采集重叠 trace 或测量延迟隐藏收益。A5 的同步 MTE 与跨 rank URMA 需要分别实施和验收。

并发预取时，请求槽退休不代表输出可复用：输出仍须保留到最后消费者完成。性能验收应记录 issue、数据可见、独立计算与消费者开始时间，并和同工作量串行执行比较。
