# Host 接入通信域：A3 双机 ACL backend

在 [#2668](https://github.com/hw-native-sys/simpler/pull/2668) 的设备域上增加每节点一块真实 Host DRAM。
沿用现有 Direct MPI launcher、GlobalDomain 事务、Buffer/Tensor 和 run 使用追踪；不引入 HIXL。

## 范围

两节点各有一个 CPU、两个 NPU，以及一块 DRAM、两块 HBM。
两个本地和两个远端 NPU 都访问同一个 Host allocation；Host 不占 NPU rank。

- CPU 读写本机 DRAM：owner CPU VA 上的 memcpy，字节操作返回时完成。
- CPU 访问其他 DRAM/HBM：本机参与 NPU 的 ACL SDMA，copy stream 同步成功后返回。
- NPU kernel 读写任意 DRAM：显式映射 Tensor，经 kernel/MTE 访问，所属 run 完成后可交接。
- NPU kernel 访问 HBM：真实目标 Buffer 的显式映射 Tensor，所属 run 完成后可交接。

不交付跨机 CPU 原生 load/store、A5 HostRegister、URMA、capture、自动故障恢复或通用 Buffer COPY。
MPI 负责启动与控制；ACL 负责内存和数据传输。

## 建域与映射

L4 在原 orchestration callback 中创建或复用域：

```python
domain = orch.get_or_allocate_domain(
    name="mixed",
    members=[(a, 0), (a, 1), (b, 0), (b, 1)],
    window_size=N,
    buffers=[CommBufferSpec("payload", "uint8", N, N)],
    host_windows={a: HostWindowSpec(0, N), b: HostWindowSpec(0, N)},
    retain_after_run=True,
)
ref = domain.ref
```

目标 L3 通过原 `orch.get_domain(ref)` 借用域，再选择本机 attachment：

```python
domain = orch.get_domain(ref)
host_tensor = domain.host(a).tensor([N], DataType.UINT8)
device_view = domain.map_view(host_tensor, domain_rank=rank)
args.add_tensor(device_view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
```

Host Buffer 的物理位置仍为 HOST；HBM 也保留其原身份。原 Tensor 与映射 Tensor 共享 backing identity，映射不复制数据。
映射必须显式选择 NONE；省略 transfer 保留原 HOST/H2D 默认，因而被映射入口拒绝。
映射只在持有对应 `local_context` 的 L3 创建；不能跨 Remote 或 nested Worker 转发。
每个 chip 在 native 导入成功后，将实际 Device VA 注册到同一 ImportRegistry，复用既有 adapter 能力表收窄权限；
解析按 `(identity, mapping_ref)` 精确命中，不回退普通 Host binding。
`mapping_ref` 是公共 Tensor 的不透明 attachment 引用；rank 编码只由域 backend 解释，
不代表本 PR 已交付 direct L2/A5 映射或通用 Buffer COPY。

## Backend 边界

L3 Host owner 通过 CANN `HOST_NUMA/P2P_HUGE` 分配物理内存，建立 CPU VA 并导出 Fabric handle。
L2 将 Host export 加在原 HBM descriptor 表之后，复用原 peer view 容器导入。
`[0,R)` 为原 HBM rank，`[R,R+H)` 为按 node 排序的 Host pool，不修改 CommContext ABI。
含 Host 的域复用逐节点 MPI 控制交换；无 Host 域保留原 collective 快路径。

CPU、SDMA 与 kernel 参数共用 `ImportRegistry.require` 的 source descriptor、权限、范围及存活校验。
L3 保存仅供委托复制使用的 binding，不提供可解引用 VA；L2 保存实际已导入的 Device VA。
L2 direct 与 chip child 继续共用原 materializer，普通 Buffer 路径保持原约束。

- `_host_domain_bytes`：校验精确本机 Host descriptor，在原 provenance 锁内读写 owner。
- `_copy_domain_buffers`：接收域内 source/destination Buffer 与相对范围；L2 再解析同一资源，native 只接收已解析 VA。
  操作限于本机 Host backing 与已声明的域 backing，stream 同步成功才返回。

Host 域的 HBM identity 由所属 L3 在 PREPARE 确认本地 backing 后创建，
暂存在原 member.buffers 中；此时不登记执行授权或发布远端 locator。
L4 复用原 Buffer 引用校验，首次 IMPORT 即携带完整来源表；L3 沿用这些 identity
构造原 VMM Buffer，L2 一次安装全部附件。COMMIT 核对同一来源表及 owner 登记，
再发布远端 locator；所有节点确认后才发布 L4 handle。重试必须保持完整表一致。
本地 IMPORT 临时复用原 command codec，L2 不保存完整 command，也不另建 grant 表或生命周期状态机。
部分安装失败立即撤销使用资格，保留映射与 owner，交由原分阶段流程清理。
Host 访问的 PayloadPart 适配、初始化/校验、CPU actor 编排和 MPI 验收矩阵位于测试模块；
backend 不依赖测试代码。

当前 A3 成功路径沿用 NoUC Device views，使用 CANN 要求的显式 VA 与 1 GiB 预约粒度。
活动预约从现有 view 推导，确认释放后可回收；外部地址冲突明确失败。
全部 HBM 是否也必须使用 NoUC 尚缺单因素对照，不能由成功矩阵反推必要性。

## 使用与回收

L4 每次提交后代前在自己的 callback 中调用 `get_domain(ref)`，沿用原 run fence 保活；
仅在 L3 获取 borrowed handle 不替代 L4 的使用登记。Host 字节操作和同步 copy 使用原控制准入，
CPU 写完后才能启动设备读取，全部设备访问完成后才能修改共享 Host 数据。
映射成功不代表数据已就绪，也不是一致性或完成 fence。

回收沿用原域流程：撤销新使用、等待 accepted work，收齐所有节点 VIEWS 确认，再释放 OWNERS。
部分导入、copy/执行完成未知或 VIEWS 失败时保留资源和上下文，不能继续 reset/free owner。
域 Buffer 是借用引用，不能用 `release_buffer` 绕过全域回收。
不承诺失联后恢复或未知结果自动转成功。

## 验收代码

`tests/st/worker/comm_domain/host_domain_matrix.py` 提供单机两卡与双机四卡同一矩阵，
分别验证 CPU memcpy、ACL SDMA、kernel/MTE，检查完整窗口、guard、同址交接与释放。
`examples/workers/l4/host_domain_mpi_direct/main.py` 保留兼容启动入口。
现场命令、日志和测试结论放本地 reports，不作为接口文档的一部分。
