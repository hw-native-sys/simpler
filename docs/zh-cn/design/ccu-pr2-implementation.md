# CCU PR2：V7 设备调用与完成跟踪

本实现按 [CCU PR2 开发需求](ccu-pr2-requirements.md) 的 V7 接口开发。当前提供设备调用、共享协议、TMR 完成跟踪和 Host 资源释放门槛；**尚不代表可在 A5 上直接执行 CCU**。PR1 的 V7 服务端和 PR3 的真实实例发布、prepare 绑定仍是集成依赖。旧 PR1 的模板 key / signature 协议不能直接替换本协议。

## 设备入口

包含 `backend/ccu/ccu_completion_kernel.h`，使用以下接口：

```cpp
auto task = get_async_ctx(args);
CcuCallArgs parameters;
parameters.add_tensor(src);
parameters.add_tensor(dst);
parameters.add_tensor(scratch);
parameters.add_scalar(uint64_t(count));
auto call = CcuCall(domain_ctx, 0, parameters);
if (!send_request_entry(task, call)) return;
```

同一 AIV 随后需要读结果时使用 `wait_request_entry`。两者都提交新请求，不能把 wait 当成已有 send 的等待操作。构造 `call` 只复制描述，不读写 Tensor 数据、占槽或提交。

首版支持域内连续、非空 FP32 Tensor 和合法子视图；每次最多 16 个 Tensor、16 个标量。参数按 Tensor 组、标量组排列，各组保持插入顺序，允许交错调用 add。标量支持 32 位或 64 位的 C++ 整数与浮点类型，记录种类、位宽和位模式；32 位值放在低 32 位，高 32 位清零，有符号解释由种类和位宽确定。指针类型在编译时拒绝。显式把指针转成整数仍属于违反接口约定，范围检查不是隔离恶意 builder 的安全边界。

容器容量、FP32 表示、连续性、整数溢出、buffer 边界和域已准备范围由 runtime 检查。数量、标量语义、动态 count、别名及算法同步由 builder / 调用方保证，不另设 signature 或读写角色。

设备上的 `AsyncCtxStorage` 保持 40 字节，保留 URMA sender 身份；CCU callable 身份放在 `GlobalContext` 原有的 4 字节 padding 中。`LocalContext` 保持 48 字节，dispatch payload 保持 512 字节和首个 64 字节控制行。scheduler 每次派发注入当前 native callable 身份，已有发布屏障及 AICore 全缓存失效覆盖该字段；`get_async_ctx` 将身份复制到本地 48 字节 `AsyncCtx`，并对不符合单逻辑块约定的任务拒绝 CCU 身份。用户无须传入额外 Host 参数。

## PR1 / PR3 共享发布契约

共享 POD 位于 `src/common/platform_comm/ccu/ccu_device_protocol.h`，版本为 2，与 Host 模块 ABI 独立。旧版本目录在访问实例前被拒绝。

1. `CommContext.workspaces[CCU]` 指向 `DomainDirectory`。CCU 为槽 3，`CommContext` 大小变为 1104 字节；所有生产者与消费者需要一起重编译。
2. directory 通过 native callable ID 查找独立 instance image。image 内按顺序 `ccu_id` 找 `ProgramBinding`。真实后端通过 `platform_context/platform_context_bytes` 引用 PR1 会话，核对 owner、domain、generation、callable、rank 与函数数量；请求参数和完成记录由 PR1 持有。两个地址字段都为零时使用可控测试后端的 image 内 Request。
3. 除独立 image、平台会话、doorbell 和已注册 Tensor 的 device 地址外，结构内部引用均使用相对 image 的 offset。`platform_program` 是平台描述信息，不能被 Host 擅自解释为 doorbell。
4. Host 在关闭 admission 且该 domain 全部运行排空后才可替换目录；新注册函数不得搬移已有 instance image。目录、image、domain context 地址和注册身份在旧引用存活期间不得复用。新 domain 使用新 generation 和 context 地址。
5. 每次接受将参数复制到运行时持有的存储，并增加不回绕的 64 位 sequence。真实后端把位置参数交给 PR1 再次验证域窗口并填入凭据；完成记录与提交信息分属独立 128 字节 cache line。TMR 的完成 token 只保存 instance、binding 与 sequence，不引用 AIV 栈。
6. PR1 必须每轮重新读取 binding 和参数、执行函数并完成算法同步。只有所有读写停止且结果可见后，才能发布 SUCCEEDED 或 FAILED；先写 generation、binding、状态，**最后发布 sequence**。旧 sequence、错误 generation 或错误 binding 不满足新请求。
7. 真实后端由 PR1 的 submit 操作触发其已准备的 doorbell；可控后端使用 `ProgramBinding.doorbell_addr/value`。不能把任意 MMIO 地址代入。A5 cache / barrier 与 doorbell 约定仍须由 CANN 9.3 真机验证。

每个实例的所有函数共用一个固定请求槽，不同实例有各自的槽，运行中不分配内存。调用者用排序 Tensor 串行化同一实例，保证每 rank 一个逻辑 AIV 块提交。跨 rank 的调用顺序与次数必须一致。

## 接受、完成和失败

send 在触发前完成：容量检查 → reserve → 参数快照 → 登记并发布完成条件 → submit。登记失败取消 RESERVED 槽，不留下已触发请求。相同本地描述再次接受会产生新 sequence；对同一个底层请求再次 submit 或 retire 会被拒绝。

wait 在当前 AIV 轮询本轮结果，处理数据可见性并回收槽后才成功。send 将完成条件交给 TMR；TMR 只有在本轮成功、槽已回收以及普通 AIV 完成都成立时，才调用任务完成逻辑。已确认停止的 FAILED 可回收，但不解除消费者依赖；超时或 UNCONFIRMED 保留活动槽并关闭后续 admission。

调度器报错不等于硬件停止。包括 AIV 在 send 后报错、没有发布 FIN、或完成 slab 被错误截断的情况，都需要 Host 兜底。因此增加 Host-only `ccu_run_lease.h`：

- PR3 在 prepare 阶段将真实 `RunLease` 赋给 `PreparedExecution.ccu_readers.lease`。它必须持有本次用到的实例、模块、域和窗口，并排他保护实例。null 仅表示普通、未声明 CCU 的 callable。
- Host 在普通执行流结束之后调用 `poll()` 或 `drain()`，分别用于非阻塞观察和有界停止。STOPPED 是全部 CCU 读写者均已停止的证明，与成功/失败错误码分开。
- onboard 的 `poll_run_fence`、`wait_run_fence`、`wait_execution_completion` 都检查这一证明；A5 `cleanup_execution` 也拒绝释放尚未确认停止的 CCU run。此时保留整个调用载体及内存，并禁止 runner 继续接收任务。
- 未安装 PR1 provider 和 prepare 绑定时，PR3 必须在 Python 与 native 注册入口拒绝 CCU 执行，不能依靠空 lease 绕过检查。

## 验证范围与尚缺证据

CPU 用例包含参数快照、合法子视图、越界/类型/容量拒绝、callable 隔离、旧完成信号、重复提交、超时保留、正常与失败回收，以及 TMR 不提前释放任务。假后端的 1024 轮测试每轮改变区域、长度和输入并检查输出保护区；它验证 runtime 协议，不证明 CCU 服务端或跨 rank 数据路径。

Host 用例检查持有未完成 lease 时不发布物理完成，以及 STOPPED 错误不会变成成功。还需要在实际 PR1 / PR3 集成后验证：真实 prepare 挂载、域关闭、模块卸载、异常退出后的全资源保留、AIV 编译和 A5 双卡 Broadcast / 用户 Copy。

硬件验收须按仓库流程经 `task-submit` 执行：CANN 9.3、双卡 UB_CTP、两个 Broadcast root、64 / 1024 个 FP32 元素；单个 L2 运行时决定至少 1000 轮，每轮改变数据/区域/长度，轮间没有 Host prepare、拷回或流同步。当前未完成这些项目。

## 本轮验证记录

2026-11-06，Linux aarch64 / GCC 10.3：11 个 C++ 测试程序全部通过，覆盖 CCU 的 12 个用例、Host RunLease、通信布局、实际 dispatch、scheduler retirement、SDMA/URMA/RDMA 完成跟踪，以及 A5/A2A3 的 native admission。完整回归代码快照为 `88e4a5f4cbdb413d9ce5017726d7ced2b9cdf2fc`。随后仅调整 CCU/URMA 头文件的设备编译包含顺序，在快照 `0d3e517dc05b7be413dccf76f6ead013bf211766` 上重编译、重跑受影响的两组用例，均通过。快照为本地验证提交，未发布 PR。

Python `tests/ut/py/test_comm_workspaces.py`：34 passed。Windows GCC 的 C++17 严格语法检查通过，裸指针 scalar 的编译拒绝检查通过；新增 C++ 文件格式检查与 6 项相关 lint 通过。中文说明位于仓库允许的 `docs/zh-cn/`；用户提供的原始中文需求文件保持原样，不纳入新增源码的英文检查。

复现 C++ 用例可先按 `docs/testing.md` 配置 `tests/ut/cpp/build`，构建下列 targets，再运行同名 ctest：

```text
test_a5_tmr_ccu_completion
test_a5_tmr_urma_completion_kernel
test_a5_tmr_urma_completion_scheduler
test_a5_tmr_rdma_completion_scheduler
test_a5_tmr_scheduler_state
test_a5_tmr_mix_spare_lane
test_a5_tmr_scheduler_retirement
test_ccu_run_lease
test_comm_workspaces
test_a5_tmr_native_admission
test_a2a3_tmr_native_admission
```
