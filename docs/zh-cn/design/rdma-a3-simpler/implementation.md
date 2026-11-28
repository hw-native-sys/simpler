# A3 RDMA 实施与验收

[返回设计入口](../rdma-a3-simpler.md)

## 1. 源文件与工作拆分

| 顺序 | 增改位置 | 完成判据 |
| ---- | -------- | -------- |
| 1 | PTO pin、下载/安装与 package headers 清单 | 固定包含 PR 2155 的 commit；source 和 wheel 都能找到公开 Host API、Host-safe 队列布局及 AIV intrinsic |
| 2 | `python/simpler/workspace_config.py`、worker control 编解码、`comm.h`、native binding | RDMA 配置、84 B Host 配置结构、端口请求/实际值一致；旧接口调用仍得到正确默认配置 |
| 3 | 现有 `src/a2a3/platform/onboard/host/comm_hccl.cpp` 的 DomainAllocation、分配/释放/abort 分支 | 仅增加私有 domain_alloc_via_rdma 和 HostExchange 适配；复用 PTO BuildComm/TryDestroyComm、context 上传、Worker join/abort；补端口协调和清理保留，不新建 owner 生命周期 API |
| 4 | `src/common/rdma/rdma_request.h` 与 A3 completion adapters | 48 B ABI、提交前容量检查、复用现有完成登记、IN_DIE 事件检查、只读 observer、有界进度和错误传播 |
| 5 | A3 TMR `async_wait.h`、mailbox type 与 completion owner | type4/ROCE 分发、FIN 与 RDMA 合并，未改变 SDMA/counter 语义 |
| 6 | `builtins/rdma/`、kernel/runtime compiler、RDMA 示例声明/私有 helper | 复用通用编译、CoreCallable/ChipCallable 和既有 AICPU 入口契约；独立 callable 仅提供默认符号的薄层源码，业务可在已有入口直接提交；注册一次可多次运行，不新增公共 RDMA builder；source/wheel 行为一致 |
| 7 | `tests/ut/py`、`tests/ut/cpp/a2a3`、`examples/a2a3/.../rdma_async_completion_demo` | 以下各层验收分开执行并留证 |
| 8 | `docs/comm-domain.md`、`docs/developer-guide.md`、`docs/hardware/cache-coherency.md`、A3 API/场景文档 | 更新 A3 原先“workspace_mask 必须为0”和 CQE unsupported 等支持表；只设计 PUT/GET，不引入本次范围外接口 |

上述文件已加入本实现；后续章节分别记录构建、模拟、A3 实机冒烟结果和仍需完成的设备验收，不把这些结果混为一项。

## 2. 工具链与构建接入

基线要求是可编译 `__NPU_ARCH__=2201` 的 CANN/Bisheng、匹配 A3 的 driver/firmware、匹配运行时的 HCCP 头文件和 `libra.so/libtsdclient.so`。PR 2155 明确要求 `hccp.h/hccp_common.h`，还使用 `infiniband/verbs.h`、driver 和 tsd 声明；不能自行镜像私有结构来消除编译错误。

SDK 包缺 HCCP 头文件时，使用与实际 HCOMM/HCCP runtime 对应的源码 headers，通过 PTO 已有 `PTO_HCCP_INCLUDE_DIR` 指定。文档不把一个未核对的 CANN 最低小版本写成已验证兼容表；实施应记录精确 Toolkit/HCOMM/HCCP/driver/firmware 版本、SoC 和 PTO commit。

复用既有 `SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON`；A3 产物使用 PTO 已定义的 `PTO_NPU_ARCH_A2A3`、`PTO_RDMA_SUPPORTED` 和 `PTO_RDMA_BACKEND_IN_DIE_SUPPORTED`，A5 继续选择 HNS1825。PTO 明确不允许同一编译单元同时选择两个 RDMA backend。shared `a2a3` 目录不等于所有 A2 变体都支持，Host `IsSupportedSoc()` 与真实 device 选择都需要生效。

构建产物分三类：Host provider 代码、AICPU runtime、AIV builtins。固定 PTO 的高层 Host domain header 带设备 tensor 声明，普通 g++ 无法独立解析；开启 RDMA 时，CMake 用 CANN `ccec -xcce --cce-host-only -std=c++17 -fPIC` 编译 `comm_hccl.cpp` 为普通 Host object，再链接进 Host runtime。其他 Host 源文件仍用普通工具链。AICPU 只读取 Host-safe 数据结构和设备内存，不能把 `in_die_backend.hpp` 的 AIV intrinsic 编译进 AICPU，不复制依赖布局。

构建缓存和 wheel 索引必须含 platform/runtime/backend/PTO commit/ABI 版本；从源码测试成功不能证明 wheel 内置模板和头文件齐全。非 RDMA 构建及原 A5 默认配置应有回归检查，新增字段不能让旧路径读到随机值。

## 3. 不需要设备的验证

| 类别 | 必测点 |
| ---- | ------ |
| ABI | 六个 uint64 的大小、offset、64 位 round-trip；Host 配置长度 84 B；CommContext 保持原布局；IN_DIE/PTO kind 与 simpler type 不混用 |
| 参数 | 溢出、越界、零长度、超长、未对齐、peer=self、非法 QP、高字截断、错误 workspace/backend/version |
| 调用装配 | PUT/GET 使用相同请求合同；普通 register/init/run/submit；产物缓存不固化本轮数据；默认/自定义 AICPU 入口均可装配；独立薄层拒绝非零 tensor、非六 scalar，并返回失败而不是空图成功 |
| 域复用 | 相同请求命中；members 顺序、QP、IP、端口请求不同拒绝复用；自动实际端口不导致复用 miss；命中不清零 |
| Bootstrap | 某 rank preflight/分配/建链失败、阶段号不一致、过期文件、callback 异常、超时、清理残留；所有分配请求 join 后才 abort；members 子集/重排不误调用更大的 base HCCL collective |
| 端口 | 同设备并行域冲突、显式端口占用、租约清理失败保留、多进程 owner 存活身份 |
| 完成登记 | slab 满时 provider post 次数必须为0；剩余一条时能提交并经原函数登记；单 writer 合同成立、QP 并发任务不共享 slab；无 event.Wait 兜底 |
| CQ 查询 | owner0/1、status 错误、target head 非零、多个 QP、非目标此前错误、有界32项扫描、不可达地址拒绝 |
| 并发 CQ | AIV 回收与 AICPU 读取交错；tail 重读；CQ 绕圈覆盖；observer cursor 缓存失败；旧 run token 重放；不向任何 PTO tail/doorbell 写值 |
| FIN 汇合 | 人工 gate 分别控制 FIN 和 CQE；任一个未到时 TaskId 不完成，二者皆到才释放后继 |
| 生命周期 | 清理失败仍保留 MR/窗口/端口/owner；超时不等于静默；初始化 abort 不能取消已发 DMA |

采用 fake provider 和可控制的队列快照验证状态机；这些单测只证明算法与协议，不证明 HCCP 硬件可用。并发覆盖通过显式事件/gate/注入点建立顺序，不能用 sleep 或“足够长时间没出错”代替断言。

编译检查应覆盖 Host+AICPU+AIV，链接检查使用真实 SDK 库，sim 明确标识未执行 RoCE。完成测试后只按实际执行结果报告，不能把源代码 grep 或 syntax-only 检查称作完整编译通过。

### 当前无设备验证记录

2026-11-27 的构建使用 Ubuntu 24.04、CANN `9.3.0-beta.1` 内的 CCEC 和 AArch64 工具链，Host 普通源使用 GCC 13.3。PTO 固定为 `596dc095892da42f1990622830c5209309d025cf`。SDK 分发包之外的声明来自官方 HCOMM 9.3.0 `50792bd5ed191699a544c7e71b4426bbb2be7c71`、HCCL 9.3.0 `7b8e1ac5565263efc43fa7d45ce36549b3db52dd`，以及 Ubuntu libibverbs `50.0-2ubuntu0.2`。这些是已使用的编译输入，不是未经实机核对的 SDK/driver/firmware 兼容承诺。

| 检查 | 结果 |
| ---- | ---- |
| 实际 SDK 构建 | A3 TMR 的 Host、AICPU、AIV 编译链接成功；PUT/GET 内核及标准 AICPU 编排产物生成成功 |
| 安装资产 | 最终 non-editable wheel 在隔离目录安装；实际从 wheel 导入编译器模块，12 项关键 native 资产和编译器模块与最终源码逐字节一致；编译 PUT/GET 和编排并完成 CoreCallable/ChipCallable 装配，未启动设备 |
| 相关 Python 回归 | 336 项通过；最终配置/control 子集 51 项通过，不能将有重叠的数量累加为总用例数 |
| 编译缓存 | 最终编译器子集 27 项通过，覆盖 RDMA 开关、A3/A5 后端宏与 PTO pin 改变时的缓存隔离 |
| 通信域和 Host Worker | 最终回归 482 项通过，2 项按已有条件跳过 |
| Host 资源协议 | 7 种 RDMA 场景通过：成功、初始化残留、provider 清理失败、两类 GM free 产生副作用后失败、端口冲突、双 rank HostExchange；另有原 Host lifecycle 30 项通过。均为明确替换外部 provider 的无设备测试 |
| C++ 完成与生命周期 | 8 个聚焦测试可执行文件通过，覆盖请求 ABI、CQ owner/wrap/status/tail 交错、有界扫描、FIN/CQ 两种到达顺序、取消排空、A5 完成回归及 native 配置 ABI |
| 完整模拟场景覆盖 | 全部 94 个资源场景子进程退出成功；整场命令在后续 L2 阶段触发 1200 秒总时限，因此不能宣称该单次命令通过。按运行时拆分后，HBG 为 18 通过、8 原有条件跳过、1 个 BF16 失败；TMR 为 33 通过、2 原有条件跳过、2 个 BF16 失败。以下环境修正后，这 3 个失败用例分别重跑通过；这些分阶段数量不作为无重复总数累加 |
| 提交检查 | 原仓 pre-commit 的格式、clang-tidy、lint、文档和类型检查通过 |
| A3 NIC 实机 | 已完成下节的同 Host 双 rank 冒烟；CQ 绕圈、并发和故障等完整验收仍未完成 |

BF16 模拟需要原生类型，PTO 的 `include/pto/common/type.hpp` 在不支持的工具链上将 `bfloat16_t` 占位为 `_Float16`。本机最初按 CI 的后备做法将 GCC 13 用作 `g++-15` 别名，造成三个 BF16 注意力用例数值不符。安装 Ubuntu 官方 GCC `14.2.0-4ubuntu2~24.04.1` 后，验证 BF16 与 half 类型不同且 `1.0` 的位表示为 `0x3f80`；HBG paged attention 重跑 1 项通过，TMR paged attention 和 batch paged attention 重跑 2 项通过。测试入口仍是仓库现有 `g++-15` 名称，实际版本为 GCC 14，未安装或声称使用 GCC 15。CANN Host 源在安装后也重新编译通过。没有放宽 golden 容差或修改算法。

模拟覆盖使用同一份最终源码、原有 markers 和 golden 检查。整场与拆分命令均采用 CI 的 `--pto-session-timeout 1200 --require-pto-isa --manual exclude`；拆分只选择 `--level 2` 和对应 runtime，保留完整 L2 范围。上述模拟结果没有执行 IN_DIE 网卡数据路径。

### A3 目标设备冒烟记录

2026-11-27，在两张空闲的 Ascend910_9362 上，以 CANN `9.3.0-beta.1`、driver `25.5.1` 和同一 PTO pin 重新完成 ARM64 Host、AICPU、AIV 构建及 Python 绑定、PUT/GET callable 装配。服务器默认 CANN `8.5.2` 缺少当前基线需要的接口，测试使用独立的 9.3 SDK，未升级驱动或修改默认 CANN 链接。

原始示例发现两处 SDK 使用错误：`Transfer` 不能作为 `Worker.run(args=...)` 的 `TaskArgs`，窗口拷贝也必须覆盖已声明的 buffer slices。示例使用回调捕获请求，并声明覆盖整个窗口的 `CommBufferSpec` 后，两个先失败的回归测试通过；连同 RDMA 配置和替换 provider 的资源协议测试，最终为 30 项通过。这些 Python 修复不改变 RDMA 后端本体。

设备上完成原示例四轮 PUT/GET，确认同一 domain/context 复用、动态参数和双 rank 数据校验。另用独立验收驱动覆盖 `PUT/GET × 256 B/4 KiB/64 KiB × QP 0/1` 共 12 个组合；每轮检查完整窗口，源区、目标区和其余区域均符合预期。HCCL Bootstrap 在多网卡服务器上需要通过已有 `HCCL_SOCKET_IFNAME` 选择链路正常的 Host 网口；数据仍通过 NPU RoCE。

测试前后两张设备的进程查询均为空、健康状态为 OK，成功测试日志中没有强制 reset 或错误。环境没有 `task-submit`，使用即时占用检查和本地 advisory lock，不能据此声称获得共享队列隔离。以上仅证明该设备/SDK/驱动组合的同 Host 双 rank 冒烟；没有覆盖下面验收中的 CQ 绕圈、并发 producer、FIN/CQ 人工顺序控制、故障注入或跨 Host 场景。

2026-11-28，对齐 upstream main `9f9beac34c7b62efa0abafab7eb9ad0dea3d6dcf` 后，在同一设备/SDK/驱动组合上重新构建 ARM64 Host、AICPU、AIV 和 Python 绑定，并装配 PUT/GET callable。原示例四轮及上述 12 个 payload/QP 组合再次通过，确认新 epoch retirement 基线下的完成与域复用。相关 Python 回归为 487 通过、4 个条件跳过；SDK 资源协议测试按上游新分层单独运行，7 项通过。8 个聚焦 C++ 测试与全量改动文件的 pre-commit 检查通过。这些是重对齐后的验证，不与前面的历史计数累加；剩余实机验收范围保持上述界限。

实际 SDK 构建入口如下，外部头文件目录须指向部署环境对应版本：

```bash
source /path/to/ascend-toolkit/set_env.sh
export PTO_HCCP_INCLUDE_DIR=/path/to/hcomm/src/base_comm/resources/hccp/inc/network
export CMAKE_PREFIX_PATH=/path/to/hccl-dev-prefix:/path/to/verbs-dev-prefix
export SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON
python simpler_setup/build_runtimes.py --platforms a2a3 --runtime tensormap_and_ringbuffer
```

直接 SDK 的 PUT/GET 编译装配和运行入口见 [RDMA 示例](../../../../examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/README.md)。模拟场景验证公共运行时行为；它没有模拟 IN_DIE 网卡，不能替代本节以下的实机验收。

## 4. A3 双 rank 实机验收

1. 在真实 A3 的同 Host 两个设备建立域。核对 SoC、logical→physical id、RoCE IP、业务 MR base/size、QP ready，确保未误选普通 Host NIC。
2. PUT/GET 分别覆盖至少 256 B、4 KiB、64 KiB 和较大 payload。源/目标两侧放 canary，按 rank/序列填不同内容；验证目标字节及邻接区域。
3. 一个 callable、一个 domain 连续多轮复用；交替 peer/offset/长度并验证数据确实随参数改变。复用期间 MR/QP/端口不重新创建，业务内容按显式 reset 策略保留。
4. 多 AIV task 发往同 peer/QP、多 QP 分流，累计超过实际 CQ 深度数倍，以确认 PTO 回收和只读 observer 不因 owner 绕圈误判。记录实际 HCCP 深度，不能用源码请求8192当作设备一定返回的值。
5. 用可观察阶段 gate 确认 FIN-before-CQE、CQE-before-FIN、CQE cache 刷新和 GET 后继读取；只有传输与正常执行都完成，后继才能消费。
6. 在 SQ 未满前提交少量请求，确认 observer 在 PTO tail 尚未推进时可读真实 CQE；再提交足够请求确认 AIV Reserve/Progress 回收仍工作。
7. 确认 AICPU 能访问真实 CQ buffer 和 software tail；验证 CQ owner/status 与固定 PR 源码一致。若不成立，停止集成并修正 provider 完成接口，不编造成功路径。
8. 正常关闭再创建、显式端口冲突、链路/peer 失败和超时分别验收。故障注入发生在可观察提交阶段，校验 live owner 不被提前 free；这类用例需要受控设备，不在无设备环境假装执行。

首版支持范围可以在双 rank验收后扩展到更多 rank，但 `(rankNum-1)*qp_num` 的设备资源预算、all-rank 失败协商和全互连建链必须一并验证。完成本地 PUT 不等于远端消费，跨 rank 测试要用显式业务握手证明覆盖许可。

## 5. 设计评审关口

落地代码需同时满足：依赖版本固定、HCCP 真实建链、原 API 的 A3 alloc/release 接入、48 B ABI、提交前容量保障并复用登记函数、无阻塞完成兜底、只读 CQ observer、FIN 合并、错误保留、wheel 可用和直接 SDK 场景。只有 builtin kernel 或只有 Host 资源接口都不算完成通用后端。

依赖源码已核对，但以下属于实施验证关口：HCCP 与具体 SDK/驱动匹配、AICPU 对 NIC CQ/tail 的可访问性与一致性、PTO 分配/回收在实际 A3 上的时序、GET 后继 cache 可见性、资源清理失败语义。构建和 fake 测试通过后可宣称“可编译、协议测试通过”；真实 A3 验收通过后才能宣称该设备组合上运行通过。
