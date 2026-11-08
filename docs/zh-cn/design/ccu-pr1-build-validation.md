# PR-1 CCU 执行层：构建与验证

本页记录独立 CCU 执行层的验证方法与实际结果。公开接口采用[开发计划 v7 的嵌入模块、位置参数及域实例契约](ccu-pr1-interface-conformance.md)。Host C ABI 兼容路径仍参与构建和 CPU 测试，其逐轮 prepare/arm 流程不能替代 v7 的设备发起循环。

## 可复现构建

在安装 CANN 9.3.0 的 Linux 中，以仓库根为工作目录执行。需要 C++17 编译器、CMake、OpenSSL 开发包、GoogleTest 和 SDK 自带的 ccec/ld.lld。若 GoogleTest 安装在独立前缀，向 configure 追加 `-DCMAKE_PREFIX_PATH=<GTEST_INSTALL>`。

```bash
source <CANN_INSTALL>/set_env.sh
cmake -S src/a5/platform/onboard/host/ccu -B build/ccu \
  -DSIMPLER_CCU_BUILD_TESTS=ON
cmake --build build/ccu --parallel 4
ctest --test-dir build/ccu --output-on-failure -V
cmake --install build/ccu --prefix <OUTPUT_DIRECTORY>
```

| 产物 | 用途 |
| ---- | ---- |
| `libsimpler_ccu.so` | 真实链接 HCOMM、AscendCL 的 Host 执行库，启用 `--no-undefined` |
| `ccu_trigger.o` | 独立 AIV 触发器，与 Host 库同目录安装 |
| `ccu_device_aiv.o` | 公共设备接口的 A5 `dav-c310-vec` 编译探针 |
| `test_ccu_session_onboard` | 双 rank 真机 Host harness，构建时不执行设备任务 |
| `ccu_session_onboard.o` | 接受运行时 rounds 并连续提交请求的 AIV 测试内核 |
| `libccu_positional_module.so` | 位置 ABI 2 的 root0/root1 builder 模块，harness 将其字节嵌入注册 |

配置找到 `aarch64-linux-gnu-g++` 时还生成 AICPU 公共头文件编译探针。安装产物包含公共 CCU 头文件和构建指纹。正式 A5 Host runtime 使用同一执行层源文件和触发器，Python 构建流程要求 Host `.so` 同目录存在 `ccu_trigger.o`。

独立目标构建成功不代表整个 Host runtime、全部模拟测试或远端 CI 已通过。历史 CANN 9.2.0-beta.2 全量 Host runtime 构建曾遇到 URMA 头文件缺少 `HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN`；该历史结果不用于判断这里的 CANN 9.3 独立目标。

## 已完成的 CPU 与编译验证

验证日期：2026-11-06。源码快照：`ba178c4ca3509ed3f8682bf283ff56cd01d88f90`。环境为 Linux aarch64、GCC 12.3.1、CMake 3.31.12、OpenSSL 3.5.6、GoogleTest 1.15.2 和 CANN 9.3.0。日志文件为 `build-ba178c4c.log` 与 `ctest-ba178c4c.log`。该快照接受从 0 开始的域及内存编号，1000 次请求的 CPU 回归使用编号 0，仍保留代次与所有者校验。

独立构建成功，以上 SDK 库、AIV、Host harness 和模块产物均已生成。CTest **9 组、67 个 C++ 用例全部通过**：

| 测试组 | 用例数 | 覆盖 |
| ------ | -----: | ---- |
| `ccu_execution_api` | 16 | 兼容 C ABI 的双 rank 协议和失败保留 |
| `ccu_execution_abi` | 1 | 兼容 ABI 布局 |
| `ccu_request_state` | 5 | 请求状态及退休 |
| `ccu_request_validation` | 3 | 参数校验 |
| `ccu_session` | 12 | 会话、内存、嵌入模块顺序、错误 ABI、回滚与关闭 |
| `ccu_service_program` | 12 | 常驻循环、用户 builder、V1 指令限制及真实录制指令的边界和随机数解释执行 |
| `ccu_session_sdk` | 6 | SDK 适配、CCU 模式配置、Host 参数与 AIV 启动、文件加载选项、排空和失败重试 |
| `ccu_device_contract` | 7 | 设备参数快照、请求身份与序号 |
| `comm_workspaces` | 5 | 共享通信 workspace 布局 |

本次增量覆盖共享比较临时变量的嵌套分支、输入不被覆盖及固定资源开销，并验证资源诊断查询或清理失败时仍返回原注册错误、保留待清理描述符。

测试后端可复用于 PR-2/PR-3：`tests/ut/cpp/a5/platform/support/ccu_session_fake.h/.cpp` 在 initialize 所在线程通过 `ccu_test::QueueBackend(control)` 注入 SDK 结果，通过 `Complete(request, result)` 显式注入完成。它沿用真实布局，但不能证明硬件访问资格、数据搬运或跨引擎可见性。

## 真机 harness 与当前结果

源文件位于 `tests/ut/cpp/a5/platform/support/`：`ccu_session_onboard.cpp`、`ccu_session_onboard_kernel.cpp`、`ccu_positional_module.cpp`。Host 命令为：

```text
test_ccu_session_onboard device rank rendezvous_dir rounds kernel.o module.so
```

需要两个独立进程、两个已分配设备和一个本次运行专用的空 rendezvous 目录。rank0 原子发布随机 epoch 与 HCCL root-info，双方按代次和 phase 交换控制数据并有界等待。rounds 当前允许 1–4096；已验证 4 轮和 1000 轮。以下片段只能放在已经取得两张卡的设备任务内，`DEVICE0`/`DEVICE1` 来自该任务分配：

```bash
run_dir=$(mktemp -d "${VALIDATION_DIR}/ccu-onboard.XXXXXX")
rounds=1000
build/ccu/test_ccu_session_onboard "$DEVICE0" 0 "$run_dir" "$rounds" \
  build/ccu/ccu_session_onboard.o build/ccu/libccu_positional_module.so \
  >"$run_dir/rank0.log" 2>&1 &
rank0_pid=$!
build/ccu/test_ccu_session_onboard "$DEVICE1" 1 "$run_dir" "$rounds" \
  build/ccu/ccu_session_onboard.o build/ccu/libccu_positional_module.so \
  >"$run_dir/rank1.log" 2>&1 &
rank1_pid=$!
status0=0
status1=0
wait "$rank0_pid" || status0=$?
wait "$rank1_pid" || status1=$?
test "$status0" -eq 0 && test "$status1" -eq 0
```

Host 为每个 rank 准备 24576 字节域窗口和 128 字节结果区，注册同一嵌入模块中的 root0/root1 两个入口，再发布上下文。一次 AIV 启动在设备侧循环，改变输入输出子视图、长度和 root，检查数据与保护区；轮间没有 Host prepare、拷回或流同步。Host 最终检查 `status=0` 且 `completed_rounds=requested_rounds`，待双方任务结束、释放 run 引用并安全 close 后才释放资源。未确认停止的失败路径保留 owner，不 reset device。

真机快照 `d7af83d3d6b09e75901f5d0b929cced4e70d91de` 的结果如下；每个日志目录均包含 `rank0.log`、`rank1.log` 和 `ascend/debug/plog/`：

| 轮数 | 任务 | 设备 | 日志目录 | 结果 |
| ---- | ---- | ---- | -------- | ---- |
| 4 | `task_20261106_165451_9009456358` | 0、3 | `onboard-d7af83d3-r4/` | 两个 rank 均 `status=0`、`completed_rounds=4` |
| 1000 | `task_20261106_165559_94287416066` | 0、1 | `onboard-d7af83d3-r1000/` | 两个 rank 均 `status=0`、`completed_rounds=1000` |

两次任务均完成数据和保护区检查，两个 rank 的完成轮数与请求轮数精确一致，正常 close 后释放资源。每轮由设备选择 root0/root1 用户入口、域内偏移及 64/1024 元素长度；Host 只准备一次实例并启动一次 AIV 循环。

该快照已完成以下 CANN 9.3 / CCU V1 适配，并通过上述 CPU 与真机回归：

- 使用 `HcclCommInitRootInfoConfig`，将 `HcclCommConfig.hcclOpExpansionMode` 设为该配置协议中的 CCU_SCHED 值 **6**；它与 `HcclOpExpansionMode` 枚举的同名值不是同一编码。
- V1 不支持 MUL、变量加立即数、变量间直接比较及大小条件。`ccu_v1_arithmetic.h` 仅用变量加法、赋值和与立即数的 EQ/NE 判断实现比较；相等判断固定 64 步，无符号顺序比较固定 64 轮、每轮至多 63 次倍增。常数乘法最多 16 轮，立即数加法先构造常量变量。原有范围、溢出、序号和身份检查均保留，没有无界比较循环。
- CPU 解释器执行实际录制的 helper 指令，覆盖边界值、固定随机数、六种关系、重复调用及句柄写回；生产指令记录禁止不受支持的操作，防止后续改动重新生成这些指令。
- Host 参数使用 `aclrtLaunchKernelWithHostArgs`，并明确指定 `ENGINE_TYPE_AIV`。`aclrtBinaryLoadFromFile` 的加载选项保持 `nullptr`；该文件加载接口不接受 binary magic 选项。SDK stub 同时核对参数传递方式、引擎和加载选项。

集成模块的真实注册曾要求 407 个变量，超过默认分配的 400 个。后续使用每个服务共享的 10 个比较临时变量，替代逐次分配；每次比较仍初始化自身使用的临时值，外层条件在进入分支时已被消费。更新后的集成快照 `2af036ee` 已完成服务注册；其首个通信任务报错 101 的后续定位和修复见下节。

## 验证范围与后续联调

PR-1 独立执行层已通过嵌入用户模块、一次准备后至少 1000 次设备串行请求、动态域内地址和长度、每轮数据与保护区以及正常关闭释放的真机验收。CPU 测试另覆盖失败注入、精确序号匹配和资源保留；这些不等同于所有硬件故障场景均已注入验证。

PR-2/PR-3 的 worker、provider、TMR 完成与所有权主路径已完成下述联合实机验证。域重建、跨域隔离及专项故障的独立验收见本页 2026-11-08 补充记录，不能从单域成功路径推定通过。

已只读核对 HCOMM 源码快照 `b6037c37aafb926cc767d9797abaf07190968506`：`ccu_kernel_mgr.cc` 查询内部 HBM token，translator 将其作为 `memTokenInfo`，Store 使用给定地址及该内部 token 生成搬运并追加等待。上述真机运行验证了本 harness 分配路径中的实际 CCU 数据搬运、完成写回与 AIV 读取；该证据不推广到任意分配方式或完整 AICPU 运行时的可见性。

成功释放必须同时满足匹配的 CCU 完成、请求退休和普通任务读者退出。普通 AIV FIN 或等待超时均不构成单独的资源释放依据；不能将 CPU 注入完成、DSL 指令记录、构建成功或硬件任务已提交写成真机验收通过。

新版快照 `ba178c4ca3509ed3f8682bf283ff56cd01d88f90` 的双卡 1000 轮独立验证也已通过：任务 `task_20261106_182755_41458312678`，设备 0、1，日志目录 `onboard-ba178c4c-r1000/`。两个 rank 均报告 `status=0`、`completed_rounds=1000`，数据及保护区检查通过，正常关闭并以退出码 0 结束。此结果覆盖共享比较临时变量和编号 0 的修复，仍不替代 PR2/PR3 的完整运行时联调。

## 三个 PR 的联合实机验证

2026-11-06，集成快照 `68e00c466eefaa66d5cf771c2d79c3ccd6f0109d` 在 CANN 9.3 / A5 双卡通过正式 Worker、域 provider 和 TMR 后继任务场景。PR-1 的执行代码与交付快照 `86577bee9a89212fb28271d36431e5fe7c7bc0b1` 一致；后者与已通过严格构建、67 个 CPU 用例和 clang-tidy 的 `6c7ad380d5d6baf39130291e58ad67075520d2fe` 具有相同 Git tree。

| 场景 | 任务 | 设备 | 日志目录 | 结果 |
| ---- | ---- | ---- | -------- | ---- |
| 4 轮 | `task_20261106_191642_263921111571` | 0、1 | `onboard-68e00c46-smoke/` | 通过，23.8 秒，退出码 0 |
| 1000 轮 | `task_20261106_191739_268998428471` | 0、1 | `onboard-68e00c46-thousand_rounds/` | 通过，9.1 秒，退出码 0 |

每个 rank 只发起一次 Worker 运行。设备侧逐轮提交初始化、CCU 请求和依赖后继验证任务；AIV 提交后立即返回，TMR 等待真实 CCU 完成并退休请求后才放行后继。每轮改变 root0/root1、64/1024 元素长度及域内偏移，校验源、目标、scratch 和保护区；Host 最终校验全部轮次结果、域已释放且没有存活域，然后正常关闭 Worker。两次耗时包含编译缓存、启动和队列内测试开销，不作为性能对比。

此前报错 101 的实机诊断显示首个 Tensor 的 dtype 和 ndims 读成切片偏移 16 与长度 64。测试内核改为从 GM 按字段读取 Tensor 到局部对象，替代逐字节重解释复制和按值返回后，上述两个场景均通过；数据类型、设备地址空间、步幅和边界校验没有放宽。失败退出也补齐了所有权保护：普通读者或父通信环境未确认释放时，停止收集线程并保留 owner，不通过全设备排空或 reset 强行回收。

该集成快照另通过 A5 TMR 全量构建、端到端 AIV 离线编译，以及 CCU completion（23 用例）、shutdown（5 用例）和 shared-memory 三个 CPU 测试组。这里没有宣称全部仓库 CI、任意外部分配内存、多域重建或所有故障注入已通过。

## 多域与专项故障验证（2026-11-08）

联合源码快照 `e8d6081b5264b87a879a7f6faa883d6468950e11` 在 CANN 9.3 / A5 设备 2、3 上通过全部 8 项 Worker/TMR 双卡测试。任务 `task_20261108_173610_121532929008` 退出码为 0，日志目录为 `extended-e8d6081b-full-20261108-173610/`。

| 场景 | 验收内容 | 结果 |
| ---- | -------- | ---- |
| 非法参数与跨域地址 | 设备拒绝跨域地址、长度溢出、未对齐地址、参数数量及标量宽度错误；随后合法请求成功 | PASS，5.5 秒 |
| 4 轮通信 | 数据、保护区、后继依赖及域回收 | PASS，12.0 秒 |
| 每 rank 1000 轮 | 变化参数的设备循环及正常关闭 | PASS，15.0 秒 |
| 两个 callable 的 ID 0 | 新目录发布后，各 callable 仍执行自身函数 | PASS，12.0 秒 |
| 保留域复用与重建 | 释放域 A 后域 B 数据和通信不受影响；重建 A 后旧引用被拒绝 | PASS，12.0 秒 |
| 活动请求明确失败 | 真实 CCU 返回 105；两个 rank 的已发布后继均未调度；域回收且 Worker 正常关闭 | PASS，13.0 秒 |
| AIV 结束后 CCU 仍未完成 | 延迟另一 rank，用全 AIV 并发见证任务确认请求仍为 pending，随后完成通信 | PASS，12.1 秒 |
| 单 rank 准备失败 | 两侧实际进入 builder，失败后回滚，使用同一 Worker 重新建域并完成通信 | PASS，17.0 秒 |

故障测试发现并修复了 PR-2/PR-3 的两个收尾问题。明确的 CCU 业务失败原先会被转换成设备内核异常，阻断普通完成确认；现在只有所有核心完成实际退出协议、每个回调没有其他错误且终态记录发布成功时，设备调用才正常结束，原始 105 仍保留给 Host。Host 再独立确认执行流与 CCU 读者都停止，才允许回收；错误码本身不能证明停止。

Worker 提前等待结果时的异常也曾绕过清理控制，或在关闭时被后续成功结果覆盖。现在按每个运行保存原始异常，确认其已结束且无需恢复资源所有权后继续处理清理请求；未确认停止的情况仍保留资源。失败后继的诊断按任务身份和代次检查依赖，两个 rank 均报告完整扫描、调度数 0 和未完成状态。

配套验证通过 A5 TMR 构建、所有新增设备内核离线编译、8 组 C++ 回归及 38 项 Python 控制与异常收尾回归。本次 C++ 改动的 Linux clang-tidy 通过。C++ 回归快照为 `9fd5b9ccc`，与最终实测快照的 C++ 源码一致；最终增量仅涉及 Python 异常处理及测试。这里覆盖上述明确场景，不代表断卡、任意超时或全部仓库 CI 已通过。
