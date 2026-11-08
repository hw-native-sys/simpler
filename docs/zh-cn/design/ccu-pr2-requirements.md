# CCU PR2 开发需求：AIV 调用与 TMR 完成跟踪

更新：2026-11-06。工作分支：`skx/ccu-pr2`；核对基线：`8001268ba9e82fe11c8627c89297646f09b60e8d`。

方案依据：[飞书《CCU 函数注册与调用接口方案》](https://hw-native-sys.feishu.cn/wiki/YOXZwef8QiabQfktGBYcXihVngg)，以及主工作区 `simpler-main/new-doc/index.html` v7。
公开接口按飞书方案执行。本文件是开发需求和验收清单，勾选项当前均未验收，不表示功能已经实现。

## 目标与开发边界

让普通 AIV 任务选择一个已注册的 CCU 函数，传入本轮参数，并选择在 AIV 内等待，或交给 AICPU 上的 TMR 调度器跟踪完成。
后继任务必须等通信真正完成、请求槽可复用且普通 AIV 执行也结束后才能运行。

首版范围：A5 / CANN 9.3、单机两卡 UB_CTP、TMR；以 Broadcast 和最小用户函数验证连续 FP32 数据。
每个 rank 的每次调用由一个逻辑 AIV 块触发，每个执行上下文最多一个活动请求；同一 L2 支持设备运行时决定次数的串行重复调用。

| 交付方 | 负责内容 | PR2 的接入要求 |
| ------ | -------- | -------------- |
| PR1 | CCU 平台执行、域内内存访问、请求提交 / 轮询 / 回收、可确认关闭 | 使用共享协议，不从 demo 导入正式实现 |
| PR2 | 设备 API、隐藏 callable 身份的设备侧接入、参数快照、完成跟踪与解依赖 | 本文件的主要工作 |
| PR3 | callable 内嵌模块和顺序 ID、Host 注册、orch 建域、实例准备与所有权 | 消费已准备的实例；不在 AIV 上临时创建资源 |
| PyPTO 后续 PR | 生成 CCU ID、位置参数、调用点读写方向和排序依赖 | 以本文件设备样例为生成目标 |

HBG、多请求并发、多机和更多集合通信不属于本次交付。域外任意 GM Tensor 也不属于首版能力。

## 已确定的接口与语义

以下是新增的 simpler 设备接口，声明省略设备编译修饰符；不是现成的 PTO ISA API。

```cpp
class CcuCallArgs {
public:
    void add_tensor(const Tensor& tensor);
    template<typename T> void add_scalar(T value);
};

CcuCallDescriptor CcuCall(
    const CommContext* domain_ctx,
    uint32_t ccu_id,
    const CcuCallArgs& parameters);

bool wait_request_entry(AsyncCtx& task, const CcuCallDescriptor& call);
bool send_request_entry(AsyncCtx& task, const CcuCallDescriptor& call);
```

| 对象 / 操作 | 必须保持的行为 |
| ----------- | -------------- |
| `ccu_id` | 所属 `ChipCallable.ccu_functions` 的下标，从 0 开始；与 AICore kernel ID 分开 |
| 资源定位 | 使用当前 callable 身份、所选域和 CCU ID；两个 callable 的 ID 0 不能串用 |
| 当前 callable 身份 | 从运行时隐藏任务上下文取得；不增加用户每次 Host 启动的 CCU ID 参数 |
| `CcuCallArgs` | 所有 Tensor 在前、标量在后，各组保持加入顺序；没有 input/output/workspace 分类 |
| Tensor | 输入、输出、用户 scratch 全部属于所选域，允许已准备范围内的合法子视图 |
| `CcuCall` | 复制本地参数描述；Tensor 只复制视图，不复制数据、不预留硬件槽、不触发执行 |
| wait 成功 | 本次通信已完成，结果对当前 AIV 可见，请求槽已回收 |
| send 成功 | 本次请求已接受并提交，完成跟踪已有所有者；调度任务仍保留未完成状态 |

wait 和 send 都提交请求，单次执行二选一。不能先 send，再用同一个本地描述调用 wait 来等待原请求。
再次接受本地描述代表新请求；重复投递已经接受的同一个底层请求必须报错。每轮使用独立代次，调用者不管理 ticket。

不引入 `CcuSignature / CcuTensorParam / CcuScalarParam`，不使用模板程序 key 替代顺序 ID。
builder 与调用方负责位置参数的数量、类型、动态长度、别名和算法同步。
运行时负责 ID、域状态、Tensor 归属及视图范围、容器容量和受支持传输表示的检查；范围检查不能保证任意用户程序不会越界。
内存参数通过 Tensor 传入，不能把裸地址当 scalar 绕过检查。

## 实现任务

- [ ] 实现 `CcuCallArgs`、`CcuCallDescriptor` 和 `CcuCall`，保留上述公开签名。
- [ ] 接通隐藏 callable 身份，按 callable / domain / CCU ID 找到 PR3 准备的实例；支持一个 AIV 调多个 CCU、多个 AIV 调同一个 CCU。
- [ ] 接受请求时把参数保存到运行时持有的存储中，AIV 返回后仍有效；不得保存对 AIV 局部参数容器的悬空引用。
- [ ] 在触发硬件之前取得请求槽，并确保完成跟踪已有所有者；容量不足、校验失败或登记失败不能留下无主硬件请求。
- [ ] 实现 wait 的提交、等待、结果可见性处理和槽回收；实现 send 的提交与延后完成登记。
- [ ] 在 TMR 接入 CCU 完成类型及 poll / retire；匹配本轮代次和成功状态，完成后只回收一次。
- [ ] 在通信完成、槽回收以及普通 AIV 完成都满足后解除依赖，保持 Tensor 和排序对象的生命周期。
- [ ] 区分已确认硬件停止的失败与未确认停止：前者进入错误清理，后者继续保留域、callable 和内存并禁止复用；两者都不能放行消费者。
- [ ] 与 PR1 验证每轮重新读取参数、执行函数并重置同步状态；不能用 Host 固定次数展开或仅改完成标志代替重复执行。

复用 SDMA 已有的异步完成框架，但增加 CCU 自己的完成判断与回收适配。
当前基线的 `CompletionPollState` 只有 PENDING / READY / FAILED；未确认停止的所有权如何保留，需要接入实际错误清理链，不能把 FAILED 自动解释为资源已安全可释放。
完成代次的实际存储和校验方式随 PR1 共享协议落地，不能只比较一个可被旧请求满足的普通计数器。

## 调用样例与依赖声明

样例约定：CCU ID 0 为 Copy；Tensor 0 是 root rank 0 的 src，Tensor 1 是各 rank 的 dst（含 root），Tensor 2 是用户 scratch；scalar 0 是 `uint64_t count`。
各 rank 都传本地合法域内 Tensor，调用方保证长度不越界。各 rank 的调用顺序和次数一致。

AIV 包装函数体片段，`task` 从原始调度参数取得 `AsyncCtx`：

```cpp
constexpr uint32_t CCU_COPY = 0;
CcuCallArgs parameters;
parameters.add_tensor(src);
parameters.add_tensor(dst);
parameters.add_tensor(scratch);
parameters.add_scalar(count);

auto call = CcuCall(domain_ctx, CCU_COPY, parameters);
if (!send_request_entry(task, call)) return;
```

同一 AIV 随后读取 dst 时，把 send 换成 `wait_request_entry(task, call)`，成功后才继续计算。
src 与 dst 是独立参数；`CcuCallArgs` 不建立任务依赖，也不注册或隐式复制内存。

L2 片段中，src / dst / scratch / order 均已由 Host 传入；数据视图为 64 个 FP32 元素：

```cpp
constexpr uint32_t CCU_WRAPPER_KERNEL = 1;  // AICore ID 空间。
for (uint64_t i = 0; i < rounds; ++i) {
    SIMPLER_SCOPE() {
        const uint64_t count = 16 * (1 + i % 4);
        CoreTaskArgs args;
        args.add_input(src);
        args.add_inout(dst);       // 只更新前 count 个元素。
        args.add_inout(scratch);
        args.add_inout(order);     // 串行化同一执行上下文。
        args.add_scalar(domain_ctx);
        args.add_scalar(count);
        args.launch_spec.set_block_num(1);
        rt_submit_aiv_task(CCU_WRAPPER_KERNEL, args);
    }
}
```

`order` 只参与调度，不传给 CCU。通信完成并回收请求槽前，不释放其依赖。
`SIMPLER_SCOPE` 结束不是通信完成；所有访问方向须在任务就绪前通过 `CoreTaskArgs` 声明。
部分写入、别名和 scratch 访问由调用方 / PyPTO 显式处理，不能从无方向的参数容器推断。
生产者直接写域内 src 可避免额外数据复制；已有域外结果需显式复制到域内后再调用。

## 代码入口与共享交付

以下路径均相对本 worktree；新增目录 / 文件表示开发目标。

| 位置 | 开发用途 |
| ---- | -------- |
| `src/a5/runtime/tensormap_and_ringbuffer/runtime/async_kernel_api.h` | 现有 `get_async_ctx`、完成条件登记和设备公共入口 |
| 同目录 `backend/ccu/ccu_completion_kernel.h`（新增） | CCU 设备调用和 wait / send 适配 |
| 同目录 `completion_token.h`、`async_wait.h` | 完成表示、后端 poll / retire 注册、错误传播 |
| 同目录 `backend/sdma/`、`backend/urma/` | 参考现有分层及测试方式，不复用其硬件完成判断 |
| `src/common/platform_comm/ccu/`（PR1 新增） | 共享请求协议、实例访问及参数表示 |
| `src/common/platform_comm/comm_context.h` | 与 PR1 / PR3 共同接入 CCU workspace 和一致布局 |
| `tests/ut/cpp/a5/runtime/tensormap_and_ringbuffer/` | CCU 假后端、错误和完成测试；参考 `test_urma_completion_kernel.cpp` |
| `tests/st/a5/tensormap_and_ringbuffer/` | 新增真实设备调用、重复执行与延迟完成用例 |

开工先补齐以下实现细节，不重新选择公开方案：

| 事项 | 牵头 / 配合 | 需要交付的结果 |
| ---- | ----------- | -------------- |
| 位置参数编码 | PR1 + PR2 | Tensor 表示、支持的标量类型 / 位宽 / 对齐、容量上限、错误行为 |
| 请求与完成协议 | PR1 + PR2 | 预留、接受、触发、完成、回收、取消 / 错误清理的边界及可控假后端 |
| 隐藏 callable 身份 | PR3 Host 侧 + PR2 设备侧 | 从任务提交到 AIV 的完整传递链，以及跨 callable 隔离测试 |
| 域发现与准备 | PR3 牵头，PR2 接入 | 设备执行前实例已准备的保证；不能等 AIV 发 Host RPC |
| 内嵌模块加载 ABI | PR1 + PR3 | PR2 只消费准备好的实例，不实现模块加载 |

这些内容写入共享头文件和对应测试后，两侧可用假后端并行开发。当前基线尚无上述 CCU API 和平台目录。

## 验收清单

- [ ] 构造本地描述不触发 CCU；接受后参数快照不依赖 AIV 栈，元数据传递不复制 Tensor 内容。
- [ ] 域内 Tensor / 子视图成功；域外、跨域、越界、过期域、无效 CCU ID 和参数容量不足被拒绝。
- [ ] 两个 callable 各自的 CCU ID 0 隔离；一个 AIV 可调用两个程序，两个 AIV 可调用同一程序。
- [ ] wait 成功后同一 AIV 可读正确结果；send 返回后消费者仍等待真实完成。
- [ ] 延迟 CCU 时，消费者、下一轮调用和临时内存复用都不提前发生。
- [ ] 单个 L2 中运行时 rounds 至少 1000 轮，改变域内区域、长度和数据，逐轮结果及保护区正确；轮间无 Host prepare / 拷回 / 流同步。
- [ ] 旧代次、重复投递、槽不足、完成登记失败、通信错误均不能误放行或留下无主请求。
- [ ] 未确认停止时保留全部资源；已确认停止的错误通过清理回收，不作为成功返回。
- [ ] 原有 SDMA / URMA / RDMA 异步完成路径的相关回归通过。
- [ ] A5 双卡 TMR 实测 Broadcast 两个 root、64 / 1024 个 FP32 元素及用户 Copy；按长度配置足够域窗口。

假后端和 Host 单测验证协议，不能替代 CCU 硬件数据路径证据。上板按仓库流程使用 `task-submit`。
后续实现记录应包含提交版本、执行命令、结果 / 日志和仍未验证的项目；本次仅整理需求，未运行硬件验收。
