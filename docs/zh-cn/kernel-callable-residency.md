# Kernel 模式 callable 注册调用链

本文说明 `simpler_kernel_mode_prepare_callable` 的注册、上传、容量限制、失败回滚和 launch 前查询。
实现基于 K2 提交 `abbe07f5` 的持久资源机制；注册沿用 K2 的内部 AICPU control stream 同步，不同步调用方 stream 或整个设备。

## 1. 接口与生命周期

```cpp
int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, const void *callable,
    size_t callable_size, int32_t *out_callable_id);

int simpler_kernel_mode_launch(
    DeviceContextHandle ctx, int32_t callable_id,
    const void *args, void *caller_stream);
```

`callable` 是完整、尚未修补设备地址的 `ChipCallable` 序列化镜像，包含 header、orchestration SO
和子 `CoreCallable`。每次成功 prepare 都上传、注册新 callable，返回新的 ID；即使内容和输入指针
完全相同也不会去重。上层负责保存和复用 ID，需要复用时直接 launch，不再调用 prepare。
返回值是状态码，失败时 `*out_callable_id = -1`；输出指针必须指向独立的可写存储。

```cpp
int32_t id = -1;
int rc = simpler_kernel_mode_prepare_callable(ctx, callable, size, &id);
if (rc != 0) return rc;
return simpler_kernel_mode_launch(ctx, id, args, caller_stream);
```

ID 属于创建它的 context，在 `[0, 8192)` 内顺序分配；成功返回的 ID 不释放、不覆盖、不复用，
直到 context close 时全部失效。不同 context 可能有相同 ID，调用方必须保证 ID 与 context 配对。
接口不携带 callable generation，也不检测跨 context 的 ID 误用。

调用前必须完成 `simpler_kernel_mode_init`。K2 的 `context_generation` 仍属于其 context 资源契约，
不进入 callable ID 或注册记录。prepare 必须在 capture 外执行；同一 context 的
init、prepare、launch、close 由调用方串行化。close 前必须停止 enqueue、等待所有执行完成并销毁相关 graph。

## 2. 注册与上传

[onboard C ABI](../../src/common/platform/onboard/host/c_api_shared.cpp) 调用链：

```text
simpler_kernel_mode_prepare_callable
  ├─ 校验输入/输出指针、镜像尺寸、对齐、kernel 模式和 context 状态
  ├─ adopt_borrowed_device：记录调用方已经绑定的设备身份
  ├─ KernelCallableCache::stage
  │   ├─ 校验镜像布局，拒绝越界、非法 signature、名称和子项偏移
  │   ├─ 检查未提交注册、8192 项限制与累计字节预算
  │   ├─ 分配新 ID，保存独立 Host 镜像副本
  │   └─ 在已有块内或新块取得新地址、修补子地址并上传
  ├─ record_callable_on_runner：保存 runtime 注册信息
  ├─ prepare_kernel_callable：复用 K2 的设备注册与持久参数准备机制
  ├─ cache.commit：标记 ready
  └─ 写出 ID，返回 0
```

[KernelCallableCache](../../src/common/platform/include/host/kernel_callable_cache.h)
保存注册镜像、容量和驻留状态，不做内容查找或 hash 去重。布局辅助函数仍计算 runtime 元数据中的
hash，但 hash 不用于选择已有注册。暂存项在 commit 前不能被 launch 解析。

设备代码内存通过 `kernel_callable_cache_ops().allocate → mem_alloc_.alloc` 按需分配：

- 小镜像共享 **2 MiB 块**，按 `align_up(callable_size, 64)` 在块内子分配；已有块有空间时不申请新块。
- 大于 2 MiB 的镜像按其 64 字节对齐后的大小独立分配。
- 扩容只追加块，不搬迁已发布镜像。大镜像分配不会丢弃已有小块尾部，后续小镜像仍能使用它。
- **8192 项**是注册条目上限；**2 GiB**是所有已分配代码块的总容量上限，包含对齐和块内空闲尾部。
  两者独立检查，均不是首次预分配量；不预分配 8192 套执行资源。
- 第 8193 次注册或无法在容量预算内申请新块时分类报错；已驻留地址和 ID 不变，不换入换出。

`resident_bytes()` 统计镜像的对齐占用，`allocated_bytes()` 统计物理块容量，二者不会混作预算。
共享协议常量 `MAX_REGISTERED_CALLABLE_IDS = 8192` 同时约束 Host runner、ID 校验和 A2A3/A5 TRB
AICPU 注册表及 dispatch。AICPU 表仅保存 SO 元数据；HBG 使用 Host 注册表，持久执行资源仍按 context 创建。

上传使用临时 scratch，`patch_chip_callable_scratch_for_device` 只修改子 callable 的设备地址，
调用方镜像和保留的 Host 副本均不变。当前 `Ops.copy → rtMemcpy` 是同步 H2D，未修改 K2 的同步点。

## 3. runtime 注册桥接

```text
record_callable_on_runner
  └─ register_callable_impl
      └─ upload_and_collect_child_addrs
          └─ HostApi::upload_chip_callable_buffer
              └─ DeviceRunnerBase::upload_chip_callable_buffer
                  └─ kernel 模式：cache.pending_uploaded_address()
```

注册桥接只读取本次 stage 暂存项的上传地址，不会取到旧注册的地址；独立上传已经在 stage 完成。
辅助函数据此生成子 kernel 地址映射，`record_callable_on_runner` 将结果保存在 `callables_` 表中。
program 模式继续使用原有上传和引用计数路径。

TRB 通过 `register_callable_on_device` 在 context 的 AICPU stream 上发射注册任务，加载 orchestration SO。
HBG 在 Host 加载 orchestration SO。随后 `persistent_args.prepare_once` 复用 K2 的持久参数资源；
代码镜像按注册独立上传，context 级参数块不按 callable 再分配。

## 4. 错误与回滚

- 数量超限：`CALLABLE_COUNT_EXCEEDED (-1005)`；相同内容也拒绝。
- 单项或累计字节超限：`CALLABLE_BYTES_EXCEEDED (-1006)`，已有注册不变。
- 镜像非法：`INVALID_ARGUMENT (-1004)`，上传前拒绝。
- 上一次注册仍未提交：`INVALID_STATE (-1003)`。
- stage 分配/上传失败：移除候选，不增加计费；已分配块保留供重试，仍计入 2 GiB 预算。
- `record_callable_on_runner` 失败：回滚暂存项及其在所属块内的占用，已有 ready 项不变。
- `prepare_kernel_callable` 失败：context 进入 Poisoned，保留可能被设备引用的地址，等待调用方完成
  quiescence 后显式 close；不发布候选 ID。

失败时尚未发布的未发布 ID 可以回滚，成功发布的 ID 不复用。错误由 C ABI 返回，不调用 `exit/abort`。
kernel context 拒绝通过 program 的 register/unregister 接口绕过注册管理。
正常 finalize 清理 Host 注册记录，设备块由 `MemoryAllocator` 统一释放；fatal 路径沿用 K2 的资源处理。

## 5. launch 与验证范围

```text
simpler_kernel_mode_launch(ctx, id, args, caller_stream)
  ├─ 校验指针、ID 范围、kernel 模式和 context 状态
  ├─ cache.resolve(id, residency)
  │   ├─ 未注册或尚未 ready：CALLABLE_NOT_RESIDENT (-1007)
  │   └─ 成功：得到 ID、设备地址和镜像大小
  └─ binder 尚未接入：返回 INVALID_STATE
```

Host kernel launch 接口与 ID 驻留检查保留。当前没有 callable generation 校验，
也没有单独的 `simpler_aicpu_kernel_exec` 包装入口或 payload 消费桩。
K9 invocation header 保留 ID 和 payload 信息，具体消费检查由后续真实执行链路负责。
`simpler_kernel_mode_supported()` 仍返回 0，prepare 成功不意味着 kernel launch 已能执行。

[注册单测](../../tests/ut/cpp/common/test_kernel_callable_cache.cpp) 覆盖重复内容的独立 ID/设备地址、
8192 项边界、共享块增长、大镜像独立分配、物理容量边界、对齐、失败回滚、子地址修补和 ID 驻留查询。
[C ABI 测试](../../tests/ut/py/test_kernel_mode_c_api.py) 验证真实 prepare 注册及生命周期：
两种 runtime 各自注册相同镜像 65 次，验证突破原条目边界、按块增长和首次分配远小于 512 MiB；
8191/8192 的边界检查、8192 项及超限不变性由 C++ 单测覆盖。
完整算子执行与 ACLGraph replay 上板验证仍待 binder 和消费入口接入。
