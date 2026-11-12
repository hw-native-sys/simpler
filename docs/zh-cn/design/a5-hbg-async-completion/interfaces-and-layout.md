# A5 HBG 异步完成接口与数据布局

[返回总览](../a5-hbg-async-completion.md)。本文定义 kernel 登记与 resident scheduler 之间的数据布局。

## 1. 登记接口

沿用 `get_async_ctx(args)`、`register_completion_condition(ctx, token)` 和 `save_expected_notification_counter(ctx, address, expected)`。调用方提供有效的完成记录地址；登记只保存条件，不隐式返回、不保存 continuation、不在条件未满足时阻塞。专用 waiter 登记后正常结束，消费数据的计算放在后继 task。

调用方和 runtime 按同一构建配套。常规 Callable 注册、序列化、缓存和任务参数传递保持原接口，不增加 capability manifest、资源身份快照、静态 UseSet 声明或 Host writer ABI 协商。不能凭登记函数返回推断 DMA 已结束。

`CompletionToken` 与既有 32 B `DeferredCompletionEntry` 保持原格式。COUNTER 使用 32-bit expected，SDMA post-done 使用 `backend_cookie` 保存完整 post id。登记检查支持的类型、读取宽度、地址对齐、expected 合法性及容量；地址所指 backing 的持有沿用既有 run/domain owner。

条件记录不编码 signal 与 payload 的对应关系，也不会为任意裸地址自动建立资源引用。调用方必须通过现有 Tensor/Buffer 参数、domain 使用登记或明确覆盖最后访问的外部 owner 保持 signal 和 payload 有效。异步能力及最大条件数只决定登记区能力和容量，不能证明这些数据的生命周期或发布顺序。使用既有 owner 和消费者全 DCache 获取的路径不要求额外的 signal-payload 声明；需要范围获取的后端须从现有调用参数及通信操作解析实际范围，不能只根据 signal 地址推断。

## 2. 每次物理执行的完成记录

`AsyncGroupData` 是 128 B 长、64 B 对齐的 header，记录 run epoch、task ref、block、lane、group index、容量、登记数量、首错、预留状态和封口状态。所有 header 连续排列在初始化前缀中；token 数组位于预留后缀。`entries_offset` 使用 header 的既有对齐余量，`entries_bytes` 只记录对齐后的 token 区长度，容量 1/64 对应 64/2048 B，加上 header 后分别预留 192/2176 B。header 与 token 分别检查边界和对齐，二者不得重叠。不增加资源 binding 表或每条件 binding index。

`AsyncWriter` 是 kernel 栈上的访问视图，只包含稳定 image 基址、长度、epoch、group offset/index。它不持有完成条件本体；复制视图仍操作同一个 group，容量、预留和首错不会因复制而分叉。

header 中还保存 run 内 SDMA 提交互斥数组的 offset 和数量，占用既有对齐余量，header 总长仍为 128 B。该数组按实际 cluster 数分配，每项 64 B；同 cluster 的 AIV 使用同一项，在 session 构造和提交期间互斥。它不描述外部资源身份，不持有 payload，也不参与完成等待。

header 的前 64 B 包含本执行可变字段，以及 Host 初始化后保持不变的 group_index、run_epoch、task_ref、entries_offset 和 sdma_guards_offset；后 64 B 保存只读的 block_index、subtask_slot、entries_bytes、sdma_guard_count 和 capacity。首行回写同时覆盖其内不变的 Host 字段。登记区只有当前物理执行写入。正常返回或 predicate skip 后 runtime 封口并发布；后续观察不能依赖已经释放的 dispatch payload、原核栈或 UB。首错不可覆盖，未关闭预留、越界或忽略登记失败都使封口失败。

后端提交若需要预留，在副作用前用内部 `reserve_conditions` 占用足够容量；提交后按实际公共完成记录数 commit，再关闭预留。登记普通 COUNTER 时一次调用内部完成这三个步骤，不要求业务代码提供额外绑定对象。

## 3. 内部上下文

`LocalContext` 保持 48 B，已有 slab 上下文保持 40 B，`DispatchPayload.args` 起点保持 64，完整 payload 保持 512 B。48 B `GlobalContext` 的 resident 字段提供当前执行的稳定完成记录地址与身份；`async_present` 只表示这个 dispatch 是否带 resident 上下文，不是跨 runtime 的注册协议版本。`async_output_capture_refused` 表示参数诊断请求了不能在异步完成前读取的输出；每次 resident dispatch 清零，命中诊断限制时设置，并由 kernel 注册入口拒绝。

旧调度路径仍使用原 slab；resident 路径使用稳定 group。二者由当前 dispatch 填入的上下文决定，公开登记函数不要求调用方选择或重写接口。

有效 resident `AsyncCtx` 的 `completion_count`、`completion_error_code`、`completion_entries` 和 `completion_capacity` 直接映射该 group 的真实存储，支持既有前端对登记区的检查；不另分配 slab，也不构造虚假的 `task_token`。`async_ctx_is_deferred()` 判断上下文身份、布局及 OPEN 状态，首个登记错误不使这个身份失效。后续登记与封口独立检查并保留首错，重复获取上下文不能清空已有条件或将溢出改写成其他错误。

## 4. Host 预分配

Host 已拥有展开后的普通任务和 Graph child 元数据，按 task/block/active lane 生成稳定索引。`AsyncTaskPlanInput` 只保存这些调度信息及内部容量，不携带额外资源身份。Host 每个逻辑 task 保存一个 40 B 的 `AsyncTaskHostPlan`，包含设备任务描述、三个 lane 的容量、条件区起点及每 block 的条件区步长；初始化时展开 block/lane，并复用一个 header 临时对象填充最终镜像，不为每个物理执行保存临时规划项或完整对齐 header 副本。输入 task ref 递增时不建立重复检查哈希表，其他顺序仍完整检查重复。Graph Definition、TensorData、task 参数和 callable 格式保持不变。

基础实现为每个可执行物理执行预留既有上限 `MAX_COMPLETIONS_PER_TASK`，支持没有额外声明的现有 kernel。没有登记条件的执行封口后直接释放资源，不创建 CHECK 责任。预留容量与每轮初始化、上传范围分开：Host/Device backing 保留完整容量，但 `initialized_bytes` 只覆盖原调度状态及异步元数据、join、目录、SDMA guard、连续 header。token、CHECK/FINISH lease 与进度位于后缀，首次实际发布前由唯一写者初始化。Host 使用既有 pending metadata 路径上传此前缀，不清零或上传未使用后缀。空间、bind 和 H2D 成本分别计量，不将减少上传宣称为省掉全部预留内存。

`SchedulerAsyncLayout` 为 128 B，保存 task→group 映射、连续 header 基址 `groups_offset`、lease、CHECK 进度、task join、FINISH 目录、活动位图及 run 活动标记的位置。offset 112 保存 activity_offset；标记独占 64 B 行，禁止与 cached 数据共用回写行。FINISH owner 持有 lease 直到全部后继同步传播完成；目录扫描游标位于 control state。lease、发布数据和 CHECK 进度分别隔离到 64 B cache line。

SDMA 提交互斥字在 Host 初始化为零，只在运行中使用原子操作；SDK 的 session/postId/SQ 状态仍使用原 workspace。32 个实际 cluster 对应 2048 B 互斥存储，不能按逻辑 block 数重复分配，也不扩大公共通信 context。

run bootstrap 校验整个连续 header 数组；执行入口根据 task/block/lane 计算稳定索引，不读取未使用的 header。kernel 请求 `AsyncCtx` 时校验记录格式与 OPEN 状态，返回封口时再次核对实际 task/block/lane 和登记状态。token-only 长度须与容量一致，不能接受把 header 长度混入 token 区的记录。

所有偏移、乘法、对齐与总分配长度采用 checked arithmetic。控制-only 图不分配 group；分配失败在启动前返回错误，不提交部分图。记录在整个 run 内保持稳定。

## 5. 身份与生命周期边界

run epoch、group index 和 lease generation 用于防止调度记录串用、重复完成与交权错误。这些内部状态不替代外部资源的生命周期。

外部资源 generation 即使存在，也只能识别过期引用，不能阻止旧 DMA 写入复用地址。必须由已有 allocator、run、Buffer 或 domain 的实际持有及退休机制保证不提前释放/复用；不为表达这一责任而扩展 Tensor 或 Graph 格式。
