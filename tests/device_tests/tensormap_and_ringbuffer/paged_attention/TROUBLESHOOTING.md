# Paged Attention 测试排错说明

## 1. 间歇性测试失败（有时 PASS，有时 853/131072 或 1652/131072 不匹配）

### 可能原因

- **浮点非确定性**  
  Golden 在 host 上按固定顺序做 online softmax；device 上同一 (batch, head) 的多个 block 通过依赖串行执行，但各 kernel 内部或不同 core 间的浮点运算顺序、舍入可能略有差异，导致边界元素在 `rtol=1e-3, atol=1e-3` 下偶发不匹配。

- **调度/时序**  
  依赖关系由 orchestrator 的 INOUT 链正确构建（UP(bn) 依赖 UP(bn-1)），理论上执行顺序确定。若仍出现间歇性错误，需排查：  
  - 完成信号与 GM 写回顺序：AICore 在置 `task_status` 前是否已保证对 GM 的写对后续 task 可见；  
  - 是否存在极少数路径下 fanin 未满足即被调度（需依赖与完成逻辑的审计）。

### 建议

- 多跑几次用例，若仅偶发 1 次失败，多为浮点或环境波动，可暂时放宽 tolerance 或接受偶发差异。  
- 若需严格可复现：可考虑在 device 上对同一 (batch, head) 的 online update 使用更确定的归约顺序或同一 core 串行化（会牺牲性能）。  
- 定位时可在脚本中循环跑 N 次并统计失败率，或临时关闭 work stealing 观察是否仍失败。

---

## 2. rtFree failed: 507899（finalize 阶段）

### 原因

CANN 在 **stream 已 destroy 之后** 再对 device 内存调用 `rtFree` 时，可能返回 507899（或类似错误）。原先 `DeviceRunner::finalize()` 的顺序是：先 `rtStreamDestroy`，再 `perf_collector_.finalize()` 和 `mem_alloc_.finalize()`，导致在无有效 stream 的情况下执行 rtFree。

### 修复

已在 `src/platform/a2a3/host/device_runner.cpp` 中调整顺序：

1. 先执行所有通过 `mem_alloc_` 的释放：  
   `kernel_args_.finalize_runtime_args()`, `finalize_device_args()`, `so_info_.finalize()`  
2. 再执行 `perf_collector_.finalize()` 和 `mem_alloc_.finalize()`（释放 perf 缓冲区和剩余 kernel/reg 等分配）  
3. **最后** 再 `rtStreamDestroy(stream_aicpu_)` 和 `rtStreamDestroy(stream_aicore_)`

这样所有 rtFree 都在 stream 仍存在时完成，可避免 507899。

此外在 finalize 开头增加了 `rtDeviceSynchronize(device_id)`，确保所有设备操作（包括可能的异步拷贝）完成后再释放，进一步降低 507899 出现概率。

### 若仍出现 507899

- **确认用的是新 host**：每次执行 `run_example.py` 都会在临时目录完整重编 host（含 `device_runner.cpp`），无需单独执行 `setup.py`。若你改过 `device_runner.cpp`，直接再跑一次用例即可加载新 so。
- 若重跑后仍报 507899：可查 CANN 文档该错误码含义。当前代码已将 507899 记为 WARN（CANN teardown 已知现象），不再以 ERROR 报出。

### 仅 enable profiling 时出现 507899

原因：profiling 使用的 device 内存在分配后经 **halHostRegister** 做了 host 映射。CANN 的 **halHostUnregister** 在解除映射时可能已释放该 device 内存，若再对该指针调用 **rtFree** 会返回 507899。

修复：在 `device_runner.cpp` 的 finalize 中，对 perf 资源传入的 `free_cb` 改为只做 **untrack**（从 allocator 跟踪中移除），不再对该指针调用 `rtFree`。同时为 `MemoryAllocator` 增加 **untrack(ptr)**，仅从 `ptr_set_` 移除不释放。这样 unregister 后不再对该块调用 rtFree，507899 在开启 profiling 时也应消失。

---

## 3. Performance data collection idle timeout（0 / N records）

### 现象

开启 `--enable-profiling` 时出现：

- `poll_and_collect: Performance data collection idle timeout after 30 seconds`
- `Collected 0 / 16704 records before timeout`
- `Total buffers processed: 0`

即 Host 在等 AICPU 往 perf 队列入队 buffer，超时前一直没有新数据。

### 可能原因

- **设备卡住或未真正跑完**：AICore 未完成任务，或设备 14 被其他进程占用，导致 AICPU 从未向 host 可见的队列入队任何 buffer。
- **偶发时序/负载**：同一命令有时能收满 16704 条记录，有时超时，多为环境或负载偶发。

### 建议

- **先确认不带 profiling 是否通过**：去掉 `--enable-profiling` 再跑，若用例 PASS，说明计算正确，问题仅在 profiling 采集。
- **直接重跑**：多次运行同一命令，若多数时候能收齐记录，可视为偶发，必要时稍后重试或换设备。
- 若需稳定拿 perf 数据：可在设备空闲时单独跑带 profiling 的用例，或排查是否有其他进程占用同一 NPU。

---

## 4. 为何 enable profiling 时会“卡住”或像死机？

### 执行顺序（enable profiling 时）

1. Host 分配一块 **device 上的 perf 共享内存**，并用 `halHostRegister` 映射到 host，使 host 与 AICPU 都能访问。
2. 把 `runtime.perf_data_base` 设为该块地址，并随 runtime 拷贝到设备；AICPU 侧用该地址写每条任务完成后的 perf 记录。
3. **Launch**：依次下发 AICPU Init、AICPU Main（DynTileFwkKernelServer）、AICore kernel。
4. **紧接着**（在 `rtStreamSynchronize` 之前）Host 调用 **`poll_and_collect`**：在这里 **轮询** 读 perf 共享区里的队列（`queue_heads` / `queue_tails`），每收到 AICPU 入队的一个 buffer 就处理其中的记录；直到 **已收集条数 ≥ expected_tasks** 或 **连续 30 秒没有新 buffer** 才返回。
5. 之后才执行 `rtStreamSynchronize`、copy-back、比对结果。

因此：**只要设备没有把 perf 数据写入队列，Host 就会一直停在 `poll_and_collect` 里**，看起来像“卡住”。

### “卡住”的两种含义

| 情况 | 表现 | 原因 |
|------|------|------|
| **进程卡住约 30 秒后继续** | 无新 buffer 时轮询满 30 秒，然后打印 idle timeout、0 records，接着继续跑完（可能结果仍 PASS） | 设备侧没写入 perf 队列：AICore 未完成任务、设备被占用、或 AICPU 未正确写 total_tasks/入队。Host 只是阻塞等待，并非真死机。 |
| **整机死机/无响应** | 机器完全卡死，只能断电或强制重启 | 少见：可能为 NPU 驱动在 **profiling 路径**（如 `halHostRegister`、大量 host 可见的 device 内存访问）下的 bug，或设备/驱动在特定负载下挂死。 |

### 建议

- **多数是第一种**：等满 30 秒会超时退出，属“假死”；可先去掉 `--enable-profiling` 确认用例能稳定 PASS，再在设备空闲时开 profiling 多跑几次。
- **若是整机死机**：尝试升级 CANN/驱动；或暂时不用 `--enable-profiling` 规避；若可稳定复现，需带环境信息向 CANN/设备侧反馈（profiling + halHostRegister 场景）。

---

## 5. Enable profiling 时输出与 golden 不一致（不带 profiling 则 PASS）

### 现象

- 不带 `--enable-profiling`：用例稳定 **PASS**，无 507899。
- 带 `--enable-profiling`：**TEST FAILED**，如 `Mismatched elements: 750/131072` 或 `1528/131072`，且每次跑不匹配数量可能不同；应用 untrack 修复后 finalize 不再报 507899。

### 可能原因

- **与第 1 节叠加**：profiling 开启后多了一块 host 可访问的 device 内存（halHostRegister）、以及 AICPU 侧写 perf 的额外逻辑，可能改变设备侧内存访问顺序或时序，使原本就存在的浮点非确定性或偶发可见性更易暴露，表现为“带 profiling 更容易不匹配”。
- **Profiling 路径影响主路径**：设备上写 perf 记录、切换 buffer、更新 total_tasks 等与主计算并发，在极端时序下可能影响主计算（例如总线/缓存、或与 GM 的可见性），导致输出与 golden 不一致。

### 建议

- **以不带 profiling 的 PASS 为准**：若不带 `--enable-profiling` 稳定 PASS，可认为主计算逻辑正确；带 profiling 的失败可先视为 profiling 与主路径的交互/时序问题。
- **需要 perf 数据时**：多跑几次带 profiling 的用例，有时会 PASS；或先跑不带 profiling 做正确性回归，需要 swimlane 时再单独跑带 profiling 并接受偶发不匹配。
- **根因排查**：可在 AICPU 侧临时关闭写 perf（或延后写）、或调整 poll_and_collect 与 rtStreamSynchronize 的顺序做对比，确认是否与“host 读 perf 与 device 写主结果”的并发有关。

---

## 5.1 检查 device 0 状态与是否有进程占用

### 查看 device 0 状态（npu-smi）

```bash
# 设备用量（HBM、AICore/AIV/AICPU 占用率等）
npu-smi info -t usages -i 0

# 设备概要（温度、功耗、AICore 数量等）
npu-smi info -t common -i 0

# 内存信息
npu-smi info -t memory -i 0
```

若 **Aicore Usage Rate / Aicpu Usage Rate** 等长期为 0，且无业务在跑，可认为设备空闲。

### 查看是否有进程占用 device 0

部分环境不支持 `npu-smi info proc -i 0`，可用下面方式辅助判断：

```bash
# 查看是否有进程打开 /dev/davinci0（device 0）
fuser -v /dev/davinci0
# 或
lsof /dev/davinci0
```

无输出则当前没有进程占用 device 0。

```bash
# 查看是否还有 run_example / paged_attention 在跑（按需改 -d 的 device）
ps aux | grep -E "run_example|paged_attention" | grep -v grep
```

若有卡住的用例，可 `kill <pid>` 后再复位设备。

### 复位 device 0 后重跑（需 root 执行复位）

复位命令必须由 root 执行（`sudo npu-smi set -t reset -i 0`）。若无 sudo 权限，需联系管理员执行复位；**若 device 0 当前无进程占用且 npu-smi 显示 Aicore/Aicpu 占用率为 0，可直接不复位直接重跑用例**。

```bash
sudo npu-smi set -t reset -i 0
sleep 20
cd /path/to/simpler
PA_CASE=Case1 python examples/scripts/run_example.py -k ... -g ... -d 0
```

---

## 6. Device log 位置与 Ready queue 抢锁统计

### 如何获取 device log（a2a3 真机）

AICPU 的 `DEV_ALWAYS` 通过 CANN 的 **dlog** 输出，不会出现在 run_example 的终端里，而是写入 CANN 的 device 日志目录：

- **默认路径**：`$HOME/ascend/log/debug/device-<device_id>/`
- 每次运行会生成或追加到类似 `device-<pid>_<timestamp>.log` 的文件。
- 最近一次运行的日志可按时间戳或修改时间找到；也可在运行前设置 `ASCEND_PROCESS_LOG_PATH=/tmp/ascend_log` 将应用类日志写到指定目录（部分 CANN 版本下 device 侧 dlog 仍可能落在 ascend 默认路径）。

查找包含 ready queue 统计的日志行示例：

```bash
grep -E "ready_q|lock\(ready_q\)|scheduler stats" $HOME/ascend/log/debug/device-14/*.log | tail -80
```

### 一次运行中的 Ready queue 抢锁统计（示例）

以下来自一次 paged attention 用例（`--enable-profiling`，device 14，约 16704 tasks，3 个调度线程）的 device log 汇总。

**锁级别（每线程）**

| 线程 | 拿锁总次数 locks | 总等待 wait(μs) | 总持锁 hold(μs) | 平均每次等待 avg_wait(μs) | 平均每次持锁 avg_hold(μs) | 分项 wait/hold(μs): scan / orch / complete / dispatch |
|------|------------------|----------------|-----------------|---------------------------|---------------------------|------------------------------------------------------|
| 0    | 45804            | 6983           | 1194            | 0.15                      | 0.03                      | 170/274, 0/0, 1207/776, 5605/144                     |
| 1    | 42824            | 6795           | 1170            | 0.16                      | 0.03                      | 172/241, 0/0, 1117/771, 5507/158                     |
| 2    | 38990            | 7386           | 1074            | 0.19                      | 0.03                      | 176/212, 0/0, 1272/768, 5938/94                      |

**Push/Pop 级别（每线程）**

| 线程 | push 次数 | push 平均等待(μs) | push 平均持锁(μs) | pop 次数 | pop 平均等待(μs) | pop 平均持锁(μs) | pop 中 steal 占比 |
|------|-----------|-------------------|-------------------|----------|------------------|------------------|--------------------|
| 0    | 6049      | 0.23              | 0.17              | 5898     | 0.95             | 0.02             | 31.5%              |
| 1    | 5733      | 0.22              | 0.18              | 5781     | 0.95             | 0.03             | 33.5%              |
| 2    | 4922      | 0.29              | 0.20              | 5025     | 1.18             | 0.02             | 38.0%              |

### 简要分析（Ready queue 抢锁）

- **抢锁强度**：平均每次拿锁 **wait ≈ 0.15–0.19 μs**，**hold ≈ 0.03 μs**；单次 push/pop 在 ready queue 上的访问时间约为 **(wait + hold) ≈ 0.18–0.22 μs**，抢锁开销不大。
- **时间分布**：绝大部分 wait 来自 **dispatch**（pop 路径），少量来自 **complete**（fanout push）；**scan / orch** 的 wait 和 hold 都很小。
- **Push 与 pop 开销**：单次 push 平均持锁约 **0.17–0.20 μs**，单次 pop 平均持锁约 **0.02–0.03 μs**；pop 的“平均等待”略高是因为 dispatch 路径上多次拿锁的等待被摊到 pop 次数上。
- **Work stealing**：约 **31–38%** 的 pop 来自偷取，说明 3 个 shard 间负载较均衡。

---

### 调度开销 break-down（Scheduler overhead）

同一份 device log 中，每线程 **scheduler 总时间** ≈ 34.3 ms，由四段相位 + yield 组成。下面按**相位**拆解，并标出其中 **ready queue 锁 (wait+hold)** 占该相位的比例，便于区分“抢锁”与“其它调度逻辑”。

**1）按相位汇总（以 Thread 0 为例，单位 μs）**

| 相位 | 时间(μs) | 占 total 比例 | 其中 ready_q lock (wait+hold) | lock 占该相位比例 |
|------|----------|----------------|--------------------------------|-------------------|
| dispatch | 18461.7 | **53.8%** | 5605+144=5749 | 31.2% |
| complete | 14201.0 | **41.4%** | 1207+776=1983 | 14.0% |
| scan | 1428.0 | 4.2% | 170+274=444 | 31.1% |
| orch_drain | 11.6 | 0.0% | 0 | — |
| yield | 189.6 | 0.6% | — | — |
| **合计** | **34291.9** | 100% | **8177** | **23.8%** |

Thread 1、2 数值类似：total ≈ 34.3 ms，ready_q lock 合计约 7.9–8.5 ms/线程，占 **总调度时间约 23–25%**。

**2）各相位含义与“非锁”部分在做什么**

- **Dispatch（~54%）**  
  - 总时间 ~18.4 ms；其中 **~5.7 ms 为 ready queue 锁**（pop 时抢锁），其余 **~12.7 ms** 为：轮询 AIC/AIV core 状态、从 ready 取到 task 后的 resolve、下发 kernel 到 AICore、写 perf 等。
- **Complete（~41%）**  
  - 总时间 ~14.2 ms；其中 **~2.0 ms 为 ready queue 锁**（fanout 完成后 push），其余 **~12.2 ms** 为：遍历 fanout 链表、更新 consumer fanin、判断是否 ready、写完成状态等（**lock(fanout)** 本次为 0，无争用）。
- **Scan（~4%）**  
  - 总时间 ~1.4 ms；其中 **~0.44 ms 为 ready queue 锁**（扫描时 drain 到 ready），其余为扫描 task 状态、判断是否可入队等。
- **Orch_drain**  
  - 可忽略（~10 μs）。
- **Yield（~0.6%）**  
  - 本线程无任务可做时 yield，与锁无关。

**3）结论（调度开销 break-down）**

- 单次 ready queue 访问（push/pop）的锁开销 ≈ **0.18–0.22 μs**，抢锁不重。
- 调度总时间 ~34.3 ms/线程里：
  - **~23–25%** 是 **ready queue spinlock**（wait+hold）；
  - **~75–77%** 是 **其它调度逻辑**：dispatch 的轮询+resolve+下发、complete 的 fanout 遍历与更新、scan 的扫描与入队判断等。
- 若进一步优化调度，可优先看 **dispatch 与 complete 的非锁路径**（轮询方式、fanout 遍历与缓存、resolve/launch 开销），其次才是 ready queue 锁本身。

**4）按每 task 平均（Per-task averages）**

将上述各相位时间除以该线程 **completed** 任务数，得到“每完成一个 task、该线程在调度上平均花费的时间”（单位 μs/task）。同一份 log 数据：

| 线程 | completed | total | dispatch | complete | scan | yield | ready_q lock (wait+hold) |
|------|-----------|------|----------|----------|------|-------|---------------------------|
| 0 | 5898 | **5.81** | 3.13 | 2.41 | 0.24 | 0.03 | **1.39** |
| 1 | 5781 | **5.93** | 3.19 | 2.48 | 0.23 | 0.03 | **1.38** |
| 2 | 5025 | **6.83** | 3.75 | 2.85 | 0.19 | 0.03 | **1.68** |

即：**每完成 1 个 task，调度侧平均约 5.8–6.8 μs**；其中约 **1.4–1.7 μs** 为 ready queue 锁，约 **3.1–3.8 μs** 为 dispatch（轮询+resolve+下发），约 **2.4–2.9 μs** 为 complete（fanout 遍历+push 等），scan/yield 合计约 0.2–0.3 μs/task。

**5）每 task 平均执行时间与调度开销汇总**

同一次运行中，host 端 **Task Statistics**（swimlane 输出）给出 AICore 上 kernel 的 **Total_Exec**；device log 给出三线程 **scheduler 总时间**。对 16704 个 task 做平均：

| 指标 | 计算 | 数值 |
|------|------|------|
| 总 task 数 | — | 16704 |
| **每 task 平均执行时间**（AICore kernel） | Total_Exec / 16704 | **27849.48 / 16704 ≈ 1.67 μs** |
| **每 task 平均调度开销**（AICPU 调度循环） | (Thread0+Thread1+Thread2) total / 16704 | **(34292+34289+34297) / 16704 ≈ 6.16 μs** |
| 调度/执行比 | 调度 / 执行 | **6.16 / 1.67 ≈ 3.7** |

即：平均每个 task 在 AICore 上执行约 **1.67 μs**，在 AICPU 调度循环里摊到约 **6.16 μs**；调度开销约为 kernel 执行时间的 **3.7 倍**。调度开销的细分（dispatch / complete / ready_q lock / scan / yield）见上表「4）按每 task 平均」各列。

若需复现或对比：用 `--enable-profiling` 跑一次，再到 `$HOME/ascend/log/debug/device-<id>/` 下找最新 `device-*.log`，用上述 grep 提取 `lock(ready_q)` 与各 phase 行；Task Statistics 见 run 终端输出的 `Task Statistics by Function` 表（Total_Exec、总 task 数）。

---

## 7. halMemCtl failed (rc=13) 与“运行不结束”

### 现象

启动时日志出现：

- `[ERROR] get_aicore_reg_info: halMemCtl failed with rc=13`
- `[ERROR] get_aicore_regs: get_aicore_reg_info failed, using placeholder addresses`
- `[INFO] init_aicore_register_addresses: Successfully initialized ... 72 addresses at device 0x...`（24/48 配置下为 72）

之后用例**一直不结束**，或设备无有效计算。

### 原因

`halMemCtl` 用于向 CANN HAL 查询 AICore 寄存器映射的虚拟地址。返回 **rc=13** 时（多为权限或资源被占用），host 侧会走 **fallback**：用占位地址 `0xDEADBEEF...` 填满 24 AIC + 48 AIV（72 个）寄存器基址并拷到设备。

这些占位地址**不是**真实的 AICore MMIO 基址，AICPU 下发 kernel 时写的是无效地址，导致：

- AICore 上 kernel 实际未执行，或
- 访问非法地址导致设备异常/挂起，

因此调度循环一直在等 AICore 完成，**运行无法正常结束**。

### 建议

1. **确认环境**：用与安装 CANN 一致的用户运行；同一设备不要被其他进程独占。
2. **查 CANN 文档**：针对 `halMemCtl` 错误码 13 的官方说明（常见为权限不足或设备状态异常）。
3. **不要依赖 placeholder**：出现 “using placeholder addresses” 时，当前 run 不能作为有效执行/性能数据；需解决 HAL 报错后再跑。
4. **若仅做 AICPU 或仿真**：若用例不依赖真实 AICore 寄存器（例如纯 sim 或仅测 host 路径），可暂时忽略；paged attention 依赖 AICore 执行 QK/PV 等 kernel，**必须**拿到真实寄存器地址才能跑通。

---

## 8. 卡在 rtStreamSynchronize stream_aicpu_

### 现象

日志中已出现：

- `Retrieved 24 AIC and 48 AIV register addresses`
- `=== launch_aicpu_kernel DynTileFwkKernelServerInit===`
- `=== launch_aicpu_kernel DynTileFwkKernelServer===`
- `=== launch_aicore_kernel===`
- `=== rtStreamSynchronize stream_aicpu_===`

之后进程**一直不返回**，需 ^C 中断。

### 含义

Host 在等待 **AICPU stream** 上提交的任务全部完成。卡住说明 AICPU 侧（调度循环 + AICore 执行）要么未正常结束，要么设备/驱动无响应。

### 可能原因

- **设备异常或占用**：部分 device ID（如 13、15）在该环境下可能挂起、被占用或与 HAL/驱动配合异常。
- **AICPU 调度或 AICore 执行死循环/死锁**：依赖未满足、完成信号未正确写回等（若多数设备都卡则重点排查）。

### 建议

1. **换 device**：优先使用在本机已验证能跑通的 device（例如 device 0）。例如：
   ```bash
   PA_CASE=Case1 python examples/scripts/run_example.py -k ... -g ... -p a2a3 -d 0
   ```
3. **查看 device 日志**：`$HOME/ascend/log/debug/device-<id>/` 下最新 `device-*.log`，搜索 `error`、`fail`、`507015` 等，确认设备侧是否有报错或超时。
4. **确认 24/48 配置**：日志中应为 “24 AIC and 48 AIV”；若曾出现 25/50，需确保已用当前代码重编并重跑。
