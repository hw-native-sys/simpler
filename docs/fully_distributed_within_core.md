# AICore 上的全分布式 Runtime

本文档定义 **simpler** 的一种运行模式：编排（orchestration）、调度（scheduling）
与执行（execution）全部以 SPMD 方式运行**在 AICore 自身**之上，**AICPU 完全不参与**。
不存在独立的调度器：每个核自行构建、拥有并执行自己的任务。

这是一份自洽的设计。第一部分描述系统如何工作（核的行为 + 伪代码）；第二部分列举
各数据结构及其共享特性（全局共享 / 每核私有 / 每核复制）。

本设计所替代的、当前以 AICPU 为中心的模型，参见
[chip-level-arch.md](chip-level-arch.md) 与 [scheduler.md](scheduler.md)。编排编写
API（`rt_submit_aic_task` / `rt_submit_aiv_task`，`pto_orchestration_api.h`）参见
`src/{arch}/runtime/` 下的 `tensormap_and_ringbuffer` runtime。

---

# 第一部分 — 系统设计

## 1. 概述

- 编排函数**被加载并同时运行在每一个参与的 AICore 上**（SPMD）。所有核执行完全相同
  的编排程序。
- 每个核同时是**编排器 + 调度器 + worker**。经典的“调度器↔worker”握手（任务门铃、
  ready 队列、完成邮箱、依赖连线线程）被**彻底取消**。
- 面向编排的 API 保持不变。通用原语是 `rt_submit_task(MixedKernels, args)`；
  `rt_submit_aic_task` / `rt_submit_aiv_task` 只是它的轻量便捷封装（**不存在**
  `rt_submit_mixed_task`——MIX 任务就是一个填了多个 kernel 槽的 `MixedKernels`）。
  在这些 API 背后，runtime 决定所有权、在本地构建任务，随后由同一个核执行它。
- AICPU 不在编排与调度的关键路径上。

本设计建立在以下四个支柱之上（下文逐一展开）：

1. 任务所有权的**抢占竞争（claim race）**（§2）。
2. **owner = builder = executor**，并配合核类型匹配（§3）。
3. 用于依赖发现的**每核全量复制 TensorMap**（§4）。
4. **每核私有任务环 + 一个全局完成标志环**，驱动一个采用拉取式依赖解析的
   run-ahead 执行循环（§5–§6）。

## 2. 任务所有权 —— 抢占竞争（Claim Race）

所有核走**完全相同**的、确定性的 submit 序列。任务身份就是它在该序列中的位置：第 N 次
`rt_submit_*` 调用在每个核上都是**任务 id `N`**，与最终由谁执行无关。

所有权由以下两个量驱动：

| 计数器 | 作用域 | 含义 |
| ------ | ------ | ---- |
| `claim_cursor[T]`（`cube_cursor`、`vector_cursor`） | **全局、原子** | 类型 `T` 已被认领任务 id 的高水位线。共**两个** cursor（cube = AIC-anchored，vector = AIV-only），二者都索引同一个共享 id 空间（§3.1） |
| `local_current_task_index` | **每核** | 本核走 submit 序列时当前到达的任务 id |

每次 `rt_submit_*`，匹配 anchor 类型的核执行如下逻辑（设 `T` 为此任务类型——若 AIC-anchored
则为 cube，若 AIV-only 则为 vector）：

```text
local_current_task_index++                        # 到达下一个 submit 点 = 任务 id N
if local_current_task_index > claim_cursor[T]:    # 我是否领先于 T 的高水位线？
    # 本核是 T 类型中走得最靠前的 → 它 WIN，拥有任务 N。
    claim_cursor[T] = local_current_task_index     # 发布（原子）
    own = true
else:
    # 已有一个 T 类型的核更早认领了此 id（它跑在前面）。
    own = false
```

胜者是该任务 id 的唯一 owner。所有权决定的是*谁来构建与执行*；它**不会**改变任务 id——
该 id 是处处使用的确定性 submit 序号（完成标志环的索引、以及每个核的 producer 引用）。
对于多核任务，胜者是 *anchor*；与它配对的同 block 核共同拥有其余子任务（§3.1）。

为什么需要两个 cursor（以及为什么单一共享 cursor 是错的）在 §3.1 解释：两个 cursor 扫过
同一 id 空间，各自只认领自己类型的 id，并**跨过**另一类型的 id，因此落后类型尚未认领的
id 只是在等待它自己的 cursor —— 它们绝不会被跳过。

> 确切原子原语（`atomic_fetch_max`，无则 CAS 回路）与内存序在 §11.1 定为规范；
> 语义上每个任务 id 恰好有一个 anchor 胜出。

## 3. owner = builder = executor；核类型匹配

**抢到任务的提交者就是它的 owner。** owner 同时负责任务的**创建**（构建
descriptor/payload、记录 fan-in producer id）与**执行**（调用 incore 函数）。一个核只会
认领它自己能执行的类型的任务。

任务由 `MixedKernels` 描述，最多携带三个子任务槽：

```cpp
struct MixedKernels {
    int32_t aic_kernel_id  { INVALID_KERNEL_ID };   // AIC 子任务
    int32_t aiv0_kernel_id { INVALID_KERNEL_ID };   // AIV 子任务 0
    int32_t aiv1_kernel_id { INVALID_KERNEL_ID };   // AIV 子任务 1
};
```

`active_mask` = 哪些槽有效，它恰好记录了一个 MIX 任务的 AIV 数量——**1C+1V**
（`aic` + `aiv0`）还是 **1C+2V**（`aic` + `aiv0` + `aiv1`）。这一区分对所有权很关键：
1C+1V 任务只绑定 AIV0_c，让 AIV1_c 保持空闲（§3.1）。因此任务是以下之一：AIC-only、
AIV-only（1 个或 2 个 AIV 子任务）、或 **MIX**（AIC + 1 个或 2 个 AIV 子任务）。

| 任务形态 | 子任务槽 | owner |
| -------- | -------- | ----- |
| **AIC-only** | `aic` | 任意一个 AIC 核 |
| **AIV-only (1V)** | `aiv0` | **任意一个 AIV 核（AIV0 或 AIV1）** |
| **AIV-only (2V)** | `aiv0`、`aiv1` | 同一 block 的两个 AIV 核 |
| **MIX (1C+1V)** | `aic`、`aiv0` | 一个 AIC + 同 block 一个 AIV（共同 owner） |
| **MIX (1C+2V)** | `aic`、`aiv0`、`aiv1` | 一个 AIC + 同 block 两个 AIV（共同 owner） |

单槽封装（`rt_submit_aic_task` → 填 `aic`，`rt_submit_aiv_task` → 填 `aiv0`）是常见路径；
多槽任务直接走 `rt_submit_task(MixedKernels, …)`。

**单核 vs 多核——竞争资格按“类型”而非“固定槽角色”。** 竞争一个任务的资格由任务**类型**
（cube / vector）决定，而非某个具体的 `aiv0`/`aiv1` 角色：

- **单核任务（1C、1V）**：没有配对、没有 anchor/follower。任意一个**匹配类型**的核通过 §2 的
  claim race 认领，胜者独自构建并执行那唯一的子任务。特别地，**1V（AIV-only 单核）由所有 AIV 核
  竞争——AIV0 与 AIV1 同等参与**；胜者执行 `aiv0_kernel_id`，与它在 block 中是 AIV0 还是 AIV1
  无关（两者都是 vector 核，可执行任意 AIV kernel）。
- **多核任务（2V、MIX）**：需要同一物理 block 的多个核共同拥有，走 §3.1 的固定配对（anchor 胜出
  后把其余子任务推送给同 block 伙伴）。

换言之，`aiv0`/`aiv1` 的“固定角色”**只**在多核任务里用来把子任务映射到 block 内具体的核；对单核
任务它不构成竞争限制。

### 3.1 通过固定物理配对实现多核任务的共同所有权

本节**只针对多核任务**（任意 MIX 任务，以及 2V 的 AIV-only 情况）——它们含多于一个有效子任务
槽，必须被多个核同时拥有。单核任务（1C、1V）不走本节机制：由任意匹配类型的核（1V 即任意 AIV 核
AIV0/AIV1）通过 §2 的 claim race 直接认领、独自执行，无 anchor/follower。本节规定多核任务的
共同 owner 如何被选出、如何达成一致——这是模型中最难的部分。

**配对被 FIXED（固定）到硬件 block。** 核被组织成硬件 block（cluster）；在本平台上一个
block = **1 AIC + 2 AIV**（AIV0、AIV1）。这个 block 是永久的共同所有权单位：AIC_c 与
AIV0_c、AIV1_c 静态配对。不存在动态配对选举。子任务槽到 block 内角色是固定映射：

| 子任务槽 | 由谁执行（block `c` 内） |
| -------- | ------------------------ |
| `aic_kernel_id` | AIC_c |
| `aiv0_kernel_id` | AIV0_c |
| `aiv1_kernel_id` | AIV1_c |

**Anchor + 同 block 跟随规则。** 一个多核任务只被**认领一次**，由一个 *anchor* 核认领；
其 block 的其余核跟随：

1. **谁竞争（anchor 类型）**：竞争按任务**类型** `T` 进行——含 AIC 子任务的任务（所有 MIX）
   是 **cube 类型，只有 AIC 核竞争**；纯 AIV 的 2V 是 **vector 类型，由所有 AIV 核（AIV0/AIV1）
   竞争**。胜出者即该任务的 **anchor**，它执行**自己物理角色**对应的那个槽（AIC 胜者执行 `aic`；
   2V 由某个 AIV 胜出则执行它自己角色的 `aiv0`/`aiv1`），其余激活槽推送给同 block 伙伴。
   **MIX 的 vector co-owner 绝不靠自己竞争得来**——它*完全*由“哪个 AIC 胜出”决定，即由胜者
   所在的 block 决定（一个 AIV 核绝不会因为先到达就赢得某 MIX 的 vector 子任务）。
2. 抢占竞争（§2）**仅在 anchor 类型之间**进行，竞争对象是 `cursor[T]`。胜出的 anchor 核
   所在的 **block** 成为拥有该任务的 block。anchor 在胜出时**一次性解析整个任务的 fan-in**
   producer id（从它在 `N` 处的 TensorMap 副本读取，各核内容相同——§4），把*自己*那个槽的
   子任务构建进自己的私有环，并把该任务**其余激活槽**的子任务记录**推送（deposit）**进一张
   **以任务 id 为键的 block-local 投递表** —— `block.won[N]` —— 内容为
   `{active_mask = M, 各激活槽 kernel id, args, 已解析的 fan-in producer id, 剩余子任务计数
   = popcount(M)}`。
3. 同 block 的 follower 核**既不竞争、也不在自己的编排走位上对该任务做“等待 anchor 决定”
   的判断**——它**永不因 anchor 而阻塞**。follower 的所有权完全靠 anchor 的**推送**到达：
   follower **异步地从 `block.won` 抽取（drain）**属于自己槽的子任务投递，在私有环有空槽时
   把它构建进环。follower 在自己的编排走位中遇到 MIX 任务时，只做 §4 的无条件 TensorMap
   更新，然后继续前进，**不**对该 MIX 任务做任何所有权决定、**不**等待它的 anchor。

**为什么是 anchor 推送，而不是 follower 自己走位 + 等待。** 两个 cursor 独立推进（§2），所以
cube 与 vector 的进度可能任意错位。若让 follower 在自己的走位上“走到 N 再判断我的 block 是否
赢了 N”，当它的 anchor 落后（`cube_cursor < vector_cursor`）时，follower 就无法区分“anchor
还没决定 N”与“anchor 输了 N（别的 block 赢了）”，只能**阻塞等待** anchor 推进到 N——这会把
vector 的吞吐死死耦合到 cube 的吞吐上，是不可接受的。**改为 anchor 推送即彻底消除这种 per-task
阻塞**：

- **cube 落后时**：`block.won` 里还没有给这个 AIV 的 MIX 投递 → AIV **不等待**，继续竞争并执行
  它自己的 AIV-only 任务（以及抽取已到的其他投递）。零停顿。
- **cube 领先时**：投递在 `block.won` 中累积 → AIV 有空槽就抽取构建。若 AIV 落后到填满
  `block.won`，则 anchor **暂缓认领新的多核任务**（反压；见 §6 中 anchor 转去执行 Phase B 而
  非自旋），方向正确：不让 cube 无限超前。

`block.won` 以任务 id 为键（而非单一会被覆盖的槽），既承载每任务的剩余子任务计数，也允许同一
block 多个并发多核任务的投递互不串扰。由于配对是静态的，投递的目标 follower 由 anchor 所在
block 唯一确定，无需任何跨 block 协商。

> 唯一残留的等待发生在**收尾**：若某 block 的 anchor 严重落后，它的 follower 在做完自己其余
> 全部工作、私有环清空后，可能要在终止前空转，等 anchor 把最后的多核子任务推送过来（§7）。
> 这是固定配对的固有代价——多核子任务的归属由 anchor 的认领决定；它不是 per-task 的串行阻塞，
> 而只是尾部的一次空转，且在 cube 密集（cube 领先）的常见场景下根本不出现。

**按形态的行为（设胜出 anchor 在 block `c`）：**

| 任务形态（`active_mask`） | 谁竞争 | Anchor（胜者） | 被推送子任务的 follower | 同 block 未被绑定（保持空闲） |
| ------------------------- | ------ | -------------- | ----------------------- | ----------------------------- |
| **1C + 2V**（多核） | 所有 AIC | AIC_c | AIV0_c、AIV1_c | — |
| **1C + 1V**（多核） | 所有 AIC | AIC_c | AIV0_c | **AIV1_c** |
| **2V**（多核，AIV-only） | 所有 AIV（AIV0/AIV1） | 胜出的那个 AIV_c | 同 block 的另一个 AIV_c | AIC_c |
| **1C**（单核，AIC-only） | 所有 AIC | 胜者独自执行，无配对 | — | （不涉及 block 配对） |
| **1V**（单核，AIV-only） | **所有 AIV（AIV0/AIV1）** | 胜者独自执行，无配对 | — | （不涉及 block 配对） |

多核任务（前三行）的 follower 身份都由 anchor 所在 block 唯一确定——不存在跨 block 协商。单核
任务（后两行）没有 anchor/follower，胜者是哪个核就由哪个核独自执行；**1V 由 AIV0 与 AIV1 同等
竞争**。

**未被绑定的 block 伙伴不是闲着——它对其他任务保持空闲可用。** 当一个 block 赢得一个不激活
某 block 伙伴槽位的任务时，那个核就**不被该任务占用**，且**绝不能**因它而阻塞或等待。它继续
运行自己的编排，继续竞争并拥有其类型的其他任务。具体地：

- 一个 **1C+1V** 任务只绑定 AIC_c + AIV0_c。**AIV1_c 是空闲的**，可继续竞争、认领并执行其他
  AIV 任务（它自己竞争到的任意 1V/2V AIV-only 任务，或本 block 后续某个 1C+2V 任务的 AIV1 槽）。
- 一个 **1C（AIC-only）** 任务只绑定一个 AIC 核；AIV 核**都**对 AIV 工作保持空闲。
- 一个 **1V（AIV-only）** 任务是单核：由**任意一个 AIV 核（AIV0 或 AIV1）**竞争得到并独自执行，
  其余 AIV 核与 AIC 核保持空闲。它不绑定任何固定角色。

这是模型的自然结论：每个核都走相同的确定性 submit 序列，并逐任务判断自己的槽是否激活。在某个
自己的槽未激活的 submit 点，该核就是不绑定该任务（但它仍执行 §4 的无条件 TensorMap 更新），
然后继续——去认领它下一个有资格的任务。每个任务记录的 `active_mask`（1C+1V vs 1C+2V 等）
就是告诉每个 block 伙伴自己是被绑定还是空闲的依据。

**多核任务只有一个完成标志。** 即使有多个共同 owner，一个任务也恰好只有一个全局
`task_completed_flag[N]`。每个共同 owner 执行自己的子任务后，递减 `block.won[N]` 中那个用
`popcount(active_mask)` 初始化的**per-task 剩余计数器**。（该计数器存在以 id 为键的记录里，
而非单一 block 字段，因此同一 block 的多个并发 MIX 任务不会互相串扰。）把计数器递减到零的那个
共同 owner（最后完成的子任务）执行唯一一次全局写 `task_completed_flag[N] = true`。因此无论
任务有多少个子任务，消费者都只看到一个原子的完成信号。每个共同 owner 在自己的子任务完成后
立即释放自己的私有环槽位。

**Claim 流一致性 —— 同一任务 id 空间上的两个全局 cursor。**

只有**一个**任务 id 空间——确定性 submit 序列（第 N 次 submit = id `N`），处处用于完成标志
环与 producer 引用。

所有权由**两个全局 claim cursor** 决定，二者都由所有核共享，且都索引进*同一个* id 空间：

- `cube_cursor` —— 已认领的 **cube（AIC-anchored）** 任务 id 的高水位线（AIC-only 与所有
  MIX 任务）。
- `vector_cursor` —— 已认领的 **vector（AIV-only）** 任务 id 的高水位线。

一个到达类型 `T` 的任务 `N` 的核，当且仅当 `N > cursor[T]` 时赢得它；赢得后把 `cursor[T]`
推进到 `N`。一个核只会推进它自己类型的 cursor；它**跨过**另一类型的 id 而不去碰它。

两个 cursor 在共享 id 空间上**独立**推进，因此任意时刻其中一个可能领先于另一个。**推进一个
cursor 不会认领它跨过的另一类型的 id。** 因此在领先 cursor 与落后 cursor 之间的 id 区间里
可能存在**尚未认领的空洞**——这些是*落后*类型的、还没有任何核到达的 id。这是正确的，不是 bug：
一个空洞只表示“暂时还没认领”；当一个该类型的核到达它时，落后类型的 cursor 会把它填上。

```text
任务 id:      0    1    2    3    4    5    6
类型:         C    V    C    C    V    V    C
                              ^cube_cursor=3        (cube 任务 0,2,3 已认领)
                   ^vector_cursor=1                 (vector 任务 1 已认领)
空洞: id 4 和 5 是位于 cube_cursor 之下的 vector 任务——仍 UNCLAIMED，
      等待 vector_cursor 推进到它们。没有 orphaning。
```

在单一类型内部不存在空洞：每个核按 id 递增顺序遇到该类型的任务，而 cursor（一个单调高水位线）
总是被设为刚刚认领的那个 id——因此该类型中所有 ≤ 其 cursor 的 id 都已被某个核拥有。（计数器的
确切表示属于实现细节——§11。）

**取舍。** 固定配对消除了一切跨 block 协商，并把唯一的共享协调状态保持在 **block-local**
（1 AIC + 2 AIV 共享一小块区域），而非全局 per-task。代价是多核任务没有跨 block 的负载均衡；
动态配对方案是未来的改进（§11）。

### 3.2 为什么 vector 不竞争 MIX（以及“不会缺失 co-owner”的论证）

> 这一节直接回答一个常见疑问：既然 vector 不参与 MIX 的竞争，会不会出现“cube 认领了某个 MIX
> 任务，却没有任何 vector 核作为它的 co-owner”？答案是**不会**。并解释为什么不采用“让 vector
> 也竞争 MIX”或“先到先得、由后到的同 block cube 反向认领”的替代方案。

**结论一：vector 核不参与 MIX 的竞争。** MIX 永远 cube-anchored（§3.1）。vector 核遇到一个
MIX 任务时走的是 follower 路径：它**不**碰 `vector_cursor`，只按 id 查 `block.won[N]`，看自己
所在 block 的 AIC 是否赢了。它“先到达” MIX 任务这件事不授予它任何东西。

**结论二：永远不会缺失 vector co-owner。** 原因有三条，缺一不可：

1. MIX 任务是 cube 任务，**只**会推进 `cube_cursor`。`vector_cursor` 永远不认领 MIX 任务——
   即便 `vector_cursor` 追上甚至越过 `cube_cursor`，它也只是在认领它路过的 *AIV-only* 任务，
   绝不会“占用”任何 MIX 任务。所以不存在“被 vector_cursor 抢走却没有 vector 执行者”的 MIX 任务。
2. 当某个 AIC 核 `AIC_x` 赢得 MIX 任务 `N` 时，它的 vector co-owner 由**固定物理配对**确定：
   就是同 block 的 `AIV0_x`（若 1C+2V 还有 `AIV1_x`）。这个身份在胜负确定的瞬间就被钉死，
   不需要任何额外竞争或选举。
3. 当 `AIC_x` 赢得 `N` 时，它把 `AIV0_x`（及 1C+2V 的 `AIV1_x`）的子任务**推送**进
   `block.won[N]`（§3.1）；`AIV0_x` 异步抽取并执行。**co-owner 的存在是被保证的。**

**那么 `vector_cursor` 追上 `cube_cursor` 时究竟会发生什么？会不会变成 blocking wait？**
不会。注意 MIX 归属靠 **anchor 推送**而非 follower 走位判断（§3.1），所以：

- **cube 落后（`cube_cursor < vector_cursor`）时**：AIC 还没认领 `N`，因此 `block.won` 里还没有
  给 AIV 的投递。AIV **不阻塞、不空等**——它继续竞争并执行自己的 AIV-only 任务，同时抽取已到的
  其他投递。它在自己的走位上遇到 MIX 任务时只做 TensorMap 更新就走，**不**对该任务做归属判断、
  **不**等待它的 cube 伙伴。
- 等 AIC 日后认领到 `N`，投递才出现在 `block.won`，AIV 再抽取执行。

换言之，不存在“AIV 走到 MIX 任务就 blocking wait 到 cube 追上来”的情况——这正是把旧设计的
`wait_until(block.anchor_progress >= N)` 去掉、改为推送的原因。唯一残留的等待是**尾部空转**
（§3.1、§7）：若某 block 的 AIC 严重落后，AIV 做完其余全部工作后会在终止前等 AIC 推送最后的
多核子任务。这不是 per-task 串行阻塞，且 cube 领先（常见）时根本不出现。

**为什么不让 vector 也竞争 MIX（方案 A）。** 因为 MIX 的 AIC 与 AIV 子任务必须在**同一物理
block 内协同执行**（共享 local memory / 相互配合，这正是固定配对的意义），所以所有权的单位
是 **block**，不是单个核。若允许 vector 核也去 anchor 一个 MIX 任务，会立刻破坏 §2 的 cursor
不变式：

- 若让 vector 核去推进 `cube_cursor` 来认领 MIX，它就会把位于旧 `cube_cursor` 与 `N` 之间的
  那些 **cube-only 任务 orphan 掉**（跳过且无人认领）——这正是双 cursor 设计要避免的问题。
- 若让 vector 核在 `vector_cursor` 上 anchor MIX，而某个 cube 核同时在 `cube_cursor` 上 anchor
  同一个 MIX `N`，那么同一任务会被两个 cursor 各认领一次 → **两个不同的 block 都认为自己拥有
  `N`**（跨 block 撕裂 / 双重认领）。错误。

因此结论是：**每一类任务必须只有一个 anchor 类**。MIX 选 cube 作为唯一的 anchor 类，保证
claim 是单写者、无 orphan、无跨 block 双重认领。

**为什么“先到先得 + 后到的 cube 反向认领”（方案 B）也不采用。** 这个想法只能作为 **block
内部**的“探测优化”（block 内谁先到达 `N` 谁就代表本 block 发布认领），而**不能**跨 block——
跨 block 的正确性仍然要求一条单一的 claim 流，且该流必须是 cube 的（否则就 orphan 掉 cube-only
任务，同方案 A）。也就是说，即便 block 内允许 vector 先“代发布”，真正权威的 anchor 流仍是 cube
的 `cube_cursor`。其收益只是偶尔省去 follower 的一次等待，却显著增加了 block 内两条 cursor
交叉认领的复杂度与正确性论证负担。因此当前**不采用**，仅在 §11 作为未来可选优化列出。

> 一句话总结：vector 不竞争 MIX 是**有意为之**的正确选择。co-owner 由固定配对保证存在；让
> vector 参与只会重新引入 orphan 或跨 block 双重认领。需要权衡的不是“会不会缺 co-owner”，而是
> cube 落后时 follower 的等待——这属于负载均衡/性能问题，留待动态配对方案（§11）解决。

## 4. 依赖发现 —— 每核全量复制 TensorMap

依赖与今天完全一样，从 tensor 的读/写重叠推导，途径是一个把 tensor 区域映射到其
**producer 任务 id** 的 **TensorMap**。本 runtime 的决定是：

> **TensorMap 是每核全量 DUPLICATE（复制）—— 每个核持有一份完整、相同的副本。它绝不被
> 分区，也绝不做成私有/部分。**

**为什么部分 map 是错的。** producer 条目只在处理某任务的 `OUTPUT`/`INOUT` tensor 时创建。
若一个核只为它*拥有*的任务插入，它的 map 就会缺失所有由别的核拥有的任务产出的 tensor；本核
上的某个消费者去查这样一个 tensor 会查不到——依赖发现会悄无声息地失效。

**所要求的 submit 行为（胜者 AND 败者都做）。** 为保持副本完整，submit 路径被拆分：TensorMap
维护是**无条件**的，只有 build+execute 才受所有权门控。每次 `rt_submit_*`，*每个*核都做：

1. **查**每个 `INPUT` / `INOUT` tensor → 解析出本任务的 fan-in producer 任务 id。
2. **插**每个 `OUTPUT` **以及 `INOUT`** tensor → 以**本任务 id**作为 producer 登记。`INOUT`
   两侧都算——它消费旧版本（第 1 步）并产出新版本（第 2 步）。

**胜者**额外构建并执行该任务；**败者**在 TensorMap 更新后停止并前进。

因为 submit 流与任务 id 在各核之间是确定且相同的，每个核重建出**相同**的 TensorMap。各核仅在
**进度**上不同：跑得更靠前的核有更多条目，但每个条目都与其他核在同一逻辑位置产出的一致——
**内容相同，进度不同**。

**取舍。** 每个核都要付出完整的 TensorMap 插入/查询开销与内存，即使是它永远不会执行的任务。
作为回报，解析 producer **零跨核通信**：消费者的 fan-in producer id 在本地副本里就能拿到，在
构建时存入任务的私有环槽位，执行时再对全局完成标志环轮询。

## 5. 任务存储 —— 私有环 + 全局完成标志

AICPU 模型的全局任务环被移除。两个结构替代它们：

- **每核私有任务环** —— 每个核拥有一个**小**环，存放它已认领的任务，保存每个任务的
  descriptor + payload + 本地状态（kernel id、args、fan-in producer id）。其他核都不读它；
  无锁。容量：

  ```cpp
  #define PRIVATE_TASK_SLOT_NUM 4   // 故意取小：见下方“为何要小”与 §6.1
  ```

  **这个容量是关键调优旋钮，不是越大越好。** 全系统的乱序窗口 = **核数 × `PRIVATE_TASK_SLOT_NUM`**，
  同时它也封顶了**单个核能比“当前就绪可执行”超前认领多少个任务**。把它开大会让某个快核一口气
  抢入一长串连续任务再独自串行执行，造成严重负载倾斜（详见 §6.1）。因此应**保持其很小**（如 2–4），
  让乱序能力主要来自“核数”维度；具体值按 kernel 时长 / 访存延迟实测调优。

- **全局 `task_completed_flag` 环** —— *唯一*全局共享的 per-task 状态：每个任务 id 一个
  一次性置位的布尔，标记完成。各核轮询它以检查某个 fan-in producer 是否已完成。

这使依赖解析成为**拉取（pull）**模型（消费者轮询 producer 标志），而非**推送（push）**模型
（producer 遍历 fanout 列表）。**没有 fanout 列表、没有 fanin/fanout 引用计数、没有依赖列表
池、也没有完成邮箱。**

### 5.1 私有任务环与 `block.won` 是两个分开的 ring

私有任务环与 `block.won`（§3.1、§8.1）**是两个独立的结构，职责不同，不可混为一谈**：

| | **私有任务环** | **`block.won[N]`** |
| ---- | ---- | ---- |
| 归属 | **每核私有**（每个 worker 各一个） | **block-共享**（1 AIC + 2 AIV 共一份） |
| 作用 | **执行队列**：存放本核已拥有、要*亲自执行*的（子）任务 | anchor → follower 的**投递/交接箱**：暂存多核任务中 anchor 没亲自构建的其余激活槽子任务 |
| 谁读写 | 仅本核读写，单一 owner、无锁 | anchor 插入（release）、follower 抽取（acquire）、`remaining` 原子递减 |
| 谁会用到 | 所有任务（含单核 1C/1V） | **仅多核任务（2V / MIX）**；单核任务根本不碰它 |
| 容量含义 | 默认小（如 4）：封顶“单核可超前多少”，故意取小以抑制负载倾斜（§6.1） | 默认 8：封顶“anchor 相对 follower 可超前多少”，满则触发反压（§11.2） |

**真正的执行永远只发生在各核自己的私有任务环里。** `block.won` 不是执行环，只是把多核子任务从
anchor **搬运**到 follower 私有环的中转站。两者如何配合：

```
anchor 赢下多核任务 N：
  ├─ 自己物理角色那一槽 ──→ 写进【anchor 自己的私有任务环】（亲自执行）
  └─ 其余激活槽          ──→ 写进【block.won[N]】（投递给伙伴）

follower 异步抽取：
  从【block.won[N]】取出属于自己槽的项 ──→ 写进【follower 自己的私有任务环】（再亲自执行）

子任务一旦进入某核私有环，其执行、置完成标志、block.won[N].remaining 递减都照常进行；
remaining 归零时释放该 block.won 条目。
```

单核任务（1C / 1V）的胜者直接把唯一子任务写进自己的私有环执行，**没有配对、没有投递、不写
`block.won`**。

## 6. 核执行循环（执行优先的 Run-Ahead）

每个核运行下面的循环。其核心准则是 **“执行优先、认领其次、一次只认领一个”**：每轮循环都
**先寻找执行机会**（腾空私有环里任何已就绪的任务），**再至多认领一个**新任务——而**不是**先把
私有环一口气抢满、再开始执行。编排仍会**向前跑（run ahead）**，但只在没有就绪任务可执行时才
逐个认领，借此把“单核超前认领”限制在很小的范围。这一改动的动机见 §6.1。

该循环从单个物理核 `self` 的视角写出，它在所在 block 中的角色是 `{AIC, AIV0, AIV1}` 之一。
竞争按**任务类型**进行（vector 任务由 AIV0/AIV1 同等竞争）；单核任务胜者独自执行，多核任务
胜者作 anchor 并把其余子任务推送给同 block 伙伴（§3、§3.1）。

> 术语对照：本文其余处（§3.1、§11）沿用旧称 **“Phase B”** 指代下方**步骤 1**（执行 / 腾空就绪
> 任务），**“Phase A”** 指代**步骤 2**（认领新任务）。差别仅在于:执行优先版**每轮只认领一个**、且
> **认领与执行严格交替**，不再“先把环填满再统一腾空”。

```text
# 全局（所有核共享），一个共享任务 id 空间（§2、§3.1）：
#   cube_cursor   : 已认领的 AIC-anchored 任务 id 高水位线
#   vector_cursor : 已认领的 AIV-only 任务 id 高水位线
# 每核：
#   self.role ∈ {AIC, AIV0, AIV1}
#   my_type(self) = cube  (若 self 是 AIC)  /  vector (若 self 是 AIV0 或 AIV1)
#   local_current_task_index : 本核已到达的任务 id

loop:
    # ============================================================================
    # 执行优先：每轮循环按 步骤0 → 步骤1 → 步骤2 顺序走，一轮只认领【一个】新任务。
    # 关键修正：不再“先把环填满再执行”。先腾空就绪任务（步骤1），再认领一个（步骤2）；
    # 认领后立刻回到循环顶部，下一轮又先找执行机会。核在执行一个长任务期间不推进认领，
    # 这段时间其它核会推进 cursor 认领后续任务 → 负载自然均衡（理由见 §6.1）。
    # ============================================================================

    # --- 步骤 0：抽取 anchor 推送给我的多核子任务（异步、非阻塞）---
    # 同 block 的 anchor 胜出某多核任务后，会把它没亲自构建的其余激活槽放进 block.won。
    # 本核按自己的物理角色（AIV0→aiv0 / AIV1→aiv1）抽取属于自己的那个槽。取空就停，不等待。
    while 私有环有空槽 AND block.won 有“我角色对应槽”尚未被本核构建的待处理项:
        从 block.won 取出该子任务，构建进一个空闲私有环槽    # fan-in 已由 anchor 解析好

    # --- 步骤 1：寻找执行机会，腾空就绪的（子）任务（执行优先）---
    # 每轮都先做这一步：只要 fan-in 已满足就执行，绝不等环填满才开始执行。
    freed = 0
    for each 私有环中已占用的槽:
        if 所有 fan-in producer 的 task_completed_flag == true:    # 依赖已满足（pull）
            execute(slot)                                          # 调用我的 incore 函数（长耗时）
            # 完成：多核任务只有一个全局标志，由其共同 owner 中最后完成的子任务置位（§3.1）。
            if slot.is_multicore:
                if atomic_dec(block.won[slot.task_id].remaining) == 0:
                    task_completed_flag[slot.task_id] = true       # 最后一个子任务胜出
                    free block.won[slot.task_id]                   # 回收以 id 为键的记录
            else:
                task_completed_flag[slot.task_id] = true           # 单核：直接置位
            free(slot)                                             # 释放我自己的槽；无 fanout 计数
            freed++

    # --- 步骤 2：至多认领【一个】新任务（仅当环有空槽且编排未结束）---
    # 一次只认领一个，认领后立即回到步骤 0/1 找执行机会，避免一口气把环抢满。
    # 若步骤 1 没有就绪任务可执行（freed==0），步骤 2 仍会认领一个 → 这就是受控的 run-ahead：
    # 没活可干时才逐个超前认领，且超前量被私有环容量（很小）封顶。
    if 私有环有空槽 AND 编排未结束:
        推进编排到下一个 submit 点                            # 任务 id N
        local_current_task_index = N
        M = task.active_mask                                  # 记录 1C+1V vs 1C+2V 等

        # (a) TensorMap 维护是无条件的（胜者、败者、follower 都做）—— §4：
        #     - 查 INPUT/INOUT tensor    → fan-in producer 任务 id
        #     - 插 OUTPUT + INOUT tensor → 以本任务 id 作为 producer
        update_tensormap(task)

        # (b) 确定本任务的类型与 cursor（§2、§3）：cube 任务由 AIC 竞争；
        #     vector 任务由所有 AIV 核（AIV0 与 AIV1）竞争。
        T         = (cube if M.has(aic) else vector)          # 有 AIC → cube；否则 vector（含 1V 与 2V）
        cursor[T] = (cube_cursor if T==cube else vector_cursor)

        if my_type(self) == T:
            # 我是该类型的合格竞争者（vector 任务时 AIV0/AIV1 都在此参与）。
            if popcount(M) > 1 AND block.won 已满:             # 多核反压：本轮不认领（§11.3）
                pass                                          # 留待步骤 1 腾空 block.won 后的下一轮再试
            else:
                # 单原子推进：返回旧值；旧值 < N 即我赢。恰一胜者且无跳过见 §11.1。
                old = atomic_fetch_max(cursor[T], N)          # N = local_current_task_index
                if old < N:                                   # WIN：我是 owner/anchor
                    fanin_ids = resolve_fanin(task)           # 一次性解析整任务 fan-in（本地 TensorMap）
                    if popcount(M) == 1:
                        # 单核（1C 或 1V）：独自执行那唯一子任务，与 AIV0/AIV1 身份无关，无配对、无推送。
                        把该唯一子任务构建进一个空闲私有环槽
                    else:
                        # 多核（2V / MIX）：我是 anchor。构建我自己物理角色对应的槽，
                        # 把其余激活槽推送给同 block 伙伴（以 id 为键，互不串扰）。§3.1
                        把我自己角色的槽对应的子任务构建进一个空闲私有环槽
                        block.won[N] = { active_mask:M, kernels, args, fanin_ids,
                                         remaining: popcount(M) }      # block-shared（§3.1）
                # else（old >= N）：已有一个 T 类型的核认领了 N（它跑在前面）→ 跳过
        # else: 类型不匹配（例如 AIC 核遇到 1V 任务）→ 只做了 TensorMap，跳过

    # --- 步骤 3：终止与前向进展 ---
    if 编排已结束 AND 私有环为空 AND 无针对我的待抽取投递（收尾条件见 §7）:
        break                                                 # 本核完成
    if freed == 0 AND (私有环已满 OR 编排已结束):
        # 这一轮既没执行成任何任务、也无法（或无需）再认领：
        # 唯一能取得进展的是别的核置位我等待的某个完成标志 → 自旋后重扫步骤 1。
        spin_wait()
    # 否则回到 loop 顶部：继续“执行优先、再认领一个”
```

性质：

- **MIX = anchor 推送 + follower 异步抽取（§3.1）。** AIC 核为 MIX 任务 anchor，胜出后把其余
  激活槽的子任务推送进以 id 为键的 block 投递表 `block.won[N]`；block 的 AIV 核绝不为它竞争、
  **也绝不阻塞等待**——它只异步从 `block.won` 抽取属于自己槽的投递并构建。cube 落后时 AIV 没有
  待抽取的投递，便继续做自己的 AIV-only 工作（零停顿）；cube 领先时投递累积、AIV 有空槽就抽取，
  若 AIV 落后到填满 `block.won`，anchor 暂缓认领新多核任务（反压，转去 Phase B）。槽未激活的
  block 伙伴（例如 **1C+1V 上的 AIV1**）从不收到投递，照常去认领其他工作。
- **每任务一个标志，由最后一个子任务置位。** 单核任务直接置 `task_completed_flag`；多核任务
  递减一个 block-local 计数器（= `popcount(active_mask)`），由最后完成的子任务置位。消费者
  始终看到一个原子完成信号。
- **执行优先、一次认领一个。** 每轮循环先腾空就绪任务、再至多认领一个；不再“填满环才执行”。
  这是把单核的“超前认领”量压到很小、避免负载倾斜的关键（§6.1）。
- **反压** = 私有环填满（`PRIVATE_TASK_SLOT_NUM` 个槽）。私有环很小，所以单核任何时刻最多只
  比“已就绪可执行”超前这么几个任务。
- **即时回收槽**：每个共同 owner 在*自己*的子任务完成时释放*自己*的槽。没有全局环尾推进，
  没有跨核的槽复位协调，因为环是私有的。
- **前向进展**：环满且无就绪任务时自旋重扫，直到另一个核的完成标志解锁某个任务；一旦腾出
  一个槽，该核就回到编排去竞争新任务。

### 6.1 为什么“执行优先 + 小环”——乱序窗口与负载均衡

**乱序（out-of-order, OoO）窗口 = 核数 × 私有环槽数。** 这是整个系统在任一时刻能“同时在飞”
并允许乱序执行的（子）任务上限。它决定了无依赖的后续任务能否绕过排在前面、但尚未就绪的任务
被尽早执行（避免 head-of-line blocking）。

**旧设计（填满环再执行）为什么会负载倾斜。** `claim + build` 极快，而 `execute` 很慢。若每个核
都“先把私有环填满再开始执行”，那么跑得最靠前的核会在极短时间内把**一连串连续的任务**全部
`atomic_fetch_max` 抢进自己的环（把 `cursor` 一路推高），随后独自长时间串行执行这一串任务；
其它核因 `cursor` 已被推高而**抢不到**这段连续 id → 严重负载不均衡。更糟的是 head-of-line：
环里靠前但未就绪的任务会一直占着槽，挡住它后面其实已就绪、本可被别的核分担的任务。

**两点改进。**

1. **执行优先（本节伪代码）。** 每轮先腾空就绪任务、只认领一个新任务。核在执行一个长任务期间
   **不推进认领**，这段时间里其它核会推进 `cursor` 认领后续任务 → 工作自然铺开。认领不再是
   “抢满即止”的突发，而是“没就绪活干时才逐个超前”的受控行为。
2. **保持私有环小（缩小 `PRIVATE_TASK_SLOT_NUM`）。** OoO 能力主要应由**核数**这一维度提供，
   而不是把单核的环开大——开大只会让单核一次能独吞更长的连续任务串，放大倾斜。把环取较小值
   （如 2–4）即可在保留足够乱序窗口（核数已经不小）的同时，把单核超前量压到最低。环大小应按
   访存延迟 / kernel 时长实测调优，而非默认开大。

> 一句话：乱序靠“多核 × 小环”，不靠“单核 × 大环”。执行优先确保快核在执行长任务时把后续认领
> 让给其它核；小环确保即便要超前，超前量也很小。

**实测泳道图。** 下图是 `benchmark_bgemm`（`FullCore24`，`block_dim=24` → 24 AIC + 48 AIV
共 72 条 lane，240 个 GEMM(1C) + 240 个 ADD(1V)）在 a2a3sim 上的每核执行泳道：每条横轴是一个
物理 lane（AIC / AIV0 / AIV1），每个色块是一次 incore 函数执行（蓝=GEMM、红=ADD）。可见执行优先
策略把 GEMM 较均匀地铺满了 24 个 AIC，而非堆积在少数快核上——这正是 §6.1 论证的负载均衡效果。

![fully_distributed_within_core 每核执行泳道（benchmark_bgemm FullCore24）](fully_distributed_within_core/swimlane_bgemm_fullcore.png)

> 复现：`dist_engine` 内置一个环境变量门控的 Chrome-trace 导出器（中心化 L2 采集器不适用于本
> runtime 的 AICPU 桩）。设 `PTO_DIST_SWIMLANE=<path.json>` 跑用例即生成 trace，再用
> `python -m simpler_setup.tools.dist_swimlane_render <path.json> -o <out.png>` 渲染为上图；
> 或把 JSON 直接拖入 [Perfetto](https://ui.perfetto.dev/) 交互查看。incore 函数名由 `scene_test`
> 在捕获后从 CALLABLE spec 注入（叶子 `CoreCallable` 不携带名字），故图例显示 GEMM/ADD 而非 f0/f1。

### 6.2 实测：编排/调度开销随核数的代价

全分布式模式用"无中心调度器"换来的代价是：**编排被每个核完整重放（SPMD），且认领要在共享 cursor
上原子竞争**。为了把这部分纯开销与 kernel 计算分离测量，`dist_engine` 提供一个环境变量门控
`PTO_DIST_SKIP_EXEC=1`：置位后 `execute_slot` **跳过 incore kernel 调用**（每个子任务当 0 代价
瞬时完成），但**保留全部 ownership/完成/frontier 簿记**，核循环照常终止。这样测得的片上编排墙钟
就只反映 orchestration + claim race + scheduling。

下表用 `benchmark_bgemm`（`matmul_add_task_num=480`，约 960 个任务）在 a2a3sim 上扫 `block_dim`
（1 block = 1 AIC + 2 AIV），取多轮中位数。`device` 为片上编排墙钟（PTO2 profiling），是关注指标；
`host` 含 Python/sim 启动等固定开销，仅作参照。复现：
`python examples/a2a3/fully_distributed_within_core/runtime_overhead_test/test_runtime_overhead.py -p a2a3sim`。

| blocks | cores | device 编排墙钟 (ms) | us/task | 相对 1 block |
| -----: | ----: | -------------------: | ------: | -----------: |
|      1 |     3 |                 3.93 |    4.09 |        1.00× |
|      2 |     6 |                 4.71 |    4.91 |        1.20× |
|     12 |    36 |                21.23 |   22.11 |        5.41× |
|     24 |    72 |                42.87 |   44.65 |       10.92× |

**结论。** 纯编排/调度墙钟**随核数近线性增长**（3→72 核约 11×）：核越多，重复重放的编排和 cursor
竞争越多。少核时增量很小（2 块仅比 1 块高约 20%），随核数增大才陡升。这部分固定开销要靠**真实
kernel 执行被多核并行摊薄**来回本——本实验故意跳过执行，所以只暴露开销本身。它也说明：私有环要小、
执行优先（§6.1）等设计的价值，正是让有限的核尽快投入真实执行，而不是把时间耗在超前认领/竞争上。

### 6.3 绑核（CPU 亲和）对测量噪声的影响

仿真把每个 AICore/AICPU“核”实现为一个 host 线程，默认由 OS 在全部物理核（本机 320 核 / 8 个 NUMA
节点，每节点 40 核）上自由调度。跨核迁移与跨 NUMA 访问会给 §6.2 的 `device` 墙钟带来明显抖动（单次运行间方差很
大）。`test_runtime_overhead.py` 新增 `--bind` 开关，用 `sched_setaffinity` 在**进程级**绑核（后续所有
sim 线程自动继承，无需外部 `numactl`，也避免 `--membind` 的内存压力）：

* `--bind none`（默认）：不绑核；
* `--bind node:<nodes>`：绑到指定 NUMA 节点的全部 CPU（如 `node:0,1`）；
* `--bind cpu:<list>` 或裸 `<list>`：绑到显式 CPU 列表/区间（如 `cpu:0-119`）。

> **绑核曾暴露的崩溃 bug（已修复）。** AICore kernel `.so` 每个 `run` 都 dlopen/dlclose 重载，而其
> `pthread_once` 创建的 TLS key 在 dlclose 时不被 glibc 回收，逐 `run` 泄漏；约 200 个 `run` 后耗尽
> `PTHREAD_KEYS_MAX`（1024），`pthread_key_create` 失败 → `sim_get_reg_base()` 返回 NULL → 在
> `write_reg` 上空指针 SIGSEGV（全量 1→24 扫描在 `block≈23` 必崩）。修复：在
> `src/{a2a3,a5}/platform/sim/aicore/kernel.cpp` 增加卸载析构 `__attribute__((destructor))`，于
> dlclose 时 `pthread_key_delete` 全部 key，使每轮重载对 key 池**净零占用**；绑核全量 sweep 现可稳定
> 跑完。

**为何把评估限制在单 NUMA 核范围。** 本机拓扑为 **8 个 NUMA 节点 × 40 核 = 320 核**（无超线程），
**跨 NUMA 访问代价显著**。仿真里每个 sim“核”是一个 host 线程，`cores = block_dim × 3`。当一次运行用到的
核数超过单个节点的 40 核（即 `block_dim ≥ 14`，42 核起），AICore 工作集被迫横跨多个 NUMA 节点，**跨节点
的 cursor 原子认领竞争 + 远端内存访问**会主导 `device` 墙钟：实测在 `block≈13→14` 出现明显台阶、且
`block 14–24` 在本共享机上随其它租户的突发负载剧烈抖动（同一配置重测可差 2–3×）。这类数字是**平台 NUMA
伪影**，并非引擎本身的编排复杂度。因此我们**只评估 AICore 核数落在单个 NUMA 节点内的 block 范围**
（`cores = block_dim × 3 ≤ 40 ⟹ block_dim ∈ [1, 13]`），不再做更大范围扫描。

**把 AICore 线程真正钉进同一个 NUMA 节点（线程级 1:1 绑核）。** 仅靠进程级 `--bind` 还不够：

* **绑单个 40 核节点很脆弱。** sim 的**总线程占用**远大于 AICore 数（还含每次 spawn 的 50 个 AICPU
  over-launch 线程、4 个存活 AICPU、采集与主线程），全挤进 40 核。空闲时 `--bind node:<单节点>` 尚能干净到
  `block 12`，但 `block 13`（39 AICore ≈ 节点满）即超订、`device` 跳升约 2×（见
  `build/sweep_singlenuma_node2_40cores.txt`）；更糟的是它对**外部负载极敏感**——因为该引擎用自旋式
  cursor 认领竞争，一旦该节点被其它租户占用一部分核，持锁线程被抢占、其余线程空转自旋（lock-convoy
  崩溃），`device` 会从 `block≈6` 起就抖升到 20–30 ms。两种情况都是 CPU 争抢伪影，非真实编排开销。
* **只绑多个节点（进程级）也不够干净。** 进程绑到 3 节点时，OS 会把 AICore 线程**散布到多个 NUMA 节点**，
  AICore 之间的 cursor 认领竞争又变成跨节点访问——这正是之前看到 1→13 增长偏大（~2.5×）的部分原因。

正确做法是**线程级绑核**：新增 `--aicore-numa <node>`（置 `PTO_SIM_AICORE_NUMA_NODE`），让 device_runner
在拉起 AICore 线程时把**第 i 个 AICore 线程用 `sched_setaffinity` 1:1 钉到该节点的第 i 个 CPU**，从而整个
AICore 工作集严格留在同一个 NUMA 节点、每核独占一个物理 CPU；而 AICPU/主/采集等辅助线程**不钉核**，由
进程级 `--bind`（给足若干空闲节点）承载，避免超订。要求 `cores = block_dim × 3 ≤ 单节点核数(40)`，即
`block_dim ∈ [1, 13]`。

> **绑核确认。** `PTO_SIM_AICORE_PIN_VERBOSE=1` 下逐线程打印落核情况；`block_dim=13`（39 个 AICore 线程，
> `--aicore-numa 2`）实测 **39/39 线程全部运行在 node2 的 cpu 80–118**，零越界，确认 AICore 工作集完全位于
> 单个 NUMA 内。

下表为该单 NUMA 区间的完整统计（**当前引擎，已含 §6.4 的 O(N) per-core TensorMap 优化**；`tasks=480`，
**25 轮中位数**；`--bind node:1,2,3` 承载辅助线程 + `--aicore-numa 2` 把全部 AICore 钉进 node2；归档
`build/sweep_singlenuma_aicorepin_node2.txt`）：

| blocks | cores | device 编排墙钟 (ms) | us/task | 相对 1 block |
| -----: | ----: | -------------------: | ------: | -----------: |
|      1 |     3 |                 2.09 |    2.17 |        1.00× |
|      2 |     6 |                 2.22 |    2.31 |        1.06× |
|      3 |     9 |                 2.39 |    2.49 |        1.15× |
|      4 |    12 |                 2.54 |    2.64 |        1.22× |
|      5 |    15 |                 2.80 |    2.91 |        1.34× |
|      6 |    18 |                 3.00 |    3.13 |        1.44× |
|      7 |    21 |                 3.05 |    3.18 |        1.46× |
|      8 |    24 |                 3.24 |    3.38 |        1.56× |
|      9 |    27 |                 3.39 |    3.53 |        1.62× |
|     10 |    30 |                 3.73 |    3.88 |        1.79× |
|     11 |    33 |                 3.84 |    4.00 |        1.84× |
|     12 |    36 |                 4.20 |    4.38 |        2.02× |
|     13 |    39 |                 4.25 |    4.42 |        2.04× |

**结论。**

* AICore 全部钉进单个 NUMA 节点后，单 NUMA 核范围（`block ≤ 13`，≤40 核）内编排/调度开销**平滑、单调、
  且低**地随核数上升，1→13 仅约 **2.0×**（`us/task` 2.17→4.42）——SPMD 冗余重放 + cursor 认领竞争的真实
  代价在节点内增长很温和。
* **对比"只进程级绑核（AICore 被散布到 3 节点）"**：同样 25 轮、同样 block 区间，后者 1→13 约 2.5×、
  `block 13` 的 `us/task` 5.47（见 `build/sweep_singlenuma_1_13_120cores.txt`）。线程级单 NUMA 绑核把
  `block 13` 降到 4.42（**−19%**）且整体更平——多出来的那部分增长确属**跨 NUMA 散布**，而非引擎本身。
* 低 `block`（≤4）相比优化前明显下降（如 1 块 `us/task` 3.36→2.17），印证 §6.4 的 O(N) 优化。
* **越过单节点（`block ≥ 14`，>40 核）**必然跨 NUMA：台阶 + 强抖动，是平台 NUMA + 共享机外部负载的伪影，
  本评估**不纳入**。
* **共享机注意**：本机为多租户共享，即便绑核别的任务仍可能突发占用同批核；故采用 25 轮中位数并先用
  `mpstat -P ALL 1 1` 选空闲节点。曾观察到全 8 节点 ~100% 占用时数值整体抬升数倍。

归档：AICore 单 NUMA 线程级绑核 `build/sweep_singlenuma_aicorepin_node2.txt`；仅进程级绑核对照
`build/sweep_singlenuma_1_13_120cores.txt`；单节点超订对照 `build/sweep_singlenuma_node2_40cores.txt`。
（历史全 1–24 跨 NUMA 扫描 `build/sweep_1_24*.txt` 仅作平台伪影参照。）

### 6.4 降低每任务编排开销：把 per-core TensorMap 从 O(N²) 降到 O(N)

§6.2/§6.3 测的是开销随**核数**的变化。另一条正交的轴是开销随**任务数**的变化——它暴露了单核
编排算法的复杂度。把 `block_dim=1`（3 核、无认领竞争）固定下来扫任务数，就能把 per-core 编排算法
的成本从多核竞争噪声里隔离出来。

**定位。** 每个核对每个任务都要维护一份"生产者表"（per-core duplicate TensorMap，§9）：fan-in
解析要 `lookup` 输入区间的生产者，注册输出要 `insert`。最初的 `DistTensorMap` 是一个**扁平数组 +
线性扫描**，且**从不回收**条目：

```
struct DistTensorMap { MapEntry entries[kMapCap]; int32_t count; };
// lookup / insert 都是 for (i in 0..count) 线性比对
```

对 bgemm 这类"**单个扁平输出 buffer + 大量不相交 tile**"的负载，`count` 会随整个运行近线性增长
（每个 tile 是不同的 `[lo,hi)`，精确匹配替换帮不上忙），于是每次 `lookup`/`insert` 都是 O(count)，
全程 **O(N²)**。仅靠"按 buffer 基址哈希"也救不了——所有 tile 共享同一个基址，落进同一条链。

**修复（对齐 `tensormap_and_ringbuffer` 的 `PTO2TensorMap` 方案）。** 改写 `DistTensorMap` 为该
runtime 久经验证的结构：**按 buffer 基址哈希分桶 + 桶内双向链 + 按生产者任务的 entry 链 + 空闲链表
+ lazy invalidation + `cleanup_retired` 按任务精确回收**。决定性的一步是**回收**：

> 依据 H 跨度契约（§9.5/§11.4），任务 N 的消费者 id ≤ N+H；因此 producer 早于 `N − H` 的条目
> **不可能**再被任何未来任务作为 fan-in（其 GM 堆区也已在同一界限下被回收）。每次 submit 用确定性
> 阈值 `alive_floor = N − H` 推进，沿**生产者任务链**精确释放刚离开 H 窗口的那一个任务的条目（绝不
> 扫描整池）。这把每条链长从"全程任务数"压到"H 窗口内"，O(N²) → O(N·H) ≈ O(N)。

阈值取自 N（确定性、各核一致），**不**取自 frontier（与时序相关），故每核的 map（含空闲链表与回收
进度）演化完全一致，"每核副本一致"不变量得以保持。与参考实现一样，`insert` **总是挂新条目**到其
生产者任务链（不做就地替换），`lookup` 返回区间重叠者中 producer **最大**（最新）的那个——语义上
等价于原先的就地替换，但让 `cleanup_retired` 能按任务链 O(1) 回收。

**附带优化：把认领门提前，让败者跳过赢家专属工作。** SPMD 下每个核都重放 submit，但一个任务只有
约 1/3 的核会赢得认领。原先所有核都先做了 fan-in `lookup` 和 `built[]` 组装（tc × `sizeof(Tensor)`
拷贝）才去认领。把 **anchor 类型判定 + cursor 认领提前到 map 操作之前**，则：
* **fan-in `lookup` 改为赢家专属**——败者从不消费 fanin，直接跳过 input 查找（output `insert` 仍
  无条件执行，保持各核 map 一致）；
* **`built[]` 组装移到认领成功之后**——失败的核省掉无用拷贝。

这正是"负载随核数摊销"能显现的关键：核越多，每个核赢得的任务越少、跳过的 fan-in 查找越多。实测
`dev vs 1blk`（tasks=4000）从改前的 1.7×/2.2×（2/4 block）压平到约 **0.7–1.1×**（多核档不再随核数爬升，
甚至偶尔低于 1 block）。注意它**动不了**每核必做的"地板"——堆物化 + output `insert`（每核全量副本的
固有代价），故 1-block 绝对值基本不变。

**A/B 实测（`block_dim=1`，跳过执行，7 轮中位数）。** 隔离单核编排算法成本，扫任务数：

| matmul_add_task_num | 旧 device (ms) | 旧 us/task | 新 device (ms) | 新 us/task | 加速 |
| ------------------: | -------------: | ---------: | -------------: | ---------: | ---: |
|                 480 |          3.10  |      3.23  |          2.95  |      3.08  | 1.05× |
|                1920 |         13.28  |      3.46  |          5.42  |      1.41  | 2.45× |
|                3840 |         34.76  |      4.53  |          4.01  |      0.52  | **8.66×** |

（新列为"哈希+回收"与"`built[]` 后置"两项优化叠加后的最终值。）

旧实现 device 随任务数**超线性**（任务 ×8 → device ×11.2，`us/task` 3.23↑4.53），正是线性 map 不
回收的 O(N²) 尾巴；新实现**亚线性**（任务 ×8 → device 仅 ×1.3，`us/task` 反而 3.08↓0.52），即 O(N)。
在 §6.2 关注的 480 任务规模，新版与旧版持平（略优）；规模越大优势越显著。

**结论。** per-core 编排里真正随规模恶化的是"无回收的线性生产者表"。沿用 `tensormap_and_ringbuffer`
的哈希 + 按任务回收方案、并用确定性的 `N − H` 作回收阈值，即可把单核编排从 O(N²) 降到 O(N)，同时保持
SPMD 各核 map 完全一致与全部 golden 正确性（bgemm / paged_attention / paged_attention_ringbuffer /
mix_coown 等用例校验通过）。复现：
`python examples/a2a3/fully_distributed_within_core/runtime_overhead_test/test_runtime_overhead.py -p a2a3sim --blocks 1 --tasks 3840`。

**附带优化：把认领门提前，让败者跳过 fan-in 查找。** 见 §6.4 上文同名段落——把 anchor 类型判定 +
cursor 认领提前到 map 操作之前，fan-in `lookup` 改为赢家专属，`built[]` 组装移到认领之后。这是"负载
随核数摊销"能显现的关键优化，效果见下节 §6.5。

### 6.5 核数 scale up 时 us/task 为何回升：cursor CAS 等共享原子的竞争

**测试条件（截至本节最新）。** workload=`benchmark_bgemm`，`PTO_DIST_SKIP_EXEC=1`（跳过 incore
执行，只测编排/调度墙钟），`device` 为片上编排墙钟（PTO2 profiling），多轮取中位数。`--blocks` 默认
随平台：macOS `1-4`、Linux `1-13`。运行用项目自带 `.venv` 解释器（含编译好的 `_task_interface` 绑定）。
当前代码含三项优化：哈希 + H 回收的 TensorMap（§6.4）、`built[]` 后置、**winner-only fan-in**。复现：
`./.venv/bin/python examples/a2a3/fully_distributed_within_core/runtime_overhead_test/test_runtime_overhead.py -p a2a3sim --tasks 4000`。

**结果 1：单核（block=1）随任务数仍是 O(N)。** 固定 1 block 扫 batch（`--tasks`，总任务约 2×）：

| matmul_add_task_num | ~tasks | device (ms) | us/task |
| ------------------: | -----: | ----------: | ------: |
|                1000 |  ~2000 |        2.08 |    1.04 |
|                4000 |  ~8000 |        3.99 |    0.50 |

任务量 ×4、device 仅约 ×2、`us/task` 反而下降 → per-core 编排算法是 O(N)（§6.4 的 TensorMap 改造之效）。

**结果 2：多核（Mac，tasks=4000，blocks 1–4）device 随核数回升。**

| blocks | cores | device (ms) | us/task | dev vs 1blk |
| -----: | ----: | ----------: | ------: | ----------: |
|      1 |     3 |        3.99 |    0.50 |       1.00× |
|      2 |     6 |        3.22 |    0.40 |       0.81× |
|      3 |     9 |        4.46 |    0.56 |       1.12× |
|      4 |    12 |        9.14 |    1.14 |       2.29× |

winner-only fan-in 使中低核数出现摊销（2 block 一度低于 1 block，约 0.8×；多轮中 `dev vs 1blk` 多在
0.7–1.3× 间）；但核数继续增大时 `device` 仍会**回升**（如上 4 block；Mac 上 12 线程超订使该档方差很大，
不同轮在 1.1×–2.3× 间跳）。下面分析这部分回升的算法性根因。

**根因：认领走的是对单个共享 cursor 的 CAS 循环 fetch_max。**

```text
bool claim(cursor, N):
    c = cursor.load()
    loop:
      if N <= c: return false          // 落后核只 load、不写（便宜）
      if cursor.CAS(c -> N): return true // 争胜:在同一条 cache line 上 CAS
```

认领**按类型共享同一个 cursor**：所有 AIC 核抢 `cube_cursor`、所有 AIV 核抢 `vector_cursor`。于是：

* **单一热点 cache line。** `block_dim=B` 时，cube 任务由 `B` 个 AIC 核、vector 任务由 `2B` 个 AIV 核
  对同一原子量每任务 load+CAS。该 line 在竞争核间反复转移独占权（MESI），竞争核越多 → 单次 CAS 延迟
  越高、失败重试越多、一致性流量越大。`device` 取最慢核墙钟，最慢核要排队等这条线 → device 随 B 回升。
  （AIV 数是 AIC 的 2 倍，故 vector 认领竞争更重——bgemm 的 ADD(1V) 即走此路。）
* **skip-exec 放大竞争。** 跳过执行后每任务 0 代价，各核近**锁步**推进 → 对任意任务 N 几乎同时争抢 cursor，
  达最坏竞争。真实执行时 kernel 耗时让各核去同步、认领被自然错开，竞争反而小。**故本测试是 cursor 竞争
  的悲观上界。**

**其它随核数增长的全局原子（次要但同向）：**

| 原子 | 访问模式 | 随核数扩展 |
| --- | --- | --- |
| `cube/vector_cursor` CAS（认领） | 每核每任务，单一热点线 | **强（主因）** |
| `frontier` CAS（`advance_frontier`） | 每次完成扩展前缀时 CAS 单一 `frontier` | 中–强 |
| `flags[N]` 完成标志（`uint8_t`，64 个/行） | 相邻任务标志**伪共享** | 中 |
| `block.won`（state/remaining/drained） | **每 block 局部，仅 3 核内** | 否（不随总核数涨） |

此外**仿真特有**：每个核是 host 线程，核多→线程多→在物理核上**超订** + 跨 NUMA，放大 device 抖动
（非算法因素，Mac 上尤甚；干净曲线应在 Linux 用 §6.3 的绑核测）。

**小结与缓解方向。** us/task 在核数增大时回升，主因是**全局单热点 cursor 的 CAS 竞争**（其次为 frontier
CAS 与 flag 伪共享），而非每核的 map 维护（那块已 O(N) 且被 winner-only fan-in 进一步减负）。若要把这条
曲线进一步压平，可考虑去掉"全局单热点"：

* **批量认领（claim stride）**：一次 CAS 抢一段连续 id，把 N 次 CAS 摊销成 N/stride 次；
* **分片认领（cursor sharding）**：把 `cube/vector_cursor` 各扩成 `G` 个，按 `task_index % G` 选 cursor，
  把单热点 CAS 摊到 `G` 条 cache line（详见 §6.6——认领语义与单一 cursor 等价，不引入偏差/不均衡）；
* `flags` 按 cache line 对齐分散以消伪共享。

这些都属于"认领/完成同步"层的可选优化，与 §6.4 的 map 改造正交。认领最初用最简单的全局 cursor；**现已
落地 §6.6 的 cursor 分片（`G=4`）+ winner-only fan-in**，实测见 §6.7。

### 6.6 cursor 分片（sharding）：按 `task_index % G` 切 cursor，认领效果与单一 cursor 等价

§6.5 把"分片认领"列为压平 cursor CAS 竞争的方向之一。本节给出**具体方案**并论证一个重要结论：**只要按
`task_index` 给 cursor 变量分片、而绝不对 worker 分组，分片在"认领任务"上的语义与单一全局 cursor 完全一致
——不产生额外进度偏差、不加剧 worker 间负载不均衡，仅把对 cursor 的访存竞争摊到 `G` 条 cache line 上。**

**方案。** 把今天的两个全局 cursor（`cube_cursor` / `vector_cursor`，§11.1）各扩成 `G` 个：
`cube_cursor[G]` / `vector_cursor[G]`。某任务 id `N` 做认领时，访问 `vector_cursor[N % G]`（cube 任务同理用
`cube_cursor[N % G]`），即 **shard = `N % G`**。`claim` 仍是同一套 CAS-loop fetch_max（§11.1）。关键在于：
**shard 只由 task_index `N` 决定**，而 `N` 在每个核上完全一致（各核 replay 同一条 submit 流），所以**任一核
认领 `N` 时算出的 shard 相同、访问的是同一个 `cursor[N%G]`**——**没有"哪些核只能碰哪个 shard"的核分组**。

**为什么认领效果与单一 cursor 完全一致。**

* **仍是"每任务恰好一个 owner、不漏不重"。** `vector_cursor[g]` 只承接 `N ≡ g (mod G)` 的那串 id
  （`g, g+G, g+2G, …`），它们被每个核**按序**处理 → 在该 residue 子序列上仍是单调 fetch_max，首个把它从
  `<N` 推到 `N` 的核独占 `N`。这与单 cursor 在全序列上的不变式**逐字相同**，只是把"一条单调序列"拆成 `G`
  条交织的单调子序列，每条仍单调、连续、无跳过。
* **任一核都能赢任一任务（工作窃取原样保留）。** shard 由 `N`、而非核身份决定，每个核处理到 `N` 就去抢
  `cursor[N%G]`，**没有核被排除在任何任务之外**。于是"谁空谁抢下一个 id"的窃取式负载均衡**完全保留**，
  不会出现某组核闲、另一组过载。
* **不产生额外进度偏差。** 不存在"各自独立推进的分片"：每个核都走完整条流，对连续的 `N, N+1, N+2, …`
  轮流落在 `cursor[0..G-1]` 上，故 `G` 个 cursor 始终贴着**同一条认领前沿**、彼此相差不超过约 `Δ+G`
  （`Δ` 为单核 run-ahead 上界）。整体推进仍由**同一个全局完成前沿 `F` + 同一个私有环 run-ahead 上限**封顶
  （与是否分片无关），所以偏差与单 cursor 时**一模一样**。
* **确定性不变。** 认领只决定"谁执行"，不改变 id、不改变 per-core map 的 replay/insert 顺序，golden 结果不变。

**结论（直接回答"是否等价"）。** **是。** 按 `N % G` 给 cursor 变量分片，在**认领语义、负载分布、推进/偏差、
确定性**四个方面与单一全局 cursor 等价；**唯一区别**是把对一条 cursor cache line 的 CAS 竞争分摊到 `G` 条
独立 line，降低访存争用。因此 cursor sharding **不会**带来更大的进度偏差，也**不会**加剧 worker 间负载不
均衡——它**只**降低了竞争这个 cursor 的访存代价。

**一处要点：收益何时兑现，以及 `G` 怎么取。** 对**同一个** id `N`，认领前沿上的核仍然撞同一个
`cursor[N%G]`；分摊之所以有效，是因为各核在任一时刻分布在一段**连续 id 窗口**上（核 A 在 `N`、核 B 在
`N+1` …，窗口宽约 run-ahead `Δ`），这些连续 id 落在不同的 `cursor[N%G]` 上。**只要在飞 id 窗口 ≥ G**，
CAS 写竞争就被摊到 ≈ `G` 条 line。故 `G` 取到"每条 line 的竞争核数不再是瓶颈"即可（量级上
`G ~ 同类型核数 / 期望每线核数`），不必更大；`G=1` 即退回今天的实现，零行为变化。

**务必区分：分片 cursor 变量 ≠ 给 worker 分组。** 上面的等价性**只**在"shard 由 `task_index` 决定、所有核
对所有任务一律可竞争"时成立。若改成另一种做法——**按核/按 block 把 id 空间静态切给不相交的核组、各组只
认领自己那片 id**——那就是"分 worker"，会引入**独立分片进度**（慢分片顶住全局完成前沿 `F`、拖慢回收）与
**工作窃取丢失**（某组核闲、另一组过载的负载不均衡）。那种核分组才需要额外的"显式认领窗口 + 跨分片窃取
兜底"等机制来补救，得不偿失。**本方案刻意避免它**：我们分片的是 **cursor 变量（按 `task_index % G`）**，
不是 worker——这正是它能与单一 cursor 等价、却又降竞争的原因。

### 6.7 cursor 分片实测：G=4 已落地；单 NUMA 区间收益与最优 G

§6.6 的方案已落地（`kCursorShards` 默认 **G=4**，每个子 cursor 独占一条 64B cache line；并配合 winner-only
fan-in，§6.4.1）。本节给出在**单 NUMA 区间**的实测结论。

**测量口径。** skip-exec（仅编排/调度），`~10000 tasks`（`--tasks 5000`），`rounds=15` 取中位数，AICore 线程级
钉进 node2（`--aicore-numa 2`，§6.3），辅助线程 `--bind node:1,2,3`，**空闲机器**上取干净单调曲线（共享机
偶发外部负载会污染后段 block，已剔除被污染的运行）。

**(1) 分片前（单一全局 cursor）→ 分片后（G=4）。**

| blocks | cores | 单 cursor us/task | G=4 us/task | 改善 |
|--------|-------|-------------------|-------------|------|
| 1  | 3  | 1.05 | 0.99 | −6% |
| 4  | 12 | 1.36 | 1.29 | −5% |
| 8  | 24 | 1.92 | 1.68 | −12% |
| 10 | 30 | 1.97 | 1.83 | −7% |
| 12 | 36 | 2.23 | 2.10 | −6% |
| 13 | 39 | 2.33 | 2.20 | −6% |

全程 `us/task` 一致小幅下降，**中高 block 段（8–13）改善约 6–12%**，曲线仍干净单调。方向正确——把单热点
cursor 的 CAS 竞争摊到 4 条 cache line 确实压低了 §6.5 所述的访存争用。

**(2) G=4 vs G=8：单 NUMA 内 G=4 是甜点。** 把 `G` 加倍到 8 重测（同口径）：

| blocks | cores | G=4 us/task | G=8 us/task | 差异 |
|--------|-------|-------------|-------------|------|
| 1  | 3  | 0.99 | 1.01 | +2% |
| 7  | 21 | 1.65 | 1.75 | +6% |
| 8  | 24 | 1.68 | 1.81 | +8% |
| 9  | 27 | 1.74 | 1.94 | +11% |
| 10 | 30 | 1.83 | 2.05 | +12% |
| 11 | 33 | 2.00 | 2.26 | +13% |
| 13 | 39 | 2.20 | 2.29 | +4% |

**G=8 不升反降**（中高 block 段慢 8–13%）。原因（单 NUMA、≤39 核区间）：

1. **G=4 已摊够竞争。** `block=13` 也才 13 个 AIC 核 / 26 个 AIV 核；G=4 下每 shard 平均仅 ~3 个同类型核竞争，
   已逼近"每条 line 竞争核数不再是瓶颈"（§6.6 对 `G` 的取值分析），再加倍几乎没有进一步降竞争的空间。
2. **G=8 反而增大 cursor 的 cache footprint**（每类型 8×64B=512B，更多 cache line 同时在核间弹跳），总相干
   流量与局部性变差，得不偿失。
3. 分片越多、单核能赢的任务越稀疏（只拿 `≡ s (mod G)` 的 id），动态窃取式负载均衡的交织略变差。

**结论。** 在评估约束的**单 NUMA、核数 ≤ 一个节点（≤39 核）**区间内，**`G=4` 为最优**，故保持默认 `G=4`。
更大的 `G` 要等到**跨 NUMA / 更高核数**(`block ≥ 14`)、单条 line 上竞争核数显著上升时才可能回正——但那已属
跨 NUMA 区间（§6.3 说明其数字是平台伪影，不在本评估范围）。

归档：G=4 干净扫描 `build/sweep_singlenuma_shardG4_node2.txt`；G=8 对照 `build/sweep_singlenuma_shardG8_node2.txt`。

## 7. 终止

一个核在其编排不再产生任务**且**私有环为空（所有拥有的任务都已执行）时结束。对 follower
（AIV）还有一条额外条件：它必须等到**其 block 的 anchor 编排也结束**且 `block.won` 中再无
针对它的待抽取投递——否则可能有尚未推送的多核子任务漏执行。这就是 §3.1 提到的**尾部空转**：
当某 block 的 anchor 严重落后时，它的 follower 做完自身其余全部工作后，会在终止前空转等待
anchor 推送最后的多核子任务。这不是 per-task 串行阻塞，只发生在收尾，且 cube 领先时不出现。

所有核都结束时达到全局完成；最终的图输出位置被发布以供 host 拷回（见 §8 的
`graph_output_ptr`）。一个全局“所有核完成”屏障替代了旧的单一 `orchestrator_done` 标志。

---

# 第二部分 — 数据结构与共享特性

## 8. 共享模型

每个结构被归为以下之一：

| 类别 | 含义 |
| ---- | ---- |
| **全局共享** | 唯一权威实例；多个核读/写；需要显式访问机制 |
| **block-共享** | 仅在一个固定 block（1 AIC + 2 AIV）的核之间共享；用于 MIX 共同所有权（§3.1） |
| **每核私有** | 由单个核拥有；无跨核可见性 |
| **每核复制** | 每核复制一份；内容相同、各自独立重建（或只读副本） |

### 8.1 新引入的结构

| 结构 | 类别 | 作用 | 访问机制 |
| ---- | ---- | ---- | -------- |
| `cursor[T]`：`cube_cursor` / `vector_cursor` | **全局共享** | 每个类型的 claim 高水位线；到达 `N` 时 `old < N` 即胜出并拥有该任务（§2、§3.1） | 单条 `atomic_fetch_max(cursor[T], N)`（无则 CAS 回路），acq-rel；无跳过性证明见 §11.1 |
| `task_completed_flag` 连续完成前沿 `F` / 回收前沿 `R` | **全局共享** | `F` = 全已完成前缀；`R = F − H` 决定堆/标志环回收（§9.5、§11.3、§11.4） | `F` 协作式 CAS 推进；`R` 派生；单调 |
| `local_current_task_index` | **每核私有** | 编排进度游标；每次 submit `++` | 普通标量 |
| **私有任务环**（`PRIVATE_TASK_SLOT_NUM`，默认小，如 4） | **每核私有** | 保存已拥有的（子）任务：descriptor + payload + 本地状态 + fan-in producer id；故意取小（OoO 窗口 = 核数 × 槽数，§6.1） | 无（单一 owner，无锁） |
| `task_completed_flag` 环 | **全局共享** | 每任务 id 一个一次性置位布尔；唯一共享的 per-task 状态 | 最后一个（子）任务 owner 做 release 存储；消费者做 acquire 加载（轮询） |
| **`block.won[N]` —— 以 id 为键的子任务投递表** | **block-共享** | anchor → follower 的**推送**通道，以任务 id 为键：`{active_mask M, 各激活槽 kernels/args, 已解析 fan-in, 剩余计数}`。anchor 胜出时把其余激活槽子任务投递进来；follower **异步抽取**属于自己槽的项（不阻塞、不按走位等待）。承载每任务剩余计数，互不串扰（§3.1）。填满时 anchor 暂缓认领新多核任务（反压） | anchor 插入（release）；follower 抽取（acquire）；`remaining` 原子递减；最后一个子任务完成时释放条目 |

### 8.2 TensorMap

| 结构 | 类别 | 作用 | 访问机制 |
| ---- | ---- | ---- | -------- |
| `PTO2TensorMap` / `PTO2TensorMapEntry` | **每核复制（全量）** | tensor 区域 → producer 任务 id；在每个核上相同地构建（§4） | 无跨核锁；通过重放确定性 submit 流重建。有效性由 `task_completed_flag` 环开窗 |

### 8.3 全局共享，超出 per-task 状态之外

| 结构 | 类别 | 作用 | 访问机制 |
| ---- | ---- | ---- | -------- |
| GM 输出堆（打包的输出缓冲） | **全局共享（物理）** | 任务输出/中间结果的后备存储，可被任意核作为下游输入读取 | 一块全局物理区域；分配记账（堆顶、scope arena 基址）是**每核复制、确定性**的（§9），写入由 owner 完成。完整策略见 §9 |
| `heap_top` / scope arena 基址栈 | **每核复制（确定性，非全局）** | 在确定性 submit 重放中无条件推进，使任务 N 的输出地址成为 id 的纯函数（§9） | 无原子、无跨核通信；与 TensorMap 同理（§4） |
| `heap_reclaim_frontier`（全局回收水位线） | **全局共享** | 全局最旧“仍可能被读”的任务 id；据此在 id 顺序上回收堆（§9） | 由完成标志环 + 各核进度最小值推导；单调 |
| `func_id_to_addr_`（kernel id → GM 地址） | **全局共享，只读** | 把 `kernel_id` 解析为要调用的 incore 函数 | init 时一次性设置，之后只读 |
| `graph_output_ptr` / `graph_output_size` | **全局共享** | 供 host 拷回的最终输出位置 | 产出核做原子发布 |
| 全局错误字（原 `orch_error_code`） | **全局共享** | 任意核的致命错误 → 所有核 + host | 原子；首个写者胜出 |
| “所有核完成”屏障（原 `orchestrator_done`） | **全局共享** | 全局终止检测（§7） | 原子计数器 / 屏障 |

### 8.4 每核私有的编排状态

| 结构 | 类别 | 作用 | 访问机制 |
| ---- | ---- | ---- | -------- |
| Scope 栈（`scope_stack_top` + 各层 arena 基址） | **每核复制（确定性）** | `PTO2_SCOPE` 生命周期跟踪；同时界定 GM 输出堆的 arena 栈（§9）。各核结构相同、进度不同 | 无锁；由确定性重放重建。注意：原 `scope_tasks[]`/`scope_begins[]` 用于 fanout 引用记账，新模型已不需要（§9、§10） |
| Fan-in producer-id 列表（每个环槽一份） | **每核私有** | 构建时解析出的 producer 任务 id，执行时轮询 | 无 |
| 本地致命标志 | **每核私有** | 快路径致命错误；升级到全局错误字 | 本地标志 + 原子发布 |
| 核数常量（`total_cluster_count`、`total_aiv_count`） | **每核复制（只读）** | 资格 / 合理性检查 | init 时一次性设置 |

## 9. 动态内存管理（全局输出堆）

任务的输出/中间缓冲分配在一块 GM 堆上。由于**一个核产出的 output 可能被另一个核作为输入读取**，
这块堆必须是**全局可寻址**的。本节给出分布式 runtime 下的内存管理策略与数据结构，并说明它相对
当前 AICPU 模型的“stack of ring + scope”实现需要如何更新。

### 9.1 当前（AICPU 集中式）模型回顾

- **统一分配器 `PTO2TaskAllocator`**：把**任务槽环**与**堆环（heap ring）**合并分配。单一
  orchestrator 单线程推进，用普通 store 写 `heap_top`（bump），无需 CAS。
- **回收**：调度器把“最旧已 CONSUMED 任务”推进 `last_task_alive`；分配器据该任务的
  `packed_buffer_end` 反推 `heap_tail`，环形回收（分配从 `top` bump，到尾部则在 `tail` 足够时
  绕回，缓冲不跨越绕回边界）。
- **stack of ring**：按 scope 深度复制成 `PTO2_MAX_RING_DEPTH`(=4) 套 {TaskRing, HeapRing,
  DepPool}，使内层 scope 可独立于外层回收。
- **scope（`PTO2_SCOPE`）**：用 `scope_tasks[]`/`scope_begins[]` 记录本 scope 的任务；每个任务
  持有一个 +1 的 fanout 引用，`scope_end` 才释放——从而保证输出缓冲的生命周期 =（真实消费者
  全部完成）**且**（scope_end）。`TaskOutputTensors` 的引用只在其 `PTO2_SCOPE` 内有效。

### 9.2 哪些前提失效、需要更新

新模型（§2–§7）取消了集中 orchestrator 与 scheduler，因此上面多数机制的前提不再成立：

| 旧机制 | 在新模型中的处置 |
| ------ | ---------------- |
| 单 orchestrator 普通-store bump | **失效**：现在每个核都为自己拥有的任务分配输出。多写者下 `heap_top` 不能再用普通 store。 |
| `last_task_alive`/CONSUMED 驱动回收 | **失效**：无 scheduler、无 CONSUMED 状态。回收改由全局完成前沿（§9.5）驱动。 |
| 每 scope 深度的 TaskRing / DepPool / FaninPool | **移除**（§10）：任务槽改为每核私有环（§5），无依赖列表。 |
| fanout 引用 + scope_end 释放 | **失效**：无 fanout/refcount。生命周期改由“窗口/前沿 + scope arena 折叠”界定（§9.4、§9.5）。 |
| “stack of ring” | **收敛**为“**每核私有任务环**（§5） + **scope arena 栈**（§9.4）”，后者只管 GM 输出堆。 |

结论：**stack-ring 需要更新**——任务环部分整体移除，堆部分保留但分配方式与回收方式都要改；
**scope 需要保留但语义简化**（不再做 fanout 引用记账，改为 arena 栈 + 确定性重放）。

### 9.3 分配：确定性、每核复制的布局（无原子、无通信）

核心思想与 §4 的“每核全量复制 TensorMap”一致：**因为 submit 序列与每个任务的输出大小在各核上
完全确定且相同，输出缓冲的布局也可以被每个核确定性地复算。**

- 每个核在确定性 submit 重放中，对**每一个**任务（无论自己是否拥有——胜者、败者、follower 一视同仁）
  **无条件**推进一份**每核复制**的堆顶 `heap_top`。任务 `N` 的输出偏移 = 其所在 arena 基址 +
  该 arena 内 `N` 之前所有任务输出大小的前缀和。
- 因此 `addr(N)` 是 submit 序列（及确定性大小）的**纯函数**：每个核为任务 `N` 算出**完全相同**的
  地址。owner 负责写数据；任何核都能**不经通信**算出任意任务的输出地址。

这取代了旧的“单 orchestrator bump”（多核下不可行），也**优于全局原子 bump**：原子 `fetch_add`
会让地址依赖跨核的 bump 顺序而**非确定**，消费者便无法自行算出 producer 地址，必须额外发布地址 +
读地址，引入跨核通信。确定性复制方案两者皆免。

> **TensorMap 与地址的关系。** TensorMap 把 tensor 区域映射到 producer 任务 id（§4）。消费者拿到
> producer id 后，用上面同一套确定性布局即可算出其输出地址（或在 TensorMap 条目里直接缓存这个
> 确定性地址，因为它在每个核上都相同）。无需 producer 主动发布地址。

### 9.4 Scope = 确定性复制的 arena 栈

`PTO2_SCOPE` 在新模型里仍然是确定性编排程序的一部分（每个核执行相同的嵌套结构），因此 scope 栈
是**每核复制且各核相同**的（与 TensorMap 同理）。它现在的职责是界定 GM 输出堆的 **arena 栈**：

- **scope begin**：把当前 `heap_top` 记为新 arena 的基址，压栈（这是旧“stack of ring”里
  per-depth 独立回收的分布式对应物）。
- scope 内任务：在该 arena 内确定性 bump 分配（§9.3）。
- **scope end**：把堆顶折叠回该 arena 基址，**一次性回收**该 scope 内所有“不外逃”的输出（LIFO
  栈式回收，干净且 O(1)）。**外逃输出**（被该 scope 之外的任务消费的 tensor）必须分配在/提升到
  **父 arena**，以便在折叠后存活。
- 对**长 scope**（任务很多、不能等到 scope_end 才回收），在 arena 内部用 §9.5 的窗口/前沿机制做
  环形回收，先行回收已不再被读的缓冲。

`TaskOutputTensors` 的**单 scope 有效**规则保持不变：它返回的引用指向 owner 私有环槽中的 tensor
存储，不得逃出其 `PTO2_SCOPE`；跨 scope 的数据流一律通过 TensorMap 按 id 查 producer + 上述确定性
地址完成，而非通过 `TaskOutputTensors` 句柄。

### 9.5 回收：窗口/前沿，取代 `last_task_alive`/CONSUMED

由于布局在 id 顺序上确定地 bump，回收也自然按 id 顺序进行（任务 `N` 的缓冲位于 `N+1` 之前）。
难点在于判断“`N` 的缓冲何时不再被读”。新模型用**全局完成前沿**而非 fanout 精确计数：

- 维护一个**全局回收水位线** `heap_reclaim_frontier`，由 `task_completed_flag` 环加上**各核进度
  最小值**（最慢的核/最旧未完成任务）推导。它表示“所有 id ≤ 该值的任务都已完成且其消费者也已完成”。
- 给定**有界依赖跨度** `H`（保证任务 `N` 的所有消费者 id ≤ `N + H`），当全局完成前沿越过 `F` 时，
  所有 id ≤ `F − H` 的输出可安全回收——把堆尾推进，腾出位置给后续（确定性布局中绕回到该位置的）
  更晚任务。
- 这与 §11 的 “`task_completed_flag` 环开窗”使用**同一个窗口**：该窗口同时裁剪复制的 TensorMap
  与 GM 堆。
- **scope_end** 对“不外逃”输出提供额外的、更早的粗粒度回收边界（§9.4）。
- **反压**：堆（或当前 arena）满时，想为新拥有任务分配的核**暂缓认领**并自旋等待前沿推进——与
  私有环填满的反压（§6）同一性质，方向一致（不让快核无限超前于回收）。

> **正确性要点。** 一个缓冲只有在其**全部消费者执行完毕**后才能回收。窗口法用有界跨度 `H` +
> 全局完成前沿保证这一点；若某图的依赖跨度可能超过 `H`，必须把 `H`/堆容量调大，否则属配置错误
> （类比旧模型的 heap/window 死锁诊断）。精确的“按 tensor 最后消费者”回收（利用 TensorMap 中
> 同一区域被新 producer 覆盖这一确定性事件）是更省内存的改进方向，列入 §11。

### 9.6 数据结构小结

| 结构 | 类别 | 作用 |
| ---- | ---- | ---- |
| GM 输出堆（物理区域） | **全局共享（物理）** | 唯一一块全局可寻址的输出后备存储 |
| `heap_top` | **每核复制（确定性）** | 确定性 bump 堆顶；每核相同，无原子 |
| scope arena 基址栈 + `scope_stack_top` | **每核复制（确定性）** | scope→arena 映射；scope_end 折叠回收 |
| `heap_reclaim_frontier` | **全局共享** | 回收水位线，由完成前沿推导 |
| `graph_output_ptr` / `graph_output_size` | **全局共享** | 最终图输出位置，供 host 拷回 |

被移除：`PTO2TaskAllocator` 的任务环部分、`last_task_alive`/`heap_tail`(基于 CONSUMED)、per-depth
`DepListPool`/`FaninPool`、`scope_tasks[]`/`scope_begins[]` 的 fanout 记账（§10）。

## 10. 被移除的结构（相对 AICPU 的 `tensormap_and_ringbuffer`）

统一的 worker-scheduler 模型删除了整个子系统：

| 被移除 | 为什么消失 |
| ------ | ---------- |
| `PTO2SchedulerState`、`RingSchedState` | 无调度器实体——每个核调度自己的环 |
| `PTO2ReadyQueue`、`dummy_ready_queue`、`early_dispatch_queue` | owner 执行自己的就绪任务；无分派队列 |
| `PTO2SpscQueue` + `WiringState` | 无独立连线权威；无 fanout 可连 |
| `fanout_lock`、`fanout_head`、`PTO2DepListPool`、`PTO2FaninPool` 溢出 | 无 fanout 列表——依赖经标志环拉取 |
| `fanin_refcount`、`fanout_refcount`、`completed_subtasks` | 被完成标志轮询替代 |
| `Handshake` 门铃、`Runtime::workers[]`、`AICoreCompletionMailbox` | 无调度器→worker 分派握手 |
| SM 中的全局 `PTO2TaskDescriptor` / `PTO2TaskPayload` / `PTO2TaskSlotState` 环 | 被每核私有任务环替代 |
| `current_task_index`（环头）/ `last_task_alive`（环尾）流控 | 被 claim 计数器 + 每核环空槽替代 |
| `task_state`（PENDING/COMPLETED/CONSUMED）、每线程 `sched_error_*` | 被单一全局 `task_completed_flag` 与单一错误字替代 |
| `PTO2TaskAllocator` 的**任务环**部分、`heap_tail`(基于 CONSUMED 反推) | 堆分配改为每核复制的确定性 bump；回收改为全局完成前沿（§9） |
| per-depth “stack of ring” 的 TaskRing | 收敛为每核私有环（§5）+ scope arena 栈（§9）；堆 arena 仍按 scope 分层 |
| `scope_tasks[]` / `scope_begins[]` 的 fanout 引用记账 | scope 不再持有 +1 fanout 引用；生命周期由窗口/前沿 + arena 折叠界定（§9） |

编排 API 表面（`PTO2RuntimeOps`、`rt_submit_*`）**保留**；只有 `submit_task` 背后的实现改变
（认领 → 无条件 TensorMap 更新 → 有条件的私有环构建 → 稍后执行）。

## 11. 实现规范（原开放问题的决议）

本节把先前列为开放的问题逐一定为具体方案。先约定全局常量：

| 常量 | 含义 | 默认 |
| ---- | ---- | ---- |
| `W` | 全局窗口（`task_completed_flag` 环、复制 TensorMap、GM 堆共用），2 的幂 | ≥ `Δ + H` |
| `Δ` | 任一核相对全局完成前沿可向前跑的最大 id 跨度（由反压封顶） | 由 `PRIVATE_TASK_SLOT_NUM`、堆容量决定 |
| `H` | 依赖跨度上界：任一 producer 的最后消费者 id ≤ producer id + `H`。**由 SCOPE 决定**（PC 退出 scope 即终结其内变量可见性，故 `H` = 最大 scope 任务跨度，详见 §6.6） | 真实 PYPTO 随 scope 动态定界；a2a3 原型用保守常数 `kHDefault=64`（`PTO_DIST_H` 覆盖）近似 |
| `F` | 全局连续完成前沿：使所有 id ≤ `F` 的任务都已完成的最大前缀 | 运行期推进 |
| `R` | 回收前沿 `= F − H`：id ≤ `R` 的输出可安全回收 | 由 `F` 推导 |
| `BLOCK_WON_SLOTS` | 每 block 的 `block.won` 投递环容量 | `PRIVATE_TASK_SLOT_NUM`(=8) |

### 11.1 Claim 原子性 + 两条流的无跳过（原“Claim 原子性”“每 anchor 类型 claim 计数器”）

**原语：单条 `atomic_fetch_max`。** 一个类型为 `T` 的核到达任务 `N` 时执行
`old = atomic_fetch_max(cursor[T], N)`（`cursor[T]` 为 GM 上一个字），**`old < N` 即胜出**，
否则 `N` 已被认领。单原子、无循环。若硬件无 `fetch_max`，等价 CAS 回路：
`do { c = load(cursor[T]); if (N <= c) return LOST; } while (!CAS(cursor[T], c, N)); return WON;`
内存序取 **acq-rel**（release 发布胜利，acquire 观察既有认领）。所有权判定只依赖 cursor 本身；
真正的产出数据另由完成标志同步（§11.5）。

#### 11.1.1 A5 onboard 实现：硬件 `atomicMax` 直接维护全局 cursor

**关键结论（推翻旧 §16.2/C1 假设）：A5（`dav_3510`）拥有可用的、核间一致的 GM 硬件原子。**
CANN `dav_3510` 的 `kernel_operator_atomic_impl.h` 暴露一组作用于 `__gm__` 地址的原子内建：
`atomicAdd` / `atomicMax` / `atomicMin`（支持 `int32_t/uint32_t/int64_t/uint64_t/float`）、
`atomicCAS` / `atomicExch`（`uint32_t/uint64_t`）。CANN 文档 [AtomicMax](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_00261.html)
明确给出**三核并发 `AtomicMax` 得到正确结果、且各核拿到本次原子操作前的旧值**的示例——即这是**内存级、核间序列化**的真原子，**不需要 uncacheable 内存别名**（本机 double page table 不可用，见 §16.2 修订）。因此 §11.1 中"若硬件无 `fetch_max`"的 CAS 回退分支在 A5 上**不再需要**。

**global task cursor（全局，跨核共享）** —— 每类型一个（或分片，见 §7.2）GM 上的 `int32_t`，仅由
硬件 `atomicMax` 维护。认领即一条原子：

```cpp
// T ∈ {cube, vector, alloc}；cursor[T] 为 GM int32_t，初值 -1
int32_t old = atomicMax(&cursor[T], N);   // 硬件原子：内存级、跨核序列化；返回操作前旧值
bool won = (old < N);                     // old < N ⇒ 本核把 cursor 从 old 推进到 N，独占 N
```

- **无 CAS 回路、无单独的 `load`**：`atomicMax` 一次完成"读旧值 + 取大发布"，返回值即判定依据。
  这从根本上绕开了"先 `load` 读到 cacheable 陈旧副本再 CAS"的隐患——旧路径的 `claim()` 用
  `coherent_load` 起头，在真机 cacheable GM 上会读到本核的陈旧缓存值（见下"coherent 读"）。
- **恰一胜者且无跳过**：`atomicMax` 的单调性与 §11.1 正文一致——每个 `T` id 恰被一个核置位，
  cursor 只在 `T` 子序列上单调跃进，不跳过任何 `T` id。

**local task cursor（每核私有）** —— `local_current_task_index`（实现里的 `self->local_index`）是
本核 replay submit 流时到达的任务 id，**纯每核变量、非共享、不加任何原子**，随核走位自增。它与
global cursor 的唯一交互就是上面那条 `atomicMax(&cursor[T], local_index)`。

**cursor 的"coherent 读"（非认领场景）** —— 少数地方需要**读**（而非推进）global cursor：run-ahead
节流（§6.1，比较本核 `local_index` 与最慢核进度）、以及诊断打印。真机 cacheable GM 上普通 `load`
读到的是本核缓存副本（可能陈旧）。用一条**幂等原子**做一致读，避免陈旧：

```cpp
int32_t cur = atomicMax(&cursor[T], INT32_MIN);  // 恒不推进（任何真值 ≥ INT32_MIN），返回内存中真值
```

即"以 `INT32_MIN` 取大"永不改变 cursor，却经原子单元读回内存里的当前真值。cursor 初值 `-1`、
运行期只增，故 `INT32_MIN` 是安全的 no-op 下界。（等价替代：读前对该 cache line 做 `dcci` 失效再普通
`load`；二者皆可，优先用幂等原子，省一次 cacheline 失效且与写路径同一原子单元、序更清晰。）

> 适用范围：本节把**全局 cursor 的认领与读**在 A5 上定死为硬件 `atomicMax`。其余跨核共享量
> （完成前沿 `frontier`、启动/回放 barrier `started_count`/`replay_done`、完成标志环 `flags[]`、
> `block.won` 的 `remaining`/`state`）遵循**同一原则**——RMW 走硬件原子（`atomicMax`/`atomicAdd`/
> `atomicCAS`）、纯读走幂等原子或 `dcci` 失效——实现细节与落地顺序见 §16.4 修订后的阶段计划。

**恰一胜者且无跳过（取代“claim 计数器”）。** 每个 `T` 核按 id 递增顺序遇到 `T` 任务，`cursor[T]`
只会取到真实的 `T` 任务 id 值。在任何核尝试第 `k` 个 `T` 任务 `t_k` 之前，它必先尝试过 `t_{k-1}`
（于是其时 `cursor[T] ≥ t_{k-1}`）；而 `cursor[T]` 的相邻取值之间没有别的 `T` id，故它只能从
`t_{k-1}` 跃到 `t_k`——**不跳过任何 `T` id，且每个恰被一个核置位（fetch_max 的单调性保证）**。
`cube_cursor` 与 `vector_cursor` 各自对自己的子序列单调推进、互不干扰，全局任务 id 仍是单一确定
序列。两个 cursor 的存在与必要性见 §2、§3.1。

### 11.2 `block.won` 容量与反压（原“`block.won` 投递表大小与偏移”）

- **容量**：每 block 一个小定长环，`BLOCK_WON_SLOTS`（默认 = `PRIVATE_TASK_SLOT_NUM`）个条目，
  每条目 = 一个多核任务推送给本 block 的子任务集 + 剩余计数。界限依据：anchor 的超前量本就被其
  自身私有环（很小，§5/§6.1）封顶，每赢一个多核任务至多占 anchor 1 个环槽 + 1 个 `block.won`
  条目，故与私有环同样大小即足够（可更小）。
- **反压（已落入 §6 伪代码）**：anchor 在**认领之前**（步骤 2）检查 `block.won` 是否有空位；满则
  **本轮不认领**（不执行 `fetch_max`），下一轮回到步骤 1 执行就绪任务（从而让 follower 抽取、腾空
  `block.won`）。被让出的多核任务由**另一个有空闲的 block 的 anchor 认领**（天然负载均衡）或本核
  稍后重试。
- **无死锁**：根任务无依赖恒就绪；执行持续腾空私有环与 `block.won`；DAG 无环 → 前向进展恒成立。
  唯一残留是 §8 的尾部空转。

### 11.3 完成标志环大小与回绕（原“`task_completed_flag` 环大小与回绕”）

- `task_completed_flag` 是 `W` 个一次性置位布尔的环，`flag(N)` 位于 `N & (W−1)`。
- **`W` 取 2 的幂且 ≥ `Δ + H`**：`Δ` 是最快核相对完成前沿的最大超前（由私有环 + 堆反压封顶），
  `H` 是依赖跨度上界（§11.4）。同一个 `W` 同时给复制 TensorMap 与 GM 堆开窗。
- **回绕/ABA**：当回收前沿 `R`（§11.4）越过 `N` 时，把 `flag(N)` 复位为 false，槽位让给 `N+W`。
  不变式：消费者只在构建了依赖 `N` 的任务**之后**（即走位已过 `N`）才轮询 `flag(N)`，而 `W ≥ Δ+H`
  保证 `N` 的标志仍被需要时 `N+W` 尚未被认领 → 不会别名。**更稳健的可选做法**：在槽内连同 true 写入
  producer 的 `N`（消费者校验 `slot.id == N`），用代/epoch 戳彻底杜绝 ABA，与 `W` 大小无关。

### 11.4 GM 堆细化：`H`、容量、前沿推导、外逃输出（原“GM 输出堆的细化”）

- **`H`（依赖跨度上界）**：**由 SCOPE 决定，不是固化常数**（详见 §6.6）。tensor 的可见域就是其所在
  `PTO2_SCOPE`；orchestrator 的 PC 退出该 scope 后，scope 内变量不再可见、不会被后续任务引用，故依赖
  跨度天然被"所在 scope 的任务跨度"封顶，`H` ≈ 最大 scope 任务数（+ 并发 scope 余量）。真实 PYPTO 据此
  随 scope 进出动态定界（按 scope 深度分环，内层 scope 完成即独立回收，见 a5 `MULTI_RING.md`）。
  **本 a2a3 原型**（`dist_scope_begin/end` 为空 stub）用保守常数 `kHDefault=64`（`PTO_DIST_H` 覆盖）作为
  "最大 scope 跨度"的静态上界近似。运行期校验：若某消费者的 producer id < (当前 − `H`)，或某分配将覆盖
  尚不可回收的区域，即判为容量/配置错误（类比旧模型的 heap-deadlock 诊断）→ 调大 `H`/堆，或细化 scope。
- **堆/arena 容量** ≥ 工作集 = 窗口 `(R, top]` 内各任务输出大小之和；超出则报诊断。
- **`F`（连续完成前沿）**：全局原子、单调。**协作式推进**——任一核置位 `flag(N)` 后，
  `while flag(F+1) == true: CAS(F, F, F+1)`。无锁、任意核可推进、开销摊薄。
- **`R = F − H`（回收前沿）**：全局派生量。某 arena 的 `heap_tail` = 任务 `R` 在该 arena 内的确定性
  偏移；因布局确定，每个核都算出相同的 `heap_tail`。核要在确定性偏移 `X` 上分配任务 `M` 时，须等
  `X` 处上一占用者的任务 id ≤ `R`（即回收已到位）——这就是堆侧反压。
- **外逃输出（promotion 的处置）**：**默认不做运行期提升**。堆按单一全局确定性 bump + 前沿回收
  （§9.5），它对任意依赖（含跨 scope）都正确，无需前向信息。**scope-arena 折叠**（scope_end 处
  LIFO 即时回收）只作为**可选优化**，仅施加于**静态可证/标注为“无外逃”**的 scope；含外逃输出的
  scope 退回前沿回收。如此既无需在产出时预知外逃，也保证正确。
- **“按 tensor 最后消费者”的精确回收**：**降级为可选优化，正确性不依赖它**。精确的最后消费者需要
  前向信息/两遍扫描/引用计数（已移除），故以 `H`-窗口为已定的主用机制；精确回收作为省内存改进
  留作未来工作（不阻塞）。

### 11.5 跨核标志可见性（原“跨核标志可见性”）

- **producer 次序**：写输出到 GM → 把输出区域 writeback/flush 到所有核读取的一致性点（GM/L2）→
  **release-store** `flag(N) = true`。
- **consumer 次序**：**acquire-load** `flag(N)`；见 true 后（acquire 栅栏）再读 producer 的输出区域；
  非一致缓存平台上对该区域做 invalidate 或旁路缓存读。
- **一致缓存平台**：标志字上的 release/acquire 即足够。**非一致平台**：在标志发布/观察前后，对**数据
  区域**显式 writeback（producer）/ invalidate（consumer）。
- `cursor[T]`、`F`、`R` 等原子量统一取 acq-rel（§11.1）。

### 11.6 异步 / SDMA kernel（原“异步/SDMA kernel”）

- **句柄记在私有环槽里，不是 `block.won`。** 异步算子是 owner 在执行自己**私有任务环**中的某个
  （子）任务时发起的，故异步句柄/事件记入**该私有环槽**，槽因任务尚未真正完成而**暂不释放**。
  异步本身与 `block.won` 没有直接关系——它只是把“完成动作”从*发起时刻*推迟到 *DMA 真正完成时刻*。
- Phase B 在检查依赖就绪之外，**额外轮询在飞私有环槽的句柄**；异步完成时，按 §11.5 的次序
  （先 flush）执行该（子）任务的**完成动作**，再释放槽。完成动作具体是什么取决于任务种类
  （与异步无关，沿用 §6 的完成逻辑）：
  - **单核任务（1C/1V）**：直接置 `flag(N)`。
  - **多核任务（MIX/2V）的子任务**：`atomic_dec(block.won[N].remaining)`，由把 `remaining` 减到 0
    的那个子任务最后置 `flag(N)`。**仅在此情形下，被推迟的完成动作才触及 `block.won`**——即“在
    mixed/2V 子任务内部发起异步 DMA”时。
- 消费者侧不变：仍只轮询标志，而标志只在算子（及其所属多核任务的全部子任务）**真正完成后**才被置。
- **反压**：在飞异步算子数量被私有环容量天然封顶。

**这一步轮询由谁做：每个核自己做，不专设 AICPU。**

- **决策**：在飞句柄由**发起该算子的 owner 核**在自己的 Phase B 中轮询，**不**引入一个专职轮询的
  AICPU。理由：
  1. **不违背全局目标**——本设计的根本目的就是把编排/调度从 AICPU 移除、SPMD 分布到 AI 核；专设
     AICPU 轮询器等于请回集中式部件，并制造单点。
  2. **保持单一 owner、无锁不变式**——置 `flag(N)`、释放私有环槽、递减 `block.won[N].remaining`
     都是 owner 的本地动作（owner = builder = executor = completer）。让 AICPU 代劳就要写别人“单一
     owner、无锁”的私有环与 block-共享计数，反而需要加锁/协调。
  3. **边际成本近零**——Phase B 本就逐槽遍历私有环查依赖就绪，顺带读一次在飞槽的句柄状态仅多一次
     状态读；在飞数被私有环容量（`PRIVATE_TASK_SLOT_NUM`）封顶。
  4. **异步算子本就并行**——SDMA 跑在 DMA 引擎上，核在此期间继续编排/执行其它任务，只在 Phase B
     间隙轮询，不占算力。
- **可选硬件辅助（不改变上述归属）**：若异步引擎能在完成时**自行写一个内存位**或**发事件**，则
  - 让引擎按 §11.5 的次序直接置 `flag(N)`：消费者照常轮询标志，**无核需要为“发布完成”而忙等**；
    owner 只需在下次访问该槽时**惰性**释放槽并递减 `remaining`（届时已见标志置位）。
  - 或：尾部空转的 owner（§7/§8，已无其它就绪工作）**等待该完成事件**而非忙轮询。

  两种辅助都仍由 owner 收尾，不引入集中式 AICPU 轮询器。

### 11.7 仍然开放

- **MIX 配对 —— 动态替代方案：** §3.1 规定*固定* block 配对（AIC_c + AIV0_c + AIV1_c）。
  **平台依据：在 A5 平台上，block 由硬件把 1 个 AIC + 2 个 AIV 固定绑定**，因此面向 A5（及当前
  目标核）开发时，**采用固定配对、不做动态 co-owner 匹配是合理且既定的选择**——它与硬件 block
  边界天然对齐，省去跨 block 的认领协调与正确性论证负担（§3.2）。
  动态配对方案（跨 block 均衡 MIX 工作；亦即 §3.2 讨论并暂不采用的“block 内先到先得代发布”等
  思路的归宿）**仅在未来核解除该硬件绑定时**才需要，届时再行设计，**本节不予裁定**。

## 12. TensorMap 构建与 Private / Shared 双模式（统一 ring-per-bucket）

本章新增一个**正交的运行模式开关**：TensorMap（§4、§8.2、§9）既可以保持"每核全量复制"形态
（private），也可以改为"全局共享一份"（shared）。二者由**命令行开关**在运行启动时一次性选定，
贯穿整次运行不再切换。本章先回顾 TensorMap 如何被构建，再**论证两种模式统一采用同一套
ring-per-bucket 数据结构**（§12.3 的 链表 vs ring 性能分析），随后定义两种模式仅有的差异、重点分析
**shared 模式下无缓存一致性平台上的数据一致性问题**并给出解决方案，最后比较两种模式的性能、
**论证 `auto` 定容的安全性与实现**（§12.7.2），并说明顶层入口（TensorMap 指针）如何被改造为支持双模。

> **规范约定（本章基准）**：**private 与 shared 两种模式统一采用 ring-per-bucket（每桶一个有界环形
> 缓冲，§12.7.1）作为唯一数据结构**；二者仅在"副本数 / 谁 insert / 并发纪律 / 回收阈值"四点上分叉
> （对比见 §12.3.2）。之所以统一：§12.3 的分析表明 **ring 在 private 与 shared 下均不劣于链表、且多数
> 维度更优**（§6.4 的 O(N) 收益来自"回收窗口 + 哈希分桶"而非"链表"这一存储形态，ring 完整继承之，
> 另在局部性/内存/回收常数上取胜）。早期为 shared 考虑的"桶内链表 + 空闲链"方案因空闲链 ABA /
> `next` 悬挂指针 / use-after-recycle 等一致性陷阱**已被否决**（§12.7 作对照基线保留）；§6.4 为 private
> 实现的链表结构在本章分析下**亦被 ring 取代**（其回收窗口/哈希/winner-only 等洞见原样继承）。ring 的
> 容量 `CAP`、溢出报错与命令行参数见 §12.7.2；`auto` 定容的安全性论证见 §12.7.2.3。

### 12.1 TensorMap 的构建回顾

TensorMap 把一个 tensor 区域 `[lo, hi)` 映射到其 **producer 任务 id**。它的构建规则在 §4 已定，
此处重述为可被两种模式共用的"构建原语"：

- **查（lookup）**：给定一个 `INPUT`/`INOUT` tensor 区间，找到与之重叠、且 producer id **最大**
  （最新）的条目，作为该 fan-in 的 producer。`INOUT` 两侧都参与：先查（消费旧版本）再插
  （产出新版本）。
- **插（insert）**：给定一个 `OUTPUT` 或 `INOUT` tensor 区间，以**本任务 id** `N` 作为 producer
  登记一条新条目。

§6.4 曾为 **private 模式**把 `DistTensorMap` 物理结构定为"按 buffer 基址哈希分桶 + 桶内链表 + 按
生产者任务的 entry 链 + 空闲链表 + lazy invalidation + `cleanup_retired` 按任务回收"，且 **insert 总是
挂新条目**（不做就地替换），`lookup` 返回重叠者中 producer **最大**的那个。**但本章 §12.3 的性能分析
表明该链表结构应被 ring-per-bucket 取代**——不仅 shared 必须用 ring（§12.7），private 用 ring 也全面
不劣于链表且更省更快。故**两种模式统一采用 ring-per-bucket**；§6.4 的回收窗口（`N−H`）、哈希分桶、
"多版本共存、不就地替换"、winner-only fan-in 等语义与洞见**原样继承**到 ring 实现（§12.5/§12.6）。

两种模式的差异不在于"构建原语"或"数据结构"本身（统一为 ring），而只在于：**insert/lookup 作用在
哪一份 map 上、由谁来执行、以及跨核可见性如何保证**（四点差异见 §12.3.2）。

### 12.2 命令行开关

新增一个启动期开关（环境变量与 CLI 同义，沿用 §6.3 的 `--bind` 风格）：

```
--tensormap-mode {private|shared}        # 等价环境变量 PTO_DIST_TENSORMAP_MODE
--tensormap-ring-cap {N|auto}            # 等价环境变量 PTO_DIST_TENSORMAP_RING_CAP；private/shared 均生效
```

- `private`（**默认**）：每核一份全量复制 map，数据结构为 ring-per-bucket（每核私有、单线程纪律，§12.3.2）。
- `shared`：全核共享**唯一一份** TensorMap，数据结构同为 ring-per-bucket（并发纪律 + per-slot `seq`，§12.7.1）。
- `--tensormap-ring-cap`：**两种模式均生效**（两者都用 ring），设定每桶 ring 的定长槽数 `CAP`（2 的幂）。
  默认 `auto` = 由依赖跨度 `H`（private）或 `Δ+H`（shared）与该桶静态区域分布推导（§12.7.2）。**`auto`
  在两种模式下都能给出可证充分的容量**（§12.7.2.3）。

开关在 runtime 初始化阶段被读取一次，据此构造对应形态的 TensorMap 句柄（§12.9）并选择对应的
insert/lookup 实现。**运行期不切换**，避免中途一致性灾难。所有 per-core 编排循环（§6）通过同一
组抽象 API（`tm_insert` / `tm_lookup`）访问 map，由句柄分发到 private 或 shared 的 ring 实现——上层
伪代码不变。`--tensormap-ring-cap` 影响两种模式每桶 ring 的分配尺寸，完整论证（含 `auto` 安全性）
见 §12.7.2。

### 12.3 为何统一到 ring-per-bucket：链表 vs ring 性能分析

**关键澄清（先破一个误解）。** §6.4 的 O(N) 收益**来自"按 H 窗口回收 + 哈希分桶"，与"链表 vs 数组"
这一存储形态无关**。ring 完整保留同一个回收窗口（private 用确定性 `N−H`，shared 用全局前沿 `R`）
与同一套哈希分桶，只把"桶内链表 + 空闲链"换成"桶内定长环 + 游标"。因此 **ring 继承 §6.4 的全部渐进
收益（仍是 O(N)）**，差异只在**常数因子**（局部性、内存、回收开销）与**并发友好度**上——而这些都对
ring 有利。

#### 12.3.1 逐维度对比（同一模式下，链表 vs ring）

| 维度 | 链表（原 §6.4） | ring-per-bucket | 谁更优 |
| ---- | ---- | ---- | ---- |
| **lookup 访存** | 指针跳转，节点随机布局，每跳可能一次 cache miss | 连续数组扫描，硬件预取友好，触达 cache line 更少 | **ring** |
| **insert** | 从空闲链取节点 + 挂两条链（bucket 头 + 生产者链）指针操作 | `slots[tail%CAP]=e; tail++` 一次连续写 | **ring** |
| **回收** | 沿生产者任务链逐节点摘除并归还空闲链 | `head++` 游标自增，不摘链、不归还 | **ring** |
| **每 entry 内存** | payload + ~3 指针（bucket next / 生产者链 next / freelist next，≈24B） | payload only（下标隐式，无指针） | **ring** |
| **容量弹性** | 不定长，随空闲链弹性伸缩，无"满"概念 | 定长 `CAP`，需定容；满则反压/报错 | **链表**（唯一劣势，auto 可证充分定容化解，§12.7.2.3） |
| **并发（shared）** | 空闲链 CAS / ABA / use-after-recycle（§12.7 最难处） | per-slot `seq` + 游标，**无空闲链、无 next 指针** | **ring（决定性）** |
| **并发（private）** | 无（每核私有），但仍付指针/空闲链簿记 | 无，且退化为**纯整数 head/tail**（无 `seq`、无原子） | **ring**（更省） |
| **渐进复杂度** | O(N)（H 窗口回收之效） | O(N)（同一回收窗口） | 平 |
| **确定性/回收阈值** | private `N−H` | private `N−H`，shared `R`（§12.7.1） | 平 |
| **实现/验证面** | private 链表、shared ring → **两套结构** | 两模式**同一套 ring** | **ring（工程）** |

**private 专门分析。** private 无跨核并发，ring 在此**退化为最简形态**：`head`/`tail` 是**普通整数**
（无需 `seq`、无需原子、无 ABA），insert = "写槽 + `tail++`"，lookup = "从 `tail-1` 向 `head` 连续扫"，
回收 = "`while slots[head%CAP].producer_id ≤ N−H: head++`"。相比 §6.4 链表，它**同为 O(N)**，但
①lookup 连续扫描（链表是指针跳转，局部性差）；②每 entry 省约 3 指针 + 整条空闲链；③回收从"走生产者
链 + 归还空闲池"简化为"游标自增"。**唯一代价**是定长 `CAP`——但 private 回收阈值是**确定性 `N−H`**、
任务图**静态已知**，故每桶存活槽数上界**可在构建期精确算出**，`auto` 能给出**可证不溢出**的 `CAP`
（§12.7.2.3）。故 private 下 ring **全面不劣于链表、且更省更快**。

**shared 专门分析。** 已由 §12.7 定论：链表空闲链在无缓存一致性平台上引出 ABA / use-after-recycle
两大最难陷阱，ring 用"游标自增回收 + per-slot `seq`"直接消去（§12.7.1）。故 shared **只能是 ring**。

**结论。** ring 在 **private 与 shared 下均不劣于链表**：private 赢在常数因子（局部性/内存/回收）且退化
到无原子最简形态，shared 则**只有** ring 可行；再加**只需维护/验证一套数据结构**的工程收益。因此
本章**弃用 §6.4 的 private 链表，两模式统一为 ring**（§6.4 的回收窗口、哈希分桶、winner-only fan-in
等洞见原样继承到 ring）。

#### 12.3.2 统一之后：两种模式仅存的四点差异

统一到 ring 后，private 与 shared **数据结构完全相同**，仅在下列四点分叉（其余——哈希分桶、多版本
追加、时序过滤 lookup、`N−H`/`R` 回收窗口——完全共用）：

| 分叉点 | private | shared |
| ---- | ---- | ---- |
| **副本数** | 每核一份（全量复制） | 全局唯一一份 |
| **谁 insert** | **所有核**都 insert（各写自己副本，保持各核一致） | **仅 winner** insert（每任务恰好一核） |
| **并发纪律** | 无：`head`/`tail` 为普通整数，无 `seq`、无原子、无 invalidate | per-slot `seq` acq-rel + `reserve`/`head` 原子 + writeback/invalidate（§12.7.1） |
| **回收阈值** | 确定性 `N−H`，每核本地推进（§6.4 语义） | 全局 `R = min_progress−H−1`，协作推进（§12.7.1/§9.5） |

- **一致性**：private 每核只读写自己的副本，**无跨核可见性问题**（producer 数据可见性仍由完成标志环
  §11.5 保证）；shared 是并发单副本，一致性由 §12.7.1 的 `seq`/acq-rel/游标纪律保证。
- **代价权衡**：private 内存 = `核数 × 单份`、每核为全部任务付 insert 地板；shared 内存 `1×`、insert
  仅 winner，但引入跨核 invalidate 流量与热桶 `reserve` 竞争。完整取舍见 §12.8。
- **lookup**：两模式都是"连续扫描 + 时序过滤取最新合法"（§12.6）；private 少了 per-slot `seq` 校验与
  invalidate（纯本地读）。

### 12.4 shared_tensormap —— 单副本 + winner-only insert

> **核心观察**：claim race（§2）使走得最快的核（winner）在任务 id 序列上**领先**于其它核。winner
> 先构建并执行靠前的任务，因此**它的 TensorMap 进度也领先**——它刚 insert 的条目，正是落后核稍后
> lookup 时所需要的。于是存在一种可能：**让 winner 把它 insert 的条目直接发布给所有核共享**，
> 落后核无需自己 insert、直接查这份共享 map 即可。

shared 模式据此重新划分职责：

- **形态**：全核共享**唯一一份** TensorMap，物理上驻留在一块全局可寻址的 GM 区域，组织为
  **ring-per-bucket**——按 buffer 基址哈希分桶，**每个桶是一个定长 `CAP` 槽的有界环**（`RingBucket`，
  §12.7.1），只有 `head`（回收游标）/ `tail`（发布游标）/ `reserve`（MPSC 抢槽游标）三个原子字，
  外加每槽一个 `seq` 代戳。**没有链表节点、没有 `next` 指针、没有空闲链。**
- **谁构建（insert）**：**仅 winner 做 insert**。败者与 follower 在走位到任务 `N` 时**不再** insert——
  因为 winner 的 insert 已经（或即将）对全核可见，落后核重放 insert 既冗余又会与 winner 抢同一份
  map。insert = `k = fetch_add(reserve, 1)` 抢一个确定下标 `k % CAP` → 写槽字段 → writeback →
  release-store 该槽 `seq`（§12.7.1）。这一改动直接消除了 §6.3/§6.4 中"每核为全部任务 insert"的地板
  开销，是 shared 模式在多核下的主要性能收益来源（§12.8）。
- **谁查（lookup）**：任何核在赢得任务 `N`、需要解析 fan-in 时，都查这一份共享 ring：从 `reserve`
  往 `head` 方向扫连续槽（无指针跳转），对每槽 acquire-read `seq` 校验有效后读字段，应用 §12.6
  时序过滤取合法者中 producer 最大（§12.6）。
- **回收**：由全局 reclaim 前沿 `R`（基于**各核进度最小值**，§9.5）驱动，**回收即游标自增**——当
  `slots[head % CAP].producer_id ≤ R` 时 `head++`，既不摘链也不归还空闲池（§12.7.1）。
- **溢出**：`fetch_add(reserve)` 后若 `reserve − head > CAP`（ring 满）则 winner **不覆写**、走
  反压/报错路径（§12.7.2），绝不静默丢条目。

shared 模式随即带来三个必须解决的问题：(A) INOUT 重写导致同一区域存在多个 producer 版本，如何
在共享 ring 中表达（§12.5）；(B) 落后核 lookup 时如何避免看到"未来 producer"（§12.6）；(C) AI 核
**无缓存一致性**，共享 ring 的跨核数据一致性如何保证（§12.7）。下面三节逐一展开，**均以 ring 为准**。

### 12.5 INOUT 重写：多版本以 ring 槽共存（append，不摘链、不替换）

INOUT tensor 既消费旧版本又产出新版本（§4）。在 claim race 下，同一区域可能被多个 winner 先后
以 INOUT 方式写入，从而**同一区域在共享 ring 中存在多个 producer 版本**。private 模式里这不是问题
（每核自己 insert，lookup 取最新即可）；shared 模式下，若试图"用新 producer **替换**旧槽"，
会丢失旧版本——而落后核此刻可能仍需要旧版本作为它的 fan-in（它的"现在"还没到新 producer）。

**规则（共享 ring 的多版本追加）：**

1. **绝不就地替换 producer。** 当某区域的 producer 被更新（例如 INOUT 重写），**不**修改任何已发布
   槽的字段（§12.7.1 的"发布即不可变"），而是**追加一条新槽**，其 producer id = winner 的任务 id `N`。
2. **追加落在 ring 尾部。** `k = fetch_add(reserve, 1)` 抢一个确定下标 `k % CAP`，写入
   `{producer_id=N, region, ...}`，writeback 后 release-store 该槽 `seq = k + lap*CAP`（§12.7.1）。
   旧槽**原样保留**在环内，仍携带它更老的 producer id，直到 `head` 越过它被回收。
3. **lookup 取"最新可见且合法"者。** 从 `reserve` 往 `head` 方向扫环内有效槽（seq 校验通过），在所有
   与查询区间重叠者中，选 producer id 最大、但又满足下节 §12.6 时序合法性的那一个。

如此，同一区域的多版本 producer 在共享 ring 中以**按 append 次序（≈producer id 升序）排列的连续槽**
共存，旧版本随 reclaim 前沿 `R` 推进、`head++` 而被回收（§12.7.1）。相比被否决的链表方案，ring 用
"下标 + `seq`"取代"`next` 指针 + 桶头替换"，多版本共存无需任何指针操作，扫描局部性更好。

### 12.6 跳过"未来 producer"：以本地任务索引为时序过滤

> **问题**：winner 走得快，它 insert 的条目 producer id 可能**大于**某个落后核当前的
> `local_current_task_index`。若落后核在解析自己位于 id `N` 的任务的 fan-in 时，查到了一个
> producer id `P > N`，那它就**引用了一个属于它自己未来的任务**——该未来任务可能尚未执行、其
> 完成标志未置位、其输出数据尚不可读，于是消费者会错误地阻塞等待一个"未来 producer"，或更糟，
> 读到未完成的数据。这本质上是"把 winner 的时钟强加给落后核"。

**规则（时序过滤）：** 任何核在 lookup 时，**跳过 producer id ≥ 自身 `local_current_task_index`
的条目**。设本核当前走位到任务 `N`（即 `local_current_task_index == N`），则只接受 producer id
`< N` 的条目作为合法 fan-in。ring 版 lookup 扫描的是**连续槽下标**（无指针跳转），每槽先 acquire-read
`seq` 确认有效（§12.7.1 防 ABA）再读字段：

```text
lookup(region, N):                       # N = 本核 local_current_task_index
    best = NONE
    b = bucket_of(region)                 # 定位桶（该桶的 RingBucket）
    hi = acquire_load(b.reserve)          # 已抢到的最高下标（发布上界）
    lo = acquire_load(b.head)             # 回收游标（最旧仍存活槽）
    for k in range(hi-1, lo-1, -1):       # 从最新 append 往最旧扫连续槽
        s = &b.slots[k % CAP]
        if acquire_load(s.seq) != k       # ★per-slot seq 校验：槽已被复用/未发布 → 跳过
            continue                       #   （非一致平台先 invalidate 该 slot cache line）
        if s.producer_id < N              # ★时序过滤：跳过"未来 producer"
           and overlaps(s.region, region):
            if best == NONE or s.producer_id > best.producer_id:
                best = snapshot(s)         # 取合法者中最新；拷出快照，不缓存槽指针
    return best                            # 可能返回 NONE（尚无合法 producer）
```

> ring 扫描按下标从 `reserve-1` 递减到 `head`，因追加近似按 producer id 升序，故先遇到者即较新；
> 也可一旦命中一个 `producer_id < N` 的重叠槽就提前返回（该方向上它已是最大合法者）。`seq != k`
> 表示该槽尚未发布或已被后续 lap 复用（§12.7.1），一律跳过；**全程不跨调用缓存 `head`/槽指针**。
>
> **注（落地优化）**：上面每槽 `acquire_load(s.seq)` 是**通用 MPSC** 设计所需（`reserve` 只是抢槽游标、
> 非发布水位）。实际实现采用**单一串行追加者**（§12.10(1)），`tail` 成为真正的发布水位，故 reader 只需对
> `tail` 做**一次** acquire、其下各槽 `seq` 改 relaxed 读即可——把"每槽一次 acquire"摊薄为"每 lookup 一次
> acquire"，详见 §12.10(4)。

**为什么用 `local_current_task_index` 而不是全局前沿 `F`。** 时序合法性是**每核本地的时钟**概念：
"我还没走到 id `P`，就不该把 `P` 当作我的 producer"。各核的 `local_current_task_index` 严格单调地
跟着自己的走位推进，是本核"当前时刻"的权威；而 `F` 是全局完成前沿，与"我是否已到达 `P`"无关。
用本地索引作阈值，确保每个核只引用**自己时间线上的过去**。

**返回 NONE 的处置。** 若整个环内没有任何 producer id `< N` 的重叠有效槽（winner 还没 append 到此
区域、或本核是该区域的第一个 producer），则该 fan-in 解析为"无 producer"——即本任务的该输入是
图的外部输入（host 提供的初始 tensor），不需要等待完成标志。这与 private 模式下"查不到 = 外部
输入"的语义一致，只是 shared 模式下"查不到"还可能是"winner 尚未发布"——但二者对消费者行为相同
（都不等任何 producer），且当本核确实是该区域首任 producer 时为正确；当本核并非首任、只是 winner
尚未发布时，见 §12.7 末尾的"发布保证"。

### 12.7 无缓存一致性下的数据一致性分析（核心）

> **本节定位**：shared 模式的**规范数据结构是 ring-per-bucket（§12.7.1）**，其一致性方案见 §12.7.1、
> 容量与溢出见 §12.7.2。本节 §12.7 先分析"无硬件一致性平台上共享 map"的**通用难点**，并以最初设想
> 的**"哈希桶 + 链表 + 空闲链"方案为对照基线**说明"为何不用链表"——这些难点（尤其空闲链 ABA 与
> `next` 悬挂指针）正是 ring 设计要规避的。通用的无一致性纪律（release/acquire 发布、writeback +
> invalidate）对 ring 同样适用；ring 的具体落地见 §12.7.1。**链表方案不作为实现，仅作动机保留。**

这是 shared 模式最困难的部分。**AI 核之间没有硬件缓存一致性**（§11.5 已就此为完成标志专门处理）。
若把共享 TensorMap 实现成一块被多核并发读写的复杂数据结构（哈希桶 + 链表 + 空闲链），其一致性不能
想当然——下面逐条剖析，正是这些陷阱促成了 §12.7.1 的 ring 决策。

**问题一：仅 invalidate 新插入条目的数据，足够吗？**

不够。winner insert 一条新 entry 时，若只把自己写的这条 entry 的 cache line invalidate/flush 到
GM，其它核仍可能基于**陈旧的桶 head 指针**导航——它们根本看不到新 entry 存在。一致性故障点至少
有四处，而非一处：

| 故障点 | 现象 | 仅 invalidate 新条目能解决吗 |
| ------ | ---- | ---- |
| (a) 桶 head 指针 | 其它核 cache 里仍是旧 head，永不到达新 entry | **否**——head 在另一条 cache line 上 |
| (b) 新 entry 的字段（producer id / region / next） | 其它核读到新 entry 但字段为旧值/撕裂 | 部分——需写回 + 读侧 invalidate |
| (c) 旧 entry 的字段 | 旧 entry 被 winner 保留不修改，但若被回收复用则字段被改写 | **否**——见问题二/三 |
| (d) 空闲链 / entry 池复用 | 一个被回收的 entry 被重新分配、改写，而某核仍持有旧指针在读它 | **否**——经典的 use-after-recycle |

**问题二：其它核是否会使用"过时的 tensormap 数据结构"？**

会，且有两种"过时"：

1. **结构性过时（miss 新 head）**：落后核 cache 里的桶 head 是旧值，于是它遍历的是**旧链表前缀**，
   完全错过 winner 新挂的 head 条目。后果：lookup 漏掉最新 producer，退而取到次新的合法 producer
   （§12.5 多版本链表使次新仍可用），**语义上仍正确**，但可能不是最新的过去版本——这在 INOUT
   场景下意味着消费了一个较旧版本的数据（见下文"数据正确性"）。
2. **悬挂指针过时（use-after-recycle）**：落后核正遍历链表到 entry `e`，此时 reclaim 把 `e` 回收
   并分配给另一个 winner 改写。落后核继续读 `e.next` / `e.producer_id`，读到**新写入者的内容**，
   指向完全无关的区域/任务 → 错误依赖，可能挂死或读错数据。

**问题三：数据正确性（不只是元数据）。**

TensorMap 只是元数据；真正的产出数据在 GM 堆，由 §11.5 的完成标志 + writeback/invalidate 保证可见。
§9.3 的**确定性 bump 分配**确保**每个 producer 写到自己的独立地址**（INOUT 的新版本也是新地址，
**非就地覆写**），因此消费者一旦选定 producer `P` 并 acquire 到 `flag(P)=true`，从 `addr(P)` 读到的
必是 `P` 的产出，不会被未来 producer 覆写。所以 shared 模式下的**数据正确性仍由 §11.5 兜底**，
shared 模式新增的风险只在**元数据**层：选错了 producer（或读到回收后的垃圾 entry），会引用错误的
`addr(P)` / 错误的完成标志位。

**（对照基线）链表方案的解决方案——仅说明其复杂度，非本设计实现。** 若坚持用"哈希桶 + 链表 +
空闲链"，需把它当作"无硬件一致性下的并发发布数据结构"来设计，沿用 §11.5 的发布/观察纪律并补齐
回收纪律，至少需以下五条。**读者可略过细节，只需记住：其中第 3、4 条（reclaim 前沿驱动回收、空闲链
无锁栈 + 版本防 ABA）是最易出错的部分——ring 设计（§12.7.1）通过取消空闲链与 `next` 指针，直接
消去了它们。** 五条如下：

1. **桶 head 为原子、acq-rel 发布。** `bucket_head` 用一个原子字（64 位指针 + 版本 tag，见下）。
   winner：先写回新 entry 的全部字段与 `next`（§11.5 writeback），再对 `bucket_head` 做
   **release-store**；reader：对 `bucket_head` 做 **acquire-load**（非一致平台先 invalidate 该 cache
   line），拿到 head 后再 invalidate 对应 entry 的 cache line 读其字段。这解决问题一 (a)(b)。
2. **entry 一经发布即不可变（immutable after publish）。** 一条 entry 被 head 指向、对其它核可见
   后，其 `producer_id` / `region` / `next` **永不再被改写**。可变的只有 head 指针与 entry 在空闲链
   中的 `freelist_next`（且二者用同一原子字的不同位/不同字段，发布期与空闲期互斥）。这把"读 entry
   字段"从并发读写降为并发只读，消除字段撕裂。
3. **回收仅由 reclaim 前沿 `R` 驱动，且 `R` 基于各核进度最小值。** §9.5 已定义 `heap_reclaim_frontier`
   由"完成前沿 + 各核进度最小值"推导。shared 模式的 TensorMap 复用同一前沿：仅当某 entry 的
   `producer_id ≤ R` 时才允许回收。由"依赖跨度 `H` + 各核进度 ≥ 完成前沿"可知，任何核在 lookup 时
   能引用的 producer id 下界 = `其 local_index − H` ≥ `R`（因为最慢核的 `local_index` 也已超过
   `R + H`，否则 `R` 不会推进到此），故**任何活着的 lookup 都不会触及已回收 entry**——问题二(2) 的
   use-after-recycle 在不变式下不可能发生。
4. **空闲链为无锁 Treiber 栈 + 版本指针防 ABA。** 多 winner 并发 pop 空 entry、reclaim 并发 push，
   用 CAS + 指针带版本号（`head{ptr, tag}`）杜绝"同地址被多次回收再分配"造成的 ABA。这是 shared
   模式新增的唯一热点原子（除既有 cursor/F 外）；可仿 §6.6 按 `bucket_index % G` 分片以降竞争。
5. **lookup 全程不缓存 head / entry 指针。** 每次进入 `tm_lookup` 都重新 acquire-load `bucket_head`
   （invalidate 后读），遍历过程中对每条 entry 的字段读取都遵循 acquire/invalidate。**禁止跨调用
   缓存 head 或 entry 指针**——结构性过时（问题二(1)）的根因正是缓存了旧 head；强制每次重读，让
   "最新 head"在 lookup 入口处对齐到当前前沿。

**关于"结构性过时取到次新版本"的最终判据。** 即便有上述全套方案，落后核在某一刻仍可能读到一个
尚未被 winner 发布的最新 entry 之前的旧 head——但这等价于"winner 尚未发布该版本"。此时落后核
取到的是**次新的合法 producer** `P_old < N`。由 §9.3（独立地址）+ §11.5（flag(P_old) 可见即数据
可见），`P_old` 的输出是完整且正确的旧版本。**只要该消费者对"必须消费最新版本"没有强要求**，
这就是可接受的弱一致（最终一致）语义——落后核消费了一个稍旧的版本。若某任务的语义要求它必须
消费"恰好最近的前任 producer"（典型如严格 in-place 累加序列），则需在任务图层面保证该前任已完成
且其 entry 已发布——这由 §11.5 的 flag 依赖链天然保证：消费者在 acquire `flag(P)` 后才读数据，
而 `P` 的 entry 由 `P` 的 winner 在置 flag **之前**就已 insert 并 release-head 发布（insert 发生在
build 阶段、flag 置位在 execute 完成 阶段，二者顺序固定）。因此"前任已发布 entry"是"前任已完成"
的必要前置，**不会出现"前任已完成但 entry 未发布"**。综上，shared 模式在上述五条方案下达成正确的
元数据一致性，数据正确性沿用 §9.3 + §11.5。

**发布保证（回应 §12.6 末尾"winner 尚未发布"）。** 当落后核 lookup 返回 NONE 时，除"本核是该区域
首任 producer / 外部输入"外，另一可能是"前任 winner 已认领但尚未 insert"。但 insert 发生在 winner
build 该任务的早期、远早于其 execute 完成与 flag 置位；落后核若需消费该前任，必先 acquire
`flag(P)`——而 `flag(P)` 置位晚于 `insert`。故"落后核看到 flag(P) 但看不到 entry"在 acq-rel 纪律下
不可能。NONE 即真正无前任，安全。

### 12.7.1 规范数据结构：ring-per-bucket（两种模式统一的实现基准）

> **本节是 private 与 shared 两种模式共用的规范实现。** §12.7 的五件套方案是**针对"桶内链表 + 空闲链"
> 这一（被否决的）数据结构**给出的。链表方案里**最容易出错**的不是桶 head 的 acq-rel，而是**空闲链**：
> 多 winner 并发 pop、reclaim 并发 push、指针 ABA、use-after-recycle——这些才是"处理一致性容易出错"的
> 根源。因此两种模式**统一改用同一数据结构，让这些陷阱根本不出现**：**每桶一个有界环（ring-per-bucket）**。
>
> **两模式共用同一 ring，仅并发纪律不同（§12.3.2）。** 下文的 `seq` / `reserve` / acq-rel / writeback /
> invalidate 是 **shared 模式**（并发单副本）所需；**private 模式**每核私有、单线程访问自己的副本，
> **退化为最简形态**：`head`/`tail` 是普通整数，insert = "写槽 + `tail++`"，回收阈值用确定性 `N−H`
> 取代全局 `R`，**无需 `seq`、无需 `reserve` 原子、无需 invalidate**。即 private = "把下文所有并发纪律
> 关掉"的 ring。

**思路。** TensorMap 的访问模式恰好是"**增量追加（append）+ 按前沿回收（evict from front）**"：
winner 不断往一个 bucket 里 insert 新条目，旧条目随 reclaim 前沿 `R` 推进被回收。这正是 **ring
（有界环形缓冲）** 的天然工作模式——**每个 bucket 一个 ring**，只有两个游标：

```text
struct RingBucket {
    Entry   slots[CAP];     // 定长槽数组
    atom<u64> head;         // 回收游标：slots[head % CAP] 是最旧仍存活条目
    atom<u64> tail;         // 发布游标：下一个 append 落在 slots[tail % CAP]
    atom<u64> reserve;      // 预定游标：MPSC 下 winner 用 fetch_add 抢槽位
    // 每个 slot 内含一个 seq 字（见下）
};
```

- **append（winner insert）**：`k = fetch_add(reserve, 1)` → 写 `slots[k % CAP]` 的字段 →
  writeback → release-store 该 slot 的 `seq`（标记"本 lap 已发布"）。tail 由"已连续填充前缀"推进
  （或等价地，reader 直接用 per-slot `seq` 判定有效性，无需单一 tail）。
- **evict（reclaim）**：当 `slots[head % CAP].producer_id ≤ R` 时 `head++`。回收 = **游标自增**，
  不摘链、不归还空闲池。
- **lookup**：从 `reserve`（或 tail）往 `head` 方向扫 `slots[k % CAP]`，对每个 slot 先 acquire-read
  其 `seq` 确认有效，再读字段，应用 §12.6 时序过滤与重叠判定，取合法者中 producer id 最大。

**为什么这能避开"最容易出错的部分"。**

| §12.7 链表方案的陷阱 | ring 方案的处置 |
| ---- | ---- |
| **空闲链 CAS 栈（ABA / use-after-recycle / 竞争）** | **彻底消失**——没有空闲链，"回收"只是 `head++`，"分配"只是 `fetch_add(reserve)` 抢一个确定下标 |
| **`next` 指针的跨核读（每跳一条远程 cache line，且 next 本身可能被回收）** | **彻底消失**——槽是连续数组，下标由 `k % CAP` 算出，无指针跳转；扫描局部性好，invalidate 目标地址确定 |
| **回收时"摘链"可能摘掉某核正持有的节点** | **不可能**——回收只动 `head` 游标，不动任何槽内容；被回收的槽在被 `reserve` 再次追上之前不会被改写 |
| **桶 head 指针的发布/观察** | 改为 **per-slot `seq` 字**（见下），把"head 可见性"问题局部化到"单个 slot 的发布可见性"，模式更标准、更易推理 |

**新增的、但更标准的要求。**

1. **per-slot `seq` 防 ABA（关键）。** ring 是定长的，slot `k % CAP` 会被反复复用。若读者刚读完 lap `L`
   的 slot `k`、被挂起，此时槽被回收并写入了 lap `L+1` 的新条目，读者醒来若只凭"slot `k` 有数据"就会
   把 lap `L+1` 的内容当成 lap `L` 的来用——经典 ABA。解法是 bounded-queue 标准技巧：每个 slot 带
   `seq`，发布时写 `seq = k + L*CAP`（每复用一次 `+CAP`），读者记下自己期望的 `seq` 值，acquire-read
   `seq` **等于**期望值才认为该槽有效。lap 切换后 `seq` 不等 → 读者识别为"槽已被复用，停止扫描"
   （因为它的目标旧条目已不可达）。这把 ABA 从"指针级、需带版本号的 Treiber 栈"降为"整数比较"，简单
   且可局部推理。
2. **定长容量 `CAP` 须按 H 窗口定。** 每个 bucket 的存活条目数上界 = "落进该桶、producer id 在
   `(R, 最新]` 内的不同版本数"，受依赖跨度 `H`（§11.4）封顶。取 `CAP ≈ (H × 桶均条目数) × 安全系数`
   即可。**溢出**时 winner **不覆写**、走反压/报错路径。这比链表的"无界增长 + 空闲链"更省内存、更可
   预测，代价是需要正确估 `CAP`。**`CAP` 的完整取值分析、溢出报错设计与是否引入命令行参数见 §12.7.2。**
3. **MPSC 预定游标 `reserve` 是新的热点原子。** 同一桶的多个 winner 用 `fetch_add(reserve)` 抢槽，
   是 per-bucket 的 CAS 热点（类似 §6.5 的 cursor）。缓解：`reserve` 与 `head` 落在同一 cache line 会
   伪共享，需分开对齐；热桶可按 §6.6 思路分片（同一 bucket 拆 `G` 个 sub-ring，按 producer id 取模）。

**能否连 `seq` 也省掉？** 严格条件下可以。若 (i) AI 核**无 OS 抢占**（lookup 在有界本地时间内完成，
不会"读到一半被挂起很久"）且 (ii) reclaim 不变式严格成立——"slot 被回收复用"要求其
`producer_id ≤ R`，而任何 lookup 能引用的 producer id `> R`（由 `R = min_progress − H − 1` 与读者
`local_index ≥ min_progress` 推出，§12.7 不变式）——则读者**永远不会**触及一个正被复用的 slot，
`seq` 可省。但**仿真在 host 线程上跑、会被抢占**，且 defense-in-depth 更稳，故**推荐保留 `seq`**；
`seq` 成本仅每槽一个整数 + 一次 acquire 比较，远低于链表的空闲链 CAS。

**结论。** 把 bucket 从链表换成 ring，**消掉了 shared 模式里最易错的空闲链与 next 指针**，把一致性
问题收敛到"per-slot acq-rel 发布 + `seq` 防 ABA + `head` 由 `R` 推进"这一套**有界队列标准模式**，
推理局部、实现成熟。回收从"摘链 + 归还空闲池"简化为"游标自增"，append 从"建节点 + CAS 挂头"简化
为"fetch_add 抢槽 + 写 slot + 发布 seq"。这正是用**数据结构的简化**换**一致性论证的简化**——代价是
定长 `CAP` 与新的 `reserve` 热点，二者都可调/可分片。**后续 §12.8/§12.9 的 shared 实现默认采用
ring-per-bucket。**

### 12.7.2 Ring 容量 `CAP`、溢出报错与命令行参数（两种模式）

ring 是**定长**的，这把"map 无界增长"换成了"必须正确定容"。**统一到 ring 后，private 与 shared 都需
定容**，但二者的活跃窗口不同（private 窄、shared 宽），且 private 的窗口是**确定性静态可算**的。本节
回答三个工程问题：(1) 两模式 `CAP` 各取多大；(2) 满环（溢出）如何检测与报错，绝不静默丢条目或覆写；
(3) 是否/如何把 `CAP` 暴露为命令行参数、**`auto` 是否安全、如何实现**（§12.7.2.3，本章重点之一）。

#### 12.7.2.1 `CAP` 如何定：活跃版本窗口 + 安全系数

单个 bucket 在任意时刻的**存活槽数**（`tail − head`，shared 下为 `reserve − head`）有明确上界，可据此
定容。**两模式的窗口不同：**

- **private 的存活窗口 = `H`（更窄、确定性）。** private 每核用确定性阈值 `N−H` 回收自己的副本：走位到
  `N` 时，`producer_id ≤ N−H` 的槽已被回收，故存活槽的 producer id 落在 `(N−H, N]`，**窗口恰为 `H`**。
  它**不含** run-ahead `Δ`——因为每核只对自己的单一进度 `N` 回收，无跨核进度差。
- **shared 的存活窗口 = `Δ + H`（更宽）。** shared 用全局前沿 `R = min_progress − H − 1` 回收共享副本
  （§12.7.1）。最快核可领先最慢核达 `Δ`（run-ahead 上界，§11.1），故存活槽 producer id 落在
  `(R, 最新 append]`，跨度 = `最新 − R ≤ Δ + H`。
- **落进单桶的比例来自哈希。** 全局活跃版本总数 ≤ `窗口 × 每任务平均输出条目数`（private 窗口=`H`、
  shared 窗口=`Δ+H`）；按 `B` 个桶哈希，**单桶期望存活槽 ≈ 全局活跃版本 / `B`**。哈希非理想均匀，需
  留倾斜裕度（`auto` 如何在构建期精确取代"倾斜裕度估计"见 §12.7.2.3）。
- **取值公式（建议下界）**：

```text
# W = H (private) 或 Δ+H (shared)
CAP = ceil_pow2( W * avg_outputs_per_task / B * skew_factor )
```

  其中 `skew_factor`（经验 2~4）覆盖哈希不均与 INOUT 多版本堆积；`ceil_pow2` 向上取到 2 的幂，使
  `k % CAP` 退化为位与、`seq = k + lap*CAP` 的 lap 递增用移位。`Δ`、`H`、`B` 均已是 §11 的现有常量，
  `CAP` 与 `W`（完成标志环窗口，§11.3）**同源**，可一并标定。
- **估偏的代价（非对称）**：估**大**只浪费 GM（每桶多几个槽 × `B` 桶，线性且可控）；估**小**在 shared 下
  会在热桶频繁触发反压 stall，甚至（若窗口真的不足）**死锁**——因为 winner 等 `head` 前进、而 `head`
  前进又依赖更慢核推进 `R`。**private 下估小更严重**：private 回收已用最紧的确定性 `N−H`、无更慢核可
  等，故一旦某桶在 `H` 窗口内溢出即为**真正的配置错误**，无法靠等待自解，必须直接报错（§12.7.2.2）。
  故**宁可略微估大**——好在 `auto` 能在构建期把两模式的窗口占用**精确算出**、不必"估"（§12.7.2.3）。

#### 12.7.2.2 溢出检测与报错设计（绝不静默覆写）

ring 满的语义必须是**显式失败或可恢复反压**，不能像无界链表那样"总能再挂一个"。**shared** 分两类
处置（A/B）；**private** 无跨核等待余地，溢出直接走 B 的确定性报错（见 B 末）。

**A. 可恢复反压（shared 默认，热路径）。** winner 在 `k = fetch_add(reserve, 1)` 后、写槽**之前**先检查
`k − load_acquire(head) >= CAP`。若成立即表示环满：

1. **不写槽、不发布 seq**，并把 `reserve` 回退（`fetch_sub(reserve, 1)`，或采用"先探测后提交"两段式
   抢槽以避免回退竞争）；
2. 进入**有界自旋 + backoff**，周期性推进本核可推进的 `F`/`R`（§11.4 协作式回收），给 `head` 前进创造
   条件；
3. `head` 前进后重试 `insert`。这与 §11 的**堆反压语义一致**（满则等待，不丢数据）。

**B. 不可恢复 → 结构化报错（诊断路径）。** 若反压自旋超过阈值 `T_stall`（如按最坏依赖链估算的上界
的数倍）仍无法推进，判定为**容量配置错误或依赖跨度估计错误**，触发一次**确定性、可定位**的运行时
错误，而非挂死或 UB：

```text
FATAL[tensormap-ring-overflow]
  bucket   = <bucket_index>            # 哪个桶溢出
  cap      = <CAP>                     # 当前容量
  live     = reserve - head            # 溢出时的存活槽数
  head/R   = <head> / <R>              # 回收游标与全局 reclaim 前沿
  slowest  = core <id> @ local_index   # 拖住 R 的最慢核（定位反压根因）
  hint     = "raise --tensormap-ring-cap or check H/Δ estimate; \
              possible deadlock if a producer never completes"
```

  报错要点：**(i)** 指明**是哪个桶**、当前 `CAP`、溢出时 `live` 值，便于直接调参；**(ii)** 打印**最慢核**
  及其 `local_index`，区分"真溢出（窗口不足）"与"某 producer 卡死导致 `R` 不前进"（后者是别处 bug，
  ring 只是最先撞墙的地方）；**(iii)** 给出可操作建议（调大 `--tensormap-ring-cap` 或复核 `H`/`Δ`）。
  该错误应是**确定性**的（同输入必在同一 bucket 触发），便于复现与回归。

  **private 的溢出更简单直接**：private 无 `reserve`/`R`，也无跨核可等；`tail − head` 触到 `CAP` 即
  刻判定为配置错误，**立即**抛同款 FATAL（`slowest`/`R` 字段留空，`head` 用本核 `N−H`）。但在 `auto`
  下 private **可证不溢出**（§12.7.2.3），此路仅在用户手动把 `--tensormap-ring-cap` 设得过小时触发。

**C. 调试增益（可选）。** debug build 下额外维护每桶 `high_watermark = max(reserve − head)`，运行
结束打印各桶水位分布，用于**离线标定 `CAP`**：水位远低于 `CAP` 说明可调小省内存，逼近 `CAP` 说明
需调大或该桶是热点（考虑 §12.7.1 的 sub-ring 分片）。

#### 12.7.2.3 `auto` 模式是否安全、如何实现（本章重点）

统一到 ring 后，定容的安全性是新引入的**唯一**风险（链表无"满"概念）。核心问题：**默认的 `auto` 定容
安全吗？** 结论：

> **`auto` 在 private 下可证 100% 安全（永不溢出）；在 shared 下在"按最坏单桶占用定容"时同样可证不
> 溢出，退一步即便估紧也绝不静默损坏（溢出→反压/确定性 FATAL）。** 关键在于：**决定 `CAP` 的两个量
> ——(1) 每桶落入的静态区域集合、(2) 回收窗口宽度——都在构建期已知或有硬上界**，因此 `auto` 不是
> "猜"，而是**在构建期精确计算**。

**为何安全：两个决定量都是已知/有界的。**

1. **区域集合是静态的。** SPMD 下每个核 replay **同一条确定性 submit 流**（§2/§6.4），全部任务的输出
   region 及其 `bucket_of(region)` 哈希**在构建期即完全确定**。因此"哪些 producer 落进哪个桶"不是运行
   期随机量，而是可枚举的静态事实——哈希倾斜**不需要"估"**，可直接数出每桶的真实占用。
2. **回收窗口有硬上界。** private 窗口 = `H`（确定性 `N−H`）；shared 窗口 = `Δ+H`，其中 `Δ` 是私有环
   run-ahead 的**配置上界**（§11.1）、`H` 是依赖跨度**契约上界**（§11.4）。二者都不是无界运行期量。

两点合起来：**每桶存活槽数的最大值 = "在任意长度为 `W` 的 producer-id 滑动窗口内、哈希到该桶的输出
region 条数"的最大值**（`W=H` 或 `Δ+H`）。这是一个**可在构建期精确算出**的确定值，不含任何运行期
不确定性（private 完全确定；shared 唯一的运行期量是进度差，而它被 `Δ` 硬封顶）。

**`auto` 的实现（构建期精确定容，非启发式）。**

```text
compute_auto_cap(task_graph, mode):
    W = H                      if mode == Private        # 确定性窗口
        (Δ + H)                if mode == Shared         # Δ 为 run-ahead 硬上界
    per_bucket_max = array[B] of 0
    live = sliding_multiset()                            # 以 producer-id 为键的滑动窗口
    for N in 0 .. num_tasks-1:                           # 按确定性 submit 顺序扫全图
        for r in outputs(task[N]):
            b = bucket_of(r)
            live.add(b, producer_id=N)
        live.evict(producer_id <= N - W)                 # 精确模拟 §12.7.1 的 head 回收
        for b in 0 .. B-1:
            per_bucket_max[b] = max(per_bucket_max[b], live.count(b))
    cap = ceil_pow2( max_over_b(per_bucket_max[b]) * safety )   # safety ∈ {1(可证), 小裕度}
    return cap
```

- 该过程**只依赖静态任务图**（host build-once 或各核构建期均可跑），复杂度 O(任务数 × 每任务输出数)，
  一次性、与执行无关。它**精确复刻** §12.7.1 的 `head` 回收语义（private 按 `N−W`、shared 按窗口
  `Δ+H`），因此算出的 `per_bucket_max` 就是运行期真实峰值的**上确界**。
- **private**：`safety = 1` 即已**可证不溢出**（窗口确定、无进度差、无并发追加）——`auto` 给出的 `CAP`
  就是精确峰值。这就是"private 下 `auto` 100% 安全"的证明。
- **shared**：以 `Δ` 硬上界代入窗口后，`per_bucket_max` 是**最坏进度差下**的峰值上界；取 `safety = 1`
  即得**可证不溢出**的 `CAP`（代价是按最坏 `Δ` 偏保守、略费内存）。若要更省内存，可取更小的有效
  `Δ_eff < Δ`（按实测/期望进度差）作为"紧档"，此时不再可证、但 §12.7.2.2 的反压 + FATAL 兜底保证
  **绝不静默损坏**。默认 `auto` 采用**可证档（`safety=1`、`Δ` 满值）**，安全优先。
- **全局单值 vs 每桶**：`auto` 天然算出**每桶**峰值，可直接支持"每桶独立 `CAP`"（最省内存）；一期为
  实现简洁可取 `CAP = max_over_b`（全局单值），后续再切每桶。

**是否需要命令行参数——需要，但默认 `auto`。**

- **为何仍保留 `--tensormap-ring-cap`**：`auto` 依赖 `Δ`/`H` 的取值正确；若用户想**收紧内存**（接受 shared
  下的紧档风险）或**排障时放大**容量，需一个**免重编译**的覆盖出口。与 §11 把 `W`/`Δ` 做成可配置一脉相承。
- **为何默认 `auto` 而非必填**：`auto` 既开箱即用又（private 可证 / shared 可证档）安全，强制用户填值
  徒增负担且易填错。
- **两模式均生效**（统一 ring 后 private 也是 ring）：取值**向上取 2 的幂**；设定值 **< `auto` 算出的
  可证峰值**时，private **启动期直接拒绝并报错**（因其必然溢出、无法自解），shared 则**告警并允许**
  （用户显式选择紧档，运行期由 §12.7.2.2 兜底）。

> 一句话：`auto` **不是估，是构建期按静态图 + 硬上界窗口精确算**——**private 可证永不溢出**，**shared
> 取可证档同样不溢出**、紧档也绝不静默损坏；命令行 `--tensormap-ring-cap`（默认 `auto`，两模式生效）
> 仅作收紧内存/排障的覆盖出口。

### 12.7.3 run-ahead 上界 `Δ_max`：负载均衡旋钮（两种模式）+ 平台默认值

**动机。** 认领用**单调 `fetch_max` 全局游标**（§11.1）——谁先 replay 到任务 `N` 谁就 win。若各核推进速率
不均（真机上偶发的慢核；仿真里 host 过订阅导致的调度倾斜），跑得快的核会把游标一路推到前沿，**抢走一长段
连续 id**，把落后的核饿死（落后核 replay 到那些 id 时游标已越过 → 只能 skip、领不到活）。§6 的 execute-first
只是**软**减速；要**硬**封顶就需要一个 run-ahead 上界。

**机制（`dist_runahead_throttle`，两模式通用）。** 每核在推进到任务 `N` 前，先把自己的 replay 走位发布到
`gd->core_progress[core]`，然后**等待**直到 `N − min(core_progress) ≤ Δ_max`（`min` 取全体核的最慢走位）。
等待是**协作式**的：每圈调用 `drain_block_won()` + `drain_phase_b()` 清偿本核欠下的完成事件，仅在无就绪任务时
`SPIN_WAIT_HINT`（仿真上即 `sched_yield`）。**无死锁**：最慢核的 `N − min == 0` 永不被节流 → 持续推进 → 抬升
`min` → 释放所有超前核。**它只改变"谁执行"，绝不改变确定性 replay / 依赖图**——`PTO_DIST_DEPSIG` 实测跨 `Δ_max`
取值、跨 private/shared **逐位一致**（a5 `Balanced9` `sig=33b150b1…` edges=270；a2a3 `Case0` `sig=8a877fd1…`
edges=750）。与 §12.7.2 shared 环的追加反压同源（后者另在 `tm_shared_claim_append` 里按 ring `cap` 限追加前沿）。

**平台默认值（按核数派生 ⇒ 每平台自动合理）。** `gd->runahead_max = 2 × num_workers`：
`num_workers` 在 a2/a3 为 24 AIC + 48 AIV = 72（默认 144），在 a5 为 36 AIC + 72 AIV = 108（默认 216）。
**下界约束 `Δ_max ≥ num_workers`**：否则窗口装不下"每核一个在飞任务"，健康并行会被节流成**空闲核**（实测
`Δ_max=8 ≪ 108` 时 shared 出现 10–24 个空闲核、makespan 翻倍）；`2×` 给流水线留裕度，又能把失控核封在
"领先 2×核数"以内。`PTO_DIST_RUNAHEAD=N` 覆盖，`0` 关闭节流。

**均衡验证（a5sim，`PTO_DIST_FAKE_EXEC_NS` 给每个 kernel 等长耗时）。**
- **非过订阅（`Balanced9`，9 worker ≈ 8 物理核）＋默认 `Δ_max=18`**：GEMM CV **2.4%**、`max/mean` **1.02×**、
  **0 空闲核**；ADD CV 15%、`max/mean` 1.20×——引擎+旋钮+默认值在忠实并行下**均衡极佳**。
- **满核过订阅（`FullCore36`，108 线程 / 8 物理核）**：任何 `Δ_max` 都无法均衡。根因是**过订阅 + 单调游标**：
  被 OS 调度到的线程抢光窗口内全部认领并推高游标，`sched_yield` 不能可靠把 CPU 让给被饿死的 107 个线程；落后
  线程追上时只能 skip。这是**仿真伪影**（host 只有 8 个真并行核），**非引擎缺陷**——真机上 108 个核真并行、
  速率近似，`Δ_max` 只在偶发慢核时兜底。故满核仿真泳道的不均衡应按过订阅解读，均衡结论以 `Balanced9` 这类
  ≈1:1 配置为准。

### 12.8 两种模式的性能分析（均为 ring-per-bucket）

> 两种模式**数据结构相同（ring-per-bucket）**，下表比较的是"每核私有 ring"与"全局共享 ring"这两种
> 用法在内存/insert/同步上的取舍（§12.3.2 的四点差异所致），**不是**链表 vs ring（后者见 §12.3.1）。

| 维度 | private（每核私有 ring） | shared（全局共享 ring） |
| ---- | ---- | ---- |
| **map 内存占用** | `O(核数 × 单份 ring)` | `O(单份 ring)` —— **省 `核数` 倍** |
| **每任务 insert 工作量** | **每核都 insert 全部任务**（SPMD 冗余，§6.4 的"地板"） | **仅 winner insert**（每任务恰好一核） |
| **每任务 lookup 工作量** | 本地 ring 连续扫描，**零跨核、无 `seq` 校验**；winner-only | 共享 ring，需 invalidate/acquire per-slot `seq` + 字段；同为连续数组扫描 |
| **跨核同步/原子** | **无**（`head`/`tail` 为普通整数） | per-bucket `reserve`/`head` 原子 + per-slot `seq` acq-rel（可分片） |
| **缓存一致性流量** | 无 | 随核数增长（更多 reader invalidate slot；热桶 `reserve` 竞争） |
| **ABA / 回收风险** | 无（单线程，回收 = `head++`） | ring 复用 ABA 由 per-slot `seq` 化解；回收 = `head++`，无 use-after-recycle |
| **`CAP` 定容窗口** | `H`（更窄），`auto` **可证不溢出**（§12.7.2.3） | `Δ+H`（更宽），`auto` 可证档不溢出 |
| **确定性 / golden 不变** | 各核 ring 内容严格一致 | map 内容由 winner 发布顺序决定，**弱确定**（最终一致） |
| **随核数 scale 的编排地板** | §6.2/§6.3 实测：随核数近线性上升（SPMD 重放 + 每 insert 地板） | 去掉每 insert 地板 → **有望显著 flattening** §6.2 曲线 |

**定性结论。**

- **shared 的主胜场**：在**多核**（大 `block_dim`）场景下，把"每核为全部任务 insert"的 SPMD 地板
  砍成"每任务仅一核 insert"，并把 map 内存从 `核数×` 降到 `1×`。这恰好对症 §6.2/§6.3 暴露的
  "编排墙钟随核数近线性增长"与 §6.5 的 cursor CAS 竞争之外的另一条地板——理论上 shared 模式可把
  §6.2 的 1→13 约 2× 的增长曲线明显压平（insert 工作总量从 `核数 × 任务数` 降到 `任务数`）。
- **shared 的主代价**：lookup 引入跨核 cache line 读 + invalidate 流量，且热桶 `reserve` 成为新的
  原子热点（类似 §6.5 的 cursor）。在**少核**或桶内存活槽多（需扫多条远程 line）时，lookup 延迟可能
  吃掉 insert 省下的红利。
- **private 的主胜场**：**少核**或**内存充裕**场景下，零跨核协调、零 ABA、严格确定、lookup 纯本地
  连续扫描（无 `seq` 校验、无 invalidate）。与 §6.4 的 O(N) 回收窗口、winner-only fan-in、§6.6 cursor
  分片等优化完全兼容（这些语义已在 ring 上继承）。
- **建议**：`private` 作为安全默认；`shared` 作为**大核数 / 大任务图**下的可选加速档，需在目标平台
  实测 §12.7 的 invalidate 流量与热桶 `reserve` 竞争是否可接受。二者正交于 cursor 分片（§6.6）、
  winner-only fan-in（§6.4）等其它优化，可叠加。

> 一句话：**两模式同用 ring-per-bucket**（§12.7.1），差异仅在"每核私有 / 全局共享"（§12.3.2）；
> **少核求快用 private**（零跨核、纯整数游标、`auto` 可证不溢出），**多核求省用 shared**（内存 `1×`、
> insert 仅 winner，正确性靠"per-slot `seq` acq-rel + 发布即不可变 + `R` 驱动 `head++` + lookup 不缓存
> 游标"，数据正确性仍由 §9.3 独立地址 + §11.5 完成标志兜底）。

#### 12.8.1 实测 overhead（TensorMap 操作计数，确定性、跨平台）

墙钟对比在仿真主机上不可用（`device_wall_us` 对单次冷跑报 0；且线程数 > 物理核时所有核忙等自旋，
makespan 被 OS 时间片放大到 ~33 ms/task，两模式**同等**放大，把编排逻辑差异完全淹没——实测 6 核
private 67.41 s vs shared 67.43 s，差 <0.03%，无法区分）。因此改用**确定性的 TensorMap 操作计数**
（env `PTO_DIST_OVERHEAD=1`，引擎在 DONE 打印 `[dist] TMOPS ...`；不依赖时钟、不受 SIGBUS/超额订阅
影响）作为 overhead 度量。BGEMM `Case0`（72 核 / 500 matmul-add / D=1000 个不同输出 region）实测：

| 指标 | private | shared | 说明 |
| ---- | ---- | ---- | ---- |
| **inserts**（map 写入次数） | **72000** | **1000** | private = `核数 × D`（每核为全部任务 insert，SPMD 地板）；shared = `D`（每 region 仅首达核 append 一次）→ **shared 少 `核数`(72)× 写入**，且内存 `1×` vs `72×` |
| **lookups**（fan-in 解析次数） | 3000 | 3000 | 两模式相同（均 winner-only 解析同一批边，§6.4） |
| **scans**（lookup 扫描的槽数） | 9332 | 43689 | shared **多 ~4.7×**：全局环在 `Δ+H` 窗口内堆积**所有核**的 append（比 private 每副本的 `H` 窗口更深），且**每扫一槽多付一次 `seq` 原子 acquire + 跨核读**——这是 shared 的并发税 |

**交叉验证（`runtime_overhead_test`，3 核 / tasks=100 / skip-exec）** —— 换一条完全不同的编排流、
核数从 72 降到 3，`insert` 缩减比仍**精确等于核数**，佐证该规律是结构性的、与具体模型无关：

| 指标 | private | shared | 比值 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| **inserts** | 600 | **200** | **3× = 核数** | 与 72 核例同构：private=`核数×D`，shared=`D` |
| **lookups** | 600 | 600 | 1× | 完全一致 |
| **scans** | 1581 | 1906 | ~1.2× | 该流 `Δ+H` 窗口浅，并发税小于 72 核例的 4.7× |

**结论（量化 §12.8 的定性判断）：**
- **shared 的胜场 = insert**：写入次数从 `核数×D` 砍到 `D`（此例 **72→1**，即 72× 减少），且随核数**线性**扩大；map 内存同步从 `72×` 降到 `1×`。这正是压平 §6.2 "编排墙钟随核数近线性增长"的那条地板。
- **shared 的代价 = lookup 扫得更深**：扫描槽数 ~4.7×。根因是**核间进度 skew**——private 每核一份副本、各自只装本核 `[N−H, N)` 的 `H` 深窗口；shared 是单一全局环，必须同时覆盖**最慢核的尾**到**最快核的头**，即 `Δ+H` 深（`Δ` = 快慢核回放领先量，§12.7.1）。多出的深度就是被 coalesce 进同一个环的核间 skew。
- **但这个扩大是有界的，不会无限增长**：`Δ` 被 run-ahead 节流（§12.7.2）按 `Δ_max` 封顶，故 shared 窗口恒 ≤ `Δ_max + H`，与核数**无关**；而 insert 红利随核数**线性**增长。两者一减一增 ⇒ **核数越多，shared 越划算**：insert 节省无上限地随核数放大，scan 深度却被 `Δ_max` 钉死。极端地令 `Δ_max→0`（lockstep）时 shared 窗口退化为 `H`、scan 与 private 完全一致，代价是牺牲 run-ahead 并行度——所以深度是"用并行度换来的可调量"，而非失控项。
- **单次 scan 的原子代价已摊薄至接近 private**：曾经每扫一槽一次 `seq` acquire，现已优化为**每次 lookup 仅一次 acquire**（快照 tail 后对其下所有槽用 relaxed 读，正确性依据见 §12.7.1）。残留的**跨核 cache line coherence 读**不可消除——那是"单份共享"换取"省 N× 写入 + N× 内存"的固有对价（private 靠复制 N 份让读永远 L1 命中）。
- **权衡**：故 **大核数 / insert 密集 → shared 明确胜出（如 72 核）**；**少核 / lookup 密集 / 内存充裕 → private**。此结论与墙钟无关，纯由确定性操作计数得出。

**为什么不用墙钟 us/task（macOS 仿真的深入排查记录）。** 曾尝试用 `[dist] OVERHEAD` 的
`makespan_us / tasks` 得到 us/task，结果稳定在 ~33–67 ms/task（比历史 0.5–5 µs/task 大 4 个数量级），
排查结论如下，供后来者免于重复踩坑：

1. **`SPIN_WAIT_HINT` 本已让核**：仿真构建通过 `common/platform/sim/aicpu/spin_hint.h` 展开为
   `yield + sched_yield()`（并非 no-op）。追加 macOS 专用 `nanosleep` 强制让核后重测，us/task **完全不变**
   → 放大**不是**忙等自旋造成的超额订阅。
2. **replay 阶段本身就 ~13 s**：新增 `[dist] OVERHEAD ... replay_us[min/avg/max]`（仅 `orch_func`
   回放耗时，排除 drain 循环）后发现 `replay_us ≈ busy_us ≈ makespan`，且**最快核**的 replay 也有 12.8 s。
   即跨核等待发生在**回放内部**的堆回收 / 完成前沿（frontier）背压——每次跨核交接被 OS 调度按
   ~ms 量级计时，任务再多也是**线性**累加，故与任务数成正比（tasks=100→13.5 s，tasks=500→67 s，
   恒为 ~67 ms/task），并非固定超时。TMOPS 全程仅 ~1.6 k 次 scan，证实这 13 s 与 TensorMap 计算无关。
3. **口径不同**：历史 0.5–5 µs/task 来自**设备周期剖析**（`PTO2_ORCH_PROFILING` 的 `avg/task`，计编排
   代码消耗的设备周期），与主机墙钟不是同一度量；在 3 核 macOS 仿真上无法用墙钟复现到 µs。

因此 **overhead 对比以上表 TMOPS 为准**（确定、跨平台、可复现）。若确需可比的 µs：(a) 在 Linux 多核机上
重跑墙钟（`sched_yield` 在 Linux 真让核、跨核交接是微秒级），或 (b) 用设备周期剖析构建并把周期计数埋进
`dist_engine.cpp` 编排热路径（排除自旋段）。`replay_us` 字段保留在 `PTO_DIST_OVERHEAD=1` 输出中，供在
低交接延迟平台上直接得到 replay/task。

### 12.9 顶层入口：TensorMap 指针的双模化设计

为支持两种模式共存于同一份代码、由命令行开关选定，runtime 的**顶层 TensorMap 入口**需被改造为
一个**间接句柄**，而非硬编码的"每核 array"。

**现状（private 硬编码）。** 编排循环里直接持有"每核私有 map"（原为链表 `DistTensorMap` 实例，
§6.4；本章统一后为每核私有 ring），`tm_insert` / `tm_lookup` 直接作用于它。这把"每核私有"写死在入口处。

**改造。** 引入一个抽象句柄 `TensorMapHandle`，封装**同一 ring 数据结构的两种用法**并提供统一 API：

```cpp
enum class TensorMapMode { Private, Shared };

// 两模式共用同一 ring 结构（§12.7.1）；仅并发纪律 / 副本数不同（§12.3.2）
struct RingTensorMap;   // ring-per-bucket；Shared 用 seq/reserve/acq-rel，Private 关闭并发纪律

struct TensorMapHandle {
    TensorMapMode mode;
    RingTensorMap* map;   // private：指向本核私有 ring（每核各一份，单线程纪律）
                          // shared ：指向全局唯一 ring（并发纪律 + per-slot seq，§12.7.1）
};

// 统一入口，由 mode 分发（Private 走无原子快路径，Shared 走并发路径）：
void  tm_insert (TensorMapHandle& h, const TensorRegion& r, task_id_t producer);
Entry* tm_lookup (TensorMapHandle& h, const TensorRegion& r, task_id_t local_index);
```

**初始化（按开关二选一，同一结构不同参数）。**

```text
init(runtime_args):
    mode = parse_tensormap_mode(args.tensormap_mode)          # private | shared
    cap  = resolve_ring_cap(args.tensormap_ring_cap, mode)    # auto → compute_auto_cap(graph, mode) (§12.7.2.3)
    if mode == Private:
        # 每核构造一份私有 ring（单线程纪律：head/tail 普通整数，无 seq/reserve）
        for each core c:  c.tensormap = { mode:Private, map: new RingTensorMap(cap, concurrent=false) }
    else:  # Shared
        # 全局构造唯一一份共享 ring（并发纪律：seq + reserve + acq-rel）
        shared = new RingTensorMap(cap, concurrent=true, pool=global_pool)
        for each core c:  c.tensormap = { mode:Shared, map: shared }
```

**调用点改动（§6 伪代码中的 `update_tensormap(task)`）。** 唯一变化是 insert 改为**仅 winner 调用**
（shared 模式下），lookup 仍由 winner 在解析 fan-in 时调用。这通过在 `update_tensormap` 内部按
`mode` 分发实现，**上层 §6 循环伪代码不动**：

```text
update_tensormap(task, won, N, mode):
    if mode == Private:
        # 无条件 insert（胜者、败者、follower 都做），保持各核 ring 副本一致
        for t in task.inputs:  tm_lookup (h, t, N)        # fan-in（winner-only 仍由调用方门控）
        for t in task.outputs: tm_insert (h, t, N)        # 私有 ring：写槽 + tail++（无原子/seq）
    else:  # Shared
        # 仅 winner 做 insert；lookup 仍为 winner 专属
        if won:
            for t in task.inputs:  tm_lookup (h, t, N)    # 内含 §12.6 时序过滤 + §12.7.1 per-slot seq 校验
            for t in task.outputs: tm_insert(h, t, N)     # 内含 §12.5 ring 追加 + §12.7.1 fetch_add 抢槽 + seq 发布
```

> 注意 `tm_insert`/`tm_lookup` 在 shared 用法里隐含 §12.7.1 的 writeback/invalidate、per-slot `seq`
> acq-rel 与 `reserve`/`head` 游标纪律，对上层透明；private 用法走**同一 ring 的无原子快路径**（普通
> 整数 `head`/`tail`、纯本地连续扫描）。两条路径共享 ring 的桶/槽/哈希/多版本逻辑，仅并发纪律有别，
> 不互相污染热路径。

**回收侧的双模（同一 ring、不同阈值）。** private 每核按确定性阈值 `N − H` 推进自己的 `head`（继承
§6.4 的回收窗口语义，改为游标自增）；shared 用全局 reclaim 前沿 `R`（§9.5）驱动——仅当
`slots[head % CAP].producer_id ≤ R` 时 `head++`。两者回收都是**游标自增、不归还任何空闲池**（§12.7.1）；
`R` 由 §11.4 协作式 `F` 推进后派生，可由任一核在推进 `F` 时顺带推进各 bucket 的 `head`（按 bucket
独立、低频，竞争小）。

如此，**命令行开关只决定 `TensorMapHandle` 的初始化参数（副本数 / 是否开并发纪律 / `CAP`）与
`update_tensormap` 的分发分支**，§6 主循环、§9 堆管理、§11 完成标志环均不改。两模式共用一套
ring-per-bucket 实现（§12.7.1），§6.4 的回收窗口/哈希/winner-only 等语义原样继承。这把"双模"的改动面
收敛到"同一 ring 的两种纪律配置" + insert/lookup 分发，满足"基于命令行设置支持两种方案"的设计要求。

### 12.10 落地实现与验证（a2a3 `dist_engine.cpp`）

§12.4–§12.9 给出的是 shared 的**通用设计**（MPSC `reserve` 抢槽 + winner-only insert + 最终一致）。
实际落地在 `src/a2a3/runtime/fully_distributed_within_core/runtime/dist_engine.cpp` 时，为**保证 shared
与 private 逐位一致的计算结果**（验收目标），做了两处**更强**的选择，并新增一个反压旋钮。三者都不改变
§12.1–§12.3 的统一 ring 结论，只是把"弱一致 + MPSC"收紧为"强一致 + 串行追加"。

**(1) 顺序化追加定序器（取代 MPSC winner-only insert）。** shared 用一个全局原子 `tm_insert_next`
把**所有追加严格串行化到 task-id 顺序**：任务 `N` 只由"第一个走位到 `N` 的核"追加一次，且必须在
`0..N−1` 全部追加之后（`CAS(insert_next, N, BUSY)` 抢占 → 写槽 → `store(N+1)`）。因每个核都按 id 顺序
replay、且离开任务 `K` 前必已确保 `K` 已被追加（定序器阻塞它），故**任何核解析任务 `N` 的 fan-in 时，
全局 ring 的内容恰好等于 private 副本此刻应有的内容**。这把 §12.4 的"最终一致（winner 可能尚未发布 →
lookup 命中次新/NONE）"收紧为**强一致**，代价是追加环节全局串行（但追加本身极廉价，且 execute 仍
乱序重叠）。因**只有单一追加者**，§12.7.1 的 MPSC `reserve` 不再需要：`tail` 由当前唯一追加者写，
`head`/`tail` 为原子、per-slot `seq` 仍保留以防读者在 host 线程被抢占时撞上槽复用。

**(2) lookup 双过滤 `producer ∈ [N−H, N)`。** 在 §12.6 时序过滤（`< N`，跳过未来 producer）之外，**再加
`≥ N−H` 的下界**——精确对齐 private 的 `alive_floor = N−H`。这样即便共享 ring 里同时存在快核追加的
"未来条目"（`≥N`）与尚未回收的"陈旧条目"（`<N−H`），lookup 接受的集合也恰好是 `[N−H, N)`，与 private
副本**逐位相同**。回收阈值用 `R = min_progress − H − 1`（各核 replay 进度的最小值），保证任何存活核的
`[N−H, N)` 窗口都不被驱逐。

**(3) run-ahead 反压（把 §12.7.2 的 `Δ+H` 溢出从 FATAL 转成背压）。** shared 存活窗口 = `Δ+H`
（§12.7.2.1），`Δ` 为最快/最慢核的 replay 进度差。在 skip-exec 或超额订阅的仿真主机上 `Δ` 可能暴涨到
超过 `cap`，触发 §12.7.2.2 的确定性 FATAL。为此新增 `g_tm_runahead_max`：追加前沿在**未持有定序器**
的前提下等待（并 drain），直到 `N − min_progress ≤ Δ_max`。落后核此时**照常前进**（它们对 `<N` 的
`claim_append` 立即返回，因 `insert_next > 其 id`），`min_progress` 上升即释放前沿——**无死锁，只节流
前沿**。默认 `Δ_max = 3·cap/4 − H − 1`（令窗口 ≤ ~¾ cap，留哈希倾斜裕度）；命令行 `--runahead N`（→
`PTO_DIST_RUNAHEAD=N`）覆盖，`0` 关闭（溢出回退为 FATAL）。这与 §11.4 的堆反压同源（满则等待、不丢数据）。
> 注：`PTO_DIST_RUNAHEAD` 现同时驱动**两模式通用的均衡节流** `gd->runahead_max`（§12.7.3，submit/alloc
> 入口按 `N − min_progress ≤ Δ_max` 等待，默认 `2×num_workers`）；shared 下它一并覆盖此处的追加前沿上界。
> 二者机制同构（都读 `core_progress[]` 的 `min`），前者管**认领走位**均衡、后者兼管**共享环窗口**不溢出。

**节流时 worker 不空转，而是协作式 drain（不是 park 线程）。** 被节流的前沿核**不阻塞、不睡眠**，
而是在等待窗口的每一圈都调用 `drain_block_won()`（把发到本 lane 的 block.won 存款拉进空闲 slot）+
`drain_phase_b()`（扫描本核私有 ring，把 **fan-in 已就绪** 的 slot 立即 `execute` 并发布完成标志、释放
slot）。**仅当**本核确实没有任何就绪任务可执行（`drain_phase_b` 返回 0）才 `SPIN_WAIT_HINT` 轻自旋 +
看门狗。这正是"worker 去检查窗口里的任务是否满足执行条件并 drain"的行为——而且它 drain 出的完成标志
会**解锁其它核**依赖这些数据的任务，让最慢核得以推进 replay、抬升 `min_progress`，从而释放本核的节流。
因此这是一条**协作式、无死锁**的等待：前沿核用等待时间替系统清偿它自己欠下的完成事件。

**为何 per-core task slot 不能替代该节流（回答"worker 的 task slot 本身是否已限制窗口"）。** 每核有
一个 **`kPrivateSlots = 4` 的私有执行 ring**（`kWonReserve = 2` 预留给 follower，故自领任务实际在
`occupied_count ≥ 2` 时即反压）。**但它约束的是"本核已认领、尚未执行"的任务数（执行窗口），不是
replay 走位的 run-ahead `Δ`。** 关键区别：一个核 **replay 全部任务**（shared 下还参与全部任务的顺序
追加），却**只对自己 win 的任务占用 slot**；对没抢到的任务，它只做 lookup/append 后**径直走过、不占
slot、不反压**。因此在 skip-exec（执行 0 成本）下，自领 slot 瞬间 drain 空、执行反压从不触发，走位游标
可一路冲到终点 → `Δ` 爆炸 → shared 的 `Δ+H` 窗口溢出（这正是先前观察到的 FATAL）。**在真实执行下**，
自领 slot 的执行反压只提供**软性、间接**的减速（核必须等自己认领的任务算完才能继续），能压低但**不能
封顶** `Δ`，且强度取决于认领比例与依赖结构。故 task slot 无法安全地界定 tensormap 窗口——必须有独立的
run-ahead 硬上界 `Δ_max`。二者约束**正交**：task-slot ring 限"owned-in-flight"（§3.1 完成侧），
run-ahead 限"replay 前沿领先量"（§12.7.1 tensormap 侧）。

**(4) lookup 的 acquire 摊薄：每次 lookup 一次 acquire（而非每槽一次）。** §12.7.1 的通用 MPSC 伪码里
每扫一槽都要 `acquire_load(s.seq)`（因 `reserve` 只是抢槽游标、非发布水位，无法保证其下每槽已发布）。
但落地实现是**单一串行追加者**（上文 (1)）：`tail` 是**真正的发布水位**——追加者先 `store(seq=k, release)`
再 `store(tail=k+1, release)`，且跨核的 `append(k)→append(k+1)` 由 `tm_insert_next` 的 release/acquire 链
串起、`tail` 单调。故 reader **只需对 `tail` 做一次 acquire-load**，即与"所有 `< tail` 的追加"建立
happens-before，其下每一槽的字段与 `seq` 都已可见；扫描各槽时 `seq` 改用 **relaxed 读**，仅作 ABA 护栏
（防扫描期间 `head` 并发回收把物理槽复用出去）。这把 §12.8.1 里"每扫一槽一次原子 acquire"降到"每次
lookup 一次 acquire"，使 shared 的单槽扫描逼近 private 的纯本地读；残留的**跨核 cache line coherence 读**
不可消除（单份共享的固有对价，private 靠复制 N 份规避）。**验证**：改动后 6 核（`sig=358074ac…`）与 24 核
（`sig=c01c01e9…`）的 `PTO_DIST_DEPSIG` 依旧与 private **逐位一致**，private 侧签名不变（`8a877fd1…`）。

**命令行 / 环境变量。**

| 变量 | 作用 |
| ---- | ---- |
| `PTO_DIST_TENSORMAP_MODE={private\|shared}` | 选择模式（默认 `private`）；运行期一次性读取，不中途切换 |
| `PTO_DIST_TENSORMAP_RING_CAP={N\|auto}` | 每桶 ring 深度（2 的幂，`auto` 由 `H` 派生，两模式生效） |
| `PTO_DIST_RUNAHEAD=N` | run-ahead 上界 `Δ_max`（**两模式通用**的负载均衡旋钮，§12.7.3）：任一核的 replay 走位最多领先最慢核 `N` 个任务。`0` 关闭节流；shared 下还同时覆盖前沿追加反压上界（`0` ⇒ ring 溢出回退确定性 FATAL）。默认按平台核数派生（见 §12.7.3） |
| `PTO_DIST_DEPSIG=1` | 打印依赖图签名（见下），供 private/shared 一致性验证 |
| `PTO_DIST_OVERHEAD=1` | 打印 `[dist] TMOPS`（inserts/lookups/scans 计数，§12.8.1）+ `[dist] OVERHEAD`（makespan/busy/replay 墙钟；macOS 仿真上被跨核调度延迟主导、仅供参考，理由见 §12.8.1） |

**正确性验证（依赖图签名，免疫浮点噪声）。** 因 BGEMM/PagedAttention 的 `C += A@B` 等**浮点累加顺序
随调度变化**，数值 `max_diff` **逐次运行本就波动**（private 自身即 0.0091↔0.0093），无法作逐位判据。故
引入 `PTO_DIST_DEPSIG`：对每条已解析的 fan-in 边 `(consumer, producer)` 做 **XOR 累加**——与调度顺序
无关，只取决于边的**集合**，是免疫浮点噪声的正确性判据。实测（build/lib，a2a3sim）：

| 用例 | 规模 | private 签名 | shared 签名 |
| ---- | ---- | ---- | ---- |
| 差分 UT `test_dist_tensormap_ring.cpp` | 268 万次查询（含"快核超前/滞后回收"越窗条件） | 参考=private | **== private** |
| BGEMM `Case0` | 72 核 / 500 任务 | `8a877fd1e0e02bb1` (750 边) | **相同** |
| PagedAttention `CaseSmall1` | 27 核 | `a7db56ff3de5afa6` (15 边) | **相同** |
| runtime_overhead | 12 核 / 600 · 1200 · 4000 任务 | `c01c…`·`3969…`·`4286b6f704b38748` | **相同** |

结论：**shared 与 private 解析出完全相同的依赖图**（跨 72 并发核、含 4000 任务大图），二者数值差异纯属
浮点累加顺序噪声（private 自身也有），非正确性差异。run-ahead 反压使 4000 任务的 skip-exec 大图从
"`Δ+H` 溢出 FATAL"转为顺利完成且签名一致。

## 13. 进程全局变量的跨平台处理（`global_data` 段 + base 指针寻址）

### 13.1 问题：CCEC 不支持进程全局变量

在常规 pthread 编程里，进程拥有的 file-scope 全局变量**对进程内所有线程自动可见、自动共享**。本分布式
运行时正是**依赖这一点**在 a2a3sim 上工作的：仿真平台把每个 AI 核实现为**同一进程内的一条 host 线程**，
共享同一地址空间，于是像 `DistGlobal g_dist`（含全局 task-id 定序器 `tm_insert_next`、shared TensorMap、
claim cursor、完成标志环、frontier/vend、block.won 投递表 …）这样**唯一一份 file-scope 全局对象**天然被
所有核共享，语义正确。

**但真实硬件（a5 等）不成立。** AICore 由 **CCEC 编译**，其生成的核上程序**不支持进程全局变量**
（无可共享的 `.data`/`.bss` 进程段）：file-scope 全局要么不可用，要么**每核各一份、互不共享**——而我们
恰恰需要**跨核共享**的编排状态。因此必须改造这块"进程全局空间"的表达与访问方式。

**关键分工（决定了改造边界）：**

| 阶段 | 运行在 | 能力 |
| ---- | ---- | ---- |
| `dist_engine_register()`（每次运行一次） | **AICPU（ARM A55）** | 有 `malloc`、有正常进程全局；可分配并初始化共享段 |
| `dist_core_main()` 及其被调用链（replay/claim/execute/drain、ops 回调） | **AICore（CCEC）** | **无进程全局**；只能靠传入的参数/寄存器寻址共享内存 |

即：**AICPU 负责"分配 + 初始化"共享段，AICore 只能"经指针访问"它**。当前代码里 AICore 直接引用
`g_dist.*` / `g_self` / `g_tm_*` 等 file-scope 全局（`dist_engine.cpp` 内约 180 处），这些正是在 a5 上失效的点。

### 13.2 设计：运行时分配 `global_data` 段，base 指针经 worker 参数下发

**核心方案（同一套代码在 sim 与 HW 上都走这条路）：**

1. **分配（AICPU / register）**：启动时把**所有需跨核共享的全局状态收拢进一个结构体 `DistGlobal`**，
   在一块**全局可寻址内存（GM）**上分配其唯一实例，并完成初始化（cursor=-1、flags=0、topology、模式旋钮…）。
2. **下发 base（经已有的 worker 参数）**：把段基址存入 `Runtime::dist.global_data_base`（新增字段）。
   `runtime`（一个共享 GM 上的 `Runtime*`）**本就作为参数**传给每个 worker：
   `core_main(runtime, core_idx, core_type)`。故**任何核都能从参数链取得 base**，无需任何 file-scope 符号。
3. **访问（AICore）**：`DistGlobal* gd = (DistGlobal*)runtime->dist.global_data_base;`，其后所有
   `g_dist.X` → `gd->X`。这**天然编译成 `base + 字段偏移` 的访存**——正是用户要求的"base 指针 + 各变量
   偏移量算出实际地址"的方案，且**不产生任何进程全局符号**，CCEC 安全。
4. **偏移量来自 `offsetof`，不写死魔数**：字段地址 = `base + offsetof(DistGlobal, field)`，由编译器在编译期
   给出。工程上**直接用"类型化结构指针 + 成员访问"（`gd->field`）即可**——编译器自动发射 base+offset 访存，
   既杜绝手算偏移出错，又满足 CCEC 无全局约束。§13.4 给出完整字段清单及其 `offsetof` 语义。

```text
┌─ AICPU (A55, register) ────────────────┐        ┌─ AICore #k (CCEC, core_main) ──────────┐
│ gd = alloc_global_segment(sizeof(DistGlobal)) │  │ gd = (DistGlobal*)runtime->dist.global_data_base │
│ init gd->cursors/flags/topology/...    │  base  │ self = &gd->cores[core_id()]           │
│ runtime->dist.global_data_base = gd  ──┼───────▶│ ...  gd->frontier / gd->shared_map ... │
│ store-release; publish dist.go=1       │        │ (base + offsetof 访存，无全局符号)     │
└────────────────────────────────────────┘        └────────────────────────────────────────┘
```

### 13.3 方案 B：AICore 执行路径上零 file-scope 符号（`gd`/`self` 全程参数穿引）

> **为什么不是"平台化 file-scope 指针"（方案 A）**：曾考虑用 `#if` 把 `g_gd`/`g_self` 在 sim 上编成
> `thread_local`、在 HW 上编成"每核各一份的普通指针"。但这依赖"CCEC 允许每核私有的**可写 file-scope 静态
> 存储**"这一前提；一旦 CCEC 完全禁止可写静态存储，方案 A 即失效。**方案 B 更强硬也更可移植：AICore 的
> 功能路径上不出现任何 file-scope / thread_local 符号，`gd`（段基址）与 `self`（当前核）一律作为参数穿引。**

**这正是集中式运行时早已在用的套路**：`Runtime` / `PTO2Runtime` 本身就是"AICore 经一个发下来的 base 指针
访问的 GM 结构"，其内部再以指针字段（如 `aicore_mailbox`、`sm_handle`）指向 arena 里的其它子区。方案 B
照搬这一模式，把 `DistGlobal` 段基址经 AICore **本就读取**的运行时对象下发：

| 入口 | 拿到什么 | 如何取 `gd` / `self` |
| ---- | ---- | ---- |
| `dist_core_main(runtime, core_idx, …)`（每核 worker 主函数） | `Runtime*`（arg） | `gd = (DistGlobal*)runtime->dist.global_data_base;` `self = &gd->cores[core_idx];` |
| ops 回调（`dist_submit_impl` / `dist_alloc_tensors` / `dist_get/set_tensor_data` / `dist_is_fatal` / `dist_report_fatal`） | 仅 `PTO2Runtime* rt` | `gd = (DistGlobal*)rt->dist_global;` `self = &gd->cores[pto_core_id()];`（见下） |
| 其余 helper（绝大多数持有 `self`） | `DistCore* self`（arg） | 顶部 `DistGlobal* gd = self->gd;`（`DistCore` 的回指字段） |
| 无 `self` 的自由函数（`fatal_set`/`set_fatal`/`advance_frontier`/`watchdog`/`dep_sig_add`/`tm_shared_*`/`alloc_won_slot`） | — | 显式加 `DistGlobal* gd` 形参 |

**ops 回调的回收 seam（新增 `PTO2Runtime::dist_global` 字段）**：回调只收到 `rt`，故在 `rt` 上加一个
`void* dist_global` 字段（register 时写入段基址），回调即 `pto_gd(rt) = (DistGlobal*)rt->dist_global`——
与 `rt` 已有的 `aicore_mailbox`/`sm_handle` 指针字段同构，不引入任何进程全局。

**"我是哪个核" `pto_core_id()`（编译期分支，段里唯一的 per-core 数据）：**

| 平台 | 实现 |
| ---- | ---- |
| a2a3sim | `thread_local int32_t`（在 `dist_core_main` 入口用 `pto_set_core_id(core_idx)` 写入）。注意：这是 **seam 内部**的一个 thread_local **整数**，不是 file-scope 的 `self` 指针；且**此分支在 HW 上不编译**。 |
| a5 / CCEC | 读**硬件 per-core id 寄存器**（block/core index，见 [simt-launch.md](simt-launch.md)），无任何存储。 |

为把改动局部化，`DistCore` 增加一个**回指指针 `DistGlobal* gd`**（register 时写好）：凡持有 `self` 的函数
即可 `self->gd->…`；只有少数无 `self` 的自由函数/回调需显式取 `gd`。这样"段化"后**没有任何一处功能代码
依赖 file-scope 可写符号**。

### 13.3.1 平台抽象 seam（GM 分配 / 缓存一致 / core-id）

方案 B 把所有平台差异收敛到 `dist_engine.cpp` 顶部匿名命名空间里的四个内联 seam（`#if DIST_SIM_HOST_CLOCK`
分支；`DIST_SIM_HOST_CLOCK==0` 当且仅当 CCEC/HW）。功能代码只调它们，不再出现任何平台条件：

| seam | 运行在 | a2a3sim | a5 / CCEC（TODO 联调） |
| ---- | ---- | ---- | ---- |
| `pto_gm_alloc(bytes)` / `pto_gm_free(p)` | AICPU | host `malloc`/`free` | GM 分配器：返回**所有核可寻址且一致**的一块 GM |
| `pto_gm_publish(base, bytes)` | AICPU | 空操作（同地址空间 + register 处的 release 栅栏已足够） | 对 `[base, base+bytes)` 做 flush/invalidate，使各核读到已初始化态 |
| `pto_core_id()` / `pto_set_core_id(id)` | AICore | `thread_local int32_t`（入口写入 `core_idx`） | 读硬件 core-id 寄存器；`set` 为空 |

- **分配只发生在 AICPU**（register），AICPU 有 `malloc`/GM 分配器与进程全局，故段句柄用一个**进程静态指针**
  持有（跨 run 复用、每 run 重置）——"无全局"约束只针对 AICore。
- **一致性**：sim 单地址空间 + register 末尾的 `atomic_thread_fence(release)` 即够；HW 由 `pto_gm_publish`
  在 worker 观测到 `dist.go` 之前把段刷出。
- `pto_core_id()` 是**段内唯一的 per-core 数据**，也是方案 B 对硬件的唯一新增要求。

### 13.4 完整全局变量清单与段内布局

**(A) 收拢进 `DistGlobal` 段的共享状态**（跨核共享，必须迁移；字段地址 = `base + offsetof(DistGlobal, ·)`）：

| # | 变量 / 字段 | 类型 | 用途 | 现状 |
| - | ---- | ---- | ---- | ---- |
| 1 | `cube_cursor[4]` / `vector_cursor[4]` / `alloc_cursor[4]` | `PaddedCursor`（cacheline 对齐） | 分类型 claim 高水位（cursor sharding，§6.6） | `g_dist` 内 |
| 2 | `flags[kFlagCap]` | `atomic<uint8_t>[65536]` | 每任务完成标志环（§11.5） | `g_dist` 内 |
| 3 | `frontier` / `H` / `vend[kFlagCap]` | `atomic<int32_t>` / `int32_t` / `atomic<uint64_t>[65536]` | 完成前沿 F、依赖跨度、累计虚拟堆字节（§9.5/§11.4） | `g_dist` 内 |
| 4 | `heap_base` / `heap_size` | `uint8_t*` / `size_t` | 确定性 GM 输出堆环 | `g_dist` 内 |
| 5 | `orch_func` / `orch_args` / `rt` / `runtime` | 指针 | 编排入口/参数/运行时回指 | `g_dist` 内 |
| 6 | `fatal` | `atomic<int32_t>` | 全局致命标志 | `g_dist` 内 |
| 7 | `num_workers` / `num_blocks` / `layout[]` / `blocks[]` | 标量 / `CoreLayout[]` / `BlockWon[]` | 物理拓扑 + block.won 投递表（§3.1） | `g_dist` 内 |
| 8 | `replay_done` / `started_count` | `atomic<int32_t>` | tail-idle 计数 + 启动栅栏（§7） | `g_dist` 内 |
| 9 | `shared_map` / `tm_insert_next` / `core_progress[]` | `SharedTensorMap` / `atomic<int32_t>` / `atomic<int32_t>[]` | shared 模式全局环 + 追加定序器 + 各核进度（§12） | `g_dist` 内 |
| 10 | `cores[RUNTIME_MAX_WORKER]` | `DistCore[]` | 各核私有状态（private map、task slot 环、outpool…）；**新增回指 `gd`** | `g_dist` 内 |
| 11 | `tm_shared` / `ring_cap` / `tm_runahead_max` / `runahead_max` | `bool`/`int32_t` | 模式/容量/run-ahead 旋钮（register 设定、全程只读）；`runahead_max` 为两模式通用的均衡上界 `Δ_max`（§12.7.3，默认 `2×num_workers`） | 现为独立 `g_*`，**并入段** |
| 12 | `dep_sig` / `dep_edges` | `atomic<uint64_t>` | 依赖图签名（验证用，§12.10） | 现为独立 `g_*`，**并入段** |

**(B) 仿真专属、已被 `#if DIST_SIM_HOST_CLOCK` 排除于 HW 之外**（HW 构建里根本不编译，无需迁移，但为
"同源"整洁仍建议并入段）：`overhead_on`、`skip_exec`、`trace_on`/`trace_epoch_ns`/`trace_reserve`、
`tm_inserts`/`lookups`/`scans`、`orch_t0_min…replay_sum`/`orch_recorded`。
> `DIST_SIM_HOST_CLOCK == 0` 当且仅当 `__CCE_AICORE__ || __DAV_C220__ || __CCE_KT_TEST__`（即 HW/CCEC），
> 故这批开销/追踪计数在 a5 上不存在，天然规避。

**(C) 只读常量与函数指针表**：
- `g_dist_ops`（`const PTO2RuntimeOps`，ops 函数指针表）：只读常量数据，`rt->ops = &g_dist_ops` 取址。
  只读常量不是"可变进程全局"，但 CCEC 下 const-data 的放置需按平台约定（常量区/GM）；实现时以
  `constexpr`/只读 GM 常量登记，或在 register（AICPU 侧）把表拷入段的一个 `ops` 字段并让 `rt->ops` 指向它。
- `kFlagCap`/`kRingBuckets`/`kBucketCapMax`/`kHDefault`/`kPrivateSlots` 等 `constexpr`：编译期常量，无存储，**无需迁移**。
- `g_self`（原 `thread_local`）：方案 B 已删除，由 `pto_dist_self(rt)`（`rt->dist_global` + `pto_core_id()`）取代，见 §13.3。

### 13.5 逐点访问迁移映射（约 180 处）

| 原引用（file-scope 全局） | 迁移后（base+offset 访问） | 备注 |
| ---- | ---- | ---- |
| `g_dist.<f>`（约 140 处） | `gd-><f>`（每函数顶部 `gd = self->gd` / 形参 / 局部） | 方案 B：`gd` 全程参数穿引，无 file-scope 符号 |
| `DistCore* self = g_self;`（ops 回调，4 处） | `DistCore* self = pto_dist_self(rt);` | `gd=(DistGlobal*)rt->dist_global`（新字段）+ `pto_core_id()` |
| `g_tm_shared`（13） | `gd->tm_shared` | 只读旋钮 |
| `g_dist_ring_cap`（7） | `gd->ring_cap` | 只读旋钮 |
| `g_tm_runahead_max`（7） | `gd->tm_runahead_max` | 只读旋钮 |
| `g_dep_sig` / `g_dep_edges`（10） | `gd->dep_sig` / `gd->dep_edges` | 验证累加器 |
| `g_tm_inserts/lookups/scans`（15） | `gd->tm_inserts/...` | 仅 `DIST_SIM_HOST_CLOCK` 下增量 |
| `g_orch_*`（sim-only） | `gd->orch_*` | `#if DIST_SIM_HOST_CLOCK` 内 |

**访问约定（方案 B，减小改动面）：** 全文 `g_dist.` → `gd->` 后，在每个用到 `gd` 的函数补一个来源：持有
`self` 的函数顶部 `DistGlobal* gd = self->gd;`；`dist_core_main` / `register` 用局部 `gd`；无 `self` 的自由函数
加 `DistGlobal* gd` 形参；ops 回调用 `pto_gd(rt)` / `pto_dist_self(rt)`。**编译器即是清单**——改完签名后一次
构建会精确列出所有"缺 `gd`"的函数，逐一补齐即可。

### 13.6 平台统一与验证策略

- **同一套源码、同一条路径**：sim 与 HW **都**分配段、都经 `runtime->dist.global_data_base` 访问。sim
  上段就是 host 堆的一块（`malloc`），HW 上是 GM 分配；差异仅在**分配器**与**core-id/self seam**两个被
  `#if` 隔离的点。sim 上不需要该方案，但**故意也走它**——用来在上 a5 之前，先在 a2a3sim 证明改造正确。
- **验证判据（复用现有）**：
  1. 差分 UT `test_dist_tensormap_ring.cpp` 仍通过；
  2. `PTO_DIST_DEPSIG` 下 private == shared 签名不变（6/24/72 核）；
  3. BGEMM / PagedAttention golden 与改造前一致。
  三者全绿即证明"段化 + base 寻址"未改变任何功能语义。

### 13.7 落地实现计划（方案 B，分阶段，可逐步在 a2a3sim 验证）

1. **平台 seam**：加 §13.3.1 的 `pto_gm_alloc/free` / `pto_gm_publish` / `pto_core_id/set_core_id`（`#if DIST_SIM_HOST_CLOCK`）。
2. **段结构收拢**：把 §13.4(A)(B) 的独立 `g_*` 并入 `DistGlobal`；`DistCore` 增 `DistGlobal* gd` 回指。
3. **下发通道**：`Runtime::DistHandoff` 增 `volatile uint64_t global_data_base`（给 `dist_core_main`）；
   `PTO2Runtime` 增 `void* dist_global`（给 ops 回调）；`shared/runtime.cpp` 初始化 `global_data_base=0`。
4. **参数穿引**：全文 `g_dist.`→`gd->`；每函数补 `gd` 来源（`self->gd` / 形参 / 局部 / `pto_gd(rt)`）；
   `g_self`→`pto_dist_self(rt)`。删除所有 file-scope `g_gd`/`g_self`。
5. **register / core_main**：register 经 `pto_gm_alloc` 分配、初始化、`pto_gm_publish`，写两个下发字段与
   `cores[].gd`；`dist_core_main` 入口取 `gd`、`pto_set_core_id(core_idx)`。
6. **构建 + 验证 a2a3sim**：DEPSIG（private==shared）、TMOPS、BGEMM golden 全绿。
7. **（后续）a5 落地**：实现 `pto_gm_alloc/publish` 的 GM 版与 `pto_core_id()` 的寄存器版，打通 CCEC 分支。

### 13.8 落地实现与验证（方案 B，已完成，a2a3sim）

第 1–6 步已实现于 `dist_engine.cpp` / `pto_runtime2.h` / `runtime.h` / `shared/runtime.cpp`，要点：

- **平台 seam**（§13.3.1）落在 `dist_engine.cpp` 顶部匿名命名空间；功能代码只调 seam，零平台条件。
- `DistGlobal` 收拢全部功能性共享状态（含 `tm_shared`/`ring_cap`/`tm_runahead_max`/`dep_sig`/`dep_edges`）；
  `DistCore` 增回指 `DistGlobal* gd`。
- **下发**：`Runtime::DistHandoff` 增 `global_data_base`，`PTO2Runtime` 增 `dist_global`。
  `dist_engine_register`（AICPU）经 `pto_gm_alloc` + placement-new 分配段（进程静态句柄，跨 run 复用），
  `pto_gm_publish` 刷出，写 `runtime->dist.global_data_base`、`rt->dist_global`、所有 `cores[i].gd`。
- **零 file-scope 符号**：`g_gd`/`g_self` 已彻底删除。约 140 处 `g_dist.*`→`gd->`，`gd` 一律来自
  `self->gd` / 形参 / 局部 / `pto_gd(rt)`；4 处回调的 `self` 改由 `pto_dist_self(rt)`（`rt->dist_global` +
  `pto_core_id()`）回收；`fatal_set`/`set_fatal`/`advance_frontier`/`watchdog`/`dep_sig_add`/`tm_shared_*`/
  `alloc_won_slot` 增 `gd` 形参。
- **诊断隔离**：`dist_dump_state`（SIGUSR1/watchdog 信号处理器，签名 `void(int)`）与 `dist_engine_dump_trace`
  （arg-less swimlane 导出）是 host/sim 专属调试器（`fprintf`/`chrono`，env 门控，不编到 CCEC），故用一个
  **AICPU 侧诊断句柄 `s_dump_gd`**（register 时写）访问——**不在功能路径上**，功能路径仍零符号。
- TMOPS 诊断计数仍为 `#if DIST_SIM_HOST_CLOCK` 专属，`TMOP_COUNT()` 宏在 HW 上编空。

**验证（a2a3sim，与迁移前判据一致）：**

| 判据 | 结果 |
| ---- | ---- |
| DEPSIG private vs shared（runtime_overhead 12 blocks / 36 核 / 240 任务） | 均 `edabb0ba8876d2cf`（360 边）——**逐位一致** |
| TMOPS（36 核） | private `inserts=17280` vs shared `inserts=480`（≈单环追加），`lookups` 均 `1440`——**插入下沉比符合预期** |
| kernels-enabled exec（4 blocks / 48 任务） | 正常完成，无 FATAL/abort，产出 `OVERHEAD` 指标 |
| 构建 | a2a3sim（AICPU/AICORE/HOST/SIM_CONTEXT）全部 `Build complete!` |

> a5 硬件落地（第 7 步）尚待：实现 `pto_gm_alloc`/`pto_gm_publish` 的 GM 版与 `pto_core_id()` 的寄存器版，
> 打通 CCEC 分支。sim 上"零 file-scope 符号 + 参数穿引 + base 寻址"已验证功能正确，为 a5 联调的基线。

## 14. 跨核缓存一致性抽象（A5 落地契约）

§13 解决了"进程全局变量"，但那只是把共享状态**放到哪里**的问题。真正让 SPMD 引擎能在 A5 上跑通的，是
**跨核可见性**：a2a3sim 的每个"核"是同进程 host 线程，天然缓存一致，`std::atomic` 的 `memory_order` 足以
保证一个核写、另一个核读的可见性。**但 A5 的 AICore 之间没有硬件缓存一致**——一个核写入的字，另一个核在
它**刷出到 HBM（`dcci … CACHELINE_OUT`）**且读者**失效本地陈旧副本（`dcci …`）**之前都看不到，这正是
a5 `aicore_executor.cpp` 全程手动做的事（第 66–194 行的 `dcci`/`OUT_OF_ORDER_STORE_BARRIER`）。因此在 HW 上
`std::atomic` 是**必要但不充分**的：它只排序本核访问，从不跨核发布。

### 14.1 设计：`Coherent<T>` —— `std::atomic<T>` 的一致性替身

在 `dist_engine.cpp` 顶部匿名命名空间引入一层 seam 与一个替身类型 `Coherent<T>`：

- **一致性原语（`#if DIST_SIM_HOST_CLOCK`）**：`pto_dcci_inval(p,n)` / `pto_dcci_flush(p,n)` /
  `pto_shared_fence(order)`。sim 上 `inval`/`flush` 为空（同地址空间 + 下面的 `std::atomic` 序即足够），
  HW 上分别发失效 / 刷出 + 屏障（TODO a5）。
- **`Coherent<T>`**：只含一个 `std::atomic<T> a` 成员（**size/alignment 与 `std::atomic<T>` 相同**，故
  `DistGlobal` 布局与 `sizeof` 不变）。它的每个方法都镜像调用点用到的 `std::atomic` 操作——读前 `inval`、
  写后 `flush`：

| 方法 | 语义 |
| ---- | ---- |
| `load(order)` | `dcci_inval` → `a.load(order)` |
| `store(v, order)` | `a.store(v, order)` → `dcci_flush` |
| `compare_exchange_weak/strong(…)`、`fetch_add/sub/xor(…)` | `dcci_inval` → RMW → `dcci_flush` |

**关键收益**：迁移只改**字段声明**（`std::atomic<T>` → `Coherent<T>`），约 80 处调用点的
`.load(order)/.store/.cas/.fetch_*` 语法**原样不动**。sim 上 `Coherent<T>` 就是原来的 `std::atomic` 操作
（`dcci` 编译为空）——**行为逐位不变、零成本**；HW 上则在同一处集中发 `dcci`。

### 14.2 已段化为 `Coherent<T>` 的跨核共享状态

`flags[]`、`frontier`、`vend[]`、`fatal`、`replay_done`、`started_count`、`tm_insert_next`、
`core_progress[]`、`dep_sig`/`dep_edges`（`DistGlobal`）；`cube/vector/alloc_cursor[].v`（`PaddedCursor`）；
`SharedTensorMap` 的 `seq`/`head[]`/`tail[]`；`block.won` 的 `state`/`remaining`/`drained[]`/`any_pub`。
两处显式发布点的 `std::atomic_thread_fence(release)` 改为 `pto_shared_fence(release)`。
> 仿真专属诊断计数（`g_tm_*`/`g_orch_*`，`#if DIST_SIM_HOST_CLOCK` 内）仍为普通 `std::atomic`——不在 HW 路径上。

### 14.3 A5 落地契约与遗留风险

- **落地点收敛为三处**：`pto_dcci_inval` / `pto_dcci_flush` / `pto_shared_fence` 的 HW 实现，加上 §13.3.1 的
  `pto_gm_alloc/publish` 与 `pto_core_id()`。功能代码不再有任何平台条件。
- **粒度**：当前按 `sizeof(atomic)`（≤8B，单 cacheline 内）失效/刷出。`flags[]` 是逐字节环，相邻 flag 可能
  共享 cacheline——HW 上 `dcci` 以 cacheline 为单位，需确认"失效邻居未刷出的写"不会丢数据（必要时 flag 按
  cacheline 对齐，或改用 per-core flag 分片）。
- **跨核 RMW 原子性（最大遗留风险）**：`claim()` 的 `fetch_max`-式 CAS、`remaining.fetch_sub`、`state`/
  `drained` 的 CAS 依赖**核间真原子**。`Coherent<T>` 的 "inval + `std::atomic` RMW + flush" 在 sim 正确，
  但 HW 上若无核间原子单元 / LL-SC，则**不是真原子**——认领仲裁可能需改设计（HW 原子单元，或退回 AICPU 仲裁）。
  `Coherent<T>` 把这一决策**集中到了一处**，但并未消除它。
- **`__gm__` 地址空间**：`Coherent<T>` 目前是普通指针成员；CCEC 上段与其内部访问需 `__gm__` 限定，属后续 HW 化工作。

### 14.4 验证（a2a3sim，与迁移前逐位一致）

| 判据 | 结果 |
| ---- | ---- |
| DEPSIG private vs shared（12 blocks / 36 核 / 240 任务） | 均 `edabb0ba8876d2cf`（360 边）——**与引入 `Coherent<T>` 前逐位相同** |
| TMOPS（36 核） | private `inserts=17280` vs shared `inserts=480`，`lookups` 均 `1440`——不变 |
| kernels-enabled exec（4 blocks / 48 任务） | 正常完成，无 FATAL/abort |
| 构建 | a2a3sim 全部 `Build complete!` |

> 结论：`Coherent<T>` 建立了 HW 缓存一致性的**单一接缝**且对 sim 零影响。A5 联调时只需实现三处 `dcci`/屏障
> 原语，并单独攻克 §14.3 的跨核 RMW 原子性与 `__gm__` 化——这两项是 SPMD 引擎上 A5 的下一个主攻方向。

## 15. a5sim 落地 SPMD 引擎（上真机前的基线）

上真 A5 硬件前，先让 **a5sim** 跑通同一套去中心化 SPMD 引擎（`dist_engine.cpp`）。a5sim 与 a2a3sim 共用同一
host 线程执行模型（AICore 也是同进程 `std::thread`，非独立 CCE 地址空间），故 `DIST_SIM_HOST_CLOCK=1`，§14 的
`Coherent<T>`/`dcci` 编译为空——本阶段验证的是**引擎结构能在 a5 代码树 / 类型上跑通**，真正的 HW 一致性路径要到
onboard CCEC 构建才激活。

### 15.1 共享单一源（消除双份维护）

`dist_engine.{cpp,h}` 从 `src/a2a3/.../runtime/` 迁到 **`src/common/runtime/fully_distributed_within_core/`**，
a2a3 与 a5 的 `build_config.py` 各自把该公共目录加入 **AICPU** 的 `include_dirs` + `source_dirs`（`DIST_COMMON`）。
CMake 递归 GLOB 该目录、编译器用**各 arch 自己的 `-I`** 解析 `runtime.h`/`pto_runtime2.h` 等——**同一份源、按 arch 编译**，
零重复。dist_engine 只编进 AICPU `.so`（`dist_core_main` 经函数指针在 AICore 线程上执行），与 a2a3 一致。

- **可移植接缝**：a2a3 的 `LocalContext` 用 `block_idx`/`block_num`，a5 用 `s_block_idx`/`s_block_num`（避开 CCE
  内建符号冲突）。共享源用**检测惯用法重载** `dist_set_local_block()` 设值——优先选 `s_` 字段，否则回退无前缀，
  两 arch 同源编译，**不改任一 `intrinsic.h`**。

### 15.2 a5 侧接线

- **类型对齐**：a5 `runtime.h` 的 `DistHandoff` 补 `global_data_base`，`Runtime` 补 `use_example_exec_time_`/
  `example_exec_time_ns_[]`；`pto_runtime2.h` 补 `dist_global`；`shared/runtime.cpp` 构造函数初始化上述字段。
- **AICPU**（`aicpu_executor.cpp`）：把编排直调 `(*p_func)(orch_args)` 换成 **dist handoff**——`dist_engine_register`
  写 `core_main_fn`、置 `dist.go`、等 `done_count==num_workers`、`dist_engine_dump_trace`、恢复 `rt->ops`。
- **AICore**（`aicore_executor.cpp`）：保留 a5 的 phase1–3 握手与 teardown（EXIT/EXITED 协议），把 `DATA_MAIN_BASE`
  轮询主循环换成 **等 `dist.go` → 调 `core_main_fn`**（SPMD 入口）。

### 15.3 关键修复：scheduler 的 0-task 完成判据

SPMD 下编排/调度/执行全在核上完成，**没有任务下发到 AICPU 共享内存**，故 `on_orchestration_done(total_tasks=0)`。
a2a3 的 `handle_orchestrator_exit` 对此有专门分支（`completed_tasks_ >= task_count` 即 `0>=0` → 立即 `completed_`），
而 **a5 版多了 `task_count > 0 &&` 门**，导致 SPMD 路径永不完成——engine 已跑完（9 核 START→DONE、DEPSIG 已出、
"all workers finished"），但 AICPU scheduler 线程在 `completed=0/0` **空转不退出**，整个 run 挂死。修复：a5
`scheduler_cold_path.cpp::handle_orchestrator_exit` 去掉 `task_count > 0 &&` 门，与 a2a3 对齐。

### 15.4 验证（a5sim）

移植的用例置于 **`tests/st/a5/fully_distributed_within_core/`**，用 `--platform a5sim` 运行：

| 用例 | 结果 |
| ---- | ---- |
| `vector_example`（AIV-only，5 任务 DAG，golden 精确 f=47） | golden PASSED；DEPSIG private==shared `0d7fa3a297fced17`（6 边） |
| `mix_coown`（MIX 1C+2V 协同拥有 + 消费者，12 组） | golden PASSED（rtol/atol 1e-3）；DEPSIG `300391b07de72c6f`（12 边） |

> a2a3sim 回归：迁源 + 可移植接缝后，paged_attention DEPSIG private==shared 仍 `a7db56ff3de5afa6`（15 边），无回归。
> 遗留（上真 A5）：§14.3 的跨核 `dcci`/屏障 HW 实现、跨核 RMW 真原子性、`__gm__` 化，以及物理→逻辑 `pto_core_id()`。

## 16. A5 onboard（真机）落地设计与阶段计划

§15 让 SPMD 引擎在 **a5sim** 跑通,但 a5sim 与 a5 onboard 是两个世界:sim 是同进程 host 线程(单一地址空间、天然缓存一致、`thread_local`),onboard 是 AICPU(aarch64)+ 多个 AICore(CCEC)的**异构、非一致缓存、分地址空间**真机。对 a5 onboard 代码库的调查暴露了两个**根本阻塞**,它们决定了 onboard 落地不是"填 seam",而是**重构 + 一个悬而未决的硬件设计决策**。

> 注:本机无 CCEC 工具链(`build_runtimes.py --list` 仅 a2a3sim/a5sim),onboard 无法在此编译验证;下述实现均需在 CI/真机上做 CCEC 编译与硬件验证。

### 16.1 阻塞 1（架构）：热路径编在了错误的处理器上

onboard a5 各 target 的编译器(`runtime_compiler.py::_init_a5`):`aicore`→**CCEC**;`aicpu`→**aarch64-g++**;`host`→g++。
`dist_engine.cpp` 现在只在 **aicpu** 的 `source_dirs`,故由 **aarch64-g++** 编译。sim 上"AICore worker"是同进程 host 线程,靠函数指针调用 `dist_core_main` 没问题;但**真机 AICore 是独立 CCEC 处理器,无法执行 aarch64 编出来的函数**。

推论:
- onboard AICPU 编译单元里 `__CCE_AICORE__` 未定义 → `DIST_SIM_HOST_CLOCK=1` → §13/§14 的 **HW seam `#else` 分支永不编译**(死代码)。单填 seam 不产生任何真机效果。
- `dist_core_main` 及其调用链(claim、per-core TensorMap、`Coherent<T>`、execute)**必须 CCEC 编译进 AICore 二进制**;只有 `dist_engine_register`(GM 段分配 + 发布)留在 AICPU。

**所需重构**:把引擎拆成两个编译单元 —— AICPU 侧(`register`/GM 分配/发布,aarch64)与 AICore 侧(`dist_core_main` 热路径,CCEC),平台 seam 抽到 CCEC 会编译的 header;`dist.core_main_fn` 在真机改为指向 **AICore 二进制内**的入口(而非 AICPU .so 内的函数指针)。同时需确认引擎热路径不使用 CCEC 不支持的 STL(`std::vector/string/chrono` 已被 `#if` 关闭,但需 CCEC 实编确认无残留)。

### 16.2 阻塞 2（硬件能力）：~~a5 AICore 无跨核原子 RMW~~ —— 已解除：A5 有硬件 GM 原子

> **本节结论已修订（2026-07）。** 早前基于对旧代码注释（`// No hardware fetch_max on the target`）
> 的判断，认定"a5 AICore 无跨核原子 RMW"，并把它列为总闸 blocker。**这个判断是错的。**
> A5（`dav_3510`）实测拥有可用、核间一致的 GM 硬件原子（见下）。claim 竞争可用一条硬件
> `atomicMax` 直接实现，**无需** AICPU 仲裁 / 静态分派 / 软件锁，也**无需** uncacheable 内存。

**证据。** CANN `dav_3510` 头 `asc/impl/basic_api/dav_3510/kernel_operator_atomic_impl.h` 暴露：
`atomicAdd`/`atomicMax`/`atomicMin`（`int32_t/uint32_t/int64_t/uint64_t/float`）、
`atomicCAS`/`atomicExch`（`uint32_t/uint64_t`），均作用于 `__gm__` 地址。CANN 文档
[AtomicMax](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_00261.html)
给出三核并发 `AtomicMax` 结果正确、各核返回操作前旧值的示例——**内存级、核间序列化的真原子**。
`dist_engine.cpp` 现有的 `dist_atomic_cas`/`dist_atomic_add` 已经在用 `atomicCAS`/`atomicAdd`，
说明 RMW 写侧本就走硬件原子。

**真正的真机 bug（已定位）在读侧，不在原子侧。** `coherent_load` 的 onboard 分支是**普通
`load`（无 `dcci` 失效）**，注释假设"uncacheable → 普通 load 即一致"。但本机 double page table
不可用（§16.2 附注/探针），段实际是 **cacheable**，普通 load 读到的是本核**陈旧缓存副本**。
后果：`atomicAdd`/`atomicMax` 的写在内存里是对的，但用 `coherent_load` 观察这些量的核看不到
别人的更新 → drain barrier（`replay_done`）永远读不到 `num_workers` → 挂死（实测 9 核只读到 8）。

**修复方向（取代旧的 (0)/(1)/(2)/(3) 候选）：**

1. **claim / 全局 cursor**：改用单条硬件 `atomicMax`（§11.1.1）。认领 `old = atomicMax(&cursor[T], N); won = old<N;`，一次原子完成读+写，天然绕开陈旧读。
2. **cursor 的纯读**（节流/诊断）：用幂等原子 `atomicMax(&cursor, INT32_MIN)` 读回内存真值（§11.1.1）。
3. **其余共享量的读**（`coherent_load` 通路：`frontier`/`started_count`/`replay_done`/`flags[]`/`block.won`）：
   onboard 分支**加 `dcci` 失效再 load**（对齐 sim 分支的 `pto_dcci_inval`），或对可用类型改幂等原子读。RMW 保持硬件原子。
4. **flag 环 false sharing**（C4）：`flags[]` 逐字节共享 cacheline，`dcci` 以 line 为粒度——仍需 flag 按 cacheline 对齐或 per-core 分片；这一条独立于原子能力，仍待处理。

> 附注（uncacheable 探针，2026-07）：`halMemCtl(CTRL_TYPE_GET_DOUBLE_PGTABLE_OFFSET)` 在本机
> 返回 `rc=0` 但 `offset=0`（`CTRL_TYPE_GET_DCACHE_ADDR` 亦返回 0），即驱动确认**本设备无
> cacheable/uncacheable 双页表别名**。故不能靠"把段搬到 uncacheable 别名"来获得一致性——
> 必须走上面的硬件原子 + `dcci` 读一致路线。

### 16.3 seam → a5 现成原语映射（重构后填入 CCEC 单元）

| seam(§13/§14) | a5 onboard 落点 |
| ---- | ---- |
| `pto_gm_alloc/free`(AICPU) | host `rtMalloc(RT_MEMORY_HBM)` 预分配 pooled 段(`memory_allocator.cpp`),或 AICPU `halMemAlloc`(`device_malloc.cpp`);段基址经 `runtime->dist.global_data_base` 下发 |
| `pto_gm_publish`(AICPU) | `cache_flush_range`(`dc cvac`+`dsb sy`,`cache_ops.cpp`)刷 `[base,base+bytes)` |
| `pto_dcci_inval`(AICore) | CANN `dcci(p, SINGLE_CACHE_LINE)`(无 `CACHELINE_OUT`,失效本地) |
| `pto_dcci_flush`(AICore) | CANN `dcci(p, SINGLE_CACHE_LINE, CACHELINE_OUT)`(写回 HBM) |
| `pto_shared_fence`(AICore) | `dsb(DSB_DDR)` + `pipe_barrier(PIPE_ALL)`(参见 `pto_async_kernel_api.h`) |
| `pto_core_id()`(AICore) | 逻辑 worker index = `dist_core_main` 的 `core_idx`(= launch `s_block_idx`),**由入口参数携带**,无需读寄存器;物理 id(`get_coreid()&0x0FFF`)仅用于诊断 |

> 注意 §14.3 粒度风险:`flags[]` 逐字节环,`dcci` 以 cacheline 为单位——需让 flag 按 cacheline 对齐或改 per-core 分片,否则"失效邻居未刷出的写"会丢数据。

### 16.4 阶段计划（每阶段带验证闸）

0. **P0 决策 —— 已完成。** claim 机制定为 **A5 硬件 `atomicMax`**（§11.1.1）：A5(`dav_3510`)有
   核间一致的 GM 硬件原子，无需 AICPU 仲裁 / 静态分派 / uncacheable 别名。**闸已过**：HW 原子能力
   结论（`kernel_operator_atomic_impl.h` + CANN 三核 `AtomicMax` 示例）+ 选定方案。
1. **P1 结构拆分(可 sim 验证)**:seam 抽到独立 header;引擎拆 AICPU/AICore 两单元;a5 `build_config` 把热路径加入 `aicore` `source_dirs`、`register` 留 `aicpu`。**闸**:a5sim + a2a3sim 仍 DEPSIG private==shared、golden 通过(结构不回归)。
2. **P2 CCEC 编译打通**:HW seam 分支填入(§16.3);`__gm__` 化引擎内 GM 指针;确认无 CCEC 不支持的 STL。**闸**:CCEC 编译 aicore 二进制成功(CI)。
3. **P3 claim + 读一致性落地**：
   - claim 用单条 `atomicMax`（`dist_atomic_max`，已落 `dist_engine.cpp` 的 `#if !DIST_SIM_HOST_CLOCK` claim 分支）；
   - global cursor 纯读改幂等 `atomicMax(&cursor, INT32_MIN)`（节流 §6.1 / 诊断）；
   - `coherent_load` 的 onboard 分支补 `dcci` 失效再 load（`frontier`/`started_count`/`replay_done`/`flags[]`/`block.won`），RMW 保持 `atomicAdd`/`atomicCAS`。
   **闸**:单核→多核真机小用例(vector_example / mix_coown) barrier 收敛、golden + 无重复认领/丢更新。
4. **P4 一致性 & 压测**:flag cacheline 对齐(C4，独立于原子能力)、makespan/负载均衡对比 sim。**闸**:真机 paged_attention 等 DEPSIG 与 sim 一致、性能达标。

## 17. Open Challenges（A5 onboard 未决项）

以下是把 SPMD 引擎从 a5sim 推进到 **a5 真机** 尚未解决的问题清单(§16 为对应设计)。分三档:**BLOCKER**(不解决无法在真机跑)、**CORRECTNESS**(能编但真机结果可能错)、**ENG/PROC**(工程或流程)。

| # | 挑战 | 档位 | 现状 / 根因 | 待决策 or 落地方向 | 验证依赖 |
| - | ---- | ---- | ---- | ---- | ---- |
| C1 | ~~跨核原子 RMW 缺失~~ **已解除** → 改为**读侧一致性** | ~~BLOCKER~~ CORRECTNESS | **修订**：A5(`dav_3510`)有硬件 GM 原子 `atomicAdd/Max/Min/CAS/Exch`(内存级、核间一致,CANN 文档三核 `AtomicMax` 示例佐证),claim 可用单条 `atomicMax` 实现(§11.1.1)。真机 bug 实为 `coherent_load` 的 onboard 分支是无 `dcci` 的普通 load(误设 uncacheable),读到陈旧缓存 → barrier 挂死。本机 double page table 不可用(探针 `offset=0`),不能靠 uncacheable 别名 | claim/cursor 用 `atomicMax`;cursor 纯读用幂等 `atomicMax(x,INT32_MIN)`;其余 `coherent_load` onboard 分支加 `dcci` 失效再 load(§11.1.1/§16.2) | 真机多核无重复认领 + barrier 收敛 |
| C2 | ~~热路径处理器错位~~ **结构拆分已完成** | ~~BLOCKER~~ 已落地 | **修订**：引擎已拆 AICPU(`register`,`SIMPLER_DIST_AICPU_ONLY`)/AICore(`dist_core_main`,`SIMPLER_DIST_AICORE_ONLY`)两单元；a5 `build_config` 把 `DIST_COMMON` 加入 `aicore` `source_dirs`；`aicore_execute` onboard 分支**直接调用** `dist_core_main`(非函数指针)。CCEC 已把 `dist_engine.cpp` 编成 `dist_engine_aic.o`/`dist_engine_aiv.o` 并链入 `aicore_kernel.o`(797KB)。**剩余的真机故障是 C10(ops 表函数指针重定位)，不再是编译错位** | 已完成；后续问题见 C10 | a5sim/a2a3sim 不回归(a5sim 通过；a2a3sim 见 C11);CCEC 编译成功✅ |
| C3 | **CCEC STL 支持不确定** | ENG(潜在 BLOCKER) | 引擎含 `std::vector/string/chrono`(HW 上已 `#if` 关闭),但需 CCEC 实编确认热路径无残留 STL / 异常 / RTTI 依赖 | 热路径去 STL 化;必要时改固定容量数组 | CCEC 编译通过 |
| C4 | **flag 环 cacheline 粒度 / false sharing** | CORRECTNESS | `flags[]` 逐字节环;`dcci` 以 cacheline 为单位失效/刷出 → "失效邻居未刷出的写"会丢数据 | flag 按 cacheline 对齐,或改 per-core 分片(§14.3 / §16.3 注) | 真机多核压测 |
| C5 | **计数缺配对 dcci** | CORRECTNESS | `runtime->dist.done_count` 的 `__atomic_add_fetch`(`aicore_executor.cpp`)无配对 dcci → 真机跨核语义不可靠(同属 C1 家族) | 计数改经确定的可见性协议(单写者 / dcci 配对 / 归约) | 真机 done_count 收敛 |
| C6 | **`__gm__` 地址空间化** | ENG | sim 上 `__gm__` 为空宏;真机 CCEC 上引擎内所有 GM 指针/段访问需 `__gm__` 限定 | 随 C2 重构给 `DistGlobal`/`DistCore`/段指针加 `__gm__` | CCEC 编译通过 |
| C7 | **GM 段分配 + 发布** | ENG | seam `pto_gm_alloc/publish` 真机分支为 TODO | AICPU 侧 `rtMalloc(RT_MEMORY_HBM)`/`halMemAlloc` + `cache_flush_range` 发布(§16.3) | 真机各核可读到初始化后的段 |
| C8 | **core-id 映射**(低风险,记录在案) | ENG | 逻辑 worker index 已由 `dist_core_main(core_idx)` 入口参数携带(= launch `s_block_idx`),无需读寄存器;物理 id 仅诊断用 | 沿用入口参数;`pto_core_id()` HW 分支返回该入参 | — |
| C9 | **验证缺口:无本地 CCEC/真机** | PROC | ~~本机无 CCEC~~ **修订**：本机 CCEC 可用(`cann-9.1.T500/bin/ccec`)，`build_runtimes --platforms a5` 能出 CCEC aicore 二进制；真机经 `task-submit --device auto` 跑 `vector_example`。仍无本地 npu-smi(缺 `libsecurec.so`，onboard-arch-precheck 需手动确认 a5) | 继续走 `task-submit` + `_run_probe.sh`(自动重编+抓 device log crumbs) | 已可用✅ |
| C10 | ~~ops 表函数指针 incore 重定位~~ **已解除** | ~~BLOCKER~~ 已落地 | **修订**：真机开 `dist_ops_refresh_aicore`+ops 后 `gd->rt->ops->submit_task` 经 `runtime->dist.global_data_base + offsetof(ops)` 的**整数偏移自解析**已可用；`dist_submit_impl` 被正常进入。之前"9 核全 271"的真因是 **ABI 分歧(C13)+`pto_core_id()` 桩(C14)** 联合，与 ops 重定位无关。修 C14 后 9 核 `local_index=5`(各核提交 5 任务)、`core_id` 各不相同(0-8) | 已完成 | 编排能提交任务✅ |
| C11 | **`a2a3sim` 编译回归** | ENG | 共享 `dist_engine.cpp` 引用 `runtime->dist.seg_base/seg_size`、`TensorRef::raw_addr()` —— 这些**仅 a5 定义**(前序会话加的 a5-onboard 段交付逻辑)，a2a3 的 runtime.h 无 → a2a3sim 编译失败 | 给这些 a5-only 字段访问加 `#ifdef`/特征检测隔离，或把段交付逻辑下沉到 a5-only 编译单元 | a2a3sim `Build complete!` |
| C13 | ~~`DistGlobal` ABI 分歧~~ **已解除** | ~~BLOCKER~~ 已落地 | `PTO2_PROFILING` 宏在 AICPU(g++)/AICore(CCEC)不一致 → `L2TaskArgs orch_args_gm` 尺寸差 160B → AICore 读 `gd->rt`=0 → MPU 271。**修复**：把 `orch_args_gm` 移到 `DistGlobal` 末尾，profiling 相关字段不再影响前段 offset | 已完成 | AICore 读到正确 `gd->rt`✅ |
| C14 | ~~编排未在核上执行(`pto_core_id` 桩 + 编排调度错位)~~ **已解除** | ~~BLOCKER~~ 已落地 | 两处联合：(a) `pto_core_id()` HW 分支是恒返回 0 的桩 → 所有核自认 core 0 → SPMD 崩塌；(b) `dist_core_main` onboard 分支**直调弱符号** `aicpu_orchestration_entry`(仅弱声明→链接器解析到镜像基址=空操作)，而真正的编排是 CCEC 编成的**独立 PC-relative blob**(`_compile_orchestration_dist_blob`)，入口在 `gd->orch_func`。**修复**：(a) `pto_set_core_id/pto_core_id` 用 `[[block_local]] static` 存/取入参 `core_idx`；(b) onboard 分支改与 sim 一致——先 `gd->orch_bind_func(gd->rt)` 绑定 blob 自身 `g_current_runtime`，再经 `gd->orch_func(*gd->orch_args)` 调用。移除 `DIST_DIAG_SKIP_ORCH` | 已完成 | 9 核 `local_index=5`、`core_id`=0..8✅ |
| C15 | ~~drain 死锁(flag false-sharing)~~ **已解除** | ~~BLOCKER~~ 已落地 | C14 解除后 `drain` 开始执行 kernel，但 2 核卡 `crumb 33`(非 40/41 → kernel 不挂死、无 DMA 故障)、`done_count=7/9`、507000 超时。诊断证实：卡住核 `occupied_count=1`、`ring_empty=0`，其私有环 slot 的 fanin 生产者 flag 永不置位——**`flags[2]=0` 但结构对称的 `flags[1]=1`**(t1/t2 都是 c+标量、只依赖 t0)。真因 = **C4 false-sharing**：`flags[]` 是 `Coherent<uint8_t>` 稠密环，t0..t4 五个 flag 共用一条 64B cacheline，`coherent_store`+dcci **按 cacheline 刷出会把相邻核刚写的 flag clobber 回 0** | flag 改 `int32_t` + 置位走**内存级 `atomicMax(&flags[i],1)`**(`dist_set_flag`)——直写真实 HBM 字、不回写共享行，相邻置位互不 clobber；读侧仍 `coherent_load` dcci 失效 | **9/9 完成、无 507000/异常✅**(仍差数值，见 C16) |
| C16 | **中间 `TensorCreateInfo` 栈局部地址空间损坏(现总闸)** | BLOCKER | C15 解除后 9 核全部完成、无死锁，但 `f` golden `max_diff≈31~51`。**真因已确诊(非缓存一致性)**：上板 dump 逐张量地址显示中间张量间距仅 `0x400`(1024B=256 float)而非 `0x10000`(65536B)——`buffer_size_bytes()` 用了 `ndims=0`(ALIGN_UP(4,1024)=1024)。进一步 dump `dist_submit_impl` 读到的 create_info：`@0x107c78 ndims=0 shape0=0`。根因 = **用户编排里 `TensorCreateInfo inter_ci` 是 CCEC AICore 的栈局部变量**(栈在 local/workspace 空间，`-cce-aicore-stack-size`)，而 `TensorRef` 在 CCEC 用**整数地址 seam**存指针(`addr_=reinterpret_cast<uintptr_t>(&inter_ci)`)——整数 round-trip **丢失 local 地址空间限定**，重建为 generic 指针后指向错误位置→读出 ndims=0→中间输出被分配成 4B→严重重叠(每张量覆盖后续 63 个,只剩首 256 元素正确)。外部张量(ext_a/ext_f)正常是因为它们 GM 常驻,地址可扁平 round-trip。注释"on-core 一切 GM-resident"的假设**对用户栈 create_info 不成立** | `TensorRef` 对 OUTPUT **按值内嵌** `TensorCreateInfo`(在 `add_output` 时用仍带正确空间限定的指针拷贝,不再存整数地址),`create_info()` 返回内嵌成员(经 `args` 引用做成员访问,空间正确)。host/CCEC 一致内嵌以保 `L2TaskArgs`(orch_args_gm)布局不变 | 真机 `f` golden 通过 |

> 依赖关系:**C1** claim 机制已定(硬件 `atomicMax`)；**C2** 结构拆分完成；**C10/C13/C14/C15 已解除**——编排提交任务、SPMD id 正确、**drain 无死锁、9/9 完成**。**当前总闸转为 C16**(中间张量数据跨核可见性)——最后一步数值正确性。C16 解除后收 C5(done_count)/C11(a2a3 隔离)。落地顺序见 §16.4 的 P0→P4 阶段闸。

### 17.1 本轮上板定位记录（2026-07，MPU 271 收敛）

真机 `vector_example`(device auto，9 workers = 3 AIC + 6 AIV)三次迭代的证据链：

1. **已修两处跨核陈旧指针隐患**（`dist_core_main_impl` / `drain_block_won` / `has_pending_won`）：
   - `self->gd`（`gd->cores[i].gd`）由 AICPU 写、AICore 用**裸指针 load**读（`Coherent<T>` 只包原子，不含此回指针），cacheable HBM 上可能读到上一轮 segment base 的陈旧值 → `gd->blocks[self->block_id]` 野地址。**修复**：在 `dist_core_main_impl` 用本核已验证的 local `gd` 就地 `self->gd = gd`（同核写读，天然一致）。
   - `block_id` 越界防护：`drain_block_won`/`has_pending_won` 进入前判 `0 <= block_id < num_blocks`。
2. **故障随 ops/orch 开关移动**（关键 bisection）：
   - `DIST_DIAG_SKIP_OPS=1 && SKIP_ORCH=1`：故障在 **drain 循环**（crumb 30），**7/9 存活**。
   - `DIST_DIAG_SKIP_OPS=0 && SKIP_ORCH=0`：故障前移到 **ops 表/编排回放**（crumb 26-27），**9/9 全 271**，pc 收敛到 `0x2600`/`0x2624`。
   - 结论：真正的元凶在 **`dist_ops_refresh_aicore` + `rt->ops` 调用路径**（C10），与 cursor 认领、读一致性无关。
3. **本轮正确性改动**（cursor `atomicMax`、`coherent_load/store` 补 dcci、`self->gd`、`block_id` 防护）都是必要且正确的，把故障从 drain 路径清掉；但它们不是 C10 的成因。
4. **复现命令**：`task-submit --device auto --max-time 260 --run "bash tests/st/a5/fully_distributed_within_core/vector_example/_run_probe.sh"`；device log crumbs 在 `~/ascend/log/debug/device-*/`，`progress=[...]` 按 core_idx、`dbg=[...]` 为各核读到的诊断包。诊断开关在 `dist_engine.cpp` 顶部 `DIST_DIAG_SKIP_ORCH/BIND/OPS`（当前全 0 = 完整路径）。

### 17.2 编排提交打通里程碑（2026-07，C13/C14 解除）

`vector_example`（device auto，9 workers）关键突破的证据链：

1. **C13（ABI 分歧）** 修复后，AICore 不再 271，能读到正确 `gd->rt`；但 `local_index` 全 0（无任务提交）。
2. **`pto_core_id()` 是桩**：HW 分支恒返回 0（带 TODO）→ 所有核自认 core 0，`dist_submit_impl`/`dist_is_fatal` 全走 core 0 视角，SPMD 崩塌。改用 `[[block_local]] static int32_t` 存 `dist_core_main(core_idx)` 入参后修复。
3. **编排调度错位（C14 核心）**：onboard `dist_core_main` 直调 `aicpu_orchestration_entry`——但该符号在通用运行时镜像里只有**弱声明**，链接器解析到镜像基址 `0x…be00000`（探针 `dbg99` bits0-31 实测），等于**空操作**；真正的编排是 CCEC 编成的**独立 PC-relative blob**（`kernel_compiler._compile_orchestration_dist_blob`），入口存 `gd->orch_func`（探针实测 `0x…0001d04c`，落在 DevMalloc 上传区）。`DIST_DIAG_SKIP_ORCH=1` 又把直调整个跳过（双重失效）。
4. **修复**：onboard 分支改与 sim 一致，先 `gd->orch_bind_func(gd->rt)` 绑定 blob 自身 `g_current_runtime`，再 `gd->orch_func(*gd->orch_args)`；删除 `DIST_DIAG_SKIP_ORCH`。（§16.1 的"编排编进镜像/直调符号"方案构建侧尚未接通，故沿用 blob 指针调度——与 kernel blob 一致。）
5. **结果**：9 核 `dbg=[0x50k0001d04c]` 解码 = `local_index=5`（每核提交 5 任务）+ `core_id`=0..8（各不相同）。**编排首次在核上真正提交任务、SPMD id 正确**。随后暴露 C15（kernel 执行/DMA MTE 异常），成为新总闸。

## 18. 相关文档

| 文档 | 关联性 |
| ---- | ------ |
| [chip-level-arch.md](chip-level-arch.md) | 当前 L2 host / AICPU / AICore 划分（本设计所替代的模型） |
| [scheduler.md](scheduler.md) | 当前 AICPU 侧调度器（此处移除） |
| [orchestrator.md](orchestrator.md) | Host/L3 Orchestrator DAG 构建器（不同层；仅命名重叠） |
| [simt-launch.md](simt-launch.md) | 设备上的 SPMD / 多 block 启动 |
| [tensormap_and_ringbuffer RUNTIME_LOGIC.md](../src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md) | 此处移除/修改结构的权威来源 |
