# 把 percounter 拆成第三个 runtime

## 目标

percounter 现在是 a5 `host_build_graph` 里的调度模式之一（`SCHEDULER_RUNTIME_MODE_PERCOUNTER = 5`）。
把它变成 `src/a5/runtime/percounter/`，和 `host_build_graph`、`tensormap_and_ringbuffer` 平级。

三条约束，按优先级：

1. **percounter 的后续演进不再落进 hbg 的文件里。**
2. **不复制 hbg 的整个文件再改自己的版本**（下称 fork）。
3. **这个分支要能合进上游主干。**

约束 2 和 3 互相支撑：fork 一个文件就等于背上它在上游的全部后续改动。

## 现状

### 体量

a5 hbg 除文档外约 22,300 行，其中 percounter 专属 3,877 行（17%），分在四个子目录：

| 目录 | 行数 |
| ---- | ---: |
| `aicore/percounter/` | 1,426 |
| `aicpu/percounter/` | 472 |
| `host/percounter/` | 1,288 |
| `runtime/scheduler/percounter/` | 691 |

**这八个文件全是头文件，没有一个 `.cpp`。** 这是整个方案成立的前提。

### percounter 改过的 hbg 文件

从分叉点算起共 7 个代码文件、约 93 行：

| 文件 | percounter 改动 | 上游自分叉以来的改动 |
| ---- | --------------: | -------------------: |
| `host/runtime_maker.cpp` | +24 | 4 commits，+509 −193 |
| `aicpu/aicpu_executor.cpp` | 40 | 5 commits，+161 −66 |
| `aicore/aicore_executor.cpp` | +5 | 3 commits，+297 −241 |
| `runtime/scheduler/scheduler_layout.h` | +3 | 2 commits，+25 −5 |
| `aicpu/aicore_lifecycle.h` | +7 | 无 |
| `runtime/scheduler/scheduler_topology.h` | 1 | 无 |
| `src/common/host_build_graph/runtime.h` | 12 | 无 |

**上游正在重写的，恰好是 percounter 打补丁的那四个文件。** 今天的试合并已经冲突：

```text
git merge-tree --write-tree --name-only upstream/main <percounter 分支>
CONFLICT (content): src/a5/runtime/host_build_graph/host/runtime_maker.cpp
CONFLICT (content): tests/ut/cpp/a5/runtime/host_build_graph/CMakeLists.txt
```

冲突落在 `create_scheduler_state` 的函数体内，正是上游加了 509 行的地方。

### 构建系统

runtime 是**按目录自动发现**的，加一个目录不需要改构建系统：

```python
def discover_runtimes(arch: str) -> list[str]:
    """Return sorted list of runtime names that have a build_config.py."""
    return sorted(d.name for d in runtime_base.iterdir()
                  if d.is_dir() and (d / "build_config.py").exists())
```

`build_config.py` 的 `source_dirs` / `include_dirs` 支持相对路径逃出自己的目录——hbg 自己就在用
`SHARED = "../../../common/host_build_graph"`。

## 关键机制：weak 符号覆盖

`__attribute__((weak))` 是链接器规则：同名符号存在 strong 定义时，weak 的那个被丢弃。
仓库里已经在用：

```cpp
// aicore/aicore_executor.cpp
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type)
```

于是：

- **hbg runtime 编译时**：只有这一个定义，weak 即最终定义，行为不变。
- **percounter runtime 编译时**：源文件里同时有 hbg 的 weak 版本和 percounter 目录下的 strong 版本，
  链接器选 strong。hbg 那份连调用都到不了。

两个 runtime 编出各自独立的三件套，互不影响。

## percounter 是无条件的

新 runtime 里**不存在模式选择**——它一定是 percounter。这一条大幅缩小了 host 侧的工作量。
看 `create_scheduler_state` 那 450 行实际在做什么：

| 段落 | 无条件 percounter 是否需要 |
| ---- | -------------------------- |
| 尺寸校验 | 需要 |
| GRAPH 任务预扫 → 回落 legacy | 不需要 |
| 逐任务 `scheduler_classify_task_shape` → 填 `SchedulerTaskMetadata` | **需要** |
| `inline_dispatch_task`（legacy 内联回落） | 不需要 |
| `legacy_shape_task_id` / `scheduler_resident_v0_task_shape_supported` | 不需要 |
| 模式选择、resident 布局规划与上传 | 不需要 |

而"需要"的那部分所依赖的两个函数**本来就在共享头里**，percounter 已经 include 了它们：

- `scheduler_classify_task_shape()` —— `scheduler/scheduler_graph.h`
- `scheduler_task_metadata_flags_from_submit_state()` —— `scheduler/scheduler_types.h`

所以 percounter 写自己的 host 入口不是重写难的部分：形状分类规则在共享头里，自己写的是一个
约 80 行的循环，把分类结果填进 metadata。percounter 另有一套自己的形状校验
（`percounter_graph.h` / `percounter_plan.h`），现在是重复校验，正好收归一处。

**结论：不 fork 任何文件。**

## hbg 的最终残留

| 文件 | 残留 | 性质 |
| ---- | ---: | ---- |
| `aicore/aicore_executor.cpp` | **0 行** | 现有 5 行删除，靠已有的 weak 覆盖 |
| `aicpu/aicpu_executor.cpp` | **1 行** | 给 `aicpu_execute` 加 `weak` |
| `host/runtime_maker.cpp` | **1 行** | 给 `create_scheduler_state` 加 `weak` |
| `runtime/scheduler/scheduler_layout.h` | **0 行** | 模式常量移入 percounter 自己的头 |
| `aicpu/aicore_lifecycle.h` | 7 行 | 三个 accessor；上游从未改动此文件 |
| `runtime/scheduler/scheduler_topology.h` | 1 行 | 加 `__aicore__`；上游从未改动此文件 |
| `src/common/host_build_graph/runtime.h` | **0 行** | 只改过注释，回退 |

合计约 10 行，**没有一处出现 `percounter` 字样，没有一处落在上游正在重写的函数体内**。

那两行 `weak` 做不到零：percounter 的构建会同时编进 hbg 的 `runtime_maker.cpp`
（其中 2,600 行 host 初始化要复用）和 percounter 自己的 `create_scheduler_state`，
两个 strong 定义就是重复符号。要么 hbg 那个标 weak（1 行），要么 fork 整个文件（2,600 行）。

## 分步计划

每一步都能独立停下，并且每一步的验收都包含一次对上游的试合并。

### 第 1 步：零改动的第三个 runtime

只新增 `src/a5/runtime/percounter/build_config.py`，`source_dirs` 全部指向 `../host_build_graph`，
percounter 目录本身为空。产出与 hbg 等价的第三套二进制。

**不碰 hbg 任何文件。**

验收：

- `discover_runtimes("a5")` 返回三个名字
- 三套二进制都能编出来
- `git merge-tree upstream/main HEAD` 的冲突数不增加

做完这一步就已经拿到最大的好处：**percounter 的实验不再和 hbg 共用一个二进制。**

### 第 2 步：搬空 hbg

把四个 percounter 子目录整体移到 `src/a5/runtime/percounter/` 下的对应位置，把模式常量移入
percounter 自己的头，并**在同一步删掉 hbg 对它们的全部引用**——头文件一旦移走，hbg 的
`#include` 就失效，所以搬运和拆接缝必须同时做，不能像本页最初写的那样分成两步。

同样移走的还有那四个 cpput 单元测试：它们测的是 percounter 的主机侧规划，却寄居在
`tests/ut/cpp/a5/runtime/host_build_graph/` 里，而那个 `CMakeLists.txt` 正是现存两个
上游冲突之一。

做完这一步，**hbg 干净了，而 percounter 还没接上**——它的八个头躺在新树里没人 include，
percounter runtime 编出来和 hbg 等价。这是一个可验证的中间态。

验收：

- hbg 树内 `grep -ri percounter` 为空
- 两套二进制都能编出来
- `git merge-tree upstream/main HEAD` 的冲突不增加，并且测试 CMakeLists 那一个消失

### 第 3 步：接上三个入口

| 入口 | 做法 |
| ---- | ---- |
| `aicore_execute` | hbg 的已经是 weak；percounter 提供 strong 定义 |
| `aicpu_execute` | hbg 侧加 `weak`；percounter 提供 strong 定义 |
| `create_scheduler_state` | hbg 侧加 `weak`；percounter 提供约 80 行的自己的版本 |

验收：

- hbg 树内只剩 `aicore_lifecycle.h` / `scheduler_topology.h` 的那 8 行，外加两处 `weak` 标记
- a5 hbg 的 st 全绿（证明 hbg 行为未变）
- percounter runtime 能跑通至少一个 DeepSeek 模块，数字与拆分前一致

### 第 4 步：收紧 percounter 内部

无条件化之后可做的清理，不改变行为：

- 模式常量退化为固定值
- 去掉 `SIMPLER_EXP_FORCE_SCHED` 一类的选择开关（它们属于旧的三选一世界）
- percounter 自己的形状校验与第 3 步新写的分类循环合并，去掉重复

## 这个分支不做的事

- **不加 st 测试树。** `tests/st/{arch}/{runtime}/` 的结构意味着要复制 21 个 case。
  本分支先不建；hbg 的 st 继续覆盖 hbg 的行为。
- **不给 a2a3 加 percounter。** 只有 a5 上有数据，而 a2a3 与 a5 两棵树已经是近似重复
  （见 `.claude/rules/codestyle.md` 第 10 条），再加一棵就是三倍维护成本。
- **不向上游单独提那三处通用改动。** 它们留在本分支。

## 后续：pypto / pypto-lib 侧

simpler 这边是现成的——runtime 是一个直接对应目录名的字符串：

```python
w = Worker(level=2, device_id=8, platform="a5", runtime="percounter")
```

`discover_runtimes()` 扫目录，所以 Python API 一行都不用改。

pypto 那边需要一个小改动。它的 runtime 选择是 `pypto.ir.passes.RuntimeKind`，驱动的是**代码生成**，
产物按 runtime 打标签、互不通用。而 **percounter 消费的设备图与 hbg 完全相同**——同样的
`ChipTaskSlotState`、同样的 task kind、同样的 orchestration `.so`。percounter 换的是调度器，
不是程序格式；这也正是它今天能作为 hbg 的模式 5 存在而 pypto 一行未改的原因。

所以要向 pypto 提的是：**把"用哪种 codegen"和"加载 simpler 的哪个 runtime"拆成两个轴。**
建议按"新增一个独立的 `RuntimeKind`、当前复用 HBG 的 codegen pass"来提，而不是只加一个
runtime 名覆盖——因为一旦 percounter 的设备镜像将来与 hbg 分化（例如把顺序表放进镜像），
codegen 就真的不同了，独立的 `RuntimeKind` 给这件事留了口子。

## 什么时候该放弃复用

一个明确的信号：**当第 3 步之后仍需 fork 的文件超过 hbg 的一半时**，复用已经不划算，
那时再整树复制，而且那时已经知道哪些可以删。
