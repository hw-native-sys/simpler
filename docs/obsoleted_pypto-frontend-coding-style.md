# PyPTO Frontend Coding Style and Grammar

本文档描述 PyPTO 前端的语法与编码风格：如何定义 InCore 函数、编排（Orchestration）函数、InCore 作用域与匿名 InCore、参数类型，以及编译器生成的输出格式（编排 C++、InCore 的 .pto 汇编），并说明 **pipev / set_wait** 在 PTO 汇编中的情况。

---

## 1. 模块与程序结构

```python
# pypto.program: program_name   # 可选：命名程序
import pypto.language as pl
```

- 未命名程序可写：`# pypto.program`
- 模块前缀默认 `pl`，也可配置为 `ir` 或自定义

---

## 2. 类型系统（参数与返回值）

### 2.1 标量类型

| 类别   | 类型示例 |
|--------|----------|
| 整数   | `pl.INT8`, `pl.INT16`, `pl.INT32`, `pl.INT64` |
| 无符号 | `pl.UINT8`, `pl.UINT16`, `pl.UINT32`, `pl.UINT64` |
| 浮点   | `pl.FP16`, `pl.FP32`, `pl.BF16` |
| 布尔   | `pl.BOOL` |

用法示例：`x: pl.INT64`、`scale: pl.Scalar[pl.FP32]`

### 2.2 张量与 Tile 类型

```python
# 张量：形状 + 元素类型
a: pl.Tensor[[4, 8], pl.FP32]        # 固定形状
b: pl.Tensor[[n, m], pl.INT64]       # 符号形状（如来自 closure）

# Tile：块状数据，用于 InCore 的 load/compute/store
t: pl.Tile[[16, 16], pl.FP16]
```

### 2.3 参数方向（In / Out / InOut）

| 方向 | 写法 | 说明 |
|------|------|------|
| In   | 不加包装（默认） | 只读输入 |
| Out  | `pl.Out[type]` | 只写输出 |
| InOut| `pl.InOut[type]` | 读写 |

约束：**Scalar 参数不能为 InOut**（会抛出 `ParserTypeError`）。

```python
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],              # In
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],  # Out
    acc: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],   # InOut
    scale: pl.Scalar[pl.FP32],                      # In
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

---

## 3. InCore 函数

### 3.1 定义方式

- 用 `@pl.function(type=pl.FunctionType.InCore)` 标记为 InCore。
- 在 `@pl.program` 类中作为方法定义时，第一个参数为 `self`（解析后不会出现在 IR 函数签名中）。

```python
@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return out
```

### 3.2 典型模式：load → compute → store

- `pl.load(tensor, [row, col], [height, width])` → 得到 `Tile`
- 块运算：`pl.add`, `pl.mul`, `pl.sub`, `pl.exp`, `pl.row_max`, `pl.row_sum` 等
- `pl.store(tile, [row, col], [height, width], tensor)` → 写回张量

### 3.3 内存空间（可选）

- 使用 `pl.load(..., target_memory=pl.MemorySpace.Mat)`、`pl.move(..., target_memory=pl.MemorySpace.Left)` 等可指定目标内存（Mat/Vec/Left/Right/Acc）。
- CUBE 流程常见：L1 → Left/Right → matmul → L0C → store。

---

## 4. 编排（Orchestration）函数

### 4.1 定义方式

- 用 `@pl.function(type=pl.FunctionType.Orchestration)` 标记为编排函数。
- 编排函数运行在 Host/AICPU，负责控制流和调用 InCore 内核。

```python
@pl.function(type=pl.FunctionType.Orchestration)
def BuildExampleGraph(
    self,
    a: pl.Tensor[[16, 16], pl.FP32],
    b: pl.Tensor[[16, 16], pl.FP32],
) -> pl.Tensor[[16, 16], pl.FP32]:
    c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
    c = self.kernel_add(a, b, c)
    d: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
    d = self.kernel_add_scalar(c, 1.0, d)
    # ... 更多 task，形成 DAG
    return f_result
```

### 4.2 编排中常用构造

- `pl.create_tensor(shape, dtype=...)`：分配中间张量。
- `self.kernel_xxx(...)`：调用同 Program 内的 InCore 函数（解析为 `ir.Call(GlobalVar, args)`）。
- `pl.range(start, stop, step)`、`pl.range(..., init_values=(...))`：循环与 iter_args。
- `pl.tensor.read(tensor, [index])`：读标量（如 config、context_lens）。
- `pl.view(tensor, shape, [offset_row, offset_col])`：张量视图。

---

## 5. InCore 作用域与匿名 InCore（with pl.incore()）

### 5.1 语法

在 **Opaque** 函数内用 `with pl.incore():` 标记一段“匿名” InCore 区域；解析后生成 `ScopeStmt(scope_type=InCore)`。

```python
@pl.program
class Before:
    @pl.function   # 默认 Opaque
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = x + 1
        with pl.incore():
            tile = pl.load(y, [0], [64])
            tile_sq = pl.mul(tile, tile)
            result = pl.store(tile_sq, [0], [64], x)
        z = result + 2
        return z
```

### 5.2 OutlineIncoreScopes 变换

- Pass **OutlineIncoreScopes** 会把每个 `ScopeStmt(InCore)` 抽取成独立的 **具名 InCore 函数**（如 `main_incore_0`），并把原作用域替换为对该函数的 **Call** + 对返回值的赋值。
- 输入/输出由 SSA 分析得到：进入 scope 前已定义的变量为输入，scope 内定义且在 scope 外使用的为输出。
- 因此：**匿名 InCore 在编译中会变成具名 InCore 函数**，最终代码生成按“InCore 函数”处理，而不是保留为内联的“匿名块”。

---

## 6. 函数类型汇总

| 类型 | 写法 | 用途 |
|------|------|------|
| Opaque | 默认 / `pl.FunctionType.Opaque` | 未指定，可含 `pl.incore()` 待 outline |
| Orchestration | `pl.FunctionType.Orchestration` | Host/AICPU 编排，调用 InCore |
| InCore | `pl.FunctionType.InCore` | AICore 上的子图（load/compute/store） |

---

## 7. 编译器输出概览

### 7.1 总体管线

- **Pass 阶段**：ConvertToSSA → FlattenCallExpr → InitMemRef → BasicMemoryReuse → (可选) OutlineIncoreScopes → InsertSync（CCE）等 → 分配地址等。
- **代码生成**：
  - **编排函数** → 生成 **C++**（.cpp），使用 PTO2 运行时 API。
  - **InCore 函数** → 视后端：
    - **PTO 后端**：PyPTO **PTOCodegen** 生成 **.pto**（PTO-ISA MLIR），再经 **ptoas** 编译为 C++，最后套一层 kernel 包装。
    - **CCE 后端**：**CCECodegen** 直接生成 **C++**（含 TLOAD/TSTORE/TADD/… 及 set_flag/wait_flag）。

### 7.2 编排函数输出：.cpp 格式

- **位置**：`output_dir/orchestration/<orch_func_name>.cpp`
- **内容**：C++ 源码，使用 **PTO2 运行时（pto2_rt）** 的 API。以下为 codegen 实际用到的接口及用途。

#### PTO2 运行时 API 一览

| API / 宏 | 用途 |
|----------|------|
| **PTO2OrchestrationConfig** | 编排配置结构体，由 `aicpu_orchestration_config` 返回，供运行时解析 expected_arg_count 等。 |
| **aicpu_orchestration_config(uint64_t* args, int arg_count)** | 返回编排配置；生成的 C++ 中仅设置 `expected_arg_count`，args/arg_count 未使用。 |
| **aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count)** | 编排入口：从 `args` 解出设备指针与返回值槽位，声明外部/中间张量，按控制流提交任务。 |
| **PTO2_SCOPE(rt)** | **Scope begin/end**：宏形式，用于在 C++ 中包裹一段“作用域”（如 for 循环体、if/else 分支）。运行时可在 scope 内做任务依赖与资源管理；codegen 在**每个 for 体**和**每个 if/else 分支**外自动生成 `PTO2_SCOPE(rt) { ... }`，无需用户手写。 |
| **make_tensor_external(ptr, shapes, ndim, dtype)** | 声明**外部**张量（入参或返回值槽位），绑定设备指针与 shape/dtype，得到 `Tensor ext_xxx`。 |
| **make_tensor(shapes, ndim, dtype)** | 声明**中间**张量（由 `tensor.create` 产生），在设备侧分配，得到 `Tensor xxx`。 |
| **PTOParam** | 任务参数数组类型；每个元素由下述 make_*_param 构造。 |
| **make_input_param(Tensor)** | 将张量标记为任务**输入**参数。 |
| **make_output_param(Tensor)** | 将张量标记为任务**输出**参数。 |
| **make_inout_param(Tensor)** | 将张量标记为任务**输入/输出**参数。 |
| **make_scalar_param(uint64_t)** | 标量参数（整型或经 `float_to_u64` 编码的浮点）。 |
| **pto2_rt_submit_task(rt, func_id, worker, params, n)** | 向运行时提交一条任务：`func_id` 为内核 ID，`worker` 为 `PTO2_WORKER_VECTOR` 或 `PTO2_WORKER_CUBE`，`params` 为 PTOParam 数组，`n` 为参数个数。 |
| **PTO2_WORKER_VECTOR** / **PTO2_WORKER_CUBE** | 内核执行的 worker 类型（由 InCore 内算子 pipe 推断）。 |
| **Tensor** | 运行时张量句柄类型（与 make_tensor_external / make_tensor 配合）。 |
| **float_to_u64(float)** | 辅助函数：将 float 编码为 uint64_t 以通过 make_scalar_param 传递。 |

- **依赖**：由编排逻辑对 InCore 的调用关系生成任务图，**运行时负责依赖**，不在此 C++ 里手写 set_flag/wait_flag。

### 7.3 InCore 函数输出（PTO 后端）：.pto 与 .cpp

- **.pto 文件**：每个 InCore 函数对应 `output_dir/ptoas/<func_name>.pto`，为 **PTO-ISA 方言的 MLIR**（“PTO 汇编”）。
- **.pto 再经 ptoas** 生成 `ptoas/<func_name>.cpp`，再与 **kernel 包装** 合并为最终 `kernels/aiv/<func_name>.cpp`（与 CCE 调用约定兼容的入口 `kernel_entry(__gm__ int64_t* args)`）。

### 7.4 双后端支持与主路径

PyPTO 代码生成支持两种后端：

| 后端 | 说明 | InCore 输出 | 编排输出 |
|------|------|-------------|----------|
| **PTO** | PTO 汇编代码生成：IR → PTOCodegen → .pto → ptoas → C++，再套 kernel 包装 | .pto（PTO-ISA MLIR）→ ptoas → .cpp | 与 CCE 共用同一套编排 C++ 生成（PTO2 运行时 API） |
| **CCE** | CCE 代码生成：IR → CCECodegen 直接生成 pto-isa 风格 C++（TLOAD/TSTORE/TADD/… 及 set_flag/wait_flag） | .cpp（内核 C++） | 同上 |

- **主路径**：`ir.compile(..., backend_type=...)` 的**默认**为 `BackendType.PTO`，即 **PTO 后端为主路径**。CCE 后端为可选，用于直接生成 C++ 内核、且由 InsertSync 等 pass 插入同步的流水线。
- 编排层不区分后端：无论 PTO 还是 CCE，编排函数均使用同一套 **orchestration codegen**（PTO2 运行时 API），生成 `orchestration/<orch_func_name>.cpp`。

---

## 8. PTO 汇编（.pto）格式与语法教程

### 8.1 生成顺序（PTOCodegen 固定顺序）

1. **常量**：`arith.constant`（index / 浮点）
2. **张量视图**：`pto.make_tensor_view`（为每个张量参数建 view）
3. **分配**：`pto.alloc_tile`（基于 MemRef 的 tile 缓冲区）
4. **函数体**：load / 计算 / store

### 8.2 类型与地址空间

- **张量参数**：MLIR 中为 `!pto.ptr<dtype>`，通过 `pto.make_tensor_view` 得到 `!pto.tensor_view<?x?xdtype>`。
- **Tile 缓冲区**：`!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`，其中 `loc` 来自 `MemorySpace`（如 `vec`, `mat`, `left`, `right`, `acc`, `gm`）。

### 8.3 典型指令对应（DSL → .pto）

| DSL（PyPTO） | PTO-ISA MLIR（.pto） |
|--------------|----------------------|
| `pl.load(tensor, [r,c], [h,w])` | `pto.partition_view` + `pto.tload` |
| `pl.store(tile, [r,c], [h,w], tensor)` | `pto.partition_view` + `pto.tstore` |
| `pl.mul(tile_a, tile_b)` | `pto.tmul` |
| `pl.add(a, b)`（tile） | `pto.taddc`（三目）或 `pto.tadds`（tile+scalar） |
| `pl.exp(tile)` 等 | 对应 unary/binary 的 pto.* |

### 8.4 示例：完整 .pto 片段（mul_kernel）

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    %3 = pto.make_tensor_view %arg0, shape = [%c32, %c32], strides = [%c32, %c1]
         : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32, %c32], strides = [%c32, %c1]
         : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32, %c32], strides = [%c32, %c1]
         : !pto.tensor_view<?x?xf32>

    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    %6 = pto.partition_view %3, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>) outs(%0 : !pto.tile_buf<...>)

    %7 = pto.partition_view %4, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>) outs(%1 : !pto.tile_buf<...>)

    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>) outs(%2 : !pto.tile_buf<...>)

    %8 = pto.partition_view %5, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>) outs(%8 : !pto.partition_tensor_view<32x32xf32>)

    return
  }
}
```

- 语法要点：`ins(...)` / `outs(...)` 带类型、`partition_view` 表示一块矩形区域、`tload`/`tstore` 为块搬运、`tmul` 等为块运算。

### 8.5 PTO_ISA 指令集与 PyPTO 前端基础函数对应关系

PyPTO 前端（DSL/IR）的**块级算子**（如 `pl.load`、`pl.store`、`pl.mul`）在 PTO 后端会映射到 **PTO-ISA 方言** 的指令名（如 `pto.tload`、`pto.tstore`、`pto.tmul`）。该映射由 **Backend** 的算子注册表维护（例如 910B PTO 的 `src/backend/910B_PTO/backend_910b_pto_ops.cpp`）：每个 IR 算子名对应一个 `pto_op_name` 和一段 codegen 回调。

#### 主要对应关系（节选，IR 算子名 → PTO-ISA 指令）

| PyPTO 前端（IR / pl.*） | PTO-ISA（.pto 中） |
|-------------------------|---------------------|
| `block.load` | `pto.partition_view` + `pto.tload` |
| `block.store` / `block.l0c_store` | `pto.partition_view` + `pto.tstore` |
| `block.add` | `pto.tadd` |
| `block.sub` / `block.mul` / `block.div` | `pto.tsub` / `pto.tmul` / `pto.tdiv` |
| `block.adds` / `block.subs` / `block.muls` / `block.divs` | `pto.tadds` / `pto.tsubs` / `pto.tmuls` / `pto.tdivs` |
| `block.addc`（三目） | `pto.taddc` |
| `block.exp` / `block.log` / `block.sqrt` / `block.relu` | `pto.texp` / `pto.tlog` / `pto.tsqrt` / `pto.trelu` |
| `block.row_sum` / `block.row_max` / `block.row_expand_div` 等 | `pto.trowsum` / `pto.trowmax` / `pto.trowexpanddiv` 等 |
| `block.matmul` / `block.move` | `pto.tmatmul` / `pto.tmov` |
| `block.cast` / `block.cmp` / `block.full` | `pto.tcvt` / `pto.tcmp` / `pto.texpands` |
| 其他 block.* | 见 Backend 表中 `op_name` → `pto_op_name` |

（上表仅作示例；完整映射以各 Backend 的 `kSimpleOps` 与 `REGISTER_BACKEND_OP` 为准。）

#### 是否靠人工维护一致性

**是，需要人工维护。** 当前设计下：

- **IR 侧**：块算子在 `src/ir/op/block_ops/` 等处以 `REGISTER_OP("block.xxx")` 注册，与 DSL 的 `pl.xxx` 对应。
- **PTO 侧**：每个要在 PTO 后端生成 .pto 的算子，必须在对应 Backend（如 910B_PTO）中注册一条 **BackendOpInfo**：IR 名 `op_name`、PTO 指令名 `pto_op_name`、以及可选的 `codegen_func`（复杂 op 如 load/store 有单独实现）。
- 新增或修改 IR 算子、或 PTO-ISA 指令集变更时，需要**人工**在 Backend 的 ops 表中增改条目并实现/调整 codegen，否则 PTOCodegen 会因 `GetOpInfo(op_name)` 为空而报错。**没有从 PTO-ISA 定义自动生成映射的机制**，两边的命名与语义一致性依赖人工保证。

### 8.6 PTO ISA 指令如何暴露给前端供程序员调用

每一条 PTO ISA 指令（如 `pto.tload`、`pto.tmul`）在 PyPTO 中对应一个 **IR 算子**（如 `block.load`、`block.mul`）。前端程序员**不直接写** PTO 指令名，而是通过以下两种方式之一调用，由解析器生成 `block.*` Call，再经 Backend 的 codegen 生成对应 `.pto` 指令。

| 暴露方式 | 写法示例 | 说明 |
|----------|----------|------|
| **Promoted（仅 block 有）** | `pl.load(...)`、`pl.store(...)`、`pl.move(...)`、`pl.create_tile(...)`、`pl.l0c_store(...)` | 这些操作只有块级语义，直接以 `pl.xxx` 暴露，解析为 `block.load` / `block.store` / `block.move` / `block.create_tile` / `block.l0c_store`。 |
| **Unified dispatch（按类型分发）** | `pl.add(a, b)`、`pl.mul(a, b)`、`pl.exp(x)`、`pl.matmul(a, b)`、`pl.row_max(x, tmp)` 等 | 首参为 `Tile` 时走 block 路径，为 `Tensor` 时走 tensor 路径。解析时根据首参类型选择 `block.add` / `tensor.add` 等。 |

**说明**：`pl.block.xxx` 显式命名空间已过时，**不再支持**；块级操作请统一使用 `pl.xxx`（promoted 或 unified dispatch）。

**解析链**：程序员写 `pl.load(tensor, [0,0], [32,32])` → AST 解析为 `Call(op=block.load, args=[...])` → Backend 的 codegen 生成 `pto.partition_view` + `pto.tload`。因此：**每一种 PTO ISA 指令都是通过“前端 API（pl.xxx）→ IR 算子（block.xxx）→ Backend 表（op_name → pto_op_name + codegen）”这条链暴露的**，程序员只面对 `pl`，不面对 `pto.*` 指令名。

### 8.7 当前 PTO ISA 函数集合（910B PTO Backend）

以下为当前 **910B PTO** Backend 中注册的、会生成 PTO-ISA 的**完整 IR 算子集合**（即前端可用的、会落到 .pto 的“ISA 函数”）。格式：**前端调用（pl.xxx）→ IR 名 → PTO-ISA 指令**。块级入口统一用 `pl.xxx`，**不再支持** `pl.block.xxx`。

- **访存 / 搬运**  
  `pl.load` → `block.load` → `pto.partition_view` + `pto.tload`  
  `pl.store` → `block.store` → `pto.partition_view` + `pto.tstore`  
  `pl.l0c_store` → `block.l0c_store` → `pto.partition_view` + `pto.tstore`  
  `pl.move` → `block.move` → `pto.tmov`  
  `block.move_fp`（仅 IR/后端）→ `pto.tmov.fp`  
  `block.alloc`（仅 IR/后端）→ 不直接生成指令（alloc_tile 由 MemRef 统一生成）

- **Tile 算术（二目/三目/标量）**  
  `block.add/sub/mul/div/rem` → `pto.tadd` / `pto.tsub` / `pto.tmul` / `pto.tdiv` / `pto.trem`  
  `block.and/or/xor/shl/shr` → `pto.tand` / `pto.tor` / `pto.txor` / `pto.tshl` / `pto.tshr`  
  `block.adds/subs/muls/divs/rems`、`ands/ors/xors/shls/shrs`、`maxs/mins`、`lrelu` → `pto.tadds` / `pto.tsubs` / … / `pto.tlrelu`  
  `block.addc/subc/sel`、`addsc/subsc/selc` → `pto.taddc` / `pto.tsubc` / `pto.tsel` / `pto.taddsc` / …

- **一元 / 比较 / 扩展**  
  `block.abs/exp/log/sqrt/rsqrt/recip/neg/not/relu` → `pto.tabs` / `pto.texp` / `pto.tlog` / `pto.tsqrt` / `pto.trsqrt` / `pto.trecip` / `pto.tneg` / `pto.tnot` / `pto.trelu`  
  `block.maximum/minimum/prelu` → `pto.tmax` / `pto.tmin` / `pto.tprelu`  
  `block.cmp` / `block.cmps` → `pto.tcmp` / `pto.tcmps`  
  `block.cast` / `block.full` → `pto.tcvt` / `pto.texpands`  

- **归约 / 轴扩展**  
  `block.row_sum/row_max/row_min`、`block.col_sum/col_max/col_min` → `pto.trowsum` / `pto.trowmax` / … / `pto.tcolmin`  
  `block.row_expand` / `block.col_expand`、`block.row_expand_div` / `row_expand_mul` / `row_expand_sub`、`block.col_expand_mul` / … → `pto.trowexpand` / `pto.tcolexpand` / `pto.trowexpanddiv` / …

- **矩阵 / 数据搬与工具**  
  `block.matmul` / `matmul_mx` / `matmul_mx_acc` / `matmul_mx_bias` / `matmul_acc` / `matmul_bias` → `pto.tmatmul` / `pto.tmatmul.mx` / …  
  `block.gemv` / `gemv_acc` / `gemv_bias` → `pto.tgemv` / `pto.tgemv.acc` / …  
  `block.transpose` / `block.extract` / `block.reshape` → `pto.ttrans` / `pto.textract` / `pto.treshape`  
  `block.gather` / `block.gatherb` / `block.scatter`、`block.mgather` / `block.mscatter` → `pto.tgather` / … / `pto.tmgather` / `pto.tmscatter`  
  `block.getval` / `block.setval`、`block.fillpad`、`block.assign`、`block.store_fp`、`block.ci`、`block.partadd` / `partmax` / `partmin`、`block.sort32` / `block.mrgsort`、`block.print` → 对应 `pto.tgetval` / `pto.tsetval` / `pto.tfillpad` / `pto.tassign` / `pto.tstore.fp` / `pto.tci` / …

（完整一一对应以 `src/backend/910B_PTO/backend_910b_pto_ops.cpp` 中 `kSimpleOps` 与 `REGISTER_BACKEND_OP` 为准；上表覆盖当前主要类别。）

### 8.8 前端 Tensor 与 Tile：InCore 中 PTO 参数的基础数据类型与转换

#### 前端定义的数据类型

- **Tensor**（`pl.Tensor[[...], dtype]`）：逻辑上是 **N 维、带形状与 dtype 的张量**，在设备上通常对应 **DDR 上的一块连续或跨步区域**。InCore 函数的**参数**可以是 Tensor（如 `a: pl.Tensor[[64,128], pl.FP16]`）。
- **Tile**（`pl.Tile[[rows, cols], dtype]`）：逻辑上是 **块状数据**，位于 **Unified Buffer (UB)**，并带有**内存空间**（Vec / Mat / Left / Right / Acc）。InCore 函数**内部**通过 `pl.load`、`pl.create_tile`、`pl.move` 等得到的是 Tile，不能作为函数参数类型（参数只能是 Tensor 或 Scalar）。

#### InCore 中作为 PTO ISA 参数的基础数据（.pto 侧）

在生成的 .pto 里，**不是**直接出现“matrix tile / vector tile”等类型名，而是用 **`!pto.tile_buf`** 的 **`loc`** 属性区分不同缓冲区：

| PTO 中的表示 | 含义 | 前端对应 |
|--------------|------|----------|
| `!pto.tile_buf<loc=vec, ...>` | 向量缓冲区（Vector tile） | `pl.load(..., target_memory=pl.MemorySpace.Vec)` 或默认 load 目标、`pl.create_tile(..., target_memory=pl.MemorySpace.Vec)` |
| `!pto.tile_buf<loc=mat, ...>` | 矩阵/L1 缓冲区（Matrix tile） | `pl.load(..., target_memory=pl.MemorySpace.Mat)` 等 |
| `!pto.tile_buf<loc=left, ...>` | 左矩阵缓冲（Left tile，CUBE 左操作数） | `pl.move(..., target_memory=pl.MemorySpace.Left)` |
| `!pto.tile_buf<loc=right, ...>` | 右矩阵缓冲（Right tile，CUBE 右操作数） | `pl.move(..., target_memory=pl.MemorySpace.Right)` |
| `!pto.tile_buf<loc=acc, ...>` | 累加器缓冲（Acc tile） | 如 matmul_acc 的结果、`pl.MemorySpace.Acc` |
| `!pto.tensor_view<?x?xdtype>` / `!pto.partition_tensor_view<MxNxdtype>` | 张量视图 / 分区视图（DDR 上的“一块”） | 由 **Tensor 参数** 经 `pto.make_tensor_view` + `pto.partition_view` 得到，作为 tload/tstore 的 **ins** 操作数 |

因此：**InCore 中作为 PTO ISA 参数的基础数据** = 上述 **tile_buf（vec/mat/left/right/acc）** + **partition_tensor_view（来自 Tensor）**。没有单独的“matrix tile 类型”名字，只有 `tile_buf` 的 `loc` 区分用途。

#### Tensor 和 Tile 是否需要转换？是否通过 tload / tstore 完成？

**不需要**在类型系统里做“Tensor ↔ Tile”的显式转换（没有 `pl.cast(tensor, tile)` 这类写法）。**数据在 Tensor 与 Tile 之间的迁移，只通过两类操作完成**：

1. **`pl.load(tensor, offsets, shapes [, target_memory])`**  
   - 语义：从 **Tensor** 上由 `offsets`/`shapes` 指定的区域读到 **Tile**（UB 中）。  
   - PTO：`pto.partition_view`（从 tensor_view 切出该区域）+ **`pto.tload`**（partition_tensor_view → tile_buf）。  
   - 即：**Tensor 的某块区域 → 通过 tload 变为 Tile（vector/mat/left/right 等由 target_memory 决定）**。

2. **`pl.store(tile, offsets, shapes, tensor)`**  
   - 语义：把 **Tile** 写回 **Tensor** 上由 `offsets`/`shapes` 指定的区域。  
   - PTO：`pto.partition_view` + **`pto.tstore`**（tile_buf → partition_tensor_view）。  
   - 即：**Tile → 通过 tstore 写回 Tensor 的某块区域**。

**Tile ↔ Tile（不同 UB 区域）** 用 **`pl.move(tile, target_memory=...)`**，生成 **`pto.tmov`**（例如 Vec → Left/Right 供 CUBE 使用）。

**结论**：  
- 前端 **Tensor** 与 **Tile** 是两种抽象（DDR 上的区域 vs UB 上的块）；  
- 在 InCore 中，它们**不**做类型转换，而是**通过 `tload` / `tstore` 在“Tensor 的某块”与“Tile（vec/mat/left/right/acc）”之间搬运数据**；  
- `pl.load` / `pl.store` 就是这两类搬运在前端的唯一接口，对应 PTO ISA 的 **tload** 与 **tstore**。

---

## 9. pipev 与 set_wait 在 PTO 汇编中的说明

### 9.1 结论：**PyPTO 前端的 PTO 汇编（.pto）不生成 pipev / set_wait**

- **PTOCodegen**（生成 .pto 的 C++ 前端）**只**生成：
  - 常量、`pto.make_tensor_view`、`pto.alloc_tile`
  - 以及 `pto.partition_view`、`pto.tload`、`pto.tstore`、`pto.tmul`、`pto.taddc`、`pto.tadds` 等 **计算与访存** 指令。
- 在 **`src/codegen/pto/`** 中 **没有** 对 `sync`、`pipe`、`wait`、`flag`、`barrier` 的生成逻辑；因此：
  - **.pto 输出中不会出现 pipev、set_wait（或 set_flag/wait_flag）**。
  - 管道/事件同步不在 PTO 前端的 .pto 里表达，而是由下游工具或运行时处理。

### 9.2 与 CCE 后端的对比（便于理解）

- **CCE 后端** 走另一条路径：
  - **InsertSync** Pass 在 **IR** 中插入 `system.sync_src` / `system.sync_dst`（以及 bar_v / bar_m 等）。
  - **CCECodegen** 把这些 IR 节点翻译成 **C++** 里的 `set_flag(...)`、`wait_flag(...)`、`pipe_barrier(...)`。
- 因此：
  - **set_flag / wait_flag** 出现在 **CCE 生成的 C++ 内核** 中，**不**出现在 PTO 的 **.pto 文件** 中。
  - 若“pipev”指向量管线（如 PIPE_V），那是 **PipeType** 枚举与 CCE 同步 API 的参数，不是 .pto 里的指令名。

### 9.3 若需在 .pto 或下层 ISA 中使用同步

- 当前 PyPTO 前端 **不** 在 .pto 中生成 pipev/set_wait；若目标平台或 ptoas 需要这类指令，需要：
  - 在 pto-isa 方言或 ptoas 中扩展同步指令并在下层插入，或
  - 由运行时在调用 kernel 时保证顺序与依赖（编排层已通过任务图表达依赖）。

---

## 10. In-cluster-function-group（簇内函数组）

本节描述前端语言中用于表达在**簇（cluster）**内、由本地互联的一组核心上运行的计算，簇内通信用 **push/pop** 抽象，而不是 store/load。

### 10.1 Cluster 作为哑张量

**Cluster（簇）** 用一个**哑张量**（或标量变量）表示，对应一组**本地互联的核心**。在 **A5 Ascend** 处理器上，一个簇由 **2 个 AIV** 和 **1 个 AIC** 核心组成。簇是簇内函数组的**分配与调度单位**。

### 10.2 簇内函数与通信

**簇内函数（in-cluster functions）** 是一组通过**本地互联通道**相互通信的函数，在语言层抽象为 **push** 和 **pop**。在该组内，数据通信在 incore 函数内部用 **TPUSH** 和 **TPOP** 表示，而不是 **TSTORE** 和 **TLOAD**，以反映数据在簇的本地互联上流动，而不经过全局内存。

### 10.3 作用域语法：allocate_cluster

所有 incore 函数被视作**一个簇内函数组**的作用域，由对 pto 运行时的一次**阻塞**调用开始：

- **`allocate_cluster`**  
  调用 pto 运行时分配一个可用的处理器簇，返回一个哑张量或标量变量 **`clusterID`**，用于标识所分配的簇。

- **阻塞语义**  
  若当前没有空闲簇，pto 运行时会**阻塞**编排，直到有簇可用。只有在 `allocate_cluster` 返回后，程序才继续执行簇内函数组。

- **clusterID 作为组的输入**  
  **clusterID** 是该簇内函数组中**所有函数的输入参数**。它保存在 **pto 运行时任务描述符**中，表示该任务**仅允许**在由 clusterID 指定的簇上执行。

- **调度保证**  
  pto-runtime 调度器**不会**将该组内任何任务调度到其他簇；始终使用 **clusterID** 所标识的簇。调度器仍按**数据依赖**对任务排序，保证正确性与顺序。

- **作用域结束：释放簇**  
  程序**不**显式调用 clusterID 的 free API。当 **clusterID** 张量被运行时释放（例如离开作用域或被回收）时，**pto-runtime** 会**自动**将该簇归还到可用簇池，供后续 `allocate_cluster` 再次分配。

### 10.4 Incore 参数类型：PIPE_IN 与 PIPE_OUT

**Incore 函数**的参数类型扩展为包含 **PIPE_IN** 和 **PIPE_OUT**，表示通过**本地互联管道**传递数据的变量，而不是全局内存张量。

- **语义**  
  **PIPE_IN** 与 **PIPE_OUT** 表示簇内函数组中函数之间的**生产者–消费者**数据流，数据经簇的本地互联（TPUSH/TPOP）传递，不经全局内存。

- **运行时行为**  
  对 PIPE_IN/PIPE_OUT 参数，**pto-runtime** **不**为其分配全局内存（如 ring buffer）。它们仍表达函数间的**数据依赖**，调度器据此排序任务、保证执行正确；仅存储位置在互联管道上，而非全局内存。

- **Drain 不变式**  
  **程序员**必须保证每个簇内函数组在作用域结束前**完全排空**互联管道：即每次 **push** 都有**对应的 pop** 消费该数据。作用域结束时管道中不得残留数据。

- **Tensor map 与最小形状**  
  当前设计中，尽管 PIPE_IN/PIPE_OUT 数据经管道（非全局内存）传递，仍可**按具有最小形状的普通张量**参与 **tensor map**，用于在函数间跟踪数据依赖。

- **单生产者、多消费者**  
  若某个 incore 函数需要将数据传给**两个**（或更多）后继函数，必须在函数接口上声明**两个独立的 PIPE_OUT** 变量——每个消费者一个。每个 PIPE_OUT 对应一条逻辑管道和一个消费者。

### 10.5 小结

- **Cluster**：用哑张量/标量表示一组本地互联核心（如 A5 Ascend 上 2 AIV + 1 AIC）。
- **簇内通信**：在 incore 函数内使用 TPUSH/TPOP，替代 TSTORE/TLOAD。
- **作用域**：由阻塞的 **allocate_cluster** 开始；**clusterID** 传入组内每个函数并写入任务描述符，运行时仅在该簇上调度这些任务并尊重数据依赖。程序不显式释放簇；当 **clusterID** 张量被运行时释放时，pto-runtime **自动**释放簇。
- **PIPE_IN / PIPE_OUT**：用于经本地互联管道传递数据的 incore 参数类型；运行时不为它们分配全局内存；程序员须保证管道完全排空（每次 push 有对应 pop）。为依赖跟踪，在 tensor map 中按最小形状的普通张量处理；单生产者多消费者用多个独立的 PIPE_OUT 参数表示。

---

## 11. block_incore 函数

本节为 **block_incore** 函数增加语法：一种以 **SPMD**（Single Program Multiple Data）方式执行的 incore 函数。

### 11.1 调用参数：blockdim 与 block_id

对 **block_incore** 函数的调用带有两个额外参数，用于标识块与整体并行度：

- **blockdim**  
  **块的总数**。函数按块被调用，因此有 **blockdim** 次并发调用。

- **block_id**  
  本次调用所对应的**块索引**，取值范围为 `0 .. blockdim-1`。每次调用获得不同的 **block_id**，incore 代码据此计算块内索引与数据。

每个 SPMD 核心（或逻辑块）以不同的 **block_id** 运行；它们共同构成同一 incore 函数的 **blockdim** 次并行执行。

### 11.2 与簇内函数组一起使用

当 **block_incore** 函数**与**簇内函数组一起使用时，编排层必须分配**足够数量的簇**以运行所有块：

- 分配的簇数量必须**等于 blockdim**，即每个块对应一个簇。
- 实践中，**allocate_cluster** 的用法需能提供 **blockdim** 个簇（例如运行时一次分配大小为 **blockdim** 的簇集合，或程序通过多次/批量分配使得到的 **clusterID** 或簇集合的基数为 **blockdim**）。
- **blockdim** 次 block_incore 簇内函数组的调用各自在其中一个簇上运行，**block_id** 标识该调用对应的块（进而对应哪个簇）。

从而保证有足够簇执行 SPMD 簇内函数组，既不过载也不欠载簇资源。

### 11.3 收益与编排模式

- **任务压缩与运行时开销**  
  **block_incore** 函数能有效**压缩** pto-runtime 可见的**任务数量**。不再调度大量细粒度任务（如每 tile 或每元素一个），而是调度一个（或少数）block_incore 任务，每个任务内部运行 **blockdim** 个并行块。这**降低**了 pto-runtime 的开销（更少的任务描述符、编排层更少的调度与依赖跟踪）。

- **PyTorch 即时执行模式**  
  **block_incore** 函数也可在 **PyTorch 即时执行模式**下编排。该模式下，程序直接从 Python 启动用 **pyPTO** 语法与编译链编写的 SPMD 内核，而不使用 pto-runtime。这提供了一条**简单路径**运行 pyPTO 编译的 SPMD 内核（例如原型验证或不需要完整任务图调度时），在复用同一 pyPTO 前端与编译器的同时，避免 pto-runtime 的复杂度。

---

## 12. 输出目录结构（PTO 后端，含编排时）

```
output_dir/
├── passes_dump/              # 各 pass 后的 IR
├── ptoas/
│   ├── <func_name>.pto       # PTOCodegen 生成的 PTO-ISA MLIR
│   └── <func_name>.cpp       # ptoas 编译 .pto 得到的 C++
├── kernels/aiv/
│   └── <func_name>.cpp       # 最终 kernel 包装（CCE 调用约定）
├── orchestration/
│   └── <orch_func_name>.cpp  # 编排 C++
└── kernel_config.py          # 运行时/编排/内核配置
```

---

## 13. 参考示例（本仓库）

- **InCore + 编排**：`examples/ir_parser/orchestration_example.py`、`examples/ir_parser/vector_example_dag.py`
- **Paged Attention 多 kernel 编排**：`examples/ir_parser/paged_attention_example.py`
- **程序与跨函数调用**：`examples/ir_parser/program_example.py`
- **InCore 作用域**：文档 `docs/en/dev/passes/08-outline_incore_scopes.md`

---

## 14. 参考文档

- 语言语法：`docs/en/dev/language/00-python_syntax.md`
- IR 类型：`docs/en/dev/ir/02-types.md`
- PTO 代码生成：`docs/en/dev/codegen/00-pto_codegen.md`
- CCE 代码生成（set_flag/wait_flag）：`docs/en/dev/codegen/01-cce_codegen.md`
- InsertSync（同步插入）：`docs/en/dev/passes/06-insert_sync.md`
