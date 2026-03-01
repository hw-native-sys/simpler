# PyPTO-Lib: Primitive Tensor Function Library

## 1. Purpose

**PyPTO-Lib** is a library of **primitive tensor functions** built on top of the programming framework defined in the **pypto** subfolder. It provides a fixed set of low-level, hardware-agnostic **tensor-level** operations that serve as the building blocks for higher-level kernels and model graphs—analogous to the role of **PyTorch ATen** primitive functions in the PyTorch stack.

**Where the library lives.** Defining a new function set **at the tile level** would add very little value: tile-level operations are essentially PTO-ISA instructions already. The library’s value is at the **tensor level**: primitives are operations on **tensors** (e.g. `max(x, axis)`, `exp(x)`, `sum(x, axis)`). The **compiler** is responsible for tiling those tensor ops into loops over tiles and lowering each tile to PTO-ISA. So PyPTO-Lib defines the **tensor-level** primitive set; tiling and PTO-ISA are the **lowering** of that set, not a second layer of “tile-level primitives” on top of PTO-ISA.

This document describes the **method of building** that library and the **three core purposes** it must accomplish:

1. **Tiling and conversion to PTO-ISA**  
   The **compiler** tiles tensor operations into smaller subtensors and lowers each tiled operation to PTO-ISA. Because **Tensor** and **Tile** are two different types, the lowering uses **`cast_tensor_to_tile`** and **`cast_tile_to_tensor`** for view-only conversion (no data movement). At this stage **TLOAD is not inserted**; incore boundaries and data movement are decided later by the backend.

2. **Tail blocks and padding**  
   Handle **tail blocks** and **padding** from tiling so that lowered code is correct for arbitrary tensor shapes.

3. **Enabling fusion across composite functions**  
   Composing primitives **inside** one composite (e.g. softmax = one loop with max, sub, exp, sum, div in the body) is effectively **manual fusion**: the user (or composite author) has already put multiple tile ops in the same loop body. The **remaining challenge** is **how to fuse loops across multiple composite functions**—e.g. when the user writes `relu(softmax(x))`, two separate composites each with their own tile loop—so the compiler can merge them into one loop with a larger body. The representation and compiler must support **cross-composite loop fusion**, not just a single composite with many tile ops.

---

## 2. Tensor vs Tile: Cast Without Data Movement

In the pypto framework, **Tensor** (N-D logical tensor, e.g. in DDR or global memory) and **Tile** (hardware-aware block in unified buffer / local memory) are **two different classes**. The library must bridge them without committing to when data actually moves.

### 2.1 Cast Primitives

- **`cast_tensor_to_tile(Tensor, offsets, sizes)`**  
  Produces a **Tile** that represents a **view** over a region of the Tensor identified by `offsets` and `sizes`. No data copy: the Tile is a logical descriptor (shape, stride, base pointer/view) over that region. Semantics are “this tile is this subtensor”; lowering to actual TLOAD/TSTORE is deferred until the compiler has chosen incore boundaries.

- **`cast_tile_to_tensor(Tile)`**  
  Produces a **Tensor** that represents a **view** over the same logical region as the Tile. Again no data movement: it is the inverse view for type compatibility (e.g. when a primitive expects Tensor but the producer is a Tile, or for chaining back into tensor-level ops).

At the **library level**, we do **not** insert TLOAD (or similar) instructions. The library expresses:
- **Tiling**: which subtensor regions (Tiles) each operation works on.
- **PTO-ISA calls**: the sequence of PTO-ISA operations (elementwise, reduction, etc.) on those Tiles.

**Data movement** (when to load from Tensor memory into tile buffers, when to store back) is a **later compilation concern**, once the compiler has decided incore function boundaries and placement.

---

## 3. Design Principles

### 3.1 Framework Foundation (pypto)

The library is built entirely on the programming framework in the **pypto** subfolder, which provides:

- **Multi-level IR**: Tensor-level and Tile/Block-level intermediate representation (see `pypto/docs/dev/`, `pypto/include/pypto/ir/`, `pypto/src/ir/`).
- **IR builder and ops**: Tensor ops, block ops, and sync ops registered under `pypto/src/ir/op/` (e.g. `tensor_ops/`, `block_ops/`, `sync_ops/`).
- **Code generation**: PTO codegen that emits PTO-ISA dialect MLIR from pypto IR (`pypto/include/pypto/codegen/pto/`, `pypto/docs/dev/12-pto_codegen.md`).
- **Backend abstraction**: Backend registry and platform-specific backends (e.g. 910B_PTO, 910B_CCE) without tying the primitive *semantics* to a particular execution model.
- **Python frontend**: Language layer and bindings in `pypto/python/pypto/` for building and compiling programs.

Primitives in PyPTO-Lib are expressed as **pypto IR programs** (Tensor and/or Block-level) and then compiled through this pipeline. The framework does not require the library to commit to “incore” vs “orchestration”; that split is a backend/implementation detail.

### 3.2 PTO-ISA and ptoas

The **instruction set** and **assembly format** for the generated code are defined by **ptoas** (PTO Assembler & Optimizer in the **ptoas** subfolder):

- **PTO dialect**: Operations and types in `ptoas/include/PTO/IR/` (e.g. `PTOOps.td`, `PTOTypeDefs.td`) define the PTO-ISA operations (e.g. `pto.make_tensor_view`, `pto.alloc_tile`, load/store, elementwise, reduction, sync).
- **Lowering and passes**: `ptoas/lib/PTO/Transforms/` contains passes that optimize and lower PTO IR (e.g. sync insertion, memory planning) and eventually to code that uses the pto-isa C++ library.
- **Toolchain**: The `ptoas` toolchain consumes `.pto` (PTO bytecode) and produces executables or artifacts for the target device.

PyPTO-Lib primitives are **lowered to sequences of PTO-ISA instructions** via:

1. **PyPTO IR → PTO-ISA MLIR**: Using pypto’s PTO codegen (which emits the PTO dialect understood by ptoas).
2. **PTO-ISA MLIR → binary**: Using ptoas for parsing, optimization, and code generation.

The library is thus **defined in terms of PTO-ISA semantics** (as defined in ptoas), while being **constructed and bound** through the pypto framework.

### 3.3 No Incore vs Orchestration in the Primitive Contract

The **primitive library API and semantics** do not specify:

- Which work runs on **AICore** (incore compute) and which on **AICPU** (orchestration/scheduling).
- How load/store, sync, and compute are mapped to specific hardware units.

That mapping is the responsibility of:

- The **pypto backend** and **codegen** (e.g. 910B_PTO, 910B_CCE),
- The **ptoas** lowering and target-specific codegen,
- And the **runtime** (e.g. pto-rt2/simpler) that executes the resulting binaries.

Primitives are **semantic building blocks** (e.g. “elementwise add”, “matmul”, “sum over axis”); backend and runtime decide how to place them on cores and pipelines.

### 3.4 Binding to the PyPTO Frontend

Primitives are **bound to the pypto frontend** so that:

- Users can **compose** them via the pypto Python API (e.g. `pypto.language`, `pypto.ir`) and IR builder.
- **Compilation** uses pypto’s passes and PTO codegen, then ptoas for PTO-ISA-level optimization and code generation.
- **Registration** follows pypto’s operator model (e.g. `REGISTER_OP`, tensor/block/sync op categories) so that primitives appear as first-class ops in the IR and in the language layer.

So: **PyPTO-Lib = a curated set of primitive tensor functions, implemented as pypto IR (and thus as PTO-ISA), exposed through the pypto frontend, without fixing incore vs orchestration.**

### 3.5 Tail Blocks and Padding

Tiling divides tensors into full tiles and possibly a **tail** (remaining dimensions not filling a full tile). The library must:

- **Detect tail blocks**: When a dimension is not divisible by the tile size, the last tile along that dimension has a smaller logical size.
- **Padding or mask**: Either (a) pad the tail tile to the full tile shape and use **masking** in PTO-ISA so that padded elements do not affect results (e.g. reduction, max), or (b) generate a separate code path for the tail with a smaller tile and no padding. The library exposes a consistent interface (e.g. `valid_row` / `valid_col` in PTO-ISA terms) so the backend can lower to the appropriate form.
- **Correctness**: All primitives (reductions, elementwise, etc.) must respect tail semantics so that composite functions (e.g. softmax over rows) remain correct at boundaries.

### 3.6 Fusion: Within Composite vs Across Composites

- **Within a single composite**: Putting multiple tile (PTO-ISA) operations inside the same loop body—e.g. softmax as one loop over tiles with body (max, sub, exp, sum, div)—is **manual fusion**. The author of the composite has already fused the sequence of tile ops into one loop. There is no separate “fusion pass” required for that; the value is in having a clear tensor-level primitive set and a lowering that emits one tile loop per composite.

- **Across multiple composite functions**: The **real challenge** is **fusing loops across composites**. The user may write `z = relu(softmax(x))`: two distinct composite functions, each lowered to its **own** tile loop (loop over tiles for softmax; loop over tiles for relu). Without fusion, the generated code does “full pass of softmax over the tensor” then “full pass of relu over the tensor,” with the intermediate result materialized. To fuse, the compiler must **merge the two loops** into one: for each tile, execute (softmax body for that tile; then relu body for that tile), so the tile never leaves the fast memory between the two. That requires:
  - **Representation**: Composites must be expressed so that their **loop structure** (tile bounds, iteration space) and **body** are visible to the compiler. A form like “composite = single tile loop + body (sequence of tensor-op lowerings)” allows the compiler to match two consecutive composites and try to merge their loops.
  - **Data flow**: The compiler must see that the output of the first composite (softmax) is the sole input of the second (relu) on the same logical tensor/tiling, so merging the loops is semantics-preserving.
  - **Iteration-space alignment**: Both loops must iterate over the same tile grid (same shape, same tiling policy). If they do, fusion is “replace loop_A; loop_B by loop { body_A; body_B }”; if not (e.g. different tiling), fusion may be illegal or require more sophisticated transformation.
  - **Legality**: No intervening use of the first composite’s full-tensor output between the two loops; no dependency that would be violated by reordering.

So the library and IR should be designed to **ease cross-composite loop fusion**: explicit, canonical tile-loop structure and explicit data flow between composites, so that a **fusion pass** can identify “loop of composite A” and “loop of composite B,” prove same iteration space and producer–consumer relationship, and merge them into one loop. The difficulty is in the **analysis and transformation** (matching loops, proving alignment, handling tails and padding consistently), not in defining yet another set of tile-level ops on top of PTO-ISA.

---

## 4. Incore Scope

The **incore scope** is a way for the user to specify **where the boundary between orchestration (e.g. host/AICPU) and incore compute (e.g. AICore)** lies, without changing the logical algorithm. By inserting an **incore scope directive** into the source code of a Python program, the user can define the boundaries of incore scopes and **adjust them with ease** (e.g. move a few lines in or out of the scope) to tune placement and data movement.

### 4.1 What an Incore Scope Is

- An incore scope defines an **anonymous incore function**: a region of code that will be compiled as a separate **incore** kernel (e.g. running on AICore), **without the user explicitly specifying the function's arguments**. The compiler derives the arguments from how variables are used across the scope boundary (see below).

- **Definition and call in one place**: The incore scope does **not** only define the function. It also defines a **reference (call)** to this function from the **parent function** at the **exact location** where the scope appears. So the parent (orchestration) code runs until it hits the scope, then **calls** the generated incore function (with the derived arguments), then continues. No separate "define elsewhere, call here" is required; the scope is both the definition and the call site.

- **Naming**: The compiler automatically generates a **distinctive name** for the anonymous incore function, using the **parent function's name as a prefix** to keep names human-readable and debuggable (e.g. `my_kernel_incore_0`, `my_kernel_incore_1` if there are multiple scopes in `my_kernel`).

### 4.2 Compiler-Derived Arguments

The compiler **automatically derives** the arguments of the anonymous incore function using the following criteria:

| Role | Rule | Meaning |
|------|------|--------|
| **Input** | Defined **outside** the scope, **referenced inside** the scope **without modification** (read-only) | Passed as an **input** argument to the incore function. Data is read by incore; no obligation to write back. |
| **Inout** | Defined **outside** the scope, **modified inside** the scope | Passed as an **inout** argument (e.g. by reference). The incore function may read and write; changes are visible to the parent after the call. |
| **Output** | Defined **outside** the scope; **unassigned** by the parent; **assigned (written)** by the incore scope; **read** by the parent after the scope | Treated as an **output** argument. It is passed into the incore function **by reference**. The **memory space** for this symbol is **allocated by the runtime when the incore function is called (submitted)**. The incore function writes into that buffer; after the call, the parent reads the symbol. |

**Output in more detail:** A symbol that is defined outside the incore scope, left **unassigned** by the parent, **written** inside the incore scope, and **read** by the parent after the scope is classified as **output**. The compiler passes it as an output argument by reference. The **runtime allocates** its memory when the incore function is **called (submitted)**; the incore function receives the reference and writes into that buffer, and the parent then uses the same symbol (backed by that buffer) after the call.

### 4.3 How to choose incore_scope

There is an essential **rule of thumb**: the **intermediate data usage within the incore scope must not exceed the size of the SRAM buffer inside the processor core**. The working space of the incore function (inputs, outputs, and any temporaries used inside the scope) needs to **fit into that on-core SRAM**. If it does not, data will **spill to global memory**, which leads to a **severe performance penalty** because of memory bandwidth limits and the latency of global memory access—the **memory wall**. Therefore, placing the incore scope boundary so that all live data inside the scope fits in core SRAM is the primary constraint for both correctness of placement and performance.

To support this:

- **Compiler**: The compiler can provide a **tool to estimate the memory usage within a given scope** (e.g. by summing the sizes of all live tensors/tiles and temporaries at each point in the scope, possibly with a simple allocation model). That estimate can be compared against the target core’s SRAM size to guide or validate scope boundaries.
- **Programmer**: The programmer can **estimate memory usage manually** (e.g. from tensor/tile shapes and known temporary buffers) and use that estimate as the **most important guideline** when choosing where to start and end incore scopes. If the estimated working set exceeds SRAM, the scope should be narrowed (e.g. move some computation or loops outside the scope, or tile so that each incore invocation uses less data).

In short: **keep the incore scope’s working set within core SRAM**; use compiler-based or manual memory estimates as the main guide for setting scope boundaries.

**Runtime and IDE integration.** The runtime **pto-rt2** (simpler) already provides **scope_begin()**, **scope_end()**, and **ring-heap–based memory allocation**, so it can perform **accurate accounting of memory workspace usage per scope**. The runtime can **collect statistics** on memory usage size for each scope (or each scope/level). This information should be **incorporated into the IDE** so that programmers can **inspect it easily** (e.g. per-scope or per-level usage, peak usage, or comparison with SRAM capacity). The same statistics can be used for **manual tuning** of incore scope boundaries and for **automatic adjustment** of frontend incore scope boundaries (e.g. a tool that suggests moving scope boundaries inward or outward based on measured usage versus SRAM size).

### 4.4 Example: Defining an Incore Scope and Equivalent After Analysis

**User source (Python with incore scope directive):**

Assume a directive such as `with incore_scope():` (or `@incore` on a block, or a comment/directive the compiler recognizes). The following is conceptual.

```python
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    # x, y are inputs from parent/caller
    n, c = x.shape[0], x.shape[1]
    tmp: Tensor   # defined outside, unassigned by parent; written inside incore_scope; read after
    with incore_scope():
        # Inside: read x, y (read-only); write tmp (output).
        for i in range(n):
            for j in range(c):
                tmp[i, j] = x[i, j] + y[i, j]   # tmp is written inside; x, y only read
    # After scope: read tmp
    result = reduce_sum(tmp, axis=1)
    return result
```

**Equivalent after compiler argument analysis:**

The compiler (1) generates a named incore function with **explicit** arguments derived from the rules above, and (2) inserts a **call** at the scope location in the parent, passing those arguments.

**Generated anonymous incore function (conceptual):**

```python
# Compiler-generated name: my_kernel_incore_0
def my_kernel_incore_0(
    x: Tensor,      # input: defined outside, read-only inside
    y: Tensor,      # input: defined outside, read-only inside
    tmp: Tensor,    # output: defined outside, unassigned by parent, written inside, read after; passed by reference
) -> None:
    n, c = x.shape[0], x.shape[1]
    for i in range(n):
        for j in range(c):
            tmp[i, j] = x[i, j] + y[i, j]
```

**Parent function with explicit call (conceptual):**

```python
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    n, c = x.shape[0], x.shape[1]
    tmp: Tensor   # output: unassigned by parent; runtime allocates when incore is called
    my_kernel_incore_0(x, y, tmp)       # call at scope location: x, y inputs; tmp output by reference
    result = reduce_sum(tmp, axis=1)
    return result
```

So: **Inputs** `x` and `y` (defined outside, only read inside) become **input** parameters of `my_kernel_incore_0`. **Output** `tmp` (defined outside, unassigned by parent, written inside the scope, read after) becomes an **output** argument passed by reference; the **runtime allocates** its memory when the incore function is **called (submitted)**. If `tmp` were read and then written inside the scope, the compiler would classify it as **inout**.

---

**More complicated example: multiple inputs, inout, and output**

User source: one incore scope with two inputs, one **inout** buffer (read and updated inside the scope), and one output. The parent initializes the inout buffer before the scope and reads it after.

```python
def fused_update(x: Tensor, y: Tensor, scale: float) -> Tensor:
    n, c = x.shape[0], x.shape[1]
    acc: Tensor = zeros((n, c), dtype=float32)   # parent assigns: acc is inout, not output
    out: Tensor      # defined outside, unassigned by parent; output
    with incore_scope():
        # Inputs: x, y, scale (read-only).
        # Inout: acc (read and written: acc[i,j] += x[i,j] * scale).
        # Output: out (written here, read by parent after).
        for i in range(n):
            for j in range(c):
                acc[i, j] += x[i, j] * scale    # inout: read then write
                out[i, j] = y[i, j] + acc[i, j] # output: write; uses updated acc
    # After scope: read acc and out
    result = reduce_sum(out, axis=1) + reduce_sum(acc, axis=1)
    return result
```

**Equivalent after compiler argument analysis:**

Generated incore function (multiple inputs, one inout, one output):

```python
# Compiler-generated name: fused_update_incore_0
def fused_update_incore_0(
    x: Tensor,       # input: read-only inside
    y: Tensor,       # input: read-only inside
    scale: float,    # input: read-only inside
    acc: Tensor,     # inout: read and written inside; parent assigned before, reads after
    out: Tensor,     # output: unassigned by parent, written inside, read after; runtime allocates
) -> None:
    n, c = x.shape[0], x.shape[1]
    for i in range(n):
        for j in range(c):
            acc[i, j] += x[i, j] * scale
            out[i, j] = y[i, j] + acc[i, j]
```

Parent with explicit call:

```python
def fused_update(x: Tensor, y: Tensor, scale: float) -> Tensor:
    n, c = x.shape[0], x.shape[1]
    acc = zeros((n, c), dtype=float32)   # inout: parent creates and initializes
    out: Tensor      # output: runtime allocates when incore is called
    fused_update_incore_0(x, y, scale, acc, out)
    result = reduce_sum(out, axis=1) + reduce_sum(acc, axis=1)
    return result
```

So: **Inputs** `x`, `y`, `scale` (read-only inside) → input args. **Inout** `acc` (assigned by parent before scope, read and written inside, read after) → inout arg passed by reference; parent creates and initializes it. **Output** `out` (unassigned by parent, written inside, read after) → output arg by reference; **runtime allocates** when the incore is called.

### 4.5 Summary

- **Incore scope directive** in the source marks a region that becomes an **anonymous incore function** plus a **call** at that point in the parent.
- **Arguments are derived**: **input** (outside, read-only inside), **inout** (outside, modified inside), **output** (defined outside, unassigned by parent, written inside scope, read by parent after; passed by reference; memory allocated by runtime when incore is called/submitted).
- The compiler generates a **readable name** (parent name as prefix) and **explicit** function parameters and call site so that the runtime can allocate and pass buffers and the incore boundary is clear and easy to adjust.

---

## 5. Primitive Set: ATen-Like Scope (Tensor-Level)

The set of primitive functions in PyPTO-Lib is **tensor-level**, similar in scope to **PyTorch ATen**: the API operates on tensors (shapes, axes, dtypes). The compiler then tiles these and lowers to PTO-ISA; there is no separate “tile-level primitive set” on top of PTO-ISA. The roster includes:

- **Elementwise binary**: add, sub, mul, div (with broadcasting where applicable).
- **Elementwise unary**: sqrt, exp, log, neg, etc.
- **Reductions**: sum, max, min (over axes, with optional keepdim).
- **Linear algebra**: matmul, batch matmul (and related).
- **Memory and indexing**: load/store between tensor and tile buffers; indexing, slice, view-like operations.
- **Type and layout**: cast, reshape, broadcast (as primitives that the backend can map to PTO-ISA).
- **Sync/control**: synchronization primitives where the IR exposes them (e.g. sync_src, sync_dst, barriers), still without specifying whether they are implemented as incore or orchestration.

The **exact roster** can be expanded or trimmed to match ATen’s usage and the capabilities of the PTO-ISA (as defined in ptoas) and pypto’s existing tensor/block ops. The important point is that **PyPTO-Lib is the closed set of primitives** on top of which fused ops and full model graphs are built, just as ATen is for PyTorch.

---

## 6. Example: Softmax from Tensor-Level Primitives

### 6.1 Tensor-Level Primitives Used

Softmax is expressed at the **tensor level** using these primitives:

- **max**(x, axis, keepdim) — reduction.
- **sub**(x, y) — elementwise.
- **exp**(x) — elementwise.
- **sum**(x, axis, keepdim) — reduction.
- **div**(x, y) — elementwise.

Formula (over last axis): **softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))**.

### 6.2 Softmax as One Composite (One Loop = Manual Fusion)

At the **tensor level**, softmax is a single composite that **calls** these primitives in sequence. The **compiler** lowers this to **one** tile loop whose body is the lowered sequence (max → sub → exp → sum → div) for each tile. So “multiple tile ops in one loop” is **by construction**: the composite is defined as one logical function, and the lowering produces one loop. That is effectively **manual fusion** of the tile sequence into one loop—no cross-composite fusion is involved.

Conceptually (tensor-level API):

```
softmax(x) = div(exp(sub(x, max(x, axis=-1, keepdim=True))),
                 sum(exp(sub(x, max(x, axis=-1, keepdim=True))), axis=-1, keepdim=True))
```

The compiler tiles this, introduces cast_tensor_to_tile / cast_tile_to_tensor as needed, and emits one loop over tiles with that body. Tail handling is part of the lowering.

### 6.3 The Real Challenge: Fusing Loops Across Multiple Composite Functions

The **hard** problem is when the user writes **two (or more) composite functions** in sequence, e.g.:

- `y = relu(softmax(x))`  
- or `z = add(softmax(x), bias)`

Then the compiler has:

- **Composite 1 (softmax)**: one tile loop, body = (max, sub, exp, sum, div) for each tile, writing an output tensor.
- **Composite 2 (relu or add)**: another tile loop, body = (read tile, op, write) for each tile, reading that output.

Without fusion, code is: **loop_softmax** (over all tiles); then **loop_relu** (over all tiles). The full softmax output is materialized between the two. To avoid that, the compiler must **fuse the two loops** into one: for each tile, run (softmax body for that tile; then relu body for that tile), so the tile is never written back to full tensor memory between the two. That is **cross-composite loop fusion**.

**What the compiler must do:**

1. **Recognize two consecutive tile loops** that correspond to two composites (e.g. softmax and relu).
2. **Prove same iteration space**: Same tensor shape and same tiling strategy, so the two loops run over the same tile indices.
3. **Prove producer–consumer**: The output of the first composite is the only input of the second for that tensor; no other consumer of the first output sits between the two loops.
4. **Merge**: Replace `for tile: body_softmax(tile); for tile: body_relu(tile)` by `for tile: body_softmax(tile); body_relu(tile)`.
5. **Cleanup**: Remove materialization of the full-tensor output of the first composite between the two; keep only the fused loop and later insert TLOAD/TSTORE at incore boundaries.

**What makes this difficult:** The representation must expose **per-composite** loop structure and **data flow between composites** so the compiler can match loops, check alignment, and merge. Different tiling or shapes between the two composites can make fusion illegal or require more complex transformations. So the library and IR design should aim to **ease this cross-composite fusion analysis**, rather than adding another layer of tile-level “primitives” on top of PTO-ISA.

### 6.4 Strategies to Ease Cross-Composite Fusion

Two complementary strategies can reduce the difficulty of establishing producer/consumer relationships and avoiding unnecessary main-memory traffic:

#### Strategy 1: Full loop unrolling

**Idea:** Fully unroll both tile loops (softmax and relu) so that **all loop indices and tile descriptors become constants**. Each iteration is then a distinct, constant-index instance of the body.

**Why it helps:**

- **Producer/consumer becomes trivial**: For each unrolled iteration index `i`, the “output of softmax at tile `i`” and “input of relu at tile `i`” are explicit, constant-sized buffers. The compiler can treat them as a single logical tile and eliminate the write from softmax and the read into relu for that tile—no main-memory round-trip.
- **No symbolic reasoning**: No need to prove “for all tile indices, output of A is input of B”; each (softmax_i, relu_i) pair is a concrete instance, so matching and fusion are local.

**Implementation involved:**

- **Unroll pass**: A transformation that takes a loop `for i in 0..N` and replaces it with `N` copies of the body, with `i` replaced by constant `0`, `1`, …, `N-1`. Requires `N` to be constant or bounded at compile time (or a user/compiler-chosen cap for partial unroll).
- **Representation**: Loop bounds and tile counts must be known at fusion time (static or symbolically constant). Dynamic “number of tiles” would require runtime unrolling or a different strategy.
- **Cost**: Code size grows linearly (or more) with the number of tiles; acceptable only for small tile counts (e.g. small tensors or large tiles). Often used for a **single** composite’s inner loop or for small 1D cases; for large 2D/3D tile grids, full unroll is usually impractical.
- **After unroll**: A second pass identifies producer/consumer pairs (softmax_instance_i → relu_instance_i), removes the intermediate buffer materialization, and optionally fuses the two bodies into one block per index. No loop structure left to “align”; fusion is per-index code merging.

#### Strategy 2: Power-of-2 predicated expansion (without full unroll)

**Idea:** Do **not** fully unroll. Instead, rewrite the loop into a **binary expansion**: a nest of loops whose ranges are **powers of 2**, with **predicates** (masks or conditionals) so that only the “used” iterations execute. After this expansion, the compiler sees **chunks of equal shape** (each power-of-2 block has the same structure), so it can **fuse chunk pairs** (softmax chunk with relu chunk) without ever fully unrolling.

**Why it helps:**

- **Uniform chunk shape**: All chunks in the expanded form have the same power-of-2 extent (e.g. 1, 2, 4, 8, …). So “body of composite A for chunk of size 4” and “body of composite B for chunk of size 4” have the same iteration shape; the compiler can recognize **equal-shape** regions and fuse them (merge the two bodies for that chunk size) without expanding to individual indices.
- **Bounded expansion**: Only a logarithmic number of “chunk sizes” (e.g. 1, 2, 4, …, 2^k) appear, so the number of fused chunk-pair templates is small. Tail (non-power-of-2 remainder) can be handled by one extra predicated chunk or a small loop.
- **Producer/consumer per chunk**: For each power-of-2 chunk, the output of A and the input of B are the same logical range; the compiler can fuse the two chunk bodies and drop the intermediate buffer for that chunk.

**Implementation involved:**

- **Binary decomposition of the loop range**: Express `for i in 0..N` as a combination of power-of-2 segments. For example, if `N = 10`, decompose into: 8 (full power-of-2) + 2 (remainder). The loop is rewritten into:
  - A loop for “full” power-of-2 blocks (e.g. `i in 0..8` with step 8, or equivalently one chunk of size 8),
  - Plus a predicated or smaller loop for the remainder (e.g. 2 iterations). More generally, a recursive or iterative split: largest power of 2 ≤ N, then recurse on N - 2^k. Each level yields a **predicated loop** over a power-of-2 range (predicate = “index < N” or “index in valid range”).
- **Predication**: For each power-of-2 chunk, iterations that go past the actual bound (e.g. `i >= N`) must be disabled by a predicate (mask or `if`). So the IR must support **predicated execution** (masked ops, or explicit conditionals), and the backend/PTO-ISA must support predication for the tile ops (e.g. masked load/store, or branch around the chunk).
- **Canonical chunk representation**: The compiler needs a single representation for “a chunk of size 2^k”: e.g. a loop `for j in 0..2^k` (with optional predicate) plus the body parameterized by the chunk base. Then “softmax for chunk (base, size 2^k)” and “relu for chunk (base, size 2^k)” have the **same shape**; the fusion pass matches them by (base, size) and merges the two bodies into one predicated chunk.
- **Fusion pass on expanded form**: After expansion, the program has a nest of power-of-2 predicated loops (and possibly one remainder loop). The fusion pass:
  - Identifies pairs (composite A chunk, composite B chunk) with the **same** (base, size) and same predicate.
  - Verifies producer/consumer: A’s output for that chunk is B’s input.
  - Merges the two chunk bodies into one, removes the intermediate buffer for that chunk, and retains the shared predication.
- **Tail handling**: The non-power-of-2 remainder (e.g. 2 iterations when N=10) can be a separate, smaller predicated chunk or a scalar loop. It should still be fused (softmax tail + relu tail) so the representation for “tail chunk” is also regular (e.g. “chunk of size r” with predicate), allowing the same fusion logic.
- **Cost**: Code size and number of loop nests grow only **logarithmically** with the range (one “level” per power-of-2). No explosion as with full unroll. Predication may add overhead (mask updates, branches, or masked instructions); the backend must support it efficiently.

**Summary of what each strategy requires**

| Aspect | Full unroll | Power-of-2 predicated expansion |
|--------|-------------|----------------------------------|
| **Loop transformation** | Unroll pass: replace loop by N copies of body with constant index. | Decompose loop into power-of-2 segments + predicates; represent as nested predicated chunks. |
| **Symbols** | All indices and tile descriptors become constants. | Chunk base and size (2^k) are the key symbols; iteration inside chunk is regular. |
| **Producer/consumer** | Per-iteration matching; eliminate intermediate buffer per index. | Per-chunk matching (same base, same size); eliminate intermediate buffer per chunk. |
| **Representation** | Bounds and N must be compile-time constant (or capped). | IR must represent predicated loops and power-of-2 chunk structure. |
| **Backend** | No extra requirement. | Predication support (masked ops or conditionals) for tile ops. |
| **Code size** | Linear (or worse) in number of tiles; only for small N. | Logarithmic in range; scalable. |
| **Fusion pass** | After unroll: match and merge per-index (softmax_i, relu_i). | Match equal-shape chunks (same 2^k), verify data flow, merge chunk bodies. |

Together, full unroll is a simple option for small tile counts where constant propagation and per-index fusion suffice; power-of-2 predicated expansion keeps the structure regular and scalable while still giving the compiler equal-shape chunks so it can fuse **across** composites without full expansion.

### 6.5 Further analysis: power-of-2 predicated expansion

#### Does it support loop fusion for non-parallel for?

**Short answer:** Yes, but the *kind* of fusion differs.

- **Parallel tile loop**: Each iteration is independent (e.g. softmax on tile `i` does not depend on tile `i-1`). Power-of-2 expansion yields chunks; fusion of two composites means for each chunk we can run (body_A for chunk; body_B for chunk), and optionally the compiler could even interleave A and B per iteration within the chunk if both are parallel. Producer/consumer is clear: A’s chunk output feeds B’s chunk input.

- **Non-parallel tile loop**: The loop has loop-carried dependencies (e.g. tile `i` depends on tile `i-1`—sequential scan, recurrence, or reduction that cannot be split arbitrarily). Power-of-2 expansion still applies: the loop is decomposed into power-of-2 *chunks*, and **within each chunk** iterations are ordered (chunk base, base+1, …, base+2^k-1). So we preserve the original order inside the chunk. Fusion of **two** such loops (composite A and composite B) is still possible, but only in a **chunk-wise** form:
  - Fused structure: one loop over chunks; for each chunk, run **all** of A for that chunk (in order), then **all** of B for that chunk (in order). No interleaving of A and B iterations within the chunk, because A’s iteration `j+1` may depend on A’s iteration `j`.
  - So: “fusion” here is merging the **outer** loop (over chunks) and grouping the two bodies per chunk, not reordering across A and B. The compiler still gets one loop with a larger body (A_chunk then B_chunk), and can remove the intermediate materialization of A’s output for that chunk before B reads it. Dependencies *within* A and *within* B are preserved.

**Conclusion:** Power-of-2 predicated expansion supports both parallel and non-parallel loops. For non-parallel loops, fusion is **chunk-wise** (run full A over the chunk, then full B over the chunk); the compiler does not reorder iterations across the two composites.

#### Is it easier to define a limited parallel loop expression?

**Yes.** Introducing a **dedicated loop construct** for this approach simplifies the compiler:

- **Semantics**: A construct such as `parallel_for_power2(extent; min_chunk, max_chunk)` (or named equivalent) explicitly means:
  - The loop is **parallel** (no loop-carried dependency), or at least that the compiler may treat it as such for the purpose of chunking and fusion.
  - The loop is to be expanded using **power-of-2 predicated chunks** with chunk sizes between `min_chunk` and `max_chunk` (e.g. 1 and 64).
  - All loops that use the same (min_chunk, max_chunk) and the same extent (or same shape) have a **canonical expansion**: the same set of chunk sizes and the same structure. So the compiler can match two consecutive such loops by construction: same construct ⇒ same chunk layout ⇒ straightforward fusion.

- **Benefits for the compiler**:
  - No need to **infer** whether the loop is parallel or to **discover** a chunk policy; both are fixed by the construct.
  - Chunk bounds (min/max) are part of the IR, so the compiler only ever generates a **finite** set of chunk templates (see below).
  - Fusion pass becomes: “two adjacent `parallel_for_power2` loops, same extent, same (min_chunk, max_chunk) ⇒ merge into one loop; merge bodies per chunk.” No symbolic proof of independence or alignment beyond matching the construct and parameters.

- **Design**: The construct can be “limited” in the sense that it is *only* for loops that fit this model (extent and chunk bounds known or symbolically fixed). General `for` loops remain for other cases; the compiler can still apply power-of-2 expansion to them if desired, but without the same guarantees and with more analysis.

#### Smallest and largest chunk sizes

**Necessity:** Without bounds, the set of chunk sizes is unbounded (1, 2, 4, …, 2^k for arbitrarily large k), so the compiler would need to handle arbitrarily many chunk “templates” and codegen would be open-ended.

**Recommendation:** The power-of-2 expansion should take **min_chunk** and **max_chunk** (both powers of 2), e.g. `min_chunk = 1` or `2`, `max_chunk = 64` or `1024`. Then:

- Only chunk sizes in `{ min_chunk, 2*min_chunk, …, max_chunk }` are used (e.g. 1, 2, 4, 8, 16, 32, 64).
- The **number of chunk types** is fixed: `log2(max_chunk) - log2(min_chunk) + 1`. The compiler has one fusion/codegen template per chunk size.
- Any larger range is decomposed into a sequence of max-sized chunks plus a remainder (which is a chunk of size ≤ max_chunk, possibly with predicate). No new “sizes” appear.
- Tail (non-power-of-2 remainder) is represented as a chunk with a predicate or as one of the allowed sizes (e.g. 3 iterations → chunk size 4 with predicate, or a dedicated “remainder” chunk of size 3 if the IR allows a small set of fixed remainder sizes). So the compiler still only handles a **bounded** set of cases.

This keeps the expansion predictable and the fusion pass simple: only a small, fixed set of chunk shapes need to be recognized and fused.

#### Full unroll within each chunk

**Idea:** For each power-of-2 chunk of size 2^k, **fully unroll** the inner loop over the chunk (the 2^k iterations). So instead of “loop over j in 0..2^k; body(base + j)”, we have 2^k copies of the body with constant indices `base+0`, `base+1`, …, `base+2^k-1`.

**Does it help?**

- **Yes, for dependency analysis:** Inside the chunk there is no loop variable anymore—every index is constant. So:
  - Data flow and producer/consumer within the chunk are explicit (each iteration is a distinct static instance).
  - The compiler can do **constant propagation** and **dead store/load elimination** without reasoning about loop-carried dependencies; no need to prove anything symbolically over the loop index.
  - When fusing two composites, the “chunk body” for A and the “chunk body” for B are both fixed-size blocks of constant-index statements; matching and merging them is the same as merging two straight-line code blocks (plus predicates if needed).

- **Cost:** Code size for one chunk of size 2^k is 2^k copies of the body. With **bounded** chunk sizes (min/max), we only have a few k values (e.g. 1..6 for max_chunk=64), so the total number of “chunk body” templates is small. Each template is a fixed unrolled block of size 2^k; the compiler generates one such block per chunk size and reuses it for every chunk of that size (parameterized by chunk base). So we get:
  - **Bounded** code growth: only a few unrolled chunk templates (one per allowed 2^k).
  - **Easier** dependency and fusion analysis inside the chunk, because everything is constant-index.

- **When not to unroll inside the chunk:** If 2^k is large (e.g. max_chunk=1024) and the body is big, unrolling can blow up code size for that template. So either keep max_chunk moderate (e.g. 64) when using inner unroll, or make inner unroll optional (e.g. only for chunk sizes ≤ 16) and keep a loop for larger chunks. That way the compiler can still do dependency analysis by unrolling only the “small” chunk sizes and treat larger chunks with a loop (and possibly more conservative fusion).

**Summary:** Full unroll **within** each power-of-2 chunk makes dependency analysis and fusion inside the chunk easier (all indices constant); combining that with **bounded** min/max chunk sizes keeps the number of chunk templates and code size under control. A dedicated **limited parallel loop** construct (with min/max chunk sizes) further simplifies the compiler by making chunk policy and parallelism explicit.

---

## 7. Build Method Summary

1. **Define primitives in pypto IR**  
   Implement each primitive as one or more pypto IR programs using the existing op set (tensor ops, block ops, sync ops) and registration mechanism in `pypto/src/ir/op/`.

2. **Lower to PTO-ISA**  
   Use pypto’s PTO codegen to emit PTO dialect MLIR (PTO-ISA) from that IR. Ensure the emitted ops and types conform to the PTO dialect and semantics defined in **ptoas**.

3. **Compile with ptoas**  
   Feed the generated PTO-ISA (e.g. `.pto` or MLIR) into the **ptoas** toolchain for optimization and target code generation. The resulting binaries are what run on device (or in simulation).

4. **Expose via pypto frontend**  
   Bind the primitive set to the pypto Python API and language layer so that user code can call them by name and compose them; compilation and backend selection remain within the pypto + ptoas pipeline.

5. **Keep semantics backend-agnostic**  
   Do not bake “incore” vs “orchestration” into the primitive library contract; let backends and runtime (e.g. pto-rt2) decide the mapping to hardware.

---

## 8. Relationship to Other Repositories

| Component   | Role relative to PyPTO-Lib |
|------------|-----------------------------|
| **pypto**  | Programming framework: IR, ops, codegen, backend registry, Python frontend. PyPTO-Lib is a **library built on this framework** and bound to this frontend. |
| **ptoas**  | Defines and implements **PTO-ISA** (dialect, ops, passes, lowering). PyPTO-Lib primitives **lower to PTO-ISA** and are compiled by ptoas. |
| **pto-rt2** (simpler) | Runtime for executing compiled task graphs on Ascend (AICPU + AICore). Consumes binaries produced by pypto + ptoas; **does not define** the primitive library. |

---

## 9. Summary

**PyPTO-Lib** is a **tensor-level** primitive library (not a tile-level one): defining yet another function set at the tile level would add little value over PTO-ISA. Primitives are **tensor** ops (max, exp, sum, add, div, etc.); the **compiler** tiles them and lowers to PTO-ISA.

**Three purposes:**

1. **Tiling and PTO-ISA**: The compiler tiles tensor ops and lowers to PTO-ISA; **cast_tensor_to_tile** / **cast_tile_to_tensor** give view-only Tensor↔Tile conversion; **TLOAD** is not inserted until incore boundaries are set.
2. **Tail blocks and padding**: Lowering handles non-divisible dimensions and padding for correct behavior.
3. **Cross-composite loop fusion**: Putting many tile ops in **one** composite (e.g. softmax) is **manual fusion**—one loop by design. The **remaining challenge** is **fusing loops across multiple composite functions** (e.g. `relu(softmax(x))` → one loop that does softmax then relu per tile). The representation and compiler must support **cross-composite** fusion: same iteration space, producer–consumer data flow, and loop merging.

**Incore scope** (Section 4): The user inserts an **incore scope directive** in Python source to mark the boundary between orchestration and incore compute. The scope defines an **anonymous incore function** (no explicit args) and a **call** at that location. The compiler derives **input** (outside, read-only inside), **inout** (outside, modified inside), and **output** (defined outside, unassigned by parent, written inside scope, read by parent after; passed by reference; memory allocated by runtime when incore is called/submitted), and generates a readable name (parent name as prefix) and explicit parameters/call site.

This document describes the **method** of building that library; concrete primitive lists, build scripts, fusion-pass design, and incore-scope implementation can be added in the same folder as the library is implemented.

---

## 10. In-cluster-function-group

This section describes front-end language features for expressing computation that runs on a **cluster** of locally interconnected cores, with in-cluster communication expressed as push/pop instead of store/load.

### 10.1 Cluster as a dummy tensor

A **cluster** is represented as a dummy tensor (or scalar variable) that corresponds to a cluster of locally interconnected cores. On **A5 Ascend** processors, one cluster consists of **2 AIV** and **1 AIC** cores. The cluster is the unit of allocation and scheduling for in-cluster function groups.

### 10.2 In-cluster functions and communication

**In-cluster functions** are a group of functions that communicate with each other using **local interconnect channels**, abstracted as **push** and **pop** operations. Within this group, data communication is expressed as **TPUSH** and **TPOP** operations inside incore functions, instead of **TSTORE** and **TLOAD**. This reflects the fact that data stays on the cluster’s local interconnect rather than going through global memory.

### 10.3 Scope grammar: allocate_cluster

The scope within which all incore functions are treated as one **in-cluster function group** is started by a **blocking** call to the pto runtime:

- **`allocate_cluster`**  
  The pto runtime is called to allocate an available processor cluster. It returns a dummy tensor or a scalar variable **`clusterID`** that identifies the allocated cluster.

- **Blocking semantics**  
  If no free cluster is available, the pto runtime **blocks** the orchestration until a free cluster becomes available. Only after `allocate_cluster` returns does the program proceed with the in-cluster function group.

- **clusterID as input to the group**  
  The **clusterID** is an **input argument** to all functions within this in-cluster function group. It is stored on the **pto runtime task descriptor** and indicates the **only valid cluster** on which a given task may execute.

- **Scheduling guarantee**  
  The pto-runtime scheduler **does not** schedule any task in this group to any other cluster; it always uses the cluster identified by **clusterID**. The scheduler still tracks **data dependencies** between tasks so that correctness and ordering are preserved.

- **End of scope: free cluster**  
  The program does **not** explicitly call a clusterID free API. When the **clusterID** tensor is freed by the runtime (e.g. when it goes out of scope or is deallocated), the **pto-runtime** automatically returns that cluster to the pool of available clusters so it can be allocated again by a subsequent `allocate_cluster` call.

### 10.4 Incore argument types: PIPE_IN and PIPE_OUT

The argument types of **incore functions** are extended to include **PIPE_IN** and **PIPE_OUT**. These denote variables that pass data using **local interconnect pipes**, instead of global-memory tensors.

- **Semantics**  
  **PIPE_IN** and **PIPE_OUT** represent producer–consumer data flow between functions in the in-cluster function group. Data is moved via the cluster’s local interconnect (TPUSH/TPOP), not via global memory.

- **Runtime behavior**  
  The **pto-runtime** does **not** allocate global memory (e.g. ring buffer) for PIPE_IN/PIPE_OUT arguments. They still express **data dependency** between functions, so the scheduler uses them to order tasks and ensure correct execution; only the storage is on the interconnect pipes, not in global memory.

- **Drain invariant**  
  The **programmer** must ensure that every in-cluster function group **completely drains** the interconnect pipe by the end of the scope. In other words: every **push** into the pipe must have a **corresponding pop** that removes that data. No data may remain in the pipe when the scope ends.

- **Tensor map and minimum shape**  
  For now, even though PIPE_IN and PIPE_OUT data pass through the pipes (not global memory), they can still be **treated as normal tensors of minimum shape** so that the **tensor map** can be used to track data dependency between functions.

- **One producer, multiple consumers**  
  If an incore function needs to pass data to **two** (or more) successor functions, it must be expressed as **two separate PIPE_OUT** variables at the function interface—one per consumer. Each PIPE_OUT corresponds to one logical pipe and one consumer.

### 10.5 Summary

- **Cluster**: dummy tensor/scalar representing a set of locally connected cores (e.g. 2 AIV + 1 AIC on A5 Ascend).
- **In-cluster communication**: TPUSH/TPOP within incore functions, instead of TSTORE/TLOAD.
- **Scope**: started by blocking **allocate_cluster**; **clusterID** is passed into every function in the group and recorded in the task descriptor so the runtime schedules those tasks only on that cluster while respecting data dependencies. The program does not explicitly free the cluster; the pto-runtime **automatically** frees it when the **clusterID** tensor is freed by the runtime.
- **PIPE_IN / PIPE_OUT**: incore argument types for data passed over local interconnect pipes; no global memory allocation by the runtime; programmer must ensure the pipe is fully drained (every push has a matching pop). For dependency tracking they are treated as normal tensors of minimum shape in the tensor map; one producer feeding multiple consumers is expressed as multiple separate PIPE_OUT arguments.

---

## 11. block_incore function

This section adds grammar to express a **block_incore** function: an incore function that is executed in **SPMD** (Single Program Multiple Data) manner.

### 11.1 Call arguments: blockdim and block_id

A call to a **block_incore** function has two additional arguments that identify the block and the overall parallelism:

- **blockdim**  
  The **total number of blocks**. The function is invoked once per block, so there are **blockdim** concurrent invocations.

- **block_id**  
  The **index of the block** for this call, in the range `0 .. blockdim-1`. Each invocation receives a distinct **block_id** so that the incore code can compute block-local indices and data.

Every SPMD core (or logical block) runs with a distinct **block_id**; together they form **blockdim** parallel executions of the same incore function.

### 11.2 Use with in-cluster function groups

When a **block_incore** function is used **together with** in-cluster function groups, the orchestration must allocate **enough clusters** to run all blocks. Specifically:

- The number of clusters allocated must **equal blockdim**, so that there is one cluster per block.
- In practice, **allocate_cluster** is used in a way that provides **blockdim** clusters (e.g. the runtime allocates a set of clusters of size **blockdim**, or the program calls allocation so that the resulting **clusterID** or cluster set has cardinality **blockdim**).
- Each of the **blockdim** invocations of the block_incore in-cluster function group then runs on one of these clusters, with **block_id** identifying which block (and thus which cluster) that invocation uses.

This ensures there are enough clusters to execute the SPMD incore function groups without oversubscribing or undersubscribing clusters.

### 11.3 Benefits and orchestration modes

- **Task compression and runtime overhead**  
  The **block_incore** function provides an effective way to **compress the number of tasks** seen by the pto-runtime. Instead of scheduling many fine-grained tasks (e.g. one per tile or per element), the runtime schedules one (or a few) block_incore tasks, each of which runs **blockdim** parallel blocks internally. This **lowers the overhead** of the pto-runtime (fewer task descriptors, less scheduling and dependency tracking at the orchestration level).

- **PyTorch eager execution mode**  
  The **block_incore** function can also be **orchestrated in PyTorch eager execution mode**. In this mode, the program launches SPMD kernels written in the **pyPTO** grammar and compile chain directly from Python, without using the pto-runtime. This gives a **simple path** to run pyPTO-compiled SPMD kernels (e.g. for prototyping or when full task-graph scheduling is not needed), avoiding the complexity of the pto-runtime while still reusing the same pyPTO front-end and compiler.
