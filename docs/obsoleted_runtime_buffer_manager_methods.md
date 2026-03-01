# PTO Runtime Buffer Management

## Overview of PTO Runtime

### Runtime Input

The PTO Runtime accepts two types of functions as input:

1. **Orchestration Functions** - Control flow and task submission logic
2. **InCore Functions** - Computational kernels executed on hardware units

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PTO RUNTIME INPUT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────┐    ┌─────────────────────────────────┐    │
│   │  Orchestration Function │    │      InCore Functions           │    │
│   │  ─────────────────────  │    │      ─────────────────          │    │
│   │  • Control flow logic   │    │  • gemm_tile (Cube)             │    │
│   │  • Memory allocation    │    │  • vector_add (Vector)          │    │
│   │  • Task submission      │    │  • dma_copy (Accelerator)       │    │
│   │  • Dependency building  │    │  • ...                          │    │
│   └─────────────────────────┘    └─────────────────────────────────┘    │
│              │                                    │                      │
│              └──────────────┬─────────────────────┘                      │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │   PTO Runtime   │                                   │
│                    └─────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Orchestration Function

An **Orchestration Function** is a **Turing-complete program** that controls the execution flow:

1. **Supports full programming constructs**:
   - **Function nesting** - Call other orchestration functions
   - **Recursion** - Recursive function calls
   - **Loops** - for, while, do-while
   - **Conditionals** - if-else, switch-case
   - **Local variables and data structures**

2. **Operates on device-side buffers** - Input/output buffers are already in device memory
3. **Allocates intermediate buffers** - For temporary data during computation
4. **Submits tasks via asynchronous InCore function calls** - Each async call creates a task
5. **Builds dynamic task dependency graph** - Through producer-consumer relationships of data

The orchestration function is executed by the **Orchestrator**, which can run on either:
- **Host CPU** - Lower latency for task submission, but requires host-device communication
- **Device AICPU** - Closer to compute units, but AICPU resources are limited

The optimal choice depends on workload characteristics (discussed in later sections).

```c
// Orchestration Function Signature
void orchestration_func(
    PTORuntime* rt,           // Runtime context
    void** input_buffers,     // Device-side input buffers
    void** output_buffers,    // Device-side output buffers
    void* params              // Scalar parameters
);

// Example: Orchestration with nesting, loops, conditionals, recursion
void transformer_layer(PTORuntime* rt, void** inputs, void** outputs, LayerParams* p) {
    // Conditional: choose attention type
    if (p->use_flash_attention) {
        flash_attention(rt, inputs, outputs, p);  // Nested orchestration call
    } else {
        standard_attention(rt, inputs, outputs, p);
    }
    
    // Loop: MLP with multiple layers
    for (int i = 0; i < p->mlp_layers; i++) {
        mlp_block(rt, ...);  // Nested orchestration call
    }
    
    // Recursion example: tree reduction
    if (p->need_reduction) {
        tree_reduce(rt, outputs, p->reduction_depth);  // Recursive call
    }
}
```

---

### Runtime Startup and Host-Device Memory Management

**Host-Device memory transfer is handled by the Runtime startup code**, not the orchestration function itself. The orchestration function operates entirely on device-side buffers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RUNTIME EXECUTION FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [Host Side]                          [Device Side]                     │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   RUNTIME STARTUP CODE                           │   │
│   │                                                                   │   │
│   │  1. Allocate device buffers for inputs/outputs                   │   │
│   │  2. Copy input data: Host → Device (H2D)                         │   │
│   │  3. Launch orchestration function on AICPU                       │   │
│   │                                                                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│                ┌────────────────────────────────────┐                    │
│                │     Orchestration Function         │                    │
│                │     (executes on device AICPU)     │                    │
│                │                                    │                    │
│                │  • Allocate intermediate buffers   │                    │
│                │  • Submit tasks (async calls)      │                    │
│                │  • Build dependency graph          │                    │
│                │  • Wait for all tasks complete     │                    │
│                │                                    │                    │
│                └────────────────────────────────────┘                    │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   RUNTIME SHUTDOWN CODE                          │   │
│   │                                                                   │   │
│   │  4. Copy output data: Device → Host (D2H)                        │   │
│   │  5. Free device buffers                                          │   │
│   │  6. Return to caller                                             │   │
│   │                                                                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```c
// Runtime API (called from host)
void pto_runtime_execute(
    PTORuntime* rt,
    OrchestrationFunc orch_func,
    void** host_inputs,       // Host-side input buffers
    void** host_outputs,      // Host-side output buffers
    void* params
) {
    // === STARTUP: Host-side runtime code ===
    
    // 1. Allocate device buffers
    void** device_inputs = allocate_device_buffers(host_inputs, ...);
    void** device_outputs = allocate_device_buffers(host_outputs, ...);
    
    // 2. Copy inputs: Host → Device
    for (int i = 0; i < num_inputs; i++) {
        memcpy_h2d(device_inputs[i], host_inputs[i], sizes[i]);
    }
    
    // 3. Launch orchestration on device AICPU
    launch_on_device(orch_func, rt, device_inputs, device_outputs, params);
    
    // 4. Wait for orchestration and all tasks to complete
    wait_for_completion(rt);
    
    // === SHUTDOWN: Host-side runtime code ===
    
    // 5. Copy outputs: Device → Host
    for (int i = 0; i < num_outputs; i++) {
        memcpy_d2h(host_outputs[i], device_outputs[i], sizes[i]);
    }
    
    // 6. Free device buffers
    free_device_buffers(device_inputs);
    free_device_buffers(device_outputs);
}
```

#### Task Dependency Graph Construction

The orchestration function dynamically constructs a **task dependency graph** through data producer-consumer relationships:

```c
// Example: Orchestration function building dependency graph
void bgemm_orchestration(PTORuntime* rt, ...) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                // Declare intermediate buffer P - will be allocated by runtime
                // IMPORTANT: P is a pointer that will receive runtime-allocated address
                void* P = NULL;
                
                // Task 1: gemm_tile produces P (async call)
                // OUTPUT uses &P (pointer-to-pointer) so runtime can write allocated address
                pto_submit_task(rt, "gemm_tile", {
                    PTO_INPUT(A[m,k], tile_idx_a, tile_size),
                    PTO_INPUT(B[k,n], tile_idx_b, tile_size),
                    PTO_OUTPUT(&P, tile_idx_p, tile_size)  // Runtime allocates and writes to P
                });
                // After submit_task returns, P now contains runtime-allocated address
                
                // Task 2: tile_add consumes P (async call)
                // INPUT uses P directly (now contains the runtime-allocated address)
                // Dependency: Task 2 depends on Task 1 (TensorMap tracks P's producer)
                pto_submit_task(rt, "tile_add", {
                    PTO_INPUT(C[m,n], tile_idx_c, tile_size),
                    PTO_INPUT(P, tile_idx_p, tile_size),   // Uses runtime-allocated address
                    PTO_INOUT(&C[m,n], tile_idx_c, tile_size)
                });
                
            }  // End of scope - P's fanout decremented
        }
    }
}
```

The runtime uses **TensorMap** to track producer-consumer relationships and automatically build task dependencies. TensorMap uses a ring buffer pool with lazy invalidation - entries for retired tasks are automatically ignored, keeping the map size bounded.

---

### InCore Function

An **InCore Function** is a computational kernel that:

1. **Runs to completion** on a single core or hardware accelerator
2. **Is compiled to the target's physical ISA** (Instruction Set Architecture)
3. **Accepts input, output, and in-out buffers**
4. **Executes atomically** - No preemption during execution

```c
// InCore Function for programmable cores (AICore Cube/Vector)
void incore_func(
    void** input_buffers,     // Read-only input data
    void** output_buffers,    // Write-only output data
    void** inout_buffers,     // Read-write data (e.g., accumulation)
    void* params              // Scalar parameters
);
```

#### Fixed-Function Accelerators

For **fixed-function accelerators** (DMA, special-purpose units), the function is defined by the hardware. The runtime uses a **Work Request Descriptor** to specify:

```c
// Work Request Descriptor for fixed-function accelerators
typedef struct {
    uint32_t opcode;              // Accelerator operation type
    void** input_buffers;         // Input buffer pointers
    void** output_buffers;        // Output buffer pointers
    void** inout_buffers;         // In-out buffer pointers
    uint32_t params[16];          // Operation-specific parameters
} WorkRequestDescriptor;
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INCORE FUNCTION TYPES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────┐  ┌───────────────────────────────┐  │
│   │   Programmable Core           │  │   Fixed-Function Accelerator  │  │
│   │   (AICore Cube/Vector)        │  │   (DMA, ...)                  │  │
│   ├───────────────────────────────┤  ├───────────────────────────────┤  │
│   │                               │  │                               │  │
│   │  • Compiled kernel binary     │  │  • Work Request Descriptor    │  │
│   │  • Flexible computation       │  │  • Fixed operation set        │  │
│   │  • ISA: AICore instructions   │  │  • Hardware-defined params    │  │
│   │                               │  │                               │  │
│   │  Example:                     │  │  Example:                     │  │
│   │    gemm_tile(A, B) → C        │  │    DMA_COPY(src, dst, size)   │  │
│   │    vector_add(X, Y) → Z       │  │    SCATTER(src, indices, dst) │  │
│   │                               │  │                               │  │
│   └───────────────────────────────┘  └───────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Runtime Roles

The PTO Runtime consists of three key roles:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PTO RUNTIME ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┐                                                    │
│   │   ORCHESTRATOR  │  (Host CPU or Device AICPU)                        │
│   │   ─────────────  │                                                    │
│   │  • Execute orchestration function                                    │
│   │  • Allocate intermediate buffers                                     │
│   │  • Submit tasks (async InCore calls)                                 │
│   │  • Build dependency graph via TensorMap                              │
│   │  • Manage buffer scopes                                              │
│   └────────┬────────┘                                                    │
│            │ submit tasks                                                │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │    SCHEDULER    │  (Device AICPU)                                    │
│   │   ─────────────  │                                                    │
│   │  • Maintain task ready queue                                         │
│   │  • Resolve task dependencies (fanin tracking)                        │
│   │  • Dispatch ready tasks to workers                                   │
│   │  • Track buffer lifecycle (fanout tracking)                          │
│   │  • Release buffers when fanout reaches zero                          │
│   │  • Release task metadata when no longer needed                       │
│   └────────┬────────┘                                                    │
│            │ dispatch tasks                                              │
│            ▼                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                              WORKERS                                   │ │
│   ├───────────────┬───────────────┬───────────────┬───────────────────────┤ │
│   │ AICore_CUBE   │ AICore_VECTOR │    AI_CPU     │    ACCELERATORS       │ │
│   │ ───────────── │ ───────────── │ ───────────── │ ───────────────────── │ │
│   │ • Matrix ops  │ • Vector ops  │ • Scalar ops  │ • DMA Engine          │ │
│   │ • GEMM tiles  │ • Element-wise│ • Complex ctrl│ • Fixed-function HW   │ │
│   │ • Convolution │ • Reduction   │ • Data-depend │ • Special-purpose     │ │
│   └───────────────┴───────────────┴───────────────┴───────────────────────┘ │
│                             │                                            │
│                             │ signal completion                          │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │    SCHEDULER    │  (receives completion signal)     │
│                    └─────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Role 1: Orchestrator

The **Orchestrator** executes the orchestration function and can run on either **Host CPU** or **Device AICPU**:

| Execution Location | Pros | Cons |
|-------------------|------|------|
| **Host CPU** | Abundant compute/memory, easier debugging | Higher latency for task submission (PCIe) |
| **Device AICPU** | Low latency to scheduler/workers | Limited AICPU resources, harder debugging |

The choice depends on workload characteristics:
- **Host CPU**: Better for complex control flow, large data structures, or when AICPU is bottleneck
- **Device AICPU**: Better for fine-grained tasks where submission latency matters

| Responsibility | Description |
|----------------|-------------|
| Execute orchestration function | Run the Turing-complete control flow (loops, conditionals, recursion, nesting) |
| Memory allocation | Allocate intermediate buffers for computation |
| Task submission | Create tasks via async InCore function calls |
| Graph construction | Build dependency graph using TensorMap |
| Scope management | Track buffer scopes for lifecycle management |

#### Role 2: Scheduler

The **Scheduler** executes on an AICPU thread and is responsible for:

| Responsibility | Description |
|----------------|-------------|
| Ready queue management | Maintain queue of tasks with satisfied dependencies |
| Dependency resolution | Track fanin count, decrement when producers complete |
| Task dispatch | Send ready tasks to appropriate workers |
| Buffer lifecycle | Track fanout count, release when all consumers done |
| Resource recycling | Return freed buffers and tasks to pools |

#### Role 3: Workers

**Workers** execute on hardware compute units:

| Worker Type | Hardware | Capabilities |
|-------------|----------|--------------|
| AICore_CUBE | Cube Unit | Matrix multiplication, convolution |
| AICore_VECTOR | Vector Unit | Element-wise ops, activation, reduction |
| AI_CPU | Device AICPU | Scalar ops, complex control flow, data-dependent ops |
| Accelerators | DMA, etc. | Data movement, fixed-function operations |

Workers poll for tasks, execute them to completion, and signal the scheduler upon completion.

**Note**: AI_CPU workers are distinct from the Orchestrator and Scheduler roles. While Orchestrator/Scheduler manage control flow and task dispatch, AI_CPU workers execute specific computational tasks (e.g., operations that are inefficient on Cube/Vector units, or require complex branching).

---

### Task Dependency Data Structures

To fully express the bidirectional dependency graph, each task maintains:

```c
typedef struct Task {
    int32_t kernel_id;            // Which InCore function to execute
    
    // ========== FANIN (for dependency resolution) ==========
    int32_t fanin_count;          // Number of unsatisfied dependencies
    int32_t* fanin_list;          // List of producer task IDs
    int32_t fanin_list_size;      // Size of fanin_list
    
    // ========== FANOUT (for buffer lifecycle management) ==========
    int32_t fanout_count;         // Number of consumer tasks
    int32_t* fanout_list;         // List of consumer task IDs
    int32_t fanout_list_size;     // Size of fanout_list
    
    // ========== BUFFER ASSOCIATION ==========
    PackedOutputBuffer* output_buf;  // Buffers produced by this task
    
    // ... other fields
} Task;
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL TASK DEPENDENCY STRUCTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│         Task 0 (Producer)                    Task 2 (Producer)          │
│         ┌─────────────────┐                  ┌─────────────────┐        │
│         │ fanout_count: 2 │                  │ fanout_count: 1 │        │
│         │ fanout_list:    │                  │ fanout_list:    │        │
│         │   [1, 3]        │                  │   [3]           │        │
│         └────────┬────────┘                  └────────┬────────┘        │
│                  │                                    │                  │
│          ┌───────┴───────┐                           │                  │
│          ▼               ▼                           ▼                  │
│   ┌─────────────┐  ┌─────────────────────────────────────┐              │
│   │   Task 1    │  │              Task 3                 │              │
│   │ (Consumer)  │  │            (Consumer)               │              │
│   ├─────────────┤  ├─────────────────────────────────────┤              │
│   │fanin_count:1│  │ fanin_count: 2                      │              │
│   │fanin_list:  │  │ fanin_list: [0, 2]                  │              │
│   │  [0]        │  │                                     │              │
│   └─────────────┘  └─────────────────────────────────────┘              │
│                                                                          │
│   Usage:                                                                │
│   • fanin_count/fanin_list  → Dependency resolution (scheduling)       │
│   • fanout_count/fanout_list → Buffer lifecycle management             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Buffer Lifecycle with Scope Management

Buffer lifecycle is determined by two factors:

1. **Fanout count reaches zero** - All consumer tasks have completed
2. **Scope ends** - The buffer's defining scope in the orchestration function exits

#### Implementation: Fanout Count Starts from Scope Depth

To handle nested scope-based lifecycle, **fanout_count starts from the current scope stack depth**.

**Key Design Goal: Simplify `scope_end()` Processing**

By initializing `fanout_count = scope_depth`, the `scope_end()` implementation becomes very simple:
- Just iterate over all buffers from `scope_begin_pos` to `scope_end_pos`
- Decrement each buffer's fanout_count by 1
- Pop the scope stack

No need to filter by depth or track ownership — every buffer in the range gets exactly one decrement from each enclosing scope.

```c
// When buffer is allocated
void* pto_buffer_alloc(PTORuntime* rt, size_t size) {
    void* buf = buffer_pool_alloc(rt, size);
    
    // Get current scope depth (number of active scopes)
    int32_t scope_depth = rt->scope_stack_top + 1;  // 0-indexed stack
    
    BufferMetadata* meta = get_metadata(buf);
    meta->fanout_count = scope_depth;  // Each enclosing scope holds a reference
    
    return buf;
}
```

**Why this works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCOPE RANGE-BASED DECREMENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   scope_begin()  ─────────────────────────────► push pos=0 to stack         │
│   │                                             scope_stack = [0]            │
│   │   Buffer A allocated (pos=0)                A.fanout = 1                 │
│   │   Buffer B allocated (pos=1)                B.fanout = 1                 │
│   │                                                                          │
│   │   scope_begin()  ─────────────────────────► push pos=2 to stack         │
│   │   │                                         scope_stack = [0, 2]         │
│   │   │   Buffer C allocated (pos=2)            C.fanout = 2                 │
│   │   │   Buffer D allocated (pos=3)            D.fanout = 2                 │
│   │   │                                                                      │
│   │   scope_end()  ───────────────────────────► pop: begin=2, end=4         │
│   │   │   for i in [2,4): buf[i].fanout--       C.fanout: 2→1               │
│   │   │                                         D.fanout: 2→1               │
│   │   │                                         scope_stack = [0]            │
│   │                                                                          │
│   scope_end()  ───────────────────────────────► pop: begin=0, end=4         │
│       for i in [0,4): buf[i].fanout--           A.fanout: 1→0 → FREE        │
│                                                 B.fanout: 1→0 → FREE        │
│                                                 C.fanout: 1→0 → FREE        │
│                                                 D.fanout: 1→0 → FREE        │
│                                                                              │
│   Note: Each buffer receives exactly scope_depth decrements from scopes     │
│         Plus decrements from consumers when tasks complete                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation details:**

```c
// When task adds this buffer as input (consumer)
void pto_task_add_input(PTORuntime* rt, int32_t task_id, void* buffer, ...) {
    BufferMetadata* meta = get_metadata(buffer);
    meta->fanout_count++;      // Increment for each consumer
    
    // ... add dependency
}

// When scope ends - simple range-based decrement
void pto_scope_end(PTORuntime* rt) {
    // Pop scope stack to get the begin position
    int32_t scope_begin_pos = rt->scope_stack[rt->scope_stack_top--];
    int32_t scope_end_pos = rt->buffer_count;  // Current position
    
    // Simply decrement fanout for ALL buffers in [begin_pos, end_pos)
    // No filtering needed - every buffer in this range was allocated 
    // when this scope was active, so it has a reference from this scope
    for (int32_t i = scope_begin_pos; i < scope_end_pos; i++) {
        BufferMetadata* meta = &rt->buffer_metadata[i];
        meta->fanout_count--;  // This scope releases its reference
        
        if (meta->fanout_count == 0) {
            buffer_pool_free(rt, i);
        }
    }
}

// Why fanout_count = scope_depth works:
//   - Buffer allocated at depth D has D enclosing scopes
//   - Each scope_end() decrements fanout for buffers in its range
//   - After all D scopes end, buffer receives exactly D decrements
//   - Combined with consumer decrements, buffer freed when all refs gone

// When consumer task completes
void on_task_complete(PTORuntime* rt, int32_t task_id) {
    for (each input buffer of task) {
        BufferMetadata* meta = get_metadata(buffer);
        meta->fanout_count--;  // Consumer releases its reference
        
        if (meta->fanout_count == 0) {
            buffer_pool_free(rt, buffer);
        }
    }
}
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 BUFFER LIFECYCLE WITH SCOPE MANAGEMENT                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   void orchestration() {                                                │
│       scope_begin();              // depth = 1                          │
│       void* A = alloc();          // A.fanout = 1 (depth=1)             │
│       {                                                                  │
│           scope_begin();          // depth = 2                          │
│           void* P = alloc();      // P.fanout = 2 (depth=2)             │
│                                                                          │
│           submit(gemm, out=P);    // P.fanout = 2 (no change, producer) │
│           submit(add, in=P);      // P.fanout = 3 (2 + 1 consumer)      │
│                                                                          │
│           scope_end();            // P.fanout = 3 - 1 = 2 (depth 2 ref) │
│       }                           // P not freed (consumer + outer scope)│
│                                                                          │
│       // ... later, add task completes                                  │
│       // P.fanout = 2 - 1 = 1 (consumer releases)                       │
│                                                                          │
│       scope_end();                // P.fanout = 1 - 1 = 0 → P freed!    │
│   }                               // A.fanout = 1 - 1 = 0 → A freed     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### The Buffer Lifecycle Challenge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BUFFER LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ALLOCATE ──► WRITE ──► READ ──► READ ──► ... ──► LAST READ ──► FREE      │
│       │          │         │        │                   │          │        │
│       │          │         │        │                   │          │        │
│   ref_count=1    │     ref_count++              ref_count--   ref_count=0   │
│   (scope ref)    │   (each consumer)          (consumer done)    ↓          │
│                  │                                            RELEASE       │
│              Producer task                      Last consumer task          │
│                                                                              │
│   ════════════════════════════════════════════════════════════════════      │
│                                                                              │
│                      SCOPE EXIT ALSO DECREMENTS REF_COUNT                   │
│                                                                              │
│   {  // Scope begins                                                        │
│       P = alloc()        // P.ref_count = 1 (scope holds reference)         │
│       submit(out=P)      // P.ref_count = 1 (producer, no change)           │
│       submit(in=P)       // P.ref_count = 2 (consumer adds ref)             │
│       submit(in=P)       // P.ref_count = 3 (another consumer)              │
│   }  // Scope ends       // P.ref_count = 3 - 1 = 2 (scope releases ref)    │
│                                                                              │
│   // Task 1 completes    // P.ref_count = 2 - 1 = 1                         │
│   // Task 2 completes    // P.ref_count = 1 - 1 = 0 → FREE!                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Reference Count Events:**

| Event | ref_count Change | Description |
|-------|------------------|-------------|
| Buffer allocated | = 1 | Scope holds initial reference |
| Consumer task added | +1 | Each consumer increments ref |
| **Scope exits** | **-1** | **Scope releases its reference** |
| Consumer task completes | -1 | Consumer releases its reference |
| ref_count reaches 0 | → FREE | Buffer can be safely released |

**Key Tradeoffs:**
- Release too early → Use-after-free, data corruption
- Release too late → Memory bloat, OOM for large models
- Track liveness precisely → Runtime overhead (ref counting, analysis)

---

### Runtime API and Internal Components

### Architecture Overview

The PTO Runtime provides a **simplified API** with only 4 functions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PTO RUNTIME                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ╔═══════════════════════════════════════════════════════════════════════╗ │
│   ║                      RUNTIME API (Public Interface)                   ║ │
│   ╠═══════════════════════════════════════════════════════════════════════╣ │
│   ║                                                                       ║ │
│   ║  Called by Orchestrator:              Called by Workers:              ║ │
│   ║  ┌─────────────────────────────┐      ┌─────────────────────────┐    ║ │
│   ║  │ • pto_scope_begin()         │      │ • pto_task_complete()   │    ║ │
│   ║  │ • pto_submit_task(kernel,   │      │                         │    ║ │
│   ║  │       params[])             │      │                         │    ║ │
│   ║  │ • pto_scope_end()           │      │                         │    ║ │
│   ║  └─────────────────────────────┘      └─────────────────────────┘    ║ │
│   ║                                                                       ║ │
│   ╚═══════════════════════════════════════════════════════════════════════╝ │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    INTERNAL COMPONENTS                               │   │
│   │                                                                      │   │
│   │   ORCHESTRATOR-OWNED (Write: Orchestrator, Read-Only: Scheduler)     │   │
│   │   ══════════════════════════════════════════════════════════════     │   │
│   │   ┌─────────────────┬─────────────────┬────────────────────────┐    │   │
│   │   │  TaskDescriptor │   TensorMap     │     DependencyGraph    │    │   │
│   │   │  ─────────────  │   ─────────     │     ───────────────    │    │   │
│   │   │  • kernel_id    │   • Ring buffer │     • fanin_list[]     │    │   │
│   │   │  • worker_type  │     pool        │     • fanout_list[]    │    │   │
│   │   │  • output_buf   │   • Lazy        │     • fanin_count[]    │    │   │
│   │   │  • scope_depth  │     invalidate  │     • fanout_count[]   │    │   │
│   │   │                 │   • Region→task │     • scope_stack      │    │   │
│   │   └─────────────────┴─────────────────┴────────────────────────┘    │   │
│   │                                                                      │   │
│   │   SCHEDULER-OWNED (Dynamic during execution)                         │   │
│   │   ══════════════════════════════════════════                         │   │
│   │   ┌─────────────────┬─────────────────────────┬────────────────┐    │   │
│   │   │  RefCountState  │   ReadyQueues           │  ResourcePools │    │   │
│   │   │  ─────────────  │   (per worker type)     │  ────────────  │    │   │
│   │   │  •fanin_refcnt[]│   ┌──────────────────┐  │  •buffer_free  │    │   │
│   │   │  •fanout_refcnt[]│  │ CUBE_queue       │  │  •task_free    │    │   │
│   │   │  • task_state[] │   │ VECTOR_queue     │  │  •tile_pool    │    │   │
│   │   │                 │   │ AI_CPU_queue     │  │                │    │   │
│   │   │                 │   │ ACCELERATOR_queue│  │                │    │   │
│   │   │                 │   └──────────────────┘  │                │    │   │
│   │   └─────────────────┴─────────────────────────┴────────────────┘    │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Runtime API

#### API Summary

| API | Caller | Description |
|-----|--------|-------------|
| `pto_scope_begin()` | Orchestrator | Begin scope, push current position to scope stack |
| `pto_submit_task(kernel, params[])` | Orchestrator | Submit task with InCore function + parameters; internally handles input/output/inout registration, dependency tracking, and buffer allocation |
| `pto_scope_end()` | Orchestrator | End scope, decrement fanout for all tasks in `[begin_pos, end_pos)` range |
| `pto_task_complete(task_id)` | Worker | Signal task completion, update fanin/fanout counters |

---

### Data Structure Separation: Orchestrator vs Scheduler

The Runtime data structures are divided into two parts based on **write ownership**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA STRUCTURE OWNERSHIP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ORCHESTRATOR-OWNED (Read-Only for Scheduler)                              │
│   ════════════════════════════════════════════════                          │
│   Written by: Orchestrator (pto_scope_begin(), pto_submit_task())           │
│   - fanout_list, fanout_count can be incrementally updated (append/incr)    │
│   - As new consumers are submitted, producer's fanout data grows            │
│   Read by: Scheduler (never modified by Scheduler)                          │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  TaskDescriptor[]     - Task definitions (kernel, worker_type)      │   │
│   │  TensorMap            - Producer lookup (ring buffer, lazy invalid) │   │
│   │  FaninList[]          - Static list of producer task IDs            │   │
│   │  FanoutList[]         - Static list of consumer task IDs            │   │
│   │  fanin_count[]        - Total dependency count (static)             │   │
│   │  fanout_count[]       - Total consumers + scope_depth (static)      │   │
│   │  ScopeStack           - Stack of scope begin positions              │   │
│   │  OutputBufferMap[]    - Task → output buffer associations           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   SCHEDULER-OWNED (Dynamic during execution)                                │
│   ══════════════════════════════════════════                                │
│   Written by: Scheduler during execution                                    │
│   Read by: Scheduler only                                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  fanin_refcount[]     - Remaining unsatisfied dependencies          │   │
│   │  fanout_refcount[]    - Remaining consumers + scope references      │   │
│   │  task_state[]         - PENDING / READY / RUNNING / COMPLETED       │   │
│   │  ready_queues[]       - Per-worker-type queues (CUBE, VECTOR, etc.) │   │
│   │  buffer_free_list     - Available buffer slots                      │   │
│   │  task_free_list       - Available task slots                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   TASK STATE TRANSITIONS:                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │  PENDING ──► READY ──► RUNNING ──► COMPLETED ──► CONSUMED           │   │
│   │     │          │          │            │             │              │   │
│   │   fanin     fanin      dispatch    execution    fanout_refcount     │   │
│   │   refcount  ==fanin    to worker    done        ==fanout_count      │   │
│   │   <fanin    _count                              (release buffers)   │   │
│   │   _count                                                            │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   LIFECYCLE CONDITIONS (Scheduler checks):                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PENDING→READY:    fanin_refcount == fanin_count                    │   │
│   │                    (all producers have completed)                   │   │
│   │                                                                     │   │
│   │  COMPLETED→CONSUMED: fanout_refcount == fanout_count                │   │
│   │                      (all consumers + scopes have released)         │   │
│   │                      → release output buffers                       │   │
│   │                                                                     │   │
│   │  Initialization: fanin_refcount  = 0  (increments as producers done)│   │
│   │                  fanout_refcount = 0  (increments as consumers done)│   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   KEY PRINCIPLE:                                                            │
│   • Orchestrator can incrementally update fanout_list/fanout_count          │
│   • Scheduler treats Orchestrator data as READ-ONLY                         │
│   • Scheduler refcounts start from 0, increment towards target              │
│   • Comparison with Orchestrator counts determines lifecycle transitions    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Core Data Structures

    #### Orchestrator Data Structures (Read-Only for Scheduler)

```c
// ========== TASK DESCRIPTOR (Read-Only for Scheduler) ==========
// Orchestrator can incrementally update fanout_list/fanout_count
// as new consumer tasks are submitted
typedef struct TaskDescriptor {
    int32_t task_id;              // Unique task identifier
    int32_t kernel_id;            // Which InCore function to execute
    int32_t worker_type;          // Target: CUBE, VECTOR, AI_CPU, ACCELERATOR
    
    // Input/Output buffer descriptors
    TensorRegion* inputs;         // Array of input tensor regions
    int32_t num_inputs;
    TensorRegion* outputs;        // Array of output tensor regions
    int32_t num_outputs;
    TensorRegion* inouts;         // Array of in-out tensor regions
    int32_t num_inouts;
    
    // Dependency graph (static, determined at submission time)
    int32_t* fanin_list;          // Array of producer task IDs
    int32_t fanin_list_size;
    int32_t* fanout_list;         // Array of consumer task IDs
    int32_t fanout_list_size;
    
    // Static counts (used to initialize Scheduler's refcounts)
    int32_t fanin_count;          // Total number of dependencies
    int32_t fanout_count;         // Total consumers + scope_depth
    
    // Scope information
    int32_t scope_depth;          // Depth of scope when task was created
    
    // Packed output buffer (allocated from GM Heap)
    // All outputs packed into single contiguous buffer
    void*    packed_buffer_base;  // Start of packed buffer in GM Heap
    void*    packed_buffer_end;   // End of packed buffer (used for heap reclamation)
    int32_t  output_offsets[MAX_OUTPUTS];  // Offset of each output within packed buffer
    
} TaskDescriptor;

// ========== TENSOR REGION ==========
typedef struct TensorRegion {
    void* base_ptr;               // Buffer base pointer
    int32_t tile_index;           // Tile index within buffer
    int32_t offset;               // Byte offset within tile
    int32_t size;                 // Region size in bytes
} TensorRegion;

// ========== TENSOR MAP (Ring Buffer Design with Lazy Invalidation) ==========
// 
// Key insight: Tasks are created and retired in FIFO order.
// TensorMap entries become stale when their producer task retires.
// Instead of explicit removal, we use lazy invalidation:
//   - Entry is valid only if producer_task_id >= last_task_alive
//   - Stale entries are simply ignored during lookup
//   - Pool wraps around, overwriting stale entries automatically
//
// ========== CRITICAL: OVERLAPPING REGION DETECTION ==========
//
// The TensorMap must detect dependencies not only for EXACT region matches,
// but also for OVERLAPPING sub-tensor regions. Two regions create a dependency
// if they share the same base tensor (raw_tensor pointer) AND their sub-regions
// (defined by offset and shape/size) partially or fully overlap.
//
// Overlap Detection Rules:
//   1. Same base_ptr (raw_tensor): REQUIRED for any dependency
//   2. Region overlap: Check if [offset1, offset1+size1) ∩ [offset2, offset2+size2) ≠ ∅
//   3. Dependency types:
//      - RAW (Read-After-Write): Consumer reads region that overlaps producer's output
//      - WAR (Write-After-Read): Producer writes region that overlaps consumer's input
//      - WAW (Write-After-Write): Both write to overlapping regions
//
// Example:
//   Task A writes: base=X, offset=0, size=256     (bytes 0-255)
//   Task B reads:  base=X, offset=128, size=256   (bytes 128-383)
//   → Overlap detected (bytes 128-255), B depends on A
//
// Implementation Options:
//   1. Interval Tree: O(log n + k) lookup, O(log n) insert
//      - Best for many overlapping regions
//   2. Tile-based bucketing: Hash by (base_ptr, tile_index)
//      - Efficient when tiles rarely overlap across tile boundaries
//   3. Conservative: Assume any same-base regions conflict
//      - Simplest, may create unnecessary dependencies
//
// Current implementation uses tile-based bucketing with exact offset match.
// For production use, consider upgrading to interval-based overlap detection.
//
typedef struct TensorMapEntry {
    TensorRegion region;
    int32_t producer_task_id;     // Used for validity check
    int32_t next_in_bucket;       // Offset to next entry in hash bucket (-1 = end)
    int32_t next_in_task;         // Offset to next entry for same task (-1 = end)
    bool    in_bucket;            // True if entry is linked in a bucket chain
                                  // CRITICAL: Must be set false before overwriting!
} TensorMapEntry;

typedef struct TensorMap {
    // Hash table buckets (fixed size, power of 2)
    int32_t* buckets;             // Array of offsets into entry_pool (-1 = empty)
    int32_t num_buckets;          // Must be power of 2 for fast modulo
    
    // Entry pool as ring buffer
    TensorMapEntry* entry_pool;   // Ring buffer of entries
    int32_t pool_size;            // Total pool capacity
    int32_t pool_head;            // Next allocation position (wraps around)
    
    // Per-task entry tracking (for efficient bucket cleanup)
    int32_t* task_entry_head;     // Per-task head offset (-1 = no entries)
                                  // Indexed by task_id % TASK_WINDOW_SIZE
    
    // Validity threshold (read from shared memory)
    int32_t last_task_alive;      // Cached value for validity checks
    
} TensorMap;

// ========== TENSOR MAP OPERATIONS (Ring Buffer with Lazy Invalidation) ==========

// Hash function for tensor region
// 
// CRITICAL DESIGN DECISION: Hash ONLY by base_ptr
// ================================================
// 
// For overlap detection to work correctly, ALL regions accessing the same
// base tensor MUST be in the SAME hash bucket. This allows the lookup
// function to find and check for overlaps.
//
// If we included offset in the hash, overlapping regions with different
// offsets would end up in different buckets and never be compared!
//
// Example:
//   Region A: base=X, offset=0,   size=256  → hash(X) → bucket 5
//   Region B: base=X, offset=128, size=256  → hash(X) → bucket 5 (SAME!)
//   Now lookup can detect overlap [128, 256)
//
// Trade-off: This increases average chain length for tensors with many
// sub-regions, but correctness requires all same-base regions be comparable.
//
static inline uint32_t tensormap_hash(TensorMap* tm, TensorRegion* region) {
    // Hash ONLY by base_ptr for correct overlap detection
    // All regions of the same tensor will be in the same bucket
    uint64_t key = (uint64_t)region->base_ptr;
    
    // Improve distribution by mixing bits (base_ptr often has low-order zeros)
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    return (uint32_t)(key & (tm->num_buckets - 1));
}

// Check if entry is valid (producer task has not retired)
static inline bool tensormap_entry_valid(TensorMap* tm, TensorMapEntry* entry) {
    return entry->producer_task_id >= tm->last_task_alive;
}

// Update validity threshold from shared memory
static inline void tensormap_sync_validity(TensorMap* tm, SharedMemoryHeader* sm) {
    tm->last_task_alive = sm->last_task_alive;
}

// Lookup producer for a tensor region
// Returns producer_task_id if found and valid, -1 otherwise
//
// ========== OPTIMIZATION: Chain Truncation on Stale Entry ==========
//
// Key insight: Since we always INSERT at HEAD, the chain is naturally 
// sorted by task_id from NEWEST to OLDEST.
//
// Invariant: For any chain, task_id[i] > task_id[i+1] (decreasing order)
//
// Implication: If entry[i] is STALE (task_id < last_task_alive), then
// ALL subsequent entries entry[i+1], entry[i+2], ... are also STALE
// (they have even smaller task_ids).
//
// Optimization: When we encounter the FIRST stale entry, we can:
//   1. Stop traversal immediately (no valid entries after this)
//   2. Optionally truncate the chain here (lazy cleanup)
//
int32_t tensormap_lookup(TensorMap* tm, TensorRegion* region) {
    uint32_t bucket = tensormap_hash(tm, region);
    int32_t* prev_ptr = &tm->buckets[bucket];  // For optional truncation
    int32_t offset = *prev_ptr;
    
    while (offset >= 0) {
        TensorMapEntry* entry = &tm->entry_pool[offset];
        
        // Check validity first
        if (!tensormap_entry_valid(tm, entry)) {
            // ========== STALE ENTRY: Truncate chain here ==========
            // All subsequent entries are guaranteed to be stale too!
            // Truncate: unlink this and all following entries
            *prev_ptr = -1;  // Terminate chain at previous entry
            
            // Mark truncated entries as not in bucket (for correct reuse)
            while (offset >= 0) {
                TensorMapEntry* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            
            return -1;  // Not found (and cleaned up stale tail)
        }
        
        // Entry is valid - check if region OVERLAPS
        // 
        // Since we hash only by base_ptr, all entries in this bucket have
        // the potential to overlap with our query region. We must check:
        //   1. Same base_ptr (raw tensor pointer)
        //   2. Same tile_index (different tiles are disjoint memory regions)
        //   3. Byte ranges [offset, offset+size) intersect within the tile
        //
        // Overlap condition (1D interval intersection):
        //   [e_start, e_end) ∩ [r_start, r_end) ≠ ∅
        //   ⟺ (e_start < r_end) AND (r_start < e_end)
        //
        if (entry->region.base_ptr == region->base_ptr &&
            entry->region.tile_index == region->tile_index) {
            
            int32_t e_start = entry->region.offset;
            int32_t e_end = e_start + entry->region.size;
            int32_t r_start = region->offset;
            int32_t r_end = r_start + region->size;
            
            // Check for overlap within the same tile
            if (e_start < r_end && r_start < e_end) {
                return entry->producer_task_id;  // FOUND (overlapping regions)
            }
        }
        
        // Move to next entry
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return -1;  // Not found
}

// Insert a new entry (called when task produces output)
//
// ========== DESIGN NOTE: Hash Chain vs Ring Buffer Conflict ==========
// 
// Problem: TensorMap entries are both:
//   1. Linked in hash bucket chains (via next_in_bucket)
//   2. Stored in a ring buffer that wraps around
//
// When ring buffer wraps and overwrites a slot, we MUST unlink the old entry
// from its bucket chain FIRST. Otherwise, the chain becomes corrupted.
//
// Solution: ALWAYS remove from bucket chain before overwriting, even for stale entries.
// This adds O(chain_length) overhead when pool wraps, but guarantees correctness.
//
void tensormap_insert(TensorMap* tm, TensorRegion* region, int32_t producer_task_id) {
    // Allocate entry from ring buffer pool
    int32_t entry_offset = tm->pool_head;
    TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    
    // Advance pool head (wrap around)
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;
    
    // ========== CRITICAL: MUST remove old entry from its bucket chain ==========
    // Even if entry is STALE (producer retired), it's still linked in its old 
    // bucket chain. If we overwrite without unlinking, the chain gets corrupted!
    //
    // We use 'in_bucket' flag to track if entry needs removal.
    // Alternative: Always call remove (idempotent if not in chain).
    if (entry->in_bucket) {
        tensormap_remove_from_bucket(tm, entry);
        entry->in_bucket = false;
    }
    
    // Initialize new entry
    entry->region = *region;
    entry->producer_task_id = producer_task_id;
    
    // Insert at head of hash bucket
    uint32_t bucket = tensormap_hash(tm, region);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;  // Mark as linked in bucket chain
    
    // Link to task's entry list (for potential cleanup)
    int32_t task_slot = producer_task_id % TASK_WINDOW_SIZE;
    entry->next_in_task = tm->task_entry_head[task_slot];
    tm->task_entry_head[task_slot] = entry_offset;
}

// Remove entry from its bucket chain (called during pool wrap-around)
void tensormap_remove_from_bucket(TensorMap* tm, TensorMapEntry* entry) {
    uint32_t bucket = tensormap_hash(tm, &entry->region);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    int32_t target_offset = entry - tm->entry_pool;
    
    while (offset >= 0) {
        if (offset == target_offset) {
            *prev_ptr = entry->next_in_bucket;
            entry->in_bucket = false;  // Mark as unlinked
            entry->next_in_bucket = -1;
            return;
        }
        prev_ptr = &tm->entry_pool[offset].next_in_bucket;
        offset = *prev_ptr;
    }
}

// Cleanup stale entries for retired tasks (optional optimization)
// Called periodically by Orchestrator when last_task_alive advances significantly
void tensormap_cleanup_retired(TensorMap* tm, int32_t old_last_task_alive, 
                                int32_t new_last_task_alive) {
    // Iterate through retired tasks and remove their entries from bucket chains
    for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
        int32_t task_slot = task_id % TASK_WINDOW_SIZE;
        int32_t offset = tm->task_entry_head[task_slot];
        
        while (offset >= 0) {
            TensorMapEntry* entry = &tm->entry_pool[offset];
            // Only remove if this entry belongs to the retiring task
            // (slot may have been reused by a newer task)
            if (entry->producer_task_id == task_id) {
                tensormap_remove_from_bucket(tm, entry);
            }
            offset = entry->next_in_task;
        }
        
        // Clear task's entry head (slot will be reused by task_id + TASK_WINDOW_SIZE)
        tm->task_entry_head[task_slot] = -1;
    }
}
```

#### Hash Chain vs Ring Buffer Conflict Resolution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               HASH CHAIN vs RING BUFFER DESIGN CONFLICT                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   问题: TensorMap 同时使用两种数据结构                                       │
│   ─────────────────────────────────────                                      │
│   1. Hash Table (链表式): entries 通过 next_in_bucket 链接                  │
│   2. Ring Buffer: entries 按 pool_head 顺序分配，wrap around 复用           │
│                                                                              │
│   矛盾场景:                                                                  │
│   ────────                                                                   │
│   T1: E5 inserted, hash(R1)=bucket[3], chain: bucket[3]→E5→E2→NULL         │
│   T2: E5's producer retires, E5 becomes STALE (但仍在链表中!)               │
│   T3: Ring wraps, pool_head 到达 E5 位置，要复用 E5 存 region R2            │
│                                                                              │
│   ⚠️ 如果直接覆盖 E5:                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ bucket[3]→E5'(new)→??? (next_in_bucket 被新值覆盖，链断裂!)      │       │
│   │ bucket[7]→E5'(new)→...  (新 entry 加入 bucket[7])               │       │
│   │                                                                  │       │
│   │ 结果: bucket[3] 的链表被破坏，后续查询可能找不到有效 entry       │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   解决方案: 在复用 slot 前，必须先从旧链表中移除                            │
│   ─────────────────────────────────────────────────                         │
│   1. 添加 in_bucket 标志位，追踪 entry 是否在链表中                         │
│   2. 复用 slot 时，若 in_bucket==true，先调用 remove_from_bucket()          │
│   3. 插入新 entry 后，设置 in_bucket=true                                   │
│                                                                              │
│   开销分析:                                                                  │
│   ──────────                                                                 │
│   • 额外内存: 1 byte per entry (in_bucket flag)                             │
│   • 时间开销: O(chain_length) per insert when slot was previously used      │
│   • 摊销开销: 每个 slot 被复用时才需要 remove，频率 = 1/pool_size           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Alternative Design Options:**

| 设计方案 | 优点 | 缺点 |
|---------|------|------|
| **当前方案: in_bucket flag** | 简单，正确性保证 | 复用时有 O(chain_length) 开销 |
| **显式 Free List** | 无需 remove，分配 O(1) | 不是 ring buffer，内存管理复杂 |
| **Open Addressing** | 无链表，覆盖安全 | 删除复杂，load factor 限制 |
| **Generation Numbers** | 无需 remove | 需要检查 generation，更大 entry |

**推荐**: 当前方案 (in_bucket flag) 在正确性和性能之间取得了良好平衡。复用时的 remove 开销由 periodic cleanup 分摊，实际影响很小。

#### TensorMap Lookup Method and Complexity Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TENSORMAP LOOKUP ALGORITHM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   关键优化: 链表按 task_id 从新到老排序 (HEAD INSERT)                        │
│   ──────────────────────────────────────────────────                        │
│   Insert 总是插入到链表头部 → 链表天然有序: task_id 递减                    │
│   遇到第一个 STALE entry → 后面的一定都是 STALE → 直接截断!                 │
│                                                                              │
│   tensormap_lookup(region):                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. HASH: bucket = hash(region) & (num_buckets - 1)     O(1)        │   │
│   │                                                                      │   │
│   │  2. CHAIN WALK:                                                      │   │
│   │     for each entry in bucket chain:                                  │   │
│   │                                                                      │   │
│   │       if (entry.task_id < last_task_alive):  // STALE!              │   │
│   │         ┌─────────────────────────────────────────────────────┐     │   │
│   │         │ TRUNCATE: 截断链表，后面的都是更老的 stale entries  │     │   │
│   │         │ prev->next = -1                                     │     │   │
│   │         │ mark all truncated entries: in_bucket = false       │     │   │
│   │         └─────────────────────────────────────────────────────┘     │   │
│   │         return -1  (not found, but cleaned up!)                      │   │
│   │                                                                      │   │
│   │       if (entry.region == region):                                   │   │
│   │         return entry.task_id  // FOUND                               │   │
│   │                                                                      │   │
│   │  3. return -1  // Chain exhausted, not found                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Chain Truncation Visualization:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CHAIN TRUNCATION ON STALE ENTRY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   last_task_alive = 50                                                       │
│                                                                              │
│   BEFORE lookup (chain sorted by task_id, newest first):                    │
│   ─────────────────────────────────────────────────────                     │
│   bucket[3] → [T=80] → [T=65] → [T=42] → [T=30] → [T=15] → NULL            │
│                 ↑         ↑         ↑         ↑         ↑                   │
│               VALID     VALID     STALE     STALE     STALE                 │
│                                     │                                        │
│                                     └── 首个 stale，后面一定都是 stale!     │
│                                                                              │
│   Lookup 遍历到 T=42 时发现是 STALE → 直接截断:                             │
│                                                                              │
│   AFTER lookup (truncated):                                                 │
│   ────────────────────────                                                  │
│   bucket[3] → [T=80] → [T=65] → NULL                                        │
│                 ↑         ↑                                                  │
│               VALID     VALID                                                │
│                                                                              │
│   [T=42], [T=30], [T=15] 被标记 in_bucket=false，可安全复用                 │
│                                                                              │
│   收益:                                                                      │
│   • 查找提前终止，无需遍历完整链表                                           │
│   • 顺便完成 lazy cleanup，缩短链表长度                                      │
│   • 后续查找更快 (链表变短了)                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Complexity Analysis:**

| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| **Lookup** | O(1) | O(active_entries_in_bucket) | O(n) |
| **Insert** | O(1) | O(1) | O(1) + O(remove) on reuse |
| **Truncation** | O(1) | O(stale_count) | O(n) |

**Key Improvement with Chain Truncation:**

```
传统方案 (无截断):
  Lookup 复杂度 = O(total_chain_length) = O(valid + stale entries)
  
Chain Truncation 优化:
  Lookup 复杂度 = O(valid_entries_only)
  
  原因: 链表按 task_id 降序排列，遇到首个 stale 即停止
        stale entries 在 lookup 过程中被顺便清理
```

Where:
- **α (load factor)** = pool_size / num_buckets = average chain length
- **effective_α** = active_entries / num_buckets (only valid entries count!)
- **n** = total entries in a single bucket (worst case: all hash to same bucket)

```
Load Factor Analysis:

  Theoretical α = pool_size / num_buckets = 4096 / 1024 = 4
  
  With chain truncation, effective α only counts VALID entries:
    effective_α = (active_tasks × avg_outputs) / num_buckets
                = (1024 × 2) / 1024 = 2
    
  Average lookup: O(1 + 4) = O(5) comparisons
  
  Note: Stale entries remain in chains but are skipped during lookup.
  After periodic cleanup, effective α decreases.
  
  Effective α (after cleanup) ≈ (active_tasks × avg_outputs) / num_buckets
                              = (1024 × 2) / 1024 = 2
```

**Hash Function Design (base_ptr only for overlap detection):**

```c
// Hash ONLY by base_ptr to enable overlap detection
// All regions of the same base tensor will be in the same bucket
uint32_t tensormap_hash(TensorMap* tm, TensorRegion* region) {
    uint64_t key = (uint64_t)region->base_ptr;
    
    // Improve distribution by mixing bits (pointers often have aligned low bits)
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    // Use bitwise AND for power-of-2 modulo (faster than %)
    return (uint32_t)(key & (tm->num_buckets - 1));
}

// CRITICAL: Why hash only by base_ptr?
// ====================================
// For overlap detection, ALL sub-regions of the same tensor MUST be
// in the SAME hash bucket. If we included offset in the hash:
//   Region A: base=X, offset=0   → bucket 5
//   Region B: base=X, offset=128 → bucket 12  (WRONG! Can't detect overlap)
//
// With base_ptr-only hash:
//   Region A: base=X, offset=0   → bucket 5
//   Region B: base=X, offset=128 → bucket 5   (CORRECT! Same bucket)
//
// Trade-off: Higher chain length for tensors with many sub-regions,
// but correctness requires all same-base regions be in same bucket.
//
// Requirements:
//   - num_buckets MUST be power of 2 for fast modulo
//   - Bit mixing improves distribution for aligned pointers
```

**Lookup Performance Optimization:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   OPTIMIZATION: Periodic Cleanup reduces chain length                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Without cleanup (stale entries accumulate in chains):                     │
│                                                                              │
│   Bucket 5: [E7:VALID] → [E3:STALE] → [E1:STALE] → [E0:STALE] → NULL       │
│             ↑                                                                │
│           4 comparisons to find valid entry                                 │
│                                                                              │
│   After cleanup (stale entries removed from chains):                        │
│                                                                              │
│   Bucket 5: [E7:VALID] → NULL                                               │
│             ↑                                                                │
│           1 comparison to find valid entry                                  │
│                                                                              │
│   Cleanup frequency: Every TENSORMAP_CLEANUP_INTERVAL (64) retired tasks    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### TensorMap Sizing Guidelines

```
TensorMap Pool Size Calculation:

  Active tasks in window = TASK_WINDOW_SIZE
  Average outputs per task = AVG_OUTPUTS
  
  Minimum pool_size = TASK_WINDOW_SIZE × AVG_OUTPUTS
  Recommended pool_size = 2 × TASK_WINDOW_SIZE × AVG_OUTPUTS (headroom)
  
  Example:
    TASK_WINDOW_SIZE = 1024
    AVG_OUTPUTS = 2
    pool_size = 2 × 1024 × 2 = 4096 entries
    
  Memory footprint:
    TensorMapEntry ≈ 32 bytes
    Pool: 4096 × 32 = 128 KB
    Buckets: 1024 × 4 = 4 KB  (num_buckets = TASK_WINDOW_SIZE)
    task_entry_head: 1024 × 4 = 4 KB
    Total: ~136 KB
    
  num_buckets selection:
    - Must be power of 2 (for fast hash modulo)
    - Recommended: num_buckets = TASK_WINDOW_SIZE
    - This gives α ≈ AVG_OUTPUTS (e.g., 2)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TENSOR MAP RING BUFFER DESIGN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   HASH BUCKETS (fixed size)                                                  │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                         │
│   │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  ...                    │
│   └──┬──┴──┬──┴─────┴──┬──┴─────┴─────┴──┬──┴─────┘                         │
│      │     │           │                 │                                   │
│      ▼     ▼           ▼                 ▼                                   │
│                                                                              │
│   ENTRY POOL (ring buffer)                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ E0  │ E1  │ E2  │ E3  │ E4  │ E5  │ E6  │ E7  │ ... │ En-1│        │   │
│   │     │     │     │     │     │     │     │     │     │     │        │   │
│   │ T=3 │ T=5 │ T=7 │ T=8 │ T=9 │ T=10│ T=11│ T=12│     │ T=4 │        │   │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │   │
│     ▲                                   ▲                                    │
│     │                                   │                                    │
│   STALE                              pool_head                               │
│  (T < last_task_alive=6)            (next alloc)                            │
│                                                                              │
│   VALIDITY CHECK:                                                            │
│   - E0: T=3 < 6 → STALE (ignored in lookup, will be overwritten)            │
│   - E1: T=5 < 6 → STALE                                                      │
│   - E2: T=7 >= 6 → VALID                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Tensor Extraction Operations and Memory Layout

This section describes how tensor extraction operations affect memory layout and dependency checking in the TensorMap.

##### Raw Tensor vs Logical Tensor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAW TENSOR vs LOGICAL TENSOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RAW TENSOR (Storage):                                                      │
│   ════════════════════                                                       │
│   - Provides the actual contiguous memory allocation                         │
│   - Described by: base_ptr, total_size                                       │
│   - Single owner, may be shared by multiple logical tensors                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ base_ptr                                            total_size  │       │
│   │ ▼                                                             ▼ │       │
│   │ ┌───────────────────────────────────────────────────────────┐   │       │
│   │ │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   │       │
│   │ └───────────────────────────────────────────────────────────┘   │       │
│   │   Raw storage: contiguous bytes in memory                       │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   LOGICAL TENSOR (View):                                                     │
│   ══════════════════════                                                     │
│   - A "window" into the raw tensor with specific layout                      │
│   - Described by: raw_tensor_ptr, offset, shape[], strides[]                 │
│   - Multiple logical tensors can share the same raw tensor                   │
│                                                                              │
│   struct LogicalTensor {                                                     │
│       void*    raw_base;        // Pointer to raw tensor's base              │
│       int64_t  storage_offset;  // Byte offset from raw_base                 │
│       int64_t  shape[MAX_DIM];  // Size in each dimension                    │
│       int64_t  strides[MAX_DIM];// Byte stride in each dimension             │
│       int      ndim;            // Number of dimensions                      │
│   };                                                                         │
│                                                                              │
│   Example: 2D tensor (3x4) with row-major layout                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                   │       │
│   │   shape = [3, 4]                                                  │       │
│   │   strides = [4*sizeof(float), sizeof(float)] = [16, 4]           │       │
│   │                                                                   │       │
│   │   Memory layout:                                                  │       │
│   │   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐              │       │
│   │   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │              │       │
│   │   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘              │       │
│   │   ├───────────────┤ row 0 (stride[0]=16 bytes to next row)       │       │
│   │       ├───┤ element stride[1]=4 bytes                            │       │
│   │                                                                   │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Primary Extraction Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR EXTRACTION OPERATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Three primary operations create new logical tensors that SHARE storage    │
│   with the original raw tensor (shallow copy / aliasing):                    │
│                                                                              │
│   1. VIEW (Slicing/Indexing)                                                 │
│   ══════════════════════════                                                 │
│   Creates a sub-tensor by selecting a subset of elements.                    │
│   Changes: offset, shape (strides typically unchanged)                       │
│                                                                              │
│   Original A (4x4):           View B = A[1:3, 1:3] (2x2):                    │
│   ┌───┬───┬───┬───┐          ┌───┬───┐                                      │
│   │ 0 │ 1 │ 2 │ 3 │          │ 5 │ 6 │  offset = 1*stride[0] + 1*stride[1] │
│   ├───┼───┼───┼───┤          ├───┼───┤  shape = [2, 2]                      │
│   │ 4 │[5]│[6]│ 7 │    →     │ 9 │10 │  strides = same as A                 │
│   ├───┼───┼───┼───┤          └───┴───┘                                      │
│   │ 8 │[9]│[10]│11│                                                         │
│   ├───┼───┼───┼───┤                                                         │
│   │12 │13 │14 │15 │                                                         │
│   └───┴───┴───┴───┘                                                         │
│                                                                              │
│   2. RESHAPE                                                                 │
│   ══════════════                                                             │
│   Changes the logical shape without moving data.                             │
│   Requirement: tensor must be contiguous (no gaps).                          │
│   Changes: shape, strides (recomputed for new shape)                         │
│                                                                              │
│   Original A (2x6):           Reshape B = A.reshape(3, 4):                   │
│   ┌───┬───┬───┬───┬───┬───┐  ┌───┬───┬───┬───┐                              │
│   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │  │ 0 │ 1 │ 2 │ 3 │  Same memory,               │
│   ├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┤  different interpretation    │
│   │ 6 │ 7 │ 8 │ 9 │10 │11 │  │ 4 │ 5 │ 6 │ 7 │                              │
│   └───┴───┴───┴───┴───┴───┘  ├───┼───┼───┼───┤                              │
│                               │ 8 │ 9 │10 │11 │                              │
│                               └───┴───┴───┴───┘                              │
│                                                                              │
│   3. TRANSPOSE (Permute)                                                     │
│   ══════════════════════                                                     │
│   Reorders dimensions without moving data.                                   │
│   Changes: shape (reordered), strides (reordered)                            │
│                                                                              │
│   Original A (2x3):           Transpose B = A.T (3x2):                       │
│   shape=[2,3], strides=[12,4] shape=[3,2], strides=[4,12]                   │
│                                                                              │
│   Logical view A:             Logical view B:                                │
│   ┌───┬───┬───┐               ┌───┬───┐                                     │
│   │ 0 │ 1 │ 2 │               │ 0 │ 3 │  Same memory!                       │
│   ├───┼───┼───┤               ├───┼───┤  B[i,j] = A[j,i]                    │
│   │ 3 │ 4 │ 5 │               │ 1 │ 4 │  via stride swap                    │
│   └───┴───┴───┘               ├───┼───┤                                     │
│                               │ 2 │ 5 │                                     │
│   Memory: [0,1,2,3,4,5]       └───┴───┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Shallow vs Deep Extraction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHALLOW vs DEEP EXTRACTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SHALLOW EXTRACTION (Aliasing):                                             │
│   ══════════════════════════════                                             │
│   - view(), reshape(), transpose()                                           │
│   - Creates a new logical tensor that SHARES the raw tensor                  │
│   - No data copy, O(1) operation                                             │
│   - Any write through one view affects all other views                       │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Raw Tensor (shared storage)                                     │       │
│   │  ┌─────────────────────────────────────────────────────────┐    │       │
│   │  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │       │
│   │  └─────────────────────────────────────────────────────────┘    │       │
│   │       ▲              ▲              ▲                           │       │
│   │       │              │              │                           │       │
│   │  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐                     │       │
│   │  │ View A  │    │ View B  │    │ View C  │                     │       │
│   │  │(original)│   │(slice)  │    │(transpose)                    │       │
│   │  └─────────┘    └─────────┘    └─────────┘                     │       │
│   │                                                                  │       │
│   │  All views ALIAS the same memory!                               │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   DEEP EXTRACTION (Copy):                                                    │
│   ═══════════════════════                                                    │
│   - deep_view(), deep_reshape(), deep_transpose(), clone(), contiguous()    │
│   - Allocates a NEW raw tensor                                               │
│   - Copies data from source to new tensor                                    │
│   - Independent of original - writes don't affect other views               │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Original Raw Tensor              New Raw Tensor (deep copy)    │       │
│   │  ┌───────────────────────┐       ┌───────────────────────┐     │       │
│   │  │░░░░░░░░░░░░░░░░░░░░░░│       │████████████████████████│     │       │
│   │  └───────────────────────┘       └───────────────────────┘     │       │
│   │       ▲                               ▲                         │       │
│   │       │           COPY                │                         │       │
│   │  ┌────┴────┐    ─────────────►  ┌────┴────┐                    │       │
│   │  │ View A  │                    │ Deep B  │                    │       │
│   │  │(original)│                   │(independent)                 │       │
│   │  └─────────┘                    └─────────┘                    │       │
│   │                                                                  │       │
│   │  Deep copy has its own storage - no aliasing                    │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   WHEN TO USE DEEP COPY:                                                     │
│   - When you need to modify data without affecting original                  │
│   - When transposed/strided tensor needs contiguous memory for kernels      │
│   - When aliasing would create false dependencies                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### TensorMap Design for Extracted Tensors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSORMAP DESIGN FOR EXTRACTED TENSORS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CHALLENGE:                                                                 │
│   ═══════════                                                                │
│   TensorMap must track dependencies for tensors that may:                    │
│   - Share the same raw storage (shallow extractions)                         │
│   - Have complex, non-contiguous memory access patterns                      │
│   - Overlap partially (not just exact matches)                               │
│                                                                              │
│   TENSORMAP ENTRY STRUCTURE (Extended):                                      │
│   ═══════════════════════════════════════                                    │
│                                                                              │
│   struct TensorMapEntry {                                                    │
│       // Raw tensor identification                                           │
│       void*    raw_base_ptr;      // Base of raw tensor (for grouping)      │
│       int64_t  raw_total_size;    // Total size of raw tensor               │
│                                                                              │
│       // Logical tensor layout (for overlap detection)                       │
│       int64_t  storage_offset;    // Byte offset from raw_base              │
│       int64_t  shape[MAX_DIM];    // Shape in each dimension                │
│       int64_t  strides[MAX_DIM];  // Strides in each dimension              │
│       int      ndim;              // Number of dimensions                   │
│                                                                              │
│       // Bounding box (precomputed for fast overlap check)                  │
│       int64_t  min_byte_offset;   // First byte accessed                    │
│       int64_t  max_byte_offset;   // Last byte accessed                     │
│                                                                              │
│       // Producer tracking                                                   │
│       int32_t  producer_task_id;  // Task that produced this tensor         │
│       int32_t  next_in_bucket;    // Hash chain                             │
│       bool     is_deep_copy;      // True if this is a deep copy            │
│   };                                                                         │
│                                                                              │
│   HASH STRATEGY:                                                             │
│   ══════════════                                                             │
│   Primary key: raw_base_ptr (groups all views of same storage)              │
│   Within bucket: linear scan checking bounding box overlap                   │
│                                                                              │
│   hash(tensor) = hash(tensor.raw_base_ptr) % NUM_BUCKETS                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Memory Overlap Detection Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY OVERLAP DETECTION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PROBLEM:                                                                   │
│   ═════════                                                                  │
│   Given two tensors A and B (potentially with complex strides from          │
│   view/reshape/transpose), determine if they access overlapping memory.     │
│                                                                              │
│   ALGORITHM 1: BOUNDING BOX (Fast, may have false positives)                │
│   ════════════════════════════════════════════════════════════              │
│                                                                              │
│   Step 1: Check if same raw storage                                         │
│   ─────────────────────────────────                                         │
│   if (A.raw_base_ptr != B.raw_base_ptr) {                                   │
│       return NO_OVERLAP;  // Different storage, definitely no overlap       │
│   }                                                                          │
│                                                                              │
│   Step 2: Calculate bounding box for each tensor                            │
│   ──────────────────────────────────────────────                            │
│   For a tensor T with shape[d] and strides[d]:                              │
│                                                                              │
│   min_offset = storage_offset                                                │
│   max_offset = storage_offset                                                │
│   for d in 0..ndim:                                                          │
│       if strides[d] > 0:                                                     │
│           max_offset += (shape[d] - 1) * strides[d]                         │
│       else:  // negative stride (flipped dimension)                         │
│           min_offset += (shape[d] - 1) * strides[d]                         │
│                                                                              │
│   Step 3: Check bounding box intersection                                   │
│   ───────────────────────────────────────                                   │
│   overlap = (A.min_offset <= B.max_offset) && (B.min_offset <= A.max_offset)│
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Memory range visualization:                                     │       │
│   │                                                                   │       │
│   │  Tensor A: [min_A ████████████████████ max_A]                   │       │
│   │  Tensor B:           [min_B ████████████████████ max_B]         │       │
│   │                           ↑               ↑                      │       │
│   │                           └───────────────┘                      │       │
│   │                           Bounding boxes overlap                 │       │
│   │                                                                   │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   IMPLEMENTATION (Pseudocode):                                               │
│   ═════════════════════════════                                              │
│                                                                              │
│   ```python                                                                  │
│   def get_bounding_box(tensor):                                             │
│       """Calculate min and max byte offset for tensor's memory footprint""" │
│       min_offset = tensor.storage_offset                                     │
│       max_offset = tensor.storage_offset                                     │
│                                                                              │
│       for dim in range(tensor.ndim):                                        │
│           extent = (tensor.shape[dim] - 1) * tensor.strides[dim]            │
│           if tensor.strides[dim] > 0:                                       │
│               max_offset += extent                                           │
│           else:                                                              │
│               min_offset += extent  # extent is negative                    │
│                                                                              │
│       return (min_offset, max_offset)                                        │
│                                                                              │
│   def check_overlap_fast(A, B):                                             │
│       """Fast bounding box overlap check (may have false positives)"""      │
│       # Different storage => no overlap                                      │
│       if A.raw_base_ptr != B.raw_base_ptr:                                  │
│           return False                                                       │
│                                                                              │
│       # Get bounding boxes                                                   │
│       min_A, max_A = get_bounding_box(A)                                    │
│       min_B, max_B = get_bounding_box(B)                                    │
│                                                                              │
│       # Check interval intersection                                          │
│       return (min_A <= max_B) and (min_B <= max_A)                          │
│   ```                                                                        │
│                                                                              │
│   FALSE POSITIVE EXAMPLE:                                                    │
│   ═══════════════════════                                                    │
│   Strided tensors may have bounding box overlap but not actual overlap:     │
│                                                                              │
│   A: elements at bytes [0, 8, 16, 24]  (stride=8, shape=4)                  │
│   B: elements at bytes [4, 12, 20, 28] (stride=8, shape=4, offset=4)        │
│                                                                              │
│   Bounding boxes: A=[0,24], B=[4,28] → OVERLAP (false positive!)            │
│   Actual bytes:   A touches {0,8,16,24}, B touches {4,12,20,28}             │
│   No actual overlap! They interleave.                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│   ALGORITHM 2: GCD-BASED EXACT CHECK (Slower, no false positives)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For 100% accuracy with strided tensors, check if the integer lattices     │
│   defined by the strides ever intersect.                                     │
│                                                                              │
│   MATHEMATICAL FORMULATION:                                                  │
│   ════════════════════════                                                   │
│   Tensor A touches byte offset: offset_A + Σ(i_d * stride_A[d])             │
│   Tensor B touches byte offset: offset_B + Σ(j_d * stride_B[d])             │
│   where 0 <= i_d < shape_A[d], 0 <= j_d < shape_B[d]                        │
│                                                                              │
│   Overlap exists iff there exist valid indices (i, j) such that:            │
│   offset_A + Σ(i_d * stride_A[d]) = offset_B + Σ(j_d * stride_B[d])         │
│                                                                              │
│   This is a Diophantine equation solvable via GCD analysis.                  │
│                                                                              │
│   1D CASE:                                                                   │
│   ─────────                                                                  │
│   A: offset_A, stride_A, size_A                                              │
│   B: offset_B, stride_B, size_B                                              │
│                                                                              │
│   Overlap iff exists integers i, j where:                                    │
│   - 0 <= i < size_A                                                          │
│   - 0 <= j < size_B                                                          │
│   - offset_A + i*stride_A = offset_B + j*stride_B                           │
│                                                                              │
│   Rearranging: i*stride_A - j*stride_B = offset_B - offset_A                │
│                                                                              │
│   Solution exists iff: gcd(stride_A, stride_B) divides (offset_B - offset_A)│
│   Then check if solution falls within valid index ranges.                    │
│                                                                              │
│   ```python                                                                  │
│   def check_overlap_exact_1d(A, B):                                         │
│       """Exact 1D overlap check using GCD"""                                │
│       if A.raw_base_ptr != B.raw_base_ptr:                                  │
│           return False                                                       │
│                                                                              │
│       delta = B.storage_offset - A.storage_offset                            │
│       g = gcd(A.stride, B.stride)                                           │
│                                                                              │
│       # No solution if delta not divisible by gcd                            │
│       if delta % g != 0:                                                     │
│           return False                                                       │
│                                                                              │
│       # Find one solution using extended Euclidean algorithm                │
│       # Then check if any solution falls in valid range                      │
│       # (Implementation details omitted for brevity)                         │
│       ...                                                                    │
│   ```                                                                        │
│                                                                              │
│   MULTI-DIMENSIONAL CASE:                                                    │
│   ═══════════════════════                                                    │
│   For N-dimensional tensors, the problem becomes significantly harder.       │
│   Practical approach: flatten to 1D using combined stride/shape info.        │
│                                                                              │
│   RECOMMENDATION:                                                            │
│   ════════════════                                                           │
│   - Use BOUNDING BOX for most cases (fast, conservative)                    │
│   - False positives create unnecessary dependencies (safe, but less parallel)│
│   - Use GCD only if false positives significantly hurt performance          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### TensorMap Lookup with Overlap Detection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSORMAP LOOKUP WITH OVERLAP                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   When a task declares an INPUT tensor, TensorMap must find ALL producers   │
│   whose outputs OVERLAP with this input (not just exact matches).           │
│                                                                              │
│   LOOKUP ALGORITHM:                                                          │
│   ═══════════════════                                                        │
│                                                                              │
│   ```c                                                                       │
│   // Find all producers that overlap with the input tensor                   │
│   int tensormap_find_overlapping_producers(                                  │
│       TensorMap* tm,                                                         │
│       LogicalTensor* input,                                                  │
│       int32_t* producer_ids,    // output: array of producer task IDs       │
│       int max_producers         // max size of output array                  │
│   ) {                                                                        │
│       int count = 0;                                                         │
│                                                                              │
│       // Hash by raw_base_ptr to find candidate bucket                       │
│       uint32_t bucket = hash(input->raw_base_ptr) % tm->num_buckets;        │
│                                                                              │
│       // Scan bucket for overlapping entries                                 │
│       int32_t entry_idx = tm->buckets[bucket];                              │
│       while (entry_idx >= 0 && count < max_producers) {                     │
│           TensorMapEntry* entry = &tm->entry_pool[entry_idx];               │
│                                                                              │
│           // Skip stale entries                                              │
│           if (entry->producer_task_id < tm->last_task_alive) {              │
│               entry_idx = entry->next_in_bucket;                            │
│               continue;                                                      │
│           }                                                                  │
│                                                                              │
│           // Check overlap (using bounding box or GCD method)               │
│           if (check_overlap(input, entry)) {                                │
│               producer_ids[count++] = entry->producer_task_id;              │
│           }                                                                  │
│                                                                              │
│           entry_idx = entry->next_in_bucket;                                │
│       }                                                                      │
│                                                                              │
│       return count;                                                          │
│   }                                                                          │
│   ```                                                                        │
│                                                                              │
│   COMPLEXITY ANALYSIS:                                                       │
│   ═════════════════════                                                      │
│   - Hash lookup: O(1)                                                        │
│   - Bucket scan: O(k) where k = entries with same raw_base_ptr             │
│   - Overlap check per entry: O(ndim) for bounding box, O(ndim²) for GCD    │
│   - Total: O(k * ndim) typical case                                         │
│                                                                              │
│   OPTIMIZATION: INTERVAL TREE                                                │
│   ════════════════════════════                                               │
│   For tensors with many views of same storage, use interval tree:           │
│   - Build tree on (min_offset, max_offset) intervals                        │
│   - Query: O(log n + m) where m = overlapping intervals                     │
│   - Useful when many slices of large tensor                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Summary: Tensor Extraction Impact on TensorMap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SUMMARY: EXTRACTION OPERATIONS IMPACT                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Operation     │ Storage │ Offset  │ Shape   │ Strides │ TensorMap Impact │
│   ══════════════╪═════════╪═════════╪═════════╪═════════╪══════════════════│
│   view/slice    │ shared  │ changed │ changed │ same    │ may overlap      │
│   reshape       │ shared  │ same    │ changed │ changed │ same region      │
│   transpose     │ shared  │ same    │ swapped │ swapped │ same region      │
│   deep_view     │ new     │ 0       │ changed │ compact │ independent      │
│   deep_reshape  │ new     │ 0       │ changed │ compact │ independent      │
│   deep_transpose│ new     │ 0       │ swapped │ compact │ independent      │
│   clone         │ new     │ 0       │ same    │ compact │ independent      │
│                                                                              │
│   KEY INSIGHTS:                                                              │
│   ══════════════                                                             │
│   1. Shallow ops (view/reshape/transpose) create ALIASES → need overlap check│
│   2. Deep ops create INDEPENDENT tensors → no overlap with original         │
│   3. TensorMap must handle both exact matches AND partial overlaps          │
│   4. Bounding box is fast but conservative (safe for correctness)           │
│   5. GCD method is exact but slower (for performance-critical paths)        │
│                                                                              │
│   DESIGN DECISION:                                                           │
│   ═════════════════                                                          │
│   CURRENT: BOUNDING BOX APPROACH (Implemented in runtime2):                  │
│                                                                              │
│   1. Simple Tensor (is_contiguous = true):                                   │
│      → Bounding Box is EXACT (no false positives)                           │
│                                                                              │
│   2. Complex Tensor (is_contiguous = false):                                 │
│      → Bounding Box is CONSERVATIVE (safe, may have false positives)        │
│                                                                              │
│   NOTE: GCD-based methods are DISABLED because:                              │
│   • They don't work when lowest dimension has stride=1 (GCD becomes 1)      │
│   • They can produce false negatives in certain cases                       │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────────  │
│   FUTURE: HIERARCHICAL BOUNDING BOX (HBB) METHOD                             │
│   ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│   Track tensor derivation history (max 8 levels of view/reshape/transpose)  │
│   Compare two tensors by walking their derivation paths:                     │
│                                                                              │
│   1. Same level, same op type:                                               │
│      - VIEW: check bbox intersection, if disjoint → NO OVERLAP (exact)      │
│      - RESHAPE/TRANSPOSE: check if identical, continue if same              │
│                                                                              │
│   2. Different op type or different reshape/transpose:                       │
│      → CONSERVATIVE (assume overlap, safe)                                   │
│                                                                              │
│   Example: A[10:50].reshape(8,5) vs A[60:80]                                │
│     Level 0: VIEW [0,399] == VIEW [0,399] → skip                            │
│     Level 1: VIEW [40,199] vs VIEW [240,319] → disjoint! → NO OVERLAP       │
│                                                                              │
│   Benefits:                                                                  │
│   • O(depth) complexity, depth ≤ 8                                          │
│   • NO FALSE NEGATIVES (safe)                                               │
│   • Eliminates many false positives for common patterns                     │
│   • Works for any dimension/stride combination                              │
│                                                                              │
│   Status: Design complete, implementation pending                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Implementation: Logical Tensor API (runtime2)

The following APIs are implemented in `pto_logical_tensor.h` and `pto_logical_tensor.c`:

**Data Structures:**

```c
// Maximum dimensions supported
#define PTO2_MAX_TENSOR_DIM   8

// Extraction type enumeration
typedef enum {
    PTO2_TENSOR_RAW = 0,           // Original raw tensor (owns storage)
    PTO2_TENSOR_VIEW = 1,          // view() - subset selection, shared storage
    PTO2_TENSOR_RESHAPE = 2,       // reshape() - shape change, shared storage
    PTO2_TENSOR_TRANSPOSE = 3,     // transpose() - dimension permute, shared storage
    PTO2_TENSOR_DEEP_VIEW = 4,     // deep_view() - copied subset, new storage
    PTO2_TENSOR_DEEP_RESHAPE = 5,  // deep_reshape() - copied reshape, new storage
    PTO2_TENSOR_DEEP_TRANSPOSE = 6 // deep_transpose() - copied transpose, new storage
} PTO2TensorExtractionType;

// Logical tensor structure
typedef struct {
    void*    raw_base;            // Pointer to raw tensor's base
    int64_t  raw_total_size;      // Total size of raw tensor in bytes
    int64_t  storage_offset;      // Byte offset from raw_base to first element
    int64_t  shape[PTO2_MAX_TENSOR_DIM];    // Size in each dimension
    int64_t  strides[PTO2_MAX_TENSOR_DIM];  // Byte stride in each dimension
    int32_t  ndim;                          // Number of dimensions
    int64_t  min_byte_offset;     // Bounding box: first byte accessed
    int64_t  max_byte_offset;     // Bounding box: last byte accessed
    int64_t  elem_size;           // Size of each element in bytes
    int64_t  numel;               // Total number of elements
    PTO2TensorExtractionType extraction_type;
    bool     is_contiguous;       // True if memory is contiguous
} PTO2LogicalTensor;
```

**Core APIs:**

```c
// === Tensor Creation ===
// Create a raw contiguous tensor (row-major layout)
void pto2_logical_tensor_init_raw(
    PTO2LogicalTensor* tensor,
    void* base_ptr,
    const int64_t* shape,
    int32_t ndim,
    int64_t elem_size
);

// === Shallow Extraction (share storage) ===
// Create a view (slice) - select subset of elements
bool pto2_logical_tensor_view(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* start,     // Start index in each dimension
    const int64_t* shape,     // Shape of the view
    int32_t ndim
);

// Reshape - change logical shape (requires contiguous tensor)
bool pto2_logical_tensor_reshape(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* shape,
    int32_t ndim
);

// Transpose - permute dimensions
bool pto2_logical_tensor_transpose(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int32_t* perm       // NULL = reverse all dimensions
);

// === Deep Extraction (copy to new storage) ===
bool pto2_logical_tensor_clone(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    void* new_base            // Pre-allocated buffer
);

// === Overlap Detection ===
// Fast bounding box check (may have false positives for non-contiguous tensors)
bool pto2_logical_tensor_overlap_fast(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

// === OVERLAP DETECTION (RECOMMENDED) ===
// Uses bounding box for all cases:
// - Contiguous tensors: bounding box is exact (no false positives)
// - Non-contiguous tensors: bounding box is conservative (safe, may have false positives)
bool pto2_logical_tensor_overlap_hybrid(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

// Overlap detection for tensor vs TensorMapEntry
bool pto2_tensor_entry_overlap_hybrid(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);

// === Interval Tree for O(log n) Queries ===
// (in pto_interval_tree.h)
bool pto2_interval_tree_insert(tree, low, high, producer_task_id, entry_index);
int32_t pto2_interval_tree_query(tree, low, high, results, max_results);
int32_t pto2_interval_tree_remove_stale(tree, task_threshold);
```

**Note on GCD-Based Methods:**

GCD-based overlap detection was considered but is currently DISABLED because:

1. **stride=1 problem**: When the lowest dimension has stride equal to elem_size,
   the GCD of all strides becomes small (often 1), making the test ineffective.

2. **False negative risk**: When one tensor is contiguous and the other is 
   non-contiguous, GCD methods can incorrectly report "no overlap".

3. **Safety priority**: Bounding box is guaranteed safe (no false negatives).
   False positives only add extra dependencies, which is safe for correctness.

**Interval Tree for O(log n + k) Queries:**

When there are many views of the same raw tensor, hash table lookup degrades to O(n).
The Interval Tree provides O(log n + k) queries where k is the number of results.

```c
// Interval Tree API
typedef struct {
    int64_t low, high;        // Bounding box interval
    int32_t producer_task_id; // Associated producer
    int32_t entry_index;      // Index into TensorMapEx
} PTO2Interval;

// Insert: O(log n) - AVL balanced
bool pto2_interval_tree_insert(tree, low, high, producer_task_id, entry_index);

// Query: O(log n + k) - find all overlapping intervals
int32_t pto2_interval_tree_query(tree, low, high, results, max_results);

// Lazy invalidation: remove stale entries
int32_t pto2_interval_tree_remove_stale(tree, task_threshold);
```

**Performance Benchmark Results:**

```
Intervals: 1000, Queries: 10000
Interval Tree: 0.002 s
Linear Scan:   0.027 s
Speedup:       11.9x
Tree height:   12 (balanced, optimal ~14)
```

**When to use each method:**

| Method | Complexity | Use Case |
|--------|------------|----------|
| Hash Table (TensorMapEx) | O(k) per bucket | Few views per tensor, fast insert |
| Interval Tree | O(log n + k) | Many views of same tensor |
| Linear Scan | O(n) | Very small datasets (<100) |

**Extended TensorMap APIs (for LogicalTensor support):**

```c
// Extended TensorMap structure
typedef struct {
    int32_t* buckets;
    PTO2TensorMapEntryEx* entry_pool;
    int32_t pool_size, pool_head, num_buckets;
    int32_t* task_entry_head;
    int32_t last_task_alive;
} PTO2TensorMapEx;

// Insert tensor output
void pto2_tensormapex_insert(
    PTO2TensorMapEx* tm, 
    const PTO2LogicalTensor* tensor, 
    int32_t producer_task_id
);

// Lookup single overlapping producer (returns first found)
int32_t pto2_tensormapex_lookup(
    PTO2TensorMapEx* tm, 
    const PTO2LogicalTensor* tensor
);

// Lookup ALL overlapping producers (for complete dependency graph)
int32_t pto2_tensormapex_lookup_all(
    PTO2TensorMapEx* tm, 
    const PTO2LogicalTensor* tensor,
    int32_t* producer_ids,    // Output array
    int32_t max_producers     // Max size of output array
);
```

**Usage Example:**

```c
// Create a 4x4 matrix
PTO2LogicalTensor A;
int64_t shape[] = {4, 4};
float* data = malloc(16 * sizeof(float));
pto2_logical_tensor_init_raw(&A, data, shape, 2, sizeof(float));

// Task 100 produces A
pto2_tensormapex_insert(&tm, &A, 100);

// Create a view of top-left 2x2 submatrix
PTO2LogicalTensor view;
int64_t start[] = {0, 0};
int64_t view_shape[] = {2, 2};
pto2_logical_tensor_view(&A, &view, start, view_shape, 2);

// Task 200 wants to read the view - find dependencies
int32_t producers[16];
int32_t count = pto2_tensormapex_lookup_all(&tm, &view, producers, 16);
// count=1, producers[0]=100 (view overlaps with A)
```

---

#### TensorMap Overlapping Region Detection (Advanced Design)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               OVERLAPPING SUB-TENSOR DEPENDENCY DETECTION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PROBLEM: Dependencies exist not just for exact region matches, but also   │
│   when sub-tensor regions OVERLAP within the same base tensor.              │
│                                                                              │
│   Base Tensor X (1024 bytes):                                                │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  0       128      256      384      512      640      768      1024│    │
│   │  ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤ │    │
│   │                                                                    │    │
│   │  Task A writes: [0, 256)      ████████                             │    │
│   │  Task B reads:  [128, 384)         ████████                        │    │
│   │                              ↑      ↑                              │    │
│   │                              └──────┘ OVERLAP: [128, 256)          │    │
│   │                                                                    │    │
│   │  Result: Task B depends on Task A (RAW dependency)                 │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   OVERLAP FORMULA (1D interval intersection):                                │
│   ─────────────────────────────────────────                                  │
│   Region A: [start_a, end_a)  where end_a = start_a + size_a                │
│   Region B: [start_b, end_b)  where end_b = start_b + size_b                │
│                                                                              │
│   Overlap exists if: (start_a < end_b) AND (start_b < end_a)                │
│                                                                              │
│   Prerequisite: Both regions must have SAME base_ptr (raw_tensor)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY TYPES WITH OVERLAPPING REGIONS                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. RAW (Read-After-Write):                                                 │
│      Consumer READS region that overlaps with Producer's OUTPUT              │
│      → Consumer must wait for Producer                                       │
│                                                                              │
│      Producer (Task A): output region [0, 256)                               │
│      Consumer (Task B): input region [128, 384)                              │
│      Overlap: [128, 256) → Task B depends on Task A                         │
│                                                                              │
│   2. WAW (Write-After-Write):                                                │
│      Task B WRITES to region that overlaps with Task A's OUTPUT              │
│      → Task B must wait for Task A (preserve write order)                    │
│                                                                              │
│      Task A: output region [0, 256)                                          │
│      Task B: output region [128, 384)                                        │
│      Overlap: [128, 256) → Task B depends on Task A                         │
│                                                                              │
│   3. WAR (Write-After-Read):                                                 │
│      Task B WRITES to region that overlaps with Task A's INPUT               │
│      → Task B must wait for Task A (A must read before B writes)            │
│      Note: This is tracked via OUTPUT registration, not input                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION OPTIONS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   OPTION 1: Exact Match Only (Current Simple Implementation)                │
│   ──────────────────────────────────────────────────────────                │
│   - Match only when base_ptr, tile_index, AND offset are identical          │
│   - Pros: O(1) lookup per entry, simple implementation                      │
│   - Cons: May miss dependencies for overlapping but non-identical regions   │
│   - Use case: Tile-aligned access patterns where tiles don't overlap        │
│                                                                              │
│   OPTION 2: Same-Base Conservative (All regions with same base conflict)    │
│   ──────────────────────────────────────────────────────────────────────    │
│   - Any two regions with same base_ptr create a dependency                  │
│   - Pros: Very simple, never misses dependencies                            │
│   - Cons: Creates unnecessary dependencies, reduces parallelism             │
│   - Use case: Single output per tensor, or when parallelism not critical    │
│                                                                              │
│   OPTION 3: Interval Overlap Check (Recommended for Production)             │
│   ─────────────────────────────────────────────────────────────             │
│   - Check actual byte-range overlap: (s1 < e2) && (s2 < e1)                │
│   - Pros: Accurate dependency tracking, maximum parallelism                 │
│   - Cons: O(n) per lookup in worst case (all same base_ptr)                │
│   - Optimization: Use Interval Tree for O(log n + k) lookup                 │
│                                                                              │
│   OPTION 4: Tile-Based with Overlap (Hybrid Approach)                       │
│   ────────────────────────────────────────────────────                      │
│   - Primary hash: (base_ptr, tile_index)                                    │
│   - Within same tile: check offset/size overlap                             │
│   - Pros: Good balance of accuracy and performance                          │
│   - Cons: Requires tiles to be well-defined boundaries                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Overlap Detection Code (tensormap_lookup and region_overlap):**

```c
// Check if two regions OVERLAP (not just exact match)
// 
// Overlap condition:
//   1. Same base_ptr (raw tensor pointer)
//   2. Same tile_index (different tiles are disjoint memory regions)
//   3. Byte ranges [offset, offset+size) intersect within the tile
//
// Note: tile_index represents a tile/block subdivision of the tensor.
// Different tile indices are treated as non-overlapping memory regions,
// even if their offsets are the same (offset is relative to tile start).
//
bool region_overlap(TensorRegion* a, TensorRegion* b) {
    // Must be same base tensor
    if (a->base_ptr != b->base_ptr) {
        return false;
    }
    
    // Must be same tile (different tiles don't overlap)
    if (a->tile_index != b->tile_index) {
        return false;
    }
    
    // Check 1D interval overlap within tile
    int32_t a_start = a->offset;
    int32_t a_end = a_start + a->size;
    int32_t b_start = b->offset;
    int32_t b_end = b_start + b->size;
    
    // Overlap exists if: (a_start < b_end) AND (b_start < a_end)
    return (a_start < b_end) && (b_start < a_end);
}

// Lookup with overlap detection
int32_t tensormap_lookup(TensorMap* tm, TensorRegion* region) {
    uint32_t bucket = tensormap_hash(tm, region);  // Hash only by base_ptr
    int32_t offset = tm->buckets[bucket];
    
    while (offset >= 0) {
        TensorMapEntry* entry = &tm->entry_pool[offset];
        
        // Check validity first (lazy invalidation)
        if (!tensormap_entry_valid(tm, entry)) {
            // Chain truncation (stale entries at tail)
            // ...
            return -1;
        }
        
        // Check for OVERLAP (not just exact match)
        if (region_overlap(&entry->region, region)) {
            return entry->producer_task_id;  // FOUND (overlapping)
        }
        
        offset = entry->next_in_bucket;
    }
    
    return -1;  // Not found
}

// For multi-dimensional tensors, extend overlap check:
bool regions_overlap_nd(TensorRegion* a, TensorRegion* b, int ndim) {
    if (a->base_ptr != b->base_ptr) return false;
    if (a->tile_index != b->tile_index) return false;
    
    // Check overlap in each dimension
    for (int d = 0; d < ndim; d++) {
        int32_t a_start = a->offset[d];
        int32_t a_end = a_start + a->shape[d];
        int32_t b_start = b->offset[d];
        int32_t b_end = b_start + b->shape[d];
        
        // No overlap in this dimension = no overlap at all
        if (a_start >= b_end || b_start >= a_end) {
            return false;
        }
    }
    return true;  // Overlap in all dimensions
}
```

#### Scheduler State (Dynamic, owned by Scheduler)

```c
// ========== WORKER TYPES ==========
typedef enum {
    WORKER_CUBE = 0,      // AICore CUBE unit
    WORKER_VECTOR = 1,    // AICore VECTOR unit
    WORKER_AI_CPU = 2,    // AI_CPU
    WORKER_ACCELERATOR = 3, // Fixed-function accelerators
    NUM_WORKER_TYPES = 4
} WorkerType;

// ========== PER-WORKER-TYPE READY QUEUE ==========
typedef struct {
    int32_t* task_ids;    // Circular buffer of task IDs
    int32_t head;         // Dequeue position
    int32_t tail;         // Enqueue position
    int32_t capacity;
} ReadyQueue;

// ========== TASK STATE ==========
typedef enum {
    TASK_PENDING   = 0,  // Waiting for dependencies (fanin_refcount < fanin_count)
    TASK_READY     = 1,  // All dependencies satisfied, waiting for dispatch
    TASK_RUNNING   = 2,  // Currently executing on a worker
    TASK_COMPLETED = 3,  // Execution finished, but output may still be in use
    TASK_CONSUMED  = 4,  // Output fully consumed (fanout_refcount == fanout_count)
                         // Output buffers can now be released
} TaskState;

// Lifecycle conditions (compare Scheduler refcount with Orchestrator count):
//   Task is READY when:    fanin_refcount[id] == task_descriptors[id].fanin_count
//   Task is CONSUMED when: fanout_refcount[id] == task_descriptors[id].fanout_count
//                          AND task_state[id] == TASK_COMPLETED
```

#### Orchestrator and Scheduler State Structures

```c
// ========== ORCHESTRATOR STATE (Private to Orchestrator) ==========
typedef struct OrchestratorState {
    // Shared memory access
    void* sm_ptr;                     // Orchestrator's view of shared memory
    SharedMemoryHeader* sm_header;    // Quick access to header
    TaskDescriptor* task_descriptors; // Points into shared memory
    int32_t* dep_list_pool;           // Points into shared memory
    int32_t dep_list_pool_next;       // Next free slot in pool
    
    // Heap ring (local state - only top is local, tail read from shared memory)
    HeapRing heap_ring;
    
    // Task ring (local state - only current_index is local)
    int32_t current_task_index;
    
    // === PRIVATE DATA (not in shared memory) ===
    TensorMap tensor_map;             // Producer lookup (ring buffer design)
    int32_t tensormap_last_cleanup;   // Last last_task_alive value when cleanup was done
    
    int32_t* scope_stack;             // Scope tracking (only Orchestrator uses)
    int32_t scope_stack_top;
    int32_t scope_stack_capacity;
    
} OrchestratorState;

// ========== SCHEDULER STATE (Private to Scheduler) ==========
typedef struct SchedulerState {
    // Shared memory access
    void* sm_ptr;                     // Scheduler's view of shared memory
    SharedMemoryHeader* sm_header;    // Quick access to header
    TaskDescriptor* task_descriptors; // Points into shared memory (read-only)
    
    // Local ring pointers (written to shared memory after update)
    int32_t last_task_alive;          // Task ring tail
    int32_t heap_tail;                // Heap ring tail
    
    // === PRIVATE DATA (not in shared memory) ===
    // Per-task state arrays (indexed by task_id % TASK_WINDOW_SIZE)
    int32_t* task_state;              // PENDING/READY/RUNNING/COMPLETED/CONSUMED
    int32_t* fanin_refcount;          // Dynamic: counts completed producers
    int32_t* fanout_refcount;         // Dynamic: counts released references
    
    // Ready queues (one per worker type)
    ReadyQueue ready_queues[NUM_WORKER_TYPES];
    
    // Worker pools
    WorkerPool worker_pools[NUM_WORKER_TYPES];
    
} SchedulerState;
```

**Memory Space Separation:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY SPACE SEPARATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ORCHESTRATOR PROCESS              SCHEDULER PROCESS                        │
│   ───────────────────              ─────────────────                         │
│                                                                              │
│   ┌─────────────────────┐         ┌─────────────────────┐                   │
│   │ OrchestratorState   │         │ SchedulerState      │                   │
│   │ (private memory)    │         │ (private memory)    │                   │
│   │                     │         │                     │                   │
│   │ • TensorMap         │         │ • task_state[]      │                   │
│   │ • scope_stack       │         │ • fanin_refcount[]  │                   │
│   │ • heap_ring.top     │         │ • fanout_refcount[] │                   │
│   │ • current_task_idx  │         │ • ready_queues[]    │                   │
│   │                     │         │ • last_task_alive   │                   │
│   │                     │         │ • heap_tail         │                   │
│   └─────────────────────┘         └─────────────────────┘                   │
│            │                               │                                 │
│            │     ┌─────────────────────┐   │                                 │
│            │     │   SHARED MEMORY     │   │                                 │
│            └────►│                     │◄──┘                                 │
│                  │ • Header (pointers) │                                     │
│                  │ • TaskDescriptor[]  │                                     │
│                  │ • dep_list_pool     │                                     │
│                  └─────────────────────┘                                     │
│                                                                              │
│   Flow Control:                                                              │
│   • Orchestrator writes: current_task_index, heap_top                       │
│   • Scheduler writes: last_task_alive, heap_tail                            │
│   • Each reads the other's pointers for back-pressure                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Implementation Architecture

This section provides a comprehensive description of the runtime implementation, covering:
- Orchestrator and Scheduler decoupling
- Shared memory communication
- Heap-based buffer management
- Efficient buffer release with LastTaskAlive tracking

#### 1. Orchestrator-Scheduler Decoupling

The Orchestrator workload can be demanding (graph construction, dependency tracking, memory allocation). To maximize flexibility and performance, the Orchestrator is **decoupled** from the Scheduler:

- **Orchestrator** can run on **device AI CPU** or **host CPU**
- **Scheduler** runs on device (typically AI CPU)
- Communication via **shared memory window**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR-SCHEDULER DECOUPLING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Option A: Orchestrator on Host CPU                                        │
│   ┌──────────────┐        PCIe/CXL       ┌──────────────┐                   │
│   │  Host CPU    │◄────────────────────► │   Device     │                   │
│   │ Orchestrator │   Shared Memory       │  Scheduler   │                   │
│   └──────────────┘                       │  + Workers   │                   │
│                                          └──────────────┘                   │
│                                                                              │
│   Option B: Orchestrator on Device AI CPU                                   │
│   ┌──────────────────────────────────────────────────────┐                  │
│   │                      Device                           │                  │
│   │  ┌──────────────┐  On-chip Memory  ┌──────────────┐  │                  │
│   │  │   AI CPU     │◄───────────────► │   AI CPU     │  │                  │
│   │  │ Orchestrator │  Shared Memory   │  Scheduler   │  │                  │
│   │  └──────────────┘                  └──────────────┘  │                  │
│   │                                     + AICore Workers │                  │
│   └──────────────────────────────────────────────────────┘                  │
│                                                                              │
│   Both options use IDENTICAL data structures and protocols!                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2. Shared Memory Window

All data structures written by Orchestrator and read by Scheduler reside in a **contiguous shared memory window**.

```c
// ========== SHARED MEMORY SETUP ==========

typedef struct {
    void* sm_orchestrator_ptr;   // Orchestrator's view of shared memory
    void* sm_scheduler_ptr;      // Scheduler's view of shared memory
    int32_t sm_size;             // Size of shared memory window
} SharedMemoryHandle;

// Create shared memory window between Orchestrator and Scheduler
// Returns pointers for both sides (may be same or different depending on environment)
SharedMemoryHandle pto_runtime_create_sm(int32_t size) {
    SharedMemoryHandle handle;
    handle.sm_size = size;
    
#ifdef PTO_SIMULATED_ENVIRONMENT
    // Simulation: single process, same address space
    // Just allocate memory and return same pointer for both
    void* buffer = aligned_alloc(64, size);
    handle.sm_orchestrator_ptr = buffer;
    handle.sm_scheduler_ptr = buffer;  // Same pointer!
    
#elif defined(PTO_HOST_ORCHESTRATOR)
    // Host orchestrator: allocate PCIe-accessible shared memory
    // Device maps same physical memory
    handle.sm_orchestrator_ptr = pcie_alloc_shared(size);
    handle.sm_scheduler_ptr = device_map_shared(handle.sm_orchestrator_ptr);
    
#else
    // Device orchestrator: allocate on-chip shared memory
    // Both Orchestrator and Scheduler AI CPUs access same memory
    void* buffer = device_shared_alloc(size);
    handle.sm_orchestrator_ptr = buffer;
    handle.sm_scheduler_ptr = buffer;
#endif
    
    return handle;
}

void pto_runtime_destroy_sm(SharedMemoryHandle* handle) {
#ifdef PTO_SIMULATED_ENVIRONMENT
    free(handle->sm_orchestrator_ptr);
#elif defined(PTO_HOST_ORCHESTRATOR)
    pcie_free_shared(handle->sm_orchestrator_ptr);
#else
    device_shared_free(handle->sm_orchestrator_ptr);
#endif
}
```

**Shared Memory Layout:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SHARED MEMORY LAYOUT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SM_Base ──────────────────────────────────────────────────────────────►   │
│   │                                                                          │
│   │  ONLY data needed for Orchestrator↔Scheduler communication              │
│   │                                                                          │
│   │  ┌────────────────────────────────────────────────────────────────┐     │
│   │  │  HEADER (flow control + sync)                                  │     │
│   │  │  - current_task_index (Orch→Sched)                             │     │
│   │  │  - heap_top (Orch→Sched)                                       │     │
│   │  │  - last_task_alive (Sched→Orch, for back-pressure)             │     │
│   │  │  - heap_tail (Sched→Orch, for back-pressure)                   │     │
│   │  │  - orchestrator_done flag                                      │     │
│   │  └────────────────────────────────────────────────────────────────┘     │
│   │                                                                          │
│   │  ┌────────────────────────────────────────────────────────────────┐     │
│   │  │  TASK DESCRIPTORS[TASK_WINDOW_SIZE]                            │     │
│   │  │  - Orchestrator writes, Scheduler reads                        │     │
│   │  │  - Contains: kernel_id, worker_type, fanin/fanout lists,       │     │
│   │  │    packed_buffer_base/end, etc.                                │     │
│   │  └────────────────────────────────────────────────────────────────┘     │
│   │                                                                          │
│   │  ┌────────────────────────────────────────────────────────────────┐     │
│   │  │  DEPENDENCY LIST POOL (fanin_list, fanout_list storage)        │     │
│   │  │  - Orchestrator writes, Scheduler reads                        │     │
│   │  └────────────────────────────────────────────────────────────────┘     │
│   │                                                                          │
│   └──────────────────────────────────────────────────────────────────────►  │
│                                                                              │
│   NOTE: TensorMap, scope_stack, ready_queues, refcounts are NOT here!       │
│         They are in private memory spaces (see below).                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

ORCHESTRATOR PRIVATE MEMORY (not shared):
┌─────────────────────────────────────────────────────────────────────────────┐
│  - TensorMap (ring buffer pool with lazy invalidation):                     │
│      • entry_pool[TENSORMAP_POOL_SIZE]  - Ring buffer of TensorMapEntry     │
│      • buckets[NUM_BUCKETS]             - Hash table buckets                │
│      • task_entry_head[TASK_WINDOW_SIZE]- Per-task entry list heads         │
│      • pool_head                        - Next allocation position          │
│      • last_task_alive                  - Cached validity threshold         │
│  - scope_stack[] (only Orchestrator uses for scope tracking)                │
│  - tensormap_last_cleanup (tracks last cleanup point)                       │
│  - Local heap_ring state (top pointer)                                      │
│  - Local task_ring state (current_index)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

SCHEDULER PRIVATE MEMORY (not shared):
┌─────────────────────────────────────────────────────────────────────────────┐
│  - task_state[TASK_WINDOW_SIZE] (Scheduler manages task lifecycle)          │
│  - fanin_refcount[TASK_WINDOW_SIZE] (dynamic dependency tracking)           │
│  - fanout_refcount[TASK_WINDOW_SIZE] (dynamic reference counting)           │
│  - ready_queues[NUM_WORKER_TYPES] (per-worker-type dispatch queues)         │
│  - Local last_task_alive, heap_tail (before writing to shared memory)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

```c
// ========== SHARED MEMORY STRUCTURE ==========
// ONLY contains data needed for Orchestrator↔Scheduler communication

typedef struct SharedMemoryHeader {
    // === FLOW CONTROL POINTERS ===
    // Written by Orchestrator, Read by Scheduler
    volatile int32_t current_task_index;  // Task ring head (next to allocate)
    volatile int32_t heap_top;            // Heap ring allocation pointer
    volatile int32_t orchestrator_done;   // Flag: orchestration complete
    
    // Written by Scheduler, Read by Orchestrator (for back-pressure)
    volatile int32_t last_task_alive;     // Task ring tail (oldest active task)
    volatile int32_t heap_tail;           // Heap ring free pointer
    
    // === LAYOUT INFO (set once at init) ===
    int32_t task_window_size;             // TASK_WINDOW_SIZE
    int32_t heap_size;                    // Total heap size
    int32_t dep_list_pool_size;           // Dependency list pool size
    
    // Offsets into shared memory (relative to SM_Base)
    int32_t task_descriptors_offset;
    int32_t dep_list_pool_offset;
    
} SharedMemoryHeader;
```

#### Dependency List Pool (dep_list_pool) Design

The `fanout_list` of a task needs to **grow dynamically** as new consumer tasks are submitted:

```
Timeline:
  1. Task A submitted (produces output O)    → A.fanout_list = []
  2. Task B submitted (consumes O)           → A.fanout_list = [B]
  3. Task C submitted (consumes O)           → A.fanout_list = [B, C]
  4. Task D submitted (consumes O)           → A.fanout_list = [B, C, D]
```

**Design: Linked List with Pool Allocation**

Each dependency list entry contains a task_id and a pointer to the next entry. Adding a new entry **prepends** to the list head, making it O(1).

```c
// ========== DEPENDENCY LIST ENTRY ==========
// Singly-linked list node for fanin/fanout lists
typedef struct DepListEntry {
    int32_t task_id;              // The dependent/dependency task ID
    int32_t next_offset;          // Offset to next entry (0 = end of list)
} DepListEntry;

// ========== DEPENDENCY LIST POOL ==========
// Ring buffer pool for allocating DepListEntry nodes
typedef struct DepListPool {
    DepListEntry* base;           // Pool base address (in shared memory)
    int32_t capacity;             // Total number of entries
    int32_t top;                  // Next allocation position
    // tail advances with last_task_alive (implicitly reclaimed)
} DepListPool;

// ========== TASK DESCRIPTOR (updated) ==========
typedef struct TaskDescriptor {
    // ... other fields ...
    
    // Dependency lists as linked list heads (offset into dep_list_pool)
    int32_t fanin_head;           // Offset to first fanin entry (0 = empty)
    int32_t fanin_count;          // Number of entries in fanin list
    int32_t fanout_head;          // Offset to first fanout entry (0 = empty)
    int32_t fanout_count;         // Number of entries in fanout list
    
    // ... other fields ...
} TaskDescriptor;
```

**Prepend Operation (O(1)):**

```c
// Allocate a single entry from the pool
static int32_t dep_list_pool_alloc_one(DepListPool* pool) {
    if (pool->top >= pool->capacity) {
        // Wrap around to beginning (old entries reclaimed with task ring)
        pool->top = 1;  // Start from 1, 0 means NULL/empty
    }
    return pool->top++;
}

// Prepend a task_id to a dependency list (O(1) operation)
// Returns new head offset
static int32_t dep_list_prepend(DepListPool* pool, 
                                 int32_t current_head,
                                 int32_t task_id) {
    // Allocate new entry
    int32_t new_offset = dep_list_pool_alloc_one(pool);
    DepListEntry* new_entry = &pool->base[new_offset];
    
    // Fill in new entry: points to old head
    new_entry->task_id = task_id;
    new_entry->next_offset = current_head;  // Link to previous head
    
    return new_offset;  // New head
}

// Usage: Add consumer to producer's fanout list
void add_consumer_to_producer(DepListPool* pool,
                               TaskDescriptor* producer,
                               int32_t consumer_id) {
    producer->fanout_head = dep_list_prepend(pool, 
                                              producer->fanout_head,
                                              consumer_id);
    producer->fanout_count++;
}
```

**Traversal (for Scheduler):**

```c
// Iterate through a dependency list
void iterate_dep_list(DepListPool* pool, int32_t head_offset,
                      void (*callback)(int32_t task_id, void* ctx), void* ctx) {
    int32_t current = head_offset;
    while (current != 0) {
        DepListEntry* entry = &pool->base[current];
        callback(entry->task_id, ctx);
        current = entry->next_offset;
    }
}

// Example: Scheduler iterates producer's fanout_list
void on_task_complete(SchedulerState* sched, int32_t task_id) {
    TaskDescriptor* task = &sched->task_descriptors[task_id];
    
    // Iterate fanout_list to update consumers' fanin_refcount
    int32_t current = task->fanout_head;
    while (current != 0) {
        DepListEntry* entry = &sched->dep_list_pool->base[current];
        int32_t consumer_id = entry->task_id;
        
        sched->fanin_refcount[consumer_id]++;
        // ... check if consumer is ready ...
        
        current = entry->next_offset;
    }
}
```

**Visualization:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY LIST PREPEND OPERATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Before: Producer A has fanout_list = [B, C]                               │
│                                                                              │
│   TaskDescriptor A:                     DepListPool:                        │
│   ┌──────────────────┐                  ┌─────┬─────┬─────┬─────┬─────┐    │
│   │ fanout_head: 5   │─────────────────►│ ... │  B  │  C  │     │     │    │
│   │ fanout_count: 2  │                  │     │ →6  │ →0  │     │     │    │
│   └──────────────────┘                  └─────┴─────┴─────┴─────┴─────┘    │
│                                           idx:  5     6     7               │
│                                                                              │
│   After: dep_list_prepend(pool, A.fanout_head, D)                           │
│          Producer A has fanout_list = [D, B, C]                             │
│                                                                              │
│   TaskDescriptor A:                     DepListPool:                        │
│   ┌──────────────────┐                  ┌─────┬─────┬─────┬─────┬─────┐    │
│   │ fanout_head: 7   │─────────────────►│ ... │  B  │  C  │  D  │     │    │
│   │ fanout_count: 3  │                  │     │ →6  │ →0  │ →5  │     │    │
│   └──────────────────┘                  └─────┴─────┴─────┴─────┴─────┘    │
│                                           idx:  5     6     7     8         │
│                                                 ▲                 │         │
│                                                 └─────────────────┘         │
│                                                    new entry points         │
│                                                    to old head              │
│                                                                              │
│   KEY: Prepend is O(1) - just allocate one entry and update head pointer   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Handling fanout_list Growth:**

When a new consumer is added to a producer's fanout_list, we have two options:

**Option 1: Copy-on-Grow (Simple)**
```c
// When adding consumer to producer's fanout_list
static void add_to_fanout_list(OrchestratorState* orch, 
                                TaskDescriptor* producer, 
                                int32_t consumer_id) {
    int32_t old_size = producer->fanout_list_size;
    int32_t new_size = old_size + 1;
    
    // Allocate new space (old space becomes garbage, reclaimed with ring wrap)
    int32_t* new_list = dep_list_pool_alloc(&orch->dep_list_pool, new_size);
    
    // Copy old entries + add new one
    for (int i = 0; i < old_size; i++) {
        new_list[i] = producer->fanout_list[i];
    }
    new_list[old_size] = consumer_id;
    
    // Update producer's pointer
    producer->fanout_list = new_list;
    producer->fanout_list_size = new_size;
}
```

**Option 2: Pre-allocated Chunks with Overflow (Efficient)**
```c
// Pre-allocate chunk with capacity, grow when needed
#define FANOUT_CHUNK_SIZE 8  // Initial capacity per task

typedef struct FanoutListHeader {
    int32_t capacity;         // Current chunk capacity
    int32_t size;             // Current number of entries
    int32_t entries[];        // Flexible array member
} FanoutListHeader;

static void add_to_fanout_list_chunked(OrchestratorState* orch,
                                        TaskDescriptor* producer,
                                        int32_t consumer_id) {
    FanoutListHeader* header = (FanoutListHeader*)producer->fanout_list_ptr;
    
    if (header->size < header->capacity) {
        // Room available - just append
        header->entries[header->size++] = consumer_id;
    } else {
        // Need to grow - allocate larger chunk
        int32_t new_capacity = header->capacity * 2;
        int32_t alloc_size = sizeof(FanoutListHeader)/sizeof(int32_t) + new_capacity;
        
        FanoutListHeader* new_header = (FanoutListHeader*)
            dep_list_pool_alloc(&orch->dep_list_pool, alloc_size);
        
        new_header->capacity = new_capacity;
        new_header->size = header->size + 1;
        
        // Copy old entries + add new
        for (int i = 0; i < header->size; i++) {
            new_header->entries[i] = header->entries[i];
        }
        new_header->entries[header->size] = consumer_id;
        
        producer->fanout_list_ptr = new_header;
    }
}
```

**Memory Reclamation:**

The key insight is that dep_list_pool memory is **implicitly reclaimed** when the task ring wraps around:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEP_LIST_POOL RING BUFFER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Task Ring:    [CONS|CONS|CONS|COMP|READY|RUN|PEND|FREE|FREE]              │
│                              ↑                   ↑                          │
│                       last_task_alive    current_task_index                 │
│                                                                              │
│   Dep List Pool:                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ OLD LISTS │ ACTIVE LISTS (for tasks 3-6) │ FREE SPACE          │       │
│   │ (garbage) │ (still needed by Scheduler)  │                     │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│        ↑                                            ↑                        │
│   Implicitly                                       top                       │
│   reclaimed when                              (next alloc)                   │
│   tasks consumed                                                             │
│                                                                              │
│   NOTE: "Garbage" from copy-on-grow is fine - it's reclaimed when           │
│   task ring wraps around and those tasks become CONSUMED                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pool Sizing:**

```
dep_list_pool_size = TASK_WINDOW_SIZE * AVG_FANOUT * GROWTH_FACTOR
                   = 1024 * 4 * 2
                   = 8192 int32_t elements
                   = 32KB
```

---

#### 3. Ring Buffer Design for Zero-Overhead Memory Management

All intermediate data (task outputs) are allocated from a **contiguous heap** (GM Heap) in Global Memory. To eliminate memory allocation/free overhead entirely, both the **Heap** and **Task Table** are organized as **wrap-around ring buffers**. This provides natural flow control (back-pressure) between the Orchestrator and Scheduler.

```c
// ========== HEAP RING BUFFER ==========
// Organized as a wrap-around ring buffer for zero-overhead allocation/free
typedef struct HeapRing {
    void*    base;            // GM_Heap_Base
    int32_t  size;            // GM_Heap_Size (total heap size in bytes)
    int32_t  top;             // Allocation pointer (advances on alloc, wraps around)
    // tail is read from shared memory (written by Scheduler)
} HeapRing;

// ========== TASK RING BUFFER ==========
// Fixed-size wrap-around task window for flow control
#define TASK_WINDOW_SIZE 1024  // Power of 2 for efficient modulo

typedef struct TaskRing {
    TaskDescriptor* descriptors;  // Task descriptor array [TASK_WINDOW_SIZE]
    int32_t current_index;        // Next task to allocate (wraps around)
    // last_task_alive is read from shared memory (written by Scheduler)
} TaskRing;
```

**Key Design Principles:**
- **No explicit free operation** - Memory is reclaimed automatically when `last_task_alive` advances
- **Allocator stalls** if insufficient space - provides back-pressure to Orchestrator
- **Wrap-around handling** - Never split a buffer across ring boundary; skip to beginning if needed
- **Shared memory pointers** - `last_task_alive` and `heap_tail` in shared memory for Orchestrator to read

#### Ring Buffer Flow Control Diagram

The runtime uses **five ring buffers** that implement flow control. When any buffer is exhausted, the orchestrator blocks until the scheduler frees resources.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RING BUFFER FLOW CONTROL (ALL 5 RINGS)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ORCHESTRATOR (Producer)                   SCHEDULER (Consumer)            │
│   ───────────────────────                   ────────────────────            │
│   • Allocates task slots                    • Completes tasks               │
│   • Inserts TensorMap entries               • Advances last_task_alive      │
│   • Allocates DepList nodes                 • Frees all resources at once   │
│   • Allocates heap buffers                                                   │
│   • Enqueues ready tasks                                                     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  1. TASK RING (PTO_TASK_WINDOW_SIZE = 8192)                      │      │
│   │     Condition: window_not_full                                    │      │
│   │                                                                   │      │
│   │     last_task_alive              next_task_id                     │      │
│   │           │                           │                           │      │
│   │           ▼                           ▼                           │      │
│   │     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │      │
│   │     │CONS │COMP │READY│ RUN │PEND │FREE │FREE │                  │      │
│   │     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │      │
│   │     ◄────── tasks_in_flight ──────►                              │      │
│   │                                                                   │      │
│   │     STALL: tasks_in_flight >= PTO_TASK_WINDOW_SIZE               │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  2. TENSORMAP POOL (PTO_TENSORMAP_POOL_SIZE = 262144)            │      │
│   │     Condition: tensormap_not_full                                 │      │
│   │                                                                   │      │
│   │              pool_head (allocate)                                 │      │
│   │                   │                                               │      │
│   │                   ▼                                               │      │
│   │     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │      │
│   │     │stale│stale│LIVE │LIVE │LIVE │ ... │next │                  │      │
│   │     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │      │
│   │           ▲                                                       │      │
│   │     Stale entries (producer_task_id < last_task_alive)           │      │
│   │     can be reused; LIVE entries block allocation                 │      │
│   │                                                                   │      │
│   │     STALL: entry.producer_task_id >= last_task_alive             │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  3. DEPLIST POOL (PTO_DEP_LIST_POOL_SIZE = 131072)               │      │
│   │     Condition: deplist_not_full                                   │      │
│   │                                                                   │      │
│   │          dep_list_tail                    dep_list_top            │      │
│   │               │                               │                   │      │
│   │               ▼                               ▼                   │      │
│   │     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │      │
│   │     │FREE │fanin│fanin│fano │fano │ ... │next │                  │      │
│   │     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │      │
│   │     ◄free►◄──────── ACTIVE ENTRIES ────────►                     │      │
│   │                                                                   │      │
│   │     STALL: (dep_list_top + 1) % SIZE == dep_list_tail            │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  4. HEAP RING (PTO_HEAP_SIZE_BYTES = 1GB)                        │      │
│   │     Condition: heap_not_full                                      │      │
│   │                                                                   │      │
│   │         heap_tail                         heap_top                │      │
│   │              │                               │                    │      │
│   │              ▼                               ▼                    │      │
│   │     ┌────────┬───────────────────────────────┬─────────────┐     │      │
│   │     │  FREE  │     PACKED OUTPUT BUFFERS     │  FREE SPACE │     │      │
│   │     └────────┴───────────────────────────────┴─────────────┘     │      │
│   │                        (wraps around)                            │      │
│   │                                                                   │      │
│   │     STALL: No contiguous space >= request_size                   │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  5. READY QUEUE (PTO_MAX_READY_QUEUE = 65536)                    │      │
│   │     Condition: ready_queue_not_full                               │      │
│   │                                                                   │      │
│   │          ready_head                        ready_tail             │      │
│   │              │                                │                   │      │
│   │              ▼                                ▼                   │      │
│   │     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │      │
│   │     │t123 │t124 │t125 │ ... │    │    │    │                     │      │
│   │     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │      │
│   │     ◄──────── ready_count ────────►                              │      │
│   │                                                                   │      │
│   │     STALL: ready_count >= PTO_MAX_READY_QUEUE (rare)             │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════       │
│   RESOURCE RELEASE: When scheduler calls pto_advance_last_task_alive():     │
│                                                                              │
│   1. Task slot becomes FREE (task.is_consumed = true)                        │
│   2. TensorMap entries become stale (producer_task_id < last_task_alive)    │
│   3. DepList tail advances (dep_list_tail = task.dep_pool_end)              │
│   4. Heap tail advances (heap_tail = task.packed_buffer_end)                │
│   5. Broadcast ALL condition variables to wake blocked orchestrator         │
│   ═══════════════════════════════════════════════════════════════════       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Flow Control Implementation Details

**1. Task Ring Flow Control**
```c
// In pto_task_alloc_impl()
if (tasks_in_flight >= PTO_TASK_WINDOW_SIZE) {
    if (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE) {
        rt->flow_stats.current_stall = PTO_STALL_TASK_RING;
        rt->flow_stats.task_ring_stalls++;
        int64_t stall_start = pto_get_time_ns();
        
        pthread_mutex_lock(&rt->task_mutex);
        while ((rt->next_task_id - rt->last_task_alive) >= PTO_TASK_WINDOW_SIZE) {
            pthread_cond_wait(&rt->window_not_full, &rt->task_mutex);
        }
        pthread_mutex_unlock(&rt->task_mutex);
        
        rt->flow_stats.task_ring_stall_ns += pto_get_time_ns() - stall_start;
        rt->flow_stats.current_stall = PTO_STALL_NONE;
    }
}
```

**2. TensorMap Pool Flow Control**
```c
// In pto_tensormap_insert()
TensorMapEntry* entry = &tm->entry_pool[tm->pool_head];
if (entry->in_bucket && entry->producer_task_id >= rt->last_task_alive) {
    // Entry still in use by live task - wait
    if (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE) {
        rt->flow_stats.current_stall = PTO_STALL_TENSORMAP_POOL;
        rt->flow_stats.tensormap_pool_stalls++;
        
        while (entry->in_bucket && entry->producer_task_id >= rt->last_task_alive) {
            pthread_cond_wait(&rt->tensormap_not_full, &rt->task_mutex);
            tm->last_task_alive = rt->last_task_alive;  // Refresh
        }
        
        rt->flow_stats.current_stall = PTO_STALL_NONE;
    }
}
```

**3. DepList Pool Flow Control**
```c
// In pto_dep_list_alloc_one_locked()
int32_t next = (rt->dep_list_top + 1) % PTO_DEP_LIST_POOL_SIZE;
if (next == rt->dep_list_tail) {
    if (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE) {
        rt->flow_stats.current_stall = PTO_STALL_DEPLIST_POOL;
        rt->flow_stats.deplist_pool_stalls++;
        
        while (next == rt->dep_list_tail) {
            pthread_cond_wait(&rt->deplist_not_full, &rt->task_mutex);
            next = (rt->dep_list_top + 1) % PTO_DEP_LIST_POOL_SIZE;
        }
        
        rt->flow_stats.current_stall = PTO_STALL_NONE;
    }
}
```

**4. Heap Ring Flow Control**
```c
// In pto_heap_alloc_locked()
while (1) {
    // Try to allocate...
    if (/* no space */) {
        if (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE) {
            if (!stall_recorded) {
                rt->flow_stats.current_stall = PTO_STALL_HEAP_RING;
                rt->flow_stats.heap_ring_stalls++;
                stall_start = pto_get_time_ns();
                stall_recorded = true;
            }
            pthread_cond_wait(&rt->heap_not_full, &rt->task_mutex);
            continue;
        }
        return NULL;  // Non-threaded mode: fail immediately
    }
    // Success - record stall time if we waited
    if (stall_recorded) {
        rt->flow_stats.heap_ring_stall_ns += pto_get_time_ns() - stall_start;
        rt->flow_stats.current_stall = PTO_STALL_NONE;
    }
    return ptr;
}
```

**5. Resource Release and Broadcast**
```c
// In scheduler task completion path
bool window_advanced = pto_advance_last_task_alive_locked(rt);
if (window_advanced) {
    // Wake up ALL blocked orchestrator threads
    pthread_cond_broadcast(&rt->window_not_full);
    pthread_cond_broadcast(&rt->tensormap_not_full);
    pthread_cond_broadcast(&rt->deplist_not_full);
    pthread_cond_broadcast(&rt->heap_not_full);
}
```

> **Note**: For detailed flow control statistics API and performance tuning guidance, see the [Flow Control and Backpressure Mechanism](#flow-control-and-backpressure-mechanism) section.

#### 4. Heap Ring Allocation (with Wrap-Around and Stall)

```c
// ========== HEAP RING ALLOCATION ==========

// Allocate from heap ring (O(1), stalls if insufficient space)
// IMPORTANT: Never split buffer across ring boundary - skip to beginning instead
void* heap_ring_alloc(HeapRing* ring, int32_t size, volatile int32_t* tail_ptr) {
    size = ALIGN_UP(size, 64);  // Align for DMA efficiency
    
    // Spin-wait if insufficient space (back-pressure from Scheduler)
    while (true) {
        // Read latest tail from shared memory (Scheduler updates this)
        int32_t tail = *tail_ptr;
        int32_t top = ring->top;
        
        if (top >= tail) {
            // Case 1: top is at or ahead of tail (normal case)
            //   [....tail====top......]
            //                   ^-- space_at_end = size - top
            
            int32_t space_at_end = ring->size - top;
            
            if (space_at_end >= size) {
                // Enough space at end - allocate here
                void* ptr = (char*)ring->base + top;
                ring->top = top + size;
                return ptr;
            }
            
            // Not enough space at end - check if we can wrap to beginning
            // IMPORTANT: Don't split buffer, skip remaining space at end
            if (tail > size) {
                // Wrap to beginning (space available: [0, tail))
                ring->top = size;
                return ring->base;
            }
            
            // Not enough space anywhere - STALL and wait for tail to advance
            continue;
            
        } else {
            // Case 2: top has wrapped, tail is ahead
            //   [====top....tail=====]
            //         ^-- free space = tail - top
            
            int32_t gap = tail - top;
            if (gap >= size) {
                void* ptr = (char*)ring->base + top;
                ring->top = top + size;
                return ptr;
            }
            
            // Not enough space - STALL and wait
            continue;
        }
    }
}

// NO EXPLICIT FREE OPERATION!
// Tail is advanced by Scheduler when last_task_alive moves forward
```

#### 5. Task Ring Allocation (with Wrap-Around and Stall)

```c
// ========== TASK RING ALLOCATION ==========

// Get task slot with flow control (stalls if task window full)
int32_t task_ring_alloc(TaskRing* ring, volatile int32_t* last_alive_ptr) {
    while (true) {
        // Read latest last_task_alive from shared memory
        int32_t last_alive = *last_alive_ptr;
        int32_t current = ring->current_index;
        
        // Calculate number of active tasks (handles wrap-around)
        int32_t active_count;
        if (current >= last_alive) {
            active_count = current - last_alive;
        } else {
            // Wrapped: current < last_alive
            active_count = TASK_WINDOW_SIZE - last_alive + current;
        }
        
        // Check if there's room for one more task
        // Leave at least 1 slot empty to distinguish full from empty
        if (active_count < TASK_WINDOW_SIZE - 1) {
            int32_t task_id = current;
            ring->current_index = (current + 1) % TASK_WINDOW_SIZE;
            return task_id;
        }
        
        // Task window full - STALL and wait for Scheduler to free slots
        continue;
    }
}

// NO EXPLICIT FREE OPERATION!
// Task slots are freed when Scheduler advances last_task_alive
```

#### 6. Scheduler Ring Pointer Advancement

```c
// ========== SCHEDULER ADVANCES RING POINTERS ==========

// Called when a task transitions to CONSUMED state
static void scheduler_on_task_consumed(PTORuntime* rt, int32_t task_id) {
    SchedulerState* sched = &rt->sched_state;
    
    // Check if this enables ring pointer advancement
    // (only advance if the CONSUMED task is at the tail of the ring)
    if (task_id == sched->last_task_alive) {
        advance_ring_pointers(rt);
    }
}

// Advance both task ring and heap ring pointers
static void advance_ring_pointers(PTORuntime* rt) {
    SchedulerState* sched = &rt->sched_state;
    SharedMemoryHeader* sm = (SharedMemoryHeader*)rt->sm_scheduler_ptr;
    TaskDescriptor* tasks = rt->task_descriptors;
    
    int32_t old_last_alive = sched->last_task_alive;
    int32_t current_index = sm->current_task_index;
    
    // Advance last_task_alive until we find a non-CONSUMED task
    while (sched->last_task_alive != current_index &&
           sched->task_state[sched->last_task_alive] == TASK_CONSUMED) {
        
        sched->last_task_alive = (sched->last_task_alive + 1) % TASK_WINDOW_SIZE;
    }
    
    // Update heap_tail based on the NEW last_task_alive position
    // All tasks before last_task_alive are CONSUMED → their buffers are free
    if (sched->last_task_alive != old_last_alive) {
        // Get the task just before last_task_alive (the last consumed task)
        int32_t last_consumed = (sched->last_task_alive + TASK_WINDOW_SIZE - 1) % TASK_WINDOW_SIZE;
        TaskDescriptor* task = &tasks[last_consumed];
        
        if (task->packed_buffer_end != NULL) {
            // heap_tail = end of last consumed task's buffer
            int32_t new_tail = (char*)task->packed_buffer_end - (char*)rt->heap_ring.base;
            sched->heap_tail = new_tail;
            
            // Write to shared memory (Orchestrator will read this for flow control)
            sm->heap_tail = new_tail;
        }
        
        // Write last_task_alive to shared memory (for flow control)
        sm->last_task_alive = sched->last_task_alive;
    }
}
```

#### 7. Heap Ring Wrap-Around Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEAP RING WRAP-AROUND                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Initial state (linear allocation):                                        │
│                                                                              │
│      tail=0                                top                               │
│        │                                    │                                │
│        ▼                                    ▼                                │
│   ┌────┬─────────────────────────────────────┬────────────────────────┐     │
│   │    │ Task0  Task1  Task2  Task3  Task4   │      FREE SPACE        │     │
│   └────┴─────────────────────────────────────┴────────────────────────┘     │
│   0                                                                  SIZE   │
│                                                                              │
│   After some tasks consumed (tail advances):                                │
│                                                                              │
│               tail                          top                              │
│                │                             │                               │
│                ▼                             ▼                               │
│   ┌────────────┬─────────────────────────────┬────────────────────────┐     │
│   │  RETIRED   │ Task3  Task4  Task5  Task6  │      FREE SPACE        │     │
│   │  (freed)   │ (still active)              │                        │     │
│   └────────────┴─────────────────────────────┴────────────────────────┘     │
│                                                                              │
│   After wrap-around (new allocation at beginning):                          │
│                                                                              │
│                              tail                                            │
│                               │                                              │
│                               ▼                                              │
│   ┌────────────────┬──────────┬──────────────────────────────────────┐      │
│   │ Task7  Task8   │  FREE    │ Task5  Task6 (still active)         │      │
│   └────────────────┴──────────┴──────────────────────────────────────┘      │
│         ▲                                                                    │
│         │                                                                    │
│        top (wrapped to beginning)                                            │
│                                                                              │
│   IMPORTANT: When buffer would cross end boundary, skip to beginning        │
│   Never split a single buffer across the wrap-around point!                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 8. Benefits of Ring Buffer Design

| Aspect | Benefit |
|--------|---------|
| **Zero alloc overhead** | O(1) bump allocation, no search or bookkeeping |
| **Zero free overhead** | No explicit free - tail advances automatically |
| **Natural flow control** | Back-pressure when rings are full (stall) |
| **No fragmentation** | Buffers are contiguous, freed in order |
| **Bounded memory** | Fixed heap size and task window size |
| **Lock-free reads** | Orchestrator reads volatile pointers (no lock needed) |
| **Simple implementation** | Modulo arithmetic, no complex data structures |

---

### Concurrency Analysis and Fine-Grained Locking

When Orchestrator and Scheduler operate **in parallel**, we need to carefully analyze data access patterns to identify race conditions and minimize lock overhead.

#### Data Structure Access Pattern Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONCURRENT ACCESS ANALYSIS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DATA STRUCTURE              ORCHESTRATOR         SCHEDULER       LOCK?    │
│   ══════════════              ════════════         ═════════       ═════    │
│                                                                              │
│   SHARED MEMORY HEADER                                                       │
│   ────────────────────                                                       │
│   current_task_index          WRITE                READ            NO (1)   │
│   heap_top                    WRITE                READ            NO (1)   │
│   last_task_alive             READ                 WRITE           NO (1)   │
│   heap_tail                   READ                 WRITE           NO (1)   │
│   orchestrator_done           WRITE                READ            NO (1)   │
│                                                                              │
│   TASK DESCRIPTORS                                                           │
│   ────────────────                                                           │
│   TaskDescriptor[i] fields    WRITE (new tasks)    READ            NO (2)   │
│   (except fanout_head)                                                       │
│                                                                              │
│   DEPENDENCY LISTS (CRITICAL!)                                               │
│   ─────────────────────────────                                              │
│   task.fanout_head            WRITE (prepend)      READ (iterate)  YES (3)  │
│   task.fanout_count           WRITE (increment)    READ            YES (3)  │
│   DepListPool entries         WRITE (new entries)  READ            NO (4)   │
│                                                                              │
│   ORCHESTRATOR PRIVATE                                                       │
│   ────────────────────                                                       │
│   TensorMap                   READ/WRITE           -               NO       │
│   scope_stack                 READ/WRITE           -               NO       │
│   heap_ring.top               READ/WRITE           -               NO       │
│   task_ring.current_index     READ/WRITE           -               NO       │
│                                                                              │
│   SCHEDULER PRIVATE                                                          │
│   ─────────────────                                                          │
│   task_state[]                -                    READ/WRITE      NO       │
│   fanin_refcount[]            -                    READ/WRITE      NO       │
│   fanout_refcount[]           -                    READ/WRITE      NO       │
│   ready_queues[]              -                    READ/WRITE      NO       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

NOTES:
(1) Single-word volatile read/write - atomic on most architectures
(2) Orchestrator completes write before updating current_task_index;
    Scheduler only reads tasks where index < current_task_index
(3) RACE CONDITION - see detailed analysis below
(4) Append-only pool - new entries don't affect existing reads
```

#### Critical Race Condition 1: fanout_list Iteration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RACE CONDITION: LIST ITERATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Timeline showing concurrent access to Task A's fanout_list:               │
│                                                                              │
│   ORCHESTRATOR                          SCHEDULER                            │
│   ════════════                          ═════════                            │
│                                                                              │
│   T1: Submit Task A                                                         │
│       A.fanout_head = 0 (empty)                                             │
│                                                                              │
│   T2: Submit Task B (consumes A)                                            │
│       A.fanout_head = [B]                                                   │
│                                                                              │
│   T3:                                   Task A starts executing              │
│                                                                              │
│   T4: Submit Task C (consumes A)                                            │
│       prepend C to A.fanout_list ──┐                                        │
│       A.fanout_head = [C]→[B]      │    Task A completes                    │
│                               ─────┼───►on_task_complete(A):                │
│                                    │      iterate A.fanout_list             │
│                                    │      current = A.fanout_head ◄── RACE! │
│       (writing new head)     ──────┘                                        │
│                                                                              │
│   PROBLEM: Scheduler may see partially updated fanout_head                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Critical Race Condition 2: fanout_count Consistency (MORE SEVERE!)

**This is a correctness bug, not just a data race!**

The `fanout_count` and `fanout_list` must be updated atomically together. Otherwise:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RACE CONDITION: COUNT INCONSISTENCY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Initial state: Task A with fanout_count = 2 (scope_depth=1, 1 consumer)   │
│                  fanout_refcount = 1 (one consumer completed)               │
│                                                                              │
│   ORCHESTRATOR                          SCHEDULER                            │
│   ════════════                          ═════════                            │
│                                                                              │
│   Submit Task C (consumes A):           scope_end() for A's scope:          │
│                                                                              │
│   // Step 1: increment count            // Check if A is CONSUMED:          │
│   A.fanout_count++;  ─────────┐         fanout_refcount[A]++;  // now = 2   │
│   // A.fanout_count now = 3   │                    │                        │
│                               │         if (fanout_refcount == fanout_count)│
│                               │              │                              │
│                               └─────────────►│ reads fanout_count = 2 !!!   │
│                                              │ (before increment visible)   │
│                                              │                              │
│   // Step 2: prepend to list               2 == 2 → TRUE!                   │
│   A.fanout_head = prepend(C)               A is marked CONSUMED!            │
│                                            A's buffer is RELEASED!          │
│                                                                              │
│   // Step 3: ... but C still needs A's output!                              │
│   C.fanin_list includes A                  USE-AFTER-FREE BUG!              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ROOT CAUSE: fanout_count increment and fanout_list update are NOT atomic  │
│                                                                              │
│   REQUIRED ATOMIC OPERATIONS:                                               │
│                                                                              │
│   ORCHESTRATOR must atomically:          SCHEDULER must atomically:         │
│   ┌──────────────────────────┐          ┌──────────────────────────┐       │
│   │ 1. fanout_count++        │          │ 1. read fanout_count     │       │
│   │ 2. prepend to fanout_list│          │ 2. compare with refcount │       │
│   └──────────────────────────┘          │ 3. decide CONSUMED or not│       │
│   These must appear atomic              └──────────────────────────┘       │
│   to Scheduler                          Must see consistent count          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this is critical:**

| If not atomic... | Consequence |
|------------------|-------------|
| Scheduler sees old `fanout_count` | Task marked CONSUMED too early → **use-after-free** |
| Scheduler sees new `fanout_count` but old list | Misses notifying new consumer → **deadlock** |
| Counter and list out of sync | Data corruption, undefined behavior |

**Conclusion: Per-task lock is REQUIRED for correctness, not just performance!**

#### Solution: Per-Task Fine-Grained Lock (REQUIRED for Correctness)

The lock is **mandatory** to ensure atomicity between `fanout_count` and `fanout_list`:

```c
// ========== PER-TASK SPINLOCK ==========
// REQUIRED: Protects fanout_count + fanout_list consistency

typedef struct TaskDescriptor {
    // ... other fields ...
    
    volatile int32_t fanout_lock;     // 0 = unlocked, 1 = locked
    volatile int32_t fanout_head;     // Protected by fanout_lock
    volatile int32_t fanout_count;    // Protected by fanout_lock
    
    // Note: fanin_list and fanin_count do NOT need lock
    // because they are set once at task submission and never modified
    int32_t fanin_head;               // Set once, read-only after
    int32_t fanin_count;              // Set once, read-only after
    
    // ... other fields ...
} TaskDescriptor;

// Spinlock acquire
static inline void task_fanout_lock(TaskDescriptor* task) {
    while (__atomic_exchange_n(&task->fanout_lock, 1, __ATOMIC_ACQUIRE) != 0) {
        // Spin - optionally add pause instruction
        __builtin_ia32_pause();
    }
}

// Spinlock release
static inline void task_fanout_unlock(TaskDescriptor* task) {
    __atomic_store_n(&task->fanout_lock, 0, __ATOMIC_RELEASE);
}

// ORCHESTRATOR: Add consumer to producer's fanout_list
void add_consumer_to_producer_locked(DepListPool* pool,
                                      TaskDescriptor* producer,
                                      int32_t consumer_id) {
    task_fanout_lock(producer);
    
    producer->fanout_head = dep_list_prepend(pool, 
                                              producer->fanout_head,
                                              consumer_id);
    producer->fanout_count++;
    
    task_fanout_unlock(producer);
}

// SCHEDULER: Iterate producer's fanout_list when task completes
void iterate_fanout_list_locked(DepListPool* pool,
                                 TaskDescriptor* task,
                                 void (*callback)(int32_t, void*),
                                 void* ctx) {
    task_fanout_lock(task);
    
    int32_t current = task->fanout_head;
    int32_t count = task->fanout_count;
    
    task_fanout_unlock(task);
    
    // Now iterate without lock (entries are immutable once written)
    while (current != 0 && count-- > 0) {
        DepListEntry* entry = &pool->base[current];
        callback(entry->task_id, ctx);
        current = entry->next_offset;
    }
}

// SCHEDULER: Check if task can transition to CONSUMED (CRITICAL!)
// Must read fanout_count atomically with the check
bool check_and_mark_consumed(SchedulerState* sched, 
                              TaskDescriptor* task,
                              int32_t task_id) {
    // Must hold lock to read fanout_count consistently
    task_fanout_lock(task);
    
    int32_t fanout_count = task->fanout_count;
    int32_t fanout_refcount = sched->fanout_refcount[task_id];
    
    bool is_consumed = (fanout_refcount == fanout_count) && 
                       (sched->task_state[task_id] == TASK_COMPLETED);
    
    task_fanout_unlock(task);
    
    if (is_consumed) {
        sched->task_state[task_id] = TASK_CONSUMED;
        scheduler_on_task_consumed(sched, task_id);
    }
    
    return is_consumed;
}

// SCHEDULER: Update fanout_refcount and check CONSUMED
// Called when a consumer completes or scope ends
void increment_fanout_refcount_and_check(SchedulerState* sched,
                                          TaskDescriptor* producer,
                                          int32_t producer_id) {
    // Increment refcount (Scheduler-private, no lock needed)
    sched->fanout_refcount[producer_id]++;
    
    // Check CONSUMED with lock (reads fanout_count)
    if (sched->task_state[producer_id] == TASK_COMPLETED) {
        check_and_mark_consumed(sched, producer, producer_id);
    }
}
```

**Why fanin_list/fanin_count DON'T need lock:**

```
fanin_list and fanin_count are set ONCE when task is submitted:
  - Orchestrator: sets fanin_list, fanin_count during pto_submit_task()
  - Scheduler: reads fanin_list, fanin_count after task is visible
  - No concurrent modification → No lock needed

fanout_list and fanout_count CAN be modified after task submission:
  - Orchestrator: adds new consumer anytime before producer is CONSUMED
  - Scheduler: reads fanout_count to check CONSUMED condition
  - Concurrent modification possible → Lock REQUIRED
```

**Lock Overhead Analysis:**

```
Per-task lock overhead:
  - Lock acquisition: ~10-50 cycles (uncontended spinlock)
  - Lock release: ~5-10 cycles
  - Contention: Rare (different tasks have independent locks)
  
Expected contention:
  - TASK_WINDOW_SIZE = 1024 tasks
  - Probability of Orchestrator and Scheduler accessing SAME task: 1/1024
  - Contention is extremely rare!
  
Memory overhead:
  - 4 bytes per task for lock
  - 1024 tasks × 4 bytes = 4KB
```

#### Solution 2: Lock-Free Prepend with Atomic CAS (Advanced)

Since prepend only modifies the head pointer, we can use atomic compare-and-swap:

```c
// ========== LOCK-FREE PREPEND ==========
// Uses atomic CAS to safely prepend without locks

// ORCHESTRATOR: Lock-free prepend
void add_consumer_to_producer_lockfree(DepListPool* pool,
                                        TaskDescriptor* producer,
                                        int32_t consumer_id) {
    // Allocate new entry first (allocation is single-threaded by Orchestrator)
    int32_t new_offset = dep_list_pool_alloc_one(pool);
    DepListEntry* new_entry = &pool->base[new_offset];
    new_entry->task_id = consumer_id;
    
    // CAS loop to atomically update head
    int32_t old_head;
    do {
        old_head = __atomic_load_n(&producer->fanout_head, __ATOMIC_ACQUIRE);
        new_entry->next_offset = old_head;  // Point to current head
    } while (!__atomic_compare_exchange_n(
        &producer->fanout_head,
        &old_head,
        new_offset,
        false,  // strong CAS
        __ATOMIC_RELEASE,
        __ATOMIC_RELAXED
    ));
    
    // Increment count (atomic, but not strictly necessary for correctness)
    __atomic_fetch_add(&producer->fanout_count, 1, __ATOMIC_RELAXED);
}

// SCHEDULER: Safe iteration (no lock needed!)
void iterate_fanout_list_lockfree(DepListPool* pool,
                                   TaskDescriptor* task,
                                   void (*callback)(int32_t, void*),
                                   void* ctx) {
    // Atomically read head - we may miss newly prepended entries,
    // but that's OK because those consumers haven't been submitted yet
    int32_t current = __atomic_load_n(&task->fanout_head, __ATOMIC_ACQUIRE);
    
    while (current != 0) {
        DepListEntry* entry = &pool->base[current];
        callback(entry->task_id, ctx);
        current = entry->next_offset;  // next_offset is immutable once written
    }
}
```

**Why Lock-Free Works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCK-FREE PREPEND SAFETY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Key insight: Prepend only modifies HEAD, not existing entries             │
│                                                                              │
│   Before prepend:                                                            │
│     fanout_head ──► [B|→] ──► [C|→0]                                        │
│                                                                              │
│   During prepend of D:                                                       │
│     1. Allocate new entry [D|?]                                             │
│     2. Set new_entry.next = fanout_head (points to B)                       │
│     3. CAS: fanout_head = new_entry                                         │
│                                                                              │
│   After prepend:                                                             │
│     fanout_head ──► [D|→] ──► [B|→] ──► [C|→0]                              │
│                                                                              │
│   CONCURRENT READER (Scheduler):                                            │
│     - If reads head BEFORE CAS: sees [B, C] - correct                       │
│     - If reads head AFTER CAS: sees [D, B, C] - correct                     │
│     - If reads head DURING CAS: atomic, sees either old or new - correct    │
│                                                                              │
│   KEY: Once an entry is written, its fields (task_id, next) are IMMUTABLE  │
│   Reader never sees partial entry or corrupted pointer                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Solution 3: Sequencing Constraint (Design-Level)

If we can guarantee that **all consumers of a task are submitted before the task completes**, no lock is needed:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCING CONSTRAINT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CONSTRAINT: Producer's fanout_list is FROZEN when producer completes      │
│                                                                              │
│   This is naturally satisfied in many cases:                                │
│                                                                              │
│   1. Static graphs: All tasks submitted before any execution                │
│      → fanout_lists complete before Scheduler starts                        │
│                                                                              │
│   2. Scope-based submission: Tasks in same scope share lifetime             │
│      → Producer and all its consumers in same scope                         │
│      → scope_end() waits for all tasks to complete                          │
│                                                                              │
│   3. Streaming with barriers: Insert barrier between phases                 │
│      → Phase N fully submitted before Phase N-1 results consumed            │
│                                                                              │
│   When NOT satisfied (need locks):                                          │
│   - Dynamic graphs where new consumers added after producer completes       │
│   - Pipeline parallelism where submission overlaps execution                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Recommendation Summary

| Approach | Complexity | Overhead | Best For |
|----------|------------|----------|----------|
| **No lock (sequencing)** | Low | Zero | Static graphs, scope-based |
| **Per-task spinlock** | Medium | ~50 cycles/op | General, low contention |
| **Lock-free CAS** | High | ~20 cycles/op | High performance, parallel |

**Recommended: Per-task spinlock** for simplicity with negligible overhead:
- Contention probability: ~0.1% (1/TASK_WINDOW_SIZE)
- Lock overhead: ~50 cycles (negligible vs task execution)
- Code complexity: Simple, easy to verify

```c
// Final recommended implementation
typedef struct TaskDescriptor {
    // ... other fields ...
    volatile int32_t fanout_lock;     // Per-task spinlock
    volatile int32_t fanout_head;
    int32_t fanout_count;
    // ... other fields ...
} TaskDescriptor;
```

---

### Runtime API Implementationn

#### 1. pto_scope_begin() - Begin a New Scope

Called by Orchestrator to mark the beginning of a scope. Pushes current task position to scope stack.

```c
void pto_scope_begin(PTORuntime* rt) {
    // Push current task index to scope stack (using ring buffer index)
    int32_t current_pos = rt->task_ring.current_index;
    
    if (rt->scope_stack_top >= rt->scope_stack_capacity - 1) {
        return;  // Stack overflow
    }
    
    rt->scope_stack[++rt->scope_stack_top] = current_pos;
}

// Get current scope depth (useful for fanout initialization)
int32_t pto_get_scope_depth(PTORuntime* rt) {
    return rt->scope_stack_top + 1;  // 0 = global scope
}
```

#### 2. pto_submit_task() - Submit Task with Ring Buffer Allocation

Called by Orchestrator to submit a task. Uses ring buffer allocation for both task slot and output buffer. **May stall** if rings are full (back-pressure from Scheduler).

##### CRITICAL: Output Buffer Memory Allocation Mechanism

**All intermediate data memory allocation is handled by `pto_submit_task()` for OUTPUT parameters.**

The key insight is that the runtime (not the orchestration function) is responsible for allocating output buffers from the HeapRing. This design requires:

1. **OUTPUT parameter must be a pointer-to-pointer (`void**`)** - The orchestration function passes a reference to a buffer pointer, allowing the runtime to write back the allocated address.

2. **Runtime allocates from HeapRing** - During `pto_submit_task()`, the runtime allocates a packed buffer for all outputs and writes the allocated addresses back to the caller's references.

3. **Orchestration function receives allocated address** - After `pto_submit_task()` returns, the orchestration function's output buffer variable contains the runtime-allocated address.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OUTPUT BUFFER ALLOCATION FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Orchestration Function                    Runtime (pto_submit_task)        │
│   ──────────────────────                    ─────────────────────────        │
│                                                                              │
│   void* P = NULL;     ─────────────────────►  1. Receives &P (pointer-to-   │
│                                                   pointer to output buffer)  │
│   pto_submit_task(rt, "gemm", {                                              │
│       PTO_INPUT(A, ...),                      2. Calculate total output      │
│       PTO_INPUT(B, ...),                         size from all OUTPUT params │
│       PTO_OUTPUT(&P, size)  ◄──────────────                                  │
│   });                                         3. Allocate packed buffer      │
│                                                  from HeapRing (may stall)   │
│                                                                              │
│   // After submit_task returns:               4. Write allocated address     │
│   // P now points to allocated buffer            back to *(&P) = P           │
│                                                  ───────────────────────►    │
│   pto_submit_task(rt, "add", {                                               │
│       PTO_INPUT(P, ...),  ◄── Uses the       5. Register allocated address   │
│       ...                     runtime-           in TensorMap for dependency │
│   });                         allocated          tracking                    │
│                               address                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this design?**

1. **Centralized memory management** - Runtime manages the HeapRing, tracking allocations and deallocations for proper lifecycle management.

2. **Automatic dependency tracking** - The TensorMap uses the runtime-allocated address to track producer-consumer relationships across tasks.

3. **Flow control** - If HeapRing is full, `pto_submit_task()` can stall (back-pressure), preventing unbounded memory growth.

4. **Packed buffer optimization** - Multiple outputs can be packed into a single contiguous allocation, improving cache locality.

##### Two Memory Allocation Modes

There are two valid approaches for managing intermediate buffers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MODE A vs MODE B: Buffer Management                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MODE A: Pre-allocated Large Buffer (Current bgemm implementation)          │
│   ═══════════════════════════════════════════════════════════════           │
│                                                                              │
│   // In main():                                                              │
│   float* P = calloc(max_tiles * tile_size, sizeof(float));  // Pre-allocate │
│                                                                              │
│   // In orchestration:                                                       │
│   pto_task_add_output(rt, t, P, tile_index, 0, 32, 128);    // Use offset    │
│                                                                              │
│   Pros: Simple, no dynamic allocation overhead                               │
│   Cons: Must know max memory at compile time, wastes memory if over-sized   │
│   Use when: Static task graph, known memory bounds                           │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│   MODE B: Runtime Dynamic Allocation (Document-described design)             │
│   ══════════════════════════════════════════════════════════════            │
│                                                                              │
│   // In orchestration:                                                       │
│   void* P = NULL;                                                            │
│   pto_submit_task(rt, "gemm", {                                              │
│       PTO_OUTPUT(&P, tile_idx, size)    // Runtime allocates, writes back   │
│   });                                                                        │
│   // P now contains runtime-allocated address                                │
│                                                                              │
│   Pros: Flexible, memory-efficient, supports dynamic task graphs             │
│   Cons: Requires pointer-to-pointer parameter, more complex                  │
│   Use when: Dynamic task graph, memory-constrained, unknown bounds           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Impact on Code Generation (`pto_compile.py`):**

To support Mode B (dynamic allocation), the code generator needs to:

1. **Generate `void* P = NULL;` declarations** for intermediate buffers
2. **Generate `PTO_OUTPUT(&P, ...)` calls** with pointer-to-pointer
3. **Use `P` (runtime-allocated address) in subsequent INPUT calls**

Example generated code for Mode B:

```c
// Mode B: Dynamic allocation (generated by updated pto_compile.py)
void bgemm_dynamic(PTORuntime* rt, void* user_data) {
    void** params = (void**)user_data;
    float* A = (float*)params[0];
    float* B = (float*)params[1];
    float* C = (float*)params[2];
    // NOTE: P is NOT passed from outside - will be dynamically allocated
    
    for (...) {
        void* P = NULL;  // Local variable, will receive runtime allocation
        
        // gemm_tile: runtime allocates P and writes back to &P
        pto_submit_task(rt, "gemm_tile", {
            PTO_INPUT(A, a_idx, size),
            PTO_INPUT(B, b_idx, size),
            PTO_OUTPUT(&P, c_idx, size)   // &P passed, runtime writes allocation
        });
        // Now P contains runtime-allocated address
        
        // tile_add: uses P (the allocated address)
        pto_submit_task(rt, "tile_add", {
            PTO_INPUT(C, c_idx, size),
            PTO_INPUT(P, c_idx, size),    // P is now the allocated buffer
            PTO_INOUT(&C, c_idx, size)
        });
    }
}
```

**Current Implementation Note:**

The current `examples/bgemm` uses **Mode A** (pre-allocated buffer). To use **Mode B**, 
updates are needed in:
- `pto_bgemm_func.py` - Change memref("P", ...) to local variable pattern
- `pto_codegen_*.py` - Generate void* declarations and PTO_OUTPUT with &
- Runtime API - Support buffer_ref (void**) writeback

```c
// InCore function call parameter descriptor
typedef struct TaskParam {
    enum { PARAM_INPUT, PARAM_OUTPUT, PARAM_INOUT } type;
    
    // For INPUT:  buffer is the data source (read-only)
    // For OUTPUT: buffer is a POINTER TO POINTER (void**), runtime writes allocated address
    // For INOUT:  buffer is the data source AND runtime writes allocated address back
    union {
        void*  buffer;        // For INPUT: direct buffer pointer
        void** buffer_ref;    // For OUTPUT/INOUT: reference to receive allocated address
    };
    
    int32_t tile_index;       // Tile index for TensorMap lookup
    int32_t size;             // Size in bytes
} TaskParam;

// Convenience macros for parameter construction
#define PTO_INPUT(buf, idx, sz)    { .type = PARAM_INPUT,  .buffer = (buf), .tile_index = (idx), .size = (sz) }
#define PTO_OUTPUT(ref, idx, sz)   { .type = PARAM_OUTPUT, .buffer_ref = (void**)(ref), .tile_index = (idx), .size = (sz) }
#define PTO_INOUT(ref, idx, sz)    { .type = PARAM_INOUT,  .buffer_ref = (void**)(ref), .tile_index = (idx), .size = (sz) }

// Submit a task with InCore function and parameters
// May STALL if task ring or heap ring is full (waiting for Scheduler to free space)
int32_t pto_submit_task(PTORuntime* rt, 
                        int32_t kernel_id,
                        int32_t worker_type,
                        TaskParam* params,
                        int32_t num_params) {
    
    SharedMemoryHeader* sm = rt->sm_header;
    
    // === STEP 0: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory
    int32_t new_last_task_alive = sm->last_task_alive;
    
    // Update TensorMap validity threshold (stale entries will be ignored in lookup)
    tensormap_sync_validity(&rt->tensor_map, sm);
    
    // Periodically cleanup TensorMap to remove stale entries from bucket chains
    // This prevents bucket chains from growing indefinitely with stale entries
    #define TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every 64 retired tasks
    if (new_last_task_alive - rt->tensormap_last_cleanup >= TENSORMAP_CLEANUP_INTERVAL) {
        tensormap_cleanup_retired(&rt->tensor_map, 
                                   rt->tensormap_last_cleanup, 
                                   new_last_task_alive);
        rt->tensormap_last_cleanup = new_last_task_alive;
    }
    
    // === STEP 1: Allocate task slot from Task Ring (may stall) ===
    // This provides back-pressure if task window is full
    int32_t task_id = task_ring_alloc(&rt->task_ring, &sm->last_task_alive);
    
    TaskDescriptor* task = &rt->task_descriptors[task_id];
    SchedulerState* sched = &rt->sched_state;
    
    // Initialize task descriptor (Orchestrator data)
    task->task_id = task_id;
    task->kernel_id = kernel_id;
    task->worker_type = worker_type;
    task->scope_depth = rt->scope_stack_top + 1;
    task->fanin_list = NULL;
    task->fanin_list_size = 0;
    task->fanout_list = NULL;
    task->fanout_list_size = 0;
    task->fanin_count = 0;
    task->fanout_count = rt->scope_stack_top + 1;   // Initial: scope_depth
    task->packed_buffer_base = NULL;
    task->packed_buffer_end = NULL;
    
    // Initialize scheduler state
    sched->task_state[task_id] = TASK_PENDING;
    
    // Temporary lists for building fanin/fanout
    int32_t fanin_temp[32];
    int32_t local_fanin_count = 0;
    
    // Temporary storage for output sizes (to compute packed buffer size)
    int32_t output_sizes[MAX_OUTPUTS];
    int32_t num_outputs = 0;
    int32_t total_output_size = 0;
    
    // === STEP 2: First pass - collect output sizes and process inputs ===
    for (int i = 0; i < num_params; i++) {
        TaskParam* p = &params[i];
        TensorRegion region = {p->buffer, p->tile_index, 0, p->size};
        
        switch (p->type) {
            case PARAM_INPUT: {
                // Look up producer via TensorMap
                int32_t producer_id = tensormap_lookup(&rt->tensor_map, &region);
                
                if (producer_id >= 0) {
                    // Add to fanin list (this task depends on producer)
                    fanin_temp[local_fanin_count++] = producer_id;
                    
                    // Increment producer's fanout_count (Orchestrator data)
                    rt->task_descriptors[producer_id].fanout_count++;
                    
                    // Add this task to producer's fanout list
                    task_descriptor_add_fanout(&rt->task_descriptors[producer_id], task_id);
                }
                break;
            }
            
            case PARAM_OUTPUT: {
                // Collect output size for packed buffer allocation
                output_sizes[num_outputs++] = p->size;
                total_output_size += ALIGN_UP(p->size, 64);
                break;
            }
            
            case PARAM_INOUT: {
                // INOUT = INPUT + OUTPUT
                
                // Handle as input (get dependency on previous writer)
                int32_t producer_id = tensormap_lookup(&rt->tensor_map, &region);
                if (producer_id >= 0) {
                    fanin_temp[local_fanin_count++] = producer_id;
                    rt->task_descriptors[producer_id].fanout_count++;
                    task_descriptor_add_fanout(&rt->task_descriptors[producer_id], task_id);
                }
                
                // Collect output size for packed buffer
                output_sizes[num_outputs++] = p->size;
                total_output_size += ALIGN_UP(p->size, 64);
                break;
            }
        }
    }
    
    // === STEP 3: Allocate packed buffer from Heap Ring (may stall) ===
    // This provides back-pressure if heap is full
    if (total_output_size > 0) {
        task->packed_buffer_base = heap_ring_alloc(&rt->heap_ring, total_output_size, &sm->heap_tail);
        task->packed_buffer_end = (char*)task->packed_buffer_base + total_output_size;
        
        // Calculate offsets for each output within packed buffer
        int32_t offset = 0;
        for (int i = 0; i < num_outputs; i++) {
            task->output_offsets[i] = offset;
            offset += ALIGN_UP(output_sizes[i], 64);
        }
        
        // Update shared memory with new heap top (for Scheduler visibility)
        sm->heap_top = rt->heap_ring.top;
    }
    
    // === STEP 4: Second pass - register outputs in TensorMap AND write back addresses ===
    int32_t output_idx = 0;
    for (int i = 0; i < num_params; i++) {
        TaskParam* p = &params[i];
        
        if (p->type == PARAM_OUTPUT || p->type == PARAM_INOUT) {
            // Compute actual buffer address for this output
            void* output_addr = (char*)task->packed_buffer_base + task->output_offsets[output_idx];
            
            // CRITICAL: Write allocated address back to caller's buffer reference
            // This allows orchestration function to use the runtime-allocated address
            // in subsequent task submissions (e.g., as input to dependent tasks)
            if (p->buffer_ref != NULL) {
                *(p->buffer_ref) = output_addr;  // Write back to caller: *(&P) = output_addr
            }
            
            // Register in TensorMap using the ORIGINAL buffer reference for dependency tracking
            // This allows consumers to find the producer via their local variable address
            TensorRegion region = {
                .base_ptr = p->buffer_ref,    // Use original reference for TensorMap lookup
                .tile_index = p->tile_index,
                .offset = 0,
                .size = p->size
            };
            
            // Register in TensorMap: this region is produced by task_id
            tensormap_insert(&rt->tensor_map, &region, task_id);
            output_idx++;
        }
    }
    
    // === STEP 5: Finalize Orchestrator data ===
    task->fanin_count = local_fanin_count;
    if (local_fanin_count > 0) {
        task->fanin_list = pool_alloc_array(local_fanin_count);
        memcpy(task->fanin_list, fanin_temp, local_fanin_count * sizeof(int32_t));
        task->fanin_list_size = local_fanin_count;
    }
    task->num_outputs = num_outputs;
    
    // === STEP 6: Initialize Scheduler refcounts (start from 0, increment up) ===
    sched->fanin_refcount[task_id] = 0;   // Will increment as producers complete
    sched->fanout_refcount[task_id] = 0;  // Will increment as consumers/scopes release
    
    // === STEP 7: Check if task is immediately ready ===
    // Task is ready when fanin_refcount == fanin_count (all producers done)
    if (sched->fanin_refcount[task_id] == task->fanin_count) {
        sched->task_state[task_id] = TASK_READY;
        // Push to the ready queue for this task's worker type
        ready_queue_push(&sched->ready_queues[worker_type], task_id);
    }
    
    // === STEP 8: Update shared memory with current task index ===
    // Scheduler reads this to know how many tasks have been submitted
    sm->current_task_index = rt->task_ring.current_index;
    
    return task_id;
}

// Helper: Get pointer to specific output of a task
void* pto_task_get_output(PTORuntime* rt, int32_t task_id, int32_t output_idx) {
    TaskDescriptor* task = &rt->task_descriptors[task_id];
    return (char*)task->packed_buffer_base + task->output_offsets[output_idx];
}

// Helper: Add consumer to producer's fanout list
static void task_descriptor_add_fanout(TaskDescriptor* producer, int32_t consumer_id) {
    // Grow fanout list (simplified - in practice use pool allocator)
    int32_t new_size = producer->fanout_list_size + 1;
    int32_t* new_list = pool_realloc_array(producer->fanout_list, new_size);
    new_list[producer->fanout_list_size] = consumer_id;
    producer->fanout_list = new_list;
    producer->fanout_list_size = new_size;
}
```

#### 3. pto_scope_end() - End Scope and Release Reference

Called by Orchestrator when a scope ends. The implementation is **very simple**:
- Pop the scope stack to get the range `[begin_pos, end_pos)`
- Iterate over all tasks in the range and increment `fanout_refcount` by 1
- Compare with `fanout_count` to check if buffer can be released

```c
void pto_scope_end(PTORuntime* rt) {
    if (rt->scope_stack_top < 0) {
        return;  // No scope to end
    }
    
    // Pop scope stack to get begin position
    int32_t scope_begin_pos = rt->scope_stack[rt->scope_stack_top--];
    int32_t scope_end_pos = rt->task_count;  // Current position is end
    
    SchedulerState* sched = &rt->sched_state;
    
    // Simple: just increment fanout_refcount for ALL tasks in [begin, end)
    // No need to filter - every task in range has a reference from this scope
    for (int32_t task_id = scope_begin_pos; task_id < scope_end_pos; task_id++) {
        sched->fanout_refcount[task_id]++;
        
        // Transition to CONSUMED when fanout_refcount == fanout_count
        TaskDescriptor* task = &rt->task_descriptors[task_id];
        if (sched->fanout_refcount[task_id] == task->fanout_count && 
            sched->task_state[task_id] == TASK_COMPLETED) {
            sched->task_state[task_id] = TASK_CONSUMED;
            scheduler_on_task_consumed(rt, task_id);
        }
    }
}
```

**Scope Reference Lifecycle:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCOPE REFERENCE MANAGEMENT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ORCHESTRATOR DATA (static):                                               │
│     fanout_count = scope_depth + num_consumers (set at submission)          │
│                                                                              │
│   SCHEDULER STATE (dynamic):                                                │
│     fanout_refcount starts at 0, incremented as refs are released           │
│     Buffer freed when: fanout_refcount == fanout_count                      │
│                                                                              │
│   pto_scope_begin()                   pto_scope_end()                       │
│         │                                   │                                │
│         │  Push scope_begin_pos             │  Pop and get range             │
│         │  to scope_stack                   │  [begin_pos, end_pos)          │
│         │                                   │                                │
│         ▼                                   ▼                                │
│   Example: Tasks created at scope depth = 2                                 │
│                                                                              │
│   ┌───────────────────┐             ┌───────────────────┐                   │
│   │ Task A            │             │ Task A            │                   │
│   │ fanout_count = 2  │   scope_end │ fanout_refcount   │                   │
│   │ fanout_refcount=0 │  ────────►  │   = 0 + 1 = 1     │                   │
│   ├───────────────────┤             ├───────────────────┤                   │
│   │ Task B            │             │ Task B            │                   │
│   │ fanout_count = 2  │             │ fanout_refcount   │                   │
│   │ fanout_refcount=0 │             │   = 0 + 1 = 1     │                   │
│   ├───────────────────┤             ├───────────────────┤                   │
│   │ Task C            │             │ Task C            │                   │
│   │ fanout_count = 3  │             │ fanout_refcount   │                   │
│   │ fanout_refcount=0 │             │   = 0 + 1 = 1     │                   │
│   └───────────────────┘             └───────────────────┘                   │
│                                                                              │
│   Note: fanout_count is IMMUTABLE (Orchestrator data)                       │
│         fanout_refcount starts at 0, INCREMENTS (Scheduler state)           │
│         Buffer freed when: fanout_refcount == fanout_count                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4. pto_task_complete() - Signal Task Completion

Called by Worker when task execution finishes. Updates Scheduler state (refcounts).

```c
void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    TaskDescriptor* task = &rt->task_descriptors[task_id];  // Read-only (Orchestrator)
    SchedulerState* sched = &rt->sched_state;               // Read/Write (Scheduler)
    
    // Mark task as completed (execution done, but output may still be in use)
    sched->task_state[task_id] = TASK_COMPLETED;
    
    // === STEP 1: Update fanin_refcount of all consumers (make them ready) ===
    // Read: Orchestrator data (fanout_list)
    // Write: Scheduler state (fanin_refcount)
    for (int i = 0; i < task->fanout_list_size; i++) {
        int32_t consumer_id = task->fanout_list[i];
        TaskDescriptor* consumer = &rt->task_descriptors[consumer_id];
        
        sched->fanin_refcount[consumer_id]++;
        
        // Task is READY when fanin_refcount == fanin_count (all producers done)
        if (sched->fanin_refcount[consumer_id] == consumer->fanin_count && 
            sched->task_state[consumer_id] == TASK_PENDING) {
            sched->task_state[consumer_id] = TASK_READY;
            // Push to the ready queue for the consumer's worker type
            ready_queue_push(&sched->ready_queues[consumer->worker_type], consumer_id);
        }
    }
    
    // === STEP 2: Update fanout_refcount of all producers (for buffer lifecycle) ===
    // Read: Orchestrator data (fanin_list)
    // Write: Scheduler state (fanout_refcount)
    for (int i = 0; i < task->fanin_list_size; i++) {
        int32_t producer_id = task->fanin_list[i];
        TaskDescriptor* producer = &rt->task_descriptors[producer_id];
        
        sched->fanout_refcount[producer_id]++;
        
        // Transition to CONSUMED when fanout_refcount == fanout_count
        if (sched->fanout_refcount[producer_id] == producer->fanout_count &&
            sched->task_state[producer_id] == TASK_COMPLETED) {
            sched->task_state[producer_id] = TASK_CONSUMED;
            scheduler_on_task_consumed(rt, producer_id);
        }
    }
    
    // === STEP 3: Check if this task can transition to CONSUMED ===
    if (sched->fanout_refcount[task_id] == task->fanout_count) {
        sched->task_state[task_id] = TASK_CONSUMED;
        scheduler_on_task_consumed(rt, task_id);
    }
}
```

#### Internal: scheduler_on_task_consumed()

Called when task transitions to `TASK_CONSUMED` state. Updates `last_task_alive` for heap reclamation.

```c
// Called when a task transitions to CONSUMED state
static void scheduler_on_task_consumed(PTORuntime* rt, int32_t task_id) {
    SchedulerState* sched = &rt->sched_state;
    
    // Task state already set to CONSUMED by caller
    // No individual buffer free needed!
    
    // Check if this enables heap compaction
    // If the consumed task is the current last_task_alive, advance the pointer
    if (task_id == sched->last_task_alive) {
        advance_last_task_alive(rt);
    }
}

// Advance last_task_alive until we find a non-CONSUMED task
static void advance_last_task_alive(PTORuntime* rt) {
    SchedulerState* sched = &rt->sched_state;
    int32_t task_count = rt->task_count;
    
    while (sched->last_task_alive < task_count &&
           sched->task_state[sched->last_task_alive] == TASK_CONSUMED) {
        sched->last_task_alive++;
    }
    
    // Update heap free_offset based on last_task_alive
    // All tasks before last_task_alive are CONSUMED → their buffers are reclaimable
    if (sched->last_task_alive > 0) {
        TaskDescriptor* last_consumed = &rt->task_descriptors[sched->last_task_alive - 1];
        if (last_consumed->packed_buffer_end != NULL) {
            // free_offset = end of last consumed task's buffer
            rt->gm_heap.free_offset = 
                (char*)last_consumed->packed_buffer_end - (char*)rt->gm_heap.base;
        }
    }
}
```

**Key insight**: No individual `free()` calls are needed. The heap is compacted by simply advancing `last_task_alive` and updating `free_offset`. This eliminates per-task deallocation overhead.

---

### Internal: Scheduler Dispatch Loop

The Scheduler runs continuously, dispatching ready tasks to workers. Since workers of the same type are interchangeable, each worker type has its own ready queue.

```c
void scheduler_dispatch_loop(PTORuntime* rt) {
    SchedulerState* sched = &rt->sched_state;
    
    while (!runtime_is_done(rt)) {
        bool dispatched_any = false;
        
        // Poll each worker type's ready queue
        for (int wtype = 0; wtype < NUM_WORKER_TYPES; wtype++) {
            ReadyQueue* ready_q = &sched->ready_queues[wtype];
            
            // Try to dispatch tasks to available workers of this type
            while (!ready_queue_empty(ready_q) && worker_available(rt, wtype)) {
                int32_t task_id = ready_queue_pop(ready_q);
                
                // Update scheduler state
                sched->task_state[task_id] = TASK_RUNNING;
                
                // Dispatch to any available worker of this type
                worker_dispatch(rt, wtype, task_id);
                dispatched_any = true;
            }
        }
        
        if (!dispatched_any) {
            // No tasks dispatched - wait for worker completion signals
            runtime_wait_for_completion(rt);
        }
    }
}

// Helper: Check if any worker of given type is available
bool worker_available(PTORuntime* rt, int worker_type) {
    return rt->worker_pools[worker_type].available_count > 0;
}

// Helper: Dispatch task to an available worker of given type
void worker_dispatch(PTORuntime* rt, int worker_type, int32_t task_id) {
    WorkerPool* pool = &rt->worker_pools[worker_type];
    int worker_id = pool->available_workers[--pool->available_count];
    
    // Send task to worker (e.g., via task queue or message)
    worker_send_task(rt, worker_type, worker_id, task_id);
}
```

---

### Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCHEDULER EXECUTION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ORCHESTRATOR                    SCHEDULER                    WORKERS      │
│   ────────────                    ─────────                    ───────      │
│                                                                              │
│   task_id = submit_task() ─────►  Allocate from task pool                   │
│                                   task.fanout_count = scope_depth           │
│                                                                              │
│   add_output(task, P) ──────────► TensorMap[P] = task_id                    │
│                                   Allocate output buffer                    │
│                                                                              │
│   add_input(task, A) ───────────► producer = TensorMap[A]                   │
│                                   task.fanin_count++                        │
│                                   task.fanin_list.add(producer)             │
│                                   producer.fanout_count++                   │
│                                   producer.fanout_list.add(task)            │
│                                                                              │
│   submit_task(task) ────────────► if fanin_count == 0:                      │
│                                     ready_queue.push(task)                  │
│                                                                              │
│                                   ┌─────────────────────┐                   │
│                                   │ Dispatch Loop:      │                   │
│                                   │ task = ready_queue  │ ─────────────────►│
│                                   │        .pop()       │    Execute        │
│                                   │ dispatch(task)      │    kernel         │
│                                   └─────────────────────┘                   │
│                                                                              │
│                                   ◄──────────────────────── signal_complete │
│                                   on_task_complete(task):                   │
│                                     for each consumer:                      │
│                                       consumer.fanin_count--                │
│                                       if fanin_count == 0:                  │
│                                         ready_queue.push(consumer)          │
│                                     for each producer:                      │
│                                       producer.fanout_count--               │
│                                       if fanout_count == 0:                 │
│                                         release(producer)                   │
│                                                                              │
│   } // scope ends ──────────────► for each task in scope:                   │
│                                     task.fanout_count--                     │
│                                     if fanout_count == 0:                   │
│                                       release(task)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Example: BGEMM Task Graph

```
Orchestration code (assuming scope_depth = 1):
  scope_begin()                      // depth = 1
  for k in range(K):
      P = alloc()                    // P.fanout = 1 (depth=1)
      submit(gemm, in=[A[k],B[k]], out=P)
      submit(add, in=[C, P], out=C)
  scope_end()

Task dependencies (for K=2):

  ┌───────────┐      ┌───────────┐
  │ gemm_0    │      │ gemm_1    │
  │ out: P0   │      │ out: P1   │
  │ fanout: 2 │      │ fanout: 2 │  (depth=1 + 1 consumer)
  └─────┬─────┘      └─────┬─────┘
        │                  │
        ▼                  ▼
  ┌───────────┐      ┌───────────┐
  │ add_0     │      │ add_1     │
  │ in: C,P0  │      │ in: C,P1  │
  │ fanin: 1  │─────►│ fanin: 1  │  (depends on previous add for C)
  │ fanout: 2 │      │ fanout: 1 │
  └───────────┘      └───────────┘

Execution order:
  1. gemm_0 ready (fanin=0) → dispatch → complete
     - add_0.fanin: 1→0 → add_0 ready
  2. gemm_1 ready (fanin=0) → dispatch → complete
     - add_1 not ready yet (depends on add_0 for C)
  3. add_0 complete
     - gemm_0.fanout: 2→1 (consumer done)
     - add_1.fanin: 1→0 → add_1 ready
  4. scope_end()
     - gemm_0.fanout: 1→0 → RELEASE gemm_0 + P0
     - gemm_1.fanout: 2→1
  5. add_1 complete
     - gemm_1.fanout: 1→0 → RELEASE gemm_1 + P1
```

---

### Overhead Analysis

```
Per-task memory overhead:
  - fanin_list_inline[4]:     16 bytes
  - fanin_list_size:           4 bytes
  - fanin_list_overflow:       8 bytes (pointer)
  - fanout_list_inline[4]:    16 bytes
  - fanout_list_size:          4 bytes
  - fanout_list_overflow:      8 bytes (pointer)
  - fanout_count:              4 bytes
  - fanin_count:               4 bytes
  Total: ~64 bytes/task for dependency tracking

Per-task CPU overhead:
  - Task submission: O(num_inputs) for dependency setup
  - Task completion: O(num_inputs + num_outputs) for fanin/fanout updates

For BGEMM with 2 inputs, 1 output per task:
  - Submission: ~20 cycles (2 fanin updates + 2 fanout updates)
  - Completion: ~30 cycles (2 fanout decrements + 1 fanin decrement per consumer)

At 10M tasks/sec:
  - Total overhead: ~500M cycles/sec ≈ 250ms/sec on 2GHz AICPU
  - Acceptable for most workloads
```

---

### Characteristics

| Aspect | Description |
|--------|-------------|
| **Memory efficiency** | Optimal - buffers released at earliest safe point |
| **Correctness** | Provably correct - ref counting ensures no use-after-free |
| **Bounded resources** | Task pool + buffer pool with fixed capacity |
| **Nested orchestration** | Fully supported via scope-based fanout management |
| **Dynamic graphs** | No static analysis needed - dependencies resolved at runtime |

---

## Summary

### Ring Buffer Design Benefits

The ring buffer design eliminates the need for traditional memory allocators:

| Aspect | Ring Buffer Approach |
|--------|---------------------|
| **Allocation** | O(1) bump pointer, no search |
| **Deallocation** | Implicit - `last_task_alive` advances, no explicit free |
| **Fragmentation** | Zero - contiguous allocation, FIFO reclamation |
| **Overhead** | Minimal - just pointer arithmetic |
| **Flow Control** | Natural back-pressure when rings full |

### Memory Layout Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE MEMORY LAYOUT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SHARED MEMORY (Orchestrator ↔ Scheduler communication)                    │
│   ─────────────────────────────────────────────────────                     │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ Header: current_task_index, heap_top, last_task_alive, heap_tail│      │
│   ├─────────────────────────────────────────────────────────────────┤      │
│   │ TaskDescriptor[TASK_WINDOW_SIZE] (ring buffer)                  │      │
│   ├─────────────────────────────────────────────────────────────────┤      │
│   │ DepListPool (ring buffer for fanin/fanout linked lists)         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   GLOBAL MEMORY HEAP (task output buffers)                                  │
│   ─────────────────────────────────────────                                 │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ HeapRing: [RETIRED | ACTIVE BUFFERS | FREE SPACE] (ring buffer) │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ORCHESTRATOR PRIVATE                  SCHEDULER PRIVATE                   │
│   ────────────────────                  ─────────────────                   │
│   • TensorMap (ring buffer pool)        • task_state[]                      │
│   • scope_stack                         • fanin_refcount[]                  │
│   • Local ring pointers                 • fanout_refcount[]                 │
│   • tensormap_last_cleanup              • ready_queues[]                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **All dynamic data uses ring buffers** - No malloc/free needed
2. **Implicit memory reclamation** - `last_task_alive` advancement frees memory
3. **TensorMap with lazy invalidation** - Stale entries auto-ignored when producer retires
4. **Per-task spinlock for fanout** - Required for correctness (count + list consistency)
5. **Shared memory for flow control** - Orchestrator and Scheduler communicate via pointers
6. **Private data separation** - TensorMap, refcounts, ready_queues not in shared memory

### Ring Buffer Components

| Component | Allocation | Reclamation | Size Bound |
|-----------|-----------|-------------|------------|
| **Task Ring** | Bump `current_task_index` | `last_task_alive` advances | TASK_WINDOW_SIZE |
| **Heap Ring** | Bump `heap_top` | `heap_tail` follows `last_task_alive` | HEAP_SIZE |
| **DepListPool** | Prepend to list | Ring wrap overwrites | DEP_LIST_POOL_SIZE |
| **TensorMap** | Bump `pool_head` | Lazy invalidation + periodic cleanup | TENSORMAP_POOL_SIZE |

### Sizing Guidelines

```
TASK_WINDOW_SIZE = 1024                          // Power of 2 for efficient modulo
HEAP_SIZE = depends on workload                  // e.g., 64MB for typical BGEMM
DEP_LIST_POOL_SIZE = TASK_WINDOW_SIZE × AVG_FANOUT × 2 = ~8K entries
TENSORMAP_POOL_SIZE = TASK_WINDOW_SIZE × AVG_OUTPUTS × 2 = ~4K entries
TENSORMAP_NUM_BUCKETS = TASK_WINDOW_SIZE         // 1024 buckets for good hash distribution
```

### TensorMap Lazy Invalidation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TENSORMAP LIFECYCLE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. Task T submits with output buffer B:                                    │
│      - tensormap_insert(B, T)                                               │
│      - Entry added to TensorMap pool                                         │
│                                                                              │
│   2. Consumer task C needs B:                                                │
│      - tensormap_lookup(B) → returns T                                       │
│      - C adds T to its fanin_list                                            │
│                                                                              │
│   3. Task T retires (CONSUMED, last_task_alive advances past T):             │
│      - NO explicit removal needed!                                           │
│      - TensorMap entry for B becomes STALE (T < last_task_alive)            │
│      - Future lookups for B will find entry but ignore it (producer invalid)│
│                                                                              │
│   4. TensorMap pool wraps around:                                            │
│      - Stale entry slot is reused for new entry                              │
│      - Automatic memory reclamation, no explicit free                        │
│                                                                              │
│   5. Periodic cleanup (optional optimization):                               │
│      - Remove stale entries from hash bucket chains                          │
│      - Keeps lookup O(1) amortized                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 总结：PTO Runtime 高性能设计

### 一、架构设计要素

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PTO RUNTIME ARCHITECTURE OVERVIEW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐                    ┌─────────────────┐                │
│   │  ORCHESTRATOR   │  ═══ Shared ═══►   │   SCHEDULER     │                │
│   │  (图构建者)      │      Memory        │   (任务调度器)   │                │
│   └────────┬────────┘                    └────────┬────────┘                │
│            │                                      │                          │
│            │ submit tasks                         │ dispatch tasks           │
│            ▼                                      ▼                          │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                        WORKERS                                   │      │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │      │
│   │   │ AICore   │  │ AICore   │  │  AI_CPU  │  │Accelerator│       │      │
│   │   │  CUBE    │  │ VECTOR   │  │          │  │          │       │      │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 核心设计原则

| 原则 | 描述 | 收益 |
|------|------|------|
| **解耦设计** | Orchestrator 与 Scheduler 通过共享内存通信 | 可独立部署在 Host CPU 或 Device AI_CPU |
| **异步流水** | 任务提交与执行并行 | 最大化硬件利用率 |
| **动态依赖** | 运行时构建任务图，无需静态分析 | 支持图灵完备的控制流 |
| **精确生命周期** | Fanin/Fanout 引用计数 | Buffer 在最早安全点释放 |

---

### 二、关键数据结构优化

#### 1. 全环形缓冲区设计 (Zero Allocation Overhead)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RING BUFFER UNIFIED DESIGN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   传统设计问题:                         Ring Buffer 解决方案:                │
│   ─────────────                         ─────────────────────                │
│   • malloc/free 每次 100-500 cycles    • Bump allocation: ~5 cycles         │
│   • 内存碎片化                          • 零碎片 (连续分配)                   │
│   • 无界增长风险                        • 固定容量，自动回压                  │
│   • 显式释放开销                        • 隐式回收 (指针推进即释放)           │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                           HEAP RING                                  │   │
│   │   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │   │
│   │   │ T0  │ T1  │ T2  │ T3  │ T4  │     │     │     │ Tn-1│ Tn  │    │   │
│   │   │ buf │ buf │ buf │ buf │ buf │FREE │FREE │FREE │ buf │ buf │    │   │
│   │   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘    │   │
│   │     ▲                       ▲                             ▲         │   │
│   │     │                       │                             │         │   │
│   │   heap_tail              heap_top                  (wrap around)    │   │
│   │   (Scheduler             (Orchestrator                              │   │
│   │    writes)                writes)                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   性能对比:                                                                  │
│   ┌────────────────┬──────────────────┬──────────────────┐                  │
│   │ 操作           │ malloc/free      │ Ring Buffer      │                  │
│   ├────────────────┼──────────────────┼──────────────────┤                  │
│   │ 分配           │ 100-500 cycles   │ ~5 cycles        │                  │
│   │ 释放           │ 100-500 cycles   │ 0 cycles (隐式)  │                  │
│   │ 碎片           │ 高               │ 无               │                  │
│   │ 10M tasks/sec  │ 2-10B cycles/sec │ 50M cycles/sec   │                  │
│   └────────────────┴──────────────────┴──────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2. TensorMap 惰性失效 + 链表截断 (Lazy Invalidation + Chain Truncation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            TENSORMAP LAZY INVALIDATION + CHAIN TRUNCATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   双重优化设计:                                                              │
│                                                                              │
│   优化1: 惰性失效 (Lazy Invalidation)                                        │
│   ─────────────────────────────────────                                      │
│   • 只 insert，不显式 remove                                                 │
│   • 查询时检查 task_id >= last_task_alive                                   │
│   • Ring buffer 自动覆盖旧 entries                                          │
│                                                                              │
│   优化2: 链表截断 (Chain Truncation)                                         │
│   ─────────────────────────────────                                          │
│   • Insert 总是在链表头部 → 链表按 task_id 降序排列                         │
│   • 遇到首个 stale entry → 后续一定都是 stale                               │
│   • 直接截断链表，无需遍历到末尾                                             │
│                                                                              │
│   效果:                                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  bucket[3] → [T=80:VALID] → [T=65:VALID] → [T=42:STALE] → ...   │       │
│   │                                              │                   │       │
│   │                                    遇到 stale 即截断，O(2) 完成  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   收益:                                                                      │
│   • 查找复杂度 = O(valid_entries_only)，不含 stale entries                  │
│   • 查找过程顺便完成 cleanup，无需单独清理                                   │
│   • 后续查找更快 (链表自动变短)                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3. Packed Output Buffer (减少分配次数)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PACKED OUTPUT BUFFER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Before (per-output):              After (packed):                         │
│   ────────────────────              ────────────────                        │
│   Task 有 3 个输出:                 Task 有 3 个输出:                        │
│   → 3 次 alloc                     → 1 次 alloc                             │
│   → 3 次 free                      → 1 次 free                              │
│   → 3 个独立 fanout                → 1 个统一 fanout                        │
│                                                                              │
│   ┌────┐ ┌────┐ ┌────┐            ┌──────────────────────┐                 │
│   │Out0│ │Out1│ │Out2│     →      │ Out0 │ Out1 │ Out2  │                 │
│   └────┘ └────┘ └────┘            └──────────────────────┘                 │
│                                         ↑ 一次分配                          │
│                                                                              │
│   节省: 若每 task 平均 N 个输出，减少 (N-1)/N 的 alloc/free 开销            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 三、并发与同步优化

#### 1. 数据所有权分离 (避免锁竞争)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA OWNERSHIP SEPARATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   核心原则: 谁写谁拥有，读者不加锁                                           │
│                                                                              │
│   ORCHESTRATOR 独占写入:              SCHEDULER 独占写入:                    │
│   ───────────────────────            ────────────────────                    │
│   • TaskDescriptor 内容               • task_state[]                         │
│   • fanin_count, fanin_list          • fanin_refcount[]                     │
│   • fanout_count, fanout_list*       • fanout_refcount[]                    │
│   • TensorMap                        • ready_queues[]                       │
│   • heap_top, current_task_index     • last_task_alive, heap_tail           │
│                                                                              │
│   *fanout_count/list 例外: 需要 per-task spinlock (见下文)                   │
│                                                                              │
│   收益:                                                                      │
│   • 大部分数据无需加锁                                                       │
│   • Orchestrator 和 Scheduler 可完全并行                                     │
│   • Cache 局部性好 (各自私有数据)                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2. 细粒度 Per-Task 自旋锁 (仅保护 fanout)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINE-GRAINED SPINLOCK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   问题: fanout_count 和 fanout_list 需要原子更新                            │
│                                                                              │
│   场景分析:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Orchestrator:                 Scheduler:                        │       │
│   │  add_consumer(producer, C)     check_consumed(producer)          │       │
│   │  {                             {                                 │       │
│   │    producer.fanout_count++;      if (fanout_refcount ==          │       │
│   │    producer.fanout_list +=C;         fanout_count &&             │       │
│   │  }                                   state == COMPLETED)         │       │
│   │                                    → CONSUMED                    │       │
│   │                                }                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   若不加锁: Scheduler 可能读到旧 fanout_count，导致过早判定 CONSUMED        │
│   → Use-After-Free!                                                          │
│                                                                              │
│   解决方案: Per-task spinlock (仅保护 fanout_head + fanout_count)            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  volatile int32_t fanout_lock;  // 每个 TaskDescriptor 一个      │       │
│   │                                                                  │       │
│   │  临界区极短: ~10 cycles (increment + list prepend)               │       │
│   │  锁竞争低: 同一 producer 不太可能被同时访问                       │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   fanin_count/fanin_list 不需要锁: 提交时一次性设置，之后只读                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 四、流控与背压机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLOW CONTROL & BACK-PRESSURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   设计目标: 防止 Orchestrator 产生过快导致内存溢出                           │
│                                                                              │
│   机制: Ring Buffer 满时 Orchestrator 自动 stall                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                  │       │
│   │   Task Ring Full:                                                │       │
│   │   ──────────────                                                 │       │
│   │   current_task_index == last_task_alive (modulo)                 │       │
│   │   → Orchestrator stalls in pto_submit_task()                     │       │
│   │   → Waits for Scheduler to advance last_task_alive               │       │
│   │                                                                  │       │
│   │   Heap Ring Full:                                                │       │
│   │   ──────────────                                                 │       │
│   │   heap_top + alloc_size > heap_tail                              │       │
│   │   → Orchestrator stalls in heap_ring_alloc()                     │       │
│   │   → Waits for Scheduler to advance heap_tail                     │       │
│   │                                                                  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   收益:                                                                      │
│   • 自动适应 Scheduler 处理速度                                              │
│   • 内存使用有界，不会 OOM                                                   │
│   • 无需复杂的拥塞控制算法                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 五、Scope-Based 生命周期管理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCOPE-BASED LIFECYCLE MANAGEMENT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   巧妙设计: fanout_count 初始值 = scope_depth                               │
│                                                                              │
│   目的: 确保 buffer 生命周期不早于其 scope                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  scope_begin()  // depth = 1                                     │       │
│   │    P = alloc()  // P.fanout_count = 1 (scope reference)          │       │
│   │    submit(gemm, out=P)                                           │       │
│   │    submit(add, in=P)  // P.fanout_count++ → 2                    │       │
│   │  scope_end()    // P.fanout_refcount++ → if ==2: release P       │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   scope_end 简化: 只需对 [scope_begin, scope_end) 范围内所有 task           │
│                   的 fanout_refcount++，无需复杂过滤                         │
│                                                                              │
│   嵌套 scope 支持:                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  scope_begin()  // depth=1                                       │       │
│   │    scope_begin()  // depth=2                                     │       │
│   │      P = alloc()  // P.fanout_count = 2                          │       │
│   │    scope_end()    // P.fanout_refcount++ → 1                     │       │
│   │  scope_end()      // P.fanout_refcount++ → 2 == count → release  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 六、Per-Worker-Type Ready Queue

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-WORKER-TYPE READY QUEUES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   设计: 同类型 Worker 可互换，按类型分 Ready Queue                           │
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────┐         │
│   │                                                                │         │
│   │   ready_queues[WORKER_CUBE]    ─────► [T1, T5, T9, ...]       │         │
│   │   ready_queues[WORKER_VECTOR]  ─────► [T2, T7, T12, ...]      │         │
│   │   ready_queues[WORKER_AI_CPU]  ─────► [T3, T8, ...]           │         │
│   │   ready_queues[WORKER_ACCEL]   ─────► [T4, T6, ...]           │         │
│   │                                                                │         │
│   └───────────────────────────────────────────────────────────────┘         │
│                                                                              │
│   收益:                                                                      │
│   • 避免单一全局队列的锁竞争                                                 │
│   • 天然负载均衡 (同类型 Worker 从同一队列取任务)                            │
│   • 支持异构计算单元高效调度                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 七、性能特征总结

| 设计要素 | 优化技术 | 性能收益 |
|---------|---------|---------|
| **内存分配** | Ring Buffer + Bump Pointer | Alloc: 100x faster, Free: 0 cost |
| **内存回收** | Implicit via pointer advance | No per-buffer free overhead |
| **TensorMap** | Lazy Invalidation + Chain Truncation | O(valid_only) lookup, auto cleanup |
| **数据同步** | Ownership separation | Mostly lock-free |
| **Fanout 保护** | Per-task spinlock | Minimal contention, ~10 cycles |
| **流量控制** | Ring full = stall | Auto back-pressure, no OOM |
| **生命周期** | Scope-based fanout init | Precise release, no leaks |
| **任务调度** | Per-worker-type queues | Load balanced, low contention |

---

### 八、完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. SCOPE BEGIN (Orchestrator)                                              │
│   ─────────────────────────────                                              │
│   pto_scope_begin():                                                         │
│     ├─ Push current_task_index to scope_stack                                │
│     ├─ scope_stack_top++ (depth increases)                                   │
│     └─ All subsequent tasks will have fanout_count init = new depth          │
│                                                                              │
│   2. TASK SUBMISSION (Orchestrator)                                          │
│   ─────────────────────────────────                                          │
│   pto_submit_task():                                                         │
│     ├─ Sync TensorMap (lazy invalidation threshold)                          │
│     ├─ Alloc task slot from Task Ring (may stall)                            │
│     ├─ Alloc packed buffer from Heap Ring (may stall)                        │
│     ├─ Lookup inputs in TensorMap → build fanin_list                         │
│     ├─ Update producer's fanout_count/list (with spinlock)                   │
│     ├─ Insert outputs in TensorMap                                           │
│     ├─ Init fanout_count = scope_depth (+ consumers as they submit)          │
│     └─ If fanin_refcount == fanin_count → push to ready_queue                │
│                                                                              │
│   3. TASK DISPATCH (Scheduler)                                               │
│   ────────────────────────────                                               │
│   scheduler_dispatch_loop():                                                 │
│     ├─ Pop task from ready_queue[worker_type]                                │
│     ├─ task_state = RUNNING                                                  │
│     └─ Dispatch to available worker                                          │
│                                                                              │
│   4. TASK COMPLETION (Worker → Scheduler)                                    │
│   ───────────────────────────────────────                                    │
│   pto_task_complete():                                                       │
│     ├─ task_state = COMPLETED                                                │
│     ├─ For each consumer: fanin_refcount++                                   │
│     │   └─ If fanin_refcount == fanin_count → push to ready_queue            │
│     ├─ For each producer: fanout_refcount++ (with spinlock check)            │
│     │   └─ If fanout_refcount == fanout_count && COMPLETED → CONSUMED        │
│     └─ Advance last_task_alive if possible                                   │
│                                                                              │
│   5. SCOPE END (Orchestrator)                                                │
│   ───────────────────────────                                                │
│   pto_scope_end():                                                           │
│     ├─ Get scope_begin_pos from scope_stack top                              │
│     ├─ For each task in [scope_begin_pos, current_task_index):               │
│     │   └─ fanout_refcount++ → check CONSUMED                                │
│     ├─ Pop scope_stack (depth decreases)                                     │
│     └─ Triggers release for tasks whose fanout_refcount == fanout_count      │
│                                                                              │
│   6. MEMORY RECLAMATION (Automatic)                                          │
│   ─────────────────────────────────                                          │
│   When last_task_alive advances:                                             │
│     ├─ Heap Ring: heap_tail follows → buffers implicitly freed               │
│     ├─ Task Ring: old slots available for reuse                              │
│     ├─ DepListPool: old entries overwritten on wrap                          │
│     └─ TensorMap: old entries become invalid (lazy)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 九、设计原则总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PRINCIPLES                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. FIFO 生命周期利用                                                       │
│      Tasks are created and retired in roughly FIFO order.                   │
│      → Use ring buffers everywhere, implicit reclamation.                   │
│                                                                              │
│   2. 最小化共享状态                                                          │
│      Only share what must be communicated.                                  │
│      → Private data stays private, reduces cache thrashing.                 │
│                                                                              │
│   3. 惰性处理                                                                │
│      Defer work until absolutely necessary.                                 │
│      → Lazy invalidation, no eager cleanup.                                 │
│                                                                              │
│   4. 批量操作                                                                │
│      Pack multiple outputs into one buffer.                                 │
│      → Reduce alloc/free count by Nx.                                       │
│                                                                              │
│   5. 细粒度锁                                                                │
│      Lock only what needs atomic access.                                    │
│      → Per-task spinlock, not global lock.                                  │
│                                                                              │
│   6. 自然流控                                                                │
│      Let resource exhaustion provide back-pressure.                         │
│      → Ring full = stall, no explicit rate limiting.                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Runtime2 Multi-Threaded Implementation (ascend_a2a3_sim)

This section describes the actual multi-threaded implementation for the `ascend_a2a3_sim` platform, where Orchestrator, Scheduler, and Workers run in **independent threads**.

### 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-THREADED ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────┐                                                    │
│   │  MAIN THREAD        │                                                    │
│   │  ───────────────    │                                                    │
│   │  • Create runtime   │                                                    │
│   │  • Start threads    │                                                    │
│   │  • Wait completion  │                                                    │
│   │  • Cleanup          │                                                    │
│   └─────────┬───────────┘                                                    │
│             │ pthread_create()                                               │
│             ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      WORKER THREADS                                  │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │   │
│   │  │ Cube 0  │  │ Cube 1  │  │ Vector 0│  │ Vector 1│  ...            │   │
│   │  │ ─────── │  │ ─────── │  │ ──────  │  │ ──────  │                 │   │
│   │  │ Wait on │  │ Wait on │  │ Wait on │  │ Wait on │                 │   │
│   │  │ ready_q │  │ ready_q │  │ ready_q │  │ ready_q │                 │   │
│   │  │ Execute │  │ Execute │  │ Execute │  │ Execute │                 │   │
│   │  │ Signal  │  │ Signal  │  │ Signal  │  │ Signal  │                 │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                 │   │
│   └───────┼────────────┼────────────┼────────────┼──────────────────────┘   │
│           │            │            │            │                           │
│           └────────────┴────────────┴────────────┘                           │
│                                │                                             │
│                    Completion Queue (MPSC)                                   │
│                                │                                             │
│                                ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SCHEDULER THREAD                                                    │   │
│   │  ─────────────────                                                   │   │
│   │  Loop:                                                               │   │
│   │    1. Poll current_task_index for new tasks                          │   │
│   │    2. Initialize task state, enqueue ready tasks                     │   │
│   │    3. Process completion queue                                       │   │
│   │    4. Update dependencies, advance ring pointers                     │   │
│   │    5. Signal all_done when orchestrator_done && all consumed         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                ▲                                             │
│                                │ Reads current_task_index                    │
│                                │                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  ORCHESTRATOR THREAD                                                 │   │
│   │  ────────────────────                                                │   │
│   │  • Execute user orchestration function                               │   │
│   │  • Submit tasks (writes to Task Ring)                                │   │
│   │  • Call scope_begin/scope_end                                        │   │
│   │  • Set orchestrator_done flag                                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Thread Context Structure

```c
/**
 * Thread context for managing all runtime threads
 */
typedef struct {
    // Orchestrator thread
    pthread_t orchestrator_thread;
    void* (*orchestrator_func)(void*);
    void* orchestrator_arg;
    volatile bool orchestrator_done;
    
    // Scheduler thread
    pthread_t scheduler_thread;
    volatile bool scheduler_running;
    
    // Worker threads
    PTO2WorkerContext workers[PTO2_MAX_WORKERS];
    int32_t num_cube_workers;
    int32_t num_vector_workers;
    int32_t num_workers;              // Total workers
    
    // Ready queue synchronization (per worker type)
    pthread_mutex_t ready_mutex[PTO2_NUM_WORKER_TYPES];
    pthread_cond_t  ready_cond[PTO2_NUM_WORKER_TYPES];
    
    // Completion queue (workers -> scheduler)
    PTO2CompletionQueue completion_queue;
    pthread_cond_t completion_cond;   // Signal scheduler when completions ready
    
    // Global shutdown signal
    volatile bool shutdown;
    
    // All-done signaling
    pthread_mutex_t done_mutex;
    pthread_cond_t  all_done_cond;
    volatile bool   all_done;
    
    // Cycle counter for simulation (atomic)
    volatile int64_t global_cycle;
    
} PTO2ThreadContext;
```

### 3. Synchronization Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SYNCHRONIZATION SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DATA STRUCTURE              SYNC MECHANISM          ACCESS PATTERN         │
│   ══════════════              ══════════════          ══════════════         │
│                                                                              │
│   Shared Memory Pointers                                                     │
│   ──────────────────────                                                     │
│   current_task_index          volatile + atomic       Orch WRITE, Sched READ │
│   last_task_alive             volatile + atomic       Sched WRITE, Orch READ │
│   heap_top/heap_tail          volatile + atomic       WRITE/READ each side   │
│   orchestrator_done           volatile + atomic       Orch WRITE, Sched READ │
│                                                                              │
│   Task Descriptors                                                           │
│   ────────────────                                                           │
│   Most fields                 Write-before-publish    Orch WRITE → Sched READ│
│   fanout_head/count           per-task spinlock       Orch/Sched R/W         │
│                                                                              │
│   Ready Queues                                                               │
│   ────────────                                                               │
│   queue[worker_type]          mutex + cond_var        Sched PUSH, Worker POP │
│                                                                              │
│   Completion Queue                                                           │
│   ────────────────                                                           │
│   entries[]                   mutex (MPSC)            Workers PUSH, Sched POP│
│                                                                              │
│   Scheduler Private State                                                    │
│   ───────────────────────                                                    │
│   task_state[]                NO SYNC (private)       Sched only             │
│   fanin_refcount[]            NO SYNC (private)       Sched only             │
│   fanout_refcount[]           NO SYNC (private)       Sched only             │
│                                                                              │
│   Global Control                                                             │
│   ──────────────                                                             │
│   shutdown                    volatile                Main → All threads     │
│   all_done                    mutex + cond_var        Sched → Main           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Thread-Safe Ready Queue Implementation

```c
/**
 * Push task to ready queue with thread synchronization
 */
bool pto2_ready_queue_push_threadsafe(PTO2ReadyQueue* queue, int32_t task_id,
                                       pthread_mutex_t* mutex, pthread_cond_t* cond) {
    pthread_mutex_lock(mutex);
    
    bool success = pto2_ready_queue_push(queue, task_id);
    
    if (success) {
        // Signal one waiting worker that a task is available
        pthread_cond_signal(cond);
    }
    
    pthread_mutex_unlock(mutex);
    return success;
}

/**
 * Pop task from ready queue with blocking wait
 */
int32_t pto2_ready_queue_pop_threadsafe(PTO2ReadyQueue* queue,
                                         pthread_mutex_t* mutex, pthread_cond_t* cond,
                                         volatile bool* shutdown) {
    pthread_mutex_lock(mutex);
    
    // Wait while queue is empty and not shutting down
    while (pto2_ready_queue_empty(queue) && !(*shutdown)) {
        pthread_cond_wait(cond, mutex);
    }
    
    // Check if we woke up due to shutdown
    if (*shutdown && pto2_ready_queue_empty(queue)) {
        pthread_mutex_unlock(mutex);
        return -1;
    }
    
    int32_t task_id = pto2_ready_queue_pop(queue);
    
    pthread_mutex_unlock(mutex);
    return task_id;
}
```

### 5. Worker Thread Implementation

```c
/**
 * Worker thread main function (simulation mode)
 */
void* pto2_worker_thread_func_sim(void* arg) {
    PTO2WorkerContext* worker = (PTO2WorkerContext*)arg;
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    while (!worker->shutdown) {
        // Get next task (blocks if queue empty)
        int32_t task_id = pto2_worker_get_task(worker);
        if (task_id < 0) {
            break;  // Shutdown signaled
        }
        
        // Record start time
        int64_t start_cycle = PTO2_LOAD_ACQUIRE(&ctx->global_cycle);
        worker->task_start_cycle = start_cycle;
        
        // Simulate the task (estimate cycles)
        int64_t cycles = pto2_worker_simulate_task(worker, task_id);
        
        // Advance global cycle counter
        PTO2_FETCH_ADD(&ctx->global_cycle, cycles);
        
        // Signal completion
        pto2_worker_task_complete(worker, task_id, cycles);
    }
    
    return NULL;
}
```

### 6. Scheduler Thread Implementation

```c
/**
 * Scheduler thread main function
 */
void* pto2_scheduler_thread_func(void* arg) {
    PTO2SchedulerContext* ctx = (PTO2SchedulerContext*)arg;
    PTO2SchedulerState* sched = ctx->scheduler;
    PTO2ThreadContext* thread_ctx = ctx->thread_ctx;
    
    int32_t last_processed_task = 0;
    
    while (!thread_ctx->shutdown) {
        bool did_work = false;
        
        // === STEP 1: Process new tasks from orchestrator ===
        // Poll current_task_index and initialize scheduler state
        int32_t new_tasks = process_new_tasks_threadsafe(sched, thread_ctx, 
                                                          &last_processed_task);
        if (new_tasks > 0) did_work = true;
        
        // === STEP 2: Process completions from workers ===
        int32_t completions = pto2_scheduler_process_completions(ctx);
        if (completions > 0) did_work = true;
        
        // === STEP 3: Advance ring pointers ===
        pto2_scheduler_advance_ring_pointers(sched);
        
        // === STEP 4: Check if all done ===
        if (pto2_scheduler_is_done(sched)) {
            pthread_mutex_lock(&thread_ctx->done_mutex);
            thread_ctx->all_done = true;
            pthread_cond_broadcast(&thread_ctx->all_done_cond);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
            break;
        }
        
        // If no work, wait with timeout to avoid busy-waiting
        if (!did_work) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 1000000;  // 1ms timeout
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            
            pthread_mutex_lock(&thread_ctx->done_mutex);
            pthread_cond_timedwait(&thread_ctx->completion_cond, 
                                   &thread_ctx->done_mutex, &ts);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
        }
    }
    
    return NULL;
}
```

### 7. Multi-Threaded vs Single-Threaded Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                SINGLE-THREADED vs MULTI-THREADED                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SINGLE-THREADED (Original)           MULTI-THREADED (New)                  │
│   ══════════════════════════           ═════════════════════                 │
│                                                                              │
│   main() {                             main() {                              │
│       // Phase 1: Build graph              rt = create_threaded();           │
│       for (all tasks)                      run_threaded(rt, orch_func);      │
│           submit_task();                   // ^ starts all threads           │
│       orchestration_done();                // ^ waits for completion         │
│                                            destroy_threaded(rt);             │
│       // Phase 2: Execute              }                                     │
│       sim_run(); // after phase 1                                            │
│   }                                    orch_func(rt) {                       │
│                                            for (all tasks)                   │
│                                                submit_task();                │
│   Problem:                                 // Scheduler processes            │
│   - Deadlock when task ring full           // tasks concurrently!            │
│   - sim_run() starts AFTER all         }                                     │
│     tasks submitted                                                          │
│   - No flow control testing            Benefit:                              │
│                                        - True concurrency                    │
│                                        - Flow control/back-pressure works    │
│                                        - Realistic simulation                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8. Key Design Decisions

#### 8.1 Orchestrator-Scheduler Decoupling

In multi-threaded mode, the orchestrator **does not** directly call scheduler methods for task initialization:

```c
// In multi-threaded mode:
pto2_orchestrator_set_scheduler_mode(&rt->base.orchestrator, 
                                      &rt->base.scheduler, 
                                      false);  // init_on_submit = false

// Scheduler thread polls current_task_index to discover new tasks
// This avoids race condition where:
//   - Orchestrator calls scheduler_init_task()
//   - Scheduler thread also processes the same task
```

However, `scope_end` **still** calls `pto2_scheduler_on_scope_end()` because:
- It only modifies `fanout_refcount[]` (scheduler-private)
- Workers don't modify refcounts (they signal via completion queue)
- Scheduler is the sole writer of refcount arrays

#### 8.2 Avoiding fanout_refcount Reset

Critical fix for multi-threaded mode:

```c
// In process_new_tasks_threadsafe():
sched->task_state[slot] = PTO2_TASK_PENDING;
sched->fanin_refcount[slot] = 0;
// sched->fanout_refcount[slot] = 0;  // DO NOT reset!
// ^ scope_end may have already incremented this before scheduler processes
```

Timeline that causes bug if refcount is reset:
```
T1: Orchestrator submits tasks 0-3
T2: Orchestrator calls scope_end() → fanout_refcount[0..3] += 1
T3: Scheduler processes new tasks → if reset, fanout_refcount = 0 (BUG!)
T4: Tasks complete but never reach CONSUMED state
```

### 9. Usage Example

```c
// User orchestration function
void bgemm_orchestration(PTO2Runtime* rt, void* arg) {
    BgemmParams* p = (BgemmParams*)arg;
    
    for (int b = 0; b < p->batch; b++) {
        pto2_rt_scope_begin(rt);
        
        for (int m = 0; m < p->m_tiles; m++) {
            for (int n = 0; n < p->n_tiles; n++) {
                pto2_rt_scope_begin(rt);
                
                for (int k = 0; k < p->k_tiles; k++) {
                    // Submit gemm_tile (CUBE)
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL,
                                        "gemm_tile", gemm_params, 3);
                    
                    // Submit tile_add (VECTOR)
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL,
                                        "tile_add", add_params, 3);
                }
                
                pto2_rt_scope_end(rt);
            }
        }
        
        pto2_rt_scope_end(rt);
    }
}

int main() {
    // Create threaded runtime
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(
        4,    // num_cube_workers
        4,    // num_vector_workers
        true  // simulation_mode
    );
    
    // Run with multi-threading
    BgemmParams params = { .batch = 4, .m_tiles = 4, ... };
    pto2_runtime_run_threaded(rt, bgemm_orchestration, &params);
    
    // Write trace
    pto2_runtime_write_trace(rt, "trace.json");
    
    // Cleanup
    pto2_runtime_destroy_threaded(rt);
    return 0;
}
```

### 10. Performance Results

Test configuration: BGEMM with batch=4, m=4, n=4, k=4 (512 tasks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE SUMMARY                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Metric                      Value                                          │
│   ══════                      ═════                                          │
│   Total tasks:                512                                            │
│   Total time:                 2.191 ms                                       │
│   Throughput:                 233.71 tasks/ms                                │
│   Simulated cycles:           38,400                                         │
│                                                                              │
│   Worker Statistics:                                                         │
│   ──────────────────                                                         │
│   CUBE workers (4):           256 tasks total, avg 100 cycles/task           │
│   VECTOR workers (4):         256 tasks total, avg 50 cycles/task            │
│                                                                              │
│   Resource Usage:                                                            │
│   ───────────────                                                            │
│   Heap Ring:                  65,536 / 67,108,864 bytes                      │
│   Task Ring:                  512 / 1024 slots                               │
│   DepList Pool:               896 / 8192 entries                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11. Runtime Creation APIs

The runtime2 provides two APIs for creating multi-threaded runtime instances:

#### Standard Creation API

```c
/**
 * Create a threaded runtime with default resource sizes
 * 
 * Uses compile-time constants from pto_runtime2_types.h:
 *   - PTO2_TASK_WINDOW_SIZE (default: 1024)
 *   - PTO2_HEAP_SIZE (default: 64MB)
 *   - PTO2_DEP_LIST_POOL_SIZE (default: 8192)
 * 
 * @param num_cube_workers   Number of CUBE worker threads (e.g., 4)
 * @param num_vector_workers Number of VECTOR worker threads (e.g., 4)
 * @param simulation_mode    true = cycle simulation, false = real execution
 * @return Runtime context, or NULL on failure
 */
PTO2RuntimeThreaded* pto2_runtime_create_threaded(
    int32_t num_cube_workers,
    int32_t num_vector_workers,
    bool simulation_mode
);
```

**Example:**
```c
// Simple creation with defaults
PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(4, 4, true);
```

#### Custom Creation API

```c
/**
 * Create a threaded runtime with custom resource sizes
 * 
 * Use this API when:
 *   - Default task_window_size causes deadlock (see Flow Control Deadlock section)
 *   - Workload requires larger heap or dependency pools
 *   - Fine-tuning resource usage for specific algorithms
 * 
 * @param num_cube_workers   Number of CUBE worker threads
 * @param num_vector_workers Number of VECTOR worker threads
 * @param simulation_mode    true = cycle simulation, false = real execution
 * @param task_window_size   Task Ring size (MUST be power of 2)
 * @param heap_size          Heap Ring size in bytes
 * @param dep_list_size      DepList pool size (number of entries)
 * @return Runtime context, or NULL on failure
 */
PTO2RuntimeThreaded* pto2_runtime_create_threaded_custom(
    int32_t num_cube_workers,
    int32_t num_vector_workers,
    bool simulation_mode,
    int32_t task_window_size,
    int32_t heap_size,
    int32_t dep_list_size
);
```

**Example - Avoiding Deadlock:**
```c
// BGEMM with batch=2, m=8, n=8, k=8 generates 2048 tasks
// Default window (1024) causes deadlock!
// Solution: Use custom API with larger window

PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
    4,                      // num_cube_workers
    4,                      // num_vector_workers
    true,                   // simulation_mode
    4096,                   // task_window_size (increased from 1024!)
    64 * 1024 * 1024,       // heap_size (64MB)
    8192                    // dep_list_size
);

if (!rt) {
    fprintf(stderr, "Failed to create runtime\n");
    return 1;
}

// Run orchestration
pto2_runtime_run_threaded(rt, bgemm_orchestration, &params);

// Cleanup
pto2_runtime_destroy_threaded(rt);
```

#### API Comparison

| Feature | `pto2_runtime_create_threaded` | `pto2_runtime_create_threaded_custom` |
|---------|-------------------------------|--------------------------------------|
| Resource sizes | Compile-time defaults | Runtime configurable |
| Recompilation needed | Yes (to change sizes) | No |
| Use case | Standard workloads | Large/custom workloads |
| Deadlock avoidance | May require recompile | Adjust at runtime |

#### Sizing Guidelines

```
task_window_size:
  - MUST be power of 2 (for efficient modulo)
  - MUST be > max tasks in any single scope
  - Recommended: 2x expected max concurrent tasks
  
  Example: BGEMM(batch=2, m=8, n=8, k=8)
    tasks = 2 * 8 * 8 * 8 * 2 = 2048
    min window = 2049
    recommended = 4096 (power of 2, with headroom)

heap_size:
  - Total memory for all task output buffers
  - Calculate: max_concurrent_tasks * avg_output_size
  
dep_list_size:
  - Pool for dependency list entries
  - Calculate: max_concurrent_tasks * avg_fanout
```

### 12. File Structure

```
src/runtime2/
├── pto_runtime2_types.h      # Core types + thread context definitions
├── pto_runtime2.h/c          # Base runtime (single-threaded)
├── pto_runtime2_threaded.h/c # Multi-threaded runtime extension
├── pto_worker.h/c            # Worker thread implementation
├── pto_scheduler.h/c         # Scheduler + thread-safe operations
├── pto_orchestrator.h/c      # Orchestrator + multi-thread mode support
├── pto_shared_memory.h/c     # Shared memory structures
├── pto_ring_buffer.h/c       # Ring buffer implementations
├── pto_tensormap.h/c         # TensorMap for dependency discovery
└── tests/
    └── test_bgemm_runtime2.c # BGEMM test (single/multi-threaded modes)
```

---

## Flow Control and Backpressure Mechanism

### Overview

The PTO Runtime implements comprehensive **flow control** to handle resource exhaustion gracefully in multi-threaded execution mode. When any ring buffer runs out of space, the orchestrator thread will **block and wait** until the scheduler frees up resources by completing tasks.

### Ring Buffer Flow Control Points

There are **five** ring buffers that implement flow control:

| Ring Buffer | Purpose | Blocking Condition |
|-------------|---------|-------------------|
| **Task Ring** | Sliding window of active tasks | `tasks_in_flight >= PTO_TASK_WINDOW_SIZE` |
| **TensorMap Pool** | Tracks output tensor → producer task mapping | Entry still in use by live task |
| **DepList Pool** | Stores fanin/fanout dependency lists | `next_offset == tail` (ring full) |
| **Heap Ring** | Allocates packed output buffers | Insufficient contiguous space |
| **Ready Queue** | Tasks ready for execution | Queue full |

### Flow Control Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLOW CONTROL ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────┐           ┌──────────────────────┐              │
│   │   Orchestrator       │           │      Scheduler       │              │
│   │   Thread             │           │      (Workers)       │              │
│   └──────────┬───────────┘           └──────────┬───────────┘              │
│              │                                   │                          │
│              ▼                                   │                          │
│   ┌──────────────────────────────────────────────▼──────────────────────┐  │
│   │                        Ring Buffers                                  │  │
│   │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │  │
│   │  │ Task Ring  │ │ TensorMap  │ │  DepList   │ │   Heap     │       │  │
│   │  │            │ │   Pool     │ │   Pool     │ │   Ring     │       │  │
│   │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘       │  │
│   │        │              │              │              │               │  │
│   │        └──────────────┴──────────────┴──────────────┘               │  │
│   │                              │                                       │  │
│   └──────────────────────────────┼───────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │               Flow Control Mechanism                                 │  │
│   │                                                                      │  │
│   │   Orchestrator WRITE:              Scheduler ADVANCE:                │  │
│   │   • Allocate task slot    ◄────►   • last_task_alive (completed)    │  │
│   │   • Insert TensorMap      ◄────►   • TensorMap entries invalidated   │  │
│   │   • Allocate DepList      ◄────►   • dep_list_tail                   │  │
│   │   • Allocate heap buffer  ◄────►   • heap_tail                       │  │
│   │                                                                      │  │
│   │   If resource exhausted:           On task completion:               │  │
│   │   • Record stall reason            • Advance pointers                │  │
│   │   • pthread_cond_wait()            • pthread_cond_broadcast()        │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stall Tracking and Statistics

The runtime tracks detailed statistics about flow control stalls for performance tuning:

```c
// Stall reasons enumeration
typedef enum {
    PTO_STALL_NONE = 0,           // No stall
    PTO_STALL_TASK_RING,          // Task window full
    PTO_STALL_TENSORMAP_POOL,     // TensorMap pool full
    PTO_STALL_DEPLIST_POOL,       // DepList pool full
    PTO_STALL_HEAP_RING,          // Heap full
    PTO_STALL_READY_QUEUE,        // Ready queue full
} PTOStallReason;

// Flow control statistics structure
typedef struct {
    // Stall counts per resource
    int64_t task_ring_stalls;         // Times blocked on task window
    int64_t tensormap_pool_stalls;    // Times blocked on TensorMap pool
    int64_t deplist_pool_stalls;      // Times blocked on DepList pool
    int64_t heap_ring_stalls;         // Times blocked on heap
    int64_t ready_queue_stalls;       // Times blocked on ready queue
    
    // Total stall time (nanoseconds)
    int64_t task_ring_stall_ns;
    int64_t tensormap_pool_stall_ns;
    int64_t deplist_pool_stall_ns;
    int64_t heap_ring_stall_ns;
    int64_t ready_queue_stall_ns;
    
    // High water marks
    int32_t task_ring_hwm;            // Max tasks in flight
    int32_t tensormap_pool_hwm;       // Max TensorMap entries used
    int32_t deplist_pool_hwm;         // Max DepList entries used
    int32_t heap_hwm;                 // Max heap bytes allocated
    int32_t ready_queue_hwm;          // Max ready queue size
    
    // Current stall state
    volatile PTOStallReason current_stall;
} PTOFlowControlStats;
```

### Flow Control API

```c
// Get flow control statistics
const PTOFlowControlStats* pto_get_flow_stats(PTORuntime* rt);

// Reset statistics
void pto_reset_flow_stats(PTORuntime* rt);

// Print detailed statistics with recommendations
void pto_print_flow_stats(PTORuntime* rt);

// Get current stall reason (useful for debugging)
PTOStallReason pto_get_current_stall(PTORuntime* rt);

// Get stall reason name
const char* pto_stall_reason_name(PTOStallReason reason);
```

### Example Statistics Output

```
[PTO Flow Control Statistics]
================================================================================

Stall Counts (times orchestration was blocked):
  Task Ring (window full):              5
  TensorMap Pool:                       0
  DepList Pool:                         0
  Heap Ring:                           12
  Ready Queue:                          0

Stall Times (nanoseconds):
  Task Ring (window full):        1234567 ns  (1.23 ms)
  TensorMap Pool:                       0 ns
  DepList Pool:                         0 ns
  Heap Ring:                      5678901 ns  (5.68 ms)
  Ready Queue:                          0 ns

Total Stall Time:                 6913468 ns  (6.91 ms)

High Water Marks (peak usage):
  Task Ring:           7890 / 8192  (96.3%)
  TensorMap Pool:    120000 / 262144  (45.8%)
  DepList Pool:       50000 / 131072  (38.1%)
  Heap Ring:       900000000 / 1073741824 bytes  (83.8%)
  Ready Queue:           50 / 65536  (0.1%)

Sizing Recommendations:
  [!] Task Ring was exhausted 5 times.
      Consider increasing PTO_TASK_WINDOW_SIZE (current: 8192)
  [!] Heap Ring was exhausted 12 times.
      Consider increasing PTO_HEAP_SIZE_BYTES (current: 1024 MB)
  [WARN] Task Ring usage reached 96.3% - consider increasing size

================================================================================
```

### Performance Tuning Based on Statistics

| Statistic | Interpretation | Action |
|-----------|---------------|--------|
| High `task_ring_stalls` | Too many tasks in flight, scheduler can't keep up | Increase `PTO_TASK_WINDOW_SIZE` or optimize task granularity |
| High `tensormap_pool_stalls` | Many output tensors, pool is too small | Increase `PTO_TENSORMAP_POOL_SIZE` |
| High `deplist_pool_stalls` | Dense dependencies, pool is too small | Increase `PTO_DEP_LIST_POOL_SIZE` |
| High `heap_ring_stalls` | Large output buffers, heap is too small | Increase `PTO_HEAP_SIZE_BYTES` |
| High `ready_queue_stalls` | Many tasks ready at once, rare | Increase `PTO_MAX_READY_QUEUE` |
| HWM > 90% | Resource approaching limit | Proactively increase size |
| Stall time significant | Orchestrator blocked waiting | Tune resource sizes or reduce parallelism |

### Condition Variables for Flow Control

The runtime uses dedicated condition variables for each resource:

```c
// In PTORuntime structure
pthread_cond_t window_not_full;       // Task ring has space
pthread_cond_t tensormap_not_full;    // TensorMap pool has space
pthread_cond_t deplist_not_full;      // DepList pool has space
pthread_cond_t heap_not_full;         // Heap ring has space
pthread_cond_t ready_queue_not_full;  // Ready queue has space
```

When the scheduler completes a task and advances `last_task_alive`, it broadcasts **all** flow control condition variables to wake up any blocked orchestrator thread:

```c
// In scheduler task completion path
bool window_advanced = pto_advance_last_task_alive_locked(rt);
if (window_advanced) {
    pthread_cond_broadcast(&rt->window_not_full);
    pthread_cond_broadcast(&rt->tensormap_not_full);
    pthread_cond_broadcast(&rt->deplist_not_full);
    pthread_cond_broadcast(&rt->heap_not_full);
}
```

### Task Window Deadlock Detection (runtime2)

#### The Deadlock Problem

In multi-threaded mode, there is a potential **circular dependency deadlock** when `TASK_WINDOW_SIZE` is too small:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TASK WINDOW DEADLOCK SCENARIO                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Task created with fanout_count = scope_depth (e.g., 2 for nested scope) │
│                                                                              │
│  2. Task completes execution (COMPLETED state)                               │
│     └── But cannot transition to CONSUMED because:                           │
│         fanout_refcount (0) != fanout_count (2)                              │
│                                                                              │
│  3. scope_end() would release references (increment fanout_refcount)         │
│     └── But scope_end() is called by Orchestrator                            │
│                                                                              │
│  4. Orchestrator is BLOCKED waiting for task ring space!                     │
│     └── Because task_window is full (active_tasks >= TASK_WINDOW_SIZE)       │
│                                                                              │
│  DEADLOCK:                                                                   │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                                                                   │     │
│    │   Orchestrator ──waits for──► Task Ring Space                    │     │
│    │        ▲                           │                              │     │
│    │        │                           │                              │     │
│    │   scope_end()               Requires last_task_alive              │     │
│    │   releases refs                 to advance                        │     │
│    │        │                           │                              │     │
│    │        │                           ▼                              │     │
│    │   Tasks need ◄──blocked by──  Tasks cannot                       │     │
│    │   scope_end                    become CONSUMED                    │     │
│    │                                                                   │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Deadlock Detection Mechanism

The runtime detects this deadlock by monitoring spin-wait iterations. If the orchestrator spins for more than `PTO2_FLOW_CONTROL_SPIN_LIMIT` (100,000 iterations) without progress, it indicates a deadlock:

```c
// In pto_ring_buffer.c
#define PTO2_FLOW_CONTROL_SPIN_LIMIT  100000

int32_t pto2_task_ring_alloc(PTO2TaskRing* ring) {
    int spin_count = 0;
    
    while (1) {
        int32_t task_id = pto2_task_ring_try_alloc(ring);
        if (task_id >= 0) {
            return task_id;
        }
        
        // Check for potential deadlock
        if (spin_count >= PTO2_FLOW_CONTROL_SPIN_LIMIT) {
            // Report deadlock and abort
            pto2_report_flow_control_deadlock(ring);
            exit(1);
        }
        
        spin_count++;
        PTO2_SPIN_PAUSE();
    }
}
```

#### Error Message and User Guidance

When deadlock is detected, the runtime prints a detailed error message:

```
========================================
FATAL: Flow Control Deadlock Detected!
========================================

Task Ring is FULL and no progress after 100000 spins.

Flow Control Status:
  - Current task index:  1023
  - Last task alive:     0
  - Active tasks:        1023
  - Window size:         1024
  - Window utilization:  99.9%

Root Cause:
  Tasks cannot transition to CONSUMED state because:
  - fanout_count is initialized to scope_depth
  - scope_end() requires orchestrator to continue
  - But orchestrator is blocked waiting for task ring space
  This creates a circular dependency (deadlock).

Solution:
  Increase PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h
  Current value: 1024
  Recommended:   2046 (at least 2x current active tasks)

  Or use pto2_runtime_create_threaded_custom() with larger
  task_window_size parameter.
========================================
```

#### Sizing Guidelines

To avoid deadlock, ensure `TASK_WINDOW_SIZE` is larger than the maximum number of tasks in any single scope:

```
TASK_WINDOW_SIZE > MAX_TASKS_IN_SCOPE

Where MAX_TASKS_IN_SCOPE depends on your algorithm:
  - BGEMM example: batch * m_tiles * n_tiles * k_tiles * 2
  - For (2, 8, 8, 8): 2 * 8 * 8 * 8 * 2 = 2048 tasks
  - Minimum window: 2048 + 1 = 2049
  - Recommended:    4096 (power of 2, with headroom)
```

**Configuration Options:**

1. **Compile-time** (in `pto_runtime2_types.h`):
   ```c
   #define PTO2_TASK_WINDOW_SIZE  4096  // Increase as needed
   ```

2. **Runtime** (using custom creation API):
   ```c
   PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
       num_cube_workers,
       num_vector_workers,
       simulation_mode,
       4096,    // task_window_size - increase this!
       PTO2_HEAP_SIZE,
       PTO2_DEP_LIST_POOL_SIZE
   );
   ```

#### Why Not Auto-Grow the Window?

The task window is a **fixed-size ring buffer** for several design reasons:

1. **Deterministic memory usage** - Important for embedded/constrained environments
2. **Zero allocation overhead** - No malloc/realloc during execution
3. **Cache efficiency** - Fixed array is more cache-friendly than linked structures
4. **Bounded latency** - No garbage collection pauses

If your workload exceeds the window size, the recommended approach is:
- Calculate required window size based on algorithm parameters
- Configure appropriate size at initialization time
- Use `pto2_runtime_create_threaded_custom()` for runtime configuration

---

## Single-Threaded Orchestration-Only Mode Limitations

### Resource Sizing for Large Task Graphs

When running in orchestration-only mode (build graph first, simulate later), **all tasks must fit in memory simultaneously** because `heap_tail` and `last_task_alive` don't advance until simulation runs.

**Critical Configuration Parameters** (`pto_runtime_common.h`):

```c
// TensorMap pool must hold ALL output entries from all tasks
// Each task with N outputs needs N TensorMap entries
#define PTO_TENSORMAP_POOL_SIZE (PTO_TASK_WINDOW_SIZE * 32)  // 262144 entries

// Heap must hold ALL packed output buffers from all tasks
// Calculate: num_tasks_with_output * avg_output_size
#define PTO_HEAP_SIZE_BYTES    (1024 * 1024 * 1024)  // 1GB
```

**Symptoms of Insufficient Resources**:

| Resource | Symptom | Solution |
|----------|---------|----------|
| TensorMap pool overflow | `fanin_count = 0` for tasks that should have dependencies | Increase `PTO_TENSORMAP_POOL_SIZE` |
| Heap overflow | `heap alloc failed` errors, tasks not submitted | Increase `PTO_HEAP_SIZE_BYTES` |
| Task window overflow | Only partial task graph visible in dump | Increase `PTO_TASK_WINDOW_SIZE` |

### Recommended Approach: Multi-Threaded Mode

For large task graphs, use **multi-threaded mode** instead of orchestration-only:

```c
// Multi-threaded: orchestration and simulation run in parallel
// - heap_tail advances as tasks complete → heap space recycled
// - last_task_alive advances → TensorMap entries recycled
// - Flow control works naturally via ring buffer stalls
PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(
    num_cube_workers, num_vector_workers, true);
pto2_runtime_run_threaded(rt, orchestration_func, params);
```

Benefits:
- No need for oversized buffers
- Flow control/backpressure works correctly
- Realistic simulation of actual hardware behavior

---

## Multi-Threaded Simulation: Data Race Issues and Solutions

本节记录在 A2A3_SIM 多线程仿真模式开发过程中遇到的 data race 问题及其解决方案。

### 1. 问题背景

在多线程仿真模式下，有多个并发执行的组件：
- **Orchestrator 线程**：提交任务，构建依赖关系
- **Scheduler 线程**：处理完成事件，转换任务状态
- **Worker 线程（多个）**：执行任务，报告完成

这些线程共享以下关键数据结构：
- Task descriptors (`task_state`, `fanin_refcount`, `fanout_refcount`)
- TensorMap (producer-consumer tracking)
- Ready queues (per worker type)
- Dependency lists (fanin/fanout chains)

### 2. Data Race 问题及解决方案

#### 2.1 Producer-Consumer 注册竞争

**问题**：Producer 任务可能在 Consumer 完全注册之前就完成了。

```
时序问题：
  Orchestrator                Producer Worker
  ────────────                ───────────────
  1. submit(Consumer)
  2. lookup producer
  3. add_consumer_to_producer ────────────────> Producer completes!
     (此时 consumer 尚未完全注册)              fanout_refcount++
                                               on_task_complete()
                                               遍历 fanout_head (Consumer 不在其中!)
  4. 完成注册
     (Consumer 永远不会收到 fanin 通知!)
```

**解决方案**：使用 Per-Task Spinlock 保护 fanout 链表

```c
// 在 task descriptor 中添加
typedef struct {
    // ...
    volatile int32_t fanout_lock;  // Spinlock for fanout list
} PTO2TaskDescriptor;

// add_consumer_to_producer 中获取锁
void pto2_add_consumer_to_producer(orchestrator, producer_id, consumer_id, ...) {
    // 获取 spinlock
    while (__atomic_exchange_n(&producer_task->fanout_lock, 1, __ATOMIC_ACQUIRE) != 0) {
        PTO2_SPIN_PAUSE();
    }
    
    // 检查 producer 是否已完成
    int32_t producer_state = PTO2_LOAD_ACQUIRE(&producer_task->task_state);
    
    if (producer_state >= PTO2_TASK_COMPLETED) {
        // Producer 已完成，直接增加 consumer 的 fanin_refcount
        __atomic_fetch_add(&consumer_fanin_refcount, 1, __ATOMIC_ACQ_REL);
        if (fanin_refcount == fanin_count) {
            // Consumer 变为 READY
        }
    }
    
    // 添加到 fanout 链表
    // ...
    
    // 释放 spinlock
    __atomic_store_n(&producer_task->fanout_lock, 0, __ATOMIC_RELEASE);
}
```

#### 2.2 Task 状态转换竞争

**问题**：多个线程可能同时尝试转换同一任务的状态。

```
场景：两个 Producer 同时完成，都尝试将同一 Consumer 从 PENDING 转为 READY

  Worker 1 (Producer A)        Worker 2 (Producer B)
  ────────────────────         ────────────────────
  fanin_refcount++              fanin_refcount++
  check: refcount == count      check: refcount == count
  if (state == PENDING) {       if (state == PENDING) {
      state = READY             // 两者都尝试设置！
```

**解决方案**：使用 CAS (Compare-And-Swap) 进行原子状态转换

```c
void check_and_transition_to_ready(scheduler, task_id) {
    int32_t current = PTO2_LOAD_ACQUIRE(&task->task_state);
    
    // 只有 PENDING 状态才能转为 READY
    if (current != PTO2_TASK_PENDING) return;
    
    // CAS: 原子地 PENDING -> READY
    if (__atomic_compare_exchange_n(
            &task->task_state,
            &current,           // expected
            PTO2_TASK_READY,    // desired
            false,              // weak
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE)) {
        // 成功：加入 ready queue
        enqueue_ready(scheduler, task_id);
    }
    // 失败：另一个线程已经处理了
}
```

#### 2.3 线程启动顺序问题

**问题**：Orchestrator 在 Worker/Scheduler 准备好之前就开始提交任务。

**解决方案**：使用条件变量同步启动顺序

```c
typedef struct {
    pthread_mutex_t startup_mutex;
    pthread_cond_t startup_cond;
    int32_t workers_ready;
    bool scheduler_ready;
} PTO2ThreadContext;

// Worker 线程启动时
void* worker_thread_func(void* arg) {
    // 初始化...
    
    pthread_mutex_lock(&ctx->startup_mutex);
    ctx->workers_ready++;
    pthread_cond_broadcast(&ctx->startup_cond);
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    // 开始工作循环...
}

// Orchestrator 启动时等待
void orchestrator_start(ctx) {
    pthread_mutex_lock(&ctx->startup_mutex);
    while (ctx->workers_ready < total_workers || !ctx->scheduler_ready) {
        pthread_cond_wait(&ctx->startup_cond, &ctx->startup_mutex);
    }
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    // 现在可以安全提交任务
}
```

#### 2.4 Completion Queue 满载问题

**问题**：Worker 完成任务后无法推入 completion queue（队列满）。

**解决方案**：重试机制 + 主动唤醒 Scheduler

```c
void pto2_worker_task_complete(worker, task_id, start_cycle, end_cycle) {
    int retry_count = 0;
    while (!pto2_completion_queue_push(&completion_queue, ...)) {
        // 队列满，唤醒 scheduler 处理
        pthread_mutex_lock(&done_mutex);
        pthread_cond_signal(&completion_cond);
        pthread_mutex_unlock(&done_mutex);
        
        retry_count++;
        if (retry_count % 1000 == 0) {
            fprintf(stderr, "[Worker %d] Completion queue full, retrying...\n",
                    worker->worker_id);
        }
        PTO2_SPIN_PAUSE();
    }
    
    // 成功后通知 scheduler
    pthread_mutex_lock(&done_mutex);
    pthread_cond_signal(&completion_cond);
    pthread_mutex_unlock(&done_mutex);
}
```

### 3. 原子操作宏定义

为了代码可读性和跨平台兼容性，定义了以下宏：

```c
// 原子加载（acquire 语义）
#define PTO2_LOAD_ACQUIRE(ptr) \
    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)

// 原子存储（release 语义）
#define PTO2_STORE_RELEASE(ptr, val) \
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE)

// 自旋等待提示
#define PTO2_SPIN_PAUSE() \
    do { __asm__ volatile("yield" ::: "memory"); } while(0)
```

---

## Clock-Based Load Balancing in Simulation Mode

在仿真模式下，所有任务执行时间是瞬时的（无真实延迟），这导致 OS 线程调度的不公平性会被放大。本节描述基于时钟的负载均衡机制。

### 1. 问题描述

```
没有负载均衡时的典型分布（24 Cube workers, 2048 任务）：

  Worker 0:  342 tasks (16.7%)
  Worker 1:  285 tasks (13.9%)
  Worker 2:   89 tasks (4.3%)
  ...
  Worker 23:  12 tasks (0.6%)

问题：某些 worker 获得大量任务，其他几乎空闲
原因：pthread_cond_signal 倾向于唤醒同一个 worker（OS 调度偏好）
```

### 2. 设计目标

**核心规则**：
> 当时钟相同时，可以竞争不公平；但时钟不同时，不允许把任务分配给时钟更大的 worker。

这意味着：
- Clock 更小的 worker 应该优先获得任务
- Clock 相同时允许多线程自由竞争
- Clock 更大的 worker 必须等待，直到其 clock 变为最小

### 3. 实现机制

#### 3.1 Per-Worker 时钟追踪

```c
typedef struct {
    // 每个 worker 的当前仿真时钟
    volatile int64_t worker_current_cycle[PTO2_MAX_WORKERS];
    
    // 每个 worker 的等待标志
    volatile bool worker_waiting[PTO2_MAX_WORKERS];
    
    // 每个 worker 的独立条件变量
    pthread_cond_t worker_cond[PTO2_MAX_WORKERS];
} PTO2ThreadContext;
```

#### 3.2 Worker 获取任务时的时钟检查

```c
// 检查当前 worker 是否有最小时钟
static bool worker_has_min_clock(PTO2WorkerContext* worker, PTO2ThreadContext* ctx) {
    int64_t my_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
    
    // 与所有同类型 worker 比较（不只是等待中的）
    for (int32_t i = start_id; i < end_id; i++) {
        if (i == worker->worker_id) continue;
        
        int64_t other_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
        if (other_clock < my_clock) {
            return false;  // 存在更小时钟的 worker
        }
        // 时钟相等时允许竞争，不返回 false
    }
    return true;
}
```

#### 3.3 Worker 获取任务的完整流程

```c
int32_t pto2_worker_get_task(PTO2WorkerContext* worker) {
    // 标记为等待状态
    __atomic_store_n(&ctx->worker_waiting[worker->worker_id], true, __ATOMIC_RELEASE);
    
    pthread_mutex_lock(mutex);
    
    while (!worker->shutdown) {
        if (pto2_ready_queue_empty(queue)) {
            // 队列空，等待
            pthread_cond_wait(my_cond, mutex);
        } else if (!worker_has_min_clock(worker, ctx)) {
            // 队列有任务但我不是最小时钟
            // 广播唤醒所有 worker，然后等待
            pthread_cond_broadcast(&ctx->ready_cond[worker->worker_type]);
            pthread_cond_timedwait(my_cond, mutex, &timeout);
        } else {
            // 队列有任务且我是最小时钟 - 获取任务
            break;
        }
    }
    
    // 不再等待
    __atomic_store_n(&ctx->worker_waiting[worker->worker_id], false, __ATOMIC_RELEASE);
    
    // 获取任务
    int32_t task_id = pto2_ready_queue_pop(queue);
    
    // 如果还有任务，唤醒其他最小时钟的等待 worker
    if (!pto2_ready_queue_empty(queue)) {
        wake_min_clock_workers(ctx, worker->worker_type);
    }
    
    pthread_mutex_unlock(mutex);
    return task_id;
}
```

#### 3.4 任务入队时的选择性唤醒

```c
bool pto2_ready_queue_push_wake_min_clock(queue, task_id, mutex,
                                           worker_clocks, worker_waiting,
                                           worker_conds, worker_start, worker_end) {
    pthread_mutex_lock(mutex);
    
    bool success = pto2_ready_queue_push(queue, task_id);
    
    if (success) {
        // 唤醒所有等待中的 worker
        // 它们会自己检查时钟是否最小
        for (int32_t i = worker_start; i < worker_end; i++) {
            if (__atomic_load_n(&worker_waiting[i], __ATOMIC_ACQUIRE)) {
                pthread_cond_signal(&worker_conds[i]);
            }
        }
    }
    
    pthread_mutex_unlock(mutex);
    return success;
}
```

#### 3.5 任务完成时更新时钟

```c
void* pto2_worker_thread_func_sim(void* arg) {
    while (!worker->shutdown) {
        int32_t task_id = pto2_worker_get_task(worker);
        
        // 计算任务开始时间（考虑依赖）
        int64_t start_cycle = max(earliest_dependency_end, worker_free_cycle);
        
        // 模拟执行
        int64_t cycles = pto2_worker_simulate_task(worker, task_id);
        int64_t end_cycle = start_cycle + cycles;
        
        // ========== 关键：立即更新时钟 ==========
        // 其他 worker 可以看到我的预期完成时间
        PTO2_STORE_RELEASE(&ctx->worker_current_cycle[worker->worker_id], end_cycle);
        
        // 报告完成
        pto2_worker_task_complete(worker, task_id, start_cycle, end_cycle);
    }
}
```

### 4. 负载均衡效果

实施后的典型分布：

```
负载均衡后（24 Cube workers, 2048 任务）：

  Cube 0:  86 tasks    Cube 6:  85 tasks    Cube 12: 86 tasks    Cube 18: 85 tasks
  Cube 1:  86 tasks    Cube 7:  85 tasks    Cube 13: 85 tasks    Cube 19: 86 tasks
  Cube 2:  86 tasks    Cube 8:  85 tasks    Cube 14: 85 tasks    Cube 20: 86 tasks
  Cube 3:  86 tasks    Cube 9:  85 tasks    Cube 15: 85 tasks    Cube 21: 85 tasks
  Cube 4:  85 tasks    Cube 10: 85 tasks    Cube 16: 85 tasks    Cube 22: 85 tasks
  Cube 5:  86 tasks    Cube 11: 85 tasks    Cube 17: 85 tasks    Cube 23: 85 tasks

  均值: 85.3  标准差: 0.5  范围: 85-86 ✓
```

### 5. 调试支持

编译时启用 `PTO2_DEBUG_LOAD_BALANCE` 可打印详细的任务分配日志：

```bash
make debug_balance    # 启用调试
make bgemm            # 运行测试

# 输出示例：
[TASK_GET] Worker 3 (clock=500) got task 42
  All CUBE workers:
    Worker  0: clock=    500  WAITING
    Worker  1: clock=    500  WAITING
    Worker  2: clock=    600  WAITING
    Worker  3: clock=    500  GOT_TASK
```

如果出现 `<-- SMALLER CLOCK!` 标记，说明存在违规（不应该发生）。

### 6. 性能考量

- **开销**：每次获取任务时需要遍历同类型所有 worker 的时钟
- **适用场景**：仅仿真模式需要，真实硬件执行有自然延迟
- **可关闭**：通过不定义 `PTO2_DEBUG_LOAD_BALANCE` 减少调试开销

---

## TensorMap 详细设计

TensorMap 的完整设计文档请参阅：[tensormap_design.md](tensormap_design.md)

主要内容包括：
- 哈希策略（仅按 base_ptr 哈希）
- 重叠检测算法（1D 区间 / 边界盒）
- View/Reshape/Transpose 支持
- 惰性失效与链截断优化
- Ring Buffer 池管理
