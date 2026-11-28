# Fixed device memory

WSE establishes its memory windows when the Worker initializes. The platform
provides one local application window per Core and an optional global memory
window per chip. Addresses and lengths belong to the
device context and stay valid until it closes.

The simulation configures each Core's total SRAM capacity and runtime
reservation in bytes. In the simulated address space each Core occupies one
contiguous block, starting with Core 0. Its application area starts at the
beginning of the block and its private runtime area occupies the tail.
A Core's block starts after the **full capacities** of preceding Cores.
Applications query each window so their layouts also work on platforms that
map the windows at different device addresses.

The simulation spec sets the chip-global window capacity with
`hardware.global_memory_bytes`. A zero capacity disables this window.
Host copies can populate both local and global windows. Moving data from
global memory to a Core during execution requires a separate transfer.

## Worker usage

```python
import torch
from simpler import Worker
from simpler.task_interface import DataType

worker = Worker(level=2, platform="wsesim", runtime="host_build_graph", device_id=0)
worker.init()
try:
    core3 = worker.get_base_addr(coreid=3)
    data = torch.arange(16, dtype=torch.float32)
    address = core3 + 128
    worker.copy_to(address, data)
    tensor = address.tensor(shapes=(16,), dtype=DataType.FLOAT32)

    global_memory = worker.get_base_addr(memory="global")
    worker.copy_to(global_memory + 128, data)
finally:
    worker.close()
```

`get_base_addr(coreid=0)` also selects local SRAM. The explicit form is
`get_base_addr(memory="local", coreid=0)`. Global memory has chip scope and
takes no Core ID. When global memory is disabled, querying it reports that
the window is unavailable.

Address arithmetic and copy offsets are in bytes. A derived address retains
its original window identity and available length. Copies must fit in that
window, including when the adjacent address belongs to another Core.
Application copies cannot write the runtime reservation. Layouts that need
object-level overlap checks must also track the offset and length of each
object placed within a window.

## Lifetime

The Worker owns the fixed windows. `malloc` and `free` are unsupported on
this platform. Window identity and the existing run-borrowing rules protect
in-flight device work. Closing the Worker invalidates its addresses after
execution drains. Device contexts have independent windows and ownership
registries.

## Runtime reservation

Each Core stores its own `core_control` record. An idle Core needs only this
record. A Core with work also stores its `block_queue`, the `local_task` record
for each distinct task it executes, and that task's `dependencies`. Each Block's
Tensor and scalar descriptors reside with the Core that executes it. Task
records contain references to their predecessors' completion signals.

An SS stores its `task_signal` on its submitted Core. For a Group, the Core
assigned to logical Block 0 holds one signal shared by all members. That signal
counts unfinished Blocks and, for `sync_start`, members yet to arrive. The
last Block to finish publishes task completion, whichever Core executes it.

Every Core executing a kernel holds a `region_map` of the fixed SRAM windows
and the `allocations` admitted for this run's Tensor arguments. A kernel can
resolve an address in another Core's admitted window, including an address
passed as a scalar. Preparing these lookup tables does not copy Tensor data.

Set `CallConfig.output_prefix` to obtain `wse_sram.json`. In `sections`, each
entry gives its `name`, numeric `kind`, owning Core, address and byte length.
In `cores`, compare each `runtime_used` with its `reserve`. The valid runtime
address interval is `[runtime_base, runtime_base + reserve)`.
The `global_memory` entry gives the chip-global base and capacity. The simulator reserves
virtual memory at init and the OS backs pages as they are touched. The
committed-memory count describes modeled capacity, and `application_used`
includes the whole registered local application window. Tracking written bytes
requires observing individual transfers and stores.

As an example, place one SS on Core 6, its synchronized Group on Cores 2 and
6, and a final SS on Core 3. Each of the three participating Cores holds its
own task records. Core 2 holds the Group signal when Block 0 is assigned there.
Adding SS tasks on Core 6 grows Core 6's task state. Core 0 remains at its fixed
control-record size. The same rules apply when the caller assigns work to Core 0.

Admission checks each Core against its own reservation before GO. On overflow,
`wse_prepare_error.txt` identifies the Core and section, the requested bytes,
remaining bytes and configured reservation. Releasing the rejected run's state
allows a later valid submission on the same Worker.

All runs use the configured Core count. Tensor ownership remains fixed when
task placement changes.
