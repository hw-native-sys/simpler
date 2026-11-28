# Fixed SRAM layout and ownership

WSE configuration fixes each Core's SRAM capacity and runtime reservation before
the device is initialized. The configuration file is a build input; there is no
runtime Python configuration or per-Core allocation API.

For Core `i`, define:

- `capacity[i]`: total modeled SRAM bytes;
- `reserve[i]`: bytes reserved for runtime state;
- `application[i] = capacity[i] - reserve[i]`;
- `offset[i] = sum(application[0:i])`.

The simulator creates one backing allocation. All application windows are
contiguous at its beginning; the private runtime windows follow. Each Core still
owns exactly `application[i] + reserve[i]` bytes. The two windows of one Core need
not be adjacent in the simulator's Host virtual address space. Runtime descriptors
therefore contain separate application and runtime bases.

## Ordinary Worker usage

```python
import json
import torch
from simpler import Worker
from simpler.task_interface import DataType
from simpler_setup.environment import PROJECT_ROOT

spec = json.loads((PROJECT_ROOT / "src/wse/platform/config/default_simulation.json").read_text())
cores = spec["hardware"]["core_count"]
def expand(value):
    return [value] * cores if isinstance(value, int) else value
capacities = expand(spec["hardware"]["sram_bytes_per_core"])
reserves = expand(spec["runtime"]["sram_reserve_bytes_per_core"])
application = [capacity - reserve for capacity, reserve in zip(capacities, reserves)]

worker = Worker(level=2, platform="wsesim", runtime="host_build_graph", device_id=0)
worker.init()
try:
    sram = worker.malloc(sum(application))
    offset = sum(application[:3])  # Core 3's application window
    data = torch.arange(16, dtype=torch.float32)
    worker.copy_to(sram, data, dst_offset=offset)
    tensor = sram.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=offset)
    # Add tensor to ordinary TaskArgs and submit a registered callable.
    # Wait for every run that borrows this Buffer before releasing it.
    worker.free(sram)
finally:
    worker.close()
```

`malloc` accepts exactly the complete application size and permits one live
Buffer claim per device context. It returns the existing fixed backing; it does
not choose a Core or allocate individual objects. A duplicate claim or a request
for another size fails through the existing Worker allocation error path.

Tensor offsets describe the workload's fixed data layout. Each Tensor must fit
inside one Core's application window, although any execution Core may access
that Tensor. Host bulk copies may span adjacent application windows. Copies and
kernel arguments cannot expose runtime-reserved storage.

## Lifetime

The Buffer has the same identity, provenance and run-borrowing rules as other
Worker allocations. `free` refuses an in-flight or retained borrow. A successful
free invalidates that Buffer claim; old handles remain invalid even if a later
claim receives the same address. Free does not zero or release the device's
physical backing. Device shutdown releases the backing after execution drains.

Independent device contexts own independent backings. An address or view from
one Worker cannot be substituted into another Worker's ownership registry.

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
