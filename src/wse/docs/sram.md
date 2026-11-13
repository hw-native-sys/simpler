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

Each run uses private Core control records, task tables, Block queues and other
execution metadata placed in the fixed runtime windows. Preparation rejects if
any required section exceeds its Core's reservation. This happens before the
start gate opens. A rejected preparation must release its temporary state and
permit a later valid run.

`wse_sram.json` records application `base`, total `capacity`, `reserve`,
`runtime_base`, runtime usage and the locations of its typed sections. Compare
runtime addresses against `[runtime_base, runtime_base + reserve)`, not against
an assumed suffix of the application's virtual address window.

All runs use the configured Core count. There is no per-run reduced-width
configuration and no memory-owner reassignment when task placement changes.
