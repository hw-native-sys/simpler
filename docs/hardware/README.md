# Hardware

Every hardware-level reference lives in this directory. Start with the chip
architecture overview if you need the host/chip model, or jump to the
cache-coherency and MMIO references when working on a data or notification
path.

## Architecture and memory

| Document | What it covers |
| -------- | -------------- |
| [Ascend Chip Architecture](chip-architecture.md) | Host, AICPU, and AICore execution tiers; chip generations; task flow; memory hierarchy |
| [Cache Coherency](cache-coherency.md) | GM visibility across AICore, AICPU, DMA, and SDMA paths, including required barriers and cache operations |
| [AICore Exit Freshness](aicore-exit-freshness.md) | Frozen validation, baselines and measured cost of current-request retirement |

## Control-path investigation

| Document | What it covers |
| -------- | -------------- |
| [MMIO Performance](mmio-performance.md) | AIC control-register memory attributes, measured access costs, concurrency limits, and notification-channel tradeoffs |
| [CANN Source References](cann-source-references.md) | Upstream driver, HCCL, and HCOMM source locations used to investigate hardware and communication behavior |

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [Chip-Level Architecture](../chip-level-arch.md) | The three-program software model layered onto the host, AICPU, and AICore tiers |
| [Task Flow](../task-flow.md) | End-to-end callable, argument, and task flow across the execution tiers |
| [AICore Kernel Programming](../aicore-kernel-programming.md) | Kernel authoring rules for the compute tier described here |
