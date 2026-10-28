# A5 local Host DDR: mapped MTE and URMA qualification

**Date**: 2026-10-25.
**Verdict**: PR1 uses synchronous mapped MTE for this process's Host table on one
A5. Cross-rank URMA and its interface extensions are deferred to a later PR.

## Evidence and scope

Qualification used simpler `2522f02bfead6071a70603d0ccb35431ca03139b`, PTO
`b1d460ee8c001585ede36b68ba3678dc5a226bbf`, and A5DT Ascend950DT_9581 with CANN 9.3.0.
POSIX_SHM mapping succeeds with `aclrtHostRegisterV2(ACL_HOST_REG_MAPPED)` and
`aclrtHostGetDevicePointer`. A mapped-MTE BF16 gather passes first/last/repeated
IDs, invalid/zero requests, output guards, synchronized CPU updates and reuse
(`task_20261024_113346_27326516584`). Device payload is 3584 bytes for a 2 MiB
Host table; the table is not copied H2D.

A single-rank HCCL communicator accepts memory registrations, but the pinned PTO
manager fails initialization: self-rank graph lookup has no links and channel
acquisition fails. Registering memory alone does not establish a usable local
URMA channel. Direct CANN same-device endpoint probes reach READY and return
issue=0, but separate Drain returns 514 for both Host and Device sources
(`task_20261024_113323_272978021792`, `task_20261024_113007_254212931814`).
These failures qualify neither asynchronous local reads nor their teardown.
They do not prove a fundamental silicon limitation or isolate the fault to PTO.

The reference-source audit agrees with the local MTE path:

- [ElasticBuffer](https://github.com/hicann/ops-transformer/blob/6bff82496a8720343fc6144218f059a06a7b9b48/mc2/common/torch_extension/csrc/elastic_buffer.cpp)
  maps local Host backing through ACL and builds a local context without a channel.
- The [fetch kernel](https://github.com/hicann/ops-transformer/blob/6bff82496a8720343fc6144218f059a06a7b9b48/mc2/engram_fetch/op_kernel/arch35/engram_fetch_arch35.h)
  uses MTE copies through UB and completes those transfers before local fetch
  returns. Its other-rank path uses Hcomm URMA.
- The [wait kernel](https://github.com/hicann/ops-transformer/blob/6bff82496a8720343fc6144218f059a06a7b9b48/mc2/engram_fetch_wait/op_kernel/arch35/engram_fetch_wait_arch35.h)
  drains remote channels. A shared fetch/wait interface does not make local
  mapped copies detached URMA operations.

These source-mirror revisions do not identify the exact installed CANN library
build. The libraries were not replaced for qualification.

## PR1 decision and deferred investigation

PR1 provides owned Host backing, binary capability checks, stable Tensor-view
policy, local Device-address resolution and safe use/release. Kernels choose rows
and synchronously finish their MTE transfers before dependent consumers execute.
URMA may be OFF. See the [implementation and acceptance record](../design/a5-host-ddr-pr1-results.md)
and [runnable example](../../examples/a5/tensormap_and_ringbuffer/host_ddr_gather/README.md).

The direct CANN probes, cross-rank comparison, PTO candidate and detailed logs are
preserved with the separate follow-up worktree and review artifacts; they are not
product code in this PR. A future asynchronous local implementation would require
a supported channel configuration with successful transfer and drain, followed by
runtime completion/lifetime integration and hardware overlap evidence. That is
not a prerequisite for the accepted local synchronous PR1 scope.
