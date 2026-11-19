# Scene-test device-memory fixtures

[Testing overview](../testing.md)

<!-- markdownlint-disable-next-line MD033 -->
<a id="explicit-l2-child-memory"></a>

This page describes fixture ownership and validation in `SceneTestCase` and
`TaskArgsBuilder`. For application allocation and transfer APIs, see the
[Buffer memory model](../buffer-abi.md#allocating-buffers). For host graph
construction, see [HBG host data access](../hbg-host-data-access.md).

`TensorArg(name, value, child_memory=True)` keeps a case-owned device buffer
across all rounds, including `--rounds 1`. `TaskArgsBuilder.add_tensor` accepts
the same keyword. The default remains host memory; host-memory IN and INOUT
tensors are copied in on every round, while pure OUT buffers skip the copy.
Child-memory IN and INOUT fixtures are uploaded once at setup.

| Declaration / direction | Setup | Between rounds | Validation |
| ----------------------- | ----- | -------------- | ---------- |
| Host memory (default) | Existing path | Restore OUT/INOUT host fixtures | Existing per-round copy-back |
| Child-memory IN | Allocate and upload once | Keep device address and input contents | No output readback |
| Child-memory OUT | Allocate without upload | Keep device contents; the case must define all compared elements | Final readback |
| Child-memory INOUT | Allocate and upload once | Keep device state | Final readback |

Golden evaluation follows the same state evolution: child-memory outputs retain
state and host-memory outputs reset. Cases with child-memory outputs compare after
the final round; other cases continue comparing every round.

For HBG, keep controls that the host orchestration reads or writes in separate
host fixtures with `transfer=TensorTransfer.NONE`. A `child_memory=True`
fixture supplies a device argument; host scalar access does not map or copy it
implicitly. The [runtime contract](../hbg-host-data-access.md#location-and-transfer)
defines the supported combinations independently of the test runner.

Declarations currently require L2, contiguous CPU fixtures, and non-overlapping
storage. Empty fixtures allocate no device buffer; the existing transport
still rejects zero-shaped Tensor arguments. Clone and rehost operations preserve
declaration metadata. An in-repo standalone driver that owns its own `Worker`
reuses the same owner, `simpler_setup.scene_test.ChildMemoryTaskArgs`, as a
context manager, adding one fixture at a time so each large fixture can be
released before the next is materialized. It is a scene-test helper, not part of
the `simpler_setup` public surface — it builds `TaskArgs` and copies back through
a `TaskArgsBuilder`, so it has no meaning outside this corpus.

The HBG `paged_attention_unroll_manual_scope` examples and `paged_attention`
scene tests include matched `HostStaged` and `ChildMemory` cases. Both provide
explicit host-only controls for orchestration; the device fixtures exercise
allocation reuse and round-state validation.
