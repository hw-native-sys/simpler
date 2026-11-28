# scalar_data — the orchestration reads and writes tensor data itself

Every other example here treats the orchestration as a *scheduler*: it submits
kernels and touches no data. This one has the orchestration poke individual
tensor elements with `get_tensor_data` / `set_tensor_data`, interleaved with
kernel submits, and checks that the runtime orders the two correctly.

Read it for one question: **when the orchestration writes a tensor a kernel is
using, who waits?**

## The answer, and the trap

`set_tensor_data` auto-waits — but only for tensors TensorMap knows about.

| Tensor kind | Added as | Does `set_tensor_data` wait? |
| ----------- | -------- | ---------------------------- |
| runtime-created | anything | **Yes.** TensorMap holds a producer entry, so it waits for the producer to complete (WAW) and for `fanout_refcount` to drain (WAR). |
| external | `add_output` / `add_inout` | **Yes.** These create a TensorMap entry, so the same lookup works. |
| external | `add_input` | **No — and this is a data race.** `add_input` on an external tensor creates no TensorMap entry. `set_tensor_data` finds no producer, so it writes immediately, while the reader kernel may still be running. |

That last row is the whole reason this example exists. There is no error and no
warning; the value is simply overwritten under a running kernel. **On an
external tensor, use `add_inout` when a later `set_tensor_data` must not race
the reader**, even though `add_input` describes the kernel's access accurately.

Step 12 in the orchestration demonstrates the safe form: submit `noop` with
`ext_b` as an output (`noop` reads nothing), then `set_tensor_data(ext_b, 55.0)`
— which now waits, because the output registered an entry.

## The thirteen steps

`kernels/orchestration/scalar_data_orch.cpp` walks through them in order, each
landing a value in `check[0..8]`:

| Steps | What is exercised | Lands in |
| ----- | ----------------- | -------- |
| 1–3 | `get_tensor_data` on a runtime-created tensor at two indices | `check[0]` = 2.0, `check[1]` = 102.0 |
| 4–5 | `alloc_tensors(TensorCreateInfo)` then `set_tensor_data(…, 77.0f)` — seeding a runtime-created output from the orchestration | `check[2]` = 77.0 |
| 6–7 | `add_inout` on the same tensor: the buffer already exists, so the submit only registers a dependency; the value survives | `check[3]` = 77.0 |
| 8 | Plain arithmetic in the orchestration on a value it read back | `check[4]` = 79.0 |
| 9 | `set_tensor_data` → `get_tensor_data` round-trip | `check[5]` = 42.0 |
| 10 | Orchestration → AICore RAW: write 10.0, then a kernel reads it | `check[6]` = 12.0 |
| 11 | WAW + WAR on a runtime-created tensor — `set_tensor_data` waits for both the producer and the consumer | `check[7]` = 88.0 |
| 12 | External WAR done correctly, via `add_inout` | `check[8]` = 55.0 |
| 13 | `result = a + b`, external output through `add_inout` | `result` |

`kernel_noop` exists precisely because several steps need a task that
*registers a dependency* without touching data.

## Cases

One, `default`, for `platforms=["a5"]`. The default AICPU configuration uses
auto selection.

Note the comment on the `check` tensor in `generate_args`: it is **exactly 9
slots**, matching `check[0..8]`. Output-tensor slots are not seeded from the
host, so a tenth slot would read undefined device memory and the golden
comparison would fail nondeterministically.

## Run

```bash
pytest examples/a5/tensormap_and_ringbuffer/scalar_data --platform a5 --manual include --device 0
```

Onboard only. Wrap in `task-submit` on a shared box.

Each step also logs its observed-vs-expected value with `LOG_INFO`, so a
failure tells you which step diverged before you look at `check`.

## See also

- [`docs/war-anti-dependency.md`](../../../../docs/war-anti-dependency.md) — write-after-read hazards and how the runtime orders them
- [`docs/orchestrator.md`](../../../../docs/orchestrator.md) — TensorMap, producer/consumer entries, `fanout_refcount`
