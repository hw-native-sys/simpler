# Writing New Tests

[Testing overview](../testing.md)

## New Scene Test

An L3 scene that qualifies default automatic admission declares
`@scene_test(level=3, runtime=..., automatic_admission=True)`. Both pytest and
standalone factories then omit `pipeline_depth`, `launch_depth`,
`pending_run_depth`, and `workspace_budget_bytes`; supplying any of them in its
case config is an error, including `None` or zero. Existing scenes retain their
explicit launch-depth policy. Each automatic class gets a fresh standalone
Worker so cold-start assertions cannot inherit another class's capacity.
This is scene metadata, not a Worker configuration option. Assert the actual
Worker mode and use endpoint capacity/identity evidence in the scene.

Create a `test_*.py` file using the `@scene_test` decorator:

```python
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMyKernel(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aiv/my_kernel.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.ones(1024, dtype=torch.float32)),
            TensorArg("y", torch.zeros(1024, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.y[:] = args.x + 1

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
```

Run it:

```bash
# Via pytest (batch, ChipWorker reuse across tests)
pytest examples tests/st --platform a2a3sim

# Standalone (single case)
python test_my_kernel.py -p a2a3sim

# On hardware (SDMA demos and the isolated fault case use separate tasks)
pytest examples tests/st -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4
pytest examples tests/st -m "sdma and not sdma_fault" --platform a2a3 --device 4-5
pytest examples tests/st -m sdma_fault --platform a2a3 --device 4
```

Key fields:

- `level`: 2 = single ChipWorker, 3 = node-level Worker coordinating ChipWorkers and SubWorkers
- `CASES[].platforms`: which platforms each case supports (sim names end in "sim")
- `runtime`: which runtime to use
- `CALLABLE.orchestration.source` / `CALLABLE.incores[].source`: paths relative to the test file
- `CALLABLE.incores[].extra_include_dirs`: extra `-I` paths for that one kernel,
  on top of the runtime/platform includes every kernel already gets. Entries may
  be relative to the test file, or use `$VAR` — `$ASCEND_HOME_PATH/aarch64-linux/asc/include`
  keeps a CANN-linked kernel off machine-specific paths. They are resolved when
  the kernel is **compiled**, not when its module is imported, so a case that
  declares an SDK dependency is still collectable on sim/macOS runners that do
  not have that SDK. Paths that do not exist are dropped, letting one candidate
  list span SDK layouts; an unset variable raises, and so does a declared set
  where nothing survived (that means the SDK is missing). Used by
  [`qwen3_14b_decode`](../../examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/)
  for its CANN attention extern.
- `CASES[].config` is optional. Ordinary cases should omit `aicpu_thread_num`
  and use the architecture default; set it only when the test specifically
  exercises a thread-count or scheduler-topology behavior.
- The framework owns the `config` namespace. Its keys are
  `aicpu_thread_num`, `runtime_env`, `device_count`, and `num_sub_workers`;
  `runtime_env` accepts `ring_task_window`, `ring_heap`, and `ring_dep_pool`.
  Which of those the runtime under test reads differs: `host_build_graph` reads
  only `ring_task_window` and warns when `ring_heap` is set, since it commits its
  graph heap to the size orchestration measured.
  Unknown keys fail when the `@scene_test` class is imported instead of being
  silently ignored. Case-level extension keys outside `config` remain allowed.

## Kernels That Wait for Siblings (Mix, sync_start)

A Mix task's 1C2V and a sync_start cohort's blocks are co-residency
requirements: the subtasks are meant to be on cores at the same instant, and a
kernel that proves it does so by waiting for its siblings to arrive. Two rules
govern such a kernel, and they pull in opposite directions — **bound the wait
by time, but never decide the verdict by time.**

**Bound the wait by elapsed time, not by a poll count.** A poll costs a
different amount in the CPU-stub build than on silicon, and it costs more again
when the machine is busy, so a count calibrated while the case runs alone can
run out when it runs beside other work. What the case then reports is not its
own give-up value but a device op-execute timeout — the op is killed, and the
failure looks like a runtime hang rather than the missing rendezvous it is.
Convert a millisecond budget with the rate of the clock being read: nanoseconds
on the CPU stub, and the platform counter frequency on device:

```cpp
#ifdef PTO_CPUSTUB_HPP
constexpr uint64_t TICKS_PER_SECOND = 1'000'000'000u;
static uint64_t now_ticks() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
#else
constexpr uint64_t TICKS_PER_SECOND = PLATFORM_PROF_SYS_CNT_FREQ;
static __aicore__ uint64_t now_ticks() { return get_sys_cnt(); }
#endif

constexpr uint64_t WAIT_MS = 2000;
const uint64_t deadline = WAIT_MS * (TICKS_PER_SECOND / 1000u);
const uint64_t started = now_ticks();
while (now_ticks() - started < deadline) { /* ... */ }
```

Note the host-side guard spelling. Select the stub path on
`#ifdef PTO_CPUSTUB_HPP`, which is how the kernels in this tree already do it —
**not** on `#ifndef dcci`. `dcci` is a *function* in both the CCE and
`cpu_stub.hpp` builds, so `#ifndef dcci` is unconditionally true and silently
compiles the whole cache-maintenance path out of the onboard build.

**Write a poison value when the wait expires** so the give-up is visible to
`compute_golden` (see [Output Validation](#output-validation)) rather than
only to someone reading a log. A value no correct run can produce — `-99.0` in
the existing cases — turns a missed rendezvous into a result mismatch naming
the lane.

**Do not judge co-residency by whether the wait expired.** That measures a race
against a wall clock, so the verdict moves with machine load: the same build can
report the defect when the case runs alone and report it fixed when the case
runs after the rest of the suite. Decide from the run's own record instead —
the `assert_cohorts_were_coresident` fixture in
`tests/st/a5/host_build_graph/conftest.py` reads `chip_swimlane_records.json`
and checks that each cohort's subtask intervals intersect (its latest start
precedes its earliest end). Load shifts where the intervals sit, never whether
they intersect. The bounded wait is what keeps a non-co-resident run from
hanging; the interval check is what decides whether it was co-resident.

That fixture is automatically available only below
`tests/st/a5/host_build_graph`. Tests elsewhere must register it in their own
test scope, for example by importing it into their local `conftest.py`:

```python
from tests.st.a5.host_build_graph.conftest import (
    assert_cohorts_were_coresident as assert_cohorts_were_coresident,
)
```

## Output Validation

By default every output tensor is compared against `compute_golden` with
`torch.allclose` at the class `RTOL` / `ATOL`. Two hooks cover the cases that
don't fit:

- **`CASES[].skip_golden`** (defaults to the class attribute `SKIP_GOLDEN`,
  itself `False`) — the case has no host-computable expected output and passes
  by running clean; `compute_golden` is never called for it. Use it when the
  output depends on a shape only the device knows, such as the SPMD grid width
  the runtime resolved. `--skip-golden` on the command line forces this on for
  every case (benchmark mode).
- **`compare_outputs(self, test_args, golden_args, output_names, params)`** —
  override to compare a subset of the outputs, apply a per-tensor tolerance, or
  derive the valid extent from the output itself (a task that writes one slot
  per SPMD block reports the grid width it actually ran with, and only that
  prefix is defined). Call `super().compare_outputs(...)` with the names you
  still want checked the default way.

Prefer a `compare_outputs` override to `skip_golden`: a self-describing output
still gets checked, whereas a skipped case only proves the run didn't crash.

## Case Selection and Manual Cases

`--case` is repeatable and accepts compound forms (useful when a test file declares multiple `SceneTestCase` classes):

| Form | Meaning |
| ---- | ------- |
| `--case Foo` | case named `Foo` in any class |
| `--case ClassA::Foo` | `Foo` in `ClassA` only |
| `--case ClassA::` | all cases in `ClassA` |
| `--case A::x --case B::y` | multiple selectors (repeatable) |

`--manual exclude` (default) skips `manual: True` cases and standalone tests
marked `@pytest.mark.manual`; `--manual include` runs them alongside normal
tests; `--manual only` runs only manual tests. The `only` filter applies to the
entire pytest session, so it also deselects ordinary standalone tests without a
`manual` marker. These compose orthogonally with `--case`: explicit selectors
still respect the manual filter — to run a manual case by name, pass
`--manual include`.

The separate `daily.yml` workflow runs the full corpus with `--manual include`
once per day on A2/A3 and A5, simulation and onboard. Mark a whole standalone
pytest test with `@pytest.mark.manual`; mark only one case in a `SceneTestCase`
by setting `"manual": True` on that `CASES` entry. The main Per-PR scene sweep
excludes those tests, while Daily runs them together with the regular corpus.
Dedicated DFX steps are the exception: they use `--manual include` in normal
Per-PR and Daily jobs, so marking a case under their target path removes only
its duplicate main-sweep execution. A caller selecting `--manual only` keeps
that mode in the DFX steps. The A2/A3 `network1` cases run in the same Daily
workflow through their existing two-machine job, and the DeepSeek smokes run
Per-PR in their own parallel job (`st-deepseek-onboard-a2a3`, `--manual only`
over the two `deepseek_v4_flash_decode` directories).

To move only selected platforms, pass the platform list to the same marker:
use `@pytest.mark.manual(["a2a3sim", "a5sim"])` (or the equivalent
`@pytest.mark.manual(platforms=["a2a3sim", "a5sim"])`) for a standalone test,
or `"manual": ["a2a3sim", "a5sim"]` for a scene-test case. Do not mix the two
standalone marker forms. An onboard execution remains in the default Per-PR
sweep only when that case also declares the corresponding onboard platform;
otherwise the selected architecture becomes Daily-only.

## Sharing an Example Between examples/ and tests/st/

If similar coverage exists in both `examples/` and `tests/st/`, collapse it into a single `test_*.py`: small cases get `platforms: ["a2a3sim", "a2a3"]`; large benchmark cases get `platforms: ["a2a3"], "manual": True`.
