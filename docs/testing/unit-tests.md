# Test suites and unit-test execution

[Testing overview](../testing.md)

## Test Types

### C++ Unit Tests (`tests/ut/cpp/`)

GoogleTest-based tests for shared components (`src/common/task_interface/` and `src/{arch}/runtime/common/`):

- `test_data_type.cpp` — DataType enum, get_element_size(), get_dtype_name()

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build --output-on-failure
```

#### How many times a case is built

Most trees under `src/` are compiled more than once by the product: each arch's
platform tree is its own code, and the arch-independent trees compile into every
runtime's image, resolving their bare-name headers to that runtime's copy. A
case covering such a tree therefore has to be built for each value of the axis
it varies over, and the declaration says which:

| Declaration | Builds |
| ----------- | ------ |
| `simpler_ut_platform_case(... PER_ARCH)` | one per arch |
| `simpler_ut_platform_case(... PER_RUNTIME)` | one per runtime |
| `simpler_ut_platform_case(... PER_ARCH PER_RUNTIME)` | one per pair |
| `simpler_ut_platform_case(... SINGLE)` | one; declares no axis variation, checked through the include paths below |
| `hbg_case(...)` / `tmr_case(...)` | one per arch; `ARCHS` narrows it |

None of these is a default — omitting the axis is a configure error, because a
case built once looks exactly like a case that does not vary, and ctest reports
both as passing.

`tests/lint/check_ut_cpp_axis.py` keeps those declarations honest by
measurement. It preprocesses every target's translation units with `-H`, reads
which files each one opened, and reports a case that opens differing files
along an applicable axis but is not built for every value of it. Code selected
by architecture macros without changing opened paths can escape this check;
a `SINGLE` declaration can therefore omit an architecture without detection.
Known gaps live in its
`WAIVERS` table with a reason, and a waiver whose gap has been closed is an
error, so the table only shrinks.

It needs the compile lines, so configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`:

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build tests/ut/cpp/build --parallel 32
python tests/lint/check_ut_cpp_axis.py
```

CI runs it in the `ut` job, right after ctest, on the Linux leg only — the
answer is a property of the include paths the CMakeLists compose rather than of
the host, and the Linux build graph is the superset (a non-Linux configure drops
`test_thread_scheduling.cpp`, whose wrapped `sched_*` symbols are a GNU ld
feature). It is not a pre-commit hook — it reads a configured build, which
pre-commit does not have.

#### Where the stand-ins come from

`tests/ut/cpp/support/` holds the stand-ins for the CANN platform API, built
into one static archive per arch. A case takes what it needs and says nothing:
an archive member is pulled only to resolve a symbol nothing else defined, so a
case that compiles the real `unified_log_host.cpp`, `device_time.cpp` or
`cache_ops.cpp` simply never pulls that stub.

That is why each file there holds one group of symbols. The linker's unit is
the object file, so a file bundling four groups is pulled for any one of them
and brings the other three along — which is what forced a case wanting one real
implementation to give up the rest.

One thing defeats it: **a weak definition already satisfies the reference, so
the linker never searches the archive.** Product code has such fallbacks on
purpose — the tmr runtime defines `get_sys_cnt_aicpu` weakly as `return 0` so a
host build links without the AICPU side. A test that takes that fallback
compiles, links, and then hangs. Where that applies, the stub is compiled
straight into the library that also compiles the weak definition, and left in
the archive as well: the archive copy is not pulled there, and everything else
still gets it.

`tests/lint/check_ut_cpp_stub_linkage.py` holds that line. For every symbol the
archive defines strongly, it fails if any test binary resolved it to a weak
definition:

```bash
python tests/lint/check_ut_cpp_stub_linkage.py
```

### Python Unit Tests (`tests/ut/`)

Tests for the nanobind extension and the Python build pipeline:

- `test_task_interface.py` — DataType, Tensor, ChipStorageTaskArgs, torch integration
- `test_runtime_builder.py` — RuntimeBuilder discovery, error handling, build logic (mocked), and real compilation integration tests

```bash
# No-hardware runner (hw tests auto-skip, no-hw tests run)
pytest tests/ut

# a2a3 hardware runner (no-hw tests skip, hw + a2a3-specific tests run)
pytest tests/ut --platform a2a3
```

### Examples (`examples/{arch}/`)

Small, fast examples that run on both simulation and real hardware. Organized by runtime:

- `host_build_graph/` — HBG examples
- `tensormap_and_ringbuffer/` — TMR examples

Each example has a `golden.py` with `generate_inputs()` and `compute_golden()` for result validation.

### Device Scene Tests (`tests/st/{arch}/`)

Hardware-only scene tests for large-scale and feature-rich scenarios that are too slow or unsupported on simulation. Organized by runtime. Same structure as examples but focused on testing specific runtime behaviors and edge cases.

| Attribute | `examples/` | `tests/st/` |
| --------- | ----------- | ----------- |
| Runs on sim | Yes | No |
| Runs on device | Yes | Yes |
| Scale | Small, fast | Large, thorough |
| Purpose | Examples + basic regression | Deep functionality/performance |

## New C++ Unit Test

Put the file in the mirror of the tree it covers and stop: the directory gives
it a target, a name, an include contract, and how many times it is built. A
case that needs nothing beyond that is never named in a CMakeLists.

[Adding a C++ unit test](adding-a-cpp-unit-test.md) covers the rest —
which directory, the per-directory macros and their keywords, how many builds
a case gets and how to have that measured rather than guessed, and where the
CANN stand-ins come from.

### C++ hardware tests needing NPU devices

Tests that need specific NPU devices use CTest's [resource allocation](https://cmake.org/cmake/help/latest/prop_test/RESOURCE_GROUPS.html). The label and the devices are keywords on the case, usually already carried by the directory's macro:

```cmake
simpler_ut_add_target(NAME my_hw_test
    SOURCES my_hw_test.cpp
    LABEL requires_hardware_a2a3
    RESOURCES "2,npus:1")    # Two distinct devices when each NPU has one slot
```

The CI generates a resource spec file from `${DEVICE_RANGE}` and passes it to ctest:

```bash
# Generate resource spec (CI does this automatically)
python3 -c "
import json
npus = [{'id': str(i), 'slots': 1} for i in range(8)]
json.dump({'version': {'major': 1, 'minor': 0}, 'local': [{'npus': npus}]},
          open('resources.json', 'w'))
"

# Run with resource allocation — CTest assigns devices, no oversubscription
ctest --test-dir tests/ut/cpp/build \
    -L "^requires_hardware(_a2a3)?$" \
    --resource-spec-file "$(pwd)/resources.json" \
    -j$(nproc) --output-on-failure
```

CTest passes allocated device ids via environment variables:

- `CTEST_RESOURCE_GROUP_COUNT` — number of groups
- `CTEST_RESOURCE_GROUP_<n>_NPUS` — `"id:<device_id>,slots:1"` per group

Tests read these to determine which devices to use. See `test_comm_lifecycle.cpp::read_ctest_devices()` for the parsing pattern.
