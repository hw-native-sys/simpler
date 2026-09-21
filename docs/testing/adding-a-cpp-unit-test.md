# Adding a C++ unit test

The short version: **put the file where the tree it covers is, and stop.**
The directory it lands in gives it a target, a name, an include contract, and
how many times it is built. A case that needs nothing else is never mentioned
in a CMakeLists at all.

This page is what to do when it does need something else.

## 1. Choose the directory

`tests/ut/cpp/` mirrors `src/`. A case goes in the mirror of the tree it
covers, and that placement is not cosmetic — it is what decides the case's
include path and how many builds it gets.

| The case covers | It goes in |
| --------------- | ---------- |
| `src/common/platform/...` | `tests/ut/cpp/common/platform/` |
| `src/common/host_build_graph/...` | `tests/ut/cpp/common/host_build_graph/` |
| `src/a5/runtime/host_build_graph/...` | `tests/ut/cpp/a5/runtime/host_build_graph/` |
| `src/a2a3/platform/...` | `tests/ut/cpp/a2a3/platform/` |
| the part of `src/a5/runtime/` every runtime carries | `tests/ut/cpp/a5/runtime/` |

Getting this wrong usually fails to compile rather than passing quietly: the
directory picks the include path, so a host_build_graph case filed under
`tensormap_and_ringbuffer/` resolves its bare-name headers — `runtime.h`,
`task_id.h`, `types.h`, which exist in both trees — to the other runtime's
copies, and the API it calls is not there.

Name the file `test_<what it covers>.cpp`. It must register at least one
`TEST` / `TEST_F`; `tests/lint/check_ut_cpp_case_naming.py` fails a
`test_*.cpp` that registers none, because the glob would otherwise build it as
a target that passes vacuously.

The directory has to be one of the mirrors that ends in
`simpler_ut_glob_cases` — a `test_*.cpp` at an intermediate level, or in a
directory left over from an older layout, is built by nothing at all. That
failure is silent in every direction: the tree configures, the build succeeds,
and ctest reports the cases that registered, so the suite is green and smaller
by exactly the case nobody can see is missing. `simpler_ut_assert_no_orphan_cases()`
at the end of the root `CMakeLists.txt` fails the configure and names the file
instead.

## 2. The ordinary case: write nothing

Every directory ends in `simpler_ut_glob_cases(<its macro>)`. A `test_*.cpp`
no declaration above has claimed is compiled and registered with that
directory's shape. Drop the file in, and it is a test.

That covers the majority. What the directory's shape is, and what its cases
are therefore built against, is in the comment at the top of its CMakeLists.

## 3. When the case needs more

Declare it **above** the glob, which takes it out of the sweep — the
declaration is the record, there is no separate exclusion list.

Use the directory's own macro. Every directory has one, named after it:

| Directory | Macro |
| --------- | ----- |
| `common/platform/` | `platform_case` / `platform_single_case` / `platform_cases` |
| `common/host_build_graph/` | `hbg_case` / `hbg_cases` |
| `common/tensormap_and_ringbuffer/` | `tmr_case` |
| `common/{log,utils,worker,task_interface,runtime_status,hierarchical,platform_comm}/` | `<dir>_case` |
| `{a2a3,a5}/platform/` | `<arch>_platform_case` |
| `{a2a3,a5}/runtime/` | `<arch>_runtime_case` |
| `{a2a3,a5}/runtime/{host_build_graph,tensormap_and_ringbuffer}/` | `<arch>_<tag>_case` |

The keywords they forward to `simpler_ut_add_target` (see
`tests/ut/cpp/cmake/ut_test.cmake` for the full list):

| Keyword | For |
| ------- | --- |
| `SOURCES` | the production sources the case compiles in — **what it covers** |
| `INCLUDES` | directories beyond the contract |
| `INCLUDES_BEFORE` | directories that must precede even the gtest prefix |
| `DEFINES` | compile definitions |
| `COMPILE_OPTIONS` / `LINK_OPTIONS` | extra compile and link flags, for a case whose assertion is about what the build produces |
| `LINK` / `DEPENDS` | libraries, and targets to build first (a case that dlopens one) |
| `TIMEOUT` / `LABEL` / `RESOURCES` | ctest properties |
| `A2A3_SOURCES` / `A5_SOURCES` | sources only one arch's build needs |

`@arch@` and `@runtime@` inside `SOURCES` and `INCLUDES` expand per build.

**Name only what the case covers.** The CANN platform stand-ins are not listed
by anyone — see [the support archive](#5-the-support-archive).

### Several cases of one kind

Where a group differs only in which file it is, declare them together:

```cmake
platform_cases(
    test_profiler_device_engine.cpp
    test_worker_chip_message_queue.cpp
    test_worker_chip_orch_endpoint.cpp
    test_region_instance_view.cpp
    PER_ARCH)
```

Four cases, one line of shape between them, and nothing about the CANN
stand-ins they all need — those come from the archive.

This asserts they are the same *kind*, not that their arguments match today.
Two cases can both be `PER_ARCH` for unrelated reasons — one because
`kernel_args.h` differs between the arches, another because
`platform_config.h` does. Group those and the file claims a relationship that
is not there; declare them apart and each keeps its own reason.

## 4. How many times the case is built

Under `common/platform/`, this is the one thing a declaration **must** state.
`simpler_ut_platform_case` has no default: omit all three and configure fails.
Nor does the directory's glob supply one — an undeclared `test_*.cpp` there is
refused by name rather than swept up as `SINGLE`, which is the only way the
requirement reaches a file nobody declared.

| Keyword | Means |
| ------- | ----- |
| `PER_ARCH` | the case reaches `src/<arch>/platform`, whose two copies are not the same file |
| `PER_RUNTIME` | the case reaches a source that ships inside every runtime's image and resolves its bare-name headers to that runtime's copy |
| `SINGLE` | it reaches neither, so one build is every build |

`PER_ARCH` and `PER_RUNTIME` combine; neither subsumes the other.

There is no default because a silent one turns an unjudged case into a single
build that afterwards looks deliberate — which is how ten cases ended up
covering one configuration and reporting on all.

Every other directory does have a default, and is entitled to one: its own
position fixes the answer. A case under `a2a3/platform/` is a2a3's because it
sits there, one under `common/host_build_graph/` covers that runtime, and each
directory's glob hands its cases exactly that shape. `common/platform/` is the
only directory where both axes are open at once, which is why it is the only
one that refuses to guess.

### When a pytest drives the binary

A case ctest is not the only reader of takes `PYUT_EXPORT <handle>`. Each build
is then recorded in `<build>/pyut_binaries.json` as `{handle, arch, runtime,
path}`, written from `$<TARGET_FILE:...>`:

```cmake
platform_case(test_chip_swimlane_collector.cpp
    PER_ARCH PER_RUNTIME TIER aicpu
    PYUT_EXPORT chip_swimlane_collector
    SOURCES ${CHIP_SWIMLANE_CASE_SOURCES})
```

The pytest looks the binary up by `(handle, arch, runtime)` and never spells a
target name. That matters more than it looks: target names are assembled from
the case file and the combination, so a name spelled on the Python side is a
second implementation of that rule with nothing tying it to the first — and
when the two drift, the pytest finds no file, which is indistinguishable from
an unbuilt tree and skips. The manifest also separates the two questions a
reader has: it lists what configure intended to build, so a missing entry is a
declaration that lost a combination (fail), while a listed path that does not
exist yet is an unbuilt tree (skip).

**You do not have to guess.** Build, then ask:

```bash
python tests/lint/check_ut_cpp_axis.py
```

It preprocesses every translation unit, reads which trees the compiler opened,
and compares the combinations built against the ones the code varies over. A
case short one combination is named, with the missing cell. CI runs it, so
guessing wrong is caught rather than shipped.

Where a combination is deliberately not built, add it to that script's
`WAIVERS` with the reason. Most entries are combinations that cannot exist —
two are there because a5's tensormap_and_ringbuffer Runtime has no
device-initialized tail to assert about. One is different in kind and its
reason says so: `test_chip_swimlane_run_export` compares against a golden
artifact captured on a2a3, carrying that capture's own `clock_freq_hz` and
`platform`, so an a5 build has nothing correct to compare against until an a5
golden is captured. That is a gap to close, not an absent combination, and
writing which one it is keeps the table honest — a reader must be able to tell
"impossible" from "not done yet".

Elsewhere the axes come from the directory: a case under `a2a3/` is that
arch's, one under `common/tensormap_and_ringbuffer/` is that runtime's and is
built for both arches.

## 5. The support archive

`tests/ut/cpp/support/` holds the stand-ins for the CANN platform API —
`unified_log_*`, `get_sys_cnt_aicpu`, the cache-maintenance no-ops,
`get_reg_ptr`, `assert_impl` — as one static archive per arch, linked by every
directory's macro.

**A case says nothing about them.** An archive member is pulled only to resolve
a symbol nothing else defined, so:

- need the stubs → they are already there;
- want the real thing instead → put it in `SOURCES`
  (`${HOST_LOG_TEST_SOURCES}`, `sim/aicpu/device_time.cpp`,
  `sim/aicpu/cache_ops.cpp`) and the stub is simply not pulled;
- want your own behaviour → define the symbol in the test source.
  `assert_impl` and `get_stacktrace` are weak for exactly this.

Each file there holds one group of symbols, because the linker's unit is the
object file: a file bundling four groups is pulled for any one of them and
brings the other three along.

One thing defeats this, and it is worth knowing before you debug it: **a weak
definition already satisfies the reference, so the linker never searches the
archive.** Product code has such fallbacks — the tmr runtime defines
`get_sys_cnt_aicpu` weakly as `return 0` so a host build links without the
AICPU side. A case that takes the fallback compiles, links, and then hangs: the
runtime's 500 ms reclaim backstop counts ticks that never advance. Where that
applies the stub is compiled straight into the library that also compiles the
weak definition. `tests/lint/check_ut_cpp_stub_linkage.py` holds the line and
CI runs it.

### Writing a new stand-in

Nothing in the tree is named `stubs/` any more: a support file goes under a
`support/` directory, and **which one says who may reach it**.

| Put it in | When | Reached as |
| --------- | ---- | ---------- |
| `tests/ut/cpp/support/` | every directory may want it | already linked, via the archive |
| `tests/ut/cpp/common/support/` | shared inside `common/` | named by path from `${CMAKE_SOURCE_DIR}` |
| `<the case's directory>/support/` | only that directory's cases | named by path from `${CMAKE_CURRENT_SOURCE_DIR}` |

Do not name it `test_*.cpp`. The glob reads that prefix as "this is a case",
and `tests/lint/check_ut_cpp_case_naming.py` fails a `test_*.cpp` that
registers no gtest — which a stub does not.

**One group of symbols per file.** This is the rule the archive rests on:
a member is pulled for any symbol in it and drags the rest along, so a file
holding four groups cannot be partially replaced. `support/` therefore keeps
the logger, the device clock, cache maintenance, the register shim, the assert
hook and the weak-link placeholders in six files rather than one — which is
what lets a case take the real logger and keep the other stand-ins.

**Then decide whether the archive can carry it**, by what the product does
with that symbol:

| Product code | Where the stub goes |
| ------------ | ------------------- |
| defines it nowhere | `tests/ut/cpp/support/`, in the archive — pulled on demand |
| defines it **weakly** | archive **and** compiled into the library that also compiles the weak definition; the weak one would otherwise win silently |
| defines it strongly, and some case wants that | archive — the strong definition wins wherever it is linked |

Check with `grep -rn "weak.*<symbol>" src/` before assuming the first row.
Exactly one symbol in the archive today — `get_sys_cnt_aicpu` — is in the
second row, and finding that out cost ten hanging tests.

When the stand-in is only right for one arch, hand it over on that arch's
keyword rather than branching:

```cmake
hbg_case(test_hbg_scheduler_drain.cpp
    SOURCES      ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
    A2A3_SOURCES ${SIMPLER_SRC}/a2a3/platform/sim/aicpu/inner_platform_regs.cpp
    A5_SOURCES   ${CMAKE_CURRENT_SOURCE_DIR}/support/scheduler_drain_a5_stubs.cpp)
```

### A stand-in the case loads at run time

Some cases need a whole shared object to `dlopen` — a fake host runtime, a
module that compiles the production logger. Build it as a `SHARED` library
from `support/`, and let the case find it by generator expression rather than
by a guessed path:

```cmake
add_library(test_device_fault_client_a SHARED
    ${CMAKE_SOURCE_DIR}/common/support/device_fault_client.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/device_fault_monitor_binding.cpp)

platform_single_case(test_device_fault_cross_dso.cpp
    DEFINES TEST_DEVICE_FAULT_CLIENT_A_PATH="$<TARGET_FILE:test_device_fault_client_a>"
    LINK    ${CMAKE_DL_LIBS}
    DEPENDS test_device_fault_client_a)
```

`DEPENDS` is what makes the library exist before the case runs;
`$<TARGET_FILE:...>` is what keeps the path right without the case knowing the
build layout.

## 6. Hardware cases

A case needing a device declares its label and its devices through the same
factory:

```cmake
platform_comm_case(test_comm_lifecycle.cpp
    DEFINES PTO_HOST_RUNTIME_LIB_PATH="${HOST_RUNTIME_LIB}")
```

The directory's macro already carries `LABEL requires_hardware_a2a3` and
`RESOURCES "2,npus:1"`. `common/platform_comm/` is entered only under
`-DSIMPLER_ENABLE_HARDWARE_TESTS=ON`, which is what lets everything in it link
CANN freely without breaking the no-hardware build.

CTest hands the allocated device ids to the test through
`CTEST_RESOURCE_GROUP_COUNT` and `CTEST_RESOURCE_GROUP_<n>_NPUS`; see
`test_comm_lifecycle.cpp::read_ctest_devices()` for the parsing.

## 7. If you added a runtime

Configure will stop you:

```text
Runtime 'foo' has a build_config.py but no unit tests.
Create tests/ut/cpp/common/foo/ for the parts both arches share, and
tests/ut/cpp/{a2a3,a5}/runtime/foo/ for each arch's own, then add_subdirectory
them. Each needs the object library its cases link, the way
common/tensormap_and_ringbuffer/ declares <arch>_tmr_objs.
```

This is on purpose. Discovery gives the new runtime the cases that repeat per
runtime — the platform tree's — for free, and those start building for it
immediately. Without the mirror directory they cannot compile, because they
resolve `task_id.h` into `src/common/foo/`; before this check the first sign of
that was a `fatal error: task_id.h: No such file or directory` reported against
a platform header nobody had touched.

What the new directories need is what `common/tensormap_and_ringbuffer/` and
`{a2a3,a5}/runtime/tensormap_and_ringbuffer/` already show: an object library
per arch, a `<rt>_case` macro over `simpler_ut_runtime_case`, and a closing
`simpler_ut_glob_cases`. Give the runtime a short tag in
`simpler_runtime_tag()` if you want its targets named like `hbg` and `tmr`;
without one they carry the full name, which is fine.

## 8. Before you push

```bash
rm -rf tests/ut/cpp/build
cmake -B tests/ut/cpp/build -S tests/ut/cpp -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build tests/ut/cpp/build --parallel 8
ctest --test-dir tests/ut/cpp/build -LE requires_hardware -j8 --output-on-failure
python tests/lint/check_ut_cpp_axis.py
python tests/lint/check_ut_cpp_stub_linkage.py
```

A clean configure matters: the glob is `CONFIGURE_DEPENDS`, but an object
library's source list is not re-read on every build.

Two things these catch that a passing ctest does not: a case built for one
configuration and reporting on all, and a stub the archive lost to a weak
fallback. Both are green-looking failures, which is why they are checks rather
than conventions.

## What not to do

- **Do not write `add_executable`.** There are none left. A bare target also
  fails to register the case file it consumes, so the directory's glob builds
  a second target from the same source.
- **Do not add a global `simpler_ut_*` helper named after an (arch, runtime)
  pair.** Sixteen of those are gone; a directory's own `hbg_case` /
  `a2a3_platform_case` macro is the shape it hands its cases, and the keywords
  on `simpler_ut_add_target` are how one case differs from it.
- **Do not reach for `NO_OBJS`** before checking whether the colliding symbol
  should be weak instead. It drops the whole object library to avoid one
  symbol. Three uses remain, each with its reason written beside it.
- **Do not hand-write an include path list.** `cmake/contract.cmake` is the
  only place that says what an `(arch, runtime, tier)` may reach.
