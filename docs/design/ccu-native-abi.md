# Native CCU module and three-Event contract

A `ChipCallable` packages native HCOMM CCU definitions alongside its AIV
functions. The module owns the communication algorithm and finite control flow;
the runtime owns registration, per-execution parameters, Events and completion.
The C/POD interface is in `src/common/platform_comm/ccu/ccu_native_abi.h`.
Modules, bindings, runtime libraries and device kernels must be rebuilt together.

## Building a Host module

Use the same `KernelCompiler` instance for AICore kernels, orchestration and
native CCU modules. `CCUToolchain` lives beside `CCECToolchain` in
`simpler_setup.toolchain`; `KernelCompiler.ccu` shares the selected Host G++
compiler with Host orchestration, including the sanitizer compiler selection.
Native CCU builds require `platform="a5"`. Other platforms keep `ccu=None`
and reject CCU queries/builds. Constructing an A5 compiler does not discover
CCU dependencies until a CCU query or build needs them.

Set `ASCEND_HOME_PATH` to the CANN installation for the Linux Host that will
load the module. Discovery searches both the installation root and the current
Host's architecture subdirectory for each required header and library. A build
or cache-token query may fetch the pinned PTO-ISA checkout if a usable local
checkout is unavailable.

```python
from simpler_setup.kernel_compiler import KernelCompiler

compiler = KernelCompiler(platform="a5")
module_bytes = compiler.compile_ccu("native_ccu.cpp")
# Use compiler.compile_incore(...) and compiler.compile_orchestration(...)
# for the other parts of this callable.
# Pass ccu_modules=[module_bytes] to ChipCallable.build().
```

`compile_ccu()` generates `ccu_manifest_identity.h`, compiles and links the
sources, and returns the complete Host shared-library bytes. Each invocation
uses a private temporary directory, removed on success or failure. Optional
`build_dir` selects its parent directory, which must already exist.
`extra_sources`, `extra_include_dirs`, `extra_compile_flags` and
`extra_link_flags` support modules with multiple translation units and custom
build inputs. Queries and builds resolve relative paths and options from the
caller's working directory. CCU modules retain their full `.so`; AICore kernels retain the
existing ELF `.text` extraction into `CoreCallable` children.

The sources still provide the native kernel entries and the manifest factory.
Include `ccu_manifest_identity.h` and copy its two arrays into the manifest:

```cpp
std::memcpy(manifest.sdk_build_id, simpler::ccu::sdk_build_id, sizeof(manifest.sdk_build_id));
std::memcpy(manifest.pto_isa_build_id, simpler::ccu::pto_isa_build_id, sizeof(manifest.pto_isa_build_id));
```

### Caller-owned compilation and caching

Code generators such as PyPTO can keep their compiler invocation and cache.
`compiler.get_ccu_include_dirs()` and `compiler.get_ccu_library_paths()` expose
paths, while `compiler.ccu` exposes the base compiler and flags, corresponding
to the AICore toolchain interface. `compiler.ccu_compile_cache_token()` returns
the effective ordered inputs, including selected sanitizer flags, working
directory, manifest identity header and content fingerprints. Pass the same extra flags and include
directories to the token query and to `compile_ccu()` if using simpler to build.

For an external build, use the effective inputs from that token:

```python
from pathlib import Path
import subprocess

from simpler_setup.kernel_compiler import KernelCompiler

compiler = KernelCompiler(platform="a5")
config = compiler.ccu_compile_cache_token()
build = Path("build/ccu").resolve()
build.mkdir(parents=True, exist_ok=True)
(build / "ccu_manifest_identity.h").write_text(config["manifest_identity_header"], encoding="ascii")
source = str(Path("native_ccu.cpp").resolve())
compile_args = [
    config["cxx_path"],
    *config["compile_flags"],
    f"-I{build}",
    *(f"-I{path}" for path in (*config["include_dirs"], *config["extra_include_dirs"])),
]
# Before each cache lookup, scan every source, including system headers.
subprocess.run(
    [*compile_args, source, *config["link_flags"], "-M", "-MF", str(build / "native_ccu.d"), "-MT", "native_ccu"],
    check=True,
    cwd=config["working_directory"],
)
# On a cache miss:
subprocess.run(
    [*compile_args, source, *config["link_flags"], *config["library_paths"], "-o", str(build / "native_ccu.so")],
    check=True,
    cwd=config["working_directory"],
)
```

The cache token is not a complete module cache key. Combine it with the paths
and content digests of every source and prerequisite in the dependency rules,
plus other caller-owned build inputs. Run a fresh scan using the same flags,
include paths and working directory as the build. Use `-M`, since `-MM` and
`-MMD` omit system headers such as the C++ standard library. This catches
changes to header contents and header resolution. `dependency_digests` lists
the files already fingerprinted by the query, including the compiler, SDK,
ABI and selected runtime libraries. Individual UBSan checks such as
`-fsanitize=signed-integer-overflow` also fingerprint their UBSan runtime.
The query links an empty shared module with the build's ordered flags and SDK
libraries, then fingerprints the inputs reported by GNU ld (bfd), including
archives selected through custom `-L` directories and files referenced by
linker scripts. Other linkers are rejected because their trace may omit archives
from which the empty probe extracts no members. The probe does not execute the
module; source and other caller-owned inputs still belong in the module key.
The probe permits unresolved symbols and omits `--no-undefined-version`, since
it has no user symbol definitions. Real module builds keep the original flags,
including version-script export checks. Response files retain their original
arguments in the returned configuration and real build command.

Pass `library_paths` directly to the linker to use the exact SDK libraries
fingerprinted by the query. CPU options such as `-march=native`, `-mcpu=native`
and `-mtune=native` are rejected; use an explicit CPU target when needed.
Deploy the complete shared-library bytes in callable order. At load time the
dynamic loader must be able to find the matching CANN/HCOMM libraries, for
example through the CANN environment setup on the deployment Host.

## Packaging and validation

A module exports one entry, which must not throw:

```c
const SimplerCcuModuleManifest *simpler_ccu_module_manifest(void);
```

The returned manifest is immutable and contiguous. Kernel and string tables
use checked byte offsets from its base. The loader checks lengths, integer
overflow, table overlap, entry symbols, parameter alignment, capabilities and
SDK/PTO-ISA build identities. The generated header and runtime validation use
the same SDK and pinned PTO-ISA header digests.
The initial interface has no historical layout negotiation or compatibility
branches. Structure and record sizes validate the single expected layout.

The package preserves exact opaque module bytes. The native loader opens them
using an executable anonymous Linux memfd (or a private unlinked file in
`TMPDIR`) and `RTLD_NOW | RTLD_LOCAL`. Its file descriptor
remains alive until after `dlclose`, preventing reuse of a live `/proc/self/fd`
path. Cache identity includes exact bytes, manifest and build identities; a
candidate is validated before sharing an existing owner.

Kernel records concatenate in module-package order and record order. Their
zero-based position is the callable-local `ccu_id`. Each kernel has one local
control resource set. Module-local kernel IDs are metadata, unique within each
module, and do not select device resources. There is no user group,
mission-index translation or separate instance table.

## Registration topology

Hardware registration and domain allocation have separate explicit APIs:
`Worker.register(ChipCallable)` stores definitions without acquiring CCU hardware
resources. `Worker.register_ccu_callable(handle)` is the explicit hardware
registration step; it uses the established L3/L2 communication context and does
not require a communication window. A domain declaring
`workspace_engines=("ccu",)` only enables W and its access descriptors.
Registration identity and ownership belong to the callable and the complete
L3/L2 communication context, independently of W. All L3 chip workers must have
the definition installed.
No mission is launched by domain creation or registration.

Every L3 chip worker prepares channels to all other workers in that context.
Changing W's members, allocation or lifetime does not create a new registration.
For example, the same callable serves A={0,1} and B={0,2} with one registration
at rank 0; releasing A leaves B's registration intact. Each domain still must
explicitly enable the CCU workspace. An unenabled domain is rejected before
run preparation or launch. The first version
supports one selected CCU domain per orchestration run. Its context must appear
exactly once in each submitted task's scalar arguments. Tensor domain references
are enrolled before selecting that context. General multi-domain execution in a
single run is rejected.

This whole-context topology implements the author's requested decoupling and
[YunjiQin's proposal](https://github.com/hw-native-sys/simpler/pull/2986#issuecomment-6769250443).
It extends the domain-scoped topology accepted by
[Chao's clarification](https://github.com/hw-native-sys/simpler/pull/2986#issuecomment-6790302878),
which gates backend initialization on domain enablement. Independent explicit
registration can initialize the backend before W exists, so that comment does not
constitute approval of this extension. Issue #2971 still describes domain-scoped
registration.

The native definition must not throw and has this signature:

```c
int32_t entry(const SimplerCcuRegistrationBinding *binding);
```

Query and registration receive the same one-element array `{&binding}`.
The Host binding contains communication-context rank and size, callable-local `ccu_id`,
a channel table indexed by L2 rank, and three distinct opaque Event handles:
ready, trigger and done. Self and unused channel entries are zero. Each Event
uses bit 0. SDK handles are not device addresses.

The base communicator uses the ordered L2 ranks 0..N-1; registration validates
that complete order. A domain may use a reordered subset of those ranks.
The current backend supports at most 64 ranks and one common IO die per
registration. Missing links, unsupported die placement or insufficient HCOMM
resources fail explicitly; the backend must not silently substitute a ring.
The user kernel chooses its peer from the channel table.

## Windows and device workspace

`allocate_domain(..., workspace_engines=("ccu",))` allocates W, obtains each
owner's `HcommCcuGetMemToken(address, size)`, and exchanges address, valid byte
range, token and physical L2 rank across domain members. Empty workspace
configuration requests no CCU resources. The Python domain association prepares
no callable registration; the native window allocator never launches CCU
programs. CCU Host windows and RDMA combinations
are currently unsupported.

`CommContext.workspaces[CCU] = {addr, size}` points to a
`SimplerCcuWorkspace`. Its W table remains valid throughout the window's
lifetime. Each `CommCcuWindow` supplies owner CCU address, range, token and
L2 rank. These addresses differ from imported AIV mappings in `windowsIn`;
URMA descriptors cannot substitute for CCU credentials.

At the execution boundary the workspace also references the current run's
ordered control table and `run_generation`. Each control contains the
parameter slot's address, size, alignment and token, a run command address, plus ready/trigger/done
device Event descriptors. Host-only module owners, streams, SDK handles and
registration caches stay on Host. `CommContext` has no CCU-specific top-level
fields and is 1104 bytes in this build.

AIV resolves `comm_ctx -> workspaces[CCU] -> controls[ccu_id]` with size,
alignment, bounds and generation checks. The typed
`submit_ccu_mission(async_ctx, comm_ctx, ccu_id, call_args)` takes positional
`CcuCallArg` values. A window reference expands to two words: owner address
plus byte offset, and the W token. A scalar expands to one word; a peer-rank
reference resolves the domain rank through the W descriptor's actual L2 rank,
which indexes the complete channel table. References are checked before
completion admission or parameter publication. The native module declares
the expanded parameter size and consumes these words in the same order.

## Prepare and activation

Prepare allocates an independent parameter buffer P, its token, descriptors
and launch arguments for each execution. Preparing B while A executes does not
clear Events, write A's P, change a domain's active controls or launch B.
Each preparation retains its own generation and resource references.

`SimplerCcuLaunchArguments` contains two 64-bit launch words:

| Word | Value |
| ---- | ----- |
| 0 | `parameter_addr` |
| 1 | `parameter_token` |

`HcommCcuGetTaskArgsNum` must report 2. P's token authorizes reading P; W tokens
arrive dynamically in P and authorize actual data windows. They have separate
lifetimes and are not interchangeable.

The existing L2 transaction admits activation only after the predecessor's
ordinary work, asynchronous completion records and CCU streams have retired.
Activation resets and confirms local Events, launches the callable's missions,
and publishes that run's controls to the selected CCU domain workspace. The
prepared run's workspace is also passed directly into its AICPU runtime image.
Preparing a successor does not change that pointer.

Every L2 execution's AICPU initialization waits for its local required READY
Events, clears and confirms them, then admits AIV work. Startup has a bounded
timeout and propagates initialization failure. Host does not perform a READY
roundtrip and the runtime adds no cross-rank READY barrier. Cross-rank
synchronization belongs to the user's CCU algorithm.

## Completion and retirement

The runtime records READY immediately before entering the native instruction definition.
The definition owns its initialization, trigger waits, communication, completion,
loop and normal exit. READY indicates the mission has started. It does not
represent the user definition's initialization or any peer's readiness.
Resource sizing and registration use the same READY prefix and user definition.
Modules declare the THREE_EVENTS capability and the two launch words above.

1. AIV registers the asynchronous done condition, publishes parameters to P and
   then records trigger.
2. The user kernel consumes trigger, reads P, executes the request and publishes
   done with output visibility. It can then wait for another request or return.
3. AICPU waits for ordinary task completion and done, clears and confirms done,
   and retires the asynchronous condition before releasing dependent tasks.
4. Users arrange normal termination through an ordinary submission, including
   for a mission with no business work. Its user-defined exit request publishes
   the same done Event before returning.
5. At L2 END, Host confirms all mission streams have terminated before releasing
   execution credit and P. Domain roots no longer reference the retired controls.

A pending stream after done can complete normally within the Host wait budget.
If the stream stays active, Host reports an exit-protocol timeout and aborts
the run. A missing trigger or done instead expires the L2 execution timeout
and reaches the same abort path through execution failure. Stream query errors
retain their device error in platform diagnostics. Unconfirmed termination
closes admission and retains modules, Events, P, W and streams. A prepared
successor cannot activate or reset its predecessor's Events.

Users serialize requests to a single mission through full asynchronous task
completion before publishing new parameters. Independent missions can execute
concurrently. The first version uses three Events and one parameter slot per
mission.
