# Native CCU module and three-Event contract

A `ChipCallable` packages native HCOMM CCU definitions alongside its AIV
functions. The module owns the communication algorithm and request control flow;
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
The manifest declares ABI version 2, with domain-logical channel indexing.
The loader rejects older versions and incompatible structure sizes. Each kernel
record declares `required_rank_count`; zero permits a dynamic domain size.

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

`Worker.register(ChipCallable)` stores definitions without acquiring CCU hardware
resources. Allocate a domain with `workspace_engines=("ccu",)`, then explicitly
register selected definitions through `domain.get_or_register_ccu(handle, ccu_id=..., worker=...)`.
Each registration belongs to one target L2 Worker, which must be a domain member.
Only that Worker needs the callable definition and receives the registration request.
`worker` is the L2 Worker ID used for submission, not the domain-logical rank.
The domain caches registrations by target Worker, selected module content, entry and build ABI identity,
plus the registration configuration. Unrelated modules elsewhere in a callable
do not change this key. The callable-local `ccu_id` remains registration configuration
because the native build binding exposes it; the same definition at different
callable-local IDs therefore has separate registrations until explicit alias routing
is supported. Repeated calls reuse that domain instance's registration; releasing and recreating the
domain invalidates the old handles even when its name and members are unchanged.
Domain creation does not register kernels, and registration does not launch them.
Complete all cache-miss registrations before the first task submission in an
orchestration run, following the existing direct-control ordering rule.

```python
ag_a = domain_a.get_or_register_ccu(chip, ccu_id=0, worker=rank)
ag_b = domain_b.get_or_register_ccu(chip, ccu_id=0, worker=rank)
reduce_b = domain_b.get_or_register_ccu(chip, ccu_id=1, worker=rank)
orch.submit_next_level(chip, args, config, worker=rank,
                       ccu_bindings=(ag_a, ag_b, reduce_b))
```

For a group, provide a dictionary from submitted Worker ID to that Worker's
registration sequence. Every submitted Worker must have an entry; an empty
sequence means no local CCU missions. Each binding must belong to a live domain
and must have been registered on the submitted Worker. Algorithms,
not the runtime, determine whether all domain members need to issue requests.
The runtime does not infer bindings from scalar arguments. Multiple domain
contexts and repeated occurrences of the same context are allowed.

Bindings select `(domain, ccu_id)` directly, with no separate device slot space.
Conflicting definitions at the same domain and ID are rejected. A registration
referenced more than once in a submission produces one mission execution.
An explicitly acknowledged, clean registration failure permits retry. An unknown
registration outcome poisons the domain to prevent unsafe reuse. The backend uses
a single-member registration communicator for local program resources while
retaining the full domain channel table and logical rank in the kernel binding.

The native definition must not throw and has this signature:

```c
int32_t entry(const SimplerCcuRegistrationBinding *binding);
```

Query and registration receive the same one-element array `{&binding}`.
The Host binding contains domain-logical rank and size, callable-local `ccu_id`,
a channel table indexed by domain-logical rank, and three distinct opaque Event handles:
ready, trigger and done. Self and unused channel entries are zero. Each Event
uses bit 0. SDK handles are not device addresses.

Domain member order determines logical rank: `[0, 2]` and `[2, 0]` have
different channel orders. The base communicator mapping remains internal.
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
plus byte offset, and the W token. A scalar expands to one word. Peer indices
are ordinary scalar domain-logical ranks; the kernel or generated code must
ensure they are within the channel table bounds. References are checked before
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
Activation resets and confirms local Events, launches the locally selected
missions, and publishes each domain's sparse `ccu_id` control table. Missing
entries fail device lookup. All selected missions belong to one execution collection. The
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
4. At L2 END, Host finishes the execution collection only after all asynchronous
   requests have completed and every associated mission has confirmed termination.
5. Host allows a bounded natural-exit wait after L2 completion. If a mission
   remains resident, `stop_completed_stream` calls `aclrtStreamAbort`, then
   requires a successful `aclrtStreamQuery` reporting `COMPLETE`. Only confirmed
   termination permits releasing this run's parameters and controls. Expected
   shutdown does not turn successful request execution into an error.

Each L2 communication owner keeps a pool of ordinary CCU mission streams.
Registration creates the kernel program and Events; activation borrows one stream
per unique mission, creating more only when the idle pool is exhausted. Streams
return to the pool after confirmed termination, retain their identities between
runs, and are destroyed during L2 finalization after the registration
communicators and their HCOMM threads are released. Each registration caches its
HCOMM thread binding for each pool slot it uses. A new launch checks the borrowed
stream is complete; it does not register the kernel again.

`aclrtStreamAbort` does not destroy a stream. The Ascend 950 runtime
[implementation](https://github.com/hicann/runtime/blob/8f16f26e162fd995ceb36c7c51d4c65414cc3063/src/runtime/core/src/stream/stream.cc#L1291)
waits for device abort completion, clears task resources and queues, then restores
normal stream state. The HCOMM backend additionally checks the returned stream
state. No common cross-rank mission-stop barrier is introduced; the algorithm
must make its request completion safe before the local L2 execution ends.

A missing trigger or done is an execution failure, not an expected shutdown.
Stream query errors retain their device error in platform diagnostics.
Unconfirmed termination closes admission and retains modules, Events, P, W and
streams. A prepared successor cannot activate or reset its predecessor's Events.

Users serialize requests to a single mission through full asynchronous task
completion before publishing new parameters. The runtime does not automatically queue competing requests to one mission and
does not currently promise to detect violations. Independent missions can execute
concurrently. The first version uses three Events and one parameter slot per
mission.
