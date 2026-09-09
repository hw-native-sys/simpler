# simpler Documentation

Index of every document under `docs/`, grouped by what you are trying to do.
The top-level [README](../README.md) links only the handful of entry-point docs;
this page is the complete map.

These pages are also published as a searchable site at
<https://www.pypto.ai/simpler/>, which adds a generated API reference
for `simpler.worker`, `simpler.task_interface` and `simpler.orchestrator`. The
site is built by `.github/workflows/docs.yml`; `mkdocs.yml` owns its navigation,
so a new page needs a `nav` entry there as well as a row here.

New docs belong in one of the groups below, and in a subdirectory when the group
already has one (`dfx/`, `hardware/`, `troubleshooting/`, `investigations/`).
Add the row here in the same commit — an unlisted doc is invisible.

**File names are kebab-case** (`chip-level-arch.md`), including when the doc is
named after a code identifier that uses underscores.

## Start here

Two audiences use this tree. If you are **building on** simpler — writing
kernels and orchestration, running and measuring them — start at
**[user/](user/README.md)**, which routes you through install → first run →
write a kernel → profile → debug. Everything else here is written for people
changing simpler's own internals.

| Document | What it covers |
| -------- | -------------- |
| [**Using simpler**](user/README.md) | The user-facing entry point: how-to guides and the Python / CLI reference |
| [Getting Started](getting-started.md) | Prerequisites, PTO-ISA setup, first example run |
| [Installation and Runtime Environment](install.md) | Install layout and the runtime environment variables |
| [Developer Guide](developer-guide.md) | Directory structure, role ownership, when to rebuild |
| [Capability Survey](capability-survey.md) | **Status snapshot** — what is shipped, gated, or design-only across topology, launch, and communication. Read this before assuming a mechanism works |

## Architecture

| Document | What it covers |
| -------- | -------------- |
| [Chip-Level Architecture (L2)](chip-level-arch.md) | Three-program model (host / AICPU / AICore), API layers, handshake |
| [Hierarchical Level Runtime](hierarchical-level-runtime.md) | The L0–L6 level model and component composition |
| [Task Flow](task-flow.md) | Callable / TaskArgs / CallConfig pass-through, `IWorker` |
| [Buffer Memory Model](buffer-abi.md) | How L3+ tasks name data: canonical identity, backend descriptor, strided view |
| [Orchestrator](orchestrator.md) | DAG submission: TensorMap, Scope, Ring, task state machine |
| [Scheduler](scheduler.md) | DAG dispatch: wiring / ready / completion queues, dispatch loop |
| [`tensormap_and_ringbuffer`: A2/A3 vs A5](tensormap-and-ringbuffer-a2a3-vs-a5.md) | Per-file comparison split into hardware/PTO-ISA architecture differences and implementation differences |
| [Worker Manager](worker-manager.md) | Worker pool, THREAD/PROCESS modes, fork + mailbox mechanics |
| [hardware/](hardware/README.md) | Hardware substrate: chip architecture, cache coherency, MMIO performance, CANN source references |

## Kernels and task authoring

| Document | What it covers |
| -------- | -------------- |
| [AICore Kernel Programming](aicore-kernel-programming.md) | Writing AICore kernels for this runtime |
| [a5 AICore SIMT Launch](simt-launch.md) | a5 SIMT launch metadata and the `ChipCallable` alignment constraint |
| [Manual Scope V0](manual-scope.md) | Explicit scope control from orchestration code |
| [WAR Anti-Dependencies](war-anti-dependency.md) | Write-after-read hazards and how the runtime orders them |
| [TPUSH/TPOP Guidelines](tpush-tpop.md) | Advisory usage rules for the push/pop instructions |

## Launch, linking, and callable registration

| Document | What it covers |
| -------- | -------------- |
| [AICPU Kernel Launch Mechanisms](aicpu-kernel-launch-mechanisms.md) | How the AICPU SO is bootstrapped, registered, and launched via CANN |
| [Dynamic Linking and TLS](dynamic-linking.md) | Binary registration, relocation, thread-local storage on device |
| [Callable Identity Registration](callable-identity-registration.md) | How a callable is identified and registered across tiers |
| [Dynamic Callable Registration over IPC](callable-ipc-dynamic-register.md) | Registering callables into a running child process |
| [Python Callable Serialization](python-callable-serialization.md) | Serializing Python callables for L3+ registration |

## Multi-chip, multi-host, and communication

| Document | What it covers |
| -------- | -------------- |
| [Communication Domains](comm-domain.md) | Dynamic `CommDomain` allocation and the symmetric window |
| [L3-L2 Orchestrator Communication](l3-l2-orch-comm.md) | Host-side L3 talking directly to the L2 AICPU orchestrator |
| [L3-L2 Message Queue](l3-l2-message-queue.md) | The queue channel between an L3 host and L2 |
| [Directed NEXT_LEVEL Scheduling](directed-next-level-scheduling.md) | Targeting a specific next-level child instead of any free one |
| [Remote L3 Worker Design](remote-l3-worker-design.md) | L4 host-to-host workers — protocol, transports, status |
| [remote-l3-worker-design/](remote-l3-worker-design/README.md) | Full design set: protocol, buffers and transports, implementation plan and record |

## Profiling, logging, and DFX

| Document | What it covers |
| -------- | -------------- |
| **[dfx/](dfx/README.md)** | **Every profiling and diagnostics reference**, indexed: framework and naming rules, L2/core swimlanes, PMU, host trace, device phases, scheduler-overhead model, args dump, dep_gen, scope stats, backpressure, buffer-capacity audit |
| [Log System](logging.md) | Log levels, sinks, and the host/device logging split |

## Building, testing, and packaging

| Document | What it covers |
| -------- | -------------- |
| [Testing](testing.md) | Test types (st / pyut / cpput), how to run them, writing new tests |
| [CI Pipeline](ci.md) | Jobs, gating, and what each pipeline stage covers |
| [Python Packaging](python-packaging.md) | Wheel layout, `simpler` vs `simpler_setup`, asset packaging |
| [Compiler Sanitizers](sanitizers.md) | ASAN / UBSan / TSAN builds |
| [Sim Multi-Device Isolation](sim-multi-device-isolation.md) | How the simulator isolates concurrent virtual devices |

## When something is broken

| Document | What it covers |
| -------- | -------------- |
| [troubleshooting/](troubleshooting/README.md) | Device error codes, local timeout defaults, AICPU shared-SO faults, sim oversubscription hangs, macOS build issues, cpput ABI issues |
| [investigations/](investigations/README.md) | Considered-and-dropped proposals and measured dead ends. **Check here before proposing an optimization or refactor** |
