# HBG graph preparation and packet execution sequence

HBG separates Host graph construction, resource ownership, packet parsing,
execution binding validation, and device execution preparation. The common
kernel resources and launch binder provide fixed storage and stream ordering.
The A2/A3 onboard `hbg_kernel_launch.cpp` connects these modules to the public
kernel C API and graph resource tokens.

The diagram describes module responsibilities. Requests marked **owner** are
coordinated by the Host runner or AICPU entry; they do not imply a Host round trip
during replay. The dedicated launch is supported on A2/A3 HBG only. A5, TMR
dedicated launch and dedicated HBG diagnostics are outside this path.

```mermaid
sequenceDiagram
    autonumber
    participant App as Caller
    participant O as Launch/execution owner
    participant B as Host graph builder
    participant R as Context/resources
    participant T as Host graph template
    participant L as Streams/launch binder
    participant S as Device slot control
    participant P as Packet reader/parser
    participant V as Execution binding validation
    participant E as Device execution preparation
    participant C as Scheduler/AICore

    App->>O: Request graph execution
    O-->>B: owner: arrange arguments and Host workspace
    B->>B: build_graph(): invoke Host orchestration
    B->>B: resolve_graph_task_definitions()
    B->>B: measure_definition_packing(), queue sizing
    B-->>O: GraphBuild (records and borrowed workspace)

    alt Existing native HBG path
        O->>B: Continue native bind/retained prepare plan
        B->>B: bind_graph_definitions(), commit resources
        B->>B: publish_run_image_impl()
        Note over B,L: Existing native image publication and launch remain separate from kernel packets
        O-->>L: Existing native launch owner
        L-->>C: Native runtime execution
        C-->>O: Native completion and output handling
    else Fixed-slot kernel packet composition
        O-->>R: owner: initialize context with device/generation
        O-->>R: owner: measure_graph_resources(), KernelResourceCapacity::create()
        O-->>R: owner: capacity.prepare(context, allocation ops)
        R->>R: prepare_resources() -> KernelDeviceResources::prepare()
        O-->>R: owner: freeze_resources()
        O-->>R: owner: capacity.make_registration()
        R->>R: inspect_frozen_resources()
        R-->>O: GraphSlotRegistration (addresses, capacities, identities)

        O-->>S: owner: submit register_execution_slot control entry
        S->>S: install_graph_execution_slot()
        S->>S: initialize fresh generation or preserve same-generation state
        S->>S: register_graph_execution_slot(), bind_graph_slot_registry()
        S-->>O: Registration ready, or conflict/poison rejection

        O-->>T: owner: make_graph_launch_template(build, registration, identity)
        T->>T: Validate tensors, compact image, copy Definitions
        T->>P: Host parser: validate completed packet format/checksum
        Note over T,P: Host calls the shared parser only; no device reader/cache operation
        T-->>O: Immutable GraphLaunchTemplate

        O->>T: make_graph_host_args(): copy validated packet bytes
        T-->>O: Per-call packet
        O->>O: pack_hbg_launch_payload(): add launch envelope
        O->>L: launch_bound_kernel(packet, caller, gate, ops)
        L->>L: gate.acquire(), validate handles, query prior tail if needed
        L->>L: enqueue_kernel_launch_sequence()
        Note over L,C: Start -> AICPU wait/clear -> AICoreStart -> AICore launch<br/>AICPU launch -> join AICore -> AICpuDone -> caller join -> SerialTail
        L->>L: gate.finish(): release submission lease; poison on partial enqueue
        Note over O,L: capture records tasks; replay reuses those tasks without rerunning Host template submission

        loop Each device invocation, including replay
            L-->>P: owner: AICPU leader receives packet after Start
            P->>S: current_graph_slot_registry(), acquire_graph_execution_slot()
            S-->>P: Trusted registration snapshot, or rejection
            P->>P: Check size/source bounds/overlap before reading
            P->>P: Invalidate source cache, parse_graph_packet()
            P-->>O: ParsedGraphPacket + registration, or failure
            Note over O,E: Owner denotes coordination on the current side, not a mandatory Host round trip
            O-->>V: owner: validate_graph_execution_binding(parsed, registration, callable)
            V->>V: Compare callable/counts/device/binary/generation/destinations
            V-->>O: ValidatedGraphExecutionBinding, or rejection

            alt Read/parse/binding rejected
                Note over O,C: Do not initialize graph destinations or open dispatch; complete required failure coordination
                O->>O: Preserve the boot owner's restore error through participant teardown
                O->>S: Poison the registered slot before publishing completion
                Note over S,C: A queued successor cannot restore or dispatch, even with a valid packet
            else Binding accepted
                O-->>E: owner: prepare_graph_execution(binding)
                E->>E: validate_images(): graph/task/Definition semantics
                Note over E: InvalidImage/Busy/Quarantined reject before destination writes;<br/>the following initialization requires valid images and an Idle slot
                E->>E: Claim Idle -> Preparing and increment attempt
                E->>E: Zero internal heap, copy images, wire pointers
                E->>E: Initialize queues/mailbox, flush caches
                alt Preparation succeeds
                    E-->>O: Ready + generation + RuntimeContext
                    O-->>E: owner: peers acquire_graph_execution(generation)
                    E-->>O: Same-generation prepared execution
                    O-->>C: owner: open dispatch and execute tasks
                    C-->>O: Completion, shutdown and reader quiescence
                    O-->>E: owner: retire_graph_execution(completion)
                    alt Successful execution and cleanup
                        E->>E: Ready -> Idle
                    else Fatal runtime or teardown error
                        E->>S: poison_graph_execution_slot()
                    end
                else Partial initialization fails
                    E-->>O: Failed / quarantined
                    O-->>E: owner: retire after controlled cleanup and quiescence
                    Note over E,S: Controlled failure may return to Idle; fatal failure poisons
                end
            end
        end

        O-->>O: owner: destroy captured graphs and establish external quiescence
        O-->>S: owner: submit detach_execution_slot control entry
        S->>S: Validate detach identity, detach_graph_slot_registry()
        O-->>R: owner: close context and release owned resources
        Note over R: Failed release retains ownership for explicit retry
    end
    O-->>App: Result/status through the enclosing API
```

## Failed device restoration

`restore_kernel_graph()` preserves the boot owner's result independently of
whether it claimed an execution attempt. Framing, callable, packet and image
validation can fail before the slot leaves `Idle`; that absence of an attempt
is not evidence of a successful invocation. The cleanup participant can also
have a zero local return while the boot owner failed.

After the participants shut down, `retire_kernel_graph()` consumes that saved
result and poisons the registered slot before the invocation completes. A later
entry cannot restore a valid packet over the failure. Successful restoration
and retirement still return the slot to `Idle` for the next replay. This is a
controlled restore-failure gate; a hung device still requires the caller's
bounded completion monitoring and retained resources.

Host argument or capacity refusal before enqueue does not enter this device
path and remains retryable. Ordinary native runs do not consume the saved
kernel restore result. This restore snapshot does not replace the executor's
existing accumulation of other participants' execution and shutdown errors.

## File responsibilities

Paths below are relative to `src/common/host_build_graph` unless stated otherwise.

| Participant | Files and responsibilities |
| ----------- | -------------------------- |
| Host graph builder | `src/{a2a3,a5}/runtime/host_build_graph/host/runtime_maker.cpp` implements `build_graph()` and preserves the native bind/publication path; `host_graph_build.h` defines the result contract. |
| Definition resolution | `host/graph_task_definition_binding.h` resolves task/Definition relationships and queue populations; `host/definition_packing.h` measures section offsets and sizes. |
| Resource planning | `kernel_resource_requirements.h` measures graph needs, aggregates fixed capacities and constructs registration from frozen resources. Actual resource ownership belongs to `KernelExecutionState` and `KernelDeviceResources` under `src/common/platform`. |
| Template | `kernel_graph_template.h`, `host/kernel_graph_template.cpp` own canonical packet bytes and per-submission writable copies. `kernel_graph_tensor_validation.h` supplies tensor descriptor/binding predicates. |
| Packet format/parser | `kernel_graph_packet.h` defines format and parsed views; `kernel_graph_packet_parser.h`, `shared/kernel_graph_packet_parser.cpp` implement shared parsing/checksum rules; `kernel_graph_packet_bounds.h` supplies bounded arithmetic. |
| Slot control | `kernel_graph_slot_protocol.h` defines shared records/states; `kernel_graph_slot_control.h`, `device/kernel_graph_slot_control.cpp` implement install, bind, detach and poison. |
| Packet read boundary | `kernel_graph_packet_reader.h`, `device/kernel_graph_packet_reader.cpp` obtain trusted registration and check source safety before cache maintenance and parsing. |
| Binding validation | `kernel_graph_execution_validation.h`, `device/kernel_graph_execution_validation.cpp` compare parsed claims with trusted registration and callable identity. |
| Execution preparation | `kernel_graph_execution.h`, `device/kernel_graph_execution.cpp` validate image semantics, initialize execution state, publish generation, acquire results and retire. They do not launch or synchronize tasks. |
| Common launch | `src/common/platform/include/host/kernel_launch_binder.h`, `src/common/platform/onboard/host/kernel_launch_binder.cpp` and `kernel_launch_sequence.h` implement the generic submission contract and stream ordering. A2/A3 `hbg_kernel_launch.cpp` supplies the gate and launch callbacks. |

## Ownership and integration boundaries

- `GraphBuild` borrows Host workspace; its lease must outlive its consumers.
  `GraphLaunchTemplate` owns copied packet bytes but does not own the device
  resources or caller tensors named by those bytes.
- Registration must finish before device preparation. Template construction can
  occur before or after registration installation once its resource addresses
  have been fixed; the diagram chooses one valid ordering.
- Parsing proves format validity. Binding validation proves compatibility with
  trusted current identities. Neither replaces graph image validation or the
  owner's obligation to keep packet/resource storage alive and immutable.
- The owner retains the device execution lease from reading through consumption
  and retirement; Host submission and device execution are different lifetimes.
  An atomic phase field alone does not serialize all control operations.
- AICore dispatcher startup may precede completion of AICPU preparation. Actual
  graph task dispatch waits for successful preparation and participant ordering.
- Each dedicated Host call builds a fresh template. Captured replay reuses the
  recorded packet; changed Host arguments require a new capture. The argument
  snapshot hash is not an automatic template cache.
- CPU contract tests exercise module composition. They do not establish a
  connected production capture/replay path or A5 resident hardware execution.
