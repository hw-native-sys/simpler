# A5 Host peer reads with MTE

This example uses the same Host-source domain API on A5 PR and DT, with
`workspace_engines=()` and `enable_sdma=False`. Each requesting device maps its
peers' same-host POSIX shared memory. Its AIV reads the peer address from
`CommContext.windowsIn`, adds the named buffer's byte offset, and copies through
UB into ordinary local HBM. The peer card does not service the request.

```text
peer rank's Host DDR -> requesting device's mapping -> MTE2 -> UB -> MTE3 -> local HBM
                                                                           |
                                                        dependent consumer: output + 1
```

The producer finishes both MTE pipelines before returning. The consumer uses the
runtime's ordinary tensor dependency; there is no deferred completion or release
of AIV while the copy is in flight. No SDMA or URMA workspace is allocated, and
the kernel verifies that all workspace descriptors are empty.

The two cases select `tensormap_and_ringbuffer` and `host_build_graph`. Each case checks:

- Two ranks reading 64 KiB from a non-self rank into ordinary HBM, with a
  256-byte buffer offset and a dependent consumer adding one.
- Independent Host domains with forward and reversed member order, distinct
  payloads, and an independent Device-source domain alive at the same time.
- Retained domain identities across runs, one-time initialization, and CPU
  updates between completed runs becoming visible to the next MTE read.
- Both devices acknowledge an HBM gate after reading the first half of the
  Host payload. L3 requests domain release while the second-half reads are
  blocked, checks that the domain is not physically freed, then releases the
  gates and verifies the entire output and dependent consumer. The gate has a
  bounded device deadline; expiry fails the check rather than proving ordering.
- Further reads from the surviving Host and Device domains after each Host
  domain is released, and explicit final domain and output-buffer release.

## Run on two devices

Install the repository in a project-local venv on the target machine. The three
runtime cases are standard L3 `SceneTestCase` classes: `CALLABLE` declares the
kernels, `CASES` requests two devices, and the lifecycle hook uses the framework's
Worker and registered callable.

Use the same precompiler as the existing A5 CI job **before** reserving devices:

```bash
source .venv/bin/activate
python -m simpler_setup.tools.scene_test_compile examples/a5/host_mte_domain \
  --platform a5 --exclude-level 4 --require-pto-isa --manual exclude -q
```

This selects and compiles all three classes into the standard scene-test kernel
cache. Each class's compilation hook also prepares its input arrays under
`build/host_mte_domain/<runtime>`. Input checksums cover the runtime, host
architecture, example sources and PTO-ISA pin; prepare on each target machine.
Execution loads those inputs without a generation fallback. Missing, stale or
corrupt inputs require rerunning the precompiler.

The A5 onboard job in PR CI and Daily discovers these classes through its existing
`examples tests/st` precompile and pytest commands. No example-specific workflow
step is needed. Runtime and manual filters use the normal scene-test selection.

Run the cases through the device queue after preparation:

```bash
task-submit --device auto --device-num 2 --run \
  'bash .claude/skills/onboard-arch-precheck/check.sh a5 &&
   .venv/bin/python -m pytest examples/a5/host_mte_domain \
     --platform a5 --device "$TASK_DEVICE"'
```

Use `--runtime host_build_graph` (or either other runtime) on both commands to
select just that runtime.

See [Host-source communication domains](../../../docs/a5-host-comm-domain.md)
for registration, CPU update ordering and lifetime rules.
