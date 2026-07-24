# Remote L4 two-machine NPU compute smoke

This smoke is the real-device baseline imported from
`codex/remote-l4-npu-machineA`. It validates:

```text
L4 parent -> machine A remote L3 -> local NPU 0 + NPU 1 vector group
          -> machine B remote L3 -> local NPU 0 + NPU 1 vector group
```

The parent allocates six remote buffers on each machine, uploads different
inputs, dispatches both remote L3 tasks, downloads both outputs, and checks the
vector-example golden result. No `mpirun` is used.

This baseline proves remote startup, remote buffers, L4 task dispatch, local L3
group scheduling, NPU execution, and result copy-back. It does not itself read
peer-machine memory. Run the sibling
[`a3_l4_tcp_smoke`](../a3_l4_tcp_smoke/README.md) for the Global CommDomain
Fabric-handle exchange and peer `TLOAD` test.

## Prepare both NPU machines

Use the same source revision on both machines:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"

SIMPLER_REMOTE_L4_NPU_ROLE=machineA \
  bash tools/remote_l4_npu/start_machine_daemon.sh
```

Use `machineB` for the role on the second server. The default daemon address is
`0.0.0.0:19073`; override `SIMPLER_REMOTE_L4_NPU_HOST` or
`SIMPLER_REMOTE_L4_NPU_PORT` when needed.

The daemon starts a session runner on random TCP command and health ports.
Firewalls between the parent and both NPU machines must allow those returned
ports in addition to the daemon port.

## Run on the L4 parent

The parent needs the A3 compilation toolchain because it builds the vector
orchestration and AIV kernels before opening the remote sessions.

```bash
export SIMPLER_REMOTE_L4_NPU_MACHINE_A=10.0.0.11:19073
export SIMPLER_REMOTE_L4_NPU_MACHINE_B=10.0.0.12:19073
export SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES=0,1
export SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES=0,1

bash tools/remote_l4_npu/run_parent_smoke.sh
```

Each machine must provide exactly two free A3 device ids. Success requires all
four downloaded outputs to match the vector golden result.
