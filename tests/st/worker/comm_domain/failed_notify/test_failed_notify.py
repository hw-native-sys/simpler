# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local failure drains peer counters without granting domain reuse."""

import os
import re
import signal
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import (
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = Path(__file__).parent
RUNTIME = "tensormap_and_ringbuffer"
pytestmark = [
    pytest.mark.platforms(["a2a3", "a2a3sim"]),
    pytest.mark.device_count(2),
    pytest.mark.runtime(RUNTIME),
    pytest.mark.timeout(900),
]


def build_callable(platform, directory):
    compiler = KernelCompiler(platform)
    children = []
    for consumer in (0, 1):
        source = directory / f"notify_{consumer}.cpp"
        source.write_text((HERE / "kernels/notify.cpp").read_text().replace("NOTIFY_CONSUMER", str(consumer)))
        binary = compiler.compile_incore(
            str(source),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        )
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        children.append((consumer, CoreCallable.build(signature=[D.OUT], binary=binary)))
    return ChipCallable.build(
        signature=[D.OUT, D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(RUNTIME, str(HERE / "kernels/notify_orchestration.cpp")),
        children=children,
    )


# Hardware failures retain communication owners and can leave a child alive
# after close. Run each fault case in its own device allocation; the allocation
# must discard the process tree before reusing its cards. Simulation needs no
# such isolation. A failed Worker is never reused to run a normal control.
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(0, id="normal"),
        pytest.param(1, id="notify_after_error", marks=pytest.mark.manual(["a2a3"])),
        pytest.param(2, id="missing_notify", marks=pytest.mark.manual(["a2a3"])),
    ],
)
def test_local_failure_with_peer_counter(st_platform, st_device_ids, tmp_path, mode):
    logs = tmp_path / "ascend"
    logs.mkdir()
    output = tmp_path / "probe.log"
    # CANN reads its logging environment during library initialization. Set it
    # before importing the binding in a fresh interpreter, including fork parents.
    env = {**os.environ, "ASCEND_PROCESS_LOG_PATH": str(logs), "ASCEND_GLOBAL_LOG_LEVEL": "3"}
    with output.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                st_platform,
                ",".join(map(str, st_device_ids[:2])),
                str(mode),
                str(tmp_path),
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            # Compilation and Worker startup precede the run under test. Worker
            # startup alone permits 300 s; do not charge that setup against the
            # separate 160 s run and 80 s close bounds below.
            setup_deadline = time.monotonic() + 600
            while not (tmp_path / "probe.ready").exists() and process.poll() is None:
                if time.monotonic() >= setup_deadline:
                    raise subprocess.TimeoutExpired(process.args, 600)
                time.sleep(0.1)
            result = process.wait(timeout=240)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            discard_probe_session(process.pid)
            # Preserve the actual child phase and failure even on a timeout.
            print(output.read_text(errors="replace"), flush=True)
    evidence = output.read_text(errors="replace")
    assert result == 0, evidence
    assert_failure_evidence(st_platform, mode, evidence)


def discard_probe_session(session_id):
    # Chip children create their own process groups inside the probe's session.
    # Only this session belongs to the test; no device-wide reset is attempted.
    # Discarding these processes is not evidence of Worker recovery.
    deadline = time.monotonic() + 10
    while True:
        remaining = []
        processes = subprocess.check_output(["ps", "-e", "-o", "pid=,stat="], text=True)
        for row in processes.splitlines():
            raw_pid, state = row.split()
            if state.startswith("Z"):
                continue
            pid = int(raw_pid)
            try:
                if os.getsid(pid) == session_id:
                    os.kill(pid, signal.SIGKILL)
                    remaining.append(pid)
            except ProcessLookupError:
                pass
        if not remaining:
            return
        assert time.monotonic() < deadline, f"probe session did not exit: {remaining}"
        time.sleep(0.05)


def run_case(st_platform, st_device_ids, tmp_path, mode):
    configure_logging("error")
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=RUNTIME,
        device_ids=[int(d) for d in st_device_ids[:2]],
        num_sub_workers=0,
    )
    domains = []
    outputs = []
    try:
        print("PEER_COUNTER_STAGE compile", time.monotonic(), flush=True)
        handle = worker.register(build_callable(st_platform, tmp_path))
        print("PEER_COUNTER_STAGE init", time.monotonic(), flush=True)
        worker.init()
        outputs = [worker.create_buffer(64) for _ in range(2)]
        for output in outputs:
            output.shm.buf[:] = bytes([0x55]) * 64

        def run(orch, _args, config):
            with orch.allocate_domain(
                name="peer_failure",
                workers=[0, 1],
                window_size=4096,
                buffers=[CommBufferSpec("signals", "uint32", 32, 128)],
            ) as domain:
                domains.append(domain)
                for rank in range(2):
                    orch.copy_to_domain(domain, rank, bytes(128), buffer="signals")
                for rank in range(2):
                    args = TaskArgs()
                    args.add_tensor(outputs[rank].tensor((16,), DataType.UINT32), TensorArgType.OUTPUT_EXISTING)
                    args.add_tensor(domain[rank].buffers["signals"].tensor((32,), DataType.UINT32), TensorArgType.INPUT)
                    args.add_scalar(domain[rank].device_ctx)
                    args.add_scalar(mode)
                    args.add_scalar(rank)
                    orch.submit_next_level(handle, args, config, worker=rank)

        config = CallConfig()
        config.aicpu_thread_num = 3
        config.enable_chip_swimlane = 0
        print("PEER_COUNTER_STAGE run", time.monotonic(), flush=True)
        (tmp_path / "probe.ready").touch()
        started = time.monotonic()
        if mode:
            with pytest.raises(RuntimeError) as failure:
                worker.run(run, config=config)
            assert domains, str(failure.value)
            assert not domains[0].freed
            assert struct.unpack_from("I", outputs[0].shm.buf)[0] == 0x55555555
            assert struct.unpack_from("I", outputs[1].shm.buf)[0] == 37
            with pytest.raises(RuntimeError):
                worker.run(lambda *_: None)
            assert not domains[0].freed
        else:
            worker.run(run, config=config)
            assert [struct.unpack_from("I", output.shm.buf)[0] for output in outputs] == [37, 37]
            assert domains[0].freed
        # A failed run leaves rank 0 quarantined. The healthy rank can wait
        # 120 s in domain release (comm_sim/comm_hccl), separate from the local
        # drain deadline. Cross-rank abort is not supplied here.
        run_limit = 160 if mode else 60
        assert time.monotonic() - started < run_limit
    finally:
        print("PEER_COUNTER_STAGE close", time.monotonic(), flush=True)
        started = time.monotonic()
        try:
            worker.close()
        except RuntimeError as error:
            if not mode or not domains or domains[0].freed:
                raise
            assert "unconfirmed" in str(error), str(error)
        # Failed hardware children can consume the 60 s teardown-report wait.
        assert time.monotonic() - started < 80
        print("PEER_COUNTER_STAGE done", time.monotonic(), flush=True)


def assert_failure_evidence(st_platform, mode, evidence):
    if mode:
        assert "runtime_status=-106" in evidence, evidence
        if st_platform.endswith("sim"):
            assert "PEER_NOTIFY_CONSUMER_ENTERED" not in evidence
        else:
            # Failed L3 children can retain native owners before CANN flushes
            # their device logs. Hardware checks the stream failure and peer
            # output; the executor handshake and simulation check drain fields.
            assert "507018" in evidence, evidence
            return
        records = re.findall(
            r"kernel retirement: core_error=(\d+) async_error=(\d+) cores=\d+/\d+ "
            r"drained=(\d+) complete=(\d+) primary_code=(\d+)",
            evidence,
        )
        assert len(records) == 1, evidence
        core_error, async_error, drained, complete, primary = map(int, records[0])
        assert core_error == 0 and primary == 106, records
        assert (drained, complete) == ((1, 1) if mode == 1 else (0, 0)), records
        assert async_error == (0 if mode == 1 else 100), records
        print("PEER_COUNTER_RETIREMENT", mode, records, flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        run_case(sys.argv[1], sys.argv[2].split(","), Path(sys.argv[4]), int(sys.argv[3]))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # Do not retry an unconfirmed Worker.close from a destructor during interpreter
    # shutdown. The pytest parent discards this probe's entire session.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
