# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TMR kernel-mode SDMA ownership and real TPREFETCH_ASYNC eager/replay coverage."""

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (
    _bind_capture_functions,
    _check,
    _check_resident,
    _TensorIO,
)
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.test_kernel_mode_capture import (
    MODULE,
    ROOT,
    RUNTIME,
    _bind_acl,
    _bind_observer_guards,
    _fail,
    capture_observer,  # noqa: F401 -- shared native observer fixture
)
from tests.ut.py.test_kernel_mode_c_api import _binaries, _load

_COUNT = 128
_ROUNDS = 100


@pytest.mark.sdma
@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", ["eager", "replay", "recreate", "disabled", "program_isolation"])
def test_kernel_prefetch(st_platform, st_device_ids, scenario, capture_observer):  # noqa: F811
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "kernel_mode"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="prefetch-" + scenario + "-", dir=artifacts))
    logs = output / "ascend"
    logs.mkdir()
    env = dict(os.environ)
    env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    env["LD_PRELOAD"] = str(capture_observer) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", MODULE + ".test_kernel_prefetch", str(st_device_ids[0]), scenario, str(output)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    text = (output / "run.log").read_text()
    assert result.returncode == 0, f"{text}\nArtifacts: {output}"
    expected = f"PASS prefetch {scenario} rounds=100"
    if scenario != "program_isolation":
        expected += " forbidden_sync=0 allocations=0"
    assert expected in text


def _build_prefetch(build_dir):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    source = ROOT / "examples/a2a3/tensormap_and_ringbuffer/prefetch_async_demo/kernels"
    compiler = KernelCompiler("a2a3")
    orchestration = compiler.compile_orchestration(
        RUNTIME, str(source / "orchestration/prefetch_async_orch.cpp"), build_dir=str(build_dir)
    )
    incore = compiler.compile_incore(
        str(source / "aiv/kernel_prefetch_copy.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        build_dir=str(build_dir),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature, func_name="prefetch_async_orchestration", binary=orchestration, children=[(0, child)]
    )


def _run_program_isolation(device, build_dir, *, runtime=RUNTIME, build_prefetch=_build_prefetch):
    from simpler.task_interface import (  # noqa: PLC0415
        CallConfig,
        ChipStorageTaskArgs,
        ChipTensor,
        ChipWorker,
        DataType,
    )

    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    chip = build_prefetch(build_dir)
    bins = RuntimeBuilder("a2a3").get_binaries(runtime)
    lib = _load("a2a3", "onboard", runtime)
    _bind_acl(lib)
    config = CallConfig()
    config.runtime_env.ring_task_window = 64
    if runtime == RUNTIME:
        config.runtime_env.ring_heap = 1 << 20
        config.runtime_env.ring_dep_pool = 1024
    program = ChipWorker()
    program.init(device, bins, prewarm_config=config, enable_sdma=True)
    handle = program.register_callable(chip)
    borrower = ChipWorker()
    # Its AICPU init publishes zero in the resident SO's DMA globals.
    borrower.kernel_init(device, bins, config, enable_sdma=False)
    # The resident executor rejects program dispatch while a kernel context is
    # live. Revoke it first; revocation does not restore the DMA globals.
    _check(lib.aclrtSynchronizeDevice(), "external drain before borrower close")
    borrower.finalize()
    allocations = []
    io = _TensorIO(lib, allocations)
    io.array = ctypes.c_float * _COUNT
    io.nbytes = ctypes.sizeof(io.array)
    source, destination = io.allocate(), io.allocate()
    args = ChipStorageTaskArgs()
    for address in (source, destination):
        args.add_tensor(ChipTensor.make(address.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
    for round_id in range(_ROUNDS):
        values = [float(round_id * 257 + i) for i in range(_COUNT)]
        io.write(source, values)
        io.write(destination, [-999.0] * _COUNT)
        program.run(handle, args, config)
        io.verify(destination, values)
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free test tensor before device owner closes")
    program.finalize()
    print("PASS prefetch program_isolation rounds=100", flush=True)


def _run(device, scenario, build_dir, *, runtime=RUNTIME, build_prefetch=_build_prefetch):
    from simpler.task_interface import (  # noqa: PLC0415
        CallConfig,
        ChipStorageTaskArgs,
        ChipTensor,
        ChipWorker,
        DataType,
    )

    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    chip = build_prefetch(build_dir)
    bins = RuntimeBuilder("a2a3").get_binaries(runtime)
    assert bins.sdma_warmup_path is not None
    lib = _load("a2a3", "onboard", runtime)
    _bind_acl(lib)
    _check(lib.aclInit(None), "init ACL")
    _check(lib.aclrtSetDevice(device), "bind framework device")
    caller = ctypes.c_void_p()
    _check(lib.aclrtCreateStream(ctypes.byref(caller)), "create caller stream")
    config = CallConfig()
    config.runtime_env.ring_task_window = 64
    if runtime == RUNTIME:
        config.runtime_env.ring_heap = 1 << 20
        config.runtime_env.ring_dep_pool = 1024
    worker = ChipWorker()
    worker.kernel_init(device, bins, config, enable_sdma=scenario != "disabled")
    cid = worker.kernel_prepare_callable(chip)
    allocations = []
    io = _TensorIO(lib, allocations)
    io.array = ctypes.c_float * _COUNT
    io.nbytes = ctypes.sizeof(io.array)
    source, destination = io.allocate(), io.allocate()
    args = ChipStorageTaskArgs()
    for address in (source, destination):
        args.add_tensor(ChipTensor.make(address.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
    observer = _bind_capture_functions(lib, hbg=runtime != RUNTIME)
    _bind_observer_guards(observer)
    observer.capture_observer_invocation_allocations.restype = ctypes.c_uint64
    committed = worker.committed_device_memory
    graph = ctypes.c_void_p()
    host_launches = 0

    def enqueue(operation):
        observer.capture_observer_guard_sync(1)
        observer.capture_observer_invocation_scope(1)
        try:
            operation()
        finally:
            observer.capture_observer_invocation_scope(0)
            observer.capture_observer_guard_sync(0)
        assert observer.capture_observer_sync_calls() == 0
        assert observer.capture_observer_invocation_allocations() == 0

    def launch():
        nonlocal host_launches
        enqueue(lambda: worker.kernel_launch(cid, args, caller.value))
        host_launches += 1

    if scenario in ("replay", "recreate"):
        # First invocation is captured, without an eager priming call.
        _check(lib.aclmdlRICaptureBegin(caller, 0), "capture begin")
        launch()
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value
    for round_id in range(_ROUNDS):
        values = [float(round_id * 257 + i) for i in range(_COUNT)]
        io.write(source, values)
        io.write(destination, [-999.0] * _COUNT)
        if graph.value:
            enqueue(lambda: _check(lib.aclmdlRIExecuteAsync(graph, caller), "replay"))
        else:
            launch()
        _check(lib.aclrtSynchronizeStreamWithTimeout(caller, 10000), "external caller sync")
        # The child writes only after a non-null workspace AND a successful
        # AsyncEvent.Wait. Sentinel seeding prevents a prior replay masking a failure.
        io.verify(destination, [-999.0] * _COUNT if scenario == "disabled" else values)
        assert worker.committed_device_memory == committed
    _check_resident(observer, host_launches)
    if graph.value:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph before SDMA release")
    worker.finalize()
    if scenario == "recreate":
        # The runtime permits only one live context for this device/SO pair.
        # Reuse the borrowed device after graph destruction and healthy close.
        for enabled in (False, True):
            worker = ChipWorker()
            worker.kernel_init(device, bins, config, enable_sdma=enabled)
            cid = worker.kernel_prepare_callable(chip)
            if runtime == RUNTIME:
                observer.capture_observer_begin()
            else:
                observer.capture_observer_begin_hbg()
            io.write(destination, [-999.0] * _COUNT)
            launch()
            _check(lib.aclrtSynchronizeStreamWithTimeout(caller, 10000), "replacement caller sync")
            io.verify(destination, values if enabled else [-999.0] * _COUNT)
            _check_resident(observer, 1)
            worker.finalize()
    # A fresh framework allocation after close proves the borrowed device still works.
    io.allocate()
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free framework tensor")
    _check(lib.aclrtDestroyStream(caller), "destroy caller stream")
    _check(lib.aclrtResetDevice(device), "framework reset")
    _check(lib.aclFinalize(), "framework finalize")
    print(f"PASS prefetch {scenario} rounds=100 forbidden_sync=0 allocations=0", flush=True)


if __name__ == "__main__":
    try:
        if sys.argv[2] == "program_isolation":
            _run_program_isolation(int(sys.argv[1]), Path(sys.argv[3]))
        else:
            _run(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    except BaseException:
        _fail()
