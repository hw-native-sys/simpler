# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real native modules, Worker registration, READY, AIV/TMR and confirmed stop."""

import os
import select
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.ccu import ChipCcuBinding, ChipCcuRun
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.ccu_toolchain import get_ccu_toolchain
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_ROOT = Path(__file__).resolve().parent
_RUNTIME = "tensormap_and_ringbuffer"


def _build_callable(tmp_path, variant=0, source_root=_ROOT, runtime=_RUNTIME):
    tmp_path.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler(platform="a5")
    host = compiler.host_gxx
    sanitizer_flags = compiler._sanitizer_flags(host)
    toolchain = get_ccu_toolchain(host, extra_compile_flags=sanitizer_flags, extra_link_flags=sanitizer_flags)
    isa_root = Path(ensure_pto_isa_root())
    (tmp_path / "ccu_manifest_identity.h").write_text(toolchain.manifest_identity_header(), encoding="ascii")
    variant_source = tmp_path / "variant.cpp"
    variant_source.write_text(
        f"#include <cstdint>\nextern const uint32_t ccu_test_variant = {variant};\n", encoding="ascii"
    )
    module = tmp_path / "native_module.so"
    subprocess.run(
        [
            toolchain.cxx_path,
            *toolchain.compile_flags,
            f"-I{tmp_path}",
            *(f"-I{path}" for path in toolchain.include_dirs),
            str(source_root / "native_module.cpp"),
            str(variant_source),
            *toolchain.link_flags,
            *(str(path) for path in toolchain.library_paths),
            "-o",
            str(module),
        ],
        check=True,
    )
    orch_source = source_root / "kernels/orchestration/ccu_orch.cpp"
    if not orch_source.exists():
        orch_source = _ROOT / "kernels/orchestration/ccu_orch.cpp"
    orchestration = compiler.compile_orchestration(runtime, str(orch_source))
    signatures = [[D.OUT, D.IN, D.IN, D.IN], [D.IN, D.OUT, D.IN, D.IN, D.IN], [D.IN, D.IN, D.OUT, D.IN, D.IN]]
    children = []
    for index, name in enumerate(("trigger_b", "trigger_a", "verify")):
        binary = compiler.compile_incore(
            str(source_root / f"kernels/aiv/{name}.cpp"),
            core_type="aiv",
            pto_isa_root=str(isa_root),
            extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
        )
        children.append((index, CoreCallable.build(signature=signatures[index], binary=extract_text_section(binary))))
    return ChipCallable.build(
        signature=[D.OUT, D.OUT, D.OUT, D.IN, D.IN],
        func_name="ccu_orchestration",
        config_name="ccu_orchestration_config",
        binary=orchestration,
        children=children,
        ccu_modules=[module.read_bytes()],
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_registration_reused_across_two_live_domains(st_platform, st_device_ids, tmp_path):
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    domains = []
    try:
        callable_handle = worker.register(_build_callable(tmp_path))
        worker.init()
        worker.register_ccu_callable(callable_handle)

        def allocate(orch, _args, _config):
            for name in ("first", "second"):
                domains.append(
                    orch.allocate_domain(
                        name=name,
                        members=[0, 1],
                        window_size=4 * 1024 * 1024,
                        workspace_engines=("ccu",),
                        retain_after_run=True,
                    )
                )

        worker.run(allocate)
        first, second = domains
        assert first.allocation_id != second.allocation_id
        initial = worker._ccu_context_registrations[(worker._worker, callable_handle.digest)]
        assert worker._ccu_context_registrations[(worker._worker, callable_handle.digest)] == initial
        first.release()
        assert first.freed
        assert worker._ccu_context_registrations[(worker._worker, callable_handle.digest)] == initial
        second.release()
        assert second.freed
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_unenabled_domain_rejects_ccu_registration(st_platform, st_device_ids, tmp_path):
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    try:
        callable_handle = worker.register(_build_callable(tmp_path))
        worker.init()
        worker.register_ccu_callable(callable_handle)
        domains = []

        def allocate(orch, _args, _config):
            domains.append(
                orch.allocate_domain(
                    name="disabled", members=[0, 1], window_size=4 * 1024 * 1024, retain_after_run=True
                )
            )

        worker.run(allocate)
        assert not worker._ccu_run_registrations
        with pytest.raises(ValueError, match="CCU-enabled"):
            worker._ccu_domain_collective(domains[0], 6, callable_handle.digest)
        domains[0].release()
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize("replace_callable", [False, True])
def test_two_missions_b_before_a(st_platform, st_device_ids, tmp_path, request, replace_callable):
    rounds = request.config.getoption("--rounds")
    assert rounds > 0
    target = _build_callable(tmp_path)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    retained = []
    try:
        callable_handle = worker.register(target)
        replacement_handle = (
            worker.register(_build_callable(tmp_path / "replacement", variant=1))
            if replace_callable
            else callable_handle
        )
        worker.init()
        worker.register_ccu_callable(callable_handle)
        if replacement_handle is not callable_handle:
            worker.register_ccu_callable(replacement_handle)
        assert not worker._ccu_run_registrations
        outputs = [[worker.create_buffer(64) for _ in range(3)] for _ in range(2)]

        def run_round(orch, _args, config):
            domain = orch.get_or_allocate_domain(
                name="native_ccu",
                members=[0, 1],
                window_size=4 * 1024 * 1024,
                workspace_engines=("ccu",),
                retain_after_run=True,
            )
            if retained:
                assert domain is retained[0]
            else:
                retained.append(domain)
            for rank in range(2):
                args = TaskArgs()
                for output in outputs[rank]:
                    args.add_tensor(output.tensor((8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_context(rank))
                args.add_scalar(round_index * 100 + rank * 10)
                selected = replacement_handle if round_index > max(1, rounds // 2) else callable_handle
                orch.submit_next_level(selected, args, config, worker=rank)

        for round_index in range(1, rounds + 1):
            worker.run(run_round)
            for rank in range(2):
                value = round_index * 100 + rank * 10
                assert [struct.unpack_from("<Q", output.shm.buf)[0] for output in outputs[rank]] == [
                    value + 2,
                    value + 1,
                    value,
                ]
                assert struct.unpack_from("<Q", outputs[rank][0].shm.buf, 8)[0] == value + 3
            if round_index in (1, rounds) or round_index % 100 == 0:
                print(
                    f"CCU round={round_index}/{rounds}: B-before-A, dynamic parameters and confirmed stop passed",
                    flush=True,
                )
            if round_index == max(1, rounds // 2) and rounds > 1:
                retained[0].release()
                assert retained[0].freed
                retained.clear()
                print("Next round enables CCU on a fresh window allocation", flush=True)
        # Merely borrowing a retained domain does not launch CCU work.
        worker.run(lambda orch, _args, _cfg: orch.get_domain("native_ccu"))
        retained[0].release()
        assert retained[0].freed
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_interleaved_callable_association_during_active_run(st_platform, st_device_ids, tmp_path, monkeypatch):
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    started_reader, started_writer = os.pipe()
    begin = ChipCcuRun.begin

    def observe_begin(run):
        begin(run)
        os.write(started_writer, b"1")

    monkeypatch.setattr(ChipCcuRun, "begin", observe_begin)
    try:
        handles = [worker.register(_build_callable(tmp_path / str(index), variant=index)) for index in range(2)]
        worker.init()
        for handle in handles:
            worker.register_ccu_callable(handle)
        outputs = [[[worker.create_buffer(64) for _ in range(3)] for _ in range(2)] for _ in handles]
        for callable_outputs in outputs:
            for rank_outputs in callable_outputs:
                for output in rank_outputs:
                    output.shm.buf[:] = bytes(64)

        def interleave(orch, _args, config):
            domain = orch.allocate_domain(
                name="interleaved_ccu",
                members=[0, 1],
                window_size=4 * 1024 * 1024,
                workspace_engines=("ccu",),
            )

            def submit(index, rank, predecessor=None):
                args = TaskArgs()
                for output in outputs[index][rank]:
                    args.add_tensor(output.tensor((8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_context(rank))
                args.add_scalar((index + 1) * 100 + rank * 10)
                if predecessor is not None:
                    args.add_dep(predecessor)
                return orch.submit_next_level(handles[index], args, config, worker=rank)

            first_rank0 = submit(0, 0)
            # A started run cannot finish its exit handshake until the peer is submitted.
            ready, _, _ = select.select([started_reader], [], [], 30.0)
            assert ready, "rank 0 did not begin the first CCU run"
            assert os.read(started_reader, 1) == b"1"
            submit(1, 0, first_rank0)
            first_rank1 = submit(0, 1)
            submit(1, 1, first_rank1)

        worker.run(interleave)
        for index, callable_outputs in enumerate(outputs):
            for rank, rank_outputs in enumerate(callable_outputs):
                value = (index + 1) * 100 + rank * 10
                assert [struct.unpack_from("<Q", output.shm.buf)[0] for output in rank_outputs] == [
                    value + 2,
                    value + 1,
                    value,
                ]
                assert struct.unpack_from("<Q", rank_outputs[0].shm.buf, 8)[0] == value + 3
    finally:
        try:
            worker.close()
        finally:
            os.close(started_reader)
            os.close(started_writer)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize("fault", ["missing_done", "missing_trigger", "missing_exit"])
def test_mission_failure_closes_admission_and_retains_domain(st_platform, st_device_ids, tmp_path, monkeypatch, fault):
    log_root = tmp_path / "ascend"
    log_root.mkdir()
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(log_root))
    target = _build_callable(tmp_path)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    retained = []
    child_pids = set()
    exit_reader, exit_writer = os.pipe()
    os.set_blocking(exit_reader, False)
    child_exit = os._exit

    def observe_child_exit(code):
        try:
            os.write(exit_writer, f"{os.getpid()}:{code}\n".encode())
        finally:
            child_exit(code)

    # The native scheduler may reap first; a signal never reaches os._exit.
    monkeypatch.setattr(os, "_exit", observe_child_exit)

    if fault == "missing_exit":

        def short_exit_wait(chip_worker, communicator, registration, _timeout):
            return ChipCcuBinding(chip_worker, communicator, registration, 10)

        monkeypatch.setattr("simpler.ccu.ChipCcuBinding", short_exit_wait)
    try:
        callable_handle = worker.register(target)
        worker.init()
        worker.register_ccu_callable(callable_handle)
        child_pids.update(worker._chip_pids)
        assert len(child_pids) == 2
        outputs = [[worker.create_buffer(64) for _ in range(3)] for _ in range(2)]
        for rank_outputs in outputs:
            for output in rank_outputs:
                output.shm.buf[:] = bytes(64)

        def fail_round(orch, _args, config):
            domain = orch.allocate_domain(
                name="failed_ccu",
                members=[0, 1],
                window_size=4 * 1024 * 1024,
                workspace_engines=("ccu",),
                retain_after_run=True,
            )
            retained.append(domain)
            for rank in range(2):
                args = TaskArgs()
                for output in outputs[rank]:
                    args.add_tensor(output.tensor((8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_context(rank))
                args.add_scalar({"missing_done": (1 << 64) - 2, "missing_trigger": 2, "missing_exit": 1}[fault])
                orch.submit_next_level(callable_handle, args, config, worker=rank)

        expected_error = "CCU exit protocol timed out" if fault == "missing_exit" else None
        with pytest.raises(RuntimeError, match=expected_error):
            worker.run(fail_round)
        assert retained and not retained[0].freed
        assert worker._ordered_cleanup_error is not None
        for device in st_device_ids[:2]:
            logs = list(log_root.glob(f"debug/device-{device}/device-*.log"))
            assert logs, f"no device logs for {fault} on device {device} under {log_root}"
            evidence = "\n".join(log.read_text(errors="replace") for log in logs)
            assert evidence.strip(), f"empty device logs for {fault} on device {device} under {log_root}"
            assert (fault != "missing_exit") == any(
                all(
                    field in line
                    for field in (
                        "ASYNC_WAIT reason=scheduler_timeout",
                        "normal_done=1",
                        "engine=CCU type=4",
                        "cookie=0x1",
                        "retired=0",
                    )
                )
                for line in evidence.splitlines()
            ), f"unexpected scheduler timeout evidence for {fault}"
        with pytest.raises(RuntimeError):
            worker.run(lambda *_: None)
        assert not retained[0].freed
        if fault == "missing_exit":
            for rank_outputs in outputs:
                assert struct.unpack_from("<Q", rank_outputs[2].shm.buf)[0] == 1
        print(f"CCU {fault} observed; admission closed and domain retained", flush=True)
    finally:
        exit_records = bytearray()
        try:
            try:
                worker.close()
            except RuntimeError:
                # Whole-tree teardown may report the original unconfirmed run error.
                assert worker._ordered_cleanup_error is not None
            try:
                while chunk := os.read(exit_reader, 4096):
                    exit_records.extend(chunk)
            except BlockingIOError:
                pass
        finally:
            os.close(exit_reader)
            os.close(exit_writer)
        child_exits = {
            int(pid): int(code) for record in exit_records.decode().splitlines() for pid, code in [record.split(":")]
        }
        assert child_pids <= child_exits.keys(), "CCU chip child crashed or did not reach controlled exit"
        for pid in child_pids:
            assert child_exits[pid] in (0, 1), f"CCU chip child {pid} has unexpected exit code {child_exits[pid]}"
        print(f"CCU chip children reached controlled exit: {child_exits}", flush=True)
