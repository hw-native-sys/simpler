# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import hashlib
import os
import socket
import subprocess
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from typing import cast

import pytest
from simpler import callable_identity
from simpler.callable_identity import (
    CallableHandle,
    CallableKindName,
    TargetNamespaceName,
    build_chip_callable_descriptor,
    build_python_import_descriptor,
    build_python_serialized_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
    parse_python_import_target,
    validate_hashid,
)
from simpler.orchestrator import Orchestrator
from simpler.remote_l3_protocol import (
    CallableKind,
    ChipCallableBlobLocation,
    RemoteChipCallablePayload,
    RemoteRegistryTarget,
    encode_register_callable_command,
    encode_remote_chip_callable_payload,
)
from simpler.remote_l3_session import (
    _install_manifest_inner_registry,
    _prepare_register_callable,
    _unpublish_inner_handle,
    get_inner_handle,
)
from simpler.task_interface import (
    ChipCallable,
    ContinuousTensor,
    DataType,
    RemoteAddressSpace,
    RemoteBufferExport,
    RemoteBufferHandle,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import (
    RemoteCallable,
    RemoteWorkerSpec,
    Worker,
    _pack_py_callable_payload,
    _read_raw_payload_from_shm,
)


def _py_target(args):
    return args


def _remote_noop_orch(orch, args, cfg):
    return None


def _remote_raises_orch(orch, args, cfg):
    raise RuntimeError("remote boom")


def _remote_sleep_orch(orch, args, cfg):
    time.sleep(1.0)


def _remote_increment_u8_orch(orch, args, cfg):
    tensor = args.tensor(0)
    data = (ctypes.c_ubyte * tensor.nbytes()).from_address(tensor.data)
    for i in range(tensor.nbytes()):
        data[i] = (int(data[i]) + 1) & 0xFF


def _remote_fail_before_write_orch(orch, args, cfg):
    raise RuntimeError("remote producer failed before write")


def _remote_mark_u8_orch(orch, args, cfg):
    tensor = args.tensor(0)
    data = (ctypes.c_ubyte * tensor.nbytes()).from_address(tensor.data)
    for i in range(tensor.nbytes()):
        data[i] = 99


def _remote_exit_orch(orch, args, cfg):
    os._exit(70)


def _remote_sum_u8_orch(orch, args, cfg):
    src = args.tensor(0)
    dst = args.tensor(1)
    src_data = (ctypes.c_ubyte * src.nbytes()).from_address(src.data)
    dst_data = (ctypes.c_ubyte * dst.nbytes()).from_address(dst.data)
    dst_data[0] = sum(int(src_data[i]) for i in range(src.nbytes())) & 0xFF


def _remote_inner_sub_noop(args):
    if args.scalar_count() != 1 or args.scalar(0) != 17:
        raise RuntimeError("inner sub args mismatch")


_INNER_SUB_HASHID = compute_callable_hashid(
    build_python_import_descriptor("tests.ut.py.test_callable_identity", "_remote_inner_sub_noop")
)


def _remote_submit_inner_sub_orch(orch, args, cfg):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    sub_args = TaskArgs()
    sub_args.add_scalar(17)
    orch.submit_sub(get_inner_handle(_INNER_SUB_HASHID), sub_args)


def _remote_chip_register_payload(chip: ChipCallable, *, platform: str, runtime: str) -> tuple[bytes, bytes]:
    blob = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    descriptor = build_chip_callable_descriptor(
        target=chip,
        platform=platform,
        runtime=runtime,
    )
    digest = hashid_to_digest(compute_callable_hashid(descriptor))
    payload = encode_remote_chip_callable_payload(
        RemoteChipCallablePayload(
            descriptor_bytes=descriptor,
            blob_location=ChipCallableBlobLocation.INLINE_BLOB,
            blob_size=len(blob),
            blob_sha256=hashlib.sha256(blob).digest(),
            inline_blob=blob,
            staged_blob_token=b"",
        )
    )
    return digest, payload


def _free_tcp_port() -> int:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError as exc:
        pytest.skip(f"local TCP sockets are not permitted in this sandbox: {exc}")
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_python_descriptor_hash_is_stable_for_same_serialized_payload():
    payload = _pack_py_callable_payload(_py_target)
    descriptor = build_python_serialized_descriptor(payload)

    hashid = compute_callable_hashid(descriptor)

    assert hashid == compute_callable_hashid(build_python_serialized_descriptor(payload))
    assert len(hashid_to_digest(hashid)) == 32


def test_chip_descriptor_changes_when_callable_blob_changes():
    first = ChipCallable.build(signature=[], func_name="x", binary=b"\x01", children=[])
    second = ChipCallable.build(signature=[], func_name="y", binary=b"\x02", children=[])

    assert compute_callable_hashid(build_chip_callable_descriptor(target=first)) != compute_callable_hashid(
        build_chip_callable_descriptor(target=second)
    )


def test_remote_inner_chip_callable_payload_validates_descriptor_and_context():
    chip = ChipCallable.build(signature=[], func_name="x", binary=b"\x01", children=[])
    digest, payload = _remote_chip_register_payload(
        chip,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
    )
    command = encode_register_callable_command(
        RemoteRegistryTarget.INNER_L3_WORKER,
        CallableKind.CHIP_CALLABLE,
        digest,
        1,
        payload,
    )

    got_digest, got_kind, got_registry, got_target = _prepare_register_callable(
        command,
        {"platform": "a2a3sim", "runtime": "tensormap_and_ringbuffer"},
    )

    assert got_digest == digest
    assert got_kind == CallableKind.CHIP_CALLABLE
    assert got_registry == RemoteRegistryTarget.INNER_L3_WORKER
    assert isinstance(got_target, ChipCallable)


def test_remote_dispatcher_rejects_chip_callable_target():
    command = encode_register_callable_command(
        RemoteRegistryTarget.REMOTE_TASK_DISPATCHER,
        CallableKind.CHIP_CALLABLE,
        b"\x00" * 32,
        1,
        b"",
    )

    with pytest.raises(ValueError, match="REMOTE_TASK_DISPATCHER only accepts PYTHON_IMPORT"):
        _prepare_register_callable(command, {"platform": "a2a3sim", "runtime": "tensormap_and_ringbuffer"})


def test_remote_register_rejects_python_serialized_without_negotiation():
    command = encode_register_callable_command(
        RemoteRegistryTarget.REMOTE_TASK_DISPATCHER,
        CallableKind.PYTHON_SERIALIZED,
        b"\x00" * 32,
        1,
        b"serialized",
    )

    with pytest.raises(ValueError, match="PYTHON_SERIALIZED is not negotiated"):
        _prepare_register_callable(command, {"platform": "a2a3sim", "runtime": "tensormap_and_ringbuffer"})


def test_remote_inner_chip_callable_rejects_staged_blob_without_negotiation():
    payload = encode_remote_chip_callable_payload(
        RemoteChipCallablePayload(
            descriptor_bytes=b"descriptor",
            blob_location=ChipCallableBlobLocation.STAGED_BLOB,
            blob_size=4,
            blob_sha256=hashlib.sha256(b"abcd").digest(),
            inline_blob=b"",
            staged_blob_token=b"token",
        )
    )
    command = encode_register_callable_command(
        RemoteRegistryTarget.INNER_L3_WORKER,
        CallableKind.CHIP_CALLABLE,
        b"\x00" * 32,
        1,
        payload,
    )

    with pytest.raises(ValueError, match="STAGED_BLOB is unsupported"):
        _prepare_register_callable(command, {"platform": "a2a3sim", "runtime": "tensormap_and_ringbuffer"})


def test_remote_manifest_inner_python_import_installs_session_handle():
    worker = Worker(level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", num_sub_workers=1)
    digest = hashid_to_digest(_INNER_SUB_HASHID)
    handles = {}
    try:
        worker.init()
        handles = _install_manifest_inner_registry(
            {
                "platform": "a2a3sim",
                "runtime": "tensormap_and_ringbuffer",
                "inner_l3_worker": [
                    {
                        "hashid": digest.hex(),
                        "kind": "PYTHON_IMPORT",
                        "target_registry": "INNER_L3_WORKER",
                        "target": "tests.ut.py.test_callable_identity:_remote_inner_sub_noop",
                    }
                ],
            },
            worker,
        )

        handle = get_inner_handle(_INNER_SUB_HASHID)

        assert handles[(CallableKind.PYTHON_IMPORT, digest)] is handle
        assert handle.kind == "PYTHON_IMPORT"
        assert handle.target_namespace == "LOCAL_PYTHON"
    finally:
        for handle in handles.values():
            try:
                worker.unregister(handle)
            except Exception:
                pass
        _unpublish_inner_handle(digest)
        worker.close()


def test_remote_manifest_inner_chip_callable_installs_session_handle():
    chip = ChipCallable.build(signature=[], func_name="x", binary=b"\x01", children=[])
    digest, payload = _remote_chip_register_payload(
        chip,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
    )
    worker = Worker(level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", num_sub_workers=0)
    handles = {}
    try:
        worker.init()
        handles = _install_manifest_inner_registry(
            {
                "platform": "a2a3sim",
                "runtime": "tensormap_and_ringbuffer",
                "inner_l3_worker": [
                    {
                        "hashid": digest.hex(),
                        "kind": "CHIP_CALLABLE",
                        "target_registry": "INNER_L3_WORKER",
                        "payload_version": 1,
                        "payload_hex": payload.hex(),
                    }
                ],
            },
            worker,
        )

        handle = get_inner_handle(digest.hex())

        assert handles[(CallableKind.CHIP_CALLABLE, digest)] is handle
        assert handle.kind == "CHIP_CALLABLE"
        assert handle.target_namespace == "LOCAL_CHIP"
    finally:
        for handle in handles.values():
            try:
                worker.unregister(handle)
            except Exception:
                pass
        _unpublish_inner_handle(digest)
        worker.close()


def test_python_import_descriptor_hash_is_stable():
    module, qualname = parse_python_import_target("pkg.mod:Class.method")
    descriptor = build_python_import_descriptor(module, qualname)

    assert compute_callable_hashid(descriptor) == compute_callable_hashid(
        build_python_import_descriptor("pkg.mod", "Class.method")
    )


def test_raw_control_payload_uses_explicit_size():
    payload = b"tests.ut.py.test_callable_identity:_remote_inner_sub_noop"
    shm = SharedMemory(create=True, size=len(payload) + 16)
    shm_buf = None
    try:
        shm_buf = shm.buf
        assert shm_buf is not None
        shm_buf[: len(payload)] = payload
        shm_buf[len(payload) :] = b"\x00" * (len(shm_buf) - len(payload))

        assert _read_raw_payload_from_shm(shm.name, len(payload)) == payload
    finally:
        if shm_buf is not None:
            shm_buf.release()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize("target", ["pkg.mod.fn", ":fn", "pkg:", ".pkg:fn", "pkg.mod:<locals>.fn", "pkg.mod:bad-name"])
def test_python_import_target_validation_rejects_invalid_targets(target):
    with pytest.raises((TypeError, ValueError)):
        RemoteCallable(target)


@pytest.mark.parametrize("hashid", ["", "sha256:ABC", "md5:" + "0" * 64, "sha256:" + "0" * 63])
def test_hashid_validation_rejects_noncanonical_values(hashid):
    with pytest.raises(ValueError, match="HASHID_FORMAT_INVALID"):
        validate_hashid(hashid)


def test_callable_identity_public_exports_do_not_include_worker_state():
    assert "CallableHandle" in callable_identity.__all__
    assert "_CallableIdentityState" not in callable_identity.__all__


def test_worker_register_returns_opaque_handle_and_deduplicates_same_identity():
    worker = Worker(level=3, num_sub_workers=0)
    try:
        first = worker.register(_py_target)
        second = worker.register(_py_target)

        assert isinstance(first, CallableHandle)
        assert not isinstance(first, int)
        assert first.hashid == second.hashid
        assert first.digest == second.digest
        assert first._handle_id != second._handle_id
        assert worker._identity_registry[first.digest].slot_id >= 0
        assert len(worker._callable_registry) == 1
        assert worker._identity_registry[first.digest].ref_count == 2
    finally:
        worker.close()


def test_remote_callable_register_requires_explicit_remote_workers():
    worker = Worker(level=4, num_sub_workers=0)
    try:
        with pytest.raises(RuntimeError, match="add at least one remote worker"):
            worker.register(RemoteCallable("pkg.remote:orch"), workers=[0])

        endpoint = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        with pytest.raises(ValueError, match="workers must be an explicit non-empty list"):
            worker.register(RemoteCallable("pkg.remote:orch"))

        handle = worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint])
        state = worker._resolve_handle(handle)
        assert handle.kind == "PYTHON_IMPORT"
        assert handle.target_namespace == "REMOTE_TASK_DISPATCHER"
        assert state.eligible_endpoint_ids == (endpoint,)
        assert state.slot_id == -1
        assert worker._callable_registry == {}
    finally:
        worker.close()


def test_remote_callable_submit_passes_remote_sidecar_to_cpp_facade():
    class FakeCOrchestrator:
        def submit_next_level(self, *args):
            self.submit_next_level_args = args

        def submit_next_level_group(self, *args):
            self.submit_next_level_group_args = args

    worker = Worker(level=4, num_sub_workers=0)
    try:
        endpoint = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        handle = worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint])
        fake = FakeCOrchestrator()
        orch = Orchestrator(fake, worker=worker)  # type: ignore[arg-type]

        buf = RemoteBufferHandle._from_remote_allocation(
            endpoint_id=endpoint,
            buffer_id=1,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=16,
        )
        args = TaskArgs()
        args.add_tensor(RemoteTensorRef(buf, shape=(4,), dtype=DataType.UINT8), TensorArgType.INPUT)
        orch.submit_next_level(handle, args)

        call = fake.submit_next_level_args
        assert call[1] == "PYTHON_IMPORT"
        assert call[2] == "REMOTE_TASK_DISPATCHER"
        assert call[6] == [endpoint]
        sidecar = call[7]
        assert len(sidecar.tensors) == 1
        assert sidecar.tensors[0].present
        assert sidecar.tensors[0].desc.owner_endpoint_id == endpoint

        bare = TaskArgs()
        bare.add_tensor(ContinuousTensor.make(0x1234, (1,), DataType.UINT8), TensorArgType.INPUT)
        orch.submit_next_level(handle, bare)
        bare_sidecar = fake.submit_next_level_args[7]
        assert len(bare_sidecar.tensors) == 1
        assert bare_sidecar.tensors[0] is None
    finally:
        worker.close()


def test_remote_callable_submit_intersects_remote_buffer_owner_endpoint():
    class FakeCOrchestrator:
        def submit_next_level(self, *args):
            self.submit_next_level_args = args

        def submit_next_level_group(self, *args):
            self.submit_next_level_group_args = args

    worker = Worker(level=4, num_sub_workers=0)
    try:
        endpoint0 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        endpoint1 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19074", platform="a2a3sim"))
        handle = worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint0, endpoint1])
        fake = FakeCOrchestrator()
        orch = Orchestrator(fake, worker=worker)  # type: ignore[arg-type]

        buf = RemoteBufferHandle._from_remote_allocation(
            endpoint_id=endpoint1,
            buffer_id=1,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=16,
        )
        args = TaskArgs()
        args.add_tensor(RemoteTensorRef(buf, shape=(4,), dtype=DataType.UINT8), TensorArgType.INPUT)
        orch.submit_next_level(handle, args)

        assert fake.submit_next_level_args[6] == [endpoint1]
    finally:
        worker.close()


def test_remote_callable_submit_rejects_remote_buffer_outside_callable_endpoints():
    class FakeCOrchestrator:
        def submit_next_level(self, *args):
            self.submit_next_level_args = args

        def submit_next_level_group(self, *args):
            self.submit_next_level_group_args = args

    worker = Worker(level=4, num_sub_workers=0)
    try:
        endpoint0 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        endpoint1 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19074", platform="a2a3sim"))
        handle = worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint0])
        fake = FakeCOrchestrator()
        orch = Orchestrator(fake, worker=worker)  # type: ignore[arg-type]

        buf = RemoteBufferHandle._from_remote_allocation(
            endpoint_id=endpoint1,
            buffer_id=1,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=16,
        )
        args = TaskArgs()
        args.add_tensor(RemoteTensorRef(buf, shape=(4,), dtype=DataType.UINT8), TensorArgType.INPUT)

        with pytest.raises(ValueError, match="no eligible remote endpoint"):
            orch.submit_next_level(handle, args)
    finally:
        worker.close()


def test_remote_callable_group_submit_intersects_each_member_endpoint_set():
    class FakeCOrchestrator:
        def submit_next_level(self, *args):
            self.submit_next_level_args = args

        def submit_next_level_group(self, *args):
            self.submit_next_level_group_args = args

    worker = Worker(level=4, num_sub_workers=0)
    try:
        endpoint0 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        endpoint1 = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19074", platform="a2a3sim"))
        handle = worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint0, endpoint1])
        fake = FakeCOrchestrator()
        orch = Orchestrator(fake, worker=worker)  # type: ignore[arg-type]

        buf0 = RemoteBufferHandle._from_remote_allocation(
            endpoint_id=endpoint0,
            buffer_id=1,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=16,
        )
        buf1 = RemoteBufferHandle._from_remote_allocation(
            endpoint_id=endpoint1,
            buffer_id=2,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=16,
        )
        args0 = TaskArgs()
        args0.add_tensor(RemoteTensorRef(buf0, shape=(4,), dtype=DataType.UINT8), TensorArgType.INPUT)
        args1 = TaskArgs()
        args1.add_tensor(RemoteTensorRef(buf1, shape=(4,), dtype=DataType.UINT8), TensorArgType.INPUT)
        orch.submit_next_level_group(handle, [args0, args1])

        assert fake.submit_next_level_group_args[6] == [[endpoint0], [endpoint1]]
    finally:
        worker.close()


def test_remote_worker_requires_reachable_daemon_before_endpoint_registration():
    worker = Worker(level=4, num_sub_workers=0)
    try:
        endpoint = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:1", platform="a2a3sim"))
        worker.register(RemoteCallable("pkg.remote:orch"), workers=[endpoint])
        with pytest.raises((ConnectionRefusedError, OSError, RuntimeError)):
            worker.init()
    finally:
        worker.close()


def test_remote_sim_noop_task_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_noop_orch"),
            workers=[endpoint],
        )
        worker.init()

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_prepare_callable_control_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_noop_orch"),
            workers=[endpoint],
        )
        worker.init()
        worker._start_hierarchical()  # noqa: SLF001 -- exercise PREPARE_CALLABLE before TASK dispatch.
        assert worker._worker is not None
        worker._worker.control_prepare(endpoint, handle.digest)

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_error_completion_raises_root_error():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_raises_orch"),
            workers=[endpoint],
        )
        worker.init()

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        with pytest.raises(RuntimeError, match="remote boom"):
            worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_post_init_register_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        worker.init()
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_noop_orch"),
            workers=[endpoint],
        )

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_unregister_then_reregister_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        worker.init()

        def run_handle(handle):
            def parent_orch(orch, _args, cfg):
                orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

            worker.run(parent_orch)

        first = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_noop_orch"),
            workers=[endpoint],
        )
        run_handle(first)
        worker.unregister(first)

        second = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_noop_orch"),
            workers=[endpoint],
        )
        run_handle(second)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_health_lane_stays_live_during_long_task():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=5)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_sleep_orch"),
            workers=[endpoint],
        )
        worker.init()

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_inner_python_import_register_runs_sub_task():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    inner_committed = False
    endpoint = -1
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"127.0.0.1:{port}",
                platform="a2a3sim",
                transport="sim",
                num_sub_workers=1,
            )
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_submit_inner_sub_orch"),
            workers=[endpoint],
        )
        worker.init()
        worker._start_hierarchical()
        assert worker._worker is not None
        inner_digest = hashid_to_digest(_INNER_SUB_HASHID)
        target = b"tests.ut.py.test_callable_identity:_remote_inner_sub_noop"
        result = worker._worker.remote_prepare_register(
            endpoint,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            target,
            inner_digest,
        )
        assert result.ok, result.error_message
        result = worker._worker.remote_commit_register(
            endpoint,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            inner_digest,
        )
        assert result.ok, result.error_message
        inner_committed = True

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        worker.run(parent_orch)

        result = worker._worker.remote_unregister(
            endpoint,
            "INNER_L3_WORKER",
            "PYTHON_IMPORT",
            inner_digest,
        )
        assert result.ok, result.error_message
        inner_committed = False
    finally:
        if inner_committed and worker._worker is not None:
            try:
                worker._worker.remote_unregister(
                    endpoint,
                    "INNER_L3_WORKER",
                    "PYTHON_IMPORT",
                    hashid_to_digest(_INNER_SUB_HASHID),
                )
            except Exception:
                pass
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_buffer_copy_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_increment_u8_orch"),
            workers=[endpoint],
        )
        worker.init()

        remote_buf = worker.remote_malloc(worker=endpoint, nbytes=4)
        src = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        worker.remote_copy_to(remote_buf, src, 4)

        def parent_orch(orch, _args, cfg):
            task_args = TaskArgs()
            task_args.add_tensor(
                RemoteTensorRef(remote_buf, shape=(4,), dtype=DataType.UINT8),
                TensorArgType.INOUT,
            )
            orch.submit_next_level(handle, task_args, cfg, worker=endpoint)

        worker.run(parent_orch)

        dst = (ctypes.c_ubyte * 4)()
        worker.remote_copy_from(remote_buf, dst, 4)
        assert bytes(dst) == b"\x02\x03\x04\x05"
        worker.remote_free(remote_buf)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_imported_buffer_runs_on_peer_endpoint():
    owner_port = _free_tcp_port()
    peer_port = _free_tcp_port()
    owner_daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(owner_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    peer_daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(peer_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        owner_endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{owner_port}", platform="a2a3sim", transport="sim")
        )
        peer_endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{peer_port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_increment_u8_orch"),
            workers=[peer_endpoint],
        )
        worker.init()

        owner_buf = worker.remote_malloc(worker=owner_endpoint, nbytes=4)
        src = (ctypes.c_ubyte * 4)(10, 11, 12, 13)
        worker.remote_copy_to(owner_buf, src, 4)
        exported = worker.remote_export(owner_buf, access="readwrite")
        peer_buf = worker.remote_import(exported, worker=peer_endpoint)

        def parent_orch(orch, _args, cfg):
            task_args = TaskArgs()
            task_args.add_tensor(
                RemoteTensorRef(peer_buf, shape=(4,), dtype=DataType.UINT8),
                TensorArgType.INOUT,
            )
            orch.submit_next_level(handle, task_args, cfg, worker=peer_endpoint)

        worker.run(parent_orch)

        worker.remote_release_import(peer_buf)
        dst = (ctypes.c_ubyte * 4)()
        worker.remote_copy_from(owner_buf, dst, 4)
        assert bytes(dst) == b"\x0b\x0c\x0d\x0e"
        worker.remote_free(owner_buf)
    finally:
        worker.close()
        for daemon in (owner_daemon, peer_daemon):
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)


def test_remote_owner_free_waits_for_import_release():
    worker = Worker(level=4, num_sub_workers=0)
    owner = RemoteBufferHandle._from_remote_allocation(
        endpoint_id=0,
        buffer_id=1,
        generation=1,
        address_space=RemoteAddressSpace.REMOTE_DEVICE,
        nbytes=4,
    )
    exported = RemoteBufferExport(
        owner_endpoint_id=0,
        buffer_id=1,
        generation=1,
        address_space=RemoteAddressSpace.REMOTE_WINDOW,
        offset=0,
        nbytes=4,
        export_id=1,
        remote_addr=0,
        rkey_or_token=1,
        ub_ldst_va=0,
        access_flags=3,
        transport_profile="sim",
        _owner_handle=owner,
    )

    class FakeRemoteWorker:
        def remote_import(self, *args):
            return (1, 0, 1, 1, 7, int(RemoteAddressSpace.REMOTE_WINDOW), 4, 0, 0, 7, 0, 3)

        def remote_release_import(self, *args):
            self.released = args

        def remote_free(self, *args):
            self.freed = args

    fake = FakeRemoteWorker()
    fake.released = None
    fake.freed = None
    worker._worker = fake
    worker._initialized = True
    worker._hierarchical_start_state = "started"
    worker._remote_worker_specs = [object(), object()]

    imported = worker.remote_import(exported, worker=1)
    worker.remote_free(owner)
    assert owner.released
    assert fake.freed is None
    assert worker._pending_remote_buffer_frees == [owner]

    worker.remote_release_import(imported)
    assert fake.released == (1, 0, 1, 1, 7)
    assert fake.freed == (0, 1, 1)
    assert worker._pending_remote_buffer_frees == []


def test_remote_sim_failed_dependency_skips_consumer():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        fail_handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_fail_before_write_orch"),
            workers=[endpoint],
        )
        mark_handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_mark_u8_orch"),
            workers=[endpoint],
        )
        worker.init()

        remote_buf = worker.remote_malloc(worker=endpoint, nbytes=4)
        src = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        worker.remote_copy_to(remote_buf, src, 4)

        def parent_orch(orch, _args, cfg):
            producer_args = TaskArgs()
            producer_args.add_tensor(
                RemoteTensorRef(remote_buf, shape=(4,), dtype=DataType.UINT8),
                TensorArgType.OUTPUT,
            )
            consumer_args = TaskArgs()
            consumer_args.add_tensor(
                RemoteTensorRef(remote_buf, shape=(4,), dtype=DataType.UINT8),
                TensorArgType.INPUT,
            )
            orch.submit_next_level(fail_handle, producer_args, cfg, worker=endpoint)
            orch.submit_next_level(mark_handle, consumer_args, cfg, worker=endpoint)

        with pytest.raises(RuntimeError, match="remote producer failed before write"):
            worker.run(parent_orch)

        dst = (ctypes.c_ubyte * 4)()
        worker.remote_copy_from(remote_buf, dst, 4)
        assert bytes(dst) == b"\x01\x02\x03\x04"
        worker.remote_free(remote_buf)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_session_exit_becomes_endpoint_failure():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=3)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_exit_orch"),
            workers=[endpoint],
        )
        worker.init()

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(handle, TaskArgs(), cfg, worker=endpoint)

        with pytest.raises(RuntimeError, match="RemoteL3Endpoint::run|socket closed|health lane"):
            worker.run(parent_orch)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_input_free_is_deferred_until_slot_refs_drop():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_sum_u8_orch"),
            workers=[endpoint],
        )
        worker.init()

        remote_in = worker.remote_malloc(worker=endpoint, nbytes=4)
        remote_out = worker.remote_malloc(worker=endpoint, nbytes=1)
        src = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        worker.remote_copy_to(remote_in, src, 4)

        def parent_orch(orch, _args, cfg):
            task_args = TaskArgs()
            task_args.add_tensor(
                RemoteTensorRef(remote_in, shape=(4,), dtype=DataType.UINT8),
                TensorArgType.INPUT,
            )
            task_args.add_tensor(
                RemoteTensorRef(remote_out, shape=(1,), dtype=DataType.UINT8),
                TensorArgType.OUTPUT,
            )
            orch.submit_next_level(handle, task_args, cfg, worker=endpoint)
            worker.remote_free(remote_in)

        worker.run(parent_orch)
        assert remote_in.released

        dst = (ctypes.c_ubyte * 1)()
        worker.remote_copy_from(remote_out, dst, 1)
        assert bytes(dst) == b"\x0a"
        worker.remote_free(remote_out)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_remote_sim_host_inline_descriptor_roundtrip():
    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=10)
    try:
        time.sleep(0.3)
        endpoint = worker.add_remote_worker(
            RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim", transport="sim")
        )
        handle = worker.register(
            RemoteCallable("tests.ut.py.test_callable_identity:_remote_sum_u8_orch"),
            workers=[endpoint],
        )
        worker.init()

        remote_out = worker.remote_malloc(worker=endpoint, nbytes=1)

        def parent_orch(orch, _args, cfg):
            task_args = TaskArgs()
            task_args.add_tensor(
                RemoteTensorRef.host_inline(b"\x01\x02\x03\x04", shape=(4,), dtype=DataType.UINT8),
                TensorArgType.INPUT,
            )
            task_args.add_tensor(
                RemoteTensorRef(remote_out, shape=(1,), dtype=DataType.UINT8),
                TensorArgType.OUTPUT,
            )
            orch.submit_next_level(handle, task_args, cfg, worker=endpoint)

        worker.run(parent_orch)

        dst = (ctypes.c_ubyte * 1)()
        worker.remote_copy_from(remote_out, dst, 1)
        assert bytes(dst) == b"\x0a"
        worker.remote_free(remote_out)
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_worker_remote_memory_api_returns_opaque_handle_and_routes_controls():
    class FakeRemoteCWorker:
        def __init__(self):
            self.calls = []

        def remote_malloc(self, endpoint_id, size):
            self.calls.append(("malloc", endpoint_id, size))
            return (endpoint_id, 7, 1, int(RemoteAddressSpace.REMOTE_DEVICE), size, 0xCAFE, 0xBEEF, 0)

        def remote_copy_to(self, endpoint_id, buffer_id, generation, offset, src, size, handle_nbytes):
            self.calls.append(("copy_to", endpoint_id, buffer_id, generation, offset, src, size, handle_nbytes))

        def remote_copy_from(self, dst, endpoint_id, buffer_id, generation, offset, size, handle_nbytes):
            self.calls.append(("copy_from", dst, endpoint_id, buffer_id, generation, offset, size, handle_nbytes))

        def remote_free(self, endpoint_id, buffer_id, generation):
            self.calls.append(("free", endpoint_id, buffer_id, generation))

        def close(self):
            self.calls.append(("close",))

    worker = Worker(level=4, num_sub_workers=0)
    endpoint = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
    fake = FakeRemoteCWorker()
    worker._initialized = True
    worker._worker = fake  # type: ignore[assignment]
    worker._hierarchical_started = True
    worker._hierarchical_start_state = "started"
    try:
        handle = worker.remote_malloc(worker=endpoint, nbytes=4)
        assert handle.endpoint_id == endpoint
        assert handle.nbytes == 4
        assert not hasattr(handle, "rkey_or_token")

        src = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        dst = (ctypes.c_ubyte * 4)()
        worker.remote_copy_to(handle, src, 4)
        worker.remote_copy_from(handle, dst, 4)
        worker.remote_free(handle)

        assert handle.released
        assert fake.calls[:4] == [
            ("malloc", endpoint, 4),
            ("copy_to", endpoint, 7, 1, 0, ctypes.addressof(src), 4, 4),
            ("copy_from", ctypes.addressof(dst), endpoint, 7, 1, 0, 4, 4),
            ("free", endpoint, 7, 1),
        ]
        with pytest.raises(RuntimeError, match="released"):
            worker.remote_copy_to(handle, src, 1)

        deferred = worker.remote_malloc(worker=endpoint, nbytes=8)
        deferred._acquire_slot_ref()
        worker.remote_free(deferred)
        assert deferred.released
        assert ("free", endpoint, 7, 1) in fake.calls
        free_count = fake.calls.count(("free", endpoint, 7, 1))
        deferred._release_slot_ref()
        worker._flush_pending_remote_frees()
        assert fake.calls.count(("free", endpoint, 7, 1)) == free_count + 1
    finally:
        worker.close()


def test_callable_handle_public_constructor_returns_unbound_handle():
    handle = CallableHandle(
        "sha256:" + "0" * 64,
        cast(CallableKindName, "PYTHON_SERIALIZED"),
        cast(TargetNamespaceName, "LOCAL_PYTHON"),
    )

    assert handle.digest == bytes(32)
    assert handle._handle_id == -1
    assert handle._owner_id is None
    assert not hasattr(handle, "slot_id")
    assert not hasattr(handle, "cid")


def test_forged_public_handle_is_rejected_by_worker_apis():
    worker = Worker(level=3, num_sub_workers=0)
    real = worker.register(_py_target)
    forged = CallableHandle(
        real.hashid,
        cast(CallableKindName, real.kind),
        cast(TargetNamespaceName, real.target_namespace),
    )
    try:
        with pytest.raises(KeyError, match="does not belong|not live"):
            worker.unregister(forged)
        with pytest.raises(KeyError, match="does not belong|not live"):
            worker._resolve_handle(forged)
    finally:
        worker.close()


def test_mutated_handle_fields_are_rejected():
    worker = Worker(level=3, num_sub_workers=0)
    handle = worker.register(_py_target)
    try:
        handle.kind = "CHIP_CALLABLE"
        with pytest.raises(RuntimeError, match="CALLABLE_HANDLE_MUTATED"):
            worker._resolve_handle(handle)
    finally:
        worker.close()


def test_uncertain_cleanup_hashid_blocks_live_handle_resolution():
    worker = Worker(level=3, num_sub_workers=0)
    handle = worker.register(_py_target)
    try:
        worker._uncertain_hashids.add(handle.digest)
        with pytest.raises(RuntimeError, match="REGISTER_CLEANUP_UNCERTAIN"):
            worker._resolve_handle(handle)
    finally:
        worker.close()
