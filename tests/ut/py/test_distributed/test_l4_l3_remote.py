import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

from simpler.distributed.l3_daemon import L3Daemon
from simpler.distributed.catalog import Catalog
from simpler.distributed.proto import dispatch_pb2
from simpler.distributed.remote_proxy import RemoteUnavailable, RemoteWorkerProxy
from simpler.distributed.transport_backend import TransportBackendError, TransportUnavailable
from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


def _scalar_value(args: TaskArgs) -> int:
    return int(args.scalar(0)) if args is not None and args.scalar_count() else 1


def _make_shared_counter():
    shm = SharedMemory(create=True, size=4)
    buf = shm.buf
    assert buf is not None
    struct.pack_into("i", buf, 0, 0)
    return shm, buf


def _read_counter(buf) -> int:
    return struct.unpack_from("i", buf, 0)[0]


def _increment_counter(buf) -> None:
    value = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, value + 1)


def _start_daemon():
    daemon = L3Daemon(0, lambda: Worker(level=3, num_sub_workers=1))
    port = daemon.start()
    return daemon, f"127.0.0.1:{port}"


def _make_file_counter(path):
    path.write_text("0")

    def read() -> int:
        return int(path.read_text())

    def add(amount: int) -> None:
        path.write_text(str(read() + int(amount)))

    return read, add


def test_l4_remote_init_close_no_dispatch():
    daemon, endpoint = _start_daemon()
    try:
        w4 = Worker(level=4, num_sub_workers=0)
        w4.add_remote_worker(endpoint)
        w4.init()
        w4.close()
    finally:
        daemon.stop()


def test_l4_remote_single_dispatch(tmp_path):
    read_counter, add_counter = _make_file_counter(tmp_path / "remote_counter.txt")
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)
        def l3_sub(args):
            add_counter(_scalar_value(args))

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid, args)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            sub_args.add_scalar(3)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert read_counter() == 3
    finally:
        daemon.stop()


def test_l4_remote_multiple_dispatches(tmp_path):
    read_counter, add_counter = _make_file_counter(tmp_path / "remote_counter.txt")
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)
        def l3_sub(args):
            add_counter(_scalar_value(args))

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid, args)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            for value in (1, 2, 4):
                sub_args = TaskArgs()
                sub_args.add_scalar(value)
                orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert read_counter() == 7
    finally:
        daemon.stop()


def test_l4_remote_with_local_sub(tmp_path):
    read_remote_counter, add_remote_counter = _make_file_counter(tmp_path / "remote_counter.txt")
    local_shm, local_buf = _make_shared_counter()
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=1)
        def l3_sub(args):
            add_remote_counter(1)

        l3_sub_cid = w4.register(l3_sub)
        local_cid = w4.register(lambda args: _increment_counter(local_buf))

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            orch.submit_next_level(l3_cid, TaskArgs(), CallConfig())
            orch.submit_sub(local_cid)

        w4.run(l4_orch)
        w4.close()
        assert read_remote_counter() == 1
        assert _read_counter(local_buf) == 1
    finally:
        daemon.stop()
        local_shm.close()
        local_shm.unlink()


def test_l4_remote_multiple_runs(tmp_path):
    read_counter, add_counter = _make_file_counter(tmp_path / "remote_counter.txt")
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)
        def l3_sub(args):
            add_counter(1)

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            orch.submit_next_level(l3_cid, TaskArgs(), CallConfig())

        for _ in range(5):
            w4.run(l4_orch)
        w4.close()
        assert read_counter() == 5
    finally:
        daemon.stop()


def test_l4_remote_inline_tensor_dispatch(tmp_path):
    result = tmp_path / "remote_tensor_sum.txt"
    payload = b"abcdef"
    buf = ctypes.create_string_buffer(payload)
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            tensor = args.tensor(0)
            data = ctypes.string_at(int(tensor.data), int(tensor.nbytes()))
            result.write_text(str(sum(data)))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.INPUT)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert int(result.read_text()) == sum(payload)
    finally:
        daemon.stop()


def test_l4_remote_handle_tensor_dispatch(tmp_path):
    result = tmp_path / "remote_tensor_sum.txt"
    payload = bytes(range(256)) * 32
    buf = ctypes.create_string_buffer(payload)
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            tensor = args.tensor(0)
            data = ctypes.string_at(int(tensor.data), int(tensor.nbytes()))
            result.write_text(str(len(data)) + ":" + str(sum(data)))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.INPUT)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert result.read_text() == f"{len(payload)}:{sum(payload)}"
    finally:
        daemon.stop()


def test_l4_remote_inline_output_tensor_writeback():
    buf = ctypes.create_string_buffer(b"\x00" * 6)
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            tensor = args.tensor(0)
            ctypes.memmove(int(tensor.data), b"fedcba", 6)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (6,), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert bytes(buf.raw) == b"fedcba\x00"
    finally:
        daemon.stop()


def test_l4_remote_handle_output_tensor_writeback():
    payload = bytes((255 - (i % 256) for i in range(8192)))
    buf = ctypes.create_string_buffer(b"\x00" * len(payload))
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            tensor = args.tensor(0)
            ctypes.memmove(int(tensor.data), payload, len(payload))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert bytes(buf.raw) == payload + b"\x00"
    finally:
        daemon.stop()


def test_l4_remote_inout_tensor_writeback():
    payload = bytearray(b"abcde")
    buf = ctypes.create_string_buffer(bytes(payload))
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            tensor = args.tensor(0)
            data = bytearray(ctypes.string_at(int(tensor.data), int(tensor.nbytes())))
            data.reverse()
            ctypes.memmove(int(tensor.data), bytes(data), len(data))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.INOUT)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert bytes(buf.raw) == b"edcba\x00"
    finally:
        daemon.stop()


def test_l4_remote_tensor_input_reaches_l3_sub(tmp_path):
    result = tmp_path / "remote_sub_tensor_sum.txt"
    payload = b"subtensor"
    buf = ctypes.create_string_buffer(payload)
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_sub(args):
            tensor = args.tensor(0)
            data = ctypes.string_at(int(tensor.data), int(tensor.nbytes()))
            result.write_text(str(sum(data)))

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid, args)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.INPUT)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert int(result.read_text()) == sum(payload)
    finally:
        daemon.stop()


def test_l4_remote_tensor_output_from_l3_sub_writeback():
    payload = b"sub-output"
    buf = ctypes.create_string_buffer(b"\x00" * len(payload))
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_sub(args):
            tensor = args.tensor(0)
            ctypes.memmove(int(tensor.data), payload, len(payload))

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid, args)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            tensor = ContinuousTensor.make(ctypes.addressof(buf), (len(payload),), DataType.UINT8)
            sub_args.add_tensor(tensor, TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert bytes(buf.raw) == payload + b"\x00"
    finally:
        daemon.stop()


def test_l4_remote_l3_multiple_subs(tmp_path):
    read_counter, add_counter = _make_file_counter(tmp_path / "remote_counter.txt")
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)
        def l3_sub(args):
            add_counter(1)

        l3_sub_cid = w4.register(l3_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid)
            orch.submit_sub(l3_sub_cid)

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            orch.submit_next_level(l3_cid, TaskArgs(), CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert read_counter() == 2
    finally:
        daemon.stop()


def test_l4_remote_error_propagates():
    daemon, endpoint = _start_daemon()

    def broken_l3_orch(orch, args, config):
        raise ValueError("remote failure")

    try:
        w4 = Worker(level=4, num_sub_workers=0)
        l3_cid = w4.register(broken_l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            orch.submit_next_level(l3_cid, TaskArgs(), CallConfig())

        try:
            w4.run(l4_orch)
        except RuntimeError as e:
            assert "remote failure" in str(e)
        else:
            raise AssertionError("remote failure did not propagate")
        finally:
            w4.close()
    finally:
        daemon.stop()

