import struct
from multiprocessing.shared_memory import SharedMemory

from simpler.distributed.l3_daemon import L3Daemon
from simpler.task_interface import CallConfig, TaskArgs
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
