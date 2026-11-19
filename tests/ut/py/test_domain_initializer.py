# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import struct
import threading
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import simpler.worker as worker_mod
from simpler.buffer import AddressSpace
from simpler.orchestrator import _admit_task_submission
from simpler.task_interface import ChipCallable, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


@pytest.fixture(params=["a5", "a2a3"])
def host_domain_worker(request, monkeypatch, native_host_memory):
    platform = request.param
    worker = Worker(
        level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0
    )
    try:
        worker.init()
        monkeypatch.setitem(worker._config, "platform", platform)
        monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
        owners = {}
        allocations = []

        def dispatch(*, workers, request_shms, reply_shms, op, allocation_id):
            if op == "release":
                for chip in workers:
                    owner = owners.pop((allocation_id, chip))
                    owner.close()
                    owner.unlink()
                return
            allocations.append(allocation_id)
            for chip in workers:
                request = request_shms[chip].buf
                identity, count, rank, size, buffer_count = worker_mod._DOMAIN_REQ_HEADER.unpack_from(request)
                assert identity == allocation_id and rank == chip and count == 2
                tail = worker_mod._DOMAIN_REQ_HEADER.size + buffer_count * 8 + count * 4
                name = worker_mod._read_shm_name(request, tail)
                owners[allocation_id, chip] = native_host_memory(name=name, size=size)
                base = 0x100000 * (chip + 1)
                response = reply_shms[chip].buf
                worker_mod._DOMAIN_REPLY_HEADER.pack_into(response, 0, 1, base + size, base, size, buffer_count)
                struct.pack_into("<2Q", response, worker_mod._DOMAIN_REPLY_HEADER.size, base, base + 64)

        monkeypatch.setattr(worker, "_dispatch_control_domain", dispatch)
        config = dict(
            name="engram",
            members=[0, 1],
            window_size=4096,
            buffers=[CommBufferSpec("prefix", "uint8", 64, 64), CommBufferSpec("table", "uint8", 64, 64)],
            memory_source=AddressSpace.HOST,
            retain_after_run=True,
        )
        yield SimpleNamespace(worker=worker, request=config, owners=owners, allocations=allocations)
    finally:
        worker.close()


def _fill(domain):
    offset, nbytes = domain.buffer_range("table")
    for rank in range(2):
        with domain.member(rank).buffers["table"].shm.buf[offset : offset + nbytes] as view:
            view[:] = bytes([rank + 17]) * nbytes


@pytest.mark.parametrize("retained", [False, True], ids=["initializer", "retained"])
def test_close_drains_admitted_host_access(host_domain_worker, retained):
    env = host_domain_worker
    worker = env.worker
    entered = threading.Event()
    domains = []

    def access(domain):
        domains.append(domain)
        entered.set()
        with worker._hierarchical_start_cv:
            assert worker._hierarchical_start_cv.wait_for(
                lambda: worker._lifecycle is worker_mod._Lifecycle.CLOSED, timeout=10
            )
        _fill(domain)

    if retained:
        worker.run(lambda orch, *_: orch.allocate_domain(**env.request))

    def model(orch, *_):
        if retained:
            access(orch.get_domain("engram"))
        else:
            orch.allocate_domain(**env.request, initializer=access)

    with ThreadPoolExecutor(max_workers=1) as pool:
        run = pool.submit(worker.run, model)
        assert entered.wait(10)
        worker.close()
        run.result(timeout=10)
    assert domains[0].freed and not env.owners


@pytest.mark.parametrize("writing", [True, False], ids=["write", "read"])
def test_host_window_rejects_device_copy(host_domain_worker, monkeypatch, writing):
    env = host_domain_worker

    def unexpected_copy(*_):
        pytest.fail("Host window reached the native device copy")

    def model(orch, *_):
        domain = orch.allocate_domain(**env.request)
        window = domain.member(0).buffers["table"]
        staging = env.worker.create_buffer(64)
        with monkeypatch.context() as patch:
            native_type = type(env.worker._worker)
            patch.setattr(native_type, "copy_to", unexpected_copy)
            patch.setattr(native_type, "copy_from", unexpected_copy)
            patch.setattr(window, "shm", None)
            if writing:
                with pytest.raises(ValueError, match=r"Buffer\.shm\.buf"):
                    orch.copy_to_domain(domain, 0, b"test", buffer="table")
                with pytest.raises(ValueError, match=r"Buffer\.shm\.buf"):
                    env.worker.copy_to(window, staging)
            else:
                with pytest.raises(ValueError, match=r"Buffer\.shm\.buf"):
                    orch.copy_from_domain(domain, 0, 4, buffer="table")
                with pytest.raises(ValueError, match=r"Buffer\.shm\.buf"):
                    env.worker.copy_from(staging, window)

    env.worker.run(model)


@pytest.mark.parametrize("api", ["allocate_domain", "get_or_allocate_domain"])
def test_initializer_once_per_domain_instance(host_domain_worker, api):
    env = host_domain_worker
    initialized, seen = [], []

    def initialize(domain):
        assert len(domain.contexts) == 2 and len(domain._host_mappings) == 2
        _fill(domain)
        initialized.append(domain)

    def model(orch, _args, _config):
        create = getattr(orch, api) if not seen else orch.get_or_allocate_domain
        domain = create(**env.request, initializer=initialize)
        seen.append(domain)
        assert bytes(env.owners[domain.allocation_id, 1].buf[64:128]) == bytes([18]) * 64
        assert orch.get_domain("engram") is domain
        _admit_task_submission(env.worker)

    for _ in range(3):
        env.worker.run(model)
    assert initialized == [seen[0]]
    assert all(domain is seen[0] for domain in seen)
    seen[0].release()
    seen.clear()
    env.worker.run(model)
    assert len(initialized) == 2 and initialized[0] is not initialized[1]


def test_initializer_is_not_retained_or_compared_on_reuse(host_domain_worker):
    env = host_domain_worker

    class Loader:
        def __call__(self, domain):
            _fill(domain)

    loader = Loader()
    reference = weakref.ref(loader)
    env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request, initializer=reference()))
    del loader
    assert reference() is None

    def unexpected(_domain):
        pytest.fail("reuse must not invoke a replacement initializer")

    env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request, initializer=unexpected))
    env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request))


def test_initializer_does_not_backfill_existing_domain(host_domain_worker):
    env = host_domain_worker
    env.worker.run(lambda orch, *_: orch.allocate_domain(**env.request))
    env.worker.run(
        lambda orch, *_: orch.get_or_allocate_domain(
            **env.request, initializer=lambda _: pytest.fail("existing domain")
        )
    )


@pytest.mark.parametrize("api", ["allocate_domain", "get_or_allocate_domain"])
def test_initializer_failure_rolls_back_all_ranks(host_domain_worker, api):
    env = host_domain_worker
    saved = []

    def fail(domain):
        saved.append(domain)
        _fill(domain)
        raise ValueError("checkpoint truncated")

    with pytest.raises(ValueError, match="checkpoint truncated"):
        env.worker.run(lambda orch, *_: getattr(orch, api)(**env.request, initializer=fail))
    assert saved[0].freed and saved[0].released
    assert not env.owners and "engram" not in env.worker.live_domains
    env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request, initializer=_fill))
    assert len(env.allocations) == 2


@pytest.mark.parametrize(
    "operation",
    [
        "lookup",
        "reuse",
        "release",
        "context",
        "submit",
        "run",
        "control",
        "scope_begin",
        "scope_end",
        "scope",
        "alloc",
        "shared_alloc",
        "submit_next_level",
        "submit_next_level_group",
        "submit_sub",
        "submit_sub_group",
    ],
)
def test_initializer_reentry_is_rejected_without_breaking_fill(host_domain_worker, operation):
    env = host_domain_worker

    def model(orch, *_):
        def initialize(domain):
            actions = {
                "lookup": lambda: orch.get_domain("engram"),
                "reuse": lambda: orch.get_or_allocate_domain(**env.request),
                "release": domain.release,
                "context": lambda: domain.device_context(0),
                "submit": lambda: _admit_task_submission(env.worker),
                "run": lambda: env.worker.run(lambda *_: None),
                "control": lambda: orch.alloc_child_tensor(0, (64,), DataType.UINT8),
                "scope_begin": orch.scope_begin,
                "scope_end": orch.scope_end,
                "scope": lambda: orch.scope().__enter__(),
                "alloc": lambda: orch.alloc([64], DataType.UINT8),
                "shared_alloc": lambda: env.worker.alloc_shared_tensor((64,), DataType.UINT8),
                "submit_next_level": lambda: orch.submit_next_level(None, TaskArgs(), worker=0),
                "submit_next_level_group": lambda: orch.submit_next_level_group(None, [TaskArgs()], workers=[0]),
                "submit_sub": lambda: orch.submit_sub(None),
                "submit_sub_group": lambda: orch.submit_sub_group(None, [TaskArgs()]),
            }
            with pytest.raises(RuntimeError, match="initializ"):
                actions[operation]()
            _fill(domain)

        orch.get_or_allocate_domain(**env.request, initializer=initialize)

    env.worker.run(model)


@pytest.mark.parametrize("bad", [False, 123, object()])
def test_initializer_type_is_checked_before_allocation(host_domain_worker, bad):
    env = host_domain_worker
    with pytest.raises(TypeError, match="initializer"):
        env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request, initializer=bad))
    assert not env.allocations


def test_initializer_rejects_device_source_before_allocation(host_domain_worker):
    env = host_domain_worker
    request = dict(env.request, memory_source=AddressSpace.DEVICE)
    with pytest.raises(ValueError, match="Host-source"):
        env.worker.run(lambda orch, *_: orch.allocate_domain(**request, initializer=_fill))
    assert not env.allocations


def test_initializer_must_complete_synchronously(host_domain_worker):
    env = host_domain_worker

    async def asynchronous(_domain):
        pass

    with pytest.raises(TypeError, match="synchronous"):
        env.worker.run(lambda orch, *_: orch.allocate_domain(**env.request, initializer=asynchronous))
    assert not env.allocations


@pytest.mark.parametrize("operation", ["submit", "shared_alloc"])
def test_caught_initializer_failure_cannot_continue(host_domain_worker, operation):
    env = host_domain_worker

    def model(orch, *_):
        def fail(_domain):
            raise ValueError("broken loader")

        with pytest.raises(ValueError, match="broken loader"):
            orch.allocate_domain(**env.request, initializer=fail)
        with pytest.raises(RuntimeError, match="initializ"):
            if operation == "submit":
                _admit_task_submission(env.worker)
            else:
                env.worker.alloc_shared_tensor((64,), DataType.UINT8)

    env.worker.run(model)
    assert not env.owners


@pytest.mark.parametrize("caught", [False, True])
def test_failed_initializer_keeps_live_export_owned(host_domain_worker, caught):
    env = host_domain_worker
    saved, views = [], []

    def fail(domain):
        saved.append(domain)
        views.append(domain.member(0).buffers["table"].shm.buf)
        raise ValueError("loader failed with a live view")

    def model(orch, *_):
        if caught:
            with pytest.raises(ValueError, match="loader failed"):
                orch.get_or_allocate_domain(**env.request, initializer=fail)
            with pytest.raises(RuntimeError, match="initializ"):
                orch.get_domain("engram")
        else:
            orch.get_or_allocate_domain(**env.request, initializer=fail)

    try:
        with pytest.raises((ValueError, BufferError, RuntimeError)) as error:
            env.worker.run(model)
        assert "loader failed" in str(error.value) or "export" in str(error.value)
        domain = saved[0]
        assert not domain.freed and not domain.released
        assert env.worker._live_domains["engram"] is domain and len(env.owners) == 2
        assert "engram" not in env.worker.live_domains
        with pytest.raises(RuntimeError, match="initialization failed"):
            domain.member(0)
        assert bytes(views[0][:4]) == bytes(4)
    finally:
        for view in views:
            view.release()
        views.clear()
    env.worker.close()
    assert domain.freed and not env.owners


def test_initializer_return_value_failure_rolls_back(host_domain_worker):
    env = host_domain_worker
    with pytest.raises(TypeError, match="return None"):
        env.worker.run(lambda orch, *_: orch.allocate_domain(**env.request, initializer=lambda _: 1))
    assert not env.owners


def test_initializer_not_called_for_configuration_mismatch(host_domain_worker):
    env = host_domain_worker
    env.worker.run(lambda orch, *_: orch.get_or_allocate_domain(**env.request, initializer=_fill))
    with pytest.raises(ValueError, match="configuration differs"):
        env.worker.run(
            lambda orch, *_: orch.get_or_allocate_domain(
                **dict(env.request, window_size=8192), initializer=lambda _: pytest.fail("configuration mismatch")
            )
        )
    assert len(env.allocations) == 1


def test_nonretained_domain_initializes_on_each_creation(host_domain_worker):
    env = host_domain_worker
    calls = []
    request = dict(env.request, retain_after_run=False)
    for _ in range(2):
        env.worker.run(
            lambda orch, *_: orch.get_or_allocate_domain(
                **request, initializer=lambda domain: calls.append(domain.domain_id)
            )
        )
        assert not env.owners
    assert len(calls) == 2 and calls[0] != calls[1]


@pytest.mark.parametrize("caught", [False, True])
def test_initializer_cannot_release_buffer_before_serializer(host_domain_worker, monkeypatch, caught):
    env = host_domain_worker
    buffer = env.worker.create_buffer(64)

    def check_release():
        def unexpected_lock():
            pytest.fail("initializer buffer release reached the already-held serializer")

        with monkeypatch.context() as patch:
            patch.setattr(env.worker._submit_mu, "exclusive", unexpected_lock)
            with pytest.raises(RuntimeError, match="initializ"):
                env.worker.release_buffer(buffer)
        assert not buffer.closed

    def model(orch, *_):
        def initialize(domain):
            if caught:
                raise ValueError("broken loader")
            check_release()
            _fill(domain)

        if caught:
            with pytest.raises(ValueError, match="broken loader"):
                orch.allocate_domain(**env.request, initializer=initialize)
            check_release()
        else:
            orch.allocate_domain(**env.request, initializer=initialize)

    env.worker.run(model)
    env.worker.release_buffer(buffer)
    assert buffer.closed


@pytest.mark.parametrize("as_array", [False, True], ids=["memoryview", "numpy"])
def test_initializer_returned_export_does_not_block_rollback(host_domain_worker, as_array):
    env = host_domain_worker
    references = []

    def bad_result(domain):
        view = domain.member(0).buffers["table"].shm.buf
        result = np.frombuffer(view, dtype=np.uint8) if as_array else view
        references.append(weakref.ref(result))
        return result

    with pytest.raises(TypeError, match="return None") as caught:
        env.worker.run(lambda orch, *_: orch.allocate_domain(**env.request, initializer=bad_result))
    try:
        # Retain the exception to prove runtime-owned traceback locals cannot pin the export.
        released = references[0]() is None
        assert released
        assert not env.owners and not env.worker.live_domains
        env.worker.run(lambda orch, *_: orch.allocate_domain(**env.request, initializer=_fill))
    finally:
        traceback.clear_frames(caught.value.__traceback__)


def test_initializer_submits_domain_tensor_in_same_orchestration(host_domain_worker, monkeypatch):
    env = host_domain_worker
    # The host registry is real; intercept device installation and the final native submit.
    monkeypatch.setattr(env.worker, "_post_init_register", lambda *args, **kwargs: None)
    entry = env.worker.register(ChipCallable.build(signature=[], func_name="consume_table", binary=b"x", children=[]))
    calls, initialized = [], []

    def initialize(domain):
        _fill(domain)
        initialized.append(domain.domain_id)

    def model(orch, *_):
        domain = orch.get_or_allocate_domain(**env.request, initializer=initialize)
        buffer = domain.member(1).buffers["table"]
        args = TaskArgs()
        args.add_tensor(buffer.tensor((64,), DataType.UINT8), TensorArgType.INPUT)

        def native_submit(*call):
            assert bytes(env.owners[domain.allocation_id, 1].buf[64:128]) == bytes([18]) * 64
            submitted = call[3]
            assert submitted.tensor(0).buffer == buffer.to_descriptor()
            assert submitted.tensor(0).byte_offset == 0
            assert tuple(submitted.tensor(0).shapes) == (64,)
            assert submitted.tag(0) == TensorArgType.INPUT
            calls.append(call)

        with monkeypatch.context() as patch:
            patch.setattr(orch, "_o", SimpleNamespace(submit_next_level=native_submit))
            orch.submit_next_level(entry, args, worker=1)

    for _ in range(2):
        env.worker.run(model)
    assert len(initialized) == 1 and len(calls) == 2
    for call in calls:
        assert call[0] == entry.digest
        assert call[1:3] == ("CHIP_CALLABLE", "LOCAL_CHIP")
        assert call[5] == 1
