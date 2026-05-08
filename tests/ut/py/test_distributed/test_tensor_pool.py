import ctypes
import time

import pytest

from simpler.distributed.proto import dispatch_pb2
from simpler.distributed.serialization import (
    decode_task_args_with_tensor_refs,
    encode_output_tensor_refs,
    encode_tensor_ref,
)
from simpler.distributed.tensor_pool import DEFAULT_INLINE_THRESHOLD, TensorPool, TensorPoolFull
from simpler.distributed.transport_backend import (
    TransportUnavailable,
    build_tensor_transport,
)
from simpler.task_interface import ContinuousTensor, DataType, TaskArgs, TensorArgType


def test_tensor_pool_inline_bytes():
    pool = TensorPool(inline_threshold=8)
    ref = pool.put_bytes(b"abc")
    assert ref.inline_data == b"abc"


def test_tensor_pool_handle_bytes():
    pool = TensorPool(inline_threshold=2)
    ref = pool.put_bytes(b"abcdef")
    assert ref.HasField("handle")
    assert pool.get_bytes(ref.handle) == b"abcdef"
    assert ref.handle.nbytes == 6
    assert ref.handle.remote_addr != 0
    assert ref.handle.transport == "grpc"
    assert ref.handle.lease_deadline_unix_ms > 0


def test_build_tensor_transport_auto_falls_back_to_grpc(monkeypatch):
    monkeypatch.delenv("SIMPLER_HCOMM_LIB", raising=False)
    monkeypatch.delenv("SIMPLER_HCOMM_ENDPOINT_HANDLE", raising=False)
    backend = build_tensor_transport("auto")
    assert backend.name == "grpc"


def test_build_tensor_transport_explicit_hcomm_requires_configuration(monkeypatch):
    monkeypatch.delenv("SIMPLER_HCOMM_LIB", raising=False)
    monkeypatch.delenv("SIMPLER_HCOMM_ENDPOINT_HANDLE", raising=False)
    with pytest.raises(TransportUnavailable):
        build_tensor_transport("hcomm")


def test_tensor_pool_default_inline_threshold_is_four_kb():
    pool = TensorPool()
    assert pool.inline_threshold == DEFAULT_INLINE_THRESHOLD
    assert pool.put_bytes(b"x" * DEFAULT_INLINE_THRESHOLD).HasField("inline_data")
    assert pool.put_bytes(b"x" * (DEFAULT_INLINE_THRESHOLD + 1)).HasField("handle")


def test_tensor_pool_alloc_write_read_free():
    pool = TensorPool(capacity_bytes=16)
    handle = pool.alloc(6, shape=(2, 3), dtype=DataType.UINT8.value, tag=TensorArgType.INPUT.value)
    pool.write_bytes(handle, b"abcdef")
    assert pool.read_bytes(handle, offset=1, nbytes=3) == b"bcd"
    assert pool.get_bytes(handle) == b"abcdef"
    pool.free(handle)
    with pytest.raises(KeyError):
        pool.get_bytes(handle)


def test_tensor_pool_capacity_and_lease_gc():
    pool = TensorPool(capacity_bytes=4, default_ttl_ms=10)
    handle = pool.alloc(4, ttl_ms=1000)
    with pytest.raises(TensorPoolFull):
        pool.alloc(1)
    pool.free(handle)
    handle = pool.alloc(4, ttl_ms=1)
    time.sleep(0.01)
    assert pool.gc_expired() == 1
    assert pool.used_bytes == 0
    with pytest.raises(KeyError):
        pool.get_bytes(handle)


def test_tensor_pool_refresh_extends_lease():
    pool = TensorPool(default_ttl_ms=100)
    handle = pool.alloc(1, ttl_ms=100)
    refreshed = pool.refresh(handle, ttl_ms=1000)
    assert refreshed.handle_id == handle.handle_id
    assert refreshed.lease_deadline_unix_ms >= handle.lease_deadline_unix_ms
    time.sleep(0.01)
    assert pool.gc_expired() == 0


def test_tensor_pool_service_pull():
    pool = TensorPool(inline_threshold=1)
    ref = pool.put_bytes(b"abcdef")
    service = pool.service()
    chunks = list(service.PullTensor(ref.handle, None))
    assert b"".join(chunk.data for chunk in chunks) == b"abcdef"
    assert chunks[-1].last


def test_tensor_pool_service_push():
    pool = TensorPool(inline_threshold=1)
    service = pool.service()
    handle = service.PushTensor(
        iter(
            [
                dispatch_pb2.TensorChunk(offset=0, data=b"abc"),
                dispatch_pb2.TensorChunk(offset=3, data=b"def", last=True),
            ]
        ),
        None,
    )
    assert pool.get_bytes(handle) == b"abcdef"


def test_tensor_pool_service_alloc_free_refresh_push_to_handle():
    pool = TensorPool(capacity_bytes=16)
    service = pool.service()
    handle = service.AllocTensor(dispatch_pb2.TensorAllocReq(nbytes=6, ttl_ms=100), None)
    refreshed = service.RefreshTensor(dispatch_pb2.TensorRefreshReq(handle=handle, ttl_ms=1000), None)
    assert refreshed.handle_id == handle.handle_id
    out = service.PushTensor(
        iter(
            [
                dispatch_pb2.TensorChunk(handle=handle, offset=0, data=b"abc"),
                dispatch_pb2.TensorChunk(handle=handle, offset=3, data=b"def", last=True),
            ]
        ),
        None,
    )
    assert out.handle_id == handle.handle_id
    assert pool.get_bytes(handle) == b"abcdef"
    assert list(service.PullTensor(handle, None))[-1].last
    service.FreeTensor(dispatch_pb2.TensorFreeReq(handle=handle), None)
    with pytest.raises(KeyError):
        pool.get_bytes(handle)


def test_encode_decode_tensor_refs_inline_and_handle():
    pool = TensorPool(inline_threshold=4)
    inline = encode_tensor_ref(
        b"abc",
        shape=(3,),
        dtype=DataType.UINT8,
        tag=TensorArgType.INPUT,
        pool=pool,
    )
    handled = encode_tensor_ref(
        b"abcdef",
        shape=(6,),
        dtype=DataType.UINT8,
        tag=TensorArgType.INPUT,
        pool=pool,
        force_handle=True,
    )
    args, keepalive = decode_task_args_with_tensor_refs([inline, handled], [7], pool)
    assert args.tensor_count() == 2
    assert args.scalar(0) == 7
    assert args.tensor(0).nbytes() == 3
    assert args.tensor(1).nbytes() == 6
    assert args.tag(0) == TensorArgType.INPUT
    assert len(keepalive) == 2


def test_encode_output_tensor_refs():
    pool = TensorPool(inline_threshold=4)
    in_buf = ctypes.create_string_buffer(b"abc")
    out_buf = ctypes.create_string_buffer(b"abcdef")
    args = TaskArgs()
    args.add_tensor(ContinuousTensor.make(ctypes.addressof(in_buf), (3,), DataType.UINT8), TensorArgType.INPUT)
    args.add_tensor(ContinuousTensor.make(ctypes.addressof(out_buf), (6,), DataType.UINT8), TensorArgType.OUTPUT)
    refs = encode_output_tensor_refs(args, pool)
    assert len(refs) == 1
    assert refs[0].HasField("handle")
    assert pool.get_bytes(refs[0].handle) == b"abcdef"
