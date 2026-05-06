from simpler.distributed.proto import dispatch_pb2
from simpler.distributed.tensor_pool import TensorPool


def test_tensor_pool_inline_bytes():
    pool = TensorPool(inline_threshold=8)
    ref = pool.put_bytes(b"abc")
    assert ref.inline_data == b"abc"


def test_tensor_pool_handle_bytes():
    pool = TensorPool(inline_threshold=2)
    ref = pool.put_bytes(b"abcdef")
    assert ref.HasField("handle")
    assert pool.get_bytes(ref.handle) == b"abcdef"


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
