def test_distributed_imports():
    import simpler.distributed
    from simpler.distributed.proto import dispatch_pb2

    assert simpler.distributed is not None
    assert dispatch_pb2.DispatchReq(task_id=1).task_id == 1
