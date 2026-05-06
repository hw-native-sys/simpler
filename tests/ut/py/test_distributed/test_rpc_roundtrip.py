import pytest

from simpler.distributed.proto import dispatch_pb2, dispatch_pb2_grpc
from simpler.distributed.rpc import RpcClient, RpcError, RpcServer


class EchoL3(dispatch_pb2_grpc.L3WorkerServicer):
    def Dispatch(self, request, context):  # noqa: N802
        return dispatch_pb2.DispatchResp(task_id=request.task_id, error_code=0)

    def Heartbeat(self, request, context):  # noqa: N802
        return dispatch_pb2.Health(ok=True, message="ok")


class FailingL3(dispatch_pb2_grpc.L3WorkerServicer):
    def Dispatch(self, request, context):  # noqa: N802
        context.abort(13, "boom")

    def Heartbeat(self, request, context):  # noqa: N802
        return dispatch_pb2.Health(ok=True, message="ok")


def test_rpc_roundtrip():
    server = RpcServer()
    server.add_l3_worker(EchoL3())
    port = server.start(0)
    client = RpcClient(f"127.0.0.1:{port}")
    try:
        resp = client.call_unary(
            "L3Worker.Dispatch",
            dispatch_pb2.DispatchReq(task_id=42, callable_id=7),
            timeout=2,
        )
        assert resp.task_id == 42
        assert resp.error_code == 0
    finally:
        client.close()
        server.stop(0)


def test_rpc_error_maps_to_exception():
    server = RpcServer()
    server.add_l3_worker(FailingL3())
    port = server.start(0)
    client = RpcClient(f"127.0.0.1:{port}")
    try:
        with pytest.raises(RpcError, match="boom"):
            client.dispatch(dispatch_pb2.DispatchReq(task_id=1), timeout=2)
    finally:
        client.close()
        server.stop(0)


def test_port_conflict_reports_clear_error():
    first = RpcServer()
    port = first.start(0)
    second = RpcServer()
    try:
        with pytest.raises(RpcError, match="failed to bind"):
            second.start(port)
    finally:
        first.stop(0)
        second.stop(0)
