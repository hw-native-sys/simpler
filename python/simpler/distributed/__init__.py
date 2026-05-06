"""Python-first distributed L4 -> L3 dispatch support."""

__all__ = [
    "Catalog",
    "CatalogError",
    "L3Daemon",
    "RemoteUnavailable",
    "RemoteWorkerProxy",
    "RpcClient",
    "RpcError",
    "RpcServer",
]


def __getattr__(name):
    if name in {"Catalog", "CatalogError"}:
        from .catalog import Catalog, CatalogError

        return {"Catalog": Catalog, "CatalogError": CatalogError}[name]
    if name == "L3Daemon":
        from .l3_daemon import L3Daemon

        return L3Daemon
    if name in {"RemoteUnavailable", "RemoteWorkerProxy"}:
        from .remote_proxy import RemoteUnavailable, RemoteWorkerProxy

        return {"RemoteUnavailable": RemoteUnavailable, "RemoteWorkerProxy": RemoteWorkerProxy}[name]
    if name in {"RpcClient", "RpcError", "RpcServer"}:
        from .rpc import RpcClient, RpcError, RpcServer

        return {"RpcClient": RpcClient, "RpcError": RpcError, "RpcServer": RpcServer}[name]
    raise AttributeError(name)
