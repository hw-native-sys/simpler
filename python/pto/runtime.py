"""Unified Runtime entry point.

Routes by ``level`` parameter to the appropriate runtime implementation:
    - ``"chip"`` → L2Runtime (single chip)
    - ``"host"`` → L3Runtime (single host, multi chip)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from .types import Arg, CompiledPackage


class Runtime:
    """Unified runtime interface.

    Usage::

        # L2 — single chip
        rt = pto.Runtime(level="chip", platform="a2a3", device=0)
        rt.register("vector_add", orch="orch.cpp", kernels=[...])
        rt.run("vector_add", args=[Arg.input(x), Arg.output(y), Arg.scalar(n)])
        rt.close()

        # L3 — single host, multi chip
        rt = pto.Runtime(level="host", platform="a2a3", devices=[0, 1, 2, 3])
        pkg = pto.compile(platform="a2a3", orch="orch.cpp", kernels=[...])
        rt.register("pipeline", orch=my_orch_func, kernels={"compute": pkg})
        rt.run("pipeline", args={"input": data})
        rt.close()
    """

    def __init__(self, level: str = "chip", **kwargs):
        self._level = level.lower()
        self._impl = _create_impl(self._level, **kwargs)

    @property
    def level(self) -> str:
        return self._level

    def register(self, name: str, **kwargs) -> None:
        """Register a named computation."""
        self._impl.register(name, **kwargs)

    def run(self, name: str, args: Any = None) -> Any:
        """Execute a registered computation."""
        return self._impl.run(name, args=args)

    def close(self) -> None:
        """Release resources."""
        self._impl.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _create_impl(level: str, **kwargs):
    """Instantiate the right runtime backend for the given level."""
    if level == "chip":
        from .l2_runtime import L2Runtime
        platform = kwargs.get("platform", "a2a3")
        device = kwargs.get("device", 0)
        return L2Runtime(platform=platform, device=device)

    if level == "host":
        from .l3_runtime import L3Runtime
        platform = kwargs.get("platform", "a2a3")
        devices = kwargs.get("devices", [0])
        return L3Runtime(platform=platform, devices=devices)

    raise ValueError(f"Unknown level '{level}'. Supported: 'chip', 'host'")
