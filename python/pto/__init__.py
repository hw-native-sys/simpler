"""PTO — Python Tensor Orchestration runtime.

Provides a unified API for L2 (single chip) through L3+ (multi-chip)
execution on Ascend NPU devices.

Quick start::

    import pto

    # Single chip (L2)
    rt = pto.Runtime(level="chip", platform="a2a3", device=0)
    rt.register("vector_add", orch="orch.cpp", kernels=[...])
    rt.run("vector_add", args=[pto.Arg.input(x), pto.Arg.output(y)])
    rt.close()

    # Multi-chip (L3)
    rt = pto.Runtime(level="host", platform="a2a3", devices=[0, 1, 2, 3])
    pkg = pto.compile(platform="a2a3", orch="orch.cpp", kernels=[...])
    rt.register("pipeline", orch=my_orch_func, kernels={"compute": pkg})
    rt.run("pipeline", args={"input": data})
    rt.close()
"""

from .types import Arg, CompiledPackage, KernelSource, ParamType, TensorHandle
from .runtime import Runtime
from .compiler import compile

__all__ = [
    "Runtime",
    "Arg",
    "compile",
    "CompiledPackage",
    "KernelSource",
    "ParamType",
    "TensorHandle",
]
