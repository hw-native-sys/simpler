"""Regression tests for simulation kernel compilation."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from kernel_compiler import KernelCompiler
from toolchain import Gxx15Toolchain, ToolchainType


@pytest.mark.parametrize(
    ("core_type", "expected_flags", "unexpected_flags"),
    [
        ("aic", ["-D__AIC__", "-D__DAV_CUBE__"], ["-D__AIV__", "-D__DAV_VEC__"]),
        ("aiv", ["-D__AIV__", "-D__DAV_VEC__"], ["-D__AIC__", "-D__DAV_CUBE__"]),
    ],
)
def test_gxx15_flags_define_core_macros(core_type, expected_flags, unexpected_flags):
    """Simulation builds must enable the correct PTOAS sections."""
    flags = Gxx15Toolchain().get_compile_flags(core_type=core_type)

    assert "-D__CPU_SIM" in flags
    for flag in expected_flags:
        assert flag in flags
    for flag in unexpected_flags:
        assert flag not in flags


@patch.object(KernelCompiler, "_compile_incore_sim", return_value=b"sim_kernel")
@patch.object(KernelCompiler, "_get_toolchain", return_value=ToolchainType.HOST_GXX_15)
def test_compile_incore_sim_forwards_core_type(mock_get_toolchain, mock_compile_incore_sim, tmp_path):
    """Simulation builds must preserve the requested core type."""
    compiler = KernelCompiler(platform="a2a3sim")
    source_path = tmp_path / "kernel.cpp"
    source_path.write_text("extern \"C\" void kernel_entry() {}\n", encoding="utf-8")

    result = compiler.compile_incore(str(source_path), core_type="aic")

    assert result == b"sim_kernel"
    assert mock_get_toolchain.called
    assert mock_compile_incore_sim.call_args.kwargs["core_type"] == "aic"
