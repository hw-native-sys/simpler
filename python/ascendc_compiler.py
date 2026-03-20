"""
AscendC Kernel Compiler for PTO Runtime Integration

Compiles AscendC operator sources into PTO-compatible kernel binaries by
generating a merged source file that includes both the user's AscendC kernel
and a wrapper ``kernel_entry`` that bridges PTO's Tensor-pointer convention
to AscendC's raw GM-address convention.

Workflow:
  1. Generate merged .cpp: kernel_entry wrapper (at offset 0) + #include user source
  2. Compile with AscendC toolchain flags (--cce-aicore-lang) → single .o
  3. Link with ld.lld (-e kernel_entry -Ttext=0) to resolve block-local relocations
  4. extract_text_section(linked_elf) → kernel binary (done by caller)

The merged source suppresses ``__global__`` when including the user's kernel
source, so the compiler allows ``kernel_entry`` to call the user's kernel
function.  kernel_entry is defined first in the source to ensure it sits at
.text offset 0 (PTO dispatches to offset 0).

Usage:
    compiler = AscendCCompiler(platform="a2a3")
    kernel_o = compiler.compile_ascendc_kernel(
        ascendc_kernel_source="/path/to/add_custom.cpp",
        ascendc_kernel_symbol="add_custom",
        tensor_args=[
            {"name": "x", "direction": "input"},
            {"name": "y", "direction": "input"},
            {"name": "z", "direction": "output"},
        ],
        core_type="aiv",
    )
    # caller does: kernel_bin = extract_text_section(kernel_o)
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import env_manager
from toolchain import AscendCToolchain, CCECToolchain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper source generation
# ---------------------------------------------------------------------------

def generate_wrapper_source(
    ascendc_kernel_symbol: str,
    tensor_args: List[Dict[str, str]],
    tiling_data: Optional[bytes] = None,
    has_workspace: bool = False,
) -> str:
    """Generate a C++ wrapper that bridges PTO kernel_entry to an AscendC kernel.

    The generated source:
      - Defines ``kernel_entry(__gm__ int64_t* args)`` (PTO dispatch convention)
      - Unpacks each ``args[i]`` as a ``Tensor*``, extracts ``buffer.addr``
      - Forwards the raw GM byte-addresses to the AscendC kernel function
      - Embeds static tiling data as a ``constexpr`` array when provided

    Args:
        ascendc_kernel_symbol: The ``extern "C"`` symbol name of the AscendC
            kernel entry function in the compiled .o (e.g. ``"add_custom"``).
        tensor_args: Ordered list of tensor descriptors.  Each dict has:
            ``name``  -- human-readable label (used in comments)
            ``direction`` -- ``"input"`` | ``"output"`` | ``"inout"``
        tiling_data: Raw bytes of the tiling data blob.  If *None*, the wrapper
            passes a nullptr for the tiling parameter.
        has_workspace: Whether the AscendC kernel expects a workspace pointer.

    Returns:
        Complete C++ source string ready for ccec compilation.
    """
    lines: List[str] = []

    # --- header / includes ---------------------------------------------------
    lines.append('#include "tensor.h"')
    lines.append('')

    # --- embedded tiling data ------------------------------------------------
    if tiling_data is not None and len(tiling_data) > 0:
        lines.append(f'static const uint8_t TILING_DATA[{len(tiling_data)}] = {{')
        chunk = 16
        for i in range(0, len(tiling_data), chunk):
            segment = tiling_data[i:i + chunk]
            hex_segment = ', '.join(f'0x{b:02x}' for b in segment)
            trailing = ',' if i + chunk < len(tiling_data) else ''
            lines.append(f'    {hex_segment}{trailing}')
        lines.append('};')
        lines.append('')

    # --- kernel_entry --------------------------------------------------------
    lines.append('extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {')

    # Unpack Tensor* from args[]
    for idx, ta in enumerate(tensor_args):
        lines.append(f'    __gm__ Tensor* t_{ta["name"]} = '
                     f'reinterpret_cast<__gm__ Tensor*>(args[{idx}]);')

    lines.append('')

    # Extract raw GM byte-addresses
    for ta in tensor_args:
        lines.append(
            f'    __gm__ uint8_t* gm_{ta["name"]} = '
            f'reinterpret_cast<__gm__ uint8_t*>(t_{ta["name"]}->buffer.addr);'
        )

    lines.append('')

    # Build call arguments
    call_args = [f'gm_{ta["name"]}' for ta in tensor_args]
    if has_workspace:
        call_args.append('nullptr')
    if tiling_data is not None and len(tiling_data) > 0:
        call_args.append('(__gm__ uint8_t*)TILING_DATA')
    else:
        call_args.append('nullptr')
    call_str = ', '.join(call_args)
    lines.append(f'    {ascendc_kernel_symbol}({call_str});')

    lines.append('}')
    lines.append('')

    return '\n'.join(lines)


def generate_merged_source(
    ascendc_kernel_source: str,
    ascendc_kernel_symbol: str,
    tensor_args: List[Dict[str, str]],
    tiling_data: Optional[bytes] = None,
    has_workspace: bool = False,
) -> str:
    """Generate a single-TU source: kernel_entry wrapper + #include user kernel.

    kernel_entry is defined FIRST so it sits at offset 0 in .text (PTO
    dispatches to offset 0).  The user's AscendC kernel source is included
    AFTER, with ``__global__`` suppressed so the compiler treats the user's
    kernel as a regular callable function rather than a top-level entry point.

    Compiled as one translation unit with AscendC flags, so there are no
    cross-TU relocations to resolve.
    """
    lines: List[str] = []

    # Include kernel_operator.h first for AscendC types and intrinsics
    lines.append('#include "kernel_operator.h"')
    lines.append('#include "tensor.h"')
    lines.append('')

    # Forward-declare user's kernel function (without __global__)
    param_types = ['__gm__ uint8_t*'] * len(tensor_args)
    if has_workspace:
        param_types.append('__gm__ uint8_t*')
    param_types.append('__gm__ uint8_t*')  # tiling
    fwd_params = ', '.join(param_types)
    lines.append(f'extern "C" __aicore__ void {ascendc_kernel_symbol}({fwd_params});')
    lines.append('')

    # Embedded tiling data
    if tiling_data is not None and len(tiling_data) > 0:
        lines.append(f'static const uint8_t TILING_DATA[{len(tiling_data)}] = {{')
        chunk = 16
        for i in range(0, len(tiling_data), chunk):
            segment = tiling_data[i:i + chunk]
            hex_segment = ', '.join(f'0x{b:02x}' for b in segment)
            trailing = ',' if i + chunk < len(tiling_data) else ''
            lines.append(f'    {hex_segment}{trailing}')
        lines.append('};')
        lines.append('')

    # kernel_entry wrapper — defined FIRST so it's at .text offset 0
    # NOT __global__ — the PTO executor calls kernel functions via function
    # pointers using the standard calling convention, not the __global__ entry
    # convention.
    lines.append('extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {')

    for idx, ta in enumerate(tensor_args):
        lines.append(f'    __gm__ Tensor* t_{ta["name"]} = '
                     f'reinterpret_cast<__gm__ Tensor*>(args[{idx}]);')
    lines.append('')

    for ta in tensor_args:
        lines.append(
            f'    __gm__ uint8_t* gm_{ta["name"]} = '
            f'reinterpret_cast<__gm__ uint8_t*>(t_{ta["name"]}->buffer.addr);'
        )
    lines.append('')

    call_args = [f'gm_{ta["name"]}' for ta in tensor_args]
    if has_workspace:
        call_args.append('nullptr')
    if tiling_data is not None and len(tiling_data) > 0:
        call_args.append('(__gm__ uint8_t*)TILING_DATA')
    else:
        call_args.append('nullptr')
    call_str = ', '.join(call_args)
    lines.append(f'    {ascendc_kernel_symbol}({call_str});')

    lines.append('}')
    lines.append('')

    # Include user's AscendC kernel source AFTER kernel_entry.
    # Suppress __global__ so user's kernel becomes a regular __aicore__ function
    # (the compiler forbids calling __global__ functions from other functions).
    lines.append('#define __global__')
    lines.append(f'#include "{ascendc_kernel_source}"')
    lines.append('#undef __global__')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# AscendC compilation helpers
# ---------------------------------------------------------------------------

def extract_kernel_artifacts(
    kernel_meta_dir: str,
) -> Tuple[Optional[bytes], Optional[bytes]]:
    """Extract compiled kernel .o and tiling data from an AscendC kernel_meta dir.

    The kernel_meta directory is produced by ``npu_op_kernel_options`` with
    ``--save-temp-files``.  It typically contains:
      - ``*.o`` -- the compiled AICore kernel binary
      - ``*.json`` -- metadata including tiling information

    Args:
        kernel_meta_dir: Absolute path to a kernel_meta directory

    Returns:
        (kernel_o_bytes, tiling_bytes) -- either may be None if not found
    """
    meta_dir = Path(kernel_meta_dir)
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"kernel_meta directory not found: {kernel_meta_dir}")

    kernel_o_bytes: Optional[bytes] = None
    tiling_bytes: Optional[bytes] = None

    # Find the .o file (there should be exactly one)
    o_files = sorted(meta_dir.glob("*.o"))
    if o_files:
        with open(o_files[0], 'rb') as f:
            kernel_o_bytes = f.read()
        logger.info(f"[AscendC] Loaded kernel .o: {o_files[0]} ({len(kernel_o_bytes)} bytes)")

    # Look for tiling data -- could be a .bin or embedded in a .json
    tiling_bins = sorted(meta_dir.glob("*tiling*.bin"))
    if tiling_bins:
        with open(tiling_bins[0], 'rb') as f:
            tiling_bytes = f.read()
        logger.info(f"[AscendC] Loaded tiling data: {tiling_bins[0]} ({len(tiling_bytes)} bytes)")
    else:
        json_files = sorted(meta_dir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    meta = json.load(f)
                if "tiling_data" in meta:
                    td = meta["tiling_data"]
                    if isinstance(td, str):
                        tiling_bytes = bytes.fromhex(td)
                    elif isinstance(td, list):
                        tiling_bytes = bytes(td)
                    logger.info(f"[AscendC] Extracted tiling from {jf.name}")
                    break
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return kernel_o_bytes, tiling_bytes


# ---------------------------------------------------------------------------
# Main compiler class
# ---------------------------------------------------------------------------

class AscendCCompiler:
    """Compile AscendC operators into PTO-compatible kernel binaries.

    Typical usage::

        compiler = AscendCCompiler(platform="a2a3")
        kernel_o = compiler.compile_ascendc_kernel(
            ascendc_kernel_source="path/to/add_custom.cpp",
            ascendc_kernel_symbol="add_custom",
            tensor_args=[
                {"name": "x", "direction": "input"},
                {"name": "y", "direction": "input"},
                {"name": "z", "direction": "output"},
            ],
            core_type="aiv",
        )
        kernel_bin = extract_text_section(kernel_o)
    """

    def __init__(self, platform: str = "a2a3"):
        self.platform = platform
        self.project_root = Path(__file__).parent.parent

        if platform in ("a2a3", "a2a3sim"):
            self.platform_dir = self.project_root / "src" / "a2a3" / "platform"
        elif platform in ("a5", "a5sim"):
            self.platform_dir = self.project_root / "src" / "a5" / "platform"
        else:
            raise ValueError(f"Unknown platform: {platform}")

        if platform in ("a2a3", "a5"):
            env_manager.ensure("ASCEND_HOME_PATH")
            self.ascendc_toolchain = AscendCToolchain(platform)
            self.pto_toolchain = CCECToolchain(platform)
        else:
            self.ascendc_toolchain = None
            self.pto_toolchain = None

    def _get_runtime_include_dir(self, runtime_name: str = "tensormap_and_ringbuffer") -> str:
        arch = "a2a3" if self.platform in ("a2a3", "a2a3sim") else "a5"
        return str(self.project_root / "src" / arch / "runtime" / runtime_name / "runtime")

    def compile_ascendc_kernel(
        self,
        ascendc_kernel_source: Optional[str] = None,
        ascendc_kernel_o: Optional[bytes] = None,
        ascendc_kernel_symbol: str = "ascendc_kernel",
        tensor_args: Optional[List[Dict[str, str]]] = None,
        tiling_data: Optional[bytes] = None,
        core_type: str = "aiv",
        has_workspace: bool = False,
        extra_include_dirs: Optional[List[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Compile AscendC kernel into a PTO-compatible .o (single-TU).

        Generates a merged source file that ``#include``s the user's AscendC
        kernel and appends a ``kernel_entry`` wrapper, then compiles everything
        as one translation unit with AscendC flags.

        The caller should run ``extract_text_section(result)`` on the output.

        Args:
            ascendc_kernel_source: Path to AscendC kernel ``.cpp`` source.
            ascendc_kernel_o: Pre-compiled AscendC kernel ``.o`` bytes.
                If provided, a two-step compile+link is used instead.
            ascendc_kernel_symbol: ``extern "C"`` symbol of the AscendC kernel.
            tensor_args: Ordered tensor descriptor list for wrapper generation.
            tiling_data: Static tiling data blob (None = no tiling).
            core_type: ``"aiv"`` (vector) or ``"aic"`` (cube).
            has_workspace: Whether AscendC kernel expects a workspace pointer.
            extra_include_dirs: Extra -I paths for compilation.
            build_dir: Working directory for intermediates.

        Returns:
            Compiled .o bytes.
        """
        if self.ascendc_toolchain is None:
            raise RuntimeError(
                "AscendC kernel compilation requires a hardware platform (a2a3/a5). "
                "Simulation platforms are not supported."
            )

        if ascendc_kernel_source is None and ascendc_kernel_o is None:
            raise ValueError("Provide either ascendc_kernel_source or ascendc_kernel_o")
        if ascendc_kernel_source is not None and ascendc_kernel_o is not None:
            raise ValueError("Provide only one of ascendc_kernel_source or ascendc_kernel_o")

        if tensor_args is None:
            tensor_args = []

        work_dir = build_dir or tempfile.mkdtemp(prefix="ascendc_build_")
        os.makedirs(work_dir, exist_ok=True)

        if ascendc_kernel_o is not None:
            return self._compile_with_prebuilt_o(
                ascendc_kernel_o, ascendc_kernel_symbol, tensor_args,
                tiling_data, core_type, has_workspace, extra_include_dirs, work_dir,
            )

        # --- Single-TU approach: merge kernel source + wrapper ---
        ascendc_kernel_source = os.path.abspath(ascendc_kernel_source)

        merged_src = generate_merged_source(
            ascendc_kernel_source=ascendc_kernel_source,
            ascendc_kernel_symbol=ascendc_kernel_symbol,
            tensor_args=tensor_args,
            tiling_data=tiling_data,
            has_workspace=has_workspace,
        )

        merged_cpp_path = os.path.join(work_dir, "ascendc_merged.cpp")
        with open(merged_cpp_path, 'w') as f:
            f.write(merged_src)
        logger.info(f"[AscendC] Generated merged source: {merged_cpp_path}")

        output_o_path = os.path.join(work_dir, "ascendc_kernel.o")

        # Build include dirs: AscendC SDK + runtime (for tensor.h) + source dir
        ascendc_includes = self.ascendc_toolchain.get_ascendc_include_dirs()
        ascendc_includes.append(self._get_runtime_include_dir())
        ascendc_includes.append(os.path.dirname(ascendc_kernel_source))
        if extra_include_dirs:
            ascendc_includes.extend(
                os.path.abspath(d) for d in extra_include_dirs
            )

        compile_cmd = [self.ascendc_toolchain.cxx_path]
        compile_cmd += self.ascendc_toolchain.get_compile_flags(core_type=core_type)
        for inc in ascendc_includes:
            compile_cmd.append(f"-I{inc}")
        compile_cmd.extend(["-o", output_o_path, merged_cpp_path])

        logger.info(f"[AscendC] Compiling AscendC kernel: {ascendc_kernel_source}")
        logger.debug(f"  Command: {' '.join(compile_cmd)}")
        self._run(compile_cmd, "AscendC-Compile")

        # Link to resolve block-local relocations (R_AICORE_BLOCK_LOCAL_OFFSET12
        # for g_vecTPipePtr etc.).  extract_text_section extracts raw bytes without
        # applying relocations, so we must link first.
        linked_path = os.path.join(work_dir, "ascendc_linked.elf")
        link_cmd = [
            self.pto_toolchain.linker_path,
            "-e", "kernel_entry", "-Ttext=0",
            "-o", linked_path, output_o_path,
        ]
        logger.info("[AscendC] Linking to resolve block-local relocations")
        self._run(link_cmd, "AscendC-Link")

        with open(linked_path, 'rb') as f:
            result_bytes = f.read()

        logger.info(f"[AscendC] Linked binary: {len(result_bytes)} bytes")
        return result_bytes

    def _compile_with_prebuilt_o(
        self,
        ascendc_kernel_o: bytes,
        ascendc_kernel_symbol: str,
        tensor_args: List[Dict[str, str]],
        tiling_data: Optional[bytes],
        core_type: str,
        has_workspace: bool,
        extra_include_dirs: Optional[List[str]],
        work_dir: str,
    ) -> bytes:
        """Fallback: pre-compiled .o needs wrapper + link."""
        kernel_o_path = os.path.join(work_dir, "ascendc_kernel.o")
        with open(kernel_o_path, 'wb') as f:
            f.write(ascendc_kernel_o)
        logger.info(f"[AscendC] Using pre-compiled kernel .o ({len(ascendc_kernel_o)} bytes)")

        # Generate + compile PTO wrapper with AscendC flags
        wrapper_src = generate_wrapper_source(
            ascendc_kernel_symbol=ascendc_kernel_symbol,
            tensor_args=tensor_args,
            tiling_data=tiling_data,
            has_workspace=has_workspace,
        )

        wrapper_cpp_path = os.path.join(work_dir, "ascendc_wrapper.cpp")
        with open(wrapper_cpp_path, 'w') as f:
            f.write(wrapper_src)

        wrapper_o_path = os.path.join(work_dir, "ascendc_wrapper.o")
        wrapper_includes = [self._get_runtime_include_dir()]
        if extra_include_dirs:
            wrapper_includes.extend(extra_include_dirs)

        wrapper_cmd = [self.ascendc_toolchain.cxx_path]
        wrapper_cmd += self.ascendc_toolchain.get_compile_flags(core_type=core_type)
        for inc in wrapper_includes:
            wrapper_cmd.append(f"-I{os.path.abspath(inc)}")
        wrapper_cmd.extend(["-o", wrapper_o_path, wrapper_cpp_path])
        self._run(wrapper_cmd, "AscendC-Wrapper")

        # Full link to resolve cross-TU calls
        combined_o_path = os.path.join(work_dir, "ascendc_combined.o")
        link_cmd = [
            self.pto_toolchain.linker_path,
            "-e", "kernel_entry", "-Ttext=0",
            "-o", combined_o_path,
            wrapper_o_path, kernel_o_path,
        ]
        self._run(link_cmd, "AscendC-Link")

        with open(combined_o_path, 'rb') as f:
            result_bytes = f.read()

        logger.info(f"[AscendC] Combined binary: {len(result_bytes)} bytes")
        return result_bytes

    def compile_from_kernel_meta(
        self,
        kernel_meta_dir: str,
        ascendc_kernel_source: str,
        ascendc_kernel_symbol: str,
        tensor_args: List[Dict[str, str]],
        core_type: str = "aiv",
        has_workspace: bool = False,
        extra_include_dirs: Optional[List[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Convenience: extract tiling from kernel_meta, then compile source."""
        _, tiling_data = extract_kernel_artifacts(kernel_meta_dir)

        return self.compile_ascendc_kernel(
            ascendc_kernel_source=ascendc_kernel_source,
            ascendc_kernel_symbol=ascendc_kernel_symbol,
            tensor_args=tensor_args,
            tiling_data=tiling_data,
            core_type=core_type,
            has_workspace=has_workspace,
            extra_include_dirs=extra_include_dirs,
            build_dir=build_dir,
        )

    @staticmethod
    def _run(cmd: List[str], label: str) -> subprocess.CompletedProcess:
        """Run a subprocess with logging and error handling."""
        logger.debug(f"[{label}] {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                logger.debug(f"[{label}] stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"[{label}] stderr:\n{result.stderr}")
            if result.returncode != 0:
                raise RuntimeError(
                    f"{label} failed (exit {result.returncode}):\n{result.stderr}"
                )
            return result
        except FileNotFoundError as exc:
            raise RuntimeError(f"{label}: tool not found -- {exc}") from exc
