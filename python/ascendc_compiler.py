"""
AscendC Kernel Compiler for PTO Runtime Integration

Wraps pre-compiled AscendC operator binaries (.o) into PTO-compatible kernel
binaries.  The key bridge is a generated "wrapper kernel" that adapts PTO's
unified kernel entry (int64_t* args with Tensor pointers) to AscendC's
per-tensor GM_ADDR calling convention.

The AscendC kernel .o is compiled externally (e.g. via tikcpp_smoke +
npu_op_kernel_options --save-temp-files).  This module only consumes the
pre-compiled .o — it does NOT compile AscendC source code.

Workflow (compile wrapper + link):
  1. Read pre-compiled kernel .o  (from kernel_meta/ or provided directly)
  2. Generate wrapper .cpp  (kernel_entry + embedded tiling data)
  3. Compile wrapper with PTO compiler (ccec -x cce) -> wrapper.o
  4. Link: ld.lld -e kernel_entry -Ttext=0 wrapper.o kernel.o -> combined.elf
  5. extract_text_section(combined.elf) -> kernel binary  (done by caller)

The wrapper is always compiled with PTO flags (-x cce) because it only needs
tensor.h and contains no AscendC code.

Usage:
    compiler = AscendCCompiler(platform="a2a3")
    kernel_o, tiling = extract_kernel_artifacts("/path/to/kernel_meta")
    combined = compiler.compile_ascendc_kernel(
        ascendc_kernel_o=kernel_o,
        ascendc_kernel_symbol="add_custom",
        tensor_args=[
            {"name": "x", "direction": "input"},
            {"name": "y", "direction": "input"},
            {"name": "z", "direction": "output"},
        ],
        tiling_data=tiling,
        core_type="aiv",
    )
    # caller does: kernel_bin = extract_text_section(combined)
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import env_manager
from toolchain import CCECToolchain

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

    Compiled with PTO flags (ccec -x cce).  Only needs tensor.h, no AscendC SDK.

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
    # Ensure __gm__ and __aicore__ are defined (PTO ccec -x cce uses [aicore])
    lines.append('#ifndef __gm__')
    lines.append('#define __gm__')
    lines.append('#endif')
    lines.append('#ifndef __aicore__')
    lines.append('#define __aicore__ [aicore]')
    lines.append('#endif')
    lines.append('')

    # --- AscendC kernel forward declaration ----------------------------------
    # Tensor pointers are in GM address space; workspace and tiling use plain
    # uint8_t* because the wrapper may embed tiling in const data (not GM) and
    # PTO's ccec treats __gm__ as a real address-space qualifier.  The linker
    # resolves symbols by name only, so parameter types don't need to match
    # the AscendC side exactly.
    param_strs = ['__gm__ uint8_t*'] * len(tensor_args)
    if has_workspace:
        param_strs.append('uint8_t*')  # workspace (nullptr from wrapper)
    param_strs.append('uint8_t*')      # tiling (may be const data, not GM)
    decl_params = ', '.join(param_strs)
    lines.append(f'extern "C" __aicore__ void {ascendc_kernel_symbol}({decl_params});')
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
        call_args.append('(uint8_t*)TILING_DATA')
    else:
        call_args.append('nullptr')
    call_str = ', '.join(call_args)
    lines.append(f'    {ascendc_kernel_symbol}({call_str});')

    lines.append('}')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# AscendC artifact extraction
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
    """Wrap pre-compiled AscendC kernel binaries into PTO-compatible kernel entries.

    Compile wrapper + link:
      1. Write pre-compiled kernel .o to build dir
      2. Generate + compile wrapper.o with PTO compiler (ccec -x cce)
      3. Link wrapper.o + kernel.o via ld.lld (wrapper first -> offset 0)

    The AscendC kernel .o must be compiled externally (e.g. via tikcpp_smoke
    + npu_op_kernel_options --save-temp-files).

    Typical usage::

        compiler = AscendCCompiler(platform="a2a3")
        combined = compiler.compile_ascendc_kernel(
            ascendc_kernel_o=kernel_o_bytes,
            ascendc_kernel_symbol="add_custom",
            tensor_args=[...],
            tiling_data=tiling_bytes,
        )
        kernel_bin = extract_text_section(combined)
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
            self.pto_toolchain = CCECToolchain(platform)
        else:
            self.pto_toolchain = None

    def _get_runtime_include_dir(self, runtime_name: str = "tensormap_and_ringbuffer") -> str:
        arch = "a2a3" if self.platform in ("a2a3", "a2a3sim") else "a5"
        return str(self.project_root / "src" / arch / "runtime" / runtime_name / "runtime")

    def compile_ascendc_kernel(
        self,
        ascendc_kernel_o: bytes,
        ascendc_kernel_symbol: str = "ascendc_kernel",
        tensor_args: Optional[List[Dict[str, str]]] = None,
        tiling_data: Optional[bytes] = None,
        core_type: str = "aiv",
        has_workspace: bool = False,
        extra_include_dirs: Optional[List[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Wrap a pre-compiled AscendC kernel .o into a PTO-compatible linked ELF.

        Generates a PTO wrapper, compiles it with PTO flags, then links it with
        the kernel .o so that ``kernel_entry`` sits at .text offset 0.

        The caller should run ``extract_text_section(result)`` on the output.

        Args:
            ascendc_kernel_o: Pre-compiled kernel ``.o`` bytes (from external
                AscendC compilation, e.g. tikcpp_smoke kernel_meta/).
            ascendc_kernel_symbol: ``extern "C"`` symbol of the AscendC kernel.
            tensor_args: Ordered tensor descriptor list for wrapper generation.
            tiling_data: Static tiling data blob (None = no tiling).
            core_type: ``"aiv"`` (vector) or ``"aic"`` (cube).
            has_workspace: Whether AscendC kernel expects a workspace pointer.
            extra_include_dirs: Extra -I paths for wrapper compilation.
            build_dir: Working directory for intermediates.

        Returns:
            Linked ELF bytes (pass to ``extract_text_section``).
        """
        if self.pto_toolchain is None:
            raise RuntimeError(
                "AscendC kernel wrapping requires a hardware platform (a2a3/a5). "
                "Simulation platforms are not supported."
            )

        if tensor_args is None:
            tensor_args = []

        work_dir = build_dir or tempfile.mkdtemp(prefix="ascendc_build_")
        os.makedirs(work_dir, exist_ok=True)

        # --- Step 1: Write pre-compiled kernel .o -----------------------------
        kernel_o_path = os.path.join(work_dir, "ascendc_kernel.o")
        with open(kernel_o_path, 'wb') as f:
            f.write(ascendc_kernel_o)
        logger.info(f"[AscendC] Using pre-compiled kernel .o "
                    f"({len(ascendc_kernel_o)} bytes)")

        # --- Step 2: Generate + compile wrapper with PTO flags ---------------
        wrapper_src = generate_wrapper_source(
            ascendc_kernel_symbol=ascendc_kernel_symbol,
            tensor_args=tensor_args,
            tiling_data=tiling_data,
            has_workspace=has_workspace,
        )

        wrapper_cpp_path = os.path.join(work_dir, "ascendc_wrapper.cpp")
        with open(wrapper_cpp_path, 'w') as f:
            f.write(wrapper_src)
        logger.info(f"[AscendC] Generated wrapper: {wrapper_cpp_path}")

        wrapper_o_path = os.path.join(work_dir, "ascendc_wrapper.o")
        wrapper_includes = [self._get_runtime_include_dir()]
        if extra_include_dirs:
            wrapper_includes.extend(extra_include_dirs)

        wrapper_cmd = [self.pto_toolchain.cxx_path]
        wrapper_cmd += self.pto_toolchain.get_compile_flags(core_type=core_type)
        for inc in wrapper_includes:
            wrapper_cmd.append(f"-I{os.path.abspath(inc)}")
        wrapper_cmd.extend(["-o", wrapper_o_path, wrapper_cpp_path])

        logger.info("[AscendC] Compiling wrapper with PTO flags")
        self._run(wrapper_cmd, "PTO-Wrapper")

        # --- Step 3: Link wrapper.o + kernel.o -------------------------------
        # wrapper.o listed FIRST so kernel_entry lands at .text offset 0
        combined_path = os.path.join(work_dir, "ascendc_combined.elf")
        link_cmd = [
            self.pto_toolchain.linker_path,
            "-e", "kernel_entry", "-Ttext=0",
            "-o", combined_path,
            wrapper_o_path, kernel_o_path,
        ]

        logger.info("[AscendC] Linking wrapper.o + kernel.o")
        self._run(link_cmd, "AscendC-Link")

        with open(combined_path, 'rb') as f:
            result_bytes = f.read()

        logger.info(f"[AscendC] Combined binary: {len(result_bytes)} bytes")
        return result_bytes

    def compile_from_kernel_meta(
        self,
        kernel_meta_dir: str,
        ascendc_kernel_symbol: str,
        tensor_args: List[Dict[str, str]],
        core_type: str = "aiv",
        has_workspace: bool = False,
        extra_include_dirs: Optional[List[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Convenience: extract .o and tiling from kernel_meta, then wrap+link."""
        kernel_o, tiling_data = extract_kernel_artifacts(kernel_meta_dir)
        if kernel_o is None:
            raise FileNotFoundError(
                f"No .o file found in kernel_meta: {kernel_meta_dir}"
            )

        return self.compile_ascendc_kernel(
            ascendc_kernel_o=kernel_o,
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
