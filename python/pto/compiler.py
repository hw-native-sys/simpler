"""Compilation utilities for PTO runtime.

Wraps the existing RuntimeBuilder + KernelCompiler pipeline and adds
SHA256-based caching.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from .types import CompiledPackage, KernelSource

logger = logging.getLogger(__name__)

# Ensure simpler's python/ is importable
_PYTHON_DIR = str(Path(__file__).resolve().parent.parent)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# Also need examples/scripts for elf_parser
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _cache_key(platform: str, runtime_name: str, orch_source: str,
               kernel_sources: list[dict], orch_func: str) -> str:
    """Compute a SHA256 cache key from source contents + config."""
    h = hashlib.sha256()
    h.update(platform.encode())
    h.update(runtime_name.encode())
    h.update(orch_func.encode())

    # Hash orchestration source content
    orch_path = Path(orch_source)
    if orch_path.is_file():
        h.update(orch_path.read_bytes())
    else:
        h.update(orch_source.encode())

    # Hash kernel source contents
    for ks in sorted(kernel_sources, key=lambda k: k.get("func_id", 0)):
        src_path = Path(ks["source"])
        if src_path.is_file():
            h.update(src_path.read_bytes())
        else:
            h.update(ks["source"].encode())
        h.update(ks.get("core_type", "aiv").encode())
        h.update(str(ks.get("func_id", 0)).encode())

    return h.hexdigest()


def compile(
    platform: str,
    runtime_name: str = "tensormap_and_ringbuffer",
    orch_source: str = "",
    kernel_sources: Optional[list[dict]] = None,
    orch_func: str = "aicpu_orchestration_entry",
    block_dim: int = 1,
    aicpu_thread_num: int = 1,
    orch_thread_num: int = 1,
    cache_dir: Optional[str] = None,
    build_dir: Optional[str] = None,
    extra_include_dirs: Optional[list[str]] = None,
) -> CompiledPackage:
    """Compile all artifacts for an L2 execution package.

    Wraps RuntimeBuilder.build() + KernelCompiler to produce a
    CompiledPackage containing all binaries needed for a single L2 run.

    Args:
        platform: "a2a3" | "a2a3sim" | "a5" | "a5sim"
        runtime_name: Which L2 runtime variant to compile
        orch_source: Path to orchestration .cpp source
        kernel_sources: List of dicts with keys: source, core_type, func_id
        orch_func: Orchestration entry function name
        block_dim: Number of AICore blocks
        aicpu_thread_num: AICPU thread count
        orch_thread_num: Orchestration thread count
        cache_dir: Optional cache directory for compiled artifacts
        build_dir: Optional build directory for intermediate files
        extra_include_dirs: Additional include directories for kernel compilation

    Returns:
        CompiledPackage with all binaries ready for L2 execution
    """
    if kernel_sources is None:
        kernel_sources = []

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        key = _cache_key(platform, runtime_name, orch_source,
                         kernel_sources, orch_func)
        cached_file = cache_path / f"{key}.pkg"
        if cached_file.exists():
            logger.info(f"Cache hit: {cached_file}")
            with open(cached_file, "rb") as f:
                return pickle.load(f)

    from runtime_builder import RuntimeBuilder
    from elf_parser import extract_text_section

    builder = RuntimeBuilder(platform=platform)
    kernel_compiler = builder.get_kernel_compiler()

    # Build runtime (host.so, aicpu.so, aicore.o)
    logger.info("Compiling runtime...")
    host_binary, aicpu_binary, aicore_binary = builder.build(
        runtime_name, build_dir)

    # Compile orchestration
    orch_binary = b""
    if orch_source:
        logger.info(f"Compiling orchestration: {orch_source}")
        orch_binary = kernel_compiler.compile_orchestration(
            runtime_name, orch_source, build_dir=build_dir)

    # Compile kernels
    compiled_kernels = []
    for ks in kernel_sources:
        src = ks["source"]
        core_type = ks.get("core_type", "aiv")
        func_id = ks.get("func_id", 0)

        logger.info(f"Compiling kernel: {src} (core_type={core_type}, func_id={func_id})")

        pto_isa_root = os.environ.get("PTO_ISA_ROOT", "")
        incore_o = kernel_compiler.compile_incore(
            src, core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=extra_include_dirs or [],
            build_dir=build_dir,
        )

        # Extract .text section for hardware platforms
        if not platform.endswith("sim"):
            kernel_bin = extract_text_section(incore_o)
        else:
            kernel_bin = incore_o

        compiled_kernels.append((func_id, kernel_bin))

    pkg = CompiledPackage(
        platform=platform,
        runtime_name=runtime_name,
        host_binary=host_binary,
        aicpu_binary=aicpu_binary,
        aicore_binary=aicore_binary,
        orch_binary=orch_binary,
        orch_func=orch_func,
        kernel_binaries=compiled_kernels,
        block_dim=block_dim,
        aicpu_thread_num=aicpu_thread_num,
        orch_thread_num=orch_thread_num,
    )

    # Save to cache
    if cache_dir:
        logger.info(f"Caching compiled package: {cached_file}")
        with open(cached_file, "wb") as f:
            pickle.dump(pkg, f)

    return pkg
