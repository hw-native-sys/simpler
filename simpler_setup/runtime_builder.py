# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import fcntl
import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .environment import PROJECT_ROOT
from .platform_info import TARGETS, load_build_config, parse_platform
from .runtime_compiler import RuntimeCompiler

logger = logging.getLogger(__name__)

_GIT_COMMIT_FILE = ".git_commit"
_DEFAULT_PROFILING_CONFIG = {
    "SIMPLER_DFX": "1",
    "SIMPLER_ORCH_PROFILING": "0",
    "SIMPLER_SCHED_PROFILING": "0",
    "SIMPLER_TENSORMAP_PROFILING": "0",
}


def platform_embeds_pto_isa(platform: str) -> bool:
    """Return True when this platform's onboard host embeds PTO-ISA headers.

    a2a3 and a5 onboard always embed a PTO-ISA async workspace. Simulation
    platforms do not embed PTO-ISA host headers.
    """
    arch, variant = parse_platform(platform)
    if variant != "onboard":
        return False
    return arch in {"a2a3", "a5"}


def _runtime_source_inputs(repo_root: Path) -> dict:
    """Snapshot build inputs independently of Git history and file mtimes."""
    if not (repo_root / "src").is_dir():
        return {}
    files = []
    for directory in (repo_root / "src", repo_root / "cmake"):
        for base, directories, names in os.walk(directory):
            directories[:] = sorted(d for d in directories if d not in {"build", "__pycache__", ".git"})
            for name in names:
                path = Path(base) / name
                if name != "compile_commands.json" and path.suffix not in {".md", ".pyc", ".pyo"}:
                    files.append(path)
    for name in (
        "runtime_builder.py",
        "runtime_compiler.py",
        "toolchain.py",
        "platform_info.py",
        "environment.py",
        "pto_isa.py",
    ):
        path = repo_root / "simpler_setup" / name
        if path.is_file():
            files.append(path)
    return {
        str(path.relative_to(repo_root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(files)
    }


def _runtime_source_stamp(repo_root: Path) -> str:
    inputs = _runtime_source_inputs(repo_root)
    if not inputs:
        return ""
    digest = hashlib.sha256()
    for name, entry in inputs.items():
        digest.update(name.encode() + b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def _runtime_compiler_inputs(compiler) -> dict:
    """Tool replacements invalidate even custom compiler commands at the same path."""
    result = {}
    for name in ("host", "aicpu", "aicore"):
        target = getattr(compiler, name + "_target", None)
        toolchain = getattr(target, "toolchain", None)
        for field in ("cxx_path", "linker_path"):
            path = getattr(toolchain, field, None)
            if isinstance(path, str) and Path(path).is_file():
                result[f"@toolchain/{name}/{field}"] = {
                    "sha256": hashlib.sha256(path.encode() + Path(path).read_bytes()).hexdigest(),
                    "mtime_ns": 0,
                }
    return result


def _normalize_overlay_env(name: str) -> Optional[str]:
    """Return "ON"/"OFF" for a CMake-style bool env var, or None if unset/invalid."""
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES", "Y"}:
        return "ON"
    if normalized in {"0", "OFF", "FALSE", "NO", "N"}:
        return "OFF"
    return None


def _a5_async_workspace_overlay_defines() -> dict[str, str]:
    """Forward the optional RDMA provider selection, clearing stale cached ON values.

    A5 SDMA and URMA providers are always compiled. Compilation creates no
    resources; communication domains select their engines at allocation time.
    """
    return {"SIMPLER_ENABLE_PTO_RDMA_WORKSPACE": _normalize_overlay_env("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE") or "OFF"}


def _abbrev_stamp(stamp: str) -> str:
    """Abbreviate each commit in a (possibly composite) cache stamp for logging.

    The stamp is ``<runtime_sha>`` or ``<runtime_sha>:pto-isa=<isa_sha>``.
    Truncating the whole string (e.g. ``stamp[:20]``) hides a pto-isa-only
    change: the 40-char runtime sha prefix is identical, so both the old and
    new stamps render the same — masking exactly the bump the log is meant to
    surface (issue #1139). Shorten each sha segment independently so a
    pto-isa-only change stays visible.
    """
    runtime, sep, isa = stamp.partition(":pto-isa=")
    if sep:
        return f"{runtime[:12]}:pto-isa={isa[:12]}"
    return runtime[:12]


def _invalidate_cache_if_stale(target_cache_dir: Path, current_stamp: str, *, inputs: Optional[dict] = None) -> None:
    """Preserve CMake incrementality; clean when timestamps cannot prove freshness.

    Changed inputs newer than every object are handled by normal CMake depfiles.
    Backdated/removed inputs, tool replacements, pin changes, and missing identity
    require a clean build. The stored input map is tied to its source stamp.
    """
    commit_file = target_cache_dir / _GIT_COMMIT_FILE
    inputs_file = target_cache_dir / ".runtime_inputs.json"
    cached_stamp = commit_file.read_text().strip() if commit_file.is_file() else ""
    try:
        previous = json.loads(inputs_file.read_text()) if inputs_file.is_file() else {}
    except (ValueError, OSError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    raw_files = previous.get("files")
    valid_inputs = isinstance(raw_files, dict) and all(
        isinstance(entry, dict) and isinstance(entry.get("sha256"), str) and isinstance(entry.get("mtime_ns"), int)
        for entry in raw_files.values()
    )
    old_inputs = raw_files if isinstance(raw_files, dict) else {}
    trusted = inputs is not None and previous.get("stamp") == cached_stamp and valid_inputs
    changed = []
    if trusted and inputs is not None:
        changed = [
            name
            for name in old_inputs.keys() | inputs.keys()
            if old_inputs.get(name, {}).get("sha256") != inputs.get(name, {}).get("sha256")
        ]
    clean = (
        not current_stamp
        or cached_stamp != current_stamp
        or (trusted and bool(changed))
        or (inputs is not None and not trusted)
    )
    if clean and current_stamp and cached_stamp and trusted and inputs is not None:
        same_pin = cached_stamp.partition(":pto-isa=")[2] == current_stamp.partition(":pto-isa=")[2]
        newest_object = max(
            (
                p.stat().st_mtime_ns
                for p in target_cache_dir.rglob("*")
                if p.is_file() and p.suffix in {".o", ".so", ".a"}
            ),
            default=0,
        )
        cmake_can_detect = all(
            name in inputs and not name.startswith("@toolchain/") and inputs[name]["mtime_ns"] > newest_object
            for name in changed
        )
        if same_pin and cmake_can_detect:
            clean = False
    if clean and target_cache_dir.is_dir():
        logger.info(
            "Runtime cache requires clean rebuild (%s -> %s): %s",
            _abbrev_stamp(cached_stamp),
            _abbrev_stamp(current_stamp),
            target_cache_dir,
        )
        shutil.rmtree(target_cache_dir)
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    if not current_stamp:
        return
    commit_file.write_text(current_stamp + "\n")
    if inputs is not None:
        inputs_file.write_text(json.dumps({"stamp": current_stamp, "files": inputs}, sort_keys=True) + "\n")


@dataclass
class RuntimeBinaries:
    """Paths to the compiled runtime binaries.

    ``dispatcher_path`` points at ``libsimpler_aicpu_dispatcher.so`` and is
    required for onboard platforms (host bootstrap reads its bytes and ships
    them to the device alongside the inner SO). Sim platforms have no
    dispatcher; the field is ``None`` there. ``_lookup_binaries`` resolves
    and validates the path against the build output directory.

    ``sdma_warmup_path`` points at ``sdma_warmup_kernel.o``, the vector-only ELF
    that warms the SDMA control path once at worker init. Unlike the dispatcher
    it is optional: only arches whose aicore CMakeLists builds it have one, and
    a missing warmup ELF costs latency on the first TPREFETCH_ASYNC rather than
    breaking the worker, so the field is left ``None`` instead of raising.
    """

    host_path: Path
    aicpu_path: Path
    aicore_path: Path
    sim_context_path: Optional[Path] = None
    dispatcher_path: Optional[Path] = None
    sdma_warmup_path: Optional[Path] = None


class RuntimeBuilder:
    """Discovers and builds runtime implementations from src/runtime/.

    Accepts a platform selection to provide correctly configured
    RuntimeCompiler and KernelCompiler instances. Runtime and platform
    are orthogonal — the same runtime (e.g., host_build_graph) can
    be compiled for any platform (e.g., a2a3, a2a3sim).
    """

    _CACHE_DIR = PROJECT_ROOT / "build" / "cache"
    _LIB_DIR = PROJECT_ROOT / "build" / "lib"

    # Defaults for compile_commands.json placement (matches old gen_compile_commands.py).
    # Platform dirs get compdb from the most feature-rich runtime;
    # runtime dirs get compdb from the onboard (real hardware) variant.
    _COMPDB_RUNTIME = "tensormap_and_ringbuffer"
    _COMPDB_VARIANT = "onboard"

    def __init__(self, platform: str = "a2a3", *, aicore_build_profile: Optional[str] = None):
        """
        Initialize RuntimeBuilder with platform selection.

        Args:
            platform: Target platform ("a2a3", "a2a3sim", "a5", or "a5sim")
            aicore_build_profile: "dev" selects A5 HBG debug compilation in
                separate cache and artifact directories. None reads
                SIMPLER_AICORE_BUILD_PROFILE, which defaults to "release".
        """
        if aicore_build_profile is None:
            aicore_build_profile = os.environ.get("SIMPLER_AICORE_BUILD_PROFILE", "release")
        if aicore_build_profile not in ("release", "dev"):
            raise ValueError(f"Unknown AICore build profile: {aicore_build_profile}")
        if aicore_build_profile == "dev" and platform != "a5":
            raise ValueError("The dev AICore build profile requires platform='a5'")
        self.aicore_build_profile = aicore_build_profile
        if aicore_build_profile != "release":
            self._CACHE_DIR = self._CACHE_DIR / "profiles" / aicore_build_profile
            self._LIB_DIR = self._LIB_DIR / "profiles" / aicore_build_profile
        self.platform = platform
        self._arch, self._variant = parse_platform(platform)

        runtime_root = PROJECT_ROOT
        self.runtime_root = runtime_root

        self.runtime_dir = runtime_root / "src" / self._arch / "runtime"

        # Discover available runtime implementations
        self._runtimes = {}
        if self.runtime_dir.is_dir():
            for entry in sorted(self.runtime_dir.iterdir()):
                config_path = entry / "build_config.py"
                if entry.is_dir() and config_path.is_file():
                    self._runtimes[entry.name] = config_path

        # Create platform-configured compiler
        self._runtime_compiler = RuntimeCompiler.get_instance(platform=platform)

    def list_runtimes(self) -> list:
        """Return names of discovered runtime implementations."""
        return list(self._runtimes.keys())

    def _validate_runtime(self, name: str) -> None:
        if name not in self._runtimes:
            available = ", ".join(self._runtimes.keys()) or "(none)"
            raise ValueError(
                f"Runtime '{name}' is not available for platform '{self.platform}'.\n"
                f"Available runtimes for {self.platform}: {available}\n"
                f"Note: Different platforms may support different runtimes. "
                f"Check {self.runtime_dir} for available implementations."
            )

    def _resolve_target_dirs(self, config_dir: Path, build_config: dict, target: str):
        """Resolve include and source dirs for a target from build_config."""
        cfg = build_config[target]
        include_dirs = [str((config_dir / p).resolve()) for p in cfg["include_dirs"]]
        source_dirs = [str((config_dir / p).resolve()) for p in cfg["source_dirs"]]
        if self._arch == "a5" and config_dir.name == "tensormap_and_ringbuffer":
            from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

            include_dirs.append(str(Path(ensure_pto_isa_root()) / "include"))
        return include_dirs, source_dirs

    def _requires_pto_isa_metadata_validation(self) -> bool:
        """Return True when this runtime embeds PTO-ISA headers into host code.

        Driven by :func:`platform_embeds_pto_isa`: a2a3 and a5 onboard always
        embed async workspace backends; simulation platforms do not.
        """
        return platform_embeds_pto_isa(self.platform)

    def _resolve_build_pto_isa_commit(self) -> str:
        """Return the pinned pto-isa commit baked into this build.

        When host code embeds pto-isa headers (a2a3 or a5 onboard), a pto-isa
        bump must invalidate that build's
        cmake cache even when the runtime source content is unchanged (issue #1139:
        a stale host_runtime.so compiled against the old headers otherwise
        survives a reinstall). A5 simulation includes ISA workspace types too,
        so its cache stamp includes the pin. Other simulation builds return ""
        and leave the
        stamp keyed on the runtime source content.

        ``pto_isa.pin`` is the single source of truth. If the pin is missing or
        invalid, let ``read_pto_isa_pin`` raise so an embedding onboard build
        cannot silently proceed with unknown PTO-ISA headers.
        """
        if not self._requires_pto_isa_metadata_validation() and self.platform != "a5sim":
            return ""
        from .pto_isa import read_pto_isa_pin  # noqa: PLC0415

        return read_pto_isa_pin()

    def _build_cache_stamp(self, pto_isa_commit: Optional[str] = None) -> str:
        """Identify runtime source contents and the pinned PTO-ISA headers."""
        runtime_commit = _runtime_source_stamp(PROJECT_ROOT)
        if not runtime_commit:
            return ""
        if pto_isa_commit is None:
            pto_isa_commit = self._resolve_build_pto_isa_commit()
        if pto_isa_commit:
            return f"{runtime_commit}:pto-isa={pto_isa_commit}"
        return runtime_commit

    def _lookup_binaries(self, name: str, output_dir: Path) -> RuntimeBinaries:
        """Look up pre-built binaries from output_dir.

        Resolves binary names from the compiler's target configs, then
        checks that each file exists.

        Raises:
            FileNotFoundError: If any binary is missing.
        """
        if self._requires_pto_isa_metadata_validation():
            from . import pto_isa  # noqa: PLC0415

            runtime_key = pto_isa.pto_isa_runtime_artifact_key(self._arch, self._variant, name)
            pto_isa.validate_runtime_pto_isa_current_pin(self._LIB_DIR, runtime_key=runtime_key)

        compiler = self._runtime_compiler
        paths = {}
        missing = []
        for target in TARGETS:
            target_obj = getattr(compiler, f"{target}_target")
            binary = output_dir / target_obj.get_binary_name()
            paths[target] = binary
            if not binary.is_file():
                missing.append(str(binary))

        if missing:
            raise FileNotFoundError(
                f"Pre-built runtime binaries not found for '{name}' "
                f"(platform={self.platform}):\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\nRun 'pip install --no-build-isolation .' to compile them."
            )

        # Validate sim_context SO exists for sim platforms
        sim_context_path = self._resolve_sim_context_path()
        if sim_context_path is not None and not sim_context_path.is_file():
            raise FileNotFoundError(
                f"Pre-built libcpu_sim_context.so not found at {sim_context_path}.\n"
                "Run 'pip install --no-build-isolation .' to compile it."
            )

        # Resolve and validate libsimpler_aicpu_dispatcher.so for onboard
        # platforms. runtime_compiler stages one copy per arch into
        # <LIB_DIR>/<arch>/dispatcher/ (shared across all runtimes); sim
        # platforms have no dispatcher.
        dispatcher_path = self._resolve_dispatcher_path()
        if dispatcher_path is not None and not dispatcher_path.is_file():
            raise FileNotFoundError(
                f"Pre-built libsimpler_aicpu_dispatcher.so not found at {dispatcher_path}.\n"
                "Run 'pip install --no-build-isolation .' to compile it."
            )

        return RuntimeBinaries(
            host_path=paths["host"],
            aicpu_path=paths["aicpu"],
            aicore_path=paths["aicore"],
            sim_context_path=sim_context_path,
            dispatcher_path=dispatcher_path,
            sdma_warmup_path=self._resolve_sdma_warmup_path(),
        )

    def get_binaries(
        self,
        name: str,
        build: bool = False,
        *,
        build_shared: bool = True,
        profiling_config: Optional[dict[str, str]] = None,
    ) -> RuntimeBinaries:
        """Return paths to compiled runtime binaries.

        By default, looks up pre-built binaries from build/lib/. When
        build=True, runs cmake configure + make using persistent build
        directories under build/cache/ for incremental compilation.

        Args:
            name: Name of the runtime implementation (e.g. 'host_build_graph')
            build: If True, compile the runtime before returning paths.
                If False (default), return pre-built binary paths.
            build_shared: If True (default), build the process-global helper
                libraries as part of this call. ``build_runtimes.py`` sets this
                to False after pre-building them once for the whole batch.
            profiling_config: Complete profiling macro configuration passed to
                each runtime target's CMake configure step. Simulation builds
                explicitly use source defaults when this is omitted, preventing
                values from an earlier profiling build remaining in CMake cache.

        Returns:
            RuntimeBinaries with paths to host, aicpu, and aicore binaries.

        Raises:
            FileNotFoundError: If build=False and pre-built binaries are missing.
        """
        if self.aicore_build_profile == "dev" and name != "host_build_graph":
            raise ValueError("The dev AICore build profile supports only A5 host_build_graph")
        self._validate_runtime(name)

        arch, variant = self._arch, self._variant
        output_dir = self._LIB_DIR / arch / variant / name
        # Per-arch shared destination for libsimpler_aicpu_dispatcher.so. The
        # dispatcher has no runtime-specific code, so all runtimes on a given
        # arch reuse the same SO instead of carrying a copy each (~50 KB × N).
        # None on sim — sim variants have no dispatcher.
        dispatcher_staging_dir = self._LIB_DIR / arch / "dispatcher" if variant != "sim" else None
        # Same reasoning for the vector-only SDMA warmup ELF: no runtime-specific
        # code, so one copy per arch. None on sim — sim has no device SDMA.
        sdma_warmup_staging_dir = self._LIB_DIR / arch / "sdma_warmup" if variant != "sim" else None

        if not build:
            return self._lookup_binaries(name, output_dir)

        config_path = self._runtimes[name]
        config_dir = config_path.parent
        build_config = load_build_config(config_path)

        compiler = self._runtime_compiler

        build_pto_isa_commit = self._resolve_build_pto_isa_commit()
        cache_stamp = self._build_cache_stamp(build_pto_isa_commit)
        cache_inputs = {**_runtime_source_inputs(PROJECT_ROOT), **_runtime_compiler_inputs(compiler)}
        # Every build gets a fully specified configuration, so a value from an earlier
        # profiling build cannot survive in the CMake cache as a default. Merged rather
        # than substituted: cmake/profiling_config.cmake defaults only the variables
        # that are *undefined*, and a name left out of a partial mapping is still
        # defined in the target's cache from whatever configured it last.
        effective_profiling_config = {**_DEFAULT_PROFILING_CONFIG, **(profiling_config or {})}

        def _compile_target(target: str) -> Path:
            include_dirs, source_dirs = self._resolve_target_dirs(config_dir, build_config, target)
            defines = dict(effective_profiling_config or {})
            # Pin-resolved checkout path for CMake include dirs (#1403). Needed by
            # host (SDMA workspace manager) and by aicore (the SDMA warmup kernel's
            # vector-only target). Prefer the path already resolved on the
            # RuntimeCompiler; fall back to ensure_pto_isa_root when embedding but
            # the compiler field is unset (e.g. mocked UT / partial construction).
            if target in ("host", "aicore") and self._requires_pto_isa_metadata_validation():
                pto_root = getattr(compiler, "pto_isa_root", None)
                if not isinstance(pto_root, str) or not pto_root:
                    from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

                    pto_root = ensure_pto_isa_root(verbose=True)
                defines["PTO_ISA_ROOT"] = pto_root
            if target == "host":
                # host_runtime.so is built once per runtime implementation, so
                # bake that immutable artifact identity into the shared host
                # platform code.  Runtime code must not grow a virtual/static
                # API merely to report the directory it was compiled from.
                defines["SIMPLER_RUNTIME_NAME"] = name
                if build_pto_isa_commit:
                    defines["SIMPLER_PTO_ISA_BUILD_COMMIT"] = build_pto_isa_commit
                # Forward the full a5 backend selection — one explicit value per
                # overlay — so a value left in the CMake cache by an earlier
                # build cannot survive a reconfigure (see
                # _a5_async_workspace_overlay_defines).
                if self._arch == "a5" and self._variant == "onboard":
                    defines.update(_a5_async_workspace_overlay_defines())
            if target == "aicore" and arch == "a5" and variant == "onboard":
                defines["SIMPLER_AICORE_BUILD_PROFILE"] = self.aicore_build_profile
            cmake_defines = defines or None
            # compile() adds a {target}/ subdirectory inside build_dir
            cache_dir = self._CACHE_DIR / arch / variant / name
            cache_dir.mkdir(parents=True, exist_ok=True)

            # File lock to prevent concurrent cmake runs in the same build dir.
            # Each target gets its own lock so host/aicpu/aicore build in parallel,
            # but two processes building the same target are serialized.
            lock_path = cache_dir / f".{target}.lock"
            with open(lock_path, "w") as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                _invalidate_cache_if_stale(cache_dir / target, cache_stamp, inputs=cache_inputs)
                return compiler.compile(  # type: ignore[return-value]
                    target,
                    include_dirs,
                    source_dirs,
                    build_dir=str(cache_dir),
                    output_dir=output_dir,
                    dispatcher_dest=dispatcher_staging_dir if target == "aicpu" else None,
                    sdma_warmup_dest=sdma_warmup_staging_dir if target == "aicore" else None,
                    cmake_defines=cmake_defines,
                )

        logger.info("Compiling AICore, AICPU, Host in parallel...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_host = executor.submit(_compile_target, "host")
            fut_aicpu = executor.submit(_compile_target, "aicpu")
            fut_aicore = executor.submit(_compile_target, "aicore")
            fut_sim_ctx = executor.submit(self.ensure_sim_context, build=build_shared) if variant == "sim" else None

            host_path = fut_host.result()
            aicpu_path = fut_aicpu.result()
            aicore_path = fut_aicore.result()
            sim_context_path = fut_sim_ctx.result() if fut_sim_ctx else None

        self._place_compile_commands(name)
        if self._requires_pto_isa_metadata_validation():
            from . import pto_isa  # noqa: PLC0415

            runtime_key = pto_isa.pto_isa_runtime_artifact_key(self._arch, self._variant, name)
            pto_isa.write_pto_isa_build_metadata(
                self._LIB_DIR,
                pto_isa.ensure_pto_isa_root(verbose=True),
                [runtime_key],
            )
        logger.info("Build complete!")
        # runtime_compiler stages libsimpler_aicpu_dispatcher.so into the
        # per-arch shared directory when target=='aicpu'. Surface it through
        # RuntimeBinaries so ChipWorker.init can pass the path to
        # LoadAicpuOp::BootstrapDispatcher.
        dispatcher_path = self._resolve_dispatcher_path()
        if dispatcher_path is not None and not dispatcher_path.is_file():
            dispatcher_path = None
        return RuntimeBinaries(
            host_path=host_path,
            aicpu_path=aicpu_path,
            aicore_path=aicore_path,
            sim_context_path=sim_context_path,
            dispatcher_path=dispatcher_path,
            sdma_warmup_path=self._resolve_sdma_warmup_path(),
        )

    def _resolve_sdma_warmup_path(self) -> Optional[Path]:
        """Return path to sdma_warmup_kernel.o, or None when this build has none.

        Staged per arch under ``build/lib/<arch>/sdma_warmup/`` by
        runtime_compiler when the arch's aicore CMakeLists builds the vector-only
        warmup ELF. Returns ``None`` for sim and whenever the file is absent:
        without it the first TPREFETCH_ASYNC pays the cold SDMA control path
        (~4.4 ms) but nothing breaks, so this is deliberately not validated the
        way ``_resolve_dispatcher_path`` is.

        A missing ELF for an arch that *does* carry the source is a build or
        staging regression, and every consumer of it degrades silently, so it is
        warned about here rather than left to be noticed as lost performance.
        """
        if self._variant == "sim":
            return None
        path = self._LIB_DIR / self._arch / "sdma_warmup" / "sdma_warmup_kernel.o"
        if path.is_file():
            return path
        source = PROJECT_ROOT / "src" / self._arch / "platform" / "onboard" / "aicore" / "sdma_warmup_kernel.cpp"
        if source.is_file():
            logger.warning(
                "SDMA warmup ELF not staged at %s though %s carries its source; "
                "an enable_sdma Worker will pay the cold SDMA control path on its first TPREFETCH_ASYNC",
                path,
                self._arch,
            )
        return None

    def _resolve_dispatcher_path(self) -> Optional[Path]:
        """Return path to libsimpler_aicpu_dispatcher.so for onboard variants.

        Returns ``None`` for sim variants (no dispatcher needed: sim's AICPU
        runs in-process). For onboard, runtime_compiler stages one shared
        copy per arch under ``build/lib/<arch>/dispatcher/`` (the dispatcher
        has no runtime-specific code, so all onboard runtimes on a given
        arch use the same SO). Validated separately by ``_lookup_binaries``.
        """
        if self._variant == "sim":
            return None
        return self._LIB_DIR / self._arch / "dispatcher" / "libsimpler_aicpu_dispatcher.so"

    def _resolve_sim_context_path(self) -> Optional[Path]:
        """Return path to libcpu_sim_context.so for sim platforms, None for onboard.

        The library is process-global and has no arch-specific code, so one
        shared copy per host toolchain is enough.
        Lives at build/lib/libcpu_sim_context.so.
        """
        if self._variant != "sim":
            return None
        return self._LIB_DIR / "libcpu_sim_context.so"

    def ensure_sim_context(self, build: bool = False) -> Optional[Path]:
        """Build or locate the process-global cpu_sim_context SO (sim only)."""
        if self._variant != "sim":
            return None

        output_dir = self._LIB_DIR
        so_path = output_dir / "libcpu_sim_context.so"

        if not build and so_path.is_file():
            return so_path
        if not build:
            raise FileNotFoundError(
                f"Pre-built libcpu_sim_context.so not found at {so_path}.\n"
                "Run 'pip install --no-build-isolation .' to compile it."
            )

        cache_dir = self._CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / ".sim_context.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            result = self._runtime_compiler.compile_sim_context(
                build_dir=str(cache_dir),
                output_dir=output_dir,
            )
            return Path(result)  # type: ignore[arg-type]

    def _place_compile_commands(self, runtime_name: str) -> None:
        """Merge compile_commands.json from build/cache/ targets into source dirs.

        Placement follows the old gen_compile_commands.py defaults:
        - Runtime dirs get compdb only from the preferred variant (onboard),
          so clangd sees real-hardware compile flags.
        - Platform dirs get compdb only from the preferred runtime
          (tensormap_and_ringbuffer), the most feature-rich runtime.

        When the preferred variant/runtime isn't available for this arch,
        falls back to writing unconditionally.
        """
        arch, variant = self._arch, self._variant
        entries = []
        for target in TARGETS:
            cc = self._CACHE_DIR / arch / variant / runtime_name / target / "compile_commands.json"
            if cc.exists():
                try:
                    entries.extend(json.loads(cc.read_text()))
                except (json.JSONDecodeError, OSError):
                    pass

        if not entries:
            return

        merged = json.dumps(entries, indent=2) + "\n"

        # Place in runtime source directory (prefer onboard variant, fallback if unavailable)
        write_runtime = (
            variant == self._COMPDB_VARIANT
            or not (self.runtime_root / "src" / arch / "platform" / self._COMPDB_VARIANT).is_dir()
        )
        if write_runtime:
            runtime_dir = self.runtime_root / "src" / arch / "runtime" / runtime_name
            if runtime_dir.is_dir():
                (runtime_dir / "compile_commands.json").write_text(merged)

        # Place in platform variant source directory (prefer tensormap_and_ringbuffer, fallback if unavailable)
        write_platform = runtime_name == self._COMPDB_RUNTIME or self._COMPDB_RUNTIME not in self._runtimes
        if write_platform:
            platform_dir = self.runtime_root / "src" / arch / "platform" / variant
            if platform_dir.is_dir():
                (platform_dir / "compile_commands.json").write_text(merged)
