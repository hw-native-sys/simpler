"""
DistributedCodeRunner — compile, prepare data, launch workers, and verify
results for distributed (multi-card) PTO kernel tests.

Parallel to CodeRunner, but handles DISTRIBUTED_CONFIG and spawns N
Python worker processes (one per rank) via distributed_worker.py.

Usage:
    runner = DistributedCodeRunner(
        kernels_dir="path/to/distributed_test/kernels",
        golden_path="path/to/distributed_test/golden.py",
        platform="a2a3", nranks=8,
    )
    runner.run()
"""

import importlib.util
import logging
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SIMPLER_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

DTYPE_FORMAT = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint32": ("I", 4),
    "uint64": ("Q", 8),
    "float16": ("e", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
}


def _load_module(path, name="mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DistributedCodeRunner:

    def __init__(
        self,
        kernels_dir: str,
        golden_path: Optional[str] = None,
        platform: str = "a2a3",
        nranks: Optional[int] = None,
        device_ids: Optional[list[int]] = None,
        root: Optional[int] = None,
        build_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        orch_func: Optional[str] = None,
        pto_isa_commit: Optional[str] = None,
        clone_protocol: str = "ssh",
    ):
        self.kernels_dir = Path(kernels_dir).resolve()
        self.platform = platform
        self.build_dir = Path(build_dir).resolve() if build_dir else \
            SIMPLER_ROOT / "build" / "distributed" / "cache"
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else \
            SIMPLER_ROOT / "build" / "distributed" / "artifacts"
        self.pto_isa_commit = pto_isa_commit
        self.clone_protocol = clone_protocol

        self._load_kernel_config()
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})

        self.nranks = nranks if nranks is not None else dist.get("nranks", 8)
        self.root = root if root is not None else dist.get("root", 0)
        self.orch_func = orch_func or self.kcfg.ORCHESTRATION["function_name"]
        if self.nranks <= 0:
            raise ValueError(f"Distributed nranks must be positive, got {self.nranks}")
        if self.root < 0 or self.root >= self.nranks:
            raise ValueError(
                f"Distributed root must be in [0, {self.nranks}), got {self.root}"
            )

        if device_ids is None:
            self.device_ids = list(range(self.nranks))
        else:
            if len(device_ids) != self.nranks:
                raise ValueError(
                    f"Expected {self.nranks} device ids, got {len(device_ids)}: {device_ids}"
                )
            self.device_ids = list(device_ids)

        self.golden_path = Path(golden_path).resolve() if golden_path else None
        self.golden_mod = None

    def _load_kernel_config(self):
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"kernel_config.py not found in {self.kernels_dir}")
        self.kcfg = _load_module(config_path, "kernel_config")

    def _load_golden(self):
        if self.golden_mod is None and self.golden_path and self.golden_path.exists():
            self.golden_mod = _load_module(self.golden_path, "golden")
        return self.golden_mod

    def _orch_artifact_name(self):
        src = Path(self.kcfg.ORCHESTRATION["source"])
        return src.stem + ".so"

    def _kernel_artifact_name(self, kernel_cfg):
        src = Path(kernel_cfg["source"])
        return src.stem + ".bin"

    def _get_buffer_config(self, name: str):
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        for buf_cfg in dist.get("buffers", []):
            if buf_cfg["name"] == name:
                return buf_cfg
        raise ValueError(
            f"Buffer '{name}' from golden.py not found in DISTRIBUTED_CONFIG['buffers']"
        )

    def _get_dtype_format(self, dtype: str, buffer_name: str):
        fmt = DTYPE_FORMAT.get(dtype)
        if fmt is None:
            raise ValueError(
                f"Unsupported dtype '{dtype}' for buffer '{buffer_name}'"
            )
        return fmt

    # ------------------------------------------------------------------
    # compile()
    # ------------------------------------------------------------------

    def compile(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("aicore", "aicpu", "host"):
            p = self.build_dir / sub
            if p.exists():
                shutil.rmtree(p)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        python_dir = SIMPLER_ROOT / "python"
        sys.path.insert(0, str(python_dir))
        sys.path.insert(0, str(SCRIPTS_DIR))

        from runtime_builder import RuntimeBuilder
        from elf_parser import extract_text_section
        from code_runner import _ensure_pto_isa_root

        pto_isa_root = _ensure_pto_isa_root(
            verbose=True, commit=self.pto_isa_commit,
            clone_protocol=self.clone_protocol)
        if pto_isa_root is None:
            raise EnvironmentError("PTO_ISA_ROOT could not be resolved.")

        runtime_name = self.kcfg.RUNTIME_CONFIG.get("runtime", "host_build_graph")
        builder = RuntimeBuilder(platform=self.platform)
        kernel_compiler = builder.get_kernel_compiler()

        logger.info("=== Phase 1: Building runtime ===")
        host_binary, aicpu_binary, aicore_binary = builder.build(
            runtime_name, str(self.build_dir))

        logger.info("=== Phase 2: Compiling orchestration ===")
        orch_source = self.kcfg.ORCHESTRATION["source"]
        if not os.path.isabs(orch_source):
            orch_source = str(self.kernels_dir / orch_source)
        orch_binary = kernel_compiler.compile_orchestration(
            runtime_name, orch_source, build_dir=str(self.build_dir))

        logger.info("=== Phase 3: Compiling kernels ===")
        if self.platform in ("a2a3", "a2a3sim"):
            arch = "a2a3"
        elif self.platform in ("a5", "a5sim"):
            arch = "a5"
        else:
            arch = "a2a3"

        runtime_include_dirs = [
            str(SIMPLER_ROOT / "src" / arch / "runtime" / runtime_name / "runtime")
        ]

        dist_config = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        extra_includes = list(runtime_include_dirs) + [
            str(SIMPLER_ROOT / "src" / arch / "platform" / "include"),
        ]
        for d in dist_config.get("comm_include_dirs", []):
            p = Path(pto_isa_root) / d if not os.path.isabs(d) else Path(d)
            extra_includes.append(str(p))

        kernel_bins = {}
        for k in self.kcfg.KERNELS:
            src = k["source"]
            if not os.path.isabs(src):
                src = str(self.kernels_dir / src)
            incore_o = kernel_compiler.compile_incore(
                src,
                core_type=k.get("core_type", "aiv"),
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_includes,
                build_dir=str(self.build_dir),
            )
            if self.platform.endswith("sim"):
                kernel_bins[k["func_id"]] = (k, incore_o)
            else:
                kernel_bins[k["func_id"]] = (k, extract_text_section(incore_o))

        logger.info("=== Phase 4: Saving artifacts ===")

        def save(name, data):
            path = self.artifact_dir / name
            path.write_bytes(data)
            logger.info(f"  {name}: {len(data)} bytes")

        save("libhost_runtime.so", host_binary)
        save("libaicpu_kernel.so", aicpu_binary)
        save("aicore_kernel.o", aicore_binary)
        save(self._orch_artifact_name(), orch_binary)
        for func_id, (kcfg, data) in kernel_bins.items():
            save(self._kernel_artifact_name(kcfg), data)

        logger.info(f"All artifacts saved to {self.artifact_dir}")

    # ------------------------------------------------------------------
    # prepare_data()
    # ------------------------------------------------------------------

    def prepare_data(self):
        golden = self._load_golden()
        if not golden or not hasattr(golden, "generate_distributed_inputs"):
            logger.info("No golden.py or generate_distributed_inputs — skipping data prep")
            return

        for r in range(self.nranks):
            rank_dir = self.artifact_dir / f"rank_{r}"
            rank_dir.mkdir(parents=True, exist_ok=True)

            inputs = golden.generate_distributed_inputs(r, self.nranks, self.root)
            for name, data in inputs:
                if isinstance(data, (list, tuple)):
                    buf_cfg = self._get_buffer_config(name)
                    fmt_char, _ = self._get_dtype_format(buf_cfg["dtype"], name)
                    bin_data = struct.pack(f"<{len(data)}{fmt_char}", *data)
                    path = rank_dir / f"{name}.bin"
                    path.write_bytes(bin_data)
                    logger.debug(f"  rank_{r}/{name}.bin: {len(bin_data)} bytes")

        logger.info(f"Prepared data for {self.nranks} ranks in {self.artifact_dir}")

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def _build_worker_cmd(self, r):
        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "distributed_worker.py"),
            "--device-id", str(self.device_ids[r]),
            "--rank", str(r),
            "--nranks", str(self.nranks),
            "--root", str(self.root),
            "--artifact-dir", str(self.artifact_dir),
            "--rootinfo-file", str(rootinfo_file),
            "--data-dir", str(self.artifact_dir / f"rank_{r}"),
            "--orch-file", self._orch_artifact_name(),
            "--orch-func", self.orch_func,
        ]

        rt_cfg = getattr(self.kcfg, "RUNTIME_CONFIG", {})
        cmd += ["--aicpu-thread-num", str(rt_cfg.get("aicpu_thread_num", 1))]
        cmd += ["--block-dim", str(rt_cfg.get("block_dim", 1))]
        cmd += ["--orch-thread-num", str(rt_cfg.get("orch_thread_num", 0))]

        win_sync = dist.get("win_sync_prefix", 0)
        if win_sync:
            cmd += ["--win-sync-prefix", str(win_sync)]

        for buf in dist.get("buffers", []):
            spec = f"{buf['name']}:{buf['dtype']}:{buf['count']}"
            if buf["placement"] == "window":
                cmd += ["--win-buffer", spec]
            else:
                cmd += ["--dev-buffer", spec]

        for name in dist.get("inputs", []):
            cmd += ["--load", name]

        for name in dist.get("outputs", []):
            cmd += ["--save", name]

        for tok in dist.get("args", []):
            cmd += ["--arg", tok]

        for k in self.kcfg.KERNELS:
            cmd += ["--kernel-bin",
                     f"{k['func_id']}:{self._kernel_artifact_name(k)}"]

        return cmd

    def run(self):
        rootinfo_file = self.artifact_dir / "rootinfo.bin"

        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()
        if rootinfo_file.exists():
            rootinfo_file.unlink()

        shm_dir = Path("/dev/shm")
        if shm_dir.is_dir():
            for f in shm_dir.glob("simpler_comm_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

        logger.info(f"=== Launching {self.nranks} workers ===")

        procs = []
        log_files = []
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            log_f = open(log_path, "w")
            log_files.append(log_f)

            cmd = self._build_worker_cmd(r)
            env = os.environ.copy()
            runtime_env = getattr(self.kcfg, "RUNTIME_ENV", None)
            if isinstance(runtime_env, dict):
                env.update(runtime_env)

            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            procs.append(proc)

        fail_count = 0
        for r, proc in enumerate(procs):
            proc.wait()
            log_files[r].close()
            if proc.returncode != 0:
                fail_count += 1
                logger.error(f"Rank {r}: FAILED (exit code {proc.returncode})")
            else:
                logger.info(f"Rank {r}: OK")

        print()
        for r in range(self.nranks):
            log_path = self.artifact_dir / f"rank{r}.log"
            lines = log_path.read_text().strip().split("\n")
            print(f"--- RANK {r} (last 5 lines) ---")
            for line in lines[-5:]:
                print(line)

        print()
        if fail_count == 0:
            print(f"=== ALL {self.nranks} RANKS COMPLETED ===")
        else:
            print(f"=== {fail_count}/{self.nranks} RANKS FAILED ===")

        for f in self.artifact_dir.glob("barrier_*.ready"):
            f.unlink()

        self._run_ok = (fail_count == 0)
        return self._run_ok

    # ------------------------------------------------------------------
    # verify()
    # ------------------------------------------------------------------

    def verify(self):
        golden = self._load_golden()
        if not golden or not hasattr(golden, "compute_golden"):
            logger.info("No golden.py or compute_golden — skipping verification")
            return True

        dist = getattr(self.kcfg, "DISTRIBUTED_CONFIG", {})
        output_names = dist.get("outputs", [])
        buf_map = {b["name"]: b for b in dist.get("buffers", [])}

        # Compute expected outputs once for the distributed verification step.
        seed_dir = self.artifact_dir / f"rank_{self.root}"
        seed_outputs = {}
        for name in output_names:
            path = seed_dir / f"{name}.bin"
            if not path.exists():
                logger.error(f"Output file not found: {path}")
                return False
            raw = path.read_bytes()
            dtype = buf_map.get(name, {}).get("dtype", "float32")
            fmt_char, elem_sz = DTYPE_FORMAT.get(dtype, ("f", 4))
            count = len(raw) // elem_sz
            seed_outputs[name] = list(struct.unpack(f"<{count}{fmt_char}", raw))

        expected_outputs = {n: v.copy() for n, v in seed_outputs.items()}
        params = {"nranks": self.nranks, "root": self.root}
        golden.compute_golden(expected_outputs, params)

        rtol = getattr(golden, "RTOL", 1e-5)
        atol = getattr(golden, "ATOL", 1e-5)

        all_ok = True
        for rank in range(self.nranks):
            rank_dir = self.artifact_dir / f"rank_{rank}"
            for name in output_names:
                path = rank_dir / f"{name}.bin"
                if not path.exists():
                    logger.error(f"Output file not found: {path}")
                    all_ok = False
                    continue
                raw = path.read_bytes()
                dtype = buf_map.get(name, {}).get("dtype", "float32")
                fmt_char, elem_sz = DTYPE_FORMAT.get(dtype, ("f", 4))
                count = len(raw) // elem_sz
                actual = list(struct.unpack(f"<{count}{fmt_char}", raw))
                expected = expected_outputs[name]

                mismatches = 0
                for i, (a, e) in enumerate(zip(actual, expected)):
                    if abs(a - e) > atol + rtol * abs(e):
                        if mismatches < 3:
                            logger.error(f"  rank {rank} {name}[{i}]: got {a}, expected {e}")
                        mismatches += 1
                if mismatches > 0:
                    logger.error(f"VERIFY FAILED: rank {rank} {name} — {mismatches}/{len(actual)} mismatches")
                    all_ok = False
                else:
                    logger.info(f"VERIFY PASSED: rank {rank} {name} — {len(actual)} elements correct")
                    if rank == 0 and len(actual) >= 5:
                        logger.info(f"  Sample: {actual[:5]}")

        if all_ok:
            print("\n=== VERIFICATION PASSED ===\n")
        else:
            print("\n=== VERIFICATION FAILED ===\n")

        return all_ok

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_all(self, skip_compile=False, skip_verify=False):
        if not skip_compile:
            self.compile()

        if self.golden_path:
            self.prepare_data()

        success = self.run()

        if success and self.golden_path and not skip_verify:
            success = self.verify()

        return success
