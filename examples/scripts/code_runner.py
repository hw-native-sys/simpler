"""
CodeRunner - Simplified test framework for PTO runtime tests.

This module provides a simplified interface for writing runtime tests.
Users only need to provide:
1. A kernels directory with kernel_config.py
2. A golden.py script with generate_inputs() and compute_golden()

Usage:
    # Command line
    python examples/scripts/run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py

    # In Python
    from code_runner import CodeRunner
    runner = CodeRunner("./kernels", "./golden.py")
    runner.run()

Golden.py interface:
    # Required functions
    def generate_inputs(params: dict) -> list:
        '''Return flat argument list — tensors as (name, tensor) tuples, scalars as ctypes typed values'''
        a = torch.tensor(...)
        b = torch.tensor(...)
        out_f = torch.zeros(...)
        return [
            ("a", a),
            ("b", b),
            ("out_f", out_f),
            ("size_a", ctypes.c_int64(a.nbytes)),
            ("size_b", ctypes.c_int64(b.nbytes)),
            ("size_f", ctypes.c_int64(out_f.nbytes)),
            ("SIZE",   ctypes.c_int64(a.numel())),
        ]

    def compute_golden(tensors: dict, params: dict) -> None:
        '''Compute expected outputs in-place'''
        tensors["out_f"][:] = tensors["a"] + tensors["b"]

    # Optional configuration
    ALL_CASES = {"Case1": {"size": 1024}, "Case2": {"size": 2048}}  # Multiple test cases
    DEFAULT_CASE = "Case1"  # Default case to run
    RTOL = 1e-5  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance
    __outputs__ = ["out_f"]  # Output tensor names
"""

import ctypes
import importlib.util
import fcntl
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# =============================================================================
# TaskArg construction — uses nanobind bindings from task_interface
# =============================================================================

from task_interface import (
    TaskArgArray,
    make_tensor_arg,
    make_scalar_arg,
)

logger = logging.getLogger(__name__)


def _setup_logging_if_needed() -> None:
    """
    Setup logging if not already configured (for direct CodeRunner usage).
    Uses PTO_LOG_LEVEL environment variable or defaults to 'info'.
    """
    # Only setup if logging hasn't been configured yet
    if not logging.getLogger().hasHandlers():
        level_str = os.environ.get('PTO_LOG_LEVEL', 'info')
        level_map = {
            'error': logging.ERROR,
            'warn': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG,
        }
        log_level = level_map.get(level_str.lower(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='[%(levelname)s] %(message)s',
            force=True
        )


def _to_torch(tensor) -> torch.Tensor:
    """Convert tensor to torch.Tensor, handling bfloat16 and other tensor types."""
    if isinstance(tensor, torch.Tensor):
        # Already a torch tensor, ensure it's on CPU and contiguous
        return tensor.cpu().contiguous()

    # For any non-torch tensor, try direct torch conversion first
    # This handles most array-like objects including numpy arrays
    try:
        return torch.as_tensor(tensor)
    except (TypeError, RuntimeError):
        # If direct conversion fails, fall back to numpy path
        import numpy as np
        arr = np.asarray(tensor)
        return torch.from_numpy(arr)


def _load_module_from_path(module_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent  # examples/scripts/ -> examples/ -> simpler/


def _get_pto_isa_clone_path() -> Path:
    """Get the expected path to pto-isa clone."""
    return _get_project_root() / "examples" / "scripts" / "_deps" / "pto-isa"


def _is_pto_isa_cloned() -> bool:
    """
    Check if pto-isa is cloned.

    A clone is considered valid if:
    1. The directory exists
    2. It contains the include directory (essential content)
    """
    clone_path = _get_pto_isa_clone_path()
    if not clone_path.exists():
        return False

    # Check for essential content
    include_dir = clone_path / "include"
    return include_dir.exists() and include_dir.is_dir()


def _is_git_available() -> bool:
    """Check if git command is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_PTO_ISA_HTTPS = "https://github.com/PTO-ISA/pto-isa.git"
_PTO_ISA_SSH = "git@github.com:PTO-ISA/pto-isa.git"


def _pto_isa_repo_url(clone_protocol: str = "ssh") -> str:
    """Return the pto-isa clone URL for the given protocol."""
    if clone_protocol == "https":
        return _PTO_ISA_HTTPS
    return _PTO_ISA_SSH


def _clone_pto_isa(verbose: bool = False, commit: Optional[str] = None,
                   clone_protocol: str = "ssh") -> bool:
    """
    Clone pto-isa repository, optionally at a specific commit.

    Args:
        verbose: Print detailed progress information
        commit: If provided, checkout this commit after cloning

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    if not _is_git_available():
        if verbose:
            logger.warning("git command not available, cannot clone pto-isa")
        return False

    clone_path = _get_pto_isa_clone_path()

    # Create parent deps directory if it doesn't exist
    deps_dir = clone_path.parent
    try:
        deps_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to create deps directory: {e}")
        return False

    try:
        if verbose:
            logger.info(f"Cloning pto-isa to {clone_path}...")
            logger.info("This may take a few moments on first run...")

        repo_url = _pto_isa_repo_url(clone_protocol)
        result = subprocess.run(
            [
                "git", "clone",
                repo_url,
                str(clone_path)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            if verbose:
                logger.warning(f"Failed to clone pto-isa:\n{result.stderr}")
            return False

        # Checkout specific commit if requested
        if commit:
            result = subprocess.run(
                ["git", "checkout", commit],
                capture_output=True, text=True,
                cwd=str(clone_path), timeout=30
            )
            if result.returncode != 0:
                if verbose:
                    logger.warning(f"Failed to checkout pto-isa commit {commit}:\n{result.stderr}")
                return False

        if verbose:
            suffix = f" at commit {commit}" if commit else ""
            logger.info(f"pto-isa cloned successfully{suffix}: {clone_path}")

        return True

    except subprocess.TimeoutExpired:
        if verbose:
            logger.warning("Clone operation timed out")
        return False
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to clone pto-isa: {e}")
        return False


def _checkout_pto_isa_commit(clone_path: Path, commit: str, verbose: bool = False) -> None:
    """Checkout the specified commit if the existing clone is at a different revision."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(clone_path), timeout=5
        )
        current = result.stdout.strip() if result.returncode == 0 else ""
        if current and not commit.startswith(current) and not current.startswith(commit):
            if verbose:
                logger.info(f"pto-isa at {current}, checking out {commit}...")
            subprocess.run(
                ["git", "fetch", "origin"], capture_output=True, text=True,
                cwd=str(clone_path), timeout=120, check=True
            )
            subprocess.run(
                ["git", "checkout", commit], capture_output=True, text=True,
                cwd=str(clone_path), timeout=30, check=True
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to checkout pto-isa commit {commit}: "
                       f"{e.stderr if hasattr(e, 'stderr') else e}")
    except Exception as e:
        logger.warning(f"Unexpected error checking out pto-isa commit {commit}: {e}")


def _update_pto_isa_to_latest(clone_path: Path, verbose: bool = False) -> None:
    """Fetch and reset existing clone to the remote default branch."""
    import subprocess
    try:
        if verbose:
            logger.info("Updating pto-isa to latest...")
        subprocess.run(
            ["git", "fetch", "origin"], capture_output=True, text=True,
            cwd=str(clone_path), timeout=120, check=True
        )
        # Use origin/HEAD which tracks the remote's default branch
        subprocess.run(
            ["git", "reset", "--hard", "origin/HEAD"], capture_output=True, text=True,
            cwd=str(clone_path), timeout=30, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to update pto-isa to latest: "
                       f"{e.stderr if hasattr(e, 'stderr') else e}")
    except Exception as e:
        logger.warning(f"Unexpected error updating pto-isa: {e}")


def _ensure_pto_isa_root(verbose: bool = False, commit: Optional[str] = None,
                         clone_protocol: str = "ssh") -> Optional[str]:
    """
    Ensure PTO_ISA_ROOT is available, either from environment or cloned repo.

    This function:
    1. Checks if PTO_ISA_ROOT is already set
    2. If not, tries to clone pto-isa repository
    3. Returns the resolved path

    Uses a file lock to prevent parallel processes from racing on the clone.

    Args:
        verbose: Print detailed progress information
        commit: If provided, checkout this specific commit

    Returns:
        PTO_ISA_ROOT path if successful, None otherwise
    """
    # Check if already set in environment
    existing_root = os.environ.get("PTO_ISA_ROOT")
    if existing_root:
        if verbose:
            logger.info(f"Using existing PTO_ISA_ROOT: {existing_root}")
        return existing_root

    # Try to use cloned repository
    clone_path = _get_pto_isa_clone_path()

    # Use a file lock so only one process clones at a time
    lock_path = clone_path.parent / ".pto-isa.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _ensure_pto_isa_root_locked(clone_path, verbose=verbose, commit=commit,
                                          clone_protocol=clone_protocol)


def _ensure_pto_isa_root_locked(
    clone_path: Path, verbose: bool = False, commit: Optional[str] = None,
    clone_protocol: str = "ssh",
) -> Optional[str]:
    """Inner logic for _ensure_pto_isa_root, called while holding the file lock."""

    # Clone if needed
    if not _is_pto_isa_cloned():
        if verbose:
            logger.info("PTO_ISA_ROOT not set, cloning pto-isa repository...")
        if not _clone_pto_isa(verbose=verbose, commit=commit,
                              clone_protocol=clone_protocol):
            # Another parallel process may have completed the clone
            if not _is_pto_isa_cloned():
                if verbose:
                    logger.warning("Failed to automatically clone pto-isa.")
                    logger.warning("You can manually clone it with:")
                    logger.warning(f"  mkdir -p {clone_path.parent}")
                    logger.warning(f"  git clone {_pto_isa_repo_url(clone_protocol)} {clone_path}")
                    logger.warning("Or set PTO_ISA_ROOT to an existing pto-isa installation:")
                    logger.warning("  export PTO_ISA_ROOT=/path/to/pto-isa")
                return None
            if verbose:
                logger.info("pto-isa already cloned by another process")
            # Recovered from race — apply commit/update below
            if commit:
                _checkout_pto_isa_commit(clone_path, commit, verbose=verbose)
            else:
                _update_pto_isa_to_latest(clone_path, verbose=verbose)
    elif commit:
        _checkout_pto_isa_commit(clone_path, commit, verbose=verbose)
    else:
        _update_pto_isa_to_latest(clone_path, verbose=verbose)

    # Verify clone has expected content
    include_dir = clone_path / "include"
    if not include_dir.exists():
        if verbose:
            logger.warning(f"pto-isa cloned but missing include directory: {include_dir}")
        return None

    return str(clone_path.resolve())


def _kernel_config_runtime_env(kernel_config_module, kernels_dir: Path) -> Dict[str, str]:
    """
    Optional per-example environment variables for runtime compilation.

    `kernel_config.py` may define:
        RUNTIME_ENV = {"ENV_KEY": "value", ...}

    If a value looks like a path (ENV key ends with _DIR/_PATH)
    and is not absolute, it is resolved relative to
    `kernels_dir`.
    """
    runtime_env = getattr(kernel_config_module, "RUNTIME_ENV", None)
    if not isinstance(runtime_env, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in runtime_env.items():
        if not isinstance(k, str):
            continue
        s = str(v)
        is_path_like = k.endswith("_DIR") or k.endswith("_PATH")
        if is_path_like and s:
            p = Path(s)
            if not p.is_absolute():
                s = str((kernels_dir / p).resolve())
        out[k] = s
    return out


@contextmanager
def _temporary_env(env_updates: Dict[str, str]):
    """Temporarily apply env vars for the duration of the context."""
    old = {k: os.environ.get(k) for k in env_updates.keys()}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


class CodeRunner:
    """
    Simplified test runner that loads kernel config and golden script.

    This class automates:
    - Loading kernel_config.py and golden.py dynamically
    - Building TaskArgArray automatically from torch tensors
    - Converting numpy arrays to torch tensors
    - Separating inputs and outputs based on naming convention
    - Running the full test flow

    Args:
        kernels_dir: Path to kernels directory containing kernel_config.py
        golden_path: Path to golden.py script
        device_id: Device ID (defaults to 0)
        platform: Platform name ("a2a3" for hardware, "a2a3sim" for simulation, default: "a2a3")
    """

    def __init__(
        self,
        kernels_dir: str,
        golden_path: str,
        device_id: Optional[int] = None,
        platform: str = "a2a3",
        enable_profiling: bool = False,
        run_all_cases: bool = False,
        case_name: Optional[str] = None,
        pto_isa_commit: Optional[str] = None,
        build_dir: Optional[str] = None,
        repeat_rounds: Optional[int] = None,
        clone_protocol: str = "ssh",
        skip_golden: bool = False,
    ):
        # Setup logging if not already configured (e.g., when used directly, not via run_example.py)
        _setup_logging_if_needed()

        self.kernels_dir = Path(kernels_dir).resolve()
        self.golden_path = Path(golden_path).resolve()
        self.platform = platform
        self.enable_profiling = enable_profiling
        self.skip_golden = skip_golden
        self.project_root = _get_project_root()

        # Resolve device ID
        self.device_id = device_id if device_id is not None else 0
        self.pto_isa_commit = pto_isa_commit
        self.clone_protocol = clone_protocol
        self.build_dir = build_dir

        # Load configurations
        self._kernel_config = self._load_kernel_config()
        self._golden_module = self._load_golden_module()

        # Extract kernel configuration
        self.kernels = self._kernel_config.KERNELS
        self.orchestration = self._kernel_config.ORCHESTRATION

        # Extract golden configuration — determine which cases to run
        all_cases = getattr(self._golden_module, 'ALL_CASES', {"Default": {}})
        default_case = getattr(self._golden_module, 'DEFAULT_CASE', "Default")

        if run_all_cases:
            self.params_list = [{"name": name, **params} for name, params in all_cases.items()]
            logger.info(f"Running all {len(self.params_list)} cases: {list(all_cases.keys())}")
        elif case_name is not None:
            if case_name not in all_cases:
                raise ValueError(f"Case '{case_name}' not found. Available: {list(all_cases.keys())}")
            self.params_list = [{"name": case_name, **all_cases[case_name]}]
        else:
            self.params_list = [{"name": default_case, **all_cases[default_case]}]

        self.rtol = getattr(self._golden_module, 'RTOL', 1e-5)
        self.atol = getattr(self._golden_module, 'ATOL', 1e-5)
        self.output_names = getattr(self._golden_module, '__outputs__', None)
        self.tensor_order = getattr(self._golden_module, 'TENSOR_ORDER', None)

        # Runtime configuration - read from kernel_config or use defaults
        runtime_config = getattr(self._kernel_config, 'RUNTIME_CONFIG', {})
        self.aicpu_thread_num = runtime_config.get('aicpu_thread_num', 3)
        self.orch_thread_num = runtime_config.get('orch_thread_num', 1)
        self.block_dim = runtime_config.get('block_dim', 24)
        self.runtime_name = runtime_config.get('runtime', 'host_build_graph')
        self.repeat_rounds = repeat_rounds if repeat_rounds is not None else runtime_config.get('rounds', 1)

    def _load_kernel_config(self):
        """Load kernel_config.py from kernels directory."""
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(
                f"kernel_config.py not found in {self.kernels_dir}\n"
                f"Expected: {config_path}"
            )
        return _load_module_from_path(config_path, f"kernel_config_{id(self)}")

    def _load_golden_module(self):
        """Load golden.py script."""
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden script not found: {self.golden_path}")

        module = _load_module_from_path(self.golden_path, f"golden_{id(self)}")

        # Validate required functions
        if not hasattr(module, 'generate_inputs'):
            raise AttributeError(
                f"golden.py must define generate_inputs(params) function\n"
                f"File: {self.golden_path}"
            )
        if not hasattr(module, 'compute_golden'):
            raise AttributeError(
                f"golden.py must define compute_golden(tensors, params) function\n"
                f"File: {self.golden_path}"
            )

        return module

    def _identify_outputs(self, tensors: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        """
        Separate inputs and outputs from tensor dict using __outputs__.

        Returns:
            Tuple of (inputs_dict, outputs_dict)
        """
        if not self.output_names:
            raise ValueError(
                "No output tensors identified. "
                "Define __outputs__ = ['tensor_name'] in golden.py"
            )

        output_set = set(self.output_names)
        outputs = {k: v for k, v in tensors.items() if k in output_set}
        inputs = {k: v for k, v in tensors.items() if k not in output_set}

        if not outputs:
            raise ValueError(
                f"None of __outputs__ = {self.output_names} found in tensors: {list(tensors.keys())}"
            )

        return inputs, outputs

    def _build_func_args_from_list(
        self, args_list: list
    ) -> Tuple[list, Dict[str, Any], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build TaskArgArray from an explicit argument list returned by generate_inputs.

        Every element must be a (name, value) pair where value is either:
        - torch.Tensor / numpy array: a tensor argument
        - ctypes scalar (ctypes.c_int64, ctypes.c_float, etc.): a scalar argument

        All named items (tensors and scalars) are collected into the args dict
        passed to compute_golden, so compute_golden can reference any arg by name.

        Returns:
            Tuple of (orch_args, args, inputs, outputs)
            where args contains all named items, inputs/outputs contain tensor-only subsets.
        """
        import numpy as np

        if not self.output_names:
            raise ValueError(
                "No output tensors identified. "
                "Define __outputs__ = ['tensor_name'] in golden.py"
            )
        output_set = set(self.output_names)

        orch_args = TaskArgArray()
        args = {}    # all named items: tensors + scalars → passed to compute_golden
        inputs = {}  # tensor inputs only → for logging
        outputs = {} # tensor outputs (and inouts) → for comparison

        for item in args_list:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(
                    f"Each element in generate_inputs() list must be a (name, value) pair, "
                    f"got: {type(item)}\n"
                    f"Tensors: ('name', tensor)  Scalars: ('name', ctypes.c_int64(...))"
                )

            name, value = item

            if isinstance(value, (torch.Tensor, np.ndarray)):
                tensor = _to_torch(value)
                tensor = tensor.cpu().contiguous()
                args[name] = tensor

                orch_args.append(make_tensor_arg(tensor))

                if name in output_set:
                    outputs[name] = tensor
                else:
                    inputs[name] = tensor

            elif isinstance(value, ctypes._SimpleCData):
                orch_args.append(make_scalar_arg(value))
                args[name] = value.value

            else:
                raise TypeError(
                    f"Unsupported value type for arg '{name}': {type(value)}\n"
                    f"Expected torch.Tensor, numpy array, or ctypes scalar (ctypes.c_int64, ctypes.c_float, etc.)"
                )

        if not outputs:
            raise ValueError(
                f"None of __outputs__ = {self.output_names} found in generate_inputs args"
            )

        return orch_args, args, inputs, outputs

    def _build_func_args(self, tensors: Dict[str, torch.Tensor]) -> list:
        """
        Build orch_args from tensors dict (legacy path).

        Convention for orchestration function signature:
            int BuildGraph(Runtime* runtime, uint64_t* args, int arg_count)

        Where args layout is:
            [ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, count]

        Args:
            tensors: Dict of torch tensors (will be modified to ensure contiguous)

        Returns:
            orch_args TaskArgArray
        """

        # Determine tensor order
        if self.tensor_order:
            order = self.tensor_order
        else:
            order = list(tensors.keys())

        # Identify outputs
        if not self.output_names:
            raise ValueError(
                "No output tensors identified. "
                "Define __outputs__ = ['tensor_name'] in golden.py"
            )

        # First pass: ensure all tensors are CPU and contiguous (update dict in place)
        for name in order:
            if name not in tensors:
                raise KeyError(
                    f"Tensor '{name}' from TENSOR_ORDER not found in generate_inputs() result.\n"
                    f"Available tensors: {list(tensors.keys())}"
                )
            tensors[name] = tensors[name].cpu().contiguous()

        orch_args = TaskArgArray()

        # Add tensor pointers
        for name in order:
            tensor = tensors[name]
            orch_args.append(make_tensor_arg(tensor))

        # Add sizes (as scalars)
        for name in order:
            tensor = tensors[name]
            orch_args.append(make_scalar_arg(tensor.element_size() * tensor.numel()))

        # Add element count (as scalar)
        count = tensors[order[0]].numel()
        orch_args.append(make_scalar_arg(count))

        return orch_args

    def run(self) -> None:
        """
        Execute the full test flow:
        1. Check environment
        2. Build runtime
        3. Load runtime and set device
        4. Compile orchestration
        5. Compile and register kernels
        6. For each params in params_list:
           - Generate inputs using golden.py
           - Initialize and launch runtime
           - Finalize and compare with golden
        """
        # Import runtime modules (deferred import to avoid top-level dependency)
        from runtime_builder import RuntimeBuilder
        from bindings import bind_host_binary, set_device, launch_runtime
        from elf_parser import extract_text_section

        # Auto-setup PTO_ISA_ROOT if needed (for all platforms, since kernels may use PTO ISA headers)
        pto_isa_root = _ensure_pto_isa_root(verbose=True, commit=self.pto_isa_commit,
                                          clone_protocol=self.clone_protocol)
        if pto_isa_root is None:
            raise EnvironmentError(
                "PTO_ISA_ROOT could not be resolved.\n"
                "Please set it to the PTO-ISA root directory, e.g.:\n"
                "  export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa"
            )

        # Step 1: Build runtime, orchestration, and kernels in parallel
        # (they are independent — all only need kernel_compiler which is ready)
        logger.info(f"=== Building Runtime: {self.runtime_name} (platform: {self.platform}) ===")
        builder = RuntimeBuilder(platform=self.platform)

        # Validate runtime exists before starting any compilation
        available_runtimes = builder.list_runtimes()
        if self.runtime_name not in available_runtimes:
            available_str = ", ".join(available_runtimes) or "(none)"
            raise ValueError(
                f"Runtime '{self.runtime_name}' is not available for platform '{self.platform}'.\n"
                f"Available runtimes for {self.platform}: {available_str}\n"
                f"Note: Different platforms may support different runtimes."
            )

        kernel_compiler = builder.get_kernel_compiler()

        from concurrent.futures import ThreadPoolExecutor, Future

        # Map platform to runtime architecture
        if self.platform in ("a2a3", "a2a3sim"):
            arch = "a2a3"
        elif self.platform in ("a5", "a5sim"):
            arch = "a5"  # Phase 2: A5 uses A5 runtime
        else:
            arch = "a2a3"

        runtime_include_dirs = [
            os.path.join(self.project_root, "src", arch, "runtime", self.runtime_name, "runtime"),
            os.path.join(self.project_root, "src", "common", "task_interface"),
        ]

        def _build_runtime():
            return builder.build(self.runtime_name, self.build_dir)

        def _compile_orchestration():
            return kernel_compiler.compile_orchestration(
                self.runtime_name,
                self.orchestration["source"],
                build_dir=self.build_dir,
            )

        def _compile_one_kernel(kernel):
            logger.info(f"Compiling kernel: {kernel['source']} (func_id={kernel['func_id']})")
            incore_o = kernel_compiler.compile_incore(
                kernel["source"],
                core_type=kernel["core_type"],
                pto_isa_root=pto_isa_root,
                extra_include_dirs=runtime_include_dirs,
                build_dir=self.build_dir,
            )
            if self.platform.endswith("sim"):
                kernel_bin = incore_o
            else:
                kernel_bin = extract_text_section(incore_o)
            return (kernel["func_id"], kernel_bin)

        # Launch all compilations concurrently
        max_workers = 2 + len(self.kernels)  # runtime + orchestration + kernels
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_runtime = executor.submit(_build_runtime)
            fut_orch = executor.submit(_compile_orchestration)
            fut_kernels = [executor.submit(_compile_one_kernel, k) for k in self.kernels]

            try:
                host_binary, aicpu_binary, aicore_binary = fut_runtime.result()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build runtime '{self.runtime_name}' for platform '{self.platform}'.\n"
                    f"Error: {e}"
                ) from e

            orch_so_binary = fut_orch.result()
            kernel_binaries = [f.result() for f in fut_kernels]

        logger.info(f"Compiled {len(kernel_binaries)} kernel(s)")

        # Step 2: Load runtime and set device
        logger.info(f"=== Loading Runtime ({len(host_binary)} bytes) ===")
        Runtime = bind_host_binary(host_binary)

        logger.info(f"=== Setting Device {self.device_id} ===")
        set_device(self.device_id)

        # Step 5: Run each parameter set
        total_cases = len(self.params_list)
        for case_idx, params in enumerate(self.params_list):
            logger.info("=" * 60)
            logger.info(f"=== Case {case_idx + 1}/{total_cases}: {params} ===")
            logger.info("=" * 60)

            # Generate tensors using golden.py
            logger.info("=== Generating Inputs ===")
            result = self._golden_module.generate_inputs(params)

            if isinstance(result, list):
                # New-style: generate_inputs returns flat argument list
                orch_args, args, inputs, outputs = \
                    self._build_func_args_from_list(result)
                tensors = args  # args contains all named items; compute_golden receives all
            else:
                # Legacy: generate_inputs returns dict of tensors
                tensors = {k: _to_torch(v) for k, v in result.items()}
                orch_args = self._build_func_args(tensors)
                inputs, outputs = self._identify_outputs(tensors)

            logger.info(f"Inputs: {list(inputs.keys())}")
            logger.info(f"Outputs: {list(outputs.keys())}")

            # Determine actual tensor order for debugging
            logger.debug(f"Tensor order: {list(tensors.keys())}")
            logger.debug(f"orch_args count: {len(orch_args)}")

            # Create and initialize runtime (including kernel registration)
            logger.info("=== Initializing Runtime ===")
            runtime = Runtime()

            # Build environment for runtime initialization
            run_env = _kernel_config_runtime_env(self._kernel_config, self.kernels_dir)
            if run_env:
                logger.debug(f"Runtime init env overrides: {run_env}")

            # Golden
            if not self.skip_golden:
                golden = {k: v.clone() for k, v in outputs.items()}
                golden_with_inputs = {**inputs, **golden}
                _t_golden_start = time.perf_counter()
                self._golden_module.compute_golden(golden_with_inputs, params)
                _t_golden_end = time.perf_counter()
                logger.info(f">>> compute_golden() took {_t_golden_end - _t_golden_start:.3f}s")

            initial_outputs = {k: v.clone() for k, v in outputs.items()}

            for round_idx in range(self.repeat_rounds):
                if self.repeat_rounds > 1:
                    logger.info(f"--- Round {round_idx + 1}/{self.repeat_rounds} ---")

                for k, v in initial_outputs.items():
                    outputs[k].copy_(v)

                runtime = Runtime()

                # Enable profiling if requested (only first round)
                if self.enable_profiling and round_idx == 0:
                    runtime.enable_profiling(True)
                    logger.info("Profiling enabled")

                with _temporary_env(run_env):
                    runtime.initialize(
                        orch_so_binary,
                        self.orchestration["function_name"],
                        orch_args,
                        kernel_binaries=kernel_binaries,
                    )

                launch_runtime(
                    runtime,
                    aicpu_thread_num=self.aicpu_thread_num,
                    block_dim=self.block_dim,
                    device_id=self.device_id,
                    aicpu_binary=aicpu_binary,
                    aicore_binary=aicore_binary,
                    orch_thread_num=self.orch_thread_num,
                )

                runtime.finalize()
                if not self.skip_golden:
                    self._compare_with_golden(outputs, golden)

            logger.info(f"=== Case {case_idx + 1}/{total_cases} Passed ===")

        logger.info("=" * 60)
        logger.info(f"=== All {total_cases} cases passed ===")
        logger.info("=" * 60)

    def _compare_with_golden(
        self,
        outputs: Dict[str, torch.Tensor],
        golden: Dict[str, torch.Tensor],
    ) -> None:
        """Compare hardware outputs with pre-computed golden values."""
        # Compare each output
        for name in outputs:
            actual = outputs[name]
            expected = golden[name]
            logger.info(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

            # Ensure both are on CPU for comparison
            actual = actual.cpu()
            expected = expected.cpu()

            # Show first 10 values
            if actual.numel() > 0:
                flat_actual = actual.flatten()
                flat_expected = expected.flatten()
                n_show = min(10, flat_actual.numel())
                logger.debug(f"  First {n_show} actual:   {flat_actual[:n_show].tolist()}")
                logger.debug(f"  First {n_show} expected: {flat_expected[:n_show].tolist()}")

            # Use torch for comparison
            if not torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol):
                # Find mismatches for better error reporting
                close_mask = torch.isclose(actual, expected, rtol=self.rtol, atol=self.atol)
                mismatches = (~close_mask).sum().item()
                total = actual.numel()
                raise AssertionError(
                    f"Output '{name}' does not match golden.\n"
                    f"Mismatched elements: {mismatches}/{total}\n"
                    f"rtol={self.rtol}, atol={self.atol}"
                )

            matched = torch.isclose(actual, expected, rtol=self.rtol, atol=self.atol).sum().item()
            logger.info(f"  {name}: PASS ({matched}/{actual.numel()} elements matched)")


def create_code_runner(kernels_dir, golden_path, device_id=None, platform="a2a3",
                       enable_profiling=False, run_all_cases=False, case_name=None,
                       pto_isa_commit=None, build_dir=None, repeat_rounds=None,
                       clone_protocol="ssh", skip_golden=False):
    """Factory: creates a CodeRunner based on kernel_config."""
    return CodeRunner(kernels_dir=kernels_dir, golden_path=golden_path,
                      device_id=device_id, platform=platform,
                      enable_profiling=enable_profiling,
                      run_all_cases=run_all_cases, case_name=case_name,
                      pto_isa_commit=pto_isa_commit, build_dir=build_dir,
                      repeat_rounds=repeat_rounds,
                      clone_protocol=clone_protocol,
                      skip_golden=skip_golden)
