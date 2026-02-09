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
    def generate_inputs(params: dict) -> dict:
        '''Return dict of numpy arrays (inputs + outputs)'''
        return {"a": np.array(...), "b": np.array(...), "out_f": np.zeros(...)}

    def compute_golden(tensors: dict, params: dict) -> None:
        '''Compute expected outputs in-place'''
        tensors["out_f"][:] = tensors["a"] + tensors["b"]

    # Optional configuration
    PARAMS_LIST = [{"size": 1024}, {"size": 2048}]  # Multiple test cases
    RTOL = 1e-5  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance
    __outputs__ = ["out_f"]  # Explicit output names (or use 'out_' prefix)
"""

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.testing import assert_allclose

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


def _has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _to_numpy(tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling PyTorch tensors."""
    if hasattr(tensor, 'detach'):
        # PyTorch tensor
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, '__array__'):
        return np.asarray(tensor)
    return tensor


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


def _clone_pto_isa(verbose: bool = False) -> bool:
    """
    Clone pto-isa repository.

    Args:
        verbose: Print detailed progress information

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

        # Clone with shallow depth for faster download
        result = subprocess.run(
            [
                "git", "clone",
                "--branch", "master",
                "--depth", "1",
                "https://gitcode.com/cann/pto-isa.git",
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

        if verbose:
            if result.stdout:
                logger.debug(result.stdout)
            logger.info(f"pto-isa cloned successfully to: {clone_path}")

        return True

    except subprocess.TimeoutExpired:
        if verbose:
            logger.warning("Clone operation timed out")
        return False
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to clone pto-isa: {e}")
        return False


def _ensure_pto_isa_root(verbose: bool = False) -> Optional[str]:
    """
    Ensure PTO_ISA_ROOT is available, either from environment or cloned repo.

    This function:
    1. Checks if PTO_ISA_ROOT is already set
    2. If not, tries to clone pto-isa repository
    3. Returns the resolved path

    Args:
        verbose: Print detailed progress information

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

    # Clone if needed
    if not _is_pto_isa_cloned():
        if verbose:
            logger.info("PTO_ISA_ROOT not set, cloning pto-isa repository...")
        if not _clone_pto_isa(verbose=verbose):
            if verbose:
                logger.warning("Failed to automatically clone pto-isa.")
                logger.warning("You can manually clone it with:")
                logger.warning(f"  mkdir -p {clone_path.parent}")
                logger.warning(f"  git clone --branch master https://gitcode.com/cann/pto-isa.git {clone_path}")
                logger.warning("Or set PTO_ISA_ROOT to an existing pto-isa installation:")
                logger.warning("  export PTO_ISA_ROOT=/path/to/pto-isa")
            return None

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
    - Building func_args automatically from numpy arrays
    - Converting PyTorch tensors to numpy
    - Separating inputs and outputs based on naming convention
    - Running the full test flow

    Args:
        kernels_dir: Path to kernels directory containing kernel_config.py
        golden_path: Path to golden.py script
        device_id: Device ID (defaults to PTO_DEVICE_ID env var or 0)
        platform: Platform name ("a2a3" for hardware, "a2a3sim" for simulation, default: "a2a3")
    """

    def __init__(
        self,
        kernels_dir: str,
        golden_path: str,
        device_id: Optional[int] = None,
        platform: str = "a2a3",
        enable_profiling: bool = False,
    ):
        # Setup logging if not already configured (e.g., when used directly, not via run_example.py)
        _setup_logging_if_needed()

        self.kernels_dir = Path(kernels_dir).resolve()
        self.golden_path = Path(golden_path).resolve()
        self.platform = platform
        self.enable_profiling = enable_profiling
        self.project_root = _get_project_root()

        # Resolve device ID
        self.device_id = device_id if device_id is not None else 0

        # Load configurations
        self._kernel_config = self._load_kernel_config()
        self._golden_module = self._load_golden_module()

        # Extract kernel configuration
        self.kernels = self._kernel_config.KERNELS
        self.orchestration = self._kernel_config.ORCHESTRATION

        # Extract golden configuration
        self.params_list = getattr(self._golden_module, 'PARAMS_LIST', [{}])
        self.rtol = getattr(self._golden_module, 'RTOL', 1e-5)
        self.atol = getattr(self._golden_module, 'ATOL', 1e-5)
        self.output_names = getattr(self._golden_module, '__outputs__', None)
        self.tensor_order = getattr(self._golden_module, 'TENSOR_ORDER', None)

        # Runtime configuration - read from kernel_config or use defaults
        runtime_config = getattr(self._kernel_config, 'RUNTIME_CONFIG', {})
        self.aicpu_thread_num = runtime_config.get('aicpu_thread_num', 3)
        self.block_dim = runtime_config.get('block_dim', 24)
        self.runtime_name = runtime_config.get('runtime', 'host_build_graph')

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

    def _identify_outputs(self, tensors: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Separate inputs and outputs from tensor dict.

        Uses either explicit __outputs__ list or 'out_' prefix convention.

        Returns:
            Tuple of (inputs_dict, outputs_dict)
        """
        if self.output_names:
            # Use explicit output names
            outputs = {k: v for k, v in tensors.items() if k in self.output_names}
            inputs = {k: v for k, v in tensors.items() if k not in self.output_names}
        else:
            # Use 'out_' prefix convention
            outputs = {k: v for k, v in tensors.items() if k.startswith('out_')}
            inputs = {k: v for k, v in tensors.items() if not k.startswith('out_')}

        if not outputs:
            raise ValueError(
                "No output tensors identified. Either:\n"
                "1. Define __outputs__ = ['tensor_name'] in golden.py, or\n"
                "2. Use 'out_' prefix for output tensor names (e.g., 'out_result')"
            )

        return inputs, outputs

    def _build_func_args(self, tensors: Dict[str, np.ndarray]) -> Tuple[List[int], List[int], List[int]]:
        """
        Build func_args, arg_types, and arg_sizes from tensors automatically.

        Convention for orchestration function signature:
            int BuildGraph(Runtime* runtime, uint64_t* args, int arg_count)

        Where args layout is:
            [ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, count]

        Args:
            tensors: Dict of numpy arrays

        Returns:
            Tuple of (func_args, arg_types, arg_sizes)
        """
        from bindings import ARG_SCALAR, ARG_INPUT_PTR, ARG_OUTPUT_PTR

        # Determine tensor order
        if self.tensor_order:
            order = self.tensor_order
        else:
            order = list(tensors.keys())

        # Identify outputs
        if self.output_names:
            output_set = set(self.output_names)
        else:
            output_set = {k for k in tensors.keys() if k.startswith('out_')}

        func_args = []
        arg_types = []
        arg_sizes = []

        # Add pointers
        for name in order:
            if name not in tensors:
                raise KeyError(
                    f"Tensor '{name}' from TENSOR_ORDER not found in generate_inputs() result.\n"
                    f"Available tensors: {list(tensors.keys())}"
                )
            arr = tensors[name]
            func_args.append(arr.ctypes.data)

            # Determine arg type based on whether it's an output
            if name in output_set:
                arg_types.append(ARG_OUTPUT_PTR)
            else:
                arg_types.append(ARG_INPUT_PTR)

            arg_sizes.append(arr.nbytes)

        # Add sizes (as scalars)
        for name in order:
            arr = tensors[name]
            func_args.append(arr.nbytes)
            arg_types.append(ARG_SCALAR)
            arg_sizes.append(0)

        # Add element count (as scalar)
        count = tensors[order[0]].size
        func_args.append(count)
        arg_types.append(ARG_SCALAR)
        arg_sizes.append(0)

        return func_args, arg_types, arg_sizes

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
        pto_isa_root = _ensure_pto_isa_root(verbose=True)
        if pto_isa_root is None:
            raise EnvironmentError(
                "PTO_ISA_ROOT could not be resolved.\n"
                "Please set it to the PTO-ISA root directory, e.g.:\n"
                "  export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa"
            )

        # Step 1: Build runtime
        logger.info(f"=== Building Runtime: {self.runtime_name} (platform: {self.platform}) ===")
        builder = RuntimeBuilder(platform=self.platform)
        kernel_compiler = builder.get_kernel_compiler()

        try:
            host_binary, aicpu_binary, aicore_binary = builder.build(
                self.runtime_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build runtime '{self.runtime_name}' for platform '{self.platform}'.\n"
                f"Error: {e}"
            ) from e

        # Step 2: Load runtime and set device
        logger.info(f"=== Loading Runtime ({len(host_binary)} bytes) ===")
        Runtime = bind_host_binary(host_binary)

        logger.info(f"=== Setting Device {self.device_id} ===")
        set_device(self.device_id)

        # Step 3: Compile orchestration
        logger.info("=== Compiling Orchestration ===")

        orch_so_binary = kernel_compiler.compile_orchestration(
            self.runtime_name,
            self.orchestration["source"],
        )

        # Step 4: Compile kernels (will be registered during runtime.initialize)
        logger.info("=== Compiling Kernels ===")

        # Build list of (func_id, binary) tuples for passing to runtime.initialize()
        kernel_binaries = []
        # Prepare runtime include directories for kernel compilation
        runtime_include_dirs = []
        runtime_dir = os.path.join(self.project_root, "src", "runtime", self.runtime_name, "runtime")
        runtime_include_dirs.append(runtime_dir)

        for kernel in self.kernels:
            logger.info(f"Compiling kernel: {kernel['source']} (func_id={kernel['func_id']})")
            incore_o = kernel_compiler.compile_incore(
                kernel["source"],
                core_type=kernel["core_type"],
                pto_isa_root=pto_isa_root,
                extra_include_dirs=runtime_include_dirs,
            )
            # For sim platform: keep complete .so for dlopen (supports external symbols like std::exp)
            # For real hardware: extract .text section (ccec compiled kernels don't depend on external symbols)
            if self.platform == "a2a3sim":
                kernel_bin = incore_o  # Complete .so for dlopen
            else:
                kernel_bin = extract_text_section(incore_o)  # .text only for mmap

            kernel_binaries.append((kernel["func_id"], kernel_bin))

        logger.info(f"Compiled {len(kernel_binaries)} kernel(s)")

        # Step 5: Run each parameter set
        total_cases = len(self.params_list)
        for case_idx, params in enumerate(self.params_list):
            logger.info("=" * 60)
            logger.info(f"=== Case {case_idx + 1}/{total_cases}: {params} ===")
            logger.info("=" * 60)

            # Generate tensors using golden.py
            logger.info("=== Generating Inputs ===")
            tensors = self._golden_module.generate_inputs(params)

            # Convert any PyTorch tensors to numpy
            tensors = {k: _to_numpy(v) for k, v in tensors.items()}

            # Identify inputs and outputs
            inputs, outputs = self._identify_outputs(tensors)
            logger.info(f"Inputs: {list(inputs.keys())}")
            logger.info(f"Outputs: {list(outputs.keys())}")

            # Build func_args automatically
            func_args, arg_types, arg_sizes = self._build_func_args(tensors)

            # Determine actual tensor order for debugging
            order = self.tensor_order if self.tensor_order else list(tensors.keys())
            logger.debug(f"Tensor order: {order}")
            logger.debug(f"func_args count: {len(func_args)}")

            # Create and initialize runtime (including kernel registration)
            logger.info("=== Initializing Runtime ===")
            runtime = Runtime()

            # Build environment for runtime initialization
            run_env = _kernel_config_runtime_env(self._kernel_config, self.kernels_dir)
            if run_env:
                logger.debug(f"Runtime init env overrides: {run_env}")

            # Enable profiling if requested (must be before initialize)
            if self.enable_profiling:
                runtime.enable_profiling(True)
                logger.info("Profiling enabled")

            with _temporary_env(run_env):
                runtime.initialize(
                    orch_so_binary,
                    self.orchestration["function_name"],
                    func_args,
                    arg_types=arg_types,
                    arg_sizes=arg_sizes,
                    kernel_binaries=kernel_binaries,
                )

            # Save expected values BEFORE hardware execution (outputs will be overwritten)
            golden = {k: v.copy() for k, v in outputs.items()}
            self._golden_module.compute_golden({**inputs, **golden}, params)

            # Launch runtime
            logger.info("=== Launching Runtime ===")
            logger.debug(f"Device ID: {self.device_id}")
            logger.debug(f"AICPU threads: {self.aicpu_thread_num}, Block dim: {self.block_dim}")
            import sys
            sys.stdout.flush()  # Ensure output is visible before potential hang

            launch_runtime(
                runtime,
                aicpu_thread_num=self.aicpu_thread_num,
                block_dim=self.block_dim,
                device_id=self.device_id,
                aicpu_binary=aicpu_binary,
                aicore_binary=aicore_binary,
            )

            logger.info("Launch completed successfully")  # Will only print if not hung

            # Finalize
            logger.info("=== Finalizing Runtime ===")
            runtime.finalize()

            # Compute golden and compare
            logger.info("=== Comparing Results ===")
            self._compare_with_golden(outputs, golden)

            logger.info(f"=== Case {case_idx + 1}/{total_cases} Passed ===")

        logger.info("=" * 60)
        logger.info(f"=== All {total_cases} cases passed ===")
        logger.info("=" * 60)

    def _compare_with_golden(
        self,
        outputs: Dict[str, np.ndarray],
        golden: Dict[str, np.ndarray],
    ) -> None:
        """Compare hardware outputs with pre-computed golden values."""
        # Compare each output
        for name in outputs:
            actual = outputs[name]
            expected = golden[name]
            logger.info(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

            # Show first 10 values
            if actual.size > 0:
                flat_actual = actual.flatten()
                flat_expected = expected.flatten()
                n_show = min(10, flat_actual.size)
                logger.debug(f"  First {n_show} actual:   {flat_actual[:n_show]}")
                logger.debug(f"  First {n_show} expected: {flat_expected[:n_show]}")

            assert_allclose(
                actual,
                expected,
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Output '{name}' does not match golden",
            )
            matched = np.sum(np.isclose(actual, expected, rtol=self.rtol, atol=self.atol))
            logger.info(f"  {name}: PASS ({matched}/{actual.size} elements matched)")


def create_code_runner(kernels_dir, golden_path, device_id=None, platform="a2a3",
                       enable_profiling=False):
    """Factory: creates a CodeRunner based on kernel_config."""
    return CodeRunner(kernels_dir=kernels_dir, golden_path=golden_path,
                      device_id=device_id, platform=platform,
                      enable_profiling=enable_profiling)
