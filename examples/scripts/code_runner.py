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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.testing import assert_allclose


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


def _check_ascend_env() -> bool:
    """Check if ASCEND_HOME_PATH environment is set."""
    return bool(os.environ.get("ASCEND_HOME_PATH"))


def _check_pto_isa_root() -> bool:
    """Check if PTO_ISA_ROOT environment is set."""
    return bool(os.environ.get("PTO_ISA_ROOT"))


def _get_device_id() -> int:
    """Get device ID from environment variables."""
    device_id = os.environ.get("PTO_DEVICE_ID")
    if device_id is None:
        device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return int(device_id)


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
        runtime_name: Runtime implementation name (default: "host_build_graph")
        device_id: Device ID (defaults to PTO_DEVICE_ID env var or 0)
        platform: Platform name ("a2a3" for hardware, "a2a3sim" for simulation, default: "a2a3")
    """

    def __init__(
        self,
        kernels_dir: str,
        golden_path: str,
        runtime_name: str = "host_build_graph",
        device_id: Optional[int] = None,
        platform: str = "a2a3",
    ):
        self.kernels_dir = Path(kernels_dir).resolve()
        self.golden_path = Path(golden_path).resolve()
        self.runtime_name = runtime_name
        self.platform = platform
        self.project_root = _get_project_root()

        # Resolve device ID
        if device_id is None:
            device_id = _get_device_id()
        self.device_id = device_id

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

        # Runtime configuration
        self.aicpu_thread_num = 3
        self.block_dim = 3

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

    def _build_func_args(self, tensors: Dict[str, np.ndarray]) -> List[int]:
        """
        Build func_args from tensors automatically.

        Convention for orchestration function signature:
            int BuildGraph(Runtime* runtime, uint64_t* args, int arg_count)

        Where args layout is:
            [ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, count]

        Args:
            tensors: Dict of numpy arrays

        Returns:
            List of func_args values (pointers, sizes, count)
        """
        # Determine tensor order
        if self.tensor_order:
            order = self.tensor_order
        else:
            order = list(tensors.keys())

        ptrs = []
        sizes = []

        for name in order:
            if name not in tensors:
                raise KeyError(
                    f"Tensor '{name}' from TENSOR_ORDER not found in generate_inputs() result.\n"
                    f"Available tensors: {list(tensors.keys())}"
                )
            arr = tensors[name]
            ptrs.append(arr.ctypes.data)
            sizes.append(arr.nbytes)

        # Get element count from first tensor
        count = tensors[order[0]].size

        return ptrs + sizes + [count]

    def skip_if_no_env(self) -> None:
        """Raise error if required environment is not available."""
        if not _check_ascend_env():
            raise EnvironmentError("ASCEND_HOME_PATH not set")
        if not _check_pto_isa_root():
            raise EnvironmentError(
                "PTO_ISA_ROOT environment variable is not set.\n"
                "Please set it to the PTO-ISA root directory, e.g.:\n"
                "  export PTO_ISA_ROOT=$(pwd)/_deps/pto-isa-src"
            )

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
        # Import runtime modules (deferred to allow skip_if_no_env to work)
        from runtime_builder import RuntimeBuilder
        from bindings import bind_host_binary, register_kernel, set_device, launch_runtime
        from elf_parser import extract_text_section

        # Skip if environment not available (only for a2a3 platform)
        if self.platform == "a2a3":
            self.skip_if_no_env()

        # Step 1: Build runtime
        print(f"\n=== Building Runtime: {self.runtime_name} (platform: {self.platform}) ===")
        builder = RuntimeBuilder(runtime_root=self.project_root, platform=self.platform)
        pto_compiler = builder.get_pto_compiler()
        try:
            host_binary, aicpu_binary, aicore_binary = builder.build(self.runtime_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build runtime '{self.runtime_name}' for platform '{self.platform}'.\n"
                f"Error: {e}"
            ) from e

        # Step 2: Load runtime and set device
        print(f"\n=== Loading Runtime ({len(host_binary)} bytes) ===")
        Runtime = bind_host_binary(host_binary)

        print(f"\n=== Setting Device {self.device_id} ===")
        set_device(self.device_id)

        # Step 3: Compile orchestration
        print("\n=== Compiling Orchestration ===")

        # Build include directories for orchestration
        orch_include_dirs = [
            str(self.project_root / "src" / "runtime" / self.runtime_name / "runtime"),
        ] + pto_compiler.get_platform_include_dirs()

        orch_so_binary = pto_compiler.compile_orchestration(
            self.orchestration["source"],
            extra_include_dirs=orch_include_dirs,
        )
        print(f"Compiled orchestration: {len(orch_so_binary)} bytes")

        # Step 4: Compile and register kernels
        print("\n=== Compiling and Registering Kernels ===")

        # Get PTO_ISA_ROOT (use default for sim platform)
        pto_isa_root = os.environ.get("PTO_ISA_ROOT", "/tmp/unused")

        for kernel in self.kernels:
            print(f"Compiling kernel: {kernel['source']} (func_id={kernel['func_id']})")
            incore_o = pto_compiler.compile_incore(
                kernel["source"],
                core_type=kernel["core_type"],
                pto_isa_root=pto_isa_root,
            )
            kernel_bin = extract_text_section(incore_o)
            register_kernel(kernel["func_id"], kernel_bin)

        print("All kernels compiled and registered")

        # Step 5: Run each parameter set
        total_cases = len(self.params_list)
        for case_idx, params in enumerate(self.params_list):
            print(f"\n{'='*60}")
            print(f"=== Case {case_idx + 1}/{total_cases}: {params} ===")
            print(f"{'='*60}")

            # Generate tensors using golden.py
            print("\n=== Generating Inputs ===")
            tensors = self._golden_module.generate_inputs(params)

            # Convert any PyTorch tensors to numpy
            tensors = {k: _to_numpy(v) for k, v in tensors.items()}

            # Identify inputs and outputs
            inputs, outputs = self._identify_outputs(tensors)
            print(f"Inputs: {list(inputs.keys())}")
            print(f"Outputs: {list(outputs.keys())}")

            # Build func_args automatically
            func_args = self._build_func_args(tensors)

            # Determine actual tensor order for debugging
            order = self.tensor_order if self.tensor_order else list(tensors.keys())
            print(f"Tensor order: {order}")
            print(f"func_args count: {len(func_args)}")

            # Create and initialize runtime
            print("\n=== Initializing Runtime ===")
            runtime = Runtime()
            runtime.initialize(orch_so_binary, self.orchestration["function_name"], func_args)

            # Launch runtime
            print("\n=== Launching Runtime ===")
            print(f"Device ID: {self.device_id}")
            print(f"AICPU threads: {self.aicpu_thread_num}, Block dim: {self.block_dim}")
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

            print("Launch completed successfully")  # Will only print if not hung

            # Finalize
            print("\n=== Finalizing Runtime ===")
            runtime.finalize()

            # Compute golden and compare
            print("\n=== Comparing Results ===")
            self._compare_with_golden(tensors, inputs, outputs, params)

            print(f"\n=== Case {case_idx + 1}/{total_cases} Passed ===")

        print(f"\n{'='*60}")
        print(f"=== All {total_cases} cases passed ===")
        print(f"{'='*60}")

    def _compare_with_golden(
        self,
        tensors: Dict[str, np.ndarray],
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        params: Dict[str, Any],
    ) -> None:
        """Compare outputs with golden values."""
        # Create copies for golden computation
        golden_outputs = {k: v.copy() for k, v in outputs.items()}
        golden_tensors = {**inputs, **golden_outputs}

        # Compute golden
        self._golden_module.compute_golden(golden_tensors, params)

        # Compare each output
        for name in outputs:
            actual = outputs[name]
            expected = golden_outputs[name]
            print(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

            # Show first 10 values
            if actual.size > 0:
                flat_actual = actual.flatten()
                flat_expected = expected.flatten()
                n_show = min(10, flat_actual.size)
                print(f"  First {n_show} actual:   {flat_actual[:n_show]}")
                print(f"  First {n_show} expected: {flat_expected[:n_show]}")

            assert_allclose(
                actual,
                expected,
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Output '{name}' does not match golden",
            )
            matched = np.sum(np.isclose(actual, expected, rtol=self.rtol, atol=self.atol))
            print(f"  {name}: PASS ({matched}/{actual.size} elements matched)")
