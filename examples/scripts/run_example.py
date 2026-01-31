#!/usr/bin/env python3
"""
Simplified test runner for PTO runtime tests.

This script provides a command-line interface to run PTO runtime tests
with minimal configuration. Users only need to provide:
1. A kernels directory with kernel_config.py
2. A golden.py script

Usage:
    python examples/scripts/run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py
    python examples/scripts/run_example.py -k ./kernels -g ./golden.py --device 0 --platform a2a3sim

Examples:
    # Run hardware example (requires Ascend device)
    python examples/scripts/run_example.py -k examples/host_build_graph_example/kernels \
                                      -g examples/host_build_graph_example/golden.py

    # Run simulation example (no hardware required)
    python examples/scripts/run_example.py -k examples/host_build_graph_sim_example/kernels \
                                      -g examples/host_build_graph_sim_example/golden.py \
                                      -p a2a3sim

    # Run with specific device
    python examples/scripts/run_example.py -k ./kernels -g ./golden.py -d 0
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run PTO runtime test with kernel config and golden script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/scripts/run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py
    python examples/scripts/run_example.py -k ./kernels -g ./golden.py -d 0

Golden.py interface:
    def generate_inputs(params: dict) -> dict:
        '''Return dict of numpy arrays (inputs + outputs)'''
        return {"a": np.array(...), "out_f": np.zeros(...)}

    def compute_golden(tensors: dict, params: dict) -> None:
        '''Compute expected outputs in-place'''
        tensors["out_f"][:] = tensors["a"] + 1

    # Optional
    PARAMS_LIST = [{"size": 1024}]  # Multiple test cases
    RTOL = 1e-5  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance
    __outputs__ = ["out_f"]  # Or use 'out_' prefix
        """
    )

    parser.add_argument(
        "-k", "--kernels",
        required=True,
        help="Path to kernels directory containing kernel_config.py"
    )

    parser.add_argument(
        "-g", "--golden",
        required=True,
        help="Path to golden.py script"
    )

    parser.add_argument(
        "-d", "--device",
        type=int,
        default=None,
        help="Device ID (default: from PTO_DEVICE_ID env or 0)"
    )

    parser.add_argument(
        "-r", "--runtime",
        default="host_build_graph",
        help="Runtime implementation name (default: host_build_graph)"
    )

    parser.add_argument(
        "-p", "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim"],
        help="Platform name: 'a2a3' for hardware, 'a2a3sim' for simulation (default: a2a3)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Add python directory to path
    script_dir = Path(__file__).parent.resolve()  # examples/scripts/
    project_root = script_dir.parent.parent        # simpler/
    python_dir = project_root / "python"

    if python_dir.exists():
        sys.path.insert(0, str(python_dir))

    # Also add script_dir for code_runner (now co-located)
    sys.path.insert(0, str(script_dir))

    # Validate paths
    kernels_path = Path(args.kernels)
    golden_path = Path(args.golden)

    if not kernels_path.exists():
        print(f"Error: Kernels directory not found: {kernels_path}")
        return 1

    if not golden_path.exists():
        print(f"Error: Golden script not found: {golden_path}")
        return 1

    kernel_config_path = kernels_path / "kernel_config.py"
    if not kernel_config_path.exists():
        print(f"Error: kernel_config.py not found in {kernels_path}")
        return 1

    # Import and run
    try:
        from code_runner import CodeRunner

        runner = CodeRunner(
            kernels_dir=str(args.kernels),
            golden_path=str(args.golden),
            runtime_name=args.runtime,
            device_id=args.device,
            platform=args.platform,
        )

        runner.run()
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("=" * 60)
        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure you're running from the project root directory.")
        return 1

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
