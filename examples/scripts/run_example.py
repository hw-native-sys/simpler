#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
    python examples/scripts/run_example.py -k examples/host_build_graph/vector_example/kernels \
                                      -g examples/host_build_graph/vector_example/golden.py

    # Run simulation example (no hardware required)
    python examples/scripts/run_example.py -k examples/host_build_graph/vector_example/kernels \
                                      -g examples/host_build_graph/vector_example/golden.py \
                                      -p a2a3sim

    # Run with specific device
    python examples/scripts/run_example.py -k ./kernels -g ./golden.py -d 0
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from simpler_setup.code_runner import create_code_runner
from simpler_setup.log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging

project_root = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


def _get_device_log_dir(device_id):
    """Return the device log directory using the same logic as device_log_resolver."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if root.exists():
            return root / f"device-{device_id}"
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"


def _wait_for_new_device_log(log_dir, pre_run_logs, timeout=15, interval=0.5):
    """Wait for a new device log file that wasn't present before the run.

    CANN dlog writes device logs asynchronously, so the file may appear
    a few seconds after the run completes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_dir.exists():
            current_logs = set(log_dir.glob("*.log"))
            new_logs = current_logs - pre_run_logs
            if new_logs:
                return max(new_logs, key=lambda p: p.stat().st_mtime)
        time.sleep(interval)
    return None


def main():  # noqa: PLR0912
    import warnings  # noqa: PLC0415

    warnings.warn(
        "run_example.py is deprecated. Use 'python test_*.py' with the same CLI options instead. "
        "See docs/testing.md for details.",
        DeprecationWarning,
        stacklevel=1,
    )

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

    # Optional — for parameterized test cases:
    ALL_CASES = {"Case1": {"size": 1024}, "Case2": {"size": 2048}}
    DEFAULT_CASE = "Case1"
    RTOL = 1e-5  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance
    __outputs__ = ["out_f"]  # Or use 'out_' prefix
        """,
    )

    parser.add_argument(
        "-k",
        "--kernels",
        required=True,
        help="Path to kernels directory containing kernel_config.py",
    )

    parser.add_argument("-g", "--golden", required=True, help="Path to golden.py script")

    parser.add_argument("-d", "--device", type=int, default=0, help="Device ID (default: 0)")

    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="Platform name: 'a2a3'/'a5' for hardware, 'a2a3sim'/'a5sim' for simulation (default: a2a3)",
    )

    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help=f"Root logger level (default: {DEFAULT_LOG_LEVEL})",
    )

    parser.add_argument(
        "--enable-profiling",
        type=int,
        nargs="?",
        const=3,
        default=0,
        metavar="LEVEL",
        help="Swimlane profiling mode: 1=AICore only, 2=task+fanout, 3=full (default when flag given: 3)",
    )

    parser.add_argument(
        "--dump-tensor",
        action="store_true",
        help="Dump per-task tensor I/O at runtime (controlled by enable_dump_tensor flag)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test cases defined in ALL_CASES (default: run only DEFAULT_CASE)",
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run a specific test case by name (e.g., --case Case2)",
    )

    parser.add_argument(
        "-c",
        "--pto-isa-commit",
        type=str,
        default=None,
        help="Checkout PTO-ISA at this commit (e.g., -c 1b22fea)",
    )

    parser.add_argument(
        "-n",
        "--rounds",
        type=int,
        default=None,
        metavar="ROUNDS",
        help="Number of rounds to run per case (overrides kernel_config RUNTIME_CONFIG['rounds'])",
    )

    parser.add_argument(
        "--clone-protocol",
        choices=["ssh", "https"],
        default="ssh",
        help="Git protocol for cloning pto-isa (default: ssh)",
    )

    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip golden computation and comparison (for benchmarking)",
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Compile runtime from source instead of using pre-built binaries",
    )

    args = parser.parse_args()

    if args.all and args.case:
        parser.error("--all and --case are mutually exclusive")

    configure_logging(args.log_level)

    # Validate paths
    kernels_path = Path(args.kernels)
    golden_path = Path(args.golden)

    if not kernels_path.exists():
        logger.error(f"Kernels directory not found: {kernels_path}")
        return 1

    if not golden_path.exists():
        logger.error(f"Golden script not found: {golden_path}")
        return 1

    kernel_config_path = kernels_path / "kernel_config.py"
    if not kernel_config_path.exists():
        logger.error(f"kernel_config.py not found in {kernels_path}")
        return 1

    try:
        runner = create_code_runner(
            kernels_dir=str(args.kernels),
            golden_path=str(args.golden),
            device_id=args.device,
            platform=args.platform,
            enable_profiling=args.enable_profiling,
            enable_dump_tensor=args.dump_tensor,
            run_all_cases=args.all,
            case_name=args.case,
            pto_isa_commit=args.pto_isa_commit,
            build_runtime=args.build,
            repeat_rounds=args.rounds,
            clone_protocol=args.clone_protocol,
            skip_golden=args.skip_golden,
        )

        # Snapshot existing device logs before the run so we can identify the
        # new log created by this run (CANN writes device logs asynchronously).
        pre_run_device_logs = set()
        device_log_dir = None
        if args.enable_profiling and args.platform == "a2a3":
            device_log_dir = _get_device_log_dir(args.device)
            if device_log_dir.exists():
                pre_run_device_logs = set(device_log_dir.glob("*.log"))

        runner.run()
        logger.info("=" * 60)
        logger.info("TEST PASSED")
        logger.info("=" * 60)

        # If profiling was enabled, generate merged swimlane JSON
        if args.enable_profiling:
            logger.info("Generating swimlane visualization...")
            kernel_config_path = kernels_path / "kernel_config.py"
            swimlane_script = project_root / "tools" / "swimlane_converter.py"

            if swimlane_script.exists():
                import subprocess  # noqa: PLC0415

                try:
                    cmd = [
                        sys.executable,
                        str(swimlane_script),
                        "-k",
                        str(kernel_config_path),
                    ]

                    # Find the device log created by this run via snapshot diff
                    if device_log_dir is not None:
                        device_log_file = _wait_for_new_device_log(device_log_dir, pre_run_device_logs)
                        if device_log_file:
                            cmd += ["--device-log", str(device_log_file)]
                        else:
                            logger.warning("No new device log found, falling back to device-id")
                            cmd += ["-d", str(args.device)]
                    else:
                        cmd += ["-d", str(args.device)]

                    if logger.isEnabledFor(logging.DEBUG):
                        cmd.append("-v")

                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(result.stdout)
                    logger.info("Swimlane JSON generation completed")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to generate swimlane JSON: {e}")
                    logger.debug(f"stderr: {e.stderr}")
            else:
                logger.warning(f"Swimlane converter script not found: {swimlane_script}")

        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root directory.")
        return 1

    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback  # noqa: PLC0415

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
