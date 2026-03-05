#!/usr/bin/env python3
"""
Multi-card test runner for PTO runtime tests.

This script provides a command-line interface to run PTO runtime tests
with multi-card support (compile once, run on N devices in parallel).
Users only need to provide:
1. A kernels directory with kernel_config.py
2. A golden.py script

Usage:
    python examples/scripts/multi_card_run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py
    python examples/scripts/multi_card_run_example.py -k ./kernels -g ./golden.py --device 0 --platform a2a3sim

Examples:
    # Run hardware example (requires Ascend device)
    python examples/scripts/multi_card_run_example.py -k examples/host_build_graph/vector_example/kernels \
                                      -g examples/host_build_graph/vector_example/golden.py

    # Run simulation example (no hardware required)
    python examples/scripts/multi_card_run_example.py -k examples/host_build_graph/vector_example/kernels \
                                      -g examples/host_build_graph/vector_example/golden.py \
                                      -p a2a3sim

    # Run with specific device
    python examples/scripts/multi_card_run_example.py -k ./kernels -g ./golden.py -d 0

    # Multi-card (e.g. multi_bgemm)
    python examples/scripts/multi_card_run_example.py -k examples/host_build_graph/multi_bgemm/kernels \
                                      -g examples/host_build_graph/multi_bgemm/golden.py \
                                      --n-devices 2 --first-device 0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Get script and project directories
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
python_dir = project_root / "python"
if python_dir.exists():
    sys.path.insert(0, str(python_dir))

logger = logging.getLogger(__name__)


def _get_device_log_dir(device_id):
    """Return the device log directory using the same logic as device_log_resolver."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if root.exists():
            return root / f"device-{device_id}"
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"


def _run_profiling_swimlane(args, kernels_path, project_root, device_log_dir, pre_run_device_logs, log_level_str):
    """Run swimlane converter after test. Returns 0 on success."""
    swimlane_script = project_root / "tools" / "swimlane_converter.py"
    if not swimlane_script.exists():
        logger.warning("Swimlane converter script not found")
        return 0
    import subprocess
    try:
        cmd = [sys.executable, str(swimlane_script), "-k", str(kernels_path)]
        if device_log_dir is not None:
            device_log_file = _wait_for_new_device_log(device_log_dir, pre_run_device_logs)
            if device_log_file:
                cmd += ["--device-log", str(device_log_file)]
            else:
                cmd += ["-d", str(args.device)]
        else:
            cmd += ["-d", str(args.device)]
        if log_level_str == "debug":
            cmd.append("-v")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Swimlane JSON generation completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Swimlane conversion failed: {e}")
    return 0


def _wait_for_new_device_log(log_dir, pre_run_logs, timeout=15, interval=0.5):
    """Wait for a new device log file that wasn't present before the run.

    CANN dlog writes device logs asynchronously, so the file may appear
    a few seconds after the run completes.
    """
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_dir.exists():
            current_logs = set(log_dir.glob("*.log"))
            new_logs = current_logs - pre_run_logs
            if new_logs:
                return max(new_logs, key=lambda p: p.stat().st_mtime)
        time.sleep(interval)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run PTO runtime test with multi-card support (kernel config and golden script)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/scripts/multi_card_run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py
    python examples/scripts/multi_card_run_example.py -k ./kernels -g ./golden.py -d 0

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
        default=0,
        help="Device ID (default: 0)"
    )

    parser.add_argument(
        "--n-devices",
        type=int,
        default=None,
        help="Number of devices to run on (multi-card). Overrides kernel_config RUNTIME_CONFIG. Default from config or 1."
    )

    parser.add_argument(
        "--first-device",
        type=int,
        default=None,
        help="First device ID for multi-card (e.g. 4 with --n-devices 4 uses devices 4,5,6,7). Overrides kernel_config."
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
        help="Enable verbose output (equivalent to --log-level debug)"
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Silent mode - only show errors (equivalent to --log-level error)"
    )

    parser.add_argument(
        "--log-level",
        choices=["error", "warn", "info", "debug"],
        help="Set log level explicitly (overrides --verbose and --silent)"
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable profiling and generate swimlane.json"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test cases defined in ALL_CASES (default: run only DEFAULT_CASE)"
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run a specific test case by name (e.g., --case Case2)"
    )

    args = parser.parse_args()

    if args.all and args.case:
        parser.error("--all and --case are mutually exclusive")

    # Determine log level from arguments
    log_level_str = None
    if args.log_level:
        log_level_str = args.log_level
    elif args.verbose:
        log_level_str = "debug"
    elif args.silent:
        log_level_str = "error"
    else:
        log_level_str = "info"
    
    # Setup logging before any other operations
    level_map = {
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
    }
    log_level = level_map.get(log_level_str.lower(), logging.INFO)
    
    # Configure Python logging
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        force=True
    )
    
    # Set environment variable for C++ side
    os.environ['PTO_LOG_LEVEL'] = log_level_str

    # Add script_dir for multi_card_code_runner (now co-located)
    sys.path.insert(0, str(script_dir))

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

    # Import and run
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from multi_card_code_runner import create_code_runner, create_compiler, run_on_device, run_on_device_comm

        # Compile first
        compiler = create_compiler(kernels_dir=str(args.kernels), platform=args.platform)
        artifacts = compiler.compile()

        # Resolve n_devices and first_device_id (args override config)
        import importlib.util
        spec = importlib.util.spec_from_file_location("kernel_config", kernel_config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        runtime_config = getattr(cfg, "RUNTIME_CONFIG", {})
        n_devices = args.n_devices if args.n_devices is not None else runtime_config.get("n_devices", 1)
        first_device_id = args.first_device if args.first_device is not None else runtime_config.get("first_device_id", 0)
        requires_comm = runtime_config.get("requires_comm", False)
        root = runtime_config.get("root", 0)

        if requires_comm and n_devices > 1:
            # Multi-card with HCCL: use Barrier + shared root_info
            import multiprocessing as mp
            from hccl_bindings import hccl_get_root_info, HCCL_ROOT_INFO_BYTES

            root_info_arr = mp.Array("b", HCCL_ROOT_INFO_BYTES)
            barrier = mp.Barrier(n_devices)

            def _comm_worker(rank_id):
                device_id = rank_id % n_devices + first_device_id
                if rank_id == 0:
                    root_info = hccl_get_root_info(device_id)
                    root_info_arr[:] = root_info[:HCCL_ROOT_INFO_BYTES]
                barrier.wait()
                root_info = bytes(root_info_arr[:])
                run_on_device_comm(
                    rank_id=rank_id,
                    device_id=device_id,
                    root_info=root_info,
                    artifacts=artifacts,
                    kernels_dir=str(args.kernels),
                    golden_path=str(args.golden),
                    n_ranks=n_devices,
                    n_devices=n_devices,
                    first_device_id=first_device_id,
                    root=root,
                    platform=args.platform,
                    enable_profiling=args.enable_profiling,
                    run_all_cases=args.all,
                    case_name=args.case,
                )

            procs = [mp.Process(target=_comm_worker, args=(r,)) for r in range(n_devices)]
            for p in procs:
                p.start()
            failed = []
            for r, p in enumerate(procs):
                p.join()
                if p.exitcode != 0:
                    failed.append((r, RuntimeError(f"Rank {r} exited with code {p.exitcode}")))
            if failed:
                err_msg = "; ".join(f"rank {d}: {e}" for d, e in failed)
                raise RuntimeError(f"Multi-card comm run failed: {err_msg}")
            logger.info("=" * 60)
            logger.info("TEST PASSED (all ranks)")
            logger.info("=" * 60)
            return 0
        elif n_devices > 1:
            # Multi-device: create N CodeRunner instances, run in parallel via ProcessPoolExecutor
            device_ids = list(range(first_device_id, first_device_id + n_devices))
            logger.info(f"=== Multi-device: compile done, running on devices {device_ids} (parallel) ===")

            failed = []
            with ProcessPoolExecutor(max_workers=n_devices) as executor:
                futures = {
                    executor.submit(
                        run_on_device,
                        did,
                        artifacts,
                        str(args.kernels),
                        str(args.golden),
                        args.platform,
                        args.enable_profiling,
                        args.all,
                        args.case,
                    ): did
                    for did in device_ids
                }
                for fut in as_completed(futures):
                    did = futures[fut]
                    try:
                        fut.result()
                        logger.info(f"Device {did}: PASS")
                    except Exception as e:
                        failed.append((did, e))
                        logger.error(f"Device {did} failed: {e}")

            if failed:
                err_msg = "; ".join(f"device {d}: {e}" for d, e in failed)
                raise RuntimeError(f"Multi-device run failed: {err_msg}")

            logger.info("=" * 60)
            logger.info("TEST PASSED (all devices)")
            logger.info("=" * 60)
            return 0
        else:
            # Single device: run in-process with compiled artifacts
            runner = create_code_runner(
                kernels_dir=str(args.kernels),
                golden_path=str(args.golden),
                device_id=args.device,
                platform=args.platform,
                enable_profiling=args.enable_profiling,
                run_all_cases=args.all,
                case_name=args.case,
                n_devices=1,
                first_device_id=args.device,
                compiled_artifacts=artifacts,
            )

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

            if args.enable_profiling:
                logger.info("Generating swimlane visualization...")
                _run_profiling_swimlane(args, kernels_path, project_root, device_log_dir, pre_run_device_logs, log_level_str)

        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root directory.")
        return 1

    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        if log_level_str == "debug":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
