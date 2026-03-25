#!/usr/bin/env python3
"""Generate compile_commands.json for clangd IDE support.

Runs cmake configure (no build) via RuntimeCompiler with
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON, then merges per-target results and
places them in source directories so clangd resolves includes correctly.

For each arch, generates:
  - Every runtime × default variant → src/{arch}/runtime/{runtime}/compile_commands.json
  - Default runtime × every variant → src/{arch}/platform/{variant}/compile_commands.json

Usage:
    python tools/gen_compile_commands.py                                    # all archs
    python tools/gen_compile_commands.py --default-runtime host_build_graph # override default runtime
    python tools/gen_compile_commands.py --default-variant sim              # override default variant
    python tools/gen_compile_commands.py --list
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

TARGETS = ("host", "aicpu", "aicore")

ARCHS = ("a2a3", "a5")

PLATFORM_MAP = {
    ("a2a3", "onboard"): "a2a3",
    ("a2a3", "sim"): "a2a3sim",
    ("a5", "onboard"): "a5",
    ("a5", "sim"): "a5sim",
}


def load_build_config(config_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("build_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUILD_CONFIG


def cmake_configure(cmake_source_dir: str, cmake_args: list, build_dir: str) -> list:
    """Run cmake configure only and return compile_commands entries."""
    cmd = ["cmake", cmake_source_dir, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"] + cmake_args
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    cc_path = Path(build_dir) / "compile_commands.json"
    if cc_path.exists():
        return json.loads(cc_path.read_text())
    return []


def discover_runtimes(arch: str) -> list:
    runtime_base = PROJECT_ROOT / "src" / arch / "runtime"
    if not runtime_base.is_dir():
        return []
    return sorted(
        d.name for d in runtime_base.iterdir()
        if d.is_dir() and (d / "build_config.py").exists()
    )


def discover_variants(arch: str) -> list:
    platform_base = PROJECT_ROOT / "src" / arch / "platform"
    if not platform_base.is_dir():
        return []
    return sorted(
        d.name for d in platform_base.iterdir()
        if d.is_dir() and d.name not in ("include", "src")
    )


def generate(arch: str, runtime_name: str, variant: str) -> list:
    """Generate compile_commands entries by running cmake configure per target."""
    from runtime_compiler import RuntimeCompiler

    platform = PLATFORM_MAP.get((arch, variant))
    if platform is None:
        return []

    try:
        compiler = RuntimeCompiler.get_instance(platform=platform)
    except (ValueError, FileNotFoundError) as e:
        print(f"    skip {platform}: {e}", file=sys.stderr)
        return []

    runtime_dir = PROJECT_ROOT / "src" / arch / "runtime" / runtime_name
    config_path = runtime_dir / "build_config.py"
    if not config_path.exists():
        return []

    build_config = load_build_config(config_path)
    all_entries = []

    for target_name in TARGETS:
        if target_name not in build_config:
            continue

        target_cfg = build_config[target_name]
        include_dirs = [str((runtime_dir / p).resolve()) for p in target_cfg["include_dirs"]]
        source_dirs = [str((runtime_dir / p).resolve()) for p in target_cfg["source_dirs"]]

        target = getattr(compiler, f"{target_name}_target")
        cmake_args = target.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = target.get_root_dir()

        with tempfile.TemporaryDirectory(prefix=f"compdb_{target_name}_") as build_dir:
            entries = cmake_configure(cmake_source_dir, cmake_args, build_dir)
            all_entries.extend(entries)

    return all_entries


def write_compile_commands(entries: list, output_path: Path) -> None:
    output_path.write_text(json.dumps(entries, indent=2) + "\n")


def generate_for_arch(arch: str, default_runtime: str, default_variant: str) -> None:
    """Generate compile_commands.json for all runtimes and variants of one arch."""
    runtimes = discover_runtimes(arch)
    variants = discover_variants(arch)

    if not runtimes:
        print(f"  {arch}: no runtimes found, skipping", file=sys.stderr)
        return

    if default_runtime not in runtimes:
        print(f"  {arch}: default runtime '{default_runtime}' not found, using {runtimes[0]}", file=sys.stderr)
        default_runtime = runtimes[0]

    if default_variant not in variants:
        print(f"  {arch}: default variant '{default_variant}' not found, using {variants[0]}", file=sys.stderr)
        default_variant = variants[0]

    # Cache: (runtime, variant) → entries, to avoid duplicate cmake runs
    cache = {}

    def get_entries(runtime: str, variant: str) -> list:
        key = (runtime, variant)
        if key not in cache:
            cache[key] = generate(arch, runtime, variant)
        return cache[key]

    # Every runtime × default variant → runtime dir
    for runtime in runtimes:
        entries = get_entries(runtime, default_variant)
        if entries:
            output = PROJECT_ROOT / "src" / arch / "runtime" / runtime / "compile_commands.json"
            write_compile_commands(entries, output)
            print(f"  {output.relative_to(PROJECT_ROOT)}  ({len(entries)} entries, variant={default_variant})", file=sys.stderr)

    # Default runtime × every variant → platform variant dir
    for variant in variants:
        entries = get_entries(default_runtime, variant)
        if entries:
            output = PROJECT_ROOT / "src" / arch / "platform" / variant / "compile_commands.json"
            write_compile_commands(entries, output)
            print(f"  {output.relative_to(PROJECT_ROOT)}  ({len(entries)} entries, runtime={default_runtime})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate compile_commands.json for clangd via cmake"
    )
    parser.add_argument(
        "--default-runtime", default="tensormap_and_ringbuffer",
        help="Default runtime for platform variant generation (default: tensormap_and_ringbuffer)",
    )
    parser.add_argument(
        "--default-variant", default="onboard",
        help="Default platform variant for runtime generation (default: onboard)",
    )
    parser.add_argument("--list", action="store_true", help="List available options per arch")
    args = parser.parse_args()

    if args.list:
        for arch in ARCHS:
            arch_dir = PROJECT_ROOT / "src" / arch
            if not arch_dir.is_dir():
                continue
            runtimes = discover_runtimes(arch)
            variants = discover_variants(arch)
            print(f"{arch}:")
            print(f"  runtimes: {', '.join(runtimes) or '(none)'}")
            print(f"  variants: {', '.join(variants) or '(none)'}")
        return

    for arch in ARCHS:
        arch_dir = PROJECT_ROOT / "src" / arch
        if not arch_dir.is_dir():
            continue
        print(f"{arch}:", file=sys.stderr)
        generate_for_arch(arch, args.default_runtime, args.default_variant)


if __name__ == "__main__":
    main()
