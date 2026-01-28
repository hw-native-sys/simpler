import importlib.util
from pathlib import Path
from typing import Optional

from binary_compiler import BinaryCompiler


class RuntimeBuilder:
    """Discovers and builds runtime implementations from src/runtime/."""

    def __init__(self, runtime_root: Optional[Path] = None):
        """
        Scan src/runtime/ for subdirectories containing build_config.py.

        Args:
            runtime_root: Root directory of the project. Defaults to parent of python/.
        """
        if runtime_root is None:
            runtime_root = Path(__file__).parent.parent
        self.runtime_root = runtime_root
        self.runtime_dir = runtime_root / "src" / "runtime"

        # Discover available runtime implementations
        self._runtimes = {}
        if self.runtime_dir.is_dir():
            for entry in sorted(self.runtime_dir.iterdir()):
                config_path = entry / "build_config.py"
                if entry.is_dir() and config_path.is_file():
                    self._runtimes[entry.name] = config_path

    def list_runtimes(self) -> list:
        """Return names of discovered runtime implementations."""
        return list(self._runtimes.keys())

    def build(self, name: str) -> tuple:
        """
        Build a specific runtime implementation by name.

        Args:
            name: Name of the runtime implementation (e.g. 'host_build_graph')

        Returns:
            Tuple of (host_binary, aicpu_binary, aicore_binary) as bytes

        Raises:
            ValueError: If the named runtime is not found
        """
        if name not in self._runtimes:
            available = ", ".join(self._runtimes.keys()) or "(none)"
            raise ValueError(
                f"Runtime '{name}' not found. Available runtimes: {available}"
            )

        config_path = self._runtimes[name]
        config_dir = config_path.parent

        # Load build_config.py
        spec = importlib.util.spec_from_file_location("build_config", config_path)
        build_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_config_module)
        build_config = build_config_module.BUILD_CONFIG

        compiler = BinaryCompiler()

        # Compile AICore kernel
        print("\n[1/3] Compiling AICore kernel...")
        aicore_cfg = build_config["aicore"]
        aicore_include_dirs = [str((config_dir / p).resolve()) for p in aicore_cfg["include_dirs"]]
        aicore_source_dirs = [str((config_dir / p).resolve()) for p in aicore_cfg["source_dirs"]]
        aicore_binary = compiler.compile("aicore", aicore_include_dirs, aicore_source_dirs)

        # Compile AICPU kernel
        print("\n[2/3] Compiling AICPU kernel...")
        aicpu_cfg = build_config["aicpu"]
        aicpu_include_dirs = [str((config_dir / p).resolve()) for p in aicpu_cfg["include_dirs"]]
        aicpu_source_dirs = [str((config_dir / p).resolve()) for p in aicpu_cfg["source_dirs"]]
        aicpu_binary = compiler.compile("aicpu", aicpu_include_dirs, aicpu_source_dirs)

        # Compile Host runtime
        print("\n[3/3] Compiling Host runtime...")
        host_cfg = build_config["host"]
        host_include_dirs = [str((config_dir / p).resolve()) for p in host_cfg["include_dirs"]]
        host_source_dirs = [str((config_dir / p).resolve()) for p in host_cfg["source_dirs"]]
        host_binary = compiler.compile("host", host_include_dirs, host_source_dirs)

        print("\nBuild complete!")
        return (host_binary, aicpu_binary, aicore_binary)
