# Runtime build configuration
# All paths are relative to this file's directory (src/runtime/)

"""
Runtime build configuration for aicpu_build_graph.

Note: AICPU graph-building logic is loaded at runtime as a device-side `.so` plugin
via `dlopen+dlsym`, so example builder sources should NOT be compiled into the
runtime AICPU binary.
"""

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime"]
    }
}
