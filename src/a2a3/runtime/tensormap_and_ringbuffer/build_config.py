import os

# Tensormap and Ringbuffer Runtime build configuration
# All paths are relative to this file's directory (src/runtime/tensormap_and_ringbuffer/)
#
# This is a device-orchestration runtime where:
# - AICPU thread 3 runs the orchestrator (builds task graph on device)
# - AICPU threads 0/1/2 run schedulers (dispatch tasks to AICore)
# - AICore executes tasks via an aligned PTO2DispatchPayload + pre-built dispatch_args
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).

def _resolve_pto_isa_include_dir() -> str:
    env_root = os.environ.get("PTO_ISA_ROOT")
    if env_root:
        include_dir = os.path.join(env_root, "include")
        if not os.path.isdir(include_dir):
            raise RuntimeError(
                f"PTO_ISA_ROOT is set but include directory does not exist: {include_dir}\n"
                "Please point PTO_ISA_ROOT to the pto-isa repository root."
            )
        return include_dir

    fallback_root = os.path.join(os.path.dirname(__file__), "../../../../3rd/pto-isa")
    fallback_include = os.path.join(fallback_root, "include")
    if os.path.isdir(fallback_include):
        return "../../../../3rd/pto-isa/include"

    raise RuntimeError(
        "PTO_ISA_ROOT is not set and the default fallback path does not exist:\n"
        f"  {fallback_include}\n"
        "Please export PTO_ISA_ROOT to the pto-isa repository root, for example:\n"
        "  export PTO_ISA_ROOT=/path/to/pto-isa"
    )


PTO_ISA_INCLUDE_DIR = _resolve_pto_isa_include_dir()

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "orchestration"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime", "orchestration"]
    },
    "host": {
        "include_dirs": ["runtime", PTO_ISA_INCLUDE_DIR],
        "source_dirs": ["host", "runtime", "orchestration"]
    },
    "orchestration": {
        "include_dirs": ["runtime", "orchestration"],
        "source_dirs": ["orchestration"]
    }
}
