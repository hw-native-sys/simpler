# Tensormap and Ringbuffer Runtime build configuration
# All paths are relative to this file's directory (src/runtime/tensormap_and_ringbuffer/)
#
# This is a device-orchestration runtime where:
# - AICPU thread 3 runs the orchestrator (builds task graph on device)
# - AICPU threads 0/1/2 run schedulers (dispatch tasks to AICore)
# - AICore executes tasks via PTO2DispatchPayload
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "runtime", "orchestration"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime", "orchestration"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime", "orchestration"]
    },
    "orchestration": {
        "include_dirs": ["runtime", "orchestration"],
        "source_dirs": ["orchestration"]
    }
}
