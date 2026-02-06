# Tensormap and Ringbuffer Runtime build configuration
# All paths are relative to this file's directory (src/runtime/tensormap_and_ringbuffer/)
#
# This is a device-orchestration runtime where:
# - AICPU thread 3 runs the orchestrator (builds task graph on device)
# - AICPU threads 0/1/2 run schedulers (dispatch tasks to AICore)
# - AICore executes tasks via PTO2DispatchPayload

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
