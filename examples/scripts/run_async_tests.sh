#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEVICE_LIST="6,7"
PLATFORM="a2a3"
PTO_ISA_COMMIT=""
CLONE_PROTOCOL="https"
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-}"

usage() {
  cat <<'EOF'
Usage: examples/scripts/run_async_tests.sh [options]

Run the two async distributed hardware test cases:
  1. async_completion_demo
  2. async_notify_demo

Options:
  --devices <list>           Device list for distributed run (default: 6,7)
  --platform <name>          Platform passed to run_example.py (default: a2a3)
  --pto-isa-commit <sha>     PTO-ISA commit to pin (default: latest)
  --clone-protocol <proto>   PTO-ISA clone protocol: ssh|https (default: https)
  --cann-env <path>          CANN environment script to source
  -h, --help                 Show this help

Examples:
  CANN_ENV_SCRIPT=/path/to/set_env.sh examples/scripts/run_async_tests.sh
  examples/scripts/run_async_tests.sh --devices 6,7 --cann-env /path/to/set_env.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --devices)
      DEVICE_LIST="$2"
      shift 2
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --pto-isa-commit)
      PTO_ISA_COMMIT="$2"
      shift 2
      ;;
    --clone-protocol)
      CLONE_PROTOCOL="$2"
      shift 2
      ;;
    --cann-env)
      CANN_ENV_SCRIPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CANN_ENV_SCRIPT}" ]]; then
  echo "CANN env script is required. Pass --cann-env or set CANN_ENV_SCRIPT." >&2
  exit 1
fi

if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "CANN env script not found: ${CANN_ENV_SCRIPT}" >&2
  exit 1
fi

source "${CANN_ENV_SCRIPT}"
export PTO_PLATFORM="${PLATFORM}"

run_case() {
  local name="$1"
  local kernels="$2"
  local golden="$3"
  local -a cmd

  echo
  echo "============================================================"
  echo "Running ${name}"
  echo "============================================================"

  cmd=(
    python "${REPO_ROOT}/examples/scripts/run_example.py"
    -k "${kernels}"
    -g "${golden}"
    -p "${PLATFORM}"
    --devices "${DEVICE_LIST}"
    --clone-protocol "${CLONE_PROTOCOL}"
  )

  if [[ -n "${PTO_ISA_COMMIT}" ]]; then
    cmd+=(-c "${PTO_ISA_COMMIT}")
  fi

  "${cmd[@]}"
}

run_case \
  "async_completion_demo" \
  "${REPO_ROOT}/examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/kernels" \
  "${REPO_ROOT}/examples/a2a3/tensormap_and_ringbuffer/async_completion_demo/golden.py"

run_case \
  "async_notify_demo" \
  "${REPO_ROOT}/examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/kernels" \
  "${REPO_ROOT}/examples/a2a3/tensormap_and_ringbuffer/async_notify_demo/golden.py"

echo
echo "All async tests passed."
