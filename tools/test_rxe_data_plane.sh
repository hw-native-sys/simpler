#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# Default to the local rdma-core build that is known to back ibv_rc_pingpong on this host.
export SIMPLER_RXE_INCLUDE_DIR="${SIMPLER_RXE_INCLUDE_DIR:-/home/ntlab/rdma-build/rdma-core-50.0/build/include}"
export SIMPLER_RXE_LIB_DIR="${SIMPLER_RXE_LIB_DIR:-/home/ntlab/rdma-build/rdma-core-50.0/build/lib}"

echo "[1/4] Python import/compile sanity"
python -m py_compile \
  python/simpler/distributed/transport_backend.py \
  python/simpler/distributed/remote_proxy.py \
  python/simpler/distributed/l3_daemon.py \
  tests/ut/py/test_distributed/test_real_e2e_smoke.py \
  tests/ut/py/test_distributed/test_transport_backend.py \
  tools/benchmark_rxe_data_plane.py

echo "[2/4] Distributed unit tests"
python -m pytest \
  tests/ut/py/test_distributed/test_catalog.py \
  tests/ut/py/test_distributed/test_heartbeat.py \
  tests/ut/py/test_distributed/test_import.py \
  tests/ut/py/test_distributed/test_l4_l3_remote.py \
  tests/ut/py/test_distributed/test_rpc_roundtrip.py \
  tests/ut/py/test_distributed/test_tensor_pool.py \
  tests/ut/py/test_distributed/test_transport_backend.py \
  -q

echo "[3/4] RXE/ibverbs RC pingpong smoke"
SIMPLER_RXE_REAL_TEST=1 \
python -m pytest tests/ut/py/test_distributed/test_rxe_real.py -q -s

echo "[4/4] L4 -> L3 RXE tensor data-plane E2E"
SIMPLER_REAL_E2E_TEST=1 \
SIMPLER_TENSOR_TRANSPORT=rxe \
python -m pytest tests/ut/py/test_distributed/test_real_e2e_smoke.py -q -s -k "rxe"

if [[ "${SIMPLER_RUN_RXE_BENCHMARK:-0}" == "1" ]]; then
  echo "[optional] RXE vs gRPC short benchmark"
  tools/benchmark_rxe_data_plane.py --sizes 8192,65536 --repeats 3 --warmup 1
fi

echo "RXE data-plane tests passed."
