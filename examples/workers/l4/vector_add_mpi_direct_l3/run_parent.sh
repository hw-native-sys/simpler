#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_LOCAL_HOST:?set the L4 machine address}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_REMOTE_HOST:?set the peer L3 machine address}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_MPI_PYTHON:?set an absolute launcher path valid on both machines}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_LOCAL_DEVICES:=0,1}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_REMOTE_DEVICES:=0,1}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_PLATFORM:=a2a3}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_RUNTIME:=tensormap_and_ringbuffer}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_STARTUP_TIMEOUT:=180}"
: "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_SESSION_TIMEOUT:=120}"

cd "${ROOT_DIR}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "error: ${ROOT_DIR}/.venv/bin/activate not found; create the virtual environment first" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

exec python -m examples.workers.l4.vector_add_mpi_direct_l3.main \
  --local-host "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_LOCAL_HOST}" \
  --remote-host "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_REMOTE_HOST}" \
  --python "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_MPI_PYTHON}" \
  --local-devices "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_LOCAL_DEVICES}" \
  --remote-devices "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_REMOTE_DEVICES}" \
  --platform "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_PLATFORM}" \
  --runtime "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_RUNTIME}" \
  --startup-timeout "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_STARTUP_TIMEOUT}" \
  --session-timeout "${SIMPLER_VECTOR_ADD_MPI_DIRECT_L3_SESSION_TIMEOUT}" \
  --mpirun-path "${MPIRUN:-mpirun}" \
  --launcher-family "${MPI_LAUNCHER_FAMILY:-auto}"
