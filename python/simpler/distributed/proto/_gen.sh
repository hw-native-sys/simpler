#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m grpc_tools.protoc \
  -I "${HERE}" \
  --python_out="${HERE}" \
  --grpc_python_out="${HERE}" \
  "${HERE}/dispatch.proto"

python - <<'PY' "${HERE}/dispatch_pb2_grpc.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("import dispatch_pb2 as dispatch__pb2", "from . import dispatch_pb2 as dispatch__pb2")
path.write_text(text)
PY
