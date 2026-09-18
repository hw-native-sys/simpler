#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Smoke-run this tool on one device. tools/build_cann_examples.sh --run calls it
# after building, with $1 = device id; it is also the shortest by-hand run.
#
# Kept next to the tool because the env var names it reads, the directory it must
# run from, and the argv that makes a short run are all specific to it -- so a
# shared runner cannot know them, and a by-name list in CI would have to.
set -euo pipefail
DEV="${1:?device id required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"

# Host-only: no dispatcher, no device SO. `version` is the cheapest subcommand
# that still calls into CANN.
exec "$HERE/build/query" version
