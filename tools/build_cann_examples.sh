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
# Build every tool under tools/cann-examples, discovering each one's shape rather
# than naming it.
#
# Discovery is the point. The UT workflows used to name three tools and gained an
# entry per rot incident, so a tool nobody had listed was a tool nobody compiled:
# aicore-notification-perf stopped building on CANN 9.x unnoticed (#2333), and
# aicpu-mmio-probes was failing to link at the same time with nothing watching.
# Adding a tool should not require editing a workflow for it to be covered.
#
# The three shapes present, keyed on which directories a tool has:
#
#   <tool>/CMakeLists.txt                                 host-only, built in place
#   <tool>/device/            + <tool>/host/              AICPU cross-compile + host
#   <tool>/device-aicore/ + <tool>/device-aicpu/ + host/   ccec + AICPU cross + host
#
# A tool in none of those shapes is reported rather than skipped, so a new layout
# is a visible failure instead of silent non-coverage.
#
# Usage:
#   ASCEND_HOME_PATH=... tools/build_cann_examples.sh [cube-arch]
#   ASCEND_HOME_PATH=... tools/build_cann_examples.sh [cube-arch] --run <device-id>
#
# `cube-arch` is the ccec target for the AICore halves: dav-c220-cube for a2a3
# (the default), dav-c310-cube for a5.
#
# With --run, each tool's own smoke.sh is executed after it builds. That file
# lives next to the tool because the env var names it reads, the directory it
# runs from and the argv that makes a short run differ per tool — a shared runner
# cannot know them. A tool with no smoke.sh is reported as build-only rather than
# passed over silently, so the gap in run coverage is a number in the summary.
set -uo pipefail

ARCH="${1:-dav-c220-cube}"
RUN_DEVICE=""
if [ "${2:-}" = "--run" ]; then
    RUN_DEVICE="${3:?--run needs a device id}"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Per-tool wall-clock bound on a smoke run. Every tool here finishes in a few
# seconds, so this is pure headroom — its job is to cap what a wedged tool costs
# a locked NPU. Keep it far above a healthy run and far below the CI job budget.
SMOKE_TIMEOUT=120

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ASCEND_HOME_PATH must be set to the CANN toolkit root" >&2
    exit 1
fi
CROSS="${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu"

# The smoke runs need this repo's dispatcher SO, whose path is per-arch. Derive
# it from the ccec target so a5 does not silently reach for the a2a3 build; an
# explicit SIMPLER_DISPATCHER_SO in the environment still wins.
case "$ARCH" in
    dav-c220-*) DISPATCHER_ARCH=a2a3 ;;
    dav-c310-*) DISPATCHER_ARCH=a5 ;;
    *) DISPATCHER_ARCH="" ;;
esac
export SIMPLER_SMOKE_ARCH="$DISPATCHER_ARCH"
if [ -n "$RUN_DEVICE" ] && [ -z "${SIMPLER_DISPATCHER_SO:-}" ]; then
    if [ -z "$DISPATCHER_ARCH" ]; then
        echo "cannot derive the dispatcher arch from '$ARCH'; set SIMPLER_DISPATCHER_SO" >&2
        exit 1
    fi
    export SIMPLER_DISPATCHER_SO="$ROOT/build/lib/$DISPATCHER_ARCH/dispatcher/libsimpler_aicpu_dispatcher.so"
fi

# Build one cmake project, printing the tail of the real error on failure —
# a bare non-zero exit says only that something broke, not what.
build_one() {
    local dir="$1"
    shift
    local label="${dir#"$ROOT"/tools/cann-examples/}"
    if ! cmake -B "$dir/build" -S "$dir" "$@" >/dev/null 2>&1; then
        echo "  CONFIGURE FAILED: $label"
        cmake -B "$dir/build" -S "$dir" "$@" 2>&1 | tail -8 | sed 's/^/    /'
        return 1
    fi
    if ! cmake --build "$dir/build" >/dev/null 2>&1; then
        echo "  BUILD FAILED: $label"
        cmake --build "$dir/build" 2>&1 | grep -iE 'error|cannot find' | head -8 | sed 's/^/    /'
        return 1
    fi
    echo "  ok: $label"
    return 0
}

failed=()
ran=()
no_smoke=()
skipped=()
built=()

for tool in "$ROOT"/tools/cann-examples/*/; do
    name="$(basename "$tool")"
    # cmake/ holds shared modules, not a tool.
    [ "$name" = "cmake" ] && continue

    echo "=== $name"
    shapes=0
    status=0

    if [ -f "${tool}CMakeLists.txt" ]; then
        shapes=$((shapes + 1))
        build_one "${tool%/}" || status=1
    fi
    if [ -d "${tool}device-aicore" ]; then
        shapes=$((shapes + 1))
        build_one "${tool}device-aicore" -DCCE_AICORE_ARCH="$ARCH" || status=1
    fi
    for dev in "${tool}device" "${tool}device-aicpu"; do
        [ -d "$dev" ] || continue
        shapes=$((shapes + 1))
        build_one "$dev" -DCMAKE_C_COMPILER="${CROSS}-gcc" -DCMAKE_CXX_COMPILER="${CROSS}-g++" || status=1
    done
    if [ -d "${tool}host" ]; then
        shapes=$((shapes + 1))
        build_one "${tool}host" || status=1
    fi

    if [ "$shapes" -eq 0 ]; then
        echo "  UNRECOGNISED LAYOUT: no CMakeLists.txt, device*/ or host/ — not covered"
        status=1
    fi

    if [ "$status" -eq 0 ] && [ -n "$RUN_DEVICE" ]; then
        if [ -x "${tool}smoke.sh" ]; then
            smoke_out="$(timeout "$SMOKE_TIMEOUT" "${tool}smoke.sh" "$RUN_DEVICE" 2>&1)"
            smoke_rc=$?
            if [ "$smoke_rc" -eq 0 ]; then
                echo "  ran ok"
                ran+=("$name")
            elif [ "$smoke_rc" -eq 77 ]; then
                # 77 = the tool declares it cannot run on this arch and says why.
                echo "  skipped: $(printf '%s' "$smoke_out" | tail -1)"
                skipped+=("$name")
            elif [ "$smoke_rc" -eq 124 ]; then
                # Say that the bound fired, and why there is nothing below it: a
                # tool killed by SIGTERM never flushes its stdio buffer, so the
                # captured output of a hung run is normally empty. The by-hand
                # re-run is the way to see how far it got.
                echo "  RUN TIMED OUT: $name exceeded ${SMOKE_TIMEOUT}s and was killed"
                echo "    re-run to see where it stops: ${tool}smoke.sh $RUN_DEVICE"
                printf '%s\n' "$smoke_out" | tail -10 | sed 's/^/    /'
                status=1
            else
                echo "  RUN FAILED: $name (exit $smoke_rc)"
                printf '%s\n' "$smoke_out" | tail -10 | sed 's/^/    /'
                status=1
            fi
        else
            echo "  no smoke.sh — built but not run"
            no_smoke+=("$name")
        fi
    fi

    if [ "$status" -eq 0 ]; then
        built+=("$name")
    else
        failed+=("$name")
    fi
done

echo
echo "built  (${#built[@]}): ${built[*]:-none}"
if [ -n "$RUN_DEVICE" ]; then
    echo "ran    (${#ran[@]}): ${ran[*]:-none}"
    echo "skipped by arch (${#skipped[@]}): ${skipped[*]:-none}"
    echo "no smoke.sh (${#no_smoke[@]}): ${no_smoke[*]:-none}"
fi
echo "failed (${#failed[@]}): ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]
