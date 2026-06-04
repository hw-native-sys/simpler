#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# onboard-arch-precheck: gate onboard --platform invocations against actual Ascend silicon.
#
# Usage:   check.sh <requested_platform>
# Exits:   0 = match, or sim variant (pass through)
#          1 = unable to detect (npu-smi missing, CANN ini missing, unknown SoC, etc.)
#          2 = silicon mismatch
#
# Detection path mirrors tools/cann-examples/query/ — single source of truth
# is the CANN platform_config/<SoC>.ini Short_SoC_version field. See
# docs/hardware/chip-architecture.md for the canonical table. SKILL.md
# holds the why (failure-mode context for 507018 / 507899).

set -u

REQUESTED="${1:-}"
CACHE="/tmp/onboard-arch-precheck.cache"
TTL=3600   # 1 hour

if [ -z "$REQUESTED" ]; then
    cat >&2 <<EOF
onboard-arch-precheck: missing platform argument.
Usage: $0 <platform>   (one of: a2a3, a2a3sim, a5, a5sim)
EOF
    exit 1
fi

# Sim variants are silicon-agnostic — pass through.
case "$REQUESTED" in
    a2a3sim|a5sim) exit 0 ;;
    a2a3|a5) ;;
    *)
        echo "onboard-arch-precheck: unknown platform '$REQUESTED' — expected a2a3|a5|a2a3sim|a5sim" >&2
        exit 1
        ;;
esac

# detect_silicon prints "arch|soc_name|short_soc_version" on stdout.
# Uses npu-smi to identify the chip + the CANN platform_config ini to map
# SoC → Short_SoC_version → repo arch. Same Short_SoC_version → arch table
# as docs/hardware/chip-architecture.md and tools/cann-examples/query/.
detect_silicon() {
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "onboard-arch-precheck: npu-smi not on PATH — cannot determine silicon. Install the Ascend driver or run on a host with one." >&2
        return 1
    fi
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        echo "onboard-arch-precheck: ASCEND_HOME_PATH not set — needed to locate CANN platform_config/." >&2
        return 1
    fi

    local board chip npu
    board=$(npu-smi info -t board -i 0 -c 0 2>/dev/null)
    chip=$(awk -F: '/Chip Name/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' <<<"$board")
    npu=$(awk -F:  '/NPU Name/  { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' <<<"$board")
    if [ -z "$chip" ] || [ -z "$npu" ]; then
        echo "onboard-arch-precheck: npu-smi did not return Chip Name + NPU Name (chip='$chip' npu='$npu'). Try \`npu-smi info -t board -i 0 -c 0\` manually." >&2
        return 1
    fi

    # CANN ships platform_config under {arch}-linux/data/platform_config/.
    # The arch subdir matches the install target, not the host — but on a
    # given install only one exists.
    local config_dir=""
    for sub in "aarch64-linux" "x86_64-linux"; do
        if [ -d "${ASCEND_HOME_PATH}/${sub}/data/platform_config" ]; then
            config_dir="${ASCEND_HOME_PATH}/${sub}/data/platform_config"
            break
        fi
    done
    if [ -z "$config_dir" ]; then
        echo "onboard-arch-precheck: no CANN platform_config dir found under ${ASCEND_HOME_PATH}/{aarch64,x86_64}-linux/data/" >&2
        return 1
    fi

    # Construct candidate SoC ini filename. Naming differs per family:
    #   910B series:  Ascend910B3   (chip + npu, no separator)
    #   910 9xxx:     Ascend910_9392 (chip _ npu)
    #   950 DT/PR:    Ascend950DT_9586 / Ascend950PR_9579 (need glob)
    local soc=""
    case "$chip" in
        Ascend910)
            if [[ "$npu" =~ ^B ]]; then
                soc="${chip}${npu}"
            else
                soc="${chip}_${npu}"
            fi
            ;;
        Ascend950)
            for prefix in DT PR; do
                if [ -f "${config_dir}/${chip}${prefix}_${npu}.ini" ]; then
                    soc="${chip}${prefix}_${npu}"
                    break
                fi
            done
            [ -z "$soc" ] && soc="${chip}_${npu}"
            ;;
        *)
            echo "onboard-arch-precheck: unrecognized Chip Name '$chip'. Update check.sh + docs/hardware/chip-architecture.md if a new family was added." >&2
            return 1
            ;;
    esac

    local ini="${config_dir}/${soc}.ini"
    if [ ! -f "$ini" ]; then
        echo "onboard-arch-precheck: CANN ini not found at ${ini} (Chip='$chip', NPU='$npu'). The SoC name pattern in check.sh may be stale." >&2
        return 1
    fi

    local short
    short=$(awk -F= '/^Short_SoC_version=/ { gsub(/\r$/, "", $2); print $2; exit }' "$ini")
    if [ -z "$short" ]; then
        echo "onboard-arch-precheck: ${ini} missing Short_SoC_version line" >&2
        return 1
    fi

    # Short_SoC_version → repo arch. Table mirrors:
    #   - docs/hardware/chip-architecture.md (Identifying which chip generation)
    #   - tools/cann-examples/query/query.cpp::arch_from_short_soc
    # Keep these three in sync if a new SoC family is added.
    local arch
    case "$short" in
        Ascend910B)   arch=a2a3 ;;   # Atlas A2 series (910B1..B4)
        Ascend910_93) arch=a2a3 ;;   # Atlas A3 series (910_9362..9392)
        Ascend950)    arch=a5 ;;
        *)
            echo "onboard-arch-precheck: unsupported Short_SoC_version '$short' (SoC=$soc). Update check.sh + docs/hardware/chip-architecture.md." >&2
            return 1
            ;;
    esac

    printf '%s|%s|%s\n' "$arch" "$soc" "$short"
}

# Cache (1hr TTL). Format: "arch|soc|short_soc"
DETECTED_LINE=""
if [ -f "$CACHE" ]; then
    NOW=$(date +%s)
    MTIME=$(stat -c %Y "$CACHE" 2>/dev/null || echo 0)
    AGE=$((NOW - MTIME))
    if [ "$AGE" -lt "$TTL" ]; then
        DETECTED_LINE=$(cat "$CACHE" 2>/dev/null || true)
    fi
fi
if [ -z "$DETECTED_LINE" ]; then
    if ! DETECTED_LINE=$(detect_silicon); then
        exit 1
    fi
    printf '%s\n' "$DETECTED_LINE" > "$CACHE" 2>/dev/null || true
fi

DETECTED=$(echo "$DETECTED_LINE" | cut -d'|' -f1)
SOC=$(echo "$DETECTED_LINE" | cut -d'|' -f2)
SHORT=$(echo "$DETECTED_LINE" | cut -d'|' -f3)

if [ "$REQUESTED" = "$DETECTED" ]; then
    exit 0
fi

# Mismatch — refuse with the failure-mode context.
cat >&2 <<EOF

==================================================================
onboard-arch-precheck: ARCH MISMATCH — refusing to run.

  Requested platform : --platform $REQUESTED   (onboard)
  Detected silicon   : $DETECTED  (SoC=$SOC, Short_SoC_version=$SHORT)

Running onboard tests built for the wrong silicon produces error
cascades that LOOK LIKE genuine bugs but aren't. Typical signatures:

  - CANN 507018 (ACL_ERROR_RT_AICPU_EXCEPTION) at aclrtSynchronizeStreamWithTimeout
  - CANN 507899 at rtStreamCreate / rtMalloc

Both codes can ALSO be genuine bugs — which is precisely what makes
the misdiagnosis costly. CI is fine (matched runners); only local
invocations hit this. See .claude/skills/onboard-arch-precheck/SKILL.md.

Options:
  - Run on this silicon:    --platform $DETECTED
  - Run silicon-agnostic:   --platform ${REQUESTED}sim   (host CPU)
  - Use a different box     (one with --platform $REQUESTED silicon)
==================================================================
EOF
exit 2
