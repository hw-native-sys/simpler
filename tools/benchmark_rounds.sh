#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware,
# then parse device-log timing lines to report per-round latency.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-c <config>] [-p <platform>] [-d <device>] [-n <rounds>] [-w <warmup>] [-v]
#
# Runs all examples listed in the config file and prints timing for each.
# Configuration is loaded from benchmark_config.json (next to this script by default).
# CLI arguments override config file values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Load configuration from JSON file
# ---------------------------------------------------------------------------
CONFIG_FILE="$SCRIPT_DIR/benchmark_config.json"

load_config() {
    local cfg="$1"
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: config file not found: $cfg" >&2
        exit 1
    fi
    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 is required to parse the config file" >&2
        exit 1
    fi
    # Parse JSON config via python3 (always available in this project)
    eval "$(python3 -c "
import json, sys, os
with open('$cfg') as f:
    c = json.load(f)
print('CFG_PROJECT_ROOT=' + repr(str(c.get('project_root', '..'))))
print('CFG_EXAMPLES_SUBDIR=' + repr(str(c.get('examples_subdir', 'tests/device_tests/\${platform}/tensormap_and_ringbuffer'))))
print('CFG_DEVICE_ID=' + repr(str(c.get('device_id', 0))))
print('CFG_ROUNDS=' + repr(str(c.get('rounds', 10))))
print('CFG_WARMUP_ROUNDS=' + repr(str(c.get('warmup_rounds', 2))))
print('CFG_PLATFORM=' + repr(str(c.get('platform', 'a2a3'))))
print('CFG_VERBOSE=' + ('true' if c.get('verbose', False) else 'false'))
print('CFG_PLOT=' + ('true' if c.get('plot', False) else 'false'))
print('CFG_LOG=' + ('true' if c.get('log', False) else 'false'))
examples = c.get('examples', [])
print('CFG_EXAMPLES=(' + ' '.join(repr(str(e)) for e in examples) + ')')
")"
}

apply_config() {
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/$CFG_PROJECT_ROOT" && pwd)"
    RUN_EXAMPLE="$PROJECT_ROOT/examples/scripts/run_example.py"
    DEVICE_ID="$CFG_DEVICE_ID"
    ROUNDS="$CFG_ROUNDS"
    WARMUP_ROUNDS="$CFG_WARMUP_ROUNDS"
    PLATFORM="$CFG_PLATFORM"
    EXAMPLES=("${CFG_EXAMPLES[@]}")
    EXAMPLES_SUBDIR="$CFG_EXAMPLES_SUBDIR"
    VERBOSE="$CFG_VERBOSE"
    PLOT="$CFG_PLOT"
    LOG="$CFG_LOG"
}

load_config "$CONFIG_FILE"
apply_config
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            load_config "$CONFIG_FILE"
            apply_config
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_ROUNDS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --plot)
            PLOT=true
            shift
            ;;
        --log)
            LOG=true
            shift
            ;;
        --help|-h)
            cat <<USAGE
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-c <config>] [-p <platform>] [-d <device>] [-n <rounds>] [-w <warmup>] [-v]

Options:
  -c, --config   Path to JSON config file (default: $SCRIPT_DIR/benchmark_config.json)
  -p, --platform Platform to run on (config default: a2a3)
  -d, --device   Device ID (config default: 0)
  -n, --rounds   Number of measured rounds per example (config default: 10)
  -w, --warmup   Number of warm-up rounds to discard (config default: 2)
  -v, --verbose  Print detailed run_example.py output (config default: false)
  --plot         Generate scatter plot PNG for each example (config default: false)
  --log          Save statistics to benchmark_logs/ for each example (config default: true)
  -h, --help     Show this help

CLI arguments override values from the config file.
All other options are passed through to run_example.py (e.g. --case).

Output:
  Mean and median elapsed time in microseconds for each example.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directory
# ---------------------------------------------------------------------------
# Substitute ${platform} placeholder in examples_subdir
RESOLVED_SUBDIR="${EXAMPLES_SUBDIR//\$\{platform\}/$PLATFORM}"
EXAMPLES_DIR="$PROJECT_ROOT/$RESOLVED_SUBDIR"

# ---------------------------------------------------------------------------
# Resolve device log directory (mirrors run_example.py / device_log_resolver.py)
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# ---------------------------------------------------------------------------
# parse_timing <log_file> <warmup_rounds>
#   Grep for orch_start / end lines, compute per-round elapsed, print summary.
#   Discards the first <warmup_rounds> rounds, then reports median, trimmed
#   mean (excluding min & max), and arithmetic mean for the remaining rounds.
# ---------------------------------------------------------------------------
parse_timing() {
    local log_file="$1"
    local warmup="${2:-0}"
    local example_name="${3:-unknown}"

    local timing
    timing=$(grep -E 'Thread=[0-9]+ (orch_start|end)=' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    echo "$timing" | awk -v warmup="$warmup" -v example="$example_name" '
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            results[round] = (max_end - min_start) / 50.0
            count++
        }
    }
    BEGIN { round = 0; min_start = 0; max_end = 0; count = 0 }
    /orch_start=/ {
        match($0, /Thread=([0-9]+)/, tm)
        tid = tm[1] + 0
        if (tid in seen) {
            flush_round()
            round++
            min_start = 0
            max_end = 0
            delete seen
        }
        seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, sm)
        val = sm[1] + 0
        if (min_start == 0 || val < min_start) min_start = val
    }
    /end=/ {
        match($0, /end=([0-9]+)/, em)
        val = em[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        flush_round()
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }

        # Print all rounds, marking warm-up rounds
        printf "  %-8s  %12s  %s\n", "Round", "Elapsed (us)", ""
        printf "  %-8s  %12s  %s\n", "-----", "------------", ""
        for (i = 0; i < count; i++) {
            if (i < warmup)
                printf "  %-8d  %12.1f  (warmup)\n", i, results[i]
            else
                printf "  %-8d  %12.1f\n", i, results[i]
        }

        # Collect measured (non-warmup) rounds
        m = 0
        for (i = warmup; i < count; i++) {
            measured[m] = results[i]
            m++
        }

        if (m == 0) {
            printf "\n  (all %d rounds were warm-up — no measured data)\n", count
            exit 1
        }

        # Sort measured[] (insertion sort — tiny array)
        for (i = 1; i < m; i++) {
            key = measured[i]
            j = i - 1
            while (j >= 0 && measured[j] > key) {
                measured[j + 1] = measured[j]
                j--
            }
            measured[j + 1] = key
        }

        # Arithmetic mean
        sum_v = 0
        for (i = 0; i < m; i++) sum_v += measured[i]
        mean_v = sum_v / m

        # Median
        if (m % 2 == 1)
            median_v = measured[int(m / 2)]
        else
            median_v = (measured[m / 2 - 1] + measured[m / 2]) / 2.0

        # Trimmed mean (drop one min and one max if we have >= 3 samples)
        if (m >= 3) {
            trim_sum = 0
            for (i = 1; i < m - 1; i++) trim_sum += measured[i]
            trimmed_v = trim_sum / (m - 2)
        } else {
            trimmed_v = mean_v
        }

        # Mean absolute deviation
        mad_sum = 0
        for (i = 0; i < m; i++) {
            diff = measured[i] - mean_v
            mad_sum += (diff < 0 ? -diff : diff)
        }
        mad_v = mad_sum / m

        # Standard deviation
        sq_sum = 0
        for (i = 0; i < m; i++) {
            diff = measured[i] - mean_v
            sq_sum += diff * diff
        }
        stddev_v = sqrt(sq_sum / m)

        printf "\n  Mean: %.1f us  |  Median: %.1f us  |  Trimmed-mean: %.1f us  (%d rounds, %d warmup)\n", \
            mean_v, median_v, trimmed_v, m, warmup
        range_v = measured[m - 1] - measured[0]
        fluct_v = 0
        if (mean_v > 0) fluct_v = (stddev_v / mean_v) * 100
        printf "  Range(Max-Min): %.1f us  |  Avg variation (MAD): %.1f us  |  Std deviation: %.1f us  |  Fluctuation rate (CV): %.2f%%\n", \
            range_v, mad_v, stddev_v, fluct_v
        printf "  result: %.1f us\n", mean_v

        # Emit machine-readable plot data (one line per measured round, original round numbers)
        for (i = warmup; i < count; i++)
            printf "PLOT_DATA:%s,%d,%.1f,%.1f,%.1f\n", example, i, results[i], mean_v, trimmed_v
    }'
}

# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>
#   Wait up to 15s for a new .log file in DEVICE_LOG_DIR. Prints the path.
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local new_log=""
    local deadline=$((SECONDS + 15))

    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done

    # Fallback: newest log
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        new_log=$(ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true)
        if [[ -n "$new_log" ]]; then
            echo "$new_log"
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
PLOT_DATA_FILE=""
if [[ "$PLOT" == "true" ]]; then
    PLOT_DATA_FILE=$(mktemp)
fi
LOG_DIR="$SCRIPT_DIR/benchmark_logs"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$LOG" == "true" || "$PLOT" == "true" ]]; then
    mkdir -p "$LOG_DIR"
fi

for example in "${EXAMPLES[@]}"; do
    EXAMPLE_DIR="$EXAMPLES_DIR/$example"
    KERNELS_DIR="$EXAMPLE_DIR/kernels"
    GOLDEN="$EXAMPLE_DIR/golden.py"

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ ! -f "$GOLDEN" || ! -d "$KERNELS_DIR" ]]; then
        echo "  SKIP: missing kernels/ or golden.py"
        ((FAIL++)) || true
        continue
    fi

    # Snapshot existing logs
    PRE_LOG_FILE=$(mktemp)
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$PRE_LOG_FILE" || true

    # Run example (measured rounds + warm-up rounds)
    TOTAL_ROUNDS=$(( ROUNDS + WARMUP_ROUNDS ))
    run_exit=0
    if [[ "$VERBOSE" == "true" ]]; then
        python3 "$RUN_EXAMPLE" \
                -k "$KERNELS_DIR" -g "$GOLDEN" \
                -p "$PLATFORM" -d "$DEVICE_ID" \
                -n "$TOTAL_ROUNDS" \
                "${EXTRA_ARGS[@]}" || run_exit=$?
    else
        python3 "$RUN_EXAMPLE" \
                -k "$KERNELS_DIR" -g "$GOLDEN" \
                -p "$PLATFORM" -d "$DEVICE_ID" \
                -n "$TOTAL_ROUNDS" \
                "${EXTRA_ARGS[@]}" > /dev/null 2>&1 || run_exit=$?
    fi

    if [[ $run_exit -ne 0 ]]; then
        echo "  FAILED: run_example.py returned non-zero"
        rm -f "$PRE_LOG_FILE"
        ((FAIL++)) || true
        continue
    fi

    # Find new device log
    NEW_LOG=$(wait_for_new_log "$PRE_LOG_FILE")
    rm -f "$PRE_LOG_FILE"

    if [[ -z "$NEW_LOG" ]]; then
        echo "  FAILED: no device log found in $DEVICE_LOG_DIR"
        ((FAIL++)) || true
        continue
    fi

    echo "  Log: $NEW_LOG"
    timing_output=$(parse_timing "$NEW_LOG" "$WARMUP_ROUNDS" "$example")
    timing_exit=$?
    # Print non-PLOT_DATA lines to stdout
    echo "$timing_output" | grep -v '^PLOT_DATA:'
    # Collect PLOT_DATA lines if plotting enabled
    if [[ "$PLOT" == "true" && -n "$PLOT_DATA_FILE" ]]; then
        echo "$timing_output" | grep '^PLOT_DATA:' >> "$PLOT_DATA_FILE" || true
    fi
    # Save statistics to log file if logging enabled
    if [[ "$LOG" == "true" && $timing_exit -eq 0 ]]; then
        LOG_FILE="$LOG_DIR/${example}_${LOG_TIMESTAMP}.log"
        echo "$timing_output" | grep -v '^PLOT_DATA:' > "$LOG_FILE"
        echo "  Log saved: $LOG_FILE"
    fi
    if [[ $timing_exit -eq 0 ]]; then
        ((PASS++)) || true
    else
        ((FAIL++)) || true
    fi
done

# ---------------------------------------------------------------------------
# Generate scatter plots (if enabled)
# ---------------------------------------------------------------------------
if [[ "$PLOT" == "true" && -n "$PLOT_DATA_FILE" && -s "$PLOT_DATA_FILE" ]]; then
    PLOT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    python3 -c "
import sys, os
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('  WARNING: matplotlib not available, skipping plot generation', file=sys.stderr)
    sys.exit(0)

from collections import OrderedDict

data = OrderedDict()  # example -> [(round, elapsed, mean, trimmed)]
with open('$PLOT_DATA_FILE') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('PLOT_DATA:'):
            continue
        parts = line[len('PLOT_DATA:'):].split(',')
        name, rnd, elapsed, mean, trimmed = parts[0], int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        data.setdefault(name, []).append((rnd, elapsed, mean, trimmed))

outdir = '$LOG_DIR'
ts = '$PLOT_TIMESTAMP'
for name, points in data.items():
    rounds = [p[0] for p in points]
    values = [p[1] for p in points]
    mean_val = points[0][2]
    trimmed_val = points[0][3]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(rounds, values, c='steelblue', s=40, zorder=3, label='Per-round elapsed')
    ax.axhline(y=mean_val, color='tomato', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f} us')
    ax.axhline(y=trimmed_val, color='blue', linestyle='--', linewidth=1.5, label=f'Trimmed-mean: {trimmed_val:.1f} us')
    ax.set_xlabel('Round')
    ax.set_ylabel('Elapsed (us)')
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(outdir, f'benchmark_{name}_{ts}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved: {outpath}')
" || echo "  WARNING: plot generation failed"
    rm -f "$PLOT_DATA_FILE"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Benchmark complete: $PASS passed, $FAIL failed (${#EXAMPLES[@]} total)"
echo "================================================================"

[[ $FAIL -eq 0 ]]
