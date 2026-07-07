#!/bin/bash
# Onboard probe runner for fdwc vector_example. Runs the ST test, then scrapes
# the device debug logs for the AICore progress crumbs + any exception dump.
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$REPO"
export PATH="${HOME}/.local/bin:${PATH}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source .venv/bin/activate
export PTO_ISA_ROOT="${REPO}/build/pto-isa"

# /tmp may be owned by another user on this shared host — use a per-user scratch dir.
TMP="${PROBE_TMP:-$HOME/.cache/fdwc_probe}"
mkdir -p "$TMP"
MARK="$TMP/_probe_marker"
touch "$MARK"

# pytest resolves binaries with build=False and does NOT recompile; build here.
# ABORT on build failure — a stale .o silently runs the previous binary otherwise.
echo "=== (re)building a5 runtime ==="
if ! python -m simpler_setup.build_runtimes --platforms a5 > "$TMP/probe_build.txt" 2>&1; then
  echo "BUILD_FAILED — aborting probe"
  grep -aoE "[a-z_]+\.(cpp|h):[0-9]+:[0-9]+: error: .{0,120}" "$TMP/probe_build.txt" | head -20
  tail -20 "$TMP/probe_build.txt"
  exit 3
fi
tail -2 "$TMP/probe_build.txt"

python -m pytest tests/st/a5/fully_distributed_within_core/vector_example/test_vector_example.py \
  --platform a5 --device "${TASK_DEVICE:-0}" --log-level v9 -v -s > "$TMP/probe.out" 2>&1
echo "PYTEST_EXIT=$?"
echo "=== pytest tail ==="
grep -aiE "507000|507046|error|FAIL|PASS|mismatch|golden|max_diff|assert|progress=|DIAG|self-test|reserve" "$TMP/probe.out" | tail -50
echo "=== orch_args tensor/scalar addresses AICPU handed to orchestration ==="
grep -aE "orch_args\[|TENSOR\(data|sm_ptr=|arg_count=" "$TMP/probe.out" | tail -30
echo "=== C16 DBGOUT (per-task out[0] chain) + actual f values ==="
grep -aE "DBGOUT|CI\(t0|DIAG _compare_outputs" "$TMP/probe.out" | tail -8

echo "=== device log crumbs + exception (device ${TASK_DEVICE:-0}) ==="
# Let the device dlog flush before we scrape.
sleep 3
LOGDIR="$HOME/ascend/log/debug/device-${TASK_DEVICE:-0}"
[ -d "$LOGDIR" ] || LOGDIR=$(dirname "$(ls -t $HOME/ascend/log/debug/device-*/*.log 2>/dev/null | head -1)")
echo "LOGDIR=$LOGDIR"
# Only logs newer than the run marker (this run), newest first.
FILES=$(find "$LOGDIR" -name "*.log" -newer "$MARK" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -4)
echo "FILES(this run)=$FILES"
echo "--- distinct progress=+dbg vectors across ALL newer logs (dedup, last 40) ---"
cat $FILES 2>/dev/null | grep -aE "progress=" | sed -E 's/.*(progress=\[[^]]*\])( dbg=\[[^]]*\])?.*/\1\2/' | uniq -c | tail -40
echo "--- total progress samples ---"
cat $FILES 2>/dev/null | grep -acE "progress="
echo "--- register: addresses AICPU handed to AICore ---"
cat $FILES 2>/dev/null | grep -aE "\[dist\] register:|host reserve:" | tail -10
echo "--- dist-engine / register / seg lines ---"
cat $FILES 2>/dev/null | grep -aiE "\[dist\]|host reserve|global_data_base|core_main|num_workers" | tail -20
echo "--- exception / fault context ---"
cat $FILES 2>/dev/null | grep -aiE "except|fault|MPU|error info|core dump|dump info| pc |0x[0-9a-f]{9,}|aicore error|507|retcode|su error|mte error|para base|STALL|completed=" | tail -50
echo "=== AICORE hardware exception dump (all ascend logs newer than mark) ==="
# The AICore MPU/instruction fault dump (faulting PC + access address) lands in
# the device-os / aicore plog, NOT the AICPU debug log. Scan the whole tree.
ALL=$(timeout 20 find "$HOME/ascend/log" -maxdepth 5 -name "*.log" -newer "$MARK" 2>/dev/null)
echo "scanned $(echo "$ALL" | wc -w) newer log files"
timeout 30 cat $ALL 2>/dev/null | grep -aiE "aicore.*exception|aic error|aiv error|exception occur|ExceptionInfo|core_id.*pc|fault.*addr|addr.*fault|MPU|mte|bus error|illegal|trap|dump_exception|kernel_name|para_base|block_dim|0x1000[0-9a-f]{6,}" | tail -60
