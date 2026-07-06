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
# Decide whether a self-hosted *fallback* job should DEFER to its GitHub-hosted
# *primary* job (because GitHub is running it) or SELF-RUN the tests because
# GitHub never scheduled the primary within the start deadline.
#
# This is the core of the "GitHub-first, fail over to self-hosted on a
# scheduling stall, only one actually runs the tests" pattern used by the
# ubuntu-fallback job in ci.yml. The GitHub primaries are the required checks;
# this convenience job DEFERS (exits, does nothing) as soon as GitHub starts a
# primary, and only self-runs a leg when GitHub never schedules it (a stall).
#
# Usage:
#   ci-wait-or-fallback.sh "<primary job name>" <start_timeout_secs> <expect_fast>
#
#   <primary job name>   Exact job name as it appears in the run's jobs list,
#                        e.g. "st-sim-a2a3 (ubuntu-latest, 3.10)". Matrix legs
#                        include the "(os, py)" suffix -- if the matrix changes,
#                        this string must change too (see RENAME_ERROR below).
#   <start_timeout_secs> How long to wait for the primary to leave "queued"
#                        before declaring a stall (600 = 10 min).
#   <expect_fast>        1 if the primary has no `needs` and should appear in
#                        the jobs list almost immediately (e.g. `ut`); 0 if it
#                        is gated by `needs: detect-changes` and may appear late.
#
# Requires env: GITHUB_TOKEN (with actions:read), GITHUB_API_URL,
#               GITHUB_REPOSITORY, GITHUB_RUN_ID.
#
# Prints exactly one decision token as the LAST line of stdout; all diagnostics
# go to stderr. Exit code is always 0 -- the caller branches on the token:
#   DEFER         primary is running or finished on GitHub: it gates itself,
#                 nothing for this job to do
#   SELFRUN       GitHub never scheduled the primary (stall): run the tests locally
#   RENAME_ERROR  primary never appeared though it should have (likely a
#                 matrix/name change broke matching): log, do not self-run
#   FETCH_ERROR   could not reach the jobs API for the whole deadline: log, do
#                 not self-run (state unknown)
#
set -uo pipefail

PRIMARY="${1:?primary job name required}"
START_TIMEOUT="${2:-600}"
EXPECT_FAST="${3:-0}"

: "${GITHUB_TOKEN:?}" "${GITHUB_API_URL:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_RUN_ID:?}"

API="${GITHUB_API_URL}/repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100"
POLL=15
elapsed=0
seen=0   # have we ever observed the primary job in the jobs list?

log() { printf '[fallback] %s\n' "$*" >&2; }

fetch() {
  # -S surfaces errors on stderr; the timeouts bound each call so a stalled TCP
  # connection can never hang past the poll interval.
  curl -sSf --connect-timeout 10 --max-time 30 \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${API}"
}

field() {  # field <json> <job-name> <field>  (e.g. status, conclusion)
  # python3 (always present on these runners) instead of jq (often absent).
  printf '%s' "$1" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
name, key = sys.argv[1], sys.argv[2]
for j in data.get("jobs", []):
    if j.get("name") == name:
        v = j.get(key)
        print("" if v is None else v)
        break
' "$2" "$3"
}

while true; do
  JSON="$(fetch || true)"
  if [ -n "${JSON}" ]; then
    STATUS="$(field "${JSON}" "${PRIMARY}" 'status')"
    CONCL="$(field "${JSON}" "${PRIMARY}" 'conclusion')"
    if [ -n "${STATUS}" ] && [ "${STATUS}" != "null" ]; then
      seen=1
      case "${STATUS}" in
        in_progress|completed)
          # GitHub is running or has finished the primary. The primary is the
          # required check and gates itself (its pass/fail/skip is authoritative),
          # so this convenience job has nothing to do.
          log "primary status=${STATUS} (conclusion=${CONCL:-n/a}) -> DEFER (primary gates)"
          echo "DEFER"; exit 0 ;;
        *)
          # queued / waiting / pending / requested -> not started yet.
          : ;;
      esac
    fi
  else
    log "jobs API fetch failed (elapsed=${elapsed}s)"
  fi

  # Enforce the start deadline. Reached on EVERY iteration -- including when the
  # fetch failed -- so a persistent API failure can never loop forever.
  if [ "${elapsed}" -ge "${START_TIMEOUT}" ]; then
    if [ "${seen}" -eq 0 ] && [ -z "${JSON}" ]; then
      # Never reached the API and never saw the primary: we cannot tell whether
      # GitHub ran it, so refuse to self-run blindly.
      printf '::error::Fallback could not reach the jobs API within %ss (last fetch failed); not self-running blindly for %s.\n' "${START_TIMEOUT}" "${PRIMARY}" >&2
      echo "FETCH_ERROR"; exit 0
    fi
    if [ "${seen}" -eq 0 ]; then
      # Never once observed. Distinguish "stall" from "matching broke".
      if [ "${EXPECT_FAST}" -eq 1 ]; then
        printf '::error::Fallback never observed primary job %s within %ss. It has no `needs` and should appear immediately -- the job name almost certainly drifted (matrix os/python change). Update the primary name in ci.yml.\n' "${PRIMARY}" "${START_TIMEOUT}" >&2
        echo "RENAME_ERROR"; exit 0
      fi
      DC_STATUS="$(field "${JSON}" 'detect-changes' 'status')"
      if [ "${DC_STATUS}" = "completed" ]; then
        printf '::error::Fallback never observed primary job %s though detect-changes has completed -- likely a renamed matrix leg or a pruned primary. Refusing to self-run blindly.\n' "${PRIMARY}" >&2
        echo "RENAME_ERROR"; exit 0
      fi
      # detect-changes has NOT completed (stuck or still running on GitHub), so
      # the primary was never created. This fallback already confirmed via its
      # own arch-diff that the work is needed, so a stalled detect-changes is
      # just another GitHub scheduling stall -- take over rather than wait
      # forever for a primary that will never appear.
      printf '::warning::Primary %s never appeared and detect-changes has not completed (status=%s) within %ss -- detect-changes itself appears stalled; self-running on this self-hosted runner.\n' "${PRIMARY}" "${DC_STATUS:-absent}" "${START_TIMEOUT}" >&2
      echo "SELFRUN"; exit 0
    fi
    printf '::warning::GitHub did not start primary job %s within %ss -- self-running the tests on this self-hosted runner.\n' "${PRIMARY}" "${START_TIMEOUT}" >&2
    echo "SELFRUN"; exit 0
  fi

  sleep "${POLL}"; elapsed=$((elapsed + POLL))
done
