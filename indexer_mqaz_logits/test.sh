#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false

# KUPL's pthread backend discovers executors from the caller's affinity.
# Keep the full taskset mask visible until KUPL initializes its thread pool.
unset OMP_PROC_BIND
unset OMP_PLACES
export KUPL_EXECUTOR_BACKEND=pthread
export KUPL_EXECUTOR_COUNT=32
export KUPL_KERNEL_CONCURRENCY=32

readonly HPCKIT_ROOT="${HPCKIT_ROOT:-/fs_real_a800/PAC2026/HPCKit26/HPCKit/latest}"
readonly KUPL_LIB="${KUPL_LIB:-${HPCKIT_ROOT}/kupl/bisheng/release/lib}"
readonly INDEXER_BIN="${INDEXER_BIN:-${TEST_BIN:-./main}}"

test -x "${INDEXER_BIN}"
test -e "${KUPL_LIB}/libkupl.so"
export LD_LIBRARY_PATH="${KUPL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

resolved_kupl="$({ ldd "${INDEXER_BIN}" || true; } | awk '
  $1 ~ /^libkupl\.so/ { print $3; exit }
')"
case "${resolved_kupl}" in
  "${KUPL_LIB}"/libkupl.so*) ;;
  *)
    echo "ERROR: wrong libkupl resolved: ${resolved_kupl:-<not found>}" >&2
    echo "EXPECTED: ${KUPL_LIB}/libkupl.so" >&2
    exit 1
    ;;
esac

exec numactl --membind=0 --cpunodebind=0 \
  taskset -c 0-31 "${INDEXER_BIN}"
