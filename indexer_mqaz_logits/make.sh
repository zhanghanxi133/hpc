#!/usr/bin/env bash
set -eo pipefail

hpckit_root="${HPCKIT_ROOT:-/fs_real_a800/PAC2026/HPCKit26/HPCKit/latest}"
source "${hpckit_root}/setvars.sh"
set -u

readonly KUPL_INCLUDE="${KUPL_INCLUDE:-${hpckit_root}/kupl/bisheng/release/include}"
readonly KUPL_LIB="${KUPL_LIB:-${hpckit_root}/kupl/bisheng/release/lib}"
readonly OUTPUT="${OUTPUT:-main}"

test -f "${KUPL_INCLUDE}/kupl.h"
test -e "${KUPL_LIB}/libkupl.so"
readelf -Ws "${KUPL_LIB}/libkupl.so" | awk '
  $8 == "kupl_parallel_for" { parallel_for = 1 }
  END { exit !parallel_for }
'

clang++ -O3 -stdlib=libc++ \
  -march=armv9-a+sme+bf16+sve2 \
  -flto \
  -fopenmp -rtlib=compiler-rt -unwindlib=libunwind \
  -DINDEXER_KUPL -I"${KUPL_INCLUDE}" "$@" \
  main.cpp -L"${KUPL_LIB}" -Wl,--disable-new-dtags \
  -Wl,-rpath,"${KUPL_LIB}" -lkupl -lnuma -o "${OUTPUT}"

clang++ --version | head -2
sha256sum indexer_mqa_logits.h "${OUTPUT}"
readelf -d "${OUTPUT}" | grep -E 'RPATH|RUNPATH|NEEDED'
readelf -Ws "${OUTPUT}" | grep -E 'kupl_parallel_for' | sort -u

resolved_kupl="$({ ldd "${OUTPUT}" || true; } | awk '
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
