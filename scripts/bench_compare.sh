#!/bin/sh
# Reproduce the numgo-vs-numpy comparison in docs/perf-baseline.md.
#
# Go side:   go test benchmarks over root/linalg/fft/ma/random, median of 3.
# numpy side: scripts/bench_numpy.py, best of 5 (op names match the doc table).
# Ratio = numgo_ns / 1e6 / numpy_ms  (>1 means numgo slower).
set -e
cd "$(dirname "$0")/.."

echo "== numgo (median of -count 3) =="
go test -run='^$' -bench=. -benchtime=1s -count=3 . ./linalg ./fft ./ma ./random \
  | awk '
    /^Benchmark/ {
      n = $1; sub(/-[0-9]+$/, "", n)
      cnt[n]++; v[n, cnt[n]] = $3 + 0
      if ($5 != "") alloc[n] = $7
    }
    END {
      for (n in cnt) {
        m = cnt[n]
        for (i = 2; i <= m; i++) {          # insertion sort, m is tiny
          x = v[n, i]; j = i - 1
          while (j > 0 && v[n, j] > x) { v[n, j+1] = v[n, j]; j-- }
          v[n, j+1] = x
        }
        med = (m % 2) ? v[n, (m+1)/2] : (v[n, m/2] + v[n, m/2+1]) / 2
        printf "%-28s %14.0f ns/op   allocs/op=%s\n", n, med, alloc[n]
      }
    }' | sort

echo
echo "== numpy (best of 5) =="
python3 scripts/bench_numpy.py
