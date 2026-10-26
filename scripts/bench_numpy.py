#!/usr/bin/env python3
"""numpy-side timing for the numgo Phase 5 performance baseline.

Runs the same operations as the *_bench_test.go files (float64), best of 5
loops with time.perf_counter, and prints one table row per op so the output
can be placed side-by-side with the numgo ns/op table in docs/perf-baseline.md.

Usage: python3 scripts/bench_numpy.py
"""

import time

import numpy as np
import numpy.ma as ma

REPEATS = 5


def timeit(fn, number=1):
    """Best-of-REPEATS wall time (ms) of fn() run `number` times per trial."""
    fn()  # warmup
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) * 1e3 / number)
    return best


def ms_to_ratio(numgo_ns, numpy_ms):
    return numgo_ns / 1e6 / numpy_ms


rows = []


def row(op, size, ms):
    rows.append((op, size, ms))
    print(f"{op:<24} {size:>16} {ms:>12.3f}")


print(f"{'op':<24} {'size':>16} {'numpy_ms':>12}")
print("-" * 56)

# ---- root package ----
a = np.ones((1000, 1000))
b = np.ones((1000, 1000))
row("Add contiguous", "1000x1000", timeit(lambda: a + b))
r = np.ones((1, 1000))
row("Add broadcast row", "1000x1000+1x1000", timeit(lambda: a + r))
row("Add strided (A+At)", "1000x1000", timeit(lambda: a + a.T))
row("Sum all", "1000x1000", timeit(lambda: a.sum()))
row("Sum axis0", "1000x1000", timeit(lambda: a.sum(axis=0)))
row("Cumsum last", "1000x1000", timeit(lambda: np.cumsum(a, axis=-1)))
row("Sort last axis", "1000x1000", timeit(lambda: np.sort(a, axis=-1)))
u = np.arange(100_000) * 2654435761 % 65536
u = u.astype(np.float64)
row("Unique", "100k 1-d", timeit(lambda: np.unique(u)))
row("Matmul 200", "200x200x200", timeit(lambda: a[:200, :200] @ b[:200, :200]))
batch = np.arange(16 * 64 * 64, dtype=np.float64).reshape(16, 64, 64) / 13
row("Matmul batch", "16x64x64x64", timeit(lambda: batch @ batch))
row("Einsum ij,jk->ik", "100x100x100", timeit(lambda: np.einsum("ij,jk->ik", a[:100, :100], b[:100, :100])))


def at_set():
    c = a.copy()
    for i in range(0, 1000):  # 1e6 scalar reads/writes would take minutes in
        for j in range(1000):  # numpy; run 1e6 as numgo but scaled window
            c[i, j] = c[i, j] + 1
    return c


# numpy scalar At/Set is interpreted, not vectorized — time 1000x1000 like numgo.
row("At/Set 1e6 scalar", "1000x1000", timeit(at_set))
row("Slice view", "500x500 step2", timeit(lambda: a[:500, ::2]))
row("Reshape", "1000x1000->10000x100", timeit(lambda: a.reshape(10_000, 100)))
row("AsType f64->f32", "1e6", timeit(lambda: np.ones(1_000_000, dtype=np.float32)))

# ---- linalg ----
rng = np.random.default_rng(42)
m100 = rng.random((100, 100))
m50 = rng.random((50, 50))
pd100 = m100 @ m100.T + 100 * np.eye(100)
x = np.ones(100)
row("Solve 100", "100x100", timeit(lambda: np.linalg.solve(m100, x)))
row("Inv 100", "100x100", timeit(lambda: np.linalg.inv(m100)))
row("Svd 100", "100x100", timeit(lambda: np.linalg.svd(m100, full_matrices=True)))
row("Eig 50", "50x50", timeit(lambda: np.linalg.eig(m50)))
row("Cholesky 100", "100x100", timeit(lambda: np.linalg.cholesky(pd100)))

# ---- fft ----
sig = np.arange(65_536, dtype=np.float64) % 97 / 97
row("FFT 65536", "2^16", timeit(lambda: np.fft.fft(sig)))
sig2 = np.arange(40_960, dtype=np.float64) % 97 / 97
row("FFT 40960", "2^13*5", timeit(lambda: np.fft.fft(sig2)))
sig64 = sig[:64 * 64].reshape(64, 64)
row("FFTN 64x64", "64x64", timeit(lambda: np.fft.fftn(sig64)))
row("IRFFT 65536", "2^16", timeit(lambda: np.fft.irfft(np.fft.fft(sig))))

# ---- random ----
g = np.random.default_rng(42)
row("StandardNormal 1M", "1e6", timeit(lambda: g.standard_normal(1_000_000)))
row("Random 1M", "1e6", timeit(lambda: g.random(1_000_000)))
row("Integers 1M", "1e6", timeit(lambda: g.integers(0, 1000, 1_000_000)))
row("StandardGamma 1M", "1e6", timeit(lambda: g.standard_gamma(3.0, 1_000_000)))
row("Weibull 1M", "1e6", timeit(lambda: g.weibull(1.5, 1_000_000)))

# ---- ma ----
mask = np.zeros((1000, 1000), dtype=bool)
mask[::2, :] = True
maa = ma.array(np.ones((1000, 1000)), mask=mask)
mac = ma.array(np.ones((1000, 1000)), mask=mask)
row("ma Add half mask", "1000x1000", timeit(lambda: maa + mac))

print(f"\n# numpy {np.__version__}, best of {REPEATS}")
