# Performance

This page summarizes numgo's performance relative to NumPy and the opt-in SIMD build mode. Everything below is reproducible from this repository.

## Methodology

- **Reproduce:** `sh scripts/bench_compare.sh` — runs the Go benchmarks (`go test -bench`, `-count 3`, median) and the NumPy side (`scripts/bench_numpy.py`, best of 5); ratio = numgo ms / numpy ms, so >1 means numgo is slower.
- **Hardware/software of the recorded runs:** AMD Ryzen 9 7950X, go1.27 linux/amd64, zero-dependency pure-Go kernel; NumPy 2.5.1 with OpenBLAS 0.3.33. NumPy matmul/linalg run multi-threaded BLAS; numgo runs single-threaded.
- Go-side benchmarks are stable (<5% run-to-run except where noted). NumPy linalg numbers on this machine fluctuate with thread-pool behavior (e.g. SVD has been observed between 1.2 ms and 53 ms), so treat the linalg ratios as indicative.

## Current numgo vs NumPy (single-threaded pure Go vs NumPy 2.5.1/OpenBLAS)

| Operation | Size | numgo | numpy | Ratio |
|---|---|---:|---:|---:|
| Add contiguous | 1000×1000 | 2.19 ms | 0.204 | **10.7×** |
| Add broadcast row | 1000×1000 + 1×1000 | 1.91 ms | 0.258 | **7.4×** |
| Add strided (A + Aᵀ) | 1000×1000 | 1.84 ms | 0.692 | 2.7× |
| Sum (full reduction) | 1000×1000 | 2.28 ms | 0.107 | **21.3×** |
| Sum axis=0 | 1000×1000 | 1.35 ms | 0.125 | **10.8×** |
| CumSum (last axis) | 1000×1000 | 2.82 ms | 2.302 | 1.2× |
| Sort (last axis) | 1000×1000 | 2.78 ms | 0.324 | **8.6×** |
| Unique | 100k 1-d | 29.3 ms | 2.882 | **10.2×** |
| Matmul | 200×200×200 | 1.91 ms | 0.092 | **20.8×** |
| Matmul batched | 16×64×64×64 | 1.02 ms | 0.116 | **8.8×** |
| Einsum ij,jk→ik | 100×100×100 | 0.29 ms | 0.168 | 1.7× |
| At/Set per element | 1e6 | 9.38 ms | 114.0 | 0.082× (**12× faster**) |
| Slice view / Reshape | — | ~50 ns | ~0 | same order |
| AsType f64→f32 | 1e6 | 0.56 ms | 0.094 | **6.0×** |
| Solve | 100×100 | 0.24 ms | 5.730 | 0.042× (**24× faster**) |
| Inv | 100×100 | 1.98 ms | 9.114 | 0.22× (**4.6× faster**) |
| SVD | 100×100 | 35.8 ms | 1.214 | **29.5×** (high variance, see methodology) |
| Eig | 50×50 | 10.9 ms | 0.583 | **18.6×** |
| Cholesky | 100×100 | 0.22 ms | 0.030 | 7.3× |
| FFT | 2^16 | 5.18 ms | 0.878 | **5.9×** |
| FFT mixed radix | 2^13·5 | 2.91 ms | 0.449 | **6.5×** |
| FFTN | 64×64 | 0.32 ms | 0.029 | **11.0×** |
| IRFFT | 2^16 | 56.7 ms | 3.741 | **15.2×** |
| StandardNormal | 1e6 | 4.53 ms | 6.563 | 0.69× (faster) |
| Random | 1e6 | 2.72 ms | 1.545 | 1.8× |
| Integers | 1e6 | 4.17 ms | 1.649 | 2.5× |
| StandardGamma | 1e6 | 11.1 ms | 12.093 | **0.92× (parity)** |
| Weibull | 1e6 | 41.0 ms | 11.037 | **3.7×** |
| ma.Add half-masked | 1000×1000 | 2.59 ms | 0.888 | 2.9× |

Reading the table:

- **numgo is faster** on small-matrix Solve/Inv (OpenBLAS thread-pool startup dominates there), ziggurat StandardNormal, StandardGamma, and At/Set (compiled indexing beats the interpreter by an order of magnitude).
- **Near parity** on einsum, cumsum, and masked-array arithmetic.
- **Elementwise and reductions are 3–21× behind** single-threaded NumPy. These are pure-Go loops without SIMD versus NumPy's vectorized C kernels — a structural gap on this workload, which is exactly what the opt-in SIMD build addresses below.

## Opt-in SIMD build (`GOEXPERIMENT=simd`)

Go 1.27's experimental `simd` package is wired into selected hot paths behind `GOEXPERIMENT=simd`, with build-tagged plain-Go twins (`internal/simdvec`) keeping default builds unchanged. Measured on the same machine (`-count 2`):

| Benchmark | Default build | GOEXPERIMENT=simd |
|---|---:|---:|
| Add contiguous 1000×1000 | ~1.1 ms | ~1.0 ms |
| Matmul 200³ | ~1.9 ms | ~0.45 ms (≈4.2×) |
| Matmul batched 16×64×64×64 | ~1.0 ms | ~0.26 ms (≈4×) |
| Einsum 100×100×100 | ~0.29 ms | ~0.12 ms (≈2.4×) |
| Sum full reduction 1000×1000 | ~2.24 ms | **~0.072 ms (beats NumPy's 0.107 ms)** |
| Sum axis=0 1000×1000 | ~1.35 ms | **~0.088 ms (beats NumPy's 0.125 ms)** |
| Matmul 200³ float32 | — | ~0.51 ms (3.4× over scalar) |
| Matmul batched float32 | — | ~0.129 ms (7.4× over scalar) |

Notes:

- SIMD reduction kernels (lane accumulation with ordered merging) turn SumAll/SumAxis0 from closure-callback-bound to memory-bandwidth-bound, overtaking NumPy.
- float32 matmul gains are real but remain behind BLAS single-precision (which enjoys dual-issue FMA throughput).
- FMA/lane-ordered accumulation can differ from the default build in the last ULP on general inputs — same class of variation NumPy shows across BLAS backends. The library's internal bit-identity contracts (einsum fold ≡ general path, cross-build axis-0 sum) hold in both builds; see [architecture.md](architecture.md).

## FFT SIMD: measured, no gain, rolled back

A full SIMD implementation of the FFT real/imaginary-split paths was built and verified correct, but measured no meaningful gain (FFT 2^16 ≈1.05×, IRFFT ≈1.0×, FFTN 64×64 8% slower). Profiling attributed this to the scalar small-radix leaves (n≤32), which account for 44–73% of runtime, with split/interleave round-trips consuming the rest of the benefit. The code was rolled back and the measurement retained as the record; see the [roadmap](roadmap.md) for the reopening conditions.
