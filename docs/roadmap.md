# Roadmap

numgo implements NumPy's numerical capabilities with Go idioms. This page states what exists today and what is planned.

## Status

The core is complete and CI-tested in two build modes (default and `GOEXPERIMENT=simd`):

- **N-dimensional core** — `Array[T]` with views/strides, broadcasting, basic and fancy indexing, ufuncs, reductions with `axis`/`keepdims`, sorting, shape manipulation, einsum/matmul.
- **linalg** — direct methods and decompositions (`Solve`, `Inv`, `SVD`, `Eig`/`Eigh`, `Cholesky`, `QR`, …) on a pure-Go LAPACK kernel, including batched matrix stacks.
- **fft** — mixed-radix FFT and the real/Hermitian families, three `norm` modes.
- **random** — `SeedSequence`, MT19937/PCG64, and the `Generator` distribution layer with NumPy-compatible stream semantics (see [numpy-parity.md](numpy-parity.md)).
- **lib layer** — IO (`.npy`/`.npz`/txt), stride tricks, set operations, padding, NaN family, statistics, histograms.
- **polynomial, strings, ma, emath, assert** — the remaining NumPy submodules numgo covers.

See [numpy-parity.md](numpy-parity.md) for the per-module list and documented deviations.

## Planned

- **float16** — software implementation (no hardware conversion on all targets); the core is dtype-generic, so this is additive.
- **More SIMD coverage** — the `GOEXPERIMENT=simd` opt-in build currently accelerates contiguous float64 add, matmul (f64/f32), einsum, and pure sums. Widening coverage follows the Go `simd` package as it stabilizes.
- **FFT SIMD** — attempted and measured: the vectorized implementation was correct but showed no real gain (small-radix scalar leaves dominate the runtime, and split/interleave round-trips ate the combination-layer benefit). It was rolled back. Revisiting requires vectorized small-radix leaf kernels or shuffle/permute-class instructions in the Go `simd` package.
- **Structured-array conversion** — `Array[T]` ↔ `[]struct` helpers, if demand materializes (see non-goals for why structured dtypes themselves are out).

## Non-goals

- **`np.matrix` and legacy shims** — deprecated upstream; Go's fixed-shape `Array[T]` covers the use case.
- **Structured dtypes (`void`), recarrays** — Go's `[]struct` is the idiomatic representation.
- **Runtime dtype containers / implicit type promotion** — generics are the dtype dispatch; conversion is explicit via `AsType`.
- **`longdouble`/`clongdouble`** — Go has no native 80/128-bit floats; those dtypes are downgraded and documented.
- **Third-party math/BLAS dependencies** — the library stays stdlib-only (see [architecture.md](architecture.md)); LAPACK and FFT are implemented in-tree.
- **Global random state** — no `np.random`-style process-global singleton; generators are explicit values.
- **f2py, the array API compatibility layer, and Python toolchain surfaces** — Python-ecosystem specific.
