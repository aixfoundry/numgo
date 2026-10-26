# NumPy parity

numgo tracks NumPy 2.x semantics. This page records what is ported, what is intentionally skipped, and what accuracy guarantees hold.

The porting approach: NumPy's Python layer is an API protocol and validation shell; the algorithms live in C (plus Cython for `random`). Go replication copies the **semantics**, not the build system. Where NumPy uses code generation to expand its dtype × operation matrix, numgo uses generics and compile-time monomorphization. The root package corresponds to NumPy's top-level namespace; subpackages mirror NumPy submodules with the same names.

## What is ported

| NumPy | numgo | Coverage |
|---|---|---|
| top level (`numpy.*`) | `numgo` (root) | `Array[T]` views/strides, constructors (incl. `Geomspace`/`Logspace`/`Empty`), ufuncs (four arithmetic plus `Maximum`/`Minimum`/`Fmax`/`Fmin`, `Mod`/`Fmod`/`FloorDivide`, `Pow`), comparisons returning `[]bool` (`Greater`… + `Where`/`NonZero`/`ArgWhere`), NaN predicates (`IsNaN`/`IsInf`/`IsFinite`/`IsClose`), reductions (axis + keepdims), sorting and `SearchSorted`/`Digitize`, shape manipulation, fancy/boolean indexing, `Dot`/`Matmul`/`Einsum`/`TensorDot`/`Convolve`/`Correlate`, `Diff`, IO (`.npy`/`.npz`/txt), stride tricks, set operations (`Union1d`/`Intersect1d`/`SetDiff1d`/`SetXor1d`), padding, window functions (`Hanning`/`Hamming`/`Bartlett`/`Blackman`/`Kaiser`), NaN family, statistics, histograms, type checks |
| `numpy.linalg` | `numgo/linalg` | `Solve`/`Inv`/`Det`/`Slogdet`/`Cholesky`/`QR`/`Lstsq`/`SVD`/`Eig`/`Eigh`/`Norm`/`Cond`/`MatrixRank`/`MultiDot`/`TensorSolve`/`TensorInv`/`MatrixPower`, batched matrix stacks, named result types |
| `numpy.fft` | `numgo/fft` | `FFT`/`IFFT`/`RFFT`/`HFFT` families + nD variants, `FFTFreq`/`FFTShift`, three `norm` modes |
| `numpy.random` | `numgo/random` | `SeedSequence` (entropy pooling + spawn), MT19937 and PCG64 bit generators, `Generator` with the common distribution set (`Normal`, `StandardNormal`, `StandardGamma`, `Uniform`, `Integers`, `Choice`, `Shuffle`, `Permutation`, `Permuted`, `Bytes`, Beta, Binomial, Exponential, Gamma, Poisson, Chisquare, Weibull, …) |
| `numpy.polynomial` | `numgo/polynomial` | `Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE` — arithmetic, `Der`/`Int`/`Roots`/`FromRoots`/`Fit`/`Vander`/`Companion`/`Convert` |
| `numpy.strings` | `numgo/strings` | vectorized string operations on `StrArray` (`Upper`/`Split`/`Replace`/…) |
| `numpy.ma` | `numgo/ma` | `MaskedArray` = data + mask, mask-propagating arithmetic |
| `numpy.emath` | `numgo/emath` | scimath functions that promote to complex outside the real domain |
| `numpy.testing` | `numgo/assert` | `AssertAllClose`, `AssertArrayEqual`, and the rest of the assert family, `*testing.T` style |

## Accuracy and verification

Parity is enforced by golden fixtures generated from NumPy (`scripts/gen_fixtures.py`), covering representative functions of every submodule; `.npy` files round-trip with NumPy bit-exactly. The stronger claims, stated precisely:

- **Bit-exact random streams (libm-free paths).** For identical seeds, distributions whose algorithms do not call libm transcendentals produce bit-identical streams to NumPy on linux/amd64 and windows/amd64 — the fixture platforms. On arm64 the Go compiler fuses `x*y+z` into a single FMA (the Go spec permits cross-statement fusion, and the C toolchains building NumPy there fuse the same expressions), so even libm-free draws drift a bounded number of ULP against the amd64 fixtures; the bit-generator state machines (MT19937, PCG64) and SeedSequence are exact ports on every platform.
- **ULP-close random streams (libm-sensitive paths).** Distributions that depend on `exp`/`log`/`pow`/`erf`-class functions use the Go standard library math instead of libm. These match NumPy to within a few ULP, not bit-for-bit. This is a deliberate trade: no vendored libm code, at the cost of last-bit differences in those distributions. The golden fixtures were generated from NumPy built against glibc/Linux; the bit-exact guarantee for a given distribution therefore applies when comparing against a NumPy linked against the same platform libm — elsewhere draws are only ULP-close. The parity tests enforce bit equality on amd64 and accept bounded drift (≤64 double-ULP / ≤32 float32-ULP) on other architectures.
- **linalg** is verified against LAPACK routine fixtures (~40 cases from the reference implementations numgo's internal LAPACK is derived from).
- **fft** transformation accuracy is verified against NumPy to 1e-12.

## Documented deviations

These are deliberate, not oversights:

- **Explicit type conversion instead of implicit promotion.** Go's static typing replaces NumPy's runtime dtype promotion: use `AsType[To, From]`. There is no runtime dtype container and no implicit upcasting.
- **Panics instead of exceptions.** Invalid shapes/axes/indices are programming errors and panic with NumPy-worded messages; IEEE special cases (divide-by-zero → Inf, empty reductions) do not raise.
- **No `np.random` global singleton.** Generators are explicit values; there is no process-global default stream.
- **`longdouble`/`clongdouble` are downgraded** to `float64`/`complex128` — Go has no native 80/128-bit floating point.
- **Errors keep NumPy wording**, so NumPy experience (and NumPy-derived tests) transfers directly.

## Intentionally skipped

| NumPy surface | Reason |
|---|---|
| `rec` / structured dtypes (`void`) | Go's natural mapping is `[]struct`; numgo prefers `Array[T]` ↔ `[]struct` conversion over a void dtype. May be reconsidered only on real demand. |
| `matrixlib` (`np.matrix`) | deprecated upstream; Go's fixed-dimension `Array[T]` fills the role |
| `char` (deprecated shim) | superseded by `strings` |
| `f2py` | Fortran→Python toolchain, meaningless in Go |
| `doc` module | documentation only |
| array API compatibility layer | Python-ecosystem protocol |
| legacy `RandomState` | Python heritage; explicit `Generator` injection is the Go idiom |
| implicit dtype promotion | static languages do runtime-free dispatch; see deviations above |
| `float16` | needs a software implementation (no hardware conversion instruction on all targets); planned, see the [roadmap](roadmap.md) |
| `longdouble`/`clongdouble` | no native Go equivalent; downgraded and documented |
| integer dtypes (`int8…int64`, `uint*`, and integer-backed `datetime64`) | Go generics cannot vary a method's return type per element type: numpy's `mean(int)->float64` and `true_divide(int)->float64` semantics would silently truncate as `Array[T].Mean() *Array[T]`. Revisit via a parallel `IntegerArray[T Integer]` on the dtype-neutral `internal/nditer` layer if real demand appears (unlocks bitwise ufuncs, integer `.npy` round-trip, datetime64). Until then, integer `.npy` files are rejected at load and `numgo/random` keeps its private `arrayI` wrapper. |
