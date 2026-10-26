# Architecture

This document describes how numgo is built: the module layout, the core `Array[T]` type, the iteration engine, API semantics, the bit-exactness contracts, and the zero-dependency policy.

## Module layout

The root package mirrors the NumPy top-level namespace; heavier subdomains live in their own packages. Dependency direction is strictly one-way: subpackages depend on the root package, which depends on `internal/` — no cycles.

```
numgo/
├── array.go, index.go, create.go, convert.go   # Array[T], views, constructors
├── ufunc.go, math.go, reduce.go, nan.go        # ufuncs, reductions, NaN family
├── sort.go, manip.go, dot.go, einsum*.go       # sorting, shape ops, matmul/einsum
├── io*.go, stridetricks.go, setops.go, pad.go  # .npy/.npz/txt, stride tricks, set ops
├── errors.go                                   # centralized panic messages
├── internal/
│   ├── nditer/      # the only strided-index algebra in the library
│   ├── lapack/      # LU, Cholesky, QR, Jacobi-SVD, Hessenberg+QR eigenvalues
│   ├── simdvec/     # opt-in SIMD kernels (GOEXPERIMENT=simd), no-op by default
│   └── fixtures/    # golden-fixture helpers (parity tests against NumPy)
├── linalg/ fft/ random/ polynomial/ strings/ ma/ emath/ assert/
└── scripts/ docs/ testdata/
```

The execution engine is unique: all cross-stride indexing algebra exists in exactly one place, `internal/nditer`. Operation code never hand-writes index formulas such as `i*c+j`; every ufunc and reduction goes through the engine. This rule came out of the historical bug record — nearly every stride bug in the previous 2D implementation was a hand-rolled index formula using the wrong dimension.

## Array[T]: the view/stride model

```go
type Number interface {
    ~float32 | ~float64 | ~complex64 | ~complex128
}

type Array[T Number] struct {
    shape   []int
    strides []int // in elements, not bytes
    offset  int
    data    []T
}
```

- **Views are first-class.** `Slice`, `Transpose`, `Broadcast`, and `Reshape` (when a view is possible) share the underlying `data` and are expressed as `offset + strides`; `Copy()` is what produces a new backing array. Constructors always copy their inputs. Contiguity is derived from the strides (there is no flag field).
- **One index formula.** `data[offset + Σ(iₖ·stridesₖ)]` lives in nditer and nowhere else.
- **Read-only internals.** `Shape()` returns a copy; `shape`/`strides` are not mutable from outside.
- **The zero value is unusable.** `var a *Array[T]` is nil; constructors are the only entry point, and a nil receiver panics rather than misbehaving silently.
- **Generics are the dtype dispatch.** NumPy expands a dtype × operation matrix with code generation at build time; `Array[T]` reaches the same effect by compile-time monomorphization. There is no runtime dtype container. Operations that need an ordering use `cmp.Ordered` as a separate constraint.

## Semantics categories

Every API falls into exactly one of three categories, stated in its documentation:

1. **View** — shares data with the source array (`Slice`, `Transpose`, `Broadcast`, `Reshape`). Always documented as such.
2. **Pure** — returns new data. This is the default for all ufuncs and reductions (`Add`, `Sum`, …).
3. **In-place** — only via an explicit `Into` suffix (`AddInto(dst, b)`). There is no implicit mutation; methods that need to reorder input copy first.

Errors follow NumPy wording and Go conventions: invalid shapes, axes, or indices panic with messages centralized in `errors.go` (`operands could not be broadcast together with shapes (2,3) (1,2)`, `Singular matrix`, `index (3,1) is out of bounds for axis 0 with size 2`). IEEE special cases do not panic: division by zero produces Inf, empty reductions follow the NumPy convention.

## The nditer engine

`internal/nditer` is aligned in responsibility with NumPy's nditer: multi-array broadcast alignment, stride stepping, and contiguous fast paths.

- `Zip` drives elementwise operations: destination and sources are broadcast-aligned and iterated together; the callback receives contiguous blocks (the engine flattens transposed/strided layouts; contiguous input is passed through zero-copy).
- `Reduce` drives axis reductions; `keepdims` semantics are handled above the engine.
- Fast paths: row-broadcast patterns (stride-0 tail dimension), strided inputs, and C-contiguous axis-0 reductions are recognized and turned into linear scans, while the general odometer path remains the fallback. Callback block-length contracts are identical across paths.

## Bit-exactness contracts

Several internal paths have more than one implementation (a general path and a folded/fast path). The library maintains these contracts:

- **Einsum fold ≡ general path.** Einsum with ≥2 operands folds pairwise into dense matmuls. The K-flattening order matches the general path's summation order, so results are bit-identical; a randomized cross-check test asserts this. Unsupported patterns fall back to the general transpose-broadcast-multiply-sum path.
- **Matmul accumulation order.** The matmul kernel keeps k-ascending accumulation with a fixed register blocking, bit-identical to the naive triple loop under the default build.
- **Default vs `GOEXPERIMENT=simd` builds.** SIMD builds use FMA accumulation and lane-ordered reductions (pairwise-like merging). On general float64 input the last ULP may therefore differ from the default build — the same class of variation NumPy exhibits across BLAS backends. Each contract above holds within a build: einsum-fold ≡ general-path bit-identity is tested in both builds, and the SIMD axis-0 sum is bit-identical across builds (both are covered by tests). float32 matmul in SIMD mode is on the same last-ULP exception list.
- **SIMD coverage is deliberately narrow.** SIMD kernels exist only for contiguous float64 add, matmul (f64/f32), einsum-via-matmul, and pure sums (`SumAll`/`SumAxis0`). Mean/Var/Std/Nan\*/Prod/Max/Min stay on the generic path — Var's `(Σx² − (Σx)²/n)` form requires both sums to be computed in the same order to avoid catastrophic cancellation.
- **FFT caching does not change results.** Twiddle tables and Bluestein kernels are cached by (n, direction) with identical expressions, so cached and freshly computed paths are bit-identical.

## Zero-dependency policy

`go.mod` contains only the Go standard library — no third-party modules, no cgo. Concretely:

- **linalg**: LAPACK routines (LU, Cholesky, QR, Jacobi-SVD, Hessenberg + QR eigenvalues) implemented in `internal/lapack`.
- **fft**: mixed-radix Cooley–Tukey plus Bluestein for prime factors, implemented in the `fft` package.
- **random**: SeedSequence, MT19937, PCG64, and the Generator distribution layer, all implemented in the `random` package.
- **math**: the Go standard library. No glibc/libm-derived kernels are vendored; see [numpy-parity.md](numpy-parity.md) for what this means for random-stream reproducibility.
- **SIMD**: the opt-in fast paths use the Go 1.27 experimental `simd` package behind `GOEXPERIMENT=simd` (still stdlib). Default builds compile a plain-Go twin with identical behavior; build tags select between them.

Rationale: NumPy's Python layer is an API-and-validation shell over C kernels, so porting NumPy semantics does not require porting its build system — and a Go library in this space is most useful when `go get` is all it takes.
