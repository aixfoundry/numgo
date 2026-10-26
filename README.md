# numgo

N-dimensional arrays with NumPy semantics for Go.

[![CI](https://github.com/aixfoundry/numgo/actions/workflows/ci.yml/badge.svg)](https://github.com/aixfoundry/numgo/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/Go-1.27%2B-00ADD8)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

[中文文档](README.zh-CN.md)

## Why numgo

- **NumPy semantics, Go ergonomics.** Broadcasting rules, type promotion, NaN/Inf propagation, axis handling, and even error-message wording match NumPy — while the API uses Go idioms: generics for dtype dispatch, explicit conversion, views as first-class values.
- **Zero dependencies.** `go.mod` contains only the Go standard library. LAPACK, FFT, random, and the math kernels are implemented in-tree; no cgo, no BLAS.
- **Reproducible random streams.** `SeedSequence` + MT19937/PCG64 are exact ports: for identical seeds, distributions that don't depend on libm produce bit-identical streams to NumPy; libm-sensitive distributions are ULP-close via Go's stdlib math.
- **Two build modes.** A default pure-Go build, and an opt-in `GOEXPERIMENT=simd` build that accelerates hot paths (with documented last-ULP notes). Results contracts hold in both.

## Install

```
go get github.com/aixfoundry/numgo
```

Requires Go 1.27 or later.

## Quick start

```go
package main

import (
	"os"

	"github.com/aixfoundry/numgo"
)

func main() {
	a := numgo.Arange(0.0, 6.0, 1.0).Reshape(2, 3)    // [[0 1 2], [3 4 5]]
	b := numgo.Ones[float64](1, 3)                     // broadcast add (2,3)+(1,3)
	c := a.Add(b)                                      // pure operation: new data

	m := numgo.Ones[float64](3, 2)
	p := c.Matmul(m)                                   // (2,3) @ (3,2) → (2,2)

	numgo.SaveFile("a.npy", a)                         // .npy write/read
	if v, err := numgo.LoadFile("a.npy"); err == nil {
		arr := v.AsFloat64()
		_ = arr.Sum()                                  // 15
	}
	_ = p
	os.Remove("a.npy")
}
```

## Packages

| Package | Contents |
|---|---|
| `numgo` (root) | `Array[T]` views/constructors, ufuncs, reductions, sorting, manipulation, dot products, einsum, IO (`.npy`/`.npz`/txt), interop |
| `numgo/linalg` | `numpy.linalg`: solve, least squares, SVD/QR/Cholesky/eigen decompositions, norms (pure-Go LAPACK kernels) |
| `numgo/fft` | `numpy.fft`: mixed-radix FFT/inverse/rfft families + fftshift, three `norm` modes |
| `numgo/random` | `numpy.random`: SeedSequence + MT19937/PCG64 + Generator distributions (streams match NumPy: libm-free paths bit-identical, libm-sensitive distributions ULP-close via Go stdlib math) |
| `numgo/polynomial` | `numpy.polynomial`: Polynomial/Chebyshev/Legendre/Laguerre/Hermite/HermiteE |
| `numgo/strings` | `numpy.strings`: vectorized string operations on StrArray (upper/split/replace/…) |
| `numgo/ma` | `numpy.ma`: MaskedArray = data + mask with mask-propagating operations |
| `numgo/emath` | `numpy.emath`: scimath functions that promote to complex outside the real domain |
| `numgo/assert` | `numpy.testing`: AssertAllClose/AssertArrayEqual and friends, for downstream tests |

## Performance

- Faster than NumPy on small-matrix `Solve` (24×) and `Inv` (4.6×), `StandardNormal`, `StandardGamma`, and element-level `At`/`Set` (12×).
- Near parity on einsum, cumsum, and masked-array arithmetic.
- Single-threaded elementwise and reductions currently trail NumPy's vectorized kernels by 3–21×.
- The opt-in `GOEXPERIMENT=simd` build turns Matmul 4.2× faster and makes `SumAll`/`SumAxis0` **faster than NumPy**.

Full methodology and tables: [docs/performance.md](docs/performance.md).

## NumPy parity

numgo tracks NumPy 2.x semantics module by module, verified against golden fixtures generated from NumPy. Some surfaces are intentionally skipped (recarrays, `np.matrix`, legacy `RandomState`), and some behavior deviates by design (explicit `AsType` instead of implicit promotion, panics instead of exceptions).

Per-module list, accuracy claims, and deviations: [docs/numpy-parity.md](docs/numpy-parity.md). Design and contracts: [docs/architecture.md](docs/architecture.md). What's next: [docs/roadmap.md](docs/roadmap.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands, and PR expectations.

## License

Apache-2.0, Copyright 2024-2026 kaisawind. Zero third-party dependencies; the math is based on the pure-Go standard library. This product includes software translated from NumPy (BSD-3-Clause); see [NOTICE](NOTICE) and [LICENSES/](LICENSES/) for details.
