// Package linalg provides numpy.linalg-semantics dense linear algebra over
// numgo arrays.
//
// Element type is generic over numgo.Number; compute happens in float64
// internally (via numgo.AsType) and results convert back to T. Complex element
// types are accepted for forward compatibility but panic at run time:
// complex kernels (Z-prefix) are not wired up yet
// (numgo/linalg: complex dtypes not supported yet).
//
// Real general eigenproblems (Eig/Eigvals) intrinsically produce complex
// values; they always return *numgo.Array[complex128] regardless of T —
// matching numpy's behaviour of upcasting to complex for non-symmetric input.
//
// All stacked-matrix operations loop over the leading batch dimensions
// (numpy gufunc semantics for the matrix axes). Batched BLAS is deferred.
//
// Backend: own pure-Go kernels in internal/lapack (no external LAPACK):
//
//	Solve/Inv/Det/Slogdet   LU (partial pivoting) + LuSolve/LuInvert
//	Cholesky                Potrf (lower)
//	QR (reduced/complete)   Geqrf + Orgqr (Householder)
//	SVD                     one-sided Jacobi (Hestenes)
//	Eig/Eigvals             Hessenberg + shifted QR + inverse iteration
//	Eigh/Eigvalsh           cyclic Jacobi
//	Lstsq/Pinv/Cond/Rank    via the Jacobi SVD
//	Norm                    direct reduction loops (no kernel dependency)
//
// Kernel buffers are C-order (row-major) with the leading dimension as the
// row stride, so numgo's C-order rows pass straight through with no
// transposition.
package linalg

import (
	"reflect"

	"github.com/aixfoundry/numgo"
)

// isComplex reports whether T is a complex element type. Complex support is
// deferred to the Z-prefix LAPACK phase.
func isComplex[T numgo.Number]() bool {
	var v T
	return reflect.ValueOf(v).Kind() == reflect.Complex64 ||
		reflect.ValueOf(v).Kind() == reflect.Complex128
}

// f64 converts a to a C-order float64 array, panicking on complex element
// types.
func f64[T numgo.Number](a *numgo.Array[T]) *numgo.Array[float64] {
	if isComplex[T]() {
		panicComplex()
	}
	return numgo.AsType[T, float64](a)
}

// wrap builds a C-order T array from float64 data shaped shape.
func wrap[T numgo.Number](data []float64, shape []int) *numgo.Array[T] {
	f := numgo.FromSlice(data, shape...)
	return numgo.AsType[float64, T](f)
}

// last2 extracts the two trailing dimensions (m, n) of shape.
func last2(shape []int) (m, n int) {
	return shape[len(shape)-2], shape[len(shape)-1]
}

// batchShape returns the leading dimensions of shape.
func batchShape(shape []int) []int {
	return shape[:len(shape)-2]
}

// eachBatch iterates the leading dimensions of shape in C order, calling f
// with the batch coordinate; coords is reused between calls. Shapes with
// fewer than two dimensions iterate once with a nil coordinate.
func eachBatch(shape []int, f func(batch []int)) {
	if len(shape) < 2 {
		f(nil)
		return
	}
	lead := shape[:len(shape)-2]
	if len(lead) == 0 {
		f(nil)
		return
	}
	batch := make([]int, len(lead))
	var rec func(k int)
	rec = func(k int) {
		if k == len(lead) {
			f(batch)
			return
		}
		for i := 0; i < lead[k]; i++ {
			batch[k] = i
			rec(k + 1)
		}
	}
	rec(0)
}

// linearIndex converts C-order coords into a flat offset.
func linearIndex(shape, coords []int) int {
	idx := 0
	for i, c := range coords {
		idx = idx*shape[i] + c
	}
	return idx
}

// scalarOf converts a float64 to any real Number element type.
func scalarOf[T numgo.Number](v float64) T {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(float32(v)).(T)
	case float64:
		return any(v).(T)
	default:
		panicComplex()
		return zero
	}
}

// aMatrixAt copies the (m,n) matrix at batch coords of a (row-major).
func aMatrixAt(a *numgo.Array[float64], batch []int, m, n int) []float64 {
	mat := make([]float64, m*n)
	coords := append(append([]int(nil), batch...), 0, 0)
	k := len(batch)
	for i := 0; i < m; i++ {
		coords[k] = i
		for j := 0; j < n; j++ {
			coords[k+1] = j
			mat[i*n+j] = a.At(coords...)
		}
	}
	return mat
}

// setVecAt writes a length-m vector at batch coords of out.
func setVecAt[T numgo.Number](out *numgo.Array[T], batch []int, v []float64) {
	coords := append(append([]int(nil), batch...), 0)
	k := len(batch)
	for i := 0; i < len(v); i++ {
		coords[k] = i
		out.Set(coords, scalarOf[T](v[i]))
	}
}

// setMatAt writes an (m,n) row-major block at batch coords of out.
func setMatAt[T numgo.Number](out *numgo.Array[T], batch []int, m, n int, mat []float64) {
	coords := append(append([]int(nil), batch...), 0, 0)
	k := len(batch)
	for i := 0; i < m; i++ {
		coords[k] = i
		for j := 0; j < n; j++ {
			coords[k+1] = j
			out.Set(coords, scalarOf[T](mat[i*n+j]))
		}
	}
}

// broadcastable reports whether right-aligned broadcast of a's dims into
// lead dims succeeds (numpy rules: equal or one).
func broadcastable(a, lead []int) bool {
	_, ok := broadcastShapes(a, lead)
	return ok
}

// broadcastCoord aligns a's batch dims with b's lead dims (right-aligned
// broadcast): dims of size 1 in a repeat.
func broadcastCoord(aShape, lead, bbatch []int) []int {
	abatch := make([]int, len(aShape))
	off := len(lead) - len(aShape)
	for i := range aShape {
		c := 0
		if aShape[i] != 1 {
			c = bbatch[off+i]
		}
		abatch[i] = c
	}
	return abatch
}
