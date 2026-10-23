package numgo

import (
	"github.com/kaisawind/numgo/internal/nditer"
)

// Matrix products: Matmul, Vdot and Dot. All are pure (new data / value) and
// follow numpy's dispatch rules. Element access goes through At/Set so all
// index arithmetic stays inside nditer (P0-1).
// shortcut: naive triple loops, no blocking or BLAS; upgrade when profiling
// demands it.

// promote1D lifts 1-D operands the numpy matmul way: (K,) on the left becomes
// a (1,K) row, on the right a (K,1) column. Two 1-D operands are an error
// (numpy: use dot/vdot).
func promote1D[T Number](a, b *Array[T]) (*Array[T], *Array[T]) {
	if a.Ndim() == 1 && b.Ndim() == 1 {
		panicMatmul1D()
	}
	if a.Ndim() == 1 {
		a = a.Reshape(1, a.Size())
	}
	if b.Ndim() == 1 {
		b = b.Reshape(b.Size(), 1)
	}
	return a, b
}

// leadShapes broadcasts the operands' leading dimensions, panicking with the
// broadcast message on incompatibility.
func leadShapes[T Number](a, b *Array[T]) []int {
	shape, _, err := nditer.Broadcast(a.Shape()[:a.Ndim()-2], b.Shape()[:b.Ndim()-2])
	if err != nil {
		panicBroadcast(a.Shape(), b.Shape())
	}
	return shape
}

// Pure: Matmul returns the numpy matmul product: the last two dimensions are
// the matrix dimensions (M,K)@(K,N), leading dimensions broadcast. A 1-D left
// operand is treated as a (1,K) row and a 1-D right operand as a (K,1)
// column, that dimension dropped from the result. Two 1-D operands panic
// (use Vdot). Incompatible inner dimensions panic.
func (a *Array[T]) Matmul(b *Array[T]) *Array[T] {
	promA, promB := a.Ndim() == 1, b.Ndim() == 1
	a, b = promote1D(a, b)
	M, K := a.shape[a.Ndim()-2], a.shape[a.Ndim()-1]
	N := b.shape[b.Ndim()-1]
	if K != b.shape[b.Ndim()-2] {
		panicMatmulShape("matmul", a.Shape(), b.Shape())
	}
	lead := leadShapes(a, b)
	av := a.BroadcastTo(append(append([]int(nil), lead...), M, K)...)
	bv := b.BroadcastTo(append(append([]int(nil), lead...), K, N)...)
	outShape := append(append([]int(nil), lead...), M, N)
	if promA {
		outShape = append(outShape[:len(outShape)-2], outShape[len(outShape)-1])
	}
	if promB {
		outShape = outShape[:len(outShape)-1]
	}
	out := Zeros[T](outShape...)
	batchAt := func(arr *Array[T], bc []int, i, j int) T {
		return arr.At(append(append([]int(nil), bc...), i, j)...)
	}
	// batchSet folds the promoted row/col dim back in: the output has fewer
	// dims than the broadcast operands whenever a 1-D operand was promoted.
	batchSet := func(bc []int, i, j int, v T) {
		coords := append(append([]int(nil), bc...), i, j)
		if promA {
			coords = append(coords[:len(coords)-2], coords[len(coords)-1])
		}
		if promB {
			coords = coords[:len(coords)-1]
		}
		out.Set(coords, v)
	}
	odometer(lead, func(bc []int) {
		// Copy the batch coordinate: append may alias the odometer's buffer.
		base := append([]int(nil), bc...)
		row, col := M, N
		if promA {
			row = 1
		}
		for i := 0; i < row; i++ {
			for j := 0; j < col; j++ {
				var sum T
				for k := 0; k < K; k++ {
					sum += batchAt(av, base, i, k) * batchAt(bv, base, k, j)
				}
				batchSet(base, i, j, sum)
			}
		}
	})
	return out
}

// odometer iterates every coordinate combination of shape in C order, calling
// f with a coordinate slice valid only during the call. An empty shape
// iterates once with an empty coordinate (scalar batch).
func odometer(shape []int, f func(coords []int)) {
	if len(shape) == 0 {
		f(nil)
		return
	}
	if nditer.Size(shape) == 0 {
		return
	}
	idx := make([]int, len(shape))
	for {
		f(idx)
		k := len(idx) - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}

// Pure: Vdot returns the conjugate dot product of the flattened operands:
// sum(conj(a[i]) * b[i]). Sizes must match.
func (a *Array[T]) Vdot(b *Array[T]) T {
	if a.Size() != b.Size() {
		panicMatmulShape("vdot", a.Shape(), b.Shape())
	}
	ra, rb := a.Ravel(), b.Ravel()
	var sum T
	for i := 0; i < a.Size(); i++ {
		re, im := parts(ra.At(i))
		sum += scalarFromParts[T](re, -im) * rb.At(i)
	}
	return sum
}

// Pure: Dot dispatches exactly like numpy's dot (PyArray_MatrixProduct_int in
// numpy/_core/src/multiarray/multiarraymodule.c): 1-D·1-D gives the scalar
// product as a one-element array; otherwise the last axis of a contracts with
// the second-to-last axis of b (axis 0 when b is 1-D). The result shape is
// a.shape[:-1] ++ b.shape[:b.ndim-2] ++ b.shape[-1:]: b's leading dimensions
// become MIDDLE dimensions of the result (an outer product of batches, not
// matmul's broadcasting — numpy deprecates this placement for a.ndim>=2 with
// b.ndim>2 and suggests tensordot), and b is shared whenever b is 1-D or 2-D.
// Inner dimension mismatches panic.
func (a *Array[T]) Dot(b *Array[T]) *Array[T] {
	an, bn := a.Ndim(), b.Ndim()
	if an == 1 && bn == 1 {
		return New1DArray([]T{a.Vdot(b)})
	}
	if bn == 1 {
		// Column vector shared across a's leading dims: (...,M,K)@(K,).
		// Matmul with the promoted (K,1) already drops the column dim, so
		// the result shape is exactly numpy's a.shape[:-1].
		return a.Matmul(b)
	}
	if an == 1 {
		// Row vector over b's leading dims: (K,)@(...,K,N). Matmul broadcasts
		// the promoted (1,K) row across b's leads and already returns
		// (...,N) — no reshape needed.
		return a.Matmul(b)
	}
	K := a.shape[an-1]
	if K != b.shape[bn-2] {
		panicMatmulShape("dot", a.Shape(), b.Shape())
	}
	if bn == 2 {
		// Shared b: a's leading dims are the batch dims.
		return a.Matmul(b)
	}
	// Both >=2-D: the C code (multiarraymodule.c:1073-1082) builds the result
	// as a.shape[:-1] ++ b.shape[:bn-2] ++ (N): a's leads, then a's row
	// index, then b's leads, then N. Layout: out[ia, i, jb, j] =
	// sum_k a[ia,i,k] * b[jb,k,j]. Coordinate slices are rebuilt per use:
	// odometer reuses its buffer and append must not alias it.
	// shortcut: naive loops; numpy deprecates this b-leads-in-the-middle
	// placement for a.ndim>=2, b.ndim>2 and points at tensordot.
	M, N := a.shape[an-2], b.shape[bn-1]
	outShape := append(append(append([]int(nil), a.Shape()[:an-1]...), b.Shape()[:bn-2]...), N)
	out := Zeros[T](outShape...)
	ac, bc := an-2, bn-2 // number of leading batch dims of a and b
	acIdx := make([]int, ac+2)
	bcIdx := make([]int, bc+2)
	ocIdx := make([]int, len(outShape))
	odometer(a.Shape()[:ac], func(ia []int) {
		copy(acIdx, ia)
		copy(ocIdx[:ac], ia)
		odometer(b.Shape()[:bc], func(jb []int) {
			copy(bcIdx, jb)
			copy(ocIdx[ac+1:ac+1+bc], jb)
			for i := 0; i < M; i++ {
				acIdx[ac], acIdx[ac+1] = i, 0
				ocIdx[ac] = i // a's row index sits after a's leads
				for j := 0; j < N; j++ {
					var sum T
					for k := 0; k < K; k++ {
						acIdx[ac+1] = k
						bcIdx[bc], bcIdx[bc+1] = k, j
						sum += a.At(append([]int(nil), acIdx...)...) * b.At(append([]int(nil), bcIdx...)...)
					}
					ocIdx[ac+1+bc] = j
					out.Set(append([]int(nil), ocIdx...), sum)
				}
			}
		})
	})
	return out
}

// shapesEqual reports whether two shapes are identical.
func shapesEqual(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}
