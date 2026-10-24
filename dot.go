package numgo

import (
	"github.com/aixfoundry/numgo/internal/nditer"
)

// Matrix products: Matmul, Vdot and Dot. All are pure (new data / value) and
// follow numpy's dispatch rules. The Matmul compute path runs on dense raw
// slices (non-contiguous operands are copied first); all other index
// arithmetic stays inside nditer (P0-1).

// matmulKernel computes c = a@b for dense C-order blocks: a is M×K, b is K×N,
// c is M×N and must be pre-zeroed. 4×8 register blocking over (i,j): each k
// step FMA-updates a 4×8 register panel, so per output element the sum still
// accumulates k in ascending order — float results are bit-identical to the
// naive i-j-k loop. Edges (M%4, N%8) fall back to per-element k-ascending
// dots, same order.
func matmulKernel[T Number](a, b, c []T, M, K, N int) {
	M4, N8 := M-M%4, N-N%8
	for i0 := 0; i0 < M4; i0 += 4 {
		c0 := c[(i0+0)*N:]
		c1 := c[(i0+1)*N:]
		c2 := c[(i0+2)*N:]
		c3 := c[(i0+3)*N:]
		for j0 := 0; j0 < N8; j0 += 8 {
			var r [4][8]T
			for k := 0; k < K; k++ {
				bk := b[k*N+j0 : k*N+j0+8 : k*N+j0+8]
				a0, a1 := a[(i0+0)*K+k], a[(i0+1)*K+k]
				a2, a3 := a[(i0+2)*K+k], a[(i0+3)*K+k]
				r0, r1 := &r[0], &r[1]
				r2, r3 := &r[2], &r[3]
				r0[0] += a0 * bk[0]
				r0[1] += a0 * bk[1]
				r0[2] += a0 * bk[2]
				r0[3] += a0 * bk[3]
				r0[4] += a0 * bk[4]
				r0[5] += a0 * bk[5]
				r0[6] += a0 * bk[6]
				r0[7] += a0 * bk[7]
				r1[0] += a1 * bk[0]
				r1[1] += a1 * bk[1]
				r1[2] += a1 * bk[2]
				r1[3] += a1 * bk[3]
				r1[4] += a1 * bk[4]
				r1[5] += a1 * bk[5]
				r1[6] += a1 * bk[6]
				r1[7] += a1 * bk[7]
				r2[0] += a2 * bk[0]
				r2[1] += a2 * bk[1]
				r2[2] += a2 * bk[2]
				r2[3] += a2 * bk[3]
				r2[4] += a2 * bk[4]
				r2[5] += a2 * bk[5]
				r2[6] += a2 * bk[6]
				r2[7] += a2 * bk[7]
				r3[0] += a3 * bk[0]
				r3[1] += a3 * bk[1]
				r3[2] += a3 * bk[2]
				r3[3] += a3 * bk[3]
				r3[4] += a3 * bk[4]
				r3[5] += a3 * bk[5]
				r3[6] += a3 * bk[6]
				r3[7] += a3 * bk[7]
			}
			b0 := c0[j0 : j0+8 : j0+8]
			b1 := c1[j0 : j0+8 : j0+8]
			b2 := c2[j0 : j0+8 : j0+8]
			b3 := c3[j0 : j0+8 : j0+8]
			for jj := 0; jj < 8; jj++ {
				b0[jj] = r[0][jj]
				b1[jj] = r[1][jj]
				b2[jj] = r[2][jj]
				b3[jj] = r[3][jj]
			}
		}
	}
	// Edge columns/rows outside the 4×8 panels: plain k-ascending dot per
	// element (same accumulation order, bit-identical).
	for i := 0; i < M; i++ {
		for j := N8; j < N; j++ {
			var sum T
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
	for i := M4; i < M; i++ {
		for j := 0; j < N8; j++ {
			var sum T
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
}

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
	denseShape := append(append([]int(nil), lead...), M, N)
	av := a.BroadcastTo(append(append([]int(nil), lead...), M, K)...)
	bv := b.BroadcastTo(append(append([]int(nil), lead...), K, N)...)
	outShape := append([]int(nil), denseShape...)
	if promA {
		outShape = append(outShape[:len(outShape)-2], outShape[len(outShape)-1])
	}
	if promB {
		outShape = outShape[:len(outShape)-1]
	}
	out := Zeros[T](outShape...)
	// Dense raw-slice kernel path. Non-contiguous operands fall back to a
	// Copy (O(n) vs the n³ compute). Promotion folds the broadcast row/col
	// dim back, so the output has fewer dims than the operands whenever a 1-D
	// operand was promoted; those results are computed into a dense (…,M,N)
	// buffer and the promoted dims are folded back via CopyInto.
	if !av.IsContiguous() {
		av = av.Copy()
	}
	if !bv.IsContiguous() {
		bv = bv.Copy()
	}
	dst := out
	if promA || promB {
		dst = Zeros[T](denseShape...)
	}
	aData := av.data[av.offset:]
	bData := bv.data[bv.offset:]
	dData := dst.data
	dstLeads := dst.strides[:len(lead)]
	odometer(lead, func(bc []int) {
		// Broadcast operands repeat planes (stride 0 on broadcast dims), so
		// each batch reads its own plane; the kernel writes the plane fully.
		abase, bbase, obase := 0, 0, 0
		for d, c := range bc {
			abase += c * av.strides[d]
			bbase += c * bv.strides[d]
			obase += c * dstLeads[d]
		}
		matmulKernel(aData[abase:], bData[bbase:], dData[obase:], M, K, N)
	})
	if dst != out {
		// dst has the broadcast (…,M,N) shape, out the promoted shape: fold
		// the broadcast row/col dim back through a same-size view.
		CopyInto(out, dst.Reshape(append([]int(nil), out.shape...)...))
	}
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
	ad, bd := ra.Data()[:ra.Size()], rb.Data()[:rb.Size()]
	var sum T
	switch sa := any(ad).(type) {
	case []float64:
		sb := any(bd).([]float64)
		for i, v := range sa {
			sum += scalarFromParts[T](v*sb[i], 0)
		}
	case []float32:
		sb := any(bd).([]float32)
		for i, v := range sa {
			sum += scalarFromParts[T](float64(v*sb[i]), 0)
		}
	case []complex128:
		sb := any(bd).([]complex128)
		for i, v := range sa {
			w := complex(real(v), -imag(v)) * sb[i]
			sum += scalarFromParts[T](real(w), imag(w))
		}
	case []complex64:
		sb := any(bd).([]complex64)
		for i, v := range sa {
			w := complex(real(v), -imag(v)) * sb[i]
			sum += scalarFromParts[T](float64(real(w)), float64(imag(w)))
		}
	default:
		for i := range ad {
			re, im := parts(ad[i])
			sum += scalarFromParts[T](re, -im) * bd[i]
		}
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
