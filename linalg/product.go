package linalg

import (
	"github.com/aixfoundry/numgo"
)

// Tensor products, matrix powers and small structural helpers: MultiDot,
// MatrixPower, TensorSolve, TensorInv, Trace, Diagonal, Cross, Outer, Vecdot,
// Kron. Batch axes loop in C order; element access goes through At/Set so no
// index algebra appears here (P0-1).

// MultiDot multiplies two or more matrices with numpy's optimal
// parenthesization (matrix-chain DP over the 2-D cores, matching numpy's
// cost heuristic). Operands are 2-D; a 1-D first or last operand participates
// as a vector like numpy multi_dot allows; interior operands must be 2-D.
func MultiDot[T numgo.Number](as ...*numgo.Array[T]) *numgo.Array[T] {
	if len(as) < 2 {
		panicDims("multi_dot", "Expecting at least two arrays.")
	}
	for i := 1; i < len(as)-1; i++ {
		if as[i].Ndim() != 2 {
			panicDims("multi_dot", "1-dimensional array given. Array must be two-dimensional")
		}
	}
	if len(as) == 2 {
		return as[0].Matmul(as[1])
	}
	order := chainOrder(dims2(as))
	return chainMultiply(as, order, 0, len(as)-1)
}

// dims2 extracts the (r, c) pairs, treating a 1-D first/last operand as its
// vector dimension.
func dims2[T numgo.Number](as []*numgo.Array[T]) [][2]int {
	dims := make([][2]int, len(as))
	for i, a := range as {
		switch a.Ndim() {
		case 1:
			dims[i] = [2]int{a.Size(), a.Size()}
		case 2:
			m, n := last2(a.Shape())
			dims[i] = [2]int{m, n}
		default:
			panicDims("multi_dot", "multi_dot: each array must be 1-D or 2-D")
		}
	}
	// Vector ends contribute only their shared dimension to the cost.
	if as[0].Ndim() == 1 {
		dims[0] = [2]int{1, as[0].Size()}
	}
	if last := len(as) - 1; as[last].Ndim() == 1 {
		dims[last] = [2]int{as[last].Size(), 1}
	}
	return dims
}

// chainOrder returns the root split table of the matrix-chain DP (entry
// [i][j] = k means: multiply (i..k) then (k+1..j)).
func chainOrder(dims [][2]int) [][]int {
	n := len(dims)
	p := make([]int, n+1)
	p[0] = dims[0][0]
	for i := range n {
		p[i+1] = dims[i][1]
	}
	cost := make([][]int, n)
	split := make([][]int, n)
	for i := range cost {
		cost[i] = make([]int, n)
		split[i] = make([]int, n)
	}
	for l := 1; l < n; l++ {
		for i := 0; i+l < n; i++ {
			j := i + l
			cost[i][j] = int(^uint(0) >> 1)
			for k := i; k < j; k++ {
				c := cost[i][k] + cost[k+1][j] + p[i]*p[k+1]*p[j+1]
				if c < cost[i][j] {
					cost[i][j] = c
					split[i][j] = k
				}
			}
		}
	}
	return split
}

// chainMultiply evaluates arrs[i..j] under the DP splits.
func chainMultiply[T numgo.Number](arrs []*numgo.Array[T], split [][]int, i, j int) *numgo.Array[T] {
	if i == j {
		return arrs[i]
	}
	k := split[i][j]
	return chainMultiply(arrs, split, i, k).Matmul(chainMultiply(arrs, split, k+1, j))
}

// MatrixPower raises the square (..., M, M) matrix a to the integer power n
// by repeated squaring. n == 0 gives the identity; n < 0 inverts a first
// (singular a panics).
func MatrixPower[T numgo.Number](a *numgo.Array[T], n int) *numgo.Array[T] {
	shape := a.Shape()
	m, k := last2(shape)
	if a.Ndim() < 2 || m != k {
		panicSquare("matrix_power", shape)
	}
	base := a
	if n < 0 {
		base = Inv(a)
		n = -n
	}
	result := IdentityLike(base, shape)
	b := base
	for e := n; e > 0; e >>= 1 {
		if e&1 == 1 {
			result = result.Matmul(b)
		}
		b = b.Matmul(b)
	}
	return result
}

// IdentityLike builds the identity with a's batch shape.
func IdentityLike[T numgo.Number](a *numgo.Array[T], shape []int) *numgo.Array[T] {
	m, _ := last2(shape)
	out := numgo.Zeros[T](shape...)
	eachBatch(shape, func(batch []int) {
		for i := range m {
			coords := append(append([]int(nil), batch...), i, i)
			out.Set(coords, 1)
		}
	})
	return out
}

// TensorSolve solves the tensor equation a @ x = b for x, where a's rightmost
// b.Ndim() axes and b's axes are contracted (numpy semantics: reshape to the
// prod-square system and Solve). axes optionally permutes a's axes first.
func TensorSolve[T numgo.Number](a, b *numgo.Array[T], axes []int) *numgo.Array[T] {
	aa := a
	if len(axes) > 0 {
		perm := make([]int, 0, a.Ndim())
		for i := 0; i < a.Ndim(); i++ {
			perm = append(perm, i)
		}
		for _, k := range axes {
			perm = append(perm[:k], perm[k+1:]...)
			perm = append(perm, k)
		}
		aa = a.Transpose(perm...)
	}
	an := aa.Ndim()
	bn := b.Ndim()
	if bn >= an {
		panicDims("tensorsolve", "b must have fewer dimensions than a")
	}
	shape := aa.Shape()
	prod := 1
	for _, d := range shape[an-bn:] {
		prod *= d
	}
	if aa.Size() != prod*prod {
		panicDims("tensorsolve", "Input arrays must satisfy the requirement prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])")
	}
	flat := aa.Reshape(prod, prod)
	fb := b.Reshape(prod)
	x := Solve(flat, fb)
	return x.Reshape(shape[an-bn:]...)
}

// TensorInv computes the "inverse" of an N-D array relative to tensordot:
// with ind first axes defining the contracted side, prod(shape[:ind]) must
// equal prod(shape[ind:]) and the result has shape shape[ind:] ++ shape[:ind].
func TensorInv[T numgo.Number](a *numgo.Array[T], ind int) *numgo.Array[T] {
	oldShape := a.Shape()
	if ind <= 0 {
		panicDims("tensorinv", "Invalid ind argument.")
	}
	prod := 1
	for _, d := range oldShape[ind:] {
		prod *= d
	}
	invshape := append(append([]int(nil), oldShape[ind:]...), oldShape[:ind]...)
	flat := a.Reshape(prod, a.Size()/prod)
	ia := Inv(flat)
	return ia.Reshape(invshape...)
}

// Trace returns the sum along the diagonals at the given offset of the two
// trailing axes. Batched input sums per batch; the result is a scalar in a
// [1]-shaped array for 2-D input (house scalar convention).
func Trace[T numgo.Number](a *numgo.Array[T], offset int) *numgo.Array[T] {
	shape := a.Shape()
	if a.Ndim() < 2 {
		panicDims("trace", "trace requires at least 2 dimensions")
	}
	outShape := append([]int(nil), shape[:a.Ndim()-2]...)
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := numgo.Zeros[T](outShape...)
	eachBatch(shape, func(batch []int) {
		sum := traceAt(a, batch, shape[a.Ndim()-2], shape[a.Ndim()-1], offset)
		out.Data()[linearIndex(outShape, clampBatch(outShape, batch))] = sum
	})
	return out
}

// traceAt sums the diagonal at offset of the (m,n) matrix at batch.
func traceAt[T numgo.Number](a *numgo.Array[T], batch []int, m, n, offset int) T {
	i0, i1 := 0, 0
	if offset >= 0 {
		i1 = offset
	} else {
		i0 = -offset
	}
	var sum T
	coords := append(append([]int(nil), batch...), 0, 0)
	k := len(batch)
	for i0 < m && i1 < n {
		coords[k], coords[k+1] = i0, i1
		sum += a.At(coords...)
		i0++
		i1++
	}
	return sum
}

// clampBatch right-aligns batch coords against outShape (batch dims absent
// from the output collapse to 0).
func clampBatch(outShape, batch []int) []int {
	if len(batch) >= len(outShape) {
		return batch[len(batch)-len(outShape):]
	}
	return make([]int, len(outShape))
}

// Diagonal extracts the diagonal at offset from the two trailing axes.
// The diagonal axis is appended after the batch axes (numpy places it last).
func Diagonal[T numgo.Number](a *numgo.Array[T], offset int) *numgo.Array[T] {
	shape := a.Shape()
	if a.Ndim() < 2 {
		panicDims("diagonal", "diagonal requires at least 2 dimensions")
	}
	m, n := shape[a.Ndim()-2], shape[a.Ndim()-1]
	i0, i1 := 0, 0
	if offset >= 0 {
		i1 = offset
	} else {
		i0 = -offset
	}
	d := max(min(m-i0, n-i1), 0)
	outShape := append(append([]int(nil), shape[:a.Ndim()-2]...), d)
	out := numgo.Zeros[T](outShape...)
	eachBatch(shape, func(batch []int) {
		coords := append(append([]int(nil), batch...), 0, 0)
		k := len(batch)
		for t := 0; t < d; t++ {
			coords[k], coords[k+1] = i0+t, i1+t
			out.Set(append(append([]int(nil), batch...), t), a.At(coords...))
		}
	})
	return out
}

// Cross returns the cross product of a and b along axis (default -1). Both
// operands must have size 3 along that axis; leading dimensions broadcast.
func Cross[T numgo.Number](a, b *numgo.Array[T], axis int) *numgo.Array[T] {
	shapeA, shapeB := a.Shape(), b.Shape()
	ax := axis
	if ax < 0 {
		ax += len(shapeA)
	}
	if ax < 0 || ax >= len(shapeA) || shapeA[ax] != 3 {
		panicDims("cross", "Both input arrays must be (arrays of) 3-dimensional vectors")
	}
	axB := axis
	if axB < 0 {
		axB += len(shapeB)
	}
	if axB < 0 || axB >= len(shapeB) || shapeB[axB] != 3 {
		panicDims("cross", "Both input arrays must be (arrays of) 3-dimensional vectors")
	}
	leadA := append(append([]int(nil), shapeA[:ax]...), shapeA[ax+1:]...)
	leadB := append(append([]int(nil), shapeB[:axB]...), shapeB[axB+1:]...)
	lead, ok := broadcastShapes(leadA, leadB)
	if !ok {
		panicDims("cross", "operands could not be broadcast together")
	}
	outShape := make([]int, 0, len(lead)+1)
	outShape = append(outShape, lead[:ax]...)
	outShape = append(outShape, 3)
	outShape = append(outShape, lead[ax:]...)
	avShape := make([]int, 0, len(lead)+1)
	avShape = append(avShape, lead[:ax]...)
	avShape = append(avShape, 3)
	avShape = append(avShape, lead[ax:]...)
	bvShape := make([]int, 0, len(lead)+1)
	bvShape = append(bvShape, lead[:axB]...)
	bvShape = append(bvShape, 3)
	bvShape = append(bvShape, lead[axB:]...)
	av := a.BroadcastTo(avShape...)
	bv := b.BroadcastTo(bvShape...)
	out := numgo.Zeros[T](outShape...)
	// Sweep all coordinates except the cross axis; per slot write all three
	// outputs. sweep holds the non-axis coordinates in output order.
	sweepShape := make([]int, 0, len(outShape)-1)
	sweepShape = append(sweepShape, outShape[:ax]...)
	sweepShape = append(sweepShape, outShape[ax+1:]...)
	sweep := make([]int, len(sweepShape))
	full := make([]int, len(outShape))
	var walk func(k int)
	walk = func(k int) {
		if k == len(sweepShape) {
			lead := append([]int(nil), sweep...)
			x1, x2, x3 := compAt(av, lead, ax, 0, 0), compAt(av, lead, ax, 0, 1), compAt(av, lead, ax, 0, 2)
			y1, y2, y3 := compAt(bv, lead, axB, 0, 0), compAt(bv, lead, axB, 0, 1), compAt(bv, lead, axB, 0, 2)
			copy(full, sweep)
			copy(full[ax+1:], sweep[ax:]) // shift right by one for the axis slot
			full[ax] = 0
			out.Set(full, x2*y3-x3*y2)
			full[ax] = 1
			out.Set(full, x3*y1-x1*y3)
			full[ax] = 2
			out.Set(full, x1*y2-x2*y1)
			return
		}
		for i := 0; i < sweepShape[k]; i++ {
			sweep[k] = i
			walk(k + 1)
		}
	}
	walk(0)
	return out
}

// compAt reads (lead..., comp) from a broadcast vector array with its
// 3-axis at position ax.
func compAt[T numgo.Number](v *numgo.Array[T], lead []int, ax, i, comp int) T {
	coords := append(append([]int(nil), lead[:ax]...), comp)
	coords = append(coords, lead[ax:]...)
	return v.At(coords...)
}

// broadcastShapes right-aligns two shapes (numpy rules); ok is false on
// incompatible dims.
func broadcastShapes(a, b []int) (out []int, ok bool) {
	if len(a) < len(b) {
		a, b = b, a
	}
	out = append([]int(nil), a...)
	off := len(a) - len(b)
	for i := range b {
		switch {
		case b[i] == 1:
		case a[off+i] == 1:
			out[off+i] = b[i]
		case a[off+i] != b[i]:
			return nil, false
		}
	}
	return out, true
}

// Outer returns the outer product of the flattened operands:
// out[i,j] = a_ravel[i] * b_ravel[j].
func Outer[T numgo.Number](a, b *numgo.Array[T]) *numgo.Array[T] {
	ra, rb := a.Ravel(), b.Ravel()
	out := numgo.Zeros[T](ra.Size(), rb.Size())
	coords := make([]int, 2)
	for i := 0; i < ra.Size(); i++ {
		coords[0] = i
		for j := 0; j < rb.Size(); j++ {
			coords[1] = j
			out.Set(coords, ra.At(i)*rb.At(j))
		}
	}
	return out
}

// Vecdot computes the dot product of a and b along axis (default -1),
// broadcasting the leading dimensions. The axis dimensions must match.
func Vecdot[T numgo.Number](a, b *numgo.Array[T], axis int) *numgo.Array[T] {
	shapeA, shapeB := a.Shape(), b.Shape()
	ax := axis
	if ax < 0 {
		ax += len(shapeA)
	}
	axB := axis
	if axB < 0 {
		axB += len(shapeB)
	}
	if ax < 0 || ax >= len(shapeA) || axB < 0 || axB >= len(shapeB) {
		panicAxis(axis, len(shapeA))
	}
	if shapeA[ax] != shapeB[axB] {
		panicDims("vecdot", "vecdot: incompatible axis lengths")
	}
	// Move the contraction axis last on both operands, then broadcast the
	// remaining leading dimensions (numpy semantics).
	permA := make([]int, 0, len(shapeA))
	for i := range shapeA {
		if i != ax {
			permA = append(permA, i)
		}
	}
	permA = append(permA, ax)
	permB := make([]int, 0, len(shapeB))
	for i := range shapeB {
		if i != axB {
			permB = append(permB, i)
		}
	}
	permB = append(permB, axB)
	a = a.Transpose(permA...)
	b = b.Transpose(permB...)
	leadA := a.Shape()[:a.Ndim()-1]
	leadB := b.Shape()[:b.Ndim()-1]
	lead, ok := broadcastShapes(leadA, leadB)
	if !ok {
		panicDims("vecdot", "operands could not be broadcast together")
	}
	k := shapeA[ax]
	a = a.BroadcastTo(append(append([]int(nil), lead...), k)...)
	b = b.BroadcastTo(append(append([]int(nil), lead...), k)...)
	outShape := lead
	if len(outShape) == 0 {
		// Scalar result kept in a [1] array (house convention).
		outShape = []int{1}
	}
	out := numgo.Zeros[T](outShape...)
	slot := make([]int, len(lead))
	var walk func(d int)
	walk = func(d int) {
		if d == len(lead) {
			var sum T
			for c := range k {
				sum += a.At(append(slot[:len(slot):len(slot)], c)...) * b.At(append(slot[:len(slot):len(slot)], c)...)
			}
			if len(lead) == 0 {
				out.Data()[0] = sum
			} else {
				out.Set(slot, sum)
			}
			return
		}
		for i := 0; i < lead[d]; i++ {
			slot[d] = i
			walk(d + 1)
		}
	}
	walk(0)
	return out
}

// Kron returns the Kronecker product of a and b: the result shape is the
// per-dimension concatenation of the two shapes and
// out[i_a*sb+i_b] = a[ia]*b[ib] per dimension pair.
func Kron[T numgo.Number](a, b *numgo.Array[T]) *numgo.Array[T] {
	sa, sb := a.Shape(), b.Shape()
	if len(sa) != len(sb) {
		panicDims("kron", "kron operands must have the same number of dimensions")
	}
	outShape := make([]int, len(sa))
	for i := range sa {
		outShape[i] = sa[i] * sb[i]
	}
	out := numgo.Zeros[T](outShape...)
	ia := make([]int, len(sa))
	ib := make([]int, len(sa))
	coords := make([]int, len(sa))
	var walk func(k int)
	walk = func(k int) {
		if k == len(sa) {
			for i := range sa {
				coords[i] = ia[i]*sb[i] + ib[i]
			}
			out.Set(coords, a.At(ia...)*b.At(ib...))
			return
		}
		for ib[k] = 0; ib[k] < sb[k]; ib[k]++ {
			walk(k + 1)
		}
	}
	var walkA func(k int)
	walkA = func(k int) {
		if k == len(sa) {
			for ib[0] = 0; ib[0] < sb[0]; ib[0]++ {
				walk(1)
			}
			return
		}
		for ia[k] = 0; ia[k] < sa[k]; ia[k]++ {
			walkA(k + 1)
		}
	}
	walkA(0)
	return out
}
