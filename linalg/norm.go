package linalg

import (
	"fmt"
	"math"

	"github.com/aixfoundry/numgo"
)

// Norms, condition numbers, ranks and pseudo-inverses.
// Norm follows numpy.linalg.norm's ord table; Cond, MatrixRank and Pinv go
// through the SVD (Dgesvd); Norm itself uses direct reduction loops — the
// index algebra stays behind At/Set (P0-1) and plain flat slices.

// Norm returns the matrix or vector norm of a, mirroring numpy.linalg.norm:
//
//   - ord nil or 2 with a 1-D array (or axis of length 1): the 2-norm.
//   - ord nil with a 2-D array: the Frobenius norm.
//   - 1-D vectors accept int/float orders including 0, ±Inf, and fractional.
//   - 2-D matrices accept 1, -1, 2, -2, ±Inf, "fro" and "nuc".
//   - axis of length 2 selects matrix-norm axes inside an N-D array; length 1
//     selects a vector axis; nil uses the numpy defaults above (flat vector
//     norm when ndim > 2).
//   - keepdims keeps reduced axes as size 1.
//
// The result is a scalar held in a size-[1] array (house scalar convention).
func Norm[T numgo.Number](a *numgo.Array[T], ord any, axis []int, keepdims bool) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	switch {
	case len(axis) == 2:
		r1, r2 := axis[0], axis[1]
		r1 = normAxis(r1, len(shape))
		r2 = normAxis(r2, len(shape))
		if r1 == r2 {
			panicDims("norm", "Duplicate axes given.")
		}
		r := matrixNormAxes(af, ord, r1, r2, keepdims)
		return wrap[T](r.Data(), r.Shape())
	case len(axis) == 1:
		ax := normAxis(axis[0], len(shape))
		r := vectorNormAxes(af, ord, ax, keepdims)
		return wrap[T](r.Data(), r.Shape())
	case len(axis) == 0:
		if len(shape) == 0 {
			panicDims("norm", "norm is not defined on 0-d arrays")
		}
		if len(shape) == 2 {
			r := matrixNormAxes(af, ord, 0, 1, keepdims)
			return wrap[T](r.Data(), r.Shape())
		}
		r := vectorNormFlat(af, ord, keepdims)
		return wrap[T](r.Data(), r.Shape())
	default:
		panicNormAxis()
		return nil
	}
}

// normAxis resolves a negative axis.
func normAxis(axis, ndim int) int {
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		panicAxis(axis, ndim)
	}
	return axis
}

// vectorNormFlat computes the norm over all elements as a flat vector.
func vectorNormFlat(a *numgo.Array[float64], ord any, keepdims bool) *numgo.Array[float64] {
	v := 0.0
	switch o := ord.(type) {
	case nil:
		v = flatOrdData(a.Copy().Data(), 2)
	case int:
		v = flatOrdData(a.Copy().Data(), float64(o))
	case float64:
		v = flatOrdData(a.Copy().Data(), o)
	case string:
		panicNormOrder(o)
	default:
		panicNormOrder(fmt.Sprint(o))
	}
	outShape := []int{1}
	if keepdims {
		outShape = ones(len(a.Shape()))
	}
	return numgo.FromSlice([]float64{v}, outShape...)
}

// flatOrdData applies a vector order over an explicit slice.
func flatOrdData(data []float64, ord float64) float64 {
	return flatOrd1(data, ord)
}

// vectorNormAxes computes a vector norm along one axis, keepdims-aware.
// Each output slot is visited in C order and the vector along ax is folded
// by rebuilding the full coordinate from the slot coordinate.
func vectorNormAxes(a *numgo.Array[float64], ord any, ax int, keepdims bool) *numgo.Array[float64] {
	shape := a.Shape()
	outShape := reduceShapeDrop(shape, ax)
	kept := keptShapeOf(shape, ax)
	vals := make([]float64, sizeOf(outShape))
	var fold func(sub []float64) float64
	switch o := ord.(type) {
	case nil:
		fold = func(sub []float64) float64 { return flatOrd1(sub, 2) }
	case int:
		fold = func(sub []float64) float64 { return flatOrd1(sub, float64(o)) }
	case float64:
		fold = func(sub []float64) float64 { return flatOrd1(sub, o) }
	case string:
		panicNormOrder(o)
	default:
		panicNormOrder(fmt.Sprint(o))
	}
	sub := make([]float64, shape[ax])
	// slot[k'] maps output axis k' to input axis (skipping ax).
	mapIn := make([]int, 0, len(outShape))
	for k := range shape {
		if k != ax {
			mapIn = append(mapIn, k)
		}
	}
	slot := make([]int, len(outShape))
	coords := make([]int, len(shape))
	var walk func(k int)
	walk = func(k int) {
		if k == len(outShape) {
			for i := 0; i < shape[ax]; i++ {
				coords[ax] = i
				sub[i] = a.At(coords...)
			}
			vals[linearIndex(outShape, slot)] = fold(sub)
			return
		}
		for i := 0; i < outShape[k]; i++ {
			slot[k] = i
			coords[mapIn[k]] = i
			walk(k + 1)
		}
	}
	walk(0)
	outShape2 := outShape
	if keepdims {
		outShape2 = kept
		vals = padKeepdims1(vals, outShape, outShape2, ax)
	}
	return numgo.FromSlice(vals, outShape2...)
}

// padKeepdims1 re-expands vals (axis dropped) into the keepdims shape with a
// single size-1 axis at ax.
func padKeepdims1(vals []float64, dropped, kept []int, ax int) []float64 {
	out := make([]float64, sizeOf(kept))
	coords := make([]int, len(kept))
	var walk func(k int)
	walk = func(k int) {
		if k == len(kept) {
			dc := make([]int, 0, len(kept)-1)
			for i := range kept {
				if i != ax {
					dc = append(dc, coords[i])
				}
			}
			out[linearIndex(kept, coords)] = vals[linearIndex(dropped, dc)]
			return
		}
		for i := 0; i < kept[k]; i++ {
			coords[k] = i
			walk(k + 1)
		}
	}
	walk(0)
	return out
}

// flatOrd1 is flatOrd over an explicit slice.
func flatOrd1(data []float64, ord float64) float64 {
	switch ord {
	case 2:
		s := 0.0
		for _, v := range data {
			s += v * v
		}
		return math.Sqrt(s)
	case 1:
		s := 0.0
		for _, v := range data {
			s += math.Abs(v)
		}
		return s
	case 0:
		n := 0
		for _, v := range data {
			if v != 0 {
				n++
			}
		}
		return float64(n)
	case math.Inf(1):
		m := 0.0
		for _, v := range data {
			m = math.Max(m, math.Abs(v))
		}
		return m
	case math.Inf(-1):
		if len(data) == 0 {
			return math.Inf(1)
		}
		m := math.Abs(data[0])
		for _, v := range data[1:] {
			m = math.Min(m, math.Abs(v))
		}
		return m
	default:
		s := 0.0
		for _, v := range data {
			s += math.Pow(math.Abs(v), ord)
		}
		return math.Pow(s, 1/ord)
	}
}

// matrixNormAxes computes the matrix norm over axes r1×r2 of an N-D array.
func matrixNormAxes(a *numgo.Array[float64], ord any, r1, r2 int, keepdims bool) *numgo.Array[float64] {
	shape := a.Shape()
	outShape := reduceShapeDrop2(shape, r1, r2)
	vals := make([]float64, sizeOf(outShape))
	eachBatch2(shape, r1, r2, func(lead []int) {
		// Build the (m,n) slice at this lead coordinate.
		coords := make([]int, len(shape))
		copy(coords, lead)
		// lead indices fill non-matrix axes in order.
		li := 0
		for k := range shape {
			if k != r1 && k != r2 {
				coords[k] = lead[li]
				li++
			}
		}
		m, n := shape[r1], shape[r2]
		mat := make([]float64, m*n)
		for i := 0; i < m; i++ {
			coords[r1] = i
			for j := 0; j < n; j++ {
				coords[r2] = j
				mat[i*n+j] = a.At(coords...)
			}
		}
		vals[slotIndex2(outShape, lead)] = matrixOrd(ord, m, n, mat)
	})
	out := outShape
	if keepdims {
		out = keptShapeOf2(shape, r1, r2)
		vals = padKeepdims(vals, out, min(r1, r2), max(r1, r2))
	}
	return numgo.FromSlice(vals, out...)
}

// matrixOrd computes one matrix norm value. ord accepts nil (fro), int,
// float64 (whole-valued) and the strings "fro"/"f"/"nuc".
func matrixOrd(ord any, m, n int, mat []float64) float64 {
	switch o := ord.(type) {
	case nil:
		return frobenius(m, n, mat)
	case string:
		switch o {
		case "fro", "f":
			return frobenius(m, n, mat)
		case "nuc":
			_, sv, _ := svdMatrix(mat, m, n)
			s := 0.0
			for _, v := range sv {
				s += v
			}
			return s
		default:
			panicNormOrderMatrix()
			return 0
		}
	case int:
		switch {
		case o == 1:
			return maxColSum(m, n, mat)
		case o == -1:
			return minColSum(m, n, mat)
		case o == 2:
			_, sv, _ := svdMatrix(mat, m, n)
			return sv[0]
		case o == -2:
			_, sv, _ := svdMatrix(mat, m, n)
			return sv[len(sv)-1]
		default:
			panicNormOrderMatrix()
			return 0
		}
	case float64:
		switch {
		case o == math.Inf(1):
			return maxRowSum(m, n, mat)
		case o == math.Inf(-1):
			return minRowSum(m, n, mat)
		case o == float64(int(o)):
			return matrixOrd(int(o), m, n, mat)
		default:
			panicNormOrderMatrix()
			return 0
		}
	default:
		panicNormOrderMatrix()
		return 0
	}
}

func frobenius(m, n int, mat []float64) float64 {
	s := 0.0
	for _, v := range mat {
		s += v * v
	}
	return math.Sqrt(s)
}

func maxColSum(m, n int, mat []float64) float64 {
	best := 0.0
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < m; i++ {
			s += math.Abs(mat[i*n+j])
		}
		best = math.Max(best, s)
	}
	return best
}

func minColSum(m, n int, mat []float64) float64 {
	best := math.Inf(1)
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < m; i++ {
			s += math.Abs(mat[i*n+j])
		}
		best = math.Min(best, s)
	}
	return best
}

func maxRowSum(m, n int, mat []float64) float64 {
	best := 0.0
	for i := 0; i < m; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += math.Abs(mat[i*n+j])
		}
		best = math.Max(best, s)
	}
	return best
}

func minRowSum(m, n int, mat []float64) float64 {
	best := math.Inf(1)
	for i := 0; i < m; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += math.Abs(mat[i*n+j])
		}
		best = math.Min(best, s)
	}
	return best
}

// MatrixNorm returns the matrix norm of x over two axes (numpy 2.x
// matrix_norm; ord defaults to 'fro').
func MatrixNorm[T numgo.Number](x *numgo.Array[T], ord any, axis []int, keepdims bool) *numgo.Array[T] {
	if len(axis) != 2 {
		if x.Ndim() < 2 {
			panicMatrixNormAxis()
		}
		axis = []int{x.Ndim() - 2, x.Ndim() - 1}
	}
	xf := f64(x)
	r1, r2 := normAxis(axis[0], xf.Ndim()), normAxis(axis[1], xf.Ndim())
	if r1 == r2 {
		panicDims("matrix_norm", "Duplicate axes given.")
	}
	r := matrixNormAxes(xf, ord, r1, r2, keepdims)
	return wrap[T](r.Data(), r.Shape())
}

// VectorNorm returns the vector norm of x (numpy 2.x vector_norm; ord
// defaults to 2). axis of length 1 selects the axis; nil flattens.
func VectorNorm[T numgo.Number](x *numgo.Array[T], ord any, axis []int, keepdims bool) *numgo.Array[T] {
	xf := f64(x)
	var r *numgo.Array[float64]
	if len(axis) == 0 {
		r = vectorNormFlat(xf, ord, keepdims)
	} else {
		r = vectorNormAxes(xf, ord, normAxis(axis[0], xf.Ndim()), keepdims)
	}
	return wrap[T](r.Data(), r.Shape())
}

// Cond computes the condition number of a (..., M, M) matrix: p nil or 2 uses
// the ratio of extreme singular values; other orders (1, -1, 2, -2, ±Inf,
// "fro") use the numpy formulas. Singular matrices give +Inf.
func Cond[T numgo.Number](a *numgo.Array[T], p any) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m != n {
		panicSquare("cond", shape)
	}
	var val float64
	switch pv := p.(type) {
	case nil:
		_, sv, _ := svdMatrix(aMatrixAt(af, nil, m, m), m, m)
		val = sv[0] / sv[len(sv)-1]
	case int:
		pi := pv
		_, sv, _ := svdMatrix(aMatrixAt(af, nil, m, m), m, m)
		switch pi {
		case 2:
			val = sv[0] / sv[len(sv)-1]
		case -2:
			val = sv[len(sv)-1] / sv[0]
		case 1, -1:
			inv := Inv[T](a)
			invf := f64(inv)
			val = matrixOrd(pi, m, m, aMatrixAt(af, nil, m, m)) * matrixOrd(pi, m, m, aMatrixAt(invf, nil, m, m))
		default:
			panicDims("cond", "Invalid norm order for matrices.")
		}
	case float64:
		_, sv, _ := svdMatrix(aMatrixAt(af, nil, m, m), m, m)
		am := aMatrixAt(af, nil, m, m)
		switch {
		case pv == math.Inf(1), pv == math.Inf(-1), pv == float64(int(pv)) && (int(pv) == 1 || int(pv) == -1):
			inv := Inv[T](a)
			invf := f64(inv)
			val = matrixOrd(pv, m, m, am) * matrixOrd(pv, m, m, aMatrixAt(invf, nil, m, m))
		case pv == 2:
			val = sv[0] / sv[len(sv)-1]
		case pv == -2:
			val = sv[len(sv)-1] / sv[0]
		default:
			panicDims("cond", "Invalid norm order for matrices.")
		}
	case string:
		inv := Inv[T](a)
		invf := f64(inv)
		val = matrixOrd(pv, m, m, aMatrixAt(af, nil, m, m)) * matrixOrd(pv, m, m, aMatrixAt(invf, nil, m, m))
	default:
		panicDims("cond", "Invalid norm order for matrices.")
	}
	return numgo.FromSlice([]T{scalarOf[T](val)}, 1)
}

// MatrixRank returns the number of singular values of a greater than tol
// (numpy 2.x: strict >). tol <= 0 selects the default cutoff
// eps * max(M, N) * s[0]. Backend: Dgesvd.
func MatrixRank[T numgo.Number](a *numgo.Array[T], tol float64) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	if af.Ndim() < 2 {
		panicDims("matrix_rank", "matrix_rank requires at least 2 dimensions")
	}
	m, n := last2(shape)
	lead := batchShape(shape)
	if len(lead) == 0 {
		lead = []int{1}
	}
	out := numgo.Zeros[T](lead...)
	var slot int
	eachBatch(shape, func(batch []int) {
		_, sv, _ := svdMatrix(aMatrixAt(af, batch, m, n), m, n)
		if len(sv) == 0 {
			out.Data()[slot] = 0
			slot++
			return
		}
		t := tol
		if t <= 0 {
			t = 2.220446049250313e-16 * float64(max(m, n)) * sv[0]
		}
		rank := 0
		for _, v := range sv {
			if v > t {
				rank++
			}
		}
		out.Data()[slot] = scalarOf[T](float64(rank))
		slot++
	})
	if a.Ndim() == 2 {
		return numgo.FromSlice([]T{out.At(0)}, 1)
	}
	return out
}

// Pinv computes the Moore-Penrose pseudo-inverse via the SVD: singular
// values <= rcond * s[0] are truncated. rcond < 0 selects numpy's default
// 1e-15. Backend: Dgesvd.
func Pinv[T numgo.Number](a *numgo.Array[T], rcond float64) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	if af.Ndim() < 2 {
		panicDims("pinv", "pinv requires at least 2 dimensions")
	}
	m, n := last2(shape)
	k := min(m, n)
	if rcond < 0 {
		rcond = 1e-15
	}
	out := numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), n, m)...)
	eachBatch(shape, func(batch []int) {
		u, sv, vt := svdMatrix(aMatrixAt(af, batch, m, n), m, n)
		// pinv = Vt^T diag(cut) U^T (row-major n×m):
		// pinv[row, col] = Σ_i vt[i*n+row] * cut_i * u[col*k+i].
		pinv := make([]float64, n*m)
		for i := 0; i < k; i++ {
			s := sv[i]
			if s > rcond*sv[0] && s > 0 {
				s = 1 / s
			} else {
				s = 0
			}
			for row := 0; row < n; row++ {
				vc := vt[i*n+row] * s
				for col := 0; col < m; col++ {
					pinv[row*m+col] += vc * u[col*k+i]
				}
			}
		}
		setMatAt(out, batch, n, m, pinv)
	})
	return out
}

// reduceShapeDrop drops axis ax.
func reduceShapeDrop(shape []int, ax int) []int {
	out := make([]int, 0, len(shape)-1)
	for i, d := range shape {
		if i != ax {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		out = []int{1}
	}
	return out
}

// reduceShapeDrop2 drops axes r1 and r2.
func reduceShapeDrop2(shape []int, r1, r2 int) []int {
	out := make([]int, 0, len(shape)-2)
	for i, d := range shape {
		if i != r1 && i != r2 {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		out = []int{1}
	}
	return out
}

// keptShapeOf sets axis ax to 1.
func keptShapeOf(shape []int, ax int) []int {
	out := append([]int(nil), shape...)
	out[ax] = 1
	return out
}

// keptShapeOf2 sets axes r1, r2 to 1.
func keptShapeOf2(shape []int, r1, r2 int) []int {
	out := append([]int(nil), shape...)
	out[r1], out[r2] = 1, 1
	return out
}

// sizeOf is nditer.Size for local slices.
func sizeOf(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// ones builds an all-ones shape.
func ones(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// slotIndex maps batch coords (with axis ax collapsed) into the dropped
// shape's flat index.
func slotIndex(outShape, batch []int, ax int) int {
	coords := make([]int, len(outShape))
	li := 0
	for i := range batch {
		if i != ax {
			coords[li] = batch[i]
			li++
		}
	}
	return linearIndex(outShape, coords)
}

// slotIndex2 maps lead coords into the doubly-dropped shape.
func slotIndex2(outShape, lead []int) int {
	if len(outShape) == 0 {
		return 0
	}
	return linearIndex(outShape, lead)
}

// eachBatch2 iterates all non-matrix axes in C order.
func eachBatch2(shape []int, r1, r2 int, f func(lead []int)) {
	leadShape := make([]int, 0, len(shape)-2)
	for i, d := range shape {
		if i != r1 && i != r2 {
			leadShape = append(leadShape, d)
		}
	}
	if len(leadShape) == 0 {
		f(nil)
		return
	}
	lead := make([]int, len(leadShape))
	var rec func(k int)
	rec = func(k int) {
		if k == len(leadShape) {
			f(lead)
			return
		}
		for i := 0; i < leadShape[k]; i++ {
			lead[k] = i
			rec(k + 1)
		}
	}
	rec(0)
}

// padKeepdims re-expands vals (dropped shape) into the keepdims shape by
// re-inserting size-1 axes at their original positions (given by drop
// coordinates d1 < d2 into the kept shape).
func padKeepdims(vals []float64, kept []int, d1, d2 int) []float64 {
	out := make([]float64, sizeOf(kept))
	coords := make([]int, len(kept))
	var walk func(k int)
	walk = func(k int) {
		if k == len(kept) {
			dc := make([]int, 0, len(kept)-2)
			for i := range kept {
				if i != d1 && i != d2 {
					dc = append(dc, coords[i])
				}
			}
			if len(dc) == 0 {
				dc = []int{0}
			}
			out[linearIndex(kept, coords)] = vals[linearIndex(keptShapeless(droppedOf(kept, d1, d2), dc), dc)]
			return
		}
		for i := 0; i < kept[k]; i++ {
			coords[k] = i
			walk(k + 1)
		}
	}
	walk(0)
	return out
}

// droppedOf removes the two keepdims axes from kept.
func droppedOf(kept []int, d1, d2 int) []int {
	out := make([]int, 0, len(kept)-2)
	for i, d := range kept {
		if i != d1 && i != d2 {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		out = []int{1}
	}
	return out
}

// keptShapeless replaces any trailing empty shape with [1].
func keptShapeless(dropped, dc []int) []int {
	if len(dropped) == len(dc) {
		return dropped
	}
	return []int{1}
}
