package linalg

import (
	"fmt"
	"math"

	"github.com/kaisawind/numgo"
	"gonum.org/v1/gonum/lapack"
)

// Solvers and determinant functions: Solve, Inv, Det, Slogdet, Lstsq.
// Backend: LU via gonum's Dgesv/Dgetrf+Dgetri. Batched systems loop over the
// leading dimensions (shortcut: batched BLAS later). Gonum buffers are
// row-major (C order), matching numgo layout directly.

// Solve solves the linear system a @ x = b. a is (..., M, M); b is either a
// (M,) vector, shared across every batch, or (..., M, N) with leading
// dimensions broadcast against a's. Singular a panics
// (numgo/linalg: Singular matrix). Backend: LAPACK Dgesv.
func Solve[T numgo.Number](a, b *numgo.Array[T]) *numgo.Array[T] {
	const op = "solve"
	af := f64(a)
	bf := f64(b)
	ashape := af.Shape()
	if af.Ndim() < 2 {
		panicDims(op, "1-dimensional array given. Array must be at least two-dimensional")
	}
	m, n := last2(ashape)
	if m != n {
		panicSquare(op, ashape)
	}
	if bf.Ndim() == 1 {
		// Vector right-hand side: shared across the batch, result (..., M).
		if bf.Size() != m {
			panicDims(op, fmt.Sprintf("solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size %d is different from %d)", bf.Size(), m))
		}
		out := numgo.Zeros[T](append(batchShape(ashape), m)...)
		b0 := bf.Copy().Data()[:m]
		eachBatch(ashape, func(batch []int) {
			am := aMatrixAt(af, batch, m, n)
			bm := append([]float64(nil), b0...)
			if !gesvJob(m, 1, am, bm) {
				panicSingular()
			}
			setVecAt(out, batch, bm)
		})
		return out
	}
	bshape := bf.Shape()
	if bshape[len(bshape)-2] != m {
		panicDims(op, fmt.Sprintf("solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size %d is different from %d)",
			bshape[len(bshape)-2], m))
	}
	bn := bshape[len(bshape)-1]
	lead := batchShape(bshape)
	if !broadcastable(af.Shape()[:af.Ndim()-2], lead) {
		panicDims(op, "solve: operands could not be broadcast together with remapped shapes")
	}
	outShape := append(append([]int(nil), lead...), m, bn)
	out := numgo.Zeros[T](outShape...)
	eachBatch(bshape, func(bbatch []int) {
		abatch := broadcastCoord(af.Shape()[:af.Ndim()-2], lead, bbatch)
		am := aMatrixAt(af, abatch, m, n)
		bm := aMatrixAt(bf, bbatch, m, bn)
		if !gesvJob(m, bn, am, bm) {
			panicSingular()
		}
		setMatAt(out, bbatch, m, bn, bm)
	})
	return out
}

// gesvJob wraps one Dgesv call on a single square system with nrhs
// right-hand sides row-major in b (n×nrhs). a and b are overwritten.
// Returns false when the matrix is exactly singular. Gonum's LAPACK buffers
// are C-order: a has leading dimension n (row stride) and b has leading
// dimension nrhs.
func gesvJob(n, nrhs int, a, b []float64) bool {
	ipiv := make([]int, n)
	return lapack64().Dgesv(n, nrhs, a, n, ipiv, b, max(nrhs, 1))
}

// lapack64 is the shared gonum LAPACK implementation.
func lapack64() gonumImpl { return gonumImpl{} }

// Inv computes the multiplicative inverse of a (..., M, M). Singular input
// panics. Backend: LAPACK Dgetrf + Dgetri.
func Inv[T numgo.Number](a *numgo.Array[T]) *numgo.Array[T] {
	const op = "inv"
	af := f64(a)
	shape := af.Shape()
	if af.Ndim() < 2 {
		panicDims(op, "1-dimensional array given. Array must be at least two-dimensional")
	}
	m, n := last2(shape)
	if m != n {
		panicSquare(op, shape)
	}
	out := numgo.Zeros[T](shape...)
	impl := gonumImpl{}
	eachBatch(shape, func(batch []int) {
		mat := aMatrixAt(af, batch, m, n)
		ipiv := make([]int, m)
		if !impl.Dgetrf(m, m, mat, m, ipiv) {
			panicSingular()
		}
		work := []float64{0}
		impl.Dgetri(m, mat, m, ipiv, work, -1)
		work = make([]float64, int(work[0]))
		if !impl.Dgetri(m, mat, m, ipiv, work, len(work)) {
			panicSingular()
		}
		setMatAt(out, batch, m, m, mat)
	})
	return out
}

// Det computes the determinant of a (..., M, M). Backend: LU (Dgetrf);
// sign accumulates from pivots, value from the log-diagonal.
func Det[T numgo.Number](a *numgo.Array[T]) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m != n {
		panicSquare("det", shape)
	}
	outShape := append([]int(nil), batchShape(shape)...)
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := numgo.Zeros[T](outShape...)
	var slot int
	eachBatch(shape, func(batch []int) {
		sign, logdet := luLogDet(af, batch, m)
		v := math.Exp(logdet) * sign
		out.Data()[slot] = scalarOf[T](v)
		slot++
	})
	return out
}

// SlogdetResult holds numpy's slogdet pair: Sign is one of -1, 0, +1 and
// LogAbsDet is log(|det|) (-Inf when the matrix is singular). This phase
// pins Slogdet to 2-D input (numpy's batched version returns arrays).
type SlogdetResult[T numgo.Number] struct {
	Sign      T
	LogAbsDet T
}

// Slogdet returns the sign and natural log of the absolute value of the
// determinant of a 2-D square matrix, computed via LU to avoid overflow.
// Singular matrices give Sign 0 and LogAbsDet -Inf.
func Slogdet[T numgo.Number](a *numgo.Array[T]) SlogdetResult[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() != 2 || m != n {
		panicSquare("slogdet", shape)
	}
	s, ld := luLogDet(af, nil, m)
	if math.IsInf(ld, -1) {
		s = 0
	}
	return SlogdetResult[T]{Sign: scalarOf[T](s), LogAbsDet: scalarOf[T](ld)}
}

// luLogDet computes (sign, log|det|) of the matrix at batch via Dgetrf.
func luLogDet(a *numgo.Array[float64], batch []int, m int) (sign, logabs float64) {
	mat := aMatrixAt(a, batch, m, m)
	ipiv := make([]int, m)
	impl := gonumImpl{}
	if !impl.Dgetrf(m, m, mat, m, ipiv) {
		// Exactly singular: numpy reports sign 0, log|det| -inf.
		return 0, math.Inf(-1)
	}
	sign = 1
	logabs = 0
	for i := 0; i < m; i++ {
		v := mat[i*m+i]
		if v < 0 {
			sign = -sign
		}
		if ipiv[i] != i {
			sign = -sign
		}
		logabs += math.Log(math.Abs(v))
	}
	return sign, logabs
}

// Lstsq computes a least-squares solution to a @ x = b. a is (M, K) and b is
// (M,) or (M, N). It returns the solution, the sums of squared residuals
// (empty when M <= K), the rank and the singular values of a.
// rcond < 0 means numpy's default cutoff eps * max(M, K); a value in [0, 1]
// truncates singular values below rcond * s[0].
//
// Backend: LAPACK Dgesvd (SVD), like numpy's gelsd driver; the rank-deficient
// pseudo-inverse truncation matches gelsd for exactly rank-deficient systems
// but, unlike gelsd, does not implement its rank-revealing precision loop.
func Lstsq[T numgo.Number](a, b *numgo.Array[T], rcond float64) LstsqResult[T] {
	const op = "lstsq"
	af := f64(a)
	bf := f64(b)
	if af.Ndim() != 2 {
		panicDims(op, "1-dimensional array given. Array must be two-dimensional")
	}
	m, k := last2(af.Shape())
	if bf.Ndim() != 1 && bf.Ndim() != 2 {
		panicDims(op, "b must be 1-D or 2-D")
	}
	if bf.Ndim() == 1 {
		if bf.Size() != m {
			panicDims(op, "Incompatible dimensions")
		}
	} else if bf.Shape()[0] != m {
		panicDims(op, "Incompatible dimensions")
	}
	u, s, vt := svdMatrix(af.Copy().Data(), m, k)
	cut := rcond
	if cut < 0 {
		// numpy: rcond=None -> eps * max(M, N).
		cut = 2.220446049250313e-16 * float64(max(m, k))
	}
	rank := 0
	sinv := make([]float64, len(s))
	for i, sv := range s {
		if sv > cut*s[0] && sv > 0 {
			sinv[i] = 1 / sv
			rank++
		}
	}
	bn := 1
	if bf.Ndim() == 2 {
		bn = bf.Shape()[1]
	}
	bm := bf.Copy().Data()
	// x = Vt^T * (sinv * (U^T b))
	ub := make([]float64, len(s)*bn)
	for j := 0; j < bn; j++ {
		for i := 0; i < len(s); i++ {
			sum := 0.0
			for r := 0; r < m; r++ {
				bv := 0.0
				if bf.Ndim() == 1 {
					bv = bm[r]
				} else {
					bv = bm[r*bn+j]
				}
				sum += u[r*len(s)+i] * bv
			}
			ub[i*bn+j] = sum * sinv[i]
		}
	}
	x := make([]float64, k*bn)
	for j := 0; j < bn; j++ {
		for c := 0; c < k; c++ {
			sum := 0.0
			for i := 0; i < len(s); i++ {
				sum += vt[i*k+c] * ub[i*bn+j]
			}
			x[c*bn+j] = sum
		}
	}
	// Residuals: only when the system is overdetermined AND full rank
	// (numpy returns an empty array when rank < N).
	var res []float64
	if m > k && rank == k {
		res = make([]float64, bn)
		adata := af.Copy().Data()
		for j := 0; j < bn; j++ {
			for r := 0; r < m; r++ {
				bv := 0.0
				if bf.Ndim() == 1 {
					bv = bm[r]
				} else {
					bv = bm[r*bn+j]
				}
				pred := 0.0
				for c := 0; c < k; c++ {
					pred += adata[r*k+c] * x[c*bn+j]
				}
				res[j] += (bv - pred) * (bv - pred)
			}
		}
	}
	xShape := []int{k}
	if bf.Ndim() == 2 {
		xShape = []int{k, bn}
	}
	svArr := wrap[T](s, []int{len(s)})
	rankArr := numgo.FromSlice([]T{scalarOf[T](float64(rank))}, 1)
	var resArr *numgo.Array[T]
	if len(res) == 0 {
		resArr = numgo.FromSlice([]T{}, 0)
	} else {
		resArr = wrap[T](res, []int{len(res)})
	}
	return LstsqResult[T]{X: wrap[T](x, xShape), Residuals: resArr, Rank: rankArr, SingularValues: svArr}
}

// LstsqResult mirrors numpy.linalg.lstsq's returned tuple.
type LstsqResult[T numgo.Number] struct {
	X              *numgo.Array[T]
	Residuals      *numgo.Array[T]
	Rank           *numgo.Array[T]
	SingularValues *numgo.Array[T]
}

// svdMatrix runs LAPACK Dgesvd (thin SVD) on a row-major m×n matrix,
// returning row-major U (m×minmn), s (minmn, descending) and Vt (minmn×n).
func svdMatrix(a []float64, m, n int) (u, s, vt []float64) {
	minmn := min(m, n)
	cm := append([]float64(nil), a...)
	uCM := make([]float64, m*minmn)
	vtCM := make([]float64, minmn*n)
	sv := make([]float64, minmn)
	impl := gonumImpl{}
	work := []float64{0}
	impl.Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, cm, n, sv, uCM, minmn, vtCM, n, work, -1)
	work = make([]float64, int(work[0]))
	if !impl.Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, cm, n, sv, uCM, minmn, vtCM, n, work, len(work)) {
		panicConvergence("svd")
	}
	return uCM, sv, vtCM
}
