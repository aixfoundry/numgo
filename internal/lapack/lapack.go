// Package lapack provides pure-Go numeric kernels for the linalg package:
// dense LU, Cholesky, Householder QR, one-sided Jacobi SVD, cyclic-Jacobi
// symmetric eigendecomposition, and general eigenvalues via Hessenberg
// reduction plus shifted QR iteration. All kernels are float64, row-major
// (C order) with an explicit leading dimension = row stride, matching numgo's
// buffer layout. Algorithms follow Golub & Van Loan, Matrix Computations;
// no LAPACK Fortran is copied.
package lapack

import (
	"math"
	"math/cmplx"
)

// eps is the double-precision unit roundoff.
const eps = 2.220446049250313e-16

// LU computes the LU factorization with partial pivoting of the n×n
// row-major matrix a (row stride lda): P A = L U, with L unit lower
// triangular stored below the diagonal and U on/above it. piv receives the
// row-swap record: at step k row k is swapped with row piv[k] (LAPACK ipiv
// convention). It returns info = 0 on success or k+1 when a[k][k] is exactly
// singular (U has a zero pivot; L and U are still partially defined).
func LU(a []float64, n, lda int, piv []int) (info int) {
	for k := 0; k < n; k++ {
		// Pivot: largest |entry| in column k at or below the diagonal.
		p := k
		best := math.Abs(a[k*lda+k])
		for i := k + 1; i < n; i++ {
			if v := math.Abs(a[i*lda+k]); v > best {
				best = v
				p = i
			}
		}
		piv[k] = p
		if best == 0 {
			return k + 1
		}
		if p != k {
			for j := 0; j < n; j++ {
				a[k*lda+j], a[p*lda+j] = a[p*lda+j], a[k*lda+j]
			}
		}
		pivv := a[k*lda+k]
		for i := k + 1; i < n; i++ {
			l := a[i*lda+k] / pivv
			a[i*lda+k] = l
			if l != 0 {
				for j := k + 1; j < n; j++ {
					a[i*lda+j] -= l * a[k*lda+j]
				}
			}
		}
	}
	return 0
}

// LuSolve solves A X = B in place given the LU factors from LU: b (n×nrhs,
// row-major with stride nrhs) is overwritten with X. It applies the row
// interchanges, then forward- and back-substitutes. Returns false when lu
// carries a singular factorization (info != 0 from LU).
func LuSolve(lu []float64, n, lda int, piv []int, b []float64, nrhs int) bool {
	for k := 0; k < n; k++ {
		if p := piv[k]; p != k {
			for j := 0; j < nrhs; j++ {
				b[k*nrhs+j], b[p*nrhs+j] = b[p*nrhs+j], b[k*nrhs+j]
			}
		}
	}
	// Forward substitution with unit L: x_i = b_i - Σ L_ij x_j.
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			l := lu[i*lda+j]
			if l != 0 {
				for c := 0; c < nrhs; c++ {
					b[i*nrhs+c] -= l * b[j*nrhs+c]
				}
			}
		}
	}
	// Back substitution with U: U x = y for the already-substituted rows.
	for i := n - 1; i >= 0; i-- {
		u := lu[i*lda+i]
		if u == 0 {
			return false
		}
		for c := 0; c < nrhs; c++ {
			b[i*nrhs+c] /= u
		}
		for r := 0; r < i; r++ {
			u2 := lu[r*lda+i]
			if u2 != 0 {
				for c := 0; c < nrhs; c++ {
					b[r*nrhs+c] -= u2 * b[i*nrhs+c]
				}
			}
		}
	}
	return true
}

// LuInvert computes the inverse of the matrix factored by LU, overwriting lu
// with the n×n inverse (row-major, stride lda). Column j of the inverse is
// the solution of A x = e_j; solves run on a copy of the factors since the
// output overwrites them in place. Returns false when singular.
func LuInvert(lu []float64, n, lda int, piv []int) bool {
	fac := make([]float64, n*n)
	for i := 0; i < n; i++ {
		copy(fac[i*lda:i*lda+n], lu[i*lda:i*lda+n])
	}
	b := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := range b {
			b[i] = 0
		}
		b[j] = 1
		if !LuSolve(fac, n, lda, piv, b, 1) {
			return false
		}
		for i := 0; i < n; i++ {
			lu[i*lda+j] = b[i]
		}
	}
	return true
}

// Potrf computes the Cholesky factorization A = L Lᵀ of the n×n
// positive-definite symmetric matrix, reading and writing only the lower
// triangle of the row-major n×n buffer a. It returns false when a leading
// minor is not positive definite.
func Potrf(a []float64, n int) bool {
	for j := 0; j < n; j++ {
		d := a[j*n+j]
		for k := 0; k < j; k++ {
			d -= a[j*n+k] * a[j*n+k]
		}
		if d <= 0 {
			return false
		}
		d = math.Sqrt(d)
		a[j*n+j] = d
		// Zero the strict upper triangle so the full buffer reads as L.
		for i := 0; i < j; i++ {
			a[i*n+j] = 0
		}
		for i := j + 1; i < n; i++ {
			s := a[i*n+j]
			for k := 0; k < j; k++ {
				s -= a[i*n+k] * a[j*n+k]
			}
			a[i*n+j] = s / d
		}
	}
	return true
}

// Geqrf computes the Householder QR factorization A = Q R of the m×n
// row-major matrix a (row stride lda, m >= n). On exit the upper triangle
// and diagonal of a hold R; below the diagonal a holds the reflector vectors
// (v_i has v_i[i] = 1 implicit). tau holds the n scale factors: each
// reflector is H_i = I - tau_i v vᵀ. This is the same (H, tau) convention
// numpy's mode='raw' exposes.
func Geqrf(a []float64, m, n, lda int, tau []float64) {
	for k := 0; k < n; k++ {
		// Norm of the active column below (and including) the diagonal.
		normx := 0.0
		for i := k; i < m; i++ {
			normx += a[i*lda+k] * a[i*lda+k]
		}
		normx = math.Sqrt(normx)
		if normx == 0 {
			tau[k] = 0
			continue
		}
		alpha := a[k*lda+k]
		beta := -math.Copysign(normx, alpha)
		tau[k] = (beta - alpha) / beta
		s := 1 / (alpha - beta)
		a[k*lda+k] = beta
		for i := k + 1; i < m; i++ {
			a[i*lda+k] *= s
		}
		// Apply H = I - tau v vᵀ to the remaining columns. v has v[k] = 1
		// and v[i] = stored (scaled) entry for i > k.
		for j := k + 1; j < n; j++ {
			sum := a[k*lda+j]
			for i := k + 1; i < m; i++ {
				sum += a[i*lda+k] * a[i*lda+j]
			}
			sum *= tau[k]
			a[k*lda+j] -= sum
			for i := k + 1; i < m; i++ {
				a[i*lda+j] -= sum * a[i*lda+k]
			}
		}
	}
}

// Orgqr generates the m×qCols matrix Q with orthonormal columns from the
// Geqrf output: a holds the reflectors below the diagonal of a factored
// m×n matrix (k = min(m,n) reflectors, row stride n), tau the scale factors.
// The result is written row-major with row stride qCols into q. qCols must
// satisfy k <= qCols <= m; columns beyond the reflector range stay zero and
// receive identity rows (complete mode).
func Orgqr(a []float64, m, n, k int, tau []float64, q []float64, qCols int) {
	// Start from identity, then post-multiply reflectors H_k .. H_0 so that
	// Q = H_0 H_1 ... H_{k-1}.
	for i := 0; i < m; i++ {
		for j := 0; j < qCols; j++ {
			q[i*qCols+j] = 0
		}
		if i < qCols {
			q[i*qCols+i] = 1
		}
	}
	for c := k - 1; c >= 0; c-- {
		tc := tau[c]
		if tc == 0 {
			continue
		}
		for j := 0; j < qCols; j++ {
			// w = vᵀ q[:,j] over rows c..m-1 (v[c] = 1).
			sum := q[c*qCols+j]
			for i := c + 1; i < m; i++ {
				sum += a[i*n+c] * q[i*qCols+j]
			}
			sum *= tc
			q[c*qCols+j] -= sum
			for i := c + 1; i < m; i++ {
				q[i*qCols+j] -= sum * a[i*n+c]
			}
		}
	}
}

// SvdJob selects how much of the SVD factors to produce.
type SvdJob int

const (
	// SVDNone computes singular values only.
	SVDNone SvdJob = iota
	// SVDThin computes the leading min(m,n) columns/rows of U and Vt.
	SVDThin
	// SVDFull computes complete square U (m×m) and Vt (n×n).
	SVDFull
)

// Svd computes the singular value decomposition A = U diag(s) Vᵀ of the
// m×n row-major matrix a (row stride n) with one-sided Jacobi (Hestenes)
// rotations: pairs of columns are orthogonalized until every off-diagonal
// entry of Aᵀ A vanishes to machine precision, then singular values are the
// column norms and U/V are the normalized columns. s is sorted descending.
// u (m×uCols) and vt (vtRows×n, row-major) receive the factors per job;
// s has min(m,n) entries. a is destroyed. Returns false when sweeps do not
// converge. Rank-deficient input converges: orthogonalized zero columns
// produce zero singular values and the corresponding factor columns keep
// their pre-rotation (numerically orthogonal) direction, giving
// reconstruction and orthogonality to ~1e-14 rather than machine zero.
func Svd(a []float64, m, n int, jobU, jobVT SvdJob, u, s, vt []float64) bool {
	k := min(m, n)
	if m < n {
		// One-sided Jacobi needs tall-or-square input; reduce wide matrices
		// by factoring aᵀ: Aᵀ = U2 diag(s) V2ᵀ, hence A = V2 diag(s) U2ᵀ.
		// The caller's Vt is U2ᵀ (thin k×n / full n×n) and the caller's U is
		// V2 (m×k thin / m×m full).
		at := make([]float64, n*m)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				at[j*m+i] = a[i*n+j]
			}
		}
		var u2, vt2 []float64
		u2Cols := k
		vt2Rows := k
		if jobU != SVDNone {
			// U2 of Aᵀ: thin n×k, full n×n.
			if jobU == SVDFull {
				u2Cols = n
			}
			u2 = make([]float64, n*u2Cols)
		}
		if jobVT != SVDNone {
			// V2ᵀ of Aᵀ: thin k×m, full m×m.
			if jobVT == SVDFull {
				vt2Rows = m
			}
			vt2 = make([]float64, vt2Rows*m)
		}
		if !svdTall(at, n, m, jobU, jobVT, u2, s, vt2) {
			return false
		}
		// caller Vt rows are U2's columns: vt (vtRows×n).
		if jobU != SVDNone {
			vtRows := k
			if jobU == SVDFull {
				vtRows = n
			}
			for r := 0; r < vtRows; r++ {
				for c := 0; c < n; c++ {
					vt[r*n+c] = u2[c*u2Cols+r]
				}
			}
		}
		// caller U = V2 = (V2ᵀ)ᵀ, m×vt2Rows.
		if jobVT != SVDNone {
			for r := 0; r < m; r++ {
				for c := 0; c < vt2Rows; c++ {
					u[r*vt2Rows+c] = vt2[c*m+r]
				}
			}
		}
		return true
	}
	return svdTall(a, m, n, jobU, jobVT, u, s, vt)
}

// svdTall is one-sided Jacobi SVD for m >= n. a (m×n, row-major) is
// destroyed. Per job, u (m×cols) and vt (rows×n) receive factors. Singular
// values are sorted descending.
func svdTall(a []float64, m, n int, jobU, jobVT SvdJob, u, s, vt []float64) bool {
	k := n
	// Workspace columns of the current B = A (m×n), normalized per sweep.
	uCols := 0
	if jobU != SVDNone {
		uCols = k
		if jobU == SVDFull {
			uCols = m
		}
	}
	b := make([]float64, m*n)
	copy(b, a)
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1
	}
	const maxSweeps = 60
	for sweep := 0; sweep < maxSweeps; sweep++ {
		off := 0.0
		normB := 0.0
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				// Gram entries of columns p, q.
				var app, aqq, apq float64
				for i := 0; i < m; i++ {
					x, y := b[i*n+p], b[i*n+q]
					app += x * x
					aqq += y * y
					apq += x * y
				}
				normB = math.Max(normB, math.Max(app, aqq))
				if apq == 0 {
					continue
				}
				off += apq * apq
				// Jacobi rotation on columns p, q annihilating apq.
				theta := (aqq - app) / (2 * apq)
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				sin := t * c
				for i := 0; i < m; i++ {
					bp, bq := b[i*n+p], b[i*n+q]
					b[i*n+p] = c*bp - sin*bq
					b[i*n+q] = sin*bp + c*bq
				}
				for i := 0; i < n; i++ {
					vp, vq := v[i*n+p], v[i*n+q]
					v[i*n+p] = c*vp - sin*vq
					v[i*n+q] = sin*vp + c*vq
				}
			}
		}
		if off <= 1e-30*normB {
			break
		}
		if sweep == maxSweeps-1 {
			return false
		}
	}
	// Singular values = column norms; U = normalized columns.
	type pair struct {
		sv  float64
		col int
	}
	ord := make([]pair, k)
	for j := 0; j < n; j++ {
		nrm := 0.0
		for i := 0; i < m; i++ {
			nrm += b[i*n+j] * b[i*n+j]
		}
		ord[j] = pair{math.Sqrt(nrm), j}
	}
	// Sort descending (insertion sort, k small).
	for i := 1; i < k; i++ {
		for j := i; j > 0 && ord[j].sv > ord[j-1].sv; j-- {
			ord[j], ord[j-1] = ord[j-1], ord[j]
		}
	}
	for j, p := range ord {
		s[j] = p.sv
	}
	if jobU != SVDNone {
		for i := 0; i < m; i++ {
			for j := 0; j < uCols; j++ {
				u[i*uCols+j] = 0
			}
		}
		for j, p := range ord {
			nrm := p.sv
			for i := 0; i < m; i++ {
				u[i*uCols+j] = b[i*n+p.col] / nrm
			}
		}
		// Full mode: fill the remaining m-k columns with an orthonormal
		// completion via Gram-Schmidt over identity candidates.
		if jobU == SVDFull && m > n {
			fill := 0
			for e := 0; e < m && fill < m-k; e++ {
				// Try identity vector e_e, orthogonalized against used cols.
				w := make([]float64, m)
				w[e] = 1
				for pass := 0; pass < 2; pass++ {
					for j := 0; j < n+fill; j++ {
						dot := 0.0
						for i := 0; i < m; i++ {
							dot += u[i*uCols+j] * w[i]
						}
						for i := 0; i < m; i++ {
							w[i] -= dot * u[i*uCols+j]
						}
					}
				}
				nrm := 0.0
				for i := 0; i < m; i++ {
					nrm += w[i] * w[i]
				}
				nrm = math.Sqrt(nrm)
				if nrm > 1e-8 {
					for i := 0; i < m; i++ {
						u[i*uCols+n+fill] = w[i] / nrm
					}
					fill++
				}
			}
		}
	}
	if jobVT != SVDNone {
		vtRows := k
		if jobVT == SVDFull {
			vtRows = n
		}
		for i := 0; i < vtRows; i++ {
			for j := 0; j < n; j++ {
				vt[i*n+j] = 0
			}
		}
		// Vt rows are the right singular vectors: row j = column ord[j] of V.
		for r, p := range ord {
			for i := 0; i < n; i++ {
				vt[r*n+i] = v[i*n+p.col]
			}
		}
	}
	return true
}

// Syev computes the eigenvalues and (optionally) eigenvectors of the n×n
// real symmetric row-major matrix a (row stride n) by cyclic Jacobi
// rotations. Eigenvalues are returned ascending in w. When wantVec is true
// a is overwritten with the orthonormal eigenvectors as columns; otherwise a
// is destroyed. Returns false when sweeps do not converge.
func Syev(a []float64, n int, w []float64, wantVec bool) bool {
	v := make([]float64, n*n)
	if wantVec {
		for i := 0; i < n; i++ {
			v[i*n+i] = 1
		}
	}
	const maxSweeps = 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		off := 0.0
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if apq == 0 {
					continue
				}
				off += apq * apq
				theta := (a[q*n+q] - a[p*n+p]) / (2 * apq)
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for i := 0; i < n; i++ {
					aip, aiq := a[i*n+p], a[i*n+q]
					a[i*n+p] = c*aip - s*aiq
					a[i*n+q] = s*aip + c*aiq
				}
				for i := 0; i < n; i++ {
					api := a[p*n+i]
					aqi := a[q*n+i]
					a[p*n+i] = c*api - s*aqi
					a[q*n+i] = s*api + c*aqi
				}
				if wantVec {
					for i := 0; i < n; i++ {
						vp, vq := v[i*n+p], v[i*n+q]
						v[i*n+p] = c*vp - s*vq
						v[i*n+q] = s*vp + c*vq
					}
				}
			}
		}
		// Convergence: off-diagonal mass below eps² relative to the matrix.
		diag := 0.0
		for i := 0; i < n; i++ {
			diag += a[i*n+i] * a[i*n+i]
		}
		if off <= 1e-30*diag {
			break
		}
		if sweep == maxSweeps-1 {
			return false
		}
	}
	for i := 0; i < n; i++ {
		w[i] = a[i*n+i]
	}
	if wantVec {
		copy(a, v)
	}
	// Sort ascending, carrying vectors.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < n; i++ {
		for j := i; j > 0 && w[idx[j]] < w[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	ws := make([]float64, n)
	for i, id := range idx {
		ws[i] = w[id]
	}
	copy(w, ws)
	if wantVec {
		out := make([]float64, n*n)
		for j, id := range idx {
			for i := 0; i < n; i++ {
				out[i*n+j] = v[i*n+id]
			}
		}
		copy(a, out)
	}
	return true
}

// Eig computes the eigenvalues — and, when wantVec, the right eigenvectors —
// of the n×n general real row-major matrix a (row stride n; destroyed).
// wr/wi receive the real/imaginary eigenvalue parts; complex conjugate pairs
// share wr. When wantVec, vr (n×n, row-major with row stride n) receives the
// eigenvectors columnwise, real columns in vr[:,j], conjugate pairs as
// v_j = vr[:,j] + i·vr[:,j+1], v_{j+1} = vr[:,j] − i·vr[:,j+1] (LAPACK
// Dgeev packing). Returns false when the QR iteration fails to converge.
func Eig(a []float64, n int, wr, wi []float64, wantVec bool, vr []float64) bool {
	// Hessenberg reduction by Householder reflectors: v = (1, h[k+2..,k]/s)
	// with s = α − β, P = I − τ v vᵀ, τ = 2/(1 + Σ v_i²). Reflectors are
	// accumulated into z (Schur basis) and their storage zeroed before the
	// QR iteration.
	h := make([]float64, n*n)
	copy(h, a)
	var z []float64
	if wantVec {
		z = make([]float64, n*n)
		for i := 0; i < n; i++ {
			z[i*n+i] = 1
		}
	}
	for k := 0; k < n-2; k++ {
		normx := 0.0
		for i := k + 1; i < n; i++ {
			normx += h[i*n+k] * h[i*n+k]
		}
		normx = math.Sqrt(normx)
		if normx == 0 {
			continue
		}
		alpha := h[(k+1)*n+k]
		beta := -math.Copysign(normx, alpha)
		s := alpha - beta
		if s == 0 {
			continue
		}
		v := make([]float64, n)
		v[k+1] = 1
		vtv := 1.0
		for i := k + 2; i < n; i++ {
			v[i] = h[i*n+k] / s
			vtv += v[i] * v[i]
		}
		tau := 2 / vtv
		// Left update runs on the original column k too; then the reduced
		// subdiagonal (beta) is written and the column zeroed — this kernel
		// keeps no reflector storage.
		for j := 0; j < n; j++ {
			sum := v[k+1] * h[(k+1)*n+j]
			for i := k + 2; i < n; i++ {
				sum += v[i] * h[i*n+j]
			}
			sum *= tau
			h[(k+1)*n+j] -= sum * v[k+1]
			for i := k + 2; i < n; i++ {
				h[i*n+j] -= sum * v[i]
			}
		}
		h[(k+1)*n+k] = beta
		for i := k + 2; i < n; i++ {
			h[i*n+k] = 0
		}
		for i := 0; i < n; i++ {
			sum := h[i*n+k+1] * v[k+1]
			for j := k + 2; j < n; j++ {
				sum += h[i*n+j] * v[j]
			}
			sum *= tau
			h[i*n+k+1] -= sum * v[k+1]
			for j := k + 2; j < n; j++ {
				h[i*n+j] -= sum * v[j]
			}
		}
		// Accumulate the similarity into the Schur basis: z := z P.
		if z != nil {
			for r := 0; r < n; r++ {
				sum := z[r*n+k+1] * v[k+1]
				for j := k + 2; j < n; j++ {
					sum += z[r*n+j] * v[j]
				}
				sum *= tau
				z[r*n+k+1] -= sum * v[k+1]
				for j := k + 2; j < n; j++ {
					z[r*n+j] -= sum * v[j]
				}
			}
		}
	}
	if !hessenbergQR(h, n, z) {
		return false
	}
	// Extract eigenvalues from the triangular blocks.
	for i := 0; i < n; i++ {
		wr[i], wi[i] = 0, 0
	}
	for i := 0; i < n; i++ {
		if wi[i] != 0 { // already consumed by its pair
			continue
		}
		if i == n-1 {
			wr[i] = h[i*n+i]
			continue
		}
		a11, a12 := h[i*n+i], h[i*n+i+1]
		a21 := h[(i+1)*n+i]
		a22 := h[(i+1)*n+(i+1)]
		// 1×1 block when the subdiagonal is negligible and the block is not
		// a hidden nilpotent pair (both diagonals zero with a12 ≠ 0 keeps
		// the 2×2 path so the discriminant decides).
		s := math.Abs(a11) + math.Abs(a22)
		if a21 == 0 && !(a11 == 0 && a22 == 0 && a12 != 0) || s > 0 && math.Abs(a21) <= eps*s && !(a11 == 0 && a22 == 0 && a12 != 0) {
			wr[i] = a11
			continue
		}
		// 2×2 block: conjugate pair or real pair via the stable quadratic.
		tr, det := a11+a22, a11*a22-a12*a21
		disc := tr*tr/4 - det
		if disc >= 0 {
			sq := math.Sqrt(disc)
			r1 := tr/2 + math.Copysign(sq, tr/2)
			var r2 float64
			if r1 == 0 {
				r2 = 0
			} else {
				r2 = det / r1
			}
			wr[i], wr[i+1] = r1, r2
		} else {
			wr[i], wr[i+1] = tr/2, tr/2
			wi[i] = math.Sqrt(-disc)
			wi[i+1] = -math.Sqrt(-disc)
		}
		i++
	}
	if !wantVec {
		return true
	}
	// Eigenvectors: inverse iteration on the quasi-triangular H with a
	// complex shift per eigenvalue, then back-transform through the Schur
	// basis Z.
	for j := 0; j < n; j++ {
		if wi[j] == 0 {
			v := eigVectorReal(h, n, complex(wr[j], 0))
			for i := 0; i < n; i++ {
				vr[i*n+j] = realOf(v[i])
			}
		} else {
			// LAPACK packing: col_j + i·col_{j+1} from the +imag eigenvector;
			// compute the +imag vector whichever slot the pair occupies.
			var vPlus []complex128
			if wi[j] > 0 {
				vPlus = eigVectorReal(h, n, complex(wr[j], wi[j]))
			} else {
				vPlus = eigVectorReal(h, n, complex(wr[j], -wi[j]))
			}
			for i := 0; i < n; i++ {
				vr[i*n+j] = real(vPlus[i])
				vr[i*n+j+1] = imag(vPlus[i])
			}
			j++
		}
	}
	// Back-transform: eigenvectors of H map to eigenvectors of A via the
	// Schur vectors Z (columns of the similarity that reduced A to T).
	out := make([]float64, n*n)
	for col := 0; col < n; col++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for kk := 0; kk < n; kk++ {
				sum += z[i*n+kk] * vr[kk*n+col]
			}
			out[i*n+col] = sum
		}
	}
	copy(vr, out)
	return true
}

func realOf(z complex128) float64 { return real(z) }

// hessenbergQR runs shifted QR iterations on the n×n upper Hessenberg h
// (row-major, destroyed), accumulating the orthogonal transformations into
// z (may be nil). Wilkinson shifts from the trailing 2×2; the iteration
// deflates 1×1 and 2×2 blocks at the bottom. It leaves h quasi-triangular:
// real eigenvalues on the diagonal, conjugate pairs in 2×2 blocks whose
// subdiagonal is kept (well above eps).
func hessenbergQR(h []float64, n int, z []float64) bool {
	const maxIter = 5000
	hi := n - 1 // active bottom index; eigenvalue extraction range [0..hi]
	iter := 0
	for hi > 0 {
		// Zero tiny subdiagonals and find the active top of the block.
		l := hi
		for l > 0 {
			s := math.Abs(h[(l-1)*n+l-1]) + math.Abs(h[l*n+l])
			if s == 0 {
				s = 1
			}
			if math.Abs(h[l*n+l-1]) <= eps*s {
				h[l*n+l-1] = 0
				break
			}
			l--
		}
		if l == hi {
			// 1×1 block deflated.
			hi--
			iter = 0
			continue
		}
		if l == hi-1 {
			// Bottom 2×2 block. Complex conjugate pair: the real-shift
			// iteration can never reduce it further — accept it as an
			// eigenvalue block, keeping the subdiagonal so extraction sees
			// the full block. Real pair: triangularize by a Givens
			// similarity Qᵀ A Q whose first column is the normalized
			// eigenvector (a12, λ₁−a11) — the subdiagonal then vanishes
			// exactly. The rotation accumulates into z.
			a11, a12 := h[l*n+l], h[l*n+hi]
			a21, a22 := h[hi*n+l], h[hi*n+hi]
			tr, det := a11+a22, a11*a22-a12*a21
			disc := tr*tr/4 - det
			if disc < 0 {
				hi = l - 1
				iter = 0
				continue
			}
			sq := math.Sqrt(disc)
			lam1 := tr/2 + math.Copysign(sq, tr/2)
			r := math.Hypot(a12, lam1-a11)
			var c, s float64 = 1, 0
			if r != 0 {
				c = a12 / r
				s = (a11 - lam1) / r
			}
			// Q = [[c, s], [-s, c]]: rows via Qᵀ (c·r_l − s·r_hi),
			// columns via Q (c·c_l − s·c_hi); z := z Q (same col mix).
			for j := 0; j < n; j++ {
				x, y := h[l*n+j], h[hi*n+j]
				h[l*n+j] = c*x - s*y
				h[hi*n+j] = s*x + c*y
			}
			for i := 0; i < n; i++ {
				x, y := h[i*n+l], h[i*n+hi]
				h[i*n+l] = c*x - s*y
				h[i*n+hi] = s*x + c*y
			}
			if z != nil {
				for i := 0; i < n; i++ {
					x, y := z[i*n+l], z[i*n+hi]
					z[i*n+l] = c*x - s*y
					z[i*n+hi] = s*x + c*y
				}
			}
			h[hi*n+l] = 0
			hi -= 2
			iter = 0
			continue
		}
		// Wilkinson shift from the trailing 2×2 of the active block.
		a11, a12 := h[(hi-1)*n+hi-1], h[(hi-1)*n+hi]
		a21, a22 := h[hi*n+hi-1], h[hi*n+hi]
		tr, det := a11+a22, a11*a22-a12*a21
		disc := tr*tr/4 - det
		var shift float64
		if disc >= 0 {
			sq := math.Sqrt(disc)
			r1 := tr/2 + math.Copysign(sq, tr/2)
			r2 := tr - r1
			if math.Abs(r1-a22) < math.Abs(r2-a22) {
				shift = r1
			} else {
				shift = r2
			}
		} else {
			shift = tr / 2
			// Exceptional shift every 10 stalled iterations (EISPACK hqr
			// trick): a real shift cannot deflate complex-pair blocks, so
			// occasionally perturb to break symmetric limit cycles.
			if iter%10 == 0 {
				shift = 1.5 * (math.Abs(h[hi*n+hi-1]) + math.Abs(h[(hi-1)*n+hi-2]))
			} else if iter%10 == 5 {
				// Second flavor: random-ish perturbation off the diagonal.
				shift = tr/2 + 0.75*math.Abs(h[hi*n+hi-1])
			}
		}
		iter++
		if iter > maxIter {
			return false
		}
		// Implicit single-shift QR step on rows l..hi via Givens rotations.
		// Factor M = H[l..hi] − σI = QR by Givens, then H' = RQ + σI. The
		// similarity G acts on full rows l..hi (all n columns) and full
		// columns l..hi (all n rows) — the deflated coupling entries are
		// zero, so only the window block changes in h, but the rotations
		// are applied to h's bordering rows/cols via cs/sn below and to z.
		sz := hi - l + 1
		m := make([]float64, sz*sz)
		for i := 0; i < sz; i++ {
			for j := 0; j < sz; j++ {
				v := h[(l+i)*n+(l+j)]
				if i == j {
					v -= shift
				}
				m[i*sz+j] = v
			}
		}
		cs := make([]float64, sz-1)
		sn := make([]float64, sz-1)
		for i := 0; i < sz-1; i++ {
			aa, bb := m[i*sz+i], m[(i+1)*sz+i]
			r := math.Hypot(aa, bb)
			if r == 0 {
				cs[i], sn[i] = 1, 0
				continue
			}
			cs[i], sn[i] = aa/r, bb/r
			for j := i; j < sz; j++ {
				x, y := m[i*sz+j], m[(i+1)*sz+j]
				m[i*sz+j] = cs[i]*x + sn[i]*y
				m[(i+1)*sz+j] = -sn[i]*x + cs[i]*y
			}
		}
		// RQ: Q = G₀ᵀ G₁ᵀ … G_{sz−2}ᵀ, so the rotations apply on the right
		// in forward order i = 0..sz−2 (reverse order would transpose the
		// product and break the similarity).
		for i := 0; i <= sz-2; i++ {
			c, s := cs[i], sn[i]
			for r := 0; r <= i+1; r++ {
				x, y := m[r*sz+i], m[r*sz+i+1]
				m[r*sz+i] = c*x + s*y
				m[r*sz+i+1] = -s*x + c*y
			}
		}
		// Row rotations (G) over ALL columns, then column rotations (Gᵀ)
		// over ALL rows: h := Gᵀᵀ... explicitly h := Gᵀ(H)G with Qᵀ =
		// G_{sz−2}…G₀ applied to rows and Q = G₀ᵀ…G_{sz−2}ᵀ to columns.
		// This is the RQ iteration in place; since h still holds H (σ was
		// only subtracted in the scratch window m), the result is already
		// QᵀHQ = RQ + σ — NO shift add-back here.
		for i := 0; i <= sz-2; i++ {
			c, s := cs[i], sn[i]
			for j := 0; j < n; j++ {
				x, y := h[(l+i)*n+j], h[(l+i+1)*n+j]
				h[(l+i)*n+j] = c*x + s*y
				h[(l+i+1)*n+j] = -s*x + c*y
			}
		}
		for i := 0; i <= sz-2; i++ {
			c, s := cs[i], sn[i]
			for r := 0; r < n; r++ {
				x, y := h[r*n+l+i], h[r*n+l+i+1]
				h[r*n+l+i] = c*x + s*y
				h[r*n+l+i+1] = -s*x + c*y
			}
		}
		// z accumulates Q (columns i, i+1 mix per Gᵢᵀ), forward order.
		if z != nil {
			for i := 0; i <= sz-2; i++ {
				c, s := cs[i], sn[i]
				for r := 0; r < n; r++ {
					x, y := z[r*n+l+i], z[r*n+l+i+1]
					z[r*n+l+i] = c*x + s*y
					z[r*n+l+i+1] = -s*x + c*y
				}
			}
		}
	}
	return true
}

// eigVectorReal finds a unit eigenvector of the quasi-triangular H for the
// complex eigenvalue λ by inverse iteration: solve (H−λ'I)x = b with complex
// Gaussian elimination and partial pivoting, iterating a few passes. λ' is
// perturbed with a tiny imaginary part (real shift + eps·i) — the classic
// inverse-iteration trick: a purely real shift on a real matrix is exactly
// singular and the unperturbed solve returns a large but wrong-direction x.
func eigVectorReal(h []float64, n int, lambda complex128) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(1, 0)
	}
	shift := lambda + complex(0, eps*1024)
	// A couple of inverse-iteration passes converge fast for isolated λ.
	const passes = 3
	for p := 0; p < passes; p++ {
		m := make([]complex128, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				m[i*n+j] = complex(h[i*n+j], 0)
				if i == j {
					m[i*n+j] -= shift
				}
			}
		}
		// Perturb a zero pivot to eps (inverse-iteration standard) so an
		// exact eigenvalue still yields a bounded solution.
		// Partial-pivot Gaussian elimination on (m | x).
		lu := make([]complex128, n*n)
		copy(lu, m)
		piv := make([]int, n)
		for k := 0; k < n; k++ {
			p := k
			best := cmplx.Abs(lu[k*n+k])
			for i := k + 1; i < n; i++ {
				if v := cmplx.Abs(lu[i*n+k]); v > best {
					best = v
					p = i
				}
			}
			piv[k] = p
			if best == 0 {
				lu[k*n+k] = complex(eps, 0)
			} else if p != k {
				for j := 0; j < n; j++ {
					lu[k*n+j], lu[p*n+j] = lu[p*n+j], lu[k*n+j]
				}
			}
			pv := lu[k*n+k]
			for i := k + 1; i < n; i++ {
				l := lu[i*n+k] / pv
				lu[i*n+k] = 0
				for j := k + 1; j < n; j++ {
					lu[i*n+j] -= l * lu[k*n+j]
				}
			}
		}
		// Apply the same pivots to x and solve.
		for k := 0; k < n; k++ {
			if p := piv[k]; p != k {
				x[k], x[p] = x[p], x[k]
			}
		}
		for i := 1; i < n; i++ {
			for j := 0; j < i; j++ {
				x[i] -= lu[i*n+j] * x[j]
			}
		}
		for i := n - 1; i >= 0; i-- {
			for j := i + 1; j < n; j++ {
				x[i] -= lu[i*n+j] * x[j]
			}
			x[i] /= lu[i*n+i]
		}
		// Normalize; pick the phase making the largest component real
		// positive (stable across runs, helps conjugate packing).
		nrm := 0.0
		maxIdx := 0
		for i, v := range x {
			if a := cmplx.Abs(v); a > nrm {
				nrm = a
				maxIdx = i
			}
		}
		if nrm == 0 {
			for i := range x {
				x[i] = 1
			}
			continue
		}
		ph := x[maxIdx] / complex(cmplx.Abs(x[maxIdx]), 0)
		for i := range x {
			x[i] = x[i] / ph / complex(nrm, 0)
		}
	}
	return x
}
