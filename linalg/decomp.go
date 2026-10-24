package linalg

import (
	"fmt"

	"github.com/kaisawind/numgo"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Matrix decompositions: Cholesky, QR, SVD, Eig/Eigvals, Eigh/Eigvalsh.
// Backend: gonum's Dpotrf, Dgeqrf+Dorgqr, Dgesvd, Dgeev, Dsyev. Batched
// inputs loop over leading dims (shortcut: batched BLAS later). Gonum
// buffers are row-major (C order), matching numgo layout.

// Cholesky returns the lower-triangular Cholesky factor L of a (..., M, M)
// with a = L @ L.T. Non-positive-definite input panics (Singular matrix).
// Backend: LAPACK Dpotrf (lower triangle).
func Cholesky[T numgo.Number](a *numgo.Array[T]) *numgo.Array[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m != n {
		panicSquare("cholesky", shape)
	}
	out := numgo.Zeros[T](shape...)
	impl := gonumImpl{}
	eachBatch(shape, func(batch []int) {
		mat := aMatrixAt(af, batch, m, m)
		if !impl.Dpotrf(blas.Lower, m, mat, m) {
			panicSingular()
		}
		// Zero the upper triangle.
		for i := 0; i < m; i++ {
			for j := i + 1; j < m; j++ {
				mat[i*m+j] = 0
			}
		}
		setMatAt(out, batch, m, m, mat)
	})
	return out
}

// QRMode selects how much of Q and R QR returns, mirroring numpy's mode
// strings.
type QRMode int

const (
	// QRReduced (numpy mode='reduced', the default) returns Q (M,K) and
	// R (K,N) with K = min(M,N).
	QRReduced QRMode = iota
	// QRComplete (numpy mode='complete') returns Q (M,M) and R (M,N).
	QRComplete
	// QRRaw (numpy mode='raw'): returns H (K,N) — R in its upper triangle
	// with the elementary reflectors below the diagonal — in R, and the tau
	// scale factors (K) as a vector in Q.
	QRRaw
)

// QRResult holds the QR factorization outputs.
type QRResult[T numgo.Number] struct {
	Q *numgo.Array[T]
	R *numgo.Array[T]
}

// QR computes the QR factorization of a (..., M, N) with M >= N. Backend:
// LAPACK Dgeqrf + Dorgqr (complete: Dorgqr over all m columns; reduced: the
// first k columns). shortcut: wide matrices (m < n) need the LQ path —
// not implemented this phase and panic.
func QR[T numgo.Number](a *numgo.Array[T], mode QRMode) QRResult[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m < n {
		panicDims("qr", "qr requires a 2-D array with rows >= columns (numpy's mode set)")
	}
	k := min(m, n)
	qCols, rRows := k, k
	if mode == QRComplete {
		qCols, rRows = m, m
	}
	qOut := numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), m, qCols)...)
	if mode == QRRaw {
		qOut = numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), k)...)
	}
	rOut := numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), rRows, n)...)
	impl := gonumImpl{}
	eachBatch(shape, func(batch []int) {
		mat := aMatrixAt(af, batch, m, n)
		tau := make([]float64, n)
		work := []float64{0}
		impl.Dgeqrf(m, n, mat, n, tau, work, -1)
		work = make([]float64, int(work[0]))
		impl.Dgeqrf(m, n, mat, n, tau, work, len(work))
		switch mode {
		case QRRaw:
			// numpy's mode='raw' returns (H, tau): H carries R in its upper
			// triangle plus the reflector vectors below the diagonal. We
			// return H (K,N) in R and tau (K) as a vector in Q.
			h := make([]float64, k*n)
			for i := 0; i < k; i++ {
				copy(h[i*n:(i+1)*n], mat[i*n:(i+1)*n])
			}
			setMatAt(rOut, batch, k, n, h)
			setVecAt(qOut, batch, tau)
		default:
			// Build Q from the reflectors. Dorgqr expects an m×qCols matrix
			// in C order (leading dim qCols >= k), so copy row slices of
			// length qCols from mat (leading dim n). For complete mode
			// qCols == m >= n and the missing reflector columns are zero.
			qMat := make([]float64, m*qCols)
			for i := 0; i < m; i++ {
				for j := 0; j < qCols; j++ {
					if j < n {
						qMat[i*qCols+j] = mat[i*n+j]
					}
				}
			}
			work2 := []float64{0}
			impl.Dorgqr(m, qCols, k, qMat, qCols, tau, work2, -1)
			work2 = make([]float64, int(work2[0]))
			impl.Dorgqr(m, qCols, k, qMat, qCols, tau, work2, len(work2))
			setMatAt(qOut, batch, m, qCols, qMat)
			// R: upper rRows×n part of mat with zeros below the diagonal.
			r := make([]float64, rRows*n)
			copy(r, mat[:rRows*n])
			for i := 1; i < rRows; i++ {
				for j := 0; j < min(i, n); j++ {
					r[i*n+j] = 0
				}
			}
			setMatAt(rOut, batch, rRows, n, r)
		}
	})
	return QRResult[T]{Q: qOut, R: rOut}
}

// SVDResult holds the singular value decomposition outputs.
type SVDResult[T numgo.Number] struct {
	U  *numgo.Array[T]
	S  *numgo.Array[T]
	Vt *numgo.Array[T]
}

// Svd computes the singular value decomposition a = U @ diag(S) @ Vt of
// a (..., M, N). S has shape (..., K) with K = min(M,N) in descending order.
// With fullMatrices, U is (..., M, M) and Vt is (..., N, N); otherwise the
// thin factors (..., M, K) and (..., K, N). With computeUV false only S is
// populated. Backend: LAPACK Dgesvd.
func Svd[T numgo.Number](a *numgo.Array[T], fullMatrices bool, computeUV bool) SVDResult[T] {
	af := f64(a)
	shape := af.Shape()
	if af.Ndim() < 2 {
		panicDims("svd", "svd requires at least 2 dimensions")
	}
	m, n := last2(shape)
	k := min(m, n)
	uCols, vtRows := k, k
	if fullMatrices {
		uCols, vtRows = m, n
	}
	sShape := append(append([]int(nil), batchShape(shape)...), k)
	sOut := numgo.Zeros[T](sShape...)
	var uOut, vtOut *numgo.Array[T]
	if computeUV {
		uOut = numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), m, uCols)...)
		vtOut = numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), vtRows, n)...)
	}
	impl := gonumImpl{}
	jobU, jobVT := lapack.SVDNone, lapack.SVDNone
	if computeUV {
		if fullMatrices {
			jobU, jobVT = lapack.SVDAll, lapack.SVDAll
		} else {
			jobU, jobVT = lapack.SVDStore, lapack.SVDStore
		}
	}
	eachBatch(shape, func(batch []int) {
		cm := aMatrixAt(af, batch, m, n)
		sv := make([]float64, k)
		var uMat, vtMat []float64
		ldu, ldvt := 1, 1
		if computeUV {
			uMat = make([]float64, m*uCols)
			vtMat = make([]float64, vtRows*n)
			// C-order leading dims: U rows stride uCols, Vt rows stride n.
			// Dgesvd with SVDStore wants ldu >= minmn rows of U; with
			// SVDAll ldu >= m.
			ldu = uCols
			ldvt = n
		} else {
			uMat, vtMat = make([]float64, 1), make([]float64, 1)
		}
		work := []float64{0}
		impl.Dgesvd(jobU, jobVT, m, n, cm, n, sv, uMat, ldu, vtMat, ldvt, work, -1)
		work = make([]float64, int(work[0]))
		if !impl.Dgesvd(jobU, jobVT, m, n, cm, n, sv, uMat, ldu, vtMat, ldvt, work, len(work)) {
			panicConvergence("svd")
		}
		base := linearIndex(sShape, append(append([]int(nil), batch...), 0))
		for i, v := range sv {
			sOut.Data()[base+i] = scalarOf[T](v)
		}
		if computeUV {
			setMatAt(uOut, batch, m, uCols, uMat)
			setMatAt(vtOut, batch, vtRows, n, vtMat)
		}
	})
	return SVDResult[T]{U: uOut, S: sOut, Vt: vtOut}
}

// EigResult holds the general eigen decomposition. Eigenproblems of real
// non-symmetric matrices have complex results, so Values and Vectors are
// always complex128, regardless of T (numpy upcasts too).
type EigResult[T numgo.Number] struct {
	Values  *numgo.Array[complex128]
	Vectors *numgo.Array[complex128]
}

// Eig computes the right eigenvalues and eigenvectors of a general square
// (..., M, M) matrix. Results are complex128 even for real T because real
// matrices may have complex conjugate eigen pairs. With computeVectors false
// Vectors is nil. Backend: LAPACK Dgeev.
func Eig[T numgo.Number](a *numgo.Array[T], computeVectors bool) EigResult[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m != n {
		panicSquare("eig", shape)
	}
	impl := gonumImpl{}
	jobvr := lapack.RightEVNone
	var vOut *numgo.Array[complex128]
	if computeVectors {
		jobvr = lapack.RightEVCompute
		vOut = numgo.Zeros[complex128](shape...)
	}
	wOut := numgo.Zeros[complex128](append(append([]int(nil), batchShape(shape)...), m)...)
	eachBatch(shape, func(batch []int) {
		cm := aMatrixAt(af, batch, m, m)
		wr, wi := make([]float64, m), make([]float64, m)
		var vr []float64
		if computeVectors {
			vr = make([]float64, m*m)
		}
		work := []float64{0}
		impl.Dgeev(lapack.LeftEVNone, jobvr, m, cm, m, wr, wi, nil, max(m, 1), vr, max(m, 1), work, -1)
		work = make([]float64, int(work[0]))
		first := impl.Dgeev(lapack.LeftEVNone, jobvr, m, cm, m, wr, wi, nil, max(m, 1), vr, max(m, 1), work, len(work))
		if first != 0 {
			panicConvergence("eig")
		}
		wb := linearIndex(wOut.Shape(), append(append([]int(nil), batch...), 0))
		for i := range wr {
			wOut.Data()[wb+i] = complex(wr[i], wi[i])
		}
		if computeVectors {
			// LAPACK packs conjugate pairs: v_j = VR[:,j] + i*VR[:,j+1] and
			// v_{j+1} = VR[:,j] - i*VR[:,j+1]. VR is stored row-major with
			// leading dim m, so column j is vr[j*m:(j+1)*m].
			for j := 0; j < m; j++ {
				if wi[j] == 0 {
					for i := 0; i < m; i++ {
						setBatch3(vOut, batch, i, j, complex(vr[j*m+i], 0))
					}
					continue
				}
				for i := 0; i < m; i++ {
					setBatch3(vOut, batch, i, j, complex(vr[j*m+i], vr[(j+1)*m+i]))
					setBatch3(vOut, batch, i, j+1, complex(vr[j*m+i], -vr[(j+1)*m+i]))
				}
				j++
			}
		}
	})
	return EigResult[T]{Values: wOut, Vectors: vOut}
}

// setBatch3 writes v at (batch..., i, j) of an (..., m, n) array.
func setBatch3(a *numgo.Array[complex128], batch []int, i, j int, v complex128) {
	coords := append(append([]int(nil), batch...), i, j)
	a.Set(coords, v)
}

// Eigvals returns only the eigenvalues of a general square matrix (complex).
func Eigvals[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return Eig[T](a, false).Values
}

// eighUplo validates the UPLO selector ("L" lower — the default — or "U").
func eighUplo(uplo string) blas.Uplo {
	switch uplo {
	case "L", "l", "":
		return blas.Lower
	case "U", "u":
		return blas.Upper
	default:
		panicDims("eigh", fmt.Sprintf("UPLO must be 'L' or 'U', got %q", uplo))
		return 0
	}
}

// Eigh computes the eigenvalues and eigenvectors of a real symmetric
// (..., M, M) matrix, reading only the selected triangle (uplo "L" or "U").
// Eigenvalues are ascending. Values and Vectors keep element type T.
// Backend: LAPACK Dsyev.
func Eigh[T numgo.Number](a *numgo.Array[T], computeVectors bool, uplo string) EighResult[T] {
	af := f64(a)
	shape := af.Shape()
	m, n := last2(shape)
	if af.Ndim() < 2 || m != n {
		panicSquare("eigh", shape)
	}
	impl := gonumImpl{}
	up := eighUplo(uplo)
	jobz := lapack.EVNone
	var vOut *numgo.Array[T]
	if computeVectors {
		jobz = lapack.EVCompute
		vOut = numgo.Zeros[T](shape...)
	}
	wOut := numgo.Zeros[T](append(append([]int(nil), batchShape(shape)...), m)...)
	eachBatch(shape, func(batch []int) {
		cm := aMatrixAt(af, batch, m, m)
		w := make([]float64, m)
		work := []float64{0}
		impl.Dsyev(jobz, up, m, cm, m, w, work, -1)
		work = make([]float64, int(work[0]))
		if !impl.Dsyev(jobz, up, m, cm, m, w, work, len(work)) {
			panicConvergence("eigh")
		}
		wb := linearIndex(wOut.Shape(), append(append([]int(nil), batch...), 0))
		for i, v := range w {
			wOut.Data()[wb+i] = scalarOf[T](v)
		}
		if computeVectors {
			setMatAt(vOut, batch, m, m, cm)
		}
	})
	return EighResult[T]{Values: wOut, Vectors: vOut}
}

// EighResult holds the symmetric eigen decomposition outputs.
type EighResult[T numgo.Number] struct {
	Values  *numgo.Array[T]
	Vectors *numgo.Array[T]
}

// Eigvalsh returns only the eigenvalues of a real symmetric matrix.
func Eigvalsh[T numgo.Number](a *numgo.Array[T], uplo string) *numgo.Array[T] {
	return Eigh[T](a, false, uplo).Values
}
