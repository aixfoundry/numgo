package lapack

import (
	"math"
	"testing"
)

// tol for kernel checks.
const tol = 1e-9

func eq(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		d := math.Abs(got[i] - want[i])
		if d > tol*(1+math.Abs(want[i])) {
			t.Fatalf("%s[%d] = %v, want %v (diff %v)", name, i, got[i], want[i], d)
		}
	}
}

func id(n int) []float64 {
	e := make([]float64, n*n)
	for i := range n {
		e[i*n+i] = 1
	}
	return e
}

func matmul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			s := 0.0
			for c := range k {
				s += a[i*k+c] * b[c*n+j]
			}
			out[i*n+j] = s
		}
	}
	return out
}

// rowMajor transposes a flat row-major m×n buffer (helper for column ops).
func TestLUSolve(t *testing.T) {
	a := []float64{3, 1, 1, 2}
	piv := make([]int, 2)
	if info := LU(a, 2, 2, piv); info != 0 {
		t.Fatalf("info = %d", info)
	}
	b := []float64{9, 8}
	if !LuSolve(a, 2, 2, piv, b, 1) {
		t.Fatal("solve failed")
	}
	eq(t, "lu solve", b, []float64{2, 3})
}

func TestLUSingular(t *testing.T) {
	a := []float64{1, 2, 2, 4}
	piv := make([]int, 2)
	if info := LU(a, 2, 2, piv); info == 0 {
		t.Fatal("expected singular info")
	}
}

func TestLUInvert(t *testing.T) {
	orig := []float64{2.1416476008704612, -0.4064150163846156, 0.6159794225754956, 1.1289722927208916}
	lu := append([]float64(nil), orig...)
	piv := make([]int, 2)
	if info := LU(lu, 2, 2, piv); info != 0 {
		t.Fatalf("info = %d", info)
	}
	if !LuInvert(lu, 2, 2, piv) {
		t.Fatal("invert failed")
	}
	eq(t, "lu invert roundtrip", matmul(orig, lu, 2, 2, 2), id(2))
}

func TestPotrf(t *testing.T) {
	spd := []float64{4, 2, 2, 3}
	if !Potrf(spd, 2) {
		t.Fatal("potrf failed")
	}
	// L = [[2,0],[1, sqrt2]]
	eq(t, "chol", spd, []float64{2, 0, 1, math.Sqrt2})
}

func TestGeqrfOrgqr(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6}
	m, n := 3, 2
	q := make([]float64, m*m)
	tau := make([]float64, n)
	Geqrf(a, m, n, n, tau)
	Orgqr(a, m, n, n, tau, q, m)
	// Orthogonality.
	qtq := matmul(transpose(q, m, m), q, m, m, m)
	eq(t, "Q^T Q", qtq, id(m))
	// Reconstruction: R = upper triangle of a, Q R = A.
	r := make([]float64, m*n)
	copy(r, a)
	for i := 1; i < m; i++ {
		for j := 0; j < min(i, n); j++ {
			r[i*n+j] = 0
		}
	}
	orig := []float64{1, 2, 3, 4, 5, 6}
	eq(t, "QR recon", matmul(q, r, m, m, n), orig)
}

func transpose(a []float64, m, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			out[j*m+i] = a[i*n+j]
		}
	}
	return out
}

func TestSvdKnownValues(t *testing.T) {
	// Known SVD: A = [[3,0],[0,2]] → s = {3,2}.
	a := []float64{3, 0, 0, 2}
	s := make([]float64, 2)
	u := make([]float64, 4)
	vt := make([]float64, 4)
	if !Svd(a, 2, 2, SVDThin, SVDThin, u, s, vt) {
		t.Fatal("svd failed")
	}
	eq(t, "s", s, []float64{3, 2})
	// Reconstruction with factors (order of tied columns is algorithm-defined;
	// check via factors) — use a non-tied matrix instead.
	b := []float64{1, 2, 3, 4, 5, 6}
	sb := make([]float64, 2)
	ub := make([]float64, 6)
	vtb := make([]float64, 6)
	if !Svd(b, 3, 2, SVDThin, SVDThin, ub, sb, vtb) {
		t.Fatal("svd failed")
	}
	eq(t, "s tall", sb, []float64{9.525518091565106, 0.5143005805324749})
	d := []float64{sb[0], 0, 0, sb[1]}
	recon := matmul(matmul(ub, d, 3, 2, 2), vtb, 3, 2, 2)
	eq(t, "recon", recon, b)
	// Orthogonality: UᵀU = I, Vt Vtᵀ = I.
	utU := matmul(transpose(ub, 3, 2), ub, 2, 3, 2)
	eq(t, "U^T U", utU, id(2))
	vvt := matmul(transpose(vtb, 2, 2), vtb, 2, 2, 2)
	eq(t, "Vt Vt^T", vvt, id(2))
}

func TestSvdWide(t *testing.T) {
	// Wide 2×3 via transpose path.
	a := []float64{1, 2, 3, 4, 5, 6}
	s := make([]float64, 2)
	u := make([]float64, 4)
	vt := make([]float64, 6)
	if !Svd(a, 2, 3, SVDThin, SVDThin, u, s, vt) {
		t.Fatal("svd failed")
	}
	eq(t, "s wide", s, []float64{9.5080320006957, 0.7728696356734853})
	// Reconstruction A = U diag(s) Vt (thin: 2×2 diag between factors).
	d := []float64{s[0], 0, 0, s[1]}
	recon := matmul(matmul(u, d, 2, 2, 2), vt, 2, 2, 3)
	eq(t, "recon wide", recon, a)
}

func TestSvdRankDeficient(t *testing.T) {
	a := []float64{1, 1, 1, 1, 1, 1}
	s := make([]float64, 2)
	u := make([]float64, 6)
	vt := make([]float64, 4)
	if !Svd(a, 3, 2, SVDThin, SVDThin, u, s, vt) {
		t.Fatal("svd failed")
	}
	// s = {sqrt6, 0}; reconstruction and orthogonality still hold.
	eq(t, "rd s", s, []float64{math.Sqrt(6), 0})
	d := []float64{s[0], 0, 0, s[1]}
	recon := matmul(matmul(u, d, 3, 2, 2), vt, 3, 2, 2)
	eq(t, "rd recon", recon, a)
	utU := matmul(transpose(u, 3, 2), u, 2, 3, 2)
	eq(t, "rd U^T U", utU, id(2))
}

func TestSyevKnown(t *testing.T) {
	a := []float64{2, 1, 0, 1, 2, 1, 0, 1, 2}
	w := make([]float64, 3)
	if !Syev(a, 3, w, true) {
		t.Fatal("syev failed")
	}
	eq(t, "syev w", w, []float64{2 - math.Sqrt2, 2, 2 + math.Sqrt2})
	// Residual: A v = λ v per column.
	v := a
	for j := range 3 {
		for i := range 3 {
			sum := 0.0
			for k := range 3 {
				sum += []float64{2, 1, 0, 1, 2, 1, 0, 1, 2}[i*3+k] * v[k*3+j]
			}
			if d := math.Abs(sum - w[j]*v[i*3+j]); d > tol {
				t.Fatalf("eigh residual[%d][%d] = %v", i, j, d)
			}
		}
	}
}

func TestEigResiduals(t *testing.T) {
	// Rotation-scale block: eigenvalues ±i, 2.5.
	a := []float64{0, -1, 0, 1, 0, 0, 0, 0, 2.5}
	wr := make([]float64, 3)
	wi := make([]float64, 3)
	vr := make([]float64, 9)
	if !Eig(a, 3, wr, wi, true, vr) {
		t.Fatal("eig failed")
	}
	got := make([]complex128, 3)
	for i := range got {
		got[i] = complex(wr[i], wi[i])
	}
	sortC(got)
	want := []complex128{-1i, 1i, 2.5}
	sortC(want)
	for i := range want {
		if d := absC(got[i] - want[i]); d > tol {
			t.Fatalf("eig[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// Residual A v = λ v.
	for j := range 3 {
		lam := got[j]
		_ = lam
	}
	checkEigVecs(t, a, 3, wr, wi, vr)
}

// checkEigVecs verifies A v_j = λ_j v_j with LAPACK conjugate-pair packing.
func checkEigVecs(t *testing.T, a []float64, n int, wr, wi, vr []float64) {
	t.Helper()
	for j := 0; j < n; j++ {
		if wi[j] == 0 {
			for i := range n {
				sum := 0.0
				for k := range n {
					sum += a[i*n+k] * vr[k*n+j]
				}
				if d := math.Abs(sum - wr[j]*vr[i*n+j]); d > tol {
					t.Fatalf("eigvec real[%d][%d] residual %v", i, j, d)
				}
			}
			continue
		}
		for i := range n {
			sum := 0.0
			for k := range n {
				sum += a[i*n+k] * vr[k*n+j]
			}
			// (A v)_i = λ v_i with λ = wr ± i wi and v = col_j ± i col_{j+1}.
			lhs := complex(sum, 0)
			lam := complex(wr[j], wi[j])
			// For the +imag eigenvalue column pair:
			lhs2 := complex(0, 0)
			for k := range n {
				lhs2 += complex(a[i*n+k], 0) * complex(vr[k*n+j+1], 0)
			}
			res := lhs2 - (lam-0)*0 // placeholder replaced below
			_ = res
			_ = lhs
			_ = lam
		}
		j++
	}
	// Full complex check with packing: for each conjugate pair pick the
	// +imag eigenvalue λ = wr + i|wi|, v = col_j + i·col_{j+1}.
	for j := 0; j < n; j++ {
		if wi[j] == 0 {
			continue
		}
		lam := complex(wr[j], math.Abs(wi[j]))
		for i := range n {
			var sum complex128
			for k := range n {
				sum += complex(a[i*n+k], 0) * complex(vr[k*n+j], vr[k*n+j+1])
			}
			want := lam * complex(vr[i*n+j], vr[i*n+j+1])
			if d := absC(sum - want); d > tol {
				t.Fatalf("eigvec pair[%d][%d] residual %v (sum %v want %v)", i, j, d, sum, want)
			}
		}
		j++
	}
}

func sortC(v []complex128) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && lessC(v[j], v[j-1]); j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

func lessC(a, b complex128) bool {
	if real(a) != real(b) {
		return real(a) < real(b)
	}
	return imag(a) < imag(b)
}

func absC(z complex128) float64 { return math.Hypot(real(z), imag(z)) }

func TestEigRealVals(t *testing.T) {
	// Diagonalizable with real spectrum.
	a := []float64{2, 1, 0, 0, 3, 4, 0, 0, 1}
	wr := make([]float64, 3)
	wi := make([]float64, 3)
	if !Eig(a, 3, wr, wi, false, nil) {
		t.Fatal("eig failed")
	}
	sort2(wr, wi)
	if math.Abs(wr[0]-1) > tol || math.Abs(wr[1]-2) > tol || math.Abs(wr[2]-3) > tol {
		t.Fatalf("eig = %v %v %v", wr[0], wr[1], wr[2])
	}
	for i := range wi {
		if math.Abs(wi[i]) > tol {
			t.Fatalf("imag[%d] = %v", i, wi[i])
		}
	}
}

func TestEigRandom4x4(t *testing.T) {
	a := []float64{
		0.03419276725318417, 1.3597475403099617, 1.2247210785859324, -0.5103070767876675,
		-0.2979695111064471, -0.5273841930334252, 0.5697263575719601, -0.056064439045617594,
		0.7468856162565439, -1.8473247989741095, 1.5665487746995206, -0.09643216015562055,
		0.6803784532741461, -0.13656633397682774, -0.3790985670748533, 0.46311015859758675,
	}
	wr := make([]float64, 4)
	wi := make([]float64, 4)
	vr := make([]float64, 16)
	if !Eig(a, 4, wr, wi, true, vr) {
		t.Fatal("eig failed")
	}
	got := make([]complex128, 4)
	for i := range got {
		got[i] = complex(wr[i], wi[i])
	}
	sortC(got)
	want := []complex128{
		-0.32561570289182684 - 1.1222845326006274i,
		-0.32561570289182684 + 1.1222845326006274i,
		0.3289831300544176,
		1.8587157832461054,
	}
	sortC(want)
	for i := range want {
		if d := absC(got[i] - want[i]); d > tol {
			t.Fatalf("eig[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// sort2 sorts wr with wi carried along, ascending by (wr, wi).
func sort2(wr, wi []float64) {
	for i := 1; i < len(wr); i++ {
		for j := i; j > 0 && less2(wr[j], wi[j], wr[j-1], wi[j-1]); j-- {
			wr[j], wr[j-1] = wr[j-1], wr[j]
			wi[j], wi[j-1] = wi[j-1], wi[j]
		}
	}
}

func less2(r1, i1, r2, i2 float64) bool {
	if r1 != r2 {
		return r1 < r2
	}
	return i1 < i2
}

func TestSvdFull(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6}
	s := make([]float64, 2)
	u := make([]float64, 9)
	vt := make([]float64, 4) // full Vt for 3×2 is (n,n) = 2×2
	if !Svd(a, 3, 2, SVDFull, SVDFull, u, s, vt) {
		t.Fatal("svd failed")
	}
	utU := matmul(transpose(u, 3, 3), u, 3, 3, 3)
	eq(t, "full U^T U", utU, id(3))
	vvt := matmul(transpose(vt, 2, 2), vt, 2, 2, 2)
	eq(t, "full V^T V", vvt, id(2))
	// Reconstruction: U[:, :2] diag(s) Vt.
	us := make([]float64, 6)
	for i := range 3 {
		us[i*2+0] = u[i*3+0] * s[0]
		us[i*2+1] = u[i*3+1] * s[1]
	}
	eq(t, "full recon", matmul(us, vt, 3, 2, 2), a)
}

func TestSyevDiag(t *testing.T) {
	a := id(4)
	w := make([]float64, 4)
	if !Syev(a, 4, w, false) {
		t.Fatal("syev failed")
	}
	eq(t, "diag w", w, []float64{1, 1, 1, 1})
}
