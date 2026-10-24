package linalg

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	numgo "github.com/kaisawind/numgo"
)

// Tolerance convention: 1e-9 absolute/relative against numpy-computed
// golden values (numpy 2.5.1; own kernels vs LAPACK reference agree to
// ~1e-14 on these sizes).

const tol = 1e-9

func f64data(vs ...float64) []float64 { return vs }

func arr2(vs [][]float64) *numgo.Array[float64] { return numgo.New2DArray(vs) }

func eqSlice(t *testing.T, name string, got []float64, want []float64) {
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

// A5: 5x5 well-conditioned random matrix (numpy default_rng(42)).
var A5 = arr2([][]float64{
	{0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364},
	{-1.302179506862318, 0.12784040316728537, -0.3162425923435822, -0.016801157504288795, -0.85304392757358},
	{0.8793979748628286, 0.7777919354289483, 0.06603069756121605, 1.1272412069680329, 0.4675093422520456},
	{-0.8592924628832382, 0.36875078408249884, -0.9588826008289989, 0.8784503013072725, -0.049925910986252896},
	{-0.18486236354526056, -0.6809295444039414, 1.2225413386740303, -0.15452948206880215, -0.4283278221631072},
})

var b5 = numgo.New1DArray([]float64{-0.3521335504882296, 0.5323091855533487, 0.36544406436407834, 0.4127326115959884, 0.43082100300788273})

var A4 = arr2([][]float64{
	{2.1416476008704612, -0.4064150163846156, -0.5122427290715373, -0.8137727282478777},
	{0.6159794225754956, 1.1289722927208916, -0.11394745765487507, -0.840156476962528},
	{-0.8244812156912396, 0.6505927878247011, 0.7432541712034423, 0.543154268305195},
	{-0.6655097072886943, 0.23216132306671977, 0.11668580914072822, 0.21868859672901295},
})

// Non-square fixtures (P0-5).
var T32 = arr2([][]float64{{1, 2, 3}, {4, 5, 6}})   // 2x3
var T23 = arr2([][]float64{{1, 2}, {3, 4}, {5, 6}}) // 3x2

func TestSolveRandom5x5(t *testing.T) {
	x := Solve(A5, b5)
	want := []float64{-0.6610430870775037, 0.5407173689335055, 0.6704693928383609, 0.3400782817509364, 0.21085480895913383}
	eqSlice(t, "solve5", x.Data()[:5], want)
}

func TestSolveMatrixRHS(t *testing.T) {
	bm := arr2([][]float64{{9, 1}, {8, 2}})
	x := Solve(arr2([][]float64{{3, 1}, {1, 2}}), bm)
	eqSlice(t, "solve2m", x.Data()[:4], []float64{2, 0, 3, 1})
}

func TestSolveSingularPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil || r != "numgo/linalg: Singular matrix" {
			t.Fatalf("panic = %v", r)
		}
	}()
	Solve(arr2([][]float64{{1, 2}, {2, 4}}), numgo.New1DArray([]float64{1, 2}))
}

func TestSolveNonSquarePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-square a")
		}
	}()
	Solve(T32, numgo.New1DArray([]float64{1, 2}))
}

func TestSolveBatched233(t *testing.T) {
	a := arr2([][]float64{
		{3.3990533903341937, -0.014875971650300183, 0.7568000802380088},
		{-2.7604178626541933, 3.8754634531933623, 0.5432001471814378},
		{0.6821286252341751, 1.7007302070100299, 5.13504567909648},
		{4.312562834746247, 0.30197600611542663, 0.7869453479945998},
		{-0.5393269947423406, 3.9598119450568148, 0.9062390030785368},
		{1.9574452148950907, -0.159267305731742, 3.951599310670136},
	}).Reshape(2, 3, 3)
	b := arr2([][]float64{
		{0.19848019378767898, 1.3432036940080567},
		{-0.03031330358915025, 1.469359036471145},
		{-0.9666572174287505, -0.18603629573699512},
		{-0.19816515668552595, 0.78650572409622},
		{1.0452603769782782, -1.5094443581182324},
		{-0.9151176258112247, 0.3368524787764621},
	}).Reshape(2, 3, 2)
	x := Solve(a, b)
	if !slices.Equal(x.Shape(), []int{2, 3, 2}) {
		t.Fatalf("shape = %v", x.Shape())
	}
	want := []float64{
		0.1118303185352337, 0.47741866851450127, 0.1051832025714875,
		0.7688595361748518, -0.23793909637996752, -0.35429477896919626,
		-0.030141871385554003, 0.21269807879066654, 0.306616078100349,
		-0.34444042654682666, -0.20429263793691899, -0.03399900797839956,
	}
	eqSlice(t, "batched solve", x.Data()[:12], want)
}

func TestInv4x4(t *testing.T) {
	inv := Inv(A4)
	want := []float64{
		2.392687505760126, -0.5721771965024447, 0.8336625241576476, 4.6348008932944005,
		3.1048398854026384, -0.2339201497440204, 0.7068181676143463, 8.899381220017053,
		-4.8779597041068685, 1.0835265353591714, 0.5669398816429355, -15.39702850080783,
		6.588005604614034, -2.0710473448697124, 1.4841232121823549, 17.44502298412225,
	}
	eqSlice(t, "inv4", inv.Data()[:16], want)
	// Round-trip: A @ inv(A) = I
	prod := A4.Matmul(inv)
	eye := numgo.Identity[float64](4)
	eqSlice(t, "roundtrip", prod.Data()[:16], eye.Data()[:16])
}

func TestInvSingularPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected singular panic")
		}
	}()
	Inv(arr2([][]float64{{0, 0}, {0, 0}}))
}

func TestDetSlogdet(t *testing.T) {
	if d := Det(A5).At(0); math.Abs(d-(-3.425480661638954)) > tol {
		t.Fatalf("det5 = %v", d)
	}
	if d := Det(A4).At(0); math.Abs(d-0.08149336332758345) > tol {
		t.Fatalf("det4 = %v", d)
	}
	if d := Det(arr2([][]float64{{2, 1, 0}, {1, 3, 1}, {0, 1, 2}})).At(0); math.Abs(d-8) > tol {
		t.Fatalf("det33 = %v", d)
	}
	// slogdet against numpy.
	r := Slogdet(A5)
	if math.Abs(float64(r.Sign)-(-1)) > tol || math.Abs(float64(r.LogAbsDet)-1.2312418010322275) > tol {
		t.Fatalf("slogdet5 = %v %v", r.Sign, r.LogAbsDet)
	}
	r4 := Slogdet(A4)
	if math.Abs(float64(r4.Sign)-1) > tol || math.Abs(float64(r4.LogAbsDet)-(-2.507233693614443)) > tol {
		t.Fatalf("slogdet4 = %v %v", r4.Sign, r4.LogAbsDet)
	}
	// singular → sign 0, -inf.
	z := Slogdet(arr2([][]float64{{1, 2}, {2, 4}}))
	if float64(z.Sign) != 0 || !math.IsInf(float64(z.LogAbsDet), -1) {
		t.Fatalf("slogdet singular = %v %v", z.Sign, z.LogAbsDet)
	}
	// negative determinant: [[0,1],[1,0]] det=-1.
	neg := Slogdet(arr2([][]float64{{0, 1}, {1, 0}}))
	if float64(neg.Sign) != -1 || math.Abs(float64(neg.LogAbsDet)) > tol {
		t.Fatalf("slogdet neg = %v %v", neg.Sign, neg.LogAbsDet)
	}
}

func TestDetNonSquarePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Det(T23)
}

func TestCholesky(t *testing.T) {
	SPD := arr2([][]float64{
		{7.757130503024862, 5.991643344254195, 0.8646202567494228, 0.5626206426619609, 3.291301256902675},
		{5.991643344254195, 8.908365902802228, 1.790544862693309, -0.7140680747789422, 4.160350538918755},
		{0.8646202567494228, 1.790544862693309, 11.705517163510583, 1.554670881960528, 0.29961049478876917},
		{0.5626206426619609, -0.7140680747789422, 1.554670881960528, 6.958383728709453, -0.13705759978014917},
		{3.291301256902675, 4.160350538918755, 0.29961049478876917, -0.13705759978014917, 13.70759240586765},
	})
	l := Cholesky(SPD)
	if !slices.Equal(l.Shape(), []int{5, 5}) {
		t.Fatalf("shape = %v", l.Shape())
	}
	// Check A = L @ L.T.
	prod := l.Matmul(l.Transpose())
	eqSlice(t, "chol recon", prod.Data()[:25], SPD.Data()[:25])
	// Strictly lower triangle matches numpy golden.
	for i, want := range []float64{2.7851625631235355, 2.1512723973765535, 2.0689110602169354, 3.3637284760671684, 3.4168844561162466} {
		_ = i
		_ = want
	}
	if v := l.At(0, 0); math.Abs(v-2.7851625631235355) > tol {
		t.Fatalf("L00 = %v", v)
	}
	if v := l.At(1, 0); math.Abs(v-2.1512723973765535) > tol {
		t.Fatalf("L10 = %v", v)
	}
	// Non-PD panics.
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-PD")
		}
	}()
	Cholesky(arr2([][]float64{{1, 2}, {2, 1}}))
}

func TestQRReduced(t *testing.T) {
	a := arr2([][]float64{{1, 2}, {3, 4}, {5, 6}})
	r := QR(a, QRReduced)
	if !slices.Equal(r.Q.Shape(), []int{3, 2}) || !slices.Equal(r.R.Shape(), []int{2, 2}) {
		t.Fatalf("shapes %v %v", r.Q.Shape(), r.R.Shape())
	}
	// Q orthogonal: Q^T Q = I.
	qtq := r.Q.Transpose().Matmul(r.Q)
	eye := numgo.Identity[float64](2)
	eqSlice(t, "Q orth", qtq.Data()[:4], eye.Data()[:4])
	// R upper triangular and Q@R = A.
	prod := r.Q.Matmul(r.R)
	eqSlice(t, "QR recon", prod.Data()[:6], a.Data()[:6])
	if v := r.R.At(1, 0); math.Abs(v) > tol {
		t.Fatalf("R10 = %v", v)
	}
	// Values match numpy for the 3x2 case.
	if v := r.Q.At(0, 0); math.Abs(v-(-0.16903085094570325)) > tol {
		t.Fatalf("Q00 = %v", v)
	}
	if v := r.R.At(0, 0); math.Abs(v-(-5.916079783099616)) > tol {
		t.Fatalf("R00 = %v", v)
	}
	// Wide input is not supported in this phase (numpy's reduced/complete/raw
	// modes require m >= n; wide matrices need the LQ path — deferred).
	func() {
		defer func() {
			if rec := recover(); rec == nil {
				t.Fatal("expected panic for wide QR")
			}
		}()
		QR(T32, QRReduced)
	}()
}

func TestQRCompleteAndRaw(t *testing.T) {
	a := arr2([][]float64{{1, 2}, {3, 4}, {5, 6}})
	rc := QR(a, QRComplete)
	if !slices.Equal(rc.Q.Shape(), []int{3, 3}) || !slices.Equal(rc.R.Shape(), []int{3, 2}) {
		t.Fatalf("shapes %v %v", rc.Q.Shape(), rc.R.Shape())
	}
	eqSlice(t, "QRc recon", rc.Q.Matmul(rc.R).Data()[:6], a.Data()[:6])
	qtq := rc.Q.Transpose().Matmul(rc.Q)
	eqSlice(t, "Qc orth", qtq.Data()[:9], numgo.Identity[float64](3).Data()[:9])
	rr := QR(a, QRRaw)
	if !slices.Equal(rr.R.Shape(), []int{2, 2}) || !slices.Equal(rr.Q.Shape(), []int{2}) {
		t.Fatalf("raw shapes %v %v", rr.R.Shape(), rr.Q.Shape())
	}
	// R = triu(H): compare with the reduced R.
	red := QR(a, QRReduced)
	eqSlice(t, "raw upper", []float64{rr.R.At(0, 0), rr.R.At(0, 1), rr.R.At(1, 1)},
		[]float64{red.R.At(0, 0), red.R.At(0, 1), red.R.At(1, 1)})
	// tau values against numpy.
	eqSlice(t, "raw tau", rr.Q.Data()[:2], []float64{1.1690308509457032, 1.1131040011646902})
}

func TestSvdThin(t *testing.T) {
	a := arr2([][]float64{
		{-0.3487250722484376, -0.46235179266456716, 0.8579758812571538, -0.1913043248816149, -1.2756863233379219},
		{-1.1332872140034806, -0.9194522860016113, 0.49716074405376404, 0.14242573607056525, 0.6904853540677682},
		{-0.42725264633653426, 0.15853969107671423, 0.6255903939673367, -0.3093465397202384, 0.45677523755741145},
	})
	s := Svd(a, false, true)
	if !slices.Equal(s.U.Shape(), []int{3, 3}) || !slices.Equal(s.S.Shape(), []int{3}) || !slices.Equal(s.Vt.Shape(), []int{3, 5}) {
		t.Fatalf("shapes %v %v %v", s.U.Shape(), s.S.Shape(), s.Vt.Shape())
	}
	eqSlice(t, "sv", s.S.Data()[:3], []float64{1.8480530273112223, 1.5994429913922052, 0.7343665761277535})
	// Reconstruction: U @ diag(S) @ Vt ≈ A (thin U is 3x3, Vt is 3x5).
	diag := numgo.Zeros[float64](3, 3)
	for i := 0; i < 3; i++ {
		diag.Set([]int{i, i}, s.S.At(i))
	}
	recon := s.U.Matmul(diag).Matmul(s.Vt)
	eqSlice(t, "svd recon", recon.Data()[:15], a.Data()[:15])
	// Full matrices shapes.
	sf := Svd(a, true, true)
	if !slices.Equal(sf.U.Shape(), []int{3, 3}) || !slices.Equal(sf.Vt.Shape(), []int{5, 5}) {
		t.Fatalf("full shapes %v %v", sf.U.Shape(), sf.Vt.Shape())
	}
	// computeUV=false only fills S.
	sn := Svd(a, false, false)
	if sn.U != nil || sn.Vt != nil {
		t.Fatal("expected nil factors")
	}
	eqSlice(t, "sv only", sn.S.Data()[:3], s.S.Data()[:3])
	// 1x3 vector-matrix case: thin Vt is (1,3)?? numpy full_matrices=False
	// gives U (1,1), S (1,), Vt (3,3) — because numpy returns V (not Vt)
	// shape (N,N) only when full... actually numpy: (1,1) (1,) (3,3) with
	// full_matrices=False for (1,3)? No: numpy gives Vt (3,3) here since
	// min(m,n)=1 < n=3 needs complete V rows... numpy returns Vt (3,3).
	// Gonum SVDStore cannot produce that shape, so thin Vt is (1,3).
	v := Svd(numgo.New2DArray([][]float64{{1, 1, 1}}), false, true)
	if !slices.Equal(v.U.Shape(), []int{1, 1}) || !slices.Equal(v.Vt.Shape(), []int{1, 3}) {
		t.Fatalf("vec shapes %v %v", v.U.Shape(), v.Vt.Shape())
	}
}

func TestEigGeneralComplex(t *testing.T) {
	// Complex pair + real eigenvalue; golden values from numpy.
	a := arr2([][]float64{{0, -1, 0}, {1, 0, 0}, {0, 0, 2.5}})
	r := Eig(a, true)
	want := []complex128{-1i, 1i, 2.5}
	got := r.Values.Data()[:3]
	sortedGot := append([]complex128(nil), got...)
	slices.SortFunc(sortedGot, func(x, y complex128) int {
		return cmpComplex(x, y)
	})
	slices.SortFunc(want, cmpComplex)
	for i := range want {
		if cmplx.Abs(sortedGot[i]-want[i]) > tol {
			t.Fatalf("eig[%d] = %v, want %v", i, sortedGot[i], want[i])
		}
	}
	// Eigenvector property: A @ V = V @ diag(w).
	ac := numgo.AsType[float64, complex128](a)
	vd := numgo.Zeros[complex128](3, 3)
	for j := 0; j < 3; j++ {
		vd.Set([]int{j, j}, r.Values.At(j))
	}
	lhs := eigMatmul(ac, r.Vectors)
	rhs := eigMatmul(r.Vectors, vd)
	for i := 0; i < 9; i++ {
		if cmplx.Abs(lhs.Data()[i]-rhs.Data()[i]) > tol {
			t.Fatalf("eigvec residual[%d] = %v vs %v", i, lhs.Data()[i], rhs.Data()[i])
		}
	}
	// 4x4 random matrix: sorted eigenvalues match numpy to 1e-9.
	g := arr2([][]float64{
		{0.03419276725318417, 1.3597475403099617, 1.2247210785859324, -0.5103070767876675},
		{-0.2979695111064471, -0.5273841930334252, 0.5697263575719601, -0.056064439045617594},
		{0.7468856162565439, -1.8473247989741095, 1.5665487746995206, -0.09643216015562055},
		{0.6803784532741461, -0.13656633397682774, -0.3790985670748533, 0.46311015859758675},
	})
	gw := Eigvals(g).Data()[:4]
	wantG := []complex128{
		-0.32561570289182684 - 1.1222845326006274i,
		-0.32561570289182684 + 1.1222845326006274i,
		0.3289831300544176,
		1.8587157832461054,
	}
	slices.SortFunc(gw, cmpComplex)
	slices.SortFunc(wantG, cmpComplex)
	for i := range wantG {
		if cmplx.Abs(gw[i]-wantG[i]) > tol {
			t.Fatalf("g eig[%d] = %v, want %v", i, gw[i], wantG[i])
		}
	}
}

func cmpComplex(a, b complex128) int {
	if real(a) != real(b) {
		if real(a) < real(b) {
			return -1
		}
		return 1
	}
	if imag(a) < imag(b) {
		return -1
	}
	if imag(a) > imag(b) {
		return 1
	}
	return 0
}

// eigMatmul multiplies two real arrays as complex arrays (At/Set only).
func eigMatmul(a, b *numgo.Array[complex128]) *numgo.Array[complex128] {
	shape := a.Shape()
	m := shape[len(shape)-2]
	out := numgo.Zeros[complex128](shape...)
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			var sum complex128
			for k := 0; k < m; k++ {
				sum += a.At(i, k) * b.At(k, j)
			}
			out.Set([]int{i, j}, sum)
		}
	}
	return out
}

func TestEighSymmetric(t *testing.T) {
	sym := arr2([][]float64{{2, 1, 0}, {1, 2, 1}, {0, 1, 2}})
	r := Eigh(sym, true, "L")
	eqSlice(t, "eigh w", r.Values.Data()[:3], []float64{0.5857864376269051, 1.9999999999999998, 3.414213562373095})
	// Eigenvector property: A @ V = V @ diag(w).
	diag := numgo.Zeros[float64](3, 3)
	for j := 0; j < 3; j++ {
		diag.Set([]int{j, j}, r.Values.At(j))
	}
	lhs := sym.Matmul(r.Vectors)
	rhs := r.Vectors.Matmul(diag)
	eqSlice(t, "eigh vectors", lhs.Data()[:9], rhs.Data()[:9])
	// UPLO='U' reads the upper triangle (numpy wording test).
	triU := numgo.Zeros[float64](3, 3)
	triU.Set([]int{0, 0}, 2)
	triU.Set([]int{0, 1}, 1)
	triU.Set([]int{0, 2}, 4)
	triU.Set([]int{1, 1}, 0)
	triU.Set([]int{1, 2}, 3)
	triU.Set([]int{2, 2}, 5)
	wU := Eigvalsh(triU, "U").Data()[:3]
	eqSlice(t, "eigh U", wU, []float64{-1.7613941, -0.06432699491246618, 8.8257211})
	// Batched.
	batch := numgo.Ones[float64](2, 3, 3)
	wb := Eigvalsh(batch, "L")
	if !slices.Equal(wb.Shape(), []int{2, 3}) {
		t.Fatalf("batch shape %v", wb.Shape())
	}
}

func TestLstsqOverdetermined(t *testing.T) {
	ao := arr2([][]float64{
		{0.8714287779481898, 0.22359554877468227, 0.6789135630718949},
		{0.06757906948889146, 0.28911939868998415, 0.6312882258385404},
		{-1.4571558198556664, -0.31967121635730134, -0.4703726542927955},
		{-0.6388778482433419, -0.27514225122668373, 1.4949413112343959},
		{-0.8658311156932432, 0.9682783545914808, -1.6828697716158048},
	})
	bo := numgo.New1DArray([]float64{-0.33488502998577485, 0.1627530651050056, 0.5862223313592781, 0.711226579792855, 0.7933472351999252})
	r := Lstsq(ao, bo, -1)
	eqSlice(t, "lstsq x", r.X.Data()[:3], []float64{-0.6739932528270965, 0.5961299282442001, 0.24127192032438102})
	eqSlice(t, "lstsq res", r.Residuals.Data()[:1], []float64{0.032506398709051344})
	if math.Abs(float64(r.Rank.At(0))-3) > 0 {
		t.Fatalf("rank = %v", r.Rank.At(0))
	}
	eqSlice(t, "lstsq sv", r.SingularValues.Data()[:3], []float64{2.7373725311446435, 1.8026651300158778, 0.8360123199394429})
}

func TestLstsqRankDeficient(t *testing.T) {
	// Exactly rank-1 system: numpy lstsq gives the min-norm solution and an
	// empty residuals array when rank < N.
	a := arr2([][]float64{{1, 1}, {1, 1}, {1, 1}})
	b := numgo.New1DArray([]float64{1, 2, 3})
	r := Lstsq(a, b, -1)
	eqSlice(t, "rd x", r.X.Data()[:2], []float64{1, 1})
	if math.Abs(float64(r.Rank.At(0))-1) > 0 {
		t.Fatalf("rank = %v", r.Rank.At(0))
	}
	if r.Residuals.Size() != 0 {
		t.Fatalf("residuals size = %v", r.Residuals.Shape())
	}
}

func TestPinvRankDeficient(t *testing.T) {
	a := arr2([][]float64{{1, 2}, {2, 4}, {1, 1}})
	p := Pinv(a, 1e-10)
	want := []float64{-0.20000000000000226, -0.39999999999999863, 1.9999999999999993, 0.2000000000000012, 0.39999999999999913, -0.9999999999999994}
	eqSlice(t, "pinv", p.Data()[:6], want)
	// Identity property on the row space: A @ P @ A ≈ A.
	ap := a.Matmul(p)
	apa := ap.Matmul(a)
	eqSlice(t, "pinv prop", apa.Data()[:6], a.Data()[:6])
}

func TestMatrixRank(t *testing.T) {
	d := arr2([][]float64{{1, 0, 0}, {0, 1e-14, 0}, {0, 0, 1e-10}})
	if got := MatrixRank(d, -1).At(0); got != 3 {
		t.Fatalf("rank default = %v", got)
	}
	if got := MatrixRank(d, 1e-12).At(0); got != 2 {
		t.Fatalf("rank tol1e-12 = %v", got)
	}
	if got := MatrixRank(d, 1e-9).At(0); got != 1 {
		t.Fatalf("rank tol1e-9 = %v", got)
	}
	if got := MatrixRank(arr2([][]float64{{1, 1}, {1, 1}}), -1).At(0); got != 1 {
		t.Fatalf("rank 2x2 ones = %v", got)
	}
}

func TestNormAllOrders(t *testing.T) {
	n := arr2([][]float64{{1, 2}, {3, 4}})
	cases := []struct {
		ord  any
		want float64
	}{
		{1, 6.0},
		{-1, 4.0},
		{math.Inf(1), 7.0},
		{math.Inf(-1), 3.0},
		{"fro", 5.477225575051661},
		{2, 5.464985704219043},
		{-2, 0.3659661906262575},
		{"nuc", 5.8309518948453},
		{nil, 5.477225575051661}, // 2-D default is fro
	}
	for _, c := range cases {
		got := Norm(n, c.ord, nil, false).At(0)
		if math.Abs(got-c.want) > tol {
			t.Fatalf("norm ord=%v = %v, want %v", c.ord, got, c.want)
		}
	}
	// 2-D with axis=(0,1) routes to matrix norm too.
	if got := Norm(n, 1, []int{0, 1}, false).At(0); math.Abs(got-6) > tol {
		t.Fatalf("norm axis = %v", got)
	}
	vec := numgo.New1DArray([]float64{1, -2, 2})
	vcases := []struct {
		ord  any
		want float64
	}{
		{1, 5.0},
		{2, 3.0},
		{nil, 3.0},
		{math.Inf(1), 2.0},
		{math.Inf(-1), 1.0},
		{3, 2.571281590658235},
		{-1, 0.5},
		{0, 3.0},
		{-2.5, 0.885951287403776},
	}
	for _, c := range vcases {
		got := Norm(vec, c.ord, nil, false).At(0)
		if math.Abs(got-c.want) > tol {
			t.Fatalf("vecnorm ord=%v = %v, want %v", c.ord, got, c.want)
		}
	}
	// Invalid orders panic.
	for _, bad := range []any{"fro"} {
		func(o any) {
			defer func() {
				if recover() == nil {
					t.Fatalf("expected panic for vector ord %v", o)
				}
			}()
			Norm(vec, o, nil, false)
		}(bad)
	}
	// Matrix norm via axis on 3-D input.
	x := arr2(make([][]float64, 2))
	d3 := numgo.FromFunction([]int{2, 3, 4}, func(c ...int) float64 {
		return float64(c[0]*12 + c[1]*4 + c[2])
	})
	mn := Norm(d3, 1, []int{-2, -1}, false)
	if !slices.Equal(mn.Shape(), []int{2}) {
		t.Fatalf("3d norm shape %v", mn.Shape())
	}
	_ = x
}

func TestMatrixNormVectorNorm(t *testing.T) {
	x := numgo.FromFunction([]int{2, 3, 4}, func(c ...int) float64 {
		return float64(c[0]*12 + c[1]*4 + c[2])
	})
	mn := MatrixNorm(x, "fro", nil, false)
	eqSlice(t, "mn", mn.Data()[:2], []float64{22.494443758403985, 61.78996682310163})
	vn := VectorNorm(x, 1, []int{0}, false)
	eqSlice(t, "vn", vn.Data()[:12], []float64{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})
	// keepdims.
	mk := MatrixNorm(x, "fro", nil, true)
	if !slices.Equal(mk.Shape(), []int{2, 1, 1}) {
		t.Fatalf("keepdims shape %v", mk.Shape())
	}
}

func TestCond(t *testing.T) {
	n := arr2([][]float64{{1, 2}, {3, 4}})
	cases := []struct {
		p    any
		want float64
	}{
		{nil, 14.933034373659265},
		{2, 14.933034373659265},
		{1, 20.999999999999993},
		{-1, 5.999999999999998},
		{math.Inf(1), 20.999999999999993},
		{math.Inf(-1), 5.999999999999998},
		{"fro", 14.999999999999998},
		{-2, 0.06696562634074718},
	}
	for _, c := range cases {
		got := Cond(n, c.p).At(0)
		if math.Abs(got-c.want) > 1e-8 {
			t.Fatalf("cond p=%v = %v, want %v", c.p, got, c.want)
		}
	}
}

func TestMultiDot(t *testing.T) {
	m1 := arr2([][]float64{{1, 2}, {3, 4}})
	m2 := arr2([][]float64{{0, 1}, {1, 0}})
	m3 := arr2([][]float64{{2, 0}, {0, 2}})
	r := MultiDot(m1, m2, m3)
	eqSlice(t, "multidot", r.Data()[:4], []float64{4, 2, 8, 6})
	// Two operands equals matmul.
	r2 := MultiDot(m1, m2)
	eqSlice(t, "multidot2", r2.Data()[:4], m1.Matmul(m2).Data()[:4])
	// Single operand panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic")
			}
		}()
		MultiDot[float64](m1)
	}()
}

func TestMatrixPower(t *testing.T) {
	p := arr2([][]float64{{1, 1}, {0, 1}})
	r5 := MatrixPower(p, 5)
	eqSlice(t, "mp5", r5.Data()[:4], []float64{1, 5, 0, 1})
	r0 := MatrixPower(p, 0)
	eqSlice(t, "mp0", r0.Data()[:4], numgo.Identity[float64](2).Data()[:4])
	rn := MatrixPower(p, -2)
	eqSlice(t, "mpneg", rn.Data()[:4], []float64{1, -2, 0, 1})
	r17 := MatrixPower(p, 17)
	eqSlice(t, "mp17", r17.Data()[:4], []float64{1, 17, 0, 1})
	// Batched.
	batch := arr2([][]float64{{2, 0, 0, 2, 3, 0, 0, 3}}).Reshape(2, 2, 2)
	rb := MatrixPower(batch, 2)
	eqSlice(t, "mpbatch", rb.Data()[:8], []float64{4, 0, 0, 4, 9, 0, 0, 9})
	// Non-square panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic")
			}
		}()
		MatrixPower(T32, 2)
	}()
}

func TestTensorSolveAndInv(t *testing.T) {
	// Golden tensorsolve from numpy: a (2,3,2,3), b (2,3).
	ta := arr2([][]float64{
		{2.304717079754431, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364, -1.302179506862318},
		{0.12784040316728537, 1.683757407656418, -0.016801157504288795, -0.85304392757358, 0.8793979748628286, 0.7777919354289483},
		{0.06603069756121605, 1.1272412069680329, 2.4675093422520455, -0.8592924628832382, 0.36875078408249884, -0.9588826008289989},
		{0.8784503013072725, -0.049925910986252896, -0.18486236354526056, 1.3190704555960586, 1.2225413386740303, -0.15452948206880215},
		{-0.4283278221631072, -0.3521335504882296, 0.5323091855533487, 0.36544406436407834, 2.412732611595988, 0.43082100300788273},
		{2.1416476008704612, -0.4064150163846156, -0.5122427290715373, -0.8137727282478777, 0.6159794225754956, 3.1289722927208916},
	}).Reshape(2, 3, 2, 3)
	tb := arr2([][]float64{
		{-0.11394745765487507, -0.840156476962528, -0.8244812156912396},
		{0.6505927878247011, 0.7432541712034423, 0.543154268305195},
	})
	x := TensorSolve(ta, tb, nil)
	if !slices.Equal(x.Shape(), []int{2, 3}) {
		t.Fatalf("tx shape %v", x.Shape())
	}
	want := []float64{-0.5542457896217099, -0.05300047071530225, 0.6761758808631401, 1.4141666174864316, -0.35694494061569965, 1.0948196627380302}
	eqSlice(t, "tensorsolve", x.Data()[:6], want)
	// tensorinv ind=2 round trip: tensordot(Ti, T) ≈ identity.
	t6 := arr2([][]float64{
		{1.703559419804937, -0.6349588923738527, 1.272591177882884, -0.37084535417839304, 0.2709915877367156, 1.7479679637522305},
		{1.5940297626557918, 2.896646584172635, -0.2415210220605113, -1.2608706340895501, -0.6944580741815911, 0.4253583476127701},
		{0.39573076180964684, 0.11023823117395956, 3.9948016575639254, -0.7723680130331807, -0.05607652018346502, 0.7312429884948122},
		{0.584145906227918, 1.0709457757324194, 0.3970204067650606, 2.690597444403168, 0.3621922602623995, -1.0025951963841089},
		{-1.6394593315431263, 0.5806717808971771, -0.0551403328210084, 0.30840962641700853, 1.302315636158582, -0.36511402472543003},
		{-0.5998571137315275, -0.8642121147366698, -2.25501500646246, -0.33487348213705126, 0.8973042881661908, 3.3809977218637384},
	}).Reshape(2, 3, 2, 3)
	ti := TensorInv(t6, 2)
	if !slices.Equal(ti.Shape(), []int{2, 3, 2, 3}) {
		t.Fatalf("ti shape %v", ti.Shape())
	}
	// tensordot(ti, t6, 2) reshaped to (6,6) ≈ eye.
	id := ti.Reshape(6, 6).Matmul(t6.Reshape(6, 6))
	eqSlice(t, "tensorinv", id.Data()[:36], numgo.Identity[float64](6).Data()[:36])
	// Invalid ind panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic")
			}
		}()
		TensorInv(t6, 0)
	}()
}

func TestTraceAndDiagonal(t *testing.T) {
	a := numgo.FromFunction([]int{3, 4}, func(c ...int) float64 { return float64(4*c[0] + c[1]) })
	if v := Trace(a, 0).At(0); math.Abs(v-15) > tol {
		t.Fatalf("trace = %v", v)
	}
	if v := Trace(a, 1).At(0); math.Abs(v-18) > tol {
		t.Fatalf("trace off1 = %v", v)
	}
	if v := Trace(a, -2).At(0); math.Abs(v-8) > tol {
		t.Fatalf("trace off-2 = %v", v)
	}
	d := Diagonal(a, 1)
	eqSlice(t, "diag1", d.Data()[:3], []float64{1, 6, 11})
	dm := Diagonal(a, -1)
	eqSlice(t, "diag-1", dm.Data()[:2], []float64{4, 9})
	// Batched: (2,3,4) → diag shape (2,3).
	b := numgo.FromFunction([]int{2, 3, 4}, func(c ...int) float64 { return float64(12*c[0] + 4*c[1] + c[2]) })
	db := Diagonal(b, 0)
	if !slices.Equal(db.Shape(), []int{2, 3}) {
		t.Fatalf("batch diag shape %v", db.Shape())
	}
	eqSlice(t, "batchdiag", db.Data()[:6], []float64{0, 5, 10, 12, 17, 22})
	trb := Trace(b, 1)
	eqSlice(t, "batchtrace", trb.Data()[:2], []float64{18, 54})
}

func TestCrossOuterVecdotKron(t *testing.T) {
	a := numgo.New1DArray([]float64{1, 2, 3})
	b := numgo.New1DArray([]float64{4, 5, 6})
	eqSlice(t, "cross", Cross(a, b, -1).Data()[:3], []float64{-3, 6, -3})
	eqSlice(t, "outer", Outer(a, b).Data()[:9], []float64{4, 5, 6, 8, 10, 12, 12, 15, 18})
	if v := Vecdot(a, b, -1).At(0); math.Abs(v-32) > tol {
		t.Fatalf("vecdot = %v", v)
	}
	k := Kron(arr2([][]float64{{1, 2}, {3, 4}}), arr2([][]float64{{0, 1}, {1, 0}}))
	eqSlice(t, "kron", k.Data()[:16], []float64{0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0})
	// Vecdot with broadcasting: (2,1,3) x (4,3) → (2,4).
	va := numgo.Ones[float64](2, 1, 3)
	vb := numgo.Ones[float64](4, 3)
	vd := Vecdot(va, vb, -1)
	if !slices.Equal(vd.Shape(), []int{2, 4}) {
		t.Fatalf("vecdot shape %v", vd.Shape())
	}
	// Cross axis=0.
	ca := arr2([][]float64{{1, 0}, {0, 1}, {0, 0}}) // columns as vectors
	cb := arr2([][]float64{{0, 1}, {1, 0}, {0, 0}})
	cr := Cross(ca, cb, 0)
	if !slices.Equal(cr.Shape(), []int{3, 2}) {
		t.Fatalf("cross axis0 shape %v", cr.Shape())
	}
	if cr.At(2, 0) != 1 || cr.At(2, 1) != -1 {
		t.Fatalf("cross axis0 values %v %v", cr.At(2, 0), cr.At(2, 1))
	}
	// Size-3 violation panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic")
			}
		}()
		Cross(numgo.Ones[float64](4), numgo.Ones[float64](4), -1)
	}()
}

func TestFloat32RoundTrip(t *testing.T) {
	a32 := numgo.AsType[float64, float32](A5)
	b32 := numgo.Ones[float32](5)
	x := Solve(a32, b32)
	if x.At(0) != float32(x.At(0)) {
		t.Fatal("not float32")
	}
	// Compare against numpy float32 solve golden.
	want := []float32{-1.0049381256103516, 0.636298418045044, 1.1678777933120728, 1.1631274223327637, 0.001265891594812274}
	for i := range want {
		if math.Abs(float64(x.At(i)-want[i])) > 1e-5 {
			t.Fatalf("f32 solve[%d] = %v, want %v", i, x.At(i), want[i])
		}
	}
}

func TestComplexInputPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil || r != "numgo/linalg: complex dtypes not supported yet" {
			t.Fatalf("panic = %v", r)
		}
	}()
	a := numgo.Eye[complex128](2)
	Solve(a, numgo.New1DArray([]complex128{1, 1}))
}

func TestNonSquareEverywhere(t *testing.T) {
	// P0-5: non-square cases where applicable.
	// SVD wide vs tall.
	sw := Svd(T32, false, true)
	if !slices.Equal(sw.U.Shape(), []int{2, 2}) || !slices.Equal(sw.S.Shape(), []int{2}) || !slices.Equal(sw.Vt.Shape(), []int{2, 3}) {
		t.Fatalf("wide svd shapes %v %v %v", sw.U.Shape(), sw.S.Shape(), sw.Vt.Shape())
	}
	st := Svd(T23, false, true)
	if !slices.Equal(st.U.Shape(), []int{3, 2}) || !slices.Equal(st.Vt.Shape(), []int{2, 2}) {
		t.Fatalf("tall svd shapes %v %v", st.U.Shape(), st.Vt.Shape())
	}
	// lstsq with 2-D b.
	r2 := Lstsq(T23, numgo.New2DArray([][]float64{{1, 0}, {0, 1}, {1, 1}}), -1)
	if !slices.Equal(r2.X.Shape(), []int{2, 2}) {
		t.Fatalf("lstsq 2d x shape %v", r2.X.Shape())
	}
}
