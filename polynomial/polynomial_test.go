package polynomial

import (
	"math"
	"testing"

	"github.com/aixfoundry/numgo"
)

const tol10 = 1e-10

func approx(name string, t *testing.T, got []float64, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d (%v vs %v)", name, len(got), len(want), got, want)
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol10 {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func rootsReal(t *testing.T, name string, got []complex128, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(real(got[i])-want[i]) > tol10 || math.Abs(imag(got[i])) > tol10 {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// numpy golden: np.polynomial.Polynomial.fit([0,1,2,3,4], [1,2,4,8,16], 2)
func TestPolynomialFit(t *testing.T) {
	x := numgo.New1DArray([]float64{0, 1, 2, 3, 4})
	y := numgo.New1DArray([]float64{1, 2, 4, 8, 16})
	p := NewPolynomial[float64](nil).Fit(x, y, 2)
	approx("polyfit coef", t, p.Coef(),
		[]float64{3.914285714285713, 7.200000000000001, 4.571428571428575})
	if p.Domain() != ([2]float64{0, 4}) {
		t.Fatalf("polyfit domain = %v", p.Domain())
	}
	if p.Window() != ([2]float64{-1, 1}) {
		t.Fatalf("polyfit window = %v", p.Window())
	}
}

// numpy: Polynomial([1,2,3], domain=[0,4])([[0,2,4]]) = [2,1,6]; raw val at
// [[0,1,2]] = Horner of [1,2,3].
func TestPolynomialValCall(t *testing.T) {
	p := NewPolynomialDomain([]float64{1, 2, 3}, [2]float64{0, 4}, [2]float64{-1, 1})
	got := p.Call(numgo.FromSlice([]float64{0, 2, 4}, 3))
	approx("Call", t, got.Data(), []float64{2, 1, 6})
	raw := p.Val(numgo.FromSlice([]float64{0, 1, 2}, 3))
	approx("Val", t, raw.Data(), []float64{1, 6, 17})
	// non-square 2x3 input keeps shape
	got2 := p.Val(numgo.FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3))
	if got2.Shape()[0] != 2 || got2.Shape()[1] != 3 {
		t.Fatalf("Val shape = %v", got2.Shape())
	}
}

// numpy: Polynomial([2,0,1]).roots() = [0±1.4142135623730951j] (complex);
// fromroots roundtrip.
func TestPolynomialRoots(t *testing.T) {
	r := NewPolynomial([]float64{2, 0, 1}).Roots().Data()
	if len(r) != 2 {
		t.Fatalf("roots len = %d", len(r))
	}
	for _, z := range r {
		if math.Abs(real(z)) > tol10 || math.Abs(math.Abs(imag(z))-math.Sqrt2) > tol10 {
			t.Fatalf("roots = %v, want 0±sqrt(2)i", r)
		}
	}
	if r[0] != complex(0, -math.Sqrt2) {
		t.Fatalf("roots sort = %v, want ascending imag", r)
	}
	rt := NewPolynomialFromRoots([]float64{1, 2, 3}).Roots().Data()
	rootsReal(t, "fromroots roundtrip", rt, []float64{1, 2, 3})
}

// numpy goldens for Polynomial([1,2,3]) vs Polynomial([0,1]).
func TestPolynomialArith(t *testing.T) {
	p, o := NewPolynomial([]float64{1, 2, 3}), NewPolynomial([]float64{0, 1})
	approx("add", t, p.Add(o).Coef(), []float64{1, 3, 3})
	approx("sub", t, o.Sub(p).Coef(), []float64{-1, -1, -3})
	approx("mul", t, p.Mul(o).Coef(), []float64{0, 1, 2, 3})
	q, r := p.Div(o)
	approx("div q", t, q.Coef(), []float64{2, 3})
	approx("div r", t, r.Coef(), []float64{1})
	approx("pow2", t, p.Pow(2).Coef(), []float64{1, 4, 10, 12, 9})
	approx("pow0", t, p.Pow(0).Coef(), []float64{1})
	approx("der2", t, p.Deriv(2).Coef(), []float64{6})
	approx("integ k7", t, p.Integ(1, 7).Coef(), []float64{7, 1, 1, 1})
	approx("integ2 k1,2", t, p.Integ(2, 1, 2).Coef(),
		[]float64{2, 1, 0.5, 1.0 / 3, 0.25})
	approx("pow3", t, p.Pow(3).Coef(), []float64{1, 6, 21, 44, 63, 54, 27})
}

// numpy: Polynomial.fromroots([1,2,3]).coef = [-6, 11, -6, 1].
func TestPolynomialFromRoots(t *testing.T) {
	approx("fromroots", t, NewPolynomialFromRoots([]float64{1, 2, 3}).Coef(),
		[]float64{-6, 11, -6, 1})
}

// numpy: Polynomial([1,0,2e-12,0]).trim(1e-9).coef = [1]; truncate(2) of
// [1,2,3] = [1,2].
func TestTrimTruncate(t *testing.T) {
	p := NewPolynomial([]float64{1, 0, 2e-12, 0})
	approx("trim", t, p.Trim(1e-9).Coef(), []float64{1})
	approx("truncate", t, NewPolynomial([]float64{1, 2, 3}).Truncate(2).Coef(), []float64{1, 2})
	if got := NewPolynomial([]float64{1, 2, 3}).Truncate(9); got.Len() != 3 {
		t.Fatalf("truncate beyond len = %v", got.Coef())
	}
}

// numpy: Polynomial([1,2,3], domain=[0,1]).convert(domain=[-1,1]).coef
// = [2,-8,12]; Chebyshev counterpart = [14,-20,12].
func TestConvert(t *testing.T) {
	p := NewPolynomialDomain([]float64{1, 2, 3}, [2]float64{0, 1}, [2]float64{-1, 1})
	approx("poly convert", t, p.Convert([2]float64{-1, 1}, [2]float64{-1, 1}).Coef(),
		[]float64{2, -8, 12})
	c := NewChebyshevDomain([]float64{1, 2, 3}, [2]float64{0, 1}, [2]float64{-1, 1})
	approx("cheb convert", t, c.Convert([2]float64{-1, 1}, [2]float64{-1, 1}).Coef(),
		[]float64{14, -20, 12})
}

// numpy: np.polynomial.Chebyshev.fit([0..4], [1,2,4,8,16], 3).coef.
func TestChebyshevFit(t *testing.T) {
	x := numgo.New1DArray([]float64{0, 1, 2, 3, 4})
	y := numgo.New1DArray([]float64{1, 2, 4, 8, 16})
	c := NewChebyshev[float64](nil).Fit(x, y, 3)
	approx("chebfit coef", t, c.Coef(),
		[]float64{6.2, 6.999999999999998, 2.2857142857142874, 0.4999999999999993})
}

// numpy: Chebyshev([1,2,3], domain=[0,4])(2x3) = [[2,-1.5,-2],[0.5,6,-0.125]];
// raw chebval on the same points = [[-2,6,26],[58,102,0.5]].
func TestChebyshevValCall(t *testing.T) {
	c := NewChebyshevDomain([]float64{1, 2, 3}, [2]float64{0, 4}, [2]float64{-1, 1})
	got := c.Call(numgo.FromSlice([]float64{0, 1, 2, 3, 4, 0.5}, 2, 3))
	approx("cheb Call 2x3", t, got.Data(), []float64{2, -1.5, -2, 0.5, 6, -0.125})
	raw := c.Val(numgo.FromSlice([]float64{0, 1, 2, 3, 4, 0.5}, 2, 3))
	approx("cheb Val 2x3", t, raw.Data(), []float64{-2, 6, 26, 58, 102, 0.5})
}

// numpy goldens for Chebyshev([1,2,3]) vs Chebyshev([0,1]).
func TestChebyshevArith(t *testing.T) {
	c, o := NewChebyshev([]float64{1, 2, 3}), NewChebyshev([]float64{0, 1})
	approx("cheb add", t, c.Add(o).Coef(), []float64{1, 3, 3})
	approx("cheb mul", t, c.Mul(o).Coef(), []float64{1, 2.5, 1, 1.5})
	q, r := c.Div(o)
	approx("cheb div q", t, q.Coef(), []float64{2, 6})
	approx("cheb div r", t, r.Coef(), []float64{-2})
	approx("cheb pow2", t, c.Pow(2).Coef(), []float64{7.5, 10, 8, 6, 4.5})
	approx("cheb der2", t, c.Deriv(2).Coef(), []float64{12})
	approx("cheb integ k5", t, c.Integ(1, 5).Coef(), []float64{5.5, -0.5, 0.5, 0.5})
	approx("cheb integ2 k1,2", t, c.Integ(2, 1, 2).Coef(),
		[]float64{1.6875, 1.25, -0.25, 1.0 / 12, 0.0625})
	approx("cheb pow3", t, c.Pow(3).Coef(), []float64{29.5, 57, 53.25, 33.5, 22.5, 13.5, 6.75})
}

// numpy: Chebyshev([-3,2,1]).roots() = [-2, 1];
// Chebyshev.fromroots([1,2]).coef = [2.5,-3,0.5].
func TestChebyshevRootsFromRoots(t *testing.T) {
	rootsReal(t, "cheb roots", NewChebyshev([]float64{-3, 2, 1}).Roots().Data(),
		[]float64{-2, 1})
	approx("cheb fromroots", t, NewChebyshevFromRoots([]float64{1, 2}).Coef(),
		[]float64{2.5, -3, 0.5})
}

// numpy: legvander([0,1,2,-1], 3).
func TestLegendreVander(t *testing.T) {
	v := NewLegendre[float64](nil).Vander(numgo.New1DArray([]float64{0, 1, 2, -1}), 3)
	approx("legvander", t, v.Data(), []float64{1, 0, -0.5, -0, 1, 1, 1, 1, 1, 2, 5.5, 17, 1, -1, 1, -1})
	if v.Shape()[0] != 4 || v.Shape()[1] != 4 {
		t.Fatalf("legvander shape = %v", v.Shape())
	}
}

// numpy goldens for Legendre([1,2,3]) vs Legendre([0,1]).
func TestLegendreArith(t *testing.T) {
	l, o := NewLegendre([]float64{1, 2, 3}), NewLegendre([]float64{0, 1})
	approx("leg add", t, l.Add(o).Coef(), []float64{1, 3, 3})
	approx("leg mul", t, l.Mul(o).Coef(), []float64{0.6666666666666666, 2.2, 1.3333333333333333, 1.8})
	q, r := l.Div(o)
	approx("leg div q", t, q.Coef(), []float64{2, 4.5})
	approx("leg div r", t, r.Coef(), []float64{-0.5})
	approx("leg der2", t, l.Deriv(2).Coef(), []float64{9})
	approx("leg integ k5", t, l.Integ(1, 5).Coef(), []float64{5.333333333333333, 0.4, 0.6666666666666666, 0.6})
	approx("leg integ2 k1,2", t, l.Integ(2, 1, 2).Coef(),
		[]float64{1.9916666666666667, 1.2, 0.047619047619047616, 0.13333333333333333, 0.08571428571428572})
	// numpy hermval([0,1,2],[1,2,3]) analog: legval check via Call identity
	vals := NewLegendre([]float64{1, 2, 3}).Val(numgo.New1DArray([]float64{0, 1, 2}))
	approx("leg val", t, vals.Data(), []float64{-0.5, 6, 21.5})
}

// numpy: hermval([0,1,2],[1,2,3]) = [-5,11,51].
func TestHermiteValArith(t *testing.T) {
	h := NewHermite([]float64{1, 2, 3})
	vals := h.Val(numgo.New1DArray([]float64{0, 1, 2}))
	approx("herm val", t, vals.Data(), []float64{-5, 11, 51})
	o := NewHermite([]float64{0, 1})
	approx("herm add", t, h.Add(o).Coef(), []float64{1, 3, 3})
	approx("herm mul", t, h.Mul(o).Coef(), []float64{4, 13, 2, 3})
	q, r := h.Div(o)
	approx("herm div q", t, q.Coef(), []float64{2, 3})
	approx("herm div r", t, r.Coef(), []float64{-5})
	approx("herm der2", t, h.Deriv(2).Coef(), []float64{24})
	approx("herm integ k5", t, h.Integ(1, 5).Coef(), []float64{6, 0.5, 0.5, 0.5})
	approx("herm integ2 k1,2", t, h.Integ(2, 1, 2).Coef(),
		[]float64{1.5, 1, 0.125, 1.0 / 12, 0.0625})
	rootsReal(t, "herm fromroots roundtrip",
		NewHermiteFromRoots([]float64{1, 2}).Roots().Data(), []float64{1, 2})
}

// numpy: hermeval([0,1,2],[1,2,3]) = [-2,3,14].
func TestHermiteEValArith(t *testing.T) {
	h := NewHermiteE([]float64{1, 2, 3})
	vals := h.Val(numgo.New1DArray([]float64{0, 1, 2}))
	approx("herme val", t, vals.Data(), []float64{-2, 3, 14})
	o := NewHermiteE([]float64{0, 1})
	approx("herme add", t, h.Add(o).Coef(), []float64{1, 3, 3})
	approx("herme mul", t, h.Mul(o).Coef(), []float64{2, 7, 2, 3})
	q, r := h.Div(o)
	approx("herme div q", t, q.Coef(), []float64{2, 3})
	approx("herme div r", t, r.Coef(), []float64{-2})
	approx("herme der2", t, h.Deriv(2).Coef(), []float64{6})
	approx("herme integ k5", t, h.Integ(1, 5).Coef(), []float64{6, 1, 1, 1})
	approx("herme integ2 k1,2", t, h.Integ(2, 1, 2).Coef(),
		[]float64{1.75, 2, 0.5, 1.0 / 3, 0.25})
	rootsReal(t, "herme fromroots roundtrip",
		NewHermiteEFromRoots([]float64{1, 2}).Roots().Data(), []float64{1, 2})
}

// numpy: lagval([0,1,2],[1,2,3]) = [6,-0.5,-4].
func TestLaguerreValArith(t *testing.T) {
	l := NewLaguerre([]float64{1, 2, 3})
	vals := l.Val(numgo.New1DArray([]float64{0, 1, 2}))
	approx("lag val", t, vals.Data(), []float64{6, -0.5, -4})
	o := NewLaguerre([]float64{0, 1})
	approx("lag add", t, l.Add(o).Coef(), []float64{1, 3, 3})
	approx("lag mul", t, l.Mul(o).Coef(), []float64{2, 3, -8, 9})
	q, r := l.Div(o)
	approx("lag div q", t, q.Coef(), []float64{5, 1.5})
	approx("lag div r", t, r.Coef(), []float64{-0.5})
	approx("lag der2", t, l.Deriv(2).Coef(), []float64{3})
	approx("lag integ k5", t, l.Integ(1, 5).Coef(), []float64{6, 1, 1, -3})
	approx("lag integ2 k1,2", t, l.Integ(2, 1, 2).Coef(),
		[]float64{4, -1, 0, -4, 3})
	rootsReal(t, "lag fromroots roundtrip",
		NewLaguerreFromRoots([]float64{1, 2}).Roots().Data(), []float64{1, 2})
}

// float32: legvander([0.5,1.5], 2) = [[1,0.5,-0.125],[1,1.5,2.875]].
func TestFloat32(t *testing.T) {
	l := NewLegendre([]float32{1, 2, 3})
	v := l.Vander(numgo.New1DArray([]float32{0.5, 1.5}), 2)
	want := []float32{1, 0.5, -0.125, 1, 1.5, 2.875}
	for i := range want {
		d := v.Data()[i] - want[i]
		if d < -1e-6 || d > 1e-6 {
			t.Fatalf("f32 vander[%d] = %v, want %v", i, v.Data()[i], want[i])
		}
	}
}
