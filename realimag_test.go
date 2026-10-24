package numgo

import (
	"math"
	"slices"
	"testing"
)

// numpy: np.real([1+4j, 2+5j]) etc. on a 2x3 complex matrix.
func c2x3() *Array[complex128] {
	return New2DArray([][]complex128{
		{1 + 4i, 2 + 5i, 3 - 1i},
		{-1 + 0.5i, 0 - 2i, 7},
	})
}

func TestRealImag(t *testing.T) {
	a := c2x3()
	re := a.Real()
	if !slices.Equal(re.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", re.Shape())
	}
	wantRe := []float64{1, 2, 3, -1, 0, 7}
	wantIm := []float64{4, 5, -1, 0.5, -2, 0}
	if !slices.Equal(re.Ravel().Data(), wantRe) {
		t.Fatalf("real = %v, want %v", re.Ravel().Data(), wantRe)
	}
	if !slices.Equal(a.Imag().Ravel().Data(), wantIm) {
		t.Fatalf("imag = %v, want %v", a.Imag().Ravel().Data(), wantIm)
	}
	// Real element types: Real converts (float32 -> float64), Imag is zeros.
	f := New1DArray([]float32{1, -2})
	if got := f.Real().Ravel().Data(); !slices.Equal(got, []float64{1, -2}) {
		t.Fatalf("float32 real = %v", got)
	}
	if got := f.Imag().Ravel().Data(); !slices.Equal(got, []float64{0, 0}) {
		t.Fatalf("float32 imag = %v", got)
	}
}

// numpy goldens: tol=1000 drops imag 4e-14/3e-15 (4e-14 < 1000*eps ≈ 2.2e-13)
// but tol=100 keeps them significant (4e-14 >= 100*eps ≈ 2.2e-14).
func TestRealIfClose(t *testing.T) {
	a := New1DArray([]complex128{2.1 + 4e-14i, 5.2 + 3e-15i})
	if got := a.RealIfClose(1000).Ravel().Data(); !slices.Equal(got, []float64{2.1, 5.2}) {
		t.Fatalf("tiny imag = %v", got)
	}
	assertPanics(t, "tol=100 significant", func() { a.RealIfClose(100) })
	// tol <= 1 is an absolute tolerance (numpy doc): 4e-14 < 1e-13 passes.
	if got := a.RealIfClose(1e-13).Ravel().Data(); !slices.Equal(got, []float64{2.1, 5.2}) {
		t.Fatalf("abs tol = %v", got)
	}
	assertPanics(t, "abs tol too small", func() { a.RealIfClose(1e-15) })
	// Significant imaginary parts panic (documented deviation: numpy returns
	// the complex array).
	assertPanics(t, "big imag", func() {
		New1DArray([]complex128{1 + 1e-10i}).RealIfClose(100)
	})
	// Real input passes through as Real regardless of tol.
	f := New1DArray([]float64{3, 4})
	if got := f.RealIfClose(100).Ravel().Data(); !slices.Equal(got, []float64{3, 4}) {
		t.Fatalf("real passthrough = %v", got)
	}
}

// numpy: np.nan_to_num([[inf, -inf, nan], [1, nan(2x3 pattern)]]) with the
// default big replacements, then with explicit replacements.
func TestNanToNum(t *testing.T) {
	a := New2DArray([][]float64{
		{math.Inf(1), math.Inf(-1), math.NaN()},
		{1, math.NaN(), -5},
	})
	got := a.NanToNumDefault().Ravel().Data()
	want := []float64{math.MaxFloat64, -math.MaxFloat64, 0, 1, 0, -5}
	if !slices.Equal(got, want) {
		t.Fatalf("default = %v, want %v", got, want)
	}
	got2 := a.NanToNum(-1, 111, -222).Ravel().Data()
	want2 := []float64{111, -222, -1, 1, -1, -5}
	if !slices.Equal(got2, want2) {
		t.Fatalf("explicit = %v, want %v", got2, want2)
	}
	// Complex: each component replaced independently.
	c := New1DArray([]complex128{complex(math.Inf(1), 3), complex(math.NaN(), -2)})
	cgot := c.NanToNumDefault().Ravel().Data()
	cwant := []complex128{complex(math.MaxFloat64, 3), complex(0, -2)}
	for i := range cwant {
		if cgot[i] != cwant[i] {
			t.Fatalf("complex[%d] = %v, want %v", i, cgot[i], cwant[i])
		}
	}
	// float32 uses its own max finite value.
	finf := float32(math.Inf(1))
	f := New1DArray([]float32{finf})
	if got := f.NanToNumDefault().At(0); got != math.MaxFloat32 {
		t.Fatalf("float32 posinf = %v, want %v", got, math.MaxFloat32)
	}
}

// numpy: np.isin(arange(6).reshape(2,3), [2,4,7]) -> [[F F T],[F T F]].
// IsIn returns []bool in C order of the TEST elements: each test element is
// checked for membership in a (numpy isin flattens elements(arange(6)) as the
// test set). We document the deviation and test the documented order.
func TestIsIn(t *testing.T) {
	a := New2DArray([][]float64{{0, 1, 2}, {3, 4, 5}})
	test := New1DArray([]float64{2, 4, 7})
	// test element 2 is in a, 4 is in a, 7 is not.
	got := a.IsIn(test, false)
	if !slices.Equal(got, []bool{true, true, false}) {
		t.Fatalf("isin = %v", got)
	}
	if inv := a.IsIn(test, true); !slices.Equal(inv, []bool{false, false, true}) {
		t.Fatalf("invert = %v", inv)
	}
	// Duplicates in the test set each get an entry.
	if dup := a.IsIn(New1DArray([]float64{2, 2, 9}), false); !slices.Equal(dup, []bool{true, true, false}) {
		t.Fatalf("dup test = %v", dup)
	}
	// NaN matches only NaN.
	n := New1DArray([]float64{math.NaN(), 1})
	if got := n.IsIn(New1DArray([]float64{math.NaN(), 1}), false); !slices.Equal(got, []bool{true, true}) {
		t.Fatalf("nan = %v", got)
	}
	if got := New1DArray([]float64{2}).IsIn(New1DArray([]float64{3}), false); !slices.Equal(got, []bool{false}) {
		t.Fatalf("absent = %v", got)
	}
}
