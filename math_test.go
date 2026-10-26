package numgo

import (
	"math"
	"math/cmplx"
	"testing"
)

const tol = 1e-12

func wantClose(t *testing.T, got, want float64) bool {
	t.Helper()
	return math.Abs(got-want) <= tol
}

// applyAll runs every unary math function over a 2x3 real array and checks a
// spot value against math.*. Covers the P0-5 non-square rule for real T.
func applyAll(t *testing.T) {
	t.Helper()
	a := FromSlice([]float64{-4, 0.5, 1, 2, 3, 9}, 2, 3)
	check := func(name string, got *Array[float64], idx [2]int, want float64) {
		t.Helper()
		if v := got.At(idx[0], idx[1]); !wantClose(t, v, want) {
			t.Errorf("%s%v = %v, want %v", name, idx, v, want)
		}
		if !shapesEqual(got.Shape(), []int{2, 3}) {
			t.Errorf("%s shape = %v", name, got.Shape())
		}
	}
	check("Abs", a.Abs(), [2]int{0, 0}, 4)
	check("Sqrt", a.Abs().Sqrt(), [2]int{0, 0}, 2)
	check("Cbrt", a.Abs().Cbrt(), [2]int{0, 0}, math.Pow(4, 1.0/3.0))
	check("Exp", a.Exp(), [2]int{0, 1}, math.Exp(0.5))
	check("Exp2", a.Exp2(), [2]int{1, 1}, 8)
	check("Log", a.Abs().Log(), [2]int{0, 0}, math.Log(4))
	check("Log2", FromSlice([]float64{4, 8, 1, 2, 3, 9}, 2, 3).Log2(), [2]int{0, 0}, 2)
	check("Log10", FromSlice([]float64{10, 100, 1, 2, 3, 1000}, 2, 3).Log10(), [2]int{1, 2}, 3)
	check("Log1p", a.Abs().Log1p(), [2]int{1, 2}, math.Log(10))
	check("Expm1", a.Expm1(), [2]int{1, 2}, math.Expm1(9))
	check("Sin", a.Sin(), [2]int{0, 1}, math.Sin(0.5))
	check("Cos", a.Cos(), [2]int{0, 1}, math.Cos(0.5))
	check("Tan", a.Tan(), [2]int{0, 1}, math.Tan(0.5))
	check("Asin", FromSlice([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 2, 3).Asin(), [2]int{1, 1}, math.Asin(0.5))
	check("Acos", FromSlice([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 2, 3).Acos(), [2]int{1, 1}, math.Acos(0.5))
	check("Atan", a.Atan(), [2]int{0, 1}, math.Atan(0.5))
	check("Sinh", a.Sinh(), [2]int{0, 1}, math.Sinh(0.5))
	check("Cosh", a.Cosh(), [2]int{0, 1}, math.Cosh(0.5))
	check("Tanh", a.Tanh(), [2]int{0, 1}, math.Tanh(0.5))
	check("Asinh", a.Abs().Asinh(), [2]int{1, 2}, math.Asinh(9))
	check("Acosh", a.Abs().Acosh(), [2]int{1, 2}, math.Acosh(9))
	check("Atanh", FromSlice([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 2, 3).Atanh(), [2]int{0, 0}, math.Atanh(0.5))
	check("Floor", FromSlice([]float64{1.2, 1.7, -1.2, -1.7, 2, 0.5}, 2, 3).Floor(), [2]int{0, 1}, 1)
	check("Ceil", FromSlice([]float64{1.2, 1.7, -1.2, -1.7, 2, 0.5}, 2, 3).Ceil(), [2]int{0, 1}, 2)
	check("Trunc", FromSlice([]float64{1.2, 1.7, -1.2, -1.7, 2, 0.5}, 2, 3).Trunc(), [2]int{0, 1}, 1)
	check("Conj", a.Conj(), [2]int{1, 0}, 2)
}

func TestUnaryMathReal(t *testing.T) {
	applyAll(t)
}

func TestRoundHalfToEven(t *testing.T) {
	a := FromSlice([]float64{0.5, 1.5, 2.5, 3.5, -0.5, -2.5}, 2, 3)
	got := a.Round()
	want := []float64{0, 2, 2, 4, -0, -2}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestLogZeroIEEE(t *testing.T) {
	a := Zeros[float64](2, 3)
	got := a.Log()
	if !math.IsInf(got.At(0, 0), -1) || !math.IsInf(got.At(1, 2), -1) {
		t.Fatalf("Log(0) = %v, want -Inf", got.At(0, 0))
	}
	if v := FromSlice([]float64{-1, 1, 1, 1, 1, 1}, 2, 3).Sqrt().At(0, 0); !math.IsNaN(v) {
		t.Fatalf("Sqrt(-1) = %v, want NaN", v)
	}
}

func TestUnaryMathComplex(t *testing.T) {
	a := FromSlice([]complex128{3 + 4i, 1 + 1i, 0, 1i, 2 + 0i, 8 + 27i}, 2, 3)
	if s := a.Sqrt().At(0, 0); cmplx.Abs(s-cmplx.Sqrt(3+4i)) > tol {
		t.Errorf("Sqrt(3+4i) = %v", s)
	}
	if e := a.Exp().At(0, 0); cmplx.Abs(e-cmplx.Exp(3+4i)) > tol {
		t.Errorf("Exp(3+4i) = %v", e)
	}
	if l := a.Log().At(0, 0); cmplx.Abs(l-cmplx.Log(3+4i)) > tol {
		t.Errorf("Log(3+4i) = %v", l)
	}
	// Componentwise Round: 2.5 -> 2, 3.5 -> 4 (half-to-even), -1.5 -> -2.
	r := FromSlice([]complex128{2.5 + 3.5i, 0.5 - 1.5i, 0, 0, 0, 0}, 2, 3).Round()
	if got := r.At(0, 0); got != complex(2, 4) {
		t.Errorf("Round components = %v, want (2+4i)", got)
	}
	if got := r.At(0, 1); got != complex(0, -2) {
		t.Errorf("Round components = %v, want -2i", got)
	}
	// Conj.
	if c := a.Conj().At(0, 0); c != complex(3, -4) {
		t.Errorf("Conj = %v", c)
	}
	// Go's cmplx.Log(0) is +i·pi/2 (numpy's branch gives -inf+0j); the real
	// -Inf IEEE rule is asserted in TestLogZeroIEEE.
	if l := a.Log().At(1, 0); cmplx.Abs(l-complex(0, math.Pi/2)) > tol {
		t.Errorf("Log(0) = %v, want i·pi/2 (Go cmplx branch)", l)
	}
}

func TestAbsComplexPanics(t *testing.T) {
	a := FromSlice([]complex128{3 + 4i, 1, 1, 1, 1, 1}, 2, 3)
	mustPanic(t, "numgo: Abs on complex arrays returns real values; use AbsToReal", func() {
		a.Abs()
	})
}

func TestAbsToReal(t *testing.T) {
	c := FromSlice([]complex128{3 + 4i, 1, 1, 1, 1, 1}, 2, 3)
	got := AbsToReal[complex128, float64](c)
	if got.At(0, 0) != 5 {
		t.Errorf("AbsToReal[0,0] = %v, want 5", got.At(0, 0))
	}
	if !shapesEqual(got.Shape(), []int{2, 3}) {
		t.Errorf("shape = %v", got.Shape())
	}
	// Real input also works.
	r := FromSlice([]float64{-4, 3, 1, 1, 1, 1}, 2, 3)
	if v := AbsToReal[float64, float64](r).At(0, 0); v != 4 {
		t.Errorf("AbsToReal real = %v", v)
	}
}

func TestMathFloat32(t *testing.T) {
	a := FromSlice([]float32{4, 9, 1, 16, 25, 36}, 2, 3)
	got := a.Sqrt()
	if got.At(0, 0) != 2 || got.At(1, 2) != 6 {
		t.Fatalf("float32 Sqrt = %v, %v", got.At(0, 0), got.At(1, 2))
	}
	c := FromSlice([]complex64{3 + 4i, 1, 1, 1, 1, 1}, 2, 3)
	if v := AbsToReal[complex64, float32](c).At(0, 0); v != 5 {
		t.Errorf("float32 AbsToReal = %v", v)
	}
	mustPanic(t, "numgo: Abs on complex arrays returns real values; use AbsToReal", func() {
		c.Abs()
	})
}

func TestMathPure(t *testing.T) {
	a := FromSlice([]float64{4, 9, 1, 16, 25, 36}, 2, 3)
	a.Sqrt()
	if a.At(0, 0) != 4 {
		t.Fatal("math function mutated its receiver")
	}
}

func TestSquareComplex(t *testing.T) {
	c := FromSlice([]complex128{1 + 2i, 2 - 1i, 0, 3 + 0i, -1 - 1i, 0.5}, 2, 3).Square()
	for i, z := range []complex128{-3 + 4i, 3 - 4i, 0, 9, 2i, 0.25} {
		if got := c.At(i/3, i%3); got != z {
			t.Errorf("Square[%d] = %v, want %v", i, got, z)
		}
	}
}

func TestSignComplexZeros(t *testing.T) {
	c := FromSlice([]complex128{0 + 0i, -2 + 0i, 3i, 5 + 0i, 0 - 2i, 1 + 1i}, 2, 3).Sign()
	for i, z := range []complex128{0, -1, 1i, 1, -1i, complex(1/math.Sqrt2, 1/math.Sqrt2)} {
		if got := c.At(i/3, i%3); cmplx.Abs(got-z) > 1e-15 {
			t.Errorf("Sign[%d] = %v, want %v", i, got, z)
		}
	}
}
