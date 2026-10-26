package numgo

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestSign(t *testing.T) {
	a := FromSlice([]float64{-3.5, 0, math.Copysign(0, -1), 2.5}, 2, 2)
	got := a.Sign()
	if got.At(0, 0) != -1 || got.At(0, 1) != 0 || got.At(1, 0) != 0 || got.At(1, 1) != 1 {
		t.Fatalf("Sign real = %v, %v, %v, %v", got.At(0, 0), got.At(0, 1), got.At(1, 0), got.At(1, 1))
	}
	n := FromSlice([]float64{math.NaN(), 1, 1, 1}, 2, 2).Sign()
	if !math.IsNaN(n.At(0, 0)) || n.At(0, 1) != 1 {
		t.Fatalf("Sign NaN propagation = %v", n.At(0, 0))
	}
	c := FromSlice([]complex128{-3 - 4i, 0, 1 + 1i, 2}, 2, 2).Sign()
	if want := complex(-0.6, -0.8); cmplx.Abs(c.At(0, 0)-want) > tol {
		t.Errorf("Sign(-3-4i) = %v, want %v", c.At(0, 0), want)
	}
	if c.At(0, 1) != 0 {
		t.Errorf("Sign(0) = %v, want 0", c.At(0, 1))
	}
	if v := FromSlice([]float32{-2, 3, 0, 1}, 2, 2).Sign().At(0, 0); v != -1 {
		t.Errorf("float32 Sign = %v", v)
	}
}

func TestSignbit(t *testing.T) {
	// math.Copysign(1, -0) builds a negative zero that keeps its sign bit
	// through float64 storage (a plain -0.0 literal in a composite literal
	// is stored as +0 here).
	got := FromSlice([]float64{-1, 0, math.Copysign(1, -1) * 0, 4}, 2, 2).Signbit()
	if got.At(0, 0) != 1 || got.At(0, 1) != 0 || got.At(1, 0) != 1 || got.At(1, 1) != 0 {
		t.Fatalf("Signbit = %v", got.data)
	}
}

func TestSquare(t *testing.T) {
	a := FromSlice([]float64{-3, 0.5, 2, 10, 0, 1}, 2, 3)
	got := a.Square()
	if got.At(0, 0) != 9 || got.At(0, 1) != 0.25 || got.At(1, 0) != 100 {
		t.Fatalf("Square = %v, %v, %v", got.At(0, 0), got.At(0, 1), got.At(1, 0))
	}
	c := FromSlice([]complex128{1 + 2i, 0, 0, 0, 0, 0}, 2, 3).Square()
	if want := complex(-3, 4); c.At(0, 0) != want {
		t.Errorf("Square(1+2i) = %v, want %v", c.At(0, 0), want)
	}
	if v := FromSlice([]float32{-1.5, 2, 1, 1, 1, 1}, 2, 3).Square().At(0, 0); v != 2.25 {
		t.Errorf("float32 Square = %v", v)
	}
}

func TestReciprocal(t *testing.T) {
	a := FromSlice([]float64{4, -0.5, 1, 0, -1, 8}, 2, 3)
	got := a.Reciprocal()
	if got.At(0, 0) != 0.25 || got.At(0, 1) != -2 {
		t.Fatalf("Reciprocal = %v, %v", got.At(0, 0), got.At(0, 1))
	}
	if !math.IsInf(got.At(1, 0), 1) {
		t.Errorf("Reciprocal(0) = %v, want +Inf", got.At(1, 0))
	}
	c := FromSlice([]complex128{1 + 1i, 0, 0, 0, 0, 0}, 2, 3).Reciprocal()
	if want := complex(0.5, -0.5); cmplx.Abs(c.At(0, 0)-want) > tol {
		t.Errorf("Reciprocal(1+i) = %v, want %v", c.At(0, 0), want)
	}
}

func TestFabs(t *testing.T) {
	got := FromSlice([]float64{-4, 0.5, -1, 2, -3, 9}, 2, 3).Fabs()
	for i, w := range []float64{4, 0.5, 1, 2, 3, 9} {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("Fabs[%d] = %v, want %v", i, v, w)
		}
	}
	mustPanic(t, "numgo: Abs on complex arrays returns real values; use AbsToReal", func() {
		FromSlice([]complex128{1, 1, 1, 1, 1, 1}, 2, 3).Fabs()
	})
}

func TestFix(t *testing.T) {
	got := FromSlice([]float64{1.2, 1.7, -1.2, -1.7, 2, 0.5}, 2, 3).Fix()
	if got.At(0, 1) != 1 || got.At(1, 0) != -1 {
		t.Fatalf("Fix = %v, %v", got.At(0, 1), got.At(1, 0))
	}
	c := FromSlice([]complex128{-1.7 + 2.9i, 0, 0, 0, 0, 0}, 2, 3).Fix()
	if want := complex(-1, 2); c.At(0, 0) != want {
		t.Errorf("Fix complex = %v, want %v", c.At(0, 0), want)
	}
}

func TestDeg2Rad(t *testing.T) {
	got := FromSlice([]float64{0, 90, 180, 270, 45, 360}, 2, 3).Deg2Rad()
	want := []float64{0, math.Pi / 2, math.Pi, 3 * math.Pi / 2, math.Pi / 4, 2 * math.Pi}
	for i, w := range want {
		if v := got.At(i/3, i%3); !wantClose(t, v, w) {
			t.Errorf("Deg2Rad[%d] = %v, want %v", i, v, w)
		}
	}
	back := got.Rad2Deg()
	for i, w := range []float64{0, 90, 180, 270, 45, 360} {
		if v := back.At(i/3, i%3); !wantClose(t, v, w) {
			t.Errorf("Rad2Deg[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestSinc(t *testing.T) {
	a := FromSlice([]float64{0, 0.5, 1, -0.25, 2, 4}, 2, 3)
	got := a.Sinc()
	want := []float64{1, 2 / math.Pi, 0, math.Sin(math.Pi/4) / (math.Pi / 4), 0, 0}
	for i, w := range want {
		if v := got.At(i/3, i%3); !wantClose(t, v, w) {
			t.Errorf("Sinc[%d] = %v, want %v", i, v, w)
		}
	}
	c := FromSlice([]complex128{1 + 2i, 0, 0, 0, 0, 0}, 2, 3).Sinc()
	// numpy: sin(pi·z)/(pi·z) with z = 1+2i.
	z := complex(math.Pi, 0) * (1 + 2i)
	if got, w := c.At(0, 0), cmplx.Sin(z)/z; cmplx.Abs(got-w) > 1e-9 {
		t.Errorf("Sinc(1+2i) = %v, want %v", got, w)
	}
}

func TestHeaviside(t *testing.T) {
	got := FromSlice([]float64{-2, 0, 3, math.Copysign(0, -1), 1, 0}, 2, 3).Heaviside(0.75)
	want := []float64{0, 0.75, 1, 0.75, 1, 0.75}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("Heaviside[%d] = %v, want %v", i, v, w)
		}
	}
	// numpy default second value is 0.5.
	half := FromSlice([]float64{0, 0, 0, 0, 0, 0}, 2, 3).Heaviside(0.5)
	if half.At(0, 0) != 0.5 {
		t.Errorf("Heaviside(0, 0.5) = %v", half.At(0, 0))
	}
}

func TestAngle(t *testing.T) {
	a := FromSlice([]complex128{1 + 1i, -1, 1i, 2 + 2i, 3, -1 - 1i}, 2, 3)
	got := a.Angle(false)
	want := []float64{math.Pi / 4, math.Pi, math.Pi / 2, math.Pi / 4, 0, -3 * math.Pi / 4}
	for i, w := range want {
		if v := real(got.At(i/3, i%3)); !wantClose(t, v, w) {
			t.Errorf("Angle[%d] = %v, want %v", i, v, w)
		}
	}
	deg := a.Angle(true)
	if v := real(deg.At(0, 0)); !wantClose(t, v, 45) {
		t.Errorf("Angle deg = %v, want 45", v)
	}
	if v := real(got.At(1, 1)); v != 0 {
		t.Errorf("Angle(real 3) = %v, want 0", v)
	}
}

func TestWindows(t *testing.T) {
	// n = 0 and negative are empty; n = 1 is [1].
	for name, f := range map[string]func(int) *Array[float64]{
		"Bartlett": Bartlett, "Hanning": Hanning, "Hamming": Hamming,
		"Blackman": Blackman, "Kaiser0": func(n int) *Array[float64] { return Kaiser(n, 2.5) },
	} {
		if s := f(0).Size(); s != 0 {
			t.Errorf("%s(0) size = %d, want 0", name, s)
		}
		if s := f(-3).Size(); s != 0 {
			t.Errorf("%s(-3) size = %d, want 0", name, s)
		}
		if v := f(1).At(0); v != 1 {
			t.Errorf("%s(1) = %v, want 1", name, v)
		}
	}
	// n = 2: every window reduces to its endpoint values.
	if v := Hanning(2).data; v[0] != 0 || v[1] != 0 {
		t.Errorf("Hanning(2) = %v, want [0 0]", v)
	}
	if v := Hamming(2).data; math.Abs(v[0]-0.08) > 1e-15 || math.Abs(v[1]-0.08) > 1e-15 {
		t.Errorf("Hamming(2) = %v, want [0.08 0.08]", v)
	}
	if v := Bartlett(2).data; v[0] != 0 || v[1] != 0 {
		t.Errorf("Bartlett(2) = %v, want [0 0]", v)
	}

	// np.hanning(12) reference values (numpy docs example).
	h := Hanning(12)
	wantH := []float64{0, 0.079373233584409397, 0.29229249349905678, 0.57115741913664253,
		0.82743036697264249, 0.97974648680724874, 0.97974648680724874, 0.82743036697264261,
		0.57115741913664264, 0.29229249349905695, 0.079373233584409397, 0}
	for i, w := range wantH {
		if v := h.At(i); math.Abs(v-w) > 1e-15 {
			t.Errorf("Hanning(12)[%d] = %v, want %v", i, v, w)
		}
	}
	if !shapesEqual(h.Shape(), []int{12}) {
		t.Errorf("Hanning shape = %v", h.Shape())
	}

	// np.hamming(12) reference values (numpy docs example).
	hm := Hamming(12)
	wantHM := []float64{0.080000000000000016, 0.15302337489765666, 0.34890909401913228,
		0.60546482560571113, 0.84123593761483106, 0.98136676786266885, 0.98136676786266896,
		0.84123593761483129, 0.60546482560571124, 0.34890909401913245, 0.15302337489765666,
		0.080000000000000016}
	for i, w := range wantHM {
		if v := hm.At(i); math.Abs(v-w) > 1e-15 {
			t.Errorf("Hamming(12)[%d] = %v, want %v", i, v, w)
		}
	}

	// np.blackman(12) reference values (numpy docs example).
	b := Blackman(12)
	wantB := []float64{-1.3877787807814457e-17, 0.032606434624560296, 0.15990363478343397,
		0.41439798124748267, 0.73604517991077967, 0.96704676943374313, 0.96704676943374324,
		0.73604517991077978, 0.41439798124748278, 0.15990363478343411, 0.03260643462456031,
		-1.3877787807814457e-17}
	for i, w := range wantB {
		if v := b.At(i); math.Abs(v-w) > 1e-15 {
			t.Errorf("Blackman(12)[%d] = %v, want %v", i, v, w)
		}
	}

	// np.bartlett(12) reference values (numpy docs example).
	ba := Bartlett(12)
	wantBA := []float64{0, 0.18181818181818177, 0.36363636363636365, 0.54545454545454541,
		0.72727272727272729, 0.90909090909090906, 0.90909090909090906, 0.72727272727272729,
		0.54545454545454541, 0.36363636363636365, 0.18181818181818177, 0}
	for i, w := range wantBA {
		if v := ba.At(i); math.Abs(v-w) > 1e-15 {
			t.Errorf("Bartlett(12)[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestKaiser(t *testing.T) {
	// Compare against the power-series i0 evaluated by hand.
	i0ref := func(x float64) float64 {
		y := x * x / 4
		sum, term := 1.0, 1.0
		for k := 1; term > 1e-17; k++ {
			term *= y / float64(k*k)
			sum += term
		}
		return sum
	}
	beta := 2.5
	k := Kaiser(7, beta)
	want := []float64{0.30396622941536872, 0.63092016785918892, 0.89742902035049787, 1,
		0.89742902035049787, 0.63092016785918892, 0.30396622941536872}
	for i, w := range want {
		if v := k.At(i); math.Abs(v-w) > 1e-14 {
			t.Errorf("Kaiser(7, 2.5)[%d] = %v, want %v", i, v, w)
		}
	}
	// i0 spot values: i0(0)=1, i0(1)=1.2660658..., i0(5)=27.2398718...
	for _, tc := range []struct{ x, want float64 }{{0, 1}, {1, 1.2660658777520082}, {2.5, 3.2898391440501231}, {5, 27.239871823604449}} {
		if got := i0(tc.x); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("i0(%v) = %v, want %v", tc.x, got, tc.want)
		}
	}
	// beta = 0 is the rectangular window.
	rect := Kaiser(5, 0)
	for i := range 5 {
		if v := rect.At(i); math.Abs(v-1) > 1e-12 {
			t.Errorf("Kaiser(5, 0)[%d] = %v, want 1", i, v)
		}
	}
	if i0ref(3) != i0(3) {
		t.Error("i0 diverged from its own reference series")
	}
}
