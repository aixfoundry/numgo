package numgo

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

func TestAddBroadcast(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20, 30}, 3)
	got := a.Add(b)
	if !slices.Equal(got.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{11, 22, 33, 14, 25, 36}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
	if a.At(0, 0) != 1 {
		t.Fatal("Add mutated its receiver")
	}
}

func TestAddBroadcastBothSides(t *testing.T) {
	a := FromSlice([]float64{1, 2}, 2, 1)
	b := FromSlice([]float64{10, 20, 30}, 1, 3)
	got := a.Add(b)
	if !slices.Equal(got.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{11, 21, 31, 12, 22, 32}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestAdd3DBroadcast(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7}, 2, 1, 4)
	b := FromSlice([]float64{100, 200, 300}, 3, 1)
	got := a.Add(b)
	if !slices.Equal(got.Shape(), []int{2, 3, 4}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{
		100, 101, 102, 103,
		200, 201, 202, 203,
		300, 301, 302, 303,
		104, 105, 106, 107,
		204, 205, 206, 207,
		304, 305, 306, 307,
	}
	for i, w := range want {
		if v := got.At(i/12, (i/4)%3, i%4); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestSubMulDiv(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{2, 3, 4}, 3)
	closeTo := func(got *Array[float64], want []float64) {
		t.Helper()
		for i, w := range want {
			if v := got.At(i/3, i%3); math.Abs(v-w) > 1e-12 {
				t.Errorf("[%d] = %v, want %v", i, v, w)
			}
		}
	}
	closeTo(a.Sub(b), []float64{-1, -1, -1, 2, 2, 2})
	closeTo(a.Mul(b), []float64{2, 6, 12, 8, 15, 24})
	closeTo(a.Div(b), []float64{0.5, 2.0 / 3.0, 0.75, 2, 5.0 / 3.0, 1.5})
}

func TestInto(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{10, 20, 30}, 3)
	dst := Zeros[float64](2, 3)
	if got := a.AddInto(dst, b); got != dst {
		t.Fatal("AddInto did not return dst")
	}
	want := []float64{11, 22, 33, 14, 25, 36}
	for i, w := range want {
		if v := dst.At(i/3, i%3); v != w {
			t.Errorf("dst[%d] = %v, want %v", i, v, w)
		}
	}
	if a.At(0, 0) != 1 || b.At(0) != 10 {
		t.Fatal("Into mutated an operand")
	}
	mustPanic(t, "numgo: output shape (3,2) does not match result shape (2,3)", func() {
		a.SubInto(Zeros[float64](3, 2), b)
	})
}

func TestValueOps(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := a.AddValue(10); got.At(1, 2) != 16 || a.At(1, 2) != 6 {
		t.Fatalf("AddValue = %v, source = %v", got.At(1, 2), a.At(1, 2))
	}
	if got := a.SubValue(1); got.At(0, 0) != 0 {
		t.Fatalf("SubValue = %v", got.At(0, 0))
	}
	if got := a.MulValue(2); got.At(1, 2) != 12 {
		t.Fatalf("MulValue = %v", got.At(1, 2))
	}
	if got := a.DivValue(2); got.At(1, 2) != 3 {
		t.Fatalf("DivValue = %v", got.At(1, 2))
	}
	q := a.DivValue(0)
	if !math.IsInf(q.At(0, 0), 1) {
		t.Fatalf("DivValue(0) = %v, want +Inf", q.At(0, 0))
	}
}

// f64a is a 3x1 column for broadcast tests.
func f64a() *Array[float64] {
	return FromSlice([]float64{1, 2, 3}, 3, 1)
}

func TestMaximumMinimum(t *testing.T) {
	a := FromSlice([]float64{1, 5, 3, math.NaN()}, 4)
	b := FromSlice([]float64{4, 2, math.NaN(), 7}, 4)
	wantNear(t, a.Maximum(b), []int{4}, []float64{4, 5, math.NaN(), math.NaN()})
	wantNear(t, a.Minimum(b), []int{4}, []float64{1, 2, math.NaN(), math.NaN()})
	// Scalar variants propagate NaN likewise.
	wantNear(t, a.MaximumValue(2), []int{4}, []float64{2, 5, 3, math.NaN()})
	wantNear(t, a.MinimumValue(2), []int{4}, []float64{1, 2, 2, math.NaN()})
	// Broadcast (3,1) op (3,) -> (3,3).
	got := f64a().Maximum(FromSlice([]float64{0, 2, 9}, 3))
	if !slices.Equal(got.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{1, 2, 9, 2, 2, 9, 3, 3, 9}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
	// Fmax/Fmin skip NaNs instead of propagating.
	wantNear(t, a.Fmax(b), []int{4}, []float64{4, 5, 3, 7})
	wantNear(t, a.Fmin(b), []int{4}, []float64{1, 2, 3, 7})
}

func TestFmaxFminNaN(t *testing.T) {
	nan := math.NaN()
	a := FromSlice([]float64{nan, nan, 3}, 3)
	b := FromSlice([]float64{nan, 5, nan}, 3)
	wantNear(t, a.Fmax(b), []int{3}, []float64{nan, 5, 3})
	wantNear(t, a.Fmin(b), []int{3}, []float64{nan, 5, 3})
	// Maximum propagates where Fmax skips.
	if !math.IsNaN(a.Maximum(b).At(0)) {
		t.Fatal("Maximum(NaN,NaN) should be NaN")
	}
	if !math.IsNaN(a.Fmax(b).At(0)) {
		t.Fatal("Fmax(NaN,NaN) should be NaN")
	}
}

func TestModSemantics(t *testing.T) {
	a := FromSlice([]float64{-7, 7, -7, 7, 1, 5}, 6)
	b := FromSlice([]float64{3, -3, -3, 3, 0, 0}, 6)
	// Python %: sign of divisor. C fmod: sign of dividend. x/0 -> NaN.
	wantNear(t, a.Mod(b), []int{6}, []float64{2, -2, -1, 1, math.NaN(), math.NaN()})
	wantNear(t, a.Fmod(b), []int{6}, []float64{-1, 1, -1, 1, math.NaN(), math.NaN()})
	// Scalar variants agree.
	wantNear(t, FromSlice([]float64{-7, 7}, 2).ModValue(3), []int{2}, []float64{2, 1})
	wantNear(t, FromSlice([]float64{-7, 7}, 2).FmodValue(3), []int{2}, []float64{-1, 1})
	// FloorDivide is consistent with Mod: x = y*q + x%y.
	for i, x := range a.Data() {
		q := b.Data()[i]
		wantNear(t, FromSlice([]float64{x}, 1).FloorDivide(FromSlice([]float64{q}, 1)),
			[]int{1}, []float64{math.Floor(x / q)})
	}
	wantNear(t, FromSlice([]float64{-7, 7}, 2).FloorDivideValue(3), []int{2}, []float64{-3, 2})
	// Dividing a nonzero by zero gives Inf (IEEE), not a panic.
	got := FromSlice([]float64{7, -7}, 2).FloorDivide(FromSlice([]float64{0, 0}, 2))
	if !math.IsInf(got.At(0), 1) || !math.IsInf(got.At(1), -1) {
		t.Fatalf("FloorDivide by zero = %v, %v; want ±Inf", got.At(0), got.At(1))
	}
}

func TestFloorDivide(t *testing.T) {
	a := FromSlice([]float64{-7, 7, 6, -6}, 4)
	b := FromSlice([]float64{3, 3, -4, -4}, 4)
	wantNear(t, a.FloorDivide(b), []int{4}, []float64{-3, 2, -2, 1})
	// Mod self-consistency: y*FloorDivide(x,y) + Mod(x,y) == x.
	m := a.Mod(b)
	q := a.FloorDivide(b)
	for i, x := range a.Data() {
		y := b.Data()[i]
		if v := y*q.Data()[i] + m.Data()[i]; math.Abs(v-x) > 1e-12 {
			t.Errorf("[%d]: y*q + r = %v, want x = %v", i, v, x)
		}
	}
}

func TestPowUfunc(t *testing.T) {
	a := FromSlice([]float64{2, 3, -2}, 3)
	b := FromSlice([]float64{3, 2, 3}, 3)
	wantNear(t, a.Pow(b), []int{3}, []float64{8, 9, -8})
	wantNear(t, a.PowValue(2), []int{3}, []float64{4, 9, 4})
	// Broadcast with the column fixture: exponents 2,1,0 per column.
	wantNear(t, f64a().Pow(FromSlice([]float64{2, 1, 0}, 3)), []int{3, 3},
		[]float64{1, 1, 1, 4, 2, 1, 9, 3, 1})
	// Complex power goes through cmplx.Pow.
	c := New1DArray([]complex128{1 + 2i, 2 + 0i})
	got := c.Pow(New1DArray([]complex128{2 + 0i, 0.5 + 0i}))
	if cmplx.Abs(got.At(0)-(-3+4i)) > 1e-12 || math.Abs(real(got.At(1))-math.Sqrt2) > 1e-12 {
		t.Fatalf("complex Pow = %v, %v", got.At(0), got.At(1))
	}
	if cv := c.PowValue(2 + 0i); cv.At(1) != 4+0i {
		t.Fatalf("complex PowValue = %v", cv.At(1))
	}
}

func TestClip(t *testing.T) {
	a := FromSlice([]float64{-5, 0, 3, 10}, 4)
	wantNear(t, a.Clip(0, 5), []int{4}, []float64{0, 0, 3, 5})
	// Scalar-bound NaN handling follows Maximum/Minimum (NaN propagates).
	wantNear(t, a.Clip(math.Inf(-1), math.Inf(1)), []int{4}, []float64{-5, 0, 3, 10})
	mustPanic(t, "numgo: Clip: min must be less than or equal to max", func() {
		a.Clip(5, 0)
	})
	// float32 full pass.
	f := FromSlice([]float32{-5, 2, 10}, 3)
	got := f.Clip(-1, 4)
	want := []float32{-1, 2, 4}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("f32[%d] = %v, want %v", i, v, w)
		}
	}
	if fg := f.ModValue(3); fg.At(2) != 1 {
		t.Errorf("f32 ModValue = %v, want 1", fg.At(2))
	}
	if fg := f.FloorDivideValue(2); fg.At(0) != -3 {
		t.Errorf("f32 FloorDivideValue = %v, want -3", fg.At(0))
	}
	if fg := FromSlice([]float32{2, 3}, 2).Pow(FromSlice([]float32{3, 2}, 2)); fg.At(0) != 8 {
		t.Errorf("f32 Pow = %v, want 8", fg.At(0))
	}
}
