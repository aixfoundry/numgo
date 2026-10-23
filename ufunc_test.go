package numgo

import (
	"math"
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
