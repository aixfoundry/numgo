package numgo

import (
	"math"
	"slices"
	"testing"
)

func TestIsClose(t *testing.T) {
	// numpy defaults: rtol=1e-5, atol=1e-8
	rel := func(v float64) float64 { return v * 1e-5 } // smallest |a| closed to b within rtol
	a := FromSlice([]float64{1.0, math.NaN(), math.Inf(1), math.Inf(-1), 1.0 + rel(1), 0})
	b := FromSlice([]float64{1.0, math.NaN(), math.Inf(1), math.Inf(1), 1.0, 1.0})
	got := IsClose(a, b, 1e-5, 1e-8)
	want := []bool{true, false, true, false, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("IsClose = %v, want %v", got, want)
	}
	// broadcasting: scalar column against vector
	got = IsClose(FromSlice([]float64{1000}), FromSlice([]float64{999, 1000.001, 1e10}), 1e-5, 1e-8)
	want = []bool{false, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("IsClose broadcast = %v, want %v", got, want)
	}
	// complex compares by magnitude of the difference: |(3+4i)-(3+4.00001i)| ~ 1e-5
	c := IsClose(FromSlice([]complex128{3 + 4i}), FromSlice([]complex128{3 + 4.00001i}), 0, 1e-4)
	if len(c) != 1 || !c[0] {
		t.Fatalf("IsClose complex = %v", c)
	}
	// incompatible shapes panic
	mustPanic(t, "numgo: operands could not be broadcast together with shapes (2) (3)", func() {
		IsClose(FromSlice([]float64{1, 2}), FromSlice([]float64{1, 2, 3}), 1e-5, 1e-8)
	})
}
