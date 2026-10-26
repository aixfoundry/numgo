package numgo

import (
	"testing"
)

func TestSetOps(t *testing.T) {
	nan := nanScalar[float64]()
	isNaN := func(v float64) bool { return v != v }

	// Union with duplicates: numpy [1 2 3].
	assertElems(t, "Union1d", Union1d(
		FromSlice([]float64{3, 1, 2, 2}),
		FromSlice([]float64{2, 1})), []float64{1, 2, 3})
	// Union flattens: 2-D input.
	assertElems(t, "Union1d 2d", Union1d(
		FromSlice([]float64{1, 2}, 1, 2),
		FromSlice([]float64{2, 3})), []float64{1, 2, 3})
	// NaN unions to a single NaN (numpy: [1 2 nan]).
	got := Union1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{nan, 2}))
	if got.Size() != 3 || !isNaN(got.At(2)) || got.At(0) != 1 || got.At(1) != 2 {
		t.Fatalf("Union1d NaN: %v", got.Data()[:got.Size()])
	}
	// Empty ∪ empty is empty.
	if Union1d(New1DArray([]float64{}), New1DArray([]float64{})).Size() != 0 {
		t.Fatal("Union1d empty")
	}

	// Intersect with duplicates: numpy [2 3].
	assertElems(t, "Intersect1d", Intersect1d(
		FromSlice([]float64{1, 2, 2, 3}),
		FromSlice([]float64{2, 2, 3, 5})), []float64{2, 3})
	// Disjoint → empty.
	if Intersect1d(FromSlice([]float64{1}), FromSlice([]float64{2})).Size() != 0 {
		t.Fatal("Intersect1d disjoint")
	}
	// NaN in both inputs → empty intersection (numpy 2.5.1: isin(NaN)
	// is always false, so nothing intersects).
	got = Intersect1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{nan, 2}))
	if got.Size() != 0 {
		t.Fatalf("Intersect1d NaN: %v", got.Data()[:got.Size()])
	}

	// SetDiff: numpy [1 3].
	assertElems(t, "SetDiff1d", SetDiff1d(
		FromSlice([]float64{1, 2, 3, 3}),
		FromSlice([]float64{2})), []float64{1, 3})
	// a's NaN is always kept — NaN never matches b (numpy: [1 nan]).
	got = SetDiff1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{2}))
	if got.Size() != 2 || got.At(0) != 1 || !isNaN(got.At(1)) {
		t.Fatalf("SetDiff1d NaN kept: %v", got.Data()[:got.Size()])
	}
	// Even with b's NaN present, a's NaN stays (numpy: [1 nan]).
	got = SetDiff1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{nan}))
	if got.Size() != 2 || got.At(0) != 1 || !isNaN(got.At(1)) {
		t.Fatalf("SetDiff1d NaN vs NaN: %v", got.Data()[:got.Size()])
	}

	// Symmetric difference: numpy [1 3 4].
	assertElems(t, "SetXor1d", SetXor1d(
		FromSlice([]float64{1, 2, 3}),
		FromSlice([]float64{2, 4})), []float64{1, 3, 4})
	// NaN on both sides: numpy keeps one NaN per input ([1 4 nan nan]).
	got = SetXor1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{nan, 4}))
	if got.Size() != 4 || got.At(0) != 1 || got.At(1) != 4 || !isNaN(got.At(2)) || !isNaN(got.At(3)) {
		t.Fatalf("SetXor1d NaN: %v", got.Data()[:got.Size()])
	}
	// NaN in one input only appears once (numpy: [1 4 nan]).
	got = SetXor1d(FromSlice([]float64{nan, 1}), FromSlice([]float64{4}))
	if got.Size() != 3 || got.At(0) != 1 || got.At(1) != 4 || !isNaN(got.At(2)) {
		t.Fatalf("SetXor1d NaN one side: %v", got.Data()[:got.Size()])
	}

	// float32 dtype.
	u32 := Union1d(FromSlice([]float32{1, 2}), FromSlice([]float32{2, 3}))
	if got32 := u32.Data()[:u32.Size()]; got32[0] != 1 || got32[1] != 2 || got32[2] != 3 {
		t.Fatalf("Union1d float32: %v", got32)
	}
	x32 := SetXor1d(FromSlice([]float32{1, 2, 3}), FromSlice([]float32{2, 4}))
	if got32 := x32.Data()[:x32.Size()]; got32[0] != 1 || got32[1] != 3 || got32[2] != 4 {
		t.Fatalf("SetXor1d float32: %v", got32)
	}
}

// assertElems checks got equals want as a 1-D sequence.
func assertElems(t *testing.T, name string, got *Array[float64], want []float64) {
	t.Helper()
	if got.Ndim() != 1 || got.Size() != len(want) {
		t.Fatalf("%s: shape %v, want len %d", name, got.Shape(), len(want))
	}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("%s[%d] = %v, want %v", name, i, v, w)
		}
	}
}
