package numgo

import (
	"slices"
	"testing"

	"math"
)

func TestAtSetIdx(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := a.At(1, 2); got != 6 {
		t.Fatalf("At(1,2) = %v, want 6", got)
	}
	if got := a.Idx(1, 2); got != 5 {
		t.Fatalf("Idx(1,2) = %d, want 5", got)
	}
	if got := a.Set([]int{0, 1}, 42); got != a {
		t.Fatal("Set did not return the receiver")
	}
	if got := a.At(0, 1); got != 42 {
		t.Fatalf("At(0,1) = %v, want 42", got)
	}
	threed := Zeros[float64](2, 3, 4)
	if got := threed.Idx(1, 2, 3); got != 23 {
		t.Fatalf("Idx(1,2,3) = %d, want 23", got)
	}
}

func TestAtPanics(t *testing.T) {
	a := Zeros[float64](2, 2)
	mustPanic(t, "numgo: index (3,1) is out of bounds for shape (2,2)", func() { a.At(3, 1) })
	mustPanic(t, "numgo: expected 2 indices for a 2-d array, got 3", func() { a.At(0, 0, 0) })
	mustPanic(t, "numgo: expected 2 indices for a 2-d array, got 3", func() { a.Set([]int{0, 0, 0}, 1) })
	mustPanic(t, "numgo: index (3,1) is out of bounds for shape (2,2)", func() { a.Idx(3, 1) })
}

func TestRangeConstructors(t *testing.T) {
	if got := R(1, 5, 2); got != (Range{1, 5, 2}) {
		t.Fatalf("R = %+v", got)
	}
	if got := To(3); got != (Range{0, 3, 0}) {
		t.Fatalf("To = %+v", got)
	}
	if got := From(2); got != (Range{2, math.MaxInt, 0}) {
		t.Fatalf("From = %+v", got)
	}
}

func TestSliceValues(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	cases := []struct {
		ranges []Range
		shape  []int
		want   []float64
	}{
		{[]Range{To(1), To(2)}, []int{1, 2}, []float64{1, 2}},
		{[]Range{R(0, 2, 1), R(0, 3, 2)}, []int{2, 2}, []float64{1, 3, 4, 6}},
		{[]Range{From(1), From(1)}, []int{1, 2}, []float64{5, 6}},
		{[]Range{Range{}, R(0, -1, 1)}, []int{2, 2}, []float64{1, 2, 4, 5}},
	}
	for i, c := range cases {
		s := a.Slice(c.ranges...)
		if !slices.Equal(s.Shape(), c.shape) {
			t.Fatalf("case %d: shape = %v, want %v", i, s.Shape(), c.shape)
		}
		for k, w := range c.want {
			coords := []int{k / c.shape[1], k % c.shape[1]}
			if got := s.At(coords...); got != w {
				t.Errorf("case %d At%v = %v, want %v", i, coords, got, w)
			}
		}
	}
}

func TestSliceReverseNegativeStep(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	rev := a.Slice(Range{Step: -1}, Range{Step: -1})
	if !slices.Equal(rev.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", rev.Shape())
	}
	want := [][]float64{{6, 5, 4}, {3, 2, 1}}
	for i := range want {
		for j := range want[i] {
			if got := rev.At(i, j); got != want[i][j] {
				t.Errorf("At(%d,%d) = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
	cols := a.Slice(Range{}, Range{Step: -1}) // reverse columns only
	if got := cols.At(0, 0); got != 3 {
		t.Fatalf("reverse cols At(0,0) = %v, want 3", got)
	}
}

func TestSliceViewSharesData(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	s := a.Slice(To(2), From(1))
	s.Set([]int{1, 1}, 99)
	if a.At(1, 2) != 99 {
		t.Fatalf("slice did not share data: %v", a.At(1, 2))
	}
}

func TestSlice3D(t *testing.T) {
	data := make([]float64, 24)
	for i := range data {
		data[i] = float64(i)
	}
	a := FromSlice(data, 2, 3, 4)
	s := a.Slice(To(2), From(1), R(0, 4, 2))
	if !slices.Equal(s.Shape(), []int{2, 2, 2}) {
		t.Fatalf("shape = %v", s.Shape())
	}
	// s[i,j,k] = a[i, 1+j, 2k]
	if got := s.At(1, 1, 1); got != a.At(1, 2, 2) {
		t.Fatalf("At(1,1,1) = %v, want %v", got, a.At(1, 2, 2))
	}
	if got := s.At(0, 0, 1); got != 6 {
		t.Fatalf("At(0,0,1) = %v, want 6", got)
	}
}

func TestSliceWrongRangeCount(t *testing.T) {
	a := Zeros[float64](2, 3)
	mustPanic(t, "numgo: expected 2 ranges for a 2-d array, got 1", func() { a.Slice(To(1)) })
}
