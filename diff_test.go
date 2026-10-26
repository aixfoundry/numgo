package numgo

import (
	"slices"
	"testing"
)

func TestDiff(t *testing.T) {
	a := New2DArray([][]float64{{1, 2, 4}, {10, 20, 40}})
	d0 := a.Diff(0)
	if !slices.Equal(d0.Shape(), []int{1, 3}) {
		t.Fatalf("axis=0 shape = %v", d0.Shape())
	}
	want0 := []float64{9, 18, 36}
	for i, w := range want0 {
		if got := d0.At(0, i); got != w {
			t.Errorf("Diff(0)[%d] = %v, want %v", i, got, w)
		}
	}
	d1 := a.Diff(1)
	if !slices.Equal(d1.Shape(), []int{2, 2}) {
		t.Fatalf("axis=1 shape = %v", d1.Shape())
	}
	want1 := [][]float64{{1, 2}, {10, 20}}
	for i := range want1 {
		for j := range want1[i] {
			if got := d1.At(i, j); got != want1[i][j] {
				t.Errorf("Diff(1)[%d][%d] = %v, want %v", i, j, got, want1[i][j])
			}
		}
	}
	if d := a.Diff(-1); d.At(1, 0) != 10 {
		t.Errorf("Diff(-1)[1][0] = %v", d.At(1, 0))
	}
}

func TestEdiff1d(t *testing.T) {
	a := New2DArray([][]float64{{1, 4, 9}, {16, 25, 36}})
	d := a.Ediff1d()
	if !slices.Equal(d.Shape(), []int{5}) {
		t.Fatalf("shape = %v", d.Shape())
	}
	want := []float64{3, 5, 7, 9, 11}
	for i, w := range want {
		if got := d.At(i); got != w {
			t.Errorf("Ediff1d[%d] = %v, want %v", i, got, w)
		}
	}
}
