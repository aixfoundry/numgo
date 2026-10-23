package numgo

import (
	"slices"
	"testing"
)

func TestAsType(t *testing.T) {
	a := FromSlice([]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, 2, 3)
	b := AsType[float64, float32](a)
	if !slices.Equal(b.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", b.Shape())
	}
	if got := b.At(1, 2); got != 6.5 {
		t.Fatalf("At(1,2) = %v, want 6.5", got)
	}
	b.Set([]int{0, 0}, 9)
	if a.At(0, 0) != 1.5 {
		t.Fatal("AsType aliased its source")
	}
}

func TestAsArray(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	a := AsArray(data, 2, 3)
	if !slices.Equal(a.Shape(), []int{2, 3}) || a.At(1, 1) != 5 {
		t.Fatalf("shape=%v At=%v", a.Shape(), a.At(1, 1))
	}
	data[0] = 99
	if a.At(0, 0) != 1 {
		t.Fatal("AsArray aliased its input")
	}
	if got := AsArray([]float64{1, 2}).Shape(); !slices.Equal(got, []int{2}) {
		t.Fatalf("default shape = %v", got)
	}
}
