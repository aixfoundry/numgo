package numgo

import (
	"slices"
	"testing"
)

func TestZerosOnesFull(t *testing.T) {
	z := Zeros[float64](2, 3)
	if !slices.Equal(z.Shape(), []int{2, 3}) || z.At(1, 2) != 0 {
		t.Fatalf("Zeros shape=%v At=%v", z.Shape(), z.At(1, 2))
	}
	o := Ones[float64](2, 3)
	if o.At(0, 2) != 1 || o.At(1, 1) != 1 {
		t.Fatalf("Ones values %v %v", o.At(0, 2), o.At(1, 1))
	}
	f := Full(7.5, 2, 3)
	if f.At(1, 2) != 7.5 || !f.IsContiguous() {
		t.Fatalf("Full = %v", f.At(1, 2))
	}
}

func TestArange(t *testing.T) {
	cases := []struct {
		start, stop, step float64
		want              []float64
	}{
		{0, 5, 1, []float64{0, 1, 2, 3, 4}},
		{5, 0, -2, []float64{5, 3, 1}},
		{0, 1, 0.25, []float64{0, 0.25, 0.5, 0.75}},
		{5, 5, 1, []float64{}},
	}
	for _, c := range cases {
		a := Arange(c.start, c.stop, c.step)
		if a.Size() != len(c.want) {
			t.Fatalf("Arange(%v,%v,%v) size = %d, want %d", c.start, c.stop, c.step, a.Size(), len(c.want))
		}
		for i, w := range c.want {
			if got := a.At(i); got != w {
				t.Errorf("Arange(%v,%v,%v)[%d] = %v, want %v", c.start, c.stop, c.step, i, got, w)
			}
		}
	}
	mustPanic(t, "numgo: Arange: step must not be zero", func() { Arange(0.0, 1.0, 0.0) })
}

func TestLinspace(t *testing.T) {
	a := Linspace(0.0, 10.0, 5)
	want := []float64{0, 2.5, 5, 7.5, 10}
	for i, w := range want {
		if got := a.At(i); got != w {
			t.Errorf("Linspace[%d] = %v, want %v", i, got, w)
		}
	}
	if got := Linspace(3.0, 9.0, 1).At(0); got != 3 {
		t.Errorf("num=1 got %v", got)
	}
	if Linspace(0.0, 1.0, 0).Size() != 0 {
		t.Error("num=0 should be empty")
	}
	mustPanic(t, "numgo: number of samples, -1, must be non-negative", func() { Linspace(0.0, 1.0, -1) })
}

func TestEye(t *testing.T) {
	a := Eye[float64](3)
	if !slices.Equal(a.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v", a.Shape())
	}
	for i := range 3 {
		for j := range 3 {
			want := 0.0
			if i == j {
				want = 1
			}
			if got := a.At(i, j); got != want {
				t.Errorf("Eye At(%d,%d) = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestFromSliceCopies(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	a := FromSlice(data, 2, 3)
	data[0] = 99
	if a.At(0, 0) != 1 {
		t.Fatal("FromSlice aliased its input")
	}
	mustPanic(t, "numgo: cannot reshape array of size 3 into shape (2,2)", func() {
		FromSlice([]float64{1, 2, 3}, 2, 2)
	})
	if got := FromSlice([]float64{1, 2, 3}).Shape(); !slices.Equal(got, []int{3}) {
		t.Fatalf("default shape = %v", got)
	}
}

func TestNew2DArray(t *testing.T) {
	rows := [][]float64{{1, 2, 3}, {4, 5, 6}}
	a := New2DArray(rows)
	if !slices.Equal(a.Shape(), []int{2, 3}) || a.At(1, 2) != 6 {
		t.Fatalf("shape=%v At=%v", a.Shape(), a.At(1, 2))
	}
	rows[0][0] = 99
	if a.At(0, 0) != 1 {
		t.Fatal("New2DArray aliased its input")
	}
	mustPanic(t, "numgo: New2DArray: jagged input: row has 1 elements, want 2", func() {
		New2DArray([][]float64{{1, 2}, {3}})
	})
	if got := New2DArray([][]float64{}).Shape(); !slices.Equal(got, []int{0, 0}) {
		t.Fatalf("empty shape = %v", got)
	}
}

func TestNew1DArrayCopies(t *testing.T) {
	data := []float64{1, 2, 3}
	a := New1DArray(data)
	data[1] = 99
	if a.At(1) != 2 || !slices.Equal(a.Shape(), []int{3}) {
		t.Fatalf("got %v shape %v", a.At(1), a.Shape())
	}
}
