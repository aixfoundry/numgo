package numgo

import (
	"slices"
	"testing"
)

func TestWhere(t *testing.T) {
	cond := []bool{true, false, true, false, true, false}
	x := FromSlice([]float64{10, 20, 30, 40, 50, 60}, 2, 3)
	y := FromSlice([]float64{-1, -2, -3, -4, -5, -6}, 2, 3)
	got := Where(cond, x, y)
	if !slices.Equal(got.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{10, -2, 30, -4, 50, -6}
	for k, w := range want {
		if got.data[k] != w {
			t.Fatalf("data[%d] = %v, want %v", k, got.data[k], w)
		}
	}
	// Cross-check: Where(cond, x, y) equals x with !cond positions overwritten
	// by the flat y values, applied via Set per coordinate.
	xc := x.Copy()
	yf := y.Ravel().Data()
	coords := make([]int, 2)
	for k := range cond {
		if !cond[k] {
			xc.Set(coords, yf[k])
		}
		coords[1]++
		if coords[1] == 3 {
			coords[0], coords[1] = coords[0]+1, 0
		}
	}
	for k := range want {
		if xc.data[k] != want[k] {
			t.Errorf("SetMask cross-check data[%d] = %v, want %v", k, xc.data[k], want[k])
		}
	}
	if v := x.MaskSelect(cond); !slices.Equal(v.Data(), []float64{10, 30, 50}) {
		t.Errorf("MaskSelect = %v", v.Data())
	}
	// Broadcast x against (3,1) and y against (3,).
	xb := FromSlice([]float64{1, 2, 3}, 3, 1)
	yb := FromSlice([]float64{10, 20, 30})
	got2 := Where([]bool{true, false, false, true, true, false, false, false, true}, xb, yb)
	var want2 []float64
	for i := range 3 {
		for j := range 3 {
			if []bool{true, false, false, true, true, false, false, false, true}[i*3+j] {
				want2 = append(want2, float64(i+1))
			} else {
				want2 = append(want2, float64((j+1)*10))
			}
		}
	}
	if !slices.Equal(got2.Data(), want2) {
		t.Errorf("broadcast Where = %v, want %v", got2.Data(), want2)
	}
}

func TestWherePanics(t *testing.T) {
	x := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	y := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	mustPanic(t, "numgo: mask length 3 does not match selection size 4",
		func() { Where([]bool{true, false, true}, x, y) })
}

func TestNonZero(t *testing.T) {
	a := FromSlice([]float64{0, 1, 0, 3})
	if !slices.Equal(a.NonZero(), []int{1, 3}) {
		t.Fatalf("NonZero = %v, want [1 3]", a.NonZero())
	}
	// 2-D flat indices.
	m := FromSlice([]float64{0, 5, 0, 0, 0, 7}, 2, 3)
	if !slices.Equal(m.NonZero(), []int{1, 5}) {
		t.Errorf("2-D NonZero = %v, want [1 5]", m.NonZero())
	}
	// Complex zero means both components zero.
	c := FromSlice([]complex128{0, 1 + 0i, 0 + 2i})
	if !slices.Equal(c.NonZero(), []int{1, 2}) {
		t.Errorf("complex NonZero = %v", c.NonZero())
	}
	// float32.
	f32 := FromSlice([]float32{0, 0, 4})
	if !slices.Equal(f32.NonZero(), []int{2}) {
		t.Errorf("float32 NonZero = %v", f32.NonZero())
	}
}

func TestArgWhere(t *testing.T) {
	m := FromSlice([]float64{0, 5, 0, 0, 0, 7}, 2, 3)
	got := m.ArgWhere()
	want := [][]int{{0, 1}, {1, 2}}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Errorf("ArgWhere[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// 1-D coords equal flat indices.
	one := FromSlice([]float64{0, 1, 0, 3})
	got1 := one.ArgWhere()
	if !slices.Equal(got1[0], []int{1}) || !slices.Equal(got1[1], []int{3}) {
		t.Errorf("1-D ArgWhere = %v", got1)
	}
	// Round-trip with MaskSelect: collected elements match the coords.
	if v := m.At(m.ArgWhere()[1]...); v != 7 {
		t.Errorf("round-trip = %v, want 7", v)
	}
}
