package numgo

import (
	"slices"
	"testing"
)

func TestTakeRowsColsAxis1(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	cases := []struct {
		name    string
		indices []int
		axis    int
		shape   []int
		want    []float64 // C order
	}{
		{"rows", []int{1, 0, 1}, 0, []int{3, 3}, []float64{4, 5, 6, 1, 2, 3, 4, 5, 6}},
		{"cols", []int{2, 0}, 1, []int{2, 2}, []float64{3, 1, 6, 4}},
		{"axis-1 negative", []int{-1, -3}, -1, []int{2, 2}, []float64{3, 1, 6, 4}},
	}
	for _, c := range cases {
		got := a.Take(c.indices, c.axis)
		if !slices.Equal(got.Shape(), c.shape) {
			t.Fatalf("%s: shape = %v, want %v", c.name, got.Shape(), c.shape)
		}
		for k, w := range c.want {
			if v := got.data[k]; v != w {
				t.Errorf("%s: data[%d] = %v, want %v", c.name, k, v, w)
			}
		}
	}
}

func TestTake3DRepeatedNegative(t *testing.T) {
	// a[i,j,k] = 100*i + 10*j + k
	a := FromFunction([]int{2, 3, 4}, func(coords ...int) float64 {
		return float64(100*coords[0] + 10*coords[1] + coords[2])
	})
	got := a.Take([]int{2, 0, 2, -2}, 1)
	if !slices.Equal(got.Shape(), []int{2, 4, 4}) {
		t.Fatalf("shape = %v, want [2 4 4]", got.Shape())
	}
	// Row j of the result along axis 1 is source row [2,0,2,1].
	checks := []struct{ i, k int }{{0, 0}, {0, 3}, {1, 0}, {1, 3}}
	for _, c := range checks {
		want := a.At(c.i, 2, c.k) // row 2 slot of the repeated pattern
		if v := got.At(c.i, 0, c.k); v != want {
			t.Errorf("At(%d,0,%d) = %v, want %v", c.i, c.k, v, want)
		}
	}
	if v, w := got.At(1, 3, 2), a.At(1, 1, 2); v != w {
		t.Errorf("negative index At(1,3,2) = %v, want %v", v, w)
	}
}

func TestTakeOutOfBoundsPanic(t *testing.T) {
	a := Zeros[float64](2, 3, 4)
	mustPanic(t, "numgo: index 5 is out of bounds for axis 1 with size 3",
		func() { a.Take([]int{1, 5}, 1) })
	mustPanic(t, "numgo: index -5 is out of bounds for axis 2 with size 4",
		func() { a.Take([]int{-5}, 2) })
}

func TestPutBroadcast1Row(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// Broadcast a 1-row of values into both selected rows.
	got := a.Put([]int{0, 1}, 0, FromSlice([]float64{7, 8, 9}, 1, 3))
	if got != a {
		t.Fatal("Put did not return the receiver")
	}
	want := []float64{7, 8, 9, 7, 8, 9}
	for k, w := range want {
		if v := a.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
}

func TestPutRepeatedIndicesNegativeAxis(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	a.Put([]int{-1, -1}, -1, New1DArray([]float64{0, 9}))
	if got := a.At(0, 2); got != 9 {
		t.Errorf("At(0,2) = %v, want 9", got)
	}
	if got := a.At(1, 2); got != 9 {
		t.Errorf("At(1,2) = %v, want 9", got)
	}
	if got := a.At(0, 1); got != 2 {
		t.Errorf("At(0,1) = %v, want 2 (untouched)", got)
	}
}

func TestPutShapeMismatchPanic(t *testing.T) {
	a := Zeros[float64](2, 3)
	mustPanic(t, "numgo: operands could not be broadcast together with shapes (4) (2,3)",
		func() { a.Put([]int{0, 1}, 0, New1DArray([]float64{1, 2, 3, 4})) })
}

func TestCompressAxes(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	cases := []struct {
		name  string
		mask  []bool
		axis  int
		shape []int
		want  []float64
	}{
		{"axis0 mixed", []bool{true, false}, 0, []int{1, 3}, []float64{1, 2, 3}},
		{"axis1 mixed", []bool{false, true, false}, 1, []int{2, 1}, []float64{2, 5}},
		{"axis1 all-false", []bool{false, false, false}, 1, []int{2, 0}, nil},
	}
	for _, c := range cases {
		got := a.Compress(c.mask, c.axis)
		if !slices.Equal(got.Shape(), c.shape) {
			t.Fatalf("%s: shape = %v, want %v", c.name, got.Shape(), c.shape)
		}
		if len(c.want) == 0 {
			if got.Size() != 0 {
				t.Errorf("%s: Size = %d, want 0", c.name, got.Size())
			}
			continue
		}
		for k, w := range c.want {
			if v := got.data[k]; v != w {
				t.Errorf("%s: data[%d] = %v, want %v", c.name, k, v, w)
			}
		}
	}
}

func TestCompress3DMaskLengthPanic(t *testing.T) {
	a := Zeros[float64](2, 3, 4)
	mustPanic(t, "numgo: mask length 3 does not match selection size 4",
		func() { a.Compress(make([]bool, 3), -1) })
}

func TestSetMaskFlatSemantics(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	mask := []bool{false, true, false, false, true, false}
	if got := a.SetMask(mask, -1); got != a {
		t.Fatal("SetMask did not return the receiver")
	}
	want := []float64{1, -1, 3, 4, -1, 6}
	for k, w := range want {
		if v := a.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
}

func TestSetMask3DShapeMismatchPanic(t *testing.T) {
	a := Zeros[float32](2, 3, 4)
	mustPanic(t, "numgo: mask length 10 does not match selection size 24",
		func() { a.SetMask(make([]bool, 10), 1) })
}

func TestIndexOneListRows(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got := a.Index([]int{1, 0, 1})
	if !slices.Equal(got.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v, want [3 3]", got.Shape())
	}
	want := []float64{4, 5, 6, 1, 2, 3, 4, 5, 6}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
}

func TestIndexTwoListsBroadcast3x2(t *testing.T) {
	// Non-square 3x2: a[i,j] = 10*i + j.
	a := FromSlice([]float64{0, 1, 10, 11, 20, 21}, 3, 2)
	// rows [0,1,2] x cols [1,0,1] (equal lengths, no repeat).
	got := a.Index([]int{0, 1, 2}, []int{1, 0, 1})
	if !slices.Equal(got.Shape(), []int{3}) {
		t.Fatalf("shape = %v, want [3]", got.Shape())
	}
	want := []float64{1, 10, 21}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
	// len-1 col list broadcasts: rows [0,2] x col [1].
	got = a.Index([]int{0, 2}, []int{1})
	if !slices.Equal(got.Shape(), []int{2}) {
		t.Fatalf("shape = %v, want [2]", got.Shape())
	}
	if got.At(0) != 1 || got.At(1) != 21 {
		t.Errorf("broadcast pick = %v, %v, want 1, 21", got.At(0), got.At(1))
	}
}

func TestIndexNegativeRepeated2x3(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got := a.Index([]int{-1, -2, 0})
	if !slices.Equal(got.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v, want [3 3]", got.Shape())
	}
	want := []float64{4, 5, 6, 1, 2, 3, 1, 2, 3}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
}

func TestIndexTailAxisRetention3D(t *testing.T) {
	// a[i,j,k] = 100*i + 10*j + k, non-square 2x3x4.
	a := FromFunction([]int{2, 3, 4}, func(coords ...int) float64 {
		return float64(100*coords[0] + 10*coords[1] + coords[2])
	})
	got := a.Index([]int{1, 0}, []int{2, 2})
	if !slices.Equal(got.Shape(), []int{2, 4}) {
		t.Fatalf("shape = %v, want [2 4]", got.Shape())
	}
	for i := range 2 {
		for k := range 4 {
			if v, w := got.At(i, k), a.At([]int{1, 0}[i], 2, k); v != w {
				t.Errorf("At(%d,%d) = %v, want %v", i, k, v, w)
			}
		}
	}
}

func TestIndexOutOfBoundsPanic(t *testing.T) {
	a := Zeros[float64](2, 3)
	mustPanic(t, "numgo: index 5 is out of bounds for axis 0 with size 2",
		func() { a.Index([]int{1, 5}) })
	mustPanic(t, "numgo: index -4 is out of bounds for axis 1 with size 3",
		func() { a.Index([]int{0}, []int{-4}) })
}

func TestIndexTooManyListsPanic(t *testing.T) {
	a := Zeros[float64](2, 3)
	mustPanic(t, "numgo: expected 2 ranges for a 2-d array, got 3",
		func() { a.Index([]int{0}, []int{0}, []int{0}) })
}

func TestPutIndicesBroadcastInto(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// rows [0,1] x col [2]: values 1-d of len 2 broadcast over tail axis.
	if got := a.PutIndices(New1DArray([]float64{7, 8}), []int{0, 1}, []int{2}); got != a {
		t.Fatal("PutIndices did not return the receiver")
	}
	if a.At(0, 2) != 7 || a.At(1, 2) != 8 {
		t.Errorf("At col 2 = %v, %v, want 7, 8", a.At(0, 2), a.At(1, 2))
	}
	if a.At(0, 0) != 1 || a.At(1, 0) != 4 {
		t.Errorf("other elements changed: %v, %v", a.At(0, 0), a.At(1, 0))
	}
}

func TestPutIndicesRepeatedUntouchedOnPanic(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	mustPanic(t, "numgo: index 3 is out of bounds for axis 0 with size 2",
		func() { a.PutIndices(New1DArray([]float64{9}), []int{0, 3}) })
	if a.At(0, 0) != 1 {
		t.Errorf("receiver mutated before panic: At(0,0) = %v", a.At(0, 0))
	}
	// Repeated index: last write wins.
	a.PutIndices(New1DArray([]float64{42}), []int{1, 1})
	if a.At(1, 0) != 42 {
		t.Errorf("repeated scatter At(1,0) = %v, want 42", a.At(1, 0))
	}
}

func TestMaskSelectFlatOrder(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got := a.MaskSelect([]bool{false, true, false, false, true, true})
	if !slices.Equal(got.Shape(), []int{3}) {
		t.Fatalf("shape = %v, want [3]", got.Shape())
	}
	want := []float64{2, 5, 6}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
}

func TestMaskSelectLengthPanic(t *testing.T) {
	a := Zeros[float32](2, 3, 4)
	mustPanic(t, "numgo: mask length 10 does not match selection size 24",
		func() { a.MaskSelect(make([]bool, 10)) })
}

func TestIndexOnTransposeView(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	ta := a.Transpose() // 3x2 view
	got := ta.Index([]int{0, 2, 1})
	if !slices.Equal(got.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got.Shape())
	}
	want := []float64{1, 4, 3, 6, 2, 5}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
	got.Set([]int{0, 0}, 99)
	if a.At(0, 0) != 1 {
		t.Fatal("Index result aliases the source view")
	}
}

func TestTakeOnTransposeView(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	ta := a.Transpose() // 3x2 view of the same data
	got := ta.Take([]int{0, 2, 1}, 0)
	if !slices.Equal(got.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got.Shape())
	}
	want := []float64{1, 4, 3, 6, 2, 5}
	for k, w := range want {
		if v := got.data[k]; v != w {
			t.Errorf("data[%d] = %v, want %v", k, v, w)
		}
	}
	// The result must be independent of the view's underlying data.
	got.Set([]int{0, 0}, 99)
	if a.At(0, 0) != 1 {
		t.Fatal("Take result aliases the source view")
	}
}
