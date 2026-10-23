package numgo

import (
	"math"
	"slices"
	"testing"
)

func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected panic %q, got none", want)
		}
		if got, ok := r.(string); !ok || got != want {
			t.Fatalf("panic = %v, want %q", r, want)
		}
	}()
	f()
}

func TestShapeAndSize(t *testing.T) {
	a := Full(7.0, 2, 3, 4)
	if got := a.Shape(); !slices.Equal(got, []int{2, 3, 4}) {
		t.Fatalf("Shape = %v", got)
	}
	if a.Ndim() != 3 || a.Size() != 24 || !a.IsContiguous() {
		t.Fatalf("Ndim=%d Size=%d IsContiguous=%v", a.Ndim(), a.Size(), a.IsContiguous())
	}
	shape := a.Shape()
	shape[0] = 99
	if a.Shape()[0] != 2 {
		t.Fatal("Shape returned aliased storage")
	}
}

func TestCopyIndependent(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Copy()
	b.Set([]int{1, 2}, 99)
	if a.At(1, 2) != 6 {
		t.Fatalf("original mutated through Copy: %v", a.At(1, 2))
	}
	if !b.IsContiguous() || b.Size() != 6 {
		t.Fatal("copy is not compact")
	}
}

func TestReshapeViewSharesData(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Reshape(3, 2)
	b.Set([]int{2, 1}, 99)
	if a.At(1, 2) != 99 {
		t.Fatalf("reshape did not share data: %v", a.At(1, 2))
	}
	if !slices.Equal(b.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v", b.Shape())
	}
}

func TestReshapeInferAndPanics(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := a.Reshape(2, -1).Shape(); !slices.Equal(got, []int{2, 3}) {
		t.Fatalf("shape = %v", got)
	}
	if got := a.Reshape(3, -1).Shape(); !slices.Equal(got, []int{3, 2}) {
		t.Fatalf("shape = %v", got)
	}
	mustPanic(t, "numgo: cannot reshape array of size 6 into shape (4,2)", func() { a.Reshape(4, 2) })
	mustPanic(t, "numgo: can only specify one unknown dimension (-1)", func() { a.Reshape(-1, -1) })
}

func TestReshapeNonContiguousCopies(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	tr := a.Transpose() // 3x2, non-contiguous
	b := tr.Reshape(6)
	if b.IsContiguous() != true || !slices.Equal(b.Shape(), []int{6}) {
		t.Fatalf("shape=%v contiguous=%v", b.Shape(), b.IsContiguous())
	}
	// C-order traversal of tr is 1,4,2,5,3,6.
	if got := b.At(1); got != 4 {
		t.Fatalf("At(1) = %v, want 4", got)
	}
	b.Set([]int{0}, 99)
	if a.At(0, 0) != 99 {
		// A copy must be independent.
		return
	}
	t.Fatal("reshape of non-contiguous array mutated the source")
}

func TestTransposeViewSharesData(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	tr := a.Transpose()
	if !slices.Equal(tr.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v", tr.Shape())
	}
	if got := tr.At(2, 1); got != 6 {
		t.Fatalf("At(2,1) = %v, want 6", got)
	}
	tr.Set([]int{1, 0}, 77)
	if a.At(0, 1) != 77 {
		t.Fatal("transpose did not share data")
	}
	if a.IsContiguous() == false || tr.IsContiguous() {
		t.Fatal("contiguity flags wrong")
	}
}

func TestTranspose3DReverseAndPermute(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2)
	rev := a.Transpose()
	if !slices.Equal(rev.Shape(), []int{2, 3, 2}) { // square-ish; use explicit permute below
		t.Fatalf("reverse shape = %v", rev.Shape())
	}
	if got := rev.At(1, 2, 0); got != a.At(0, 2, 1) {
		t.Fatalf("reverse mapping: %v != %v", got, a.At(0, 2, 1))
	}
	perm := a.Transpose(1, 0, 2) // 3x2x2
	if !slices.Equal(perm.Shape(), []int{3, 2, 2}) {
		t.Fatalf("permute shape = %v", perm.Shape())
	}
	if got := perm.At(2, 1, 1); got != a.At(1, 2, 1) {
		t.Fatalf("permute mapping: %v != %v", got, a.At(1, 2, 1))
	}
	mustPanic(t, "numgo: axes don't match array", func() { a.Transpose(0, 1) })
	mustPanic(t, "numgo: axes don't match array", func() { a.Transpose(0, 0, 1) })
}

func TestShapeMismatchPanicMessage(t *testing.T) {
	a := Full(1.0, 2, 3)
	b := Full(1.0, 1, 2)
	mustPanic(t,
		"numgo: operands could not be broadcast together with shapes (2,3) (1,2)",
		func() { a.Add(b) })
}

func TestDivByZeroIsInf(t *testing.T) {
	a := FromSlice([]float64{1, -1, 0}, 3)
	z := Zeros[float64](3)
	q := a.Div(z)
	if !math.IsInf(q.At(0), 1) || !math.IsInf(q.At(1), -1) || !math.IsNaN(q.At(2)) {
		t.Fatalf("got %v", []float64{q.At(0), q.At(1), q.At(2)})
	}
}

func TestSqueezeExpandDimsSwapAxes(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 1, 2)
	s := a.Squeeze()
	if !slices.Equal(s.Shape(), []int{2, 2}) || s.At(1, 1) != 4 {
		t.Fatalf("squeeze all shape=%v", s.Shape())
	}
	s1 := a.Squeeze(1)
	if !slices.Equal(s1.Shape(), []int{2, 2}) {
		t.Fatalf("squeeze 1 shape=%v", s1.Shape())
	}
	mustPanic(t, "numgo: cannot select an axis to squeeze out which has size not equal to one",
		func() { a.Squeeze(0) })
	mustPanic(t, "numgo: axis 5 is out of bounds for 3-d array", func() { a.Squeeze(5) })

	e := a.Squeeze().ExpandDims(1)
	if !slices.Equal(e.Shape(), []int{2, 1, 2}) || e.At(1, 0, 1) != 4 {
		t.Fatalf("expand shape=%v", e.Shape())
	}
	if !slices.Equal(a.Squeeze().ExpandDims(-1).Shape(), []int{2, 2, 1}) {
		t.Fatal("expand -1")
	}
	mustPanic(t, "numgo: axis 3 is out of bounds for 2-d array", func() { a.Squeeze().ExpandDims(3) })

	sw := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3).SwapAxes(0, 1)
	if !slices.Equal(sw.Shape(), []int{3, 2}) || sw.At(2, 1) != 6 {
		t.Fatalf("swap shape=%v at(2,1)=%v", sw.Shape(), sw.At(2, 1))
	}
	mustPanic(t, "numgo: axis 2 is out of bounds for 2-d array", func() { sw.SwapAxes(0, 2) })
}

func TestSqueezeExpandShareData(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 1, 2)
	a.Squeeze(1).Set([]int{1, 1}, 99)
	if a.At(1, 0, 1) != 99 {
		t.Fatal("Squeeze does not share data")
	}
	e := a.Squeeze().ExpandDims(1) // still a view
	e.Set([]int{0, 0, 1}, 42)
	if a.At(0, 0, 1) != 42 {
		t.Fatal("ExpandDims does not share data")
	}
}

func TestRavelViewVsCopy(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	r := a.Ravel()
	if !slices.Equal(r.Shape(), []int{6}) || r.At(4) != 5 {
		t.Fatalf("ravel shape=%v", r.Shape())
	}
	r.Set([]int{0}, 99)
	if a.At(0, 0) != 99 {
		t.Fatal("contiguous Ravel is not a view")
	}
	v := a.Transpose() // non-contiguous
	rv := v.Ravel()
	rv.Set([]int{1}, 77)
	if v.At(0, 1) == 77 && v.At(1, 0) != 77 {
		t.Fatal("unexpected transpose sharing")
	}
	if v.Ravel().At(1) != 4 {
		t.Fatalf("ravel of transposed = %v", v.Ravel().At(1))
	}
}

func TestBroadcastTo(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 1, 3)
	b := a.BroadcastTo(2, 3, 3)
	if !slices.Equal(b.Shape(), []int{2, 3, 3}) || b.At(0, 0, 2) != 3 || b.At(1, 0, 2) != 3 {
		t.Fatalf("shape=%v at(1,0,2)=%v", b.Shape(), b.At(1, 0, 2))
	}
	if !slices.Equal(a.BroadcastTo(4, 3).Shape(), []int{4, 3}) {
		t.Fatal("same-rank broadcast")
	}
	mustPanic(t, "numgo: operands could not be broadcast together with shapes (1,3) (2)",
		func() { a.BroadcastTo(2) })
	// stride-0 aliasing: writing through the view hits every expanded slot.
	c := FromSlice([]float64{7}, 1).BroadcastTo(3)
	c.Set([]int{0}, 9)
	if c.At(1) != 9 || c.At(2) != 9 {
		t.Fatal("stride-0 expansion does not alias")
	}
}
