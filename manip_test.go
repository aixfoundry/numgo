package numgo

import (
	"slices"
	"testing"
)

func TestConcatenate(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{7, 8, 9, 10, 11, 12}, 2, 3)
	ax0 := Concatenate(0, a, b)
	if !slices.Equal(ax0.Shape(), []int{4, 3}) || ax0.At(2, 0) != 7 || ax0.At(3, 2) != 12 {
		t.Fatalf("axis0 shape=%v at(2,0)=%v", ax0.Shape(), ax0.At(2, 0))
	}
	ax1 := Concatenate(1, a, b)
	if !slices.Equal(ax1.Shape(), []int{2, 6}) || ax1.At(1, 3) != 10 || ax1.At(0, 5) != 9 {
		t.Fatalf("axis1 shape=%v at(1,3)=%v", ax1.Shape(), ax1.At(1, 3))
	}
	axn := Concatenate(-1, a, b)
	if !slices.Equal(axn.Shape(), []int{2, 6}) {
		t.Fatalf("negative axis shape=%v", axn.Shape())
	}
	three := Concatenate(0, a, b, a)
	if !slices.Equal(three.Shape(), []int{6, 3}) || three.At(4, 1) != 2 {
		t.Fatalf("3-way shape=%v", three.Shape())
	}
	c := FromSlice([]float32{1, 2, 3}, 1, 3)
	d := FromSlice([]float32{4, 5, 6}, 1, 3)
	if got := Concatenate(0, c, d); !slices.Equal(got.Shape(), []int{2, 3}) || got.At(1, 0) != 4 {
		t.Fatalf("float32 concat: %v", got.Shape())
	}
	mustPanic(t,
		"numgo: all the input array dimensions except for the concatenation axis must match exactly",
		func() { Concatenate(0, a, FromSlice([]float64{1, 2, 3}, 3)) })
	mustPanic(t, "numgo: need at least one array to concatenate", func() { Concatenate[float64](0) })
}

func TestStack(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := FromSlice([]float64{5, 6, 7, 8}, 2, 2)
	s := Stack(0, a, b)
	if !slices.Equal(s.Shape(), []int{2, 2, 2}) || s.At(1, 0, 1) != 6 {
		t.Fatalf("shape=%v at(1,0,1)=%v", s.Shape(), s.At(1, 0, 1))
	}
	s1 := Stack(1, a, b)
	if !slices.Equal(s1.Shape(), []int{2, 2, 2}) || s1.At(0, 1, 1) != 6 {
		t.Fatalf("axis1 shape=%v at(0,1,1)=%v", s1.Shape(), s1.At(0, 1, 1))
	}
	sneg := Stack(-1, a, b)
	if !slices.Equal(sneg.Shape(), []int{2, 2, 2}) || sneg.At(0, 1, 1) != 6 {
		t.Fatalf("axis-1 at(0,1,1)=%v", sneg.At(0, 1, 1))
	}
	mustPanic(t, "numgo: all input arrays must have the same shape",
		func() { Stack(0, a, FromSlice([]float64{1, 2}, 1, 2)) })
}

func TestSplit(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	parts := Split(a, []int{1}, 1)
	if len(parts) != 2 || !slices.Equal(parts[0].Shape(), []int{2, 1}) || !slices.Equal(parts[1].Shape(), []int{2, 2}) {
		t.Fatalf("shapes %v %v", parts[0].Shape(), parts[1].Shape())
	}
	parts[1].Set([]int{1, 1}, 99)
	if a.At(1, 2) != 99 {
		t.Fatal("Split views do not share data")
	}
	rows := Split(a, []int{1}, 0) // [1,2,3 | 4,99,6] after the mutation above
	if len(rows) != 2 || rows[0].At(0, 2) != 3 || rows[1].At(0, 2) != 99 || rows[1].At(0, 0) != 4 {
		t.Fatalf("axis0 split: %v", rows[1].Shape())
	}
	mustPanic(t, "numgo: split indices must be ascending and within the axis bounds",
		func() { Split(a, []int{2, 1}, 1) })
}

func TestTile(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := Tile(a, 2)
	if !slices.Equal(t2.Shape(), []int{2, 6}) || t2.At(0, 3) != 1 || t2.At(1, 5) != 6 {
		t.Fatalf("reps (2) shape=%v at(0,3)=%v", t2.Shape(), t2.At(0, 3))
	}
	t12 := Tile(a, 1, 2)
	if !slices.Equal(t12.Shape(), []int{2, 6}) || t12.At(0, 3) != 1 || t12.At(1, 3) != 4 || t12.At(0, 0) != 1 {
		t.Fatalf("reps (1,2) at(1,3)=%v", t12.At(1, 3))
	}
	t22 := Tile(a, 2, 2)
	if !slices.Equal(t22.Shape(), []int{4, 6}) || t22.At(2, 0) != 1 || t22.At(3, 5) != 6 {
		t.Fatalf("reps (2,2) at(2,0)=%v", t22.At(2, 0))
	}
	t3 := Tile(a, 1, 2, 1) // a right-aligns to (1,2,3), so axis 1 tiles: (1,4,3)
	if !slices.Equal(t3.Shape(), []int{1, 4, 3}) || t3.At(0, 1, 2) != 6 || t3.At(0, 3, 2) != 6 {
		t.Fatalf("longer reps shape=%v", t3.Shape())
	}
}

func TestRepeat(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	r0 := Repeat(a, 2, 0)
	if !slices.Equal(r0.Shape(), []int{4, 3}) || r0.At(1, 0) != 1 || r0.At(2, 0) != 4 || r0.At(3, 2) != 6 {
		t.Fatalf("axis0 at(1,0)=%v at(2,0)=%v", r0.At(1, 0), r0.At(2, 0))
	}
	r1 := Repeat(a, 3, 1)
	if !slices.Equal(r1.Shape(), []int{2, 9}) || r1.At(0, 0) != 1 || r1.At(0, 2) != 1 || r1.At(0, 3) != 2 || r1.At(1, 8) != 6 {
		t.Fatalf("axis1 at(0,3)=%v", r1.At(0, 3))
	}
	mustPanic(t, "numgo: repeat count -1 must be non-negative", func() { Repeat(a, -1, 0) })
}

func TestFlip(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	f := Flip(a)
	if f.At(0, 0) != 6 || f.At(0, 1) != 5 || f.At(0, 2) != 4 || f.At(1, 0) != 3 || f.At(1, 2) != 1 {
		t.Fatalf("flip all: %v %v", f.At(0, 0), f.At(1, 2))
	}
	f0 := Flip(a, 0)
	if f0.At(0, 0) != 4 || f0.At(1, 0) != 1 || f0.At(1, 2) != 3 {
		t.Fatalf("flip axis0: %v", f0.At(0, 0))
	}
	f1 := Flip(a, -1)
	if f1.At(0, 0) != 3 || f1.At(1, 0) != 6 {
		t.Fatalf("flip axis-1: %v", f1.At(0, 0))
	}
}

func TestRoll(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	r := Roll(a, 2)
	if r.At(0, 0) != 5 || r.At(0, 1) != 6 || r.At(1, 0) != 2 || !slices.Equal(r.Shape(), []int{2, 3}) {
		t.Fatalf("flat roll +2: %v", r.At(0, 0))
	}
	rn := Roll(a, -1)
	if rn.At(0, 0) != 2 || rn.At(1, 2) != 1 {
		t.Fatalf("flat roll -1: %v", rn.At(0, 0))
	}
	ra := Roll(a, 1, 0)
	if ra.At(0, 0) != 4 || ra.At(1, 0) != 1 || ra.At(1, 2) != 3 {
		t.Fatalf("roll axis0: %v", ra.At(0, 0))
	}
	rb := Roll(a, -1, 1)
	if rb.At(0, 0) != 2 || rb.At(0, 2) != 1 {
		t.Fatalf("roll axis1: %v", rb.At(0, 0))
	}
}

func TestCopyInto(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	dst := Zeros[float64](2, 3)
	CopyInto(dst, a)
	if dst.At(1, 2) != 6 {
		t.Fatal("plain copy failed")
	}
	b := FromSlice([]float64{10, 20}, 2, 1)
	CopyInto(dst, b)
	if dst.At(0, 0) != 10 || dst.At(0, 2) != 10 || dst.At(1, 1) != 20 {
		t.Fatalf("broadcast copy: %v %v", dst.At(0, 2), dst.At(1, 1))
	}
	mustPanic(t, "numgo: operands could not be broadcast together with shapes (2,1) (3,3)",
		func() { CopyInto(Full(0.0, 3, 3), b) })
}

func TestFromFunction(t *testing.T) {
	a := FromFunction([]int{2, 3, 4}, func(coords ...int) float64 {
		return float64(coords[0]*100 + coords[1]*10 + coords[2])
	})
	if !slices.Equal(a.Shape(), []int{2, 3, 4}) || a.At(1, 2, 3) != 123 || a.At(0, 1, 2) != 12 {
		t.Fatalf("shape=%v at(1,2,3)=%v", a.Shape(), a.At(1, 2, 3))
	}
}

func TestMeshGrid(t *testing.T) {
	x := Arange(0.0, 2.0, 1.0) // [0 1]
	y := Arange(0.0, 3.0, 1.0) // [0 1 2]
	g := MeshGrid(x, y)
	if len(g) != 2 || !slices.Equal(g[0].Shape(), []int{2, 3}) {
		t.Fatalf("len=%d shape=%v", len(g), g[0].Shape())
	}
	if g[0].At(0, 0) != 0 || g[0].At(1, 2) != 1 || g[1].At(1, 2) != 2 {
		t.Fatalf("values %v %v", g[0].At(1, 2), g[1].At(1, 2))
	}
	mustPanic(t, "numgo: MeshGrid inputs must be 1-D",
		func() { MeshGrid(x, Full(0.0, 2, 2)) })
}

func TestIdentity(t *testing.T) {
	a := Identity[float64](3)
	if a.Ndim() != 2 || a.At(1, 1) != 1 || a.At(0, 1) != 0 || a.At(2, 0) != 0 {
		t.Fatalf("values %v %v", a.At(1, 1), a.At(0, 1))
	}
	if Identity[float64](0).Size() != 0 {
		t.Fatal("n=0 should be empty")
	}
}
