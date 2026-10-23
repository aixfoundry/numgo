package numgo

import (
	"math"
	"slices"
	"testing"
)

var reduceData = []float64{1, 2, 3, 4, 5, 6} // [[1 2 3] [4 5 6]]

func reduce2x3() *Array[float64] {
	return FromSlice(reduceData, 2, 3)
}

func TestSumAxes(t *testing.T) {
	a := reduce2x3()
	cases := []struct {
		name  string
		axes  []int
		shape []int
		want  []float64
	}{
		{"axis0", []int{0}, []int{3}, []float64{5, 7, 9}},
		{"axis1", []int{1}, []int{2}, []float64{6, 15}},
		{"axis-1", []int{-1}, []int{2}, []float64{6, 15}},
		{"all", nil, []int{1}, []float64{21}},
	}
	for _, c := range cases {
		got := a.Sum(c.axes...)
		if !slices.Equal(got.Shape(), c.shape) {
			t.Errorf("%s: shape = %v, want %v", c.name, got.Shape(), c.shape)
		}
		for i, w := range c.want {
			if v := got.At(i); v != w {
				t.Errorf("%s: [%d] = %v, want %v", c.name, i, v, w)
			}
		}
	}
	if a.At(0, 0) != 1 {
		t.Error("Sum mutated its receiver")
	}
}

func TestSumKeepdims(t *testing.T) {
	a := reduce2x3()
	got := a.SumKeepdims(1)
	if !slices.Equal(got.Shape(), []int{2, 1}) {
		t.Fatalf("shape = %v, want [2 1]", got.Shape())
	}
	if got.At(0, 0) != 6 || got.At(1, 0) != 15 {
		t.Fatalf("values = %v, %v", got.At(0, 0), got.At(1, 0))
	}
	all := a.SumKeepdims()
	if !slices.Equal(all.Shape(), []int{1, 1}) || all.At(0, 0) != 21 {
		t.Fatalf("whole keepdims = shape %v val %v", all.Shape(), all.At(0, 0))
	}
}

func TestSum3DMultiAxis(t *testing.T) {
	a := FromSlice([]float64{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,

		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,
	}, 2, 3, 4)
	got := a.Sum(0, 2)
	if !slices.Equal(got.Shape(), []int{3}) {
		t.Fatalf("shape = %v, want [3]", got.Shape())
	}
	// numpy: arange(24).reshape(2,3,4).sum(axis=(0,2)) = [60 92 124]
	want := []float64{60, 92, 124}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestProdMaxMin(t *testing.T) {
	a := reduce2x3()
	if got := a.Prod(0); got.At(0) != 4 || got.At(1) != 10 || got.At(2) != 18 {
		t.Errorf("Prod(0) = %v %v %v", got.At(0), got.At(1), got.At(2))
	}
	if got := a.Prod(); got.Size() != 1 || got.At(0) != 720 {
		t.Errorf("Prod() = %v", got.At(0))
	}
	max0 := a.Max(0)
	if max0.At(0) != 4 || max0.At(1) != 5 || max0.At(2) != 6 {
		t.Errorf("Max(0) wrong: %v", max0.data)
	}
	min1 := a.Min(1)
	if min1.At(0) != 1 || min1.At(1) != 4 {
		t.Errorf("Min(1) wrong: %v", min1.data)
	}
	if v := a.Max(); v.At(0) != 6 {
		t.Errorf("Max() = %v", v.At(0))
	}
	if v := a.Min(); v.At(0) != 1 {
		t.Errorf("Min() = %v", v.At(0))
	}
	if !slices.Equal(a.MaxKeepdims(0).Shape(), []int{1, 3}) {
		t.Errorf("MaxKeepdims(0) shape = %v", a.MaxKeepdims(0).Shape())
	}
}

func TestMeanVarStd(t *testing.T) {
	a := reduce2x3()
	m0 := a.Mean(0)
	if m0.At(0) != 2.5 || m0.At(1) != 3.5 || m0.At(2) != 4.5 {
		t.Errorf("Mean(0) wrong: %v", m0.data)
	}
	mAll := a.Mean()
	if mAll.At(0) != 3.5 {
		t.Errorf("Mean() = %v", mAll.At(0))
	}
	// numpy: var(axis=1) = [2/3, 2/3]; var() = 35/12; std() = sqrt(35/12)
	v1 := a.Var(1)
	if math.Abs(v1.At(0)-2.0/3.0) > 1e-12 || math.Abs(v1.At(1)-2.0/3.0) > 1e-12 {
		t.Errorf("Var(1) = %v", v1.data)
	}
	vAll := a.Var()
	if math.Abs(vAll.At(0)-35.0/12.0) > 1e-12 {
		t.Errorf("Var() = %v, want %v", vAll.At(0), 35.0/12.0)
	}
	sAll := a.Std()
	if math.Abs(sAll.At(0)-math.Sqrt(35.0/12.0)) > 1e-12 {
		t.Errorf("Std() = %v", sAll.At(0))
	}
	// ddof=1: var() = 4.2 (numpy: a.var(ddof=1) with 6 elems = 3.5)
	vDdof := a.VarDDof(1)
	if math.Abs(vDdof.At(0)-3.5) > 1e-12 {
		t.Errorf("VarDDof(1) = %v, want 3.5", vDdof.At(0))
	}
	sDdof := a.StdDDof(1)
	if math.Abs(sDdof.At(0)-math.Sqrt(3.5)) > 1e-12 {
		t.Errorf("StdDDof(1) = %v", sDdof.At(0))
	}
	// Var of a 2x3 along axis 0 = [2.25 2.25 2.25]
	v0 := a.Var(0)
	for i := 0; i < 3; i++ {
		if v0.At(i) != 2.25 {
			t.Errorf("Var(0)[%d] = %v, want 2.25", i, v0.At(i))
		}
	}
}

func TestMeanComplex(t *testing.T) {
	a := FromSlice([]complex128{1 + 2i, 3 + 4i}, 2)
	m := a.Mean()
	if m.At(0) != 2+3i {
		t.Errorf("complex Mean = %v, want (2+3i)", m.At(0))
	}
	s := a.Sum()
	if s.At(0) != 4+6i {
		t.Errorf("complex Sum = %v", s.At(0))
	}
}

func TestStdComplexPanics(t *testing.T) {
	a := FromSlice([]complex128{1 + 2i, 3 + 4i}, 2)
	mustPanic(t, "numgo: Var/Std are not defined for complex arrays", func() { a.Var() })
	mustPanic(t, "numgo: Var/Std are not defined for complex arrays", func() { a.Std(0) })
}

func TestArgMaxMin(t *testing.T) {
	a := reduce2x3()
	if got := a.ArgMax(0); !slices.Equal(got, []int{1, 1, 1}) {
		t.Errorf("ArgMax(0) = %v, want [1 1 1]", got)
	}
	if got := a.ArgMax(1); !slices.Equal(got, []int{2, 2}) {
		t.Errorf("ArgMax(1) = %v, want [2 2]", got)
	}
	if got := a.ArgMin(0); !slices.Equal(got, []int{0, 0, 0}) {
		t.Errorf("ArgMin(0) = %v, want [0 0 0]", got)
	}
	if got := a.ArgMin(-1); !slices.Equal(got, []int{0, 0}) {
		t.Errorf("ArgMin(-1) = %v, want [0 0]", got)
	}
	// Tie-breaking: first occurrence wins.
	ties := FromSlice([]float64{5, 5, 5, 1, 1, 1}, 2, 3)
	if got := ties.ArgMax(0); !slices.Equal(got, []int{0, 0, 0}) {
		t.Errorf("ArgMax tie = %v, want [0 0 0]", got)
	}
	// 3-D: argmax/argmin over middle axis of 2x3x4 (numpy-verified).
	b := FromSlice([]float64{
		0, 8, 2, 3,
		4, 1, 6, 7,
		9, 5, 10, 11,

		12, 2, 14, 4,
		16, 17, 1, 19,
		7, 21, 22, 3,
	}, 2, 3, 4)
	if got := b.ArgMax(1); !slices.Equal(got, []int{2, 0, 2, 2, 1, 2, 2, 1}) {
		t.Errorf("ArgMax(1) 3d = %v", got)
	}
	if got := b.ArgMin(1); !slices.Equal(got, []int{0, 1, 0, 0, 2, 0, 1, 2}) {
		t.Errorf("ArgMin(1) 3d = %v", got)
	}
}

func TestInvalidAxisPanics(t *testing.T) {
	a := reduce2x3()
	mustPanic(t, "numgo: axis 2 is out of bounds for 2-d array", func() { a.Sum(2) })
	// -3 normalizes to -1 which is still out of bounds after wrap; resolveAxis
	// reports the wrapped value, matching the existing convention.
	mustPanic(t, "numgo: axis -1 is out of bounds for 2-d array", func() { a.Sum(-3) })
	mustPanic(t, "numgo: axis 5 is out of bounds for 2-d array", func() { a.ArgMax(5) })
}

func TestAnyAll(t *testing.T) {
	if !(FromSlice([]float64{0, 1}, 2)).Any() {
		t.Error("Any should be true")
	}
	if FromSlice([]float64{0, 0}, 2).Any() {
		t.Error("Any should be false")
	}
	if !FromSlice([]float64{1, 2}, 2).All() {
		t.Error("All should be true")
	}
	if FromSlice([]float64{1, 0}, 2).All() {
		t.Error("All should be false")
	}
	if !Zeros[float64]().All() {
		t.Error("zero-size All should be vacuously true")
	}
	if Zeros[float64]().Any() {
		t.Error("zero-size Any should be false")
	}
	if !(FromSlice([]complex128{1i}, 1)).Any() {
		t.Error("complex imaginary-only Any should be true")
	}
}

func TestCumSum(t *testing.T) {
	a := reduce2x3()
	// axis 0: [[1 2 3] [5 7 9]]
	got0 := a.CumSum(0)
	if !slices.Equal(got0.Shape(), []int{2, 3}) {
		t.Fatalf("CumSum shape = %v", got0.Shape())
	}
	want0 := []float64{1, 2, 3, 5, 7, 9}
	for i, w := range want0 {
		if v := got0.At(i/3, i%3); v != w {
			t.Errorf("CumSum(0)[%d] = %v, want %v", i, v, w)
		}
	}
	// axis 1 and -1: [[1 3 6] [4 9 15]]
	for _, ax := range []int{1, -1} {
		got := a.CumSum(ax)
		want := []float64{1, 3, 6, 4, 9, 15}
		for i, w := range want {
			if v := got.At(i/3, i%3); v != w {
				t.Errorf("CumSum(%d)[%d] = %v, want %v", ax, i, v, w)
			}
		}
	}
	// no axes: numpy's cumsum default folds the raveled array into 1-d.
	flat := a.CumSum()
	if !slices.Equal(flat.Shape(), []int{6}) {
		t.Fatalf("CumSum() shape = %v, want [6]", flat.Shape())
	}
	want := []float64{1, 3, 6, 10, 15, 21}
	for i, w := range want {
		if v := flat.At(i); v != w {
			t.Errorf("CumSum()[%d] = %v, want %v", i, v, w)
		}
	}
	if a.At(0, 0) != 1 {
		t.Error("CumSum mutated its receiver")
	}
}

func TestCumProd(t *testing.T) {
	a := reduce2x3()
	got := a.CumProd(1)
	// numpy: cumprod axis1 = [1 2 6 4 20 120]
	want := []float64{1, 2, 6, 4, 20, 120}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("CumProd(1)[%d] = %v, want %v", i, v, w)
		}
	}
	flat := a.CumProd()
	if !slices.Equal(flat.Shape(), []int{6}) {
		t.Fatalf("CumProd() shape = %v, want [6]", flat.Shape())
	}
	wantFlat := []float64{1, 2, 6, 24, 120, 720}
	for i, w := range wantFlat {
		if v := flat.At(i); v != w {
			t.Errorf("CumProd()[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEmptyReductions(t *testing.T) {
	// 0x3 array: reducing along the empty axis gives identity per kept slot.
	// Max/Min of zero data return ±Inf as documented (numpy errors instead).
	a := Zeros[float64](0, 3)
	if got := a.Sum(0); !slices.Equal(got.Shape(), []int{3}) || got.At(0) != 0 {
		t.Errorf("empty Sum(0) = %v %v", got.Shape(), got.data)
	}
	if got := a.Max(0); got.At(0) != math.Inf(-1) {
		t.Errorf("empty Max(0) = %v, want -Inf", got.At(0))
	}
	if got := a.Min(0); got.At(0) != math.Inf(1) {
		t.Errorf("empty Min(0) = %v, want +Inf", got.At(0))
	}
	if got := a.Prod(0); got.At(0) != 1 {
		t.Errorf("empty Prod(0) = %v, want 1", got.At(0))
	}
	if got := a.Var(0); !math.IsNaN(got.At(0)) {
		t.Errorf("empty Var(0) = %v, want NaN", got.At(0))
	}
	if got := a.Mean(0); !math.IsNaN(got.At(0)) {
		t.Errorf("empty Mean(0) = %v, want NaN", got.At(0))
	}
	// Fully empty reduction: 1-element identity.
	if got := a.Sum(); got.Size() != 1 || got.At(0) != 0 {
		t.Errorf("empty Sum() = %v", got.data)
	}
	if got := a.Max(); got.Size() != 1 || got.At(0) != math.Inf(-1) {
		t.Errorf("empty Max() = %v, want [-Inf]", got.data)
	}
	if got := a.Min(); got.Size() != 1 || got.At(0) != math.Inf(1) {
		t.Errorf("empty Min() = %v, want [+Inf]", got.data)
	}
	if got := a.Mean(); !math.IsNaN(got.At(0)) {
		t.Errorf("empty Mean() = %v, want NaN", got.At(0))
	}
	// 3x0: reducing axis 1 hits the empty axis.
	b := Zeros[float64](3, 0)
	if got := b.Sum(1); !slices.Equal(got.Shape(), []int{3}) || got.At(2) != 0 {
		t.Errorf("3x0 Sum(1) = %v %v", got.Shape(), got.data)
	}
}

func TestSumFloat32(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	got := a.Sum(0)
	if got.At(0) != 5 || got.At(1) != 7 || got.At(2) != 9 {
		t.Errorf("float32 Sum(0) = %v", got.data)
	}
	if got := a.CumSum(1); got.At(1, 0) != 4 || got.At(1, 2) != 15 || got.At(0, 1) != 3 {
		t.Errorf("float32 CumSum(1) = %v", got.data)
	}
}

func TestReductionOnView(t *testing.T) {
	// Reductions must respect strides: sum a transposed 2x3.
	a := reduce2x3().Transpose()
	// a.T is 3x2: [[1 4] [2 5] [3 6]]
	got := a.Sum(1)
	if got.At(0) != 5 || got.At(1) != 7 || got.At(2) != 9 {
		t.Errorf("transposed Sum(1) = %v", got.data)
	}
	if got := a.CumSum(1); got.At(2, 1) != 9 {
		t.Errorf("transposed CumSum(1)(2,1) = %v, want 9", got.At(2, 1))
	}
}
