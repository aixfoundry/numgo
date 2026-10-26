package numgo

import (
	"math"
	"slices"
	"testing"
)

func TestCompareGreater(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{3, 2, 1, 0, -1, 9}, 2, 3)
	got := a.Greater(b)
	want := []bool{false, false, true, true, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("Greater = %v, want %v", got, want)
	}
	if !slices.Equal(a.Less(b), []bool{true, false, false, false, false, true}) {
		t.Errorf("Less mismatch: %v", a.Less(b))
	}
	if !slices.Equal(a.Equal(b), []bool{false, true, false, false, false, false}) {
		t.Errorf("Equal mismatch: %v", a.Equal(b))
	}
	if !slices.Equal(a.NotEqual(b), complement(a.Equal(b))) {
		t.Errorf("NotEqual mismatch")
	}
	if !slices.Equal(a.GreaterEqual(b), or(got, a.Equal(b))) {
		t.Errorf("GreaterEqual mismatch")
	}
	if !slices.Equal(a.LessEqual(b), complement(a.Greater(b))) {
		t.Errorf("LessEqual mismatch")
	}
}

func TestCompareBroadcast(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3}, 3, 1)
	b := FromSlice([]float64{2, 0, 2}, 3)
	got := a.Greater(b)
	if len(got) != 9 {
		t.Fatalf("len = %d, want 9", len(got))
	}
	// a[i][0] > b[j] for i in 0..2, j in 0..2.
	var want []bool
	for i := range 3 {
		for j := range 3 {
			want = append(want, float64(i+1) > b.data[j])
		}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("broadcast Greater = %v, want %v", got, want)
	}
	// Scalar family cross-checked against a handwritten loop.
	m := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	cmpValue := func(f func(float64, float64) bool, v float64) []bool {
		var w []bool
		for _, x := range []float64{0, 1, 2, 3, 4, 5} {
			w = append(w, f(x, v))
		}
		return w
	}
	if !slices.Equal(m.GreaterValue(2), cmpValue(func(x, v float64) bool { return x > v }, 2)) {
		t.Errorf("GreaterValue mismatch")
	}
	if !slices.Equal(m.GreaterEqualValue(2), cmpValue(func(x, v float64) bool { return x >= v }, 2)) {
		t.Errorf("GreaterEqualValue mismatch")
	}
	if !slices.Equal(m.LessValue(2), cmpValue(func(x, v float64) bool { return x < v }, 2)) {
		t.Errorf("LessValue mismatch")
	}
	if !slices.Equal(m.LessEqualValue(2), cmpValue(func(x, v float64) bool { return x <= v }, 2)) {
		t.Errorf("LessEqualValue mismatch")
	}
	if !slices.Equal(m.EqualValue(2), cmpValue(func(x, v float64) bool { return x == v }, 2)) {
		t.Errorf("EqualValue mismatch")
	}
	if !slices.Equal(m.NotEqualValue(2), cmpValue(func(x, v float64) bool { return x != v }, 2)) {
		t.Errorf("NotEqualValue mismatch")
	}
}

func TestCompareNaN(t *testing.T) {
	na := math.NaN()
	a := FromSlice([]float64{1, na, 3})
	b := FromSlice([]float64{na, 2, 3})
	for name, got := range map[string][]bool{
		"Greater":    a.Greater(b),
		"Less":       a.Less(b),
		"Equal":      a.Equal(b),
		"GreaterEqu": a.GreaterEqual(b),
		"LessEqual":  a.LessEqual(b),
	} {
		if got[0] || got[1] {
			t.Errorf("%s: NaN comparison = %v, want false", name, got[:2])
		}
	}
	if !a.Equal(b)[2] {
		t.Errorf("3==3 should hold")
	}
	// NaN!=NaN is true under IEEE.
	if !a.NotEqual(b)[1] {
		t.Errorf("NaN != NaN should be true")
	}
	// NonZero counts NaN as nonzero (numpy x != 0), along with 1 and 3.
	if idx := a.NonZero(); !slices.Equal(idx, []int{0, 1, 2}) {
		t.Errorf("NonZero with NaN = %v, want [0 1 2]", idx)
	}
}

func TestPredicates(t *testing.T) {
	a := FromSlice([]float64{1, math.NaN(), math.Inf(1), math.Inf(-1), 0})
	if !slices.Equal(a.IsNaN(), []bool{false, true, false, false, false}) {
		t.Errorf("IsNaN = %v", a.IsNaN())
	}
	if !slices.Equal(a.IsInf(), []bool{false, false, true, true, false}) {
		t.Errorf("IsInf = %v", a.IsInf())
	}
	if !slices.Equal(a.IsFinite(), []bool{true, false, false, false, true}) {
		t.Errorf("IsFinite = %v", a.IsFinite())
	}
	if !slices.Equal(a.IsPosInf(), []bool{false, false, true, false, false}) {
		t.Errorf("IsPosInf = %v", a.IsPosInf())
	}
	if !slices.Equal(a.IsNegInf(), []bool{false, false, false, true, false}) {
		t.Errorf("IsNegInf = %v", a.IsNegInf())
	}
	// Complex: either component hits.
	c := FromSlice([]complex128{1 + 2i, complex(math.NaN(), 0), complex(0, math.Inf(1)), complex(math.Inf(-1), 3)})
	if !slices.Equal(c.IsNaN(), []bool{false, true, false, false}) {
		t.Errorf("complex IsNaN = %v", c.IsNaN())
	}
	if !slices.Equal(c.IsInf(), []bool{false, false, true, true}) {
		t.Errorf("complex IsInf = %v", c.IsInf())
	}
	if !slices.Equal(c.IsFinite(), []bool{true, false, false, false}) {
		t.Errorf("complex IsFinite = %v", c.IsFinite())
	}
	if !slices.Equal(c.IsPosInf(), []bool{false, false, true, false}) {
		t.Errorf("complex IsPosInf = %v", c.IsPosInf())
	}
	if !slices.Equal(c.IsNegInf(), []bool{false, false, false, true}) {
		t.Errorf("complex IsNegInf = %v", c.IsNegInf())
	}
	// float32 sanity.
	f32 := FromSlice([]float32{1, float32(math.NaN())})
	if !slices.Equal(f32.IsNaN(), []bool{false, true}) {
		t.Errorf("float32 IsNaN = %v", f32.IsNaN())
	}
	// Complex comparison uses the real part.
	cl := FromSlice([]complex128{1 + 100i, 0 - 5i})
	cr := FromSlice([]complex128{2, -1})
	if !slices.Equal(cl.Less(cr), []bool{true, false}) {
		t.Errorf("complex real-part Less = %v", cl.Less(cr))
	}
	if !slices.Equal(cl.GreaterValue(0+999i), []bool{true, false}) {
		t.Errorf("complex real-part GreaterValue = %v", cl.GreaterValue(0+999i))
	}
}

func complement(m []bool) []bool {
	out := make([]bool, len(m))
	for i, v := range m {
		out[i] = !v
	}
	return out
}

func or(a, b []bool) []bool {
	out := make([]bool, len(a))
	for i := range a {
		out[i] = a[i] || b[i]
	}
	return out
}
