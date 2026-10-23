package numgo

import (
	"slices"
	"testing"
)

func newTest2x3() *Array[float64] {
	return New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
}

func assertArray(t *testing.T, got *Array[float64], want []float64) {
	t.Helper()
	if !slices.Equal(got.Data(), want) {
		t.Fatalf("data = %v, want %v", got.Data(), want)
	}
}

func TestSwapStandaloneNonSquare(t *testing.T) {
	a := newTest2x3()
	assertArray(t, Swap(a, 0, 1, 0), []float64{4, 5, 6, 1, 2, 3})
	assertArray(t, Swap(a, 0, 2, 1), []float64{3, 2, 1, 6, 5, 4})
	assertArray(t, a, []float64{1, 2, 3, 4, 5, 6}) // input untouched
}

func TestSwapMethodNonSquare(t *testing.T) {
	assertArray(t, newTest2x3().Swap(0, 1, 0), []float64{4, 5, 6, 1, 2, 3})
	assertArray(t, newTest2x3().Swap(0, 2, 1), []float64{3, 2, 1, 6, 5, 4})
}

func TestDeleteStandaloneNonSquare(t *testing.T) {
	r, c := 1, 3
	del := Delete(newTest2x3(), 0, 0)
	if gotR, gotC := del.Dims(); gotR != r || gotC != c {
		t.Fatalf("dims = %dx%d, want %dx%d", gotR, gotC, r, c)
	}
	assertArray(t, del, []float64{4, 5, 6})

	r, c = 2, 2
	del = Delete(newTest2x3(), 1, 1)
	if gotR, gotC := del.Dims(); gotR != r || gotC != c {
		t.Fatalf("dims = %dx%d, want %dx%d", gotR, gotC, r, c)
	}
	assertArray(t, del, []float64{1, 3, 4, 6})
}

func TestDeleteMethodNonSquare(t *testing.T) {
	a := newTest2x3().Delete(0, 0)
	if gotR, gotC := a.Dims(); gotR != 1 || gotC != 3 {
		t.Fatalf("dims = %dx%d, want 1x3", gotR, gotC)
	}
	assertArray(t, a, []float64{4, 5, 6})

	a = newTest2x3().Delete(1, 1)
	if gotR, gotC := a.Dims(); gotR != 2 || gotC != 2 {
		t.Fatalf("dims = %dx%d, want 2x2", gotR, gotC)
	}
	assertArray(t, a, []float64{1, 3, 4, 6})
}

func TestRangeDeleteStandaloneNonSquare(t *testing.T) {
	assertArray(t, RangeDelete(newTest2x3(), 0, 1, 0), []float64{4, 5, 6})
	assertArray(t, RangeDelete(newTest2x3(), 1, 2, 0), []float64{1, 2, 3})
	assertArray(t, RangeDelete(newTest2x3(), 0, 2, 1), []float64{3, 6})
	assertArray(t, RangeDelete(newTest2x3(), 1, 3, 1), []float64{1, 4})
}

func TestRangeDeleteMethodNonSquare(t *testing.T) {
	assertArray(t, newTest2x3().RangeDelete(0, 1, 0), []float64{4, 5, 6})
	assertArray(t, newTest2x3().RangeDelete(1, 2, 0), []float64{1, 2, 3})
	assertArray(t, newTest2x3().RangeDelete(0, 2, 1), []float64{3, 6})
	assertArray(t, newTest2x3().RangeDelete(1, 3, 1), []float64{1, 4})
}

func TestPackageDeleteDoesNotAliasInput(t *testing.T) {
	a := newTest2x3()
	before := slices.Clone(a.Data())

	del := Delete(a, 0, 0)
	assertArray(t, a, before)
	assertArray(t, del, []float64{4, 5, 6})
	del.Set(0, 0, 99.0)
	assertArray(t, a, before)
	assertArray(t, del, []float64{99, 5, 6})

	rd := RangeDelete(a, 0, 1, 0)
	assertArray(t, a, before)
	assertArray(t, rd, []float64{4, 5, 6})
	rd.Set(0, 0, 99.0)
	assertArray(t, a, before)
	assertArray(t, rd, []float64{99, 5, 6})
}

func TestMedianNoMutation(t *testing.T) {
	a := New2DArray([][]float64{{3, 1, 2}, {6, 4, 5}})
	med := a.Median(-1)
	if got := med.At(0, 0); got != 3.5 {
		t.Fatalf("median = %v, want 3.5", got)
	}
	assertArray(t, a, []float64{3, 1, 2, 6, 4, 5})
}

func TestEqualEmptyVsZero(t *testing.T) {
	// former EmptyArray: same dims, no data
	empty := &Array[float64]{r: 2, c: 2}
	if Equal(empty, ZeroArray[float64](2, 2)) {
		t.Fatal("Equal(empty, ZeroArray) = true, want false")
	}
}

func TestReshapeEmptyMinusOne(t *testing.T) {
	ZeroArray[float64](0, 0).Reshape(-1, 0)
	ZeroArray[float64](0, 0).Reshape(0, -1)
}

func TestNew2DArrayJaggedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for jagged input")
		}
	}()
	New2DArray([][]float64{{1, 2}, {3}})
}
