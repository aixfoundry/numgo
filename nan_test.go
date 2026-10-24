package numgo

import (
	"math"
	"slices"
	"testing"
)

// wantNear checks a float64 array's shape and values against numpy goldens
// with the fixture tolerance (NaN == NaN).
func wantNear(t *testing.T, got *Array[float64], shape []int, want []float64) {
	t.Helper()
	if !slices.Equal(got.Shape(), shape) {
		t.Fatalf("shape = %v, want %v", got.Shape(), shape)
	}
	data := got.Data()
	if len(data) != len(want) {
		t.Fatalf("len = %d, want %d", len(data), len(want))
	}
	for i, w := range want {
		if !closeEnough(data[i], w) {
			t.Fatalf("data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

// nan2x3 is [[1 2 NaN] [4 NaN 6]].
func nan2x3() *Array[float64] {
	return New2DArray([][]float64{{1, 2, math.NaN()}, {4, math.NaN(), 6}})
}

// allnan3x3 has a NaN-poisoned column 1 and an all-NaN row 2.
func allnan3x3() *Array[float64] {
	return New2DArray([][]float64{{1, math.NaN(), 3}, {4, 5, math.NaN()}, {math.NaN(), math.NaN(), math.NaN()}})
}

func TestNanSumProd(t *testing.T) {
	b := nan2x3()
	wantNear(t, b.NanSum(0), []int{3}, []float64{5, 2, 6})
	wantNear(t, b.NanSum(1), []int{2}, []float64{3, 10})
	wantNear(t, b.NanSum(-1), []int{2}, []float64{3, 10})
	wantNear(t, b.NanSum(), []int{1}, []float64{13})
	wantNear(t, b.NanProd(0), []int{3}, []float64{4, 2, 6})
	wantNear(t, b.NanProd(1), []int{2}, []float64{2, 24})
	if !slices.Equal(b.NanSumKeepdims(1).Shape(), []int{2, 1}) {
		t.Fatalf("keepdims shape = %v", b.NanSumKeepdims(1).Shape())
	}
	// all-NaN folds to the identity (numpy nansum/nanprod).
	allnan := New1DArray([]float64{math.NaN(), math.NaN()})
	wantNear(t, allnan.NanSum(), []int{1}, []float64{0})
	wantNear(t, allnan.NanProd(), []int{1}, []float64{1})
	// cumulative: NaN contributes 0/1 (numpy nancumsum/nancumprod goldens).
	c := allnan3x3()
	wantNear(t, c.NanCumSum(1), []int{3, 3}, []float64{1, 1, 4, 4, 9, 9, 0, 0, 0})
	wantNear(t, c.NanCumSum(0), []int{3, 3}, []float64{1, 0, 3, 5, 5, 3, 5, 5, 3})
	wantNear(t, c.NanCumProd(1), []int{3, 3}, []float64{1, 1, 3, 4, 20, 20, 1, 1, 1})
	// flat default over the raveled array.
	wantNear(t, c.NanCumSum(), []int{9}, []float64{1, 1, 4, 8, 13, 13, 13, 13, 13})
}

func TestNanMaxMin(t *testing.T) {
	b := nan2x3()
	wantNear(t, b.NanMax(0), []int{3}, []float64{4, 2, 6})
	wantNear(t, b.NanMax(1), []int{2}, []float64{2, 6})
	wantNear(t, b.NanMin(0), []int{3}, []float64{1, 2, 6})
	wantNear(t, b.NanMin(1), []int{2}, []float64{1, 4})
	if !math.IsNaN(Zeros[float64](0).NanMax().At(0)) {
		t.Fatal("NanMax of empty should be NaN")
	}
	if !math.IsNaN(Zeros[float64](0).NanMin().At(0)) {
		t.Fatal("NanMin of empty should be NaN")
	}
	allnan := New1DArray([]float64{math.NaN(), math.NaN()})
	if !math.IsNaN(allnan.NanMax().At(0)) {
		t.Fatal("NanMax of all-NaN should be NaN")
	}
	if !slices.Equal(b.NanMaxKeepdims(0).Shape(), []int{1, 3}) {
		t.Fatalf("keepdims shape = %v", b.NanMaxKeepdims(0).Shape())
	}
}

func TestNanMeanVarStd(t *testing.T) {
	b := nan2x3()
	wantNear(t, b.NanMean(0), []int{3}, []float64{2.5, 2, 6})
	wantNear(t, b.NanMean(1), []int{2}, []float64{1.5, 5})
	wantNear(t, b.NanVar(0), []int{3}, []float64{2.25, 0, 0})
	wantNear(t, b.NanVar(1), []int{2}, []float64{0.25, 1})
	wantNear(t, b.NanStd(0), []int{3}, []float64{1.5, 0, 0})
	wantNear(t, b.NanStd(1), []int{2}, []float64{0.5, 1})
	// ddof=1 sample variance of [1 2 3] is 1.
	wantNear(t, New1DArray([]float64{1, 2, 3}).NanVarDDof(1), []int{1}, []float64{1})
	wantNear(t, New1DArray([]float64{1, 2, 3}).NanStdDDof(1), []int{1}, []float64{1})
	// all-NaN slices yield NaN.
	mixed := New2DArray([][]float64{{1, math.NaN()}, {math.NaN(), math.NaN()}})
	wantNear(t, mixed.NanMean(1), []int{2}, []float64{1, math.NaN()})
	if !math.IsNaN(New1DArray([]float64{math.NaN()}).NanVar().At(0)) {
		t.Fatal("NanVar of all-NaN should be NaN")
	}
	if !slices.Equal(b.NanVarKeepdims(0).Shape(), []int{1, 3}) {
		t.Fatalf("keepdims shape = %v", b.NanVarKeepdims(0).Shape())
	}
	if !slices.Equal(b.NanStdKeepdims(1).Shape(), []int{2, 1}) {
		t.Fatalf("keepdims shape = %v", b.NanStdKeepdims(1).Shape())
	}
}

func TestNanMedianArgCum(t *testing.T) {
	c := allnan3x3()
	wantNear(t, c.NanMedian(1), []int{3}, []float64{2, 4.5, math.NaN()})
	wantNear(t, c.NanMedian(0), []int{3}, []float64{2.5, 5, 3})
	if got := c.NanArgMax(0); !slices.Equal(got, []int{1, 1, 0}) {
		t.Fatalf("NanArgMax = %v", got)
	}
	if got := c.NanArgMin(0); !slices.Equal(got, []int{0, 1, 0}) {
		t.Fatalf("NanArgMin = %v", got)
	}
	if !slices.Equal(c.NanMedianKeepdims(1).Shape(), []int{3, 1}) {
		t.Fatalf("keepdims shape = %v", c.NanMedianKeepdims(1).Shape())
	}
	// All-NaN slice: numpy nanargmax raises ValueError; we panic.
	assertPanic(t, func() { New1DArray([]float64{math.NaN(), math.NaN()}).NanArgMax(0) })
	assertPanic(t, func() { New1DArray([]float64{math.NaN()}).NanArgMin(0) })
	// odd channel, even channel.
	odd := New2DArray([][]float64{{5, math.NaN(), 3}, {1, 2, math.NaN()}})
	wantNear(t, odd.NanMedian(1), []int{2}, []float64{4, 1.5})
}

func TestNanFloat32(t *testing.T) {
	b := New2DArray([][]float32{{1, 2, float32(math.NaN())}, {4, float32(math.NaN()), 6}})
	mean0 := AsType[float32, float64](b.NanMean(0))
	wantNear(t, mean0, []int{3}, []float64{2.5, 2, 6})
	// NaN-free even channel averages its two middle values; odd takes the
	// middle one.
	odd := New2DArray([][]float32{{5, float32(math.NaN()), 3}, {1, 2, 4}})
	wantNear(t, AsType[float32, float64](odd.NanMedian(1)), []int{2}, []float64{4, 2})
}
