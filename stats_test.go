package numgo

import (
	"math"
	"testing"
)

func TestPercentileQuantile(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	wantNear(t, a.Percentile(25, 0), []int{3}, []float64{1.75, 2.75, 3.75})
	wantNear(t, a.Percentile(25, 1), []int{2}, []float64{1.5, 4.5})
	wantNear(t, a.Percentile(50, 0), []int{3}, []float64{2.5, 3.5, 4.5})
	wantNear(t, a.Percentile(90, 0), []int{3}, []float64{3.7, 4.7, 5.7})
	wantNear(t, a.Percentile(90, 1), []int{2}, []float64{2.8, 5.8})
	wantNear(t, a.Percentile(25), []int{1}, []float64{2.25})
	wantNear(t, a.Median(0), []int{3}, []float64{2.5, 3.5, 4.5})
	wantNear(t, a.Median(1), []int{2}, []float64{2, 5})
	// numpy quantile scale: q in [0,1].
	wantNear(t, a.Quantile(0.25), []int{1}, []float64{2.25})
	wantNear(t, a.Quantile(0.5, 1), []int{2}, []float64{2, 5})
	// odd-length channel and linear interpolation of [1 2 3 4].
	odd := New1DArray([]float64{5, 1, 3})
	wantNear(t, odd.Median(), []int{1}, []float64{3})
	even := New1DArray([]float64{1, 2, 3, 4})
	wantNear(t, even.Median(), []int{1}, []float64{2.5})
	wantNear(t, even.Percentile(25), []int{1}, []float64{1.75})
	// NaN poisons the channel for all q (numpy masks NaN slices).
	poison := New1DArray([]float64{1, 2, math.NaN(), 3})
	for _, q := range []float64{0, 50, 100} {
		if !math.IsNaN(poison.Percentile(q).At(0)) {
			t.Fatalf("Percentile(%v) with NaN should be NaN", q)
		}
	}
	// bounds and out-of-range q.
	assertPanic(t, func() { a.Percentile(101) })
	assertPanic(t, func() { a.Percentile(-1) })
	assertPanic(t, func() { a.Quantile(1.5) })
	assertPanic(t, func() { a.Percentile(math.NaN()) })
	// 3x4 axis-0 golden (interpolating between rows).
	v := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4)
	wantNear(t, v.Percentile(25, 0), []int{4}, []float64{3, 4, 5, 6})
}

func TestCovCorrCoef(t *testing.T) {
	x := FromSlice([]float64{1, 2, 3, 4, 2, 4, 6, 8, 3, 1, 0, 2}, 3, 4)
	wantNear(t, Cov(x, nil), []int{3, 3}, []float64{
		1.6666666666666665, 3.333333333333333, -0.6666666666666666,
		3.333333333333333, 6.666666666666666, -1.3333333333333333,
		-0.6666666666666666, -1.3333333333333333, 1.6666666666666665})
	wantNear(t, Cov(New1DArray([]float64{1, 2, 3}), New1DArray([]float64{2, 4, 6})),
		[]int{2, 2}, []float64{1, 2, 2, 4})
	wantNear(t, CorrCoef(x, nil), []int{3, 3}, []float64{
		1, 1, -0.39999999999999997,
		1, 1, -0.39999999999999997,
		-0.39999999999999997, -0.39999999999999997, 1})
	// perfectly correlated 1-D pair.
	wantNear(t, CorrCoef(New1DArray([]float64{1, 2, 3}), New1DArray([]float64{2, 4, 6})),
		[]int{2, 2}, []float64{1, 1, 1, 1})
	// mismatched observation counts panic.
	assertPanic(t, func() {
		Cov(New1DArray([]float64{1, 2, 3}), New1DArray([]float64{1, 2}))
	})
}

func TestHistogram(t *testing.T) {
	v := FromSlice([]float64{
		0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
		0.2, 1.8, 2.9, 3.1, 4.4, 5.6, 6.9, 7.7, 8.1, 9.9})
	counts, edges := v.Histogram(5)
	wantNear(t, counts, []int{5}, []float64{4, 4, 4, 4, 4})
	wantNear(t, edges, []int{6}, []float64{0.2, 2.14, 4.08, 6.02, 7.96, 9.9})
	// last edge inclusive.
	counts2, edges2 := New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).Histogram(5)
	wantNear(t, counts2, []int{5}, []float64{2, 2, 2, 2, 2})
	wantNear(t, edges2, []int{6}, []float64{0, 1.8, 3.6, 5.4, 7.2, 9})
	// empty input: range [0, 1].
	counts3, edges3 := Zeros[float64](0).Histogram(4)
	wantNear(t, counts3, []int{4}, []float64{0, 0, 0, 0})
	wantNear(t, edges3, []int{5}, []float64{0, 0.25, 0.5, 0.75, 1})
	// degenerate range widens by 0.5; all values fall in the first bin.
	counts4, edges4 := New1DArray([]float64{3, 3, 3}).Histogram(2)
	wantNear(t, counts4, []int{2}, []float64{0, 3})
	wantNear(t, edges4, []int{3}, []float64{2.5, 3, 3.5})
	// bin edges count as left-edge members (searchsorted right).
	counts5, _ := New1DArray([]float64{0, 1, 2}).Histogram(2)
	wantNear(t, counts5, []int{2}, []float64{1, 2})
	assertPanic(t, func() { v.Histogram(0) })
}

func TestHistogram2D(t *testing.T) {
	xs := New1DArray([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	ys := New1DArray([]float64{9, 8, 7, 6, 5, 4, 3, 2, 1})
	h, xe, ye := Histogram2D(xs, ys, 2)
	wantNear(t, h, []int{2, 2}, []float64{0, 4, 4, 1})
	wantNear(t, xe, []int{3}, []float64{1, 5, 9})
	wantNear(t, ye, []int{3}, []float64{1, 5, 9})
	// diagonal grid 3x3.
	gx := New1DArray([]float64{0, 1, 2, 0, 1, 2, 0, 1, 2})
	gy := New1DArray([]float64{0, 0, 0, 1, 1, 1, 2, 2, 2})
	h2, xe2, ye2 := Histogram2D(gx, gy, 3)
	wantNear(t, h2, []int{3, 3}, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1})
	wantNear(t, xe2, []int{4}, []float64{0, 0.6666666666666666, 1.3333333333333333, 2})
	wantNear(t, ye2, []int{4}, []float64{0, 0.6666666666666666, 1.3333333333333333, 2})
	// unequal lengths panic.
	assertPanic(t, func() { Histogram2D(gx, New1DArray([]float64{0, 1}), 2) })
}

func TestHistogramDD(t *testing.T) {
	dd := New2DArray([][]float64{{1, 10, 2}, {20, 3, 4}, {5, 6, 7}, {8, 9, 1}})
	h, edges := dd.HistogramDD([]int{2, 2, 2})
	wantNear(t, h, []int{2, 2, 2}, []float64{0, 1, 2, 0, 0, 1, 0, 0})
	wantNear(t, edges[0], []int{3}, []float64{1, 10.5, 20})
	wantNear(t, edges[1], []int{3}, []float64{3, 6.5, 10})
	wantNear(t, edges[2], []int{3}, []float64{1, 4, 7})
	// explicit ranges replace the autodetected ones.
	h2, edges2 := dd.HistogramDD([]int{2, 2, 2}, [2]float64{0, 20}, [2]float64{0, 10}, [2]float64{0, 7})
	wantNear(t, h2, []int{2, 2, 2}, []float64{0, 0, 2, 1, 0, 1, 0, 0})
	wantNear(t, edges2[0], []int{3}, []float64{0, 10, 20})
	wantNear(t, edges2[1], []int{3}, []float64{0, 5, 10})
	wantNear(t, edges2[2], []int{3}, []float64{0, 3.5, 7})
	// out-of-range points are dropped (bins over a tiny 2-D window).
	h3, _ := dd.HistogramDD([]int{2, 2, 2}, [2]float64{1, 2}, [2]float64{1, 2}, [2]float64{1, 2})
	wantNear(t, h3, []int{2, 2, 2}, []float64{0, 0, 0, 0, 0, 0, 0, 0})
	assertPanic(t, func() { dd.HistogramDD([]int{2}) })
	assertPanic(t, func() { dd.HistogramDD([]int{0, 3}) })
}

func TestGradient(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	g := a.Gradient()
	if len(g) != 2 {
		t.Fatalf("len(g) = %d", len(g))
	}
	wantNear(t, g[0], []int{2, 3}, []float64{3, 3, 3, 3, 3, 3})
	wantNear(t, g[1], []int{2, 3}, []float64{1, 1, 1, 1, 1, 1})
	// ragged golden from numpy: interior central, edges one-sided.
	b := New2DArray([][]float64{{1, 2, 4, 7}, {0, 5, 6, 1}})
	gb := b.Gradient()
	wantNear(t, gb[0], []int{2, 4}, []float64{-1, 3, 2, -6, -1, 3, 2, -6})
	wantNear(t, gb[1], []int{2, 4}, []float64{1, 1.5, 2.5, 3, 5, 3, -2, -5})
	wantNear(t, New1DArray([]float64{1, 2, 4, 7}).Gradient()[0], []int{4},
		[]float64{1, 1.5, 2.5, 3})
	// axis with a single element panics (numpy ValueError).
	assertPanic(t, func() { Zeros[float64](2, 1).Gradient() })
}

func TestInterp(t *testing.T) {
	xp := New1DArray([]float64{0, 1, 2, 3})
	fp := New1DArray([]float64{0, 10, 20, 30})
	x := New1DArray([]float64{-1, 0, 0.5, 1, 1.5, 2.5, 3, 4})
	wantNear(t, Interp(x, xp, fp), []int{8},
		[]float64{0, 0, 5, 10, 15, 25, 30, 30})
	// clamping keeps the query's shape.
	xt := FromSlice([]float64{-5, 5, 2}, 1, 3)
	wantNear(t, Interp(xt, xp, fp), []int{1, 3}, []float64{0, 30, 20})
	// duplicate abscissae resolve to the last (numpy searchsorted behavior).
	wantNear(t, Interp(New1DArray([]float64{1}),
		New1DArray([]float64{0, 1, 1, 2}), New1DArray([]float64{0, 10, 20, 30})),
		[]int{1}, []float64{20})
	// a single sample point returns it everywhere.
	wantNear(t, Interp(New1DArray([]float64{0, 1, 2}),
		New1DArray([]float64{1}), New1DArray([]float64{5})), []int{3},
		[]float64{5, 5, 5})
	// NaN queries yield NaN.
	if !math.IsNaN(Interp(New1DArray([]float64{math.NaN()}), xp, fp).At(0)) {
		t.Fatal("interp of NaN should be NaN")
	}
	assertPanic(t, func() {
		Interp(x, New1DArray([]float64{0, 1}), New1DArray([]float64{0, 1, 2}))
	})
	assertPanic(t, func() { Interp(x, Zeros[float64](0), Zeros[float64](0)) })
}
