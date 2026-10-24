package nditer

import "testing"

// Benchmarks for the contiguous fast paths. All shapes are 1000x100 float64.

const benchRows, benchCols = 1000, 100

// BenchmarkZipWhole measures the pure engine fast path: dst contiguous, one fn
// call covering all elements (run inside fn, not per element).
func BenchmarkZipWhole(b *testing.B) {
	const n = benchRows * benchCols
	dst := &Layout{Shape: []int{n}, Strides: []int{1}}
	dstData := make([]float64, n)
	a := &Layout{Shape: []int{n}, Strides: []int{1}}
	aData := make([]float64, n)
	bData := make([]float64, n)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Zip(dst, dstData, []*Layout{a, dst}, [][]float64{aData, bData},
			func(d []float64, src [][]float64) {
				for i := range d {
					d[i] = src[0][i] + src[1][i]
				}
			})
	}
}

// BenchmarkZipFast is the 2-D contiguous case (also hits the fast path).
func BenchmarkZipFast(b *testing.B) {
	n, m := benchRows, benchCols
	dst := &Layout{Shape: []int{n, m}, Strides: []int{m, 1}}
	dstData := make([]float64, n*m)
	a := &Layout{Shape: []int{n, m}, Strides: []int{m, 1}}
	aData := make([]float64, n*m)
	bData := make([]float64, n*m)
	for i := range aData {
		aData[i] = float64(i)
		bData[i] = 1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Zip(dst, dstData, []*Layout{a, dst}, [][]float64{aData, bData},
			func(d []float64, src [][]float64) {
				for i := range d {
					d[i] = src[0][i] + src[1][i]
				}
			})
	}
}

// BenchmarkZipStrided forces the per-element path: aStrided is a transposed
// (column-major) view of the same 1000x100 data.
func BenchmarkZipStrided(b *testing.B) {
	n, m := benchRows, benchCols
	dst := &Layout{Shape: []int{n, m}, Strides: []int{m, 1}}
	dstData := make([]float64, n*m)
	aStrided := &Layout{Shape: []int{n, m}, Strides: []int{1, n}}
	aData := make([]float64, n*m)
	bData := make([]float64, n*m)
	for i := range aData {
		aData[i] = float64(i)
		bData[i] = 1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Zip(dst, dstData, []*Layout{aStrided, dst}, [][]float64{aData, bData},
			func(d []float64, src [][]float64) {
				for i := range d {
					d[i] = src[0][i] + src[1][i]
				}
			})
	}
}

// BenchmarkReduceLastAxis exercises the last-axis fast path.
func BenchmarkReduceLastAxis(b *testing.B) {
	n, m := benchRows, benchCols
	l := &Layout{Shape: []int{n, m}, Strides: []int{m, 1}}
	data := make([]float64, n*m)
	for i := range data {
		data[i] = float64(i)
	}
	out := make([]float64, n)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for i := range out {
			out[i] = 0
		}
		Reduce(l, data, []int{-1}, out, func(acc *float64, v float64) { *acc += v })
	}
}

// BenchmarkAccumulateLastAxis exercises the last-axis fast path.
func BenchmarkAccumulateLastAxis(b *testing.B) {
	n, m := benchRows, benchCols
	l := &Layout{Shape: []int{n, m}, Strides: []int{m, 1}}
	data := make([]float64, n*m)
	for i := range data {
		data[i] = float64(i)
	}
	out := make([]float64, n*m)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Accumulate(l, data, []int{-1}, out, func(acc *float64, v float64) { *acc += v })
	}
}
