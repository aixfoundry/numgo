package numgo_test

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Performance baseline benchmarks (Phase 5, before-optimization).
// Sizes chosen to match scripts/bench_numpy.py so the two tables align.
func benchMat[T numgo.Number](n int) *numgo.Array[T] {
	return numgo.Ones[T](n, n)
}

func BenchmarkUfuncAddContiguous(b *testing.B) {
	a := benchMat[float64](1000)
	c := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Add(c)
	}
}

func BenchmarkUfuncAddBroadcastRow(b *testing.B) {
	a := benchMat[float64](1000)
	row := numgo.Ones[float64](1, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Add(row)
	}
}

func BenchmarkUfuncAddStrided(b *testing.B) {
	a := benchMat[float64](1000)
	t := a.Transpose() // strided view operand
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Add(t)
	}
}

func BenchmarkSumAll(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Sum()
	}
}

func BenchmarkSumAxis0(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Sum(0)
	}
}

func BenchmarkCumSumLast(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.CumSum(-1)
	}
}

func BenchmarkSortLastAxis(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Sort()
	}
}

func BenchmarkUnique(b *testing.B) {
	data := make([]float64, 100_000)
	for i := range data {
		data[i] = float64(i * 2654435761 % 65536)
	}
	v := numgo.FromSlice(data, 100_000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		v.Unique()
	}
}

func BenchmarkMatmul200(b *testing.B) {
	a := benchMat[float64](200)
	c := benchMat[float64](200)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Matmul(c)
	}
}

func BenchmarkMatmulBatch(b *testing.B) {
	data := make([]float64, 16*64*64)
	for i := range data {
		data[i] = float64(i%13) / 13
	}
	a := numgo.FromSlice(data, 16, 64, 64)
	c := numgo.FromSlice(data, 16, 64, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Matmul(c)
	}
}

func BenchmarkEinsum(b *testing.B) {
	a := benchMat[float64](100)
	c := benchMat[float64](100)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		numgo.Einsum("ij,jk->ik", a, c)
	}
}

func BenchmarkAtSet(b *testing.B) {
	const n = 1_000_000
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for i := 0; i < n; i++ {
			r, c := i/1000, i%1000
			a.Set([]int{r, c}, a.At(r, c)+1)
		}
	}
}

func BenchmarkSliceView(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Slice(numgo.To(500), numgo.R(0, 1000, 2))
	}
}

func BenchmarkReshape(b *testing.B) {
	a := benchMat[float64](1000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		a.Reshape(10_000, 100)
	}
}

func BenchmarkAsTypeFloat64Float32(b *testing.B) {
	a := numgo.Ones[float64](1_000_000)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		numgo.AsType[float64, float32](a)
	}
}
