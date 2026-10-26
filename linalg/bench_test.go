package linalg

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Performance baseline benchmarks (before optimization); sizes match
// scripts/bench_numpy.py. Values are a fixed pseudo-random matrix — the SVD /
// eig paths need non-degenerate data to converge.
func benchPD(n int) *numgo.Array[float64] {
	a := numgo.Ones[float64](n, n)
	s := uint64(12345)
	for i := range n {
		for j := range n {
			s = s*6364136223846793005 + 1442695040888963407
			a.Set([]int{i, j}, float64(int64(s>>33)%1000)/1000)
		}
	}
	return a
}

func BenchmarkSolve100(b *testing.B) {
	a := benchPD(100)
	x := numgo.Ones[float64](100)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Solve(a, x)
	}
}

func BenchmarkInv100(b *testing.B) {
	a := benchPD(100)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Inv(a)
	}
}

func BenchmarkSvd100(b *testing.B) {
	a := benchPD(100)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Svd(a, true, true)
	}
}

func BenchmarkEig50(b *testing.B) {
	a := benchPD(50)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Eig(a, true)
	}
}

func BenchmarkCholesky100(b *testing.B) {
	a := benchPD(100)
	// Cholesky requires positive-definite input: use A·Aᵀ + n·I.
	n := 100
	aat := a.Matmul(a.Transpose())
	for i := range n {
		aat.Set([]int{i, i}, aat.At(i, i)+float64(n))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		Cholesky(aat)
	}
}
