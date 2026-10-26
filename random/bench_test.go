package random

import (
	"testing"
)

// Performance baseline benchmarks (Phase 5, before-optimization); sizes match
// scripts/bench_numpy.py.

func BenchmarkStandardNormal1M(b *testing.B) {
	g := DefaultRng(42)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		g.StandardNormal(1_000_000)
	}
}

func BenchmarkRandom1M(b *testing.B) {
	g := DefaultRng(42)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		g.Random(1_000_000)
	}
}

func BenchmarkIntegers1M(b *testing.B) {
	g := DefaultRng(42)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		g.Integers(0, 1000, 1_000_000)
	}
}

// libm-sensitive distributions (math via Go stdlib since the glibc-kernel
// removal); numpy counterparts are in scripts/bench_numpy.py.
func BenchmarkStandardGamma1M(b *testing.B) {
	g := DefaultRng(42)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		g.StandardGamma(3.0, 1_000_000)
	}
}

func BenchmarkWeibull1M(b *testing.B) {
	g := DefaultRng(42)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		g.Weibull(1.5, 1_000_000)
	}
}
