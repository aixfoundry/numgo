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
