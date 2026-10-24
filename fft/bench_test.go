package fft

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Performance baseline benchmarks (Phase 5, before-optimization); sizes match
// scripts/bench_numpy.py. n == 0 selects the axis length (numpy's None).

func realSignal(n int) *numgo.Array[float64] {
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i%97) / 97
	}
	return numgo.FromSlice(data, n)
}

func BenchmarkFFT65536(b *testing.B) {
	a := realSignal(1 << 16)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		FFT(a, 0, Backward)
	}
}

func BenchmarkFFT40960(b *testing.B) {
	a := realSignal(40960) // 2^13 · 5, mixed radix
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		FFT(a, 0, Backward)
	}
}

func BenchmarkFFTN64x64(b *testing.B) {
	a := realSignal(64 * 64)
	_ = a.Reshape(64, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		FFTN(a.Reshape(64, 64), nil, Backward)
	}
}

func BenchmarkIrfft65536(b *testing.B) {
	a := FFT(realSignal(1<<16), 0, Backward)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		IRFFT(a, 0, Backward)
	}
}
