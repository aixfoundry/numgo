package ma

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Performance baseline benchmark (Phase 5, before-optimization); size matches
// scripts/bench_numpy.py. The mask hides every other row (half the elements).
func BenchmarkMaAdd(b *testing.B) {
	a := numgo.Ones[float64](1000, 1000)
	c := numgo.Ones[float64](1000, 1000)
	mask := make([]bool, 1000*1000)
	for i := range mask {
		mask[i] = (i/1000)%2 == 0
	}
	ma := FromData(a, mask)
	mc := FromData(c, mask)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ma.Add(mc)
	}
}
