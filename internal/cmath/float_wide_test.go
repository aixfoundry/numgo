package cmath

import (
	"math"
	"testing"
)

func TestFloat32KernelsWide(t *testing.T) {
	bad := map[string]int{}
	for _, c := range f32Ref2 {
		x := math.Float32frombits(c.x)
		y := math.Float32frombits(c.y)
		if got := math.Float32bits(Log1pf(x)); got != c.l1 {
			bad["log1pf"]++
		}
		if x > 0 {
			if got := math.Float32bits(Logf(x)); got != c.lf {
				bad["logf"]++
			}
		}
		if got := math.Float32bits(Expf(x)); got != c.ef {
			bad["expf"]++
		}
		if got := math.Float32bits(Powf(x, y)); got != c.pf {
			if bad["powf"] < 5 {
				t.Logf("powf(%v,%v): got %x want %x", x, y, got, c.pf)
			}
			bad["powf"]++
		}
	}
	if len(bad) > 0 {
		for k, v := range bad {
			t.Errorf("%s: %d mismatches", k, v)
		}
	}
}
