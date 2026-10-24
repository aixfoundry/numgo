package cmath

import (
	"math"
	"testing"
)

func TestFloat32Kernels(t *testing.T) {
	bad := map[string]int{}
	for _, c := range f32Ref {
		x := math.Float32frombits(c.x)
		if got := math.Float32bits(Log1pf(x)); got != c.l1 {
			bad["log1pf"]++
			if bad["log1pf"] < 4 {
				t.Logf("log1pf(%v=%x): got %x want %x", x, c.x, got, c.l1)
			}
		}
		if x > 0 {
			if got := math.Float32bits(Logf(x)); got != c.lf {
				bad["logf"]++
				if bad["logf"] < 4 {
					t.Logf("logf(%v): got %x want %x", x, got, c.lf)
				}
			}
		}
		if got := math.Float32bits(Expf(x)); got != c.ef {
			bad["expf"]++
			if bad["expf"] < 4 {
				t.Logf("expf(%v): got %x want %x", x, got, c.ef)
			}
		}
		if x != 0 {
			y := 1 / x
			if got := math.Float32bits(Powf(x, y)); got != c.pf {
				bad["powf"]++
				if bad["powf"] < 4 {
					t.Logf("powf(%v,%v): got %x want %x", x, y, got, c.pf)
				}
			}
		}
	}
	if len(bad) > 0 {
		for k, v := range bad {
			t.Errorf("%s: %d mismatches", k, v)
		}
	}
}
