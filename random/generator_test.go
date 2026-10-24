package random

import (
	"math"
	"testing"
)

// Reproducibility: identical seeds produce identical streams.
func TestReproducibility(t *testing.T) {
	a := DefaultRng(42)
	b := DefaultRng(42)
	for range 100 {
		if a.bg.Next64() != b.bg.Next64() {
			t.Fatal("streams diverged for identical seed")
		}
	}
	a1 := DefaultRng(7).StandardNormal(16)
	a2 := DefaultRng(7).StandardNormal(16)
	for i := range a1.Data() {
		if a1.Data()[i] != a2.Data()[i] {
			t.Fatal("standard_normal not reproducible")
		}
	}
	// distinct seeds diverge
	if DefaultRng(1).bg.Next64() == DefaultRng(2).bg.Next64() {
		t.Fatal("distinct seeds produced identical draws")
	}
}

// Spawned children produce independent streams with numpy-consistent seeds.
func TestSpawnIndependence(t *testing.T) {
	g := DefaultRng(99)
	kids := g.Spawn(2)
	if kids[0].SeedSequence() == g.SeedSequence() {
		t.Fatal("spawn returned parent seed sequence")
	}
	s0 := kids[0].bg.Next64()
	s1 := kids[1].bg.Next64()
	if s0 == s1 {
		t.Fatal("spawned generators share streams")
	}
	// numpy: default_rng(99).spawn(1)[0] draws
	ref := moreRef["spawn99_child0"]
	if ref != nil {
		k := DefaultRng(99).Spawn(1)[0]
		if k.bg.Next64() != ref[0] {
			t.Fatalf("spawn99_child0: %d != %d", k.bg.Next64(), ref[0])
		}
	}
}

// Scalar no-size convention: methods without size return a 1-element array.
func TestScalarConvention(t *testing.T) {
	g := DefaultRng(12345)
	r := g.Random()
	if r.Size() != 1 {
		t.Fatalf("Random() size = %d, want 1", r.Size())
	}
	if g.StandardNormal().Size() != 1 || g.StandardExponential().Size() != 1 {
		t.Fatal("scalar convention broken")
	}
}

// Error paths: numpy-matching panics.
func TestErrorPaths(t *testing.T) {
	g := DefaultRng(1)
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		fn()
	}
	mustPanic("integers low>=high", func() { g.Integers(5, 5, 1) })
	mustPanic("choice sample too large", func() { g.Choice(3, 5, false, nil) })
	mustPanic("probabilities NaN", func() {
		g.Choice(3, 1, true, []float64{0.5, math.NaN(), 0.5})
	})
	mustPanic("probabilities sum", func() {
		g.Choice(3, 1, true, []float64{0.5, 0.5, 0.5})
	})
	mustPanic("negative entropy", func() { NewSeedSequence(^uint64(0)) })
	// numpy accepts high up to 2**63-1 for int64; low>high panics
	mustPanic("integers low>high", func() { g.Integers(10, 2, 1) })
}
