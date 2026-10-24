package random

import (
	"math"
	"testing"
)

// Stress parity against numpy 2.5.1 default_rng(12345): every element must be
// bit-identical (uint64 encoding of the IEEE bits, or raw integers).

func eqBits(t *testing.T, name string, got []float64, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float64bits(got[i]) != want[i] {
			t.Fatalf("%s[%d]: got %x want %x (%v vs %v)", name, i,
				math.Float64bits(got[i]), want[i], got[i], math.Float64frombits(want[i]))
		}
	}
}

func eqInts(t *testing.T, name string, got []int64, want []uint64) {
	t.Helper()
	for i := range got {
		if uint64(got[i]) != want[i] {
			t.Fatalf("%s[%d]: got %d want %d", name, i, got[i], int64(want[i]))
		}
	}
}

func TestStressParity(t *testing.T) {
	g := DefaultRng(12345)
	eqBits(t, "random", g.Random(200).Data(), stressRef["rnd200"])

	g = DefaultRng(12345)
	eqBits(t, "standard_normal", g.StandardNormal(200).Data(), stressRef["sn200_twos"])

	g = DefaultRng(12345)
	eqBits(t, "standard_exponential", g.StandardExponential(200).Data(), stressRef["se200"])

	g = DefaultRng(12345)
	eqBits(t, "standard_gamma<1", g.StandardGamma(0.1, 200).Data(), stressRef["gamma_pt1"])

	g = DefaultRng(12345)
	eqBits(t, "standard_gamma>1", g.StandardGamma(9.5, 200).Data(), stressRef["gamma_9"])

	g = DefaultRng(12345)
	eqBits(t, "beta johnk", g.Beta(0.3, 0.7, 200).Data(), stressRef["beta_j"])

	g = DefaultRng(12345)
	eqBits(t, "beta gamma", g.Beta(2.0, 3.5, 200).Data(), stressRef["beta_g"])

	g = DefaultRng(12345)
	eqInts(t, "poisson mult", g.Poisson(0.5, 200).Data(), stressRef["pois_lo"])

	g = DefaultRng(12345)
	eqInts(t, "poisson ptrs", g.Poisson(30.0, 200).Data(), stressRef["pois_hi"])

	g = DefaultRng(12345)
	eqInts(t, "binomial inv", g.Binomial(8, 0.3, 200).Data(), stressRef["binom_inv"])

	g = DefaultRng(12345)
	eqInts(t, "binomial btpe", g.Binomial(120, 0.4, 200).Data(), stressRef["binom_btpe"])

	g = DefaultRng(12345)
	eqInts(t, "binomial flipped", g.Binomial(120, 0.7, 200).Data(), stressRef["binom_flip"])

	g = DefaultRng(12345)
	eqInts(t, "integers 40-bit", g.Integers(0, 1<<40, 100).Data(), stressRef["int_b64"])

	g = DefaultRng(12345)
	eqInts(t, "integers 31-bit", g.Integers(0, 1<<31, 100).Data(), stressRef["int_u32"])

	g = DefaultRng(12345)
	eqInts(t, "integers full range", g.Integers(math.MinInt64, math.MaxInt64, 100).Data(), stressRef["int_full_twos"])

	g = DefaultRng(12345)
	eqInts(t, "integers endpoint", g.IntegersEndpoint(1, 7, 100).Data(), stressRef["int_ep"])

	g = DefaultRng(12345)
	eqBits(t, "uniform", g.Uniform(-10, 10, 150).Data(), stressRef["uni_twos"])

	g = DefaultRng(12345)
	eqBits(t, "normal", g.Normal(-2.5, 0.25, 150).Data(), stressRef["nrm_twos"])

	g = DefaultRng(12345)
	eqBits(t, "exponential", g.Exponential(3.3, 150).Data(), stressRef["exp_twos"])
}
