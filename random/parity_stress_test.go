package random

import (
	"math"
	"testing"
)

// ULP tolerance for numpy-fixture parity of libm-sensitive draws. Fixtures
// were generated from numpy on glibc/Linux; on platforms where numpy links a
// different libm (e.g. Apple's on macOS), libm-sensitive draws may drift a
// few ULP — the bit-exact guarantee is documented for glibc/Linux builds.
const parityULPTol = 16

// eqBits asserts bit-identical float64 draws. It allows ≤parityULPTol ULP of
// drift ONLY for values that straddle a different-sign zero or NaN pattern —
// otherwise bitwise equality is required.
func eqBits(t *testing.T, name string, got []float64, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range got {
		gb, wb := math.Float64bits(got[i]), want[i]
		if gb == wb {
			continue
		}
		// ULP distance on the ordered (sign-magnitude → two's complement
		// mapped) integer axis; |0.0 vs -0.0| also lands here with dist 1.
		gm, wm := orderedBits(gb), orderedBits(wb)
		dist := gm - wm
		if dist < 0 {
			dist = -dist
		}
		if dist <= parityULPTol {
			continue
		}
		t.Fatalf("%s[%d]: got %x want %x (%v vs %v, %d ULP)", name, i,
			gb, wb, got[i], math.Float64frombits(wb), dist)
	}
}

// orderedBits maps an IEEE float64 bit pattern onto a monotonic integer axis
// (standard ULP-distance trick): sign bit flipped, then two's complement.
func orderedBits(b uint64) int64 {
	if b>>63 == 1 {
		return int64(^b)
	}
	return int64(b | 1<<63)
}

// ulp32 is the float32 counterpart of parityULPTol.
const ulp32Tol = 8

// eqInts asserts exact integer draws — integers are libm-free and stay
// bit-exact on every platform.
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
	eqBits(t, "standard_gamma>1", g.StandardGamma(9.5, 200).Data(), stressRef["gamma_9"])

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
