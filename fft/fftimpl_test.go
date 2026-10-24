package fft

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// refDFT is the naive reference DFT (forward or inverse, unnormalized).
func refDFT(x []complex128, inverse bool) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	s := -2 * math.Pi / float64(n)
	if inverse {
		s = -s
	}
	for k := range n {
		for j, v := range x {
			out[k] += v * cmplx.Exp(complex(0, s*float64(j*k)))
		}
	}
	return out
}

func maxDiff(a, b []complex128) float64 {
	var m float64
	for i := range a {
		if d := cmplx.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func randC(n int, rng *rand.Rand) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(rng.NormFloat64(), rng.NormFloat64())
	}
	return x
}

func TestFFTCKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ns := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 64}
	for _, n := range ns {
		x := randC(n, rng)
		want := refDFT(x, false)
		got := make([]complex128, n)
		fftC(got, x, false)
		if d := maxDiff(got, want); d > 1e-10 {
			t.Errorf("fftC forward n=%d: max diff %g", n, d)
		}
		// Round-trip: the inverse is unnormalized, so inv == n·x.
		inv := make([]complex128, n)
		fftC(inv, got, true)
		for i := range inv {
			inv[i] /= complex(float64(n), 0)
		}
		if d := maxDiff(inv, x); d > 1e-10 {
			t.Errorf("fftC round-trip n=%d: max diff %g", n, d)
		}
	}
}

func TestBluesteinPrimes(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{13, 17, 31} {
		x := randC(n, rng)
		want := refDFT(x, false)
		got := make([]complex128, n)
		fftC(got, x, false)
		if d := maxDiff(got, want); d > 1e-10 {
			t.Errorf("bluestein n=%d: max diff %g", n, d)
		}
	}
}

func TestIrfftHalfSpectrum(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{8, 9, 16, 17} {
		// Build a real signal, take its forward half-spectrum, invert.
		sig := make([]complex128, n)
		for i := range sig {
			sig[i] = complex(rng.NormFloat64(), 0)
		}
		full := refDFT(sig, false)
		half := full[:n/2+1]
		out := make([]float64, n)
		irfftC(out, half)
		for i := range out {
			if d := math.Abs(out[i] - real(sig[i])); d > 1e-10 {
				t.Errorf("irfft n=%d [%d]: diff %g", n, i, d)
			}
		}
		// Numpy convention: Nyquist bin forced real for even n even when the
		// input half-spectrum is not hermitian-symmetric.
		if n%2 == 0 {
			nonSym := make([]complex128, n/2+1)
			for i := range nonSym {
				nonSym[i] = randC(1, rng)[0]
			}
			irfftC(out, nonSym)
			want := refDFT(func() []complex128 {
				f := make([]complex128, n)
				copy(f, nonSym)
				f[n/2] = complex(real(f[n/2]), 0)
				for i := 1; i <= (n-1)/2; i++ {
					f[n-i] = cmplx.Conj(f[i])
				}
				return f
			}(), true)
			for i := range want {
				want[i] *= complex(1/float64(n), 0)
			}
			for i := range out {
				if d := math.Abs(out[i] - real(want[i])); d > 1e-10 {
					t.Errorf("irfft nonsym n=%d [%d]: diff %g", n, i, d)
				}
			}
		}
	}
}
