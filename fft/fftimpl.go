package fft

import (
	"math"
	"math/cmplx"
)

// Own FFT backend: recursive mixed-radix Cooley-Tukey (radix 2/3/5/7) with a
// Bluestein chirp-z convolution for prime lengths. Forward: A_k =
// Σ_j a_j·exp(-2πi·jk/n); the inverse conjugates around an unnormalized
// forward pass (the 1/n scaling stays with Norm in fft.go).

// fftC transforms src into dst (same length). dst may alias src.
func fftC(dst, src []complex128, inverse bool) {
	copy(dst, src)
	fftRec(dst, inverse)
}

// fftRec computes an unnormalized transform of a in place.
func fftRec(a []complex128, inverse bool) {
	n := len(a)
	switch {
	case n <= 1:
		return
	case n <= 32:
		dftNaive(a, inverse)
	case smallFactor(n) > 0:
		fftRadix(a, smallFactor(n), inverse)
	default:
		fftBluestein(a, inverse)
	}
}

// smallFactor returns a prime factor ≤ 7 of n, or 0.
func smallFactor(n int) int {
	for _, p := range [...]int{2, 3, 5, 7} {
		if n%p == 0 {
			return p
		}
	}
	return 0
}

// fftRadix computes a radix-r Cooley-Tukey step: view a as r interleaved
// subsequences of length m = n/r; X[b] = Σ_j col_j[b mod m]·cis(sign·j·b).
func fftRadix(a []complex128, r int, inverse bool) {
	n := len(a)
	m := n / r
	sign := twiddleSign(n, inverse)
	// Gather and sub-transform each column, storing bin b of column j at
	// buf[j*m + b mod m].
	buf := make([]complex128, n)
	col := make([]complex128, m)
	for j := 0; j < r; j++ {
		for k := 0; k < m; k++ {
			col[k] = a[j+k*r]
		}
		if m > 1 {
			fftRec(col, inverse)
		}
		copy(buf[j*m:], col)
	}
	// Combine: X[k + q·m] = Σ_j buf[j·m+k]·cis(sign·j·(k+q·m)).
	for b := 0; b < n; b++ {
		var s complex128
		for j := 0; j < r; j++ {
			s += buf[j*m+b%m] * cis(sign*float64(j*b))
		}
		a[b] = s
	}
}

// fftBluestein computes an arbitrary-n DFT as a power-of-2 circular
// convolution: X[k] = c[k]·Σ_j a[j]·conj(c[j])·conj(c[k−j]) with the chirp
// c[j] = exp(s·j²/2), s = ±2π/n. The chirp is periodic in j² mod 2n and
// conj(c) is even, so the wrapped kernel b needs only indices 0..n−1.
func fftBluestein(a []complex128, inverse bool) {
	n := len(a)
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	sign := twiddleSign(n, inverse)
	chirp := make([]complex128, n)
	for j := range chirp {
		chirp[j] = cmplx.Exp(complex(0, sign/2*float64(j*j%(2*n))))
	}
	fa := make([]complex128, m)
	fb := make([]complex128, m)
	fa[0] = a[0]
	fb[0] = 1
	for j := 1; j < n; j++ {
		c := cmplx.Conj(chirp[j])
		fa[j] = a[j] * c
		fb[j] = c
		fb[m-j] = c
	}
	fftC(fa, fa, false)
	fftC(fb, fb, false)
	for i := range fa {
		fa[i] *= fb[i]
	}
	fftC(fa, fa, true)
	scale := complex(1/float64(m), 0)
	for k := range a {
		a[k] = chirp[k] * fa[k] * scale
	}
}

// dftNaive is the direct O(n²) DFT, used as the small-n base case.
func dftNaive(a []complex128, inverse bool) {
	n := len(a)
	tmp := make([]complex128, n)
	copy(tmp, a)
	sign := twiddleSign(n, inverse)
	for k := range a {
		var sum complex128
		for j, v := range tmp {
			sum += v * cis(sign*float64(j*k))
		}
		a[k] = sum
	}
}

// twiddleSign returns the raw twiddle angle factor ∓2π/n (forward/inverse).
func twiddleSign(n int, inverse bool) float64 {
	if inverse {
		return 2 * math.Pi / float64(n)
	}
	return -2 * math.Pi / float64(n)
}

func cis(theta float64) complex128 {
	return cmplx.Exp(complex(0, theta))
}

// irfftC inverts a numpy-convention half-spectrum (bins 0..n/2, the Nyquist
// bin taken real for even n) into n real samples, using only the given half
// plus implied conjugates — no hermitian symmetry of the input is assumed.
// Includes the 1/n like numpy irfft; fft.go applies no extra norm factor of
// its own beyond the Norm multiplier passed in.
func irfftC(dst []float64, half []complex128) {
	irfftUnnorm(dst, half)
	f := 1 / float64(len(dst))
	for i := range dst {
		dst[i] *= f
	}
}

// irfftUnnorm is irfftC without the 1/n (for callers that fold 1/n into the
// Norm factor themselves, as fft.go does).
func irfftUnnorm(dst []float64, half []complex128) {
	n := len(dst)
	full := make([]complex128, n)
	copy(full, half)
	if n%2 == 0 {
		h := n / 2
		full[h] = complex(real(full[h]), 0)
	}
	for i := 1; i <= (n-1)/2; i++ {
		full[n-i] = cmplx.Conj(full[i])
	}
	fftC(full, full, true)
	for i := range dst {
		dst[i] = real(full[i])
	}
}
