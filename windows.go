package numgo

import "math"

// Window functions: tapered coefficient sequences for signal processing,
// matching numpy's window functions (np.bartlett, np.hanning, np.hamming,
// np.blackman, np.kaiser). Each returns a 1-D array of length n of
// float64; n <= 0 returns an empty array and n == 1 returns [1].

// window allocates the output array and fills it with f(i, n).
func window(n int, f func(i int) float64) *Array[float64] {
	if n <= 0 {
		return Zeros[float64](0)
	}
	if n == 1 {
		return Ones[float64](1)
	}
	out := Zeros[float64](n)
	for i := range n {
		out.data[i] = f(i)
	}
	return out
}

// Bartlett returns the Bartlett window of length n, matching numpy's
// np.bartlett: a triangular taper with zero endpoints.
func Bartlett(n int) *Array[float64] {
	return window(n, func(i int) float64 {
		k := (float64(2*i) + 1 - float64(n)) / float64(n-1)
		if k <= 0 {
			return 1 + k
		}
		return 1 - k
	})
}

// Hanning returns the Hanning (Hann) window of length n, matching numpy's
// np.hanning: 0.5 - 0.5·cos(2πi/(n-1)).
func Hanning(n int) *Array[float64] {
	return window(n, func(i int) float64 {
		return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
	})
}

// Hamming returns the Hamming window of length n, matching numpy's
// np.hamming: 0.54 - 0.46·cos(2πi/(n-1)).
func Hamming(n int) *Array[float64] {
	return window(n, func(i int) float64 {
		return 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
	})
}

// Blackman returns the exact Blackman window of length n, matching numpy's
// np.blackman: 0.42 - 0.5·cos(2πi/(n-1)) + 0.08·cos(4πi/(n-1)).
func Blackman(n int) *Array[float64] {
	return window(n, func(i int) float64 {
		x := 2 * math.Pi * float64(i) / float64(n-1)
		return 0.42 - 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
	})
}

// Kaiser returns the Kaiser window of length n with shape parameter beta,
// matching numpy's np.kaiser: I0(beta·sqrt(1-((2i/(n-1))-1)²))/I0(beta).
func Kaiser(n int, beta float64) *Array[float64] {
	if n <= 0 {
		return Zeros[float64](0)
	}
	if n == 1 {
		return Ones[float64](1)
	}
	denom := i0(beta)
	out := Zeros[float64](n)
	for i := range n {
		r := (float64(i) - float64(n-1)/2.0) / (float64(n-1) / 2.0)
		out.data[i] = i0(beta*math.Sqrt(1-r*r)) / denom
	}
	return out
}

// i0 is the modified Bessel function of the first kind, order zero,
// via its power series (same expansion numpy's np.kaiser uses).
func i0(x float64) float64 {
	y := x * x / 4
	sum, term := 1.0, 1.0
	for k := 1; term > 1e-17; k++ {
		term *= y / float64(k*k)
		sum += term
	}
	return sum
}
