package fft

import (
	"math"
	"math/cmplx"
	"sync"
)

// Own FFT backend: recursive mixed-radix Cooley-Tukey (radix 2/3/5/7) with a
// Bluestein chirp-z convolution for prime lengths. Forward: A_k =
// Σ_j a_j·exp(-2πi·jk/n); the inverse conjugates around an unnormalized
// forward pass (the 1/n scaling stays with Norm in fft.go).
//
// Twiddle tables and Bluestein chirp/kernel spectra are cached per (n, sign)
// in package-level sync.Maps (bit-identical: entries are cis(sign·k) with the
// same float expression the loops used inline). Recursion work buffers are
// carved from one scratch arena per top-level call: fftRec(a) needs 2·len(a)
// scratch, since each radix level uses n for its gather buffer and leaves the
// tail for the single active child (sub-transforms run sequentially).

// twiddleKey identifies a cached table or Bluestein kernel by length and
// direction.
type twiddleKey struct {
	n       int
	inverse bool
}

var (
	// cacheMu guards the entry counts below. Eviction is deliberately crude:
	// when the total cached entries across all three caches crosses
	// maxCacheEntries, everything is dropped and rebuilt on demand. Workloads
	// with few distinct transform sizes never trip it; adversarial
	// many-sizes workloads pay a rebuild per window instead of growing
	// without bound. No LRU — eviction counts events, not bytes.
	cacheMu        sync.Mutex
	twiddleCache   sync.Map // twiddleKey -> []complex128 of cis(sign·k)
	bluesteinCache sync.Map // twiddleKey -> *bluesteinKernel
	bluesteinWork  sync.Map // int m -> *sync.Pool of []complex128 (len 3m)
	cacheEntries   int
)

// maxCacheEntries bounds the combined twiddle/bluestein/pool-key cache size.
const maxCacheEntries = 256

// noteCacheEntry records a new cache insertion and clears all caches when the
// combined entry count exceeds the bound. Called after a successful
// LoadOrStore that inserted; races only over-count, which just evicts sooner.
func noteCacheEntry() {
	cacheMu.Lock()
	cacheEntries++
	if cacheEntries > maxCacheEntries {
		twiddleCache.Range(func(key, _ any) bool {
			twiddleCache.Delete(key)
			return true
		})
		bluesteinCache.Range(func(key, _ any) bool {
			bluesteinCache.Delete(key)
			return true
		})
		bluesteinWork.Range(func(key, _ any) bool {
			bluesteinWork.Delete(key)
			return true
		})
		cacheEntries = 0
	}
	cacheMu.Unlock()
}

// cacheEntryCount reports the tracked entry count (white-box tests).
func cacheEntryCount() int {
	cacheMu.Lock()
	defer cacheMu.Unlock()
	return cacheEntries
}

// bluesteinScratch rents a workspace of 3m complex: fa = w[:m], sc = w[m:].
// Bluestein sits at recursion leaves needing ~6n elements — more than the
// per-call arena carries — so its buffers come from a size-keyed pool instead
// (concurrent transforms of one size contend only on the pool).
func bluesteinScratch(m int) (w []complex128, put func()) {
	p, loaded := bluesteinWork.LoadOrStore(m, &sync.Pool{New: func() any {
		return make([]complex128, 3*m)
	}})
	if !loaded {
		noteCacheEntry()
	}
	v := p.(*sync.Pool).Get()
	w = v.([]complex128)
	return w, func() { p.(*sync.Pool).Put(v) }
}

// bluesteinKernel holds the per-(n, sign) reusable chirp and the FFT of the
// wrapped conjugate-chirp kernel; only fa depends on the input.
type bluesteinKernel struct {
	chirp []complex128 // chirp[j] = exp(sign/2·j²/n·i), j < n
	spec  []complex128 // forward FFT of the wrapped kernel, len m
	m     int          // convolution length (next pow2 ≥ 2n−1)
}

// twiddles returns the cached table tw[k] = cis(sign·k). The table covers
// every j·b index the n-loops can form: n² for the naive base case (n ≤ 32)
// and r·n for a radix-r level. When the combined cache count crosses
// maxCacheEntries all caches are cleared (simple whole-cache eviction; no
// LRU — workloads with few sizes never trip it).
func twiddles(n int, inverse bool) []complex128 {
	key := twiddleKey{n, inverse}
	if v, ok := twiddleCache.Load(key); ok {
		return v.([]complex128)
	}
	l := n * n
	if n > 32 {
		l = smallFactor(n) * n
	}
	sign := twiddleSign(n, inverse)
	tw := make([]complex128, l)
	for k := range tw {
		tw[k] = cis(sign * float64(k))
	}
	v, loaded := twiddleCache.LoadOrStore(key, tw)
	if !loaded {
		noteCacheEntry()
	}
	return v.([]complex128)
}

// fftC transforms src into dst (same length). dst may alias src.
func fftC(dst, src []complex128, inverse bool) {
	fftScratch(dst, src, inverse, make([]complex128, 2*len(dst)))
}

// fftScratch is fftC with a caller-provided scratch arena of at least
// 2·len(dst), so batch callers transform many lines through one buffer.
func fftScratch(dst, src []complex128, inverse bool, scratch []complex128) {
	copy(dst, src)
	fftRec(dst, inverse, scratch)
}

// fftRec computes an unnormalized transform of a in place; scratch must hold
// at least 2·len(a) elements (invariant maintained by the carvings below).
func fftRec(a []complex128, inverse bool, scratch []complex128) {
	n := len(a)
	switch {
	case n <= 1:
		return
	case n <= 32:
		dftNaive(a, inverse, scratch)
	case smallFactor(n) > 0:
		fftRadix(a, smallFactor(n), inverse, scratch)
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
func fftRadix(a []complex128, r int, inverse bool, scratch []complex128) {
	n := len(a)
	m := n / r
	tw := twiddles(n, inverse)
	buf := scratch[:n]
	// Gather and sub-transform each column in place in buf: bin b of column
	// j lands at buf[j·m + b mod m].
	for j := range r {
		off := j * m
		for k := range m {
			buf[off+k] = a[j+k*r]
		}
		if m > 1 {
			fftRec(buf[off:off+m], inverse, scratch[n:])
		}
	}
	// Combine: X[k + q·m] = Σ_j buf[j·m+k]·cis(sign·j·(k+q·m)).
	for b := range n {
		var s complex128
		for j := range r {
			s += buf[j*m+b%m] * tw[j*b]
		}
		a[b] = s
	}
}

// fftBluestein computes an arbitrary-n DFT as a power-of-2 circular
// convolution: X[k] = c[k]·Σ_j a[j]·conj(c[j])·conj(c[k−j]) with the chirp
// c[j] = exp(s·j²/2), s = ±2π/n. The chirp is periodic in j² mod 2n and
// conj(c) is even, so the wrapped kernel b needs only indices 0..n−1. Chirp
// and kernel spectrum come from the cache; only fa is per-call.
func fftBluestein(a []complex128, inverse bool) {
	n := len(a)
	kern := bluesteinKernelFor(n, inverse)
	w, put := bluesteinScratch(kern.m)
	defer put()
	fa := w[:kern.m]
	sc := w[kern.m:]
	fa[0] = a[0]
	for j := 1; j < n; j++ {
		fa[j] = a[j] * cmplx.Conj(kern.chirp[j])
	}
	fftRec(fa, false, sc)
	for i := range fa {
		fa[i] *= kern.spec[i]
	}
	fftRec(fa, true, sc)
	scale := complex(1/float64(kern.m), 0)
	for k := range a {
		a[k] = kern.chirp[k] * fa[k] * scale
	}
}

// bluesteinKernelFor builds (once) and returns the cached kernel for (n, sign).
func bluesteinKernelFor(n int, inverse bool) *bluesteinKernel {
	key := twiddleKey{n, inverse}
	if v, ok := bluesteinCache.Load(key); ok {
		return v.(*bluesteinKernel)
	}
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	sign := twiddleSign(n, inverse)
	chirp := make([]complex128, n)
	chirp[0] = 1
	fb := make([]complex128, m)
	fb[0] = 1
	for j := 1; j < n; j++ {
		c := cmplx.Exp(complex(0, sign/2*float64(j*j%(2*n))))
		chirp[j] = c
		cc := cmplx.Conj(c)
		fb[j] = cc
		fb[m-j] = cc
	}
	fftC(fb, fb, false)
	kern := &bluesteinKernel{chirp: chirp, spec: fb, m: m}
	v, loaded := bluesteinCache.LoadOrStore(key, kern)
	if !loaded {
		noteCacheEntry()
	}
	return v.(*bluesteinKernel)
}

// dftNaive is the direct O(n²) DFT, used as the small-n base case; scratch
// holds its copy of a (n ≤ 32).
func dftNaive(a []complex128, inverse bool, scratch []complex128) {
	n := len(a)
	tmp := scratch[:n]
	copy(tmp, a)
	tw := twiddles(n, inverse)
	for k := range a {
		var sum complex128
		for j, v := range tmp {
			sum += v * tw[j*k]
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
	irfftUnnormBuf(dst, half, make([]complex128, n), make([]complex128, 2*n))
}

// irfftUnnormBuf is irfftUnnorm with caller-provided work buffers (full has
// len(dst) capacity, scratch 2·len(dst)) so batch callers reuse them per row.
func irfftUnnormBuf(dst []float64, half, full, scratch []complex128) {
	n := len(dst)
	copy(full, half)
	if n%2 == 0 {
		h := n / 2
		full[h] = complex(real(full[h]), 0)
	}
	for i := 1; i <= (n-1)/2; i++ {
		full[n-i] = cmplx.Conj(full[i])
	}
	fftScratch(full, full, true, scratch)
	for i := range dst {
		dst[i] = real(full[i])
	}
}
