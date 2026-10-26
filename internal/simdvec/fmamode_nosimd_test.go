//go:build !goexperiment.simd

package simdvec

// fmaMode selects the bit oracle in simdvec_test.go: the default build's
// kernel accumulates plain mul+add — oracle must match those
// bits exactly.
const fmaMode = false

// Default build: the nosimd twin uses the hardcoded 4×8 panel shape.
func f64Lanes() int { return 8 }
func f32Lanes() int { return 8 }
