//go:build !goexperiment.simd

package simdvec

// fmaMode selects the bit oracle in simdvec_test.go: the default build's
// restored pre-P1 kernel accumulates plain mul+add — oracle must match those
// bits exactly.
const fmaMode = false
