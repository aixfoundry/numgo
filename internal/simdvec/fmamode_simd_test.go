//go:build goexperiment.simd

package simdvec

// fmaMode selects the bit oracle in simdvec_test.go: the simd build's MulAdd
// lanes accumulate fused, matching a per-element math.FMA oracle.
const fmaMode = true
