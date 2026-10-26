//go:build goexperiment.simd

package simdvec

import "simd"

// fmaMode selects the bit oracle in simdvec_test.go: the simd build's MulAdd
// lanes accumulate fused, matching a per-element math.FMA oracle.
const fmaMode = true

// lane widths the simd kernel uses for its panel boundaries — the tests must
// derive region classification from the SAME values the kernel computes
// (Float64s.Len()=4 on AVX2, 8 on AVX-512; Float32s: 8/16). Resolved lazily:
// calling .Len() at package-init/link time breaks under this experiment.
func f64Lanes() int { return simd.Float64s{}.Len() }
func f32Lanes() int { return simd.Float32s{}.Len() }
