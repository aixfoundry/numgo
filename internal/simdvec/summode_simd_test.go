//go:build goexperiment.simd

package simdvec

import "simd"

// sumLaneMerge: the simd SumF64 keeps l lane accumulators merged at the end —
// pairwise-ish order, distinct from the sequential default oracle.
const sumLaneMerge = true

// sumLanes is the runtime lane width SumF64 uses (4 on AVX2, 8 on AVX-512);
// lazily resolved — .Len() at init/link time breaks under this experiment.
func sumLanes() int { return simd.Float64s{}.Len() }
