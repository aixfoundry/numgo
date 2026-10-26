//go:build goexperiment.simd

package simdvec

// sumLaneMerge: the simd SumF64 keeps l lane accumulators merged at the end —
// pairwise-ish order, distinct from the sequential default oracle.
const sumLaneMerge = true
