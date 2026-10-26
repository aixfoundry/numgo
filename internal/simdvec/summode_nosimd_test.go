//go:build !goexperiment.simd

package simdvec

// sumLaneMerge: the default SumF64 is the plain sequential loop — no lanes.
const sumLaneMerge = false
