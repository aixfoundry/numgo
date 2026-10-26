//go:build !goexperiment.simd

package simdvec

// sumLaneMerge: the default SumF64 is the plain sequential loop — no lanes.
const sumLaneMerge = false

// Default build: no lanes, but keep the loop bound derived so the tested
// lengths stay meaningful (sequential oracle is width-independent).
func sumLanes() int { return 8 }
