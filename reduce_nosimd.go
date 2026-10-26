//go:build !goexperiment.simd

package numgo

// Default-build no-op: reductions keep the nditer walk verbatim (pre-P2 bits
// and speed). The GOEXPERIMENT=simd twin lives in reduce_simd.go.
func simdSumF64[T Number](a *Array[T], out []T, axes []int, isPureSum bool) bool {
	return false
}
