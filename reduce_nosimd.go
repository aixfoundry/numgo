//go:build !goexperiment.simd

package numgo

// Default-build no-op: reductions keep the nditer walk verbatim (bits and
// speed unchanged). The GOEXPERIMENT=simd twin lives in reduce_simd.go.
func simdSumF64[T Number](a *Array[T], out []T, axes []int, isPureSum bool) bool {
	return false
}
