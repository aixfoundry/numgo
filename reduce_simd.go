//go:build goexperiment.simd

package numgo

import (
	"github.com/aixfoundry/numgo/internal/simdvec"
)

// simdSumF64 handles float64 whole-array and axis-0 Sum on contiguous input,
// mirroring nditer.Reduce's fast-path guards; reports whether it ran. The
// default-build twin (reduce_nosimd.go) is a no-op, keeping the nditer walk
// verbatim. Gated to the plain-add Sum fold ONLY: callers with any other
// fold function (NanSum's NaN skip, Mean/Var's reuse) must pass isPureSum.
//
// Semantics: whole-array Sum uses simdvec.SumF64 (lane accumulators merged at
// the end, pairwise-ish order — last ulp may differ from the default build's
// sequential walk; documented in the simdvec package doc). Axis-0 Sum is
// bit-identical to the default build in every case: lanes span columns, each
// output element accumulates rows in ascending order exactly like nditer's
// fast path. Prod/Max/Min/… never reach here — they keep the nditer walk.
func simdSumF64[T Number](a *Array[T], out []T, axes []int, isPureSum bool) bool {
	if !isPureSum || len(axes) == 0 || len(out) == 0 {
		return false
	}
	d, ok := any(a.data).([]float64)
	if !ok {
		return false
	}
	o, ook := any(out).([]float64)
	if !ook || !a.IsContiguous() || len(a.data) < a.Size()+a.offset {
		return false
	}
	n := a.Size()
	src := d[a.offset : a.offset+n]
	if len(axes) == a.Ndim() {
		// Whole-array fold: sum in one shot.
		o[0] = simdvec.SumF64(src)
		return true
	}
	if a.Ndim() > 1 && a.shape[0] > 0 && len(axes) == 1 && axes[0] == 0 {
		// Axis-0 fold: out holds row 0 (identity-prefilled by the caller),
		// rows 1.. add in order — same structure as nditer's fast path.
		w := n / a.shape[0]
		if len(out) < w {
			return false
		}
		copy(o, src[:w])
		simdvec.SumAxis0F64(o[:w], src, a.shape[0], w)
		return true
	}
	return false
}
