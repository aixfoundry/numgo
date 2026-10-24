package numgo

import (
	"math"
)

// This file ports the numpy.lib type_check subset: real/imaginary parts,
// real_if_close and nan_to_num. All are pure (new data).

// Pure: Real returns the element-wise real parts as float64. Real element
// types are converted (float32 widens to float64); complex elements drop the
// imaginary part.
func (a *Array[T]) Real() *Array[float64] {
	return mapReal[float64](a, func(re, _ float64) float64 { return re })
}

// Pure: Imag returns the element-wise imaginary parts as float64. Real
// element types yield all zeros (documented numpy behavior: .imag of a real
// array is zeros).
func (a *Array[T]) Imag() *Array[float64] {
	return mapReal[float64](a, func(_, im float64) float64 { return im })
}

// mapReal builds a P-valued array from the (re, im) parts of every element.
func mapReal[P Float, T Number](a *Array[T], f func(re, im float64) P) *Array[P] {
	out := Zeros[P](a.Shape()...)
	d := out.Data()
	for i, v := range a.data {
		re, im := parts(v)
		d[i] = f(re, im)
	}
	return out
}

// Pure: RealIfClose returns Real unless a is complex with any |imag| >= tol,
// where it panics. tol is a multiple of the machine epsilon (numpy
// real_if_close); tol <= 1 uses the absolute value as the tolerance, matching
// numpy's "if the tolerance is <=1, use the absolute tolerance". Documented
// deviation: numpy returns the COMPLEX array unchanged in that case; this
// package panics because the caller asked for real values that cannot be
// given losslessly.
func (a *Array[T]) RealIfClose(tol float64) *Array[float64] {
	if isComplexKind[T]() && len(a.data) > 0 {
		if tol > 1 {
			tol *= 2.220446049250313e-16 // numpy.finfo(complex128).eps
		}
		for _, v := range a.data {
			if _, im := parts(v); math.Abs(im) >= tol {
				panicRealIfClose()
			}
		}
	}
	return a.Real()
}

// Pure: NanToNum replaces NaN with nan and ±Inf with posinf/neginf
// element-wise (numpy nan_to_num). For complex elements each component is
// replaced independently; replacing a component leaves the other intact, so a
// complex NaN-replacement writes NaN+0i parts as (nan value, 0).
func (a *Array[T]) NanToNum(nan, posinf, neginf T) *Array[T] {
	return mapValues(a, func(v T) T {
		re, im := parts(v)
		return scalarFromParts[T](fixNonFinite(re, mustFloat(nan), mustFloat(posinf), mustFloat(neginf)),
			fixNonFinite(im, mustFloat(nan), mustFloat(posinf), mustFloat(neginf)))
	})
}

// fixNonFinite maps NaN/±Inf on one float component to the replacements.
func fixNonFinite(x, nan, posinf, neginf float64) float64 {
	switch {
	case math.IsNaN(x):
		return nan
	case math.IsInf(x, 1):
		return posinf
	case math.IsInf(x, -1):
		return neginf
	}
	return x
}

// Pure: NanToNumDefault is NanToNum with numpy's defaults: NaN becomes 0 and
// ±Inf become the largest/smallest finite values of the element type
// (math.MaxFloat64 for float64; float32 elements use math.MaxFloat32).
func (a *Array[T]) NanToNumDefault() *Array[T] {
	max := math.MaxFloat64
	if isFloat32Kind[T]() {
		max = math.MaxFloat32
	}
	return a.NanToNum(scalarFromParts[T](0, 0), scalarFromParts[T](max, 0), scalarFromParts[T](-max, 0))
}
