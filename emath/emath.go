// Package emath ports numpy.emath (lib.scimath): math functions that return
// mathematically valid answers in the complex domain instead of NaN.
//
// Translated in part from NumPy (numpy/lib/_scimath_impl.py), BSD-3-Clause;
// see LICENSES/numpy-BSD3-CLAUSE.txt.
//
// Documented deviation from numpy: numpy preserves the real dtype when no
// imaginary part would appear (emath.sqrt([1,4]) stays float64); these
// functions ALWAYS return *numgo.Array[complex128] with a zero imaginary part
// for in-domain elements. Use Array.Real to extract real parts. Ported from
// numpy/lib/_scimath_impl.py: the _fix_real_* promotion guards only decide
// numpy's result dtype, so with a fixed complex128 output the port reduces to
// applying the math/cmplx function to every element (real inputs get a +0i part,
// matching numpy's astype(cdouble)).
package emath

import (
	"math"
	"math/cmplx"

	"github.com/aixfoundry/numgo"
)

// mapC converts a to complex128 and applies f to every element.
func mapC[T numgo.Number](a *numgo.Array[T], f func(complex128) complex128) *numgo.Array[complex128] {
	src := numgo.AsType[T, complex128](a)
	out := numgo.Zeros[complex128](a.Shape()...)
	d := out.Data()
	s := src.Data()
	for i, v := range s {
		d[i] = f(v)
	}
	return out
}

// Sqrt returns the element-wise square root: negative real elements give an
// imaginary result (numpy emath.sqrt).
func Sqrt[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, cmplx.Sqrt)
}

// Log returns the element-wise natural logarithm: real negative elements give
// the principal complex value (numpy emath.log).
func Log[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, cmplx.Log)
}

// Log2 returns the element-wise base-2 logarithm (numpy emath.log2), computed
// as Log(x)/ln2 like numpy's complex log2.
func Log2[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, func(v complex128) complex128 { return cmplx.Log(v) * complex(1/math.Ln2, 0) })
}

// Log10 returns the element-wise base-10 logarithm (numpy emath.log10),
// computed as Log(x)/ln10 like numpy's complex log10.
func Log10[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, func(v complex128) complex128 { return cmplx.Log(v) * complex(1/math.Ln10, 0) })
}

// Logn returns the element-wise log base n of x (numpy emath.logn, note the
// argument order). A negative base is handled in the complex domain, like
// numpy's _fix_real_lt_zero(n).
func Logn[T numgo.Number](n float64, x *numgo.Array[T]) *numgo.Array[complex128] {
	ln := cmplx.Log(complex(n, 0))
	return mapC(x, func(v complex128) complex128 { return cmplx.Log(v) / ln })
}

// Power returns x**p element-wise after broadcasting x and p (numpy
// emath.power): negative real bases give complex results.
//
// shortcut: Go's cmplx.Pow computes exp(p*log(x)), so integral exponents on
// negative bases carry ~1e-16 imaginary residue where numpy's complex power
// yields exactly 0i; within test tolerance.
func Power[T numgo.Number](x, p *numgo.Array[T]) *numgo.Array[complex128] {
	b := numgo.BroadcastArrays(x, p)
	xs := numgo.AsType[T, complex128](b[0]).Data()
	ps := numgo.AsType[T, complex128](b[1]).Data()
	out := numgo.Zeros[complex128](b[0].Shape()...)
	d := out.Data()
	for i := range xs {
		d[i] = cmplx.Pow(xs[i], ps[i])
	}
	return out
}

// Arccos returns the element-wise inverse cosine: |x| > 1 gives the complex
// principal value (numpy emath.arccos).
func Arccos[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, cmplx.Acos)
}

// Arcsin returns the element-wise inverse sine: |x| > 1 gives the complex
// principal value (numpy emath.arcsin).
func Arcsin[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, cmplx.Asin)
}

// Arctanh returns the element-wise inverse hyperbolic tangent: |x| > 1 gives
// the complex principal value, ±1 gives ±Inf (numpy emath.arctanh).
func Arctanh[T numgo.Number](a *numgo.Array[T]) *numgo.Array[complex128] {
	return mapC(a, cmplx.Atanh)
}
