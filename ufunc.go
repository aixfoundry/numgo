package numgo

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
	"github.com/aixfoundry/numgo/internal/simdvec"
)

// Add returns a new array with the element-wise sum of a and b, broadcasting
// both operands with full numpy right-alignment.
func (a *Array[T]) Add(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	// SIMD fast path: float64 add with all three arrays C-contiguous over the
	// same shape. a[i]+b[i] is exact IEEE and order-free, so the result is
	// bit-identical to the generic path; any other layout keeps the nditer
	// route. Only Add takes it — Sub/Mul/Div stay on zip2.
	if dst.Size() > 0 && simdAddF64(dst, la, a.data, lb, b.data, shape) {
		return dst
	}
	return zip2Into(dst, la, a.data, lb, b.data, shape, func(x, y T) T { return x + y })
}

// Sub returns a new array with the element-wise difference of a and b.
func (a *Array[T]) Sub(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x - y })
}

// Mul returns a new array with the element-wise product of a and b.
func (a *Array[T]) Mul(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x * y })
}

// Div returns a new array with the element-wise quotient of a and b. Division
// by zero follows IEEE (Inf/NaN), never panics.
func (a *Array[T]) Div(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x / y })
}

// AddInto writes the broadcast sum of a and b into dst and returns dst. dst
// must already have the broadcast result shape.
func (a *Array[T]) AddInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x + y })
}

// SubInto writes the broadcast difference of a and b into dst and returns dst.
func (a *Array[T]) SubInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x - y })
}

// MulInto writes the broadcast product of a and b into dst and returns dst.
func (a *Array[T]) MulInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x * y })
}

// DivInto writes the broadcast quotient of a and b into dst and returns dst.
func (a *Array[T]) DivInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x / y })
}

// AddValue returns a new array with v added to every element.
func (a *Array[T]) AddValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x + v })
}

// SubValue returns a new array with v subtracted from every element.
func (a *Array[T]) SubValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x - v })
}

// MulValue returns a new array with every element multiplied by v.
func (a *Array[T]) MulValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x * v })
}

// DivValue returns a new array with every element divided by v. Division by
// zero follows IEEE (Inf/NaN), never panics.
func (a *Array[T]) DivValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x / v })
}

// broadcastPair aligns a and b, panicking with the numpy message when their
// shapes are incompatible.
func broadcastPair[T Number](a, b *Array[T]) (*nditer.Layout, *nditer.Layout, []int) {
	shape, strides, err := nditer.Broadcast(a.shape, b.shape)
	if err != nil {
		panicBroadcast(a.shape, b.shape)
	}
	return &nditer.Layout{Shape: shape, Strides: strides[0]},
		&nditer.Layout{Shape: shape, Strides: strides[1]}, shape
}

// zip2 broadcasts a and b, allocates fresh storage, then feeds an op over them.
func zip2[T Number](la *nditer.Layout, adata []T, lb *nditer.Layout, bdata []T,
	shape []int, op func(x, y T) T) *Array[T] {
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	return zip2Into(dst, la, adata, lb, bdata, shape, op)
}

// zip2Into feeds op over broadcast operands into the preallocated dst.
func zip2Into[T Number](dst *Array[T], la *nditer.Layout, adata []T,
	lb *nditer.Layout, bdata []T, shape []int, op func(x, y T) T) *Array[T] {
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{la, lb}, [][]T{adata, bdata},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = op(src[0][i], src[1][i])
			}
		})
	return dst
}

// simdAddF64 runs simdvec.AddF64 when dst, a and b are all C-contiguous full
// slices of float64 over the same shape; otherwise it reports false and the
// caller falls back to the generic nditer path.
func simdAddF64[T Number](dst *Array[T], la *nditer.Layout, adata []T,
	lb *nditer.Layout, bdata []T, shape []int) bool {
	n := dst.Size()
	a, aok := any(adata).([]float64)
	b, bok := any(bdata).([]float64)
	d, dok := any(dst.data).([]float64)
	if !aok || !bok || !dok {
		return false
	}
	if !contigFull(la, a, shape, n) || !contigFull(lb, b, shape, n) ||
		!nditer.IsContiguous(dst.layout()) || len(dst.data) < n+dst.offset {
		return false
	}
	simdvec.AddF64(d[dst.offset:dst.offset+n], a[la.Offset:la.Offset+n], b[lb.Offset:lb.Offset+n])
	return true
}

// contigFull reports whether l addresses exactly shape in C order with no
// broadcasting and data has room for the whole run.
func contigFull[T any](l *nditer.Layout, data []T, shape []int, n int) bool {
	return nditer.ContiguousSame(l, shape) && len(data) >= n+l.Offset
}

// into2 broadcasts a and b and writes the op result into dst.
func into2[T Number](dst, a, b *Array[T], op func(x, y T) T) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	if !slices.Equal(dst.shape, shape) {
		panicInto(dst.shape, shape)
	}
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{la, lb}, [][]T{a.data, b.data},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = op(src[0][i], src[1][i])
			}
		})
	return dst
}

// mapValues copies a and applies f to every element.
func mapValues[T Number](a *Array[T], f func(T) T) *Array[T] {
	out := newArray(a.Shape(), nditer.COrderStrides(a.shape), 0, make([]T, a.Size()))
	if out.Size() == 0 {
		return out
	}
	nditer.Zip(out.layout(), out.data, []*nditer.Layout{a.layout()}, [][]T{a.data},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = f(src[0][i])
			}
		})
	return out
}

// Binary ufuncs beyond the arithmetic four: min/max with and without NaN
// propagation, the two remainder conventions, floor division and power, plus
// scalar bounds clipping. Real-only ops compute in float64 and convert back
// (the float32 convention); division by zero stays IEEE (Inf/NaN), no panic.

// Maximum returns the element-wise maximum of a and b, broadcasting both
// operands. If one of the elements being compared is a NaN, then that
// element is returned (numpy maximum propagates NaNs).
func (a *Array[T]) Maximum(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, maximumElem[T])
}

// Minimum returns the element-wise minimum of a and b, broadcasting both
// operands. If one of the elements being compared is a NaN, then that
// element is returned (numpy minimum propagates NaNs).
func (a *Array[T]) Minimum(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, minimumElem[T])
}

// MaximumValue returns a new array with each element replaced by its maximum
// with v, propagating NaNs like Maximum.
func (a *Array[T]) MaximumValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return maximumElem(x, v) })
}

// MinimumValue returns a new array with each element replaced by its minimum
// with v, propagating NaNs like Minimum.
func (a *Array[T]) MinimumValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return minimumElem(x, v) })
}

// Fmax returns the element-wise maximum of a and b, ignoring NaNs: a NaN
// element yields the other one, and two NaNs yield NaN (numpy fmax).
func (a *Array[T]) Fmax(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, fmaxElem[T])
}

// Fmin returns the element-wise minimum of a and b, ignoring NaNs: a NaN
// element yields the other one, and two NaNs yield NaN (numpy fmin).
func (a *Array[T]) Fmin(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, fminElem[T])
}

// maximumElem is numpy maximum for one pair: NaN wins over everything,
// otherwise the larger real part.
func maximumElem[T Number](x, y T) T {
	if isNaN(x) || isNaN(y) {
		return nanScalar[T]()
	}
	if greater(y, x) {
		return y
	}
	return x
}

// minimumElem is numpy minimum for one pair.
func minimumElem[T Number](x, y T) T {
	if isNaN(x) || isNaN(y) {
		return nanScalar[T]()
	}
	if less(y, x) {
		return y
	}
	return x
}

// fmaxElem is numpy fmax for one pair: NaN loses to any number.
func fmaxElem[T Number](x, y T) T {
	switch {
	case isNaN(x):
		return y
	case isNaN(y):
		return x
	}
	if greater(y, x) {
		return y
	}
	return x
}

// fminElem is numpy fmin for one pair.
func fminElem[T Number](x, y T) T {
	switch {
	case isNaN(x):
		return y
	case isNaN(y):
		return x
	}
	if less(y, x) {
		return y
	}
	return x
}

// pyMod is Python's % (numpy mod): the remainder has the sign of the divisor,
// unlike C's fmod. r += y once flips the sign of math.Mod's result into
// divisor orientation; a zero divisor leaves the NaN untouched.
func pyMod(x, y float64) float64 {
	r := math.Mod(x, y)
	if r != 0 && ((r < 0) != (y < 0)) {
		r += y
	}
	return r
}

// floorDiv is numpy floor_divide: floor(x/y), corrected by one when the
// truncated quotient rounds the wrong way, so x = y*q + pyMod(x, y) always
// holds. y == 0 follows IEEE (±Inf/NaN), no panic.
func floorDiv(x, y float64) float64 {
	q := math.Floor(x / y)
	if r := x - q*y; r != 0 && ((r < 0) != (y < 0)) {
		q--
	}
	return q
}

// Mod returns the element-wise Python-style remainder of a and b: the result
// has the same sign as the divisor (numpy mod). Complex arrays are not
// supported (numpy leaves complex mod undefined).
func (a *Array[T]) Mod(b *Array[T]) *Array[T] { return realBin(a, b, "Mod", pyMod) }

// Fmod returns the element-wise C-style remainder of a and b: the result has
// the same sign as the dividend (numpy fmod). Complex arrays are not
// supported.
func (a *Array[T]) Fmod(b *Array[T]) *Array[T] { return realBin(a, b, "Fmod", math.Mod) }

// ModValue returns the element-wise Python-style remainder with scalar v
// (sign of v); see Mod.
func (a *Array[T]) ModValue(v T) *Array[T] { return realBinValue(a, v, "Mod", pyMod) }

// FmodValue returns the element-wise C-style remainder with scalar v (sign
// of the element); see Fmod.
func (a *Array[T]) FmodValue(v T) *Array[T] { return realBinValue(a, v, "Fmod", math.Mod) }

// FloorDivide returns the element-wise floor quotient of a and b (numpy's //
// operator), consistent with Mod: x = y*FloorDivide(x,y) + Mod(x,y). Complex
// arrays are not supported.
func (a *Array[T]) FloorDivide(b *Array[T]) *Array[T] {
	return realBin(a, b, "FloorDivide", floorDiv)
}

// FloorDivideValue returns the element-wise floor quotient with scalar v;
// see FloorDivide.
func (a *Array[T]) FloorDivideValue(v T) *Array[T] {
	return realBinValue(a, v, "FloorDivide", floorDiv)
}

// Pow returns a with every element raised to the power of the corresponding
// element of b, broadcasting both operands (numpy power). Real elements use
// math.Pow, complex ones cmplx.Pow.
func (a *Array[T]) Pow(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	if isComplexKind[T]() {
		return zip2(la, a.data, lb, b.data, shape, func(x, y T) T {
			xre, xim := parts(x)
			yre, yim := parts(y)
			w := cmplx.Pow(complex(xre, xim), complex(yre, yim))
			return scalarFromParts[T](real(w), imag(w))
		})
	}
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T {
		return scalarFromParts[T](math.Pow(mustFloat(x), mustFloat(y)), 0)
	})
}

// PowValue returns a with every element raised to the scalar power p.
func (a *Array[T]) PowValue(p T) *Array[T] {
	pre, pim := parts(p)
	pc := complex(pre, pim)
	if isComplexKind[T]() {
		return mapValues(a, func(x T) T {
			re, im := parts(x)
			w := cmplx.Pow(complex(re, im), pc)
			return scalarFromParts[T](real(w), imag(w))
		})
	}
	return mapValues(a, func(x T) T {
		return scalarFromParts[T](math.Pow(mustFloat(x), pre), 0)
	})
}

// Clip clips (limits) the values in an array: values outside [lo, hi] are
// clipped to the interval edges. lo greater than hi panics (numpy raises
// ValueError for the same input).
func (a *Array[T]) Clip(lo, hi T) *Array[T] {
	if greater(lo, hi) {
		panicClipBounds()
	}
	return mapValues(a, func(x T) T { return minimumElem(maximumElem(x, lo), hi) })
}

// realBin applies the float64 op over broadcast a and b for real element
// types only, converting each result back to T.
func realBin[T Number](a, b *Array[T], op string, f func(x, y float64) float64) *Array[T] {
	if isComplexKind[T]() {
		panicComplexMod(op)
	}
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T {
		return scalarFromParts[T](f(mustFloat(x), mustFloat(y)), 0)
	})
}

// realBinValue is realBin with a scalar second operand.
func realBinValue[T Number](a *Array[T], v T, op string, f func(x, y float64) float64) *Array[T] {
	if isComplexKind[T]() {
		panicComplexMod(op)
	}
	y := mustFloat(v)
	return mapValues(a, func(x T) T { return scalarFromParts[T](f(mustFloat(x), y), 0) })
}

// panicComplexMod reports Mod/Fmod/FloorDivide on complex arrays, which
// numpy leaves undefined.
func panicComplexMod(op string) {
	panic(fmt.Sprintf("numgo: %s is not defined for complex arrays", op))
}

// panicClipBounds reports a Clip whose bounds are inverted (numpy raises
// ValueError for min > max).
func panicClipBounds() {
	panic("numgo: Clip: min must be less than or equal to max")
}
