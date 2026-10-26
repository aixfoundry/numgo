package ma

import (
	"math"

	"github.com/aixfoundry/numgo"
)

// Constructors and masked-value helpers (numpy ma masked_* family).

// MaskedZeros returns a Masked zeros array (nomask).
func MaskedZeros[T numgo.Number](shape ...int) *Masked[T] {
	return New(numgo.Zeros[T](shape...))
}

// MaskedOnes returns a Masked ones array (nomask).
func MaskedOnes[T numgo.Number](shape ...int) *Masked[T] {
	return New(numgo.Ones[T](shape...))
}

// MaskedFull returns a Masked array filled with v (nomask).
func MaskedFull[T numgo.Number](v T, shape ...int) *Masked[T] {
	return New(numgo.Full(v, shape...))
}

// MaskedAll returns a Masked array with every element masked. The data
// carries zeros (numpy leaves whatever Empty produced; masked anyway).
func MaskedAll[T numgo.Number](shape ...int) *Masked[T] {
	m := New(numgo.Zeros[T](shape...))
	for i := range m.mask {
		m.mask[i] = true
	}
	return m
}

// MaskedEqual masks elements equal to v.
func MaskedEqual[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return eq(x, v) })
}

// MaskedNotEqual masks elements not equal to v.
func MaskedNotEqual[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return !eq(x, v) })
}

// MaskedLess masks elements strictly less than v.
func MaskedLess[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return realOf(x) < realOf(v) })
}

// MaskedLessEqual masks elements less than or equal to v.
func MaskedLessEqual[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return realOf(x) <= realOf(v) })
}

// MaskedGreater masks elements strictly greater than v.
func MaskedGreater[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return realOf(x) > realOf(v) })
}

// MaskedGreaterEqual masks elements greater than or equal to v.
func MaskedGreaterEqual[T numgo.Number](a *numgo.Array[T], v T) *Masked[T] {
	return maskedCompare(a, v, func(x, v T) bool { return realOf(x) >= realOf(v) })
}

// maskedCompare shares the compare family body.
func maskedCompare[T numgo.Number](a *numgo.Array[T], v T, hit func(x, v T) bool) *Masked[T] {
	m := New(a)
	for i, x := range a.Data() {
		m.mask[i] = hit(x, v)
	}
	return m
}

// eq compares two Numbers (complex equality allowed; NaN never equals).
func eq[T numgo.Number](x, v T) bool {
	xr, xi := parts(x)
	vr, vi := parts(v)
	return xr == vr && xi == vi
}

// MaskedWhere masks cond true positions of a (flat C order).
func MaskedWhere[T numgo.Number](cond []bool, a *numgo.Array[T]) *Masked[T] {
	if len(cond) != a.Size() {
		panicMaskedSize()
	}
	return FromData(a, cond)
}

// MaskedInvalid masks NaN and ±Inf elements (real and complex parts).
func MaskedInvalid[T numgo.Number](a *numgo.Array[T]) *Masked[T] {
	m := New(a)
	for i, x := range a.Data() {
		re, im := parts(x)
		if invalidPart(re) || invalidPart(im) {
			m.mask[i] = true
		}
	}
	return m
}

func invalidPart(f float64) bool {
	return math.IsNaN(f) || math.IsInf(f, 0)
}

// MaskedInside masks v1 <= x <= v2 (both inclusive).
func MaskedInside[T numgo.Number](a *numgo.Array[T], v1, v2 T) *Masked[T] {
	m := New(a)
	for i, x := range a.Data() {
		re := realOf(x)
		m.mask[i] = realOf(v1) <= re && re <= realOf(v2)
	}
	return m
}

// MaskedOutside masks x < v1 or x > v2.
func MaskedOutside[T numgo.Number](a *numgo.Array[T], v1, v2 T) *Masked[T] {
	m := New(a)
	for i, x := range a.Data() {
		re := realOf(x)
		m.mask[i] = re < realOf(v1) || re > realOf(v2)
	}
	return m
}

// slicesClone copies a bool slice (masks are never aliased across instances).
func slicesClone(s []bool) []bool {
	out := make([]bool, len(s))
	copy(out, s)
	return out
}
