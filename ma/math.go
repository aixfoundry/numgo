package ma

import (
	"math"
	"math/cmplx"

	"github.com/aixfoundry/numgo"
)

// Math functions with domain masking (numpy ma): positions where the real
// function is undefined (negative Sqrt/Log inputs, Log1p below -1) become
// masked instead of warning; the IEEE value stays in the data. Passthrough
// functions (Exp/Sin/Cos/Tan/Floor/Ceil/Round) only propagate the mask.

// Sqrt returns the element-wise square root; negative real elements are
// masked (complex input uses the principal branch, no domain).
func (m *Masked[T]) Sqrt() *Masked[T] {
	if isComplexKind[T]() {
		return m.unary(func(x T) T { return cmTo[T](cmplx.Sqrt(cmOf(x))) }, nil)
	}
	return m.unary(func(x T) T { return fromParts[T](math.Sqrt(realOf(x)), 0) },
		func(x T) bool { return realOf(x) < 0 })
}

// Log returns the natural logarithm; real elements < 0 are masked
// (Log(0) stays -Inf and unmasked, matching numpy's divide warning domain).
func (m *Masked[T]) Log() *Masked[T] {
	if isComplexKind[T]() {
		return m.unary(func(x T) T { return cmTo[T](cmplx.Log(cmOf(x))) }, nil)
	}
	return m.unary(func(x T) T { return fromParts[T](math.Log(realOf(x)), 0) },
		func(x T) bool { return realOf(x) < 0 })
}

// Log2 returns the base-2 logarithm with Log's domain.
func (m *Masked[T]) Log2() *Masked[T] {
	if isComplexKind[T]() {
		return m.unary(func(x T) T { return cmTo[T](cmplx.Log(cmOf(x)) * complex(1/math.Ln2, 0)) }, nil)
	}
	return m.unary(func(x T) T { return fromParts[T](math.Log2(realOf(x)), 0) },
		func(x T) bool { return realOf(x) < 0 })
}

// Log10 returns the base-10 logarithm with Log's domain.
func (m *Masked[T]) Log10() *Masked[T] {
	if isComplexKind[T]() {
		return m.unary(func(x T) T { return cmTo[T](cmplx.Log(cmOf(x)) * complex(1/math.Ln10, 0)) }, nil)
	}
	return m.unary(func(x T) T { return fromParts[T](math.Log10(realOf(x)), 0) },
		func(x T) bool { return realOf(x) < 0 })
}

// Log1p returns log(1+x); real elements < -1 are masked.
func (m *Masked[T]) Log1p() *Masked[T] {
	if isComplexKind[T]() {
		return m.unary(func(x T) T { return cmTo[T](cmplx.Log(1 + cmOf(x))) }, nil)
	}
	return m.unary(func(x T) T { return fromParts[T](math.Log1p(realOf(x)), 0) },
		func(x T) bool { return realOf(x) < -1 })
}

// Exp returns the element-wise exponential (mask propagates only).
func (m *Masked[T]) Exp() *Masked[T] {
	return m.unary(func(x T) T { return cmTo[T](cmplx.Exp(cmOf(x))) }, nil)
}

// Sin returns the element-wise sine (mask propagates only).
func (m *Masked[T]) Sin() *Masked[T] {
	return m.unary(func(x T) T { return cmTo[T](cmplx.Sin(cmOf(x))) }, nil)
}

// Cos returns the element-wise cosine (mask propagates only).
func (m *Masked[T]) Cos() *Masked[T] {
	return m.unary(func(x T) T { return cmTo[T](cmplx.Cos(cmOf(x))) }, nil)
}

// Tan returns the element-wise tangent (mask propagates only).
func (m *Masked[T]) Tan() *Masked[T] {
	return m.unary(func(x T) T { return cmTo[T](cmplx.Tan(cmOf(x))) }, nil)
}

// Floor returns the element-wise floor (mask propagates only).
func (m *Masked[T]) Floor() *Masked[T] {
	return m.unary(func(x T) T { return fromParts[T](math.Floor(realOf(x)), 0) }, nil)
}

// Ceil returns the element-wise ceiling (mask propagates only).
func (m *Masked[T]) Ceil() *Masked[T] {
	return m.unary(func(x T) T { return fromParts[T](math.Ceil(realOf(x)), 0) }, nil)
}

// Round returns the element-wise round-half-to-even (mask propagates
// only).
func (m *Masked[T]) Round() *Masked[T] {
	return m.unary(func(x T) T { return fromParts[T](math.RoundToEven(realOf(x)), 0) }, nil)
}

// realOf extracts the real part as float64.
func realOf[T numgo.Number](v T) float64 {
	re, _ := parts(v)
	return re
}

// cmOf converts any Number to complex128.
func cmOf[T numgo.Number](v T) complex128 {
	re, im := parts(v)
	return complex(re, im)
}

// cmTo builds T from a complex128 result.
func cmTo[T numgo.Number](c complex128) T {
	return fromParts[T](real(c), imag(c))
}

func cmplxPow(x, y complex128) complex128 { return cmplx.Pow(x, y) }
