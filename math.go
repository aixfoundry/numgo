package numgo

import (
	"math"
	"math/cmplx"
)

// Float is the real-valued subset of Number: the result element type of
// operations whose value is real even for complex input (Abs).
type Float interface {
	~float32 | ~float64
}

// Element-wise unary math functions: the numpy ufunc unary subset. All are
// pure (new data), preserve the input shape, and follow the IEEE semantics of
// the math and math/cmplx packages with no special cases (Log(0) = -Inf,
// Sqrt(-1) = NaN for real T, principal complex branches, and so on).
// Functions math/cmplx lacks are applied per-component to complex elements
// (numpy applies componentwise ufuncs to both parts too).

// mapUnary is the shared body of the unary math functions. It copies a and
// applies fr to real elements, fc to complex elements (per-component helpers
// pass fc built from fr). The element kind dispatch happens once; the typed
// fast paths use plain conversions.
func mapUnary[T Number](a *Array[T], fr func(float64) float64, fc func(complex128) complex128) *Array[T] {
	out := a.Copy()
	switch s := any(out.data).(type) {
	case []float64:
		for i, v := range s {
			s[i] = fr(v)
		}
	case []float32:
		for i, v := range s {
			s[i] = float32(fr(float64(v)))
		}
	case []complex64:
		for i, v := range s {
			s[i] = complex64(fc(complex128(v)))
		}
	case []complex128:
		for i, v := range s {
			s[i] = fc(v)
		}
	default:
		if isComplexKind[T]() {
			for i, v := range out.data {
				re, im := parts(v)
				w := fc(complex(re, im))
				out.data[i] = scalarFromParts[T](real(w), imag(w))
			}
		} else {
			for i, v := range out.data {
				re, _ := parts(v)
				out.data[i] = scalarFromParts[T](fr(re), 0)
			}
		}
	}
	return out
}

// perComponent lifts a real function to a complex one applying it to the real
// and imaginary parts independently.
func perComponent(f func(float64) float64) func(complex128) complex128 {
	return func(c complex128) complex128 { return complex(f(real(c)), f(imag(c))) }
}

// exp2c, log2c, log10c, log1pc, expm1c and cbrtc compose cmplx.Exp/Log/Pow,
// which lack dedicated functions.
func exp2c(c complex128) complex128 { return cmplx.Exp(c * complex(0, math.Ln2)) }
func log2c(c complex128) complex128 { return cmplx.Log(c) * complex(1/math.Ln2, 0) }
func log10c(c complex128) complex128 {
	return cmplx.Log(c) * complex(1/math.Ln10, 0)
}
func log1pc(c complex128) complex128 { return cmplx.Log(1 + c) }
func expm1c(c complex128) complex128 { return cmplx.Exp(c) - 1 }
func cbrtc(c complex128) complex128  { return cmplx.Pow(c, complex(1.0/3.0, 0)) }

// Pure: Abs returns the element-wise absolute value. For real T this is the
// magnitude; for complex T the result is real and not representable in
// Array[T], so it panics and points at AbsToReal.
func (a *Array[T]) Abs() *Array[T] {
	if isComplexKind[T]() {
		panicComplexAbs()
	}
	return mapUnary(a, math.Abs, nil)
}

// Pure: AbsToReal is the package-level element-wise modulus with an explicit
// float result type P (methods cannot add type parameters): for complex input
// it is |z| as P; for real input it is the absolute value converted to P.
// P cannot be inferred, so callers must name both types,
// e.g. AbsToReal[complex128, float64](a).
func AbsToReal[F Number, P Float](a *Array[F]) *Array[P] {
	out := Zeros[P](a.Shape()...)
	for i, v := range a.data {
		re, im := parts(v)
		out.data[i] = P(math.Hypot(re, im))
	}
	return out
}

// Pure: Sqrt returns the element-wise square root: the math/cmplx square root
// (principal branch; real Sqrt(-1) is NaN).
func (a *Array[T]) Sqrt() *Array[T] { return mapUnary(a, math.Sqrt, cmplx.Sqrt) }

// Pure: Cbrt returns the element-wise cube root. Complex elements get the
// real cube root applied per-component (math/cmplx has no Cbrt).
func (a *Array[T]) Cbrt() *Array[T] { return mapUnary(a, math.Cbrt, cbrtc) }

// Pure: Exp returns the element-wise exponential.
func (a *Array[T]) Exp() *Array[T] { return mapUnary(a, math.Exp, cmplx.Exp) }

// Pure: Exp2 returns the element-wise base-2 exponential, 2**x. Complex
// elements use exp(x·ln2).
func (a *Array[T]) Exp2() *Array[T] { return mapUnary(a, math.Exp2, exp2c) }

// Pure: Log returns the natural logarithm; Log(0) is -Inf (IEEE).
func (a *Array[T]) Log() *Array[T] { return mapUnary(a, math.Log, cmplx.Log) }

// Pure: Log2 returns the element-wise base-2 logarithm. Complex elements use
// Log(x)/ln2.
func (a *Array[T]) Log2() *Array[T] { return mapUnary(a, math.Log2, log2c) }

// Pure: Log10 returns the element-wise base-10 logarithm. Complex elements use
// Log(x)/ln10.
func (a *Array[T]) Log10() *Array[T] { return mapUnary(a, math.Log10, log10c) }

// Pure: Log1p returns log(1+x), accurate near zero. Complex elements use
// Log(1+x).
func (a *Array[T]) Log1p() *Array[T] { return mapUnary(a, math.Log1p, log1pc) }

// Pure: Expm1 returns e**x - 1, accurate near zero. Complex elements use
// Exp(x)-1.
func (a *Array[T]) Expm1() *Array[T] { return mapUnary(a, math.Expm1, expm1c) }

// Pure: Sin returns the element-wise sine.
func (a *Array[T]) Sin() *Array[T] { return mapUnary(a, math.Sin, cmplx.Sin) }

// Pure: Cos returns the element-wise cosine.
func (a *Array[T]) Cos() *Array[T] { return mapUnary(a, math.Cos, cmplx.Cos) }

// Pure: Tan returns the element-wise tangent.
func (a *Array[T]) Tan() *Array[T] { return mapUnary(a, math.Tan, cmplx.Tan) }

// Pure: Asin returns the element-wise arcsine.
func (a *Array[T]) Asin() *Array[T] { return mapUnary(a, math.Asin, cmplx.Asin) }

// Pure: Acos returns the element-wise arccosine.
func (a *Array[T]) Acos() *Array[T] { return mapUnary(a, math.Acos, cmplx.Acos) }

// Pure: Atan returns the element-wise arctangent.
func (a *Array[T]) Atan() *Array[T] { return mapUnary(a, math.Atan, cmplx.Atan) }

// Pure: Sinh returns the element-wise hyperbolic sine.
func (a *Array[T]) Sinh() *Array[T] { return mapUnary(a, math.Sinh, cmplx.Sinh) }

// Pure: Cosh returns the element-wise hyperbolic cosine.
func (a *Array[T]) Cosh() *Array[T] { return mapUnary(a, math.Cosh, cmplx.Cosh) }

// Pure: Tanh returns the element-wise hyperbolic tangent.
func (a *Array[T]) Tanh() *Array[T] { return mapUnary(a, math.Tanh, cmplx.Tanh) }

// Pure: Asinh returns the element-wise inverse hyperbolic sine.
func (a *Array[T]) Asinh() *Array[T] { return mapUnary(a, math.Asinh, cmplx.Asinh) }

// Pure: Acosh returns the element-wise inverse hyperbolic cosine.
func (a *Array[T]) Acosh() *Array[T] { return mapUnary(a, math.Acosh, cmplx.Acosh) }

// Pure: Atanh returns the element-wise inverse hyperbolic tangent.
func (a *Array[T]) Atanh() *Array[T] { return mapUnary(a, math.Atanh, cmplx.Atanh) }

// Pure: Floor returns the element-wise floor, per-component on complex
// elements (numpy applies to both parts).
func (a *Array[T]) Floor() *Array[T] { return mapUnary(a, math.Floor, perComponent(math.Floor)) }

// Pure: Ceil returns the element-wise ceiling, per-component on complex
// elements.
func (a *Array[T]) Ceil() *Array[T] { return mapUnary(a, math.Ceil, perComponent(math.Ceil)) }

// Pure: Round returns the element-wise round-half-to-even (numpy's round),
// per-component on complex elements.
func (a *Array[T]) Round() *Array[T] {
	return mapUnary(a, math.RoundToEven, perComponent(math.RoundToEven))
}

// Pure: Trunc returns the element-wise integer part, per-component on complex
// elements.
func (a *Array[T]) Trunc() *Array[T] { return mapUnary(a, math.Trunc, perComponent(math.Trunc)) }

// Pure: Conj returns the element-wise complex conjugate; a real array is
// copied unchanged.
func (a *Array[T]) Conj() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return v }, cmplx.Conj)
}
