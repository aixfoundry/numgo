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

// Abs returns the element-wise absolute value. For real T this is the
// magnitude; for complex T the result is real and not representable in
// Array[T], so it panics and points at AbsToReal.
func (a *Array[T]) Abs() *Array[T] {
	if isComplexKind[T]() {
		panicComplexAbs()
	}
	return mapUnary(a, math.Abs, nil)
}

// AbsToReal is the package-level element-wise modulus with an explicit
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

// Sqrt returns the element-wise square root: the math/cmplx square root
// (principal branch; real Sqrt(-1) is NaN).
func (a *Array[T]) Sqrt() *Array[T] { return mapUnary(a, math.Sqrt, cmplx.Sqrt) }

// Cbrt returns the element-wise cube root. Complex elements get the
// real cube root applied per-component (math/cmplx has no Cbrt).
func (a *Array[T]) Cbrt() *Array[T] { return mapUnary(a, math.Cbrt, cbrtc) }

// Exp returns the element-wise exponential.
func (a *Array[T]) Exp() *Array[T] { return mapUnary(a, math.Exp, cmplx.Exp) }

// Exp2 returns the element-wise base-2 exponential, 2**x. Complex
// elements use exp(x·ln2).
func (a *Array[T]) Exp2() *Array[T] { return mapUnary(a, math.Exp2, exp2c) }

// Log returns the natural logarithm; Log(0) is -Inf (IEEE).
func (a *Array[T]) Log() *Array[T] { return mapUnary(a, math.Log, cmplx.Log) }

// Log2 returns the element-wise base-2 logarithm. Complex elements use
// Log(x)/ln2.
func (a *Array[T]) Log2() *Array[T] { return mapUnary(a, math.Log2, log2c) }

// Log10 returns the element-wise base-10 logarithm. Complex elements use
// Log(x)/ln10.
func (a *Array[T]) Log10() *Array[T] { return mapUnary(a, math.Log10, log10c) }

// Log1p returns log(1+x), accurate near zero. Complex elements use
// Log(1+x).
func (a *Array[T]) Log1p() *Array[T] { return mapUnary(a, math.Log1p, log1pc) }

// Expm1 returns e**x - 1, accurate near zero. Complex elements use
// Exp(x)-1.
func (a *Array[T]) Expm1() *Array[T] { return mapUnary(a, math.Expm1, expm1c) }

// Sin returns the element-wise sine.
func (a *Array[T]) Sin() *Array[T] { return mapUnary(a, math.Sin, cmplx.Sin) }

// Cos returns the element-wise cosine.
func (a *Array[T]) Cos() *Array[T] { return mapUnary(a, math.Cos, cmplx.Cos) }

// Tan returns the element-wise tangent.
func (a *Array[T]) Tan() *Array[T] { return mapUnary(a, math.Tan, cmplx.Tan) }

// Asin returns the element-wise arcsine.
func (a *Array[T]) Asin() *Array[T] { return mapUnary(a, math.Asin, cmplx.Asin) }

// Acos returns the element-wise arccosine.
func (a *Array[T]) Acos() *Array[T] { return mapUnary(a, math.Acos, cmplx.Acos) }

// Atan returns the element-wise arctangent.
func (a *Array[T]) Atan() *Array[T] { return mapUnary(a, math.Atan, cmplx.Atan) }

// Sinh returns the element-wise hyperbolic sine.
func (a *Array[T]) Sinh() *Array[T] { return mapUnary(a, math.Sinh, cmplx.Sinh) }

// Cosh returns the element-wise hyperbolic cosine.
func (a *Array[T]) Cosh() *Array[T] { return mapUnary(a, math.Cosh, cmplx.Cosh) }

// Tanh returns the element-wise hyperbolic tangent.
func (a *Array[T]) Tanh() *Array[T] { return mapUnary(a, math.Tanh, cmplx.Tanh) }

// Asinh returns the element-wise inverse hyperbolic sine.
func (a *Array[T]) Asinh() *Array[T] { return mapUnary(a, math.Asinh, cmplx.Asinh) }

// Acosh returns the element-wise inverse hyperbolic cosine.
func (a *Array[T]) Acosh() *Array[T] { return mapUnary(a, math.Acosh, cmplx.Acosh) }

// Atanh returns the element-wise inverse hyperbolic tangent.
func (a *Array[T]) Atanh() *Array[T] { return mapUnary(a, math.Atanh, cmplx.Atanh) }

// Floor returns the element-wise floor, per-component on complex
// elements (numpy applies to both parts).
func (a *Array[T]) Floor() *Array[T] { return mapUnary(a, math.Floor, perComponent(math.Floor)) }

// Ceil returns the element-wise ceiling, per-component on complex
// elements.
func (a *Array[T]) Ceil() *Array[T] { return mapUnary(a, math.Ceil, perComponent(math.Ceil)) }

// Round returns the element-wise round-half-to-even (numpy's round),
// per-component on complex elements.
func (a *Array[T]) Round() *Array[T] {
	return mapUnary(a, math.RoundToEven, perComponent(math.RoundToEven))
}

// Trunc returns the element-wise integer part, per-component on complex
// elements.
func (a *Array[T]) Trunc() *Array[T] { return mapUnary(a, math.Trunc, perComponent(math.Trunc)) }

// Conj returns the element-wise complex conjugate; a real array is
// copied unchanged.
func (a *Array[T]) Conj() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return v }, cmplx.Conj)
}

// Sign returns the element-wise sign: for real x, -1 if x < 0, +1 if x > 0,
// and x itself if x is ±0 or NaN (numpy's np.sign). Complex elements get
// z/|z| (NaN if any component is NaN).
func (a *Array[T]) Sign() *Array[T] {
	return mapUnary(a, func(v float64) float64 {
		if v != v {
			return math.NaN()
		}
		if v > 0 {
			return 1
		}
		if v < 0 {
			return -1
		}
		return v
	}, func(z complex128) complex128 {
		m := cmplx.Abs(z)
		if math.IsNaN(real(z)) || math.IsNaN(imag(z)) || m == 0 {
			return z
		}
		return z / complex(m, 0)
	})
}

// Signbit returns true where the real sign bit is set (math.Signbit),
// as 1 or 0; on complex elements it applies to each component.
func (a *Array[T]) Signbit() *Array[T] {
	return mapUnary(a, sbit, perComponent(sbit))
}

func sbit(v float64) float64 {
	if math.Signbit(v) {
		return 1
	}
	return 0
}

// Square returns the element-wise square x*x.
func (a *Array[T]) Square() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return v * v },
		func(z complex128) complex128 { return z * z })
}

// Reciprocal returns the element-wise reciprocal 1/x (IEEE: 1/0 is +Inf,
// 1/-0 is -Inf).
func (a *Array[T]) Reciprocal() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return 1 / v },
		func(z complex128) complex128 { return 1 / z })
}

// Fabs returns the element-wise absolute value of the real input
// (numpy's np.fabs, which does not accept complex; output type is T).
func (a *Array[T]) Fabs() *Array[T] {
	if isComplexKind[T]() {
		panicComplexAbs()
	}
	return mapUnary(a, math.Abs, nil)
}

// Fix returns the element-wise round toward zero (numpy's np.fix),
// per-component on complex elements.
func (a *Array[T]) Fix() *Array[T] { return mapUnary(a, math.Trunc, perComponent(math.Trunc)) }

// Deg2Rad converts angles from degrees to radians, element-wise (×π/180).
func (a *Array[T]) Deg2Rad() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return v * (math.Pi / 180) },
		func(z complex128) complex128 { return z * complex(math.Pi/180, 0) })
}

// Rad2Deg converts angles from radians to degrees, element-wise (×180/π).
func (a *Array[T]) Rad2Deg() *Array[T] {
	return mapUnary(a, func(v float64) float64 { return v * (180 / math.Pi) },
		func(z complex128) complex128 { return z * complex(180/math.Pi, 0) })
}

// Angle returns the element-wise argument in radians in [-π, π] (atan2 of
// the imaginary over the real part; real elements are non-negative with
// angle 0), or degrees when deg is true (numpy's np.angle).
func (a *Array[T]) Angle(deg bool) *Array[T] {
	return mapUnary(a,
		func(v float64) float64 { return 0 },
		func(z complex128) complex128 {
			r := math.Atan2(imag(z), real(z))
			if deg {
				r *= 180 / math.Pi
			}
			return complex(r, 0)
		})
}

// Sinc returns sin(πx)/(πx), normalized so Sinc(0) = 1 (numpy's np.sinc);
// complex elements use sin(πz)/(πz).
func (a *Array[T]) Sinc() *Array[T] {
	return mapUnary(a, sincc, sincz)
}

func sincc(v float64) float64 {
	if v == 0 {
		return 1
	}
	return math.Sin(math.Pi*v) / (math.Pi * v)
}

func sincz(z complex128) complex128 {
	if z == 0 {
		return 1
	}
	p := complex(math.Pi, 0) * z
	return cmplx.Sin(p) / p
}

// Heaviside returns the element-wise step: 0 below zero, 1 above, and
// second at zero (numpy's np.heaviside); real semantics per-component on
// complex elements.
func (a *Array[T]) Heaviside(second T) *Array[T] {
	sre, _ := parts(second)
	return mapUnary(a, heaviside(sre), perComponent(heaviside(sre)))
}

func heaviside(sec float64) func(float64) float64 {
	return func(v float64) float64 {
		if v > 0 {
			return 1
		}
		if v < 0 {
			return 0
		}
		return sec
	}
}
