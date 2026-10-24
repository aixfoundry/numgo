package numgo

import (
	"reflect"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// Pure: AsType returns a new array with every element converted to P. Real values
// converted to a complex type get a zero imaginary part; a complex value
// converted to a real type drops its imaginary part.
//
// The conversion dispatches once per call on the concrete slice type and runs
// a typed copy loop (compiler-vectorizable); named types under Number keep the
// generic parts/scalarFromParts path. Same conversions as the generic path —
// bit-identical by construction.
func AsType[F, P Number](a *Array[F]) *Array[P] {
	out := newArray(a.Shape(), nditer.COrderStrides(a.shape), 0, make([]P, a.Size()))
	if out.Size() == 0 {
		return out
	}
	if a.IsContiguous() && convertTyped(out.data, a.Data()[:a.Size()]) {
		return out
	}
	src := make([]P, len(a.data))
	for i, v := range a.data {
		re, im := parts(v)
		src[i] = scalarFromParts[P](re, im)
	}
	nditer.Zip(out.layout(), out.data, []*nditer.Layout{a.layout()}, [][]P{src},
		func(d []P, src [][]P) {
			for i := range d {
				d[i] = src[0][i]
			}
		})
	return out
}

// convertTyped fills dst from src with the plain per-type conversion when both
// concrete types match one of the built-in Number cases; it reports whether it
// handled the pair (named types fall back to the reflect path).
func convertTyped[F, P Number](dst []P, src []F) bool {
	switch s := any(src).(type) {
	case []float64:
		switch d := any(dst).(type) {
		case []float64:
			copy(d, s)
		case []float32:
			for i, v := range s {
				d[i] = float32(v)
			}
		case []complex64:
			for i, v := range s {
				d[i] = complex64(complex(v, 0))
			}
		case []complex128:
			for i, v := range s {
				d[i] = complex(v, 0)
			}
		default:
			return false
		}
	case []float32:
		switch d := any(dst).(type) {
		case []float64:
			for i, v := range s {
				d[i] = float64(v)
			}
		case []float32:
			copy(d, s)
		case []complex64:
			for i, v := range s {
				d[i] = complex64(complex(v, 0))
			}
		case []complex128:
			for i, v := range s {
				d[i] = complex128(complex(v, 0))
			}
		default:
			return false
		}
	case []complex128:
		switch d := any(dst).(type) {
		case []float64:
			for i, v := range s {
				d[i] = real(v)
			}
		case []float32:
			for i, v := range s {
				d[i] = float32(real(v))
			}
		case []complex64:
			for i, v := range s {
				d[i] = complex64(v)
			}
		case []complex128:
			copy(d, s)
		default:
			return false
		}
	case []complex64:
		switch d := any(dst).(type) {
		case []float64:
			for i, v := range s {
				d[i] = float64(real(v))
			}
		case []float32:
			for i, v := range s {
				d[i] = real(v)
			}
		case []complex64:
			copy(d, s)
		case []complex128:
			for i, v := range s {
				d[i] = complex128(v)
			}
		default:
			return false
		}
	default:
		return false
	}
	return true
}

// Pure: AsArray returns a new array holding a copy of data, shaped as given. It is
// numpy's asarray naming for FromSlice.
func AsArray[T Number](data []T, shape ...int) *Array[T] {
	return FromSlice(data, shape...)
}

// parts splits any Number into real and imaginary float64 components. reflect
// handles named types in the constraint's type set.
//
// shortcut: per-element reflect (~30ns/elem), fine until profiling says
// otherwise; upgrade path is a one-time type switch dispatching to typed loops.
func parts[T Number](v T) (re, im float64) {
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Complex64 || rv.Kind() == reflect.Complex128 {
		c := rv.Complex()
		return real(c), imag(c)
	}
	return rv.Float(), 0
}

// scalarFromParts builds a Number from float64 components.
func scalarFromParts[T Number](re, im float64) T {
	var out T
	rv := reflect.ValueOf(&out).Elem()
	if rv.Kind() == reflect.Complex64 || rv.Kind() == reflect.Complex128 {
		rv.SetComplex(complex(re, im))
	} else {
		rv.SetFloat(re)
	}
	return out
}
