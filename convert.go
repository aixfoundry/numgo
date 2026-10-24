package numgo

import (
	"reflect"

	"github.com/kaisawind/numgo/internal/nditer"
)

// Pure: AsType returns a new array with every element converted to P. Real values
// converted to a complex type get a zero imaginary part; a complex value
// converted to a real type drops its imaginary part.
func AsType[F, P Number](a *Array[F]) *Array[P] {
	out := newArray(a.Shape(), nditer.COrderStrides(a.shape), 0, make([]P, a.Size()))
	if out.Size() == 0 {
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
