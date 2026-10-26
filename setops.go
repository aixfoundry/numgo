package numgo

import (
	"reflect"
)

// isFloat32Kind reports whether T is float32, for dtype-sized infinities.
func isFloat32Kind[T Number]() bool {
	return reflect.TypeOf(*new(T)).Kind() == reflect.Float32
}

// IsIn tests every element of testElements against a's flattened unique
// values (numpy isin, Unique-based) and returns a []bool in C order of
// testElements. NaN only matches NaN. invert swaps the membership test.
// The result is a plain []bool because bool is not a Number element type.
func (a *Array[T]) IsIn(testElements *Array[T], invert bool) []bool {
	vals := a.Unique().Data()
	out := make([]bool, testElements.Size())
	d := testElements.Ravel().Data()
	for i, v := range d {
		found := false
		for _, u := range vals {
			if cmpNumber[T](u, v) == 0 {
				found = true
				break
			}
		}
		out[i] = found != invert
	}
	return out
}
