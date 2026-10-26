package numgo

import (
	"reflect"
	"slices"
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

// Set operations over flattened inputs (numpy union1d / intersect1d /
// setdiff1d / setxor1d). All results are sorted ascending with Unique's
// total order (NaN last, complex by real then imag). NaN deduplicates within
// an input via Unique but never matches across inputs (numpy's sort-based
// isin treats NaN as unequal to every value, including another NaN) —
// verified against numpy 2.5.1: intersect1d([nan],[nan]) is empty,
// setdiff1d([nan,1],[nan]) keeps the NaN, setxor1d([nan,1],[nan,4]) keeps
// both NaNs. numpy's assume_unique flag is not exposed: inputs are always
// fully deduplicated.

// Union1d returns the sorted unique union of the flattened a and b
// (numpy union1d). The union of two empties is empty.
func Union1d[T Number](a, b *Array[T]) *Array[T] {
	return Concatenate[T](0, a.Ravel(), b.Ravel()).Unique()
}

// inBFlags reports, for each value of vals, whether it occurs in the
// flattened b. NaN never matches (numpy isin semantics), so the bare
// cmpNumber equality of IsIn must be masked for NaN inputs.
func inBFlags[T Number](vals []T, b *Array[T]) []bool {
	flags := b.Ravel().IsIn(New1DArray(vals), false)
	for i, v := range vals {
		if isNaN(v) {
			flags[i] = false // numpy isin: NaN matches nothing
		}
	}
	return flags
}

// Intersect1d returns the sorted unique values present in both flattened a
// and b (numpy intersect1d). NaN is in the intersection only if numpy says
// so — it never is, since isin(NaN) is always false.
func Intersect1d[T Number](a, b *Array[T]) *Array[T] {
	vals := a.Ravel().Unique().Data()
	inB := inBFlags(vals, b)
	out := make([]T, 0, len(vals))
	for i, v := range vals {
		if inB[i] {
			out = append(out, v)
		}
	}
	return New1DArray(out)
}

// SetDiff1d returns the sorted unique values of flattened a that are not in
// flattened b (numpy setdiff1d). a's NaN is always kept: NaN never matches
// anything in b, per isin.
func SetDiff1d[T Number](a, b *Array[T]) *Array[T] {
	vals := a.Ravel().Unique().Data()
	inB := inBFlags(vals, b)
	out := make([]T, 0, len(vals))
	for i, v := range vals {
		if !inB[i] {
			out = append(out, v)
		}
	}
	return New1DArray(out)
}

// SetXor1d returns the sorted unique values present in exactly one of the
// flattened a and b (numpy setxor1d). Each input keeps its own NaN, since
// NaN never matches across the two inputs.
func SetXor1d[T Number](a, b *Array[T]) *Array[T] {
	valsA := a.Ravel().Unique().Data()
	inB := inBFlags(valsA, b)
	valsB := b.Ravel().Unique().Data()
	inA := inBFlags(valsB, a)
	out := make([]T, 0, len(valsA)+len(valsB))
	for i, v := range valsA {
		if !inB[i] {
			out = append(out, v)
		}
	}
	for i, v := range valsB {
		if !inA[i] {
			out = append(out, v)
		}
	}
	slices.SortFunc(out, cmpNumber[T])
	return New1DArray(out)
}
