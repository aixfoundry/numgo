package numgo

import "slices"

// Diff returns the first-order differences of a along axis (negative
// allowed): out[i...] = a[i+1...] - a[i...], so the axis dimension shrinks
// by one. Higher-order differences are not provided; chain calls if needed.
func (a *Array[T]) Diff(axis int) *Array[T] {
	ax := a.resolveAxis(axis)
	shape := slices.Clone(a.shape)
	shape[ax]--
	if shape[ax] < 0 {
		shape[ax] = 0
	}
	return FromFunction(shape, func(coords ...int) T {
		src := slices.Clone(coords)
		src[ax]++
		return a.At(src...) - a.At(coords...)
	})
}

// Ediff1d returns the first differences of a flattened in C order,
// equivalent to a.Ravel().Diff(0).
func (a *Array[T]) Ediff1d() *Array[T] {
	return a.Ravel().Diff(0)
}
