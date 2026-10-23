package numgo

import (
	"slices"
)

// Pure: Take gathers the elements of a at indices along axis (negative indices
// count from the end, repeats allowed). The result shape is a's with the axis
// dimension replaced by len(indices); the result is new data.
func (a *Array[T]) Take(indices []int, axis int) *Array[T] {
	ax := a.resolveAxis(axis)
	norm := normalizeIndices(indices, ax, a.shape[ax])
	shape := slices.Clone(a.shape)
	shape[ax] = len(indices)
	return FromFunction(shape, func(coords ...int) T {
		src := slices.Clone(coords)
		src[ax] = norm[coords[ax]]
		return a.At(src...)
	})
}

// Into: Put mutates the receiver by writing values at the given indices along
// axis (negative indices count from the end, repeats allowed). values broadcast
// against the gathered shape (a's shape with the axis dimension replaced by
// len(indices)). Returns the receiver for chaining.
func (a *Array[T]) Put(indices []int, axis int, values *Array[T]) *Array[T] {
	ax := a.resolveAxis(axis)
	norm := normalizeIndices(indices, ax, a.shape[ax])
	gathered := slices.Clone(a.shape)
	gathered[ax] = len(indices)
	dst := Zeros[T](gathered...)
	// CopyInto broadcasts values against gathered, panicking on mismatch.
	CopyInto(dst, values)
	coords := make([]int, len(gathered))
	for i := 0; i < dst.Size(); i++ {
		src := slices.Clone(coords)
		src[ax] = norm[coords[ax]]
		a.Set(src, dst.data[i])
		for k := len(coords) - 1; k >= 0; k-- {
			coords[k]++
			if coords[k] < gathered[k] {
				break
			}
			coords[k] = 0
		}
	}
	return a
}

// Pure: Compress selects the entries of a along axis where mask is true. The
// result shape is a's with the axis dimension replaced by the number of true
// entries (possibly zero); the result is new data.
func (a *Array[T]) Compress(mask []bool, axis int) *Array[T] {
	ax := a.resolveAxis(axis)
	if len(mask) != a.shape[ax] {
		panicCompressMask(len(mask), a.shape[ax])
	}
	keep := make([]int, 0, len(mask))
	for i, m := range mask {
		if m {
			keep = append(keep, i)
		}
	}
	return a.Take(keep, ax)
}

// Into: SetMask mutates the receiver by assigning v to every element whose
// flattened C-order position has mask true (numpy a[mask] = v flat semantics).
// Returns the receiver for chaining.
func (a *Array[T]) SetMask(mask []bool, v T) *Array[T] {
	if len(mask) != a.Size() {
		panicCompressMask(len(mask), a.Size())
	}
	coords := make([]int, len(a.shape))
	for _, m := range mask {
		if m {
			a.Set(coords, v)
		}
		for k := len(coords) - 1; k >= 0; k-- {
			coords[k]++
			if coords[k] < a.shape[k] {
				break
			}
			coords[k] = 0
		}
	}
	return a
}

// normalizeIndices resolves negative indices against dim, panicking via
// panicTakeIndex on out-of-bounds entries.
func normalizeIndices(indices []int, axis, dim int) []int {
	norm := make([]int, len(indices))
	for i, idx := range indices {
		if idx < 0 {
			idx += dim
		}
		if idx < 0 || idx >= dim {
			panicTakeIndex(indices[i], axis, dim)
		}
		norm[i] = idx
	}
	return norm
}
