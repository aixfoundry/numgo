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

// gatherIndices normalizes integer fancy-index lists against a's leading axes.
// Each list is a 1-d index array; the lists broadcast to one common length k
// (len-1 lists repeat, unequal non-1 lengths panic). Returns the per-list
// resolved index arrays (negatives folded, bounds checked with the list's axis
// position) and k.
func (a *Array[T]) gatherIndices(idx [][]int) (norm [][]int, k int) {
	if len(idx) > len(a.shape) {
		panicRanges(len(idx), len(a.shape))
	}
	k = 1
	for _, ix := range idx {
		if len(ix) == 1 {
			continue
		}
		if k != 1 && len(ix) != k {
			panicBroadcast([]int{len(ix)}, []int{k})
		}
		k = len(ix)
	}
	norm = make([][]int, len(idx))
	for j, ix := range idx {
		norm[j] = normalizeIndices(ix, j, a.shape[j])
	}
	return norm, k
}

// fancyCoords maps a gathered-position coordinate (broadcast dim first, tail
// axes after) to the source coordinates implied by per-axis index arrays norm.
func fancyCoords(norm [][]int, k int, coords []int) []int {
	src := make([]int, len(norm)+len(coords)-1)
	for j, ix := range norm {
		src[j] = ix[0]
		if len(ix) == k {
			src[j] = ix[coords[0]]
		}
	}
	copy(src[len(norm):], coords[1:])
	return src
}

// Pure: Index gathers elements of a by integer fancy indexing on the leading
// axes: each list is a 1-d index array, the lists broadcast to one common
// length k, and the result shape is (k,) ++ a's trailing axes. Negative
// indices count from the end, repeats allowed; the result is new data. With no
// lists Index returns a Copy.
func (a *Array[T]) Index(idx ...[]int) *Array[T] {
	if len(idx) == 0 {
		return a.Copy()
	}
	norm, k := a.gatherIndices(idx)
	shape := append([]int{k}, a.shape[len(norm):]...)
	return FromFunction(shape, func(coords ...int) T {
		return a.At(fancyCoords(norm, k, coords)...)
	})
}

// Into: PutIndices mutates the receiver by assigning values at the positions
// selected by integer fancy indexing (same rules as Index): values are
// broadcast against the gathered shape. Returns the receiver for chaining.
func (a *Array[T]) PutIndices(values *Array[T], idx ...[]int) *Array[T] {
	norm, k := a.gatherIndices(idx)
	gathered := append([]int{k}, a.shape[len(norm):]...)
	dst := Zeros[T](gathered...)
	// CopyInto broadcasts values against gathered, panicking on mismatch.
	CopyInto(dst, values)
	coords := make([]int, len(gathered))
	for i := 0; i < dst.Size(); i++ {
		a.Set(fancyCoords(norm, k, coords), dst.data[i])
		for j := len(coords) - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < gathered[j] {
				break
			}
			coords[j] = 0
		}
	}
	return a
}

// Pure: MaskSelect selects every element whose flattened C-order position has
// mask true (numpy a[mask] for the whole array). The result is a new 1-d array.
func (a *Array[T]) MaskSelect(mask []bool) *Array[T] {
	if len(mask) != a.Size() {
		panicCompressMask(len(mask), a.Size())
	}
	vals := make([]T, 0, a.Size())
	coords := make([]int, len(a.shape))
	for _, m := range mask {
		if m {
			vals = append(vals, a.At(coords...))
		}
		for k := len(coords) - 1; k >= 0; k-- {
			coords[k]++
			if coords[k] < a.shape[k] {
				break
			}
			coords[k] = 0
		}
	}
	return New1DArray(vals)
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
