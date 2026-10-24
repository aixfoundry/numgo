package ma

import (
	"slices"

	"github.com/kaisawind/numgo"
)

// Shape operations: pure ops that rebuild the mask for the new shape. There
// are deliberately no zero-copy views this batch (Masked docs). Concatenate,
// Stack, Transpose, Reshape and Squeeze are implemented; the rest of numpy
// ma's shape family (Split, Tile, …) is pending.

// Pure: Concatenate joins masked arrays along axis (negative allowed); masks
// merge by concatenation.
func Concatenate[T numgo.Number](axis int, as ...*Masked[T]) *Masked[T] {
	if len(as) == 0 {
		panic("numgo: need at least one array to concatenate")
	}
	data := make([]*numgo.Array[T], len(as))
	for i, m := range as {
		data[i] = m.a
	}
	out := numgo.Concatenate(axis, data...)
	ax := axis
	if ax < 0 {
		ax += out.Ndim()
	}
	mask := make([]bool, out.Size())
	outShape := out.Shape()
	walk(outShape, func(coords []int) {
		// Find which input this coordinate came from along the axis.
		off := 0
		for _, m := range as {
			n := m.Shape()[ax]
			if coords[ax] < off+n {
				src := slices.Clone(coords)
				src[ax] = coords[ax] - off
				mask[flat(outShape, coords)] = m.mask[flat(m.Shape(), src)]
				return
			}
			off += n
		}
	})
	return newMasked(out, mask)
}

// Pure: Stack joins same-shaped masked arrays along a new axis (negative
// allowed, ndim+1 positions valid).
func Stack[T numgo.Number](axis int, as ...*Masked[T]) *Masked[T] {
	if len(as) == 0 {
		panic("numgo: need at least one array to stack")
	}
	data := make([]*numgo.Array[T], len(as))
	for i, m := range as {
		data[i] = m.a
	}
	out := numgo.Stack(axis, data...)
	ins := axis
	if ins < 0 {
		ins += out.Ndim()
	}
	mask := make([]bool, out.Size())
	outShape := out.Shape()
	walk(outShape, func(coords []int) {
		k := coords[ins]
		src := slices.Delete(slices.Clone(coords), ins, ins+1)
		mask[flat(outShape, coords)] = as[k].mask[flat(as[k].Shape(), src)]
	})
	return newMasked(out, mask)
}

// Pure: Transpose permutes axes (no axes = reverse); the mask permutes with
// the data.
func Transpose[T numgo.Number](m *Masked[T], axes ...int) *Masked[T] {
	out := m.a.Transpose(axes...)
	ax := axes
	if len(ax) == 0 {
		ax = make([]int, m.Ndim())
		for i := range ax {
			ax[i] = m.Ndim() - 1 - i
		}
	}
	inv := make([]int, len(ax))
	for i, a := range ax {
		inv[a] = i
	}
	mask := make([]bool, m.Size())
	outShape := out.Shape()
	srcShape := m.Shape()
	walk(outShape, func(coords []int) {
		src := make([]int, len(coords))
		for i := range src {
			src[i] = coords[inv[i]]
		}
		mask[flat(outShape, coords)] = m.mask[flat(srcShape, src)]
	})
	return newMasked(out, mask)
}

// Pure: Reshape re-shapes (at most one -1 dimension); the mask re-indexes in
// C order of the new shape. Masks are materialized in C order of the logical
// shape, so for contiguous data the mask carries over verbatim.
func Reshape[T numgo.Number](m *Masked[T], shape ...int) *Masked[T] {
	out := m.a.Reshape(shape...)
	if m.a.IsContiguous() {
		return newMasked(out, slicesClone(m.mask))
	}
	// Non-contiguous data: Reshape copied it C-order first, so both source
	// and result enumerate the same logical sequence — the mask carries over.
	return newMasked(out, slicesClone(m.mask))
}

// Pure: Squeeze drops size-1 dimensions (all when no axes given); the mask
// re-indexes in C order.
func Squeeze[T numgo.Number](m *Masked[T], axes ...int) *Masked[T] {
	out := m.a.Squeeze(axes...)
	mask := make([]bool, out.Size())
	outShape := out.Shape()
	fullShape := m.Shape()
	squeezed := make([]bool, len(fullShape))
	if len(axes) == 0 {
		for i, d := range fullShape {
			squeezed[i] = d == 1
		}
	} else {
		for _, a := range axes {
			a = ((a % len(fullShape)) + len(fullShape)) % len(fullShape)
			squeezed[a] = true
		}
	}
	walk(outShape, func(coords []int) {
		src := make([]int, len(fullShape))
		k := 0
		for i := range fullShape {
			if squeezed[i] {
				src[i] = 0
			} else {
				src[i] = coords[k]
				k++
			}
		}
		mask[flat(outShape, coords)] = m.mask[flat(fullShape, src)]
	})
	return newMasked(out, mask)
}
