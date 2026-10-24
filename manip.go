package numgo

import (
	"slices"

	"github.com/kaisawind/numgo/internal/nditer"
)

// Into: CopyInto mutates dst by filling it with src broadcast to dst's shape.
// This is the Into-style mutation primitive later operations build on.
func CopyInto[T Number](dst, src *Array[T]) {
	if dst.Size() == 0 {
		return
	}
	l := src.layout()
	if !slices.Equal(src.shape, dst.shape) {
		shape, strides, err := nditer.Broadcast(src.shape, dst.shape)
		if err != nil {
			panicBroadcast(src.shape, dst.shape)
		}
		if !slices.Equal(shape, dst.shape) {
			panicInto(dst.shape, shape)
		}
		l = &nditer.Layout{Shape: shape, Strides: strides[0]}
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{l}, [][]T{src.data},
		func(d []T, s [][]T) {
			for i := range d {
				d[i] = s[0][i]
			}
		})
}

// Pure: Concatenate joins arrays along axis (negative allowed). All dimensions
// except axis must match exactly. The result is new data.
func Concatenate[T Number](axis int, as ...*Array[T]) *Array[T] {
	if len(as) == 0 {
		panicNoArrays()
	}
	ax := as[0].resolveAxis(axis)
	for _, a := range as {
		if a.Ndim() != as[0].Ndim() {
			panicConcatenateShapes()
		}
		for i, d := range a.shape {
			if i != ax && d != as[0].shape[i] {
				panicConcatenateShapes()
			}
		}
	}
	shape := slices.Clone(as[0].shape)
	for _, a := range as[1:] {
		shape[ax] += a.shape[ax]
	}
	out := Zeros[T](shape...)
	off := 0
	for _, a := range as {
		rs := make([]Range, out.Ndim())
		rs[ax] = Range{Start: off, Stop: off + a.shape[ax]}
		CopyInto(out.Slice(rs...), a)
		off += a.shape[ax]
	}
	return out
}

// Pure: Stack joins same-shaped arrays along a new axis (negative allowed,
// ndim+1 positions valid). The result is new data.
func Stack[T Number](axis int, as ...*Array[T]) *Array[T] {
	if len(as) == 0 {
		panicNoArrays()
	}
	for _, a := range as {
		if !slices.Equal(a.shape, as[0].shape) {
			panicStackShapes()
		}
	}
	ins := axis
	if ins < 0 {
		ins += len(as[0].shape) + 1
	}
	if ins < 0 || ins > as[0].Ndim() {
		panicAxis(axis, as[0].Ndim())
	}
	outShape := slices.Insert(slices.Clone(as[0].shape), ins, len(as))
	out := Zeros[T](outShape...)
	for k, a := range as {
		CopyInto(out.Slice(axisRange(out.Ndim(), ins, k)...), a.ExpandDims(ins))
	}
	return out
}

// axisRange builds one Slice argument list with coordinate k fixed on axis ax
// and every other axis whole (a zero Range means the whole axis).
func axisRange(ndim, ax, k int) []Range {
	rs := make([]Range, ndim)
	rs[ax] = Range{Start: k, Stop: k + 1}
	return rs
}

// View: Split slices a at the given ascending boundaries along axis into
// len(indices)+1 views sharing data with a. Indices may equal the axis size.
func Split[T Number](a *Array[T], indices []int, axis int) []*Array[T] {
	ax := a.resolveAxis(axis)
	for i, at := range indices {
		if at < 0 || at > a.shape[ax] || (i > 0 && at <= indices[i-1]) {
			panicSplitIndices()
		}
	}
	bounds := append([]int{0}, append(slices.Clone(indices), a.shape[ax])...)
	out := make([]*Array[T], len(bounds)-1)
	for i := range out {
		rs := make([]Range, a.Ndim())
		rs[ax] = Range{Start: bounds[i], Stop: bounds[i+1]}
		out[i] = a.Slice(rs...)
	}
	return out
}

// Pure: Tile repeats a along each axis by reps, right-aligned like numpy.tile
// (shorter reps are padded with leading ones; a longer reps list prepends
// size-1 axes to a). The result is new data.
func Tile[T Number](a *Array[T], reps ...int) *Array[T] {
	for _, r := range reps {
		if r < 0 {
			panicRepeat(r)
		}
	}
	base := make([]int, 0, max(len(reps), a.Ndim()))
	for i := 0; i < len(reps)-a.Ndim(); i++ {
		base = append(base, 1) // implicit leading axes have size 1
	}
	base = append(base, a.shape...)
	if len(reps) < a.Ndim() {
		pad := make([]int, a.Ndim()-len(reps))
		for i := range pad {
			pad[i] = 1
		}
		reps = append(pad, reps...)
	}
	shape := make([]int, len(reps))
	for i := range shape {
		shape[i] = base[i] * reps[i]
	}
	lead := len(shape) - a.Ndim()
	return FromFunction(shape, func(coords ...int) T {
		src := make([]int, a.Ndim())
		for i := range src {
			src[i] = coords[lead+i] % a.shape[i]
		}
		return a.At(src...)
	})
}

// Pure: Repeat repeats each element count times along axis (negative allowed).
// The result is new data.
func Repeat[T Number](a *Array[T], count int, axis int) *Array[T] {
	if count < 0 {
		panicRepeat(count)
	}
	ax := a.resolveAxis(axis)
	shape := slices.Clone(a.shape)
	shape[ax] *= count
	return FromFunction(shape, func(coords ...int) T {
		src := slices.Clone(coords)
		src[ax] /= count
		return a.At(src...)
	})
}

// Pure: Flip reverses the order of elements along axes (default: all axes). The
// result is new data; a negative-stride view is deliberately not attempted.
func Flip[T Number](a *Array[T], axes ...int) *Array[T] {
	rev := make([]bool, a.Ndim())
	if len(axes) == 0 {
		for i := range rev {
			rev[i] = true
		}
	}
	for _, ax := range axes {
		rev[a.resolveAxis(ax)] = true
	}
	return FromFunction(slices.Clone(a.shape), func(coords ...int) T {
		src := slices.Clone(coords)
		for i, r := range rev {
			if r {
				src[i] = a.shape[i] - 1 - src[i]
			}
		}
		return a.At(src...)
	})
}

// Pure: Roll shifts elements by shift along axes (default: the flattened array
// in C order). Negative shifts roll backwards, wrapping around. The result is
// new data.
func Roll[T Number](a *Array[T], shift int, axes ...int) *Array[T] {
	out := Zeros[T](a.shape...)
	if out.Size() == 0 {
		return out
	}
	if len(axes) == 0 {
		flat := a
		if !flat.IsContiguous() {
			flat = flat.Copy()
		}
		f := flat.Reshape(flat.Size())
		n := f.Size()
		s := ((shift % n) + n) % n
		for i := range out.data {
			out.data[i] = f.At((i - s + n) % n)
		}
		return out
	}
	shifts := make([]int, a.Ndim())
	for _, ax := range axes {
		shifts[a.resolveAxis(ax)] = shift
	}
	return FromFunction(slices.Clone(a.shape), func(coords ...int) T {
		src := slices.Clone(coords)
		for i, s := range shifts {
			if s != 0 {
				n := a.shape[i]
				src[i] = ((coords[i]-s)%n + n) % n
			}
		}
		return a.At(src...)
	})
}
