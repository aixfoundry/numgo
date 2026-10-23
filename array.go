package numgo

import (
	"slices"

	"github.com/kaisawind/numgo/internal/nditer"
)

// Number is the element type constraint for Array. Integers are excluded:
// Mean/Div would silently truncate without numpy-style dtype promotion.
type Number interface {
	~float32 | ~float64 | ~complex64 | ~complex128
}

// Array is an N-dimensional strided view over a flat data slice. Strides are in
// element units. Views share data; use Copy for an independent array.
type Array[T Number] struct {
	shape, strides []int
	offset         int
	data           []T
}

// newArray wraps existing data without copying. Constructors and views are the
// only callers.
func newArray[T Number](shape, strides []int, offset int, data []T) *Array[T] {
	return &Array[T]{shape: shape, strides: strides, offset: offset, data: data}
}

// layout projects the array into the nditer engine's dtype-independent form.
func (a *Array[T]) layout() *nditer.Layout {
	return &nditer.Layout{Shape: a.shape, Strides: a.strides, Offset: a.offset}
}

// Pure: Shape returns a copy of the array's dimensions.
func (a *Array[T]) Shape() []int {
	return slices.Clone(a.shape)
}

// Pure: Ndim returns the number of dimensions.
func (a *Array[T]) Ndim() int {
	return len(a.shape)
}

// Pure: Size returns the total number of elements.
func (a *Array[T]) Size() int {
	return nditer.Size(a.shape)
}

// Pure: IsContiguous reports whether the array is laid out in C order.
func (a *Array[T]) IsContiguous() bool {
	return nditer.IsContiguous(a.layout())
}

// Pure: Copy returns an independent, C-order compact copy of the array.
func (a *Array[T]) Copy() *Array[T] {
	shape := slices.Clone(a.shape)
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{a.layout()}, [][]T{a.data},
		func(d []T, src [][]T) {
			d[0] = src[0][0]
		})
	return dst
}

// View: Reshape views the array with a new shape when the data is contiguous; a
// non-contiguous array is copied first, so the result may not share data.
// At most one dimension may be -1 and is inferred.
func (a *Array[T]) Reshape(shape ...int) *Array[T] {
	if !a.IsContiguous() {
		return a.Copy().Reshape(shape...)
	}
	size := a.Size()
	unknown := -1
	for i, d := range shape {
		if d < 0 {
			if d != -1 {
				panicReshape(size, shape)
			}
			if unknown >= 0 {
				panicReshapeUnknown()
			}
			unknown = i
		}
	}
	if unknown >= 0 {
		known := 1
		for i, d := range shape {
			if i != unknown {
				known *= d
			}
		}
		if known == 0 || size%known != 0 {
			panicReshape(size, shape)
		}
		shape[unknown] = size / known
	} else if nditer.Size(shape) != size {
		panicReshape(size, shape)
	}
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), a.offset, a.data)
}

// View: Transpose views the array with its axes permuted. With no axes it reverses
// them (numpy .T). The result shares data.
func (a *Array[T]) Transpose(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = make([]int, len(a.shape))
		for i := range axes {
			axes[i] = len(axes) - 1 - i
		}
	}
	if len(axes) != len(a.shape) {
		panicAxes()
	}
	seen := make([]bool, len(a.shape))
	for _, ax := range axes {
		if ax < 0 || ax >= len(a.shape) {
			panicAxis(ax, len(a.shape))
		}
		if seen[ax] {
			panicAxes()
		}
		seen[ax] = true
	}
	l := nditer.Permute(a.layout(), axes)
	return newArray(l.Shape, l.Strides, l.Offset, a.data)
}

// View: Slice views a strided sub-array. One Range per dimension is required; the
// result shares data. Bounds follow numpy: negative indices count from the end,
// a zero Range means the whole axis, and a negative Step reverses it.
func (a *Array[T]) Slice(rs ...Range) *Array[T] {
	if len(rs) != len(a.shape) {
		panicRanges(len(rs), len(a.shape))
	}
	starts := make([]int, len(rs))
	steps := make([]int, len(rs))
	shape := make([]int, len(rs))
	for i, r := range rs {
		starts[i], shape[i], steps[i] = normalizeRange(r, a.shape[i])
	}
	l := nditer.Slice(a.layout(), starts, steps, shape)
	return newArray(l.Shape, l.Strides, l.Offset, a.data)
}
