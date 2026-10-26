package numgo

import (
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
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

// Shape returns a copy of the array's dimensions.
func (a *Array[T]) Shape() []int {
	return slices.Clone(a.shape)
}

// Ndim returns the number of dimensions.
func (a *Array[T]) Ndim() int {
	return len(a.shape)
}

// Size returns the total number of elements.
func (a *Array[T]) Size() int {
	return nditer.Size(a.shape)
}

// IsContiguous reports whether the array is laid out in C order.
func (a *Array[T]) IsContiguous() bool {
	return nditer.IsContiguous(a.layout())
}

// Data returns the underlying storage starting at the view
// offset. For contiguous arrays it reads in C order; for non-contiguous
// views the element order does NOT match C order — use Ravel().Data()
// instead. Mutating it affects every array sharing the storage.
func (a *Array[T]) Data() []T {
	return a.data[a.offset:]
}

// Copy returns an independent, C-order compact copy of the array.
func (a *Array[T]) Copy() *Array[T] {
	shape := slices.Clone(a.shape)
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{a.layout()}, [][]T{a.data},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = src[0][i]
			}
		})
	return dst
}

// Reshape views the array with a new shape when the data is contiguous; a
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

// Transpose views the array with its axes permuted. With no axes it reverses
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

// Slice views a strided sub-array. Ranges cover the leading axes;
// trailing axes not mentioned are taken whole (numpy a[:2] semantics), so a
// zero Range or a missing axis means the entire axis. The result shares data.
// Bounds follow numpy: negative indices count from the end and a negative
// Step reverses the axis.
func (a *Array[T]) Slice(rs ...Range) *Array[T] {
	if len(rs) > len(a.shape) {
		panicRanges(len(rs), len(a.shape))
	}
	if len(rs) < len(a.shape) {
		full := make([]Range, len(a.shape))
		copy(full, rs)
		rs = full
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

// Squeeze drops size-1 dimensions. With no axes all size-1 dimensions are
// dropped; named axes must have size one. The result shares data.
func (a *Array[T]) Squeeze(axes ...int) *Array[T] {
	drop := make([]bool, len(a.shape))
	if len(axes) == 0 {
		for i, d := range a.shape {
			drop[i] = d == 1
		}
	} else {
		for _, ax := range axes {
			ax = a.resolveAxis(ax)
			if a.shape[ax] != 1 {
				panicSqueeze()
			}
			drop[ax] = true
		}
	}
	shape, strides := make([]int, 0, len(a.shape)), make([]int, 0, len(a.shape))
	for i := range a.shape {
		if !drop[i] {
			shape = append(shape, a.shape[i])
			strides = append(strides, a.strides[i])
		}
	}
	return newArray(shape, strides, a.offset, a.data)
}

// ExpandDims inserts a size-1 dimension at axis (negative counts from the
// end). The result shares data.
func (a *Array[T]) ExpandDims(axis int) *Array[T] {
	ax := axis
	if ax < 0 {
		ax += len(a.shape) + 1
	}
	if ax < 0 || ax > len(a.shape) {
		panicAxis(axis, len(a.shape))
	}
	return newArray(
		slices.Insert(slices.Clone(a.shape), ax, 1),
		slices.Insert(slices.Clone(a.strides), ax, 0),
		a.offset, a.data)
}

// SwapAxes exchanges dimensions i and j (negative allowed). The result
// shares data.
func (a *Array[T]) SwapAxes(i, j int) *Array[T] {
	i, j = a.resolveAxis(i), a.resolveAxis(j)
	axes := make([]int, len(a.shape))
	for k := range axes {
		axes[k] = k
	}
	axes[i], axes[j] = axes[j], axes[i]
	l := nditer.Permute(a.layout(), axes)
	return newArray(l.Shape, l.Strides, l.Offset, a.data)
}

// Ravel flattens to 1-D in C order. A contiguous array returns a view
// sharing data; a non-contiguous array is copied first, so sharing is not
// guaranteed.
func (a *Array[T]) Ravel() *Array[T] {
	return a.Reshape(a.Size())
}

// BroadcastTo expands the array to shape with stride-0 dimensions. The
// shape must be right-align-compatible: every trailing dimension is equal to the
// input's or the input dimension is one. The result shares data, and writes
// through it alias across expanded slots — numpy's broadcast_to is read-only,
// but this package cannot enforce that, so treat the result as read-only.
func (a *Array[T]) BroadcastTo(shape ...int) *Array[T] {
	if len(shape) < len(a.shape) {
		panicBroadcast(a.shape, shape)
	}
	lead := len(shape) - len(a.shape)
	for i, d := range a.shape {
		if d != 1 && shape[lead+i] != d {
			panicBroadcast(a.shape, shape)
		}
	}
	strides := make([]int, len(shape))
	for i, d := range a.shape {
		if d != 1 {
			strides[lead+i] = a.strides[i]
		}
	}
	return newArray(slices.Clone(shape), strides, a.offset, a.data)
}

// resolveAxis normalizes a possibly negative axis against the array's rank.
func (a *Array[T]) resolveAxis(axis int) int {
	if axis < 0 {
		axis += len(a.shape)
	}
	if axis < 0 || axis >= len(a.shape) {
		panicAxis(axis, len(a.shape))
	}
	return axis
}

// axesOrAll maps an optional-axes argument: empty means every axis.
func (a *Array[T]) axesOrAll(axes []int) []int {
	if len(axes) == 0 {
		return allAxes(len(a.shape))
	}
	return axes
}
