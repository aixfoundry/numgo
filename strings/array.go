// Package strings ports numpy.strings: elementwise string operations over an
// N-dimensional array of Go strings. StrArray mirrors numgo.Array's view model
// (shape/strides/offset over shared data); all index arithmetic goes through
// internal/nditer.
//
// Divergences from numpy.strings:
//   - Broadcasting: not implemented. numpy broadcasts string operands, but Go
//     callers pre-expand scalars via BroadcastTo; elementwise ops require equal
//     shapes.
//   - Mod, Encode, Decode: skipped (format-spec machinery; Go strings are UTF-8
//     already). Join/Split/SplitLines are kept even though numpy.strings removed
//     them (numpy.char still has them).
//   - Contains: a Go idiom numpy lacks, provided for parity with the strings
//     package.
package strings

import (
	"slices"

	"github.com/aixfoundry/numgo"
	"github.com/aixfoundry/numgo/internal/nditer"
)

// StrArray is an N-dimensional strided view over a flat data slice of strings.
// Strides are in element units. Views share data; use Copy for an independent
// array.
type StrArray struct {
	shape, strides []int
	offset         int
	data           []string
}

// newStrArray wraps existing data without copying. Constructors and views are
// the only callers.
func newStrArray(shape, strides []int, offset int, data []string) *StrArray {
	return &StrArray{shape: shape, strides: strides, offset: offset, data: data}
}

// layout projects the array into the nditer engine's dtype-independent form.
func (a *StrArray) layout() *nditer.Layout {
	return &nditer.Layout{Shape: a.shape, Strides: a.strides, Offset: a.offset}
}

// FromSlice returns a new array holding a copy of data. With no shape the
// result is 1-D; otherwise the element count must match.
func FromSlice(data []string, shape ...int) *StrArray {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	if nditer.Size(shape) != len(data) {
		panicReshape(len(data), shape)
	}
	return newStrArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, slices.Clone(data))
}

// Shape returns a copy of the array's dimensions.
func (a *StrArray) Shape() []int {
	return slices.Clone(a.shape)
}

// Ndim returns the number of dimensions.
func (a *StrArray) Ndim() int {
	return len(a.shape)
}

// Size returns the total number of elements.
func (a *StrArray) Size() int {
	return nditer.Size(a.shape)
}

// IsContiguous reports whether the array is laid out in C order.
func (a *StrArray) IsContiguous() bool {
	return nditer.IsContiguous(a.layout())
}

// Data returns the underlying storage starting at the view
// offset. For contiguous arrays it reads in C order; for non-contiguous
// views the element order does NOT match C order — use Ravel().Data()
// instead. Mutating it affects every array sharing the storage.
func (a *StrArray) Data() []string {
	return a.data[a.offset:]
}

// Copy returns an independent, C-order compact copy of the array.
func (a *StrArray) Copy() *StrArray {
	shape := slices.Clone(a.shape)
	dst := newStrArray(shape, nditer.COrderStrides(shape), 0, make([]string, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{a.layout()}, [][]string{a.data},
		func(d []string, src [][]string) {
			for i := range d {
				d[i] = src[0][i]
			}
		})
	return dst
}

// Reshape views the array with a new shape when the data is contiguous; a
// non-contiguous array is copied first, so the result may not share data.
// At most one dimension may be -1 and is inferred.
func (a *StrArray) Reshape(shape ...int) *StrArray {
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
	return newStrArray(slices.Clone(shape), nditer.COrderStrides(shape), a.offset, a.data)
}

// Transpose views the array with its axes permuted. With no axes it reverses
// them (numpy .T). The result shares data.
func (a *StrArray) Transpose(axes ...int) *StrArray {
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
	return newStrArray(l.Shape, l.Strides, l.Offset, a.data)
}

// Ravel flattens to 1-D in C order. A contiguous array returns a view
// sharing data; a non-contiguous array is copied first, so sharing is not
// guaranteed.
func (a *StrArray) Ravel() *StrArray {
	return a.Reshape(a.Size())
}

// Slice views a strided sub-array. Ranges cover the leading axes;
// trailing axes not mentioned are taken whole (numpy a[:2] semantics), so a
// zero Range or a missing axis means the entire axis. The result shares data.
// Bounds follow numpy: negative indices count from the end and a negative
// Step reverses the axis.
func (a *StrArray) Slice(rs ...numgo.Range) *StrArray {
	if len(rs) > len(a.shape) {
		panicRanges(len(rs), len(a.shape))
	}
	if len(rs) < len(a.shape) {
		full := make([]numgo.Range, len(a.shape))
		copy(full, rs)
		rs = full
	}
	starts := make([]int, len(rs))
	steps := make([]int, len(rs))
	shape := make([]int, len(rs))
	for i, r := range rs {
		starts[i], shape[i], steps[i] = numgo.NormalizeRange(r, a.shape[i])
	}
	l := nditer.Slice(a.layout(), starts, steps, shape)
	return newStrArray(l.Shape, l.Strides, l.Offset, a.data)
}

// resolveAxis normalizes a possibly negative axis against the array's rank.
func (a *StrArray) resolveAxis(axis int) int {
	if axis < 0 {
		axis += len(a.shape)
	}
	if axis < 0 || axis >= len(a.shape) {
		panicAxis(axis, len(a.shape))
	}
	return axis
}

// At returns the element at coords. It panics if the rank or index is invalid.
func (a *StrArray) At(coords ...int) string {
	if err := nditer.CheckCoords(a.layout(), coords); err != nil {
		panicIndex(coords, a.shape)
	}
	return a.data[nditer.Index(a.layout(), coords)]
}

// Set writes v at coords in place and returns a for chaining.
func (a *StrArray) Set(coords []int, v string) *StrArray {
	if err := nditer.CheckCoords(a.layout(), coords); err != nil {
		panicIndex(coords, a.shape)
	}
	a.data[nditer.Index(a.layout(), coords)] = v
	return a
}

// Concatenate joins arrays along axis (negative allowed). All dimensions
// except axis must match exactly. The result is new data.
func Concatenate(axis int, as ...*StrArray) *StrArray {
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
	out := newStrArray(shape, nditer.COrderStrides(shape), 0, make([]string, nditer.Size(shape)))
	off := 0
	for _, a := range as {
		rs := make([]numgo.Range, out.Ndim())
		rs[ax] = numgo.Range{Start: off, Stop: off + a.shape[ax]}
		dst := out.Slice(rs...)
		nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{a.layout()}, [][]string{a.data},
			func(d []string, src [][]string) {
				for i := range d {
					d[i] = src[0][i]
				}
			})
		off += a.shape[ax]
	}
	return out
}
