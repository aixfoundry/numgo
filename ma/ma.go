// Package ma provides numpy.ma-style masked arrays over numgo.Array
// (numpy/ma/core.py semantics; see docs/architecture.md).
//
// A Masked pairs a data array with a parallel mask: true marks an element as
// invalid. The mask is always materialized (a []bool whose length equals the
// data's Size) and is indexed in C order of the data's LOGICAL shape, so it
// stays aligned no matter what view the underlying data is.
//
// Semantics:
//   - Constructors copy their inputs; operations are pure (new Masked). There
//     are no zero-copy views: shape ops rebuild the mask for the new shape.
//   - Element-wise ops OR the operand masks. Div additionally masks positions
//     where the divisor is zero (numpy ma domain behavior: warnings become
//     mask bits, the IEEE value stays in the data).
//   - Reductions ignore masked elements. A slice with no unmasked elements
//     yields a NaN value that is itself masked (numpy returns a masked element
//     after RuntimeWarning).
//   - ArgMax/ArgMin skip masked entries and return -1 for an all-masked
//     channel (numpy returns a masked index).
//   - Any/All consider unmasked values only; a Go bool cannot express numpy's
//     masked answer, so all-masked input gives Any=false, All=true (vacuous).
//   - FilledDefault fills with numpy's default fill table for float/complex:
//     1e20 (complex real part 1e20), matching np.ma.default_fill_value.
//   - SetUnmasked writes the value and clears the mask bit; under a hard mask
//     a masked position silently refuses the write (numpy behavior).
package ma

import (
	"reflect"
	"slices"

	"github.com/aixfoundry/numgo"
	"github.com/aixfoundry/numgo/internal/nditer"
)

// Masked is a data array plus an element mask. mask[i] true means element i
// (C order of the logical shape) is invalid. The zero value is unusable; use
// the constructors.
type Masked[T numgo.Number] struct {
	a    *numgo.Array[T]
	mask []bool
	hard bool
}

// New returns a masked view-free copy of a with no elements masked
// (numpy's nomask).
func New[T numgo.Number](a *numgo.Array[T]) *Masked[T] {
	return &Masked[T]{a: a.Copy(), mask: make([]bool, a.Size())}
}

// FromData builds a Masked from data and a mask, copying both. The mask
// length must equal data's Size.
func FromData[T numgo.Number](data *numgo.Array[T], mask []bool) *Masked[T] {
	if len(mask) != data.Size() {
		panicMaskedSize()
	}
	return &Masked[T]{a: data.Copy(), mask: slices.Clone(mask)}
}

// newMasked adopts fresh data and a fresh mask (no copies) — internal fast
// path for pure operations whose outputs are new already.
func newMasked[T numgo.Number](a *numgo.Array[T], mask []bool) *Masked[T] {
	return &Masked[T]{a: a, mask: mask}
}

// Shape returns a copy of the data's dimensions.
func (m *Masked[T]) Shape() []int { return m.a.Shape() }

// Ndim returns the data's dimension count.
func (m *Masked[T]) Ndim() int { return m.a.Ndim() }

// Size returns the data's element count (and the mask length).
func (m *Masked[T]) Size() int { return m.a.Size() }

// Data returns the underlying data array.
func (m *Masked[T]) Data() *numgo.Array[T] { return m.a }

// Mask returns a copy of the mask, C order of the logical shape.
func (m *Masked[T]) Mask() []bool { return slices.Clone(m.mask) }

// CountUnmasked returns the number of valid elements.
func (m *Masked[T]) CountUnmasked() int {
	n := 0
	for _, bad := range m.mask {
		if !bad {
			n++
		}
	}
	return n
}

// Copy returns an independent copy of data and mask (hard flag kept).
func (m *Masked[T]) Copy() *Masked[T] {
	return &Masked[T]{a: m.a.Copy(), mask: slices.Clone(m.mask), hard: m.hard}
}

// Filled returns a copy of the data with every masked element replaced
// by fill.
func (m *Masked[T]) Filled(fill T) *numgo.Array[T] {
	out := m.a.Copy()
	d := out.Data()
	for i, bad := range m.mask {
		if bad {
			d[i] = fill
		}
	}
	return out
}

// FilledDefault fills masked elements with numpy's default fill value
// for the element type: 1e20 (real part 1e20 for complex), per
// np.ma.default_fill_value.
func (m *Masked[T]) FilledDefault() *numgo.Array[T] {
	return m.Filled(fromParts[T](1e20, 0))
}

// Compressed returns the unmasked elements as a new 1-D array in C order
// (numpy compressed()); possibly empty.
func (m *Masked[T]) Compressed() *numgo.Array[T] {
	vals := make([]T, 0, m.CountUnmasked())
	flat := m.a.Ravel().Data()
	for i, bad := range m.mask {
		if !bad {
			vals = append(vals, flat[i])
		}
	}
	return numgo.New1DArray(vals)
}

// HardMask enables the hard mask: SetUnmasked silently refuses
// writes to masked positions (numpy hard_mask).
func (m *Masked[T]) HardMask() *Masked[T] { m.hard = true; return m }

// SoftMask disables the hard mask (numpy default).
func (m *Masked[T]) SoftMask() *Masked[T] { m.hard = false; return m }

// SetUnmasked writes v at coords and clears the mask bit there. A
// masked position under a hard mask silently refuses the write and is returned
// unchanged (numpy behavior). Invalid coords panic.
func (m *Masked[T]) SetUnmasked(coords []int, v T) *Masked[T] {
	l := maskLayout(m.a.Shape())
	if err := nditer.CheckCoords(l, coords); err != nil {
		panicIndex(coords, m.a.Shape())
	}
	i := nditer.Index(l, coords)
	if m.hard && m.mask[i] {
		return m
	}
	m.a.Set(coords, v)
	m.mask[i] = false
	return m
}

// maskLayout builds a C-order layout over shape for mask/data indexing.
func maskLayout(shape []int) *nditer.Layout {
	return &nditer.Layout{Shape: shape, Strides: nditer.COrderStrides(shape)}
}

// walk visits every coordinate tuple of shape in C order. The coords slice is
// reused across calls; fn must not retain it.
func walk(shape []int, fn func(coords []int)) {
	coords := make([]int, len(shape))
	n := nditer.Size(shape)
	for range n {
		fn(coords)
		for k := len(coords) - 1; k >= 0; k-- {
			coords[k]++
			if coords[k] < shape[k] {
				break
			}
			coords[k] = 0
		}
	}
}

// allAxes is the no-axes convention: fold over every axis.
func allAxes(ndim int) []int {
	axes := make([]int, ndim)
	for i := range axes {
		axes[i] = i
	}
	return axes
}

// reduceShape returns shape with the axes dropped; folding everything
// collapses to a 1-element 1-D shape (root convention).
func reduceShape(shape, axes []int) []int {
	drop := make([]bool, len(shape))
	for _, ax := range axes {
		drop[ax] = true
	}
	out := make([]int, 0, len(shape))
	for i, d := range shape {
		if !drop[i] {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		return []int{1}
	}
	return out
}

// isComplexKind reports whether T is a complex type.
func isComplexKind[T numgo.Number]() bool {
	var v T
	k := reflect.TypeOf(v).Kind()
	return k == reflect.Complex64 || k == reflect.Complex128
}

// parts splits any Number into float64 components (same shortcut as the root
// package: per-element reflect, fine until profiling says otherwise).
func parts[T numgo.Number](v T) (re, im float64) {
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Complex64 || rv.Kind() == reflect.Complex128 {
		c := rv.Complex()
		return real(c), imag(c)
	}
	return rv.Float(), 0
}

// fromParts builds a Number from float64 components.
func fromParts[T numgo.Number](re, im float64) T {
	var out T
	rv := reflect.ValueOf(&out).Elem()
	if rv.Kind() == reflect.Complex64 || rv.Kind() == reflect.Complex128 {
		rv.SetComplex(complex(re, im))
	} else {
		rv.SetFloat(re)
	}
	return out
}
